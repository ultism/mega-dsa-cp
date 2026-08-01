"""Dual-GPU test: cross-rank DAG executed through the static scheduler.

0.1 tested the raw primitives; this tests codegen + scheduler + events as a
stack: SYS events flow through packed queue edges, phase targets come from
the device scalar, and the same queues run twice (phase 0 and 1) against the
same cells — monotonic reuse across ranks, no resets.

    torchrun --nproc_per_node=2 tests/test_scheduler_dual_gpu.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint64
from cutlass.cute.runtime import make_ptr
from cutlass.cute.typing import AddressSpace
import cuda.bindings.driver as cuda_drv

from mega_dsa_cp.events import (
    SCOPE_LOCAL,
    SCOPE_SYS,
    EventSpec,
    NotifyEdge,
    TaskSpec,
    WaitEdge,
)
from mega_dsa_cp.events.core import EventSet
from mega_dsa_cp.events.symm import alloc_event_buffer
from mega_dsa_cp.schedule.codegen import codegen_schedule
from mega_dsa_cp.schedule.device import StaticScheduler

N_THREADS = 128
NUM_CELLS = 8

TASKS = [
    TaskSpec("a", 0, notifies=(NotifyEdge("e", 1), NotifyEdge("g", 0))),
    TaskSpec("b", 1, waits=(WaitEdge("e"),), notifies=(NotifyEdge("f", 0), NotifyEdge("h", 1))),
    TaskSpec("c", 0, waits=(WaitEdge("f"), WaitEdge("g"))),
    TaskSpec("d", 0, waits=(WaitEdge("g"),)),
    TaskSpec("i", 1, waits=(WaitEdge("h"),)),
]
EVENTS = [
    EventSpec("e", SCOPE_SYS),
    EventSpec("f", SCOPE_SYS),
    EventSpec("g", SCOPE_LOCAL),
    EventSpec("h", SCOPE_LOCAL),
]
# global task indices: a=0 b=1 c=2 d=3 i=4
EXPECTED_ORDER = {
    0: [(0, 2), (0, 3)],  # a before c, a before d
    1: [(1, 4)],  # b before i
}


@cute.kernel
def sched_kernel(
    queue: cute.Pointer,
    scalars: cute.Pointer,
    arity: cute.Pointer,
    scopes: cute.Pointer,
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    log_buf: cute.Pointer,
    log_count: cute.Pointer,
    row_words: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    events = EventSet(base=ev_base, offsets=ev_offsets, my_rank=my_rank)
    sched = StaticScheduler(
        queue, scalars, arity, scopes, events, row_words, N_THREADS
    )
    cur = sched.init()
    t = cur.read(Int32(0))
    while t.valid:
        cur.waits(t)
        if tidx == 0:
            pos = cute.arch.atomic_add(log_count, Int32(1), scope="gpu")
            cute.arch.store(log_buf + pos, t.c0)
        cur.notifies(t)
        t = cur.advance(t)


@cute.jit
def launch(
    queue: cute.Pointer,
    scalars: cute.Pointer,
    arity: cute.Pointer,
    scopes: cute.Pointer,
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    log_buf: cute.Pointer,
    log_count: cute.Pointer,
    row_words: cutlass.Constexpr[int],
    n_ctas: cutlass.Constexpr[int],
    stream: cuda_drv.CUstream,
):
    sched_kernel(
        queue, scalars, arity, scopes, ev_base, ev_offsets,
        my_rank, log_buf, log_count, row_words,
    ).launch(grid=[n_ctas, 1, 1], block=[N_THREADS, 1, 1], stream=stream)


def ptr(dtype, tensor, align=8):
    return make_ptr(dtype, tensor.data_ptr(), AddressSpace.gmem, assumed_align=align)


def main():
    n_ctas = 2
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    torch.cuda.set_device(rank)
    group = dist.new_group(ranks=[0, 1])

    sched = codegen_schedule(TASKS, EVENTS, rank=rank, n_ctas=n_ctas, world_size=world)
    row_words = max(len(q) for q in sched.queues)
    flat = []
    for q in sched.queues:
        flat += q + [0] * (row_words - len(q))
    queue_t = torch.tensor(flat, dtype=torch.int32).cuda()
    arity_t = torch.tensor(sched.arity, dtype=torch.int32).cuda()
    scopes_t = torch.tensor(sched.scopes, dtype=torch.int32).cuda()
    scalars_t = torch.zeros(2, dtype=torch.int64).cuda()

    buf = alloc_event_buffer(NUM_CELLS, rank, world, group_name=group.group_name)
    log_buf = torch.full((8,), -1, dtype=torch.int32).cuda()
    log_count = torch.zeros(1, dtype=torch.int32).cuda()
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    args = (
        ptr(Int32, queue_t),
        ptr(Uint64, scalars_t),
        ptr(Int32, arity_t),
        ptr(Int32, scopes_t),
        ptr(Uint64, buf.tensor, 128),
        ptr(Int64, buf.offsets),
        Int32(rank),
        ptr(Int32, log_buf),
        ptr(Int32, log_count),
    )
    dist.barrier()
    compiled = cute.compile(launch, *args, row_words, n_ctas, stream)

    for phase in (0, 1):
        log_buf.fill_(-1)
        log_count.zero_()
        scalars_t[0] = phase
        dist.barrier()
        compiled(*args, stream)
        torch.cuda.synchronize()
        dist.barrier()

        order = [t for t in log_buf.cpu().tolist() if t >= 0]
        position = {t: i for i, t in enumerate(order)}
        for a, b in EXPECTED_ORDER[rank]:
            assert position[a] < position[b], (
                f"rank{rank} phase{phase}: task {a} not before {b}: {order}"
            )
        print(f"PASS rank{rank} phase={phase}: order {order}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
