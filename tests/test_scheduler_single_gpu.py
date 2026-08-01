"""Single-GPU test: randomized DAG executed through the static scheduler.

Differences from the Phase 0.1 equivalence test: tasks run through the real
scheduler (codegen -> packed queues -> smem staging -> cursor walk), test ops
are dispatched in the kernel's if-elif chain, and the SAME kernel is launched
twice against the same event cells with only the phase scalar bumped —
verifying monotonic phase reuse (no resets, no INIT tasks) end to end.

    python3 tests/test_scheduler_single_gpu.py [seed]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint64
from cutlass.cute.runtime import make_ptr
from cutlass.cute.typing import AddressSpace
import cuda.bindings.driver as cuda_drv

from mega_dsa_cp.events.core import EventSet
from mega_dsa_cp.events.symm import alloc_event_buffer
from mega_dsa_cp.events.ptx import nanosleep
from mega_dsa_cp.schedule.codegen import codegen_schedule
from mega_dsa_cp.schedule.device import StaticScheduler
from dag_gen import random_dag

N_THREADS = 128
OP_NOP, OP_LOG, OP_JITTER, OP_EXTENT = 0, 1, 2, 3
EXTENT_VALUE = 12345


@cute.kernel
def sched_kernel(
    queue: cute.Pointer,
    scalars: cute.Pointer,
    arity: cute.Pointer,
    scopes: cute.Pointer,
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    log_buf: cute.Pointer,  # i32[n_tasks]
    log_count: cute.Pointer,  # i32[1]
    log_ext: cute.Pointer,  # i32[n_tasks]
    row_words: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    events = EventSet(base=ev_base, offsets=ev_offsets, my_rank=Int32(0))
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
            if t.op == OP_JITTER:
                nanosleep(Int32((t.c0 * 37) % 200))
            elif t.op == OP_EXTENT:
                cute.arch.store(log_ext + t.c0, cur.extent(t))
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
    log_buf: cute.Pointer,
    log_count: cute.Pointer,
    log_ext: cute.Pointer,
    row_words: cutlass.Constexpr[int],
    n_ctas: cutlass.Constexpr[int],
    stream: cuda_drv.CUstream,
):
    sched_kernel(
        queue, scalars, arity, scopes, ev_base, ev_offsets,
        log_buf, log_count, log_ext, row_words,
    ).launch(grid=[n_ctas, 1, 1], block=[N_THREADS, 1, 1], stream=stream)


def ptr(dtype, tensor, align=8):
    return make_ptr(dtype, tensor.data_ptr(), AddressSpace.gmem, assumed_align=align)


def run_phase(compiled, args, scalars_t, phase, stream):
    scalars_t[0] = phase
    compiled(*args, stream)
    torch.cuda.synchronize()


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_tasks, n_ctas, edge_prob = 512, 8, 0.02

    tasks, events, _, preds, _ = random_dag(seed, n_tasks, edge_prob, n_ctas)
    op_of = {t.name: i % 4 for i, t in enumerate(tasks)}  # NOP/LOG/JITTER/EXTENT
    extent_of = {t.name: 1 for i, t in enumerate(tasks) if i % 4 == 3}
    sched = codegen_schedule(
        tasks, events, rank=0, n_ctas=n_ctas, op_of=op_of, extent_of=extent_of
    )

    row_words = max(len(q) for q in sched.queues)
    flat = []
    for q in sched.queues:
        flat += q + [0] * (row_words - len(q))
    queue_t = torch.tensor(flat, dtype=torch.int32).cuda()
    arity_t = torch.tensor(sched.arity or [0], dtype=torch.int32).cuda()
    scopes_t = torch.tensor(sched.scopes or [0], dtype=torch.int32).cuda()
    scalars_t = torch.zeros(4, dtype=torch.int64).cuda()
    scalars_t[1] = EXTENT_VALUE

    buf = alloc_event_buffer(max(sched.n_events, 1), rank=0, world_size=1)
    log_buf = torch.full((n_tasks,), -1, dtype=torch.int32).cuda()
    log_count = torch.zeros(1, dtype=torch.int32).cuda()
    log_ext = torch.zeros(n_tasks, dtype=torch.int32).cuda()
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    args = (
        ptr(Int32, queue_t),
        ptr(Uint64, scalars_t),
        ptr(Int32, arity_t),
        ptr(Int32, scopes_t),
        ptr(Uint64, buf.tensor, 128),
        ptr(Int64, buf.offsets),
        ptr(Int32, log_buf),
        ptr(Int32, log_count),
        ptr(Int32, log_ext),
    )
    compiled = cute.compile(launch, *args, row_words, n_ctas, stream)

    for phase in (0, 1):
        log_buf.fill_(-1)
        log_count.zero_()
        log_ext.zero_()
        run_phase(compiled, args, scalars_t, phase, stream)

        order = log_buf.cpu().tolist()
        assert log_count.item() == n_tasks, (
            f"phase{phase}: ran {log_count.item()} != {n_tasks}"
        )
        assert sorted(order) == list(range(n_tasks)), f"phase{phase}: task set"
        position = {t: i for i, t in enumerate(order)}
        violations = [
            (p, t) for t, ps in preds.items() for p in ps if position[p] >= position[t]
        ]
        assert not violations, f"phase{phase}: ordering violations {violations[:10]}"
        ext = log_ext.cpu().tolist()
        for i, t in enumerate(tasks):
            if t.name in extent_of:
                assert ext[i] == EXTENT_VALUE, f"phase{phase}: extent t{i}={ext[i]}"
        print(f"PASS phase={phase}: {n_tasks} tasks via scheduler, extents ok")


if __name__ == "__main__":
    main()
