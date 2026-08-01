"""Phase 0.4: skeleton pipeline end-to-end at cp=2 (skeleton freeze point).

The frozen one-layer task graph (30 tasks/rank, 14 events) executed through
the Phase 0.2 scheduler with Phase 0.1 events and Phase 0.3 comm primitives.
Compute is stubbed (sleeps / pattern fills); COMMUNICATION IS REAL:
  - candidates: 128KB per rank via multimem.st allgather
  - partials: 256KB per (spec, head_chunk) via 4x64KB smem->bulk S2G pushes
    to the owning rank's LSE inbox (head-dim reduce-scatter layout)
  - LSE "merge": owner verifies both ranks' inbox payloads (scoped reads)
Two launches (phase 0/1) exercise arena rotation + monotonic events.

    torchrun --nproc_per_node=2 tests/test_skeleton_cp2.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint64, Float32
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
from mega_dsa_cp.events.core import EventSet, SCOPE_SYS as EV_SYS
from mega_dsa_cp.events.symm import alloc_event_buffer
from mega_dsa_cp.events.ptx import nanosleep
from mega_dsa_cp.comm.buffers import alloc_arena
from mega_dsa_cp.comm.device import PeerBuffer
from mega_dsa_cp.comm.primitives import push_start, push_finish, multimem_st
from mega_dsa_cp.schedule.codegen import codegen_schedule
from mega_dsa_cp.schedule.device import StaticScheduler

N_THREADS = 128
N_CTAS = 8
Q_LEN = 2  # spec-token chains
N_HCHUNK = 8  # head chunks per layer
CAND_INTS = 32768  # 128KB candidate block per rank
PART_INTS = 65536  # 256KB partial per (src, s, h)
PART_CHUNKS = 4  # pushed as 4x64KB
NUM_CELLS = 14

OP_SLEEP, OP_MMS_CAND, OP_ATTN, OP_LSEV, OP_OUT = 1, 2, 3, 4, 5

# event ids (order in build_dag's events list): q, logits, cand, sel0, sel1,
# outdone, partin0..7 -> 14 cells total


def slot_of(src, s, h):
    return ((src * Q_LEN + s) * (N_HCHUNK // 2) + h // 2)


def part_base(src, s, h):
    return ((src * Q_LEN + s) * N_HCHUNK + h) * 1000000


def build_dag():
    """Full 2-rank DAG (codegen filters per rank). Event names are shared
    (each rank has its own cell); task names are rank-unique."""
    tasks, events = [], []
    for r in (0, 1):
        tasks += [
            TaskSpec(f"qpath_r{r}", r, notifies=(NotifyEdge("q", r),)),
            *[
                TaskSpec(f"logits_r{r}_c{c}", r, notifies=(NotifyEdge("logits", r),))
                for c in range(4)
            ],
            TaskSpec(f"kvwrite_r{r}", r),
            TaskSpec(
                f"cand_push_r{r}",
                r,
                waits=(WaitEdge("logits"),),
                notifies=(NotifyEdge("cand", 0), NotifyEdge("cand", 1)),
            ),
            *[
                TaskSpec(
                    f"merge_r{r}_s{s}",
                    r,
                    waits=(WaitEdge("cand"),),
                    notifies=(NotifyEdge(f"sel{s}", r),),
                )
                for s in range(Q_LEN)
            ],
            *[
                TaskSpec(
                    f"attn_r{r}_s{s}_h{h}",
                    r,
                    waits=(WaitEdge(f"sel{s}"), WaitEdge("q")),
                    notifies=(NotifyEdge(f"partin{h}", h % 2),),
                    coords=(s, h, 0),
                )
                for s in range(Q_LEN)
                for h in range(N_HCHUNK)
            ],
            *[
                TaskSpec(
                    f"lsemerge_r{r}_h{h}",
                    r,
                    waits=(WaitEdge(f"partin{h}"),),
                    notifies=(NotifyEdge("outdone", r),),
                    coords=(h, 0, 0),
                )
                for h in range(N_HCHUNK)
                if h % 2 == r
            ],
            TaskSpec(f"out_r{r}", r, waits=(WaitEdge("outdone"),)),
        ]
    events = [
        EventSpec("q", SCOPE_LOCAL),
        EventSpec("logits", SCOPE_LOCAL),
        EventSpec("cand", SCOPE_SYS),
        EventSpec("sel0", SCOPE_LOCAL),
        EventSpec("sel1", SCOPE_LOCAL),
        EventSpec("outdone", SCOPE_LOCAL),
        *[EventSpec(f"partin{h}", SCOPE_SYS) for h in range(N_HCHUNK)],
    ]
    op_of = {}
    for r in (0, 1):
        op_of[f"qpath_r{r}"] = OP_SLEEP
        op_of[f"kvwrite_r{r}"] = OP_SLEEP
        op_of[f"out_r{r}"] = OP_OUT
        op_of[f"cand_push_r{r}"] = OP_MMS_CAND
        op_of.update({f"logits_r{r}_c{c}": OP_SLEEP for c in range(4)})
        op_of.update({f"merge_r{r}_s{s}": OP_SLEEP for s in range(Q_LEN)})
        op_of.update({f"attn_r{r}_s{s}_h{h}": OP_ATTN for s in range(Q_LEN) for h in range(N_HCHUNK)})
        op_of.update({f"lsemerge_r{r}_h{h}": OP_LSEV for h in range(N_HCHUNK) if h % 2 == r})
    return tasks, events, op_of


@cute.kernel
def skeleton_kernel(
    queue: cute.Pointer,
    scalars: cute.Pointer,
    arity: cute.Pointer,
    scopes: cute.Pointer,
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    pay_base: cute.Pointer,
    pay_offsets: cute.Pointer,
    mc_base: Int64,
    my_rank: Int32,
    cand_off0: Int64,
    inbox_off0: Int64,
    phase_stride: Int64,
    result: cute.Pointer,  # i32[4]: cand_bad, part_bad, tasks_run, reserved
    row_words: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    events = EventSet(base=ev_base, offsets=ev_offsets, my_rank=my_rank)
    pbuf = PeerBuffer(pay_base, pay_offsets, my_rank, mc_base)
    other = Int32(1) - my_rank
    sched = StaticScheduler(queue, scalars, arity, scopes, events, row_words, N_THREADS)
    cur = sched.init()
    phase64 = Int64(cur.phase)
    cand_off = cand_off0 + phase64 * phase_stride
    inbox_off = inbox_off0 + phase64 * phase_stride
    # attn scratch: allocated ONCE at kernel top (SmemAllocator is a shared
    # bump allocator — per-task allocation would exhaust smem)
    smem = cutlass.utils.SmemAllocator()
    sbuf = smem.allocate_tensor(Int32, PART_INTS // PART_CHUNKS, byte_alignment=16)

    t = cur.read(Int32(0))
    while t.valid:
        cur.waits(t)
        if t.op == OP_SLEEP:
            if tidx == 0:
                nanosleep(Int32(2000))
        elif t.op == OP_OUT:
            pass
        elif t.op == OP_MMS_CAND:
            # 128KB candidate block -> multimem allgather (register pattern)
            for i in cutlass.range(CAND_INTS // 4 // N_THREADS + 1):
                idx = i * N_THREADS + tidx  # v4 units
                if idx < CAND_INTS // 4:
                    base_val = my_rank * 1000000 + idx * 4
                    multimem_st(
                        pbuf.mc_ptr(cand_off + my_rank * (CAND_INTS * 4) + idx * 16, Int32, align=16),
                        Int32(base_val).ir_value(),
                        Int32(base_val + 1).ir_value(),
                        Int32(base_val + 2).ir_value(),
                        Int32(base_val + 3).ir_value(),
                    )
            cute.arch.barrier()
        elif t.op == OP_ATTN:
            s, h = t.c0, t.c1
            owner = h % 2
            slot = slot_of(my_rank, s, h)
            slot_off = inbox_off + slot * (PART_INTS * 4)
            base_val = part_base(my_rank, s, h)
            for c in cutlass.range_constexpr(PART_CHUNKS):
                for i in cutlass.range(PART_INTS // PART_CHUNKS // N_THREADS + 1):
                    idx = i * N_THREADS + tidx
                    if idx < PART_INTS // PART_CHUNKS:
                        sbuf[idx] = base_val + c * (PART_INTS // PART_CHUNKS) + idx
                cute.arch.fence_view_async_shared()
                cute.arch.barrier()
                if tidx == 0:
                    push_start(
                        pbuf.peer_ptr(slot_off + c * (PART_INTS // PART_CHUNKS) * 4, owner, cutlass.Uint8, align=16),
                        sbuf.iterator,
                        Int32(PART_INTS // PART_CHUNKS * 4),
                    )
                    push_finish()
                cute.arch.barrier()
        elif t.op == OP_LSEV:
            h = t.c0
            bad = Int32(0)
            for src in cutlass.range_constexpr(2):
                for s in cutlass.range_constexpr(Q_LEN):
                    slot = slot_of(src, s, h)
                    vptr = pbuf.peer_ptr(inbox_off + slot * (PART_INTS * 4), my_rank, Int32, align=16)
                    want_base = part_base(src, s, h)
                    for i in cutlass.range(PART_INTS // N_THREADS + 1):
                        idx = i * N_THREADS + tidx
                        if idx < PART_INTS:
                            v = cute.arch.load(vptr + idx, Int32, sem="acquire", scope="sys")
                            if v != want_base + idx:
                                bad += 1
            if bad != 0:
                cute.arch.atomic_add(result + 1, bad, scope="gpu")
        # merge_s candidate verify rides on OP_SLEEP tasks named merge_s*:
        # (kept out of the hot skeleton path; see host-side cand check task)
        cur.notifies(t)
        if tidx == 0:
            cute.arch.atomic_add(result + 2, Int32(1), scope="gpu")
        t = cur.advance(t)


@cute.kernel
def candcheck_kernel(
    pay_base: cute.Pointer,
    pay_offsets: cute.Pointer,
    my_rank: Int32,
    cand_off0: Int64,
    phase_stride: Int64,
    phase: Int32,
    result: cute.Pointer,
):
    """Standalone candidate payload check (runs after the skeleton kernel):
    the other rank's multimem.st block must be intact."""
    tidx, _, _ = cute.arch.thread_idx()
    pbuf = PeerBuffer(pay_base, pay_offsets, my_rank, Int64(0))
    other = Int32(1) - my_rank
    cand_off = cand_off0 + Int64(phase) * phase_stride
    vptr = pbuf.peer_ptr(cand_off + other * (CAND_INTS * 4), my_rank, Int32, align=16)
    bad = Int32(0)
    for i in cutlass.range(CAND_INTS // N_THREADS + 1):
        idx = i * N_THREADS + tidx
        if idx < CAND_INTS:
            v = cute.arch.load(vptr + idx, Int32, sem="acquire", scope="sys")
            if v != other * 1000000 + idx:
                bad += 1
    if bad != 0:
        cute.arch.atomic_add(result, bad, scope="gpu")


@cute.jit
def launch_skeleton(
    queue: cute.Pointer,
    scalars: cute.Pointer,
    arity: cute.Pointer,
    scopes: cute.Pointer,
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    pay_base: cute.Pointer,
    pay_offsets: cute.Pointer,
    mc_base: Int64,
    my_rank: Int32,
    cand_off0: Int64,
    inbox_off0: Int64,
    phase_stride: Int64,
    result: cute.Pointer,
    row_words: cutlass.Constexpr[int],
    stream: cuda_drv.CUstream,
):
    skeleton_kernel(
        queue, scalars, arity, scopes, ev_base, ev_offsets,
        pay_base, pay_offsets, mc_base, my_rank,
        cand_off0, inbox_off0, phase_stride, result, row_words,
    ).launch(grid=[N_CTAS, 1, 1], block=[N_THREADS, 1, 1], stream=stream)


@cute.jit
def launch_candcheck(
    pay_base: cute.Pointer,
    pay_offsets: cute.Pointer,
    my_rank: Int32,
    cand_off0: Int64,
    phase_stride: Int64,
    phase: Int32,
    result: cute.Pointer,
    stream: cuda_drv.CUstream,
):
    candcheck_kernel(
        pay_base, pay_offsets, my_rank, cand_off0, phase_stride, phase, result
    ).launch(grid=[1, 1, 1], block=[N_THREADS, 1, 1], stream=stream)


def ptr(dtype, tensor, align=8):
    return make_ptr(dtype, tensor.data_ptr(), AddressSpace.gmem, assumed_align=align)


def main():
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    torch.cuda.set_device(rank)
    group = dist.new_group(ranks=[0, 1])

    tasks, events, op_of = build_dag()
    sched = codegen_schedule(tasks, events, rank=rank, n_ctas=N_CTAS, world_size=world, op_of=op_of)
    row_words = max(len(q) for q in sched.queues)
    flat = []
    for q in sched.queues:
        flat += q + [0] * (row_words - len(q))
    queue_t = torch.tensor(flat, dtype=torch.int32).cuda()
    arity_t = torch.tensor(sched.arity, dtype=torch.int32).cuda()
    scopes_t = torch.tensor(sched.scopes, dtype=torch.int32).cuda()
    scalars_t = torch.zeros(2, dtype=torch.int64).cuda()

    arena = alloc_arena(
        {"cand": CAND_INTS * 4 * 2, "inbox": PART_INTS * 4 * 16},
        phases=2,
        rank=rank,
        world_size=world,
        group_name=group.group_name,
    )
    assert arena.mc_base != 0, "skeleton requires NVLS multicast"
    evbuf = alloc_event_buffer(NUM_CELLS, rank, world, group_name=group.group_name)
    result = torch.zeros(4, dtype=torch.int32).cuda()
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    args = (
        ptr(Int32, queue_t),
        ptr(Uint64, scalars_t),
        ptr(Int32, arity_t),
        ptr(Int32, scopes_t),
        ptr(Uint64, evbuf.tensor, 128),
        ptr(Int64, evbuf.offsets),
        ptr(cutlass.Uint8, arena.tensor, 256),
        ptr(Int64, arena.offsets),
        Int64(arena.mc_base),
        Int32(rank),
        Int64(arena.region_off("cand", 0)),
        Int64(arena.region_off("inbox", 0)),
        Int64(arena.nbytes_per_phase),
        ptr(Int32, result),
    )
    dist.barrier()
    compiled = cute.compile(launch_skeleton, *args, row_words, stream)
    cand_args = (
        args[6],
        args[7],
        Int32(rank),
        args[10],
        args[12],
        Int32(0),
        ptr(Int32, result),
    )
    cand_compiled = cute.compile(launch_candcheck, *cand_args, stream)

    n_tasks_expected = len(sched.topo)
    for phase in (0, 1):
        result.zero_()
        scalars_t[0] = phase
        dist.barrier()
        t0 = time.perf_counter()
        compiled(*args, stream)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1e6
        cand_args_p = (*cand_args[:5], Int32(phase), cand_args[6])
        cand_compiled(*cand_args_p, stream)
        torch.cuda.synchronize()
        dist.barrier()
        r = result.cpu().tolist()
        assert r[0] == 0, f"rank{rank} phase{phase}: {r[0]} candidate mismatches"
        assert r[1] == 0, f"rank{rank} phase{phase}: {r[1]} partial mismatches"
        assert r[2] == n_tasks_expected, (
            f"rank{rank} phase{phase}: ran {r[2]} != {n_tasks_expected} tasks"
        )
        print(
            f"PASS rank{rank} phase={phase}: {r[2]} tasks, "
            f"cand 128KB allgather ok, partials 4MB push+verify ok, wall {dt:.0f} us"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
