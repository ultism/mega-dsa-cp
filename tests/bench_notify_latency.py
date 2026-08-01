"""Cross-GPU notify->wait latency benchmark (Phase 0.1 acceptance).

Modes (each on FRESH event cells — reusing saturated cells turns waits into
no-ops and the numbers into garbage):
  pingpong    : notify->wait RTT/2, A/B over (poll_sem, fence_after)
  atom-rt     : sys-scope atom with return, full round trip = 2x propagation
  waitoverhead: local wait on a pre-satisfied cell (1 load + barrier + fence),
                isolates per-wait fixed cost from propagation (local run only)

Note: the fence_after=False variant is flag-only — it is the production
candidate IF payload consumers carry their own ordering (scoped loads / TMA
acquire chains). A plain-data consumer needs the fence.

Acceptance: remote pingpong with fence_after=False <= 3.0us (production wait
is flag-only); fenced numbers are reported for reference. Local <= 1.5us.

    python3 tests/bench_notify_latency.py [iters]           # local only
    torchrun --nproc_per_node=2 tests/bench_notify_latency.py [iters]
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint64
from cutlass.cute.runtime import make_ptr
from cutlass.cute.typing import AddressSpace
import cuda.bindings.driver as cuda_drv

from mega_dsa_cp.events.core import EventSet, SCOPE_LOCAL, SCOPE_SYS, CELL_ELEMS
from mega_dsa_cp.events.symm import alloc_event_buffer
from mega_dsa_cp.events.ptx import atomic_add_u64_rt

N_THREADS = 128
NUM_CELLS = 16
BACKOFF_NS = 32


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@cute.kernel
def pingpong_kernel(
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    ev_a: Int32,
    ev_b: Int32,
    iters: cutlass.Constexpr[int],
    warmup: cutlass.Constexpr[int],
    remote: cutlass.Constexpr[bool],
    poll_sem: cutlass.Constexpr[str],
    fence_after: cutlass.Constexpr[bool],
    results: cute.Pointer,  # i64[iters] on the timing side
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    events = EventSet(base=ev_base, offsets=ev_offsets, my_rank=my_rank)
    other = Int32(1) - my_rank
    scope = SCOPE_SYS if cutlass.const_expr(remote) else SCOPE_LOCAL

    # local mode: CTA0 notifies first, CTA1 responds (same GPU).
    # remote mode: rank0 CTA0 vs rank1 CTA0.
    is_first = bidx == 0 if cutlass.const_expr(not remote) else (my_rank == 0)

    if is_first:
        t0 = Int64(0)  # hoisted: staged if-regions require prior definition
        for p in cutlass.range(iters + warmup):
            if tidx == 0:
                t0 = cute.arch.globaltimer()
            if cutlass.const_expr(remote):
                events.notify_peer(ev_a, other)
            else:
                events.notify_local(ev_a)
            events.wait(ev_b, Uint64(Int64(p) + 1), scope, BACKOFF_NS, poll_sem, fence_after)
            if tidx == 0:
                t1 = cute.arch.globaltimer()
                if p >= warmup:
                    cute.arch.store(results + (p - warmup), t1 - t0)
    else:
        for p in cutlass.range(iters + warmup):
            events.wait(ev_a, Uint64(Int64(p) + 1), scope, BACKOFF_NS, poll_sem, fence_after)
            if cutlass.const_expr(remote):
                events.notify_peer(ev_b, other)
            else:
                events.notify_local(ev_b)


@cute.kernel
def atom_rt_kernel(
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    ev_a: Int32,
    iters: cutlass.Constexpr[int],
    warmup: cutlass.Constexpr[int],
    results: cute.Pointer,
):
    tidx, _, _ = cute.arch.thread_idx()
    events = EventSet(base=ev_base, offsets=ev_offsets, my_rank=my_rank)
    other = Int32(1) - my_rank
    if my_rank == 0:
        if tidx == 0:
            target = events.peer_cell(ev_a, other)
            for p in cutlass.range(iters + warmup):
                t0 = cute.arch.globaltimer()
                atomic_add_u64_rt(target, Uint64(1), SCOPE_SYS)
                t1 = cute.arch.globaltimer()
                if p >= warmup:
                    cute.arch.store(results + (p - warmup), t1 - t0)


@cute.kernel
def waitoverhead_kernel(
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    ev: Int32,
    iters: cutlass.Constexpr[int],
    warmup: cutlass.Constexpr[int],
    fence_after: cutlass.Constexpr[bool],
    results: cute.Pointer,
):
    """Wait on a pre-satisfied cell: cost = 1 poll + barrier + (fence)."""
    tidx, _, _ = cute.arch.thread_idx()
    events = EventSet(base=ev_base, offsets=ev_offsets, my_rank=Int32(0))
    t0 = Int64(0)  # hoisted
    for p in cutlass.range(iters + warmup):
        if tidx == 0:
            t0 = cute.arch.globaltimer()
        events.wait(ev, Uint64(1), SCOPE_SYS, BACKOFF_NS, "acquire", fence_after)
        if tidx == 0:
            t1 = cute.arch.globaltimer()
            if p >= warmup:
                cute.arch.store(results + (p - warmup), t1 - t0)


@cute.jit
def launch_pingpong(
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    ev_a: Int32,
    ev_b: Int32,
    iters: cutlass.Constexpr[int],
    warmup: cutlass.Constexpr[int],
    remote: cutlass.Constexpr[bool],
    poll_sem: cutlass.Constexpr[str],
    fence_after: cutlass.Constexpr[bool],
    results: cute.Pointer,
    stream: cuda_drv.CUstream,
):
    grid = [1, 1, 1] if cutlass.const_expr(remote) else [2, 1, 1]
    pingpong_kernel(
        ev_base, ev_offsets, my_rank, ev_a, ev_b,
        iters, warmup, remote, poll_sem, fence_after, results,
    ).launch(grid=grid, block=[N_THREADS, 1, 1], stream=stream)


@cute.jit
def launch_atom(
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    ev_a: Int32,
    iters: cutlass.Constexpr[int],
    warmup: cutlass.Constexpr[int],
    results: cute.Pointer,
    stream: cuda_drv.CUstream,
):
    atom_rt_kernel(ev_base, ev_offsets, my_rank, ev_a, iters, warmup, results).launch(
        grid=[1, 1, 1], block=[N_THREADS, 1, 1], stream=stream
    )


@cute.jit
def launch_waitoverhead(
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    ev: Int32,
    iters: cutlass.Constexpr[int],
    warmup: cutlass.Constexpr[int],
    fence_after: cutlass.Constexpr[bool],
    results: cute.Pointer,
    stream: cuda_drv.CUstream,
):
    waitoverhead_kernel(
        ev_base, ev_offsets, ev, iters, warmup, fence_after, results
    ).launch(grid=[1, 1, 1], block=[N_THREADS, 1, 1], stream=stream)


def ptr(dtype, tensor, align=8):
    return make_ptr(dtype, tensor.data_ptr(), AddressSpace.gmem, assumed_align=align)


def stats(name, samples_ns, divisor=2):
    s = sorted(samples_ns)
    n = len(s)
    med, p10, p90 = s[n // 2] / divisor, s[n // 10] / divisor, s[9 * n // 10] / divisor
    log(f"{name}: median {med:.0f} ns  p10 {p10:.0f}  p90 {p90:.0f}")
    return med


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    warmup = iters // 10

    distributed = "RANK" in os.environ
    if distributed:
        import torch.distributed as dist

        dist.init_process_group(backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        assert world == 2
        torch.cuda.set_device(rank)
        group = dist.new_group(ranks=[0, 1])
        group_name = group.group_name
    else:
        rank, world, group_name = 0, 1, "0"

    buf = alloc_event_buffer(NUM_CELLS, rank, world, group_name=group_name)
    results = torch.zeros(iters, dtype=torch.int64).cuda()
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    base = (ptr(Uint64, buf.tensor, 128), ptr(Int64, buf.offsets), Int32(rank))

    ok = True

    if not distributed:
        # local pingpong (CTA<->CTA)
        args = (*base, Int32(0), Int32(1), ptr(Int64, results), stream)
        log("compile pingpong local")
        cute.compile(
            launch_pingpong, *args[:5], iters, warmup, False, "acquire", True, *args[5:]
        )(*args[:5], *args[5:])
        torch.cuda.synchronize()
        med = stats("local    one-way", results.cpu().tolist())
        ok &= med <= 1500

        # per-wait fixed overhead, fence A/B (cell pre-satisfied at 1)
        for fi, fence_after in enumerate((True, False)):
            buf.tensor[2 * CELL_ELEMS] = 1
            args = (base[0], base[1], Int32(2), ptr(Int64, results), stream)
            log(f"compile waitoverhead fence={fence_after}")
            cute.compile(
                launch_waitoverhead, *args[:3], iters, warmup, fence_after, *args[3:]
            )(*args[:3], *args[3:])
            torch.cuda.synchronize()
            stats(f"waitfix  fence={fence_after} per-wait", results.cpu().tolist(), 1)
    else:
        import torch.distributed as dist

        best_flag_only = None
        variants = (
            ("acquire", True),
            ("relaxed", True),
            ("relaxed", False),
        )
        for vi, (poll_sem, fence_after) in enumerate(variants):
            args = (*base, Int32(2 * vi), Int32(2 * vi + 1), ptr(Int64, results), stream)
            log(f"compile pingpong poll={poll_sem} fence={fence_after}")
            dist.barrier()
            cute.compile(
                launch_pingpong, *args[:5],
                iters, warmup, True, poll_sem, fence_after, *args[5:],
            )(*args[:5], *args[5:])
            torch.cuda.synchronize()
            dist.barrier()
            if rank == 0:
                med = stats(
                    f"remote   poll={poll_sem:8s} fence={int(fence_after)} one-way",
                    results.cpu().tolist(),
                )
                if not fence_after:
                    best_flag_only = med

        log("compile atom-rt")
        args = (*base, Int32(8), ptr(Int64, results), stream)
        dist.barrier()
        cute.compile(launch_atom, *args[:4], iters, warmup, *args[4:])(
            *args[:4], *args[4:]
        )
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            stats("atom-rt  full RTT", results.cpu().tolist(), 1)
            log(f"best flag-only remote one-way: {best_flag_only:.0f} ns")
            ok &= best_flag_only <= 3000
        dist.destroy_process_group()

    if rank == 0:
        log("PASS" if ok else "FAIL: latency above budget")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
