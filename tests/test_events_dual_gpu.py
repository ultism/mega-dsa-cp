"""Dual-GPU test: ping-pong, fan-in, fan-out over symmetric event cells.

Also validates the payload ordering chain end-to-end: producer does a
relaxed payload store to the peer's cell, then the release-ordered notify
red; consumer reads the payload after wait + fence and checks the value.

    torchrun --nproc_per_node=2 tests/test_events_dual_gpu.py [iters]
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

from mega_dsa_cp.events.core import EventSet, SCOPE_SYS
from mega_dsa_cp.events.symm import alloc_event_buffer
from mega_dsa_cp.events.ptx import st_relaxed_u64, ld_acquire_u64

N_THREADS = 128

EV_A, EV_B, EV_F, EV_G = 0, 1, 2, 3
PAY_PING, PAY_PONG = 4, 5
NUM_CELLS = 8

MODE_PINGPONG = 0
MODE_FANIN = 1
MODE_FANOUT = 2


@cute.kernel
def pattern_kernel(
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    iters: cutlass.Constexpr[int],
    mode: Int32,
    mismatch: cute.Pointer,  # i32[4]: count, first_phase, expected, got
):
    tidx, _, _ = cute.arch.thread_idx()
    events = EventSet(base=ev_base, offsets=ev_offsets, my_rank=my_rank)
    other = Int32(1) - my_rank

    if mode == MODE_PINGPONG:
        for p in cutlass.range(iters):
            phase = Int64(p) + 1
            if my_rank == 0:
                events.wait(EV_A, Uint64(phase), SCOPE_SYS)
                if tidx == 0:
                    v = ld_acquire_u64(events.cell(PAY_PING), SCOPE_SYS)
                    if v != Uint64(p):
                        pos = cute.arch.atomic_add(mismatch, Int32(1), scope="gpu")
                        if pos == 0:
                            cute.arch.store(mismatch + 1, Int32(p))
                            cute.arch.store(mismatch + 2, Int32(Uint64(p)))
                            cute.arch.store(mismatch + 3, Int32(v))
                    st_relaxed_u64(events.peer_cell(PAY_PONG, other), Uint64(p), SCOPE_SYS)
                events.notify_peer(EV_B, other)
            else:
                if tidx == 0:
                    st_relaxed_u64(events.peer_cell(PAY_PING, other), Uint64(p), SCOPE_SYS)
                events.notify_peer(EV_A, other)
                events.wait(EV_B, Uint64(phase), SCOPE_SYS)
                if tidx == 0:
                    v = ld_acquire_u64(events.cell(PAY_PONG), SCOPE_SYS)
                    if v != Uint64(p):
                        pos = cute.arch.atomic_add(mismatch, Int32(1), scope="gpu")
                        if pos == 0:
                            cute.arch.store(mismatch + 1, Int32(p))
                            cute.arch.store(mismatch + 2, Int32(Uint64(p)))
                            cute.arch.store(mismatch + 3, Int32(v))

    if mode == MODE_FANIN:
        if my_rank == 0:
            for p in cutlass.range(iters):
                events.notify_local(EV_F)
                events.wait(EV_F, Uint64(Int64(2) * (Int64(p) + 1)), SCOPE_SYS)
        else:
            for p in cutlass.range(iters):
                events.notify_peer(EV_F, Int32(0))

    if mode == MODE_FANOUT:
        if my_rank == 0:
            for p in cutlass.range(iters):
                events.notify_local(EV_G)
                events.notify_peer(EV_G, other)
                events.wait(EV_G, Uint64(Int64(p) + 1), SCOPE_SYS)
        else:
            for p in cutlass.range(iters):
                events.wait(EV_G, Uint64(Int64(p) + 1), SCOPE_SYS)


@cute.jit
def launch(
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    iters: cutlass.Constexpr[int],
    mode: Int32,
    mismatch: cute.Pointer,
    stream: cuda_drv.CUstream,
):
    pattern_kernel(ev_base, ev_offsets, my_rank, iters, mode, mismatch).launch(
        grid=[1, 1, 1], block=[N_THREADS, 1, 1], stream=stream
    )


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2, "dual-GPU test needs exactly 2 ranks"
    torch.cuda.set_device(rank)
    group = dist.new_group(ranks=[0, 1])

    buf = alloc_event_buffer(NUM_CELLS, rank, world, group_name=group.group_name)
    mismatch = torch.zeros(4, dtype=torch.int32).cuda()
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    args = (
        ptr(Uint64, buf.tensor, 128),
        ptr(Int64, buf.offsets),
        Int32(rank),
        ptr(Int32, mismatch),
        stream,
    )
    dist.barrier()

    first_args = (*args[:3], Int32(MODE_PINGPONG), *args[3:])
    compiled = cute.compile(launch, *first_args[:4], iters, *first_args[4:])

    for mode, name in (
        (MODE_PINGPONG, "ping-pong"),
        (MODE_FANIN, "fan-in"),
        (MODE_FANOUT, "fan-out"),
    ):
        mismatch.zero_()
        mode_args = (*args[:3], Int32(mode), *args[3:])
        compiled(*mode_args[:4], *mode_args[4:])
        torch.cuda.synchronize()
        dist.barrier()
        m = mismatch.cpu().tolist()
        assert m[0] == 0, (
            f"rank{rank} {name}: {m[0]} payload mismatches, "
            f"first at phase {m[1]}: expected {m[2]}, got {m[3]}"
        )
        if rank == 0:
            print(f"PASS {name}: {iters} iters, payload chain verified")
    dist.destroy_process_group()


def ptr(dtype, tensor, align=8):
    return make_ptr(dtype, tensor.data_ptr(), AddressSpace.gmem, assumed_align=align)


if __name__ == "__main__":
    main()
