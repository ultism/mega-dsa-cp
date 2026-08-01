"""Dual-GPU test: Phase 0.3 communication primitives.

mode 0: bulk S2G push to peer + flag-only wait + scoped-read payload verify
        (the full data+flag chain: bulk -> commit/wait -> fence.proxy.async
        -> red notify -> flag-only wait -> ld.acquire.sys payload reads),
        two phases to exercise region rotation + monotonic events.
mode 1: multimem.st allgather (one store lands on both ranks), NVLS only.
mode 2: multimem.ld_reduce fp32 switch reduction, NVLS only.

    torchrun --nproc_per_node=2 tests/test_comm_dual_gpu.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint64, Float32
from cutlass.cute.runtime import make_ptr
from cutlass.cute.typing import AddressSpace
import cuda.bindings.driver as cuda_drv

from mega_dsa_cp.events.core import EventSet, SCOPE_SYS
from mega_dsa_cp.events.symm import alloc_event_buffer
from mega_dsa_cp.comm.buffers import alloc_arena
from mega_dsa_cp.comm.device import PeerBuffer
from mega_dsa_cp.comm.primitives import (
    push_start,
    push_finish,
    multimem_st,
    multimem_ld_reduce,
)

N_THREADS = 128
DATA_BYTES = 4096
CAND_BYTES = 2048  # 1KB per rank
RED_ELEMS = 1024  # fp32
SUM_3F32_BITS = 0x40400000  # 1.0 + 2.0 = 3.0

EV_DATA, EV_CAND, EV_RED = 0, 1, 2
NUM_CELLS = 4


@cute.kernel
def comm_kernel(
    pay_base: cute.Pointer,
    pay_offsets: cute.Pointer,
    mc_base: Int64,
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    data_off: Int64,
    cand_off: Int64,
    red_off: Int64,
    mode: Int32,
    phase: Int32,
    result: cute.Pointer,  # i32[4]
):
    tidx, _, _ = cute.arch.thread_idx()
    events = EventSet(base=ev_base, offsets=ev_offsets, my_rank=my_rank)
    pbuf = PeerBuffer(pay_base, pay_offsets, my_rank, mc_base)
    other = Int32(1) - my_rank
    target = Uint64(Int64(2) * (Int64(phase) + 1))

    if mode == 0:
        smem = cutlass.utils.SmemAllocator()
        sbuf = smem.allocate_tensor(Int32, DATA_BYTES // 4, byte_alignment=16)
        mine = (my_rank + 1) * 1000003 + phase * 7919
        for i in cutlass.range(DATA_BYTES // 4 // N_THREADS + 1):
            idx = i * N_THREADS + tidx
            if idx < DATA_BYTES // 4:
                sbuf[idx] = mine + idx
        cute.arch.fence_view_async_shared()
        cute.arch.barrier()
        if tidx == 0:
            push_start(
                pbuf.peer_ptr(data_off, other, cutlass.Uint8, align=16),
                sbuf.iterator,
                Int32(DATA_BYTES),
            )
            push_finish()
        events.notify(Int32(EV_DATA), my_rank)
        events.notify(Int32(EV_DATA), other)
        events.wait(Int32(EV_DATA), target, SCOPE_SYS, fence_after=False)
        want = (other + 1) * 1000003 + phase * 7919
        lptr = pbuf.peer_ptr(data_off, my_rank, Int32, align=16)
        bad = Int32(0)
        for i in cutlass.range(DATA_BYTES // 4 // N_THREADS + 1):
            idx = i * N_THREADS + tidx
            if idx < DATA_BYTES // 4:
                v = cute.arch.load(lptr + idx, Int32, sem="acquire", scope="sys")
                if v != want + idx:
                    bad += 1
        if bad != 0:
            cute.arch.atomic_add(result, bad, scope="gpu")

    if mode == 1:
        # multimem.st allgather: one store lands on both ranks' cand region
        for i in cutlass.range(64 // N_THREADS + 1):
            idx = i * N_THREADS + tidx  # 16B units within my 1KB block
            if idx < 64:
                base_val = (my_rank + 1) * 100003 + idx * 4
                multimem_st(
                    pbuf.mc_ptr(cand_off + my_rank * 1024 + idx * 16, Int32, align=16),
                    Int32(base_val).ir_value(),
                    Int32(base_val + 1).ir_value(),
                    Int32(base_val + 2).ir_value(),
                    Int32(base_val + 3).ir_value(),
                )
        cute.arch.barrier()
        events.notify(Int32(EV_CAND), my_rank)
        events.notify(Int32(EV_CAND), other)
        events.wait(Int32(EV_CAND), target, SCOPE_SYS, fence_after=False)
        bad = Int32(0)
        lptr = pbuf.peer_ptr(cand_off, my_rank, Int32, align=16)
        for i in cutlass.range(CAND_BYTES // 4 // N_THREADS + 1):
            idx = i * N_THREADS + tidx
            if idx < CAND_BYTES // 4:
                owner = idx // 256  # 1KB blocks
                want = (owner + 1) * 100003 + (idx % 256)
                v = cute.arch.load(lptr + idx, Int32, sem="acquire", scope="sys")
                if v != want:
                    bad += 1
        if bad != 0:
            cute.arch.atomic_add(result + 1, bad, scope="gpu")

    if mode == 2:
        # each rank fills its partial region with float(rank+1)
        fptr = pbuf.peer_ptr(red_off, my_rank, Float32, align=16)
        for i in cutlass.range(RED_ELEMS // N_THREADS + 1):
            idx = i * N_THREADS + tidx
            if idx < RED_ELEMS:
                cute.arch.store(fptr + idx, Float32(my_rank + 1))
        cute.arch.barrier()
        events.notify(Int32(EV_RED), my_rank)
        events.notify(Int32(EV_RED), other)
        events.wait(Int32(EV_RED), target, SCOPE_SYS, fence_after=False)
        bad = Int32(0)
        for i in cutlass.range(RED_ELEMS // 4 // N_THREADS + 1):
            idx = i * N_THREADS + tidx  # 4-float units
            if idx < RED_ELEMS // 4:
                r0, r1, r2, r3 = multimem_ld_reduce(
                    pbuf.mc_ptr(red_off + idx * 16, Float32, align=16),
                    dtype=Float32,
                    num_elements=4,
                )
                if Int32(r0) != SUM_3F32_BITS or Int32(r1) != SUM_3F32_BITS:
                    bad += 1
                if Int32(r2) != SUM_3F32_BITS or Int32(r3) != SUM_3F32_BITS:
                    bad += 1
        if bad != 0:
            cute.arch.atomic_add(result + 2, bad, scope="gpu")


@cute.jit
def launch(
    pay_base: cute.Pointer,
    pay_offsets: cute.Pointer,
    mc_base: Int64,
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    data_off: Int64,
    cand_off: Int64,
    red_off: Int64,
    mode: Int32,
    phase: Int32,
    result: cute.Pointer,
    stream: cuda_drv.CUstream,
):
    comm_kernel(
        pay_base, pay_offsets, mc_base, ev_base, ev_offsets, my_rank,
        data_off, cand_off, red_off, mode, phase, result,
    ).launch(grid=[1, 1, 1], block=[N_THREADS, 1, 1], stream=stream)


def ptr(dtype, addr_or_tensor, align=8):
    if isinstance(addr_or_tensor, int):
        return make_ptr(dtype, addr_or_tensor, AddressSpace.gmem, assumed_align=align)
    return make_ptr(
        dtype, addr_or_tensor.data_ptr(), AddressSpace.gmem, assumed_align=align
    )


def main():
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    torch.cuda.set_device(rank)
    group = dist.new_group(ranks=[0, 1])

    arena = alloc_arena(
        {"data": DATA_BYTES, "cand": CAND_BYTES, "partial": RED_ELEMS * 4},
        phases=2,
        rank=rank,
        world_size=world,
        group_name=group.group_name,
    )
    evbuf = alloc_event_buffer(NUM_CELLS, rank, world, group_name=group.group_name)
    result = torch.zeros(4, dtype=torch.int32).cuda()
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    has_mc = arena.mc_base != 0
    if rank == 0:
        print(f"multicast (NVLS): {'available' if has_mc else 'UNAVAILABLE — modes 1/2 skipped'}")

    args = (
        ptr(cutlass.Uint8, arena.tensor, 256),
        ptr(Int64, arena.offsets),
        Int64(arena.mc_base),
        ptr(Uint64, evbuf.tensor, 128),
        ptr(Int64, evbuf.offsets),
        Int32(rank),
        Int64(arena.region_off("data", 0)),
        Int64(arena.region_off("cand", 0)),
        Int64(arena.region_off("partial", 0)),
        Int32(0),
        Int32(0),
        ptr(Int32, result),
    )
    dist.barrier()
    compiled = cute.compile(launch, *args, stream)

    plan = [(0, "bulk push+flag-only wait")]
    if has_mc:
        plan += [(1, "multimem.st allgather"), (2, "multimem.ld_reduce fp32")]
    for mode, name in plan:
        for phase in (0, 1):
            result.zero_()
            run_args = (
                *args[:6],
                Int64(arena.region_off("data", phase)),
                *args[7:9],
                Int32(mode),
                Int32(phase),
                args[11],
            )
            dist.barrier()
            compiled(*run_args, stream)
            torch.cuda.synchronize()
            dist.barrier()
            bad = result[mode].item()
            assert bad == 0, f"rank{rank} {name} phase{phase}: {bad} mismatches"
        print(f"PASS rank{rank} {name} (2 phases)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
