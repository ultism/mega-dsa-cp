"""Compile-only probe for Phase 1.3 kernels (single GPU, no comm)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8
from cutlass.cute.runtime import make_ptr
from cutlass.cute.typing import AddressSpace
import cuda.bindings.driver as cuda_drv

from mega_dsa_cp.tiles import compress as C

B, MAXE, MAXE128 = 8, 128, 4


def ptr(dtype, t, align=16):
    return make_ptr(dtype, t.data_ptr(), AddressSpace.gmem, assumed_align=align)


def main():
    torch.cuda.set_device(0)
    dev = "cuda"
    ring_attn = torch.zeros(B, C.RING, C.REC_ATTN, dtype=torch.float32, device=dev)
    ring_idx = torch.zeros(B, C.RING, C.REC_IDX, dtype=torch.float32, device=dev)
    c128_state = torch.zeros(B, 3, 512, dtype=torch.float32, device=dev)
    cmp_pool = torch.zeros(B, MAXE, 512, dtype=torch.uint8, device=dev)
    idxk_pool = torch.zeros(B, 1, C.IDXK_PAGE_BYTES, dtype=torch.uint8, device=dev)
    c128_pool = torch.zeros(B, MAXE128, 512, dtype=torch.uint8, device=dev)
    k_valid = torch.zeros(B, dtype=torch.int32, device=dev)
    dbg_attn = torch.zeros(B, MAXE, 512, dtype=torch.float32, device=dev)
    dbg_idx = torch.zeros(B, MAXE, 128, dtype=torch.float32, device=dev)
    dbg128 = torch.zeros(B, MAXE128, 512, dtype=torch.float32, device=dev)
    seq_len = torch.zeros(B, dtype=torch.int32, device=dev)
    ape_attn = torch.zeros(8, 512, dtype=torch.float32, device=dev)
    ape_idx = torch.zeros(8, 128, dtype=torch.float32, device=dev)
    ape128 = torch.zeros(128, 512, dtype=torch.float32, device=dev)
    nw = torch.ones(512, dtype=torch.float32, device=dev)
    nwi = torch.ones(128, dtype=torch.float32, device=dev)
    freqs = torch.zeros(4096, 64, dtype=torch.float32, device=dev)
    pay = torch.zeros(1 << 20, dtype=torch.uint8, device=dev)
    offs = torch.zeros(2, dtype=torch.int64, device=dev)
    evb = torch.zeros(64, dtype=torch.uint64, device=dev)
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    print("compiling csa_step...", flush=True)
    cute.compile(
        C.launch_csa_step, B,
        ptr(Float32, ring_attn), ptr(Float32, ring_idx),
        ptr(Float32, ring_attn), ptr(Float32, ring_idx),
        ptr(Float32, ape_attn), ptr(Float32, ape_idx), ptr(Int32, seq_len),
        ptr(cutlass.Uint8, pay, 256), ptr(Int64, offs), Int64(0), Int64(0),
        ptr(cutlass.Uint64, evb, 128), ptr(Int64, offs), Int32(0), stream,
    )
    print("compiling csa_finalize...", flush=True)
    cute.compile(
        C.launch_csa_finalize, B, MAXE,
        ptr(Int32, seq_len),
        ptr(cutlass.Uint8, pay, 256), ptr(Int64, offs), Int64(0), Int64(0),
        ptr(cutlass.Uint64, evb, 128), ptr(Int64, offs), Int32(0), Int32(0),
        ptr(Float32, nw), ptr(Float32, nwi), Float32(1e-6),
        ptr(Float32, freqs), Float32(1.0),
        ptr(Uint8, cmp_pool), ptr(Uint8, idxk_pool), ptr(Int32, k_valid),
        ptr(Float32, dbg_attn), ptr(Float32, dbg_idx), stream,
    )
    print("compiling c128_step...", flush=True)
    cute.compile(
        C.launch_c128_step, B,
        ptr(Float32, c128_state), ptr(Float32, ring_attn),
        ptr(Float32, ape128), ptr(Int32, seq_len),
        ptr(cutlass.Uint8, pay, 256), ptr(Int64, offs), Int64(0), Int64(0),
        ptr(cutlass.Uint64, evb, 128), ptr(Int64, offs), Int32(0), stream,
    )
    print("compiling c128_finalize...", flush=True)
    cute.compile(
        C.launch_c128_finalize, B, MAXE128,
        ptr(Int32, seq_len),
        ptr(cutlass.Uint8, pay, 256), ptr(Int64, offs), Int64(0), Int64(0),
        ptr(cutlass.Uint64, evb, 128), ptr(Int64, offs), Int32(0), Int32(0),
        ptr(Float32, nw), Float32(1e-6), ptr(Float32, freqs), Float32(1.0),
        ptr(Uint8, c128_pool), ptr(Int32, k_valid), ptr(Float32, dbg128),
        stream,
    )
    print("ALL COMPILE OK", flush=True)


if __name__ == "__main__":
    main()
