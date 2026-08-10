"""Bisect NVVM failure in tiles/merge.py: compile each new construct alone."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32
from cutlass.cute.runtime import make_ptr

from mega_dsa_cp.tiles.merge import make_key64, red_add_u32_gmem


class ProbeKey:
    @cute.jit
    def __call__(self, v: cute.Pointer, i: cute.Pointer, o: cute.Pointer,
                 stream: cuda.CUstream):
        self.kernel(v, i, o).launch(grid=(1, 1, 1), block=(32, 1, 1),
                                    stream=stream)

    @cute.kernel
    def kernel(self, v: cute.Pointer, i: cute.Pointer, o: cute.Pointer):
        tidx, _, _ = cute.arch.thread_idx()
        mv = cute.make_tensor(v, cute.make_layout((32,)))
        mi = cute.make_tensor(i, cute.make_layout((32,)))
        mo = cute.make_tensor(o, cute.make_layout((32,)))
        k = make_key64(mv[tidx], mi[tidx])
        # exercise shift/mask/sub/compare on Int64
        b = (k >> 27) & Int64(0xFF)
        ok = ((k - Int64(123)) >> 40) == Int64(0)
        mo[tidx] = Int64(0)
        if ok:
            mo[tidx] = b


class ProbeRed:
    @cute.jit
    def __call__(self, o: cute.Pointer, stream: cuda.CUstream):
        self.kernel(o).launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, o: cute.Pointer):
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            red_add_u32_gmem(o, Int32(1))


def main():
    import torch
    v = torch.randn(32, device="cuda")
    i = torch.randint(0, 100, (32,), device="cuda", dtype=torch.int32)
    o64 = torch.zeros(32, device="cuda", dtype=torch.int64)
    o32 = torch.zeros(1, device="cuda", dtype=torch.int32)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    for name, fn in (
        ("key64", lambda: cute.compile(
            ProbeKey(),
            make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Int64, 0, cute.AddressSpace.gmem, assumed_align=16),
            cuda.CUstream(0))),
        ("red_gmem", lambda: cute.compile(
            ProbeRed(),
            make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
            cuda.CUstream(0))),
    ):
        try:
            fn()
            print(f"PROBE {name}: compile OK")
        except Exception as e:
            print(f"PROBE {name}: FAIL {type(e).__name__}: {str(e)[:300]}")
    # runtime smoke for key64
    try:
        ck = cute.compile(
            ProbeKey(),
            make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(Int64, 0, cute.AddressSpace.gmem, assumed_align=16),
            cuda.CUstream(0))
        ck(make_ptr(Float32, v.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
           make_ptr(Int32, i.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
           make_ptr(Int64, o64.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
           stream)
        torch.cuda.synchronize()
        print("PROBE key64 run OK", o64[:4].tolist())
    except Exception as e:
        print(f"PROBE key64 run FAIL: {str(e)[:300]}")


if __name__ == "__main__":
    main()
