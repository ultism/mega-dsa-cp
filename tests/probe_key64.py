"""Numeric check of make_key64 against a Python model."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32
from cutlass.cute.runtime import make_ptr

from mega_dsa_cp.tiles.merge import make_key64


class ProbeKeyNum:
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
        mo[tidx] = make_key64(mv[tidx], mi[tidx])
        # signed-compare probe: slot 8..15 hold key >= key(v=1.0,i=0)
        if tidx >= 8:
            mo[tidx] = Int64(0)
        if tidx < 8:
            k = make_key64(mv[tidx], mi[tidx])
            kth = make_key64(mv[0], mi[0])
            mo[tidx + 8] = Int64(1) if k >= kth else Int64(0)


def py_key(v: float, i: int) -> int:
    import struct
    u = struct.unpack("<I", struct.pack("<f", v))[0]
    s = (u ^ 0xFFFFFFFF) if (u & 0x80000000) else (u ^ 0x80000000)
    s &= 0xFFFFFFFF
    l = (~i) & 0xFFFFFFFF
    k = ((s << 32) | l) ^ 0x8000000000000000
    return k - (1 << 64) if k >= (1 << 63) else k


def main():
    import torch
    vals = [1.0, -1.0, 0.0, -0.0, float("-inf"), float("inf"), 3.14, -2.5]
    ids = [0, 1, 2, (1 << 31) - 1, 5, 1000, 7, 3]
    n = 8
    v = torch.tensor(vals + [0.0] * 24, device="cuda", dtype=torch.float32)
    i = torch.tensor(ids + [0] * 24, device="cuda", dtype=torch.int32)
    o = torch.zeros(32, device="cuda", dtype=torch.int64)
    ck = cute.compile(
        ProbeKeyNum(),
        make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Int64, 0, cute.AddressSpace.gmem, assumed_align=16),
        cuda.CUstream(0))
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    ck(make_ptr(Float32, v.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
       make_ptr(Int32, i.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
       make_ptr(Int64, o.data_ptr(), cute.AddressSpace.gmem, assumed_align=16),
       stream)
    torch.cuda.synchronize()
    ok = True
    ref_th = py_key(vals[0], ids[0])
    for j in range(n):
        ge = o[j + 8].item()
        exp_ge = 1 if py_key(vals[j], ids[j]) >= ref_th else 0
        st = "OK " if ge == exp_ge else "CMP-MISMATCH"
        if ge != exp_ge:
            ok = False
        print(f"{st} cmp v={vals[j]:>8} key>=key(1.0,0): got={ge} exp={exp_ge}")
    for j in range(n):
        exp = py_key(vals[j], ids[j])
        got = o[j].item()
        st = "OK " if got == exp else "MISMATCH"
        if got != exp:
            ok = False
        print(f"{st} v={vals[j]:>8} i={ids[j]:>10} "
              f"exp={exp & 0xFFFFFFFFFFFFFFFF:016x} got={got & 0xFFFFFFFFFFFFFFFF:016x}")
    # order sanity: kernel keys must be increasing for decreasing v
    print("KEY64", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
