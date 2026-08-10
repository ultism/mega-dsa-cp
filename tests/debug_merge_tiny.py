"""Tiny hand-computable merge case with debug taps."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from mega_dsa_cp.tiles.merge import run_merge_topk
from mega_dsa_cp.tiles.merge_ref import merge_topk_ref

DEV = "cuda"


def main():
    B, q, L, maxK, K = 1, 1, 2, 16, 8
    g = torch.Generator(device=DEV).manual_seed(1)
    v = torch.randn(B, q, L, maxK, generator=g, device=DEV)
    i = torch.arange(L * maxK, device=DEV, dtype=torch.int32).reshape(1, 1, L, maxK)
    c = torch.full((B, q, L), maxK, dtype=torch.int32, device=DEV)
    out_v = torch.zeros(B, q, K, device=DEV)
    out_i = torch.zeros(B, q, K, dtype=torch.int32, device=DEV)
    out_c = torch.zeros(B, q, dtype=torch.int32, device=DEV)
    dbg = torch.zeros(64, dtype=torch.int32, device=DEV)
    run_merge_topk(v, i, c, out_v, out_i, out_c, dbg, top_k=K)
    torch.cuda.synchronize()
    rv, ri, rc = merge_topk_ref(v, i, c, K)
    print("dbg taps: m_total=%d emit=%d tau_b0=%02x tau_topbyte=%02x" % (
        dbg[4].item(), dbg[5].item(),
        dbg[6].item() & 0xFF, dbg[7].item() & 0xFF))
    for p in range(8):
        print("  pass %d: bnd=%02x c_gt=%d rem2=%d hist[bnd]=%d" % (
            p, dbg[8 + p * 4].item() & 0xFF, dbg[8 + p * 4 + 1].item(),
            dbg[8 + p * 4 + 2].item(), dbg[8 + p * 4 + 3].item()))
    print("ref count:", rc[0, 0].item(), "kernel count:", out_c[0, 0].item())
    print("ref  ids:", ri[0, 0].tolist())
    print("kern ids:", out_i[0, 0].tolist())
    # expected tau from torch: 8th largest key
    s = v[0, 0].reshape(-1).view(torch.int32)
    u = s.to(torch.int64) & 0xFFFFFFFF
    srt = torch.where(u >= 0x80000000, u ^ 0xFFFFFFFF, u ^ 0x80000000)
    key = ((srt << 32) | ((i[0, 0].reshape(-1).to(torch.int64)) ^ 0xFFFFFFFF))
    key = key ^ (1 << 63)
    key_s = torch.where(key >= (1 << 63), key - (1 << 64), key)
    tau_exp = torch.sort(key_s, descending=True).values[K - 1]
    print("expected tau: %016x" % (tau_exp.item() & 0xFFFFFFFFFFFFFFFF))


if __name__ == "__main__":
    main()
