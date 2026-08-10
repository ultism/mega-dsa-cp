"""Phase 1.4c single-GPU acceptance for the LSE merge tile.

Gates (docs/phase1.4-comm-tasks.md §7): fp32 tolerance vs same-formula torch
ref; cp in {2, 4}; -inf edges (some ranks empty, all-empty heads); random
partials at realistic scales.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from mega_dsa_cp.tiles.lse_merge import H, D, run_lse_merge
from mega_dsa_cp.tiles.lse_merge_ref import lse_merge_ref

DEV = "cuda"


def _run_case(name, B, q, cp, cp_max, mode, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    acc_o = torch.randn(B, q, cp_max, H, D, generator=g, device=DEV) * 2.0
    acc_lse = torch.randn(B, q, cp_max, H, generator=g,
                          device=DEV) * 4.0 + 2.0
    if mode == "sparse" or mode == "allempty":
        # rank r empty on alternating (b,t,h) stripes
        drop = torch.rand(B, q, cp_max, H, generator=g, device=DEV) < 0.35
        acc_lse = acc_lse.masked_fill(drop, -float("inf"))
    if mode == "allempty":
        acc_lse[:, :, :, : H // 4] = -float("inf")       # 1/4 heads all empty
    o = torch.full((B, q, H, D), float("nan"), device=DEV)
    lse = torch.full((B, q, H), float("nan"), device=DEV)
    run_lse_merge(acc_o, acc_lse, o, lse, cp)
    torch.cuda.synchronize()
    ro, rlse = lse_merge_ref(acc_o, acc_lse, cp)

    finite = torch.isfinite(rlse)
    assert bool((torch.isfinite(lse) == finite).all()), f"[{name}] -inf map differs"
    d_o = (o - ro).abs().max().item()
    d_lse = (lse - rlse)[finite].abs().max().item() if finite.any() else 0.0
    zero_ok = bool((o[~finite] == 0.0).all()) if (~finite).any() else True
    ok = d_o < 1e-5 and d_lse < 1e-4 and zero_ok
    print(f"{'PASS' if ok else 'FAIL'} {name}: B={B} q={q} cp={cp}/{cp_max} "
          f"mode={mode} dO={d_o:.2e} dLSE={d_lse:.2e} zero_ok={zero_ok}")
    assert ok, name


def main():
    _run_case("cp2", 4, 2, 2, 2, "full")
    _run_case("cp4_padded", 4, 2, 4, 8, "full")      # cp_max padding path
    _run_case("cp2_sparse", 4, 2, 2, 2, "sparse")
    _run_case("cp4_sparse", 2, 2, 4, 4, "sparse", seed=3)
    _run_case("allempty", 2, 2, 2, 2, "allempty")
    _run_case("B32", 32, 2, 2, 2, "sparse", seed=5)
    print("ALL LSE-MERGE GATES PASS")


if __name__ == "__main__":
    main()
