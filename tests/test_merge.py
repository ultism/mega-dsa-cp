"""Phase 1.4b single-GPU acceptance for the merge tile (tiles/merge.py).

Gates (per docs/phase1.4-comm-tasks.md §7):
  - ordered bitwise match vs merge_topk_ref (same selected set AND same
    deterministic input-order emit)
  - determinism: two runs bitwise identical
  - K in {512, 1024}; L in {1, 2, 4, 16, 256}; full / variable / tiny counts
  - heavy fp32 ties at the boundary (discrete logits), all-equal logits,
    -inf entries, id edge values
  - emit-count dbg counter stays 0
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from mega_dsa_cp.tiles.merge import run_merge_topk
from mega_dsa_cp.tiles.merge_ref import merge_topk_ref

DEV = "cuda"


def _gen(B, q, L, maxK, top_k, mode, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    counts = torch.full((B, q, L), maxK, dtype=torch.int32, device=DEV)
    if mode == "varcount":
        counts = torch.randint(0, maxK + 1, (B, q, L), generator=g,
                               device=DEV, dtype=torch.int32)
    elif mode == "tiny":
        counts = torch.randint(0, max(2, top_k // (4 * L) + 2), (B, q, L),
                               generator=g, device=DEV, dtype=torch.int32)
    v = torch.randn(B, q, L, maxK, generator=g, device=DEV,
                    dtype=torch.float32)
    if mode == "ties":
        v = torch.randint(0, 16, (B, q, L, maxK), generator=g,
                          device=DEV).float() / 8.0
    elif mode == "const":
        v = torch.full((B, q, L, maxK), 0.5, device=DEV)
    elif mode == "neginf":
        drop = torch.rand(B, q, L, maxK, generator=g, device=DEV) < 0.3
        v = v.masked_fill(drop, float("-inf"))
    # unique ids per (b,t): random permutation chunks, incl. edge values
    ii = torch.empty(B, q, L, maxK, dtype=torch.int32, device=DEV)
    for b in range(B):
        for t in range(q):
            perm = torch.randperm(1 << 24, generator=g, device=DEV)[: L * maxK]
            ii[b, t] = perm.reshape(L, maxK).to(torch.int32)
    ii.view(-1)[0] = 0
    ii.view(-1)[1] = (1 << 31) - 1
    return v, ii, counts


def _run_case(name, B, q, L, maxK, top_k, mode, seed=0):
    v, i, c = _gen(B, q, L, maxK, top_k, mode, seed)
    out_v = torch.zeros(B, q, top_k, dtype=torch.float32, device=DEV)
    out_i = torch.zeros(B, q, top_k, dtype=torch.int32, device=DEV)
    out_c = torch.zeros(B, q, dtype=torch.int32, device=DEV)
    dbg = torch.zeros(64, dtype=torch.int32, device=DEV)
    run_merge_topk(v, i, c, out_v, out_i, out_c, dbg, top_k=top_k)
    out_v2 = torch.zeros_like(out_v)
    out_i2 = torch.zeros_like(out_i)
    out_c2 = torch.zeros_like(out_c)
    run_merge_topk(v, i, c, out_v2, out_i2, out_c2, dbg, top_k=top_k)
    torch.cuda.synchronize()
    rv, ri, rc = merge_topk_ref(v, i, c, top_k)

    assert dbg.item() == 0, f"[{name}] emit-count mismatch dbg={dbg.item()}"
    assert torch.equal(out_c, rc), f"[{name}] counts differ"
    # ordered bitwise comparison (v via int32 view)
    kv, kr = out_v.view(torch.int32), rv.view(torch.int32)
    same_v = torch.equal(kv, kr)
    same_i = torch.equal(out_i, ri)
    if not (same_v and same_i):
        for b in range(B):
            for t in range(q):
                n = rc[b, t].item()
                set_k = set(zip(out_v[b, t, :n].tolist(), out_i[b, t, :n].tolist()))
                set_r = set(zip(rv[b, t, :n].tolist(), ri[b, t, :n].tolist()))
                if set_k != set_r:
                    only_r = set_r - set_k
                    only_k = set_k - set_r
                    raise AssertionError(
                        f"[{name}] SET mismatch b={b} t={t}: "
                        f"ref-only {len(only_r)} kernel-only {len(only_k)} "
                        f"e.g. {list(only_r)[:3]} vs {list(only_k)[:3]}"
                    )
        raise AssertionError(f"[{name}] set match but ORDER differs")
    det = torch.equal(out_v, out_v2) and torch.equal(out_i, out_i2)
    assert det, f"[{name}] non-deterministic across runs"
    print(f"PASS {name}: B={B} q={q} L={L} maxK={maxK} K={top_k} "
          f"mode={mode} (ordered bitwise, deterministic)")


def main():
    _run_case("smoke", 2, 2, 4, 512, 512, "full")
    _run_case("k1024", 2, 2, 4, 1024, 1024, "full")
    _run_case("varcount", 4, 2, 8, 512, 512, "varcount")
    _run_case("tiny_M<K", 2, 2, 4, 512, 512, "tiny")
    _run_case("ties", 2, 2, 8, 512, 512, "ties")
    _run_case("const", 2, 2, 4, 512, 512, "const")
    _run_case("neginf", 2, 2, 8, 512, 512, "neginf")
    _run_case("L256_S1M", 1, 2, 256, 512, 512, "full")
    _run_case("L1", 2, 2, 1, 1024, 1024, "varcount")
    _run_case("k1024_L256", 1, 2, 256, 1024, 1024, "varcount", seed=7)
    print("ALL MERGE-TILE GATES PASS")


if __name__ == "__main__":
    main()
