"""Torch reference for the merge tile (Phase 1.4b).

Total order = (logit desc, entry_id asc), identical to the kernel's 64-bit
key order. Output is emitted in padded-input (list, j) order to match the
kernel's deterministic strip-mined emit, so reference and kernel compare
bitwise, ordered.
"""

import torch


def merge_topk_ref(
    cand_v: torch.Tensor,   # (B, q, L, maxK) fp32
    cand_i: torch.Tensor,   # (B, q, L, maxK) int32
    cand_c: torch.Tensor,   # (B, q, L) int32
    top_k: int,
):
    B, q, L, maxK = cand_v.shape
    dev = cand_v.device
    out_v = torch.zeros(B, q, top_k, dtype=torch.float32, device=dev)
    out_i = torch.zeros(B, q, top_k, dtype=torch.int32, device=dev)
    out_c = torch.zeros(B, q, dtype=torch.int32, device=dev)
    j = torch.arange(maxK, device=dev)
    valid = j[None, None, None, :] < cand_c[:, :, :, None]     # (B,q,L,maxK)
    for b in range(B):
        for t in range(q):
            v = cand_v[b, t].reshape(-1)
            i = cand_i[b, t].reshape(-1)
            m = valid[b, t].reshape(-1)
            v = v[m]
            i = i[m]
            n = v.numel()
            k = min(n, top_k)
            if n == 0:
                continue
            # total order (v desc, i asc) via stable double sort
            oi = torch.argsort(i, stable=True)
            ov = torch.argsort(v[oi], descending=True, stable=True)
            sel = oi[ov][:k]                                    # positions in v/i
            emit = torch.zeros(n, dtype=torch.bool, device=dev)
            emit[sel] = True
            out_v[b, t, :k] = v[emit]
            out_i[b, t, :k] = i[emit]
            out_c[b, t] = k
    return out_v, out_i, out_c
