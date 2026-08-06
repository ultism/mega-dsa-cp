"""Phase 1.2: torch reference + input synthesis for the HCA fork (tiles/hca.py).

Dual-pool semantics matching the kernel:
- slots [0, 128): sliding-window pool (kv_win[b]), valid j < win_valid[row]
- slots [128, K_valid[row]): compressed pool, gathered by page table
  (paged mode: 16-entry pages; gather mode: entry ids directly)
- attn_sink: per-head extra logit in unscaled-S space with V=0
- LSE in log2 domain: log2(row_sum) + scale_log2 * row_max
Quant scales are folded into softmax_scale by the caller; reference works on
raw fp8-dequant values throughout (same as the kernel).
"""

from __future__ import annotations

import torch

H = 128
D = 512
WIN = 128
LOG2E = 1.4426950408889634


def quant_fp8(x: torch.Tensor) -> torch.Tensor:
    """Per-tensor e4m3 quantization (scale folded out by caller)."""
    amax = x.abs().amax().clamp(min=1e-6)
    scale = amax / 448.0
    return (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def dequant(q: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float32)


def make_inputs(
    B: int,
    T: int,
    s_raws: list[int],
    mode: str,  # "c128a" (page_cmp=16 dense) | "csa" (page_cmp=1 gather, K=512)
    seed: int = 0,
    k_sel: int = 512,
) -> dict:
    """Synthesize kernel inputs. Returns dict of torch tensors (cuda)."""
    assert len(s_raws) == B
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"

    q32 = torch.randn(B, T, H, D, generator=g, device=dev) * 0.5
    kwin32 = torch.randn(B, WIN, D, generator=g, device=dev) * 0.5

    rows = B * T
    if mode == "c128a":
        ratio = 128
        page_cmp = 16
        e_counts = [(s + ratio - 1) // ratio for s in s_raws]
        pages_per_b = [(e + page_cmp - 1) // page_cmp for e in e_counts]
        e_total = max(1, sum(p * page_cmp for p in pages_per_b))
        kcmp32 = torch.randn(e_total, D, generator=g, device=dev) * 0.5
        max_tiles = max((128 + e + WIN - 1) // WIN for e in e_counts)
        cols = (max_tiles - 1) * 8  # cmp pages per row (tile-rounded)
        pt_cmp = torch.zeros(rows, cols, dtype=torch.int32, device=dev)
        base = 0
        for b in range(B):
            for p in range(pages_per_b[b]):
                pt_cmp[b * T : (b + 1) * T, p] = base + p
            base += pages_per_b[b]
        n_valid = torch.tensor(e_counts, dtype=torch.int32)
    else:
        ratio = 4
        page_cmp = 1
        e_counts = [(s + ratio - 1) // ratio for s in s_raws]
        e_max = max(e_counts)
        kcmp32 = torch.randn(e_max, D, generator=g, device=dev) * 0.5
        n_sel = [min(k_sel, e) for e in e_counts]
        max_tiles = max((128 + n + WIN - 1) // WIN for n in n_sel)
        cols = (max_tiles - 1) * 128  # entry ids per row (tile-rounded)
        pt_cmp = torch.zeros(rows, cols, dtype=torch.int32, device=dev)  # pad id 0
        for b in range(B):
            n = n_sel[b]
            for t in range(T):
                n_t = max(1, n - (T - 1 - t))
                ids = torch.randperm(e_counts[b], generator=g, device=dev)[:n_t]
                pt_cmp[b * T + t, :n_t] = ids
        n_valid = torch.tensor(n_sel, dtype=torch.int32)

    # per-row k_valid: later draft tokens see more entries
    k_valid = torch.zeros(rows, dtype=torch.int32, device=dev)
    win_valid = torch.zeros(rows, dtype=torch.int32, device=dev)
    for b in range(B):
        for t in range(T):
            n_t = max(1, int(n_valid[b]) - (T - 1 - t))
            k_valid[b * T + t] = WIN + n_t
            win_valid[b * T + t] = min(WIN, s_raws[b] + t)

    if mode == "c128a":
        kv_cmp = kcmp32.view(sum(pages_per_b), page_cmp, D)
    else:
        kv_cmp = kcmp32.view(e_max, page_cmp, D)
    pt_win = torch.arange(B, dtype=torch.int32, device=dev).repeat_interleave(T).view(rows, 1)

    q_q = quant_fp8(q32)
    kwin_q = quant_fp8(kwin32)
    kv_cmp_q = quant_fp8(kv_cmp)
    out = {
        "q": q_q,
        "kv_win": kwin_q,
        "kv_cmp": kv_cmp_q,
        "pt_win": pt_win,
        "pt_cmp": pt_cmp,
        "k_valid": k_valid,
        "win_valid": win_valid,
        # raw fp8-dequant views: the kernel's exact inputs (scales folded out)
        "q32": dequant(q_q),
        "kwin32": dequant(kwin_q),
        "kcmp32": dequant(kv_cmp_q).view(-1, D),
        "mode": mode,
        "page_cmp": page_cmp,
    }
    return out


def ref_hca_online(
    inp: dict,
    sink: torch.Tensor,
    softmax_scale: float,
    output_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact simulation of the kernel's online softmax: iterate k-tiles in
    order (win tile first), fp8-quantize P per tile at the running-max scale,
    rescale accumulator/row_sum by exp2 corrections. Returns (o, lse)."""
    q32, kwin32, kcmp32 = inp["q32"], inp["kwin32"], inp["kcmp32"]
    B, T, _, _ = q32.shape
    pt_cmp, k_valid, win_valid = inp["pt_cmp"], inp["k_valid"], inp["win_valid"]
    page_cmp = inp["page_cmp"]
    dev = q32.device
    scale_log2 = softmax_scale * LOG2E
    TILE = 128

    o = torch.zeros(B, T, H, D, device=dev)
    lse = torch.full((B, T, H), -float("inf"), device=dev)
    for b in range(B):
        for t in range(T):
            row = b * T + t
            ids = pt_cmp[row]
            if page_cmp == 1:
                k_sel = kcmp32[ids.long()]
            else:
                k_sel = kcmp32.view(-1, page_cmp, D)[ids.long()].view(-1, D)
            s_win = q32[b, t] @ kwin32[b].T
            s_cmp = q32[b, t] @ k_sel.T
            S = torch.cat([s_win, s_cmp], dim=-1)
            n_tiles = (int(k_valid[row]) + TILE - 1) // TILE
            n_slots = n_tiles * TILE
            S = S[:, :n_slots]
            idx = torch.arange(n_slots, device=dev)
            mask = torch.where(
                idx < WIN, idx >= win_valid[row], idx >= k_valid[row]
            )
            S = S.masked_fill(mask[None, :], -float("inf"))
            V = torch.cat([kwin32[b], k_sel], dim=0)[:n_slots]

            run_max = sink.clone()  # (H,) sink applied on "split 0" (always here)
            acc = torch.zeros(H, D, device=dev)
            row_sum = torch.where(
                torch.isinf(sink), torch.zeros(H, device=dev), torch.ones(H, device=dev)
            )
            for j in range(n_tiles):
                tile_S = S[:, j * TILE : (j + 1) * TILE]
                new_max = torch.maximum(run_max, tile_S.amax(dim=-1))
                # skip_correction_threshold=0.0: equal max -> factor exactly 1
                corr = torch.exp2((run_max - new_max) * scale_log2)
                corr = torch.where(new_max == run_max, torch.ones_like(corr), corr)
                P = torch.exp2((tile_S - new_max[:, None]) * scale_log2)
                P = P.to(torch.float8_e4m3fn).to(torch.float32)
                acc = acc * corr[:, None] + P @ V[j * TILE : (j + 1) * TILE]
                row_sum = row_sum * corr + P.sum(dim=-1)
                run_max = new_max
            safe = row_sum > 0
            o[b, t] = torch.where(
                safe[:, None], acc / row_sum.clamp(min=1e-30)[:, None], acc * 0.0
            ) * output_scale
            lse[b, t] = torch.where(
                safe,
                torch.log2(row_sum.clamp(min=1e-30)) + scale_log2 * run_max,
                torch.full_like(run_max, -float("inf")),
            )
    return o, lse


def ref_hca(
    inp: dict,
    sink: torch.Tensor,  # (H,) fp32, -inf disables
    softmax_scale: float,
    output_scale: float = 1.0,
    fp8_p: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dual-pool fp32 reference. Returns (o (B,T,H,D), lse (B,T,H))."""
    q32, kwin32, kcmp32 = inp["q32"], inp["kwin32"], inp["kcmp32"]
    B, T, _, _ = q32.shape
    pt_cmp, k_valid, win_valid = inp["pt_cmp"], inp["k_valid"], inp["win_valid"]
    page_cmp = inp["page_cmp"]
    dev = q32.device
    scale_log2 = softmax_scale * LOG2E

    o = torch.zeros(B, T, H, D, device=dev)
    lse = torch.full((B, T, H), -float("inf"), device=dev)
    for b in range(B):
        for t in range(T):
            row = b * T + t
            ids = pt_cmp[row]
            if page_cmp == 1:
                k_sel = kcmp32[ids.long()]
            else:
                k_sel = kcmp32.view(-1, page_cmp, D)[ids.long()].view(-1, D)
            s_win = q32[b, t] @ kwin32[b].T  # (H, 128)
            s_cmp = q32[b, t] @ k_sel.T  # (H, cols)
            S = torch.cat([s_win, s_cmp], dim=-1)
            n = S.shape[-1]
            idx = torch.arange(n, device=dev)
            mask = torch.where(
                idx < WIN,
                idx >= win_valid[row],
                idx >= k_valid[row],
            )
            S = S.masked_fill(mask[None, :], -float("inf"))
            row_max = S.amax(dim=-1)  # (H,)
            row_max = torch.maximum(row_max, sink)
            P = torch.exp2((S - row_max[:, None]) * scale_log2)
            if fp8_p:
                P = P.to(torch.float8_e4m3fn).to(torch.float32)
            row_sum = P.sum(dim=-1)
            sink_p = torch.exp2((sink - row_max) * scale_log2)
            sink_p = torch.where(torch.isinf(sink), torch.zeros_like(sink_p), sink_p)
            row_sum = row_sum + sink_p
            V = torch.cat([kwin32[b], k_sel], dim=0)  # (128+cols, D)
            acc = P @ V
            safe = row_sum > 0
            o[b, t] = torch.where(
                safe[:, None], acc / row_sum.clamp(min=1e-30)[:, None], acc * 0.0
            ) * output_scale
            lse[b, t] = torch.where(
                safe,
                torch.log2(row_sum.clamp(min=1e-30)) + scale_log2 * row_max,
                torch.full_like(row_sum, -float("inf")),
            )
    return o, lse
