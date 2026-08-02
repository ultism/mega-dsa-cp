"""MXFP4 (E2M1 + UE8M0/group-32) host-side quant / dequant / reference scoring.

Quantization matches DeepGEMM's FP4 indexer path bit-for-bit (sglang PR #30546):
round through bf16, per-32-group amax, scale = UE8M0-ceil(max(amax/6, 1e-4)),
RTNE to E2M1. Packing: two nibbles per byte, element 2i low / 2i+1 high
(CUTLASS sub-byte convention). UE8M0 stored as biased exponent byte (e+127).
"""

from __future__ import annotations

import torch

E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
HEAD_DIM = 128
SF_VEC = 32


def quant_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x (..., K) float, K % 32 == 0 -> (packed uint8 (..., K//2), sf uint8 (..., K//32))."""
    assert x.shape[-1] % SF_VEC == 0
    xb = x.bfloat16().float()
    g = xb.reshape(*xb.shape[:-1], -1, SF_VEC)
    amax = g.abs().amax(dim=-1)
    s = (amax / 6.0).clamp(min=1e-4)
    e = torch.ceil(torch.log2(s))
    scale = torch.exp2(e)  # exact power of two
    q = g / scale.unsqueeze(-1)
    aq = q.abs()
    code = torch.zeros_like(aq, dtype=torch.int32)
    code = torch.where(aq > 0.25, 1, code)  # tie 0.25 -> 0 (RTNE to even code)
    code = torch.where(aq >= 0.75, 2, code)  # tie 0.75 -> 2
    code = torch.where(aq > 1.25, 3, code)  # tie 1.25 -> 2
    code = torch.where(aq >= 1.75, 4, code)  # tie 1.75 -> 4
    code = torch.where(aq > 2.5, 5, code)  # tie 2.5 -> 4
    code = torch.where(aq >= 3.5, 6, code)  # tie 3.5 -> 6
    code = torch.where(aq > 5.0, 7, code)  # tie 5.0 -> 6
    nib = (code | ((q < 0).to(torch.int32) << 3)).to(torch.uint8)
    nib = nib.reshape(*x.shape[:-1], x.shape[-1])
    packed = nib[..., 0::2] | (nib[..., 1::2] << 4)
    sf = (e.to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)
    return packed.contiguous(), sf.contiguous()


def dequant_mxfp4(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """(packed (..., K//2) uint8, sf (..., K//32) uint8) -> fp32 (..., K)."""
    levels = torch.tensor(E2M1_LEVELS, dtype=torch.float32, device=packed.device)

    def dec(nib: torch.Tensor) -> torch.Tensor:
        code = (nib & 0x7).long()
        sign = ((nib >> 3) & 0x1) != 0
        v = levels[code]
        return torch.where(sign, -v, v)

    lo, hi = dec(packed & 0xF), dec(packed >> 4)
    vals = torch.stack((lo, hi), dim=-1).flatten(-2)
    scale = torch.exp2(sf.float() - 127.0)
    return vals * scale.repeat_interleave(SF_VEC, dim=-1)


def sf_to_atom(sf: torch.Tensor) -> torch.Tensor:
    """(T, K//32) uint8 plain per-token scales -> (T//128, 512) uint8 CUTLASS
    BlockScaledBasicChunk atom layout (per 128-token block):
    byte_off(token, k_group) = (token % 32) * 16 + (token // 32) * 4 + k_group."""
    T, kg = sf.shape
    assert T % 128 == 0 and kg == HEAD_DIM // SF_VEC
    b = sf.reshape(T // 128, 4, 32, kg)  # (blk, m_mid, m_inner, k)
    return b.permute(0, 2, 1, 3).reshape(T // 128, 512).contiguous()


def ref_logits(
    q_packed: torch.Tensor,
    q_sf: torch.Tensor,
    kv_fused: torch.Tensor,
    kv_sf_plain: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    q_len: int = 2,
    heads: int = 64,
) -> torch.Tensor:
    """fp32 ground truth for the paged MQA logits.

    q_packed (B, q_len*heads, 64) uint8, q_sf (B, q_len*heads, 4) uint8 (plain);
    kv_fused (num_pages, 8704) uint8, kv_sf_plain (num_pages, 128, 4) uint8;
    weights (B, q_len*heads) fp32; block_table (B, max_pages) int64/int32.
    Returns logits (B*q_len, max_pages*128) fp32; row = b*q_len + t.
    """
    B, N, _ = q_packed.shape
    q = dequant_mxfp4(q_packed, q_sf)  # (B, N, 128)
    num_pages = kv_fused.shape[0]
    kv_packed = kv_fused[:, :8192].reshape(num_pages * 128, 64)
    kv_all = dequant_mxfp4(kv_packed, kv_sf_plain.reshape(num_pages * 128, 4))
    kv_all = kv_all.reshape(num_pages, 128, HEAD_DIM)
    max_pages = block_table.shape[1]
    out = torch.zeros(B * q_len, max_pages * 128, dtype=torch.float32, device=q.device)
    for b in range(B):
        kv = kv_all[block_table[b].long()]  # (max_pages, 128, 128)
        kv = kv.reshape(max_pages * 128, HEAD_DIM)
        for t in range(q_len):
            rows = slice(t * heads, (t + 1) * heads)
            dots = q[b, rows] @ kv.T  # (heads, S) fp32
            s = (dots.relu() * weights[b, rows].unsqueeze(1)).sum(dim=0)
            out[b * q_len + t] = s
    return out


def make_index_cache(
    B: int,
    max_pages: int,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random paged index-K cache in kernel layout: fused pages
    (num_pages, 8704) uint8 = [128x64B E2M1 data][512B atom-arranged UE8M0 sf].
    Also returns plain per-token sf (num_pages, 128, 4) for the torch reference.
    Pages are allocated per request contiguously (block_table[b] = b*max_pages + i).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    num_pages = B * max_pages
    raw = torch.randn(num_pages * 128, HEAD_DIM, generator=g, device=device)
    packed, sf_plain = quant_mxfp4(raw)
    sf_atom = sf_to_atom(sf_plain)
    fused = torch.cat(
        [packed.reshape(num_pages, 8192), sf_atom], dim=1
    ).contiguous()  # (num_pages, 8704)
    return fused, sf_plain.reshape(num_pages, 128, 4)


def make_q(
    B: int,
    q_len: int = 2,
    heads: int = 64,
    device: str = "cuda",
    seed: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random indexer q: packed (B, N, 64) uint8, sf plain (B, N, 4),
    sf atom-arranged (B, 512), weights (B, N) fp32."""
    g = torch.Generator(device=device).manual_seed(seed)
    N = q_len * heads
    raw = torch.randn(B, N, HEAD_DIM, generator=g, device=device)
    packed, sf_plain = quant_mxfp4(raw)
    sf_atom = torch.stack([sf_to_atom(sf_plain[b]) for b in range(B)]).squeeze(1)
    w = torch.rand(B, N, generator=g, device=device) * 0.1
    return packed, sf_plain, sf_atom, w
