"""Phase 1.3 compressor: torch semantics reference + input synthesis.

Defines the exact model semantics that the CuTe DSL kernels in compress.py
implement (CP-native: partial stats per rank -> push -> owner finalize).
Single-machine full-pooling references are the ground truth; the decomposed
partial/merge path must match them.

Semantics sources: sglang c4_v2.cuh / c128_online_v2.cuh /
fused_norm_rope_v2.cuh + TRT-LLM blog26 (see docs/phase1.3-compressor-tile.md).

Dimensions (V4 Flash, frozen by 1.1/1.2):
  attn stream:  head_dim 512 (448 nope | 64 pre-rope), fp8 e4m3 pool entries
  idx stream:   head_dim 128, MXFP4 68B/entry (64B E2M1 + 4B UE8M0 x4)
  CSA ratio 4:  8-slot window = overlap 4 (role 0) + normal 4 (role 1)
  C128A ratio 128: online per-channel softmax state [max | sum | kv_norm]
"""

from __future__ import annotations

import math

import torch

from mega_dsa_cp.tiles.fp4 import quant_mxfp4, dequant_mxfp4, SF_VEC

HD_ATTN = 512
HD_IDX = 128
ROPE_DIM = 64
RATIO_CSA = 4
RATIO_C128 = 128
# record strides (fp32 elements per token)
REC_ATTN = 4 * HD_ATTN  # [kv_ov | kv_nm | s_ov | s_nm]
REC_IDX = 4 * HD_IDX
REC_C128 = 2 * HD_ATTN  # [kv | score]

NEG_INF = float("-inf")

# fp32(1/sqrt(128)) nearest — hardcoded identically in the DSL kernel so the
# post-hadamard scale multiply is bit-identical on both sides.
HADAMARD_SCALE = 0.0883883461356163


def hadamard128_staged(x: torch.Tensor) -> torch.Tensor:
    """128-point Hadamard replicating the kernel's butterfly stage order:
    in-thread strides 1,2 over 4 contiguous elems/lane, then lane xor masks
    1,2,4,8,16 (element strides 4..64). Same per-element op sequence as the
    DSL kernel (sglang's arrangement), unlike the matmul form. x (..., 128).
    """
    v = x.reshape(*x.shape[:-1], 32, 4)
    a0, a1, a2, a3 = v[..., 0], v[..., 1], v[..., 2], v[..., 3]
    b0, b1, b2, b3 = a0 + a1, a0 - a1, a2 + a3, a2 - a3
    v = torch.stack([b0 + b2, b1 + b3, b0 - b2, b1 - b3], dim=-1)
    idx = torch.arange(32)
    for m in (1, 2, 4, 8, 16):
        other = v[..., idx ^ m, :]
        hi = ((idx & m) != 0).reshape(1, -1, 1)
        v = torch.where(hi, other - v, v + other)
    return (v * HADAMARD_SCALE).reshape(*x.shape)


# ---------------------------------------------------------------------------
# post-processing chain pieces
# ---------------------------------------------------------------------------

def hadamard_matrix(n: int, device=None) -> torch.Tensor:
    """Sylvester-order orthonormal Hadamard: H[i,j] = (-1)^popcount(i&j) / sqrt(n).

    Matches the kernel's stage order (in-thread strides 1,2 + lane butterfly
    strides 4..64): butterflies on distinct bits commute -> natural order.
    """
    h = torch.ones(1, 1, dtype=torch.float32, device=device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x (..., hd) fp32; RMS over full head_dim with learnable weight."""
    factor = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * factor * weight


def rope_last64(x: torch.Tensor, freqs_cis: torch.Tensor, pos: int) -> torch.Tensor:
    """Rotate the last 64 dims at `pos`. freqs_cis (P, 64) fp32 interleaved
    (cos, sin) pairs: freq row = [cos0, sin0, cos1, sin1, ...] (32 complex).
    Pairing is interleaved: (x[2i], x[2i+1]) rotate by theta_i.
    """
    out = x.clone()
    d = x[..., -ROPE_DIM:].float()
    f = freqs_cis[pos].float()  # (64,)
    c, s = f[0::2], f[1::2]  # (32,)
    xr, xi = d[..., 0::2], d[..., 1::2]
    out[..., -ROPE_DIM:][..., 0::2] = xr * c - xi * s
    out[..., -ROPE_DIM:][..., 1::2] = xr * s + xi * c
    return out


def make_freqs_cis(max_pos: int, device=None) -> torch.Tensor:
    """Standard interleaved rope table (P, 64) = [cos, sin] x 32."""
    inv = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, ROPE_DIM, 2, dtype=torch.float32, device=device)
        / ROPE_DIM
    )  # (32,)
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    ang = torch.outer(t, inv)  # (P, 32)
    return torch.stack([ang.cos(), ang.sin()], dim=-1).flatten(-2)  # (P, 64)


# ---------------------------------------------------------------------------
# CSA (ratio=4): full-window pooling reference
# ---------------------------------------------------------------------------

def csa_pool_window(kv: torch.Tensor, score: torch.Tensor, ape: torch.Tensor) -> torch.Tensor:
    """One closing group, single machine. kv/score (8, hd) fp32 (roles already
    selected per slot), ape (8, hd). Returns pooled (hd,) fp32.
    Per-channel softmax over 8 slots: w = softmax(s + ape, dim=slots); out = w.kv
    """
    w = torch.softmax(score + ape, dim=0)  # (8, hd)
    return (w * kv).sum(dim=0)


def csa_partial(kv: torch.Tensor, score: torch.Tensor, ape: torch.Tensor):
    """One rank's partial over its local slots. kv/score (k, hd), ape (k, hd).
    Returns (m, l, w) each (hd,) fp32, UNNORMALIZED w.
    """
    s = score + ape
    m = s.max(dim=0).values
    e = torch.exp(s - m)
    return m, e.sum(dim=0), (e * kv).sum(dim=0)


def csa_merge(partials: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Merge (m, l, w) partials across ranks -> pooled (hd,) fp32."""
    m = torch.stack([p[0] for p in partials]).max(dim=0).values
    num = den = None
    for pm, pl, pw in partials:
        f = torch.exp(pm - m)
        num = pw * f if num is None else num + pw * f
        den = pl * f if den is None else den + pl * f
    return num / den


# ---------------------------------------------------------------------------
# C128A (ratio=128): online state machine reference
# ---------------------------------------------------------------------------

def c128_state_init() -> tuple[None, None, None]:
    return None, None, None  # (m, l, kv_norm) empty


def c128_update(state, kv: torch.Tensor, score: torch.Tensor, bias: torch.Tensor):
    """Fold one token into the running state. kv/score/bias (hd,) fp32.
    state = (m, l, kv_norm) or Nones. Returns new state.
    """
    s = score + bias
    m, l, kvn = state
    if m is None:
        return s.clone(), torch.ones_like(s), kv.clone()
    new_m = torch.maximum(m, s)
    old_l = l * torch.exp(m - new_m)
    new_e = torch.exp(s - new_m)
    new_l = old_l + new_e
    new_kvn = (kvn * old_l + kv * new_e) / new_l
    return new_m, new_l, new_kvn


def c128_merge_states(s0, s1) -> torch.Tensor:
    """Merge two ranks' states at chunk close -> pooled (hd,) fp32."""
    m0, l0, k0 = s0
    m1, l1, k1 = s1
    m = torch.maximum(m0, m1)
    f0, f1 = torch.exp(m0 - m), torch.exp(m1 - m)
    return (k0 * (l0 * f0) + k1 * (l1 * f1)) / (l0 * f0 + l1 * f1)


# ---------------------------------------------------------------------------
# post chains (finalize): norm -> rope -> [hadamard] -> quant
# ---------------------------------------------------------------------------

def post_attn(x: torch.Tensor, norm_w: torch.Tensor, eps: float,
              freqs_cis: torch.Tensor, pos: int, fp8_scale: float) -> torch.Tensor:
    """Attention stream: RMS norm + rope(last 64 @ group start) -> e4m3."""
    y = rope_last64(rms_norm(x, norm_w, eps), freqs_cis, pos)
    return (y / fp8_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def quant_mxfp4_bitexact(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """quant_mxfp4 (fp4.py) but with the UE8M0 exponent computed by integer bit
    inspection instead of torch.ceil(torch.log2(s)) — mathematically identical,
    but deterministic against the DSL kernel's integer path (torch.log2 can
    round across an integer boundary within 1 ulp of a power of two).
    """
    assert x.shape[-1] % SF_VEC == 0
    xb = x.bfloat16().float()
    g = xb.reshape(*xb.shape[:-1], -1, SF_VEC)
    amax = g.abs().amax(dim=-1)
    s = (amax / 6.0).clamp(min=1e-4)
    bits = s.view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = torch.exp2(e.float())  # exact power of two
    q = g / scale.unsqueeze(-1)
    aq = q.abs()
    code = torch.zeros_like(aq, dtype=torch.int32)
    code = torch.where(aq > 0.25, 1, code)
    code = torch.where(aq >= 0.75, 2, code)
    code = torch.where(aq > 1.25, 3, code)
    code = torch.where(aq >= 1.75, 4, code)
    code = torch.where(aq > 2.5, 5, code)
    code = torch.where(aq >= 3.5, 6, code)
    code = torch.where(aq > 5.0, 7, code)
    nib = (code | ((q < 0).to(torch.int32) << 3)).to(torch.uint8)
    nib = nib.reshape(*x.shape[:-1], x.shape[-1])
    packed = nib[..., 0::2] | (nib[..., 1::2] << 4)
    sf = (e + 127).clamp(0, 254).to(torch.uint8)
    return packed.contiguous(), sf.contiguous()


def post_indexer_bitexact(x, norm_w, eps, freqs_cis, pos):
    """post_indexer with staged hadamard + bitexact quant: the reference for
    DSL kernel bit-comparisons (same op order, same integer exponent path)."""
    y = rope_last64(rms_norm(x, norm_w, eps), freqs_cis, pos)
    return quant_mxfp4_bitexact(hadamard128_staged(y))


def post_indexer(x: torch.Tensor, norm_w: torch.Tensor, eps: float,
                 freqs_cis: torch.Tensor, pos: int):
    """Indexer stream: RMS norm + rope(last 64) -> Hadamard128 -> MXFP4.

    Returns (packed (64,) uint8, sf (4,) uint8) per 128-dim entry.
    """
    y = rope_last64(rms_norm(x, norm_w, eps), freqs_cis, pos)
    h = hadamard_matrix(HD_IDX, device=x.device)
    y = y @ h  # H symmetric: (Hx) = x @ H
    return quant_mxfp4(y)


# ---------------------------------------------------------------------------
# input synthesis for tests (deterministic per (b, token pos, stream))
# ---------------------------------------------------------------------------

def synth_record(b: int, pos: int, hd: int, dual_role: bool, seed: int = 0) -> torch.Tensor:
    """One token's compressor record. dual_role=True (CSA): 4*hd
    [kv_ov|kv_nm|s_ov|s_nm]; False (C128A): 2*hd [kv|score]. Deterministic."""
    g = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + b * 65_537 + pos * 257 + hd)
    n = (4 if dual_role else 2) * hd
    rec = torch.randn(n, generator=g, dtype=torch.float32)
    # scores (latter half of each role pair) get a wider range to exercise softmax
    if dual_role:
        rec[2 * hd : 3 * hd] *= 3.0
        rec[3 * hd :] *= 3.0
    else:
        rec[hd:] *= 3.0
    return rec


def synth_step_batch(B: int, positions: list[int], hd: int, dual_role: bool, seed: int = 0):
    """Records for one token per sequence: (B, rec_len)."""
    return torch.stack([synth_record(b, positions[b], hd, dual_role, seed) for b in range(B)])


def synth_ape(hd: int, slots: int, seed: int = 1234) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed + hd + slots)
    return torch.randn(slots, hd, generator=g, dtype=torch.float32) * 0.5
