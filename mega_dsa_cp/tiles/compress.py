"""Phase 1.3 compressor tiles: CuTe DSL kernels (CP-native, cp=2, q_len=2).

Per decode step, per layer:
  csa_step:     ring-append this rank's new local token records (attn + idx
                streams); if the token closes a 4-token group, compute
                per-channel partial stats (m, l, w) over this rank's local
                window slots (8-slot window = overlap role 0 + normal role 1,
                first group has no overlap), push stats to the entry owner
                rank (owner = entry_id % CP), notify both ranks' cell[b].
  csa_finalize: wait cell[b] (arity CP); if a group closed and this rank owns
                the entry: merge partials, post-chain (RMS norm -> rope last 64
                @ group start 4n -> fp8 e4m3 attn / Hadamard128 + MXFP4 idx),
                write pools at local slot n//CP, bump k_valid.
  c128_step / c128_finalize: same shape for the C128A online state machine
                (ratio 128, single attn stream, state = [max|sum|kv_norm]).

All shapes static (S-bucket contract); seq_len is a device scalar. No dynamic
tensor dims (gotcha #33); no Python ternaries / dynamic-if captures — all
predication is branch-free arithmetic (gotcha #32). See
docs/phase1.3-compressor-tile.md.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8, Uint64
import cuda.bindings.driver as cuda_drv
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from mega_dsa_cp.events.core import EventSet, SCOPE_SYS
from mega_dsa_cp.comm.device import PeerBuffer
from mega_dsa_cp.comm.primitives import push_start, push_finish
from mega_dsa_cp.tiles.logits import sel_f32, sel_i32, f32_as_i32

CP = 2
N_THREADS = 128
HD_ATTN = 512
HD_IDX = 128
ROPE_DIM = 64
REC_ATTN = 4 * HD_ATTN  # [kv_ov | kv_nm | s_ov | s_nm]
REC_IDX = 4 * HD_IDX
REC_C128 = 2 * HD_ATTN  # [kv | score]
RING = 4  # local ring depth (CSA window holds exactly 4 local slots at cp=2)

# CSA stats region layout (fp32 slots), per (b, src_rank)
S_VALID = 0
S_ENTRY = 1
S_MA = 4
S_LA = S_MA + HD_ATTN
S_WA = S_LA + HD_ATTN
S_MI = S_WA + HD_ATTN
S_LI = S_MI + HD_IDX
S_WI = S_LI + HD_IDX
CSA_STATS_F32 = S_WI + HD_IDX  # 1924
CSA_STATS_BYTES = CSA_STATS_F32 * 4  # 7696 (16B multiple)

# C128 stats region layout, per (b, src_rank)
C_M = 4
C_L = C_M + HD_ATTN
C_K = C_L + HD_ATTN
C128_STATS_F32 = C_K + HD_ATTN  # 1540
C128_STATS_BYTES = C128_STATS_F32 * 4  # 6160

# index-K fused page: 128 entries x 64B packed + 512B sf atom
IDXK_PAGE_BYTES = 8704

NEG_BIG = -1.0e30
HADAMARD_SCALE = 0.0883883461356163  # fp32(1/sqrt(128)); compress_ref same


@dsl_user_op
def i32_as_f32(x: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.bitcast(
            Float32.mlir_type, Int32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
    )


@dsl_user_op
def cvt_e4m3x2(hi: Float32, lo: Float32, *, loc=None, ip=None) -> Int32:
    """Two fp32 -> packed e4m3 pair (hardware RNE satfinite): returns
    (e4m3(hi) << 8) | e4m3(lo) in the low 16 bits of an i32."""
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [Float32(hi).ir_value(loc=loc, ip=ip),
             Float32(lo).ir_value(loc=loc, ip=ip)],
            """{
            .reg .b16 h;
            cvt.rn.satfinite.e4m3x2.f32 h, $1, $2;
            cvt.u32.u16 $0, h;
            }""",
            "=r,f,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


def _e2m1_nib(q: Float32) -> Int32:
    """fp32 scaled value -> 4-bit E2M1 code (DeepGEMM thresholds, fp4.py)."""
    aq = cutlass.max(q, -q)
    code = Int32(0)
    code = code + sel_i32(aq > 0.25, Int32(1), Int32(0))
    code = code + sel_i32(aq >= 0.75, Int32(1), Int32(0))
    code = code + sel_i32(aq > 1.25, Int32(1), Int32(0))
    code = code + sel_i32(aq >= 1.75, Int32(1), Int32(0))
    code = code + sel_i32(aq > 2.5, Int32(1), Int32(0))
    code = code + sel_i32(aq >= 3.5, Int32(1), Int32(0))
    code = code + sel_i32(aq > 5.0, Int32(1), Int32(0))
    return code | sel_i32(q < 0.0, Int32(8), Int32(0))


# ---------------------------------------------------------------------------
# CSA step: ring append + partial stats + push + notify
# ---------------------------------------------------------------------------

@cute.kernel
def _csa_step_kernel(
    ring_attn: cute.Pointer,  # f32 (B, RING, REC_ATTN)
    ring_idx: cute.Pointer,  # f32 (B, RING, REC_IDX)
    new_attn: cute.Pointer,  # f32 (B, REC_ATTN) this rank's new local token
    new_idx: cute.Pointer,  # f32 (B, REC_IDX)
    ape_attn: cute.Pointer,  # f32 (8, HD_ATTN)
    ape_idx: cute.Pointer,  # f32 (8, HD_IDX)
    seq_len: cute.Pointer,  # i32 (B,) pre-step
    pay_base: cute.Pointer,
    pay_offsets: cute.Pointer,
    mc_base: Int64,
    stats_off: Int64,
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    do_push: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    b, _, _ = cute.arch.block_idx()
    events = EventSet(base=ev_base, offsets=ev_offsets, my_rank=my_rank)
    pbuf = PeerBuffer(pay_base, pay_offsets, my_rank, mc_base)

    s = cute.arch.load(seq_len + b, Int32)
    pos = s + ((my_rank - s) & 1)  # my local token position (q_len=2)
    lt = pos >> 1  # local token index (pos // CP)
    slot = lt & (RING - 1)

    # 1. append records into the local ring
    for i in cutlass.range_constexpr(REC_ATTN // N_THREADS):
        o = i * N_THREADS + tidx
        v = cute.arch.load(new_attn + b * REC_ATTN + o, Float32)
        cute.arch.store(ring_attn + (b * RING + slot) * REC_ATTN + o, v)
    for i in cutlass.range_constexpr(REC_IDX // N_THREADS):
        o = i * N_THREADS + tidx
        v = cute.arch.load(new_idx + b * REC_IDX + o, Float32)
        cute.arch.store(ring_idx + (b * RING + slot) * REC_IDX + o, v)
    cute.arch.barrier()

    closed = ((pos + 1) & 3) == 0
    n = pos >> 2  # closing group index (meaningful only when closed)

    smem = cutlass.utils.SmemAllocator()
    s_stats = smem.allocate_tensor(Float32, CSA_STATS_F32, byte_alignment=16)

    # 2. partial stats over my 4 local window slots (branch-free predication)
    m_a = [Float32(NEG_BIG)] * 4
    l_a = [Float32(0.0)] * 4
    w_a = [Float32(0.0)] * 4
    m_i = Float32(NEG_BIG)
    l_i = Float32(0.0)
    w_i = Float32(0.0)
    s_regs = [[Float32(0.0) for _ in range(4)] for _ in range(4)]
    si_regs = [Float32(0.0) for _ in range(4)]
    for jj in cutlass.range_constexpr(4):
        j = my_rank + 2 * (jj % 2) + 4 * (jj // 2)  # window slot (dynamic)
        # first group has no overlap: slots j<4 invalid -> subtract 1e30
        bad = sel_f32((n > 0) | (j >= 4), Float32(0.0), Float32(1.0))
        t_j = 4 * n - 4 + j
        rslot = (t_j >> 1) & (RING - 1)
        role = j >> 2
        ra = ring_attn + (b * RING + rslot) * REC_ATTN
        ri = ring_idx + (b * RING + rslot) * REC_IDX
        for k in cutlass.range_constexpr(4):
            ch = tidx + N_THREADS * k
            sc = cute.arch.load(ra + 2 * HD_ATTN + role * HD_ATTN + ch, Float32)
            sc = sc + cute.arch.load(ape_attn + j * HD_ATTN + ch, Float32)
            sc = sc - bad * Float32(-NEG_BIG)
            s_regs[jj][k] = sc
            m_a[k] = cutlass.max(m_a[k], sc)
        sci = cute.arch.load(ri + 2 * HD_IDX + role * HD_IDX + tidx, Float32)
        sci = sci + cute.arch.load(ape_idx + j * HD_IDX + tidx, Float32)
        sci = sci - bad * Float32(-NEG_BIG)
        si_regs[jj] = sci
        m_i = cutlass.max(m_i, sci)
    for jj in cutlass.range_constexpr(4):
        j = my_rank + 2 * (jj % 2) + 4 * (jj // 2)
        t_j = 4 * n - 4 + j
        rslot = (t_j >> 1) & (RING - 1)
        role = j >> 2
        ra = ring_attn + (b * RING + rslot) * REC_ATTN
        ri = ring_idx + (b * RING + rslot) * REC_IDX
        for k in cutlass.range_constexpr(4):
            ch = tidx + N_THREADS * k
            e = cute.math.exp(s_regs[jj][k] - m_a[k])
            kv = cute.arch.load(ra + role * HD_ATTN + ch, Float32)
            l_a[k] = l_a[k] + e
            w_a[k] = w_a[k] + e * kv
        ei = cute.math.exp(si_regs[jj] - m_i)
        kvi = cute.arch.load(ri + role * HD_IDX + tidx, Float32)
        l_i = l_i + ei
        w_i = w_i + ei * kvi

    # 3. stage to smem
    if tidx == 0:
        s_stats[S_VALID] = sel_f32(closed, Float32(1.0), Float32(0.0))
        s_stats[S_ENTRY] = Float32(n)
    for k in cutlass.range_constexpr(4):
        ch = tidx + N_THREADS * k
        s_stats[S_MA + ch] = m_a[k]
        s_stats[S_LA + ch] = l_a[k]
        s_stats[S_WA + ch] = w_a[k]
    s_stats[S_MI + tidx] = m_i
    s_stats[S_LI + tidx] = l_i
    s_stats[S_WI + tidx] = w_i
    cute.arch.barrier()
    cute.arch.fence_view_async_shared()

    # 4. push payload to the owner rank (only on close), notify both cells
    if tidx == 0:
        if closed:
            if do_push != 0:
                owner = n & (CP - 1)
                dst = pbuf.peer_ptr(
                    stats_off + (b * CP + my_rank) * CSA_STATS_BYTES,
                    owner,
                    Uint8,
                    align=16,
                )
                push_start(dst, s_stats.iterator, Int32(CSA_STATS_BYTES))
                push_finish()
        events.notify(Int32(b), Int32(0))
        events.notify(Int32(b), Int32(1))


@cute.jit
def launch_csa_step(
    b_count: cutlass.Constexpr,
    ring_attn: cute.Pointer,
    ring_idx: cute.Pointer,
    new_attn: cute.Pointer,
    new_idx: cute.Pointer,
    ape_attn: cute.Pointer,
    ape_idx: cute.Pointer,
    seq_len: cute.Pointer,
    pay_base: cute.Pointer,
    pay_offsets: cute.Pointer,
    mc_base: Int64,
    stats_off: Int64,
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    do_push: Int32,
    stream: cuda_drv.CUstream,
):
    _csa_step_kernel(
        ring_attn, ring_idx, new_attn, new_idx, ape_attn, ape_idx, seq_len,
        pay_base, pay_offsets, mc_base, stats_off, ev_base, ev_offsets,
        my_rank, do_push,
    ).launch(grid=[b_count, 1, 1], block=[N_THREADS, 1, 1], stream=stream)


# ---------------------------------------------------------------------------
# CSA finalize: wait + merge + post-chain + pool writes
# ---------------------------------------------------------------------------

@cute.kernel
def _csa_finalize_kernel(
    seq_len: cute.Pointer,  # i32 (B,) pre-step
    pay_base: cute.Pointer,
    pay_offsets: cute.Pointer,
    mc_base: Int64,
    stats_off: Int64,
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    phase: Int32,
    norm_w_attn: cute.Pointer,  # f32 (HD_ATTN,)
    norm_w_idx: cute.Pointer,  # f32 (HD_IDX,)
    norm_eps: Float32,
    freqs: cute.Pointer,  # f32 (P, ROPE_DIM)
    inv_fp8_scale: Float32,
    cmp_pool: cute.Pointer,  # u8 storage (B, MAXE, HD_ATTN) e4m3 bytes
    idxk_pool: cute.Pointer,  # u8 (B, MAXE//128, IDXK_PAGE_BYTES)
    k_valid: cute.Pointer,  # i32 (B,)
    dbg_attn: cute.Pointer,  # f32 (B, MAXE, HD_ATTN) merged fp32 (pre-norm)
    dbg_idx: cute.Pointer,  # f32 (B, MAXE, HD_IDX)
    max_e: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    b, _, _ = cute.arch.block_idx()
    events = EventSet(base=ev_base, offsets=ev_offsets, my_rank=my_rank)
    pbuf = PeerBuffer(pay_base, pay_offsets, my_rank, mc_base)

    smem = cutlass.utils.SmemAllocator()
    s_y = smem.allocate_tensor(Float32, HD_ATTN, byte_alignment=16)
    s_red = smem.allocate_tensor(Float32, 8, byte_alignment=16)

    events.wait(
        Int32(b), Uint64(Int64(CP) * (Int64(phase) + 1)), SCOPE_SYS,
        fence_after=False,
    )

    s = cute.arch.load(seq_len + b, Int32)
    # q_len=2: closing position p (if any) among {s, s+1}
    c0 = ((s + 1) & 3) == 0
    c1 = ((s + 2) & 3) == 0
    closed = c0 | c1
    p = s + sel_i32(c0, Int32(0), Int32(1))
    n = p >> 2
    owner = n & (CP - 1)

    if closed:
        if owner == my_rank:
            src0 = pbuf.peer_ptr(
                stats_off + (b * CP) * CSA_STATS_BYTES, my_rank, Float32,
                align=16,
            )
            src1 = src0 + CSA_STATS_F32

            slot = n >> 1  # local entry slot (n // CP)
            pos4n = 4 * n  # rope position = group start

            # ---- merge attn stream (4 channels per thread) ----
            out = [Float32(0.0)] * 4
            ssq = Float32(0.0)
            for k in cutlass.range_constexpr(4):
                ch = tidx + N_THREADS * k
                m0 = cute.arch.load(src0 + S_MA + ch, Float32, sem="acquire",
                                    scope="sys")
                m1 = cute.arch.load(src1 + S_MA + ch, Float32, sem="acquire",
                                    scope="sys")
                m = cutlass.max(m0, m1)
                f0 = cute.math.exp(m0 - m)
                f1 = cute.math.exp(m1 - m)
                l0 = cute.arch.load(src0 + S_LA + ch, Float32, sem="acquire",
                                    scope="sys")
                l1 = cute.arch.load(src1 + S_LA + ch, Float32, sem="acquire",
                                    scope="sys")
                w0 = cute.arch.load(src0 + S_WA + ch, Float32, sem="acquire",
                                    scope="sys")
                w1 = cute.arch.load(src1 + S_WA + ch, Float32, sem="acquire",
                                    scope="sys")
                o = (w0 * f0 + w1 * f1) / (l0 * f0 + l1 * f1)
                out[k] = o
                ssq = ssq + o * o
                cute.arch.store(dbg_attn + (b * max_e + slot) * HD_ATTN + ch, o)
            for off in (16, 8, 4, 2, 1):
                ssq = ssq + cute.arch.shuffle_sync_bfly(ssq, offset=off)
            if (tidx & 31) == 0:
                s_red[tidx >> 5] = ssq
            cute.arch.barrier()
            if tidx < 4:
                v = s_red[tidx]
                for off in (2, 1):
                    v = v + cute.arch.shuffle_sync_bfly(v, offset=off)
                if tidx == 0:
                    s_red[0] = v
            cute.arch.barrier()
            factor = cute.math.rsqrt(
                s_red[0] / Float32(HD_ATTN) + norm_eps
            )
            for k in cutlass.range_constexpr(4):
                ch = tidx + N_THREADS * k
                s_y[ch] = out[k] * factor * cute.arch.load(norm_w_attn + ch,
                                                           Float32)
            cute.arch.barrier()
            for k in cutlass.range_constexpr(4):
                ch = tidx + N_THREADS * k
                if ch >= HD_ATTN - ROPE_DIM:
                    y = s_y[ch]
                    partner = s_y[ch ^ 1]
                    i2 = ch - (HD_ATTN - ROPE_DIM)
                    cc = cute.arch.load(freqs + pos4n * ROPE_DIM + (i2 & ~1),
                                        Float32)
                    th = cute.arch.load(freqs + pos4n * ROPE_DIM + (i2 | 1),
                                        Float32)
                    sgn = Float32(2 * (ch & 1) - 1)
                    s_y[ch] = y * cc + partner * (sgn * th)
            cute.arch.barrier()
            base_ch = 4 * tidx  # 4 consecutive channels per thread
            for k in cutlass.range_constexpr(2):
                ch = base_ch + 2 * k
                v0 = s_y[ch] * inv_fp8_scale
                v1 = s_y[ch + 1] * inv_fp8_scale
                v0 = cutlass.min(cutlass.max(v0, Float32(-448.0)),
                                 Float32(448.0))
                v1 = cutlass.min(cutlass.max(v1, Float32(-448.0)),
                                 Float32(448.0))
                pk = cvt_e4m3x2(v1, v0)
                poff = (b * max_e + slot) * HD_ATTN + ch
                cute.arch.store(cmp_pool + poff, Uint8(pk & 255))
                cute.arch.store(cmp_pool + poff + 1, Uint8((pk >> 8) & 255))

            # ---- merge idx stream (1 channel per thread) ----
            ch = tidx
            m0 = cute.arch.load(src0 + S_MI + ch, Float32, sem="acquire",
                                scope="sys")
            m1 = cute.arch.load(src1 + S_MI + ch, Float32, sem="acquire",
                                scope="sys")
            m = cutlass.max(m0, m1)
            f0 = cute.math.exp(m0 - m)
            f1 = cute.math.exp(m1 - m)
            l0 = cute.arch.load(src0 + S_LI + ch, Float32, sem="acquire",
                                scope="sys")
            l1 = cute.arch.load(src1 + S_LI + ch, Float32, sem="acquire",
                                scope="sys")
            w0 = cute.arch.load(src0 + S_WI + ch, Float32, sem="acquire",
                                scope="sys")
            w1 = cute.arch.load(src1 + S_WI + ch, Float32, sem="acquire",
                                scope="sys")
            oi = (w0 * f0 + w1 * f1) / (l0 * f0 + l1 * f1)
            cute.arch.store(dbg_idx + (b * max_e + slot) * HD_IDX + ch, oi)
            ssqi = oi * oi
            for off in (16, 8, 4, 2, 1):
                ssqi = ssqi + cute.arch.shuffle_sync_bfly(ssqi, offset=off)
            if (tidx & 31) == 0:
                s_red[tidx >> 5] = ssqi
            cute.arch.barrier()
            if tidx < 4:
                v = s_red[tidx]
                for off in (2, 1):
                    v = v + cute.arch.shuffle_sync_bfly(v, offset=off)
                if tidx == 0:
                    s_red[0] = v
            cute.arch.barrier()
            factori = cute.math.rsqrt(s_red[0] / Float32(HD_IDX) + norm_eps)
            yi = oi * factori * cute.arch.load(norm_w_idx + ch, Float32)
            s_y[ch] = yi
            cute.arch.barrier()
            if ch >= HD_IDX - ROPE_DIM:
                y = s_y[ch]
                partner = s_y[ch ^ 1]
                i2 = ch - (HD_IDX - ROPE_DIM)
                cc = cute.arch.load(freqs + pos4n * ROPE_DIM + (i2 & ~1),
                                    Float32)
                th = cute.arch.load(freqs + pos4n * ROPE_DIM + (i2 | 1),
                                    Float32)
                sgn = Float32(2 * (ch & 1) - 1)
                s_y[ch] = y * cc + partner * (sgn * th)
            cute.arch.barrier()
            # Hadamard128 + MXFP4 quant on warp 0 (4 contiguous elems/lane)
            if tidx < 32:
                x0 = s_y[4 * tidx]
                x1 = s_y[4 * tidx + 1]
                x2 = s_y[4 * tidx + 2]
                x3 = s_y[4 * tidx + 3]
                b0, b1 = x0 + x1, x0 - x1
                b2, b3 = x2 + x3, x2 - x3
                d0, d1 = b0 + b2, b1 + b3
                d2, d3 = b0 - b2, b1 - b3
                for msk in (1, 2, 4, 8, 16):
                    o0 = cute.arch.shuffle_sync_bfly(d0, offset=msk)
                    o1 = cute.arch.shuffle_sync_bfly(d1, offset=msk)
                    o2 = cute.arch.shuffle_sync_bfly(d2, offset=msk)
                    o3 = cute.arch.shuffle_sync_bfly(d3, offset=msk)
                    hi = (tidx & msk) != 0
                    d0 = sel_f32(hi, o0 - d0, d0 + o0)
                    d1 = sel_f32(hi, o1 - d1, d1 + o1)
                    d2 = sel_f32(hi, o2 - d2, d2 + o2)
                    d3 = sel_f32(hi, o3 - d3, d3 + o3)
                hs = Float32(HADAMARD_SCALE)
                q0 = Float32((d0 * hs).to(cutlass.BFloat16))
                q1 = Float32((d1 * hs).to(cutlass.BFloat16))
                q2 = Float32((d2 * hs).to(cutlass.BFloat16))
                q3 = Float32((d3 * hs).to(cutlass.BFloat16))
                amax = cutlass.max(
                    cutlass.max(cutlass.max(q0, -q0), cutlass.max(q1, -q1)),
                    cutlass.max(cutlass.max(q2, -q2), cutlass.max(q3, -q3)),
                )
                for msk in (1, 2, 4):
                    amax = cutlass.max(
                        amax, cute.arch.shuffle_sync_bfly(amax, offset=msk)
                    )
                sc = cutlass.max(amax / Float32(6.0), Float32(1e-4))
                bits = f32_as_i32(sc)
                e = ((bits >> 23) & 255) - 127 + sel_i32(
                    (bits & 8388607) != 0, Int32(1), Int32(0)
                )
                scale = i32_as_f32((e + 127) << 23)
                sf_byte = cutlass.min(cutlass.max(e + 127, Int32(0)),
                                      Int32(254))
                nib0 = _e2m1_nib(q0 / scale)
                nib1 = _e2m1_nib(q1 / scale)
                nib2 = _e2m1_nib(q2 / scale)
                nib3 = _e2m1_nib(q3 / scale)
                pg = (b * (max_e // 128) + (slot >> 7)) * IDXK_PAGE_BYTES
                tok = slot & 127
                voff = pg + tok * 64 + tidx * 2
                cute.arch.store(idxk_pool + voff, Uint8(nib0 | (nib1 << 4)))
                cute.arch.store(idxk_pool + voff + 1,
                                Uint8(nib2 | (nib3 << 4)))
                if (tidx & 7) == 0:
                    soff = (
                        pg + 8192 + (tok % 32) * 16 + (tok >> 5) * 4
                        + (tidx >> 3)
                    )
                    cute.arch.store(idxk_pool + soff, Uint8(sf_byte))

            if tidx == 0:
                cute.arch.store(k_valid + b, slot + 1)


@cute.jit
def launch_csa_finalize(
    b_count: cutlass.Constexpr,
    max_e: cutlass.Constexpr,
    seq_len: cute.Pointer,
    pay_base: cute.Pointer,
    pay_offsets: cute.Pointer,
    mc_base: Int64,
    stats_off: Int64,
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    phase: Int32,
    norm_w_attn: cute.Pointer,
    norm_w_idx: cute.Pointer,
    norm_eps: Float32,
    freqs: cute.Pointer,
    inv_fp8_scale: Float32,
    cmp_pool: cute.Pointer,
    idxk_pool: cute.Pointer,
    k_valid: cute.Pointer,
    dbg_attn: cute.Pointer,
    dbg_idx: cute.Pointer,
    stream: cuda_drv.CUstream,
):
    _csa_finalize_kernel(
        seq_len, pay_base, pay_offsets, mc_base, stats_off, ev_base,
        ev_offsets, my_rank, phase, norm_w_attn, norm_w_idx, norm_eps, freqs,
        inv_fp8_scale, cmp_pool, idxk_pool, k_valid, dbg_attn, dbg_idx, max_e,
    ).launch(grid=[b_count, 1, 1], block=[N_THREADS, 1, 1], stream=stream)


# ---------------------------------------------------------------------------
# C128A step: fold local token into online state; push state on chunk close
# ---------------------------------------------------------------------------

@cute.kernel
def _c128_step_kernel(
    state: cute.Pointer,  # f32 (B, 3, HD_ATTN): [m | l | kv_norm]
    new_c128: cute.Pointer,  # f32 (B, REC_C128)
    ape128: cute.Pointer,  # f32 (128, HD_ATTN)
    seq_len: cute.Pointer,  # i32 (B,) pre-step
    pay_base: cute.Pointer,
    pay_offsets: cute.Pointer,
    mc_base: Int64,
    stats_off: Int64,
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    b_count: Int32,
):
    tidx, _, _ = cute.arch.thread_idx()
    b, _, _ = cute.arch.block_idx()
    events = EventSet(base=ev_base, offsets=ev_offsets, my_rank=my_rank)
    pbuf = PeerBuffer(pay_base, pay_offsets, my_rank, mc_base)

    smem = cutlass.utils.SmemAllocator()
    s_stats = smem.allocate_tensor(Float32, C128_STATS_F32, byte_alignment=16)

    s = cute.arch.load(seq_len + b, Int32)
    pos = s + ((my_rank - s) & 1)
    pic = pos & 127  # position in chunk
    n128 = pos >> 7
    closed = pic == 127

    st_m = state + (b * 3 + 0) * HD_ATTN
    st_l = state + (b * 3 + 1) * HD_ATTN
    st_k = state + (b * 3 + 2) * HD_ATTN

    for k in cutlass.range_constexpr(4):
        ch = tidx + N_THREADS * k
        m_old = cute.arch.load(st_m + ch, Float32)
        l_old = cute.arch.load(st_l + ch, Float32)
        k_old = cute.arch.load(st_k + ch, Float32)
        kv = cute.arch.load(new_c128 + b * REC_C128 + ch, Float32)
        sc = cute.arch.load(new_c128 + b * REC_C128 + HD_ATTN + ch, Float32)
        sc = sc + cute.arch.load(ape128 + pic * HD_ATTN + ch, Float32)
        new_m = cutlass.max(m_old, sc)
        f_old = cute.math.exp(m_old - new_m)
        e_new = cute.math.exp(sc - new_m)
        l_new = l_old * f_old + e_new
        k_new = (k_old * (l_old * f_old) + kv * e_new) / l_new
        # stage pushed state (post-fold), then store or reset
        s_stats[C_M + ch] = new_m
        s_stats[C_L + ch] = l_new
        s_stats[C_K + ch] = k_new
        rst = closed
        cute.arch.store(st_m + ch, sel_f32(rst, Float32(NEG_BIG), new_m))
        cute.arch.store(st_l + ch, sel_f32(rst, Float32(0.0), l_new))
        cute.arch.store(st_k + ch, sel_f32(rst, Float32(0.0), k_new))
    if tidx == 0:
        s_stats[S_VALID] = sel_f32(closed, Float32(1.0), Float32(0.0))
        s_stats[S_ENTRY] = Float32(n128)
    cute.arch.barrier()
    cute.arch.fence_view_async_shared()

    if tidx == 0:
        if closed:
            owner = n128 & (CP - 1)
            dst = pbuf.peer_ptr(
                stats_off + (b * CP + my_rank) * C128_STATS_BYTES,
                owner,
                Uint8,
                align=16,
            )
            push_start(dst, s_stats.iterator, Int32(C128_STATS_BYTES))
            push_finish()
        events.notify(Int32(b_count + b), Int32(0))
        events.notify(Int32(b_count + b), Int32(1))


@cute.jit
def launch_c128_step(
    b_count: cutlass.Constexpr,
    state: cute.Pointer,
    new_c128: cute.Pointer,
    ape128: cute.Pointer,
    seq_len: cute.Pointer,
    pay_base: cute.Pointer,
    pay_offsets: cute.Pointer,
    mc_base: Int64,
    stats_off: Int64,
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    stream: cuda_drv.CUstream,
):
    _c128_step_kernel(
        state, new_c128, ape128, seq_len, pay_base, pay_offsets, mc_base,
        stats_off, ev_base, ev_offsets, my_rank, Int32(b_count),
    ).launch(grid=[b_count, 1, 1], block=[N_THREADS, 1, 1], stream=stream)


# ---------------------------------------------------------------------------
# C128A finalize: wait + merge two ranks' states + post-chain + pool write
# ---------------------------------------------------------------------------

@cute.kernel
def _c128_finalize_kernel(
    seq_len: cute.Pointer,  # i32 (B,) pre-step
    pay_base: cute.Pointer,
    pay_offsets: cute.Pointer,
    mc_base: Int64,
    stats_off: Int64,
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    phase: Int32,
    norm_w: cute.Pointer,  # f32 (HD_ATTN,)
    norm_eps: Float32,
    freqs: cute.Pointer,  # f32 (P, ROPE_DIM)
    inv_fp8_scale: Float32,
    c128_pool: cute.Pointer,  # u8 storage (B, MAXE128, HD_ATTN) e4m3 bytes
    k_valid: cute.Pointer,  # i32 (B,)
    dbg: cute.Pointer,  # f32 (B, MAXE128, HD_ATTN) merged fp32
    b_count: Int32,
    max_e: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    b, _, _ = cute.arch.block_idx()
    events = EventSet(base=ev_base, offsets=ev_offsets, my_rank=my_rank)
    pbuf = PeerBuffer(pay_base, pay_offsets, my_rank, mc_base)

    smem = cutlass.utils.SmemAllocator()
    s_y = smem.allocate_tensor(Float32, HD_ATTN, byte_alignment=16)
    s_red = smem.allocate_tensor(Float32, 8, byte_alignment=16)

    events.wait(
        Int32(b_count + b), Uint64(Int64(CP) * (Int64(phase) + 1)), SCOPE_SYS,
        fence_after=False,
    )

    s = cute.arch.load(seq_len + b, Int32)
    c0 = ((s + 1) & 127) == 0
    c1 = ((s + 2) & 127) == 0
    closed = c0 | c1
    p = s + sel_i32(c0, Int32(0), Int32(1))
    n128 = p >> 7
    owner = n128 & (CP - 1)

    if closed:
        if owner == my_rank:
            src0 = pbuf.peer_ptr(
                stats_off + (b * CP) * C128_STATS_BYTES, my_rank, Float32,
                align=16,
            )
            src1 = src0 + C128_STATS_F32
            slot = n128 >> 1
            pos128 = 128 * n128

            out = [Float32(0.0)] * 4
            ssq = Float32(0.0)
            for k in cutlass.range_constexpr(4):
                ch = tidx + N_THREADS * k
                m0 = cute.arch.load(src0 + C_M + ch, Float32, sem="acquire",
                                    scope="sys")
                m1 = cute.arch.load(src1 + C_M + ch, Float32, sem="acquire",
                                    scope="sys")
                m = cutlass.max(m0, m1)
                f0 = cute.math.exp(m0 - m)
                f1 = cute.math.exp(m1 - m)
                l0 = cute.arch.load(src0 + C_L + ch, Float32, sem="acquire",
                                    scope="sys")
                l1 = cute.arch.load(src1 + C_L + ch, Float32, sem="acquire",
                                    scope="sys")
                k0 = cute.arch.load(src0 + C_K + ch, Float32, sem="acquire",
                                    scope="sys")
                k1 = cute.arch.load(src1 + C_K + ch, Float32, sem="acquire",
                                    scope="sys")
                o = (k0 * (l0 * f0) + k1 * (l1 * f1)) / (l0 * f0 + l1 * f1)
                out[k] = o
                ssq = ssq + o * o
                cute.arch.store(dbg + (b * max_e + slot) * HD_ATTN + ch, o)
            for off in (16, 8, 4, 2, 1):
                ssq = ssq + cute.arch.shuffle_sync_bfly(ssq, offset=off)
            if (tidx & 31) == 0:
                s_red[tidx >> 5] = ssq
            cute.arch.barrier()
            if tidx < 4:
                v = s_red[tidx]
                for off in (2, 1):
                    v = v + cute.arch.shuffle_sync_bfly(v, offset=off)
                if tidx == 0:
                    s_red[0] = v
            cute.arch.barrier()
            factor = cute.math.rsqrt(s_red[0] / Float32(HD_ATTN) + norm_eps)
            for k in cutlass.range_constexpr(4):
                ch = tidx + N_THREADS * k
                s_y[ch] = out[k] * factor * cute.arch.load(norm_w + ch,
                                                           Float32)
            cute.arch.barrier()
            for k in cutlass.range_constexpr(4):
                ch = tidx + N_THREADS * k
                if ch >= HD_ATTN - ROPE_DIM:
                    y = s_y[ch]
                    partner = s_y[ch ^ 1]
                    i2 = ch - (HD_ATTN - ROPE_DIM)
                    cc = cute.arch.load(freqs + pos128 * ROPE_DIM + (i2 & ~1),
                                        Float32)
                    th = cute.arch.load(freqs + pos128 * ROPE_DIM + (i2 | 1),
                                        Float32)
                    sgn = Float32(2 * (ch & 1) - 1)
                    s_y[ch] = y * cc + partner * (sgn * th)
            cute.arch.barrier()
            base_ch = 4 * tidx  # 4 consecutive channels per thread
            for k in cutlass.range_constexpr(2):
                ch = base_ch + 2 * k
                v0 = s_y[ch] * inv_fp8_scale
                v1 = s_y[ch + 1] * inv_fp8_scale
                v0 = cutlass.min(cutlass.max(v0, Float32(-448.0)),
                                 Float32(448.0))
                v1 = cutlass.min(cutlass.max(v1, Float32(-448.0)),
                                 Float32(448.0))
                pk = cvt_e4m3x2(v1, v0)
                poff = (b * max_e + slot) * HD_ATTN + ch
                cute.arch.store(c128_pool + poff, Uint8(pk & 255))
                cute.arch.store(c128_pool + poff + 1, Uint8((pk >> 8) & 255))
            if tidx == 0:
                cute.arch.store(k_valid + b, slot + 1)


@cute.jit
def launch_c128_finalize(
    b_count: cutlass.Constexpr,
    max_e: cutlass.Constexpr,
    seq_len: cute.Pointer,
    pay_base: cute.Pointer,
    pay_offsets: cute.Pointer,
    mc_base: Int64,
    stats_off: Int64,
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    my_rank: Int32,
    phase: Int32,
    norm_w: cute.Pointer,
    norm_eps: Float32,
    freqs: cute.Pointer,
    inv_fp8_scale: Float32,
    c128_pool: cute.Pointer,
    k_valid: cute.Pointer,
    dbg: cute.Pointer,
    stream: cuda_drv.CUstream,
):
    _c128_finalize_kernel(
        seq_len, pay_base, pay_offsets, mc_base, stats_off, ev_base,
        ev_offsets, my_rank, phase, norm_w, norm_eps, freqs, inv_fp8_scale,
        c128_pool, k_valid, dbg, Int32(b_count), max_e,
    ).launch(grid=[b_count, 1, 1], block=[N_THREADS, 1, 1], stream=stream)


# ---------------------------------------------------------------------------
# micro test: staged Hadamard128 + MXFP4 quant (single warp per block)
# ---------------------------------------------------------------------------

@cute.kernel
def _micro_hadamard_quant_kernel(
    x: cute.Pointer,  # f32 (N, 128)
    out68: cute.Pointer,  # u8 (N, 68): [64B packed | 4B sf]
    dbg: cute.Pointer,  # f32 (N, 128) post-hadamard pre-quant
):
    tidx, _, _ = cute.arch.thread_idx()
    b, _, _ = cute.arch.block_idx()
    if tidx < 32:
        base = x + b * HD_IDX
        x0 = cute.arch.load(base + 4 * tidx, Float32)
        x1 = cute.arch.load(base + 4 * tidx + 1, Float32)
        x2 = cute.arch.load(base + 4 * tidx + 2, Float32)
        x3 = cute.arch.load(base + 4 * tidx + 3, Float32)
        b0, b1 = x0 + x1, x0 - x1
        b2, b3 = x2 + x3, x2 - x3
        d0, d1 = b0 + b2, b1 + b3
        d2, d3 = b0 - b2, b1 - b3
        for msk in (1, 2, 4, 8, 16):
            o0 = cute.arch.shuffle_sync_bfly(d0, offset=msk)
            o1 = cute.arch.shuffle_sync_bfly(d1, offset=msk)
            o2 = cute.arch.shuffle_sync_bfly(d2, offset=msk)
            o3 = cute.arch.shuffle_sync_bfly(d3, offset=msk)
            hi = (tidx & msk) != 0
            d0 = sel_f32(hi, o0 - d0, d0 + o0)
            d1 = sel_f32(hi, o1 - d1, d1 + o1)
            d2 = sel_f32(hi, o2 - d2, d2 + o2)
            d3 = sel_f32(hi, o3 - d3, d3 + o3)
        hs = Float32(HADAMARD_SCALE)
        d0, d1 = d0 * hs, d1 * hs
        d2, d3 = d2 * hs, d3 * hs
        cute.arch.store(dbg + b * HD_IDX + 4 * tidx, d0)
        cute.arch.store(dbg + b * HD_IDX + 4 * tidx + 1, d1)
        cute.arch.store(dbg + b * HD_IDX + 4 * tidx + 2, d2)
        cute.arch.store(dbg + b * HD_IDX + 4 * tidx + 3, d3)
        q0 = Float32(d0.to(cutlass.BFloat16))
        q1 = Float32(d1.to(cutlass.BFloat16))
        q2 = Float32(d2.to(cutlass.BFloat16))
        q3 = Float32(d3.to(cutlass.BFloat16))
        amax = cutlass.max(
            cutlass.max(cutlass.max(q0, -q0), cutlass.max(q1, -q1)),
            cutlass.max(cutlass.max(q2, -q2), cutlass.max(q3, -q3)),
        )
        for msk in (1, 2, 4):
            amax = cutlass.max(amax, cute.arch.shuffle_sync_bfly(amax,
                                                                 offset=msk))
        sc = cutlass.max(amax / Float32(6.0), Float32(1e-4))
        bits = f32_as_i32(sc)
        e = ((bits >> 23) & 255) - 127 + sel_i32(
            (bits & 8388607) != 0, Int32(1), Int32(0)
        )
        scale = i32_as_f32((e + 127) << 23)
        sf_byte = cutlass.min(cutlass.max(e + 127, Int32(0)), Int32(254))
        nib0 = _e2m1_nib(q0 / scale)
        nib1 = _e2m1_nib(q1 / scale)
        nib2 = _e2m1_nib(q2 / scale)
        nib3 = _e2m1_nib(q3 / scale)
        voff = b * 68 + tidx * 2
        cute.arch.store(out68 + voff, Uint8(nib0 | (nib1 << 4)))
        cute.arch.store(out68 + voff + 1, Uint8(nib2 | (nib3 << 4)))
        if (tidx & 7) == 0:
            cute.arch.store(out68 + b * 68 + 64 + (tidx >> 3), Uint8(sf_byte))


@cute.jit
def launch_micro(
    n_count: cutlass.Constexpr,
    x: cute.Pointer,
    out68: cute.Pointer,
    dbg: cute.Pointer,
    stream: cuda_drv.CUstream,
):
    _micro_hadamard_quant_kernel(x, out68, dbg).launch(
        grid=[n_count, 1, 1], block=[N_THREADS, 1, 1], stream=stream
    )
