"""Global top-K merge tile (Phase 1.4b, design: docs/phase1.4-comm-tasks.md §5).

One kernel serves both merge levels:
  L1: per (b,t), input = C chunk candidate lists (rank-local)  -> local top-K
  L2: per (b,t), input = cp rank slices of cand_l1 (allgather) -> global top-K

Algorithm: exact radix select on a 64-bit key = (sortable(logit) << 32) | ~entry_id
(descending logit, ascending entry_id tie-break; keys unique by construction).
8 byte-passes, prefix-filtered re-scan of the source (no compaction buffers,
same shape as logits.py::_merge_heap but 64-bit), early exit when the boundary
bin exactly fills the remaining count. Deterministic two-pass strip-mined emit
(input-order output) so L2 produces bitwise-identical selections on every rank
from the allgathered identical input — atomics-based emit is FORBIDDEN here.

Entry-id convention (integration, 1.4d): ids are GLOBAL token positions
(global = local * cp + rank under token-interleaved sharding), so every rank
can filter the global selection for its local entries via id % cp.
"""

import functools

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32
from cutlass.cute.runtime import make_ptr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

import torch

from mega_dsa_cp.tiles.logits import atom_add_u32, sel_i32

N_THREADS = 256


@cute.jit
def red_add_u32_gmem(ptr: cute.Pointer, val: Int32) -> None:
    """GPU-scope posted red for the dbg counter (cute.arch.red shape, cf.
    events/ptx.py — raw inline-asm red fails NVVM on this toolchain)."""
    cute.arch.red(ptr, val, op="add", dtype="u32", sem="relaxed", scope="gpu")


@dsl_user_op
def make_key64(v: Float32, ix: Int32, *, loc=None, ip=None) -> Int64:
    """(logit, entry_id) -> signed-i64 radix key: descending logit, ascending id.

    hi = f32 sortable bits, lo = ~id; top bit flipped so that SIGNED i64
    comparisons implement the unsigned 64-bit order.
    """
    return Int64(
        llvm.inline_asm(
            Int64.mlir_type,
            [
                Float32(v).ir_value(loc=loc, ip=ip),
                Int32(ix).ir_value(loc=loc, ip=ip),
            ],
            """{
            .reg .b32 u;
            .reg .pred p;
            .reg .b32 s;
            .reg .b32 l;
            .reg .b64 k;
            mov.b32 u, $1;
            setp.lt.s32 p, u, 0;
            xor.b32 s, u, 0x80000000;
            @p xor.b32 s, u, 0xFFFFFFFF;
            not.b32 l, $2;
            mov.b64 k, {l, s};
            xor.b64 k, k, 0x8000000000000000;
            mov.b64 $0, k;
            }""",
            "=l,f,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def i64_byte(x: Int64, shift: int, *, loc=None, ip=None) -> Int32:
    """(x >> shift) & 0xFF as Int32 — avoids Int64->Int32 constructor narrowing
    (unproven on this toolchain). shift is a trace-time python int."""
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [Int64(x).ir_value(loc=loc, ip=ip)],
            """{
            .reg .b64 t;
            shr.b64 t, $1, %d;
            and.b64 t, t, 0xFF;
            cvt.u32.u64 $0, t;
            }""" % shift,
            "=r,l",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


class MergeTopK:
    """Exact top-K over L concatenated candidate lists, per (b,t)."""

    def __init__(self, top_k: int = 512, q_len: int = 2):
        self.top_k = top_k
        self.q_len = q_len

    @cute.jit
    def __call__(
        self,
        cand_v_ptr: cute.Pointer,  # fp32 (B, q_len, L, maxK)
        cand_i_ptr: cute.Pointer,  # i32  (B, q_len, L, maxK)
        cand_c_ptr: cute.Pointer,  # i32  (B, q_len, L)
        out_v_ptr: cute.Pointer,   # fp32 (B, q_len, top_k)
        out_i_ptr: cute.Pointer,   # i32  (B, q_len, top_k)
        out_c_ptr: cute.Pointer,   # i32  (B, q_len)
        dims,                      # (B, L, maxK)
        dbg_ptr: cute.Pointer,
        stream: cuda.CUstream,
    ):
        B, L, maxK = dims
        layout_v = cute.make_ordered_layout(
            (B, self.q_len, L, maxK), order=(3, 2, 1, 0)
        )
        layout_c = cute.make_ordered_layout((B, self.q_len, L), order=(2, 1, 0))
        layout_o = cute.make_ordered_layout(
            (B, self.q_len, self.top_k), order=(2, 1, 0)
        )
        layout_oc = cute.make_ordered_layout((B, self.q_len), order=(1, 0))
        mCandV = cute.make_tensor(cand_v_ptr, layout_v)
        mCandI = cute.make_tensor(cand_i_ptr, layout_v)
        mCandC = cute.make_tensor(cand_c_ptr, layout_c)
        mOutV = cute.make_tensor(out_v_ptr, layout_o)
        mOutI = cute.make_tensor(out_i_ptr, layout_o)
        mOutC = cute.make_tensor(out_c_ptr, layout_oc)
        self.kernel(
            mCandV, mCandI, mCandC, mOutV, mOutI, mOutC, L, maxK, dbg_ptr
        ).launch(
            grid=(B * self.q_len, 1, 1),
            block=(N_THREADS, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mCandV: cute.Tensor,
        mCandI: cute.Tensor,
        mCandC: cute.Tensor,
        mOutV: cute.Tensor,
        mOutI: cute.Tensor,
        mOutC: cute.Tensor,
        L: Int32,
        maxK: Int32,
        dbg_ptr: cute.Pointer,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bt, _, _ = cute.arch.block_idx()
        b = bt // self.q_len
        t = bt % self.q_len

        smem = cutlass.utils.SmemAllocator()
        sHist = smem.allocate_tensor(Int32, cute.make_layout((256,)), 16)
        sOff = smem.allocate_tensor(Int32, cute.make_layout((256,)), 16)
        sCntL = smem.allocate_tensor(Int32, cute.make_layout((256,)), 16)
        sAux = smem.allocate_tensor(Int32, cute.make_layout((8,)), 16)
        sState = smem.allocate_tensor(Int64, cute.make_layout((2,)), 16)

        # stage per-list counts + total
        sCntL[tidx] = Int32(0)
        cute.arch.barrier()
        if tidx < L:
            sCntL[tidx] = mCandC[(b, t, tidx)]
        cute.arch.barrier()
        if tidx == 0:
            tot = Int32(0)
            for l in range(256):
                tot += sCntL[l]
            sAux[0] = tot
        cute.arch.barrier()
        m_total = sAux[0]

        # ---- radix select for tau (the K-th largest key, signed-flip domain)
        if tidx == 0:
            sState[0] = Int64(0)               # prefix
            sAux[1] = cutlass.min(m_total, Int32(self.top_k))  # remaining
            sAux[3] = Int32(0)                 # done flag
            sState[1] = Int64(-9223372036854775807 - 1)        # tau = i64 min
        cute.arch.barrier()

        if m_total > self.top_k:
            for p in cutlass.range_constexpr(8):
                if sAux[3] == 0:
                    shift = 56 - 8 * p
                    if tidx < 256:
                        sHist[tidx] = Int32(0)
                    cute.arch.barrier()
                    prefix = sState[0]
                    m_pad = L * maxK
                    iters = (m_pad + N_THREADS - 1) // N_THREADS
                    for it in cutlass.range(iters, unroll=1):
                        idx = it * N_THREADS + tidx
                        if idx < m_pad:
                            l = idx // maxK
                            j = idx % maxK
                            if j < sCntL[l]:
                                key = make_key64(mCandV[(b, t, l, j)],
                                                 mCandI[(b, t, l, j)])
                                ok = cutlass.Boolean(True)
                                if p > 0:
                                    ok = ((key - prefix)
                                          >> (64 - 8 * p)) == Int64(0)
                                if ok:
                                    bin_i = i64_byte(key, shift)
                                    atom_add_u32(sHist.iterator + bin_i,
                                                 Int32(1))
                    cute.arch.barrier()
                    if tidx == 0:
                        remaining = sAux[1]
                        cum = Int32(0)
                        bnd = Int32(0)
                        c_gt = Int32(0)
                        for bin_i in range(255, -1, -1):
                            cc = sHist[bin_i]
                            hit = (cum < remaining) & (cum + cc >= remaining)
                            bnd = sel_i32(hit, Int32(bin_i), bnd)
                            c_gt = sel_i32(hit, cum, c_gt)
                            cum += cc
                        prefix = sState[0]
                        new_prefix = prefix | (Int64(bnd) << shift)
                        sState[0] = new_prefix
                        rem2 = remaining - c_gt
                        sAux[1] = rem2
                        cute.arch.store(dbg_ptr + 8 + p * 4, bnd,
                                        sem="relaxed", scope="gpu")
                        cute.arch.store(dbg_ptr + 8 + p * 4 + 1, c_gt,
                                        sem="relaxed", scope="gpu")
                        cute.arch.store(dbg_ptr + 8 + p * 4 + 2, rem2,
                                        sem="relaxed", scope="gpu")
                        cute.arch.store(dbg_ptr + 8 + p * 4 + 3, sHist[bnd],
                                        sem="relaxed", scope="gpu")
                        if rem2 == sHist[bnd]:
                            # boundary bin exactly fills: tau is final
                            sState[1] = new_prefix
                            sAux[3] = Int32(1)
                    cute.arch.barrier()
            cute.arch.barrier()

        tau = sState[1]

        # ---- deterministic emit: two-pass strip partition over padded space.
        # gotcha-32 discipline: no local-var assignment inside dynamic `if`
        # that crosses the region boundary — all predication via sel_*.
        m_pad = L * maxK
        strip = (m_pad + N_THREADS - 1) // N_THREADS
        start = cutlass.min(tidx * strip, m_pad)
        end = cutlass.min(start + strip, m_pad)
        n_walk = end - start
        l0 = start // maxK
        j0 = start % maxK
        cnt = Int32(0)
        for _u in cutlass.range(strip, unroll=1):
            live = _u < n_walk
            l0s = cutlass.min(l0, L - 1)          # dead-walk clamp
            in_range = live & (j0 < sCntL[l0s])
            v = mCandV[(b, t, l0s, j0)]            # padded space: always in-bounds
            ix = mCandI[(b, t, l0s, j0)]
            hit = in_range & (make_key64(v, ix) >= tau)
            cnt += sel_i32(hit, Int32(1), Int32(0))
            j0n = j0 + Int32(1)
            wrap = j0n == maxK
            l0 = sel_i32(wrap, l0 + Int32(1), l0)
            j0 = sel_i32(wrap, Int32(0), j0n)
        sOff[tidx] = cnt
        cute.arch.barrier()
        if tidx == 0:
            run = Int32(0)
            for i in range(256):
                c0 = sOff[i]
                sOff[i] = run
                run += c0
            sAux[4] = run
        cute.arch.barrier()
        w = sOff[tidx]
        l0 = start // maxK
        j0 = start % maxK
        for _u in cutlass.range(strip, unroll=1):
            live = _u < n_walk
            l0s = cutlass.min(l0, L - 1)
            in_range = live & (j0 < sCntL[l0s])
            v = mCandV[(b, t, l0s, j0)]
            ix = mCandI[(b, t, l0s, j0)]
            hit = in_range & (make_key64(v, ix) >= tau)
            if hit:
                mOutV[(b, t, w)] = v
                mOutI[(b, t, w)] = ix
            w = sel_i32(hit, w + Int32(1), w)
            j0n = j0 + Int32(1)
            wrap = j0n == maxK
            l0 = sel_i32(wrap, l0 + Int32(1), l0)
            j0 = sel_i32(wrap, Int32(0), j0n)
        if tidx == 0:
            expect = cutlass.min(m_total, Int32(self.top_k))
            mOutC[(b, t)] = sAux[4]
            if sAux[4] != expect:
                red_add_u32_gmem(dbg_ptr, Int32(1))
            # debug taps: [4]=m_total [5]=emit_total [6]=tau_lo [7]=tau_hi
            cute.arch.store(dbg_ptr + 4, sAux[0], sem="relaxed", scope="gpu")
            cute.arch.store(dbg_ptr + 5, sAux[4], sem="relaxed", scope="gpu")
            cute.arch.store(dbg_ptr + 6, i64_byte(tau, 0),
                            sem="relaxed", scope="gpu")
            cute.arch.store(dbg_ptr + 7, i64_byte(tau, 56),
                            sem="relaxed", scope="gpu")



# ---------------------------------------------------------------- host runner


@functools.lru_cache(maxsize=None)
def _compiled(top_k: int, q_len: int):
    kernel = MergeTopK(top_k=top_k, q_len=q_len)
    return cute.compile(
        kernel,
        make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        (Int32(0), Int32(0), Int32(0)),
        make_ptr(Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
        cuda.CUstream(0),
    )


def run_merge_topk(
    cand_v: torch.Tensor,   # (B, q_len, L, maxK) fp32
    cand_i: torch.Tensor,   # (B, q_len, L, maxK) int32
    cand_c: torch.Tensor,   # (B, q_len, L) int32
    out_v: torch.Tensor,    # (B, q_len, top_k) fp32
    out_i: torch.Tensor,    # (B, q_len, top_k) int32
    out_c: torch.Tensor,    # (B, q_len) int32
    dbg: torch.Tensor,      # (1,) int32 emit-count mismatch counter
    stream: cuda.CUstream | None = None,
    top_k: int = 512,
) -> None:
    B, q_len, L, maxK = cand_v.shape
    assert L <= 256 and out_v.shape[-1] == top_k
    if stream is None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _compiled(top_k, q_len)(
        make_ptr(Float32, cand_v.data_ptr(), cute.AddressSpace.gmem,
                 assumed_align=16),
        make_ptr(Int32, cand_i.data_ptr(), cute.AddressSpace.gmem,
                 assumed_align=16),
        make_ptr(Int32, cand_c.data_ptr(), cute.AddressSpace.gmem,
                 assumed_align=16),
        make_ptr(Float32, out_v.data_ptr(), cute.AddressSpace.gmem,
                 assumed_align=16),
        make_ptr(Int32, out_i.data_ptr(), cute.AddressSpace.gmem,
                 assumed_align=16),
        make_ptr(Int32, out_c.data_ptr(), cute.AddressSpace.gmem,
                 assumed_align=16),
        (Int32(B), Int32(L), Int32(maxK)),
        make_ptr(Int32, dbg.data_ptr(), cute.AddressSpace.gmem,
                 assumed_align=16),
        stream,
    )
