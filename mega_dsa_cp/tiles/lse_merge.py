"""LSE merge tile (Phase 1.4c, design: docs/phase1.4-comm-tasks.md §6.5).

Merges cp per-rank attention partials into the global output, per (b,t):
  g        = max_i lse_i[h]                       (log2 domain, -inf = empty)
  glse[h]  = g + log2(sum_i exp2(lse_i[h] - g))   (=-inf if all empty)
  scale_i  = exp2(lse_i[h] - glse[h])             (0 when all empty)
  O[h, d]  = sum_i O_i[h, d] * scale_i

Semantics copied from the upstream split-KV reduction kernel
(flashinfer hca_fp8.py:1591-1652); works verbatim on our NORMALIZED
partials (hca.py epilogue divides by row_sum, lse = log2(row_sum) +
scale*log2e*run_max) because exp2(l_i - glse) carries 1/Z_total.

1.4d wiring note: attn_sink must be applied by exactly ONE rank's partial
(rank 0, owner of position 0 under token interleave) — the merge assumes
sink mass appears at most once across slices.
"""

import functools

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass.cute.runtime import make_ptr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

import torch

N_THREADS = 256
H = 128
D = 512


class LseMerge:
    def __init__(self, q_len: int = 2, max_splits: int = 8):
        self.q_len = q_len
        self.max_splits = max_splits          # cp capacity (static padding)

    @cute.jit
    def __call__(
        self,
        acc_o_ptr: cute.Pointer,   # fp32 (B, q_len, cp, H, D)
        acc_lse_ptr: cute.Pointer, # fp32 (B, q_len, cp, H)
        o_ptr: cute.Pointer,       # fp32 (B, q_len, H, D)
        lse_ptr: cute.Pointer,     # fp32 (B, q_len, H)
        dims,                      # (B, cp)
        stream: cuda.CUstream,
    ):
        B, cp = dims
        mAccO = cute.make_tensor(
            acc_o_ptr,
            cute.make_ordered_layout(
                (B, self.q_len, self.max_splits, H, D), order=(4, 3, 2, 1, 0)
            ),
        )
        mAccL = cute.make_tensor(
            acc_lse_ptr,
            cute.make_ordered_layout(
                (B, self.q_len, self.max_splits, H), order=(3, 2, 1, 0)
            ),
        )
        mO = cute.make_tensor(
            o_ptr, cute.make_ordered_layout((B, self.q_len, H, D),
                                            order=(3, 2, 1, 0))
        )
        mL = cute.make_tensor(
            lse_ptr, cute.make_ordered_layout((B, self.q_len, H),
                                              order=(2, 1, 0))
        )
        self.kernel(mAccO, mAccL, mO, mL, cp).launch(
            grid=(B * self.q_len, 1, 1), block=(N_THREADS, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        mAccO: cute.Tensor,
        mAccL: cute.Tensor,
        mO: cute.Tensor,
        mL: cute.Tensor,
        cp: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bt, _, _ = cute.arch.block_idx()
        b = bt // self.q_len
        t = bt % self.q_len

        smem = cutlass.utils.SmemAllocator()
        sScale = smem.allocate_tensor(
            Float32, cute.make_layout((self.max_splits, H)), 16
        )

        # ---- per-head global lse + scales (thread h handles head h).
        # gotcha-32 discipline: constexpr loop x dynamic cp guard -> clamped
        # loads + sel masking, no cross-region assignment.
        NEG_INF = Float32(-float("inf"))
        for i in cutlass.range_constexpr(self.max_splits):
            sScale[(i, tidx % H)] = Float32(0.0)
        cute.arch.barrier()
        if tidx < H:
            g = NEG_INF
            for i in cutlass.range_constexpr(self.max_splits):
                ic = cutlass.min(Int32(i), cp - 1)
                live = cutlass.Boolean(i < cp)
                l = sel_f32_m(live, mAccL[(b, t, ic, tidx)], NEG_INF)
                g = cute.arch.fmax(g, l)
            if g == NEG_INF:
                g = Float32(0.0)
            ssum = Float32(0.0)
            for i in cutlass.range_constexpr(self.max_splits):
                ic = cutlass.min(Int32(i), cp - 1)
                live = cutlass.Boolean(i < cp)
                l = sel_f32_m(live, mAccL[(b, t, ic, tidx)], NEG_INF)
                ssum += cute.math.exp2(l - g, fastmath=True)
            has_mass = ssum > Float32(0.0)
            glse = sel_f32_m(
                has_mass, g + cute.math.log2(ssum, fastmath=True), NEG_INF
            )
            mL[(b, t, tidx)] = glse
            for i in cutlass.range_constexpr(self.max_splits):
                ic = cutlass.min(Int32(i), cp - 1)
                live = cutlass.Boolean(i < cp) & has_mass
                l = sel_f32_m(live, mAccL[(b, t, ic, tidx)], NEG_INF)
                sc = sel_f32_m(
                    live, cute.math.exp2(l - glse, fastmath=True), Float32(0.0)
                )
                if i < cp:
                    sScale[(i, tidx)] = sc
        cute.arch.barrier()

        # ---- O accumulate: 256 threads x 256 elems (elem = h*D + d)
        for j in cutlass.range_constexpr(H * D // N_THREADS):
            elem = j * N_THREADS + tidx
            h = elem // D
            acc = Float32(0.0)
            for i in cutlass.range_constexpr(self.max_splits):
                ic = cutlass.min(Int32(i), cp - 1)
                acc += mAccO[(b, t, ic, h, elem % D)] * sScale[(i, h)]
            mO[(b, t, h, elem % D)] = acc


@dsl_user_op
def sel_f32_m(pred, a, b, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.select(
            cutlass.Boolean(pred).ir_value(loc=loc, ip=ip),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------- host runner


@functools.lru_cache(maxsize=None)
def _compiled(q_len: int, max_splits: int):
    kernel = LseMerge(q_len=q_len, max_splits=max_splits)
    return cute.compile(
        kernel,
        make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=16),
        make_ptr(Float32, 0, cute.AddressSpace.gmem, assumed_align=16),
        (Int32(0), Int32(0)),
        cuda.CUstream(0),
    )


def run_lse_merge(
    acc_o: torch.Tensor,    # (B, q_len, cp_max, H, D) fp32
    acc_lse: torch.Tensor,  # (B, q_len, cp_max, H) fp32
    o: torch.Tensor,        # (B, q_len, H, D) fp32
    lse: torch.Tensor,      # (B, q_len, H) fp32
    cp: int,
    stream: cuda.CUstream | None = None,
) -> None:
    B, q_len = o.shape[0], o.shape[1]
    if stream is None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _compiled(q_len, acc_o.shape[2])(
        make_ptr(Float32, acc_o.data_ptr(), cute.AddressSpace.gmem,
                 assumed_align=16),
        make_ptr(Float32, acc_lse.data_ptr(), cute.AddressSpace.gmem,
                 assumed_align=16),
        make_ptr(Float32, o.data_ptr(), cute.AddressSpace.gmem,
                 assumed_align=16),
        make_ptr(Float32, lse.data_ptr(), cute.AddressSpace.gmem,
                 assumed_align=16),
        (Int32(B), Int32(cp)),
        stream,
    )
