"""Phase 1.1 v1: FP4 (MXFP4) paged MQA logits kernel (standalone, logits to gmem).

Design: docs/phase1-logits-tile.md. One CTA per (chunk c, request b):
streams its chunk of the paged index-K cache (128-entry pages, fused
[128x64B E2M1 data][512B atom-arranged UE8M0 scales]) through a deep TMA
pipeline, scores with MmaMXF4Op (M=128 kv, N=q_len*heads, K=128), epilogue
ReLU * head-gate weights -> fp32 logits rows in gmem. Top-K heap lands in v2.

Warp org (192 threads): warps 0-3 math (epilogue), warp 4 UMMA, warp 5 TMA.
Idioms: svdquant gemm_w4a4/kernel_v2_fa4.py (blockscaled MMA + SF plumbing),
sglang cutedsl_fp8_paged_mqa_logits.py (paged TMA + epilogue reduce).
"""

from __future__ import annotations

from typing import Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

# Page layout constants (index-K cache)
PAGE_TOKENS = 128
HEAD_DIM = 128
PAGE_DATA_BYTES = PAGE_TOKENS * 64  # 8192
PAGE_SF_BYTES = 512  # 128 tokens x 4B UE8M0 (atom-arranged)
PAGE_BYTES = PAGE_DATA_BYTES + PAGE_SF_BYTES  # 8704
PAGE_ELEMS_FP4 = PAGE_BYTES * 2  # fp4 elements per page stride

SF_VEC_SIZE = 32

# Fused top-K (v2): per-spec-slot heap + overflow in smem; threshold-filtered
# inserts; radix-select merge when overflow nears capacity.
TOP_K = 2048
OVF_CAP = 2048
OVF_TRIGGER = OVF_CAP - 128  # one block adds at most 128 inserts per spec slot


@dsl_user_op
def atom_add_u32(ptr: cute.Pointer, val: Int32, *, loc=None, ip=None) -> Int32:
    """Shared-memory atomic add (smem pointers: 32-bit shared-space address).
    Returns old value."""
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [
                ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
                Int32(val).ir_value(loc=loc, ip=ip),
            ],
            "atom.shared.add.u32 $0, [$1], $2;",
            "=r,r,r",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f32_sort_key(x: Float32, *, loc=None, ip=None) -> Int32:
    """fp32 -> sign-flipped sortable u32 (monotonic map for radix select)."""
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [Float32(x).ir_value(loc=loc, ip=ip)],
            """{
            .reg .b32 u;
            .reg .pred p;
            mov.b32 u, $1;
            setp.lt.s32 p, u, 0;
            xor.b32 $0, u, 0x80000000;
            @p xor.b32 $0, u, 0xFFFFFFFF;
            }""",
            "=r,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def f32_as_i32(x: Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.bitcast(
            Int32.mlir_type, Float32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip
        )
    )


@dsl_user_op
def sel_i32(pred, a, b, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.select(
            cutlass.Boolean(pred).ir_value(loc=loc, ip=ip),
            Int32(a).ir_value(loc=loc, ip=ip),
            Int32(b).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def sel_f32(pred, a, b, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.select(
            cutlass.Boolean(pred).ir_value(loc=loc, ip=ip),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )

def relu2_fma_f32x2(v0, v1, v2, v3, w0, w1, w2, w3, s0x, s0y, s1x, s1y):
    """ReLU + weighted accumulate of 4 lanes via packed f32x2 (DeepGEMM pattern,
    via sglang cutedsl_fp8_paged_mqa_logits.py:180). 2*relu(x) = x + |x|; the
    0.5 is folded by the caller."""
    r01 = cute.arch.add_packed_f32x2((v0, v1), (cute.math.absf(v0), cute.math.absf(v1)))
    r23 = cute.arch.add_packed_f32x2((v2, v3), (cute.math.absf(v2), cute.math.absf(v3)))
    s0x, s0y = cute.arch.fma_packed_f32x2(r01, (w0, w1), (s0x, s0y), rnd="rn")
    s1x, s1y = cute.arch.fma_packed_f32x2(r23, (w2, w3), (s1x, s1y), rnd="rn")
    return s0x, s0y, s1x, s1y


class Fp4PagedMQALogits:
    def __init__(
        self,
        n_heads: int = 64,
        q_len: int = 2,
        num_kv_stage: int = 8,
        num_acc_stage: int = 2,
        fused_topk: bool = False,
    ):
        self.n_heads = n_heads
        self.q_len = q_len
        self.n = n_heads * q_len  # MMA N
        assert self.n <= 256, "single-MMA TMEM limit (N <= 256)"
        self.num_kv_stage = num_kv_stage
        self.num_acc_stage = num_acc_stage
        self.fused_topk = fused_topk
        self.acc_dtype = cutlass.Float32
        self.ab_dtype = cutlass.Float4E2M1FN
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (PAGE_TOKENS, self.n)
        self.epi_sub_mn = (PAGE_TOKENS, 32)

        self.math_warp_ids = (0, 1, 2, 3)
        self.umma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 192

    def _setup_mma(self):
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            self.ab_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            SF_VEC_SIZE,
            self.cta_group,
            self.mma_tiler_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.ab_dtype,
            self.ab_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            SF_VEC_SIZE,
            tcgen05.CtaGroup.ONE,
            (self.mma_tiler_mn[0], cute.round_up(self.mma_tiler_mn[1], 128)),
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])  # 64 for MXF4
        self.mma_inst_tile_k = HEAD_DIM // mma_inst_shape_k  # 2
        self.mma_tiler = (*self.mma_tiler_mn, HEAD_DIM)
        self.mma_tiler_sfb = (
            self.mma_tiler_mn[0],
            cute.round_up(self.mma_tiler_mn[1], 128),
            HEAD_DIM,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.num_kv_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.ab_dtype, 1
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, SF_VEC_SIZE, self.num_kv_stage
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, SF_VEC_SIZE, 1
        )

        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (
            self.cta_tile_shape_mnk[0] // sf_atom_mn
        ) * self.mma_inst_tile_k
        self.num_sfb_tmem_cols = (
            self.mma_tiler_sfb[1] // sf_atom_mn
        ) * self.mma_inst_tile_k
        self.num_acc_tmem_cols = self.mma_tiler[1] * self.num_acc_stage
        self.num_tmem_cols_total = (
            self.num_acc_tmem_cols + self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        )
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        return tiled_mma, tiled_mma_sfb

    @cute.jit
    def __call__(
        self,
        kv_ptr: cute.Pointer,  # fp4, fused pages (data at page*8704 + 0)
        ksf_ptr: cute.Pointer,  # ue8m0, fused pages (sf at page*8704 + 8192)
        q_ptr: cute.Pointer,  # fp4 (B, N, 64B) contiguous
        qsf_ptr: cute.Pointer,  # ue8m0 (B, 512B) atom-arranged
        w_ptr: cute.Pointer,  # fp32 (B, N)
        bt_ptr: cute.Pointer,  # int32 (B, max_pages)
        logits_ptr: cute.Pointer,  # fp32 (B*q_len, max_pages*128)
        seq_lens_ptr: cute.Pointer,  # int32 (B,) — fused_topk masking
        cand_v_ptr: cute.Pointer,  # fp32 (B, q_len, n_chunks, TOP_K)
        cand_i_ptr: cute.Pointer,  # int32 (B, q_len, n_chunks, TOP_K)
        cand_c_ptr: cute.Pointer,  # int32 (B, q_len, n_chunks)
        dbg_ptr: cute.Pointer,  # int32 (8,) progress markers
        dims: Tuple[Int32, Int32, Int32, Int32, Int32],
        stream: cuda.CUstream,
    ):
        batch_size, num_pages, blocks_per_chunk, max_pages, n_chunks = dims

        # --- gmem tensors ---
        kv_layout = cute.make_layout(
            (PAGE_TOKENS, HEAD_DIM, num_pages),
            stride=(HEAD_DIM, 1, PAGE_ELEMS_FP4),
        )
        mA = cute.make_tensor(kv_ptr, kv_layout)
        sf1 = blockscaled_utils.tile_atom_to_shape_SF(
            (PAGE_TOKENS, HEAD_DIM, 1), SF_VEC_SIZE
        )
        sfa_layout = cute.make_layout(
            (sf1.shape[0], sf1.shape[1], num_pages),
            stride=(sf1.stride[0], sf1.stride[1], PAGE_BYTES),
        )
        mSFA = cute.make_tensor(ksf_ptr, sfa_layout)
        q_layout = cute.make_layout(
            (self.n, HEAD_DIM, batch_size),
            stride=(HEAD_DIM, 1, self.n * HEAD_DIM),
        )
        mB = cute.make_tensor(q_ptr, q_layout)
        sfb_layout = cute.make_layout(
            (sf1.shape[0], sf1.shape[1], batch_size),
            stride=(sf1.stride[0], sf1.stride[1], PAGE_SF_BYTES),
        )
        mSFB = cute.make_tensor(qsf_ptr, sfb_layout)
        mW = cute.make_tensor(
            w_ptr, cute.make_layout((self.n, batch_size), stride=(1, self.n))
        )
        mBT = cute.make_tensor(
            bt_ptr,
            cute.make_layout((batch_size, max_pages), stride=(max_pages, 1)),
        )
        s_max = max_pages * PAGE_TOKENS
        mLogits = cute.make_tensor(
            logits_ptr,
            cute.make_layout(
                (batch_size * self.q_len, s_max), stride=(s_max, 1)
            ),
        )
        mSeqLens = cute.make_tensor(seq_lens_ptr, cute.make_layout((batch_size,)))
        mCandV = cute.make_tensor(
            cand_v_ptr,
            cute.make_ordered_layout(
                (batch_size, self.q_len, n_chunks, TOP_K), order=(3, 2, 1, 0)
            ),
        )
        mCandI = cute.make_tensor(
            cand_i_ptr,
            cute.make_ordered_layout(
                (batch_size, self.q_len, n_chunks, TOP_K), order=(3, 2, 1, 0)
            ),
        )
        mCandC = cute.make_tensor(
            cand_c_ptr,
            cute.make_ordered_layout((batch_size, self.q_len, n_chunks), order=(2, 1, 0)),
        )


        tiled_mma, tiled_mma_sfb = self._setup_mma()

        # --- TMA atoms ---
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma.thr_id
            ),
            mA,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, tiled_mma.thr_id
            ),
            mB,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sm100_utils.cluster_shape_to_tma_atom_A(
                self.cluster_shape_mn, tiled_mma.thr_id
            ),
            mSFA,
            sfa_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_SFB(
                self.cluster_shape_mn, tiled_mma.thr_id
            ),
            mSFB,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        kv_tma_bytes = cute.size_in_bytes(
            self.ab_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        q_tma_bytes = cute.size_in_bytes(
            self.ab_dtype, b_smem_layout
        ) + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, self.num_kv_stage * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_holding_buf: cutlass.Int32

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            mW,
            mBT,
            mLogits,
            mSeqLens,
            mCandV,
            mCandI,
            mCandC,
            dbg_ptr,
            blocks_per_chunk,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            SharedStorage,
            kv_tma_bytes,
            q_tma_bytes,
        ).launch(
            grid=(n_chunks, batch_size, 1),
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        mW: cute.Tensor,
        mBT: cute.Tensor,
        mLogits: cute.Tensor,
        mSeqLens: cute.Tensor,
        mCandV: cute.Tensor,
        mCandI: cute.Tensor,
        mCandC: cute.Tensor,
        dbg_ptr: cute.Pointer,
        blocks_per_chunk: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        SharedStorage: cutlass.Constexpr,
        kv_tma_bytes: cutlass.Constexpr,
        q_tma_bytes: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, _ = cute.arch.block_idx()
        b = bidy
        blk0 = bidx * blocks_per_chunk
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # --- pipelines ---
        one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        kv_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=self.num_kv_stage,
            producer_group=one_thread,
            consumer_group=one_thread,
            tx_count=kv_tma_bytes,
            barrier_storage=storage.kv_mbar.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_pipe = pipeline.PipelineUmmaAsync.create(
            num_stages=self.num_acc_stage,
            producer_group=one_thread,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.math_warp_ids) * 32
            ),
            barrier_storage=storage.acc_mbar.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        q_pipe = pipeline.PipelineTmaUmma.create(
            num_stages=1,
            producer_group=one_thread,
            consumer_group=one_thread,
            tx_count=q_tma_bytes,
            barrier_storage=storage.q_mbar.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        kv_prod = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stage
        )
        kv_cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stage
        )
        acc_prod = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_acc_stage
        )
        acc_cons = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_acc_stage
        )
        q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2, num_threads=(len(self.math_warp_ids) + 1) * 32
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.math_warp_ids[0],
            is_two_cta=False,
        )
        sW_barrier = pipeline.NamedBarrier(
            barrier_id=3, num_threads=len(self.math_warp_ids) * 32
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = smem.allocate_tensor(
            self.ab_dtype,
            a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            self.ab_dtype,
            b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            self.sf_dtype, sfa_smem_layout_staged, byte_alignment=128
        )
        sSFB = smem.allocate_tensor(
            self.sf_dtype, sfb_smem_layout_staged, byte_alignment=128
        )
        sW = smem.allocate_tensor(Float32, cute.make_layout((self.n,)), 16)

        # --- fused top-K smem (v2) ---
        sHeapV = sHeapI = sOvfV = sOvfI = None
        sScrAV = sScrAI = sBins = sAux = sCnt = sTheta = sFlags = None
        if cutlass.const_expr(self.fused_topk):
            sHeapV = smem.allocate_tensor(
                Float32, cute.make_layout((self.q_len, TOP_K)), 16
            )
            sHeapI = smem.allocate_tensor(
                Int32, cute.make_layout((self.q_len, TOP_K)), 16
            )
            sOvfV = smem.allocate_tensor(
                Float32, cute.make_layout((self.q_len, OVF_CAP)), 16
            )
            sOvfI = smem.allocate_tensor(
                Int32, cute.make_layout((self.q_len, OVF_CAP)), 16
            )
            sScrAV = smem.allocate_tensor(Float32, cute.make_layout((TOP_K,)), 16)
            sScrAI = smem.allocate_tensor(Int32, cute.make_layout((TOP_K,)), 16)
            sBins = smem.allocate_tensor(Int32, cute.make_layout((256,)), 16)
            sAux = smem.allocate_tensor(Int32, cute.make_layout((4,)), 16)
            # per spec slot: [heap_cnt, ovf_cnt, out_cnt, bnd_cnt]
            sCnt = smem.allocate_tensor(
                Int32, cute.make_layout((self.q_len, 4), stride=(4, 1)), 16
            )
            sTheta = smem.allocate_tensor(Float32, cute.make_layout((self.q_len,)), 16)
            sFlags = smem.allocate_tensor(Int32, cute.make_layout((self.q_len,)), 16)

        # --- TMA / MMA partitions ---
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )

        thr_mma = tiled_mma.get_slice(0)
        thr_mma_sfb = tiled_mma_sfb.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

        one_layout = cute.make_layout(1)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a, 0, one_layout,
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b, 0, one_layout,
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
        )
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa, 0, one_layout,
            cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb, 0, one_layout,
            cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ===================== TMA warp =====================
        if warp_idx == self.tma_warp_id:
            cute.arch.setmaxregister_decrease(24)
            q_pipe.producer_acquire(q_prod)
            q_bar = q_pipe.producer_get_barrier(q_prod)
            cute.copy(
                tma_atom_b, tBgB[(None, 0, 0, b)], tBsB[(None, 0)],
                tma_bar_ptr=q_bar,
            )
            cute.copy(
                tma_atom_sfb, tBgSFB[(None, 0, 0, b)], tBsSFB[(None, 0)],
                tma_bar_ptr=q_bar,
            )
            if tidx % 32 == 0:
                cute.arch.store(dbg_ptr + 0, Int32(1), sem="relaxed", scope="sys")
            for i in cutlass.range(blocks_per_chunk, unroll=1):
                page = mBT[(b, blk0 + i)]
                kv_pipe.producer_acquire(kv_prod)
                bar = kv_pipe.producer_get_barrier(kv_prod)
                cute.copy(
                    tma_atom_a, tAgA[(None, 0, 0, page)],
                    tAsA[(None, kv_prod.index)], tma_bar_ptr=bar,
                )
                cute.copy(
                    tma_atom_sfa, tAgSFA[(None, 0, 0, page)],
                    tAsSFA[(None, kv_prod.index)], tma_bar_ptr=bar,
                )
                if tidx % 32 == 0:
                    cute.arch.store(dbg_ptr + 0, Int32(2 + i), sem="relaxed", scope="sys")
                kv_prod.advance()
            if tidx % 32 == 0:
                cute.arch.store(dbg_ptr + 0, Int32(100), sem="relaxed", scope="sys")

        # ===================== UMMA warp =====================
        elif warp_idx == self.umma_warp_id:
            cute.arch.setmaxregister_decrease(24)
            tmem.wait_for_alloc()
            acc_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_ptr, tCtAcc_fake.layout)
            sfa_tmem_ptr = cute.recast_ptr(
                acc_ptr + self.num_acc_tmem_cols, dtype=self.sf_dtype
            )
            tCtSFA = cute.make_tensor(
                sfa_tmem_ptr,
                blockscaled_utils.make_tmem_layout_sfa(
                    tiled_mma, self.mma_tiler, SF_VEC_SIZE,
                    cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
                ),
            )
            sfb_tmem_ptr = cute.recast_ptr(
                acc_ptr + self.num_acc_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB = cute.make_tensor(
                sfb_tmem_ptr,
                blockscaled_utils.make_tmem_layout_sfb(
                    tiled_mma, self.mma_tiler, SF_VEC_SIZE,
                    cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
                ),
            )
            (tiled_copy_s2t_sfa, tCsSFA_s2t, tCtSFA_s2t) = self._s2t_copy(sSFA, tCtSFA)
            (tiled_copy_s2t_sfb, tCsSFB_s2t, tCtSFB_s2t) = self._s2t_copy(sSFB, tCtSFB)

            cute.arch.store(dbg_ptr + 1, Int32(1), sem="relaxed", scope="sys")
            q_pipe.consumer_wait(q_cons)
            cute.arch.store(dbg_ptr + 1, Int32(2), sem="relaxed", scope="sys")
            cute.copy(
                tiled_copy_s2t_sfb,
                tCsSFB_s2t[(None, None, None, None, 0)],
                tCtSFB_s2t,
            )
            num_kblocks = self.mma_inst_tile_k
            for i in cutlass.range(blocks_per_chunk, unroll=1):
                kv_pipe.consumer_wait(kv_cons)
                cute.arch.store(dbg_ptr + 1, Int32(3), sem="relaxed", scope="sys")
                cute.copy(
                    tiled_copy_s2t_sfa,
                    tCsSFA_s2t[(None, None, None, None, kv_cons.index)],
                    tCtSFA_s2t,
                )
                acc_pipe.producer_acquire(acc_prod)
                cute.arch.store(dbg_ptr + 1, Int32(4), sem="relaxed", scope="sys")
                tCtAcc = tCtAcc_base[(None, None, None, acc_prod.index)]
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for kblk in range(num_kblocks):  # trace-time unroll (gotcha #24)
                    tiled_mma.set(
                        tcgen05.Field.SFA, tCtSFA[(None, None, kblk)].iterator
                    )
                    tiled_mma.set(
                        tcgen05.Field.SFB, tCtSFB[(None, None, kblk)].iterator
                    )
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[(None, None, kblk, kv_cons.index)],
                        tCrB[(None, None, kblk, 0)],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                cute.arch.store(dbg_ptr + 1, Int32(5), sem="relaxed", scope="sys")
                kv_pipe.consumer_release(kv_cons)
                kv_cons.advance()
                acc_pipe.producer_commit(acc_prod)
                acc_prod.advance()
                cute.arch.store(dbg_ptr + 1, Int32(6 + i), sem="relaxed", scope="sys")
            q_pipe.consumer_release(q_cons)
            cute.arch.store(dbg_ptr + 1, Int32(100), sem="relaxed", scope="sys")

        # ===================== math warps =====================
        else:
            cute.arch.setmaxregister_increase(240)
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_ptr, tCtAcc_fake.layout)

            if tidx < self.n:
                sW[tidx] = mW[(tidx, b)]
            sW_barrier.arrive_and_wait()
            if tidx == 0:
                cute.arch.store(dbg_ptr + 2, Int32(1), sem="relaxed", scope="sys")

            local_tidx = tidx % 128
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                utils.LayoutEnum.ROW_MAJOR,
                self.acc_dtype,
                self.acc_dtype,
                self.epi_sub_mn,
                False,
            )
            cC = cute.make_identity_tensor(self.epi_sub_mn)
            tAcc_ref = tCtAcc_base[(None, None, None, 0)][((None, None), 0, 0)]
            tAcc_ref_epi = cute.flat_divide(tAcc_ref, self.epi_sub_mn)
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tAcc_ref_epi[(None, None, 0, 0)]
            )
            thr_copy = tiled_copy_t2r.get_slice(local_tidx)
            tTR_cC = thr_copy.partition_D(cC)
            m_coord = tTR_cC[0][0]
            tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)

            n_sub = self.n // self.epi_sub_mn[1]
            limit = Int32(0)
            dbg_score0 = Float32(0.0)
            if cutlass.const_expr(self.fused_topk):
                if tidx < self.q_len:
                    sCnt[(tidx, 0)] = Int32(0)
                    sCnt[(tidx, 1)] = Int32(0)
                    sCnt[(tidx, 3)] = Int32(0)
                    sTheta[tidx] = Float32(-float("inf"))
                limit = mSeqLens[b] - 128
                if limit < 0:
                    limit = Int32(0)
                sW_barrier.arrive_and_wait()
            for i in cutlass.range(blocks_per_chunk, unroll=1):
                acc_pipe.consumer_wait(acc_cons)
                if tidx == 0:
                    cute.arch.store(dbg_ptr + 2, Int32(2 + i), sem="relaxed", scope="sys")
                tAcc = tCtAcc_base[(None, None, None, acc_cons.index)][
                    ((None, None), 0, 0)
                ]
                tAcc_epi = cute.flat_divide(tAcc, self.epi_sub_mn)
                tTR_tAcc = thr_copy.partition_S(tAcc_epi)
                kv_pos = (blk0 + i) * PAGE_TOKENS + m_coord
                for t in range(self.q_len):
                    s0x = Float32(0.0)
                    s0y = Float32(0.0)
                    s1x = Float32(0.0)
                    s1y = Float32(0.0)
                    for sub in range(t * (n_sub // self.q_len), (t + 1) * (n_sub // self.q_len)):
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[(None, None, None, 0, sub)],
                            tTR_rAcc,
                        )
                        cute.arch.fence_view_async_tmem_load()
                        acc_vec = tTR_rAcc.load()
                        h_base = t * self.n_heads + (sub % (n_sub // self.q_len)) * 32
                        for h in range(0, 32, 4):
                            s0x, s0y, s1x, s1y = relu2_fma_f32x2(
                                acc_vec[h], acc_vec[h + 1],
                                acc_vec[h + 2], acc_vec[h + 3],
                                sW[h_base + h], sW[h_base + h + 1],
                                sW[h_base + h + 2], sW[h_base + h + 3],
                                s0x, s0y, s1x, s1y,
                            )
                    score = (s0x + s0y + s1x + s1y) * Float32(0.5)
                    if t == 0:
                        dbg_score0 = score
                    if cutlass.const_expr(self.fused_topk):
                        if kv_pos < limit:
                            if score > sTheta[t]:
                                if sCnt[(t, 0)] < TOP_K:
                                    slot = atom_add_u32(sCnt.iterator + (t * 4 + 0), Int32(1))
                                    if slot < TOP_K:
                                        sHeapV[(t, slot)] = score
                                        sHeapI[(t, slot)] = kv_pos
                                    else:
                                        s2 = atom_add_u32(sCnt.iterator + (t * 4 + 1), Int32(1))
                                        if s2 < OVF_CAP:
                                            sOvfV[(t, s2)] = score
                                            sOvfI[(t, s2)] = kv_pos
                                else:
                                    s2 = atom_add_u32(sCnt.iterator + (t * 4 + 1), Int32(1))
                                    if s2 < OVF_CAP:
                                        sOvfV[(t, s2)] = score
                                        sOvfI[(t, s2)] = kv_pos
                    else:
                        mLogits[(b * self.q_len + t, kv_pos)] = score
                acc_pipe.consumer_release(acc_cons)
                acc_cons.advance()
                if cutlass.const_expr(self.fused_topk):
                    sW_barrier.arrive_and_wait()
                    if b == 0 and bidx == 0 and tidx == 0:
                        cute.arch.store(dbg_ptr + 16 + i * 2, sCnt[(0, 0)], sem="relaxed", scope="sys")
                        cute.arch.store(dbg_ptr + 17 + i * 2, sCnt[(0, 1)], sem="relaxed", scope="sys")
                        cute.arch.store(dbg_ptr + 64 + i, f32_as_i32(dbg_score0), sem="relaxed", scope="sys")
                        cute.arch.store(dbg_ptr + 96 + i, f32_as_i32(sW[0]), sem="relaxed", scope="sys")
                    if tidx < self.q_len:
                        sFlags[tidx] = sel_i32(
                            sCnt[(tidx, 1)] >= OVF_TRIGGER, Int32(1), Int32(0)
                        )
                    sW_barrier.arrive_and_wait()
                    for t in range(self.q_len):
                        if sFlags[t] == 1:
                            self._merge_heap(
                                t, tidx, sHeapV, sHeapI, sOvfV, sOvfI,
                                sScrAV, sScrAI, sBins, sAux, sCnt, sTheta,
                                sW_barrier,
                            )
            if cutlass.const_expr(self.fused_topk):
                self._finalize_candidates(
                    b, bidx, tidx, sHeapV, sHeapI, sOvfV, sOvfI,
                    sScrAV, sScrAI, sBins, sAux, sCnt, sTheta, sFlags,
                    mCandV, mCandI, mCandC, sW_barrier, dbg_ptr,
                )
            if tidx == 0:
                cute.arch.store(dbg_ptr + 2, Int32(100), sem="relaxed", scope="sys")

            tmem.relinquish_alloc_permit()
            tmem.free(acc_ptr)

    @staticmethod
    def _s2t_copy(sSF: cute.Tensor, tSF: cute.Tensor):
        tCsSF = cute.filter_zeros(sSF)
        tCtSF = cute.filter_zeros(tSF)
        copy_atom = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), cutlass.Float8E8M0FNU
        )
        tiled_copy = tcgen05.make_s2t_copy(copy_atom, tCtSF)
        thr_copy = tiled_copy.get_slice(0)
        tCsSF_ = thr_copy.partition_S(tCsSF)
        tCsSF_desc = tcgen05.get_s2t_smem_desc_tensor(tiled_copy, tCsSF_)
        tCtSF_part = thr_copy.partition_D(tCtSF)
        return tiled_copy, tCsSF_desc, tCtSF_part

    @cute.jit
    def _merge_heap(
        self, t, tidx, sHeapV, sHeapI, sOvfV, sOvfI,
        sScrAV, sScrAI, sBins, sAux, sCnt, sTheta, bar,
    ):
        """Exact top-TOP_K of heap[0:TOP_K] + ovf[0:ovf_cnt] back into heap,
        theta = new heap min. Byte-wise radix select (4 passes over key bytes);
        boundary-bin entries are NOT copied — later passes re-scan the source
        with a high-bit prefix filter (saves smem and the read/write race)."""
        if tidx == 0:
            sCnt[(t, 2)] = Int32(0)  # out_cnt (atomic slot for sScrA)
            sCnt[(t, 3)] = sCnt[(t, 3)] + Int32(1)  # merge counter (debug)
        bar.arrive_and_wait()
        total = TOP_K + sCnt[(t, 1)]
        iters = (total + 127) // 128
        remaining = Int32(TOP_K)
        prefix = Int32(0)
        for p in range(4):
            shift = (3 - p) * 8
            pm32 = 32 - 8 * p
            if remaining > 0:
                sBins[tidx * 2] = Int32(0)
                sBins[tidx * 2 + 1] = Int32(0)
                bar.arrive_and_wait()
                # histogram of this byte over prefix-filtered source
                for it in cutlass.range(iters, unroll=1):
                    j = it * 128 + tidx
                    v = Float32(0.0)
                    if j < total:
                        if j < TOP_K:
                            v = sHeapV[(t, j)]
                        else:
                            v = sOvfV[(t, j - TOP_K)]
                        key = f32_sort_key(v)
                        ok = cutlass.Boolean(True)
                        if p > 0:
                            # logical-shift prefix check (Int32 >> sign-extends)
                            ok = (key & Int32(-1 << pm32)) == (prefix << pm32)
                        if ok:
                            bin_i = (key >> shift) & 0xFF
                            atom_add_u32(sBins.iterator + bin_i, Int32(1))
                bar.arrive_and_wait()
                # thread0: descending scan for the boundary bin
                if tidx == 0:
                    cum = Int32(0)
                    bnd = Int32(0)
                    c_gt = Int32(0)
                    for bin_i in range(255, -1, -1):
                        c = sBins[bin_i]
                        hit = (cum < remaining) & (cum + c >= remaining)
                        bnd = sel_i32(hit, Int32(bin_i), bnd)
                        c_gt = sel_i32(hit, cum, c_gt)
                        cum += c
                    sAux[0] = bnd
                    sAux[1] = c_gt
                bar.arrive_and_wait()
                bnd = sAux[0]
                c_gt = sAux[1]
                # compact: strictly-above-boundary entries are selected
                for it in cutlass.range(iters, unroll=1):
                    j = it * 128 + tidx
                    v = Float32(0.0)
                    ix = Int32(0)
                    if j < total:
                        if j < TOP_K:
                            v = sHeapV[(t, j)]
                            ix = sHeapI[(t, j)]
                        else:
                            v = sOvfV[(t, j - TOP_K)]
                            ix = sOvfI[(t, j - TOP_K)]
                        key = f32_sort_key(v)
                        ok = cutlass.Boolean(True)
                        if p > 0:
                            ok = (key & Int32(-1 << pm32)) == (prefix << pm32)
                        if ok:
                            bin_i = (key >> shift) & 0xFF
                            if bin_i > bnd:
                                slot = atom_add_u32(sCnt.iterator + (t * 4 + 2), Int32(1))
                                sScrAV[slot] = v
                                sScrAI[slot] = ix
                            elif p == 3 and bin_i == bnd:
                                # exact-key tie: fill up to TOP_K
                                slot = atom_add_u32(sCnt.iterator + (t * 4 + 2), Int32(1))
                                if slot < TOP_K:
                                    sScrAV[slot] = v
                                    sScrAI[slot] = ix
                if p < 3:
                    prefix = (prefix << 8) | bnd
                bar.arrive_and_wait()
                remaining = remaining - c_gt
        # selected set (sScrA) -> heap; theta = heap min
        for it in cutlass.range(TOP_K // 128, unroll=1):
            j = it * 128 + tidx
            sHeapV[(t, j)] = sScrAV[j]
            sHeapI[(t, j)] = sScrAI[j]
        m = Float32(float("inf"))
        for it in range(TOP_K // 128):
            v = sHeapV[(t, it * 128 + tidx)]
            m = sel_f32(v < m, v, m)
        sScrAV[tidx] = m
        bar.arrive_and_wait()
        if tidx == 0:
            mm = sScrAV[0]
            for j in range(1, 128):
                mm = sel_f32(sScrAV[j] < mm, sScrAV[j], mm)
            sTheta[t] = mm
            sCnt[(t, 1)] = Int32(0)
        bar.arrive_and_wait()

    @cute.jit
    def _finalize_candidates(
        self, b, chunk, tidx, sHeapV, sHeapI, sOvfV, sOvfI,
        sScrAV, sScrAI, sBins, sAux, sCnt, sTheta, sFlags,
        mCandV, mCandI, mCandC, bar, dbg_ptr,
    ):
        for t in range(self.q_len):
            if b == 0 and tidx == 0:
                nc = cute.arch.grid_dim()[0]
                base = (t * nc + chunk) * 4
                cute.arch.store(dbg_ptr + base + 0, sCnt[(t, 0)], sem="relaxed", scope="sys")
                cute.arch.store(dbg_ptr + base + 1, sCnt[(t, 1)], sem="relaxed", scope="sys")
                cute.arch.store(dbg_ptr + base + 2, sCnt[(t, 3)], sem="relaxed", scope="sys")
                cute.arch.store(dbg_ptr + base + 3, f32_sort_key(sTheta[t]), sem="relaxed", scope="sys")

            total = sCnt[(t, 0)] + sCnt[(t, 1)]
            if tidx == 0:
                sFlags[t] = sel_i32(total > TOP_K, Int32(1), Int32(0))
            bar.arrive_and_wait()
            if sFlags[t] == 1:
                # pad partially-filled heap with -inf, then exact merge
                for it in cutlass.range(TOP_K // 128, unroll=1):
                    j = it * 128 + tidx
                    if j >= sCnt[(t, 0)]:
                        sHeapV[(t, j)] = Float32(-float("inf"))
                        sHeapI[(t, j)] = Int32(-1)
                bar.arrive_and_wait()
                self._merge_heap(
                    t, tidx, sHeapV, sHeapI, sOvfV, sOvfI,
                    sScrAV, sScrAI, sBins, sAux, sCnt, sTheta, bar,
                )
            else:
                # everything fits: append overflow into heap
                for it in cutlass.range(OVF_CAP // 128, unroll=1):
                    j = it * 128 + tidx
                    if j < sCnt[(t, 1)]:
                        sHeapV[(t, sCnt[(t, 0)] + j)] = sOvfV[(t, j)]
                        sHeapI[(t, sCnt[(t, 0)] + j)] = sOvfI[(t, j)]
                bar.arrive_and_wait()
            out_n = cutlass.min(total, Int32(TOP_K))
            for it in cutlass.range(TOP_K // 128, unroll=1):
                j = it * 128 + tidx
                if j < out_n:
                    mCandV[(b, t, chunk, j)] = sHeapV[(t, j)]
                    mCandI[(b, t, chunk, j)] = sHeapI[(t, j)]
            if tidx == 0:
                mCandC[(b, t, chunk)] = out_n
            bar.arrive_and_wait()


# ---------------------------------------------------------------- host runner

_compile_cache: dict = {}


def _ptr(dtype: Type[cutlass.Numeric], t: torch.Tensor, align: int) -> cute.Pointer:
    return make_ptr(dtype, t.data_ptr(), cute.AddressSpace.gmem, assumed_align=align)


def run_fp4_paged_mqa_logits(
    kv_fused: torch.Tensor,  # (num_pages, 8704) uint8 fused pages
    q_packed: torch.Tensor,  # (B, N, 64) uint8
    q_sf_atom: torch.Tensor,  # (B, 512) uint8
    weights: torch.Tensor,  # (B, N) fp32
    block_table: torch.Tensor,  # (B, max_pages) int32
    logits: torch.Tensor,  # (B*q_len, max_pages*128) fp32
    blocks_per_chunk: int,
    num_kv_stage: int = 8,
    seq_lens: torch.Tensor | None = None,  # (B,) int32 — fused_topk masking
    cand_v: torch.Tensor | None = None,  # (B, q_len, n_chunks, TOP_K) fp32
    cand_i: torch.Tensor | None = None,  # (B, q_len, n_chunks, TOP_K) int32
    cand_c: torch.Tensor | None = None,  # (B, q_len, n_chunks) int32
    dbg: torch.Tensor | None = None,  # (8,) int32 progress markers
    stream: cuda.CUstream | None = None,
) -> None:
    num_pages, B = kv_fused.shape[0], q_packed.shape[0]
    N = q_packed.shape[1]
    q_len = logits.shape[0] // B
    max_pages = block_table.shape[1]
    n_chunks = max_pages // blocks_per_chunk
    fused_topk = cand_v is not None
    if dbg is None:
        dbg = torch.zeros(8, dtype=torch.int32, device=kv_fused.device)
    if seq_lens is None:
        seq_lens = torch.full(
            (B,), max_pages * PAGE_TOKENS, dtype=torch.int32, device=kv_fused.device
        )
    if not fused_topk:
        cand_v = logits  # unused placeholders
        cand_i = block_table
        cand_c = block_table
    key = (N, q_len, blocks_per_chunk, num_kv_stage, fused_topk)
    if key not in _compile_cache:
        kernel = Fp4PagedMQALogits(
            n_heads=N // q_len,
            q_len=q_len,
            num_kv_stage=num_kv_stage,
            fused_topk=fused_topk,
        )
        compiled = cute.compile(
            kernel,
            make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=32),
            make_ptr(cutlass.Float8E8M0FNU, 0, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float4E2M1FN, 0, cute.AddressSpace.gmem, assumed_align=32),
            make_ptr(cutlass.Float8E8M0FNU, 0, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=16),
            (Int32(0), Int32(0), Int32(0), Int32(0), Int32(0)),
            cuda.CUstream(0),
        )
        _compile_cache[key] = compiled
    compiled = _compile_cache[key]
    if stream is None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    compiled(
        _ptr(cutlass.Float4E2M1FN, kv_fused, 32),
        make_ptr(
            cutlass.Float8E8M0FNU,
            kv_fused.data_ptr() + PAGE_DATA_BYTES,
            cute.AddressSpace.gmem,
            assumed_align=16,
        ),
        _ptr(cutlass.Float4E2M1FN, q_packed, 32),
        _ptr(cutlass.Float8E8M0FNU, q_sf_atom, 16),
        _ptr(cutlass.Float32, weights, 16),
        _ptr(cutlass.Int32, block_table, 16),
        _ptr(cutlass.Float32, logits, 16),
        _ptr(cutlass.Int32, seq_lens, 16),
        _ptr(cutlass.Float32, cand_v, 16),
        _ptr(cutlass.Int32, cand_i, 16),
        _ptr(cutlass.Int32, cand_c, 16),
        _ptr(cutlass.Int32, dbg, 16),
        (
            Int32(B),
            Int32(num_pages),
            Int32(blocks_per_chunk),
            Int32(max_pages),
            Int32(n_chunks),
        ),
        stream,
    )
