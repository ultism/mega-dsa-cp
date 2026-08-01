"""Peer-to-peer communication primitives (Phase 0.3).

Transport selection per payload class (docs/overview.md 通信清单):
  - small dense arrays (candidates, LSE scalars): multimem.st allgather,
    one store lands on every rank (NVLS); fallback = per-peer bulk push
  - large partials: bulk S2G push to peer / multimem.ld_reduce (fp32)
  - flags: event system (Phase 0.1), flag-only waits; payload readers carry
    their own ordering (scoped loads) so waits stay fence-free

Data+flag ordering chain (gotchas #3): bulk S2G is async-proxy — between
`cp.async.bulk.wait_group 0` and the notify red there must be a
`fence.proxy.async.global`. push_start/push_finish are split so callers can
overlap other work between commit and wait (never wait right after commit).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm

from cutlass.utils.distributed import (  # noqa: F401  (re-export)
    multimem_st,
    multimem_ld_reduce,
    multimem_red_release_sys_add1,
)


@dsl_user_op
def cp_async_bulk_s2g(dst_ptr: cute.Pointer, src_smem_ptr: cute.Pointer, nbytes: Int32, *, loc=None, ip=None) -> None:
    """`cp.async.bulk.global.shared::cta.bulk_group [dst], [src], size`
    1D bulk SMEM->GMEM copy (no tensor descriptor). Single issuing thread.
    Mirrors cutedsl_megamoe ptx_helpers.cp_async_bulk_s2g."""
    llvm.inline_asm(
        None,
        [
            dst_ptr.toint(loc=loc, ip=ip).ir_value(),
            src_smem_ptr.toint(loc=loc, ip=ip).ir_value(),
            nbytes.ir_value(),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_bulk_commit(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "cp.async.bulk.commit_group;",
        "",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_bulk_wait0(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "cp.async.bulk.wait_group 0;",
        "",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@cute.jit
def push_start(dst_ptr: cute.Pointer, src_smem_ptr: cute.Pointer, nbytes: Int32) -> None:
    """Issue an async bulk push to a peer address. Single thread only."""
    cp_async_bulk_s2g(dst_ptr, src_smem_ptr, nbytes)
    cp_async_bulk_commit()


@cute.jit
def push_finish() -> None:
    """Complete the outstanding push and order it before a following notify.

    Between start and finish the caller should do unrelated work; the
    wait_group exposes the S2G completion latency (~1-2us)."""
    cp_async_bulk_wait0()
    cute.arch.fence_proxy("async.global")
