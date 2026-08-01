"""Low-level PTX primitives for the event system.

Design rules (see docs/gotchas.md):
  - notify is always a posted `red` (fire-and-forget), never a round-trip `atom`.
  - cross-rank ops are sys scope; intra-rank ops are gpu scope.
  - waits poll a LOCAL cell with scoped acquire loads; only one thread polls,
    then a CTA barrier publishes the result.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint64
from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def nanosleep(ns: Int32, *, loc=None, ip=None) -> None:
    """`nanosleep.u32` — suspend the current thread for ~ns nanoseconds."""
    llvm.inline_asm(
        None,
        [Int32(ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )


@cute.jit
def red_add_u64(ptr: cute.Pointer, val: Uint64, scope: cutlass.Constexpr[str]) -> None:
    """Posted notify: `red.release.<scope>.global.add.u64`. Fire-and-forget."""
    cute.arch.red(ptr, val, op="add", dtype="u64", sem="release", scope=scope)


@cute.jit
def ld_acquire_u64(ptr: cute.Pointer, scope: cutlass.Constexpr[str]) -> Uint64:
    """One poll of a wait loop: `ld.acquire.<scope>.global.u64`."""
    return cute.arch.load(ptr, Uint64, sem="acquire", scope=scope)


@cute.jit
def ld_relaxed_u64(ptr: cute.Pointer, scope: cutlass.Constexpr[str]) -> Uint64:
    """Cheaper poll: `ld.relaxed.<scope>.global.u64`. Only correct when the
    wait does a full acquire fence after detection (see EventSet.wait)."""
    return cute.arch.load(ptr, Uint64, sem="relaxed", scope=scope)


@cute.jit
def st_relaxed_u64(ptr: cute.Pointer, val: Uint64, scope: cutlass.Constexpr[str]) -> None:
    """Payload write. Must be followed by a release (red/fence) before the flag."""
    cute.arch.store(ptr, val, sem="relaxed", scope=scope)


@cute.jit
def atomic_add_u64_rt(ptr: cute.Pointer, val: Uint64, scope: cutlass.Constexpr[str]) -> Uint64:
    """Round-trip `atom.acq_rel.<scope>.global.add.u64` (returns old).

    Banned on cross-rank critical paths (2-4us round trip, see gotchas.md #1).
    Legit uses: gpu-scope producer-side triggers, benchmarks.
    """
    return cute.arch.atomic_add(ptr, val, sem="acq_rel", scope=scope)


@dsl_user_op
def multimem_red_add1_s32(mc_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    """`multimem.red.release.sys.global.add.s32 [mc], 1` — switch-side fan-in.

    One instruction lands on every rank's copy of the cell. Requires NVLS
    hardware and a multicast pointer. 32-bit only (gotchas.md #2).
    """
    llvm.inline_asm(
        None,
        [mc_ptr.toint().ir_value(loc=loc, ip=ip)],
        "multimem.red.release.sys.global.add.s32 [$0], 1;",
        "l",
        has_side_effects=True,
        asm_dialect=0,
        loc=loc,
        ip=ip,
    )
