"""Device-side event cells: notify/wait on u64 monotonic phase counters.

Cell layout: event i lives at byte offset i * CELL_BYTES (128B stride) in the
rank-local symmetric buffer, so cells never share a cache line with a cell
written by remote ranks. Counters are monotonic: phase p completes when the
cell reaches `arity * (p + 1)`. No reset, no ABA, replay-safe in CUDA graphs.

Wait protocol: thread 0 of the calling CTA polls the local cell with scoped
acquire loads, then a CTA barrier publishes completion; for sys-scope waits
every thread issues an acquire fence after the barrier before touching the
payload (L1 is not coherent across GPUs — the fence is what makes remote
release-ordered writes visible to all threads of the CTA).

Notify protocol: each call site contributes exactly +1 to the cell, so the
CTA-level `notify_*` wrappers are guarded to thread 0. The raw `_1t` variants
are unguarded for caller-managed predication (e.g. a dedicated producer warp
in the megakernel) — the caller must ensure exactly one issuer per arity
contribution, or the monotonic counter overshoots and every consumer's
target becomes wrong.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint64
from cutlass.cute.typing import AddressSpace

from .ptx import nanosleep, red_add_u64, ld_acquire_u64, ld_relaxed_u64

CELL_BYTES = 128
CELL_ELEMS = CELL_BYTES // 8  # u64 elements per cell

SCOPE_LOCAL = "gpu"
SCOPE_SYS = "sys"

DEFAULT_BACKOFF_NS = 64


@dataclass(frozen=True)
class EventSet:
    """Device view over one rank's event buffer + the per-rank offset table.

    Constructed inside the kernel from raw pointer arguments. `offsets[r]`
    is (peer_r_base - my_base); mapping a local address to peer r is a single
    gmem L2 load + add (grid_constant const-bank table is a later optimization,
    see flashinfer cutedsl_megamoe sym_buffer.py).
    """

    base: cute.Pointer  # u64*, rank-local event buffer
    offsets: cute.Pointer  # i64*, length world_size
    my_rank: Int32

    @cute.jit
    def cell(self, event_id: Int32) -> cute.Pointer:
        return self.base + event_id * CELL_ELEMS

    @cute.jit
    def peer_cell(self, event_id: Int32, dst_rank: Int32) -> cute.Pointer:
        off = cute.arch.load(self.offsets + dst_rank, Int64, sem="relaxed", scope="gpu")
        addr = self.base.toint() + off + Int64(event_id) * CELL_BYTES
        return cute.make_ptr(Uint64, addr, AddressSpace.gmem, assumed_align=8)

    @cute.jit
    def notify_local_1t(self, event_id: Int32) -> None:
        """Raw local notify. CALLER must guarantee exactly one issuing thread
        (per arity contribution) — every call site adds +1 to the cell."""
        red_add_u64(self.cell(event_id), Uint64(1), SCOPE_LOCAL)

    @cute.jit
    def notify_peer_1t(self, event_id: Int32, dst_rank: Int32) -> None:
        """Raw cross-rank notify, same single-issuer contract as _1t above."""
        red_add_u64(self.peer_cell(event_id, dst_rank), Uint64(1), SCOPE_SYS)

    @cute.jit
    def notify_local(self, event_id: Int32) -> None:
        """CTA-level local notify: thread 0 issues a single +1."""
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            self.notify_local_1t(event_id)

    @cute.jit
    def notify_peer(self, event_id: Int32, dst_rank: Int32) -> None:
        """CTA-level cross-rank notify: thread 0 issues a single +1."""
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            self.notify_peer_1t(event_id, dst_rank)

    @cute.jit
    def notify(self, event_id: Int32, dst_rank: Int32) -> None:
        """Dispatch on dst == self: local notify avoids the sys-scope cost."""
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            if dst_rank == self.my_rank:
                self.notify_local_1t(event_id)
            else:
                self.notify_peer_1t(event_id, dst_rank)

    @cute.jit
    def wait(
        self,
        event_id: Int32,
        target: Uint64,
        scope: cutlass.Constexpr[str],
        backoff_ns: cutlass.Constexpr[int] = DEFAULT_BACKOFF_NS,
        poll_sem: cutlass.Constexpr[str] = "acquire",
        fence_after: cutlass.Constexpr[bool] = True,
    ) -> None:
        """CTA-level wait until the local cell reaches `target` (monotonic).

        poll_sem="acquire": every poll is ld.acquire (per-poll ordering).
        poll_sem="relaxed": polls are ld.relaxed; correctness comes from the
        post-detection acquire fence, which every thread issues anyway.
        fence_after=False skips that fence: ONLY valid when the consumer's
        payload reads carry their own ordering (e.g. scoped loads, TMA);
        flag-only signalling is fine without it.
        """
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == 0:
            p = self.cell(event_id)
            v = Uint64(0)
            while v < target:
                if cutlass.const_expr(poll_sem == "acquire"):
                    v = ld_acquire_u64(p, scope)
                else:
                    v = ld_relaxed_u64(p, scope)
                if v < target:
                    nanosleep(Int32(backoff_ns))
        cute.arch.barrier()
        if cutlass.const_expr(fence_after):
            if cutlass.const_expr(scope == SCOPE_SYS):
                cute.arch.fence_acq_rel_sys()
            else:
                cute.arch.fence_acq_rel_gpu()
