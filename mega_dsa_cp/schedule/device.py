"""Device-side static queue scheduler (Phase 0.2).

Each CTA stages its queue row from gmem to smem once, then walks it with
zero gmem traffic per fetch (tirx StaticTileScheduler pattern, adapted to
our packed descriptors with inline wait/notify edges).

Usage in the kernel main loop (mirrors cutlass persistent tile scheduler
idioms — state carried in explicit locals, no hidden mutation across staged
control-flow regions):

    sched = StaticScheduler(...)
    cur = sched.init()        # stage row, read phase scalar
    t = cur.read(Int32(0))    # first descriptor
    while t.valid:
        cur.waits(t)          # CTA-level event waits (scope dispatched)
        # ... handler dispatch on t.op, coords t.c0/c1/c2, cur.extent(t)
        cur.notifies(t)       # single-issuer event notifies
        t = cur.advance(t)    # next descriptor

Wait targets are monotonic: target = arity[event] * (phase + 1), where phase
is scalars[0] read once at init. The runner (or a graph node) bumps it
between launches; nothing is ever reset.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint64

from ..events.core import EventSet, SCOPE_LOCAL, SCOPE_SYS
from .codegen import OP_END


@dataclass(frozen=True)
class TaskInfo:
    valid: cutlass.Boolean  # dynamic: header op != OP_END
    op: Int32
    n_waits: Int32
    n_notifies: Int32
    extent_src: Int32
    c0: Int32
    c1: Int32
    c2: Int32
    pos: Int32  # word offset of this entry's header in the smem queue


@dataclass(frozen=True)
class StaticScheduler:
    queue: cute.Pointer  # i32 gmem, [n_ctas * row_words]
    scalars: cute.Pointer  # u64: [0]=phase, [1:]=dynamic extents
    arity: cute.Pointer  # i32 per event (this rank's cells)
    scopes: cute.Pointer  # i32 per event: 0=local, 1=sys
    events: EventSet
    row_words: cutlass.Constexpr[int]
    n_threads: cutlass.Constexpr[int]

    @cute.jit
    def init(self) -> "Cursor":
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        qbuf = smem.allocate_tensor(Int32, self.row_words, byte_alignment=16)
        row = self.queue + bidx * self.row_words
        for i in cutlass.range_constexpr(self.row_words // self.n_threads + 1):
            idx = i * self.n_threads + tidx
            if idx < self.row_words:
                qbuf[idx] = cute.arch.load(row + idx, Int32)
        cute.arch.barrier()
        phase = cute.arch.load(self.scalars, Uint64, sem="relaxed", scope="gpu")
        return Cursor(buf=qbuf, sched=self, phase=phase)


@dataclass(frozen=True)
class Cursor:
    """Queue walker + per-task wait/notify executor."""

    buf: cute.Tensor  # smem int32[row_words]
    sched: StaticScheduler
    phase: Uint64

    @cute.jit
    def read(self, pos: Int32) -> TaskInfo:
        w = self.buf[pos]
        op = w & 0xFF
        return TaskInfo(
            valid=(op != OP_END),
            op=op,
            n_waits=(w >> 8) & 0x3F,
            n_notifies=(w >> 14) & 0x3F,
            extent_src=(w >> 20) & 0xFF,
            c0=self.buf[pos + 1],
            c1=self.buf[pos + 2],
            c2=self.buf[pos + 3],
            pos=pos,
        )

    @cute.jit
    def extent(self, t: TaskInfo) -> Int32:
        """Dynamic trip-count bound for this task (0 when extent_src == 0)."""
        v = Int32(0)
        if t.extent_src != 0:
            v = Int32(
                cute.arch.load(self.sched.scalars + t.extent_src, Uint64, sem="relaxed", scope="gpu")
                & Uint64(0xFFFFFFFF)
            )
        return v

    @cute.jit
    def waits(self, t: TaskInfo) -> None:
        for w in cutlass.range(t.n_waits):
            ev = self.buf[t.pos + 4 + w]
            ar = cute.arch.load(self.sched.arity + ev, Int32)
            sc = cute.arch.load(self.sched.scopes + ev, Int32)
            target = Uint64(ar) * (self.phase + Uint64(1))
            if sc == 0:
                self.sched.events.wait(ev, target, SCOPE_LOCAL)
            else:
                self.sched.events.wait(ev, target, SCOPE_SYS)

    @cute.jit
    def notifies(self, t: TaskInfo) -> None:
        for n in cutlass.range(t.n_notifies):
            e = self.buf[t.pos + 4 + t.n_waits + n]
            self.sched.events.notify(e >> 8, e & 0xFF)

    @cute.jit
    def advance(self, t: TaskInfo) -> TaskInfo:
        return self.read(t.pos + 4 + t.n_waits + t.n_notifies)
