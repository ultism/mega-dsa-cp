"""Host-side event/task metadata: the frozen task-graph description.

An event has one cell per rank. A consumer on rank r always waits on rank
r's own cell; producers notify the cells of every consuming rank. The
per-phase arity of cell (event, r) is the number of notify edges targeting
it per phase — consumers wait for `arity * (phase + 1)` (monotonic u64).

These structures are pure host metadata (no torch/cutlass imports) so the
validator and tests run without a GPU environment.
"""

from dataclasses import dataclass, field

SCOPE_LOCAL = "local"  # all producers/consumers on the same rank (gpu scope)
SCOPE_SYS = "sys"  # at least one cross-rank edge (sys scope)


@dataclass(frozen=True)
class EventSpec:
    name: str
    scope: str  # SCOPE_LOCAL | SCOPE_SYS


@dataclass(frozen=True)
class WaitEdge:
    event: str


@dataclass(frozen=True)
class NotifyEdge:
    event: str
    dst_rank: int


@dataclass(frozen=True)
class TaskSpec:
    name: str
    rank: int
    waits: tuple = ()  # tuple[WaitEdge, ...]
    notifies: tuple = ()  # tuple[NotifyEdge, ...]
    coords: tuple = ()  # optional (c0, c1, c2) packed into the descriptor;
    # empty -> codegen fills (global_task_index, 0, 0)


@dataclass
class EventTable:
    """Validated event table: name -> EventSpec, plus per-cell arities."""

    events: dict = field(default_factory=dict)  # name -> EventSpec
    arities: dict = field(default_factory=dict)  # (event, rank) -> int

    def event_id(self, name: str) -> int:
        return list(self.events).index(name)
