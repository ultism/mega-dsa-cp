"""Event system (Phase 0.1): u64 monotonic phase counters + scoped red/ld."""

from .table import (
    SCOPE_LOCAL,
    SCOPE_SYS,
    EventSpec,
    EventTable,
    WaitEdge,
    NotifyEdge,
    TaskSpec,
)
from .validator import validate

__all__ = [
    "SCOPE_LOCAL",
    "SCOPE_SYS",
    "EventSpec",
    "EventTable",
    "WaitEdge",
    "NotifyEdge",
    "TaskSpec",
    "validate",
]
