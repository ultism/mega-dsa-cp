"""Static queue codegen: validated one-layer DAG -> per-CTA packed queues.

Queue format: each CTA owns an int32 word stream. A task entry is
    [header][coord0][coord1][coord2][wait_ev...][notify_ev...]
with
    header = op[7:0] | n_waits[13:8] | n_notifies[19:14] | extent_src[27:20]
    wait_ev = event_id (scope/arity looked up in the shared event tables)
    notify_ev = (event_id << 8) | dst_rank
Terminated by an OP_END header. Descriptors are dealt round-robin from a
topological order, so every CTA's subsequence is itself topological and the
static schedule is deadlock-free by construction.

Pure host code (no torch/cutlass) so it is CPU-testable. The same function
runs on every rank with identical inputs -> identical shared event tables
(topological symmetry required by event-system.md section 6).
"""

from dataclasses import dataclass

from ..events.table import TaskSpec
from ..events.validator import validate

OP_END = 0xFF
MAX_EDGES_PER_TASK = 63  # 6-bit field
MAX_EXTENT_SRC = 255  # 8-bit field


@dataclass
class Schedule:
    queues: list  # list per CTA: list[int32] incl. trailing OP_END
    arity: list  # per event id: notify count per phase on THIS rank's cell
    scopes: list  # per event id: 0=local, 1=sys
    topo: list  # task indices in topological order
    op_of: list  # per task index: op code (from extents/op assignment)
    n_events: int


def pack_header(op: int, n_waits: int, n_notifies: int, extent_src: int) -> int:
    return (
        (op & 0xFF)
        | ((n_waits & 0x3F) << 8)
        | ((n_notifies & 0x3F) << 14)
        | ((extent_src & 0xFF) << 20)
    )


def topo_sort(tasks: list, events: list) -> list:
    prod = {}
    for i, t in enumerate(tasks):
        for n in t.notifies:
            prod.setdefault(n.event, set()).add(i)
    adj = {i: [] for i in range(len(tasks))}
    indeg = {i: 0 for i in range(len(tasks))}
    for j, t in enumerate(tasks):
        for w in t.waits:
            for i in prod.get(w.event, ()):  # producer -> consumer
                if i != j:
                    adj[i].append(j)
                    indeg[j] += 1
    queue = [i for i, d in indeg.items() if d == 0]
    order = []
    while queue:
        i = queue.pop(0)
        order.append(i)
        for j in adj[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                queue.append(j)
    if len(order) != len(tasks):
        raise ValueError("dependency cycle (validator should have caught this)")
    return order


def codegen_schedule(
    tasks: list,
    events: list,
    rank: int,
    n_ctas: int,
    world_size: int = 1,
    op_of: dict = None,
    extent_of: dict = None,
) -> Schedule:
    """Build this rank's per-CTA queues from a validated multi-rank DAG.

    tasks/events cover ALL ranks (shared event ids); only tasks with
    TaskSpec.rank == rank enter this rank's queues. op_of/extent_of map task
    name -> op code / dynamic-extent scalar id (default 0 = none).
    """
    errors, table = validate(tasks, events, world_size=world_size)
    if errors:
        raise ValueError(f"invalid DAG: {errors}")

    op_of = op_of or {}
    extent_of = extent_of or {}
    ev_id = {ev.name: i for i, ev in enumerate(events)}

    my_idx = [i for i, t in enumerate(tasks) if t.rank == rank]
    order = [i for i in topo_sort(tasks, events) if i in set(my_idx)]

    queues = [[] for _ in range(n_ctas)]
    for dealt, i in enumerate(order):
        t = tasks[i]
        waits = [ev_id[w.event] for w in t.waits]
        notifies = [(ev_id[n.event] << 8) | n.dst_rank for n in t.notifies]
        assert len(waits) <= MAX_EDGES_PER_TASK, f"task {t.name}: too many waits"
        assert len(notifies) <= MAX_EDGES_PER_TASK, f"task {t.name}: too many notifies"
        op = op_of.get(t.name, 0)
        extent = extent_of.get(t.name, 0)
        assert 0 <= extent <= MAX_EXTENT_SRC
        c0, c1, c2 = t.coords if t.coords else (i, 0, 0)
        entry = [pack_header(op, len(waits), len(notifies), extent), c0, c1, c2]
        entry += waits
        entry += notifies
        queues[dealt % n_ctas].extend(entry)
    for q in queues:
        q.append(pack_header(OP_END, 0, 0, 0))

    arity = [table.arities.get((ev.name, rank), 0) for ev in events]
    scopes = [0 if ev.scope == "local" else 1 for ev in events]
    op_list = [op_of.get(t.name, 0) for t in tasks]
    return Schedule(
        queues=queues,
        arity=arity,
        scopes=scopes,
        topo=order,
        op_of=op_list,
        n_events=len(events),
    )
