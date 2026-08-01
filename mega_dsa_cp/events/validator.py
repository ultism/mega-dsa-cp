"""CPU static validator for the event/task graph.

Runs before any kernel exists (and in CI without a GPU). Catches the bug
classes that are painful to debug on-device:

  1. unknown event references
  2. scope violations: a local event notified to a remote cell, or notified
     from a rank that does not own the target cell through a local edge
  3. waits with no producer (deadlock), notifies with no consumer (waste)
  4. dependency cycles (task -> event -> task graph must be a DAG)
  5. arity inconsistency: every cell consumed in the same event must have a
     well-defined per-phase notify count (it always does structurally — each
     notify edge contributes exactly 1 per phase — so this check is about
     consumers sharing a cell having a single arity, which is automatic)
  6. rank bounds

validate() returns (errors, EventTable). errors is empty on success.
"""

from .table import (
    SCOPE_LOCAL,
    SCOPE_SYS,
    EventSpec,
    EventTable,
    TaskSpec,
)


def validate(tasks: list, events: list, world_size: int) -> tuple:
    errors = []
    ev_by_name = {}
    for ev in events:
        if ev.name in ev_by_name:
            errors.append(f"duplicate event name: {ev.name}")
        ev_by_name[ev.name] = ev
        if ev.scope not in (SCOPE_LOCAL, SCOPE_SYS):
            errors.append(f"event {ev.name}: bad scope {ev.scope!r}")

    # per-cell notify counts: (event, dst_rank) -> count; wait cells: (event, rank)
    cell_arity = {}
    wait_cells = set()
    edges = []  # (producer_task, consumer_task) for cycle check

    task_by_name = {}
    for t in tasks:
        if t.name in task_by_name:
            errors.append(f"duplicate task name: {t.name}")
        task_by_name[t.name] = t
        if not (0 <= t.rank < world_size):
            errors.append(f"task {t.name}: rank {t.rank} out of range")

    for t in tasks:
        for w in t.waits:
            if w.event not in ev_by_name:
                errors.append(f"task {t.name}: waits unknown event {w.event!r}")
                continue
            wait_cells.add((w.event, t.rank))
        for n in t.notifies:
            if n.event not in ev_by_name:
                errors.append(f"task {t.name}: notifies unknown event {n.event!r}")
                continue
            if not (0 <= n.dst_rank < world_size):
                errors.append(
                    f"task {t.name}: notify {n.event} dst_rank {n.dst_rank} out of range"
                )
                continue
            ev = ev_by_name[n.event]
            if ev.scope == SCOPE_LOCAL and n.dst_rank != t.rank:
                errors.append(
                    f"task {t.name}: local-scope event {n.event} notified to "
                    f"remote rank {n.dst_rank} (scope violation, cf. gotchas #4)"
                )
            cell_arity[(n.event, n.dst_rank)] = cell_arity.get((n.event, n.dst_rank), 0) + 1

    # producer/consumer matching per cell
    for (ev_name, rank) in sorted(wait_cells):
        if (ev_name, rank) not in cell_arity:
            errors.append(
                f"event {ev_name}: waited on rank {rank} but never notified there (deadlock)"
            )
    for (ev_name, rank) in sorted(cell_arity):
        if (ev_name, rank) not in wait_cells:
            errors.append(
                f"event {ev_name}: notified on rank {rank} but never waited there"
            )

    # sys events should actually cross ranks somewhere; local events must not
    for ev in events:
        producers_ranks = {t.rank for t in tasks for n in t.notifies if n.event == ev.name}
        consumer_ranks = {t.rank for t in tasks for w in t.waits if w.event == ev.name}
        dst_ranks = {n.dst_rank for t in tasks for n in t.notifies if n.event == ev.name}
        crosses = any(r != d for r in producers_ranks for d in dst_ranks)
        if ev.scope == SCOPE_SYS and not crosses and world_size > 1:
            errors.append(f"event {ev.name}: declared sys but has no cross-rank edge")
        if ev.scope == SCOPE_LOCAL and (producers_ranks | consumer_ranks | dst_ranks) and (
            len(producers_ranks | consumer_ranks) > 1
        ):
            # multi-rank use of a local event is fine only if each rank's cell
            # is produced and consumed purely locally
            for r in producers_ranks | consumer_ranks:
                prod_local = any(
                    t.rank == r and n.dst_rank == r
                    for t in tasks
                    for n in t.notifies
                    if n.event == ev.name
                )
                cons_local = r in consumer_ranks
                if prod_local != cons_local:
                    errors.append(
                        f"event {ev.name}: local event has unbalanced edges on rank {r}"
                    )

    # cycle check on task graph (task -> task via any event cell)
    task_event_prod = {}  # event -> set(producer task names)
    for t in tasks:
        for n in t.notifies:
            task_event_prod.setdefault(n.event, set()).add(t.name)
    for t in tasks:
        for w in t.waits:
            for p in task_event_prod.get(w.event, ()):  # producer -> consumer
                if p != t.name:
                    edges.append((p, t.name))

    indeg = {t.name: 0 for t in tasks}
    adj = {t.name: [] for t in tasks}
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    queue = [n for n, d in indeg.items() if d == 0]
    seen = 0
    while queue:
        n = queue.pop()
        seen += 1
        for m in adj[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                queue.append(m)
    if seen != len(tasks):
        errors.append("dependency cycle detected in task graph")

    table = EventTable(events=ev_by_name, arities=cell_arity)
    return errors, table
