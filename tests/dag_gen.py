"""Random DAG generator for the equivalence tests (pure python, no GPU deps).

Generates a task graph where each consumer task owns one fan-in event whose
arity is its in-degree; producers notify that event once per edge. Tasks are
numbered in topological order, then assigned round-robin to CTAs — each CTA's
subsequence is itself a topological order of its tasks, so the static
schedule is deadlock-free by construction.
"""

import random

from mega_dsa_cp.events.table import (
    SCOPE_LOCAL,
    EventSpec,
    NotifyEdge,
    TaskSpec,
    WaitEdge,
)


def random_dag(seed: int, n_tasks: int, edge_prob: float, n_ctas: int):
    rng = random.Random(seed)
    order = list(range(n_tasks))
    rng.shuffle(order)
    pos = {t: i for i, t in enumerate(order)}

    preds = {t: [] for t in range(n_tasks)}
    for j in range(n_tasks):
        for i in range(n_tasks):
            if pos[i] < pos[j] and rng.random() < edge_prob:
                preds[j].append(i)

    events = []
    tasks = []
    for t in range(n_tasks):
        ps = sorted(preds[t], key=lambda p: pos[p])
        waits = ()
        if ps:
            ev = f"e{t}"
            events.append(EventSpec(name=ev, scope=SCOPE_LOCAL))
            waits = (WaitEdge(event=ev),)
        tasks.append(
            TaskSpec(
                name=f"t{t}",
                rank=0,
                waits=waits,
                notifies=tuple(
                    NotifyEdge(event=f"e{j}", dst_rank=0)
                    for j in range(n_tasks)
                    if t in preds[j]
                ),
            )
        )

    # round-robin in topological order -> per-CTA subsequences are topological
    schedule = [[] for _ in range(n_ctas)]
    for i, t in enumerate(order):
        schedule[i % n_ctas].append(t)

    arity = {t: len(preds[t]) for t in range(n_tasks) if preds[t]}
    return tasks, events, schedule, preds, arity
