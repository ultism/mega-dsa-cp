"""CPU tests for static queue codegen (pure python).

    python3 tests/test_scheduler_codegen.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mega_dsa_cp.schedule.codegen import OP_END, codegen_schedule
from dag_gen import random_dag


def unpack_header(w):
    return w & 0xFF, (w >> 8) & 0x3F, (w >> 14) & 0x3F, (w >> 20) & 0xFF


def parse_queue(words):
    """Walk a queue word stream -> list of (op, coord0, waits, notifies, extent)."""
    out = []
    i = 0
    while True:
        op, nw, nn, ext = unpack_header(words[i])
        if op == OP_END:
            return out
        coord0 = words[i + 1]
        waits = words[i + 4 : i + 4 + nw]
        notifies = words[i + 4 + nw : i + 4 + nw + nn]
        out.append((op, coord0, waits, notifies, ext))
        i += 4 + nw + nn


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail and not cond else ""))
    return cond


def main():
    ok = True
    for seed in range(30):
        n_tasks, n_ctas = 200, 8
        tasks, events, _, preds, arity_map = random_dag(seed, n_tasks, 0.03, n_ctas)
        op_of = {t.name: (i % 3) + 1 for i, t in enumerate(tasks)}
        extent_of = {tasks[0].name: 1}
        sched = codegen_schedule(tasks, events, rank=0, n_ctas=n_ctas, op_of=op_of, extent_of=extent_of)

        # every task placed exactly once; per-CTA order is topological
        placed = []
        for q in sched.queues:
            entries = parse_queue(q)
            placed += [e[1] for e in entries]
            pos = {t: k for k, t in enumerate(e[1] for e in entries)}
            for t, ps in preds.items():
                for p in ps:
                    if t in pos and p in pos:
                        ok &= check(f"seed{seed} per-CTA topo", pos[p] < pos[t], f"p{p} after t{t}")
        ok &= check(f"seed{seed} all tasks once", sorted(placed) == list(range(n_tasks)))

        # edge tables match the DAG; arity matches in-degree
        ev_id = {ev.name: i for i, ev in enumerate(events)}
        for q in sched.queues:
            for op, coord0, waits, notifies, ext in parse_queue(q):
                t = tasks[coord0]
                ok &= check(
                    f"seed{seed} waits match",
                    waits == [ev_id[w.event] for w in t.waits],
                )
                ok &= check(
                    f"seed{seed} notifies match",
                    notifies == [(ev_id[n.event] << 8) | n.dst_rank for n in t.notifies],
                )
                ok &= check(f"seed{seed} op match", op == op_of[t.name])
        for t, a in arity_map.items():
            ok &= check(
                f"seed{seed} arity t{t}",
                sched.arity[ev_id[f"e{t}"]] == a,
            )
        ok &= check(f"seed{seed} extent", any(
            e[4] == 1 for q in sched.queues for e in parse_queue(q)
        ))
        ok &= check(f"seed{seed} scopes", all(s in (0, 1) for s in sched.scopes))

    # multi-rank: rank filtering + symmetric tables
    from mega_dsa_cp.events import EventSpec, NotifyEdge, TaskSpec, WaitEdge, SCOPE_SYS

    mr_tasks = [
        TaskSpec("a", 0, notifies=(NotifyEdge("e", 1),)),
        TaskSpec("b", 1, waits=(WaitEdge("e"),), notifies=(NotifyEdge("f", 0),)),
        TaskSpec("c", 0, waits=(WaitEdge("f"),)),
    ]
    mr_events = [EventSpec("e", SCOPE_SYS), EventSpec("f", SCOPE_SYS)]
    s0 = codegen_schedule(mr_tasks, mr_events, rank=0, n_ctas=2, world_size=2)
    s1 = codegen_schedule(mr_tasks, mr_events, rank=1, n_ctas=2, world_size=2)
    q0 = [e[1] for q in s0.queues for e in parse_queue(q)]
    q1 = [e[1] for q in s1.queues for e in parse_queue(q)]
    ok &= check("rank0 tasks {a,c}", sorted(q0) == [0, 2])
    ok &= check("rank1 tasks {b}", q1 == [1])
    ok &= check("arity symmetric", s0.arity == [0, 1] and s1.arity == [1, 0])

    print("ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
