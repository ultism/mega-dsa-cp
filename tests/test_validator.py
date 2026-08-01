"""CPU validator tests — runnable with plain python3, no GPU deps.

    python3 tests/test_validator.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mega_dsa_cp.events import (
    SCOPE_LOCAL,
    SCOPE_SYS,
    EventSpec,
    NotifyEdge,
    TaskSpec,
    WaitEdge,
    validate,
)
from dag_gen import random_dag


def check(name, errors, expect_ok):
    ok = (len(errors) == 0) == expect_ok
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if not ok:
        for e in errors:
            print(f"    {e}")
    return ok


def main():
    all_ok = True

    # valid random DAGs must validate clean
    for seed in range(50):
        tasks, events, _, _, _ = random_dag(seed, n_tasks=64, edge_prob=0.05, n_ctas=8)
        errors, table = validate(tasks, events, world_size=1)
        all_ok &= check(f"random dag seed={seed}", errors, True)
        # arity sanity: cell arity equals in-degree
        for ev in events:
            t = int(ev.name[1:])
            assert table.arities[(ev.name, 0)] == len(
                [x for x in tasks if any(n.event == ev.name for n in x.notifies)]
            )

    # cycle: a waits e1 notified by b; b waits e2 notified by a
    cyc_tasks = [
        TaskSpec("a", 0, waits=(WaitEdge("e1"),), notifies=(NotifyEdge("e2", 0),)),
        TaskSpec("b", 0, waits=(WaitEdge("e2"),), notifies=(NotifyEdge("e1", 0),)),
    ]
    cyc_events = [EventSpec("e1", SCOPE_LOCAL), EventSpec("e2", SCOPE_LOCAL)]
    errors, _ = validate(cyc_tasks, cyc_events, 1)
    all_ok &= check("cycle detected", errors, False)

    # unknown event
    bad = [TaskSpec("a", 0, waits=(WaitEdge("nope"),))]
    errors, _ = validate(bad, [], 1)
    all_ok &= check("unknown event", errors, False)

    # wait with no producer
    wp_tasks = [TaskSpec("a", 0, waits=(WaitEdge("e"),))]
    errors, _ = validate(wp_tasks, [EventSpec("e", SCOPE_LOCAL)], 1)
    all_ok &= check("wait without producer", errors, False)

    # notify with no consumer
    nc_tasks = [TaskSpec("a", 0, notifies=(NotifyEdge("e", 0),))]
    errors, _ = validate(nc_tasks, [EventSpec("e", SCOPE_LOCAL)], 1)
    all_ok &= check("notify without consumer", errors, False)

    # local event notified cross-rank
    xr_tasks = [
        TaskSpec("a", 0, notifies=(NotifyEdge("e", 1),)),
        TaskSpec("b", 1, waits=(WaitEdge("e"),)),
    ]
    errors, _ = validate(xr_tasks, [EventSpec("e", SCOPE_LOCAL)], 2)
    all_ok &= check("local event cross-rank notify", errors, False)

    # valid sys event across two ranks
    sys_tasks = [
        TaskSpec("a", 0, notifies=(NotifyEdge("e", 1),)),
        TaskSpec("b", 1, waits=(WaitEdge("e"),), notifies=(NotifyEdge("f", 0),)),
        TaskSpec("c", 0, waits=(WaitEdge("f"),)),
    ]
    sys_events = [EventSpec("e", SCOPE_SYS), EventSpec("f", SCOPE_SYS)]
    errors, _ = validate(sys_tasks, sys_events, 2)
    all_ok &= check("valid sys ping-pong", errors, True)

    # sys event with no cross-rank edge (world>1)
    fs_tasks = [
        TaskSpec("a", 0, notifies=(NotifyEdge("e", 0),)),
        TaskSpec("b", 0, waits=(WaitEdge("e"),)),
    ]
    errors, _ = validate(fs_tasks, [EventSpec("e", SCOPE_SYS)], 2)
    all_ok &= check("sys without cross-rank edge", errors, False)

    # dst_rank out of range
    oob = [TaskSpec("a", 0, notifies=(NotifyEdge("e", 7),))]
    errors, _ = validate(oob, [EventSpec("e", SCOPE_SYS)], 2)
    all_ok &= check("dst_rank out of range", errors, False)

    print("ALL PASS" if all_ok else "FAILURES PRESENT")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
