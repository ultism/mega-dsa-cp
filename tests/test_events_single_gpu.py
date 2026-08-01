"""Single-GPU equivalence test: random DAG over the event system.

Runs a randomized DAG (tests/dag_gen.py) on one GPU through the real event
primitives (gpu scope), with a static round-robin schedule. The kernel logs
the global execution order; the host checks it is a valid linearization of
the DAG — any missing or mistimed notify/wait shows up as an ordering
violation or a hang.

    python3 tests/test_events_single_gpu.py [seed]
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Uint64
from cutlass.cute.runtime import make_ptr
from cutlass.cute.typing import AddressSpace

from mega_dsa_cp.events import validate
from mega_dsa_cp.events.core import EventSet, SCOPE_LOCAL as GPU_SCOPE
from mega_dsa_cp.events.symm import alloc_event_buffer
from mega_dsa_cp.events.ptx import nanosleep
from dag_gen import random_dag

N_THREADS = 128


@cute.kernel
def dag_kernel(
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    sched: cute.Pointer,  # i32[num_ctas * sched_len], row-major per CTA
    sched_counts: cute.Pointer,  # i32[num_ctas]
    waits_flat: cute.Pointer,  # i32 event ids
    waits_off: cute.Pointer,  # i32[n_tasks + 1]
    notifies_flat: cute.Pointer,  # i32 event ids
    notifies_off: cute.Pointer,  # i32[n_tasks + 1]
    arity: cute.Pointer,  # i32[n_events]
    log_buf: cute.Pointer,  # i32[n_tasks]
    log_count: cute.Pointer,  # i32[1]
    n_ctas: cutlass.Constexpr[int],
    sched_len: cutlass.Constexpr[int],
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    events = EventSet(base=ev_base, offsets=ev_offsets, my_rank=Int32(0))

    count = cute.arch.load(sched_counts + bidx, Int32)
    for k in cutlass.range(count):
        t = cute.arch.load(sched + bidx * sched_len + k, Int32)

        w0 = cute.arch.load(waits_off + t, Int32)
        w1 = cute.arch.load(waits_off + t + 1, Int32)
        for w in cutlass.range(w1 - w0):
            ev = cute.arch.load(waits_flat + w0 + w, Int32)
            a = cute.arch.load(arity + ev, Int32)
            events.wait(ev, Uint64(a), GPU_SCOPE)

        if tidx == 0:
            pos = cute.arch.atomic_add(log_count, Int32(1), sem="acq_rel", scope="gpu")
            cute.arch.store(log_buf + pos, t)
            nanosleep(Int32((t * 37) % 200))

        n0 = cute.arch.load(notifies_off + t, Int32)
        n1 = cute.arch.load(notifies_off + t + 1, Int32)
        for n in cutlass.range(n1 - n0):
            ev = cute.arch.load(notifies_flat + n0 + n, Int32)
            events.notify_local(ev)


@cute.jit
def launch(
    ev_base: cute.Pointer,
    ev_offsets: cute.Pointer,
    sched: cute.Pointer,
    sched_counts: cute.Pointer,
    waits_flat: cute.Pointer,
    waits_off: cute.Pointer,
    notifies_flat: cute.Pointer,
    notifies_off: cute.Pointer,
    arity: cute.Pointer,
    log_buf: cute.Pointer,
    log_count: cute.Pointer,
    n_ctas: cutlass.Constexpr[int],
    sched_len: cutlass.Constexpr[int],
):
    dag_kernel(
        ev_base,
        ev_offsets,
        sched,
        sched_counts,
        waits_flat,
        waits_off,
        notifies_flat,
        notifies_off,
        arity,
        log_buf,
        log_count,
        n_ctas,
        sched_len,
    ).launch(grid=[n_ctas, 1, 1], block=[N_THREADS, 1, 1])


def i32(t):
    return t.to(torch.int32).cuda()


def ptr(dtype, tensor, align=8):
    return make_ptr(dtype, tensor.data_ptr(), AddressSpace.gmem, assumed_align=align)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    n_tasks, n_ctas, edge_prob = 512, 8, 0.02

    tasks, events, schedule, preds, arity_map = random_dag(
        seed, n_tasks, edge_prob, n_ctas
    )
    errors, table = validate(tasks, events, world_size=1)
    assert not errors, errors

    ev_names = [ev.name for ev in events]
    ev_id = {name: i for i, name in enumerate(ev_names)}
    n_events = len(ev_names)

    sched_len = max(len(s) for s in schedule)
    sched = torch.full((n_ctas * sched_len,), -1, dtype=torch.int32)
    counts = torch.zeros(n_ctas, dtype=torch.int32)
    for c, row in enumerate(schedule):
        counts[c] = len(row)
        for k, t in enumerate(row):
            sched[c * sched_len + k] = t

    waits_flat, waits_off = [], [0]
    notifies_flat, notifies_off = [], [0]
    for t in tasks:
        waits_flat += [ev_id[w.event] for w in t.waits]
        waits_off.append(len(waits_flat))
        notifies_flat += [ev_id[n.event] for n in t.notifies]
        notifies_off.append(len(notifies_flat))
    arity = torch.zeros(max(n_events, 1), dtype=torch.int32)
    for name, i in ev_id.items():
        arity[i] = table.arities[(name, 0)]

    buf = alloc_event_buffer(max(n_events, 1), rank=0, world_size=1)
    log_buf = torch.full((n_tasks,), -1, dtype=torch.int32).cuda()
    log_count = torch.zeros(1, dtype=torch.int32).cuda()

    keep = [
        sched.cuda(),
        counts.cuda(),
        i32(torch.tensor(waits_flat or [0])),
        i32(torch.tensor(waits_off)),
        i32(torch.tensor(notifies_flat or [0])),
        i32(torch.tensor(notifies_off)),
        arity.cuda(),
    ]
    args = (
        ptr(Uint64, buf.tensor, 128),
        ptr(cutlass.Int64, buf.offsets),
        ptr(Int32, keep[0]),
        ptr(Int32, keep[1]),
        ptr(Int32, keep[2]),
        ptr(Int32, keep[3]),
        ptr(Int32, keep[4]),
        ptr(Int32, keep[5]),
        ptr(Int32, keep[6]),
        ptr(Int32, log_buf),
        ptr(Int32, log_count),
    )
    compiled = cute.compile(launch, *args, n_ctas, sched_len)
    compiled(*args)
    torch.cuda.synchronize()

    order = log_buf.cpu().tolist()
    assert log_count.item() == n_tasks, f"ran {log_count.item()} != {n_tasks} tasks"
    assert sorted(order) == list(range(n_tasks)), "task set mismatch"

    position = {t: i for i, t in enumerate(order)}
    violations = [
        (p, t)
        for t, ps in preds.items()
        for p in ps
        if position[p] >= position[t]
    ]
    assert not violations, f"ordering violations: {violations[:10]}"
    print(
        f"PASS seed={seed}: {n_tasks} tasks, {n_events} events, "
        f"{sum(len(v) for v in preds.values())} edges, valid linearization"
    )


if __name__ == "__main__":
    main()
