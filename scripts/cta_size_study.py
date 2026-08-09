#!/usr/bin/env python3
"""CTA sizing / occupancy / static-schedule algebra study (Phase 1.4a pre-work).

Pure stdlib. Answers, for the unified megakernel block-size decision:
  1. feasibility: which block sizes can host every tile (threads/regs/smem)
  2. occupancy:   worker CTAs per SM under unified smem = max tile smem
  3. task mix:    per-layer-type task counts as f(B, q, S, cp)
  4. makespan:    static-deal simulation (round-robin vs greedy), with and
                  without intra-CTA warp-group packing of small tiles
  5. critical path: event-chain latency per layer; x61 layers = step latency

All per-task latencies are ANALYTICAL ESTIMATES (bandwidth terms + fixed
overhead). Run with --lat-scale to see sensitivity. Exact regs/smem per tile
need a GPU probe (cuFuncGetAttribute on the compiled standalone kernels) --
marked EST below; see docs/phase1.4 design doc for the probe plan.

Sources for constants: tiles/hca.py:92,129-137,351 (512thr, cluster(2,1,1),
CtaGroup.TWO, 208KB); tiles/logits.py:9 (192thr); tiles/compress.py:36
(128thr); docs/event-system.md; docs/overview.md (comm manifest).
"""

from dataclasses import dataclass
import argparse

# ---------------------------------------------------------------- hardware


@dataclass
class HW:
    sms: int = 148                 # B200
    smem_per_cta_max: int = 232_448  # 227 KB opt-in ceiling
    regs_per_sm: int = 65_536
    hbm_bw: float = 6.0e12         # B/s effective stream
    nvl_push_bw: float = 4.0e11    # B/s per-rank directional push
    sys_hop_us: float = 2.85       # cross-rank notify->wait (measured, Phase 0)
    loc_hop_us: float = 1.1        # local notify->wait (measured)
    task_ovh_us: float = 1.2       # dispatch + wait + notify floor per task


# ---------------------------------------------------------------- tiles

KB = 1024


@dataclass
class Tile:
    name: str
    threads: int
    ctas: int          # 2 = cluster task (CtaGroup.TWO)
    smem: int          # bytes per CTA
    regs: int | None   # per-thread, None = unmeasured (GPU probe TODO)


#                      name            thr  ctas  smem        regs
T_HCA = Tile("attn_hca", 512, 2, 208 * KB, None)          # measured smem (1.2)
T_LOG = Tile("logits", 192, 1, 70 * KB, None)          # smem EST (staged K+heap)
T_L1 = Tile("merge_l1", 256, 1, 48 * KB, None)          # EST (1.4b, does not exist)
T_L2 = Tile("merge_l2", 128, 1, 16 * KB, None)          # EST (1.4b)
T_PUSH = Tile("push", 128, 1, 32 * KB, None)          # staging EST
T_CMP = Tile("compress", 128, 1, 12 * KB, None)          # s_stats+s_y (1.3)
T_KVW = Tile("kvwrite", 128, 1, 4 * KB, None)          # EST
T_LSE = Tile("lse_merge", 128, 1, 24 * KB, None)          # EST (1.4c)
ALL_TILES = [T_HCA, T_LOG, T_L1, T_L2, T_PUSH, T_CMP, T_KVW, T_LSE]


# ---------------------------------------------------------------- task mix

@dataclass
class Ctx:
    B: int = 8
    q: int = 2                   # spec tokens per step (2 or 6-8)
    S: int = 64 * 1024           # context length bucket
    cp: int = 2
    topk: int = 512              # indexer K (512 Flash / 1024 Pro)
    bpc: int = 16                # blocks_per_chunk (logits) -> chunk = bpc*128 entries
    win: int = 2048              # SWA window entries
    lat_scale: float = 1.0
    lse_push_fused: bool = True  # LSE partials push folded into attn epilogue


def task_counts(layer: str, c: Ctx) -> dict[str, int]:
    """Tasks per rank per step for one layer of the given type."""
    n = {t.name: 0 for t in ALL_TILES}
    if layer == "CSA":
        entries = (c.S + c.cp - 1) // c.cp            # token-interleaved shard
        chunks = max(1, (entries // 128 + c.bpc - 1) // c.bpc)
        n["logits"] = c.B * c.q * chunks
        n["merge_l1"] = c.B * c.q
        n["merge_l2"] = c.B * c.q
        n["push"] = c.B * c.q                        # cand allgather
        n["attn_hca"] = c.B * c.q
        n["lse_merge"] = c.B * c.q
        n["compress"] = 2 * c.B                       # step + finalize
        n["kvwrite"] = c.B * c.q
        if not c.lse_push_fused:
            n["push"] += c.B * c.q                    # separate LSE push tasks
    elif layer == "C128A":
        n["attn_hca"] = c.B * c.q                     # dense over S/128 entries
        n["compress"] = 2 * c.B
        n["kvwrite"] = c.B * c.q
    elif layer == "SWA":
        n["attn_hca"] = c.B * c.q                     # window only
        n["kvwrite"] = c.B * c.q
    return n


# ---------------------------------------------------------------- latencies (us, analytical)


def lat(tile: str, layer: str, c: Ctx, hw: HW) -> float:
    o = hw.task_ovh_us
    x = 1.0
    if tile == "logits":
        x = o + (c.bpc * 128 * 68) / hw.hbm_bw * 1e6           # 136KB stream
    elif tile == "merge_l1":
        entries = (c.S + c.cp - 1) // c.cp
        chunks = max(1, (entries // 128 + c.bpc - 1) // c.bpc)
        x = o + 0.5 + (chunks * c.topk * 8) / hw.hbm_bw * 1e6  # radix passes
    elif tile == "merge_l2":
        x = o + (c.cp * c.topk * 8) / hw.hbm_bw * 1e6
    elif tile == "push":
        x = o + 0.3
    elif tile == "attn_hca":
        if layer == "CSA":
            x = o + 1.5 + ((c.topk + c.win) * 512) / hw.hbm_bw * 1e6
            if c.lse_push_fused:
                x += (128 * 512 * 4) / hw.nvl_push_bw * 1e6     # 256KB partials
        elif layer == "C128A":
            x = o + 1.0 + (c.S // 128 * 512) / hw.hbm_bw * 1e6
        else:
            x = o + 0.5 + (c.win * 512) / hw.hbm_bw * 1e6
    elif tile == "lse_merge":
        x = o + (2 * c.cp * 128 * 512 * 4) / hw.hbm_bw * 1e6
    elif tile == "compress":
        x = o + 0.3
    elif tile == "kvwrite":
        x = o
    return x * c.lat_scale


def crit_path(layer: str, c: Ctx, hw: HW) -> float:
    """Event-chain latency for one (b,t) through a CSA layer."""
    if layer != "CSA":
        return lat("attn_hca", layer, c, hw) + lat("kvwrite", layer, c, hw)
    p = lat("logits", layer, c, hw) + hw.loc_hop_us
    p += lat("merge_l1", layer, c, hw) + hw.loc_hop_us
    p += lat("push", layer, c, hw) + hw.sys_hop_us             # cand allgather
    p += lat("merge_l2", layer, c, hw) + hw.loc_hop_us
    p += lat("attn_hca", layer, c, hw) + hw.sys_hop_us         # partials arrive
    p += lat("lse_merge", layer, c, hw)
    return p


# ---------------------------------------------------------------- schedule sim


def simulate(layer: str, c: Ctx, hw: HW, pack: bool) -> dict:
    """Static-deal makespan. Workers = hw.sms (1 CTA/SM, unified smem=208KB).
    Cluster tasks (attn_hca) occupy an even/odd worker pair.
    pack=True: small tiles (threads<=256, ctas==1) run 512/threads-per-CTA
    concurrent warp-groups per worker (REQUIRES tile barrier rework: compress
    uses CTA-wide cute.arch.barrier, compress.py:230; logits named barriers).
    """
    W = hw.sms
    counts = task_counts(layer, c)
    # expand tasks in topological order (logits first ... lse last)
    order = ["kvwrite", "compress", "logits", "merge_l1", "push",
             "merge_l2", "attn_hca", "lse_merge"]
    tasks = []
    for name in order:
        for _ in range(counts[name]):
            tasks.append((name, lat(name, layer, c, hw)))
    load = [0.0] * W
    cur = 0

    def nxt(w):
        return (w + 1) % W

    if not pack:
        for name, d in tasks:
            t = next(t for t in ALL_TILES if t.name == name)
            if t.ctas == 2:
                if cur % 2:
                    cur += 1
                w = cur % W
                t0 = max(load[w], load[(w + 1) % W])
                load[w] = load[(w + 1) % W] = t0 + d
                cur = nxt(w + 1)
            else:
                w = cur % W
                load[w] += d
                cur = nxt(w)
    else:
        # capacity in warp-group slots per worker; cluster tasks take all 512
        for name, d in tasks:
            t = next(t for t in ALL_TILES if t.name == name)
            if t.ctas == 2:
                if cur % 2:
                    cur += 1
                w = cur % W
                t0 = max(load[w], load[(w + 1) % W])
                load[w] = load[(w + 1) % W] = t0 + d
                cur = nxt(w + 1)
            else:
                per = 512 // t.threads
                w = cur % W
                load[w] += d / per                     # ideal packing
                cur = nxt(w)
    mk = max(load)
    cp_l = crit_path(layer, c, hw)
    return {"makespan": max(mk, cp_l), "sched": mk, "crit": cp_l,
            "tasks": sum(counts.values()),
            "cta_slots": sum(v * next(t for t in ALL_TILES if t.name == k).ctas
                             for k, v in counts.items()),
            "load_mean": sum(load) / W}


# ---------------------------------------------------------------- report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--q", type=int, default=2)
    ap.add_argument("--cp", type=int, default=2)
    ap.add_argument("--S", type=int, nargs="+", default=[64 * 1024, 1024 * 1024])
    ap.add_argument("--lat-scale", type=float, default=1.0)
    ap.add_argument("--sms", type=int, default=148)
    a = ap.parse_args()
    hw = HW(sms=a.sms)

    print("== 1. block-size feasibility (unified kernel, one block size) ==")
    print(f"{'block':>6} {'regs/thr':>9} {'fits HCA':>9} {'fits all':>9}  note")
    for blk in (128, 256, 384, 512, 768, 1024):
        regs = hw.regs_per_sm // blk
        fits = {t.name: (t.threads <= blk) for t in ALL_TILES}
        hca = fits["attn_hca"]
        allf = all(fits.values())
        note = "" if allf else ("HCA needs 512" if not hca else "some tile > block")
        print(f"{blk:>6} {regs:>9} {str(hca):>9} {str(allf):>9}  {note}")
    smem_u = max(t.smem for t in ALL_TILES)
    print(f"unified smem = max tile smem = {smem_u // KB}KB (attn_hca) "
          f"-> {hw.smem_per_cta_max // smem_u} CTA/SM -> workers = {hw.sms}")

    print("\n== 2. per-CTA warp waste at block=512 (small tiles use subset) ==")
    for t in ALL_TILES:
        print(f"  {t.name:<10} thr={t.threads:>3} ctas={t.ctas} "
              f"smem={t.smem // KB:>3}KB  idle-warps@512={16 - t.threads // 32:>2}/16")

    print("\n== 3. task mix & makespan per layer (us) ==")
    hdr = f"{'layer':<6} {'S':>8} {'tasks':>6} {'slots':>6} {'sched':>8} " \
          f"{'crit':>7} {'mkspan':>8} {'packed':>8}"
    print(hdr)
    for S in a.S:
        c = Ctx(B=a.B, q=a.q, S=S, cp=a.cp, lat_scale=a.lat_scale)
        for layer in ("SWA", "CSA", "C128A"):
            r = simulate(layer, c, hw, pack=False)
            rp = simulate(layer, c, hw, pack=True)
            print(f"{layer:<6} {S:>8} {r['tasks']:>6} {r['cta_slots']:>6} "
                  f"{r['sched']:>8.1f} {r['crit']:>7.1f} {r['makespan']:>8.1f} "
                  f"{rp['makespan']:>8.1f}")

    print("\n== 4. full-step latency (61 layers, mix parameter) ==")
    for S in a.S:
        c = Ctx(B=a.B, q=a.q, S=S, cp=a.cp, lat_scale=a.lat_scale)
        for mix in ((3, 29, 29), (3, 58, 0)):
            tot = 0.0
            for layer, n in zip(("SWA", "CSA", "C128A"), mix):
                tot += n * simulate(layer, c, hw, pack=False)["makespan"]
            print(f"  S={S:>8} mix(swa,csa,c128)={mix}: step = {tot / 1e3:6.2f} ms "
                  f"({1e6 / tot:5.1f} tok/s/q-tok)")


if __name__ == "__main__":
    main()
