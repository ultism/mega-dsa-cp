"""Phase 1.2: HCA fork (tiles/hca.py) vs torch dual-pool reference (1xB200)."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from mega_dsa_cp.tiles.hca import run_hca_fp8
from mega_dsa_cp.tiles.hca_ref import H, D, make_inputs, ref_hca, ref_hca_online


def run_case(name, inp, sink, softmax_scale, use_acc, poll_markers=False):
    B, T = inp["q"].shape[0], inp["q"].shape[1]
    o = torch.full((B, T, H, D), float("nan"), device="cuda")
    lse = torch.full((B, T, H), float("nan"), device="cuda")
    acc_o = torch.full((B, T, H, 1, D), float("nan"), device="cuda") if use_acc else None
    acc_lse = (
        torch.full((B, T, H, 1), float("nan"), device="cuda") if use_acc else None
    )
    dbg = torch.zeros(8, dtype=torch.int32, pin_memory=True)
    print(f"[{name}] compiling/launching...", flush=True)
    run_hca_fp8(
        inp["q"], inp["kv_win"], inp["kv_cmp"], inp["pt_win"], inp["pt_cmp"],
        o, lse, inp["k_valid"], inp["win_valid"], sink, softmax_scale,
        acc_o=acc_o, acc_lse=acc_lse, dbg=dbg,
    )
    print(f"[{name}] launched, syncing...", flush=True)
    if poll_markers:
        for _ in range(60):
            time.sleep(2)
            vals = [dbg[i].item() for i in range(7)]
            print(f"  markers {vals}", flush=True)
            if all(v == 100 for v in vals):
                break
        else:
            print("DEADLOCK", flush=True)
            os._exit(2)
    torch.cuda.synchronize()
    print(f"[{name}] kernel done", flush=True)

    ref_o, ref_lse = ref_hca(inp, sink, softmax_scale, fp8_p=True)
    on_o, on_lse = ref_hca_online(inp, sink, softmax_scale)
    got_o, got_lse = (acc_o[:, :, :, 0], acc_lse[:, :, :, 0]) if use_acc else (o, lse)

    d_o = (got_o - on_o).abs()
    d_lse = (got_lse - on_lse).abs()
    d_o_naive = (got_o - ref_o).abs()
    finite = torch.isfinite(on_lse)
    o_tol = torch.allclose(got_o, on_o, rtol=2e-2, atol=2e-2)
    lse_ok = torch.allclose(
        got_lse[finite], on_lse[finite], rtol=1e-3, atol=1e-3
    ) and bool((torch.isfinite(got_lse) == finite).all())
    print(
        f"[{name}] O: vs online-sim max {d_o.max().item():.3e} "
        f"(vs naive-fp8P {d_o_naive.max().item():.3e}) {'OK' if o_tol else 'BAD'} | "
        f"LSE: max abs {d_lse[finite].max().item() if finite.any() else 0.0:.3e} "
        f"{'OK' if lse_ok else 'BAD'}",
        flush=True,
    )
    ok = o_tol and lse_ok
    if not ok:
        viol = d_o / (2e-2 + 2e-2 * on_o.abs())
        w = viol.argmax()
        b, t, h, dd = torch.unravel_index(w, d_o.shape)
        print(
            f"  worst o[{b.item()},{t.item()},{h.item()},{dd.item()}] "
            f"got {got_o[b,t,h,dd].item():.6f} online {on_o[b,t,h,dd].item():.6f} "
            f"viol {viol.max().item():.2f}",
            flush=True,
        )
        bad = (d_o > 1e-2).nonzero()[:5]
        for b, t, h, dd in bad.tolist():
            print(
                f"  o[{b},{t},{h},{dd}] got {got_o[b,t,h,dd].item():.6f} "
                f"online {on_o[b,t,h,dd].item():.6f} naive {ref_o[b,t,h,dd].item():.6f}",
                flush=True,
            )
    return ok


def main():
    torch.cuda.set_device(0)
    results = []

    # 1. tiny C128A smoke (single tile) with marker polling
    inp = make_inputs(1, 1, [512], "c128a", seed=0)
    sink = torch.full((H,), -float("inf"), device="cuda")
    results.append(
        run_case("tiny-c128a", inp, sink, 0.007, use_acc=False, poll_markers=True)
    )

    # 2. C128A dense, var seq lens incl. S<128 window-partial edge
    inp = make_inputs(4, 2, [4396, 100, 8229, 2048], "c128a", seed=1)
    results.append(run_case("c128a-nosink", inp, sink, 0.007, use_acc=False))

    # 3. C128A with attn sink
    sink_f = torch.randn(H, device="cuda") * 0.5 - 1.0
    results.append(run_case("c128a-sink", inp, sink_f, 0.007, use_acc=False))

    # 4. CSA gather (page_cmp=1), K=512 with E<K row
    inp = make_inputs(2, 2, [8192, 600], "csa", seed=2, k_sel=512)
    results.append(run_case("csa-gather", inp, sink, 0.007, use_acc=False))

    # 5. partials path (acc_o/acc_lse), C128A
    inp = make_inputs(2, 2, [4096, 1024], "c128a", seed=3)
    results.append(run_case("c128a-partials", inp, sink, 0.007, use_acc=True))

    n_ok = sum(results)
    print(f"{n_ok}/{len(results)} cases PASS", flush=True)
    if n_ok != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
