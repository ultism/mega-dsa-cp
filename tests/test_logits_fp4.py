"""Phase 1.1 v1: FP4 paged MQA logits vs torch fp32-dequant reference (1xB200)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from mega_dsa_cp.tiles.fp4 import make_index_cache, make_q, ref_logits
from mega_dsa_cp.tiles.logits import run_fp4_paged_mqa_logits


def main() -> None:
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    tiny = "--tiny" in sys.argv
    if tiny:
        B, max_pages, q_len, blocks_per_chunk = 1, 1, 2, 1
    else:
        B, max_pages, q_len = 4, 64, 2  # S = 8192, 4 chunks x 16 blocks
        blocks_per_chunk = 16

    kv_fused, kv_sf_plain = make_index_cache(B, max_pages, seed=0)
    q_packed, q_sf_plain, q_sf_atom, w = make_q(B, q_len=q_len, seed=1)
    bt = torch.arange(B * max_pages, dtype=torch.int32, device="cuda").reshape(
        B, max_pages
    )
    logits = torch.full(
        (B * q_len, max_pages * 128), float("nan"), device="cuda"
    )

    print("inputs ready, compiling...", flush=True)
    dbg = torch.zeros(8, dtype=torch.int32, pin_memory=True)
    run_fp4_paged_mqa_logits(
        kv_fused, q_packed, q_sf_atom, w, bt, logits, blocks_per_chunk, dbg=dbg
    )
    print("launched, syncing...", flush=True)
    if tiny:
        # Deadlock triage: markers live in pinned host memory (sys-scope
        # stores), poll without any CUDA call; hard-exit instead of hanging.
        import time

        for _ in range(90):
            time.sleep(2)
            print(f"markers tma={dbg[0].item()} umma={dbg[1].item()} math={dbg[2].item()}", flush=True)
            if dbg[0].item() == 100 and dbg[1].item() == 100 and dbg[2].item() == 100:
                break
        else:
            print("DEADLOCK", flush=True)
            os._exit(2)
    torch.cuda.synchronize()
    print("kernel done", flush=True)

    ref = ref_logits(q_packed, q_sf_plain, kv_fused, kv_sf_plain, w, bt, q_len=q_len)
    diff = (logits - ref).abs()
    rel = diff / ref.abs().clamp(min=1e-3)
    print(
        f"max abs {diff.max().item():.3e}  max rel {rel.max().item():.3e}  "
        f"mean abs {diff.mean().item():.3e}  nan={torch.isnan(logits).any().item()}"
    )
    ok = torch.allclose(logits, ref, rtol=1e-3, atol=1e-4)
    print("PASS" if ok else "FAIL")
    if not ok:
        bad = (diff > 1e-3).nonzero()[:8]
        for r, c in bad.tolist():
            print(f"  [{r},{c}] got {logits[r, c].item():.6f} ref {ref[r, c].item():.6f}")
        raise SystemExit(1)


def main_topk() -> None:
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    from mega_dsa_cp.tiles.logits import TOP_K

    B, max_pages, q_len = 2, 64, 2  # S = 8192
    blocks_per_chunk = 32  # 2 chunks x 4096 tokens (real selection: 4096 -> 2048)
    n_chunks = max_pages // blocks_per_chunk
    chunk_tokens = blocks_per_chunk * 128

    kv_fused, kv_sf_plain = make_index_cache(B, max_pages, seed=0)
    q_packed, q_sf_plain, q_sf_atom, w = make_q(B, q_len=q_len, seed=1)
    bt = torch.arange(B * max_pages, dtype=torch.int32, device="cuda").reshape(
        B, max_pages
    )
    S = max_pages * 128
    seq_lens = torch.full((B,), S, dtype=torch.int32, device="cuda")
    cand_v = torch.full((B, q_len, n_chunks, TOP_K), float("nan"), device="cuda")
    cand_i = torch.full((B, q_len, n_chunks, TOP_K), -1, dtype=torch.int32, device="cuda")
    cand_c = torch.zeros((B, q_len, n_chunks), dtype=torch.int32, device="cuda")
    logits = torch.zeros((B * q_len, S), device="cuda")  # unused in fused mode

    print("inputs ready, compiling (fused topk)...", flush=True)
    dbg = torch.zeros(192, dtype=torch.int32, pin_memory=True)
    run_fp4_paged_mqa_logits(
        kv_fused, q_packed, q_sf_atom, w, bt, logits, blocks_per_chunk,
        seq_lens=seq_lens, cand_v=cand_v, cand_i=cand_i, cand_c=cand_c, dbg=dbg,
    )
    print("launched, syncing...", flush=True)
    torch.cuda.synchronize()
    print("kernel done", flush=True)
    for i in range(blocks_per_chunk):
        print(
            f"  blk{i:2d} heap={dbg[16+i*2].item():5d} ovf={dbg[17+i*2].item():5d} "
            f"score={dbg[64+i].item() & 0xFFFFFFFF:#010x} "
            f"sW0={dbg[96+i].item() & 0xFFFFFFFF:#010x} "
            f"acc0={dbg[128+i].item() & 0xFFFFFFFF:#010x}",
            flush=True,
        )
    for t in range(q_len):
        for c in range(n_chunks):
            base = (t * n_chunks + c) * 4
            print(
                f"  cnt[t{t},c{c}] heap={dbg[base].item()} "
                f"ovf={dbg[base+1].item()} merges={dbg[base+2].item()} "
                f"theta_key={dbg[base+3].item() & 0xFFFFFFFF:#x}",
                flush=True,
            )

    ref = ref_logits(q_packed, q_sf_plain, kv_fused, kv_sf_plain, w, bt, q_len=q_len)
    limit = S - 128  # SWA window skipped by the kernel
    n_bad = 0
    for b in range(B):
        for t in range(q_len):
            for c in range(n_chunks):
                lo, hi = c * chunk_tokens, min((c + 1) * chunk_tokens, limit)
                seg = ref[b * q_len + t, lo:hi]
                k = min(TOP_K, seg.numel())
                exp_v, exp_i = torch.topk(seg, k)
                exp_i = exp_i + lo
                cnt = int(cand_c[b, t, c].item())
                assert cnt == k, f"[{b},{t},{c}] count {cnt} != {k}"
                got_v = cand_v[b, t, c, :k]
                got_i = cand_i[b, t, c, :k]
                got_v_sorted, _ = got_v.sort(descending=True)
                v_ok = torch.allclose(got_v_sorted, exp_v, rtol=1e-4, atol=1e-5)
                overlap = (
                    len(set(got_i.tolist()) & set(exp_i.tolist())) / k
                )
                print(
                    f"[{b},{t},{c}] k={k} values {'OK' if v_ok else 'BAD'} "
                    f"index overlap {overlap:.4f}",
                    flush=True,
                )
                if not v_ok or overlap < 0.999:
                    n_bad += 1
    print("PASS" if n_bad == 0 else f"FAIL ({n_bad} bad)")
    if n_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    if "--topk" in sys.argv:
        main_topk()
    else:
        main()
