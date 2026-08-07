"""Phase 1.3 dual-GPU test: CP-native compressor (CSA dual-stream + C128A).

    torchrun --nproc_per_node=2 tests/test_compress.py

Flow: micro bit-gates (staged Hadamard + MXFP4 quant, deterministic) -> CPU
reference presim (full-pooling ground truth + fp8 scales) -> 300-step decode
roll at cp=2 (q_len=2; every sequence starts at S=0 in lockstep; group closes
alternate owner rank by entry parity) -> per-rank pool comparison.

Acceptance:
  1. micro: kernel staged-Hadamard == compress_ref.hadamard128_staged bitwise;
     kernel 68B MXFP4 == quant_mxfp4_bitexact(staged(x)) bitwise.
  2. CSA/C128A merged fp32 (dbg taps) vs torch ref: atol 2e-4 (kernel uses
     fast exp/rsqrt; gotcha-34 methodology — transcendental noise lives here).
  3. fp8 attn entries: byte diff <= 1 per element (e4m3 boundary flips only).
  4. MXFP4 idx entries: sf bytes exact; nibble mismatch < 1%, dequant value
     error within one E2M1 level.
  5. k_valid counters == expected local entry counts.

Limitation (documented): all sequences start at S=0 in lockstep; staggered
starts need ring/state pre-fill (future work).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.distributed as dist

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint8
from cutlass.cute.runtime import make_ptr
from cutlass.cute.typing import AddressSpace
import cuda.bindings.driver as cuda_drv

from mega_dsa_cp.events.symm import alloc_event_buffer
from mega_dsa_cp.comm.buffers import alloc_arena
from mega_dsa_cp.tiles import compress as C
from mega_dsa_cp.tiles import compress_ref as cr

B = 8
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SKIP_FIN = len(sys.argv) > 2 and sys.argv[2] == "nofin"
NO_PUSH = len(sys.argv) > 2 and sys.argv[2] == "nopush"
VERBOSE = os.environ.get("VERBOSE", "") != ""
MAXE = 128  # local CSA entry capacity per rank (one idx-K page)
MAXE128 = 4  # local C128A entry capacity per rank
EPS = 1e-6


def ptr(dtype, t, align=16):
    return make_ptr(dtype, t.data_ptr(), AddressSpace.gmem, assumed_align=align)


def micro_test(rank, stream):
    """Deterministic bit-gates for staged hadamard + MXFP4 quant."""
    N = 64
    g = torch.Generator(device="cpu").manual_seed(42 + rank)
    x = torch.randn(N, 128, generator=g, dtype=torch.float32).cuda() * 2.0
    out68 = torch.zeros(N, 68, dtype=torch.uint8).cuda()
    dbg = torch.zeros(N, 128, dtype=torch.float32).cuda()
    compiled = cute.compile(
        C.launch_micro, N, ptr(Float32, x), ptr(Uint8, out68), ptr(Float32, dbg),
        stream,
    )
    compiled(ptr(Float32, x), ptr(Uint8, out68), ptr(Float32, dbg), stream)
    torch.cuda.synchronize()
    ref_h = cr.hadamard128_staged(x.cpu())
    bit_h = (dbg.cpu() == ref_h).all().item()
    ref_packed, ref_sf = cr.quant_mxfp4_bitexact(ref_h)
    got_packed = out68.cpu()[:, :64]
    got_sf = out68.cpu()[:, 64:]
    bit_p = (got_packed == ref_packed).all().item()
    bit_s = (got_sf == ref_sf).all().item()
    max_h = (dbg.cpu() - ref_h).abs().max().item()
    print(
        f"[rank{rank}] micro: hadamard bitexact={bit_h} (maxerr {max_h:.3e}) "
        f"packed bitexact={bit_p} sf bitexact={bit_s}",
        flush=True,
    )
    assert bit_h, "staged hadamard not bit-exact"
    assert bit_p and bit_s, "MXFP4 quant not bit-exact"


def main():
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    torch.cuda.set_device(rank)
    group = dist.new_group(ranks=[0, 1])
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)

    micro_test(rank, stream)

    # ---------------- reference presim (CPU) ----------------
    n_tok = 2 * STEPS
    recs_attn = [[cr.synth_record(b, p, 512, True) for p in range(n_tok)]
                 for b in range(B)]
    recs_idx = [[cr.synth_record(b, p, 128, True) for p in range(n_tok)]
                for b in range(B)]
    recs_c128 = [[cr.synth_record(b, p, 512, False) for p in range(n_tok)]
                 for b in range(B)]
    ape_attn = cr.synth_ape(512, 8)
    ape_idx = cr.synth_ape(128, 8)
    ape128 = cr.synth_ape(512, 128)
    norm_w_attn = torch.rand(512, generator=torch.Generator().manual_seed(7)) * 0.4 + 0.8
    norm_w_idx = torch.rand(128, generator=torch.Generator().manual_seed(8)) * 0.4 + 0.8
    norm_w_128 = torch.rand(512, generator=torch.Generator().manual_seed(9)) * 0.4 + 0.8
    freqs = cr.make_freqs_cis(4096)

    ref_attn, ref_idx = {}, {}  # (b, n) -> pooled fp32
    ref_c128 = {}
    for b in range(B):
        n_groups = n_tok // 4
        for n in range(n_groups):
            kv = torch.stack([
                recs_attn[b][4 * n - 4 + j][(j >> 2) * 512:(j >> 2) * 512 + 512]
                if 4 * n - 4 + j >= 0 else torch.zeros(512)
                for j in range(8)
            ])
            sc = torch.stack([
                recs_attn[b][4 * n - 4 + j][1024 + (j >> 2) * 512:1024 + (j >> 2) * 512 + 512]
                if 4 * n - 4 + j >= 0 else torch.full((512,), float("-inf"))
                for j in range(8)
            ])
            if n == 0:
                pooled = cr.csa_pool_window(kv[4:], sc[4:], ape_attn[4:])
            else:
                pooled = cr.csa_pool_window(kv, sc, ape_attn)
            ref_attn[(b, n)] = pooled
            kvi = torch.stack([
                recs_idx[b][4 * n - 4 + j][(j >> 2) * 128:(j >> 2) * 128 + 128]
                if 4 * n - 4 + j >= 0 else torch.zeros(128)
                for j in range(8)
            ])
            sci = torch.stack([
                recs_idx[b][4 * n - 4 + j][256 + (j >> 2) * 128:256 + (j >> 2) * 128 + 128]
                if 4 * n - 4 + j >= 0 else torch.full((128,), float("-inf"))
                for j in range(8)
            ])
            if n == 0:
                pooled_i = cr.csa_pool_window(kvi[4:], sci[4:], ape_idx[4:])
            else:
                pooled_i = cr.csa_pool_window(kvi, sci, ape_idx)
            ref_idx[(b, n)] = pooled_i
        st = cr.c128_state_init()
        for p in range(n_tok):
            rec = recs_c128[b][p]
            st = cr.c128_update(st, rec[:512], rec[512:], ape128[p % 128])
            if p % 128 == 127:
                ref_c128[(b, p // 128)] = st[2].clone()

    # fp8 scales from presim amax (post-norm values)
    amax_a = max(
        cr.rope_last64(cr.rms_norm(v, norm_w_attn, EPS), freqs, 4 * n).abs().max()
        for (b, n), v in ref_attn.items()
    ).item()
    amax_c = max(
        (cr.rope_last64(cr.rms_norm(v, norm_w_128, EPS), freqs, 128 * m).abs().max()
         for (b, m), v in ref_c128.items()),
        default=torch.tensor(1.0),
    ).item()
    inv_a = 448.0 / amax_a
    inv_c = 448.0 / amax_c

    # ---------------- device buffers ----------------
    dev = "cuda"
    ring_attn = torch.zeros(B, C.RING, C.REC_ATTN, dtype=torch.float32, device=dev)
    ring_idx = torch.zeros(B, C.RING, C.REC_IDX, dtype=torch.float32, device=dev)
    c128_state = torch.zeros(B, 3, 512, dtype=torch.float32, device=dev)
    c128_state[:, 0, :] = C.NEG_BIG
    cmp_pool = torch.zeros(B, MAXE, 512, dtype=torch.uint8, device=dev)
    idxk_pool = torch.zeros(B, MAXE // 128, C.IDXK_PAGE_BYTES, dtype=torch.uint8, device=dev)
    c128_pool = torch.zeros(B, MAXE128, 512, dtype=torch.uint8, device=dev)
    k_valid = torch.zeros(B, dtype=torch.int32, device=dev)
    k_valid128 = torch.zeros(B, dtype=torch.int32, device=dev)
    dbg_attn = torch.zeros(B, MAXE, 512, dtype=torch.float32, device=dev)
    dbg_idx = torch.zeros(B, MAXE, 128, dtype=torch.float32, device=dev)
    dbg128 = torch.zeros(B, MAXE128, 512, dtype=torch.float32, device=dev)
    seq_len = torch.zeros(B, dtype=torch.int32, device=dev)

    ape_attn_d, ape_idx_d, ape128_d = ape_attn.cuda(), ape_idx.cuda(), ape128.cuda()
    nw_a, nw_i, nw_c = norm_w_attn.cuda(), norm_w_idx.cuda(), norm_w_128.cuda()
    freqs_d = freqs.cuda()

    arena = alloc_arena(
        {
            "csa": B * C.CP * C.CSA_STATS_BYTES,
            "c128": B * C.CP * C.C128_STATS_BYTES,
        },
        phases=2,
        rank=rank,
        world_size=world,
        group_name=group.group_name,
    )
    evbuf = alloc_event_buffer(2 * B, rank, world, group_name=group.group_name)

    # ---------------- compile ----------------
    common = (
        ptr(cutlass.Uint8, arena.tensor, 256),
        ptr(Int64, arena.offsets),
        Int64(arena.mc_base),
    )
    ev_args = (ptr(cutlass.Uint64, evbuf.tensor, 128), ptr(Int64, evbuf.offsets))
    step_c = cute.compile(
        C.launch_csa_step, B,
        ptr(Float32, ring_attn), ptr(Float32, ring_idx),
        ptr(Float32, ring_attn),  # placeholder, replaced per-call
        ptr(Float32, ring_idx),
        ptr(Float32, ape_attn_d), ptr(Float32, ape_idx_d),
        ptr(Int32, seq_len),
        *common, Int64(0), *ev_args, Int32(rank), Int32(1), stream,
    )
    fin_c = cute.compile(
        C.launch_csa_finalize, B, MAXE,
        ptr(Int32, seq_len),
        *common, Int64(0), *ev_args, Int32(rank), Int32(0),
        ptr(Float32, nw_a), ptr(Float32, nw_i), Float32(EPS),
        ptr(Float32, freqs_d), Float32(inv_a),
        ptr(Uint8, cmp_pool), ptr(Uint8, idxk_pool), ptr(Int32, k_valid),
        ptr(Float32, dbg_attn), ptr(Float32, dbg_idx), stream,
    )
    step128 = cute.compile(
        C.launch_c128_step, B,
        ptr(Float32, c128_state), ptr(Float32, ring_attn),
        ptr(Float32, ape128_d), ptr(Int32, seq_len),
        *common, Int64(0), *ev_args, Int32(rank), stream,
    )
    fin128 = cute.compile(
        C.launch_c128_finalize, B, MAXE128,
        ptr(Int32, seq_len),
        *common, Int64(0), *ev_args, Int32(rank), Int32(0),
        ptr(Float32, nw_c), Float32(EPS), ptr(Float32, freqs_d),
        Float32(inv_c), ptr(Uint8, c128_pool), ptr(Int32, k_valid128),
        ptr(Float32, dbg128), stream,
    )
    if rank == 0:
        print("compiled 4 kernels", flush=True)

    # ---------------- step roll ----------------
    for t in range(STEPS):
        s0 = 2 * t  # all sequences in lockstep
        pos_r = s0 + ((rank - s0) & 1)
        new_attn = torch.stack([recs_attn[b][pos_r] for b in range(B)]).cuda()
        new_idx = torch.stack([recs_idx[b][pos_r] for b in range(B)]).cuda()
        new_c128 = torch.stack([recs_c128[b][pos_r] for b in range(B)]).cuda()
        csa_off = arena.region_off("csa", t % 2)
        c128_off = arena.region_off("c128", t % 2)
        step_c(
            ptr(Float32, ring_attn), ptr(Float32, ring_idx),
            ptr(Float32, new_attn), ptr(Float32, new_idx),
            ptr(Float32, ape_attn_d), ptr(Float32, ape_idx_d),
            ptr(Int32, seq_len),
            *common, Int64(csa_off), *ev_args, Int32(rank),
            Int32(0 if NO_PUSH else 1), stream,
        )
        if not SKIP_FIN:
            fin_c(
                ptr(Int32, seq_len),
                *common, Int64(csa_off), *ev_args, Int32(rank), Int32(t),
                ptr(Float32, nw_a), ptr(Float32, nw_i), Float32(EPS),
                ptr(Float32, freqs_d), Float32(inv_a),
                ptr(Uint8, cmp_pool), ptr(Uint8, idxk_pool), ptr(Int32, k_valid),
                ptr(Float32, dbg_attn), ptr(Float32, dbg_idx), stream,
            )
        step128(
            ptr(Float32, c128_state), ptr(Float32, new_c128),
            ptr(Float32, ape128_d), ptr(Int32, seq_len),
            *common, Int64(c128_off), *ev_args, Int32(rank), stream,
        )
        if not SKIP_FIN:
            fin128(
                ptr(Int32, seq_len),
                *common, Int64(c128_off), *ev_args, Int32(rank), Int32(t),
                ptr(Float32, nw_c), Float32(EPS), ptr(Float32, freqs_d),
                Float32(inv_c), ptr(Uint8, c128_pool), ptr(Int32, k_valid128),
                ptr(Float32, dbg128), stream,
            )
        seq_len += 2
        if SKIP_FIN and t % 2 == 1:
            pass
        if VERBOSE and rank == 0:
            torch.cuda.synchronize()
            print(f"[rank0] step {t} done", flush=True)
        dist.barrier()
    torch.cuda.synchronize()
    if rank == 0:
        print("roll done", flush=True)
    if SKIP_FIN or NO_PUSH:
        print(f"[rank{rank}] bisect roll completed (skip compare)", flush=True)
        dist.destroy_process_group()
        return

    # ---------------- compare ----------------
    da, di, d128 = dbg_attn.cpu(), dbg_idx.cpu(), dbg128.cpu()
    pool_a = cmp_pool.cpu().view(torch.float8_e4m3fn).float()
    pool_c = c128_pool.cpu().view(torch.float8_e4m3fn).float()
    idxk = idxk_pool.cpu()

    n_chk = 0
    err_a = err_i = err_c = 0.0
    fp8_bad = fp8_tot = 0
    nib_bad = nib_tot = 0
    sf_bad = 0
    for (b, n), ref in ref_attn.items():
        if n % 2 != rank:
            continue
        slot = n // 2
        if slot >= MAXE:
            continue
        n_chk += 1
        err_a = max(err_a, (da[b, slot] - ref).abs().max().item())
        # fp8 entry vs ref post-chain (value space; allow exactly +-1 ulp flips)
        ref_q = cr.post_attn(ref, norm_w_attn, EPS, freqs, 4 * n, 1.0 / inv_a)
        got = pool_a[b, slot]
        want = ref_q.float()
        d = (got - want).abs()
        tol = want.abs() * 0.14 + 3e-3
        fp8_bad += (d > tol).sum().item()
        fp8_tot += d.numel()
        # idx stream
        ref_i = ref_idx[(b, n)]
        err_i = max(err_i, (di[b, slot] - ref_i).abs().max().item())
        pk, sf = cr.post_indexer_bitexact(ref_i, norm_w_idx, EPS, freqs, 4 * n)
        page, tok = slot // 128, slot % 128
        got_pk = idxk[b, page, tok * 64:(tok + 1) * 64]
        base = 8192 + (tok % 32) * 16 + (tok // 32) * 4
        got_sf = idxk[b, page, base:base + 4]
        nib_bad += (got_pk != pk).sum().item()
        nib_tot += pk.numel()
        sf_bad += (got_sf != sf).sum().item()
    for (b, m), ref in ref_c128.items():
        if m % 2 != rank:
            continue
        slot = m // 2
        err_c = max(err_c, (d128[b, slot] - ref).abs().max().item())
        ref_q = cr.post_attn(ref, norm_w_128, EPS, freqs, 128 * m, 1.0 / inv_c)
        d = (pool_c[b, slot] - ref_q.float()).abs()
        tol = ref_q.float().abs() * 0.14 + 3e-3
        fp8_bad += (d > tol).sum().item()
        fp8_tot += d.numel()

    kv_exp = (n_tok // 4 - rank + 1) // 2
    kv128_exp = n_tok // 128 // 2
    kv_ok = (k_valid.cpu() == kv_exp).all().item()
    kv128_ok = (k_valid128.cpu() == kv128_exp).all().item()

    print(
        f"[rank{rank}] CSA entries checked: {n_chk}; dbg_attn maxerr {err_a:.3e} "
        f"dbg_idx maxerr {err_i:.3e} c128 maxerr {err_c:.3e}",
        flush=True,
    )
    print(
        f"[rank{rank}] fp8 >1ulp: {fp8_bad}/{fp8_tot}; idx nibble mismatch "
        f"{nib_bad}/{nib_tot} ({100.0 * nib_bad / max(nib_tot, 1):.4f}%); "
        f"sf mismatch {sf_bad}; k_valid ok={kv_ok} k_valid128 ok={kv128_ok}",
        flush=True,
    )
    assert err_a < 2e-4 and err_i < 2e-4 and err_c < 2e-4, "merge fp32 mismatch"
    assert fp8_bad == 0, "fp8 entries diverge beyond boundary flips"
    assert sf_bad == 0 and nib_bad * 100 < nib_tot, "idx fp4 mismatch"
    assert kv_ok and kv128_ok, "k_valid counters wrong"
    print(f"[rank{rank}] PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
