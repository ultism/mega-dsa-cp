"""Torch reference for the LSE merge tile (Phase 1.4c). Same formula as the
kernel: g = max lse, glse = g + log2(sum 2^(l-g)), O = sum O_i * 2^(l_i-glse);
all-empty heads -> O=0, glse=-inf."""

import torch


def lse_merge_ref(acc_o, acc_lse, cp):
    # acc_o (B,q,cp_max,H,D) fp32, acc_lse (B,q,cp_max,H) fp32
    o_s = acc_o[:, :, :cp]
    l_s = acc_lse[:, :, :cp]
    g = l_s.amax(dim=2)                                   # (B,q,H)
    g_safe = torch.where(torch.isinf(g), torch.zeros_like(g), g)
    w = torch.exp2(l_s - g_safe[:, :, None, :])           # (B,q,cp,H)
    ssum = w.sum(dim=2)                                   # (B,q,H)
    has = ssum > 0
    glse = torch.where(has, g + torch.log2(ssum.clamp(min=1e-30)),
                       torch.full_like(g, -float("inf")))
    scale = torch.where(has[:, :, None, :],
                        torch.exp2(l_s - glse[:, :, None, :]),
                        torch.zeros_like(w))
    o = (o_s * scale[:, :, :, :, None]).sum(dim=2)        # (B,q,H,D)
    return o, glse
