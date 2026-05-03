"""Fused 2-way cat for sparse attn decode (T4): combines gathered_kv cat + invalid_mask cat into one Triton launch.

Replaces this ref.py pattern when extra_kv_scope is not None:
    gathered_kv = torch.cat([gathered_kv, gathered_kv1], dim=2)  # [b, s_q, topk_a+topk_b, d]
    invalid_mask = torch.cat([invalid_mask, invalid_mask1], dim=2)  # [b, s_q, topk_a+topk_b]

Two CatArrayBatchedCopy launches → one fused Triton launch per sparse-decode call.
At ~22 sparse layers/forward × 2 cats saved × 1.46 us cuda-graph dispatch ≈ 64 us/forward.
Plus eliminates intermediate allocations.

Implementation: each program handles one (b*s_q, t_idx) pair, copies one kv row
(D elements) and one mask bool. Topk_a + topk_b rows total, batched into BLOCK_T groups.

Author: 2026-05-03 (T4 fusion)
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_dual_cat_kernel(
    KV_A_ptr, KV_B_ptr, KV_OUT_ptr,
    MASK_A_ptr, MASK_B_ptr, MASK_OUT_ptr,
    sKa_n, sKa_t, sKa_d,
    sKb_n, sKb_t, sKb_d,
    sKo_n, sKo_t, sKo_d,
    sMa_n, sMa_t,
    sMb_n, sMb_t,
    sMo_n, sMo_t,
    TOPK_A: tl.constexpr,
    TOPK_B: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_n = tl.program_id(0)  # which (b*s_q) row
    pid_t = tl.program_id(1)  # which combined topk index

    TOPK_TOTAL: tl.constexpr = TOPK_A + TOPK_B

    if pid_t >= TOPK_TOTAL:
        return

    is_b = pid_t >= TOPK_A
    src_t = pid_t - TOPK_A if is_b else pid_t

    # Mask copy: 1 element
    if is_b:
        m_val = tl.load(MASK_B_ptr + pid_n * sMb_n + src_t * sMb_t).to(tl.uint8)
    else:
        m_val = tl.load(MASK_A_ptr + pid_n * sMa_n + src_t * sMa_t).to(tl.uint8)
    tl.store(MASK_OUT_ptr + pid_n * sMo_n + pid_t * sMo_t, m_val)

    # KV row copy: D elements, tiled by BLOCK_D
    for d_start in range(0, D, BLOCK_D):
        offs_d = d_start + tl.arange(0, BLOCK_D)
        d_mask = offs_d < D
        if is_b:
            kv = tl.load(
                KV_B_ptr + pid_n * sKb_n + src_t * sKb_t + offs_d * sKb_d,
                mask=d_mask, other=0,
            )
        else:
            kv = tl.load(
                KV_A_ptr + pid_n * sKa_n + src_t * sKa_t + offs_d * sKa_d,
                mask=d_mask, other=0,
            )
        tl.store(
            KV_OUT_ptr + pid_n * sKo_n + pid_t * sKo_t + offs_d * sKo_d,
            kv, mask=d_mask,
        )


def fused_dual_cat(
    kv_a: torch.Tensor,           # [b, s_q, topk_a, d]
    kv_b: torch.Tensor,           # [b, s_q, topk_b, d]
    mask_a: torch.Tensor,         # [b, s_q, topk_a]  bool
    mask_b: torch.Tensor,         # [b, s_q, topk_b]  bool
    *,
    out_kv: "torch.Tensor | None" = None,    # optional pre-allocated [b, s_q, topk_a+topk_b, d]
    out_mask: "torch.Tensor | None" = None,  # optional pre-allocated [b, s_q, topk_a+topk_b]
    BLOCK_D: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (gathered_kv [b, s_q, topk_a+topk_b, d], mask [b, s_q, topk_a+topk_b]).

    If out_kv / out_mask are provided, uses them as pre-allocated output buffers
    (cuts per-call alloc; preferred under cuda-graph). Otherwise allocates fresh.
    """
    b, s_q, topk_a, d = kv_a.shape
    bb, ss, topk_b, dd = kv_b.shape
    assert b == bb and s_q == ss and d == dd
    assert mask_a.shape == (b, s_q, topk_a)
    assert mask_b.shape == (b, s_q, topk_b)
    assert kv_a.dtype == kv_b.dtype
    assert mask_a.dtype == mask_b.dtype == torch.bool

    # Make sure inputs are contiguous (for stride simplicity)
    kv_a = kv_a.contiguous()
    kv_b = kv_b.contiguous()
    mask_a = mask_a.contiguous()
    mask_b = mask_b.contiguous()

    topk_total = topk_a + topk_b
    n = b * s_q

    if out_kv is None:
        kv_out = torch.empty((b, s_q, topk_total, d), dtype=kv_a.dtype, device=kv_a.device)
    else:
        assert out_kv.shape == (b, s_q, topk_total, d)
        assert out_kv.dtype == kv_a.dtype
        kv_out = out_kv
    if out_mask is None:
        mask_out = torch.empty((b, s_q, topk_total), dtype=torch.bool, device=mask_a.device)
    else:
        assert out_mask.shape == (b, s_q, topk_total)
        assert out_mask.dtype == torch.bool
        mask_out = out_mask

    # Reshape for kernel: 2D-collapse leading dims (b*s_q, topk, d) and (b*s_q, topk)
    kv_a_2d = kv_a.view(n, topk_a, d)
    kv_b_2d = kv_b.view(n, topk_b, d)
    kv_out_2d = kv_out.view(n, topk_total, d)
    mask_a_2d = mask_a.view(n, topk_a)
    mask_b_2d = mask_b.view(n, topk_b)
    mask_out_2d = mask_out.view(n, topk_total)

    grid = (n, topk_total)
    _fused_dual_cat_kernel[grid](
        kv_a_2d, kv_b_2d, kv_out_2d,
        mask_a_2d, mask_b_2d, mask_out_2d,
        kv_a_2d.stride(0), kv_a_2d.stride(1), kv_a_2d.stride(2),
        kv_b_2d.stride(0), kv_b_2d.stride(1), kv_b_2d.stride(2),
        kv_out_2d.stride(0), kv_out_2d.stride(1), kv_out_2d.stride(2),
        mask_a_2d.stride(0), mask_a_2d.stride(1),
        mask_b_2d.stride(0), mask_b_2d.stride(1),
        mask_out_2d.stride(0), mask_out_2d.stride(1),
        TOPK_A=topk_a,
        TOPK_B=topk_b,
        D=d,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    return kv_out, mask_out
