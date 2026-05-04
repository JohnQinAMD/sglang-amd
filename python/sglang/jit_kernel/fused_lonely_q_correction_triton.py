"""Fused lonely_q correction kernel — direct successor to C1 fused_invalid_mask.

Replaces the 7-launch chain in flashmla_tests/ref.py:333-339:

    lonely_q_mask = lse == float("-inf")                                 # 1 launch (eq)
    output = torch.where(                                                 # 4 launches:
        lonely_q_mask.unsqueeze(-1).expand_as(output),                    #   - unsqueeze + expand (views)
        torch.zeros_like(output),                                         #   - zeros_like = empty + zero_ + fill_(0)
        output,                                                           #   - where on (s_q, h_q, d_v)
    )
    lse = torch.where(lonely_q_mask, torch.full_like(lse, float("+inf")), lse)  # 2 launches:
                                                                          #   - full_like = empty + fill_(+inf)
                                                                          #   - where on (s_q, h_q)

with a SINGLE Triton kernel that:
  - Loads lse[s_q, h_q]
  - Computes is_lonely = (lse == -inf)
  - For lonely positions: zeros output[s_q, h_q, :] in place AND updates lse to +inf in place

7 launches → 1. Plus eliminates the (s_q, h_q, d_v) intermediate materialization (large
HBM traffic save: from 4× tensor-pass to 1× tensor-pass for output).

Trace evidence (post-MEGA-3' Stage 1+2, chi2811): 43× aten::where((856, 1, 64, 512))
+ 43× aten::where((2812, 1, 64, 512)) per profile-window = ~96 µs/event for the big
output where = ~8 ms total per extend forward.

CONSTRAINTS:
  - output: 3D shape [s_q, h_q, d_v], any contiguous-or-strided
  - lse: 2D shape [s_q, h_q]
  - d_v ≤ 4096 (production: 512)
  - s_q × h_q ≤ 2^24 (more than enough for any production batch)

OPERATES IN PLACE on both output and lse (matches the original `output = ...` semantics
since the original assignment happened to a contiguous output tensor view).

Author: 2026-05-04 (continuation of C1 cascade pattern)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_lonely_q_correction_kernel(
    OUTPUT_ptr,       # [s_q, h_q, d_v] (rw, in place)
    LSE_ptr,          # [s_q, h_q] (rw, in place)
    sO_sq, sO_hq, sO_dv,
    sL_sq, sL_hq,
    s_q,
    h_q,
    DV: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    """One program per (s_q, h_q) row. Reads lse, masks output, writes both back."""
    pid_sq = tl.program_id(0)
    pid_hq = tl.program_id(1)
    if pid_sq >= s_q or pid_hq >= h_q:
        return

    lse_off = pid_sq * sL_sq + pid_hq * sL_hq
    lse_val = tl.load(LSE_ptr + lse_off).to(tl.float32)
    is_lonely = lse_val == -float("inf")

    # Update lse: -inf → +inf (so corresponding output below is 0; semantics: lonely q has no attendable k)
    if is_lonely:
        tl.store(LSE_ptr + lse_off, float("+inf"))

        # Zero out output[s_q, h_q, :] in BLOCK_DV chunks
        out_base = pid_sq * sO_sq + pid_hq * sO_hq
        for d_start in range(0, DV, BLOCK_DV):
            d_offs = d_start + tl.arange(0, BLOCK_DV)
            d_mask = d_offs < DV
            tl.store(
                OUTPUT_ptr + out_base + d_offs * sO_dv,
                tl.zeros((BLOCK_DV,), dtype=OUTPUT_ptr.dtype.element_ty),
                mask=d_mask,
            )


def fused_lonely_q_correction_triton(
    output: torch.Tensor,    # [s_q, h_q, d_v] bf16/fp16/fp32 (rw)
    lse: torch.Tensor,       # [s_q, h_q] fp32 (rw)
) -> bool:
    """Drop-in replacement for the lonely_q_mask + 2× where chain. Returns True if
    the kernel fired, False if shape gate failed (caller should fall back).
    """
    if output.dim() != 3 or lse.dim() != 2:
        return False
    s_q, h_q, d_v = output.shape
    if lse.shape != (s_q, h_q):
        return False
    if d_v > 4096:
        return False

    BLOCK_DV = min(triton.next_power_of_2(d_v), 512)
    grid = (s_q, h_q)
    _fused_lonely_q_correction_kernel[grid](
        output, lse,
        output.stride(0), output.stride(1), output.stride(2),
        lse.stride(0), lse.stride(1),
        s_q, h_q,
        DV=d_v, BLOCK_DV=BLOCK_DV,
        num_warps=1, num_stages=1,
    )
    return True
