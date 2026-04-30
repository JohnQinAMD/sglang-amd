"""Single-launch Triton port of debug_flash_mla_adapter._get_invalid_mask.

Replaces the 3-launch torch implementation:

    mask = indices < 0                                            # cmp<long>
    if topk_length is not None:
        arange_topk = torch.arange(topk, device=...).view(1,1,topk)  # FillFunctor + alloc
        mask = mask | (arange_topk >= topk_length.view(b,1,1))    # cmp<int> + or
    mask_2d = mask.view(b * s_q, topk)

Per the chi2774 Phase 13 trace, the `aten::ge` step alone emitted 167 launches/
window (cmp<int>) plus an `aten::lt` and an arange + or — 4 total launches per
call.

Also drops the `_invalid_mask_cache` keyed on `data_ptr()`, which is unsafe
under the caching allocator (per `feedback_data_ptr_caching_unsafe`: same
data_ptr can be reused for different content).
"""
from __future__ import annotations

from typing import Optional
import torch
import triton
import triton.language as tl


@triton.jit
def _invalid_mask_kernel(
    indices_ptr,       # (b*s_q, topk) int (or compatible)
    topk_length_ptr,   # (b,) int OR null
    mask_ptr,          # (b*s_q, topk) bool
    BS: tl.constexpr,  # b * s_q
    SQ: tl.constexpr,  # s_q
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_TL: tl.constexpr,  # 1 if topk_length is provided, 0 otherwise
):
    pid_b = tl.program_id(0)   # one program per (b*s_q) row
    pid_k = tl.program_id(1)
    cols = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    col_mask = cols < TOPK

    row_off = pid_b * TOPK + cols
    idx = tl.load(indices_ptr + row_off, mask=col_mask, other=0)
    mask_neg = idx < 0

    if HAS_TL:
        # Map flat row pid_b -> batch index b = pid_b // s_q.
        b = pid_b // SQ
        tl_b = tl.load(topk_length_ptr + b)
        mask_ge = cols >= tl_b
        mask = mask_neg | mask_ge
    else:
        mask = mask_neg

    # Store as bool (1 byte).
    tl.store(mask_ptr + row_off, mask.to(tl.int8), mask=col_mask)


def get_invalid_mask_triton(
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    b: int,
    s_q: int,
    topk: int,
) -> torch.Tensor:
    """Returns mask_2d of shape (b*s_q, topk), dtype torch.bool.

    Equivalent to:
        m = indices < 0
        if topk_length is not None:
            m = m | (arange(topk) >= topk_length.view(b,1,1))
        return m.view(b*s_q, topk)
    """
    # Fast path: no topk_length means torch's `indices < 0` is already 1 kernel
    # and beats the Triton overhead for small inputs. Skip the fused kernel.
    if topk_length is None:
        return (indices < 0).view(b * s_q, topk)

    BS = b * s_q
    # Flatten indices to (BS, topk) view (no copy if contiguous).
    if indices.dim() != 2 or indices.shape != (BS, topk):
        indices = indices.reshape(BS, topk)
    indices = indices.contiguous()

    mask = torch.empty((BS, topk), dtype=torch.bool, device=indices.device)

    # Tuned on MI355X / chi2811 (BS=6, TOPK=512): BLOCK_K=64 + num_warps=1 +
    # num_stages=1 -> 10.11 us/call (vs default 14.52 us, 30% faster). The
    # smaller BLOCK_K spreads work into multiple grid programs in the K
    # dimension (cdiv(512,64)=8), giving better CU utilization on this tiny
    # workload. num_warps=1 minimizes launch overhead.
    BLOCK_K = 64 if topk >= 64 else triton.next_power_of_2(topk)
    grid = (BS, triton.cdiv(topk, BLOCK_K))
    _invalid_mask_kernel[grid](
        indices,
        topk_length,
        mask,
        BS=BS, SQ=s_q, TOPK=topk,
        BLOCK_K=BLOCK_K,
        HAS_TL=1,
        num_warps=1,
        num_stages=1,
    )
    return mask
