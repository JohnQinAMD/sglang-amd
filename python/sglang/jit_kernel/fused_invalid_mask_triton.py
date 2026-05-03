"""Phase C1: fused invalid_mask compute Triton kernel.

Replaces the chain in ref.py:240-244 (M3 lookup miss path):
    invalid_mask = kv_scope.indices_in_kvcache == -1                       # eq
    if topk_length is not None:
        invalid_mask |= torch.arange(0, topk, ...) >= topk_length.view(...)  # arange + ge + bitwise_or

with a single Triton kernel that:
    invalid_mask[b, s_q, t] = (indices[b, s_q, t] == -1) | (t >= topk_length[b])

4-op chain → 1 kernel. Saves 3 launches per fall-through path × 22 layers/forward.

Author: 2026-05-03 (C1)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_invalid_mask_kernel(
    INDICES_ptr,       # [B, S_q, K] int32 / int64
    TOPK_LEN_ptr,      # [B] int32, or null when HAS_TOPK_LEN=False
    OUT_ptr,           # [B, S_q, K] bool / uint8
    sI_b, sI_s, sI_k,
    sO_b, sO_s, sO_k,
    sL_b,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_TOPK_LEN: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    if HAS_TOPK_LEN:
        topk_len = tl.load(TOPK_LEN_ptr + pid_b * sL_b)
    else:
        topk_len = K  # so t >= topk_len never fires

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        in_range = offs_k < K
        idx = tl.load(
            INDICES_ptr + pid_b * sI_b + pid_s * sI_s + offs_k * sI_k,
            mask=in_range, other=0,
        )
        # Two conditions OR'd:
        cond_neg = idx == -1
        cond_oob = offs_k >= topk_len  # cond_oob is True for t >= topk_len
        invalid = cond_neg | cond_oob
        # Store as uint8 (0 or 1)
        tl.store(
            OUT_ptr + pid_b * sO_b + pid_s * sO_s + offs_k * sO_k,
            invalid.to(tl.uint8),
            mask=in_range,
        )


def fused_invalid_mask(
    indices: torch.Tensor,                           # [B, S_q, K]
    topk_length: "torch.Tensor | None",              # [B] or None
    *,
    out: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Compute invalid_mask[b, s, t] = (indices[b, s, t] == -1) | (t >= topk_length[b]).

    Returns a bool tensor of shape [B, S_q, K]. If `out` is provided, writes there.
    """
    assert indices.ndim == 3
    B, S_q, K = indices.shape
    indices = indices.contiguous()
    if out is None:
        out = torch.empty((B, S_q, K), dtype=torch.bool, device=indices.device)
    else:
        assert out.shape == (B, S_q, K)
        assert out.dtype == torch.bool

    BLOCK_K = min(triton.next_power_of_2(K), 1024) if K > 0 else 1
    has_topk = topk_length is not None
    grid = (B, S_q)
    _fused_invalid_mask_kernel[grid](
        indices,
        topk_length if has_topk else indices,  # dummy when not used
        out,
        indices.stride(0), indices.stride(1), indices.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        topk_length.stride(0) if has_topk else 0,
        K=K,
        BLOCK_K=BLOCK_K,
        HAS_TOPK_LEN=has_topk,
        num_warps=2,
    )
    return out
