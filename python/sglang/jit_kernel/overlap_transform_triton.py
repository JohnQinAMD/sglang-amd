"""HIP Triton port of DeepseekV4Indexer.overlap_transform.

Replaces the 3-launch torch implementation in models/deepseek_v4.py:693-708:

    new_tensor.fill_(fill_value)              # FillFunctor launch
    new_tensor[:, r:] = tensor[:, :, d:]      # bare aten::copy_ launch
    new_tensor[1:, :r] = tensor[:-1, :, :d]   # bare aten::copy_ launch

Per the chi2774 Phase 13 trace (ROCm 7.2 + AITER HEAD), this produced 672
elementwise_kernel_manual_unroll<128,4> launches/window across all
overlap_transform call sites (lines 830/833/1080/1083).

Single Triton kernel writes all three regions of `new_tensor` in one dispatch:
    new_tensor[:, :r, :] = fill_value            (default region)
    new_tensor[:, r:, :] = tensor[:, :, d:]      (last r rows of dim=1, src is second half of dim=2)
    new_tensor[1:, :r, :] = tensor[:-1, :, :d]   (first r rows of dim=1 except row 0, src is first half of dim=2 shifted)

Layout: input  tensor      (s, r, 2*d)   contiguous
        output new_tensor  (s, 2*r, d)   contiguous

Grid: (s, cdiv(d, BLOCK_D))   one program per output (s, dim2_tile)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


MIN_S_FOR_TRITON = 1


@triton.jit
def _overlap_transform_kernel(
    src_ptr,          # (s, r, 2*d) bf16/fp16
    dst_ptr,          # (s, 2*r, d) bf16/fp16, pre-allocated persistent scratch
    fill_value,       # scalar (0.0 or -inf)
    s,                # int
    r: tl.constexpr,  # ratio (typically 4 or 64)
    d: tl.constexpr,  # head_dim (128)
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_d = tl.program_id(1)

    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < d
    rows = tl.arange(0, r)

    # Region 1: dst[pid_s, :r, :] -- always set to fill_value (overwrites everything,
    # plus row 0 stays at fill_value because Region 3 below skips pid_s == 0).
    fill_vec = tl.full((r, BLOCK_D), fill_value, dtype=dst_ptr.dtype.element_ty)
    dst_off1 = pid_s * (2 * r) * d + rows[:, None] * d + cols[None, :]
    mask1 = col_mask[None, :]
    if pid_s == 0:
        # Row 0 of dim=1's first r entries: pure fill (Region 3 doesn't apply).
        tl.store(dst_ptr + dst_off1, fill_vec, mask=mask1)
    else:
        # Region 3 source: src[pid_s - 1, :, :d]
        # = src offset (pid_s - 1) * r * (2*d) + rows * (2*d) + cols
        src_off3 = (pid_s - 1) * r * (2 * d) + rows[:, None] * (2 * d) + cols[None, :]
        v3 = tl.load(src_ptr + src_off3, mask=mask1, other=0.0)
        tl.store(dst_ptr + dst_off1, v3, mask=mask1)

    # Region 2: dst[pid_s, r:, :] = src[pid_s, :, d:]
    # src offset = pid_s * r * (2*d) + rows * (2*d) + (d + cols)
    src_off2 = pid_s * r * (2 * d) + rows[:, None] * (2 * d) + (d + cols[None, :])
    dst_off2 = pid_s * (2 * r) * d + (r + rows[:, None]) * d + cols[None, :]
    v2 = tl.load(src_ptr + src_off2, mask=col_mask[None, :], other=0.0)
    tl.store(dst_ptr + dst_off2, v2, mask=col_mask[None, :])


def overlap_transform_triton(
    tensor: torch.Tensor,
    new_tensor: torch.Tensor,
    fill_value: float,
    ratio: int,
    head_dim: int,
) -> bool:
    """Fused fill + 2-region scatter for overlap_transform.

    Equivalent to:
        new_tensor.fill_(fill_value)
        new_tensor[:, ratio:] = tensor[:, :, head_dim:]
        new_tensor[1:, :ratio] = tensor[:-1, :, :head_dim]

    Args:
        tensor: (s, ratio, 2*head_dim) source.
        new_tensor: (s, 2*ratio, head_dim) pre-allocated persistent scratch.
        fill_value: scalar to write into rows 0:ratio (only matters at pid_s=0;
            other rows are overwritten by tensor[:-1, :, :head_dim]).
        ratio, head_dim: constexpr.

    Returns:
        True if Triton fired; False if caller should fall back.
    """
    assert tensor.dim() == 3 and new_tensor.dim() == 3
    s, r, d2 = tensor.shape
    assert r == ratio and d2 == 2 * head_dim, f"src shape {tensor.shape} vs (s, {ratio}, {2*head_dim})"
    s2, r2, d = new_tensor.shape
    assert s2 == s and r2 == 2 * ratio and d == head_dim
    assert tensor.is_contiguous() and new_tensor.is_contiguous()
    assert tensor.dtype == new_tensor.dtype

    if s < MIN_S_FOR_TRITON:
        return False

    # Tuned on MI355X / chi2811 (s=8, head_dim=128): BLOCK_D=32 spreads the
    # head_dim into 4 row blocks for better CU utilization on the small workload.
    # num_warps depends on ratio: 2 for narrow (c=4 indexer), 4 for wide (c=128).
    BLOCK_D = 32
    if ratio <= 64:
        num_warps, num_stages = 2, 2   # ratio=4 best: 8.94 us (vs default 12.0 us, 26% faster)
    else:
        num_warps, num_stages = 4, 1   # ratio=128 best: 9.02 us (vs default 12.0 us, 25% faster)
    grid = (s, triton.cdiv(d, BLOCK_D))
    _overlap_transform_kernel[grid](
        tensor, new_tensor,
        float(fill_value),
        s,
        r=ratio, d=head_dim,
        BLOCK_D=BLOCK_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return True
