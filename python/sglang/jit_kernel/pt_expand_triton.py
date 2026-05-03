"""Phase A2: fused page_table expansion for AITER paged-mqa-logits.

Replaces the 3-launch chain in indexer.py:404, 437:
    arange_buf  (already cached at scratch alloc time, T3)
    intermediate = page_table.unsqueeze(-1) * block_size + arange_buf.view(1,1,-1)  # 2 launches
    pt_expanded.copy_(intermediate.view(batch_size, -1))                            # 1 launch

with a single Triton kernel that writes:
    pt_expanded[b, j*block_size + t] = page_table[b, j] * block_size + t

Per-call savings: 2 launches eliminated × ~22 c4-indexer-calls/forward × 1.46 µs cuda-graph
≈ 64 µs/forward = ~0.06 ms TPOT (microbench predicts; E2E often cascades to 0.1-0.2 ms).

Also eliminates the intermediate-tensor allocations (mul + add results), reducing
cuda-graph pool pressure.

Author: 2026-05-03 (A2)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pt_expand_kernel(
    PT_ptr, OUT_ptr,
    sP_b, sP_j,
    sO_b, sO_t,
    BLOCK_SIZE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    BLOCK_T: tl.constexpr,  # tile size along T per program
):
    pid_b = tl.program_id(0)
    pid_j = tl.program_id(1)

    if pid_j >= MAX_PAGES:
        return

    # Load page id (one int per program)
    page_id = tl.load(PT_ptr + pid_b * sP_b + pid_j * sP_j)

    # Compute base offset in OUT: pid_j * BLOCK_SIZE
    out_base = pid_j * BLOCK_SIZE

    # Tile along T inside the block_size axis
    for t_start in range(0, BLOCK_SIZE, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask = offs_t < BLOCK_SIZE
        # expanded[b, j*BS + t] = page_id * BS + t
        vals = page_id * BLOCK_SIZE + offs_t
        tl.store(
            OUT_ptr + pid_b * sO_b + (out_base + offs_t) * sO_t,
            vals,
            mask=mask,
        )


def pt_expand(
    page_table: torch.Tensor,
    block_size: int,
    *,
    out: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """Expand `page_table[b, j]` into `out[b, j*block_size + t] = page_table[b, j] * block_size + t`.

    Args:
        page_table: [batch, max_pages] int32 or int64
        block_size: page size (typically 64)
        out: optional pre-allocated [batch, max_pages * block_size] same dtype as page_table

    Returns the populated `out` tensor (allocates if None).
    """
    assert page_table.ndim == 2
    batch, max_pages = page_table.shape
    page_table = page_table.contiguous()

    if out is None:
        out = torch.empty(
            (batch, max_pages * block_size),
            dtype=page_table.dtype,
            device=page_table.device,
        )
    else:
        assert out.shape == (batch, max_pages * block_size)
        assert out.dtype == page_table.dtype

    # Pick BLOCK_T to balance occupancy
    if block_size <= 64:
        BLOCK_T = 64
    elif block_size <= 128:
        BLOCK_T = 128
    else:
        BLOCK_T = 256

    grid = (batch, max_pages)
    _pt_expand_kernel[grid](
        page_table, out,
        page_table.stride(0), page_table.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=block_size,
        MAX_PAGES=max_pages,
        BLOCK_T=BLOCK_T,
        num_warps=2,
    )
    return out
