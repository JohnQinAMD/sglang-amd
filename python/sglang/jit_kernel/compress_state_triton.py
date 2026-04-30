"""HIP Triton port for CompressStatePool.set_state_by_state_loc.

Replaces the 3-launch torch fallback in compress_state.py:188-190:

    self.kv_score_buffer[state_loc] = value      # __setitem__  -> aten::copy_ x 1 (NEW) / x 2 (OLD)
    self.kv_score_buffer[-1].clear()             # zero_() + fill_(-inf) -> 2 ewise launches

Per the chi2774 Phase 13 trace (ROCm 7.2 + AITER HEAD), the OLD-compressor path
emits 1,824 elementwise_kernel_manual_unroll<128,4> launches per 174-AR window,
all attributed to compress_state __setitem__/clear. This kernel collapses those
3 separate aten launches into a single Triton dispatch (scatter rows + write
the clean-state pattern at row size-1 in one program grid).

Skipped (returns False) when N < MIN_N — torch index_put + zero_+fill_ wins on
launch overhead. The cuda-graph captured decode replays the kernel from a
graph-pool stable buffer, so the kernel must NOT allocate.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MIN_N_FOR_TRITON = 1


@triton.jit
def _scatter_kernel(
    buf_ptr,            # (size, D)
    state_loc_ptr,      # (N,) int32/int64
    value_ptr,          # (N, D)
    N,                  # number of rows to scatter
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    row_mask = rows < N
    col_mask = cols < D
    mask = row_mask[:, None] & col_mask[None, :]

    loc = tl.load(state_loc_ptr + rows, mask=row_mask, other=0)
    # Skip rows where loc < 0 (sentinel for invalid position).
    valid = (loc >= 0) & row_mask
    scatter_mask = valid[:, None] & col_mask[None, :]

    src_off = rows[:, None] * D + cols[None, :]
    dst_off = loc[:, None].to(tl.int64) * D + cols[None, :]

    v = tl.load(value_ptr + src_off, mask=mask, other=0.0)
    tl.store(buf_ptr + dst_off, v, mask=scatter_mask)


@triton.jit
def _scatter_with_clear_kernel(
    buf_ptr,            # (size, D)
    state_loc_ptr,      # (N,) int32/int64
    value_ptr,          # (N, D)
    N,                  # number of rows to scatter
    last_row,           # = size - 1; clean-state row index
    item_size,          # = D // 2; first half is kv (zeros), second is score (-inf)
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused scatter + clean-state row in a single grid.

    pid_m in [0, n_row_blocks): handle scatter rows
    pid_m == n_row_blocks: handle the clean-state write at last_row
    """
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < D

    n_row_blocks = (N + BLOCK_M - 1) // BLOCK_M

    if pid_m < n_row_blocks:
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < N
        mask = row_mask[:, None] & col_mask[None, :]

        loc = tl.load(state_loc_ptr + rows, mask=row_mask, other=0)
        valid = (loc >= 0) & row_mask
        scatter_mask = valid[:, None] & col_mask[None, :]

        src_off = rows[:, None] * D + cols[None, :]
        dst_off_2d = loc[:, None].to(tl.int64) * D + cols[None, :]

        v = tl.load(value_ptr + src_off, mask=mask, other=0.0)
        tl.store(buf_ptr + dst_off_2d, v, mask=scatter_mask)
    else:
        # Clean-state write at last_row.
        is_score = cols >= item_size
        clean = tl.where(is_score, float("-inf"), 0.0)
        dst_off_1d = tl.cast(last_row, tl.int64) * D + cols
        tl.store(buf_ptr + dst_off_1d, clean, mask=col_mask)


def set_state_with_clear_triton(
    kv_score_buffer: torch.Tensor,
    state_loc: torch.Tensor,
    value: torch.Tensor,
    item_size: int,
) -> bool:
    """Fused scatter + clean-(-1)-row.

    Equivalent to:
        kv_score_buffer[state_loc] = value     (skipping rows where state_loc < 0)
        kv_score_buffer[-1, :item_size].zero_()
        kv_score_buffer[-1, item_size:].fill_(float('-inf'))

    Args:
        kv_score_buffer: 2D persistent pool tensor, shape (size, D).
        state_loc: 1D int tensor of target row indices (negative = skip).
        value: 2D source tensor, shape (N, D), where N = state_loc.numel().
        item_size: D // 2 (split point between kv-half and score-half).

    Returns:
        True if Triton fired, False if caller should fall back.
    """
    assert kv_score_buffer.dim() == 2, f"buffer must be 2D, got {kv_score_buffer.shape}"
    assert kv_score_buffer.is_contiguous()
    size, D = kv_score_buffer.shape
    assert D == 2 * item_size, f"D={D} != 2*item_size={2*item_size}"

    if value.dim() > 2:
        value = value.reshape(-1, D)
    assert value.shape[-1] == D, f"value last dim {value.shape[-1]} != {D}"
    assert value.is_contiguous()

    if state_loc.numel() == 0:
        # Nothing to scatter, but still need the -1 clear. Caller handles.
        return False

    state_loc = state_loc.reshape(-1)
    N = state_loc.shape[0]
    assert value.shape[0] == N, f"N mismatch: state_loc {N} vs value {value.shape[0]}"

    if N < MIN_N_FOR_TRITON:
        return False

    if state_loc.dtype not in (torch.int32, torch.int64):
        state_loc = state_loc.to(torch.int32)

    # Tuned on MI355X / chi2811 (N=6, D=512): BLOCK_M=16 + BLOCK_D=cdiv(D)
    # + num_warps=1 + num_stages=1 -> 10.26 us/call (vs default 13.95 us, 27% faster).
    # Lower num_warps reduces launch overhead for the small-N decode shape.
    BLOCK_M = 16
    if D <= 512:
        BLOCK_D = triton.next_power_of_2(D)
    else:
        BLOCK_D = 256
    n_row_blocks = (N + BLOCK_M - 1) // BLOCK_M
    # +1 row block for the clean-state row.
    grid = (n_row_blocks + 1, triton.cdiv(D, BLOCK_D))
    _scatter_with_clear_kernel[grid](
        kv_score_buffer,
        state_loc,
        value,
        N,
        size - 1,
        item_size,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        num_warps=1,
        num_stages=1,
    )
    return True
