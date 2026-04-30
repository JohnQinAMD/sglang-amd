"""HIP Triton port for KVAndScoreOld.__setitem__ and .clear().

Replaces the 2-launch torch fallback in compress_state.py:34-40:

    def __setitem__(self, index, value):
        self.kv[index] = value.kv          # 1 aten::copy_
        self.score[index] = value.score    # 1 aten::copy_

    def clear(self):
        self.kv.zero_()                    # 1 FillFunctor
        self.score.fill_(float("-inf"))    # 1 FillFunctor

Per the chi2774 Phase 13 trace, these emit 1,202 (__setitem__) + 496 (clear)
= 1,698 elementwise launches/window in compress_extend_old's per-request loop.
Each pair of kv/score launches collapses into a single Triton dispatch.

Strategy:
  - The slice-setitem path (dst[a:b] = src[c:d]) covers compress_extend_old
    lines 1400, 1401, 1407.
  - The multi-axis fancy-index path (pool[req_idx, write_pos] = src) at
    compress_decode_old:1497 is NOT covered here — that's Option B's
    multi-axis-index megakernel work.
  - For unsupported index types we fall back to torch; the kernel returns
    False and the caller does the original 2-launch dance.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# Skip Triton for trivially small slices.
MIN_ELEMS_FOR_TRITON = 64


@triton.jit
def _set_kv_score_slice_kernel(
    dst_kv_ptr, dst_score_ptr,
    src_kv_ptr, src_score_ptr,
    dst_start, src_start,
    LAST_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused slice copy: dst.kv[dst_start+i] = src.kv[src_start+i]
    AND dst.score[dst_start+i] = src.score[src_start+i] for i in [0, num_rows).

    Each program handles BLOCK_N rows × LAST_DIM cols of both tensors.
    """
    pid = tl.program_id(0)
    pid_d = tl.program_id(1)

    rows = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    cols = pid_d * LAST_DIM + tl.arange(0, LAST_DIM)
    mask = (cols < LAST_DIM)[None, :] & tl.full((BLOCK_N, 1), 1, dtype=tl.int1)

    # Compute offsets (assumes row-major contiguous: stride[0] = LAST_DIM, stride[1] = 1)
    src_off = (src_start + rows)[:, None] * LAST_DIM + cols[None, :]
    dst_off = (dst_start + rows)[:, None] * LAST_DIM + cols[None, :]

    kv = tl.load(src_kv_ptr + src_off, mask=mask)
    score = tl.load(src_score_ptr + src_off, mask=mask)
    tl.store(dst_kv_ptr + dst_off, kv, mask=mask)
    tl.store(dst_score_ptr + dst_off, score, mask=mask)


def setitem_slice_triton(
    dst_kv: torch.Tensor, dst_score: torch.Tensor,
    src_kv: torch.Tensor, src_score: torch.Tensor,
    dst_start: int, src_start: int, num_rows: int,
) -> bool:
    """Fused 2-tensor slice scatter. Returns True if Triton fired.

    Handles only the contiguous-slice case along dim 0:
        dst.kv  [dst_start : dst_start+num_rows] = src.kv  [src_start : src_start+num_rows]
        dst.score[dst_start : dst_start+num_rows] = src.score[src_start : src_start+num_rows]

    Both tensors must be 2D and contiguous, with the same dtype within each pair.
    """
    if num_rows == 0:
        return True
    if dst_kv.dim() != 2 or src_kv.dim() != 2:
        return False
    if not (dst_kv.is_contiguous() and dst_score.is_contiguous()
            and src_kv.is_contiguous() and src_score.is_contiguous()):
        return False
    if dst_kv.shape[1] != src_kv.shape[1] or dst_score.shape[1] != src_score.shape[1]:
        return False
    if dst_kv.shape[1] != dst_score.shape[1]:
        return False
    if dst_kv.dtype != src_kv.dtype or dst_score.dtype != src_score.dtype:
        return False

    LAST_DIM = dst_kv.shape[1]
    if num_rows * LAST_DIM < MIN_ELEMS_FOR_TRITON:
        return False

    # Tuned: BLOCK_N=8 gives one row block for typical small-bs decode shapes,
    # num_warps=1 minimizes launch overhead. Single grid_d block when LAST_DIM<=512.
    BLOCK_N = 8
    n_row_blocks = (num_rows + BLOCK_N - 1) // BLOCK_N
    grid = (n_row_blocks, 1)

    # NOTE: BLOCK_D is fixed at LAST_DIM (no tiling along col dim) for
    # simplicity — works for head_dim_times_coff up to 512.
    if LAST_DIM > 1024:
        return False  # fall back; single-tile too wide

    _set_kv_score_slice_kernel[grid](
        dst_kv, dst_score, src_kv, src_score,
        dst_start, src_start,
        LAST_DIM=LAST_DIM,
        BLOCK_N=BLOCK_N,
        num_warps=1, num_stages=1,
    )
    return True


@triton.jit
def _clear_kv_score_kernel(
    kv_ptr, score_ptr,
    n_elem,
    fill_score,  # python float -> tl scalar; use tl.where to coerce dtype if needed
    BLOCK_N: tl.constexpr,
):
    """Single-launch clear: kv = 0, score = fill_score (typically -inf).

    Both tensors are 1D-flattened (caller guarantees same numel).
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n_elem
    tl.store(kv_ptr + offsets,
             tl.zeros((BLOCK_N,), dtype=kv_ptr.dtype.element_ty),
             mask=mask)
    tl.store(score_ptr + offsets,
             tl.full((BLOCK_N,), fill_score, dtype=score_ptr.dtype.element_ty),
             mask=mask)


def clear_kv_score_triton(
    kv: torch.Tensor, score: torch.Tensor, fill_score: float,
) -> bool:
    """Fused clear: kv.zero_() + score.fill_(fill_score) in one launch.

    Returns True if Triton fired, False if caller should fall back.
    """
    if kv.numel() != score.numel():
        return False
    if not (kv.is_contiguous() and score.is_contiguous()):
        return False
    n_elem = kv.numel()
    if n_elem < MIN_ELEMS_FOR_TRITON:
        return False

    BLOCK_N = 1024
    grid = (triton.cdiv(n_elem, BLOCK_N),)
    _clear_kv_score_kernel[grid](
        kv.view(-1), score.view(-1),
        n_elem,
        float(fill_score),
        BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=1,
    )
    return True
