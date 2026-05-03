"""Fused freqs_idx + freqs_cis gather (B200-style page-table arithmetic absorption).

Replaces the 4-launch chain at deepseek_v4.py:1739-1740:

    _freqs_idx = (seq_lens - 1) // self.ratio * self.ratio   # 3 launches: sub, floor_divide, mul
    _freqs_per_bs = torch.view_as_real(self.freqs_cis[_freqs_idx]).contiguous()  # 1 index + view + contig

with a single Triton kernel that:
  1. Per-batch computes `idx = (seq_lens[b] - 1) // ratio * ratio`
  2. Gathers freqs_cis[idx] (a row of complex64 / float32 pairs)
  3. Writes the result as float32 (B, rope_dim, 2) → equivalent to view_as_real

Target: 4 launches → 1 launch per layer; with 22 layers × 1.5 µs ≈ 0.13 ms TPOT.

Per `feedback_data_ptr_caching_unsafe.md` the per-step memoization approach
(SGLANG_FREQS_IDX_CACHE) was tried and reverted due to id() reuse causing
Paris→London bugs. This kernel-level fusion is the safe alternative — no
caching, just fewer launches.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _freqs_idx_gather_kernel(
    seq_lens_ptr,       # int64 [B]
    freqs_cis_ptr,      # complex64 [max_seq_len, rope_dim//2] (treated as float32 [max_seq_len, rope_dim])
    out_ptr,            # float32 [B, rope_dim//2, 2] (= float32 [B, rope_dim])
    ratio,              # int (per-NSA-layer compress ratio: 4 or 128)
    B,
    ROPE_DIM_FLAT: tl.constexpr,   # rope_dim//2 * 2 = rope_dim (number of float32 per row)
    BLOCK_K: tl.constexpr,
):
    """One program per batch row.

    Computes:  freqs_per_bs[b, :] = freqs_cis_as_float[idx_b, :]
    where     idx_b = (seq_lens[b] - 1) // ratio * ratio
    """
    pid_b = tl.program_id(0)
    if pid_b >= B:
        return

    # Compute idx for this batch row. Use int64 to match seq_lens dtype.
    seq_len_b = tl.load(seq_lens_ptr + pid_b)
    idx_b = ((seq_len_b - 1) // ratio) * ratio

    # Gather a row of ROPE_DIM_FLAT float32 values from freqs_cis_ptr at row idx_b.
    # freqs_cis is viewed as float32 [max_seq_len, ROPE_DIM_FLAT].
    for k_start in range(0, ROPE_DIM_FLAT, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < ROPE_DIM_FLAT
        src_offs = idx_b * ROPE_DIM_FLAT + k_offs
        dst_offs = pid_b * ROPE_DIM_FLAT + k_offs
        v = tl.load(freqs_cis_ptr + src_offs, mask=k_mask, other=0.0)
        tl.store(out_ptr + dst_offs, v, mask=k_mask)


def freqs_idx_gather_triton(
    seq_lens: torch.Tensor,      # int64 [B]
    freqs_cis: torch.Tensor,     # complex64 [max_seq_len, rope_dim//2]
    ratio: int,
) -> torch.Tensor:
    """Fused drop-in for:

        _freqs_idx = (seq_lens - 1) // ratio * ratio
        return torch.view_as_real(freqs_cis[_freqs_idx]).contiguous()

    Returns float32 (B, rope_dim//2, 2) tensor.
    """
    assert freqs_cis.is_contiguous(), "freqs_cis must be contiguous"
    assert freqs_cis.dtype == torch.complex64, (
        f"expected complex64 freqs_cis, got {freqs_cis.dtype}"
    )
    B = seq_lens.size(0)
    max_seq_len, rope_dim_half = freqs_cis.size()
    rope_dim_flat = rope_dim_half * 2  # complex → 2 floats per element

    # View freqs_cis as float32 [max_seq_len, rope_dim_flat].
    freqs_flat = torch.view_as_real(freqs_cis).reshape(max_seq_len, rope_dim_flat)

    # Output shape (B, rope_dim_half, 2) == (B, rope_dim_flat) layout.
    out = torch.empty(
        (B, rope_dim_half, 2),
        dtype=torch.float32,
        device=seq_lens.device,
    )

    # BLOCK_K: pow2, large enough to cover most rope dims in 1-2 iterations.
    BLOCK_K = min(triton.next_power_of_2(rope_dim_flat), 256)

    grid = (B,)
    _freqs_idx_gather_kernel[grid](
        seq_lens.contiguous(),
        freqs_flat,
        out,
        int(ratio),
        B,
        ROPE_DIM_FLAT=rope_dim_flat,
        BLOCK_K=BLOCK_K,
    )
    return out
