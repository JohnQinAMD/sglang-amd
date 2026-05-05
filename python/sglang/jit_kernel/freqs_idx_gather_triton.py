"""Fused freqs_idx + gather for the decode-shape RoPE freqs lookup.

Replaces the 4-launch chain in deepseek_v4.py:

    _freqs_idx = (seq_lens - 1) // ratio * ratio        # 3 launches: sub, floor_div, mul
    freqs_cis_per_bs = self.freqs_cis[_freqs_idx]       # 1 launch: index_select

with a single Triton kernel that computes idx and gathers complex64 rows in
one pass. Returns a complex64 tensor with the same shape as
`self.freqs_cis[_freqs_idx]` so the downstream `fused_norm_rope_inplace_triton`
(which accepts complex64) is a drop-in.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_hip

_is_hip = is_hip()


@triton.jit
def _freqs_idx_gather_complex_kernel(
    seq_lens_ptr,
    freqs_re_ptr,
    out_re_ptr,
    ratio,
    ROPE_FLAT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    seq_len_b = tl.load(seq_lens_ptr + pid_b)
    idx_b = ((seq_len_b - 1) // ratio) * ratio
    for k_start in range(0, ROPE_FLAT, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < ROPE_FLAT
        v = tl.load(freqs_re_ptr + idx_b * ROPE_FLAT + k_offs, mask=k_mask, other=0.0)
        tl.store(out_re_ptr + pid_b * ROPE_FLAT + k_offs, v, mask=k_mask)


def _freqs_idx_gather_triton(
    seq_lens: torch.Tensor,
    freqs_cis: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    assert freqs_cis.is_contiguous()
    assert freqs_cis.dtype == torch.complex64
    assert seq_lens.dtype in (torch.int32, torch.int64)
    assert seq_lens.device == freqs_cis.device
    B = seq_lens.size(0)
    if B == 0:
        return torch.empty((0, freqs_cis.size(1)), dtype=torch.complex64, device=freqs_cis.device)
    max_seq_len, rope_half = freqs_cis.size()
    rope_flat = rope_half * 2

    freqs_re = torch.view_as_real(freqs_cis).reshape(max_seq_len, rope_flat)
    out = torch.empty((B, rope_half), dtype=torch.complex64, device=freqs_cis.device)
    out_re = torch.view_as_real(out).reshape(B, rope_flat)

    BLOCK_K = min(triton.next_power_of_2(rope_flat), 256)
    _freqs_idx_gather_complex_kernel[(B,)](
        seq_lens.contiguous(),
        freqs_re,
        out_re,
        int(ratio),
        ROPE_FLAT=rope_flat,
        BLOCK_K=BLOCK_K,
    )
    return out


def freqs_idx_gather(
    seq_lens: torch.Tensor,
    freqs_cis: torch.Tensor,
    ratio: int,
) -> torch.Tensor:
    """Drop-in for `freqs_cis[(seq_lens - 1) // ratio * ratio]`.

    Triton-fused on HIP; falls back to the original torch indexing on CUDA.
    """
    if _is_hip:
        return _freqs_idx_gather_triton(seq_lens, freqs_cis, ratio)
    return freqs_cis[(seq_lens - 1) // ratio * ratio]
