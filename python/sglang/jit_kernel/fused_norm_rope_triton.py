"""HIP fallback for csrc/deepseek_v4/fused_norm_rope.cuh.

Per-token in-place RMSNorm followed by RoPE on the last `kRopeDim` lanes.
Three modes (matching the .cuh enum and the deepseek_v4.py wrapper at
`compress_fused_norm_rope_inplace` / `fused_norm_rope_inplace`):

  CompressExtend  (mode=0): handle = PrefillPlan[N]; position = plan.position - cr + 1
                            input row = ragged_id (skip when ragged_id == -1)
  CompressDecode  (mode=1): handle = seq_lens int32[N]; position = seq_len - cr
                            skip when seq_len % cr != 0
  DefaultForward  (mode=2): handle = positions int64[N]; position = positions[i]

Layout:
  input:     [N, kHeadDim]  bf16 / fp32
  weight:    [kHeadDim]      same dtype
  freqs_cis: [max_pos, kRopeDim]  float32 (real/imag interleaved)

PrefillPlan struct (compress.cuh, kPrefillPlanDim=16 bytes per plan):
  uint32 ragged_id;  uint32 batch_id;  uint32 position;  uint32 window_len;
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# Mode constants must match the wrapper's `mode` argument.
_MODE_COMPRESS_EXTEND = 0
_MODE_COMPRESS_DECODE = 1
_MODE_DEFAULT_FORWARD = 2


@triton.jit
def _fused_norm_rope_kernel(
    input_ptr,             # bf16/fp32 [N, head_dim]
    weight_ptr,            # bf16/fp32 [head_dim]
    handle_ptr,            # mode-dependent
    freqs_cis_ptr,         # fp32 [max_pos, rope_dim] (real/imag interleaved)
    eps,
    compress_ratio,
    num_works,
    MODE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK: tl.constexpr,    # = HEAD_DIM (one warp/row, padded if needed)
):
    bid = tl.program_id(0)
    if bid >= num_works:
        return

    if MODE == 0:  # CompressExtend
        # handle is PrefillPlan[N]: ragged_id, batch_id, position, window_len
        plan_off = bid * 4
        ragged_id = tl.load(handle_ptr + plan_off + 0).to(tl.uint32)
        position_raw = tl.load(handle_ptr + plan_off + 2).to(tl.int32)
        if ragged_id == 0xFFFFFFFF:
            return
        row = ragged_id.to(tl.int64)
        position = position_raw + 1 - compress_ratio
    elif MODE == 1:  # CompressDecode
        seq_len = tl.load(handle_ptr + bid).to(tl.int32)
        if seq_len % compress_ratio != 0:
            return
        row = bid.to(tl.int64)
        position = seq_len - compress_ratio
    else:           # DefaultForward (mode=2)
        # handle is int64 positions
        position = tl.load(handle_ptr + bid).to(tl.int32)
        row = bid.to(tl.int64)

    col = tl.arange(0, BLOCK)
    mask = col < HEAD_DIM
    in_off = row * HEAD_DIM + col

    x = tl.load(input_ptr + in_off, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + col, mask=mask, other=0.0).to(tl.float32)

    sum_sq = tl.sum(x * x, axis=0)
    norm_factor = tl.rsqrt(sum_sq / HEAD_DIM + eps)
    y = x * norm_factor * w

    # RoPE on the last ROPE_DIM lanes
    rope_start = HEAD_DIM - ROPE_DIM
    in_rope = (col >= rope_start) & mask

    # Load freq pair (real, imag) for this position. rope_dim=64 → 32 complex pairs.
    # freqs layout is [max_pos, rope_dim] with real/imag interleaved (matches the
    # `freq_cis = torch.view_as_real(freqs_cis).flatten(-2)` in the wrapper).
    rope_lane = col - rope_start                               # 0..ROPE_DIM-1 in rope region
    pair_idx = rope_lane // 2                                  # complex-pair index (0..ROPE_DIM/2-1)
    is_imag = (rope_lane & 1) == 1
    # freal at offset 2*pair_idx, fimag at 2*pair_idx + 1
    freq_base = position.to(tl.int64) * ROPE_DIM
    freq_real = tl.load(freqs_cis_ptr + freq_base + 2 * pair_idx,
                        mask=in_rope, other=0.0)
    freq_imag = tl.load(freqs_cis_ptr + freq_base + 2 * pair_idx + 1,
                        mask=in_rope, other=0.0)

    # We have (real, imag) interleaved within `y` too: pair = (y[2k], y[2k+1])
    # Need partner: load partner via shfl by xor-1. Since BLOCK fits a warp/CTA,
    # use a second tl.load on the partner offset.
    partner_lane = col ^ 1
    partner_off = row * HEAD_DIM + partner_lane
    partner_in = tl.load(input_ptr + partner_off, mask=in_rope, other=0.0).to(tl.float32)
    partner_w = tl.load(weight_ptr + partner_lane, mask=in_rope, other=0.0).to(tl.float32)
    partner_y = partner_in * norm_factor * partner_w

    # y[2k] = real, y[2k+1] = imag → out[2k]   = real*fr - imag*fi
    #                                  out[2k+1] = real*fi + imag*fr
    # For lane (col), if even (real): partner is imag. If odd (imag): partner is real.
    real = tl.where(is_imag, partner_y, y)
    imag = tl.where(is_imag, y, partner_y)
    rope_out = tl.where(
        is_imag,
        real * freq_imag + imag * freq_real,
        real * freq_real - imag * freq_imag,
    )

    out = tl.where(in_rope, rope_out, y).to(input_ptr.dtype.element_ty)
    tl.store(input_ptr + in_off, out, mask=mask)


def fused_norm_rope_inplace_hip(
    input: torch.Tensor,        # [N, head_dim] bf16/fp32
    weight: torch.Tensor,        # [head_dim]
    handle: torch.Tensor,        # mode-dependent
    freqs_cis: torch.Tensor,     # [max_pos, rope_dim] fp32 (real/imag interleaved)
    mode: int,
    eps: float,
    compress_ratio: int,
) -> None:
    """In-place RMSNorm + RoPE — HIP Triton port."""
    assert mode in (_MODE_COMPRESS_EXTEND, _MODE_COMPRESS_DECODE, _MODE_DEFAULT_FORWARD)
    N, head_dim = input.shape
    rope_dim = freqs_cis.shape[-1]
    assert weight.shape == (head_dim,)
    assert rope_dim % 2 == 0
    assert head_dim >= rope_dim

    if mode == _MODE_COMPRESS_EXTEND:
        # handle is [N, kPrefillPlanDim/4=4] uint32
        if handle.dtype not in (torch.uint32, torch.int32):
            handle = handle.view(torch.int32)
        num_works = handle.shape[0]
    elif mode == _MODE_COMPRESS_DECODE:
        assert handle.dtype == torch.int32, f"got {handle.dtype}"
        num_works = handle.shape[0]
    else:  # default forward
        assert handle.dtype == torch.int64, f"got {handle.dtype}"
        num_works = handle.shape[0]

    BLOCK = triton.next_power_of_2(head_dim)
    grid = (num_works,)
    _fused_norm_rope_kernel[grid](
        input, weight, handle, freqs_cis,
        float(eps), int(compress_ratio), num_works,
        MODE=mode, HEAD_DIM=head_dim, ROPE_DIM=rope_dim, BLOCK=BLOCK,
    )
