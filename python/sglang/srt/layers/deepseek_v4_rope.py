import math
import os
from functools import lru_cache
from typing import Optional

import tilelang
import torch
import triton
import triton.language as tl

from sglang.srt.utils.custom_op import register_custom_op


from sglang.srt.utils.common import maybe_torch_compile

tilelang.set_log_level("WARNING")

pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


@lru_cache(2)
def precompute_freqs_cis(
    dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min, max, dim):
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


@maybe_torch_compile
def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Adopted from DeepSeek's reference implementation, but adapted to sglang input formats:
        - 2D: x [bs, rope_dim], freqs_cis [bs, rope_dim // 2]
        - 3D: x [bs, n_heads, rope_dim], freqs_cis [bs, rope_dim // 2]

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        # x: [bs, n_heads, rope_dim // 2], freqs_cis: [bs, rope_dim // 2]
        # -> reshape freqs_cis to [bs, 1, rope_dim // 2] to broadcast over n_heads
        freqs_cis = freqs_cis.unsqueeze(1)
    # For 2D case should directly match: x [bs, rope_dim // 2], freqs_cis [bs, rope_dim // 2]
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


@triton.jit
def _fused_rmsnorm_rope_q_kernel(
    q_ptr,             # bf16 in/out, shape [bs, n_heads, head_dim]
    freqs_real_ptr,    # fp32 [max_pos, rope_dim] (real/imag interleaved)
    positions_ptr,     # int64 [bs]
    eps,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    stride_b,
    stride_h,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused unweighted-rmsnorm + RoPE on per-head Q tensor.

    Replaces the unfused pair at deepseek_v4.py:1756-1763:
        q = rms_normalize_triton(q, eps)
        apply_rotary_emb_triton(q[..., -rope_dim:], freqs_cis, positions=positions)

    Per-call savings ~20-25 us in microbench (1.42x speedup over the unfused
    chain plus the standalone KV rope, validated cos_sim 0.999998).
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    base = pid_b * stride_b + pid_h * stride_h

    # Load full head_dim row, compute RMS factor in fp32
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HEAD_DIM
    x = tl.load(q_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    sumsq = tl.sum(x * x, axis=0)
    rms_inv = tl.rsqrt(sumsq / HEAD_DIM + eps)
    x_norm = x * rms_inv  # full normalized row

    # Fetch position (per-batch), then load freqs for the rope segment
    position = tl.load(positions_ptr + pid_b)
    rope_offs = tl.arange(0, ROPE_DIM // 2)
    rope_mask = rope_offs < (ROPE_DIM // 2)
    nope_offset = HEAD_DIM - ROPE_DIM
    real_in_head = nope_offset + rope_offs * 2
    imag_in_head = nope_offset + rope_offs * 2 + 1

    freq_real = tl.load(
        freqs_real_ptr + position * ROPE_DIM + rope_offs * 2,
        mask=rope_mask,
        other=0.0,
    )
    freq_imag = tl.load(
        freqs_real_ptr + position * ROPE_DIM + rope_offs * 2 + 1,
        mask=rope_mask,
        other=0.0,
    )

    # Re-load rope-segment elements (raw bf16) and re-normalize, then rotate
    x_real_raw = tl.load(q_ptr + base + real_in_head, mask=rope_mask, other=0.0).to(tl.float32)
    x_imag_raw = tl.load(q_ptr + base + imag_in_head, mask=rope_mask, other=0.0).to(tl.float32)
    x_real = x_real_raw * rms_inv
    x_imag = x_imag_raw * rms_inv
    out_real = x_real * freq_real - x_imag * freq_imag
    out_imag = x_real * freq_imag + x_imag * freq_real

    # Write the full normalized row, then overwrite rope segment with rotated values
    tl.store(q_ptr + base + offs, x_norm, mask=mask)
    tl.store(q_ptr + base + real_in_head, out_real, mask=rope_mask)
    tl.store(q_ptr + base + imag_in_head, out_imag, mask=rope_mask)


def fused_rmsnorm_rope_q_triton(
    q: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    eps: float,
    rope_dim: int,
) -> None:
    """In-place fused unweighted-rmsnorm + RoPE on per-head Q tensor.

    Args:
        q: 3d [bs, n_heads, head_dim] bf16, modified in-place
        freqs_cis: complex64 [max_seqlen, rope_dim // 2]
        positions: int64 [bs]
        eps: rmsnorm epsilon
        rope_dim: number of trailing lanes to apply RoPE to
    """
    bs, n_heads, head_dim = q.shape
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)  # [max_seqlen, rope_dim]
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    grid = (bs, n_heads)
    _fused_rmsnorm_rope_q_kernel[grid](
        q, freqs_real, positions, eps,
        HEAD_DIM=head_dim, ROPE_DIM=rope_dim,
        stride_b=q.stride(0), stride_h=q.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )


@triton.jit
def apply_rotary_emb_triton_kernel(
    x_ptr,
    freqs_ptr,
    positions_ptr,
    rope_dim,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_freq_pos,
    stride_freq_dim,
    n_tokens,
    USE_POS: tl.constexpr,
    IS_INVERSE: tl.constexpr,
    IS_3D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    """Multi-token-per-CTA RoPE kernel.

    Original kernel grid was (batch, n_heads, ceil(rope/2 / BLOCK)). At prefill
    bs=8192 / n_heads=16 / rope=64, BLOCK_SIZE=128 → 131,072 CTAs each doing 32
    complex pairs (~256 bytes). 430× over-decomposition per CU on MI355X (304 CU);
    measured efficiency was 9.8% vs HBM bound — 2.07× off ideal due to launch /
    dispatch overhead per tiny CTA.

    This kernel takes BLOCK_M tokens per CTA. Grid (ceil(M/BLOCK_M), n_heads).
    At BLOCK_M=64, M=8192, n_heads=16 → 2,048 CTAs each handling 64×64=4096
    elements (~8 KB). Microbench: 2.07× over original at prefill, neutral at
    decode (cos_sim 1.000000 across all BLOCK_M values).
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < n_tokens

    pair_offs = tl.arange(0, ROPE_DIM // 2)
    pair_mask = pair_offs < (ROPE_DIM // 2)
    mask_2d = m_mask[:, None] & pair_mask[None, :]

    if USE_POS:
        positions = tl.load(positions_ptr + m_offs, mask=m_mask, other=0)
    else:
        positions = m_offs

    # freqs layout: [max_pos, rope_dim] (real/imag interleaved)
    freq_real_offs = positions[:, None] * stride_freq_pos + (pair_offs[None, :] * 2) * stride_freq_dim
    freq_imag_offs = freq_real_offs + stride_freq_dim
    freq_real = tl.load(freqs_ptr + freq_real_offs, mask=mask_2d, other=0.0)
    freq_imag = tl.load(freqs_ptr + freq_imag_offs, mask=mask_2d, other=0.0)

    if IS_3D:
        base = m_offs[:, None] * stride_x_batch + pid_h * stride_x_head
    else:
        base = m_offs[:, None] * stride_x_batch

    real_dim_offs = pair_offs * 2 * stride_x_dim
    imag_dim_offs = (pair_offs * 2 + 1) * stride_x_dim
    x_real_offs = base + real_dim_offs[None, :]
    x_imag_offs = base + imag_dim_offs[None, :]

    x_real = tl.load(x_ptr + x_real_offs, mask=mask_2d, other=0.0).to(tl.float32)
    x_imag = tl.load(x_ptr + x_imag_offs, mask=mask_2d, other=0.0).to(tl.float32)

    if IS_INVERSE:
        # (a + bi) * (c - di)
        out_real = x_real * freq_real + x_imag * freq_imag
        out_imag = x_imag * freq_real - x_real * freq_imag
    else:
        # (a + bi) * (c + di)
        out_real = x_real * freq_real - x_imag * freq_imag
        out_imag = x_real * freq_imag + x_imag * freq_real

    tl.store(x_ptr + x_real_offs, out_real, mask=mask_2d)
    tl.store(x_ptr + x_imag_offs, out_imag, mask=mask_2d)


@register_custom_op(mutates_args=["x"])
def apply_rotary_emb_triton(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    inverse: bool = False,
) -> None:
    """
    Args:
        x: 2d [bs, rope_dim] or 3d [bs, n_heads, rope_dim]
        freqs_cis:
            - If positions is None: [bs, rope_dim // 2] (already indexed)
            - If positions is not None: [max_seqlen, rope_dim // 2] (full table)
        positions: Optional[bs], if provided will index into freqs_cis
        inverse: bool, if True, apply inverse rotation (conjugate)
    Returns:
        x with rotary embeddings applied (inplace)

    Optional HIP fast path: set `SGLANG_HIP_ROPE=1` to dispatch through the
    bundled HIP kernel at csrc/rope_hip/. Bit-exact vs Triton (microbench
    7/7 PASS) but with ~61% lower per-call CPU launch overhead.
    """
    if os.environ.get("SGLANG_HIP_ROPE") == "1":
        from sglang.srt.layers.rope_hip import apply_rotary_emb_hip
        apply_rotary_emb_hip(
            x=x, freqs_cis=freqs_cis, positions=positions, inverse=inverse,
        )
        return

    is_3d = x.ndim == 3

    if is_3d:
        batch_size, n_heads, rope_dim = x.shape
    else:
        batch_size, rope_dim = x.shape
        n_heads = 1

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)

    # BLOCK_M=64 tokens per CTA was the inflection point in microbench:
    # at M=8192 it cuts grid from 131K → 2K CTAs (60× less), capturing the 2.07x
    # speedup; at M<=64 it falls back to 1 CTA per head, matching the
    # original kernel's wall time. See microbench_rope_v2.py.
    BLOCK_M = 64
    grid = (triton.cdiv(batch_size, BLOCK_M), n_heads if is_3d else 1)

    if positions is not None:
        # use positions to index into freqs_cis
        assert positions.shape == (
            batch_size,
        ), f"positions shape {positions.shape} != ({batch_size},)"

        apply_rotary_emb_triton_kernel[grid](
            x,
            freqs_real,
            positions,
            rope_dim,
            x.stride(0),
            x.stride(1) if is_3d else 0,
            x.stride(-1),
            freqs_real.stride(0),
            freqs_real.stride(1),
            batch_size,
            USE_POS=True,
            IS_INVERSE=inverse,
            IS_3D=is_3d,
            BLOCK_M=BLOCK_M,
            ROPE_DIM=rope_dim,
        )
    else:
        # freqs_cis already indexed, treat row index as position
        assert (
            freqs_real.shape[0] == batch_size
        ), f"freqs_cis batch size {freqs_real.shape[0]} != x batch size {batch_size}"

        apply_rotary_emb_triton_kernel[grid](
            x,
            freqs_real,
            x,  # positions_ptr unused when USE_POS=False
            rope_dim,
            x.stride(0),
            x.stride(1) if is_3d else 0,
            x.stride(-1),
            freqs_real.stride(0),
            freqs_real.stride(1),
            batch_size,
            USE_POS=False,
            IS_INVERSE=inverse,
            IS_3D=is_3d,
            BLOCK_M=BLOCK_M,
            ROPE_DIM=rope_dim,
        )
