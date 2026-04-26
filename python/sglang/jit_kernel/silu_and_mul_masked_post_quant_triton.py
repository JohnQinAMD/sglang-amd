"""HIP fallback for csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh.

Per-token, per-group SwiGLU + fp8 quantization for masked-EP MoE. Each group
is `quant_group_size=128` columns of the gate_up output (`gateup`), produces
a fp8_e4m3 down-projection input plus a per-group scale.

Two layouts:
  - normal:     scale dtype=fp32, shape [E, T, G]              (G = D / quant_group_size)
  - transposed: scale dtype=int32, shape [E, G/4, T]    (4 ue8m0 scales packed as int32)

Production DSv4 path uses transposed=True + scale_ue8m0=True; that's also
what the existing CUDA kernel's compile_aot pre-builds. The non-transposed
path falls through to the existing Triton kernel in
`srt/layers/moe/ep_moe/kernels.py:silu_and_mul_masked_post_quant_fwd` which
already runs on HIP.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


_FP8_MAX = tl.constexpr(240.0)


@triton.jit
def _silu_mul_quant_transposed_ue8m0_kernel(
    input_ptr,            # bf16 [E, T, 2D]
    output_ptr,           # fp8_e4m3fn [E, T, D]
    output_scale_ptr,     # int32 [E, G/4, T] (4 ue8m0 packed per int32)
    masked_m_ptr,         # i32 [E]
    swiglu_limit,         # float; 0 means "no clamp"
    APPLY_SWIGLU_LIMIT: tl.constexpr,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,     # 128
    G: tl.constexpr,              # D // GROUP_SIZE
    T: tl.constexpr,
):
    # grid: (E, T, G)
    eid = tl.program_id(0)
    tid = tl.program_id(1)
    gid = tl.program_id(2)

    masked = tl.load(masked_m_ptr + eid)
    if tid >= masked:
        return

    # Load gate and up vectors for this group
    col = tl.arange(0, GROUP_SIZE)
    gate_off = eid * T * 2 * D + tid * 2 * D + gid * GROUP_SIZE + col
    up_off   = gate_off + D
    gate = tl.load(input_ptr + gate_off).to(tl.float32)
    up   = tl.load(input_ptr + up_off).to(tl.float32)

    if APPLY_SWIGLU_LIMIT:
        gate = tl.minimum(gate, swiglu_limit)
        up = tl.minimum(tl.maximum(up, -swiglu_limit), swiglu_limit)
        silu = gate / (1.0 + tl.exp(-gate))
        val = up * silu
    else:
        # bf16-rounded silu mirrors the non-fused CUDA path (silu_d cast to bf16
        # then multiplied with up_vec). Match that here so byte-equality with
        # the CUDA kernel is preserved.
        silu = gate / (1.0 + tl.exp(-gate))
        silu_bf16 = silu.to(tl.bfloat16).to(tl.float32)
        val = up * silu_bf16

    abs_max = tl.maximum(tl.max(tl.abs(val)), 1e-10)
    raw_scale = abs_max / _FP8_MAX

    # ue8m0: ceil-up exponent
    bits = raw_scale.to(tl.int32, bitcast=True)
    exp_field = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    ue8m0 = (exp_field + tl.where(mant != 0, 1, 0)).to(tl.int32)
    scale = (ue8m0 << 23).to(tl.float32, bitcast=True)
    inv_scale = 1.0 / scale

    quantized = val * inv_scale
    quantized = tl.maximum(tl.minimum(quantized, _FP8_MAX), -_FP8_MAX)
    fp8 = quantized.to(tl.float8e4nv)

    # Write fp8 row
    out_off = eid * T * D + tid * D + gid * GROUP_SIZE + col
    tl.store(output_ptr + out_off, fp8)

    # Write packed int32 scale into [E, G/4, T] layout
    # Physical byte address: base + eid*G*T + (gid/4)*T*4 + tid*4 + (gid%4)
    # => int32 element index: (eid*G*T + (gid/4)*T*4 + tid*4 + (gid%4)) / 4
    #    = eid*G*T/4 + (gid/4)*T + tid + (gid%4)/4   -- not aligned; use bytes view.
    # Use atomic byte write instead: cast scale_ptr to uint8.
    scale_bptr = output_scale_ptr.to(tl.pointer_type(tl.uint8))
    byte_off = eid * G * T + (gid // 4) * T * 4 + tid * 4 + (gid % 4)
    tl.store(scale_bptr + byte_off, ue8m0.to(tl.uint8))


def silu_and_mul_masked_post_quant_transposed_ue8m0(
    input: torch.Tensor,                  # bf16 [E, T, 2D]
    output: torch.Tensor,                  # fp8_e4m3fn [E, T, D]
    output_scale: torch.Tensor,            # int32 [E, G/4, T]
    quant_group_size: int,
    masked_m: torch.Tensor,                # i32 [E]
    swiglu_limit: float = 0.0,
) -> None:
    assert quant_group_size == 128
    assert input.dtype == torch.bfloat16
    assert output.dtype == torch.float8_e4m3fn
    assert output_scale.dtype == torch.int32
    assert masked_m.dtype == torch.int32
    E, T, D2 = input.shape
    D = D2 // 2
    G = D // quant_group_size
    assert output.shape == (E, T, D)
    assert output_scale.shape == (E, G // 4, T), \
        f"transposed scale must be [E, G//4, T], got {output_scale.shape}"
    assert G % 4 == 0
    assert T % 4 == 0

    grid = (E, T, G)
    _silu_mul_quant_transposed_ue8m0_kernel[grid](
        input, output, output_scale, masked_m,
        float(swiglu_limit),
        APPLY_SWIGLU_LIMIT=(swiglu_limit != 0.0),
        D=D, GROUP_SIZE=quant_group_size, G=G, T=T,
    )


def silu_and_mul_masked_post_quant_hip(
    input: torch.Tensor,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    quant_group_size: int,
    masked_m: torch.Tensor,
    scale_ue8m0: bool = False,
    topk: int = 8,
    transposed: bool = False,
    swiglu_limit: float = 0.0,
) -> None:
    """HIP entry point. Routes to the transposed+ue8m0 Triton kernel for the
    DSv4 production path; falls through to the existing ep_moe Triton kernel
    for the non-transposed path."""
    if transposed:
        assert scale_ue8m0, "transposed layout requires scale_ue8m0=True"
        silu_and_mul_masked_post_quant_transposed_ue8m0(
            input, output, output_scale, quant_group_size, masked_m,
            swiglu_limit=swiglu_limit,
        )
        return

    # Non-transposed path → reuse the existing ep_moe Triton kernel.
    from sglang.srt.layers.moe.ep_moe.kernels import (
        silu_and_mul_masked_post_quant_fwd,
    )
    silu_and_mul_masked_post_quant_fwd(
        input, output, output_scale, quant_group_size, masked_m,
        scale_ue8m0=scale_ue8m0,
    )
