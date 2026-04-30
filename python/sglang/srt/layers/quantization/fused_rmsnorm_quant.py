"""Fused (add_rmsnorm + per-1x128 fp8 quant) Triton kernel.

A2-#1 prototype for DSv4 Flash-Base FP8 on AMD MI355X (gfx950).

The unfused production path is two consecutive launches:

    add_rmsnorm(out, x, residual, residual_out, w, eps)            # bf16
    q_input, x_scale = aiter_per1x128_quant(out, fp8)              # 2nd launch

This module fuses them into one Triton launch. The output bf16 `normed` of
add_rmsnorm is consumed only by the per-1x128 quant, so it never needs to
land in HBM — the fused kernel keeps it register-resident.

Microbench (M=8, N=4096, MI355X, ROCm 7.2):
    UNFUSED aiter:    12.37 us / call (cuda graph replay)
    FUSED Triton:     10.28 us / call (this module)
    Aiter HIP fused:   9.64 us / call (upper-bound reference; not used)

At ~140 fp8-quant callsites/step, that's ~0.29 ms / step under cuda-graph
replay, ~1.45 ms / step in eager.

Gated by env var SGLANG_FUSED_RMSNORM_QUANT_PER1x128 (default OFF). Enable
when ready by exporting `SGLANG_FUSED_RMSNORM_QUANT_PER1x128=1` before launch.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


__all__ = [
    "fused_add_rmsnorm_per1x128_quant",
    "fused_add_rmsnorm_per1x128_quant_dual",
    "fused_rmsnorm_per1x128_quant",
    "fused_rmsnorm_per1x128_quant_dual",
]


# Per-1x128 group size for DSv4 FP8. This is fixed by the quant scheme.
_GROUP = 128


@triton.jit
def _fused_add_rmsnorm_per1x128_quant_kernel(
    x_ptr,          # bf16 [M, N], pre-add input
    res_ptr,        # bf16 [M, N], residual in (unused / aliased when HAS_RESIDUAL=False)
    w_ptr,          # bf16 or fp32 [N], rmsnorm weight (loaded as fp32)
    new_res_ptr,    # bf16 [M, N], residual out (= x + res)  (unused when HAS_RESIDUAL=False)
    y_ptr,          # fp8  [M, N], quantized output
    s_ptr,          # fp32 [M, N // GROUP], one scale per 128-element block
    bf16_out_ptr,   # bf16 [M, N], normed bf16 output (unused when WRITE_BF16_NORMED=False)
    eps,
    fp8_max,
    M_,
    N_: tl.constexpr,
    GROUP_: tl.constexpr,
    N_GROUPS_: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    WRITE_BF16_NORMED: tl.constexpr,
    MATCH_BF16_PRODUCTION: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M_:
        return

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N_
    row_off = pid * N_

    # 1. load (and optionally fold in residual)
    x = tl.load(x_ptr + row_off + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        r = tl.load(res_ptr + row_off + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        x = x + r
        tl.store(new_res_ptr + row_off + offs_n, x.to(tl.bfloat16), mask=mask_n)

    # 2. rmsnorm over full row (rms_inv folds in 1/sqrt(N))
    mean_sq = tl.sum(x * x, axis=0) / N_
    rms_inv = tl.rsqrt(mean_sq + eps)
    w = tl.load(w_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    normed = x * rms_inv * w  # full row, fp32, register-resident

    # 2b. (optional) emit bf16 normed-out for non-fp8 consumers (indexer/compressor).
    # Stored *before* the per-block fp8 quant — both downstream paths see the
    # exact same fp32 normed value (truncated to bf16 here, scaled+clamped to
    # fp8 below). Single launch, single fp32 reduction.
    if WRITE_BF16_NORMED:
        tl.store(bf16_out_ptr + row_off + offs_n, normed.to(tl.bfloat16), mask=mask_n)

    # 2c. F4 ROOT-CAUSE FIX FOR A2-#1: round-trip normed through bf16.
    #
    # The UNFUSED production path is two launches:
    #   - rmsnorm writes bf16 normed-out to HBM  (this truncates fp32 → bf16)
    #   - per-1x128 quant reads bf16 from HBM, casts to fp32, quantizes
    # so the fp8 codepoints are produced from a BF16-TRUNCATED normed value.
    #
    # The fused kernel keeps `normed` register-resident in fp32 — which is
    # MORE accurate but produces different fp8 codepoints. Across 43 layers
    # this drift was the suspected root cause of A2-#1's +22 ms TPOT
    # regression / garbage tokens (per feedback_microbench_vs_live_wiring.md).
    #
    # When MATCH_BF16_PRODUCTION=True, we materialize bf16 in registers and
    # cast back to fp32 BEFORE the per-block quant, making the fused fp8
    # output bit-equivalent to the unfused production result. Default ON.
    if MATCH_BF16_PRODUCTION:
        normed = normed.to(tl.bfloat16).to(tl.float32)

    # 3. per-128-block fp8 quant.
    # Reshape register-resident normed into [N_GROUPS, GROUP] for amax/group.
    normed_2d = tl.reshape(normed, (N_GROUPS_, GROUP_))
    amax = tl.max(tl.abs(normed_2d), axis=1)              # [N_GROUPS]
    amax = tl.where(amax > 0, amax, 1.0)                  # avoid div-by-0
    scale = amax / fp8_max                                # [N_GROUPS]
    inv_scale = 1.0 / scale

    # Store scales (row-major: [M, N_GROUPS])
    g_offs = tl.arange(0, N_GROUPS_)
    g_mask = g_offs < N_GROUPS_
    tl.store(s_ptr + pid * N_GROUPS_ + g_offs, scale, mask=g_mask)

    # Quantize and store fp8 row
    q = normed_2d * inv_scale[:, None]
    q = tl.minimum(tl.maximum(q, -fp8_max), fp8_max)
    q = tl.reshape(q, (BLOCK_N,))
    tl.store(y_ptr + row_off + offs_n, q.to(y_ptr.dtype.element_ty), mask=mask_n)


def _fp8_max_for(dtype: torch.dtype) -> float:
    return float(torch.finfo(dtype).max)


def fused_add_rmsnorm_per1x128_quant(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    out_fp8: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    out_residual: Optional[torch.Tensor] = None,
    match_bf16_production: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused (add + rmsnorm + per-1x128 fp8 quant) in one Triton launch.

    Args:
        x:        [..., N] bf16, pre-add input
        residual: [..., N] bf16, residual in
        weight:   [N] bf16, rmsnorm weight
        eps:      float
        fp8_dtype: target fp8 dtype (default: torch.float8_e4m3fn / gfx950)
        out_fp8 / out_scale / out_residual: optional pre-allocated buffers

    Returns:
        (q_input, x_scale, new_residual)
            q_input:      [M, N] fp8_dtype
            x_scale:      [M, N // 128] fp32 (one scale per 128-element block)
            new_residual: [..., N] bf16 (= x + residual)
    """
    assert x.dtype == torch.bfloat16, f"x dtype: {x.dtype}"
    assert residual.dtype == torch.bfloat16, f"residual dtype: {residual.dtype}"
    # weight: bf16 or fp32; kernel re-casts via .to(tl.float32) on load.
    assert weight.dtype in (torch.bfloat16, torch.float32), f"weight dtype: {weight.dtype}"
    assert x.shape == residual.shape
    assert x.shape[-1] == weight.shape[-1]

    orig_shape = x.shape
    N_ = orig_shape[-1]
    assert N_ % _GROUP == 0, f"N={N_} not divisible by group {_GROUP}"
    x2 = x.reshape(-1, N_)
    r2 = residual.reshape(-1, N_)
    M_ = x2.shape[0]
    N_GROUPS_ = N_ // _GROUP
    BLOCK_N = triton.next_power_of_2(N_)

    if out_fp8 is None:
        out_fp8 = torch.empty((M_, N_), dtype=fp8_dtype, device=x.device)
    if out_scale is None:
        out_scale = torch.empty((M_, N_GROUPS_), dtype=torch.float32, device=x.device)
    if out_residual is None:
        out_residual = torch.empty_like(x)
    out_r2 = out_residual.reshape(-1, N_)

    fp8_max = _fp8_max_for(fp8_dtype)
    num_warps = 4 if BLOCK_N <= 4096 else 8

    _fused_add_rmsnorm_per1x128_quant_kernel[(M_,)](
        x2, r2, weight, out_r2, out_fp8, out_scale, x2,  # bf16_out unused
        eps, fp8_max,
        M_, N_, _GROUP, N_GROUPS_, BLOCK_N,
        HAS_RESIDUAL=True,
        WRITE_BF16_NORMED=False,
        MATCH_BF16_PRODUCTION=match_bf16_production,
        num_warps=num_warps,
    )
    return out_fp8, out_scale, out_residual


def fused_rmsnorm_per1x128_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    out_fp8: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    match_bf16_production: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused (rmsnorm + per-1x128 fp8 quant), no residual add. For q_norm /
    kv_norm callsites where the input is already the rmsnorm-input, no residual.
    Returns (q_input, x_scale)."""
    assert x.dtype == torch.bfloat16
    assert weight.dtype in (torch.bfloat16, torch.float32)

    orig_shape = x.shape
    N_ = orig_shape[-1]
    assert N_ % _GROUP == 0, f"N={N_} not divisible by group {_GROUP}"
    x2 = x.reshape(-1, N_)
    M_ = x2.shape[0]
    N_GROUPS_ = N_ // _GROUP
    BLOCK_N = triton.next_power_of_2(N_)

    if out_fp8 is None:
        out_fp8 = torch.empty((M_, N_), dtype=fp8_dtype, device=x.device)
    if out_scale is None:
        out_scale = torch.empty((M_, N_GROUPS_), dtype=torch.float32, device=x.device)

    fp8_max = _fp8_max_for(fp8_dtype)
    num_warps = 4 if BLOCK_N <= 4096 else 8

    # Kernel ignores res_ptr / new_res_ptr / bf16_out_ptr when their flags are
    # False, but we still pass valid tensors (Triton will not deref them).
    _fused_add_rmsnorm_per1x128_quant_kernel[(M_,)](
        x2, x2, weight, x2, out_fp8, out_scale, x2,  # res / new_res / bf16_out unused
        eps, fp8_max,
        M_, N_, _GROUP, N_GROUPS_, BLOCK_N,
        HAS_RESIDUAL=False,
        WRITE_BF16_NORMED=False,
        MATCH_BF16_PRODUCTION=match_bf16_production,
        num_warps=num_warps,
    )
    return out_fp8, out_scale


def fused_rmsnorm_per1x128_quant_dual(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    out_fp8: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    out_bf16: Optional[torch.Tensor] = None,
    match_bf16_production: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused (rmsnorm + per-1x128 fp8 quant) producing BOTH fp8 and bf16
    normed outputs in a single launch.

    Designed for the input_layernorm fan-out site in DSv4 Flash-Base, where
    the same normed value is consumed by FP8 GEMMs (wq_a / wkv) AND by bf16
    consumers (indexer / compressor). Without this fusion the bf16 output
    requires either a separate rmsnorm or a dequantize.

    Returns (q_input, x_scale, x_bf16):
        q_input:  [M, N] fp8_dtype  (rmsnorm + per-1x128 quant)
        x_scale:  [M, N // 128] fp32 (one scale per 128-element block)
        x_bf16:   [M, N] bf16  (rmsnorm output truncated to bf16)
    """
    assert x.dtype == torch.bfloat16
    assert weight.dtype in (torch.bfloat16, torch.float32)

    orig_shape = x.shape
    N_ = orig_shape[-1]
    assert N_ % _GROUP == 0, f"N={N_} not divisible by group {_GROUP}"
    x2 = x.reshape(-1, N_)
    M_ = x2.shape[0]
    N_GROUPS_ = N_ // _GROUP
    BLOCK_N = triton.next_power_of_2(N_)

    if out_fp8 is None:
        out_fp8 = torch.empty((M_, N_), dtype=fp8_dtype, device=x.device)
    if out_scale is None:
        out_scale = torch.empty((M_, N_GROUPS_), dtype=torch.float32, device=x.device)
    if out_bf16 is None:
        out_bf16 = torch.empty_like(x)
    out_bf16_2 = out_bf16.reshape(-1, N_)

    fp8_max = _fp8_max_for(fp8_dtype)
    num_warps = 4 if BLOCK_N <= 4096 else 8

    _fused_add_rmsnorm_per1x128_quant_kernel[(M_,)](
        x2, x2, weight, x2, out_fp8, out_scale, out_bf16_2,
        eps, fp8_max,
        M_, N_, _GROUP, N_GROUPS_, BLOCK_N,
        HAS_RESIDUAL=False,
        WRITE_BF16_NORMED=True,
        MATCH_BF16_PRODUCTION=match_bf16_production,
        num_warps=num_warps,
    )
    return out_fp8, out_scale, out_bf16


def fused_add_rmsnorm_per1x128_quant_dual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    out_fp8: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    out_bf16: Optional[torch.Tensor] = None,
    out_residual: Optional[torch.Tensor] = None,
    match_bf16_production: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add+rmsnorm+per-1x128 fp8 quant producing fp8, scale, bf16-normed AND
    new-residual in one launch.

    NOTE the four outputs are distinct tensors:
      - new_residual = (x + residual)                    (bf16, pre-norm sum)
      - x_bf16       = rmsnorm(new_residual) * weight    (bf16, post-norm)
      - q_input      = quantize_per_1x128(x_bf16)        (fp8)
      - x_scale      = the per-1x128 scales              (fp32)

    This is the right primitive for any post-residual fan-out where one
    consumer needs the new residual (next-layer add), one needs fp8 (next
    GEMM), and one needs bf16 (auxiliary path).

    Returns (q_input, x_scale, x_bf16, new_residual).
    """
    assert x.dtype == torch.bfloat16, f"x dtype: {x.dtype}"
    assert residual.dtype == torch.bfloat16, f"residual dtype: {residual.dtype}"
    assert weight.dtype in (torch.bfloat16, torch.float32)
    assert x.shape == residual.shape
    assert x.shape[-1] == weight.shape[-1]

    orig_shape = x.shape
    N_ = orig_shape[-1]
    assert N_ % _GROUP == 0, f"N={N_} not divisible by group {_GROUP}"
    x2 = x.reshape(-1, N_)
    r2 = residual.reshape(-1, N_)
    M_ = x2.shape[0]
    N_GROUPS_ = N_ // _GROUP
    BLOCK_N = triton.next_power_of_2(N_)

    if out_fp8 is None:
        out_fp8 = torch.empty((M_, N_), dtype=fp8_dtype, device=x.device)
    if out_scale is None:
        out_scale = torch.empty((M_, N_GROUPS_), dtype=torch.float32, device=x.device)
    if out_bf16 is None:
        out_bf16 = torch.empty_like(x)
    if out_residual is None:
        out_residual = torch.empty_like(x)
    out_bf16_2 = out_bf16.reshape(-1, N_)
    out_r2 = out_residual.reshape(-1, N_)

    fp8_max = _fp8_max_for(fp8_dtype)
    num_warps = 4 if BLOCK_N <= 4096 else 8

    _fused_add_rmsnorm_per1x128_quant_kernel[(M_,)](
        x2, r2, weight, out_r2, out_fp8, out_scale, out_bf16_2,
        eps, fp8_max,
        M_, N_, _GROUP, N_GROUPS_, BLOCK_N,
        HAS_RESIDUAL=True,
        WRITE_BF16_NORMED=True,
        MATCH_BF16_PRODUCTION=match_bf16_production,
        num_warps=num_warps,
    )
    return out_fp8, out_scale, out_bf16, out_residual
