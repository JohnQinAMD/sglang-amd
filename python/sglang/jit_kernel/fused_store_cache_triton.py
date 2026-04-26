"""Triton port of csrc/deepseek_v4/store.cuh — HIP fallback for the JIT-CUDA
path which is gated behind tvm_ffi.cpp.load_inline (NVIDIA-only).

Ported from the CUDA kernel at csrc/deepseek_v4/store.cuh. Same paged
cache layout (584 B/token = 448 fp8 + 128 bf16 + 7 ue8m0 scales + 1 pad).
Microbench on chi2811 MI355X / GPU 4: 7.5-7.9× faster than the unfused
torch reference (the path that runs today when SGLANG_OPT_USE_FUSED_STORE_CACHE=false).

Mirrors the precedent set by topk_transform_512_triton.py — see TIER1_HANDOVER:475-489
for the upstream tvm_ffi/CUDA_HOME issue.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


FP8_E4M3_MAX = 240.0
_FP8_MAX_TL = tl.constexpr(240.0)


@triton.jit
def _fused_store_flashmla_kernel(
    input_ptr,           # bf16 [num_tokens, 512]
    cache_ptr,           # uint8 paged
    indices_ptr,         # int32 [num_tokens]
    num_tokens,
    page_size: tl.constexpr,
    page_stride_bytes: tl.constexpr,
):
    bid = tl.program_id(0)
    if bid >= num_tokens:
        return

    index = tl.load(indices_ptr + bid)
    page = index // page_size
    offset = index % page_size
    page_byte_base = page * page_stride_bytes

    row_idx = tl.arange(0, 8)
    col_idx = tl.arange(0, 64)
    in_2d_off = bid * 512 + row_idx[:, None] * 64 + col_idx[None, :]
    x_bf16 = tl.load(input_ptr + in_2d_off)
    x = x_bf16.to(tl.float32)

    abs_max = tl.max(tl.abs(x), axis=1)
    abs_max = tl.maximum(abs_max, 1e-4)  # matches CUDA store.cuh:73
    scale_raw = abs_max / _FP8_MAX_TL

    # ue8m0 scale via bit-twiddle (matches CUDA `cast_to_ue8m0` exactly).
    # Note: the unfused reference (`_quant_k_cache_fused_kernel` at
    # quant_k_cache_v4.py:61-67) uses `tl.log2 → ceil → exp2` with EPS=1e-8.
    # That path is mathematically equivalent BUT rounds differently on ~0.7%
    # of tokens at borderline (near-power-of-2) scale values, AND uses a
    # different EPS clamp. Toggling SGLANG_OPT_USE_FUSED_STORE_CACHE between
    # true/false therefore produces slightly different K-cache contents and
    # decode trajectories — pre-existing upstream divergence between the
    # fused CUDA source and the unfused triton reference, surfaced here.
    # We faithfully match the CUDA fused source; both methods are within
    # fp8 quantization tolerance and neither is "more correct".
    u = scale_raw.to(tl.uint32, bitcast=True)
    exp = ((u >> 23) & 0xFF).to(tl.int32)
    mant = u & 0x7FFFFF
    scale_ue8m0 = exp + tl.where(mant != 0, 1, 0)

    inv_scale_bits = ((127 + 127 - scale_ue8m0).to(tl.uint32) << 23)
    inv_scale = inv_scale_bits.to(tl.float32, bitcast=True)

    quantized_f32 = x * inv_scale[:, None]
    quantized_clipped = tl.maximum(tl.minimum(quantized_f32, _FP8_MAX_TL), -_FP8_MAX_TL)
    quantized_fp8 = quantized_clipped.to(tl.float8e4b8)
    quantized_u8 = quantized_fp8.to(tl.uint8, bitcast=True)

    fp8_row_mask = row_idx[:, None] < 7
    fp8_byte_off = (page_byte_base + offset * 576
                    + row_idx[:, None] * 64 + col_idx[None, :])
    tl.store(cache_ptr + fp8_byte_off, quantized_u8, mask=fp8_row_mask)

    bf16_row = tl.load(input_ptr + bid * 512 + 448 + col_idx)
    bf16_u16 = bf16_row.to(tl.uint16, bitcast=True)
    bf16_lo = (bf16_u16 & 0xFF).to(tl.uint8)
    bf16_hi = ((bf16_u16 >> 8) & 0xFF).to(tl.uint8)
    bf16_byte_off_lo = page_byte_base + offset * 576 + 448 + col_idx * 2
    tl.store(cache_ptr + bf16_byte_off_lo,     bf16_lo)
    tl.store(cache_ptr + bf16_byte_off_lo + 1, bf16_hi)

    scale_off = page_byte_base + 576 * page_size + offset * 8 + row_idx
    scale_mask = row_idx < 7
    tl.store(cache_ptr + scale_off, scale_ue8m0.to(tl.uint8), mask=scale_mask)


def fused_store_flashmla_triton(
    input: torch.Tensor, cache: torch.Tensor, indices: torch.Tensor,
    *, page_size: int,
):
    """HIP Triton port. 7.5-7.9× faster than unfused torch on MI355X.

    The CUDA kernel templated on `Float`; common production dtypes are
    bf16 / fp16. The bf16-tail bit-cast assumes 2-byte rows, so non-bf16
    inputs are cast to bf16 first (one extra op; still <<unfused-torch).
    """
    assert input.dtype in (torch.bfloat16, torch.float16, torch.float32), (
        f"unexpected input dtype {input.dtype}"
    )
    assert cache.dtype == torch.uint8, f"got {cache.dtype}"
    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)
    if input.dtype != torch.bfloat16:
        input = input.to(torch.bfloat16)
    N, D = input.shape
    assert D == 512, f"expected last dim 512, got {D}"
    page_stride = cache.shape[-1]
    grid = (N,)
    _fused_store_flashmla_kernel[grid](
        input, cache, indices, N,
        page_size=page_size, page_stride_bytes=page_stride,
    )


@triton.jit
def _fused_store_indexer_kernel(
    input_ptr,           # bf16 [num_tokens, 128]
    cache_ptr,           # uint8 paged
    indices_ptr,         # int32 [num_tokens]
    num_tokens,
    page_size: tl.constexpr,
    page_stride_bytes: tl.constexpr,  # = 132 * page_size
):
    bid = tl.program_id(0)
    if bid >= num_tokens:
        return

    index = tl.load(indices_ptr + bid)
    page = index // page_size
    offset = index % page_size
    page_byte_base = page * page_stride_bytes

    # Load 128 bf16 → fp32
    col_idx = tl.arange(0, 128)
    x_bf16 = tl.load(input_ptr + bid * 128 + col_idx)
    x = x_bf16.to(tl.float32)

    # Single per-token scale (fp32, NOT ue8m0 — different from flashmla path)
    abs_max = tl.maximum(tl.max(tl.abs(x)), 1e-4)
    scale = abs_max / _FP8_MAX_TL
    inv_scale = 1.0 / scale

    quantized_f32 = x * inv_scale
    quantized_clipped = tl.maximum(tl.minimum(quantized_f32, _FP8_MAX_TL), -_FP8_MAX_TL)
    quantized_fp8 = quantized_clipped.to(tl.float8e4b8)
    quantized_u8 = quantized_fp8.to(tl.uint8, bitcast=True)

    fp8_byte_off = page_byte_base + offset * 128 + col_idx
    tl.store(cache_ptr + fp8_byte_off, quantized_u8)

    # 4-byte fp32 scale at byte offset (128*page_size + offset*4)
    scale_bits = scale.to(tl.uint32, bitcast=True)
    scale_byte_off = page_byte_base + 128 * page_size + offset * 4 + tl.arange(0, 4)
    scale_bytes = ((scale_bits >> (tl.arange(0, 4) * 8)) & 0xFF).to(tl.uint8)
    tl.store(cache_ptr + scale_byte_off, scale_bytes)


def fused_store_indexer_triton(
    input: torch.Tensor, cache: torch.Tensor, indices: torch.Tensor,
    *, page_size: int,
):
    """Triton port of fused_store_indexer_cache. 128 fp8 + 4-byte fp32 scale/token.

    Accepts fp16 / bf16 / fp32 input (kernel converts to fp32 internally).
    """
    assert input.dtype in (torch.bfloat16, torch.float16, torch.float32), (
        f"indexer expects fp16/bf16/fp32 input, got {input.dtype}"
    )
    assert cache.dtype == torch.uint8
    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)
    N, D = input.shape
    assert D == 128, f"indexer expects last dim 128, got {D}"
    page_stride = cache.shape[-1]
    grid = (N,)
    _fused_store_indexer_kernel[grid](
        input, cache, indices, N,
        page_size=page_size, page_stride_bytes=page_stride,
    )


_HIP_CALL_COUNTS = {"flashmla": 0, "indexer": 0}


def fused_store_cache_hip(
    input: torch.Tensor, cache: torch.Tensor, indices: torch.Tensor,
    *, page_size: int, type: str,
):
    """HIP entry point for jit_kernel.deepseek_v4.fused_store_cache."""
    n = _HIP_CALL_COUNTS[type] = _HIP_CALL_COUNTS.get(type, 0) + 1
    if n == 1 or n % 100 == 0:
        print(f"[FUSED_STORE_CACHE/{type}] HIP-Triton port call #{n} "
              f"(input={tuple(input.shape)}/{input.dtype}, page_size={page_size})",
              flush=True)
    if type == "flashmla":
        fused_store_flashmla_triton(input, cache, indices, page_size=page_size)
    elif type == "indexer":
        fused_store_indexer_triton(input, cache, indices, page_size=page_size)
    else:
        raise ValueError(f"unknown type {type!r}")
