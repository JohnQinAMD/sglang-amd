import enum
from typing import Tuple

import torch

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

# from sglang.srt.utils import is_hip
# FP8_DTYPE = torch.float8_e4m3fnuz if is_hip() else torch.float8_e4m3fn
FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
# Representable max of the FP8 variant in use:
#   e4m3fn   (NV, MI355): 448
#   e4m3fnuz (MI300):     240
# Originally hard-coded 448.0 in quantize_k_cache's scale computation, which
# caused values in (240, 448] to round-trip to FNUZ NaN (byte 0x80) on MI300
# and forced every consumer to call `nan_to_num` downstream. Using the actual
# dtype max fixes this at the source. (R5 P1-b)
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


class FP8KVCacheLayout(enum.Enum):
    V32_FP8Sparse = 1
    MODEL1_FP8Sparse = 2

    def get_meta(self) -> Tuple[int, int, int, int, int]:
        # Return: (d, d_nope, d_rope, tile_size, num_tiles)
        return {
            FP8KVCacheLayout.V32_FP8Sparse: (576, 512, 64, 128, 4),
            FP8KVCacheLayout.MODEL1_FP8Sparse: (512, 448, 64, 64, 7),
        }[self]


def _cast_scale_inv_to_ue8m0(
    scales_inv: torch.Tensor, out_dtype=torch.float32
) -> torch.Tensor:
    return torch.pow(2, torch.clamp_min(scales_inv, 1e-4).log2().ceil()).to(out_dtype)


def quantize_k_cache(
    input_k_cache: torch.Tensor,  # (num_blocks, block_size, h_k, d)
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    """
    Quantize the k-cache
    For more detail about the layout of K/V, please refer to comments in flash_mla_interface.py
    """
    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    assert input_k_cache.shape[-1] == d
    num_blocks, block_size, h_k, _ = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)  # [num_blocks, block_size, d]
    input_elem_size = input_k_cache.element_size()

    if kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:
        bytes_per_token = d_nope + num_tiles * 4 + input_elem_size * d_rope
        result = torch.empty(
            (num_blocks, block_size + 1, bytes_per_token),
            dtype=FP8_DTYPE,
            device=input_k_cache.device,
        )[:, :block_size, :]
        result_k_nope_part = result[..., :d_nope]
        result_k_scale_factor = result[..., d_nope : d_nope + num_tiles * 4].view(
            torch.float32
        )
        result_k_rope_part = result[..., d_nope + num_tiles * 4 :].view(
            input_k_cache.dtype
        )
        result_k_rope_part[:] = input_k_cache[..., d_nope:]

        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = (
                torch.abs(
                    input_k_cache[
                        ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                    ]
                )
                .max(dim=-1)
                .values.float()
                / FP8_MAX
            )  # [num_blocks, block_size]
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv

            cur_scale_factors_inv.unsqueeze_(-1)  # [num_blocks, block_size, 1]
            cur_quantized_nope = (
                input_k_cache[
                    ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                ].float()
                / cur_scale_factors_inv.float()
            ).to(FP8_DTYPE)
            result_k_nope_part[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ] = cur_quantized_nope

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    elif kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
        bytes_per_token = d_nope + 2 * d_rope + num_tiles + 1
        size_per_block_padded = (block_size * bytes_per_token + 576 - 1) // 576 * 576
        result = torch.empty(
            (num_blocks, size_per_block_padded),
            dtype=FP8_DTYPE,
            device=input_k_cache.device,
        )[:, : block_size * bytes_per_token]
        result_k_nope_rope_part = result[:, : block_size * (d_nope + 2 * d_rope)].view(
            num_blocks, block_size, d_nope + 2 * d_rope
        )
        result_k_nope = result_k_nope_rope_part[
            :, :, :d_nope
        ]  # [num_blocks, block_size, d_nope]
        result_k_rope = result_k_nope_rope_part[:, :, d_nope:].view(
            input_k_cache.dtype
        )  # [num_blocks, block_size, d_rope]
        result_k_scale_factor = (
            result[:, block_size * (d_nope + 2 * d_rope) :]
            .view(num_blocks, block_size, 8)[:, :, :7]
            .view(torch.float8_e8m0fnu)
        )  # [num_blocks, block_size, num_tiles]

        result_k_rope[:] = input_k_cache[..., d_nope:]
        for tile_idx in range(0, num_tiles):
            cur_scale_factors_inv = (
                torch.abs(
                    input_k_cache[
                        ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                    ]
                )
                .max(dim=-1)
                .values.float()
                / FP8_MAX
            )  # [num_blocks, block_size]
            cur_scale_factors_inv = _cast_scale_inv_to_ue8m0(cur_scale_factors_inv)
            result_k_scale_factor[:, :, tile_idx] = cur_scale_factors_inv.to(
                torch.float8_e8m0fnu
            )

            cur_scale_factors_inv = cur_scale_factors_inv.view(
                num_blocks, block_size, 1
            )
            cur_quantized_nope = (
                input_k_cache[
                    ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
                ].float()
                / cur_scale_factors_inv.float()
            ).to(FP8_DTYPE)
            result_k_nope[:, :, tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_quantized_nope
            )

        result = result.view(num_blocks, block_size, 1, -1)
        return result

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")


def dequantize_k_cache(
    quant_k_cache: torch.Tensor,  # (num_blocks, block_size, 1, bytes_per_token)
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    """
    De-quantize the k-cache
    """
    # NOTE ADD
    assert quant_k_cache.dtype == FP8_DTYPE

    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1
    result = torch.empty(
        (num_blocks, block_size, d), dtype=torch.bfloat16, device=quant_k_cache.device
    )

    if kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, block_size, -1)

        input_nope = quant_k_cache[..., :d_nope]
        input_scale = quant_k_cache[..., d_nope : d_nope + num_tiles * 4].view(
            torch.float32
        )
        input_rope = quant_k_cache[..., d_nope + num_tiles * 4 :].view(torch.bfloat16)
        result[..., d_nope:] = input_rope

        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ].to(torch.float32)
            cur_scales = input_scale[..., tile_idx].unsqueeze(-1)
            result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_nope * cur_scales
            )

    elif kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
        quant_k_cache = quant_k_cache.view(num_blocks, -1)  # [num_blocks, ...]
        input_nope_rope = quant_k_cache[:, : block_size * (d_nope + 2 * d_rope)].view(
            num_blocks, block_size, d_nope + 2 * d_rope
        )
        input_nope = input_nope_rope[:, :, :d_nope]
        input_rope = input_nope_rope[:, :, d_nope:].view(torch.bfloat16)
        input_scale = (
            quant_k_cache[:, block_size * (d_nope + 2 * d_rope) :]
            .view(num_blocks, block_size, 8)[:, :, :7]
            .view(torch.float8_e8m0fnu)
        )  # [num_blocks, block_size, num_tiles]

        result[..., d_nope:] = input_rope
        for tile_idx in range(0, num_tiles):
            cur_nope = input_nope[
                ..., tile_idx * tile_size : (tile_idx + 1) * tile_size
            ].to(torch.bfloat16)
            cur_scales = input_scale[:, :, tile_idx].to(torch.bfloat16).unsqueeze(-1)
            result[..., tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_nope * cur_scales
            )

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")

    result = result.view(num_blocks, block_size, 1, d)
    return result


import os

_DEQUANT_TRITON = os.environ.get("SGLANG_GATHER_FIRST_TRITON", "1") == "1"


def _try_import_triton_dequant():
    """Lazy import the triton kernels; returns (kernels_dict, available).

    Two kernels are registered:

    * ``nope_only``: post-gather version (takes pre-gathered [G, d_nope] nope
      + [G, num_tiles] scale buffers). This is what P1-a shipped.

    * ``fused_gather_model1`` (P1-c): takes the raw packed KV buffer and the
      flat indices tensor directly and does gather + dequant + rope copy in
      one kernel — eliminates the two Python ``advanced indexing`` ops that
      P0-c/P1-a still had to run before the kernel.
    """
    try:
        import triton
        import triton.language as tl

        if not hasattr(tl, "float8e4b8"):
            return {}, False

        @triton.jit
        def _dequant_nope_kernel(
            nope_ptr, scale_ptr, out_ptr,
            g_n: tl.int32,
            out_stride_0: tl.int64,
            D_NOPE: tl.constexpr,
            NUM_TILES: tl.constexpr,
            TILE_SIZE: tl.constexpr,
            BLOCK_G: tl.constexpr,
        ):
            pid = tl.program_id(0)
            rows = pid * BLOCK_G + tl.arange(0, BLOCK_G)
            rmask = rows < g_n

            for tile_idx in tl.static_range(NUM_TILES):
                s_u8 = tl.load(scale_ptr + rows * NUM_TILES + tile_idx, mask=rmask, other=0)
                s_bf = tl.exp2(s_u8.to(tl.float32) - 127.0).to(tl.bfloat16)

                tile_cols = tile_idx * TILE_SIZE + tl.arange(0, TILE_SIZE)
                nope_fp8 = tl.load(
                    nope_ptr + rows[:, None] * D_NOPE + tile_cols[None, :],
                    mask=rmask[:, None],
                )
                tl.store(
                    out_ptr + rows[:, None] * out_stride_0 + tile_cols[None, :],
                    nope_fp8.to(tl.bfloat16) * s_bf[:, None],
                    mask=rmask[:, None],
                )

        @triton.jit
        def _fused_gather_dequant_model1_kernel(
            packed_u8_ptr,    # uint8* — packed KV, strided layout
            indices_ptr,      # int64*  [G]   flat per-row index into [num_blocks*block_size)
            out_ptr,          # bf16*   [G, D_OUT] (D_OUT == D_NOPE + D_ROPE == d)
            g_n: tl.int32,
            size_per_block: tl.int64,       # bytes per block (already -padded- to pow of 576)
            out_stride_0: tl.int64,          # bf16 elems between rows
            BLOCK_SIZE: tl.constexpr,        # KV block size (#tokens per block)
            D_NOPE: tl.constexpr,
            D_ROPE: tl.constexpr,
            NUM_TILES: tl.constexpr,
            TILE_SIZE: tl.constexpr,
            BLOCK_G: tl.constexpr,
        ):
            """One kernel launch gathers + dequants + copies rope into the bf16 output.

            MODEL1_FP8Sparse layout per block:
              0 .. block_size*(D_NOPE + 2*D_ROPE) : nope|rope interleaved per token
              ...                                 : padding, scales in the tail:
              [block_size*(D_NOPE + 2*D_ROPE), ...) : fp8_e8m0 scales, 8 bytes/token
            """
            pid = tl.program_id(0)
            rows = pid * BLOCK_G + tl.arange(0, BLOCK_G)
            rmask = rows < g_n

            idx = tl.load(indices_ptr + rows, mask=rmask, other=0)
            idx = tl.maximum(idx, 0)
            block_idx = (idx // BLOCK_SIZE).to(tl.int64)
            pos_idx   = (idx %  BLOCK_SIZE).to(tl.int64)

            nr_stride  = D_NOPE + 2 * D_ROPE
            row_base   = block_idx * size_per_block                          # start of block
            nope_base  = row_base + pos_idx * nr_stride                      # start of this token's nope
            rope_base  = nope_base + D_NOPE                                  # start of this token's rope
            scale_base = row_base + BLOCK_SIZE * nr_stride + pos_idx * 8     # start of this token's 8-byte scale group

            # --- nope: dequant tile-by-tile ---
            for tile_idx in tl.static_range(NUM_TILES):
                s_u8 = tl.load(packed_u8_ptr + scale_base + tile_idx, mask=rmask, other=0)
                s_bf = tl.exp2(s_u8.to(tl.float32) - 127.0).to(tl.bfloat16)

                tile_cols = tile_idx * TILE_SIZE + tl.arange(0, TILE_SIZE)
                # Read raw fp8 bytes from packed, reinterpret as fp8e4b8, cast to bf16.
                nope_u8 = tl.load(
                    packed_u8_ptr + nope_base[:, None] + tile_cols[None, :],
                    mask=rmask[:, None],
                )
                nope_fp8 = nope_u8.to(tl.float8e4b8, bitcast=True)
                tl.store(
                    out_ptr + rows[:, None] * out_stride_0 + tile_cols[None, :],
                    nope_fp8.to(tl.bfloat16) * s_bf[:, None],
                    mask=rmask[:, None],
                )

            # --- rope: bf16 bytes → direct copy via u8+u8 -> u16 bitcast to bf16 ---
            rope_i = tl.arange(0, D_ROPE)
            lo = tl.load(packed_u8_ptr + rope_base[:, None] + (2 * rope_i)[None, :],     mask=rmask[:, None])
            hi = tl.load(packed_u8_ptr + rope_base[:, None] + (2 * rope_i + 1)[None, :], mask=rmask[:, None])
            u16 = lo.to(tl.uint16) | (hi.to(tl.uint16) << 8)
            rope_bf = u16.to(tl.bfloat16, bitcast=True)
            tl.store(
                out_ptr + rows[:, None] * out_stride_0 + (D_NOPE + rope_i)[None, :],
                rope_bf,
                mask=rmask[:, None],
            )

        # ------------------------------------------------------------------
        # Phase G fold: explicit bit-shift + mask version of the gather kernel.
        # When BLOCK_SIZE is a power of 2 (DSv4 production: 256 = 2^8), the
        # `idx // BLOCK_SIZE` and `idx % BLOCK_SIZE` ops are reducible to a
        # right-shift and a bit-mask. Triton's compiler usually does this on
        # constexpr power-of-2 divisors, but encoding it explicitly removes
        # ambiguity and lets us also drop the redundant Python-side preamble
        # (clamp_min + // + %) at the wrapper. See `_fused_gather_dequant_
        # model1_triton` and `dequantize_k_cache_gather` in this file for the
        # SGLANG_PHASE_G_FOLD wire-in.
        #
        # Behavior contract: identical to the non-fold kernel. Negative input
        # indices are clamped to 0 (same as the old wrapper's clamp_min) and
        # produce row-0 data in the output — the downstream consumer masks
        # these rows out via `invalid_mask` at the attn step, so the kernel
        # output for invalid rows is "don't care" garbage. The kernel is
        # bit-exact with the non-fold kernel + clamp+div+rem preamble for
        # ALL inputs (negative or otherwise).
        # ------------------------------------------------------------------
        @triton.jit
        def _fused_gather_dequant_model1_kernel_g_fold(
            packed_u8_ptr,    # uint8*  packed KV, strided layout
            indices_ptr,      # int*    [G]  flat per-row index, possibly-negative
            out_ptr,          # bf16*   [G, D_OUT]
            g_n: tl.int32,
            size_per_block: tl.int64,
            out_stride_0: tl.int64,
            BLOCK_SIZE: tl.constexpr,        # MUST be a power of 2
            BLOCK_SIZE_LOG2: tl.constexpr,   # log2(BLOCK_SIZE)
            D_NOPE: tl.constexpr,
            D_ROPE: tl.constexpr,
            NUM_TILES: tl.constexpr,
            TILE_SIZE: tl.constexpr,
            BLOCK_G: tl.constexpr,
        ):
            """G-fold variant: explicit `>>` and `&` for index decomposition.

            Equivalent to:
              idx = max(idx_raw, 0)
              block_idx = idx >> BLOCK_SIZE_LOG2     # idx // BLOCK_SIZE
              pos_idx   = idx & (BLOCK_SIZE - 1)     # idx %  BLOCK_SIZE

            Allows the caller to skip the Python `torch.clamp_min + // + %`
            preamble (3 elementwise launches per call). Output bit-equality
            with the original kernel + Python preamble is preserved.
            """
            pid = tl.program_id(0)
            rows = pid * BLOCK_G + tl.arange(0, BLOCK_G)
            rmask = rows < g_n

            idx_raw = tl.load(indices_ptr + rows, mask=rmask, other=0)
            # In-register clamp + bit-shift decomposition. Operates on the
            # native dtype of indices_ptr; promotes to int64 for byte address
            # arithmetic to avoid 32-bit overflow on `block_idx * size_per_block`
            # (size_per_block can exceed 2^15 bytes, num_blocks can be 4096+).
            idx = tl.maximum(idx_raw, 0)
            block_idx = (idx >> BLOCK_SIZE_LOG2).to(tl.int64)
            pos_idx   = (idx &  (BLOCK_SIZE - 1)).to(tl.int64)

            nr_stride  = D_NOPE + 2 * D_ROPE
            row_base   = block_idx * size_per_block
            nope_base  = row_base + pos_idx * nr_stride
            rope_base  = nope_base + D_NOPE
            scale_base = row_base + BLOCK_SIZE * nr_stride + pos_idx * 8

            # --- nope: dequant tile-by-tile ---
            for tile_idx in tl.static_range(NUM_TILES):
                s_u8 = tl.load(packed_u8_ptr + scale_base + tile_idx, mask=rmask, other=0)
                s_bf = tl.exp2(s_u8.to(tl.float32) - 127.0).to(tl.bfloat16)

                tile_cols = tile_idx * TILE_SIZE + tl.arange(0, TILE_SIZE)
                nope_u8 = tl.load(
                    packed_u8_ptr + nope_base[:, None] + tile_cols[None, :],
                    mask=rmask[:, None],
                )
                nope_fp8 = nope_u8.to(tl.float8e4b8, bitcast=True)
                tl.store(
                    out_ptr + rows[:, None] * out_stride_0 + tile_cols[None, :],
                    nope_fp8.to(tl.bfloat16) * s_bf[:, None],
                    mask=rmask[:, None],
                )

            # --- rope: bf16 bytes → direct copy via u8+u8 -> u16 bitcast to bf16 ---
            rope_i = tl.arange(0, D_ROPE)
            lo = tl.load(packed_u8_ptr + rope_base[:, None] + (2 * rope_i)[None, :],     mask=rmask[:, None])
            hi = tl.load(packed_u8_ptr + rope_base[:, None] + (2 * rope_i + 1)[None, :], mask=rmask[:, None])
            u16 = lo.to(tl.uint16) | (hi.to(tl.uint16) << 8)
            rope_bf = u16.to(tl.bfloat16, bitcast=True)
            tl.store(
                out_ptr + rows[:, None] * out_stride_0 + (D_NOPE + rope_i)[None, :],
                rope_bf,
                mask=rmask[:, None],
            )

        return (
            {"nope_only": _dequant_nope_kernel,
             "fused_model1": _fused_gather_dequant_model1_kernel,
             "fused_model1_g_fold": _fused_gather_dequant_model1_kernel_g_fold},
            True,
        )
    except Exception:
        return {}, False


_DEQUANT_KERNELS, _DEQUANT_KERNEL_OK = ({}, False)
if _DEQUANT_TRITON:
    _DEQUANT_KERNELS, _DEQUANT_KERNEL_OK = _try_import_triton_dequant()
_DEQUANT_KERNEL = _DEQUANT_KERNELS.get("nope_only")              # backward-compat alias used by P1-a code path
_FUSED_MODEL1_KERNEL = _DEQUANT_KERNELS.get("fused_model1")      # P1-c
# Phase G fold: explicit shift+mask kernel; only valid when block_size is pow2.
# Wrapper falls back to the non-fold kernel + Python preamble when the env knob
# is OFF or block_size is not a power of 2.
_FUSED_MODEL1_KERNEL_G_FOLD = _DEQUANT_KERNELS.get("fused_model1_g_fold")
_PHASE_G_FOLD = os.environ.get("SGLANG_PHASE_G_FOLD", "0") == "1"


def _ilog2_pow2(x: int) -> int:
    """Return log2(x) for positive power-of-two `x`, else -1."""
    if x <= 0 or (x & (x - 1)) != 0:
        return -1
    n = 0
    while x > 1:
        x >>= 1
        n += 1
    return n


def _dequant_nope_triton(
    gathered_nope_fp8: torch.Tensor,
    gathered_scale_u8: torch.Tensor,
    result: torch.Tensor,
    d_nope: int,
    num_tiles: int,
    tile_size: int,
) -> None:
    """P1-a path: post-gather kernel. Kept for the V32 layout + debug fallback."""
    G = gathered_nope_fp8.shape[0]
    BLOCK_G = 16
    grid = ((G + BLOCK_G - 1) // BLOCK_G,)
    if gathered_scale_u8.dtype != torch.uint8:
        gathered_scale_u8 = gathered_scale_u8.view(torch.uint8)
    _DEQUANT_KERNEL[grid](
        gathered_nope_fp8, gathered_scale_u8, result,
        G, result.stride(0),
        D_NOPE=d_nope, NUM_TILES=num_tiles, TILE_SIZE=tile_size,
        BLOCK_G=BLOCK_G, num_warps=4, num_stages=2,
    )


def _fused_gather_dequant_model1_triton(
    packed_k_cache: torch.Tensor,  # [num_blocks, block_size, 1, bytes_per_token] FP8_DTYPE
    indices_flat: torch.Tensor,    # [G] int32/int64, -1 ok (kernel clamps to 0)
    result: torch.Tensor,          # [G, d] bfloat16
    block_size: int,
    d_nope: int,
    d_rope: int,
    num_tiles: int,
    tile_size: int,
) -> None:
    """P1-c path: single kernel does the two Python advanced-index gathers
    (nope+rope region gather, scale region gather) AND the per-tile dequant
    AND the rope bf16 copy. No intermediate tensors.

    Phase G fold (`SGLANG_PHASE_G_FOLD=1`): when the env knob is set AND
    block_size is a power of 2 AND the g_fold kernel compiled, dispatches
    to `_fused_gather_dequant_model1_kernel_g_fold` which uses explicit
    `>>` and `&` for the index decomposition. The wrapper signature is
    UNCHANGED — callers see no difference. The fold removes the redundant
    `clamp_min + // + %` Python preamble that lives at the call-site
    (`dequantize_k_cache_gather`); see that function for the detail.
    """
    G = indices_flat.numel()
    if G == 0:
        return
    BLOCK_G = 16
    grid = ((G + BLOCK_G - 1) // BLOCK_G,)
    # Flatten to uint8 so kernel does byte-level pointer arithmetic.
    packed_u8 = packed_k_cache.view(torch.uint8).view(packed_k_cache.shape[0], -1)
    size_per_block = packed_u8.stride(0)   # bytes between blocks (includes padding)
    # int64 indices for safer arithmetic in kernel
    if indices_flat.dtype != torch.int64:
        indices_flat = indices_flat.to(torch.int64)

    # Phase G fold dispatch: only fires when knob is on AND block_size is pow2
    # (DSv4 production: block_size=256 = 2^8). Falls back to the original
    # kernel otherwise so behavior is identical for non-pow2 block sizes.
    log2_bs = _ilog2_pow2(block_size)
    use_g_fold = (
        _PHASE_G_FOLD
        and _FUSED_MODEL1_KERNEL_G_FOLD is not None
        and log2_bs >= 0
    )
    if use_g_fold:
        _FUSED_MODEL1_KERNEL_G_FOLD[grid](
            packed_u8,
            indices_flat,
            result,
            G,
            size_per_block,
            result.stride(0),
            BLOCK_SIZE=block_size,
            BLOCK_SIZE_LOG2=log2_bs,
            D_NOPE=d_nope,
            D_ROPE=d_rope,
            NUM_TILES=num_tiles,
            TILE_SIZE=tile_size,
            BLOCK_G=BLOCK_G,
            num_warps=4,
            num_stages=2,
        )
        return

    _FUSED_MODEL1_KERNEL[grid](
        packed_u8,
        indices_flat,
        result,
        G,
        size_per_block,
        result.stride(0),
        BLOCK_SIZE=block_size,
        D_NOPE=d_nope,
        D_ROPE=d_rope,
        NUM_TILES=num_tiles,
        TILE_SIZE=tile_size,
        BLOCK_G=BLOCK_G,
        num_warps=4,
        num_stages=2,
    )


def dequantize_k_cache_gather(
    quant_k_cache: torch.Tensor,  # (num_blocks, block_size, 1, bytes_per_token), dtype=FP8_DTYPE
    indices_flat: torch.Tensor,   # [G] int32/int64, values in [0, num_blocks*block_size); -1 = invalid
    kvcache_layout: FP8KVCacheLayout,
) -> torch.Tensor:
    """
    Gather-first de-quantization: only dequant the ``G`` rows we actually need
    instead of the whole [num_blocks * block_size] table.

    This is the per-step hot-path optimisation that replaces the sparse MLA
    torch fallback's "dequantize_k_cache(whole table) then index_select(topk)"
    with "index_select(topk) then dequantize(topk)". Shrinks HBM bandwidth by
    ``num_blocks*block_size / G`` (typically 30-100x on DS-V4).

    Returns:
        gathered: [G, d] bfloat16, layout matches the ``[nope | rope]`` order
        of the whole-table dequantize_k_cache result, so callers can treat it
        the same as a pre-gathered row from that function.

    Invalid rows (indices_flat == -1) are clamped to row 0 and will be masked
    out by the caller via the usual invalid_mask (same convention as the torch
    fallback, which also clamp_min(indices, 0) before index_select).
    """
    assert quant_k_cache.dtype == FP8_DTYPE, quant_k_cache.dtype

    d, d_nope, d_rope, tile_size, num_tiles = kvcache_layout.get_meta()
    num_blocks, block_size, h_k, bytes_per_token = quant_k_cache.shape
    assert h_k == 1
    G = indices_flat.numel()
    device = quant_k_cache.device

    # Phase G fold: when the env knob is set, the MODEL1 fused kernel does
    # the clamp+div+rem in-register so the Python preamble below is dead
    # work for that fast path. We skip it when we know we'll take the fused
    # fast path; the V32 path and BF16 fallback still need block_idx /
    # pos_idx tensors so they recompute lazily.
    _g_fold_active = (
        _PHASE_G_FOLD
        and kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse
        and _FUSED_MODEL1_KERNEL_G_FOLD is not None
        and _ilog2_pow2(block_size) >= 0
        and G > 0
        and os.environ.get("SGLANG_DISABLE_FUSED_MODEL1_KERNEL", "0") != "1"
    )
    if _g_fold_active:
        # Skip the Python clamp_min + // + % preamble — kernel does it in-register.
        idx_flat = None
        block_idx = None
        pos_idx = None
    else:
        idx_flat = torch.clamp_min(indices_flat, 0).to(torch.int64)
        block_idx = idx_flat // block_size
        pos_idx = idx_flat % block_size

    if kvcache_layout == FP8KVCacheLayout.MODEL1_FP8Sparse:
        # P1-c fast path: ONE triton kernel handles gather + dequant + rope
        # copy, with no Python advanced-indexing allocations. Requires the
        # `fused_model1` triton kernel to have compiled successfully at import
        # time. Opt out by setting SGLANG_DISABLE_FUSED_MODEL1_KERNEL=1.
        _use_fused = (
            _FUSED_MODEL1_KERNEL is not None
            and G > 0
            and os.environ.get("SGLANG_DISABLE_FUSED_MODEL1_KERNEL", "0") != "1"
        )
        if _use_fused:
            result = torch.empty((G, d), dtype=torch.bfloat16, device=device)
            _fused_gather_dequant_model1_triton(
                quant_k_cache,
                indices_flat,
                result,
                block_size=block_size,
                d_nope=d_nope,
                d_rope=d_rope,
                num_tiles=num_tiles,
                tile_size=tile_size,
            )
            return result
        # Non-fused path needs the (block_idx, pos_idx) tensors that the
        # g_fold preamble skipped. Compute lazily here.
        if block_idx is None:
            idx_flat = torch.clamp_min(indices_flat, 0).to(torch.int64)
            block_idx = idx_flat // block_size
            pos_idx = idx_flat % block_size

        # P1-a fallback: gather in Python, one kernel for dequant.
        # Layout per block:
        #   [block_size * (d_nope + 2*d_rope)] bytes : nope+rope interleaved per token
        #   + [block_size * 8] bytes                 : 7 fp8_e8m0 scales + 1 pad, per token
        flat = quant_k_cache.view(num_blocks, -1)

        nope_rope_region = flat[:, : block_size * (d_nope + 2 * d_rope)].view(
            num_blocks, block_size, d_nope + 2 * d_rope
        )
        gathered_nope_rope = nope_rope_region[block_idx, pos_idx, :]
        gathered_nope = gathered_nope_rope[:, :d_nope]
        gathered_rope_bf16 = (
            gathered_nope_rope[:, d_nope:].contiguous().view(torch.bfloat16)
        )

        scale_region = (
            flat[:, block_size * (d_nope + 2 * d_rope):]
            .view(num_blocks, block_size, 8)[:, :, :num_tiles]
        )
        gathered_scale_u8 = scale_region[block_idx, pos_idx, :].contiguous()

        result = torch.empty((G, d), dtype=torch.bfloat16, device=device)
        result[:, d_nope:] = gathered_rope_bf16

        if _DEQUANT_KERNEL_OK and G > 0:
            gathered_nope_contig = gathered_nope.contiguous()
            _dequant_nope_triton(
                gathered_nope_contig, gathered_scale_u8, result,
                d_nope=d_nope, num_tiles=num_tiles, tile_size=tile_size,
            )
        else:
            # Pure-Python final fallback.
            gathered_scale_fp8 = gathered_scale_u8.view(torch.float8_e8m0fnu)
            for tile_idx in range(num_tiles):
                cur_nope = gathered_nope[
                    :, tile_idx * tile_size : (tile_idx + 1) * tile_size
                ].to(torch.bfloat16)
                cur_scale = gathered_scale_fp8[:, tile_idx].to(torch.bfloat16).unsqueeze(-1)
                result[:, tile_idx * tile_size : (tile_idx + 1) * tile_size] = cur_nope * cur_scale
        return result

    elif kvcache_layout == FP8KVCacheLayout.V32_FP8Sparse:
        # Per-token interleaved: [nope | fp32_scales*num_tiles | rope_bf16].
        per_token = quant_k_cache.view(num_blocks, block_size, -1)
        gathered = per_token[block_idx, pos_idx, :]
        input_nope = gathered[:, :d_nope]
        input_scale = gathered[:, d_nope : d_nope + num_tiles * 4].view(torch.float32)
        input_rope = gathered[:, d_nope + num_tiles * 4:].contiguous().view(torch.bfloat16)

        result = torch.empty((G, d), dtype=torch.bfloat16, device=device)
        result[:, d_nope:] = input_rope
        for tile_idx in range(num_tiles):
            cur_nope = input_nope[
                :, tile_idx * tile_size : (tile_idx + 1) * tile_size
            ].to(torch.float32)
            cur_scales = input_scale[:, tile_idx].unsqueeze(-1)
            result[:, tile_idx * tile_size : (tile_idx + 1) * tile_size] = (
                cur_nope * cur_scales
            ).to(torch.bfloat16)
        return result

    else:
        raise NotImplementedError(f"Unsupported kvcache_layout: {kvcache_layout}")


def abs_indices2indices_in_kvcache(
    abs_indices: torch.Tensor,  # [b, s_q, topk]
    block_table: torch.Tensor,  # [b, /]
    block_size: int,
) -> torch.Tensor:
    """
    Convert abs_indices (logical index, ranging from 0 to s_k-1) to index expected by the sparse attn kernel
    Equivalent to:

    b, s_q, topk = abs_indices.shape
    indices_in_kvcache = torch.empty_like(abs_indices)
    for i in range(b):
        cur_abs_indices = abs_indices[i, :, :].clone()  # [s_q, topk]
        invalid_mask = cur_abs_indices == -1
        cur_abs_indices[invalid_mask] = 0
        cur_indices_in_kvcache = block_table[i].index_select(0, cur_abs_indices.flatten()//block_size).view(s_q, topk)*block_size + cur_abs_indices%block_size
        cur_indices_in_kvcache[invalid_mask] = -1
        indices_in_kvcache[i] = cur_indices_in_kvcache
    return indices_in_kvcache

    """
    b, s_q, topk = abs_indices.shape
    _, max_blocks_per_seq = block_table.shape

    abs_indices = abs_indices.clone()
    invalid_mask = abs_indices == -1
    abs_indices[invalid_mask] = 0

    real_block_idxs = block_table.view(-1).index_select(
        0,
        (
            abs_indices // block_size
            + torch.arange(0, b).view(b, 1, 1) * max_blocks_per_seq
        ).view(-1),
    )
    indices_in_kvcache = (
        real_block_idxs.view(b, s_q, topk) * block_size + abs_indices % block_size
    )
    indices_in_kvcache[invalid_mask] = -1

    return indices_in_kvcache
