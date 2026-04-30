"""M1 megakernel — fused KV-write-with-RoPE for DSv4 Flash-Base FP8 (gfx950 MI355X).

Fuses the kv-side post-norm chain that runs every attention layer in the SWA
KV-cache write path:

    Today (SGLANG_OPT_USE_FUSED_STORE_CACHE=false), per-layer chain:
      1) apply_rotary_emb_triton(kv[..., -64:].unsqueeze(1), freqs_cis, positions)
         → 1 launch (in-place RoPE on rope segment)
      2) quant_to_nope_fp8_rope_bf16_pack_triton(kv)
         → 1 launch grid (N, 8); allocates 3 outputs (k_nope_fp8 / k_rope_bf16 /
           scale_k_nope_ue8m0)
      3) translate_loc_from_full_to_swa(raw_loc)  (1 launch; cached when
         SGLANG_OPT_CACHE_SWA_TRANSLATION=1, so often 0 launches/layer)
      4) _set_k_and_s_triton(buf, swa_loc, pack)
         → 1 launch scatter into the paged uint8 buffer

    M1 megakernel collapses ops 1 + 2 + 4 into ONE launch:
      grid = (N,), each CTA:
        * loads 512 bf16 from kv[i, :]
        * applies RoPE in-register on the trailing 64 elems (re-uses the
          existing partner-lane / xor-1 trick from fused_norm_rope_triton)
        * writes the rope-rotated 64 lanes BACK to kv[i, -64:]   (so attn
          sees it; matches in-place semantic of apply_rotary_emb_triton)
        * per-tile (7 tiles × 64 elems) FP8 quant of the nope segment with
          ue8m0 scale (matches quant_to_nope_fp8_rope_bf16_pack semantics)
        * scatters fp8 nope, bf16 rope, ue8m0 scale into the paged uint8
          buffer at loc[i] using the same byte layout as
          _set_k_and_s_triton_kernel

3 launches → 1 launch per layer × 60 layers = 180 → 60 launches/step. At
ROCm 7.2 HSA dispatch ~3.92 µs/launch the host-side budget is ~470 µs/step
saved (assuming the chain was eager-amortized; under cuda-graph replay the
saving is dominated by intermediate-allocation elision + RAW elimination).

Memory layout (matches the unfused production path's _set_k_and_s_triton):

  Each page is `BUF_NUMEL_PER_PAGE` bytes (= bytes_per_page_padded, ceil to 576).
  Within a page (page_size = 64 tokens):
    bytes [0 .. 576*page_size)             : per-token (448 fp8 nope + 64*2 bf16 rope)
    bytes [576*page_size .. 576*page_size + 8*page_size) : per-token (7 ue8m0 scale + 1 pad)

  Per-token within the data region:
    [0 .. 448)            fp8 nope (8 tiles × 56 ... actually 7 tiles × 64)
                           NB: the unfused kernel uses 7 tiles × 64 = 448
    [448 .. 576)          bf16 rope (64 elems × 2 bytes)

References:
  fused_store_cache_triton.py::_fused_store_flashmla_kernel — quant+scatter (no rope)
  index_buf_accessor_v4.py::_set_k_and_s_triton_kernel — pure scatter (used today)
  quant_k_cache_v4.py::_quant_k_cache_fused_kernel — quant only (used today)
  deepseek_v4_rope.py::apply_rotary_emb_triton_kernel — rope only (used today)
  fused_norm_rope_triton.py::_fused_norm_rope_kernel — partner-lane RoPE pattern

Author: M1 megakernel agent, 2026-04-29.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


_FP8_MAX_TL = tl.constexpr(240.0)
_FP8_MAX = 240.0


# ---------------------------------------------------------------------------
# M1 megakernel — single launch fuses RoPE + per-tile FP8 quant + paged scatter.
# ---------------------------------------------------------------------------


@triton.jit
def _m1_kv_write_with_rope_kernel(
    kv_ptr,                # bf16 [N, 512]   — IN/OUT (rope written back in-place)
    freqs_real_ptr,        # fp32 [max_pos, ROPE_DIM]   real/imag interleaved
    positions_ptr,         # int64 [N]
    loc_ptr,               # int32 [N]   — already translated to SWA index
    buf_fp8_ptr,           # cache.view(fp8_dtype)        — for fp8 stores
    buf_bf16_ptr,          # cache.view(bfloat16)         — for bf16 rope stores
    buf_uint8_ptr,         # cache.view(uint8)            — for ue8m0 scale stores
    kv_stride_0: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BUF_NUMEL_PER_PAGE: tl.constexpr,   # = bytes_per_page_padded
    HEAD_DIM: tl.constexpr,             # 512
    NOPE_DIM: tl.constexpr,             # 448
    ROPE_DIM: tl.constexpr,             # 64
    NUM_TILES: tl.constexpr,            # 7    (NOPE_DIM // TILE_SIZE)
    TILE_SIZE: tl.constexpr,            # 64
    NUM_NOPE_ROPE_BYTES_PER_TOKEN: tl.constexpr,    # 576
    PADDED_SCALE_ELEMS_PER_TOKEN: tl.constexpr,     # 8 (= 7 + 1 pad)
    S_OFFSET_NBYTES_IN_PAGE: tl.constexpr,          # PAGE_SIZE * (NOPE_DIM + ROPE_DIM*2)
    BLOCK_NOPE: tl.constexpr,                       # next-pow2 of NOPE_DIM (512 for 448)
    NUM_TILES_PADDED: tl.constexpr,                 # BLOCK_NOPE // TILE_SIZE (8)
):
    bid = tl.program_id(0)

    # ───── Step 1: load full row (NOPE part with mask) ──────────────────────
    # tl.arange requires power-of-2; NOPE_DIM=448 → use BLOCK_NOPE=512 + mask.
    # Reshape to [NUM_TILES, TILE_SIZE] = [8, 64] for the per-tile reduction;
    # tile_id 7 is the masked-out tail (NUM_TILES_PADDED = 8 = BLOCK_NOPE / TILE_SIZE).
    nope_col = tl.arange(0, BLOCK_NOPE)             # 0..511
    nope_mask = nope_col < NOPE_DIM
    rope_col = tl.arange(0, ROPE_DIM)               # 0..63

    nope_off = bid * kv_stride_0 + nope_col
    rope_off = bid * kv_stride_0 + NOPE_DIM + rope_col   # 448..511

    x_nope_bf16 = tl.load(kv_ptr + nope_off, mask=nope_mask, other=0.0)  # bf16
    x_rope_bf16 = tl.load(kv_ptr + rope_off)         # bf16

    # ───── Step 2: RoPE on the rope segment (in-register) ────────────────────
    position = tl.load(positions_ptr + bid).to(tl.int64)

    # Convert rope row to fp32 for the rotation
    x_rope_f32 = x_rope_bf16.to(tl.float32)

    # rope segment is laid out as real/imag interleaved: pairs (lane 2k, lane 2k+1)
    pair = rope_col // 2                                 # 0..ROPE_DIM/2-1
    is_imag = (rope_col & 1) == 1

    freq_base = position * ROPE_DIM
    freq_real = tl.load(freqs_real_ptr + freq_base + 2 * pair)
    freq_imag = tl.load(freqs_real_ptr + freq_base + 2 * pair + 1)

    # Use partner-lane (xor 1) trick — same as fused_norm_rope_triton.
    # Compute partner element values via direct index swap inside the same
    # vector. We have the full ROPE_DIM in `x_rope_f32`; just gather along
    # the xor-1 axis via tl.load again on the same row offsets.
    partner_lane = rope_col ^ 1                          # swap real<->imag in-pair
    partner_off = bid * kv_stride_0 + NOPE_DIM + partner_lane
    partner_bf16 = tl.load(kv_ptr + partner_off)
    partner_f32 = partner_bf16.to(tl.float32)

    # if lane is real (even): real=x, imag=partner
    # if lane is imag (odd):  real=partner, imag=x
    real = tl.where(is_imag, partner_f32, x_rope_f32)
    imag = tl.where(is_imag, x_rope_f32, partner_f32)
    rope_rotated_f32 = tl.where(
        is_imag,
        real * freq_imag + imag * freq_real,
        real * freq_real - imag * freq_imag,
    )
    rope_rotated_bf16 = rope_rotated_f32.to(tl.bfloat16)

    # ───── Step 2b: write rope back IN-PLACE so downstream attn sees it ─────
    tl.store(kv_ptr + rope_off, rope_rotated_bf16)

    # ───── Step 3: load loc, compute page indexing ──────────────────────────
    loc = tl.load(loc_ptr + bid)
    loc_page = loc // PAGE_SIZE
    loc_in_page = loc % PAGE_SIZE
    page_byte_base_fp8 = loc_page * BUF_NUMEL_PER_PAGE     # for fp8 view (unit = 1 byte)
    # bf16 view stride is half of uint8 stride; use BUF_NUMEL_PER_PAGE // 2
    page_byte_base_bf16 = loc_page * (BUF_NUMEL_PER_PAGE // 2)

    # ───── Step 4: per-tile FP8 quantization with ue8m0 scale, store fp8 ───
    # Match quant_to_nope_fp8_rope_bf16_pack_triton semantics (using log2/ceil/exp2 path
    # — *not* the bit-twiddle path in fused_store_cache_triton; this is the path
    # the unfused production today actually executes).
    nope_token_base = (
        page_byte_base_fp8
        + loc_in_page * NUM_NOPE_ROPE_BYTES_PER_TOKEN
    )
    EPS = 1e-8
    scale_off_base = (
        page_byte_base_fp8
        + S_OFFSET_NBYTES_IN_PAGE
        + loc_in_page * PADDED_SCALE_ELEMS_PER_TOKEN
    )

    # Reshape nope (padded to BLOCK_NOPE) into [NUM_TILES_PADDED, TILE_SIZE].
    # For NOPE_DIM=448, TILE_SIZE=64, BLOCK_NOPE=512, NUM_TILES_PADDED=8.
    # Tile 7 is the masked-out tail (zeroed at load); we don't store it.
    x_nope_f32 = x_nope_bf16.to(tl.float32)
    x_nope_2d = tl.reshape(x_nope_f32, (NUM_TILES_PADDED, TILE_SIZE))
    abs_x = tl.abs(x_nope_2d)
    max_abs = tl.max(abs_x, axis=1)                    # [NUM_TILES_PADDED]
    max_abs_clamped = tl.maximum(max_abs, EPS)
    scale = max_abs_clamped / _FP8_MAX_TL              # [NUM_TILES_PADDED]
    log2_scale = tl.log2(scale)
    ceil_log2 = tl.math.ceil(log2_scale)
    scale_pow2 = tl.exp2(ceil_log2)                    # [NUM_TILES_PADDED]
    scale_inv = 1.0 / scale_pow2                       # [NUM_TILES_PADDED]

    # Per-tile cast to fp8 with the per-tile scale_inv (broadcast on TILE_SIZE).
    x_scaled = x_nope_2d * scale_inv[:, None]
    x_clipped = tl.clamp(x_scaled, -_FP8_MAX_TL, _FP8_MAX_TL)
    x_fp8 = x_clipped.to(buf_fp8_ptr.dtype.element_ty)
    x_fp8_flat = tl.reshape(x_fp8, (BLOCK_NOPE,))

    # Store fp8 nope (448 elems contiguous in the page row, mask the tail)
    fp8_dst_off = nope_token_base + nope_col
    tl.store(buf_fp8_ptr + fp8_dst_off, x_fp8_flat, mask=nope_mask)

    # Store bf16 rope (64 elems, 128 bytes after the 448 fp8 elems).
    # buf_bf16_ptr stride is in bf16 elems; offset must be in elems.
    # NUM_NOPE_ROPE_BYTES_PER_TOKEN = 576 bytes = 288 bf16-elems.
    # Within a token row, rope starts after 448 bytes = 224 bf16-elems.
    rope_dst_off_bf16 = (
        page_byte_base_bf16
        + loc_in_page * (NUM_NOPE_ROPE_BYTES_PER_TOKEN // 2)
        + (NOPE_DIM // 2)
        + rope_col
    )
    tl.store(buf_bf16_ptr + rope_dst_off_bf16, rope_rotated_bf16)

    # ───── Step 5: store ue8m0 scale (7 elems, 1 byte each, +1 byte pad) ────
    # ceil_log2 / scale come back at shape [NUM_TILES_PADDED=8]; we only want
    # the first NUM_TILES=7 entries. Use the full 8-lane store with a mask
    # because tl.arange needs power-of-2.
    exponent = ceil_log2.to(tl.int32)
    scale_uint8 = (exponent + 127).to(tl.uint8)        # ue8m0 = exp + 127  [8]
    scale_lane = tl.arange(0, NUM_TILES_PADDED)        # 0..7
    scale_mask = scale_lane < NUM_TILES                # only first 7 are real
    tl.store(buf_uint8_ptr + scale_off_base + scale_lane, scale_uint8,
             mask=scale_mask)


# ---------------------------------------------------------------------------
# Python entry point
# ---------------------------------------------------------------------------


_FP8_DTYPE_CACHE = None
def _fp8_dtype():
    """Match sglang.srt.layers.quantization.fp8_kernel.is_fp8_fnuz semantics
    without importing that module (which transitively requires aiter).

    On AMD MI300/MI350 (gfx94*/gfx95*) bf8/fp8 is FNUZ; on MI200 and elsewhere
    it's the IEEE e4m3fn variant. We detect ROCm + gfx9x via torch.version.hip.
    """
    global _FP8_DTYPE_CACHE
    if _FP8_DTYPE_CACHE is None:
        is_rocm = (getattr(torch.version, "hip", None) is not None)
        if is_rocm:
            try:
                # FNUZ on gfx94x (MI300) and gfx950 (MI355X)
                arch = torch.cuda.get_device_properties(0).gcnArchName
                use_fnuz = arch.startswith("gfx94") or arch.startswith("gfx95")
            except Exception:
                use_fnuz = True
            _FP8_DTYPE_CACHE = (
                torch.float8_e4m3fnuz if use_fnuz else torch.float8_e4m3fn
            )
        else:
            _FP8_DTYPE_CACHE = torch.float8_e4m3fn
    return _FP8_DTYPE_CACHE


def m1_kv_write_with_rope_triton(
    kv: torch.Tensor,                      # [N, 512] bf16  (in-place rope write)
    freqs_cis: torch.Tensor,               # complex64 [max_pos, ROPE_DIM // 2]
    positions: torch.Tensor,               # int64 [N]
    loc: torch.Tensor,                     # int32 [N], already SWA-translated
    buf: torch.Tensor,                     # uint8 [num_pages, page_stride_bytes]
    page_size: int,
) -> None:
    """Single-launch: RoPE in-place on kv + per-tile FP8 quant + paged scatter.

    Args:
        kv: bf16 [N, 512] — in-place: kv[..., -64:] is overwritten with rope-rotated bf16.
        freqs_cis: complex64 [max_pos, 32] precomputed full table (caller provides).
        positions: int64 [N] sequence positions for this step's tokens.
        loc: int32 [N] — already-translated SWA paged-buffer indices.
        buf: uint8 [num_pages, page_stride_bytes] — paged KV cache buffer.
        page_size: int (typically 64; see DeepSeekV4SingleKVPool.create_buffer).
    """
    assert kv.dtype == torch.bfloat16, f"kv dtype must be bfloat16, got {kv.dtype}"
    assert kv.is_contiguous(), "kv must be contiguous"
    N, head_dim = kv.shape
    assert head_dim == 512, f"head_dim must be 512, got {head_dim}"

    NOPE_DIM = 448
    ROPE_DIM = 64
    TILE_SIZE = 64
    NUM_TILES = NOPE_DIM // TILE_SIZE  # 7

    assert positions.shape == (N,), (
        f"positions shape {positions.shape} != ({N},)"
    )
    assert positions.dtype == torch.int64, (
        f"positions dtype must be int64, got {positions.dtype}"
    )
    assert loc.shape == (N,), f"loc shape {loc.shape} != ({N},)"

    if loc.dtype != torch.int32:
        loc = loc.to(torch.int32)

    assert buf.dtype == torch.uint8, f"buf dtype must be uint8, got {buf.dtype}"
    assert buf.is_contiguous(), "buf must be contiguous"
    num_pages, buf_numel_per_page = buf.shape

    # Materialize the freqs_cis as fp32 [max_pos, ROPE_DIM] (real/imag interleaved).
    # Caller may pass complex64; if so, convert.
    if freqs_cis.is_complex():
        freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    else:
        freqs_real = freqs_cis
    assert freqs_real.dtype in (torch.float32, torch.float16, torch.bfloat16), (
        f"freqs dtype must be fp32/fp16/bf16, got {freqs_real.dtype}"
    )
    assert freqs_real.shape[-1] == ROPE_DIM, (
        f"freqs last dim {freqs_real.shape[-1]} != {ROPE_DIM}"
    )

    fp8_dt = _fp8_dtype()
    buf_fp8 = buf.view(fp8_dt)
    buf_bf16 = buf.view(torch.bfloat16)
    buf_u8 = buf.view(torch.uint8)

    NUM_NOPE_ROPE_BYTES_PER_TOKEN = NOPE_DIM + ROPE_DIM * 2     # 448 + 128 = 576
    PADDED_SCALE_ELEMS_PER_TOKEN = NUM_TILES + 1                # 7 + 1 = 8
    S_OFFSET_NBYTES_IN_PAGE = page_size * NUM_NOPE_ROPE_BYTES_PER_TOKEN
    # Pad NOPE_DIM up to next pow2 for tl.arange
    BLOCK_NOPE = 1
    while BLOCK_NOPE < NOPE_DIM:
        BLOCK_NOPE *= 2

    grid = (N,)
    _m1_kv_write_with_rope_kernel[grid](
        kv,
        freqs_real,
        positions,
        loc,
        buf_fp8,
        buf_bf16,
        buf_u8,
        kv_stride_0=kv.stride(0),
        PAGE_SIZE=page_size,
        BUF_NUMEL_PER_PAGE=buf_numel_per_page,
        HEAD_DIM=head_dim,
        NOPE_DIM=NOPE_DIM,
        ROPE_DIM=ROPE_DIM,
        NUM_TILES=NUM_TILES,
        TILE_SIZE=TILE_SIZE,
        NUM_NOPE_ROPE_BYTES_PER_TOKEN=NUM_NOPE_ROPE_BYTES_PER_TOKEN,
        PADDED_SCALE_ELEMS_PER_TOKEN=PADDED_SCALE_ELEMS_PER_TOKEN,
        S_OFFSET_NBYTES_IN_PAGE=S_OFFSET_NBYTES_IN_PAGE,
        BLOCK_NOPE=BLOCK_NOPE,
        NUM_TILES_PADDED=BLOCK_NOPE // TILE_SIZE,
        num_warps=4,
        waves_per_eu=2,
    )


# ---------------------------------------------------------------------------
# Torch reference (mirrors today's actual production chain).
# ---------------------------------------------------------------------------


def m1_kv_write_with_rope_torch(
    kv: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
    loc: torch.Tensor,
    buf: torch.Tensor,
    page_size: int,
) -> None:
    """Reference path — in-place RoPE + quant_pack + scatter, as runs today.

    Calls the EXISTING Triton kernels from the unfused production path:
        apply_rotary_emb_triton  →  quant_to_nope_fp8_rope_bf16_pack_triton  →
        _set_k_and_s_triton

    NB: this is a Triton chain, not a "torch op" chain — the production path
    is already a Triton chain; we measure the megakernel against the chain.

    Imports are *deferred* inside the function and direct (skipping the
    sglang model registry) so the microbench can run without a fully-
    functional aiter / MoE import graph.
    """
    # Direct kernel imports — skip apply_rotary_emb_triton's wrapper module
    # because that module imports `maybe_torch_compile`, which triggers the
    # MoE import chain via cuda_graph_runner. The wrapper logic is replicated
    # below to avoid the heavy import.
    from sglang.srt.layers.attention.nsa.quant_k_cache_v4 import (
        quant_to_nope_fp8_rope_bf16_pack_triton,
    )
    from sglang.srt.layers.attention.nsa.index_buf_accessor_v4 import (
        _set_k_and_s_triton,
    )

    # 1) RoPE in-place on kv[..., -64:].unsqueeze(1) — matches deepseek_v4.py:1990
    _apply_rotary_emb_triton_local(
        kv[..., -64:].unsqueeze(1),
        freqs_cis,
        positions=positions,
    )

    # 2) Quant + pack to nope_fp8 / rope_bf16 / scale_ue8m0
    pack = quant_to_nope_fp8_rope_bf16_pack_triton(kv)

    # 3) Scatter into paged buffer at loc
    _set_k_and_s_triton(buf, loc, pack, page_size=page_size)


# ---------------------------------------------------------------------------
# Local copy of apply_rotary_emb_triton (sglang.srt.layers.deepseek_v4_rope)
# Avoids importing maybe_torch_compile / cuda_graph_runner / MoE chain.
# Bit-identical to the production wrapper at deepseek_v4_rope.py:281.
# ---------------------------------------------------------------------------


@triton.jit
def _apply_rotary_emb_triton_kernel_local(
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
        out_real = x_real * freq_real + x_imag * freq_imag
        out_imag = x_imag * freq_real - x_real * freq_imag
    else:
        out_real = x_real * freq_real - x_imag * freq_imag
        out_imag = x_real * freq_imag + x_imag * freq_real

    tl.store(x_ptr + x_real_offs, out_real, mask=mask_2d)
    tl.store(x_ptr + x_imag_offs, out_imag, mask=mask_2d)


def _apply_rotary_emb_triton_local(x, freqs_cis, positions=None, inverse=False):
    """Local copy of apply_rotary_emb_triton — see top of file for reason."""
    is_3d = x.ndim == 3
    if is_3d:
        batch_size, n_heads, rope_dim = x.shape
    else:
        batch_size, rope_dim = x.shape
        n_heads = 1

    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    BLOCK_M = 64
    grid = (triton.cdiv(batch_size, BLOCK_M), n_heads if is_3d else 1)

    if positions is not None:
        assert positions.shape == (batch_size,)
        _apply_rotary_emb_triton_kernel_local[grid](
            x, freqs_real, positions, rope_dim,
            x.stride(0),
            x.stride(1) if is_3d else 0,
            x.stride(-1),
            freqs_real.stride(0), freqs_real.stride(1),
            batch_size,
            USE_POS=True, IS_INVERSE=inverse, IS_3D=is_3d,
            BLOCK_M=BLOCK_M, ROPE_DIM=rope_dim,
        )
    else:
        assert freqs_real.shape[0] == batch_size
        _apply_rotary_emb_triton_kernel_local[grid](
            x, freqs_real, x, rope_dim,
            x.stride(0),
            x.stride(1) if is_3d else 0,
            x.stride(-1),
            freqs_real.stride(0), freqs_real.stride(1),
            batch_size,
            USE_POS=False, IS_INVERSE=inverse, IS_3D=is_3d,
            BLOCK_M=BLOCK_M, ROPE_DIM=rope_dim,
        )
