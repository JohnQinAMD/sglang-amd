"""M2 megakernel — `compress_decode` compression core (Stage1) + RMSNorm + RoPE epilogue (Stage2).

Fuses up to 8 ops from `compress_decode_old` (deepseek_v4.py:1483-1563):

    # Stage1 (always fused):
    score = score + ape.unsqueeze(0)              # aten::add (skipped when ape=None)
    weights = score.softmax(dim=1)                 # aten::_softmax
    kv_compressed = (kv * weights).sum(dim=1)      # aten::mul + aten::sum
    # Stage2 (when norm_weight + freqs_cis_per_bs supplied):
    kv_compressed = kv_compressed * rsqrt(mean(kv_compressed**2) + eps) * norm_weight  # RMSNorm
    kv_compressed[..., -rope_dim:] = apply_rotary(kv_compressed[..., -rope_dim:], freqs)  # RoPE

Layout (post-view, pre-compression):
  kv:     (bs, S, D) where S = ratio * coff, D = head_dim
  score:  (bs, S, D)
  ape:    (S, D)  — broadcast-added to score before softmax
  out:    (bs, D)

OLD path (compress_decode_old, line 1483) has kv and score as SEPARATE
tensors (KVAndScoreOld). NEW path (compress_decode, line 1146) has them
PACKED in one tensor's last dim (KVAndScore: kv = first D, score = second D).

Kernel takes raw ptrs + element-strides so caller passes either layout
without copy.

Production-shape v2-microbench-validated: SHIPS only if v2 verdict is SHIP
under both eager and graph-replay timing on production shape histogram.
"""
from __future__ import annotations

from typing import Optional
import torch
import triton
import triton.language as tl


@triton.jit
def _compress_decode_core_kernel(
    kv_ptr,            # bf16 / fp16 / fp32; shape (bs, S, D)
    score_ptr,         # same shape as kv
    ape_ptr,           # shape (S, D); ignored when ADD_APE=False (caller passes score itself as a dummy)
    out_ptr,           # shape (bs, D)
    s_kvb, s_kvs, s_kvd,
    s_scb, s_scs, s_scd,
    s_apes, s_aped,
    s_outb, s_outd,
    bs,
    S: tl.constexpr,    # ratio * coff (typically 4 or 8)
    D: tl.constexpr,    # head_dim (e.g. 128)
    BLOCK_D: tl.constexpr,
    ADD_APE: tl.constexpr,  # True for overlap=False (c=128) path; False for c=4 path where APE was pre-applied
):
    """One program per (bs_idx, d_tile).

    Fast layout: S is small (4-8), D is the contiguous dim (128).
    We load all S rows × BLOCK_D cols into a (S, BLOCK_D) tile, do softmax
    along S, multiply by kv, reduce along S, store one row of (D,) per bs.

    When ADD_APE=True, fuses 4 ops (add+softmax+mul+sum). When False, fuses 3
    ops (softmax+mul+sum) — used for the overlap=True path where APE was
    added before overlap_transform_decode and is already baked into score.
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    cols = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    col_mask = cols < D

    s_rng = tl.arange(0, S)
    # Offsets for the (S, BLOCK_D) tile of this (pid_b, pid_d) program.
    score_off = pid_b * s_scb + s_rng[:, None] * s_scs + cols[None, :] * s_scd
    kv_off = pid_b * s_kvb + s_rng[:, None] * s_kvs + cols[None, :] * s_kvd
    tile_mask = col_mask[None, :]

    # Load score; optionally fuse APE add in fp32
    score = tl.load(score_ptr + score_off, mask=tile_mask, other=0.0).to(tl.float32)
    if ADD_APE:
        ape_off = s_rng[:, None] * s_apes + cols[None, :] * s_aped
        ape = tl.load(ape_ptr + ape_off, mask=tile_mask, other=0.0).to(tl.float32)
        score = score + ape

    # Softmax along S (axis=0 in this (S, BLOCK_D) tile)
    score_max = tl.max(score, axis=0)
    score_exp = tl.exp(score - score_max[None, :])
    score_sum = tl.sum(score_exp, axis=0)
    weights = score_exp / score_sum[None, :]

    # Load kv in fp32, weighted-sum along S
    kv = tl.load(kv_ptr + kv_off, mask=tile_mask, other=0.0).to(tl.float32)
    out = tl.sum(kv * weights, axis=0)

    # Store back as the input dtype
    out_off = pid_b * s_outb + cols * s_outd
    tl.store(out_ptr + out_off, out.to(out_ptr.dtype.element_ty), mask=col_mask)


def compress_decode_core_triton(
    kv: torch.Tensor,        # (bs, S, D)
    score: torch.Tensor,     # (bs, S, D); for ape=None, must already have APE applied externally
    ape: Optional[torch.Tensor],  # (S, D) or None (skip APE add in kernel)
    out: torch.Tensor,       # (bs, D) pre-allocated
) -> bool:
    """Fused: out = (kv * softmax(score + ape, dim=1)).sum(dim=1) when ape is set.
    With ape=None: out = (kv * softmax(score, dim=1)).sum(dim=1).

    Returns True if Triton fired, False if caller should fall back to torch.
    """
    assert kv.dim() == 3 and score.dim() == 3, f"kv/score must be 3D"
    assert out.dim() == 2, f"out must be 2D"
    bs, S, D = kv.shape
    assert score.shape == kv.shape, f"score shape {score.shape} != kv {kv.shape}"
    assert out.shape == (bs, D), f"out {out.shape} != ({bs}, {D})"
    assert kv.dtype == score.dtype == out.dtype, "all tensors must share dtype"
    if ape is not None:
        assert ape.dim() == 2, f"ape must be 2D"
        assert ape.shape == (S, D), f"ape {ape.shape} != ({S}, {D})"
        assert ape.dtype == kv.dtype, "ape dtype must match kv"

    # Constraint for the (S, BLOCK_D) tile: S must be a power of 2 for tl.arange
    if S & (S - 1) != 0:
        return False
    if S > 16:
        return False

    # Pick BLOCK_D: full D fits if power-of-2 ≤ 256
    if D <= 256:
        BLOCK_D = triton.next_power_of_2(D)
    else:
        BLOCK_D = 128

    grid = (bs, triton.cdiv(D, BLOCK_D))
    add_ape = ape is not None
    # When ADD_APE=False the kernel ignores ape_ptr; pass score as a dummy non-null
    # ptr (Triton needs a valid ptr; the load is skipped via constexpr branch).
    ape_ptr_arg = ape if add_ape else score
    ape_stride0 = ape.stride(0) if add_ape else 0
    ape_stride1 = ape.stride(1) if add_ape else 0
    _compress_decode_core_kernel[grid](
        kv, score, ape_ptr_arg, out,
        kv.stride(0), kv.stride(1), kv.stride(2),
        score.stride(0), score.stride(1), score.stride(2),
        ape_stride0, ape_stride1,
        out.stride(0), out.stride(1),
        bs,
        S=S, D=D, BLOCK_D=BLOCK_D,
        ADD_APE=add_ape,
        # Tuned for small S, small D (production shapes): minimize launch overhead
        num_warps=2, num_stages=1,
    )
    return True


def compress_decode_core_torch(
    kv: torch.Tensor,        # (bs, S, D)
    score: torch.Tensor,     # (bs, S, D)
    ape: Optional[torch.Tensor],  # (S, D) or None
    out: torch.Tensor,       # (bs, D) — written in-place
) -> None:
    """Reference torch path: 4 launches (add + softmax + mul + sum) when ape set,
    3 launches (softmax + mul + sum) when ape=None."""
    if ape is not None:
        s = score + ape.unsqueeze(0)         # 1 launch
    else:
        s = score
    w = s.softmax(dim=1)                      # 1 launch
    out.copy_((kv * w).sum(dim=1))            # 2 launches (mul + sum)


# ============================================================================
# M2-Stage2: same kernel, plus RMSNorm + RoPE epilogue
# ============================================================================

@triton.jit
def _compress_decode_full_kernel(
    kv_ptr,            # (bs, S, D) input
    score_ptr,         # (bs, S, D) input
    ape_ptr,           # (S, D) optional, ignored if ADD_APE=False
    norm_w_ptr,        # (D,) fp32 RMSNorm weight
    freqs_ptr,         # (bs, rope_dim // 2, 2) — real/imag interleaved per bs row
    out_ptr,           # (bs, D)
    s_kvb, s_kvs, s_kvd,
    s_scb, s_scs, s_scd,
    s_apes, s_aped,
    s_normw,
    s_freqsb, s_freqsd,
    s_outb, s_outd,
    bs,
    eps,
    S: tl.constexpr,
    D: tl.constexpr,        # head_dim (must = BLOCK_D for the RMSNorm reduction)
    ROPE_DIM: tl.constexpr, # rope_head_dim — RoPE applied to last ROPE_DIM cols
    BLOCK_D: tl.constexpr,
    ADD_APE: tl.constexpr,
):
    """One program per bs row. BLOCK_D MUST equal D so that the RMSNorm
    mean reduction sees the full head_dim row in a single tile.
    """
    pid_b = tl.program_id(0)

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D

    s_rng = tl.arange(0, S)
    score_off = pid_b * s_scb + s_rng[:, None] * s_scs + cols[None, :] * s_scd
    kv_off = pid_b * s_kvb + s_rng[:, None] * s_kvs + cols[None, :] * s_kvd
    tile_mask = col_mask[None, :]

    # ---- Stage1: APE add + softmax(score) + (kv * softmax).sum ----
    score = tl.load(score_ptr + score_off, mask=tile_mask, other=0.0).to(tl.float32)
    if ADD_APE:
        ape_off = s_rng[:, None] * s_apes + cols[None, :] * s_aped
        ape = tl.load(ape_ptr + ape_off, mask=tile_mask, other=0.0).to(tl.float32)
        score = score + ape

    score_max = tl.max(score, axis=0)
    score_exp = tl.exp(score - score_max[None, :])
    score_sum = tl.sum(score_exp, axis=0)
    weights = score_exp / score_sum[None, :]

    kv = tl.load(kv_ptr + kv_off, mask=tile_mask, other=0.0).to(tl.float32)
    out = tl.sum(kv * weights, axis=0)   # (BLOCK_D,) fp32

    # ---- Stage2: RMSNorm ----
    # rsqrt(mean(out^2) + eps) * out * norm_weight
    sq = out * out
    sq = tl.where(col_mask, sq, 0.0)
    mean_sq = tl.sum(sq, axis=0) / tl.cast(D, tl.float32)
    rstd = tl.rsqrt(mean_sq + eps)
    norm_w = tl.load(norm_w_ptr + cols * s_normw, mask=col_mask, other=0.0).to(tl.float32)
    out = out * rstd * norm_w

    # ---- Stage2: RoPE on last ROPE_DIM columns, in-register via tl.split ----
    # The probe at /tmp/probe_tl_split.py verified this pattern works on
    # chi2811's Triton 3.6 / ROCm 7.2 build.
    if ROPE_DIM > 0:
        # (BLOCK_D,) -> (BLOCK_D//2, 2) view, split into real/imag halves.
        out_pairs = tl.reshape(out, (BLOCK_D // 2, 2))
        real, imag = tl.split(out_pairs)   # each (BLOCK_D//2,)

        # Pair i (cols 2i, 2i+1) is in the rope half if 2i >= rope_offset.
        rope_offset = D - ROPE_DIM
        pair_i = tl.arange(0, BLOCK_D // 2)
        # Clamp at D: padding pairs (BLOCK_D > D case) have pair_i*2 >= D
        # and must NOT be marked as rope (would OOB-read freqs).
        is_rope_pair = ((pair_i * 2) >= rope_offset) & ((pair_i * 2) < D)
        # Local index within rope half (0..ROPE_DIM/2-1 for valid pairs;
        # garbage for non-rope pairs but masked out via is_rope_pair).
        rope_pair_idx = pair_i - rope_offset // 2

        # cos/sin per pair from freqs_per_bs[pid_b, rope_pair_idx, 0/1]
        freq_real_off = pid_b * s_freqsb + rope_pair_idx * s_freqsd * 2
        freq_imag_off = freq_real_off + s_freqsd
        cos_v = tl.load(freqs_ptr + freq_real_off, mask=is_rope_pair, other=1.0).to(tl.float32)
        sin_v = tl.load(freqs_ptr + freq_imag_off, mask=is_rope_pair, other=0.0).to(tl.float32)

        new_real = tl.where(is_rope_pair, real * cos_v - imag * sin_v, real)
        new_imag = tl.where(is_rope_pair, real * sin_v + imag * cos_v, imag)

        # Re-interleave (BLOCK_D//2, 2) -> (BLOCK_D,) via tl.join + reshape.
        joined = tl.join(new_real, new_imag)   # (BLOCK_D//2, 2)
        out = tl.reshape(joined, (BLOCK_D,))

    out_off = pid_b * s_outb + cols * s_outd
    tl.store(out_ptr + out_off, out.to(out_ptr.dtype.element_ty), mask=col_mask)


# ============================================================================
# M2-Stage2 LARGE-S variant — streaming online-softmax for S > 16 (e.g. c128
# layers with S=128). Avoids materializing the (S, BLOCK_D) score tile in
# registers. Mathematically equivalent to the small-S kernel above; uses
# FlashAttention-style running max + sum-of-exp + weighted-acc.
# ============================================================================

@triton.jit
def _compress_decode_full_kernel_large_s(
    kv_ptr,            # (bs, S, D)
    score_ptr,         # (bs, S, D)
    ape_ptr,           # (S, D) optional
    norm_w_ptr,        # (D,) fp32
    freqs_ptr,         # (bs, rope_dim//2, 2)
    out_ptr,           # (bs, D)
    s_kvb, s_kvs, s_kvd,
    s_scb, s_scs, s_scd,
    s_apes, s_aped,
    s_normw,
    s_freqsb, s_freqsd,
    s_outb, s_outd,
    bs,
    S,                 # runtime arg (kernel can handle any S)
    eps,
    D: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    S_CHUNK: tl.constexpr,
    ADD_APE: tl.constexpr,
):
    """One program per bs row. Streams over S in chunks of S_CHUNK using
    online softmax to bound register/LDS pressure independent of S. BLOCK_D
    must equal D so RMSNorm reduction sees the full row in one tile.
    """
    pid_b = tl.program_id(0)

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D

    NEG_INF = float("-inf")
    m = tl.full((BLOCK_D,), NEG_INF, dtype=tl.float32)   # running max  per col
    l = tl.zeros((BLOCK_D,), dtype=tl.float32)            # running sum-exp per col
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)          # running weighted-kv-sum per col

    s_chunk_rng = tl.arange(0, S_CHUNK)
    for s_start in range(0, S, S_CHUNK):
        s_offs = s_start + s_chunk_rng
        s_mask = s_offs < S

        score_off = pid_b * s_scb + s_offs[:, None] * s_scs + cols[None, :] * s_scd
        kv_off = pid_b * s_kvb + s_offs[:, None] * s_kvs + cols[None, :] * s_kvd
        chunk_mask = s_mask[:, None] & col_mask[None, :]

        # Load score; pad masked-out S positions with -inf so they vanish
        # under exp() (zero contribution to max + sum + acc).
        score_chunk = tl.load(score_ptr + score_off, mask=chunk_mask, other=NEG_INF).to(tl.float32)
        if ADD_APE:
            ape_off = s_offs[:, None] * s_apes + cols[None, :] * s_aped
            ape_chunk = tl.load(ape_ptr + ape_off, mask=chunk_mask, other=0.0).to(tl.float32)
            score_chunk = score_chunk + ape_chunk

        # Online softmax update.
        chunk_max = tl.max(score_chunk, axis=0)               # (BLOCK_D,)
        m_new = tl.maximum(m, chunk_max)
        rescale = tl.exp(m - m_new)
        l = l * rescale
        acc = acc * rescale

        score_exp = tl.exp(score_chunk - m_new[None, :])      # (S_CHUNK, BLOCK_D)
        l += tl.sum(score_exp, axis=0)
        kv_chunk = tl.load(kv_ptr + kv_off, mask=chunk_mask, other=0.0).to(tl.float32)
        acc += tl.sum(score_exp * kv_chunk, axis=0)

        m = m_new

    out = acc / l        # (BLOCK_D,) fp32 — Stage1 output

    # ---- Stage2: RMSNorm ----
    sq = out * out
    sq = tl.where(col_mask, sq, 0.0)
    mean_sq = tl.sum(sq, axis=0) / tl.cast(D, tl.float32)
    rstd = tl.rsqrt(mean_sq + eps)
    norm_w = tl.load(norm_w_ptr + cols * s_normw, mask=col_mask, other=0.0).to(tl.float32)
    out = out * rstd * norm_w

    # ---- Stage2: RoPE on last ROPE_DIM cols (in-register pair-swap) ----
    if ROPE_DIM > 0:
        out_pairs = tl.reshape(out, (BLOCK_D // 2, 2))
        real, imag = tl.split(out_pairs)

        rope_offset = D - ROPE_DIM
        pair_i = tl.arange(0, BLOCK_D // 2)
        # Clamp at D: padding pairs (BLOCK_D > D case) have pair_i*2 >= D
        # and must NOT be marked as rope (would OOB-read freqs).
        is_rope_pair = ((pair_i * 2) >= rope_offset) & ((pair_i * 2) < D)
        rope_pair_idx = pair_i - rope_offset // 2

        freq_real_off = pid_b * s_freqsb + rope_pair_idx * s_freqsd * 2
        freq_imag_off = freq_real_off + s_freqsd
        cos_v = tl.load(freqs_ptr + freq_real_off, mask=is_rope_pair, other=1.0).to(tl.float32)
        sin_v = tl.load(freqs_ptr + freq_imag_off, mask=is_rope_pair, other=0.0).to(tl.float32)

        new_real = tl.where(is_rope_pair, real * cos_v - imag * sin_v, real)
        new_imag = tl.where(is_rope_pair, real * sin_v + imag * cos_v, imag)

        joined = tl.join(new_real, new_imag)
        out = tl.reshape(joined, (BLOCK_D,))

    out_off = pid_b * s_outb + cols * s_outd
    tl.store(out_ptr + out_off, out.to(out_ptr.dtype.element_ty), mask=col_mask)


def compress_decode_full_triton(
    kv: torch.Tensor,            # (bs, S, D)
    score: torch.Tensor,         # (bs, S, D)
    ape: Optional[torch.Tensor], # (S, D) or None
    norm_weight: torch.Tensor,   # (D,) fp32 RMSNorm weight
    freqs_per_bs: torch.Tensor,  # (bs, rope_dim // 2, 2) complex stored as real/imag
    eps: float,
    rope_dim: int,
    out: torch.Tensor,           # (bs, D)
) -> bool:
    """Stage2 megakernel: Stage1 + RMSNorm + RoPE in one launch.

    `freqs_per_bs` must be pre-indexed: caller computes
        freqs_per_bs = freqs_cis[(seq_lens - 1) // ratio * ratio]
    and ensures it's stored as (bs, rope_dim // 2, 2) real/imag.

    Returns True if Triton fired, False if caller should fall back to torch.
    """
    bs, S, D = kv.shape
    # S must be reachable by EITHER variant: small-S (S ≤ 16, power of 2)
    # OR large-S (any S, processed via streaming online softmax with
    # S_CHUNK-tiled loop).
    # 2026-05-02: previously required D power-of-2 ≤ 256; now handles arbitrary
    # D ≤ 512 by padding BLOCK_D to next_power_of_2(D) and masking via col_mask.
    # Unblocks DSv4 MQALayer compressor where head_dim = qk_rope + qk_nope =
    # 64 + 448 = 512 (in 2604 mode), which previously fell back to the
    # 4-launch torch chain (softmax + sum + RMSNorm + RoPE).
    if D > 512:
        return False
    if rope_dim & 1 != 0 or rope_dim > D:
        return False
    BLOCK_D = triton.next_power_of_2(D)

    add_ape = ape is not None
    ape_arg = ape if add_ape else score
    ape_s0 = ape.stride(0) if add_ape else 0
    ape_s1 = ape.stride(1) if add_ape else 0

    grid = (bs,)

    # Pick kernel variant: small-S materializes the full (S, BLOCK_D) tile
    # (faster when register pressure allows). Large-S streams in S_CHUNK
    # chunks (lower register pressure, fits any S).
    use_small_s = (S <= 16) and (S & (S - 1) == 0)

    if use_small_s:
        _compress_decode_full_kernel[grid](
            kv, score, ape_arg, norm_weight, freqs_per_bs, out,
            kv.stride(0), kv.stride(1), kv.stride(2),
            score.stride(0), score.stride(1), score.stride(2),
            ape_s0, ape_s1,
            norm_weight.stride(0),
            freqs_per_bs.stride(0), freqs_per_bs.stride(-1),
            out.stride(0), out.stride(1),
            bs, float(eps),
            S=S, D=D, ROPE_DIM=rope_dim, BLOCK_D=BLOCK_D,
            ADD_APE=add_ape,
            num_warps=2, num_stages=1,
        )
    else:
        # Large-S: stream over S in chunks. S_CHUNK = 16 is a sweet spot for
        # MI355X (fits in registers, amortizes load latency, avoids spills).
        S_CHUNK = 16
        _compress_decode_full_kernel_large_s[grid](
            kv, score, ape_arg, norm_weight, freqs_per_bs, out,
            kv.stride(0), kv.stride(1), kv.stride(2),
            score.stride(0), score.stride(1), score.stride(2),
            ape_s0, ape_s1,
            norm_weight.stride(0),
            freqs_per_bs.stride(0), freqs_per_bs.stride(-1),
            out.stride(0), out.stride(1),
            bs, S, float(eps),
            D=D, ROPE_DIM=rope_dim, BLOCK_D=BLOCK_D,
            S_CHUNK=S_CHUNK, ADD_APE=add_ape,
            num_warps=4, num_stages=2,
        )
    return True


def compress_decode_full_torch(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: Optional[torch.Tensor],
    norm_weight: torch.Tensor,
    freqs_per_bs: torch.Tensor,
    eps: float,
    rope_dim: int,
    out: torch.Tensor,
) -> None:
    """Reference: 7-8 launches (compress_decode_core_torch + RMSNorm + RoPE)."""
    compress_decode_core_torch(kv, score, ape, out)
    # RMSNorm: 3+ launches
    out_fp32 = out.to(torch.float32)
    var = out_fp32.pow(2).mean(-1, keepdim=True)
    out_fp32 = out_fp32 * torch.rsqrt(var + eps) * norm_weight
    out.copy_(out_fp32.to(out.dtype))
    # RoPE on last rope_dim columns (in-place, complex multiply)
    rope_offset = out.shape[-1] - rope_dim
    rope_half = out[..., rope_offset:].to(torch.float32)
    real = rope_half[..., 0::2]
    imag = rope_half[..., 1::2]
    cos = freqs_per_bs[..., 0]
    sin = freqs_per_bs[..., 1]
    new_real = real * cos - imag * sin
    new_imag = real * sin + imag * cos
    rope_half[..., 0::2] = new_real
    rope_half[..., 1::2] = new_imag
    out[..., rope_offset:] = rope_half.to(out.dtype)
