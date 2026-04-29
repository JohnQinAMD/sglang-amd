"""Triton replacement for the TileLang mHC pre block.

Replaces the 2-stage `mhc_pre_gemm_sqrsum_splitk_kernel` + `mhc_pre_big_fuse_tilelang`
with 3 Triton kernels matching the original computation:

  Stage 1: out[n, j] = (x[n, :] @ fn[j, :]) for j in [0, hc_mult3)
           sqrsum[n] = sum(x[n, :] ** 2)
           x: (n, hc_hidden) bf16, fn: (hc_mult3, hc_hidden) fp32
           out: (n, hc_mult3) fp32, sqrsum: (n,) fp32
  Stage 2: rms[n] = rsqrt(sqrsum[n] / hc_hidden + rms_eps)
           mixes[n, j] = out[n, j] * rms[n]
           Split mixes -> pre_mix (hc), post_mix (hc), comb (hc, hc)
           post_mix = sigmoid(mix * scale[1] + base[hc:2hc]) * post_mult_value
           Sinkhorn iters on comb (hc, hc) -> doubly-stochastic
           Output: post_mix (n, hc), comb (n, hc, hc), pre_mix (n, hc)
  Stage 3: layer_input[n, h] = sum_hc(pre_mix[n, hc] * residual[n, hc, h])

Replaces TileLang `mhc_pre_gemm_sqrsum_splitk_stage_0` (the +21 ms TPOT
regression source on chi2811 Flash-Base FP8 — 287 µs/call vs 27 µs/call
in Phase 13).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


# ============================================================================
# Stage 1: GEMM + sqrsum (split-K + tl.dot, tuned for small n)
# ============================================================================
# Per-token CTA grid was a perf trap at small n (e.g. n=6 → 6 CTAs / 256 CUs
# = 2.3% utilization, ~28 µs/call dominating the 3-stage pipeline). Restructured
# to a 2D grid (split_k, n_blocks) where split_k provides occupancy and the
# token block lets us use real `tl.dot` (MFMA) for the GEMV. Each CTA produces
# partial (TOKEN_BLOCK, HC_MULT3_PAD) outputs and partial sqrsum; the small
# Stage 1.5 reduction kernel sums splits.
@triton.jit
def _mhc_pre_gemm_sqrsum_splitk_kernel(
    x_ptr,              # (n, hc_hidden) bf16
    fn_ptr,             # (hc_mult3, hc_hidden) fp32
    out_partial_ptr,    # (split_k, n_padded, HC_MULT3_PAD) fp32
    sqrsum_partial_ptr, # (split_k, n_padded) fp32
    n,
    hc_hidden,
    n_padded,
    HC_MULT3: tl.constexpr,
    HC_MULT3_PAD: tl.constexpr,
    TOKEN_BLOCK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,        # hc_hidden // SPLIT_K
    BLOCK_K: tl.constexpr,           # tile along K within each split
):
    pid_split = tl.program_id(0)     # 0..SPLIT_K-1
    pid_token = tl.program_id(1)     # 0..ceildiv(n, TOKEN_BLOCK)-1

    token_offs = pid_token * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
    token_mask = token_offs < n

    j_offs = tl.arange(0, HC_MULT3_PAD)
    j_mask = j_offs < HC_MULT3

    k_base = pid_split * SPLIT_SIZE

    acc = tl.zeros((TOKEN_BLOCK, HC_MULT3_PAD), dtype=tl.float32)
    sqsum = tl.zeros((TOKEN_BLOCK,), dtype=tl.float32)

    for k_start in range(0, SPLIT_SIZE, BLOCK_K):
        k_offs = k_base + k_start + tl.arange(0, BLOCK_K)

        # Load x[token_offs, k_offs] -> (TOKEN_BLOCK, BLOCK_K) bf16
        x_offs_2d = token_offs[:, None] * hc_hidden + k_offs[None, :]
        x_tile = tl.load(x_ptr + x_offs_2d,
                          mask=token_mask[:, None], other=0.0)  # bf16

        # Accumulate sqrsum per token (in fp32)
        x_tile_f32 = x_tile.to(tl.float32)
        sqsum += tl.sum(x_tile_f32 * x_tile_f32, axis=1)

        # Load fn[j, k_offs] -> (HC_MULT3_PAD, BLOCK_K) fp32, then we want
        # x_tile @ fn^T → (TOKEN_BLOCK, HC_MULT3_PAD).  Cast fp32 fn to bf16
        # so tl.dot can use MFMA (bf16 inputs + fp32 accumulator).
        fn_offs = j_offs[:, None] * hc_hidden + k_offs[None, :]
        fn_tile = tl.load(fn_ptr + fn_offs, mask=j_mask[:, None], other=0.0)  # fp32
        fn_tile_bf16 = fn_tile.to(tl.bfloat16)

        # tl.dot wants (M, K) @ (K, N) → (M, N).
        # We have x_tile: (TOKEN_BLOCK, BLOCK_K), fn_tile_bf16: (HC_MULT3_PAD, BLOCK_K).
        # Compute x_tile @ fn_tile_bf16^T via transpose_b.
        acc += tl.dot(x_tile, tl.trans(fn_tile_bf16), out_dtype=tl.float32)

    # Store partial outputs (split_k dim is the slow-varying outer dim).
    out_offs = (pid_split * n_padded * HC_MULT3_PAD
                + token_offs[:, None] * HC_MULT3_PAD
                + j_offs[None, :])
    out_mask = token_mask[:, None] & j_mask[None, :]
    tl.store(out_partial_ptr + out_offs, acc, mask=out_mask)

    sq_offs = pid_split * n_padded + token_offs
    tl.store(sqrsum_partial_ptr + sq_offs, sqsum, mask=token_mask)


@triton.jit
def _mhc_pre_splitk_reduce_kernel(
    out_partial_ptr,    # (split_k, n_padded, HC_MULT3_PAD) fp32
    sqrsum_partial_ptr, # (split_k, n_padded) fp32
    out_ptr,            # (n, HC_MULT3_PAD) fp32
    sqrsum_ptr,         # (n,) fp32
    n,
    n_padded,
    HC_MULT3: tl.constexpr,
    HC_MULT3_PAD: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    if pid_n >= n:
        return

    j_offs = tl.arange(0, HC_MULT3_PAD)
    j_mask = j_offs < HC_MULT3

    acc = tl.zeros((HC_MULT3_PAD,), dtype=tl.float32)
    sqsum = tl.zeros((), dtype=tl.float32)
    for s in tl.static_range(0, SPLIT_K):
        partial = tl.load(out_partial_ptr
                           + s * n_padded * HC_MULT3_PAD
                           + pid_n * HC_MULT3_PAD + j_offs,
                           mask=j_mask, other=0.0)
        acc += partial
        sqsum += tl.load(sqrsum_partial_ptr + s * n_padded + pid_n)

    tl.store(out_ptr + pid_n * HC_MULT3_PAD + j_offs, acc, mask=j_mask)
    tl.store(sqrsum_ptr + pid_n, sqsum)


# Backward-compat single-CTA-per-token kernel (kept as a fallback path).
@triton.jit
def _mhc_pre_gemm_sqrsum_kernel(
    x_ptr,         # (n, hc_hidden) bf16
    fn_ptr,        # (hc_mult3, hc_hidden) fp32
    out_ptr,       # (n, hc_mult3_padded=32) fp32 — padded to 32 cols
    sqrsum_ptr,    # (n,) fp32
    n,
    hc_hidden,
    HC_MULT3: tl.constexpr,         # actual = 24 (hc=4: 4*(2+4))
    HC_MULT3_PAD: tl.constexpr,     # padded to 32 for power-of-2
    BLOCK_K: tl.constexpr,          # tile along hc_hidden
):
    pid_n = tl.program_id(0)
    if pid_n >= n:
        return

    row_offs = pid_n * hc_hidden
    j_offs = tl.arange(0, HC_MULT3_PAD)
    j_mask = j_offs < HC_MULT3

    acc = tl.zeros((HC_MULT3_PAD,), dtype=tl.float32)
    sqsum = tl.zeros((), dtype=tl.float32)

    for k_start in range(0, hc_hidden, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < hc_hidden
        x_chunk = tl.load(x_ptr + row_offs + k_offs, mask=k_mask, other=0.0).to(tl.float32)  # (BLOCK_K,)

        # sqrsum
        sqsum += tl.sum(x_chunk * x_chunk)

        # gemm contribution: for each j in [0, HC_MULT3),
        # acc[j] += sum_k(x_chunk[k] * fn[j, k_offs[k]])
        # Load fn[:, k_offs] as (HC_MULT3_PAD, BLOCK_K), then reduce over BLOCK_K.
        fn_offs = j_offs[:, None] * hc_hidden + k_offs[None, :]
        fn_mask = j_mask[:, None] & k_mask[None, :]
        fn_chunk = tl.load(fn_ptr + fn_offs, mask=fn_mask, other=0.0)  # (HC_MULT3_PAD, BLOCK_K)
        acc += tl.sum(fn_chunk * x_chunk[None, :], axis=1)

    tl.store(out_ptr + pid_n * HC_MULT3_PAD + j_offs, acc, mask=j_mask)
    tl.store(sqrsum_ptr + pid_n, sqsum)


# ============================================================================
# Stage 2: rms norm + sigmoid + sinkhorn (small reductions; per-token CTA)
# ============================================================================
@triton.jit
def _mhc_pre_big_fuse_kernel(
    out_ptr,           # (n, HC_MULT3_PAD) fp32 input mixes
    sqrsum_ptr,        # (n,) fp32 input sqrsum
    hc_scale_ptr,      # (3,) fp32
    hc_base_ptr,       # (HC_MULT3,) fp32
    pre_mix_out_ptr,   # (n, HC) fp32   — internal, consumed by stage 3
    post_mix_ptr,      # (n, HC) fp32   — output of mhc_pre
    comb_mix_ptr,      # (n, HC*HC) fp32 — output of mhc_pre
    n,
    hc_hidden,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    HC: tl.constexpr,                # 4
    HC_MULT3: tl.constexpr,          # 24
    HC_MULT3_PAD: tl.constexpr,      # 32
):
    pid_n = tl.program_id(0)
    if pid_n >= n:
        return

    j_offs = tl.arange(0, HC_MULT3_PAD)
    j_mask = j_offs < HC_MULT3

    sqsum_v = tl.load(sqrsum_ptr + pid_n)
    rms = 1.0 / tl.sqrt(sqsum_v / hc_hidden + rms_eps)

    mixes = tl.load(out_ptr + pid_n * HC_MULT3_PAD + j_offs,
                     mask=j_mask, other=0.0).to(tl.float32) * rms  # (HC_MULT3_PAD,)

    # Slice mixes into pre, post, comb.
    # Layout: [pre (HC), post (HC), comb (HC*HC)]  → total = HC + HC + HC*HC = HC*(2+HC) = HC_MULT3
    hc_offs = tl.arange(0, HC)

    # Load scale/base scalars / vectors.
    s0 = tl.load(hc_scale_ptr + 0)
    s1 = tl.load(hc_scale_ptr + 1)
    s2 = tl.load(hc_scale_ptr + 2)
    base_pre = tl.load(hc_base_ptr + hc_offs)  # (HC,)
    base_post = tl.load(hc_base_ptr + HC + hc_offs)  # (HC,)
    base_comb = tl.load(hc_base_ptr + 2 * HC + tl.arange(0, HC * HC))  # (HC*HC,)

    # Reconstruct slices from mixes via masking (mixes is HC_MULT3_PAD-wide).
    # We manually pull the needed components:
    pre_raw = tl.load(out_ptr + pid_n * HC_MULT3_PAD + hc_offs).to(tl.float32) * rms
    post_raw = tl.load(out_ptr + pid_n * HC_MULT3_PAD + HC + hc_offs).to(tl.float32) * rms
    comb_raw = tl.load(out_ptr + pid_n * HC_MULT3_PAD + 2 * HC + tl.arange(0, HC * HC)).to(tl.float32) * rms

    pre_mix = tl.sigmoid(pre_raw * s0 + base_pre) + hc_pre_eps  # (HC,)
    post_mix = tl.sigmoid(post_raw * s1 + base_post) * hc_post_mult_value  # (HC,)

    # Sinkhorn on comb (HC, HC). HC=4 → tiny matrix; do row/col passes.
    comb = comb_raw * s2 + base_comb  # (HC*HC,) flat

    # softmax over rows (axis=-1): reshape (HC, HC), apply row-wise.
    comb_2d = tl.reshape(comb, (HC, HC))
    row_max = tl.max(comb_2d, axis=1)
    comb_2d = tl.exp(comb_2d - row_max[:, None])
    row_sum = tl.sum(comb_2d, axis=1)
    comb_2d = comb_2d / row_sum[:, None] + hc_sinkhorn_eps

    # column normalize
    col_sum = tl.sum(comb_2d, axis=0)
    comb_2d = comb_2d / (col_sum[None, :] + hc_sinkhorn_eps)

    # additional sinkhorn iters
    for _ in range(0, sinkhorn_repeat - 1):
        row_sum = tl.sum(comb_2d, axis=1)
        comb_2d = comb_2d / (row_sum[:, None] + hc_sinkhorn_eps)
        col_sum = tl.sum(comb_2d, axis=0)
        comb_2d = comb_2d / (col_sum[None, :] + hc_sinkhorn_eps)

    # Store.
    tl.store(pre_mix_out_ptr + pid_n * HC + hc_offs, pre_mix)
    tl.store(post_mix_ptr + pid_n * HC + hc_offs, post_mix)
    tl.store(comb_mix_ptr + pid_n * HC * HC + tl.arange(0, HC * HC),
              tl.reshape(comb_2d, (HC * HC,)))


# ============================================================================
# Stage 3: apply pre_mix to residual to produce layer_input
#   layer_input[n, h] = sum_hc(pre_mix[n, hc] * residual[n, hc, h])
# ============================================================================
@triton.jit
def _mhc_pre_apply_mix_kernel(
    layer_input_ptr,   # (n, hidden) bf16
    pre_mix_ptr,       # (n, HC) fp32
    residual_ptr,      # (n, HC, hidden) bf16
    n,
    hidden,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_n >= n:
        return

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < hidden

    pre_vec = tl.load(pre_mix_ptr + pid_n * HC + tl.arange(0, HC)).to(tl.float32)  # (HC,)

    # Load residual[n, :, h_offs] -> (HC, BLOCK_H)
    hc_offs = tl.arange(0, HC)[:, None]
    res_offs = pid_n * HC * hidden + hc_offs * hidden + h_offs[None, :]
    res_mat = tl.load(residual_ptr + res_offs, mask=h_mask[None, :], other=0.0).to(tl.float32)  # (HC, BLOCK_H)

    # sum_hc(pre[hc] * residual[hc, :])
    out_vec = tl.sum(pre_vec[:, None] * res_mat, axis=0)  # (BLOCK_H,)
    tl.store(layer_input_ptr + pid_n * hidden + h_offs,
              out_vec.to(tl.bfloat16), mask=h_mask)


# ============================================================================
# Wrapper
# ============================================================================
def mhc_pre_triton(
    residual: torch.Tensor,    # (..., hc, hidden) bf16
    fn: torch.Tensor,          # (hc_mult3, hc * hidden) fp32
    hc_scale: torch.Tensor,    # (3,) fp32
    hc_base: torch.Tensor,     # (hc_mult3,) fp32
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    """Triton port of mhc_pre. Returns (post_mix, comb_mix, layer_input) or
    None if the shape is not supported (caller falls back to TileLang).
    """
    if residual.dtype != torch.bfloat16 or fn.dtype != torch.float32:
        return None
    if hc_scale.dtype != torch.float32 or hc_base.dtype != torch.float32:
        return None

    outer_shape = residual.shape[:-2]
    hc = residual.shape[-2]
    hidden = residual.shape[-1]
    if hc not in (2, 4, 8) or hidden % 32 != 0:
        return None
    hc_mult3 = hc * (2 + hc)
    hc_hidden = hc * hidden
    if hc_mult3 > 32 or fn.shape != (hc_mult3, hc_hidden):
        return None
    if not residual.is_contiguous() or not fn.is_contiguous():
        return None

    residual_flat = residual.view(-1, hc, hidden)
    n = residual_flat.shape[0]
    device = residual.device

    # Pad hc_mult3 to next pow2 for Triton tile shapes.
    hc_mult3_pad = 32  # fits up to hc=4 (hc_mult3=24); hc=8 (80) would need 128
    if hc_mult3 > hc_mult3_pad:
        return None

    out_partial_reduced = torch.empty(n, hc_mult3_pad, dtype=torch.float32, device=device)
    sqrsum = torch.empty(n, dtype=torch.float32, device=device)

    # Stage 1 (split-K + tl.dot): pick split_k to fill ~256 CUs assuming
    # n CTAs per split. SPLIT_SIZE must divide hc_hidden cleanly.
    # For hc_hidden=8192: SPLIT_K=32 → SPLIT_SIZE=256 fits BLOCK_K=64 with 4 iters.
    # Token block: 8 covers n up to 8 (production decode max_running_request=6).
    TOKEN_BLOCK = 8
    n_padded = ((n + TOKEN_BLOCK - 1) // TOKEN_BLOCK) * TOKEN_BLOCK
    n_token_blocks = n_padded // TOKEN_BLOCK

    # Choose split_k so SPLIT_SIZE is a multiple of 64 (for tl.dot K-dim).
    candidates = [32, 16, 8, 4, 2, 1]
    SPLIT_K = 1
    for c in candidates:
        if hc_hidden % c == 0 and (hc_hidden // c) % 64 == 0:
            SPLIT_K = c
            break
    SPLIT_SIZE = hc_hidden // SPLIT_K
    BLOCK_K = min(64, SPLIT_SIZE)

    out_partial = torch.empty(SPLIT_K, n_padded, hc_mult3_pad,
                               dtype=torch.float32, device=device)
    sqrsum_partial = torch.empty(SPLIT_K, n_padded,
                                  dtype=torch.float32, device=device)

    _mhc_pre_gemm_sqrsum_splitk_kernel[(SPLIT_K, n_token_blocks)](
        residual_flat, fn, out_partial, sqrsum_partial,
        n, hc_hidden, n_padded,
        HC_MULT3=hc_mult3, HC_MULT3_PAD=hc_mult3_pad,
        TOKEN_BLOCK=TOKEN_BLOCK, SPLIT_K=SPLIT_K,
        SPLIT_SIZE=SPLIT_SIZE, BLOCK_K=BLOCK_K,
        num_warps=4,
    )

    # Stage 1.5: reduce splits → (n, HC_MULT3_PAD) + (n,)
    _mhc_pre_splitk_reduce_kernel[(n,)](
        out_partial, sqrsum_partial,
        out_partial_reduced, sqrsum,
        n, n_padded,
        HC_MULT3=hc_mult3, HC_MULT3_PAD=hc_mult3_pad, SPLIT_K=SPLIT_K,
        num_warps=1,
    )
    out_partial = out_partial_reduced  # alias for stage 2 below

    # Stage 2: rms norm + sigmoid + sinkhorn.
    pre_mix = torch.empty(n, hc, dtype=torch.float32, device=device)
    post_mix = torch.empty(n, hc, dtype=torch.float32, device=device)
    comb_mix = torch.empty(n, hc * hc, dtype=torch.float32, device=device)
    _mhc_pre_big_fuse_kernel[(n,)](
        out_partial, sqrsum, hc_scale, hc_base,
        pre_mix, post_mix, comb_mix,
        n, hc_hidden,
        rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value,
        sinkhorn_repeat,
        HC=hc, HC_MULT3=hc_mult3, HC_MULT3_PAD=hc_mult3_pad,
        num_warps=1,
    )

    # Stage 3: apply pre_mix to residual.
    layer_input = torch.empty(n, hidden, dtype=torch.bfloat16, device=device)
    BLOCK_H = 256 if hidden >= 256 else hidden
    _mhc_pre_apply_mix_kernel[(n, triton.cdiv(hidden, BLOCK_H))](
        layer_input, pre_mix, residual_flat,
        n, hidden,
        HC=hc, BLOCK_H=BLOCK_H,
        num_warps=4,
    )

    return (
        post_mix.view(*outer_shape, hc, 1),
        comb_mix.view(*outer_shape, hc, hc),
        layer_input.view(*outer_shape, hidden),
    )


def mhc_pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
):
    """Reference torch implementation (no Triton, no TileLang)."""
    outer_shape = residual.shape[:-2]
    hc = residual.shape[-2]
    hidden = residual.shape[-1]
    hc_mult3 = hc * (2 + hc)
    hc_hidden = hc * hidden

    residual_flat = residual.view(-1, hc, hidden).to(torch.float32)
    fn_f = fn.to(torch.float32)

    # GEMM: x[n, hc_hidden] @ fn[hc_mult3, hc_hidden].T → (n, hc_mult3)
    x_flat = residual_flat.reshape(-1, hc_hidden)
    out = x_flat @ fn_f.T  # (n, hc_mult3)
    sqrsum = (x_flat * x_flat).sum(-1)  # (n,)

    rms = torch.rsqrt(sqrsum / hc_hidden + rms_eps)
    mixes = out * rms[:, None]  # (n, hc_mult3)

    pre_raw = mixes[:, :hc]
    post_raw = mixes[:, hc:2 * hc]
    comb_raw = mixes[:, 2 * hc:].reshape(-1, hc, hc)

    pre_mix = torch.sigmoid(pre_raw * hc_scale[0] + hc_base[:hc]) + hc_pre_eps
    post_mix = torch.sigmoid(post_raw * hc_scale[1] + hc_base[hc:2 * hc]) * hc_post_mult_value
    comb = comb_raw * hc_scale[2] + hc_base[2 * hc:].reshape(hc, hc)

    # Sinkhorn iters
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    col_sum = comb.sum(dim=-2, keepdim=True)
    comb = comb / (col_sum + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        row_sum = comb.sum(dim=-1, keepdim=True)
        comb = comb / (row_sum + hc_sinkhorn_eps)
        col_sum = comb.sum(dim=-2, keepdim=True)
        comb = comb / (col_sum + hc_sinkhorn_eps)

    # layer_input[n, h] = sum_hc(pre[n, hc] * residual[n, hc, h])
    layer_input = (pre_mix.unsqueeze(-1) * residual_flat).sum(dim=1).to(torch.bfloat16)

    return (
        post_mix.view(*outer_shape, hc, 1),
        comb.view(*outer_shape, hc, hc),
        layer_input.view(*outer_shape, hidden),
    )
