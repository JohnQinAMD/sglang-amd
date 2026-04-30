"""Decode-body megakernel — Block A (MQA prologue) — Phase 2 v2 (scale-applying).

Status: v2 IMPLEMENTED. Replaces the production 4-op chain
    aiter::add_rmsnorm_quant_kernel
    aiter::dynamic_per_group_scaled_quant_kernel
    aiter::gemm_a8w8_blockscale  (wq_a)
    aiter::gemm_a8w8_blockscale  (wkv_a)
with ONE Triton kernel that loads `hidden[bs, HIDDEN]` once, RMSNorms in fp32,
per-1x128 quantizes to fp8, then runs the two scale-applying block-scale GEMMs
sharing the same in-flight fp8/x_scale pair.

Phase 2 semantic: matches `aiter::gemm_a8w8_blockscale` exactly:
    acc += sum_g (x_scale[:, g] * w_scale[n_group, g]) * (fp8_x_g @ fp8_w_g.T)
where each K-group is GROUP_SIZE=128 wide. x_scale is computed inside this
kernel (per-row, per-K-group, fp32) from RMSNorm output. w_scale is loaded
from the per-128x128 fp32 scale tensor passed in.

Shape histogram (DSv4 Flash-Base FP8 decode, TP=4):
  BS in {1,2,4,6,8}; HIDDEN=8192; Q_LORA_RANK=1536; HEAD_DIM=64; GROUP_SIZE=128

Grid: (num_bs_tiles, num_n_tiles_q + num_n_tiles_kv)
  Each program owns one (bs_tile, output_n_tile) and writes either q_lora or kv.
  Output discrimination is by pid_n: pid_n < NQ → q_lora, else kv.
  BLOCK_N == GROUP_SIZE == 128 ⇒ each program owns exactly one w_scale row.
"""

from __future__ import annotations

from typing import Tuple
import torch
import triton
import triton.language as tl


# --------------------------------------------------------------------------
# Phase 2 autotune sweep (small). BLOCK_HIDDEN is fixed at GROUP_SIZE=128 because
# the per-K-group scale-applying GEMM steps the K-loop at 128. Sweep num_warps
# and waves_per_eu (kpack forced 1 on AMD per feedback_amd_triton_gfx950_knobs).
# --------------------------------------------------------------------------
_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=4, num_stages=2),
    triton.Config({}, num_warps=2, num_stages=2),
    triton.Config({}, num_warps=8, num_stages=2),
    triton.Config({}, num_warps=4, num_stages=3),
]


@triton.autotune(
    configs=_AUTOTUNE_CONFIGS,
    key=["BLOCK_BS", "Q_LORA_RANK", "HEAD_DIM", "HIDDEN"],
)
@triton.jit
def _mqa_prologue_kernel(
    # Inputs
    hidden_ptr,               # bf16 (bs, hidden)
    rmsnorm_w_ptr,            # fp32 (hidden,)
    wq_a_ptr,                 # fp8 (q_lora_rank, hidden) row-major
    wkv_a_ptr,                # fp8 (head_dim, hidden) row-major
    wq_a_scale_ptr,           # fp32 (cdiv(q_lora_rank,128), hidden//128)
    wkv_a_scale_ptr,          # fp32 (cdiv(head_dim,128), hidden//128)
    # Outputs
    q_lora_ptr,               # bf16 (bs, q_lora_rank)
    kv_ptr,                   # bf16 (bs, head_dim)
    # Strides (element-strides)
    s_hidden_b, s_hidden_h,
    s_wqa_q, s_wqa_h,
    s_wkva_k, s_wkva_h,
    s_wqas_n, s_wqas_k,
    s_wkvas_n, s_wkvas_k,
    s_qlora_b, s_qlora_q,
    s_kv_b, s_kv_k,
    # Scalars
    bs,
    Q_LORA_RANK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NQ: tl.constexpr,                   # num_n_tiles for q_lora = cdiv(Q_LORA_RANK, BLOCK_N)
    HIDDEN: tl.constexpr,
    GROUP_SIZE: tl.constexpr,           # 128
    EPS: tl.constexpr,
    BLOCK_BS: tl.constexpr,             # next-pow-2 of bs (≤ 8) padded to MFMA-min 16
    BLOCK_N: tl.constexpr,              # unified output tile size (N dim) == GROUP_SIZE
):
    BLOCK_HIDDEN: tl.constexpr = 256    # Pass-1 sumsq tile size; not used in Pass-2.

    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    is_q = pid_n < NQ

    offs_bs = pid_b * BLOCK_BS + tl.arange(0, BLOCK_BS)
    bs_mask = offs_bs < bs

    # ---- Pass 1: compute rstd over hidden axis -------------------------------
    sumsq = tl.zeros((BLOCK_BS,), dtype=tl.float32)
    for k0 in range(0, HIDDEN, BLOCK_HIDDEN):
        offs_h = k0 + tl.arange(0, BLOCK_HIDDEN)
        h_ptrs = hidden_ptr + offs_bs[:, None] * s_hidden_b + offs_h[None, :] * s_hidden_h
        h = tl.load(h_ptrs, mask=bs_mask[:, None], other=0.0).to(tl.float32)
        sumsq += tl.sum(h * h, axis=1)

    mean_sq = sumsq / HIDDEN
    rstd = 1.0 / tl.sqrt(mean_sq + EPS)  # (BLOCK_BS,)

    # ---- Pass 2: streaming RMSNorm + per-1x128 quant + scale-applying GEMM ---
    # Per-K-group dot: tile is BLOCK_HIDDEN wide = NG groups; each group has its
    # own (x_scale, w_scale) pair. We do NG sub-dots per K-tile and fold the
    # scales into the per-group acc.
    if is_q:
        n_start = pid_n * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < Q_LORA_RANK
        # w_scale row index (one row per program; BLOCK_N == GROUP_SIZE).
        w_scale_n_idx = pid_n
        accumulator = tl.zeros((BLOCK_BS, BLOCK_N), dtype=tl.float32)

        # Per-group K-loop: K stride = GROUP_SIZE. Each iter loads ONE 128-wide
        # group, applies its (x_scale, w_scale), and accumulates.
        for k0 in tl.range(0, HIDDEN, GROUP_SIZE, num_stages=2):
            offs_h = k0 + tl.arange(0, GROUP_SIZE)
            h_ptrs = hidden_ptr + offs_bs[:, None] * s_hidden_b + offs_h[None, :] * s_hidden_h
            h = tl.load(h_ptrs, mask=bs_mask[:, None], other=0.0).to(tl.float32)
            w_norm = tl.load(rmsnorm_w_ptr + offs_h).to(tl.float32)
            normed = h * rstd[:, None] * w_norm[None, :]                 # (BLOCK_BS, GROUP_SIZE) fp32
            amax = tl.max(tl.abs(normed), axis=1)                        # (BLOCK_BS,)
            x_scale = amax / 448.0
            x_scale_safe = tl.where(x_scale == 0.0, 1.0, x_scale)        # (BLOCK_BS,)
            fx = (normed / x_scale_safe[:, None]).to(tl.float8e4nv)      # (BLOCK_BS, GROUP_SIZE) fp8

            # Load fp8 weight tile for THIS group. w_scale is one fp32 scalar.
            w_ptrs = (wq_a_ptr
                      + offs_n[:, None] * s_wqa_q
                      + offs_h[None, :] * s_wqa_h)
            fw = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)        # (BLOCK_N, GROUP_SIZE) fp8

            ws = tl.load(
                wq_a_scale_ptr + w_scale_n_idx * s_wqas_n + (k0 // GROUP_SIZE) * s_wqas_k
            )  # scalar fp32

            part = tl.dot(fx, tl.trans(fw))                              # fp32 (BLOCK_BS, BLOCK_N)
            accumulator += part * (x_scale_safe[:, None] * ws)

        out = accumulator.to(tl.bfloat16)
        out_ptrs = q_lora_ptr + offs_bs[:, None] * s_qlora_b + offs_n[None, :] * s_qlora_q
        tl.store(out_ptrs, out, mask=bs_mask[:, None] & n_mask[None, :])
    else:
        n_start = (pid_n - NQ) * BLOCK_N
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < HEAD_DIM
        w_scale_n_idx = pid_n - NQ
        accumulator = tl.zeros((BLOCK_BS, BLOCK_N), dtype=tl.float32)

        for k0 in tl.range(0, HIDDEN, GROUP_SIZE, num_stages=2):
            offs_h = k0 + tl.arange(0, GROUP_SIZE)
            h_ptrs = hidden_ptr + offs_bs[:, None] * s_hidden_b + offs_h[None, :] * s_hidden_h
            h = tl.load(h_ptrs, mask=bs_mask[:, None], other=0.0).to(tl.float32)
            w_norm = tl.load(rmsnorm_w_ptr + offs_h).to(tl.float32)
            normed = h * rstd[:, None] * w_norm[None, :]
            amax = tl.max(tl.abs(normed), axis=1)
            x_scale = amax / 448.0
            x_scale_safe = tl.where(x_scale == 0.0, 1.0, x_scale)
            fx = (normed / x_scale_safe[:, None]).to(tl.float8e4nv)

            w_ptrs = (wkv_a_ptr
                      + offs_n[:, None] * s_wkva_k
                      + offs_h[None, :] * s_wkva_h)
            fw = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)

            ws = tl.load(
                wkv_a_scale_ptr + w_scale_n_idx * s_wkvas_n + (k0 // GROUP_SIZE) * s_wkvas_k
            )

            part = tl.dot(fx, tl.trans(fw))
            accumulator += part * (x_scale_safe[:, None] * ws)

        out = accumulator.to(tl.bfloat16)
        out_ptrs = kv_ptr + offs_bs[:, None] * s_kv_b + offs_n[None, :] * s_kv_k
        tl.store(out_ptrs, out, mask=bs_mask[:, None] & n_mask[None, :])


def mqa_prologue_megakernel(
    hidden: torch.Tensor,
    rmsnorm_w: torch.Tensor,
    wq_a_fp8: torch.Tensor,
    wq_a_scale: torch.Tensor,            # fp32 (cdiv(q_lora_rank,128), hidden//128)
    wkv_a_fp8: torch.Tensor,
    wkv_a_scale: torch.Tensor,           # fp32 (cdiv(head_dim,128), hidden//128)
    eps: float,
    *,
    q_lora_out: torch.Tensor | None = None,
    kv_out: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor] | None:
    """Block A v2 entrypoint. Returns (q_lora bf16, kv bf16) or None on shape gate fail."""
    bs, hidden_dim = hidden.shape
    q_lora_rank = wq_a_fp8.shape[0]
    head_dim = wkv_a_fp8.shape[0]

    # Shape gates -------------------------------------------------------------
    if hidden_dim & (hidden_dim - 1) != 0:
        return None
    if hidden_dim < 128 or hidden_dim > 16384:
        return None
    if bs > 32:
        return None
    GROUP_SIZE = 128
    if hidden_dim % GROUP_SIZE != 0:
        return None
    if rmsnorm_w.dtype != torch.float32:
        return None
    if hidden.dtype != torch.bfloat16:
        return None
    if wq_a_fp8.dtype != torch.float8_e4m3fn or wkv_a_fp8.dtype != torch.float8_e4m3fn:
        return None
    if wq_a_scale.dtype != torch.float32 or wkv_a_scale.dtype != torch.float32:
        return None
    # BLOCK_N == GROUP_SIZE pre-condition for one-w_scale-row-per-program.
    BLOCK_N = GROUP_SIZE
    # Q_LORA_RANK must be divisible by BLOCK_N for the simple n-tiling. (1536 / 128 = 12.)
    if q_lora_rank % BLOCK_N != 0:
        return None

    if q_lora_out is None:
        q_lora_out = torch.empty(bs, q_lora_rank, dtype=torch.bfloat16, device=hidden.device)
    if kv_out is None:
        kv_out = torch.empty(bs, head_dim, dtype=torch.bfloat16, device=hidden.device)

    BLOCK_BS = max(16, triton.next_power_of_2(bs))   # MFMA needs M>=16; pad bs.

    NQ = triton.cdiv(q_lora_rank, BLOCK_N)
    NKV = triton.cdiv(head_dim, BLOCK_N)

    grid = (
        triton.cdiv(bs, BLOCK_BS),
        NQ + NKV,
    )

    _mqa_prologue_kernel[grid](
        hidden, rmsnorm_w,
        wq_a_fp8, wkv_a_fp8,
        wq_a_scale, wkv_a_scale,
        q_lora_out, kv_out,
        hidden.stride(0), hidden.stride(1),
        wq_a_fp8.stride(0), wq_a_fp8.stride(1),
        wkv_a_fp8.stride(0), wkv_a_fp8.stride(1),
        wq_a_scale.stride(0), wq_a_scale.stride(1),
        wkv_a_scale.stride(0), wkv_a_scale.stride(1),
        q_lora_out.stride(0), q_lora_out.stride(1),
        kv_out.stride(0), kv_out.stride(1),
        bs,
        Q_LORA_RANK=q_lora_rank,
        HEAD_DIM=head_dim,
        NQ=NQ,
        HIDDEN=hidden_dim,
        GROUP_SIZE=GROUP_SIZE,
        EPS=eps,
        BLOCK_BS=BLOCK_BS,
        BLOCK_N=BLOCK_N,
        # BLOCK_HIDDEN, num_warps, num_stages selected by autotune.
    )

    return q_lora_out, kv_out
