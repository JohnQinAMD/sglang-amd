"""Decode-shape-optimized hc_pre Triton kernel for DSv4-Flash.

Replaces `_hc_pre_torch_impl` (deepseek_v4.py:213) at decode shapes (M=1-8).
Per the post-fused-pool agent attribution (post-fused-pool-attribution-2026-05-01.md),
this site emits ~5 ms cpu_dispatch (4.92 ms / 602 calls) and is the top fusion
candidate after compress_decode_old shipped.

DESIGN — different from the prefill-tuned `hc_pre_fused_triton` kernel:
- Prefill kernel (M=8192): BLOCK_M=32, single grid program processes 32 rows
  at once with HC_MULT_OUT iterations inside the loop. Tuned for amortizing
  GEMM work across many rows.
- Decode kernel (M=1-8): grid=(M,), one program per row, BLOCK_M=1.
  Splits RMSNorm + GEMM + mul into 3 separate launches:
    1. Triton kernel: cast bf16→fp32, compute sum_sq, write x_flat, write rsqrt
    2. torch.matmul (cuBLAS/aiter-tuned for small M)
    3. Triton (or single torch mul) for `linear_out * rsqrt[:, None]`
  This separation lets each kernel use its best parallelization shape.

WHY the prefill kernel regressed at decode (-60 ms TPOT, tested 2026-05-01):
- BLOCK_M=32 with M=6 means 26 rows are masked off — 80%+ thread waste.
- HC_MULT_OUT=24 outer loop unrolls 24 hc_fn loads × HIDDEN/BLOCK_K = 24×64 =
  1536 tile loads per program.
- Single program → no grid parallelism → kernel becomes serial.

This kernel: M=6 → 6 grid programs, each independent.

STATUS: Phase 1 — kernel + microbench. Wire-in pending v2 graph-replay validation
+ E2E live smoke (per the M1 / B-pre / hc_pre_v1 microbench-pass-but-E2E-fail
lessons).
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Reference torch impl (the oracle). Mirrors deepseek_v4.py:213-219 byte-for-byte.
# ---------------------------------------------------------------------------
def _hc_pre_decode_reference(
    x: torch.Tensor,           # bf16 [M, HC_MULT_IN, HC_DIM]
    hc_fn: torch.Tensor,       # fp32 [HC_MULT_OUT, HIDDEN]
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference for v2 microbench correctness. Returns (x_flat, mixes)."""
    import torch.nn.functional as F
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + eps)
    mixes = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)
    return x_flat, mixes


# ---------------------------------------------------------------------------
# Triton kernel: per-row RMSNorm (fused: cast + square + mean + add + rsqrt)
# ---------------------------------------------------------------------------
@triton.jit
def _hc_pre_decode_rmsnorm_kernel(
    x_ptr,            # bf16 [M, HIDDEN]
    x_flat_out_ptr,   # fp32 [M, HIDDEN] — cast and copy of x
    rsqrt_out_ptr,    # fp32 [M] — per-row 1/sqrt(variance + eps)
    eps,
    M,
    HIDDEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per row. Reads HIDDEN/BLOCK_K tiles, accumulates sum_sq,
    writes x_flat (fp32 cast) along the way, computes rsqrt at the end.

    grid = (M,)
    """
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    sum_sq = tl.zeros((1,), dtype=tl.float32)
    row_offset = pid_m * HIDDEN

    for k_start in range(0, HIDDEN, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < HIDDEN
        x_block = tl.load(
            x_ptr + row_offset + k_offs, mask=k_mask, other=0.0
        ).to(tl.float32)
        # Persist x_flat (fp32 cast of input row)
        tl.store(
            x_flat_out_ptr + row_offset + k_offs, x_block, mask=k_mask
        )
        # Accumulate sum of squares (sum over the BLOCK_K axis returns scalar)
        sum_sq += tl.sum(x_block * x_block, axis=0)

    rsqrt = tl.rsqrt(sum_sq / HIDDEN + eps)
    tl.store(rsqrt_out_ptr + pid_m, rsqrt)


def hc_pre_decode_triton(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decode-shape drop-in for `_hc_pre_torch_impl`.

    Same signature: returns (x_flat, mixes) matching production.
    Internally: 1 Triton (RMSNorm) + 1 torch.matmul (GEMM) + 1 torch broadcast mul.
    Total 3 launches vs the original ~7 launches.

    Caller must ensure:
      x.shape == (M, HC_MULT_IN, HC_DIM)
      hc_fn.shape == (HC_MULT_OUT, HC_MULT_IN * HC_DIM)
      x.dtype is bf16, hc_fn.dtype is fp32
    """
    M_, HC_MULT_IN, HC_DIM_ = x.shape
    HIDDEN_ = HC_MULT_IN * HC_DIM_
    assert hc_fn.shape[1] == HIDDEN_, (
        f"hc_fn HIDDEN mismatch: hc_fn[1]={hc_fn.shape[1]} vs HIDDEN={HIDDEN_}"
    )

    # Step 1: RMSNorm via Triton — produces x_flat fp32 + per-row rsqrt
    x_contig = x.contiguous().view(M_, HIDDEN_)
    x_flat = torch.empty(M_, HIDDEN_, dtype=torch.float32, device=x.device)
    rsqrt = torch.empty(M_, dtype=torch.float32, device=x.device)
    BLOCK_K = min(triton.next_power_of_2(HIDDEN_), 2048)
    grid = (M_,)
    _hc_pre_decode_rmsnorm_kernel[grid](
        x_contig, x_flat, rsqrt, eps, M_,
        HIDDEN=HIDDEN_, BLOCK_K=BLOCK_K,
    )

    # Step 2: GEMM via torch (cuBLAS/aiter-tuned for the M-small regime)
    # F.linear(x, w) computes x @ w.t() — but our hc_fn is already (HC_MULT_OUT, HIDDEN),
    # so we want x_flat @ hc_fn.t() which equals F.linear(x_flat, hc_fn).
    import torch.nn.functional as F
    linear_out = F.linear(x_flat, hc_fn)  # [M, HC_MULT_OUT]

    # Step 3: Broadcast multiply by rsqrt; final unsqueeze for production shape
    mixes = (linear_out * rsqrt.unsqueeze(1)).unsqueeze(1)  # [M, 1, HC_MULT_OUT]

    return x_flat, mixes
