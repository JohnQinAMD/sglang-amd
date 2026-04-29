"""Triton replacement for `mhc_post_tilelang` (deepseek_v4 mHC post block).

Single-kernel fused linear-combine across hc_mult slots:
  x_out[n, hc_o, h] = c[n, hc_o] * d[n, h] + sum_hci( a[n, hci, hc_o] * b[n, hci, h] )

Inputs (Flash-Base FP8 typical: hc=4, hidden=2048):
  a (n, hc, hc)     fp32   = comb_res_mix
  b (n, hc, h)      bf16   = residual
  c (n, hc)         fp32   = post_layer_mix
  d (n, h)          bf16   = x
Output:
  x_out (n, hc, h)  bf16

One CTA per (token, h_block) tile; loops over hc_o serially and over h within
the tile in parallel. `a` and `c` slices are loaded once into registers and
reused across all h positions in the tile.

Replaces TileLang `mhc_post_tilelang_kernel` which was the source of the
+21 ms TPOT regression on chi2811 Flash-Base FP8.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mhc_post_kernel(
    out_ptr,        # (n, hc, h) bf16
    a_ptr,          # (n, hc, hc) fp32
    b_ptr,          # (n, hc, h) bf16
    c_ptr,          # (n, hc) fp32
    d_ptr,          # (n, h) bf16
    n,
    h,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_n >= n:
        return

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < h

    # Load c[n, :] (HC,) — fp32, once per CTA, broadcast to register fragment.
    c_offs = tl.arange(0, HC)
    c_vec = tl.load(c_ptr + pid_n * HC + c_offs).to(tl.float32)  # (HC,)

    # Load d[n, h_offs] — bf16 → fp32, once per CTA.
    d_vec = tl.load(d_ptr + pid_n * h + h_offs, mask=h_mask, other=0.0).to(tl.float32)  # (BLOCK_H,)

    # Load a[n, :, :] (HC, HC) once per CTA — fp32. We only need a[hci, hc_o]
    # for fixed n; layout is row-major (n, hc, hc).
    a_row_offs = tl.arange(0, HC)[:, None]  # hci
    a_col_offs = tl.arange(0, HC)[None, :]  # hc_o
    a_mat = tl.load(a_ptr + pid_n * HC * HC + a_row_offs * HC + a_col_offs).to(tl.float32)  # (HC, HC)

    # Load b[n, :, h_offs] -> (HC, BLOCK_H) bf16 → fp32
    b_row_offs = tl.arange(0, HC)[:, None]  # hci
    b_col_offs = h_offs[None, :]
    b_mat = tl.load(
        b_ptr + pid_n * HC * h + b_row_offs * h + b_col_offs,
        mask=h_mask[None, :], other=0.0,
    ).to(tl.float32)  # (HC, BLOCK_H)

    # For each output channel hc_o:
    #   out[hc_o, h] = c[hc_o] * d[h] + sum_hci(a[hci, hc_o] * b[hci, h])
    # Since HC is small (4), loop hc_o serially and write per channel.
    for hc_o in tl.static_range(0, HC):
        c_scalar = tl.sum(c_vec * (tl.arange(0, HC) == hc_o).to(tl.float32))
        # Compute sum_hci(a[hci, hc_o] * b[hci, :])
        a_col = tl.sum(a_mat * (tl.arange(0, HC)[None, :] == hc_o).to(tl.float32), axis=1)  # (HC,)
        # mat-vec: (HC,) * (HC, BLOCK_H) -> (BLOCK_H,)
        ab = tl.sum(a_col[:, None] * b_mat, axis=0)  # (BLOCK_H,)
        out_vals = c_scalar * d_vec + ab  # (BLOCK_H,)
        out_offs = pid_n * HC * h + hc_o * h + h_offs
        tl.store(out_ptr + out_offs, out_vals.to(tl.bfloat16), mask=h_mask)


def mhc_post_triton(
    x: torch.Tensor,                # (n, h) bf16        — sglang's `x` (post-MLP)
    residual: torch.Tensor,         # (n, hc, h) bf16    — sglang's `residual` (bilinear input)
    post_layer_mix: torch.Tensor,   # (n, hc) fp32       — sglang's `post_layer_mix`
    comb_res_mix: torch.Tensor,     # (n, hc, hc) fp32   — sglang's `comb_res_mix`
    out: torch.Tensor,              # (n, hc, h) bf16    — same shape as residual
) -> bool:
    """Triton port of mhc_post. Returns True if fired, False if shape unsupported.

    Semantics (matching sglang's mhc_post wrapper + TileLang implementation):
        out[n, hc_o, h] = post_layer_mix[n, hc_o] * x[n, h]
                        + sum_hci(comb_res_mix[n, hci, hc_o] * residual[n, hci, h])

    Note: `x.shape = (n, h)` and `residual.shape = (n, hc, h)` — out has the
    residual's shape, not x's. This matches `out = torch.empty_like(residual)`
    in sglang.srt.layers.mhc.mhc_post.
    """
    assert x.dtype == torch.bfloat16
    assert residual.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert out.dtype == torch.bfloat16

    assert residual.shape == out.shape
    n, hc, h = residual.shape
    assert x.shape == (n, h)
    assert post_layer_mix.shape == (n, hc)
    assert comb_res_mix.shape == (n, hc, hc)

    if not (x.is_contiguous() and residual.is_contiguous()
            and post_layer_mix.is_contiguous() and comb_res_mix.is_contiguous()
            and out.is_contiguous()):
        return False
    if hc not in (2, 4, 8):
        return False
    if h % 64 != 0:
        return False

    # Tile along h. 256-elem block fits comfortably in registers (2 KB bf16 +
    # 4 KB fp32 working set per CTA), allowing high CU occupancy.
    BLOCK_H = 256 if h >= 256 else h
    grid = (n, triton.cdiv(h, BLOCK_H))
    # Kernel signature: (out, a=comb, b=residual, c=post, d=x, n, h, ...)
    _mhc_post_kernel[grid](
        out, comb_res_mix, residual, post_layer_mix, x,
        n, h,
        HC=hc, BLOCK_H=BLOCK_H,
        num_warps=4,
    )
    return True


def mhc_post_torch(
    x: torch.Tensor,                # (n, h)
    residual: torch.Tensor,         # (n, hc, h)
    post_layer_mix: torch.Tensor,   # (n, hc)
    comb_res_mix: torch.Tensor,     # (n, hc, hc)
    out: torch.Tensor,              # (n, hc, h)
) -> None:
    """Reference torch implementation. Sglang convention.

    Computes: out[n, hc_o, h] = post_layer_mix[n, hc_o] * x[n, h]
                              + sum_hci(comb_res_mix[n, hci, hc_o] * residual[n, hci, h])
    """
    n, hc, h = residual.shape
    # post_layer_mix * x broadcast: (n, hc, 1) * (n, 1, h) -> (n, hc, h)
    base = post_layer_mix.unsqueeze(-1).to(torch.float32) * x.unsqueeze(1).to(torch.float32)
    # einsum "n hci hco, n hci h -> n hco h"
    bilin = torch.einsum("nij,nih->njh", comb_res_mix.to(torch.float32),
                          residual.to(torch.float32))
    result = (base + bilin).to(torch.bfloat16)
    out.copy_(result)
