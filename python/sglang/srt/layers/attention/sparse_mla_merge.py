"""Online-softmax merge of two sparse-attention split outputs.

Used by ``debug_flash_mla_adapter.flash_mla_with_kvcache_torch`` when
``extra_k_cache != None``. Each split is computed independently by
``ck_sparse_mla_decode_fp8_v32`` (no sink applied internally), and this
kernel merges them via the standard FlashAttention-2 two-split formula:

    lm = max(lse_a, lse_b)
    wa = exp(lse_a - lm)
    wb = exp(lse_b - lm)
    sw = wa + wb
    out = (wa * out_a + wb * out_b) / sw
    lse = lm + log(sw)

Lonely-Q rows (sw == 0) come out as out=0, lse=-inf — matches the BF16
fallback semantics.

Phase B (2026-04-28): kernel now accepts arbitrary strides for the (q, h)
axes to skip the 4× ``.contiguous()`` calls in the dispatch wrapper. The V
axis MUST stay stride-1 (kernel does flat ``+ offs`` loads). A constexpr
``IS_CONTIGUOUS`` switch keeps the original SASS for the all-contiguous
fast path. The wrapper also accepts ``out=`` / ``lse=`` kwargs for
caller-allocated destinations (no internal ``empty_like``).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _merge_kernel(
    out_ptr, out_a_ptr, out_b_ptr,
    lse_ptr, lse_a_ptr, lse_b_ptr,
    stride_a_q, stride_a_h,
    stride_b_q, stride_b_h,
    stride_out_q, stride_out_h,
    stride_lse_a_q, stride_lse_a_h,
    stride_lse_b_q, stride_lse_b_h,
    stride_lse_out_q, stride_lse_out_h,
    H: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    IS_CONTIGUOUS: tl.constexpr,
):
    pid = tl.program_id(0)
    if IS_CONTIGUOUS:
        # Original flat-indexing path — preserves the SASS the kernel had
        # before Phase B for the all-contiguous case (no perf regression).
        base_a = pid * V
        base_b = pid * V
        base_o = pid * V
        lse_a_off = pid
        lse_b_off = pid
        lse_o_off = pid
    else:
        q_idx = pid // H
        h_idx = pid % H
        base_a = q_idx * stride_a_q + h_idx * stride_a_h
        base_b = q_idx * stride_b_q + h_idx * stride_b_h
        base_o = q_idx * stride_out_q + h_idx * stride_out_h
        lse_a_off = q_idx * stride_lse_a_q + h_idx * stride_lse_a_h
        lse_b_off = q_idx * stride_lse_b_q + h_idx * stride_lse_b_h
        lse_o_off = q_idx * stride_lse_out_q + h_idx * stride_lse_out_h

    la = tl.load(lse_a_ptr + lse_a_off).to(tl.float32)
    lb = tl.load(lse_b_ptr + lse_b_off).to(tl.float32)
    lm = tl.maximum(la, lb)

    da = la - lm
    db = lb - lm
    # NaN guard: la-lm == NaN iff la==-inf and lm==-inf (both halves invalid).
    wa = tl.where(da == da, tl.exp(da), 0.0)
    wb = tl.where(db == db, tl.exp(db), 0.0)
    sw = wa + wb
    all_invalid = sw == 0.0
    sw_safe = tl.where(all_invalid, 1.0, sw)

    new_lse = tl.where(all_invalid, float("-inf"), lm + tl.log(sw))
    tl.store(lse_ptr + lse_o_off, new_lse)

    for v_start in tl.static_range(0, V, BLOCK_V):
        offs = v_start + tl.arange(0, BLOCK_V)
        mask = offs < V
        oa = tl.load(out_a_ptr + base_a + offs, mask=mask, other=0.0).to(tl.float32)
        ob = tl.load(out_b_ptr + base_b + offs, mask=mask, other=0.0).to(tl.float32)
        merged = (wa * oa + wb * ob) / sw_safe
        merged = tl.where(all_invalid, 0.0, merged)
        tl.store(out_ptr + base_o + offs, merged.to(tl.bfloat16), mask=mask)


def merge_two_sparse_attn_outputs(
    out_a: torch.Tensor,           # [total_q, H, V] bf16
    lse_a: torch.Tensor,           # [total_q, H]    fp32
    out_b: torch.Tensor,           # [total_q, H, V] bf16
    lse_b: torch.Tensor,           # [total_q, H]    fp32
    *,
    out: torch.Tensor | None = None,   # caller-allocated [total_q, H, V] bf16
    lse: torch.Tensor | None = None,   # caller-allocated [total_q, H]    fp32
):
    """Online-softmax merge of two parallel sparse-attention outputs.

    Inputs may be non-contiguous on the (q, h) axes; the V axis (innermost
    dim of ``out_a/out_b``) MUST be stride-1 — asserted below. Outputs are
    written into caller-supplied ``out``/``lse`` if provided (saves the
    per-call empty_like allocation that churns the caching allocator).

    Returns (out [total_q, H, V] bf16, lse [total_q, H] fp32).
    """
    assert out_a.shape == out_b.shape
    assert lse_a.shape == lse_b.shape
    assert out_a.dtype == torch.bfloat16 and out_b.dtype == torch.bfloat16
    assert lse_a.dtype == torch.float32 and lse_b.dtype == torch.float32

    total_q, H, V = out_a.shape
    assert lse_a.shape == (total_q, H)
    # V axis must be stride-1 in all input tensors (kernel does flat
    # `out_a_ptr + base + offs` loads); other axes may be strided.
    assert out_a.stride(-1) == 1, f"out_a V-stride must be 1, got {out_a.stride(-1)}"
    assert out_b.stride(-1) == 1, f"out_b V-stride must be 1, got {out_b.stride(-1)}"

    if out is None:
        out = torch.empty((total_q, H, V), dtype=torch.bfloat16, device=out_a.device)
    else:
        assert out.shape == (total_q, H, V) and out.dtype == torch.bfloat16
        assert out.stride(-1) == 1, f"out V-stride must be 1, got {out.stride(-1)}"
    if lse is None:
        lse = torch.empty((total_q, H), dtype=torch.float32, device=lse_a.device)
    else:
        assert lse.shape == (total_q, H) and lse.dtype == torch.float32

    is_contig = (
        out_a.is_contiguous() and out_b.is_contiguous() and out.is_contiguous()
        and lse_a.is_contiguous() and lse_b.is_contiguous() and lse.is_contiguous()
    )

    BLOCK_V = min(512, V)
    grid = (total_q * H,)
    _merge_kernel[grid](
        out, out_a, out_b,
        lse, lse_a, lse_b,
        out_a.stride(0), out_a.stride(1),
        out_b.stride(0), out_b.stride(1),
        out.stride(0),   out.stride(1),
        lse_a.stride(0), lse_a.stride(1),
        lse_b.stride(0), lse_b.stride(1),
        lse.stride(0),   lse.stride(1),
        H=H, V=V, BLOCK_V=BLOCK_V,
        IS_CONTIGUOUS=is_contig,
    )
    return out, lse
