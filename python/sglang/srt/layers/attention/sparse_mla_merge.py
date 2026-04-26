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
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _merge_kernel(
    out_ptr, out_a_ptr, out_b_ptr,
    lse_ptr, lse_a_ptr, lse_b_ptr,
    H: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    qh_id = pid

    la = tl.load(lse_a_ptr + qh_id).to(tl.float32)
    lb = tl.load(lse_b_ptr + qh_id).to(tl.float32)
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
    tl.store(lse_ptr + qh_id, new_lse)

    base = qh_id * V
    for v_start in tl.static_range(0, V, BLOCK_V):
        offs = v_start + tl.arange(0, BLOCK_V)
        mask = offs < V
        oa = tl.load(out_a_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        ob = tl.load(out_b_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        merged = (wa * oa + wb * ob) / sw_safe
        merged = tl.where(all_invalid, 0.0, merged)
        tl.store(out_ptr + base + offs, merged.to(tl.bfloat16), mask=mask)


def merge_two_sparse_attn_outputs(
    out_a: torch.Tensor,           # [total_q, H, V] bf16
    lse_a: torch.Tensor,           # [total_q, H]    fp32
    out_b: torch.Tensor,           # [total_q, H, V] bf16
    lse_b: torch.Tensor,           # [total_q, H]    fp32
):
    """Online-softmax merge of two parallel sparse-attention outputs.

    Returns (out_merged [total_q, H, V] bf16, lse_merged [total_q, H] fp32).
    """
    assert out_a.shape == out_b.shape
    assert lse_a.shape == lse_b.shape
    assert out_a.dtype == torch.bfloat16 and out_b.dtype == torch.bfloat16
    assert lse_a.dtype == torch.float32 and lse_b.dtype == torch.float32

    total_q, H, V = out_a.shape
    assert lse_a.shape == (total_q, H)

    out_a_c = out_a if out_a.is_contiguous() else out_a.contiguous()
    out_b_c = out_b if out_b.is_contiguous() else out_b.contiguous()
    lse_a_c = lse_a if lse_a.is_contiguous() else lse_a.contiguous()
    lse_b_c = lse_b if lse_b.is_contiguous() else lse_b.contiguous()

    out = torch.empty_like(out_a_c)
    lse = torch.empty_like(lse_a_c)

    BLOCK_V = min(512, V)
    grid = (total_q * H,)
    _merge_kernel[grid](
        out, out_a_c, out_b_c,
        lse, lse_a_c, lse_b_c,
        H=H, V=V, BLOCK_V=BLOCK_V,
    )
    return out, lse
