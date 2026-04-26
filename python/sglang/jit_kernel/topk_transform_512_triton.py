"""Triton port of TopK512Kernel from csrc/deepseek_v4/topk.cuh.

For each batch row b:
  - Take top-K=512 positions from scores[b, :seq_lens[b]] by descending score.
  - Convert each top-K position p into a page-table-mapped index:
        page_id = p >> page_bits
        offset  = p &  page_mask
        out[i]  = (page_tables[b, page_id] << page_bits) | offset
  - For positions beyond seq_len (when seq_len < K) emit -1 in both
    `out_page_indices` and `out_raw_indices`.

Algorithm: pack (descending-monotonic score key, position) into a single
uint64 (key in high 32 bits, position in low 32 bits) and `tl.sort` once.
The first K entries of the sorted block are the top-K. Invalid positions
get a sentinel so they sort to the bottom.

Runs natively on AMDGCN (Triton 3.x supports gfx950) — no `nvcc`/CUDA_HOME
dependency. Fixes the gap left by the JIT-CUDA path being NVIDIA-only.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _topk_transform_512_kernel(
    scores_ptr,           # [B, max_seq_len] f32
    seq_lens_ptr,         # [B] i32
    page_tables_ptr,      # [B, num_pages] i32
    out_page_indices_ptr, # [B, K] i32
    out_raw_indices_ptr,  # [B, K] i32 or null
    stride_scores_b,
    stride_pt_b,
    K: tl.constexpr,
    PAGE_BITS: tl.constexpr,
    PAGE_MASK: tl.constexpr,
    BLOCK_L: tl.constexpr,
    HAS_RAW_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + pid).to(tl.int32)

    row_offs = tl.arange(0, BLOCK_L)
    valid_mask = row_offs < seq_len

    # Load full row; padding gets -inf.
    scores = tl.load(
        scores_ptr + pid * stride_scores_b + row_offs,
        mask=valid_mask,
        other=-float("inf"),
    )

    # Build descending-monotonic uint32 key from float32 bits, mirroring
    # convert_to_uint32 in csrc/deepseek_v4/topk.cuh:
    #   if (sign): ~bits          (negative floats: bigger magnitude → smaller key)
    #   else:      bits | 0x80000000   (positives: shift above all negatives)
    # Then larger float → larger key. We want descending sort, so negate
    # via (MAX - key) so largest score → smallest packed value.
    # All bit math in uint32 to avoid int32-literal-overflow on 0x80000000.
    score_bits_u32 = scores.to(tl.uint32, bitcast=True)
    sign_set = score_bits_u32 >> 31  # 0 or 1
    high_bit = tl.full((), 0x80000000, dtype=tl.uint32)
    max_u32 = tl.full((), 0xFFFFFFFF, dtype=tl.uint32)
    key_pos = score_bits_u32 | high_bit
    key_neg = max_u32 - score_bits_u32   # equivalent to ~score_bits_u32 in uint32
    key = tl.where(sign_set == 1, key_neg, key_pos)
    inv_key = max_u32 - key
    # Sentinel for invalid positions: max possible key → sorts last.
    inv_key = tl.where(valid_mask, inv_key, max_u32)

    # Pack into uint64: [inv_key (32 bits) | row_offs (32 bits)]
    packed = (inv_key.to(tl.uint64) << 32) | row_offs.to(tl.uint64)

    # Sort ascending → values descending by score.
    sorted_packed = tl.sort(packed)

    # First K entries are the top-K (or fewer if seq_len < K, with sentinel suffix).
    out_mask = row_offs < K

    raw_pos = (sorted_packed & 0xFFFFFFFF).to(tl.int32)
    sorted_inv_key = (sorted_packed >> 32).to(tl.uint32)
    is_real = sorted_inv_key < 0xFFFFFFFF  # not the sentinel
    raw_pos = tl.where(is_real, raw_pos, -1)

    # Convert raw_pos → page_indices.
    page_id = raw_pos >> PAGE_BITS
    page_id_clamped = tl.maximum(page_id, 0)
    physical_page = tl.load(
        page_tables_ptr + pid * stride_pt_b + page_id_clamped,
        mask=out_mask & is_real,
        other=0,
    ).to(tl.int32)
    page_indices = (physical_page << PAGE_BITS) | (raw_pos & PAGE_MASK)
    page_indices = tl.where(is_real, page_indices, -1)

    tl.store(
        out_page_indices_ptr + pid * K + row_offs,
        page_indices,
        mask=out_mask,
    )
    if HAS_RAW_OUT:
        tl.store(
            out_raw_indices_ptr + pid * K + row_offs,
            raw_pos,
            mask=out_mask,
        )


def topk_transform_pytorch_parametric(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """torch.topk-based path, parametric in K (no hardcoded 512).

    Used as the large-L dispatch target for `topk_transform_512_triton`: the
    Triton kernel does a full `tl.sort(BLOCK_L)` per CTA, so its cost grows
    as L·log L. torch.topk grows as L·log K and wins on MI355X for
    scores.shape[1] ≳ 8192 (see bench_deepseek_v4.py). Capture-safe:
    `torch.where` instead of `.clone() + masked_fill_`.
    """
    B, L = scores.shape
    K = out_page_indices.shape[1]
    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    pos = torch.arange(L, device=scores.device).unsqueeze(0).expand(B, -1)
    valid = pos < seq_lens.unsqueeze(1)
    masked = torch.where(valid, scores, scores.new_full((), float("-inf")))
    actual_k = min(K, L)
    _, raw = torch.topk(masked, k=actual_k, dim=1, largest=True, sorted=False)
    raw = raw.to(torch.int32)
    if actual_k < K:
        pad = torch.full((B, K - actual_k), -1, dtype=torch.int32,
                         device=scores.device)
        raw = torch.cat([raw, pad], dim=1)

    bidx = torch.arange(B, device=scores.device).unsqueeze(1).expand(-1, K)
    gathered = scores[bidx.flatten(), raw.clamp(min=0).long().flatten()].view(B, K)
    valid_topk = gathered != float("-inf")

    page_idx_clamped = torch.clamp(raw >> page_bits, min=0).long()
    phys = torch.gather(page_tables, dim=1, index=page_idx_clamped)
    out = ((phys << page_bits) | (raw & page_mask)).to(torch.int32)
    out_page_indices.copy_(torch.where(valid_topk, out, out.new_full((), -1)))
    if out_raw_indices is not None:
        out_raw_indices.copy_(torch.where(valid_topk, raw,
                                          raw.new_full((), -1)))


def topk_transform_512_triton(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Triton equivalent of the CUDA TopK512Kernel.

    Despite the "512" in the name, K is taken from out_page_indices.shape[1]
    so this works for both Flash (K=512) and Pro (K=1024) c4_sparse_topk values.

    Shapes:
      scores:           [B, max_seq_len] f32
      seq_lens:         [B] i32 — actual valid length per batch row
      page_tables:      [B, num_pages] i32
      out_page_indices: [B, K] i32
      out_raw_indices:  [B, K] i32 or None
    """
    import os

    # Apply the indexer cap before the kernel reads scores.shape[1].
    # Without this, BLOCK_L = next_pow2(scores.shape[1]) can exceed Triton's
    # TRITON_MAX_TENSOR_NUMEL (1M) under cuda graph capture (capture pre-allocates
    # logits to the worst-case context length, e.g. 262144). The cap is also
    # enforced upstream in fp8_paged_mqa_logits, so seq_lens are always <= cap.
    cap = int(os.environ.get("SGLANG_INDEXER_MAX_SEQ_LEN", "0"))
    if cap > 0 and scores.shape[1] > cap:
        scores = scores[:, :cap]

    # Large-L dispatch: the Triton kernel sorts BLOCK_L = next_pow2(L) per CTA,
    # which scales as L·log L. For L ≳ 8192 on MI355X this is slower than
    # torch.topk (which scales as L·log K). Dispatch on `scores.shape[1]`,
    # not seq_lens.max() — the latter would force a GPU→CPU sync and break
    # cuda graph capture; under capture, scores.shape[1] is the worst-case
    # buffer size and is what the Triton kernel actually processes anyway.
    triton_max_l = int(os.environ.get("SGLANG_TOPK_TRITON_MAX_L", "8192"))
    if scores.shape[1] > triton_max_l:
        topk_transform_pytorch_parametric(
            scores, seq_lens, page_tables, out_page_indices, page_size,
            out_raw_indices,
        )
        return

    B, max_seq_len = scores.shape
    K = out_page_indices.shape[1]
    assert page_size > 0 and (page_size & (page_size - 1)) == 0, "page_size must be power of 2"
    page_bits = (page_size - 1).bit_length()
    page_mask = page_size - 1

    BLOCK_L = triton.next_power_of_2(max(max_seq_len, K))

    has_raw = out_raw_indices is not None
    raw_ptr = out_raw_indices if has_raw else out_page_indices  # dummy ptr; unused when HAS_RAW_OUT=False

    grid = (B,)
    _topk_transform_512_kernel[grid](
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        raw_ptr,
        scores.stride(0),
        page_tables.stride(0),
        K=K,
        PAGE_BITS=page_bits,
        PAGE_MASK=page_mask,
        BLOCK_L=BLOCK_L,
        HAS_RAW_OUT=has_raw,
    )
