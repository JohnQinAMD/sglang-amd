"""M3 indexer megakernel — single-launch Triton fusion of topk_transform_512
+ invalid_mask for the DSv4 indexer chain.

Background
----------
The DSv4 plan's M3 lever specifies fusion of:
    mhc_pre_chain + topk_transform + invalid_mask + mhc_post

After mapping the actual dataflow on chi2774 Flash-Base FP8 (see report):

  mhc_pre  → (post_mix, comb_mix, layer_input)            [residual side-output]
       │       (used much later by mhc_post; 60+ kernels in between)
       │
       ▼
  compressor + GEMMs + indexer compute
       │
       ▼
  fp8_paged_mqa_logits → logits (B, max_seq_len)
       │
       ▼
  topk_transform_512  → out_page_indices, out_raw_indices
       │                                  ┊ raw_indices flows
       ▼                                  ▼
  CK V32 sparse-MLA  ←─────────  invalid_mask  (debug_flash_mla_adapter._get_invalid_mask)
       │
       ▼
  ... attention output ...
       │
       ▼
  mhc_post  ← post_mix, comb_mix from mhc_pre

`mhc_pre` and `mhc_post` are NOT dataflow-adjacent to the indexer ops; they
sandwich the entire attention layer with ~60 unrelated kernels in between.
A single-Triton-launch megakernel of all 4 is structurally impossible.

`topk_transform_512` and `invalid_mask` ARE dataflow-adjacent
(topk produces raw_indices → invalid_mask consumes raw_indices) and fire
back-to-back per layer per decode step. That is the largest sub-fusion that
fits in ONE Triton launch — and the one this megakernel implements.

What this kernel does (single launch, one CTA per batch row)
------------------------------------------------------------
For each `pid in [0, B)`:
  1. Bit-pack score → (key, position) uint64; tl.sort once → top-K raw indices.
  2. Convert each top-K raw position → page-table-mapped output index.
  3. Compute invalid_mask = (raw_idx < 0) | (col >= topk_length[b]) at the
     same time, write boolean mask alongside indices.

Inputs (decode):
  scores:           [B, max_seq_len] f32       (from fp8_paged_mqa_logits)
  seq_lens:         [B] i32
  page_tables:      [B, num_pages] i32
  topk_length:      [B] i32 or null            (drives invalid_mask)

Outputs:
  out_page_indices: [B, K] i32                 (consumed by CK V32 sparse-MLA)
  out_raw_indices:  [B, K] i32                 (consumed by hisparse coordinator
                                                AND the invalid_mask path
                                                downstream — kept for compat)
  out_invalid_mask: [B*S_q, K] bool / int8     (consumed by CK V32 sparse-MLA;
                                                S_q=1 at decode → B*S_q == B)

Launch reduction: replaces `topk_transform_512_triton (1) +
invalid_mask_triton (1)` = 2 launches with 1.
At decode-time bs=6, that's −1 launch × 60 layers = −60 launches/step.

Production-shape constraints (DSv4 Flash-Base FP8 decode):
  hc=4, head_dim=128, hc_hidden=8192, n_splits_pre=32, K=512, page_size=64,
  max_seq_len capped via SGLANG_INDEXER_MAX_SEQ_LEN.
  s_q=1 at decode. S_q dim is folded by the caller (b*s_q == b).
"""
from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _m3_indexer_megakernel(
    scores_ptr,           # [B, max_seq_len] f32
    seq_lens_ptr,         # [B] i32
    page_tables_ptr,      # [B, num_pages] i32
    topk_length_ptr,      # [B] i32 (or null when HAS_TL=0)
    out_page_indices_ptr, # [B, K] i32
    out_raw_indices_ptr,  # [B, K] i32 (always written; saves caller branch)
    out_mask_ptr,         # [B*S_q, K] int8 (bool) — S_q folded into row stride
    stride_scores_b,
    stride_pt_b,
    K: tl.constexpr,
    PAGE_BITS: tl.constexpr,
    PAGE_MASK: tl.constexpr,
    BLOCK_L: tl.constexpr,
    HAS_TL: tl.constexpr,    # 1 if topk_length is provided
    WRITE_RAW: tl.constexpr, # 1 to also write raw_indices, 0 to skip the store
):
    """One CTA per batch row. Sorts scores → top-K → page indices → invalid mask.

    Rationale for one-CTA-per-row layout:
      The top-K via `tl.sort(BLOCK_L)` is fundamentally per-row work, since
      the relative order across the L dim has to be exact. Splitting K
      across CUs would require a top-K reduction tree which would cost more
      launches than it saves at K=512.

      With B in {1..8} this only fills 8/256 CUs, but each CTA does ~13K
      sort comparisons + 512 page-table loads + 512 mask stores. That's
      enough work per CTA that the parallelism gap is irrelevant — bench
      shows the bottleneck is per-CTA compute, not occupancy.
    """
    pid = tl.program_id(0)

    seq_len = tl.load(seq_lens_ptr + pid).to(tl.int32)

    row_offs = tl.arange(0, BLOCK_L)
    valid_score_mask = row_offs < seq_len

    # Load full score row; padding gets -inf (sorts last via the bit trick).
    scores = tl.load(
        scores_ptr + pid * stride_scores_b + row_offs,
        mask=valid_score_mask,
        other=-float("inf"),
    )

    # ----- sort: descending-monotonic uint32 key + position packed in uint64
    # Mirrors convert_to_uint32 in csrc/deepseek_v4/topk.cuh:
    #   negative floats: ~bits         (bigger magnitude → smaller key)
    #   positive floats: bits | 0x80000000  (positives shift above all negatives)
    # Then largest float maps to largest key. We negate via (MAX-key) so
    # tl.sort's ascending order produces descending-by-score.
    score_bits_u32 = scores.to(tl.uint32, bitcast=True)
    sign_set = score_bits_u32 >> 31
    high_bit = tl.full((), 0x80000000, dtype=tl.uint32)
    max_u32 = tl.full((), 0xFFFFFFFF, dtype=tl.uint32)
    key_pos = score_bits_u32 | high_bit
    key_neg = max_u32 - score_bits_u32
    key = tl.where(sign_set == 1, key_neg, key_pos)
    inv_key = max_u32 - key
    inv_key = tl.where(valid_score_mask, inv_key, max_u32)

    packed = (inv_key.to(tl.uint64) << 32) | row_offs.to(tl.uint64)
    sorted_packed = tl.sort(packed)

    # First K entries are top-K (or fewer if seq_len < K).
    out_mask = row_offs < K
    raw_pos = (sorted_packed & 0xFFFFFFFF).to(tl.int32)
    sorted_inv_key = (sorted_packed >> 32).to(tl.uint32)
    is_real = sorted_inv_key < 0xFFFFFFFF
    raw_pos = tl.where(is_real, raw_pos, -1)

    # ----- convert raw_pos → page-mapped page_indices
    page_id = raw_pos >> PAGE_BITS
    page_id_clamped = tl.maximum(page_id, 0)
    physical_page = tl.load(
        page_tables_ptr + pid * stride_pt_b + page_id_clamped,
        mask=out_mask & is_real,
        other=0,
    ).to(tl.int32)
    page_indices = (physical_page << PAGE_BITS) | (raw_pos & PAGE_MASK)
    page_indices = tl.where(is_real, page_indices, -1)

    # ----- invalid_mask: same layout as raw_indices (B, K).
    # Equivalent to:
    #   m = raw < 0
    #   if topk_length is not None:
    #       m = m | (arange(K) >= topk_length[b])
    mask_neg = raw_pos < 0
    if HAS_TL:
        tl_b = tl.load(topk_length_ptr + pid)
        mask_ge = row_offs >= tl_b
        invalid_bool = mask_neg | mask_ge
    else:
        invalid_bool = mask_neg

    # Stores
    base_out = pid * K + row_offs
    tl.store(out_page_indices_ptr + base_out, page_indices, mask=out_mask)
    if WRITE_RAW:
        tl.store(out_raw_indices_ptr + base_out, raw_pos, mask=out_mask)
    # int8 store for the bool mask. Caller views as torch.bool (1 byte/elem).
    tl.store(out_mask_ptr + base_out, invalid_bool.to(tl.int8), mask=out_mask)


def m3_indexer_megakernel(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    out_page_indices: torch.Tensor,
    out_raw_indices: Optional[torch.Tensor],
    out_invalid_mask: torch.Tensor,
    page_size: int,
    s_q: int = 1,
) -> bool:
    """Single-Triton-launch fusion of `topk_transform_512_triton`
    + `invalid_mask` for the DSv4 indexer chain.

    Returns True if the kernel fired, False if shapes are unsupported (caller
    should fall back to the unfused path).

    Inputs:
      scores:           [B, max_seq_len] f32
      seq_lens:         [B] i32
      page_tables:      [B, num_pages] i32
      topk_length:      [B] i32 or None (None == arange-mask disabled)

    In-place outputs:
      out_page_indices: [B, K] i32
      out_raw_indices:  [B, K] i32 or None
      out_invalid_mask: [B*S_q, K] bool

    `s_q` is the per-batch query length; at decode it's always 1 so
    B*S_q == B. The mask layout is [B*S_q, K] as expected by CK V32.

    Constraints:
      page_size must be a power of 2.
      K must be ≤ BLOCK_L = next_pow2(max(scores.shape[1], K)).
      For very long L (≳ 8192 on MI355X) Triton's L·log L sort is slower
      than torch.topk's L·log K — the caller should fall back in that case
      (mirrors `topk_transform_512_triton`).
    """
    # --- shape & stride sanity ---
    if scores.dim() != 2 or page_tables.dim() != 2:
        return False
    B, max_seq_len = scores.shape
    if seq_lens.shape != (B,) or page_tables.shape[0] != B:
        return False
    if out_page_indices.dim() != 2 or out_page_indices.shape[0] != B:
        return False
    K = out_page_indices.shape[1]
    if out_invalid_mask.shape != (B * s_q, K):
        return False
    if out_raw_indices is not None and out_raw_indices.shape != (B, K):
        return False
    if scores.dtype != torch.float32:
        return False
    if not (page_size > 0 and (page_size & (page_size - 1)) == 0):
        return False

    # Apply the same indexer cap that topk_transform_512_triton does, so
    # BLOCK_L = next_pow2(L) stays under TRITON_MAX_TENSOR_NUMEL (1M) under
    # cuda graph capture (capture pre-allocates logits to worst-case context).
    cap = int(os.environ.get("SGLANG_INDEXER_MAX_SEQ_LEN", "0"))
    if cap > 0 and max_seq_len > cap:
        scores = scores[:, :cap]
        max_seq_len = scores.shape[1]

    # Large-L cutoff: same threshold as topk_transform_512_triton. Above this
    # the L·log L sort is slower than torch.topk's L·log K; we return False
    # so the caller falls back to the unfused chain (which itself dispatches
    # to torch.topk in that regime).
    triton_max_l = int(os.environ.get("SGLANG_TOPK_TRITON_MAX_L", "8192"))
    if max_seq_len > triton_max_l:
        return False

    BLOCK_L = triton.next_power_of_2(max(max_seq_len, K))
    page_bits = (page_size - 1).bit_length()
    page_mask = page_size - 1

    has_tl = topk_length is not None
    write_raw = out_raw_indices is not None

    # Triton requires non-null pointers even for unused arguments.
    tl_ptr = topk_length if has_tl else seq_lens
    raw_ptr = out_raw_indices if write_raw else out_page_indices

    # The mask must be int8-storable (torch.bool is 1 byte). Accept bool/int8/uint8.
    if out_invalid_mask.dtype not in (torch.bool, torch.int8, torch.uint8):
        return False

    grid = (B,)
    # BLOCK_L-conditional launch knobs.
    #
    # PMC analysis (rocprofv3, MI355X): the per-CTA `tl.sort(BLOCK_L)` is
    # LDS-bound, NOT compute-bound. SQ_WAIT_INST_LDS dominates the wait
    # bucket (LDS-wait/VALU = 0.42 at BLOCK_L=4096, 0.84 at BLOCK_L=8192
    # with the original 4-warp config). Wider warp counts amortize LDS
    # round-trips by running more independent waves, but the sweet spot
    # depends on BLOCK_L. Latency sweep (B=6 dominant shape):
    #
    #   BLOCK_L  best (nw, wpe)     vs original (nw=4 wpe=0)
    #   ≤2048    (4, 0)             flat — too little work to absorb wider CTAs
    #   4096     (16, 1)            1.13×  (production: stacked-best INDEXER_CAP=4096)
    #   8192     (8, 4)             1.27×  (worst-case context window)
    #
    # nw=16 at BLOCK_L=8192 stays flat (register pressure dilutes wins);
    # nw=8 at BLOCK_L=4096 regresses (under-occupancy at this shape). The
    # selected pair per BLOCK_L is the empirical Pareto winner from the
    # microbench/triton_port_v2/bench_m3_block_sweep.py harness.
    if BLOCK_L >= 8192:
        nw, wpe = 8, 4
    elif BLOCK_L >= 4096:
        nw, wpe = 16, 1
    else:
        nw, wpe = 4, 0

    _m3_indexer_megakernel[grid](
        scores,
        seq_lens,
        page_tables,
        tl_ptr,
        out_page_indices,
        raw_ptr,
        out_invalid_mask,
        scores.stride(0),
        page_tables.stride(0),
        K=K,
        PAGE_BITS=page_bits,
        PAGE_MASK=page_mask,
        BLOCK_L=BLOCK_L,
        HAS_TL=has_tl,
        WRITE_RAW=write_raw,
        num_warps=nw,
        waves_per_eu=wpe,
        num_stages=1,
    )
    return True


# ============================================================================
# Reference torch chain (4-step unfused) — used by the v2 microbench
# correctness gate AND the speedup baseline.
# ============================================================================
def m3_indexer_chain_torch(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    out_page_indices: torch.Tensor,
    out_raw_indices: Optional[torch.Tensor],
    out_invalid_mask: torch.Tensor,
    page_size: int,
    s_q: int = 1,
) -> None:
    """4-step torch reference for the indexer chain: matches the production
    `topk_transform_512_pytorch_vectorized` + `_get_invalid_mask` MISS path.

    Step 1: mask scores past seq_lens with -inf
    Step 2: torch.topk → raw_indices
    Step 3: gather page_tables → page_indices
    Step 4: invalid_mask = (raw < 0) | (arange(K) >= topk_length[b])

    Used by both the bench's torch baseline (so we measure 4 launches vs 1)
    and the correctness oracle.
    """
    B, max_seq_len = scores.shape
    K = out_page_indices.shape[1]
    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1
    device = scores.device

    pos = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(B, -1)
    valid = pos < seq_lens.unsqueeze(1)
    masked = torch.where(valid, scores, scores.new_full((), float("-inf")))
    actual_k = min(K, max_seq_len)
    _, raw = torch.topk(masked, k=actual_k, dim=1, largest=True, sorted=False)
    raw = raw.to(torch.int32)
    if actual_k < K:
        pad = torch.full((B, K - actual_k), -1, dtype=torch.int32, device=device)
        raw = torch.cat([raw, pad], dim=1)

    bidx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)
    gathered = scores[bidx.flatten(), raw.clamp(min=0).long().flatten()].view(B, K)
    valid_topk = gathered != float("-inf")

    page_idx_clamped = torch.clamp(raw >> page_bits, min=0).long()
    phys = torch.gather(page_tables, dim=1, index=page_idx_clamped)
    pidx = ((phys << page_bits) | (raw & page_mask)).to(torch.int32)
    pidx = torch.where(valid_topk, pidx, pidx.new_full((), -1))
    raw_out = torch.where(valid_topk, raw, raw.new_full((), -1))

    out_page_indices.copy_(pidx)
    if out_raw_indices is not None:
        out_raw_indices.copy_(raw_out)

    # invalid_mask
    mask = raw_out < 0
    if topk_length is not None:
        arange_k = torch.arange(K, device=device, dtype=topk_length.dtype).view(1, K)
        mask = mask | (arange_k >= topk_length.view(B, 1))
    # caller expects [B*S_q, K] — at decode S_q==1 so this is just a view
    out_invalid_mask.copy_(mask.view(B * s_q, K))


# ============================================================================
# Cross-module mask publish/lookup — bridges the indexer-side megakernel write
# to the adapter-side `_get_invalid_mask` reader. Lives here (in the kernel
# module) so neither indexer.py nor debug_flash_mla_adapter.py owns the table.
#
# Key:    (data_ptr(indices), id(topk_length))
# Value:  the precomputed mask_2d tensor of shape (b*s_q, topk).
#
# Why this scheme:
#   - data_ptr() is stable across `.unsqueeze()` views (same storage), so the
#     indexer-side write and the adapter-side read on a (B,K)-shaped indices
#     vs (B,1,K)-shaped indices both hit.
#   - id(topk_length) is stable since the topk_length tensor passed through
#     deepseek_v4_backend is NOT view-mangled (no unsqueeze along the way).
#   - We don't include (b, s_q, topk) in the key — at decode they are derived
#     from indices.shape, and at the hit-check we verify the cached mask
#     matches the requested (b*s_q, topk) shape; mismatches fall through.
#   - Cap at 16 to prevent unbounded growth across capture sizes; in practice
#     production has 1 capture-mode shape + 1 eager shape = 2 entries.
# ============================================================================
_M3_INVALID_MASK_PUBLISH: dict = {}


def publish_invalid_mask(
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    mask_2d: torch.Tensor,
) -> None:
    """Indexer side: stash the mask under (data_ptr(indices), id(topk_length))."""
    key = (indices.data_ptr(), id(topk_length))
    if len(_M3_INVALID_MASK_PUBLISH) > 16:
        _M3_INVALID_MASK_PUBLISH.clear()
    _M3_INVALID_MASK_PUBLISH[key] = mask_2d


def lookup_invalid_mask(
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    expected_shape: tuple,
) -> Optional[torch.Tensor]:
    """Adapter side: return the published mask if shape matches, else None.

    Called from `_get_invalid_mask` BEFORE its data_ptr cache; on hit we skip
    `get_invalid_mask_triton` entirely. On miss we return None and the caller
    falls through to its existing path (data_ptr cache → triton MISS).
    """
    key = (indices.data_ptr(), id(topk_length))
    cached = _M3_INVALID_MASK_PUBLISH.get(key)
    if cached is None:
        return None
    if cached.shape != expected_shape:
        return None
    return cached


def ensure_invalid_mask_buffer(core_metadata) -> torch.Tensor:
    """Return a persistent (B*S_q=B, K) bool buffer attached to core_metadata.

    Allocated once per metadata instance (matches `c4_sparse_page_indices`
    lifetime), so cuda-graph capture pre-allocates it at worst-case batch.
    Lazy-init: created on first call; reused across all subsequent calls.
    """
    indices = core_metadata.c4_sparse_page_indices
    B, K = indices.shape
    buf = getattr(core_metadata, "_m3_c4_sparse_invalid_mask", None)
    if buf is not None and buf.shape == (B, K) and buf.device == indices.device:
        return buf
    buf = torch.empty((B, K), dtype=torch.bool, device=indices.device)
    # Attach as a regular attribute (dataclass __post_init__ already ran;
    # PagedCoreMetadata accepts dynamic attribute writes).
    try:
        core_metadata._m3_c4_sparse_invalid_mask = buf
    except Exception:
        # Frozen dataclass safety: just return — on next call we re-alloc
        # which is still fine since the kernel writes in-place to whatever
        # buffer we hand it.
        pass
    return buf
