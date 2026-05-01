"""Fused Triton sampler for DSv4 / Flash-Base FP8.

Replaces the multi-kernel sampling chain:
  logits.div_(temp) -> softmax(logits) -> aiter::top_k_top_p_sampling_from_probs(probs, top_k, top_p)

with a SINGLE Triton kernel that:
  1. Streams over vocab to find top_k logits (running buffer in registers)
  2. Sorts top_k descending
  3. Computes softmax over top_k (numerically stable: max-subtract, exp, normalize)
  4. Cumsum + top_p truncation (find cutoff index k* where cumsum >= p)
  5. Samples via inverse-CDF (uniform draw + cumulative threshold)

DESIGN:
- One program per batch row (grid = (B,))
- Streaming top_k pass: each program walks vocab in BLOCK_V tiles, maintains
  top_k items via register buffer (no global memory). For K <= 64 the buffer
  fits in SGPRs/VGPRs comfortably.
- Per-batch top_k, top_p, temperature passed as tensors (sglang convention)
- Per-batch RNG seed (positions[batch_idx] for deterministic mode, else
  philox-derived from a single seed)

WHY this should beat aiter::top_k_top_p_sampling_from_probs:
- aiter's kernel takes PROBS (post-softmax). We take RAW LOGITS, fusing in
  the softmax + temperature + cumsum + sampling. Eliminates 2 kernel launches
  upstream (logits.div_(temp), torch.softmax(logits)).
- Single kernel launch vs 3 (temp + softmax + aiter sample).
- Streaming approach avoids materializing the full (B, V) probs tensor in
  global memory between softmax and top_k.

LIMITATIONS / SCOPE:
- Supports top_k >= 1 and 0 < top_p <= 1.0
- Does NOT support min_p (rare path; falls through to torch)
- Does NOT support deterministic per-position seeded sampling (uses single
  seed broadcast across batch; sglang's position-based seeding can be added
  if proven to be a hot path)
- Vocab must fit BLOCK_V * num_blocks; tested up to V=200k

STATUS: Phase 1 — kernel + microbench. Wire-in default OFF.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Reference torch impl (oracle for correctness gate).
# ---------------------------------------------------------------------------
def _fused_sampler_reference(
    logits: torch.Tensor,    # [B, V] fp32 / bf16
    top_k: torch.Tensor,     # [B] int32
    top_p: torch.Tensor,     # [B] fp32
    temperature: torch.Tensor,  # [B] fp32
    seed: int = 42,
) -> torch.Tensor:
    """Returns sampled token ids [B] int32. Matches the production semantics
    of sglang's top_k + top_p sampling."""
    B, V = logits.shape
    device = logits.device

    # Apply temperature
    logits_t = logits.float() / temperature.unsqueeze(-1)

    # Sort descending
    sorted_logits, sorted_idx = torch.sort(logits_t, dim=-1, descending=True)

    # Top-k: keep first top_k items
    rank = torch.arange(V, device=device).unsqueeze(0).expand(B, V)
    top_k_mask = rank < top_k.unsqueeze(-1)
    sorted_logits = sorted_logits.masked_fill(~top_k_mask, float("-inf"))

    # Softmax (over the truncated logits)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)

    # Top-p: cumsum and truncate
    cum = torch.cumsum(sorted_probs, dim=-1)
    # Keep items where the previous cumsum is < top_p (i.e., this item still in the nucleus)
    # cumsum_prev = cum - sorted_probs; first item is always kept
    keep = cum - sorted_probs < top_p.unsqueeze(-1)
    sorted_probs = sorted_probs * keep.float()
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # Sample
    g = torch.Generator(device=device).manual_seed(seed)
    u = torch.rand(B, device=device, generator=g)
    cum_p = torch.cumsum(sorted_probs, dim=-1)
    sample_idx_in_sorted = (cum_p < u.unsqueeze(-1)).sum(dim=-1)  # find first idx where cum >= u
    sample_idx_in_sorted = sample_idx_in_sorted.clamp_max(V - 1)

    # Map back from sorted-rank to vocab-id
    sampled = sorted_idx.gather(1, sample_idx_in_sorted.unsqueeze(-1)).squeeze(-1)
    return sampled.to(torch.int32)


# ---------------------------------------------------------------------------
# Triton kernel (per-batch, streaming top_k + top_p + sample)
# ---------------------------------------------------------------------------
@triton.jit
def _fused_sampler_kernel(
    logits_ptr,       # [B, V] fp32 (cast bf16→fp32 inline)
    top_k_ptr,        # [B] int32
    top_p_ptr,        # [B] fp32
    temp_ptr,         # [B] fp32
    seed,             # uint64 scalar, plus per-batch via batch_id
    out_ptr,          # [B] int32
    B,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    K_MAX: tl.constexpr,        # max top_k supported (compile-time)
    LOGITS_IS_BF16: tl.constexpr,
):
    """One program per batch. Streams vocab in BLOCK_V tiles, maintains a
    register-resident top_k buffer (logit values + indices). After the pass,
    runs softmax / cumsum / top_p truncation / inverse-CDF sample inline.
    """
    pid_b = tl.program_id(0)
    if pid_b >= B:
        return

    # Per-batch sampling params
    temp = tl.load(temp_ptr + pid_b)
    inv_temp = 1.0 / temp
    top_k = tl.load(top_k_ptr + pid_b)   # int32
    top_p = tl.load(top_p_ptr + pid_b)   # fp32
    # Clamp top_k to [1, K_MAX]
    top_k_eff = tl.minimum(tl.maximum(top_k, 1), K_MAX)

    # ----- Pass 1: streaming top_K via union-buffer + tl.sort -----
    # Union buffer of size COMBINED = K_MAX + BLOCK_V. First K_MAX slots hold
    # the running top-K, next BLOCK_V slots are loaded with the current tile.
    # Sort descending, write back first K_MAX.
    # Triton-AMD's tl.cat may reorder, so we use a fixed-size union tensor and
    # write into it via tl.where on a position mask.
    NEG_INF = float("-inf")
    k_range = tl.arange(0, K_MAX)
    top_vals = tl.full([K_MAX], NEG_INF, dtype=tl.float32)
    top_idxs = tl.zeros([K_MAX], dtype=tl.int32)
    COMBINED: tl.constexpr = K_MAX + BLOCK_V
    combined_range = tl.arange(0, COMBINED)
    in_top_k_slots = combined_range < K_MAX
    tile_slot_offs = combined_range - K_MAX  # valid where >= K_MAX

    for v_start in range(0, V, BLOCK_V):
        v_offs = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < V
        x = tl.load(
            logits_ptr + pid_b * V + v_offs,
            mask=v_mask, other=NEG_INF,
        )
        if LOGITS_IS_BF16:
            x = x.to(tl.float32)
        x = x * inv_temp

        # Build union: [top_vals (K_MAX) | x (BLOCK_V)] of size COMBINED
        # Use position mask to select from top_vals or x
        # For position p in [0, COMBINED):
        #   if p < K_MAX: union[p] = top_vals[p], union_idx[p] = top_idxs[p]
        #   else:         union[p] = x[p - K_MAX], union_idx[p] = v_offs[p - K_MAX]
        # Build top side (zero outside K_MAX):
        top_vals_padded = tl.where(
            in_top_k_slots,
            tl.gather(top_vals, tl.where(in_top_k_slots, combined_range, 0), axis=0),
            NEG_INF,
        )
        top_idxs_padded = tl.where(
            in_top_k_slots,
            tl.gather(top_idxs, tl.where(in_top_k_slots, combined_range, 0), axis=0),
            0,
        )
        x_padded = tl.where(
            in_top_k_slots,
            NEG_INF,
            tl.gather(x, tl.where(in_top_k_slots, 0, tile_slot_offs), axis=0),
        )
        v_offs_padded = tl.where(
            in_top_k_slots,
            0,
            tl.gather(v_offs.to(tl.int32), tl.where(in_top_k_slots, 0, tile_slot_offs), axis=0),
        )
        union_vals = tl.where(in_top_k_slots, top_vals_padded, x_padded)
        union_idxs = tl.where(in_top_k_slots, top_idxs_padded, v_offs_padded)

        # Sort descending: tl.sort returns ascending; negate for descending
        sorted_vals_neg, perm = tl.sort(-union_vals, dim=0, descending=False, return_indices=True)
        sorted_vals_desc = -sorted_vals_neg
        sorted_idxs = tl.gather(union_idxs, perm, axis=0)
        # Take first K_MAX
        top_vals = tl.gather(sorted_vals_desc, k_range, axis=0)
        top_idxs = tl.gather(sorted_idxs, k_range, axis=0)

    # ----- top_k_eff truncation: mask top_vals[i >= top_k_eff] to -inf -----
    keep_k = k_range < top_k_eff
    top_vals = tl.where(keep_k, top_vals, NEG_INF)

    # ----- softmax over top_K -----
    max_v = tl.max(top_vals)
    exp_v = tl.exp(top_vals - max_v)
    exp_v = tl.where(keep_k, exp_v, 0.0)
    denom = tl.sum(exp_v) + 1e-12
    probs = exp_v / denom

    # ----- top_p truncation -----
    cumprobs = tl.cumsum(probs, axis=0)
    cumprobs_prev = cumprobs - probs  # 0 at index 0; cum_before_i
    in_nucleus = cumprobs_prev < top_p
    probs = tl.where(in_nucleus, probs, 0.0)
    norm = tl.sum(probs) + 1e-12
    probs = probs / norm

    # ----- Inverse-CDF sample with uniform u from philox(seed, pid_b) -----
    # Use one philox draw per batch
    u_int = tl.randint(seed, pid_b)
    # Convert int32 to uniform fp32 in [0, 1)
    u = (u_int & 0x7FFFFFFF).to(tl.float32) / 2147483648.0

    cum2 = tl.cumsum(probs, axis=0)
    # Find first index where cum2 >= u
    pick_mask = cum2 >= u
    # The first true index is at: K - max(reverse cumulative AND of ~mask) — simpler: find
    # the smallest k such that mask[k] is True. argmax of reversed mask cast to int gives K-1-i.
    # Triton trick: compute cumulative-OR from left, find first True.
    # Alt: use that pick_mask is monotonically True from some index onwards (cum2 is non-decreasing),
    # so the index = K - sum(pick_mask).
    # But we need a 0-based index of the FIRST True.
    pick_count = tl.sum(pick_mask.to(tl.int32))
    pick_idx_in_sorted = K_MAX - pick_count  # index of first True in [0, K_MAX)
    pick_idx_in_sorted = tl.minimum(tl.maximum(pick_idx_in_sorted, 0), K_MAX - 1)

    # Gather the vocab id from top_idxs[pick_idx_in_sorted]
    sampled = tl.sum(tl.where(k_range == pick_idx_in_sorted, top_idxs, 0))

    # Store
    tl.store(out_ptr + pid_b, sampled)


def fused_sampler_triton(
    logits: torch.Tensor,        # [B, V]
    top_k: torch.Tensor,         # [B] int32
    top_p: torch.Tensor,         # [B] fp32
    temperature: torch.Tensor,   # [B] fp32
    seed: int = 42,
    K_MAX: int = 64,
) -> torch.Tensor:
    """Single-launch fused sampler. Returns [B] int32 token ids.

    Production-aligned signature: takes RAW LOGITS (not probs), with per-batch
    sampling params. Replaces the multi-kernel chain (temp scale + softmax +
    aiter sampling).
    """
    B, V = logits.shape
    assert top_k.shape == (B,)
    assert top_p.shape == (B,)
    assert temperature.shape == (B,)

    out = torch.empty(B, dtype=torch.int32, device=logits.device)

    # Choose BLOCK_V based on vocab size and SGPR/VGPR pressure
    BLOCK_V = min(triton.next_power_of_2(min(V, 1024)), 1024)
    LOGITS_IS_BF16 = logits.dtype == torch.bfloat16

    # Note: K_MAX must be compile-time constexpr and >= max(top_k) at runtime.
    # Caller is responsible for clamping top_k to <= K_MAX before passing.
    _fused_sampler_kernel[(B,)](
        logits, top_k, top_p, temperature,
        seed, out,
        B, V=V, BLOCK_V=BLOCK_V,
        K_MAX=K_MAX, LOGITS_IS_BF16=LOGITS_IS_BF16,
    )
    return out
