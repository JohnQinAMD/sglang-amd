"""Fused KV-pool maintain + gather + APE add kernel for compress_decode_old.

Replaces deepseek_v4.py:1597-1626 (5+ launches per NSA layer) with a single
Triton kernel. The torch reference (used as oracle for correctness) is
preserved here for v2 microbench validation.

Per the eager-trace agent attribution (elementwise-template-attribution-2026-05-01.md),
this site owns >92% of `index_elementwise_kernel` budget (~12.4 ms eager / 9 passes
in the audit trace) plus ~5 ms of `manual_unroll` (where, add, mul).

Design (ratio=4, overlap=True case):
  - kv_pool layout: [num_reqs, coff*ratio, coff*head_dim] per channel (kv, score)
  - inputs:
      kv_pool_kv, kv_pool_score : [num_reqs, coff*ratio, coff*head_dim] (rw)
      req_indices               : [bs] int32
      seq_lens                  : [bs] int32
      new_kv, new_score         : [bs, coff*ratio, coff*head_dim] (write)
        (typically the new_kv has only ONE valid time-slot; rest are pre-filled)
        Actually L1598 writes kv_and_scores which is shape [bs, ...] one-time-slot.
      ape_score                 : [coff*ratio, coff*head_dim] fp32 (constant per layer)
      ratio, overlap, coff      : constexpr
  - output:
      kv_to_compress_kv, kv_to_compress_score : [bs, coff*ratio, coff*head_dim]

Operations performed (per batch element b):
  1. write_pos = (seq_lens[b] - 1) % ratio + overlap * ratio
  2. kv_pool[req_indices[b], write_pos, :] = new_value (kv + score)
  3. out[b] = kv_pool[req_indices[b]]  (gather all time slots)
  4. if overlap and (seq_lens[b] % ratio == 0):
        kv_pool[req_indices[b], :ratio, :] = out[b][ratio:, :]  (shift left)
  5. out[b].score += ape_score  (broadcast add along time dim)

NOT IN SCOPE (handled outside the kernel):
  - The downstream M2 megakernel call at L1670 stays as-is
  - `overlap_transform_decode` at L1633/1636 stays as-is (separate ops)

STATUS: Phase 1 — reference oracle + structural Triton kernel + microbench.
        Wire-in default OFF. Activation pending v2 graph-replay validation
        AND E2E live smoke (per the M1 / B-pre / hc_pre lessons).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Reference torch impl (the oracle for v2 correctness). Mirrors
# deepseek_v4.py:1597-1626 byte-for-byte (without the surrounding view/M2 calls).
# ---------------------------------------------------------------------------
def _compress_decode_kv_pool_reference(
    kv_pool_kv: torch.Tensor,        # [num_reqs, T, D]  (T = coff*ratio, D = coff*head_dim)
    kv_pool_score: torch.Tensor,     # [num_reqs, T, D]
    req_indices: torch.Tensor,       # [bs] int32
    seq_lens: torch.Tensor,          # [bs] int32
    new_kv: torch.Tensor,            # [bs, D] — value to write at write_pos
    new_score: torch.Tensor,         # [bs, D]
    ape_score: torch.Tensor,         # [ratio, D] fp32 — added to score after L1621 view+L1625 broadcast
    ratio: int,
    overlap: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (kv_to_compress_kv, kv_to_compress_score) at shape (bs, T=coff*ratio, D)
    with APE already added per the production L1621-L1625 view+broadcast semantics:
        view(-1, ratio, D) → score + ape.unsqueeze(0) → view(bs, T, D)
    Equivalent to: out_score[b, t, d] = pool[b, t, d] + ape[t % ratio, d]

    Side effects: writes to kv_pool_kv and kv_pool_score in place (matches
    production L1598 + L1607 semantics).
    """
    bs = req_indices.shape[0]
    T, D = kv_pool_kv.shape[1], kv_pool_kv.shape[2]
    coff = T // ratio
    # L1597
    write_pos = (seq_lens - 1) % ratio + (ratio if overlap else 0)
    # L1598
    kv_pool_kv[req_indices, write_pos] = new_kv
    kv_pool_score[req_indices, write_pos] = new_score
    # L1602 — snapshot
    kv_to_compress_kv = kv_pool_kv[req_indices].clone()
    kv_to_compress_score = kv_pool_score[req_indices].clone()
    if overlap:
        # L1604-1618
        should_shift = (seq_lens % ratio == 0)
        kv_pool_kv[req_indices, :ratio] = torch.where(
            should_shift[:, None, None],
            kv_to_compress_kv[:, ratio:],
            kv_to_compress_kv[:, :ratio],
        )
        kv_pool_score[req_indices, :ratio] = torch.where(
            should_shift[:, None, None],
            kv_to_compress_score[:, ratio:],
            kv_to_compress_score[:, :ratio],
        )
    # L1621-L1625 — view to (bs*coff, ratio, D), add APE, view back to (bs, T, D)
    kv_to_compress_score = (
        kv_to_compress_score.view(bs * coff, ratio, D) + ape_score.unsqueeze(0)
    ).view(bs, T, D)
    return kv_to_compress_kv, kv_to_compress_score


# ---------------------------------------------------------------------------
# Triton fused kernel (Phase 1 — minimum viable; correctness-first, perf later).
# ---------------------------------------------------------------------------
@triton.jit
def _compress_decode_kv_pool_kernel(
    kv_pool_kv_ptr,
    kv_pool_score_ptr,
    req_indices_ptr,
    seq_lens_ptr,
    new_kv_ptr,
    new_score_ptr,
    ape_score_ptr,
    out_kv_ptr,
    out_score_ptr,
    # strides
    pool_stride_req, pool_stride_t, pool_stride_d,
    new_stride_b, new_stride_d,
    ape_stride_t, ape_stride_d,
    out_stride_b, out_stride_t, out_stride_d,
    # constexprs
    bs,
    T: tl.constexpr,        # coff*ratio
    D: tl.constexpr,         # coff*head_dim
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,   # bool: True for ratio=4 path
    BLOCK_D: tl.constexpr,
):
    """One program per (batch, t-pos, d-tile). Race-free design:

    Each program computes its OWN final pool[req, pid_t, d] state and out[b, pid_t, d]
    by ALWAYS reading from the original pool + new_value lookup (never reading
    what another program writes). Final pool state per-position:

        if pid_t == write_pos:
            final_pool[t] = new_value
        elif OVERLAP and should_shift and pid_t < RATIO:
            # take from pid_t + RATIO; that slot might itself be the write_pos
            src_t = pid_t + RATIO
            final_pool[t] = (new_value if src_t == write_pos else original_pool[src_t])
        else:
            final_pool[t] = original_pool[t]   # unchanged

    out[b, t] (the snapshot BEFORE shift, plus APE on score):
        if pid_t == write_pos: out_kv[b, t] = new_value, out_score[b, t] = new_score + ape[t]
        else: out_kv[b, t] = original_pool[t], out_score[b, t] = orig_score[t] + ape[t]
    """
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    req = tl.load(req_indices_ptr + pid_b)
    seq = tl.load(seq_lens_ptr + pid_b)
    write_pos = (seq - 1) % RATIO + (RATIO if OVERLAP else 0)
    should_shift = (seq % RATIO == 0) if OVERLAP else False

    pool_base_t = req * pool_stride_req + pid_t * pool_stride_t

    # Always read original pool at pid_t (for snapshot output)
    orig_kv_t = tl.load(kv_pool_kv_ptr + pool_base_t + d_offs * pool_stride_d, mask=d_mask, other=0.0)
    orig_score_t = tl.load(kv_pool_score_ptr + pool_base_t + d_offs * pool_stride_d, mask=d_mask, other=0.0)

    # Load new value (always; cheap; only used if pid_t == write_pos OR shift target)
    new_kv_b = tl.load(new_kv_ptr + pid_b * new_stride_b + d_offs * new_stride_d, mask=d_mask, other=0.0).to(orig_kv_t.dtype)
    new_score_b = tl.load(new_score_ptr + pid_b * new_stride_b + d_offs * new_stride_d, mask=d_mask, other=0.0).to(orig_score_t.dtype)

    # ---- Compute snapshot at pid_t (before shift) — for OUT ----
    is_write_slot = (pid_t == write_pos)
    snap_kv = tl.where(is_write_slot, new_kv_b, orig_kv_t)
    snap_score = tl.where(is_write_slot, new_score_b, orig_score_t)

    # Add APE to score for output. Production APE has shape (ratio, D); the
    # add happens in the (bs*coff, ratio, D) view, so APE indexes by pid_t % RATIO.
    ape_idx = pid_t % RATIO
    ape_t = tl.load(ape_score_ptr + ape_idx * ape_stride_t + d_offs * ape_stride_d, mask=d_mask, other=0.0)
    out_score = snap_score.to(tl.float32) + ape_t

    out_base = pid_b * out_stride_b + pid_t * out_stride_t
    tl.store(out_kv_ptr + out_base + d_offs * out_stride_d, snap_kv, mask=d_mask)
    # Cast back to score dtype (typically bf16) for output. Match torch broadcast semantics
    # which would also stay in score dtype after add (note torch.where + add stays in
    # original dtype since ape is fp32 → upcast for add → torch keeps fp32). For
    # production parity, cast to score dtype here.
    tl.store(out_score_ptr + out_base + d_offs * out_stride_d, out_score.to(snap_score.dtype), mask=d_mask)

    # ---- Compute final pool state (in-place modification) ----
    if OVERLAP:
        # if should_shift and pid_t < RATIO: src_t = pid_t + RATIO
        # We need to compute final_pool[pid_t] = src value (either new_kv or pool[src_t])
        # Read pool[req, pid_t+RATIO, d] (the shift source)
        # Note: pid_t+RATIO is in [RATIO, 2*RATIO), valid range for OVERLAP=True
        if should_shift and pid_t < RATIO:
            src_t = pid_t + RATIO
            shift_base = req * pool_stride_req + src_t * pool_stride_t
            src_kv = tl.load(kv_pool_kv_ptr + shift_base + d_offs * pool_stride_d, mask=d_mask, other=0.0)
            src_score = tl.load(kv_pool_score_ptr + shift_base + d_offs * pool_stride_d, mask=d_mask, other=0.0)
            # If src_t IS the write_pos, the src value should be the new value (not the
            # original pool value, which hasn't been overwritten yet from this program's pov)
            src_is_write = (src_t == write_pos)
            shift_kv = tl.where(src_is_write, new_kv_b, src_kv)
            shift_score = tl.where(src_is_write, new_score_b, src_score)
            tl.store(kv_pool_kv_ptr + pool_base_t + d_offs * pool_stride_d, shift_kv, mask=d_mask)
            tl.store(kv_pool_score_ptr + pool_base_t + d_offs * pool_stride_d, shift_score, mask=d_mask)
        elif is_write_slot:
            # Write the new value at write_pos
            tl.store(kv_pool_kv_ptr + pool_base_t + d_offs * pool_stride_d, new_kv_b, mask=d_mask)
            tl.store(kv_pool_score_ptr + pool_base_t + d_offs * pool_stride_d, new_score_b, mask=d_mask)
        # else: pool[pid_t] unchanged, no write
    else:
        # No overlap: just commit the new value at write_pos
        if is_write_slot:
            tl.store(kv_pool_kv_ptr + pool_base_t + d_offs * pool_stride_d, new_kv_b, mask=d_mask)
            tl.store(kv_pool_score_ptr + pool_base_t + d_offs * pool_stride_d, new_score_b, mask=d_mask)


def compress_decode_kv_pool_fused(
    kv_pool_kv: torch.Tensor,
    kv_pool_score: torch.Tensor,
    req_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    new_kv: torch.Tensor,
    new_score: torch.Tensor,
    ape_score: torch.Tensor,
    ratio: int,
    overlap: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton drop-in replacement for the L1597-1626 chain.

    Caller must ensure:
      kv_pool_kv.shape[0] >= req_indices.max().item() + 1
      kv_pool_kv.shape[1] == ratio if not overlap else 2*ratio (= coff*ratio)
      ape_score.shape[0] == kv_pool_kv.shape[1]
      All same dtype (typically bf16 for kv/score, fp32 for ape).
    """
    bs = req_indices.shape[0]
    num_reqs, T, D = kv_pool_kv.shape
    assert kv_pool_score.shape == kv_pool_kv.shape, (
        f"score pool shape {kv_pool_score.shape} != kv pool shape {kv_pool_kv.shape}"
    )
    assert new_kv.shape == (bs, D), f"new_kv {new_kv.shape} expected ({bs}, {D})"
    assert new_score.shape == (bs, D), f"new_score {new_score.shape} expected ({bs}, {D})"
    assert ape_score.shape == (ratio, D), (
        f"ape {ape_score.shape} expected ({ratio}, {D}) — production APE is per-ratio not per-T"
    )
    assert T % ratio == 0, f"T={T} must be multiple of ratio={ratio}"

    out_kv = torch.empty(bs, T, D, dtype=kv_pool_kv.dtype, device=kv_pool_kv.device)
    out_score = torch.empty(bs, T, D, dtype=kv_pool_score.dtype, device=kv_pool_score.device)

    BLOCK_D = min(triton.next_power_of_2(D), 512)
    grid = (bs, T, triton.cdiv(D, BLOCK_D))

    _compress_decode_kv_pool_kernel[grid](
        kv_pool_kv, kv_pool_score,
        req_indices, seq_lens,
        new_kv, new_score,
        ape_score,
        out_kv, out_score,
        kv_pool_kv.stride(0), kv_pool_kv.stride(1), kv_pool_kv.stride(2),
        new_kv.stride(0), new_kv.stride(1),
        ape_score.stride(0), ape_score.stride(1),
        out_kv.stride(0), out_kv.stride(1), out_kv.stride(2),
        bs, T=T, D=D, RATIO=ratio, OVERLAP=overlap,
        BLOCK_D=BLOCK_D,
    )
    return out_kv, out_score


# ===========================================================================
# Phase 2 — extend kernel to absorb overlap_transform_decode (deepseek_v4.py:761)
# ---------------------------------------------------------------------------
# Phase 1 outputs (bs, T=coff*ratio, D=coff*head_dim). Production then runs:
#   if overlap:
#     out.kv = overlap_transform_decode(out.kv)        # (bs, T, head_dim)
#     out.score = overlap_transform_decode(out.score)  # (bs, T, head_dim)
# where overlap_transform_decode does:
#   ret = torch.cat((tensor[:, :r, :d], tensor[:, r:, d:]), dim=1)
# i.e., for output[b, t, d_out_idx]:
#   - if t < r: take input[b, t, d_out_idx]      (lower half d)
#   - if t >= r: take input[b, t, d_out_idx + d] (upper half d)
#
# This Phase 2 kernel writes that transformed output directly. For overlap=False
# (c128 path), output is identical to Phase 1 (D == head_dim).
#
# Microbench-validated -3.32 ms ship in Phase 1; Phase 2 adds another -0.2 to
# -0.4 ms est. by eliminating 2 cat launches per NSA layer.
# ===========================================================================


def _compress_decode_kv_pool_v2_reference(
    kv_pool_kv: torch.Tensor,
    kv_pool_score: torch.Tensor,
    req_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    new_kv: torch.Tensor,
    new_score: torch.Tensor,
    ape_score: torch.Tensor,
    ratio: int,
    overlap: bool,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference: phase 1 kernel + overlap_transform_decode(kv) + overlap_transform_decode(score).

    Returns out_kv, out_score at shape (bs, T=coff*ratio, head_dim).
    For overlap=False, head_dim == D (no transform); for overlap=True, head_dim = D//2.
    """
    # Phase 1 step
    out_kv_p1, out_score_p1 = _compress_decode_kv_pool_reference(
        kv_pool_kv, kv_pool_score,
        req_indices, seq_lens,
        new_kv, new_score,
        ape_score,
        ratio, overlap,
    )
    # out_p1 shape: (bs, T=coff*ratio, D=coff*head_dim)
    if overlap:
        # overlap_transform_decode: cat((t[:, :r, :d], t[:, r:, d:]), dim=1)
        # Result shape: (bs, T, head_dim) where T = 2*ratio
        r = ratio
        d = head_dim
        out_kv_xform = torch.cat((out_kv_p1[:, :r, :d], out_kv_p1[:, r:, d:]), dim=1)
        out_score_xform = torch.cat((out_score_p1[:, :r, :d], out_score_p1[:, r:, d:]), dim=1)
        return out_kv_xform, out_score_xform
    else:
        # overlap=False: head_dim == D, no transform
        return out_kv_p1, out_score_p1


@triton.jit
def _compress_decode_kv_pool_kernel_v2(
    kv_pool_kv_ptr,
    kv_pool_score_ptr,
    req_indices_ptr,
    seq_lens_ptr,
    new_kv_ptr,
    new_score_ptr,
    ape_score_ptr,
    out_kv_ptr,
    out_score_ptr,
    pool_stride_req, pool_stride_t, pool_stride_d,
    new_stride_b, new_stride_d,
    ape_stride_t, ape_stride_d,
    out_stride_b, out_stride_t, out_stride_d_out,
    bs,
    T: tl.constexpr,
    D: tl.constexpr,            # input pool d-dim (coff*head_dim)
    HEAD_DIM: tl.constexpr,     # output d-dim (head_dim if overlap else D)
    RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    BLOCK_D: tl.constexpr,      # tile size on FULL pool d-dim (D), same as Phase 1
):
    """Phase 2: per (pid_b, pid_t, pid_d). Grid = (bs, T, cdiv(D, BLOCK_D)) — same as Phase 1.

    Pool side-effects use full d_offs in [0, D) (covers all pool slots).
    Output writes are SELECTIVE based on overlap_transform_decode logic:
      - overlap=False: out[b, t, d_offs] = snap (full d, identity transform)
      - overlap=True, pid_t < RATIO: out[b, t, d_offs] = snap, ONLY where d_offs < HEAD_DIM
      - overlap=True, pid_t >= RATIO: out[b, t, d_offs - HEAD_DIM] = snap,
                                       ONLY where d_offs >= HEAD_DIM
                                       (maps source upper-half to output lower-half — overlap_transform_decode)
    """
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    req = tl.load(req_indices_ptr + pid_b)
    seq = tl.load(seq_lens_ptr + pid_b)
    write_pos = (seq - 1) % RATIO + (RATIO if OVERLAP else 0)
    should_shift = (seq % RATIO == 0) if OVERLAP else False

    pool_base_t = req * pool_stride_req + pid_t * pool_stride_t

    # Read original pool at (pid_t, d_offs)
    orig_kv = tl.load(kv_pool_kv_ptr + pool_base_t + d_offs * pool_stride_d, mask=d_mask, other=0.0)
    orig_score = tl.load(kv_pool_score_ptr + pool_base_t + d_offs * pool_stride_d, mask=d_mask, other=0.0)

    # Read new value at (pid_b, d_offs)
    new_kv_b = tl.load(new_kv_ptr + pid_b * new_stride_b + d_offs * new_stride_d, mask=d_mask, other=0.0).to(orig_kv.dtype)
    new_score_b = tl.load(new_score_ptr + pid_b * new_stride_b + d_offs * new_stride_d, mask=d_mask, other=0.0).to(orig_score.dtype)

    # Snapshot at pid_t (before shift) for OUTPUT
    is_write_slot = (pid_t == write_pos)
    snap_kv = tl.where(is_write_slot, new_kv_b, orig_kv)
    snap_score = tl.where(is_write_slot, new_score_b, orig_score)

    # APE add (production indexes APE by t % RATIO)
    ape_idx = pid_t % RATIO
    ape = tl.load(ape_score_ptr + ape_idx * ape_stride_t + d_offs * ape_stride_d, mask=d_mask, other=0.0)
    out_score = snap_score.to(tl.float32) + ape

    # ---- Selective output writes (overlap_transform_decode applied inline) ----
    if OVERLAP:
        # output_d_offs and out_write_mask depend on (pid_t < RATIO, d_offs)
        d_out_offs = tl.where(pid_t < RATIO, d_offs, d_offs - HEAD_DIM)
        out_write_mask = d_mask & tl.where(
            pid_t < RATIO,
            d_offs < HEAD_DIM,
            d_offs >= HEAD_DIM,
        )
    else:
        d_out_offs = d_offs
        out_write_mask = d_mask

    out_base = pid_b * out_stride_b + pid_t * out_stride_t
    tl.store(out_kv_ptr + out_base + d_out_offs * out_stride_d_out, snap_kv, mask=out_write_mask)
    tl.store(
        out_score_ptr + out_base + d_out_offs * out_stride_d_out,
        out_score.to(snap_score.dtype),
        mask=out_write_mask,
    )

    # ---- Pool side-effects (identical to Phase 1) — full d-dim ----
    if OVERLAP:
        if should_shift and pid_t < RATIO:
            src_t = pid_t + RATIO
            shift_base = req * pool_stride_req + src_t * pool_stride_t
            src_kv = tl.load(kv_pool_kv_ptr + shift_base + d_offs * pool_stride_d, mask=d_mask, other=0.0)
            src_score = tl.load(kv_pool_score_ptr + shift_base + d_offs * pool_stride_d, mask=d_mask, other=0.0)
            src_is_write = (src_t == write_pos)
            shift_kv = tl.where(src_is_write, new_kv_b, src_kv)
            shift_score = tl.where(src_is_write, new_score_b, src_score)
            tl.store(kv_pool_kv_ptr + pool_base_t + d_offs * pool_stride_d, shift_kv, mask=d_mask)
            tl.store(kv_pool_score_ptr + pool_base_t + d_offs * pool_stride_d, shift_score, mask=d_mask)
        elif is_write_slot:
            tl.store(kv_pool_kv_ptr + pool_base_t + d_offs * pool_stride_d, new_kv_b, mask=d_mask)
            tl.store(kv_pool_score_ptr + pool_base_t + d_offs * pool_stride_d, new_score_b, mask=d_mask)
    else:
        if is_write_slot:
            tl.store(kv_pool_kv_ptr + pool_base_t + d_offs * pool_stride_d, new_kv_b, mask=d_mask)
            tl.store(kv_pool_score_ptr + pool_base_t + d_offs * pool_stride_d, new_score_b, mask=d_mask)


def compress_decode_kv_pool_fused_v2(
    kv_pool_kv: torch.Tensor,
    kv_pool_score: torch.Tensor,
    req_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    new_kv: torch.Tensor,
    new_score: torch.Tensor,
    ape_score: torch.Tensor,
    ratio: int,
    overlap: bool,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Phase 2: includes overlap_transform_decode in the kernel.

    Output shape: (bs, T=coff*ratio, head_dim).
    For overlap=False: head_dim == D, no transform applied.
    For overlap=True: output is the post-overlap-transform shape (skips the cat ops).

    IMPORTANT: caller MUST stop calling overlap_transform_decode separately.
    """
    bs = req_indices.shape[0]
    num_reqs, T, D = kv_pool_kv.shape
    assert kv_pool_score.shape == kv_pool_kv.shape
    assert new_kv.shape == (bs, D)
    assert new_score.shape == (bs, D)
    assert ape_score.shape == (ratio, D), (
        f"ape {ape_score.shape} expected ({ratio}, {D})"
    )

    if overlap:
        coff = T // ratio
        assert coff == 2, f"v2 only handles coff=2 (overlap=True ratio=4)"
        assert head_dim * coff == D, f"head_dim*coff={head_dim*coff} vs D={D}"
        OUT_D = head_dim
    else:
        assert head_dim == D, f"overlap=False requires head_dim={D} (D); got {head_dim}"
        OUT_D = D

    out_kv = torch.empty(bs, T, OUT_D, dtype=kv_pool_kv.dtype, device=kv_pool_kv.device)
    out_score = torch.empty(bs, T, OUT_D, dtype=kv_pool_score.dtype, device=kv_pool_score.device)

    # Phase 2 grid sizes by FULL D (same as Phase 1) so pool side-effects
    # cover the entire d-dim. Output writes are selectively masked to the
    # transformed positions.
    BLOCK_D = min(triton.next_power_of_2(D), 512)
    grid = (bs, T, triton.cdiv(D, BLOCK_D))

    _compress_decode_kv_pool_kernel_v2[grid](
        kv_pool_kv, kv_pool_score,
        req_indices, seq_lens,
        new_kv, new_score,
        ape_score,
        out_kv, out_score,
        kv_pool_kv.stride(0), kv_pool_kv.stride(1), kv_pool_kv.stride(2),
        new_kv.stride(0), new_kv.stride(1),
        ape_score.stride(0), ape_score.stride(1),
        out_kv.stride(0), out_kv.stride(1), out_kv.stride(2),
        bs, T=T, D=D, HEAD_DIM=OUT_D, RATIO=ratio, OVERLAP=overlap,
        BLOCK_D=BLOCK_D,
    )
    return out_kv, out_score
