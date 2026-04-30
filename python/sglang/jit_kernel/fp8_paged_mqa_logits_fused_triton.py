# SPDX-License-Identifier: MIT
# Fused Triton port of `fp8_paged_mqa_logits_torch` (indexer.py:80-236).
#
# v2 (2026-04-26 evening, kernel-agents task
# dsv4_fp8_paged_mqa_logits_triton_integration): integration optimizations
# applied on top of v1 (which had 2.5× kernel-level speedup but -9% E2E
# regression due to wrapper overhead).
#
# v2 fixes:
#   1. Persistent scratch buffers for `out` (mirrors torch path's
#      `_FP8_PAGED_SCRATCH_CAPTURED`/`_EAGER` pattern) — eliminates per-call
#      torch.empty for the output.
#   2. Inline q_fp8 → fp32 cast inside the kernel (load uint8, bitcast to
#      tl.float8e4nv, cast to fp32) — eliminates the external
#      `q_fp8[:, 0].to(torch.float32).contiguous()` launch.
#   3. Skip `.contiguous()` on the kvcache uint8 view (the underlying buffer
#      is already contiguous as packed by the KV pool) and remove the
#      separate -inf pre-fill kernel (the main kernel writes -inf for both
#      `t >= seq_lens[b]` and `t >= eff_max_seq_len` positions in its grid
#      coverage; positions ≥ max_pages*BLOCK_SIZE never get a chance to
#      contain stale data because we size `out` to exactly that bound).
#
# Replaces a 14-op torch chain with a single Triton launch:
#   q_fp8 -> q_f32                         (cast — now INLINE in kernel)
#   index_select(kvcache, page_table)      (gather — INLINE)
#   .contiguous().view().to(f32)           (cast — INLINE)
#   torch.bmm(value_f32, q_f32.T)          (bmm — INLINE)
#   torch.relu(score)                      (elementwise — INLINE)
#   score * weight.unsqueeze(1)            (broadcast mul — INLINE)
#   score.sum(dim=2)                       (reduce — INLINE)
#   score * scale                          (elementwise — INLINE)
#   torch.arange(padded_seq_len, ...)      (arange — derived from program_id)
#   positions.unsqueeze(0) < seq_lens...   (bool compare — INLINE)
#   torch.where(valid, score, -inf)        (mask — INLINE)
#   out.fill_(-inf); out[:, :fill] = score (fill+slice — written in main kernel)
#
# Layout assumptions (matches indexer.py:fp8_paged_mqa_logits_torch):
#   q_fp8:        [B, 1, H, D]           fp8 (e4m3fn, OCP standard on gfx950)
#   kvcache_fp8:  [num_blocks, BLOCK_SIZE, 1, D + 4]  fp8-typed buffer
#                  (per-page layout: BLOCK_SIZE × D fp8 bytes,
#                   then BLOCK_SIZE × 4 bytes of fp32 scales)
#   weight:       [B, H]                 fp32
#   seq_lens:     [B]                    int (int32 or int64; kernel converts)
#   page_table:   [B, max_pages]         int (block ids)
#   out:          [B, effective_max_seq_len] fp32 (filled in-place by kernel)

import os
from typing import Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# v2 FIX #1: persistent scratch dicts (mirrors torch path's pattern).
#
# `_CAPTURED` is allocated during `torch.cuda.is_current_stream_capturing()`
# (cuda-graph capture window). Its tensors are baked into the captured kernels
# by physical address and MUST NEVER be reassigned, or replay reads freed
# memory → HSA 0x29 aperture violation.
# `_EAGER` services every other (eager / non-captured) call and is free to
# reallocate.
#
# We key by (batch_size, eff_max_seq_len, dtype, device) so a re-shape between
# calls (e.g., when bs=1 captures vs bs=8 captures share scratch) doesn't
# overwrite the stable address.
# ---------------------------------------------------------------------------
_FP8_FUSED_SCRATCH_CAPTURED: dict = {}
_FP8_FUSED_SCRATCH_EAGER: dict = {}


def _ensure_scratch(scratch_dict, name, shape, dtype, device):
    cur = scratch_dict.get(name)
    target = tuple(shape)
    if (
        cur is None
        or cur.dtype != dtype
        or cur.device != device
        or tuple(cur.shape) != target
    ):
        scratch_dict[name] = torch.empty(target, dtype=dtype, device=device)
    return scratch_dict[name]


# ---------------------------------------------------------------------------
# Triton kernel — v2: loads q from FP8 directly (inline cast), writes -inf
# for invalid positions inline (no separate fill).
# ---------------------------------------------------------------------------
@triton.jit
def _fp8_paged_mqa_logits_fused_kernel(
    # Inputs (q is now fp8/uint8 — v2 FIX #2)
    q_u8_ptr,           # [B, H, D]  uint8 (bitcast view of q_fp8[:, 0])
    kvcache_u8_ptr,     # uint8* — packed KV-cache, byte addressed
    weight_ptr,         # [B, H]     fp32
    seq_lens_ptr,       # [B]        int32 or int64 (kernel converts)
    page_table_ptr,     # [B, max_pages] int32 (block ids)
    # Output
    out_ptr,            # [B, effective_max_seq_len] fp32
    # Strides / dims (in element units for typed ptrs, byte units for u8 ptrs)
    eff_max_seq_len: tl.int32,    # output dim; positions ≥ this are not written
    max_pages: tl.int32,          # page_table dim
    block_size_x_bpt: tl.int64,   # bytes per page = BLOCK_SIZE * (D + 4)
    out_stride_b: tl.int64,
    page_table_stride_b: tl.int64,
    q_stride_b_bytes: tl.int64,   # q stride in BYTES (= H * D for contiguous fp8 q)
    # constexpr
    H: tl.constexpr,              # 64 for DSv4
    D: tl.constexpr,              # 128 for DSv4 indexer KV
    BLOCK_SIZE: tl.constexpr,     # 64 (per-page tokens)
    BLOCK_T: tl.constexpr,        # tokens per program (e.g. 32 or 64)
    SCALE_OFFSET_BYTES: tl.constexpr,  # BLOCK_SIZE * D — start of scales in page (bytes)
):
    """One program emits scores for BLOCK_T contiguous output positions for
    one batch, reading FP8 KV via the page_table.

    Output (within the kernel grid coverage [0, max_pages*BLOCK_SIZE)):
      out[b, t] for t < min(seq_lens[b], eff_max_seq_len):  computed score
      out[b, t] for seq_lens[b] ≤ t < eff_max_seq_len:      -inf
      out[b, t] for t ≥ eff_max_seq_len:                    not written

    Caller sizes `out` to exactly [B, eff_max_seq_len] so positions outside
    the grid coverage simply don't exist; no separate -inf fill is needed.
    """
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    # -- Output token range ---------------------------------------------------
    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    out_mask = t_offsets < eff_max_seq_len

    # -- Per-batch scalars ---------------------------------------------------
    seq_len = tl.load(seq_lens_ptr + pid_b).to(tl.int32)

    # -- Load q from FP8 bytes [H, D], bitcast → fp32 (v2 FIX #2) ------------
    h_offs = tl.arange(0, H)
    d_offs = tl.arange(0, D)
    q_u8 = tl.load(
        q_u8_ptr + pid_b * q_stride_b_bytes + h_offs[:, None] * D + d_offs[None, :]
    )  # [H, D] uint8
    # tl.float8e4nv = OCP e4m3 (bias 7) — matches torch.float8_e4m3fn
    # which is what the model uses on gfx950 (`is_fp8_fnuz()=False`).
    q_fp8 = q_u8.to(tl.float8e4nv, bitcast=True)
    q = q_fp8.to(tl.float32)  # [H, D]

    # -- Load weight for this batch [H] (fp32) -------------------------------
    w = tl.load(weight_ptr + pid_b * H + h_offs).to(tl.float32)  # [H]

    # -- Resolve token → page_id, pos_in_page --------------------------------
    page_idx = t_offsets // BLOCK_SIZE          # [BLOCK_T]
    pos_in_page = t_offsets % BLOCK_SIZE        # [BLOCK_T]
    page_idx_clamped = tl.minimum(page_idx, max_pages - 1)
    page_id = tl.load(
        page_table_ptr + pid_b * page_table_stride_b + page_idx_clamped,
        mask=out_mask,
        other=0,
    ).to(tl.int64)  # [BLOCK_T]

    # -- Byte offsets into kvcache_u8_ptr ------------------------------------
    page_base = page_id * block_size_x_bpt
    value_base = page_base + pos_in_page.to(tl.int64) * D
    scale_base = page_base + SCALE_OFFSET_BYTES + pos_in_page.to(tl.int64) * 4

    # -- Load FP8 KV value vector + bitcast → fp32 ---------------------------
    v_u8 = tl.load(
        kvcache_u8_ptr + value_base[:, None] + d_offs[None, :],
        mask=out_mask[:, None],
        other=0,
    )  # [BLOCK_T, D] uint8
    v_fp8 = v_u8.to(tl.float8e4nv, bitcast=True)
    v_f32 = v_fp8.to(tl.float32)  # [BLOCK_T, D]

    # -- Load fp32 scale (4 bytes assembled from uint8) ----------------------
    # NOTE: v3 attempt switched to a typed fp32 ptr load — measured 4 us
    # SLOWER than this byte-assemble pattern (37 vs 33 us at b=1 ISL=4096).
    # Triton's compiler already coalesces the 4 sequential aligned byte loads
    # into a single 32-bit access, so the shifted-OR assemble is just register
    # ops. Keeping this form.
    s_b0 = tl.load(kvcache_u8_ptr + scale_base + 0, mask=out_mask, other=0).to(tl.uint32)
    s_b1 = tl.load(kvcache_u8_ptr + scale_base + 1, mask=out_mask, other=0).to(tl.uint32)
    s_b2 = tl.load(kvcache_u8_ptr + scale_base + 2, mask=out_mask, other=0).to(tl.uint32)
    s_b3 = tl.load(kvcache_u8_ptr + scale_base + 3, mask=out_mask, other=0).to(tl.uint32)
    s_u32 = s_b0 | (s_b1 << 8) | (s_b2 << 16) | (s_b3 << 24)
    scale = s_u32.to(tl.float32, bitcast=True)  # [BLOCK_T]

    # -- BMM: [BLOCK_T, D] @ [D, H] = [BLOCK_T, H] ---------------------------
    qk = tl.dot(v_f32, tl.trans(q))  # [BLOCK_T, H]

    # -- relu, weight-mul, sum(dim=H), scale --------------------------------
    qk = tl.maximum(qk, 0.0)
    qk_w = qk * w[None, :]
    score = tl.sum(qk_w, axis=1)
    score = score * scale

    # -- Mask: t >= seq_len → -inf (also covers t ≥ eff_max_seq_len via out_mask)
    valid = (t_offsets < seq_len) & out_mask
    score = tl.where(valid, score, float("-inf"))

    # -- Write to out[b, t] -------------------------------------------------
    tl.store(
        out_ptr + pid_b * out_stride_b + t_offsets,
        score,
        mask=out_mask,
    )


# ---------------------------------------------------------------------------
# Python wrapper — v2: persistent scratch + inline cast + skip-contiguous +
# no separate -inf fill kernel.
# ---------------------------------------------------------------------------
def fp8_paged_mqa_logits_fused_triton(
    q_fp8: torch.Tensor,                # [B, 1, H, D] fp8 (e4m3fn)
    kvcache_fp8: torch.Tensor,          # [num_blocks, BLOCK_SIZE, 1, D+4] fp8-typed
    weight: torch.Tensor,               # [B, H] fp32
    seq_lens: torch.Tensor,             # [B] int (int32 or int64)
    page_table: torch.Tensor,           # [B, max_pages] int32
    deep_gemm_metadata,                 # ignored (kept for signature compat)
    max_seq_len: int,
    clean_logits: bool = True,
    *,
    out: Optional[torch.Tensor] = None,
    block_t: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """Fused-Triton drop-in for `fp8_paged_mqa_logits_torch`. v2 with
    integration optimizations (persistent scratch, inline cast, no fill kernel).

    Returns: [B, effective_max_seq_len] fp32 score buffer where:
      * out[b, t] = sum_h(relu(<q[b,h,:], kv[b,t,:]>) * weight[b,h]) * scale[b,t]
        for t < seq_lens[b]
      * out[b, t] = -inf for seq_lens[b] ≤ t < effective_max_seq_len

    If `out` is provided the caller controls the buffer width and address.
    Otherwise we route through the persistent scratch dict (graph-pool stable
    across cuda-graph captures, mirrors the torch reference's pattern).
    """
    _ = deep_gemm_metadata
    _ = clean_logits

    batch_size, next_n, num_heads, head_dim = q_fp8.shape
    assert next_n == 1, f"Only next_n=1 supported, got {next_n}"
    assert num_heads == 64, f"Kernel assumes H=64 (DSv4 indexer); got H={num_heads}"
    assert head_dim == 128, f"Kernel assumes D=128 (DSv4 indexer); got D={head_dim}"
    num_blocks, block_size, h_kv, bytes_per_token = kvcache_fp8.shape
    assert h_kv == 1
    assert block_size == 64, f"Kernel assumes BLOCK_SIZE=64; got {block_size}"
    assert bytes_per_token == head_dim + 4, (
        f"Kernel assumes per-token = D + 4 fp32 scale bytes; "
        f"got bytes_per_token={bytes_per_token}"
    )
    device = q_fp8.device

    # -- Bound math (matches torch reference) -------------------------------
    max_pages = page_table.shape[1]
    eff_max_seq_len = min(max_seq_len, max_pages * block_size)

    # -- v2 FIX #1: persistent scratch for `out` ----------------------------
    if out is None:
        scratch = (
            _FP8_FUSED_SCRATCH_CAPTURED
            if torch.cuda.is_current_stream_capturing()
            else _FP8_FUSED_SCRATCH_EAGER
        )
        out = _ensure_scratch(
            scratch, "out_fused",
            (batch_size, eff_max_seq_len),
            torch.float32, device,
        )
    else:
        eff_max_seq_len = out.shape[1]

    # -- v2 FIX #2: q stays as fp8 (uint8 view) — kernel does the cast -----
    # `q_fp8[:, 0]` returns [B, H, D] fp8; view as uint8 for byte-addressed
    # load + bitcast inside the kernel.
    q_u8 = q_fp8[:, 0].view(torch.uint8)
    # Stride: contiguous q has stride(0) = H*D bytes. View doesn't change strides.
    # Element stride is 1 byte (uint8). For [B, H, D] uint8: stride(0) = H*D.
    q_stride_b_bytes = q_u8.stride(0)  # in uint8 elements (= bytes)

    # -- v2 FIX #3a: skip .contiguous() on kvcache uint8 view --------------
    # kvcache_fp8 was allocated as a flat contiguous buffer by the KV pool.
    # `.view(torch.uint8)` is metadata-only (fp8 is 1 byte = uint8 byte-wise).
    # We assert contiguity once instead of forcing a copy each call.
    assert kvcache_fp8.is_contiguous(), (
        "kvcache_fp8 must be contiguous; got strides "
        f"{kvcache_fp8.stride()} / shape {kvcache_fp8.shape}"
    )
    kvcache_u8 = kvcache_fp8.view(torch.uint8)

    # -- Strides --
    out_stride_b = out.stride(0)
    page_table_stride_b = page_table.stride(0)
    block_size_x_bpt = block_size * bytes_per_token  # bytes per page

    # -- Launch (v2 FIX #3b: no separate _fill_neg_inf kernel; main kernel
    # writes -inf inline for both invalid (t ≥ seq_len) and out-of-bounds
    # (t ≥ eff_max_seq_len) positions in its grid coverage)
    n_token_blocks = triton.cdiv(max_pages * block_size, block_t)
    grid = (batch_size, n_token_blocks)
    _fp8_paged_mqa_logits_fused_kernel[grid](
        q_u8, kvcache_u8, weight, seq_lens, page_table, out,
        eff_max_seq_len,
        max_pages,
        block_size_x_bpt,
        out_stride_b, page_table_stride_b, q_stride_b_bytes,
        H=num_heads, D=head_dim,
        BLOCK_SIZE=block_size,
        BLOCK_T=block_t,
        SCALE_OFFSET_BYTES=block_size * head_dim,
        num_warps=num_warps, num_stages=num_stages,
    )

    return out
