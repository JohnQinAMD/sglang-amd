"""CK Tile FP8 sparse MLA decode kernel adapter (V32 + 2604 / DSv4 shapes).

This module wraps the gfx950 CK Tile kernel that lives bundled at
``sglang/srt/layers/attention/csrc/ck_v32/``. Used by
``debug_flash_mla_adapter.py`` when:
  * ``SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1`` is set, AND
  * ``head_dim_v == 512`` AND
  * ``head_dim_qk in {576, 512}``

Two production shape specializations are emitted from the same templated
kernel — one runtime-dispatch on ``q.size(-1)`` selects:
  * 576: V32 / Pro V32 mode (DSv4-Pro non-2604, DSv4 V32 reference)
  * 512: 2604 mode (DSv4-Pro & Flash in ``SGLANG_DSV4_MODE=2604``)

The kernel beats the asm `.co` baseline by ~1.7-3.4× across the typical
B×topk decode grid; see ``CK_V32_RESULTS.md`` in the kernel-agents
experiments workspace for the full bench grid.

Build prerequisite (one-time per node): the C++/HIP kernel source must be
visible at the path pointed to by ``SGLANG_CK_V32_KERNEL_SRC_DIR``
(default: ``/mnt/vast/john/rocm-dynamo/aiter-amd/csrc/ck_mla_decode_sparse_fp8``).
First import compiles the kernel (~5 s); subsequent imports hit the
torch.utils.cpp_extension cache.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# Single-launch sink fold: out *= 1 / (1 + exp(sink - lse))
#
# Replaces the previous 3-4-launch torch chain (broadcast sub → exp → add → div
# → mul → cast) with one Triton kernel that streams `out` and writes it back in
# place. Kernel is launched per (total_q, head, V-tile); each program reads one
# (q, h)'s lse + sink scalar, broadcasts the resulting scale across the V tile.
#
# `total_q` is small (B*S_q, e.g. 1-8 in decode), `H=128`, `V=512` for V32
# DSv4-Pro — so a single 1D launch over (total_q*H) with 512 lanes per
# program is well-sized for MI355X (~5 µs typical → ~1-2 µs after fusion).
@triton.jit
def _sink_fold_inplace_kernel(
    out_ptr,      # [total_q, H, V] bf16, modified in place
    lse_ptr,      # [total_q, H]    fp32
    sink_ptr,     # [H]             fp32
    H: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)            # 0 .. total_q*H
    qh_id = pid                        # row index in flattened (total_q, H)
    h_id = qh_id % H

    # Per-row scale: 1 / (1 + exp(sink[h] - lse[q, h])).
    sink = tl.load(sink_ptr + h_id).to(tl.float32)
    lse = tl.load(lse_ptr + qh_id).to(tl.float32)
    scale = 1.0 / (1.0 + tl.exp(sink - lse))

    offs = tl.arange(0, BLOCK_V)
    base = qh_id * V
    for v_start in tl.static_range(0, V, BLOCK_V):
        cur = offs + v_start
        mask = cur < V
        x = tl.load(out_ptr + base + cur, mask=mask, other=0.0).to(tl.float32)
        x = x * scale
        tl.store(out_ptr + base + cur, x.to(tl.bfloat16), mask=mask)


def _apply_sink_fold_inplace(out_bf16: torch.Tensor, lse: torch.Tensor,
                              sink: torch.Tensor) -> None:
    """In-place: out_bf16[q, h, :] *= 1 / (1 + exp(sink[h] - lse[q, h]))."""
    assert out_bf16.dtype == torch.bfloat16
    total_q, H, V = out_bf16.shape
    assert lse.shape == (total_q, H), f"lse shape {tuple(lse.shape)} vs ({total_q},{H})"
    assert sink.shape == (H,), f"sink shape {tuple(sink.shape)} vs ({H},)"
    if not out_bf16.is_contiguous():
        out_bf16 = out_bf16.contiguous()
    if lse.dtype != torch.float32:
        lse = lse.float()
    if sink.dtype != torch.float32:
        sink = sink.float()
    if not lse.is_contiguous():
        lse = lse.contiguous()
    if not sink.is_contiguous():
        sink = sink.contiguous()

    # BLOCK_V=512 covers the full V dim in one program for V32 (V=512), so the
    # static_range loop is single-iter — single 512-wide vector load+store per
    # (q, h) row. For larger V the loop tiles automatically.
    BLOCK_V = min(512, V)
    grid = (total_q * H,)
    _sink_fold_inplace_kernel[grid](
        out_bf16, lse, sink, H=H, V=V, BLOCK_V=BLOCK_V,
    )

# V32 (DSv4-Pro) head dimensions; these define the V32 reference. The kernel
# is also instantiated for QK_HEAD_DIM=512 (2604 mode); SUPPORTED_QK_DIMS
# below mirrors the static dispatch in mla_decode_fwd.cu.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576 — V32 reference
V_HEAD_DIM = KV_LORA_RANK
HEAD_GROUPS = 4  # = num_heads (128) / TILE_HEADS (32) — kernel constant
SUPPORTED_QK_DIMS = (576, 512)

# Bundled kernel source ships with the SGLang Python package at
# sglang/srt/layers/attention/csrc/ck_v32/. Override via env var if you
# want to point at a development checkout (e.g. an in-flight aiter-amd
# branch with patched kernel source).
_BUNDLED_KERNEL_SRC = os.path.join(os.path.dirname(__file__), "csrc", "ck_v32")

_ck_mod = None


def _kernel_src_dir() -> str:
    return os.environ.get("SGLANG_CK_V32_KERNEL_SRC_DIR", _BUNDLED_KERNEL_SRC)


def _get_ck_mod():
    """JIT-build the CK Tile FP8 sparse module (cached per-process)."""
    global _ck_mod
    if _ck_mod is not None:
        return _ck_mod
    from torch.utils.cpp_extension import load

    src_dir = _kernel_src_dir()
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(
            f"CK V32 kernel source dir not found at {src_dir!r}. "
            "Set SGLANG_CK_V32_KERNEL_SRC_DIR to the directory containing "
            "mla_decode_fwd.cu and mla_decode_fwd_kernel.hpp."
        )

    old_arch = os.environ.get("PYTORCH_ROCM_ARCH", None)
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
    _ck_mod = load(
        name="ck_mla_decode_sparse_fp8",
        sources=[os.path.join(src_dir, "mla_decode_fwd.cu")],
        extra_include_paths=[src_dir],
        extra_cuda_cflags=["-O3", "-std=c++20"],
        verbose=False,
    )
    if old_arch is not None:
        os.environ["PYTORCH_ROCM_ARCH"] = old_arch
    else:
        os.environ.pop("PYTORCH_ROCM_ARCH", None)
    return _ck_mod


def pick_num_splits(B: int, topk: int) -> int:
    """Pick num_kv_splits so each split has ~1 BLOCK_N=32 worth of work AND
    total workgroup count stays in the [64, 512] sweet spot for MI355X."""
    BLOCK_N = 32
    splits_by_topk = max(1, topk // BLOCK_N)
    splits_by_total = max(1, 512 // (HEAD_GROUPS * B))
    splits = min(splits_by_topk, splits_by_total)
    while topk % splits != 0 and splits > 1:
        splits -= 1
    return splits


# --------- Persistent buffer caches ---------
# All keyed by shape + device. In steady-state serving the shapes don't
# change call-to-call, so cost per call is just dict lookup.
_indptr_cache: dict = {}
_split_buf_cache: dict = {}
_reduce_meta_cache: dict = {}


def _get_uniform_indptrs(B: int, topk: int, device: torch.device):
    """qo_indptr=[0,1,..,B] (one query per batch in decode) and
    kv_indptr=[0,topk,2*topk,...,B*topk]. Cached per (B, topk, device)."""
    key = (B, topk, str(device))
    cached = _indptr_cache.get(key)
    if cached is not None:
        return cached
    qo = torch.arange(B + 1, dtype=torch.int32, device=device)
    kv = torch.arange(0, (B + 1) * topk, topk, dtype=torch.int32, device=device)
    if len(_indptr_cache) > 32:
        _indptr_cache.clear()
    _indptr_cache[key] = (qo, kv)
    return qo, kv


def _get_split_buffers(total_q: int, num_splits: int, H: int, V: int,
                       device: torch.device):
    """Return (split_data, split_lse) cached per shape. Kernel writes all
    slots since pick_num_splits picks splits with ≥ BLOCK_N=32 rows of work,
    and `pidx < 0` rows zero-fill in-kernel — so we can leave the buffers
    uninitialized between calls."""
    key = (total_q, num_splits, H, V, str(device))
    cached = _split_buf_cache.get(key)
    if cached is not None:
        return cached
    sd = torch.empty((total_q, num_splits, H, V), dtype=torch.float32, device=device)
    sl = torch.empty((total_q, num_splits, H, 1), dtype=torch.float32, device=device)
    if len(_split_buf_cache) > 32:
        _split_buf_cache.clear()
    _split_buf_cache[key] = (sd, sl)
    return sd, sl


def _build_uniform_reduce_meta(total_q: int, num_splits: int, device: torch.device):
    """Build (reduce_indptr, reduce_final_map, reduce_partial_map) for our
    uniform decode split-K scheme. Cached per (total_q, num_splits, device).

    The CK reduce kernel (`aiter.mla_reduce_v1`) consumes:
      * reduce_indptr      [total_q+1] cumulative split count per tile
      * reduce_partial_map [total_q*splits] partial-slot index per contributor
      * reduce_final_map   [total_q, 2] {q_start, q_end} per tile (uniform decode → length-1 ranges)
    """
    key = (total_q, num_splits, str(device))
    cached = _reduce_meta_cache.get(key)
    if cached is not None:
        return cached
    reduce_indptr = torch.arange(
        0, (total_q + 1) * num_splits, num_splits,
        dtype=torch.int32, device=device,
    )
    reduce_partial_map = torch.arange(
        total_q * num_splits, dtype=torch.int32, device=device,
    )
    qs = torch.arange(total_q, dtype=torch.int32, device=device)
    reduce_final_map = torch.stack([qs, qs + 1], dim=-1).contiguous()
    out = (reduce_indptr, reduce_final_map, reduce_partial_map)
    if len(_reduce_meta_cache) > 32:
        _reduce_meta_cache.clear()
    _reduce_meta_cache[key] = out
    return out


def _ck_native_reduce(split_data: torch.Tensor, split_lse: torch.Tensor,
                       max_seqlen_q: int = 1):
    """Single-launch CK Tile stage-2 reduce via aiter.mla_reduce_v1."""
    import aiter as _aiter

    total_q, num_splits, H, V = split_data.shape
    device = split_data.device

    reduce_indptr, reduce_final_map, reduce_partial_map = _build_uniform_reduce_meta(
        total_q, num_splits, device,
    )

    partial_output = split_data.view(total_q * num_splits, 1, H, V)
    partial_lse = split_lse.view(total_q * num_splits, 1, H, 1)

    final_output = torch.empty((total_q, H, V), dtype=torch.bfloat16, device=device)
    final_lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

    _aiter.mla_reduce_v1(
        partial_output, partial_lse,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        max_seqlen_q, final_output, final_lse,
    )
    return final_output, final_lse


def ck_sparse_mla_decode_fp8_v32(
    q: torch.Tensor,                  # [B, S_q, H, D=576] bf16
    k_cache: torch.Tensor,            # FP8 KV pool, normalized to [N, 1, 1, 576]
    indices: torch.Tensor,            # [B, S_q, topk] int32; -1 entries → masked
    invalid_mask: torch.Tensor,       # [B*S_q, topk] bool — currently unused (kernel handles masking)
    attn_sink: Optional[torch.Tensor],   # [H] fp32 or None
    sm_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (output[B, S_q, H, V=512] bf16, lse[B, H, S_q] fp32)."""
    assert q.dim() == 4
    B, S_q, H, D = q.shape
    assert D in SUPPORTED_QK_DIMS, (
        f"expected qk_head_dim in {SUPPORTED_QK_DIMS} (V32=576 / 2604=512), got {D}"
    )

    topk = indices.shape[-1]
    device = q.device
    total_q = B * S_q

    # KV pool layout — Phase B+ (Apr 2026): the CK launcher now accepts the 4D
    # tensor directly with native (possibly padded) strides, so we no longer
    # `.reshape()`/`.contiguous()` here for the 4D case. The previous code
    # silently copied the whole fp8 pool (~131 ms / dispatch) because c4/c128
    # caches pad each row to a 576-B multiple, making the slice-then-view
    # output non-contiguous.
    if k_cache.dim() == 2:
        n_kv, slot_stride = k_cache.shape
        kv_view = k_cache
        if not kv_view.is_contiguous():
            kv_view = kv_view.contiguous()
    elif k_cache.dim() == 3:
        n_kv = k_cache.shape[0] * k_cache.shape[1]
        slot_stride = k_cache.shape[-1]
        kv_view = k_cache.reshape(n_kv, slot_stride)
        if not kv_view.is_contiguous():
            kv_view = kv_view.contiguous()
    elif k_cache.dim() == 4:
        num_pages, page_size, h_kv, slot_stride = k_cache.shape
        assert h_kv == 1, f"expected h_kv=1, got k_cache shape {tuple(k_cache.shape)}"
        n_kv = num_pages * page_size
        # Pass the 4D tensor as-is; the launcher reads .stride(0)/.stride(1)
        # and supports padded pool rows natively.
        kv_view = k_cache
    else:
        raise AssertionError(f"unexpected k_cache shape: {tuple(k_cache.shape)}")

    assert slot_stride >= D, (
        f"k_cache slot stride {slot_stride} < q's d_qk={D}; kernel would read OOB"
    )

    # The CK kernel handles `pidx < 0` natively — invalid rows zero-fill in
    # the LDS tile loader, so we don't need to clamp / masked_fill here.
    idx_flat = indices.to(torch.int32).reshape(B * topk)
    if not idx_flat.is_contiguous():
        idx_flat = idx_flat.contiguous()

    qo_indptr, kv_indptr = _get_uniform_indptrs(B, topk, device)

    num_splits = pick_num_splits(B, topk)
    split_data, split_lse = _get_split_buffers(
        total_q, num_splits, H, V_HEAD_DIM, device,
    )

    q_2d = (
        q.view(total_q, H, D).contiguous()
        if q.is_contiguous()
        else q.reshape(total_q, H, D)
    )

    ck = _get_ck_mod()
    # FP8 decode scale: the gfx950 `cvt_pk_f32_fp8` HW intrinsic decodes bytes
    # with e4m3fn semantics (bias=7). When KV is stored as torch.float8_e4m3fn
    # (the MI355X default — `is_fp8_fnuz()=False` on gfx950) the HW reading
    # matches storage exactly, no fold needed → scale=1.0. When stored as
    # fnuz (MI300X gfx942 default, bias=8), the HW gives 2× the fnuz value →
    # scale=0.5 to fold back. Inspect the original-dtype tensor (k_cache may
    # be reinterpreted as uint8 for the kernel call but k_cache_orig keeps
    # the dtype info).
    _kv_dtype = k_cache.dtype
    if _kv_dtype == torch.uint8:
        # Caller already reinterpreted as uint8; assume fn (MI355X / OCP standard).
        # Pro V32 / Flash 2604 on MI355X both store as fp8_e4m3fn upstream.
        fp8_decode_scale = 1.0
    elif _kv_dtype == torch.float8_e4m3fn:
        fp8_decode_scale = 1.0
    elif hasattr(torch, "float8_e4m3fnuz") and _kv_dtype == torch.float8_e4m3fnuz:
        fp8_decode_scale = 0.5
    else:
        # Unknown dtype — fall back to legacy 0.5 for backward compat.
        fp8_decode_scale = 0.5
    ck.mla_decode_fwd_ck_sparse_fp8(
        q_2d, kv_view, split_data, split_lse,
        qo_indptr, kv_indptr, idx_flat,
        float(sm_scale), int(num_splits), float(fp8_decode_scale),
    )

    if num_splits == 1:
        # Kernel pre-normalizes for single-split case — no reduce needed.
        out = split_data[:, 0, :, :]
        lse = split_lse[:, 0, :, 0]
        out_bf16 = out.to(torch.bfloat16)
    else:
        out_bf16, lse = _ck_native_reduce(split_data, split_lse, max_seqlen_q=S_q)

    # Lonely-Q queries (all-invalid indices) come out of the kernel as exactly
    # zero — `pidx < 0` rows are zero-filled in the LDS tile, so Q@K=0 and
    # P@V=0. No wrapper correction needed.

    # Attention sink correction: out *= 1 / (1 + exp(sink - lse)).
    #
    # Fused into a single Triton kernel (`_sink_fold_inplace_kernel`) — one
    # launch instead of the previous 3-4-kernel torch chain (broadcast sub →
    # exp → add → div → mul → cast). The kernel streams `out_bf16` in place
    # so we avoid the f32 round-trip and the extra HBM allocation.
    if attn_sink is not None:
        # `out_bf16` may not be contiguous in the num_splits==1 fast path
        # (it's a `.to(bfloat16)` of a strided view of split_data), so make
        # it contiguous first — the kernel writes in place.
        if not out_bf16.is_contiguous():
            out_bf16 = out_bf16.contiguous()
        _apply_sink_fold_inplace(out_bf16, lse, attn_sink)

    out_bf16 = out_bf16.view(B, S_q, H, V_HEAD_DIM)
    lse_bhs = lse.view(B, S_q, H).transpose(1, 2).contiguous()  # [B, H, S_q]
    return out_bf16, lse_bhs


# ─────────────────────────────────────────────────────────────────────────────
# Phase A two-source combine helpers (2026-04-28)
# ─────────────────────────────────────────────────────────────────────────────
# Replaces the (2× ck_sparse_mla_decode_fp8_v32 + 2× aiter.mla_reduce_v1 +
# Triton merge_two_sparse_attn_outputs + Triton _sink_fold_inplace_kernel)
# pipeline with (2× ck_sparse_mla_decode_fp8_v32_to_split + 1× CK combine).
#
# Compared to the pre-Phase-A pipeline:
#   * Skips the 2× per-source `aiter.mla_reduce_v1` calls — the new combine
#     consumes both sources' raw split tensors directly and does the N-way
#     merge in one launch.
#   * Skips the Triton merge kernel + 4× `.contiguous()` reshape work in
#     `debug_flash_mla_adapter.py` — combine reads strided split tensors.
#   * Optionally fuses the `_sink_fold_inplace_kernel` into the same launch
#     (toggle via the `attn_sink` arg).
#   * Net: 8+ Triton/aiter launches per layer-pass → 1 CK launch + 2 splitkv
#     launches. Eliminates the cuda-graph re-record + small-launch dispatch
#     overhead that caused the +131 ms elementwise regression in Phase B.


def ck_sparse_mla_decode_fp8_v32_to_split(
    q: torch.Tensor,                  # [B, S_q, H, D] bf16
    k_cache: torch.Tensor,            # FP8 KV pool
    indices: torch.Tensor,            # [B, S_q, topk] int32
    invalid_mask: torch.Tensor,       # [B*S_q, topk] bool — unused (kernel masks)
    sm_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run CK V32 splitkv ONLY — return raw split tensors without reduce.

    Used by the Phase A two-shot path. The caller will pass split_data /
    split_lse from BOTH sources (main + extra) into the new ``mla_combine_fwd_ck``
    which does the N-way online-softmax merge in one launch.

    Returns (split_data [total_q, num_splits, H, V] fp32,
             split_lse  [total_q, num_splits, H, 1] fp32).

    Both tensors live in the per-shape `_split_buf_cache`; the caller MUST
    consume them before issuing another `..._to_split` call at the same
    (total_q, num_splits, H, V) signature, otherwise the buffer will be
    overwritten by the next splitkv pass. For the two-shot path (main + extra
    have different num_splits typically) this is naturally avoided because
    `pick_num_splits(B, topk_main) != pick_num_splits(B, topk_extra)` keys the
    cache to different buffers.
    """
    assert q.dim() == 4
    B, S_q, H, D = q.shape
    assert D in SUPPORTED_QK_DIMS, (
        f"expected qk_head_dim in {SUPPORTED_QK_DIMS}, got {D}"
    )

    topk = indices.shape[-1]
    device = q.device
    total_q = B * S_q

    # (KV-buffer normalization mirrors `ck_sparse_mla_decode_fp8_v32` — keep
    # the two helpers in sync.) Phase B+ (Apr 2026): pass 4D padded pool
    # tensors as-is — the C++ launcher reads .stride(0)/.stride(1) and
    # handles pool-row padding natively, eliminating a 131 ms float8_copy
    # that previously fired on every two-shot dispatch.
    if k_cache.dim() == 2:
        n_kv, slot_stride = k_cache.shape
        kv_view = k_cache
        if not kv_view.is_contiguous():
            kv_view = kv_view.contiguous()
    elif k_cache.dim() == 3:
        n_kv = k_cache.shape[0] * k_cache.shape[1]
        slot_stride = k_cache.shape[-1]
        kv_view = k_cache.reshape(n_kv, slot_stride)
        if not kv_view.is_contiguous():
            kv_view = kv_view.contiguous()
    elif k_cache.dim() == 4:
        num_pages, page_size, h_kv, slot_stride = k_cache.shape
        assert h_kv == 1
        n_kv = num_pages * page_size
        kv_view = k_cache
    else:
        raise AssertionError(f"unexpected k_cache shape: {tuple(k_cache.shape)}")
    assert slot_stride >= D

    idx_flat = indices.to(torch.int32).reshape(B * topk)
    if not idx_flat.is_contiguous():
        idx_flat = idx_flat.contiguous()

    qo_indptr, kv_indptr = _get_uniform_indptrs(B, topk, device)
    num_splits = pick_num_splits(B, topk)
    split_data, split_lse = _get_split_buffers(
        total_q, num_splits, H, V_HEAD_DIM, device,
    )

    q_2d = (
        q.view(total_q, H, D).contiguous()
        if q.is_contiguous()
        else q.reshape(total_q, H, D)
    )

    # FP8 decode scale dispatch (kept identical to the reduce-fused path).
    _kv_dtype = k_cache.dtype
    if _kv_dtype == torch.uint8 or _kv_dtype == torch.float8_e4m3fn:
        fp8_decode_scale = 1.0
    elif hasattr(torch, "float8_e4m3fnuz") and _kv_dtype == torch.float8_e4m3fnuz:
        fp8_decode_scale = 0.5
    else:
        fp8_decode_scale = 0.5

    ck = _get_ck_mod()
    ck.mla_decode_fwd_ck_sparse_fp8(
        q_2d, kv_view, split_data, split_lse,
        qo_indptr, kv_indptr, idx_flat,
        float(sm_scale), int(num_splits), float(fp8_decode_scale),
    )
    return split_data, split_lse


def ck_combine_two_splits(
    split_data_a: torch.Tensor,       # [total_q, S_a, H, V] fp32
    split_lse_a: torch.Tensor,        # [total_q, S_a, H, 1] fp32
    split_data_b: Optional[torch.Tensor] = None,  # [total_q, S_b, H, V] fp32
    split_lse_b: Optional[torch.Tensor] = None,   # [total_q, S_b, H, 1] fp32
    attn_sink: Optional[torch.Tensor] = None,     # [H] fp32 or None
    out: Optional[torch.Tensor] = None,           # [total_q, H, V] bf16
    lse: Optional[torch.Tensor] = None,           # [total_q, H]    fp32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-launch CK Tile combine of one or two split-K sources.

    For the two-source case (Phase A two-shot decode), pass split_data_b /
    split_lse_b from the second CK splitkv. For single-source N-way reduce,
    leave them as None (kernel reduces over source-A splits only — equivalent
    to ``aiter.mla_reduce_v1`` without the metadata buffer overhead).

    Output tensors are caller-allocated when provided (saves the per-call
    allocation that churns the caching allocator and helps cuda-graph capture).

    Returns (out [total_q, H, V] bf16, lse [total_q, H] fp32).
    """
    total_q, S_a, H, V = split_data_a.shape
    device = split_data_a.device
    if out is None:
        out = torch.empty((total_q, H, V), dtype=torch.bfloat16, device=device)
    else:
        assert out.shape == (total_q, H, V) and out.dtype == torch.bfloat16
        assert out.stride(-1) == 1
    if lse is None:
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)
    else:
        assert lse.shape == (total_q, H) and lse.dtype == torch.float32

    ck = _get_ck_mod()
    ck.mla_combine_fwd_ck(
        split_data_a, split_lse_a,
        split_data_b, split_lse_b,
        attn_sink,
        out, lse,
    )
    return out, lse
