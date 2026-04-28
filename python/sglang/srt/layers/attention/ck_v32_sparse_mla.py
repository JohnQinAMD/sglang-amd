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


# Single-launch sink fold + lse transpose:
#
#     out_bf16[q, h, v] *= 1 / (1 + exp(sink[h] - lse[q, h]))
#     lse_bhs[b, h, s_q] = lse_qh[b * S_q + s_q, h]                  (transpose view)
#
# Replaces the (3-4-launch torch chain → 1 sink-fold launch → 1 strided-copy
# launch) pre-Phase-E pipeline with a SINGLE Triton kernel. The original
# `_sink_fold_inplace_kernel` only handled the sink path; the strided lse
# transpose `lse_bhs.copy_(lse_qh.view(B,S_q,H).transpose(1,2))` was a
# separate launch (~3-5 µs). Folding them halves post-kernel launches.
#
# Phase E Lever 1 (2026-04-28): renamed and extended. Backward-compatible
# `_apply_sink_fold_inplace` is preserved as a thin wrapper for the no-sink
# decode path that doesn't need the lse transpose either.
@triton.jit
def _sink_fold_and_lse_transpose_kernel(
    out_ptr,        # [total_q, H, V] bf16, modified in place
    lse_qh_ptr,     # [total_q, H]    fp32, source lse in (q, h) layout
    sink_ptr,       # [H]             fp32  (or null when HAS_SINK=0)
    lse_bhs_ptr,    # [B, H, S_q]     fp32  output (transposed)
    stride_lse_bhs_b, stride_lse_bhs_h, stride_lse_bhs_s,
    B: tl.constexpr,
    S_q: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_LSE_TRANSPOSE: tl.constexpr,
):
    """One program per (q, h) row. Streams V in BLOCK_V tiles in-place.

    At V=512, BLOCK_V=512: single iter; the kernel is functionally one fused
    pass over the (q, h, V) row + one fp32 lse store at v_block==0.
    """
    qh_id = tl.program_id(0)
    h_id = qh_id % H
    q_id = qh_id // H

    lse = tl.load(lse_qh_ptr + qh_id).to(tl.float32)

    if HAS_SINK:
        sink = tl.load(sink_ptr + h_id).to(tl.float32)
        scale = 1.0 / (1.0 + tl.exp(sink - lse))
    else:
        scale = 1.0  # no-op; loop body skipped if scale is constant 1 — but we
        # still need to keep the loop for the fp32→bf16 cast path; eager
        # decode (num_splits=1, no sink) uses a different `out_bf16_cached.copy_`
        # path which is already a fp32→bf16 cast in the dispatcher.

    if HAS_SINK:
        offs = tl.arange(0, BLOCK_V)
        base = qh_id * V
        for v_start in tl.static_range(0, V, BLOCK_V):
            cur = offs + v_start
            mask = cur < V
            x = tl.load(out_ptr + base + cur, mask=mask, other=0.0).to(tl.float32)
            x = x * scale
            tl.store(out_ptr + base + cur, x.to(tl.bfloat16), mask=mask)

    # Transposed lse store: lse_bhs[b, h, s_q] = lse_qh[q_id=b*S_q+s_q, h]
    if HAS_LSE_TRANSPOSE:
        b_id = q_id // S_q
        s_id = q_id % S_q
        tl.store(
            lse_bhs_ptr
            + b_id * stride_lse_bhs_b
            + h_id * stride_lse_bhs_h
            + s_id * stride_lse_bhs_s,
            lse,
        )


@triton.jit
def _split0_to_bf16_with_sink_and_lse_transpose_kernel(
    split_data_ptr,   # [total_q, num_splits, H, V] fp32 — only [:, 0, ...] is read
    out_ptr,          # [total_q, H, V] bf16 — written
    lse_qh_ptr,       # [total_q, H] fp32 — read
    sink_ptr,         # [H] fp32 (or null when HAS_SINK=0)
    lse_bhs_ptr,      # [B, H, S_q] fp32 — written when HAS_LSE_TRANSPOSE=1
    stride_sd_q, stride_sd_s, stride_sd_h,
    stride_lse_bhs_b, stride_lse_bhs_h, stride_lse_bhs_s,
    B: tl.constexpr,
    S_q: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    BLOCK_V: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_LSE_TRANSPOSE: tl.constexpr,
):
    """Phase E Lever 1 part 2 — fuse 3 launches into 1.

    Reads split_data[q, 0, h, v] fp32, writes out_bf16[q, h, v] = bf16(x * scale)
    where scale = 1/(1+exp(sink[h]-lse[q,h])) (or 1.0 if no sink), and writes
    lse_bhs[b, h, s_q] = lse[q, h] in the same launch.

    Replaces (out_bf16.copy_(split[:,0]) + _apply_sink_fold_inplace) when
    num_splits==1 and sink is present.
    """
    qh_id = tl.program_id(0)
    h_id = qh_id % H
    q_id = qh_id // H

    lse = tl.load(lse_qh_ptr + qh_id).to(tl.float32)

    if HAS_SINK:
        sink = tl.load(sink_ptr + h_id).to(tl.float32)
        scale = 1.0 / (1.0 + tl.exp(sink - lse))
    else:
        scale = 1.0

    offs = tl.arange(0, BLOCK_V)
    sd_base = q_id * stride_sd_q + 0 * stride_sd_s + h_id * stride_sd_h
    out_base = qh_id * V
    for v_start in tl.static_range(0, V, BLOCK_V):
        cur = offs + v_start
        mask = cur < V
        x = tl.load(split_data_ptr + sd_base + cur, mask=mask, other=0.0)
        x = x * scale
        tl.store(out_ptr + out_base + cur, x.to(tl.bfloat16), mask=mask)

    if HAS_LSE_TRANSPOSE:
        b_id = q_id // S_q
        s_id = q_id % S_q
        tl.store(
            lse_bhs_ptr
            + b_id * stride_lse_bhs_b
            + h_id * stride_lse_bhs_h
            + s_id * stride_lse_bhs_s,
            lse,
        )


def _apply_split0_cast_with_sink_and_lse_transpose(
    split_data: torch.Tensor,
    out_bf16: torch.Tensor,
    lse_qh: torch.Tensor,
    sink: Optional[torch.Tensor],
    lse_bhs_out: torch.Tensor,
    B: int,
    S_q: int,
) -> None:
    """Phase E Lever 1 part 2 — single-launch fused fp32→bf16 cast + sink + lse transpose.

    Args:
        split_data:   [total_q, num_splits, H, V] fp32 — kernel writes only [:, 0, ...]
        out_bf16:     [total_q, H, V] bf16 — written directly by this kernel
        lse_qh:       [total_q, H] fp32
        sink:         [H] fp32 or None
        lse_bhs_out:  [B, H, S_q] fp32 — written
        B, S_q:       transpose split (q_id = b * S_q + s_q)
    """
    total_q, H, V = out_bf16.shape
    assert split_data.shape[0] == total_q
    assert split_data.shape[2] == H
    assert split_data.shape[3] == V
    assert lse_qh.shape == (total_q, H)
    assert lse_bhs_out.shape == (B, H, S_q)
    assert B * S_q == total_q

    has_sink = sink is not None
    if has_sink:
        if sink.dtype != torch.float32:
            sink = sink.float()
        if not sink.is_contiguous():
            sink = sink.contiguous()

    BLOCK_V = min(512, V)
    grid = (total_q * H,)
    _split0_to_bf16_with_sink_and_lse_transpose_kernel[grid](
        split_data, out_bf16, lse_qh,
        sink if has_sink else lse_qh,
        lse_bhs_out,
        split_data.stride(0), split_data.stride(1), split_data.stride(2),
        lse_bhs_out.stride(0), lse_bhs_out.stride(1), lse_bhs_out.stride(2),
        B=B, S_q=S_q, H=H, V=V, BLOCK_V=BLOCK_V,
        HAS_SINK=has_sink,
        HAS_LSE_TRANSPOSE=True,
    )


def _apply_sink_fold_inplace(
    out_bf16: torch.Tensor,
    lse: torch.Tensor,
    sink: torch.Tensor,
    lse_bhs_out: Optional[torch.Tensor] = None,
    B: Optional[int] = None,
    S_q: Optional[int] = None,
) -> None:
    """In-place sink fold + (optional) lse-transpose write.

    Args:
        out_bf16: [total_q, H, V] bf16, modified in place: out *= 1/(1+exp(sink-lse))
        lse:      [total_q, H] fp32 source.
        sink:     [H] fp32. May be None to skip sink fold.
        lse_bhs_out: optional [B, H, S_q] fp32 output. When provided, the
                     transposed lse is written in the same kernel launch
                     (Phase E Lever 1 fusion).
        B, S_q:   required when lse_bhs_out is provided (used for the
                  transpose index split q_id=b*S_q+s_q).
    """
    assert out_bf16.dtype == torch.bfloat16
    total_q, H, V = out_bf16.shape
    assert lse.shape == (total_q, H), f"lse shape {tuple(lse.shape)} vs ({total_q},{H})"
    has_sink = sink is not None
    if has_sink:
        assert sink.shape == (H,), f"sink shape {tuple(sink.shape)} vs ({H},)"
    has_lse_t = lse_bhs_out is not None
    if has_lse_t:
        assert B is not None and S_q is not None, \
            "B and S_q required when lse_bhs_out is provided"
        assert B * S_q == total_q, \
            f"B*S_q={B*S_q} vs total_q={total_q}"
        assert lse_bhs_out.shape == (B, H, S_q), \
            f"lse_bhs_out shape {tuple(lse_bhs_out.shape)} vs ({B},{H},{S_q})"
    if not out_bf16.is_contiguous():
        out_bf16 = out_bf16.contiguous()
    if lse.dtype != torch.float32:
        lse = lse.float()
    if not lse.is_contiguous():
        lse = lse.contiguous()
    if has_sink:
        if sink.dtype != torch.float32:
            sink = sink.float()
        if not sink.is_contiguous():
            sink = sink.contiguous()

    if not has_sink and not has_lse_t:
        return  # nothing to do

    BLOCK_V = min(512, V)
    grid = (total_q * H,)
    _sink_fold_and_lse_transpose_kernel[grid](
        out_bf16, lse,
        sink if has_sink else lse,            # placeholder when HAS_SINK=0
        lse_bhs_out if has_lse_t else lse,    # placeholder when HAS_LSE_TRANSPOSE=0
        lse_bhs_out.stride(0) if has_lse_t else 0,
        lse_bhs_out.stride(1) if has_lse_t else 0,
        lse_bhs_out.stride(2) if has_lse_t else 0,
        B=B if B is not None else 1,
        S_q=S_q if S_q is not None else 1,
        H=H, V=V, BLOCK_V=BLOCK_V,
        HAS_SINK=has_sink,
        HAS_LSE_TRANSPOSE=has_lse_t,
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


_PICK_NUM_SPLITS_LOGGED = False


def pick_num_splits(B: int, topk: int) -> int:
    """Pick num_kv_splits so each split has ~1 BLOCK_N=32 worth of work AND
    total workgroup count stays in the [64, 512] sweet spot for MI355X.

    Override via env `SGLANG_CK_V32_FORCE_SPLITS=<int>` for Phase C tuning.
    """
    global _PICK_NUM_SPLITS_LOGGED
    forced = os.environ.get("SGLANG_CK_V32_FORCE_SPLITS", "")
    if forced:
        s = int(forced)
        # Snap to a divisor of topk (kernel requires topk % splits == 0).
        while topk % s != 0 and s > 1:
            s -= 1
        if not _PICK_NUM_SPLITS_LOGGED:
            print(f"[ck_v32_sparse_mla] FORCED num_kv_splits={s} via "
                  f"SGLANG_CK_V32_FORCE_SPLITS (B={B} topk={topk})", flush=True)
            _PICK_NUM_SPLITS_LOGGED = True
        return s
    BLOCK_N = 32
    splits_by_topk = max(1, topk // BLOCK_N)
    splits_by_total = max(1, 512 // (HEAD_GROUPS * B))
    splits = min(splits_by_topk, splits_by_total)
    while topk % splits != 0 and splits > 1:
        splits -= 1
    if not _PICK_NUM_SPLITS_LOGGED:
        print(f"[ck_v32_sparse_mla] pick_num_splits(B={B}, topk={topk}) -> {splits} "
              f"(by_topk={splits_by_topk}, by_total={splits_by_total}, "
              f"HEAD_GROUPS={HEAD_GROUPS})", flush=True)
        _PICK_NUM_SPLITS_LOGGED = True
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


# Cache for the bf16 output buffer + transposed lse buffer. These are the
# tensors returned to the caller; without caching, each call did
# `split_data.to(bf16)` and `lse.transpose().contiguous()` as fresh
# allocations — the captured cuda-graph downstream baked the first-call
# addresses, then read stale memory on replay (root cause of garbage tokens
# in iter22). Cache by per-call shape so each call reuses the same buffer
# (memory: 32 MB total saved across 61 layers); sequential pipeline within
# each layer guarantees attention-write before downstream-read.
_OUT_BUF_CACHE: dict = {}


def _get_out_buffers(total_q: int, B: int, S_q: int, H: int, V: int,
                     device: torch.device):
    """Return (out_bf16, lse_bhs) cached per shape. `out_bf16` is the
    [total_q, H, V] bf16 attention output; `lse_bhs` is the transposed
    [B, H, S_q] log-sum-exp."""
    key = (total_q, B, S_q, H, V, str(device))
    cached = _OUT_BUF_CACHE.get(key)
    if cached is not None:
        return cached
    out_bf16 = torch.empty((total_q, H, V), dtype=torch.bfloat16, device=device)
    lse_bhs = torch.empty((B, H, S_q), dtype=torch.float32, device=device)
    if len(_OUT_BUF_CACHE) > 32:
        _OUT_BUF_CACHE.clear()
    _OUT_BUF_CACHE[key] = (out_bf16, lse_bhs)
    return out_bf16, lse_bhs


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

    # Cached output buffers (cuda-graph mempool-stable). copy_ does an
    # in-place fp32→bf16 cast into the cached buffer; the original code
    # did `split_data[:, 0].to(bfloat16)` which allocated a fresh tensor
    # every call → stale-buffer reads on cuda-graph replay (iter22 root
    # cause of garbage tokens).
    out_bf16_cached, lse_bhs_cached = _get_out_buffers(
        total_q, B, S_q, H, V_HEAD_DIM, device,
    )

    if num_splits == 1:
        lse_2d = split_lse[:, 0, :, 0]
        # Phase E Lever 1: fused fp32→bf16-cast + sink + lse-transpose Triton
        # kernel WINS only when sink is present (1 launch vs 3).
        #
        # MICROBENCH on chi2866 MI355X (V32 V=512 H=16 num_splits=1):
        #   sink:    ref 33.5 us → fused 29.0 us  (+4.5 us/call WIN)
        #   nosink:  ref 10.3 us → fused 29.0 us  (-18.7 us/call REGRESSION)
        #
        # Triton has ~30us constant overhead. torch's .copy_() (fp32→bf16) +
        # .transpose().contiguous() are HIP-fused at ~5us each — Triton can't
        # beat them without enough work to amortize launch overhead.
        # Only fuse when sink fold is actually needed.
        if attn_sink is not None:
            _apply_split0_cast_with_sink_and_lse_transpose(
                split_data, out_bf16_cached, lse_2d, attn_sink,
                lse_bhs_cached, B=B, S_q=S_q,
            )
        else:
            # No sink: 2-launch torch path beats single-Triton-launch.
            out_bf16_cached.copy_(split_data[:, 0, :, :])
            lse_bhs_cached.copy_(lse_2d.view(B, S_q, H).transpose(1, 2))
    else:
        # Multi-split: reduce path still allocates internally; copy result
        # into our cached buffer so the captured graph reads stable address.
        out_reduced, lse_2d = _ck_native_reduce(
            split_data, split_lse, max_seqlen_q=S_q,
        )
        out_bf16_cached.view_as(out_reduced).copy_(out_reduced)

        # For multi-split path: only fuse sink+transpose when sink is present.
        # When no sink, torch's strided .copy_() (5 us) beats Triton's launch
        # overhead (23 us). See bench_lever1_multisplit.py.
        if attn_sink is not None:
            _apply_sink_fold_inplace(
                out_bf16_cached, lse_2d, attn_sink,
                lse_bhs_out=lse_bhs_cached, B=B, S_q=S_q,
            )
        else:
            lse_bhs_cached.copy_(lse_2d.view(B, S_q, H).transpose(1, 2))

    # Reshape views (no allocation) into the public output shapes.
    out_bf16 = out_bf16_cached.view(B, S_q, H, V_HEAD_DIM)
    return out_bf16, lse_bhs_cached


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

    # Phase C3 / Phase E Lever 3: Triton fast-path for combine.
    #
    # MICROBENCH on chi2866 MI355X (V32, V=512, H=16):
    #   shape                CK us   Triton us   speedup
    #   DEC q=1 S_a=1          2.9      40.5      0.07x   ← Triton REGRESSES
    #   DEC q=6 S_a=8          4.4      45.1      0.10x   ← Triton REGRESSES
    #   PFL q=1024 S_a=8      79.5      45.8      1.73x   ← Triton wins
    #   PFL q=4096 S_a=4     194.4     130.7      1.49x   ← Triton wins
    #   PFL q=16384 S_a=8   1279.0     818.3      1.56x   ← Triton wins
    #
    # Triton has ~40 us constant launch overhead. CK has ~3 us. Triton only
    # wins when there's enough work to amortize. Gate: use Triton iff
    # `total_q * (S_a + S_b) >= 8192` (empirically tuned crossover).
    # Disable Triton path entirely with SGLANG_CK_V32_TRITON_COMBINE=0.
    # Force-enable regardless of work score with SGLANG_CK_V32_TRITON_COMBINE=force.
    S_a = split_data_a.size(1)
    S_b = split_data_b.size(1) if split_data_b is not None else 0
    triton_env = os.environ.get("SGLANG_CK_V32_TRITON_COMBINE", "1")
    use_triton = False
    if triton_env != "0" and S_b <= 1:
        work = total_q * (S_a + max(S_b, 1))
        threshold = int(os.environ.get("SGLANG_CK_V32_TRITON_COMBINE_MIN_WORK", "8192"))
        use_triton = (triton_env == "force") or (work >= threshold)

    if use_triton:
        if S_a == 1:
            from sglang.jit_kernel.mla_combine_triton import (
                mla_combine_two_splits_triton,
            )
            return mla_combine_two_splits_triton(
                split_data_a, split_lse_a,
                split_data_b, split_lse_b,
                attn_sink, out, lse,
            )
        # PREFILL N-way path. **ON by default** — chi2811 E2E A/B aligned with
        # shipping config (2026-04-28, dsv4-flash-base-fp8-shipping-state.md)
        # showed:
        #   median TPOT  41.59 → 41.45 ms (-0.3%, neutral)
        #   median TTFT 251.72 → 250.50 ms (-0.5%, neutral)
        #   mean   TTFT 520.28 → 441.88 ms (-15.1%, improvement)
        #   P99    TTFT 2275   → 1533    ms (-32.6%, improvement)
        #   output tput 92.54 → 93.00 tok/s (+0.5%)
        # The earlier "P99 TTFT regression" finding was an artifact of an
        # un-aligned baseline (missing TWO_SHOT=1 / TORCH=1 / 8-request warmup);
        # with shipping-aligned warmup the cold-compile tail amortizes cleanly.
        # Disable via SGLANG_CK_V32_TRITON_COMBINE_NWAY=0 if a deployment is
        # tail-sensitive and unable to do an 8-request warmup at startup.
        if os.environ.get("SGLANG_CK_V32_TRITON_COMBINE_NWAY", "1") == "1":
            from sglang.jit_kernel.mla_combine_triton import (
                mla_combine_n_way_triton,
            )
            return mla_combine_n_way_triton(
                split_data_a, split_lse_a,
                split_data_b, split_lse_b,
                attn_sink, out, lse,
            )

    ck = _get_ck_mod()
    ck.mla_combine_fwd_ck(
        split_data_a, split_lse_a,
        split_data_b, split_lse_b,
        attn_sink,
        out, lse,
    )
    return out, lse
