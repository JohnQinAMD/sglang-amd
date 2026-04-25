"""CK Tile FP8 sparse MLA decode kernel adapter (V32 / DSv4-Pro shape).

This module wraps the gfx950 CK Tile kernel that lives in the aiter-amd
checkout. Used by ``debug_flash_mla_adapter.py`` when:
  * ``SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1`` is set, AND
  * ``head_dim_v == 512`` (V32 shape — DSv4-Pro / V3-style 576/512 dims).

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

# V32 (DSv4-Pro) head dimensions; these are baked into the kernel template
# parameters and the reduce metadata builders below.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
V_HEAD_DIM = KV_LORA_RANK
HEAD_GROUPS = 4  # = num_heads (128) / TILE_HEADS (32) — kernel constant

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
    assert D == QK_HEAD_DIM, f"expected qk_head_dim={QK_HEAD_DIM}, got {D}"

    topk = indices.shape[-1]
    device = q.device
    total_q = B * S_q

    # Normalize KV pool to [num_pages, 1, 1, 576] FP8 — accepts 2D, 3D, or 4D
    # depending on how the caller materialized the cache.
    if k_cache.dim() == 2:
        n_kv = k_cache.shape[0]
        kv_view = k_cache.view(n_kv, 1, 1, QK_HEAD_DIM)
    elif k_cache.dim() == 3:
        n_kv = k_cache.shape[0] * k_cache.shape[1]
        kv_view = k_cache.view(n_kv, 1, 1, QK_HEAD_DIM)
    elif k_cache.dim() == 4:
        if k_cache.shape[-1] == QK_HEAD_DIM:
            kv_view = k_cache
        else:
            num_pages, page_size, h_kv, d = k_cache.shape
            assert h_kv == 1 and d == QK_HEAD_DIM
            kv_view = k_cache.view(num_pages * page_size, 1, 1, QK_HEAD_DIM)
    else:
        raise AssertionError(f"unexpected k_cache shape: {tuple(k_cache.shape)}")

    if not kv_view.is_contiguous():
        kv_view = kv_view.contiguous()

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
        q.view(total_q, H, QK_HEAD_DIM).contiguous()
        if q.is_contiguous()
        else q.reshape(total_q, H, QK_HEAD_DIM)
    )

    ck = _get_ck_mod()
    ck.mla_decode_fwd_ck_sparse_fp8(
        q_2d, kv_view, split_data, split_lse,
        qo_indptr, kv_indptr, idx_flat,
        float(sm_scale), int(num_splits),
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
