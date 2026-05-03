from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.jit_kernel.deepseek_v4 import topk_transform_512
from sglang.srt.environ import envs
from sglang.srt.layers.attention.compressed.metadata import (
    PagedCoreMetadata,
    PagedIndexerMetadata,
)
from sglang.srt.layers.attention.indexer_topk_capturer import (
    get_global_indexer_capturer,
)
from sglang.srt.layers.attention.nsa.triton_kernel import act_quant
from sglang.srt.layers.attention.nsa.utils import is_nsa_enable_prefill_cp
from sglang.srt.utils import is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.attention.compressed.compressor import CompressorBackend
    from sglang.srt.layers.attention.compressed.metadata import DeepseekV4Metadata
    from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4TokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v4 import C4Indexer

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

# if is_hip():
if is_fp8_fnuz():
    FP8_DTYPE = torch.float8_e4m3fnuz
    # FP8_MAX = torch.finfo(FP8_DTYPE).max
    FP8_MAX = 224.0
else:
    FP8_DTYPE = torch.float8_e4m3fn
    FP8_MAX = torch.finfo(FP8_DTYPE).max


# Per-forward-pass cache for `int(seq_lens.max().item())`. Multiple indexer
# layers in one forward pass share the same `seq_lens` tensor, but each call
# to `int(seq_lens.max().item())` forces a D2H sync (~360 us each on AMD —
# waits for all pending GPU work to drain). Cache by tensor identity to fold
# ~13 syncs per decode step → 1.
_SEQ_LENS_MAX_CACHE: "dict[tuple[int, int, int], int]" = {}


def _seq_lens_max_cached(seq_lens: torch.Tensor) -> int:
    """Return `int(seq_lens.max().item())` with single-step caching.

    Cache key combines `id()` and `data_ptr()` to defeat object reuse after
    GC. Cache is bounded — cleared when it grows beyond a few entries (well
    above the steady-state 1-2 active forward passes).
    """
    key = (id(seq_lens), seq_lens.data_ptr(), seq_lens.numel())
    cached = _SEQ_LENS_MAX_CACHE.get(key)
    if cached is not None:
        return cached
    val = int(seq_lens.max().item())
    if len(_SEQ_LENS_MAX_CACHE) > 8:
        _SEQ_LENS_MAX_CACHE.clear()
    _SEQ_LENS_MAX_CACHE[key] = val
    return val


def _ensure_scratch(scratch_dict, name, shape, dtype, device):
    """Lazily allocate a persistent scratch tensor and return a contiguous
    EXACT-shape tensor. Reallocates on any shape mismatch — by design, since
    a non-contig slice of a grow-only buffer breaks downstream `.view()`.

    THIS HELPER MUST NOT BE SHARED ACROSS THE CAPTURE/EAGER BOUNDARY. A
    captured-bs=1 graph that reads from `scratch_dict[name]` baked in the
    storage's physical address; if eager (bs > 1) work later reallocates
    `scratch_dict[name]`, the bs=1 storage is freed and the captured graph
    faults (HSA 0x29) when it next replays. Use the per-mode dicts
    `_FP8_PAGED_SCRATCH_CAPTURED` / `_FP8_PAGED_SCRATCH_EAGER` below — only
    `_EAGER` ever reallocates; `_CAPTURED` is allocated once during graph
    capture and kept frozen so its address stays valid for every replay.
    """
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


# Per-mode scratch for fp8_paged_mqa_logits_torch.
#
# `_CAPTURED` is allocated during `torch.cuda.is_current_stream_capturing()` —
# i.e. while the bs=1 cuda graph is being recorded. Its tensors are baked
# into the captured kernels by physical address and MUST NEVER be
# reassigned thereafter, or replay reads freed memory → HSA 0x29.
# `_EAGER` services every other (bs > 1) call and is free to reallocate.
_FP8_PAGED_SCRATCH_CAPTURED: dict = {}
_FP8_PAGED_SCRATCH_EAGER: dict = {}

# Cache for the per-call `pt_expanded` tensor built inside
# `fp8_paged_mqa_logits_aiter`. `page_table` is shared across every c4
# indexer layer in a single decode step, so the arange + broadcast + copy
# that produces `pt_expanded` only needs to run once per step. We key on
# `page_table.data_ptr()` + shape + dtype + block_size, which uniquely
# identifies the input bytes; when the page_table buffer rotates (e.g.
# eager calls or a new graph capture replaces the persistent buffer),
# the data_ptr changes and we rebuild.
#
# Stored as `(int_key) -> (pt_expanded_tensor, arange_buf_tensor)`. Cache
# is bounded — cleared when it grows beyond 16 entries (well above the
# steady-state of one capture-mode buffer + one eager buffer per shape).
_PT_EXPANDED_CACHE: dict = {}


def fp8_paged_mqa_logits_torch(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Vectorized pytorch fallback — CUDA-graph compatible (no .item() / .tolist()).

    The prior loop-based version had `seq_len = int(seq_lens[i].item())` inside
    the batch loop, which forces a GPU->CPU sync per iter AND blocks CUDA graph
    capture. This version is branch-free: it gathers ALL pages up to
    `max_num_pages = ceil(max_seq_len / block_size)`, masks positions >=
    seq_lens[b] to -inf, and returns `[B, max_seq_len]`. Equivalent output for
    valid positions; padding entries are -inf, which matches the downstream
    top-K's semantics (it already ignores -inf via the seq_lens mask).

    NOTE on graph-pool stability: `gathered = kvcache_flat[pages]` (fancy
    indexing), `torch.arange(padded_seq_len)` and `torch.full((B, max_seq_len),
    -inf)` each allocate fresh tensors per call. Inside `torch.cuda.graph` they
    bind to whatever caching-allocator pool slab is live at capture. After
    eager multi-bs work churns that slab, replay reads stale addresses → HSA
    0x29. We route all three allocations through persistent module-level
    scratch buffers (`_FP8_PAGED_SCRATCH`) so addresses stay stable across
    replays.
    """
    _ = deep_gemm_metadata
    batch_size, _, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    device = q_fp8.device

    assert head_dim == 128, "TODO"
    assert block_size == 64, "TODO"
    assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
    assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
    assert weight.shape == (batch_size, num_heads)
    assert seq_lens.shape == (batch_size,)
    assert page_table.shape[0] == batch_size
    assert clean_logits is False

    # Cap padded_seq_len aggressively. Caller passes
    # `max_seq_len = page_table.shape[1] * page_size` (worst-case from
    # --context-len, e.g. 1048576 at 1M). At bs=8 x H=64, transient allocs
    # (kvcache_value_f32, score pre-sum) scale as B * padded * (D | H) * 4 and
    # hit ~66 GiB before the caching allocator can reuse, OOMing the scheduler.
    #
    # Two bounds applied:
    #   1. SGLANG_INDEXER_MAX_SEQ_LEN env (host int) — caps BOTH capture and
    #      eager so the captured graph's worst-case reserve stays in budget.
    #   2. seq_lens.max() in eager (single D2H sync, cheap vs per-layer MFMA) —
    #      shrinks decode allocs to the actual content length.
    # During capture we can't sync on seq_lens, so only (1) applies there.
    env_cap = envs.SGLANG_INDEXER_MAX_SEQ_LEN.get()
    base_cap = env_cap if env_cap > 0 else max_seq_len
    if torch.cuda.is_current_stream_capturing():
        effective_max_seq_len = base_cap
    else:
        effective_max_seq_len = min(_seq_lens_max_cached(seq_lens), base_cap)
        # Round up to block_size so max_num_pages * block_size == padded len.
        effective_max_seq_len = max(
            block_size,
            ((effective_max_seq_len + block_size - 1) // block_size) * block_size,
        )

    max_num_pages = min(
        (effective_max_seq_len + block_size - 1) // block_size,
        page_table.shape[1],
    )
    padded_seq_len = max_num_pages * block_size
    row_bytes = block_size * (head_dim + 4)  # last-dim of gathered

    # Pick the captured-graph or eager scratch dict by current stream state.
    # During cuda graph capture this returns True, so the captured graph
    # binds to `_FP8_PAGED_SCRATCH_CAPTURED` and that dict is never mutated
    # again (replay doesn't run Python). Eager calls use the separate
    # `_EAGER` dict, which is free to reallocate per shape.
    _paged_scratch = (
        _FP8_PAGED_SCRATCH_CAPTURED
        if torch.cuda.is_current_stream_capturing()
        else _FP8_PAGED_SCRATCH_EAGER
    )

    # q: [B, H, D] fp32 — .to() returns fresh; route into scratch.
    q_f32_buf = _ensure_scratch(
        _paged_scratch, "q_f32",
        (batch_size, num_heads, head_dim),
        torch.float32, device,
    )
    q_f32_buf.copy_(q_fp8[:, 0].to(torch.float32))
    q_f32 = q_f32_buf

    # gathered: logically [B, max_pages, row_bytes] but stored as a 2-D
    # exact-shape scratch so that `out=` receives a contiguous tensor (a
    # 3-D slice of a grow-only scratch is non-contiguous when batch or
    # max_pages shrinks, which makes `.view()` reject it). We pick the
    # 2-D shape `[B*max_pages, row_bytes]` and re-view to 3-D after the
    # gather.
    kvcache_flat = kvcache_fp8.view(-1, row_bytes)
    pages = page_table[:, :max_num_pages]
    gathered_2d = _ensure_scratch(
        _paged_scratch, "gathered",
        (batch_size * max_num_pages, row_bytes),
        kvcache_flat.dtype, device,
    )
    torch.index_select(
        kvcache_flat, 0, pages.reshape(-1),
        out=gathered_2d,
    )
    gathered = gathered_2d.view(batch_size, max_num_pages, row_bytes)

    SCALE_OFFSET = block_size * head_dim
    value_flat = gathered[..., :SCALE_OFFSET].contiguous().view(dtype=FP8_DTYPE)
    scale_flat = gathered[..., SCALE_OFFSET:].contiguous().view(dtype=torch.float32)

    kvcache_value_f32 = value_flat.to(torch.float32).view(
        batch_size, padded_seq_len, head_dim
    )
    kvcache_scale_f32 = scale_flat.view(batch_size, padded_seq_len)

    # [B, padded, H] = [B, padded, D] @ [B, D, H]
    score = torch.bmm(kvcache_value_f32, q_f32.transpose(1, 2))
    score = torch.relu(score)
    score = score * weight.unsqueeze(1)   # [B, padded, H]
    score = score.sum(dim=2)              # [B, padded]
    score = score * kvcache_scale_f32     # [B, padded]

    # positions: [padded_seq_len] — fresh arange. Route through scratch.
    positions_buf = _ensure_scratch(
        _paged_scratch, "positions",
        (padded_seq_len,), torch.long, device,
    )
    torch.arange(padded_seq_len, dtype=torch.long, device=device, out=positions_buf)
    positions = positions_buf

    valid = positions.unsqueeze(0) < seq_lens.unsqueeze(1)  # [B, padded]
    # `score.new_full((), -inf)` routes through graph pool (per the
    # topk_transform_512 .clone() fix at indexer.py:271-279).
    neg_inf = score.new_full((), float("-inf"))
    score = torch.where(valid, score, neg_inf)

    # out: [B, effective_max_seq_len], filled with -inf, then copy
    # `score[:, :fill]` into the head. Persistent scratch + fill_(-inf) instead
    # of torch.full. Width is bounded by effective_max_seq_len (NOT the
    # caller-passed worst-case max_seq_len) — downstream topk_transform_512
    # reads scores.shape[1] off the tensor and runs min(TOPK, width), so any
    # consistent width is fine; tokens past seq_lens[b] are already -inf.
    fill = min(padded_seq_len, effective_max_seq_len)
    out = _ensure_scratch(
        _paged_scratch, "out",
        (batch_size, effective_max_seq_len), torch.float32, device,
    )
    out.fill_(float("-inf"))
    out[:, :fill] = score[:, :fill]
    return out


def fp8_paged_mqa_logits_aiter(
    q_fp8: torch.Tensor,
    kvcache_fp8: torch.Tensor,
    weight: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata: Any,
    max_seq_len: int,
    clean_logits: bool = True,
) -> torch.Tensor:
    """aiter Triton fp8_paged_mqa_logits — HIP path.

    Drop-in replacement for fp8_paged_mqa_logits_torch. Wraps
    `aiter.ops.triton.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits`, which is
    the AMD-tuned Triton (or Gluon) kernel for the same compute the
    NVIDIA-only `deep_gemm.fp8_paged_mqa_logits` performs. Output buffer is
    pre-filled with -inf so positions past `seq_lens[b]` carry the correct
    masking semantics for downstream top-K (matches the torch fallback).

    Output buffer is routed through the per-mode scratch dicts so it stays
    graph-pool stable across cuda-graph replays (same convention as
    fp8_paged_mqa_logits_torch).
    """
    _ = deep_gemm_metadata
    _ = clean_logits
    from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

    batch_size, next_n, num_heads, head_dim = q_fp8.shape
    block_size = kvcache_fp8.shape[1]
    device = q_fp8.device

    # Cap padded length using the same bound logic as the torch fallback so
    # capture-time and eager-time output shapes are consistent.
    env_cap = envs.SGLANG_INDEXER_MAX_SEQ_LEN.get()
    base_cap = env_cap if env_cap > 0 else max_seq_len
    if torch.cuda.is_current_stream_capturing():
        effective_max_seq_len = base_cap
    else:
        eff = min(_seq_lens_max_cached(seq_lens), base_cap)
        effective_max_seq_len = max(
            block_size,
            ((eff + block_size - 1) // block_size) * block_size,
        )

    _paged_scratch = (
        _FP8_PAGED_SCRATCH_CAPTURED
        if torch.cuda.is_current_stream_capturing()
        else _FP8_PAGED_SCRATCH_EAGER
    )

    # aiter writes [batch * next_n, max_model_len] fp32; downstream topk reads
    # `[batch, max_seq_len]`, which equals the same buffer when next_n == 1
    # (DSv4 decode). Allocate via persistent scratch — addresses must stay
    # stable across replays (else HSA 0x29 on graph replay).
    out = _ensure_scratch(
        _paged_scratch,
        "out_aiter",
        (batch_size * next_n, effective_max_seq_len),
        torch.float32,
        device,
    )
    if not envs.SGLANG_INDEXER_SKIP_OUT_PREFILL.get():
        # Default-on safe path: the deepgemm aiter kernel writes per-batch
        # only up to context_length (see `_deepgemm_fp8_paged_mqa_logits*`
        # store loop). Anything past per-batch seq_lens[b] would carry stale
        # values without this fill, breaking downstream top-K masking.
        out.fill_(float("-inf"))

    # On HIP with Triton 3.4.0, aiter's `_deepgemm_fp8_paged_mqa_logits` only
    # supports KVBlockSize=1 (the Gluon path that handles 64 needs Triton 3.5+).
    # Reshape kv_cache to per-token rows and expand page_table accordingly:
    #   kv_cache: [num_blocks, 64, 1, D+4] -> [num_blocks*64, 1, 1, D+4]   (free view)
    #   page_table: [B, max_pages] (block ids) -> [B, max_pages*64] (token ids)
    kv_flat = kvcache_fp8.view(-1, 1, 1, kvcache_fp8.shape[-1])

    max_pages = page_table.shape[1]

    # Cached path: `page_table` is the same object across every c4 indexer
    # layer in one decode step, so `pt_expanded` is identical every call.
    # Key on (data_ptr, batch, max_pages, dtype-id, block_size, device) so a
    # rotated page_table buffer (different storage) misses and rebuilds.
    # CRITICAL: only enable when caller guarantees `page_table` is read-only
    # within a step (true for DSv4 — the global page table is set once at
    # forward-pass start). When unsafe (eager rebuild between layers), the
    # data_ptr will differ and the cache will rebuild correctly anyway.
    if envs.SGLANG_INDEXER_PT_EXPANDED_CACHED.get():
        # Capture-mode flag is part of the key: a captured graph's allocations
        # bind to the capture pool. Replaying a captured graph that holds a
        # cached eager-allocated tensor (or vice-versa) would read freed
        # memory (HSA 0x29). Keep capture and eager caches strictly disjoint.
        is_capturing = torch.cuda.is_current_stream_capturing()
        cache_key = (
            page_table.data_ptr(),
            batch_size,
            max_pages,
            page_table.dtype,
            block_size,
            device.index if device.index is not None else -1,
            bool(is_capturing),
        )
        cached = _PT_EXPANDED_CACHE.get(cache_key)
        if cached is None:
            # Bound the cache so a long-running process with many distinct
            # page_table buffers (e.g. many capture replays + eager) can't
            # leak memory. Clear when it grows past a small threshold.
            if len(_PT_EXPANDED_CACHE) > 16:
                _PT_EXPANDED_CACHE.clear()
            pt_expanded = torch.empty(
                (batch_size, max_pages * block_size),
                dtype=page_table.dtype,
                device=device,
            )
            arange_buf = torch.arange(
                block_size, dtype=page_table.dtype, device=device
            )
            # expanded[b, j*block_size + t] = page_table[b, j] * block_size + t
            pt_expanded.copy_(
                (page_table.unsqueeze(-1) * block_size + arange_buf.view(1, 1, -1)).view(
                    batch_size, -1
                )
            )
            _PT_EXPANDED_CACHE[cache_key] = pt_expanded
        else:
            pt_expanded = cached
    else:
        pt_expanded = _ensure_scratch(
            _paged_scratch,
            "pt_aiter_expanded",
            (batch_size, max_pages * block_size),
            page_table.dtype,
            device,
        )
        arange_buf = _ensure_scratch(
            _paged_scratch,
            "pt_aiter_arange",
            (block_size,),
            page_table.dtype,
            device,
        )
        # 2026-05-03 Target 3: arange content is a pure function of block_size
        # (constant during a run), so it's stable for the lifetime of the
        # scratch tensor. Skip the kernel launch on subsequent calls — identity
        # check picks up reallocations from `_ensure_scratch` (different shape
        # / dtype / device) and refills correctly. Saves ~1 arange/lyr in the
        # AITER hot path.
        if _paged_scratch.get("pt_aiter_arange__filled_for") is not arange_buf:
            torch.arange(block_size, dtype=page_table.dtype, device=device, out=arange_buf)
            _paged_scratch["pt_aiter_arange__filled_for"] = arange_buf
        # expanded[b, j*block_size + t] = page_table[b, j] * block_size + t
        pt_expanded.copy_(
            (page_table.unsqueeze(-1) * block_size + arange_buf.view(1, 1, -1)).view(
                batch_size, -1
            )
        )

    deepgemm_fp8_paged_mqa_logits(
        q_fp8=q_fp8,
        kv_cache=kv_flat,
        weights=weight,
        out_logits=out,
        context_lens=seq_lens,
        kv_indices=pt_expanded,
        max_model_len=effective_max_seq_len,
        KVBlockSize=1,
    )
    return out


# def fp8_paged_mqa_logits_torch(
#     q_fp8: torch.Tensor,
#     kvcache_fp8: torch.Tensor,
#     weight: torch.Tensor,
#     seq_lens: torch.Tensor,
#     page_table: torch.Tensor,
#     deep_gemm_metadata: Any,
#     max_seq_len: int,
#     clean_logits: bool = True,
# ) -> torch.Tensor:
#     """
#     Vectorized PyTorch implementation of fp8_paged_mqa_logits.
#     Processes all batches in parallel without Python for loops.
#     """
#     _ = deep_gemm_metadata
#     batch_size, _, num_heads, head_dim = q_fp8.shape
#     block_size = kvcache_fp8.shape[1]
#     device = q_fp8.device

#     assert head_dim == 128, "TODO"
#     assert block_size == 64, "TODO"
#     assert q_fp8.shape == (batch_size, 1, num_heads, head_dim)
#     assert kvcache_fp8.shape[1:] == (block_size, 1, head_dim + 4)
#     assert weight.shape == (batch_size, num_heads)
#     assert seq_lens.shape == (batch_size,)
#     assert page_table.shape[0] == batch_size
#     assert clean_logits == False

#     # Prepare q: (batch_size, num_heads, head_dim)
#     q = q_fp8[:, 0].to(torch.float32)  # (batch_size, num_heads, head_dim)

#     # Calculate number of pages per batch element
#     num_pages_per_batch = (seq_lens + block_size - 1) // block_size  # (batch_size,)
#     max_num_pages = int(
#         num_pages_per_batch.max().item()
#     )  # Single sync, outside main computation

#     # Padded seq len for each batch
#     padded_seq_lens = num_pages_per_batch * block_size  # (batch_size,)
#     max_padded_seq_len = max_num_pages * block_size

#     # Reshape kvcache for gathering
#     # Original: (num_blocks_total, block_size, 1, head_dim + 4)
#     # Reshape to: (num_blocks_total, block_size * (head_dim + 4))
#     kvcache_flat = kvcache_fp8.view(-1, block_size * (head_dim + 4))

#     # Gather pages for all batches: page_table[:, :max_num_pages]
#     # Shape: (batch_size, max_num_pages)
#     pages = page_table[:, :max_num_pages]

#     # Gather kvcache for all batches
#     # Shape: (batch_size, max_num_pages, block_size * (head_dim + 4))
#     gathered_kvcache = kvcache_flat[pages]

#     # Split into values and scales
#     SCALE_OFFSET = block_size * head_dim
#     # Shape: (batch_size, max_num_pages, block_size * head_dim)
#     kvcache_value_flat = gathered_kvcache[..., :SCALE_OFFSET]
#     # Shape: (batch_size, max_num_pages, block_size * 4) -> scales are 4 bytes per position
#     kvcache_scale_flat = gathered_kvcache[..., SCALE_OFFSET:]

#     # Convert FP8 values to float32
#     kvcache_value_fp8 = kvcache_value_flat.view(dtype=FP8_DTYPE)
#     kvcache_value = kvcache_value_fp8.to(torch.float32)
#     # Reshape to (batch_size, max_padded_seq_len, head_dim)
#     kvcache_value = kvcache_value.view(batch_size, max_padded_seq_len, head_dim)

#     # Convert scales to float32
#     kvcache_scale = kvcache_scale_flat.view(dtype=torch.float32)
#     # Reshape to (batch_size, max_padded_seq_len)
#     kvcache_scale = kvcache_scale.reshape(batch_size, max_padded_seq_len)

#     # Compute attention scores: kvcache_value @ q^T
#     # kvcache_value: (batch_size, max_padded_seq_len, head_dim)
#     # q: (batch_size, num_heads, head_dim)
#     # score: (batch_size, max_padded_seq_len, num_heads)
#     score = torch.bmm(kvcache_value, q.transpose(1, 2))

#     # Apply ReLU
#     score = F.relu(score)

#     # Multiply by weight (q_scale): (batch_size, num_heads)
#     # score: (batch_size, max_padded_seq_len, num_heads)
#     score = score * weight.unsqueeze(1)

#     # Sum over heads: (batch_size, max_padded_seq_len)
#     score = score.sum(dim=2)

#     # Multiply by kvcache_scale: (batch_size, max_padded_seq_len)
#     score = score * kvcache_scale

#     # Create output logits with proper masking
#     logits = torch.full(
#         (batch_size, max_seq_len),
#         float("-inf"),  # or 0.0 depending on requirements
#         dtype=torch.float32,
#         device=device,
#     )

#     # Create position indices for masking
#     positions = torch.arange(max_seq_len, device=device).unsqueeze(
#         0
#     )  # (1, max_seq_len)
#     valid_mask = positions < seq_lens.unsqueeze(1)  # (batch_size, max_seq_len)

#     # Copy valid scores to logits
#     # We need to handle the case where max_padded_seq_len might differ from max_seq_len
#     copy_len = min(max_padded_seq_len, max_seq_len)
#     logits[:, :copy_len] = torch.where(
#         valid_mask[:, :copy_len], score[:, :copy_len], logits[:, :copy_len]
#     )

#     return logits


# Vectorized version (faster but uses more memory) - for AMD/HIP
def topk_transform_512_pytorch_vectorized(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """
    Vectorized PyTorch fallback for topk_transform_512.
    Faster than the loop version but may use more memory.
    """

    TOPK = 512
    batch_size = scores.shape[0]
    max_seq_len = scores.shape[1]
    device = scores.device

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    # Create mask for valid positions based on seq_lens
    positions = (
        torch.arange(max_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    )
    valid_mask = positions < seq_lens.unsqueeze(1)

    # Mask out invalid positions with -inf.
    #
    # NOTE: Earlier this was `scores.clone(); masked_scores[~valid_mask] = -inf`,
    # which allocates a new tensor inside the captured graph. Allocations from
    # the caching allocator inside `torch.cuda.graph` cause stale-pointer
    # APERTURE_VIOLATION (HSA 0x29) on multi-bs replay because each replay can
    # bind a different physical address while the captured kernel reads the
    # original. Using `torch.where` avoids the .clone() and routes through the
    # graph-pool allocator for the temporary, which IS replay-safe.
    masked_scores = torch.where(valid_mask, scores, scores.new_full((), float("-inf")))

    # Get top-k indices
    actual_k = min(TOPK, max_seq_len)
    _, raw_indices = torch.topk(
        masked_scores, k=actual_k, dim=1, largest=True, sorted=False
    )
    raw_indices = raw_indices.to(torch.int32)

    # Pad raw_indices to TOPK size if needed
    if actual_k < TOPK:
        padding = torch.zeros(
            (batch_size, TOPK - actual_k), dtype=torch.int32, device=device
        )
        raw_indices = torch.cat([raw_indices, padding], dim=1)

    # Check which indices are valid
    batch_indices = (
        torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, TOPK)
    )
    gathered_scores = scores[
        batch_indices.flatten(), raw_indices.clamp(min=0).flatten()
    ].view(batch_size, TOPK)

    valid_topk = gathered_scores != float("-inf")
    if actual_k < TOPK:
        pad_mask = torch.arange(TOPK, device=device).unsqueeze(0) >= actual_k
        valid_topk = valid_topk & ~pad_mask

    # For short sequences, use sequential indices.
    # NOTE: was guarded by `if needs_sequential.any():` which forces a
    # GPU->CPU sync and breaks CUDA graph capture. We now always apply the
    # where — it's a no-op when needs_sequential is all-False.
    needs_sequential = seq_lens <= TOPK
    sequential_indices = (
        torch.arange(TOPK, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    sequential_valid = sequential_indices < seq_lens.unsqueeze(1)

    # Pre-allocate the fill constants on the same device/stream to avoid
    # CUDA-graph capture issues ('capturing stream has unjoined work') that
    # `torch.tensor(-1, device=...)` inline creation can trigger.
    minus_one_i32 = raw_indices.new_full((), -1)
    raw_indices = torch.where(
        needs_sequential.unsqueeze(1).expand(-1, TOPK),
        torch.where(
            sequential_valid,
            sequential_indices,
            minus_one_i32,
        ),
        raw_indices,
    )
    valid_topk = torch.where(
        needs_sequential.unsqueeze(1).expand(-1, TOPK), sequential_valid, valid_topk
    )

    # Transform to page indices
    page_idx = raw_indices >> page_bits
    offset_in_page = raw_indices & page_mask

    page_idx_clamped = torch.clamp(page_idx, min=0)
    physical_pages = torch.gather(page_tables, dim=1, index=page_idx_clamped.long())

    page_indices = (physical_pages << page_bits) | offset_in_page
    page_indices = page_indices.to(torch.int32)

    page_indices = torch.where(
        valid_topk, page_indices, page_indices.new_full((), -1)
    )

    out_page_indices.copy_(page_indices)

    if out_raw_indices is not None:
        raw_indices = torch.where(
            valid_topk, raw_indices, torch.tensor(-1, device=device, dtype=torch.int32)
        )
        out_raw_indices.copy_(raw_indices)


@triton.jit
def _fused_scale_kernel(
    weight_ptr,  # [B, H]
    q_scale_ptr,  # [B, H, 1]
    out_ptr,  # [B, H, 1]
    numel,  # B * H
    out_scale,  # scalar
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel

    w = tl.load(weight_ptr + offs, mask=mask)
    qs = tl.load(q_scale_ptr + offs, mask=mask)

    # Compute in fp32 for better numerical stability, then cast back.
    acc = w.to(tl.float32) * out_scale * qs.to(tl.float32)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def fused_scale(
    weight: torch.Tensor,
    out_scale: float,
    q_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Triton version of:
        weight.unsqueeze(-1) * out_scale * q_scale

    Args:
        weight:  [B, H], contiguous
        q_scale: [B, H, 1], contiguous
        out_scale: Python float / scalar

    Returns:
        out: [B, H, 1]
    """
    assert weight.is_contiguous() and q_scale.is_contiguous()
    B, H = weight.shape
    numel = B * H
    out_dtype = torch.promote_types(weight.dtype, q_scale.dtype)
    out = torch.empty((B, H, 1), device=weight.device, dtype=out_dtype)
    BLOCK = 1024
    grid = (triton.cdiv(numel, BLOCK),)
    _fused_scale_kernel[grid](
        weight,
        q_scale,
        out,
        numel,
        out_scale,
        BLOCK=BLOCK,
    )
    return out


class C4IndexerBackend:
    def __init__(self):
        super().__init__()
        self.forward_metadata: DeepseekV4Metadata
        self.debug_use_external_c4_sparse_indices: bool = False

        # this method should be type method
        # see srt/layers/attention/compressed/compressor.py

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        x_for_compressor: Optional[torch.Tensor] = None,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackend)

        assert alt_streams is not None
        assert len(alt_streams) >= 2
        current_stream = torch.cuda.current_stream()
        stream_q = alt_streams[0]
        stream_weights = alt_streams[1]

        stream_q.wait_stream(current_stream)
        stream_weights.wait_stream(current_stream)

        # main stream
        self.forward_indexer_compressor(
            x=x_for_compressor if (is_nsa_enable_prefill_cp() and x_for_compressor is not None) else x,
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )
        c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=c4_indexer.layer_id,
        )

        # alt stream 0: compute q
        with torch.cuda.stream(stream_q):
            if q_lora_ready is not None:
                stream_q.wait_event(q_lora_ready)
            q = c4_indexer.compute_q(q_lora, positions=positions)
            q_fp8, q_scale = act_quant(q)
            q_scale_ready = stream_q.record_event()

        # alt stream 1: compute weights
        with torch.cuda.stream(stream_weights):
            weights = c4_indexer.compute_weights(x, skip_scale=True)
            stream_weights.wait_event(q_scale_ready)
            weights = fused_scale(weights, c4_indexer.weight_scale, q_scale)

        current_stream.wait_stream(stream_q)
        current_stream.wait_stream(stream_weights)

        return q_fp8, weights, c4_indexer_kv_cache

    def _forward_prepare_normal(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        token_to_kv_pool: DeepSeekV4TokenToKVPool,
        x_for_compressor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if TYPE_CHECKING:
            assert isinstance(self, CompressorBackend)

        q = c4_indexer.compute_q(q_lora, positions=positions)
        q_fp8, q_scale = act_quant(q)
        weights = c4_indexer.compute_weights(x, skip_scale=True)
        weights = fused_scale(weights, c4_indexer.weight_scale, q_scale)
        self.forward_indexer_compressor(
            x=x_for_compressor if (is_nsa_enable_prefill_cp() and x_for_compressor is not None) else x,
            forward_batch=forward_batch,
            layer_id=c4_indexer.layer_id,
            compressor=c4_indexer.compressor,
        )
        c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=c4_indexer.layer_id,
        )
        return q_fp8, weights, c4_indexer_kv_cache

    def forward_c4_indexer(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        c4_indexer: C4Indexer,
        forward_batch: ForwardBatch,
        x_for_compressor: Optional[torch.Tensor] = None,
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        token_to_kv_pool = forward_batch.token_to_kv_pool

        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
            assert isinstance(self, CompressorBackend)

        metadata = self.forward_metadata
        indexer_metadata = metadata.indexer_metadata
        core_metadata = metadata.core_metadata

        from sglang.srt.layers.attention.deepseek_v4_backend_radix import (
            DSV4AttnMetadataRadix,
        )

        assert isinstance(core_metadata, (PagedCoreMetadata, DSV4AttnMetadataRadix))
        assert isinstance(indexer_metadata, PagedIndexerMetadata)

        _x_comp = x_for_compressor if (is_nsa_enable_prefill_cp() and x_for_compressor is not None) else x
        # Capture-mode flatten: HIP 7.0 / gfx950 SIGSEGVs in hipGraphInstantiate when
        # torch.cuda.stream() is nested inside another torch.cuda.stream() during capture.
        # The inner 2-way fork (stream_q + stream_weights) is the second level. Outer MQA
        # 3-way fork (kv/compressor/indexer) is fine. Force the flat path inside capture.
        # See /mnt/vast/john/rocm-dynamo/hip_repro/ for minimal reproducer.
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
        if enable_multi_stream and get_is_capture_mode():
            enable_multi_stream = False
            if q_lora_ready is not None:
                torch.cuda.current_stream().wait_event(q_lora_ready)
                q_lora_ready = None
        if enable_multi_stream:
            q_fp8, weights, c4_indexer_kv_cache = self._forward_prepare_multi_stream(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
                x_for_compressor=_x_comp,
                alt_streams=alt_streams,
                q_lora_ready=q_lora_ready,
            )
        else:
            assert q_lora_ready is None
            q_fp8, weights, c4_indexer_kv_cache = self._forward_prepare_normal(
                x=x,
                q_lora=q_lora,
                c4_indexer=c4_indexer,
                positions=core_metadata.positions,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
                x_for_compressor=_x_comp,
            )

        assert len(q_fp8.shape) == 3
        q_fp8 = q_fp8.unsqueeze(1)  # the next_n dim is 1 now
        assert len(c4_indexer_kv_cache.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132

        # DeepGEMM#280 does not change test_attention.py for fp8_paged_mqa_logits, thus
        c4_indexer_kv_cache = c4_indexer_kv_cache.view(
            c4_indexer_kv_cache.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
        assert len(weights.shape) == 3
        weights = weights.squeeze(2)
        # CUDA path: use deep_gemm
        if envs.SGLANG_OPT_USE_TILELANG_INDEXER.get():
            from sglang.srt.layers.attention.nsa.tilelang_kernel import (
                tilelang_fp8_paged_mqa_logits as fn,
            )
        elif envs.SGLANG_FP8_PAGED_MQA_LOGITS_AITER.get():
            # AMD path: aiter Triton/Gluon kernel.
            fn = fp8_paged_mqa_logits_aiter
        # elif is_hip():
        elif envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get():
            fn = fp8_paged_mqa_logits_torch
        else:
            if envs.SGLANG_OPT_DG_PAGED_MQA_LOGITS_CHUNK_SIZE.get() != -1:
                from sglang.srt.layers.deep_gemm_wrapper.paged_mqa_logits import (
                    fp8_paged_mqa_logits_chunked as fn,
                )
            else:
                from deep_gemm import fp8_paged_mqa_logits as fn

        logits = fn(
            q_fp8,
            c4_indexer_kv_cache,
            weights,
            indexer_metadata.c4_seq_lens,
            indexer_metadata.page_table,
            indexer_metadata.deep_gemm_metadata,
            indexer_metadata.max_seq_len,
            False,
        )

        assert indexer_metadata.page_table is core_metadata.page_table
        if self.debug_use_external_c4_sparse_indices:
            return  # skip updating page indices

        indexer_capturer = get_global_indexer_capturer()
        capture_enabled = indexer_capturer.is_enabled()

        raw_indices = None
        if capture_enabled or forward_batch.hisparse_coordinator is not None:
            raw_indices = torch.empty_like(core_metadata.c4_sparse_page_indices)

        # M3 indexer megakernel: single-launch fusion of
        # `topk_transform_512_triton` + `invalid_mask` (the mask consumed later
        # in `debug_flash_mla_adapter._get_invalid_mask`). The megakernel
        # writes both `c4_sparse_page_indices` AND a `(B*S_q, K)` int8 mask
        # buffer; the mask is published into a module-level dict keyed by
        # `(data_ptr(indices), id(topk_length))` that `_get_invalid_mask`
        # consults BEFORE its existing data_ptr cache (preserving the legacy
        # cache HIT semantics in the OFF case and on subsequent decode layers).
        #
        # Gates:
        #   - HIP only (CK V32 path is the consumer; CUDA path is unaffected)
        #   - decode forward mode only (s_q == 1; mask shape simplifies to (B,K))
        #   - `c4_sparse_topk_lengths` available on core_metadata
        #   - `topk == 512` (megakernel matches the production K)
        #   - Falls back to legacy chain on any unsupported case (megakernel
        #     returns False).
        _m3_megakernel_on = (
            is_hip()
            and os.environ.get("SGLANG_M3_INDEXER_MEGAKERNEL", "0") == "1"
            and forward_batch.forward_mode.is_decode()
            and not envs.SGLANG_TOPK_TRANSFORM_512_TORCH.get()
            and getattr(core_metadata, "c4_sparse_topk_lengths", None) is not None
        )
        _m3_fired = False
        if _m3_megakernel_on:
            from sglang.jit_kernel.m3_indexer_megakernel_triton import (
                m3_indexer_megakernel,
                publish_invalid_mask,
                ensure_invalid_mask_buffer,
            )

            mask_buf = ensure_invalid_mask_buffer(core_metadata)
            _m3_fired = m3_indexer_megakernel(
                scores=logits,
                seq_lens=indexer_metadata.c4_seq_lens,
                page_tables=core_metadata.page_table,
                topk_length=core_metadata.c4_sparse_topk_lengths,
                out_page_indices=core_metadata.c4_sparse_page_indices,
                out_raw_indices=raw_indices,
                out_invalid_mask=mask_buf,
                page_size=indexer_metadata.c4_page_size,
                s_q=1,
            )
            if _m3_fired:
                # Publish so `_get_invalid_mask` returns this buffer (avoids
                # the unfused 4-step torch chain or the `get_invalid_mask_triton`
                # MISS fallback). Bound to the megakernel firing — when the
                # kernel falls back via False return, no stale mask is
                # published.
                publish_invalid_mask(
                    indices=core_metadata.c4_sparse_page_indices,
                    topk_length=core_metadata.c4_sparse_topk_lengths,
                    mask_2d=mask_buf,
                )

        if not _m3_fired:
            if envs.SGLANG_TOPK_TRANSFORM_512_TORCH.get():
                topk_transform_512_pytorch_vectorized(
                    logits,
                    indexer_metadata.c4_seq_lens,
                    core_metadata.page_table,
                    core_metadata.c4_sparse_page_indices,
                    indexer_metadata.c4_page_size,
                    raw_indices,
                )
            elif is_hip():
                # On HIP the tvm_ffi-backed topk_transform_512 hardcodes CUDA_HOME
                # and crashes ("Could not find CUDA installation") during JIT build.
                # Route to the native Triton port instead — same algorithm
                # (bit-pack + tl.sort), AMDGCN-compatible, no nvcc dependency.
                from sglang.jit_kernel.topk_transform_512_triton import (
                    topk_transform_512_triton,
                )

                topk_transform_512_triton(
                    logits,
                    indexer_metadata.c4_seq_lens,
                    core_metadata.page_table,
                    core_metadata.c4_sparse_page_indices,
                    indexer_metadata.c4_page_size,
                    raw_indices,
                )
            else:
                topk_transform_512(
                    logits,
                    indexer_metadata.c4_seq_lens,
                    core_metadata.page_table,
                    core_metadata.c4_sparse_page_indices,
                    indexer_metadata.c4_page_size,
                    raw_indices,
                )

        if forward_batch.hisparse_coordinator is not None:
            if forward_batch.forward_mode.is_decode():
                # todo hisparse: to coordinate with kernel signature
                core_metadata.c4_sparse_page_indices = (
                    forward_batch.hisparse_coordinator.get_front_topk_tokens(
                        forward_batch.req_pool_indices,
                        forward_batch.seq_lens,
                        raw_indices,
                    )
                )
            else:
                core_metadata.c4_sparse_page_indices = token_to_kv_pool.c4_kv_pool.translate_loc_from_compressed_to_hisparse_device(
                    core_metadata.c4_sparse_page_indices
                )

        if capture_enabled:
            compress_layer_id = token_to_kv_pool.layer_mapping[
                c4_indexer.layer_id
            ].compress_layer_id
            indexer_capturer.capture(compress_layer_id, raw_indices)
