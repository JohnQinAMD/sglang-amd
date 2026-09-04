"""gfx950 fused DSA-indexer decode step: 12 launches -> 4 kernels.

Replaces, for one indexer layer of one decode forward:

  wq_b GEMM | wk GEMM | weights_proj GEMM | k_norm | rope | Hadamard(q) |
  Hadamard(k) | act_quant(q) | indexer_k_quant_and_cache(k) | head gate |
  paged-MQA fp8 logits | top-k + page transform

with

  dual_gemv_kernel<...,cfg 61> | qk_rope_hadamard_quant_kernel<true,true,true> |
  logits_hist_m<8,12> | dsa_topk::k_scatter<2,...,16>

Each kernel source ships exactly one instantiation (see the package
docstring); this module only marshals production tensors into their ABIs and
owns the persistent workspace. It makes no numerical decision of its own.

CONTRACT PRESERVED, deliberately and non-negotiably:
  * output is ``(rows, index_topk)`` int32 **physical page_size=1 KV slots**,
    ``-1`` padded -- identical in meaning to
    ``dsa_topk_backend._topk_transform_aiter_paged``.  The slot is derived
    in-kernel from the compact ``real_page_table`` as
    ``pt64[row, p >> 6] * 64 + (p & 63)``, which is the definition of
    ``page_table_1``; nothing downstream can distinguish the two.
  * BF16/FP8 dtypes, the fused Hadamard, the ue8m0 scale format and the
    preshuffled index-K cache layout are unchanged, so the cache written here is
    byte-compatible with the cache written by the standard path.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch

from sglang.kernels.ops.attention.dsa.hip_gfx950 import loader

logger = logging.getLogger(__name__)

_WARNED_CAPTURE = False
_WARNED_TOO_NARROW = False

# Kernel-source constraints. Every one of these is enforced by a TORCH_CHECK or a
# compile-time constant in csrc/, not by taste:
HEAD_DIM = 128  # qk_rope_hadamard_quant.cu / paged_mqa_logits.cu
ROPE_DIM = 64  # qk_rope_hadamard_quant.cu
N_HEADS = 32  # validated shape (grid.y of the qk kernel, MFMA n of the logits kernel)
INDEX_TOPK = 2048  # topk_transform.cu: constexpr TOPK = 2048u
PAGE_SIZE = 64  # paged_mqa_logits.cu: PAGE_TOK
CACHE_TOK_STRIDE = 132  # paged_mqa_logits.cu: TOK_STRIDE (128 K bytes + 4 scale)
Q_LORA_RANK = 2048  # dual_gemv cfg 61, Q half
HIDDEN_SIZE = 6144  # dual_gemv cfg 61, K half
QUANT_BLOCK = 128  # qk_rope_hadamard_quant.cu: quant_block_size == head_dim
# The only per-launch row cap among the four kernels (dual_gemv_bf16.cu).
# Section A is chunked to it rather than capping the whole path, so larger
# verify batches stay fused.
DUAL_GEMV_MAX_M = 8
MAX_ROWS = 48  # bs 8 x num_draft_tokens 6


def model_shape_supported(
    *,
    head_dim: int,
    rope_head_dim: int,
    n_heads: int,
    index_topk: int,
    q_lora_rank: int,
    hidden_size: int,
    quant_block: int,
    scale_fmt: Optional[str],
    is_neox_style: bool,
    k_norm,
    num_init_tokens: int,
    num_local_tokens: int,
) -> bool:
    """Static (per-model) half of the gate. Evaluated once in ``Indexer.__init__``."""
    from sglang.srt.layers.layernorm import LayerNorm

    return (
        head_dim == HEAD_DIM
        and rope_head_dim == ROPE_DIM
        and n_heads == N_HEADS
        and index_topk == INDEX_TOPK
        and q_lora_rank == Q_LORA_RANK
        and hidden_size == HIDDEN_SIZE
        and quant_block == QUANT_BLOCK
        and scale_fmt == "ue8m0"
        # GLM-5.2 sets indexer_rope_interleave=True -> is_neox_style False, which
        # is what the qk kernel's interleaved rope implements.
        and not is_neox_style
        # The kernel does a LayerNorm with weight AND bias; RMSNorm has no bias
        # and different math (index_k_norm_type == "rms").
        and isinstance(k_norm, LayerNorm)
        # _mask_init_and_local_tokens must be a no-op: the fused pipeline has no
        # place to inject the +inf streaming-attention mask between B and C.
        and num_init_tokens == 0
        and num_local_tokens == 0
    )


class _Workspace:
    """Persistent per-Indexer buffers.

    Allocated ONCE, outside any HIP graph capture, at the maximum size the
    process can ask for (``MAX_ROWS`` x the fixed decode page-table width), so a
    captured graph never records a pointer from a graph-private pool and no
    later call can force a reallocation that would invalidate an earlier graph.

    ``ghist`` carries a zero-in / zero-out invariant: the fused ``k_scatter``
    tail restores it before exiting, so it is zeroed exactly once here.
    """

    def __init__(self, *, device, max_cols: int, fp8_dtype):
        rows = MAX_ROWS
        cap = max(INDEX_TOPK, max_cols)
        gh_stride = loader.modules()[3].hist_stride()
        self.max_cols = max_cols
        self.logits = torch.empty((rows, max_cols), dtype=torch.float32, device=device)
        self.q_fp8 = torch.empty(
            (rows, N_HEADS, HEAD_DIM), dtype=fp8_dtype, device=device
        )
        self.head_gate = torch.empty(
            (rows, N_HEADS), dtype=torch.float32, device=device
        )
        self.q_proj = torch.empty(
            (rows, N_HEADS * HEAD_DIM), dtype=torch.bfloat16, device=device
        )
        self.kw = torch.empty(
            (rows, HEAD_DIM + N_HEADS), dtype=torch.bfloat16, device=device
        )
        self.ghist = torch.zeros((rows, gh_stride), dtype=torch.int32, device=device)
        self.cursor = torch.zeros((rows, 32), dtype=torch.int32, device=device)
        self.cand_cnt = torch.zeros((rows,), dtype=torch.int32, device=device)
        self.cand_idx = torch.empty((rows, cap), dtype=torch.int32, device=device)
        self.cand_val = torch.empty((rows, cap), dtype=torch.float32, device=device)
        # Row-expanded page table: metadata is per request, the kernels index
        # per row. Flat so a narrower width still views contiguously.
        self.page_table = torch.empty(
            (rows * (max_cols // PAGE_SIZE),), dtype=torch.int32, device=device
        )


# One workspace per (device, fp8 dtype), shared by every Indexer on that
# device. The buffers are pure scratch: the layers run back-to-back on the
# current stream -- this path never uses an alternate stream -- and ``ghist``
# restores its zero-in/zero-out invariant inside the kernel itself. A per-layer
# copy therefore bought nothing and multiplied the footprint by the layer count
# (79 on GLM-5.2: 78 target + 1 draft), which is device memory the rest of the
# server had already been told was free.
_WORKSPACES: Dict[Tuple[int, torch.dtype], _Workspace] = {}
_FRESH_ALLOCATION = False


def _workspace_key(device, fp8_dtype) -> Tuple[int, torch.dtype]:
    index = device.index if isinstance(device, torch.device) else device
    if index is None:
        index = torch.cuda.current_device()
    return (int(index), fp8_dtype)


def prealloc_workspace(*, device, max_cols: int, fp8_dtype) -> bool:
    """Create the shared workspace before the server sizes its memory pools.

    Order matters more than size here. The workspace is device memory, and
    ``calculate_pool_sizes`` divides whatever ``mem_get_info`` reports at that
    moment among the KV pools. Taking the workspace afterwards -- on the first
    non-capture sparse decode, which is far later -- means the pools were sized
    against memory that no longer exists, and the server spends the rest of the
    run with no headroom: on GLM-5.2 that showed up as free device memory
    sitting at 0.6-0.8 GiB for a whole 3600 s window, Triton kernels loading
    lazily mid-serve because they no longer fit at init, and decode steps
    running ~2x slower at unchanged clock and 25% lower power -- the GPU
    stalling on the allocator rather than computing.

    ``consume_fresh_allocation`` patches the consequence (it drops the stale
    MQA-logits budget so the next prefill re-reads the real figure). This
    removes the cause. Call it from Indexer construction, which runs while
    weights load and therefore before any pool is sized.

    Idempotent and shared: the first Indexer allocates, the other 78 reuse.
    Returns False if a workspace already exists or a capture is in progress.
    """
    if torch.cuda.is_current_stream_capturing():
        return False
    key = _workspace_key(device, fp8_dtype)
    if key in _WORKSPACES:
        return False
    _WORKSPACES[key] = _Workspace(device=device, max_cols=max_cols, fp8_dtype=fp8_dtype)
    global _FRESH_ALLOCATION
    _FRESH_ALLOCATION = True
    logger.info(
        "gfx950 fused DSA indexer: workspace preallocated (max_cols=%d, "
        "%.0f MiB) before pool sizing",
        max_cols,
        sum(
            t.numel() * t.element_size()
            for t in vars(_WORKSPACES[key]).values()
            if isinstance(t, torch.Tensor)
        )
        / (1 << 20),
    )
    return True


def consume_fresh_allocation() -> bool:
    """True once after a workspace was allocated, then False.

    Callers use this to drop any cached free-memory reading: the workspace is
    taken after the server has already sized other buffers against
    ``torch.cuda.mem_get_info``, so that reading no longer holds.
    """
    global _FRESH_ALLOCATION
    fresh, _FRESH_ALLOCATION = _FRESH_ALLOCATION, False
    return fresh


class Gfx950FusedIndexer:
    """Per-``Indexer`` state: merged weights, fp32 norm params, 2-D rope cache."""

    def __init__(self, indexer):
        self.indexer = indexer
        self.workspace: Optional[_Workspace] = None
        self._const = None
        # softmax_scale * n_heads**-0.5, folded exactly as _get_logits_head_gate
        # does (weights * n_heads**-0.5, then * q_scale * softmax_scale).
        self.weights_scale = float(indexer.n_heads) ** -0.5 * indexer.softmax_scale

    # -- setup-time derived constants (weights are frozen after load) ---------
    def _constants(self):
        if self._const is None:
            ix = self.indexer
            # L1: [wk ; weights_proj] -> one [160, 6144] bf16 weight. A pure
            # weight-loader style merge: both are ReplicatedLinear over the same
            # x, so one 6144->160 GEMM is arithmetically the two GEMMs.
            w_kw = torch.cat(
                [ix.wk.weight.detach(), ix.weights_proj.weight.detach()], dim=0
            ).contiguous()
            assert w_kw.dtype == torch.bfloat16, w_kw.dtype
            w_q_b = ix.wq_b.weight.detach()
            assert w_q_b.dtype == torch.bfloat16 and w_q_b.is_contiguous()

            norm_w = ix.k_norm.weight.detach().float().contiguous()
            norm_b = ix.k_norm.bias.detach().float().contiguous()

            cos2d, sin2d = _rope_2d(ix.rotary_emb, ROPE_DIM)
            self._const = (w_q_b, w_kw, norm_w, norm_b, cos2d, sin2d)
        return self._const

    def ensure_workspace(self, *, device, max_cols: int, fp8_dtype) -> bool:
        """Allocate the workspace if absent. Returns False if it cannot be used.

        Never allocates during graph capture: a buffer created inside a capture
        lives in that graph's private pool, and reusing it from a different graph
        is exactly the stale-mapping hazard this campaign already recorded once.
        """
        key = _workspace_key(device, fp8_dtype)
        ws = _WORKSPACES.get(key)
        if ws is not None:
            self.workspace = ws
            if max_cols > ws.max_cols:
                # Say so once. This refusal turns the whole feature off for the
                # rest of the run while every other log line still reports it as
                # enabled, and finding that out cost a full 1800 s A/B that
                # measured two identical arms.
                global _WARNED_TOO_NARROW
                if not _WARNED_TOO_NARROW:
                    _WARNED_TOO_NARROW = True
                    logger.warning(
                        "gfx950 fused DSA indexer: workspace is %d columns wide "
                        "but this call needs %d, so the standard indexer runs "
                        "instead. The workspace must cover the decode graph's "
                        "page-table width; check "
                        "SGLANG_DSA_HIP_FUSED_INDEXER_MAX_CTX.",
                        ws.max_cols,
                        max_cols,
                    )
                return False
            return True
        if torch.cuda.is_current_stream_capturing():
            global _WARNED_CAPTURE
            if not _WARNED_CAPTURE:
                _WARNED_CAPTURE = True
                logger.warning(
                    "gfx950 fused DSA indexer: workspace not allocated before "
                    "CUDA graph capture; using the standard indexer path"
                )
            return False
        self._constants()
        ws = _Workspace(device=device, max_cols=max_cols, fp8_dtype=fp8_dtype)
        _WORKSPACES[key] = ws
        self.workspace = ws
        global _FRESH_ALLOCATION
        _FRESH_ALLOCATION = True
        return True

    # -- the four launches ---------------------------------------------------
    def run(
        self,
        *,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        out_cache_loc: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_cache_read: torch.Tensor,
        seqlens_int32: torch.Tensor,
        row_ends_int32: torch.Tensor,
        page_table_64: torch.Tensor,
        rows: int,
    ) -> torch.Tensor:
        gemv, qk, logits_mod, topk_mod = loader.modules()
        w_q_b, w_kw, norm_w, norm_b, cos2d, sin2d = self._constants()
        ws = self.workspace
        assert ws is not None

        cols = page_table_64.shape[1] * PAGE_SIZE
        q_proj = ws.q_proj[:rows]
        kw = ws.kw[:rows]
        q_fp8 = ws.q_fp8[:rows]
        head_gate = ws.head_gate[:rows]
        logits = ws.logits[:rows, :cols]

        # --- node 1: the two independent projections in one GEMV ------------
        # oq = q_lora @ wq_b.T  (2048 -> 4096)   ok = x @ [wk ; weights_proj].T
        # Exact, not an approximation: one GEMV per row, and the slices keep
        # the stride(0) the binding reads. rows <= 8 stays a single launch.
        for i in range(0, rows, DUAL_GEMV_MAX_M):
            j = min(i + DUAL_GEMV_MAX_M, rows)
            gemv.dual_gemv_bf16(
                q_lora[i:j],
                w_q_b,
                q_proj[i:j],
                x[i:j],
                w_kw,
                kw[i:j],
                loader.DUAL_GEMV_CFG,
            )

        # --- node 2: k_norm | rope | Hadamard(q,k) | act_quant(q) |
        #             indexer_k_quant_and_cache(k) | head gate ---------------
        # hadamard=True is the whole point of the separate gfx950 gate; see
        # dsa/utils.assert_hadamard_preserved.
        qk.indexer_qk_rope_hadamard_quant_and_cache(
            q_proj.view(rows, N_HEADS, HEAD_DIM),
            q_fp8,
            kw[:, HEAD_DIM:],
            head_gate,
            kw[:, :HEAD_DIM],
            kv_cache,
            out_cache_loc,
            norm_w,
            norm_b,
            positions,
            cos2d,
            sin2d,
            float(self.indexer.k_norm.variance_epsilon),
            QUANT_BLOCK,
            self.indexer.scale_fmt,
            float(self.weights_scale),
            True,  # preshuffle
            False,  # is_neox
            True,  # compute_all_q_rope
            True,  # hadamard  <-- NOT optional on this path
        )

        # --- node 3: paged MQA fp8 logits + fine histogram + coarse summary --
        logits_mod.logits_hist(
            q_fp8,
            kv_cache_read,
            head_gate,
            seqlens_int32,
            page_table_64,
            logits,
            ws.ghist[:rows],
            loader.LOGITS_BLOCKS_PER_ROW,
            loader.LOGITS_HIST_BITS,
            INDEX_TOPK,
            loader.LOGITS_WARPS,
        )

        # --- node 4: top-k(2048) + page transform, fused refinement ---------
        # A fresh output tensor per call, matching _topk_transform_aiter_paged:
        # consumers (cross-layer MTP index share, PD serialisation) may hold a
        # reference past this layer, so the buffer must not be recycled.
        out = torch.empty((rows, INDEX_TOPK), dtype=torch.int32, device=logits.device)
        topk_mod.topk_transform(
            logits,
            row_ends_int32,
            page_table_64,
            out,
            ws.ghist[:rows],
            ws.cursor[:rows],
            ws.cand_cnt[:rows],
            ws.cand_idx[:rows],
            ws.cand_val[:rows],
            loader.TOPK_G,
            PAGE_SIZE,
        )
        return out


def _rope_2d(rotary_emb, rope_dim: int):
    """cos/sin as 2-D ``[max_pos, rope_dim // 2]`` bf16 with unit last stride."""
    half = rope_dim // 2
    cos = getattr(rotary_emb, "cos_cache", None)
    sin = getattr(rotary_emb, "sin_cache", None)
    if cos is None or sin is None:
        cache = rotary_emb.cos_sin_cache
        cos, sin = cache[..., :half], cache[..., half:]
    cos = cos.reshape(cos.shape[0], half).contiguous()
    sin = sin.reshape(sin.shape[0], half).contiguous()
    assert cos.dtype == torch.bfloat16 and sin.dtype == torch.bfloat16, (
        "gfx950 fused indexer needs a bf16 rope cache, got " f"{cos.dtype}/{sin.dtype}"
    )
    return cos, sin
