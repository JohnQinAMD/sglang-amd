from __future__ import annotations

import concurrent.futures
import logging
import os
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, List, Literal, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

import sglang.srt.models.deepseek_v2 as deepseek_v2
from sglang.jit_kernel.deepseek_v4 import fused_rope, linear_bf16_fp32
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.debug_utils.deepseek_v4_debug_utils import deepseek_v4_moe_code_path_checker
from sglang.srt.distributed import get_pp_group, get_tensor_model_parallel_world_size
from sglang.srt.distributed.parallel_state import get_moe_expert_parallel_world_size
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation
from sglang.srt.layers.attention.nsa.utils import (
    can_cp_split,
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    is_nsa_enable_prefill_cp,
    nsa_use_prefill_cp,
    prepare_input_dp_with_cp_dsa,
)
from sglang.srt.layers.communicator import LayerScatterModes, get_attn_tp_context
from sglang.srt.layers.deepseek_v4_rope import (
    apply_rotary_emb_triton,
    fused_rmsnorm_rope_q_triton,
    fused_rmsnorm_rope_q_triton_to_out,
)
# M1 megakernel — fuses kv-side RoPE + per-tile FP8 quant + paged scatter.
# Gated by envs.SGLANG_M1_KV_WRITE_WITH_ROPE (default OFF).
try:
    from sglang.jit_kernel.m1_kv_write_with_rope_triton import (
        m1_kv_write_with_rope_triton as _m1_kv_write_with_rope_triton,
    )
except Exception:  # pragma: no cover
    _m1_kv_write_with_rope_triton = None
from sglang.srt.layers.dp_attention import (
    _DpGatheredBufferWrapper,
    dp_gather_partial,
    dp_scatter,
    get_attention_dp_size,
    get_attention_tp_rank,
    get_attention_tp_size,
    get_global_dp_buffer,
    get_local_dp_buffer,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear

# Optional aiter fused q/k RMSNorm — falls back to the two-call path if not
# available (CUDA build, older aiter releases). One launch instead of two for
# the q-lora-norm + kv-norm pair in MQALayer._forward_prepare.
try:
    from aiter.ops.fused_qk_norm_rope_cache_quant import (
        fused_qk_rmsnorm as _aiter_fused_qk_rmsnorm,
    )
except Exception:  # pragma: no cover - import-time fallback only
    _aiter_fused_qk_rmsnorm = None

# 2026-05-02: aiter `fused_qk_rmsnorm_group_quant` fuses ALL THREE in one launch:
# q_norm + per-128 fp8 quant on q + kv_norm. Strict superset of both
# `_aiter_fused_qk_rmsnorm` (only norms, no quant) and `_fused_rmsnorm_per1x128_quant`
# (only q's norm+quant, kv_norm fires separately). Output:
#   q_out_quantized: fp8 [bs, q_dim]      → wq_b((fp8, scale))
#   q_out_scale:     fp32 [bs, q_dim/128]
#   k_out:           bf16 [bs, kv_dim]     → kv path (rope + attn) unchanged
#   q_out_unquantized (optional): bf16 q   → q_lora for indexer
# Gated via SGLANG_AITER_QK_RMSNORM_GROUP_QUANT=1; default OFF.
try:
    from aiter import fused_qk_rmsnorm_group_quant as _aiter_fused_qk_rmsnorm_group_quant
except Exception:  # pragma: no cover - import-time fallback only
    _aiter_fused_qk_rmsnorm_group_quant = None
_AITER_QK_RMSNORM_GROUP_QUANT = (
    os.environ.get("SGLANG_AITER_QK_RMSNORM_GROUP_QUANT", "0") == "1"
    and _aiter_fused_qk_rmsnorm_group_quant is not None
)

# A2-#1: fuse (rmsnorm + per-1x128 fp8 quant) into one Triton launch on the
# q_norm → wq_b path. Default OFF; enable via SGLANG_FUSED_RMSNORM_QUANT_PER1x128=1.
# Microbench (M=8, N=4096, MI355X cuda-graph replay): UNFUSED 12.37 µs → FUSED
# 10.28 µs (~0.29 ms/step at ~140 callsites). When enabled here we lose the qk
# rmsnorm fusion (kv_norm runs in its own launch) but eliminate the per-1x128
# quant launch before wq_b. Stack on remaining callsites in follow-up patches.
try:
    from sglang.srt.layers.quantization.fused_rmsnorm_quant import (
        fused_rmsnorm_per1x128_quant as _fused_rmsnorm_per1x128_quant,
        fused_rmsnorm_per1x128_quant_dual as _fused_rmsnorm_per1x128_quant_dual,
    )
except Exception:  # pragma: no cover
    _fused_rmsnorm_per1x128_quant = None
    _fused_rmsnorm_per1x128_quant_dual = None
_FUSED_RMSNORM_QUANT_PER1x128 = (
    os.environ.get("SGLANG_FUSED_RMSNORM_QUANT_PER1x128", "0") == "1"
    and _fused_rmsnorm_per1x128_quant is not None
)
# Phase 24: dual-output extension also gated by the same env knob; only fires
# when the dual entry-point loaded successfully.
_FUSED_RMSNORM_QUANT_PER1x128_DUAL = (
    _FUSED_RMSNORM_QUANT_PER1x128
    and _fused_rmsnorm_per1x128_quant_dual is not None
)
# F4 Mode A (2026-04-30): parallel env knob targeting the same q_norm + per-1x128
# quant fusion as _FUSED_RMSNORM_QUANT_PER1x128, but using the F4-patched kernel
# (MATCH_BF16_PRODUCTION=True, default) which round-trips `normed` through bf16
# in registers before per-block fp8 quant — making fp8 codepoints production-
# equivalent vs the unfused two-launch path. Independent of the older knob so
# in-flight experiments aren't disturbed. Default OFF.
_F4_MODE_A = (
    os.environ.get("SGLANG_F4_MODE_A", "0") == "1"
    and _fused_rmsnorm_per1x128_quant is not None
)

# Decode-body megakernel — Block A (MQA prologue). Phase 2 wire-in stub.
# Replaces the 4-op chain (input_layernorm → fp8 quant → wq_a + wkv) with one
# Triton kernel. Default OFF; gated by SGLANG_DECODE_BODY_BLOCK_A=1. The wire-in
# only activates on the indexer-bf16 q_lora path (DSv4 indexer enabled), which
# is the v1 target case the kernel was designed for. E2E activation deferred
# until baseline regression (separate track) is resolved.
_DECODE_BODY_BLOCK_A = os.environ.get("SGLANG_DECODE_BODY_BLOCK_A", "0") == "1"
try:
    from sglang.jit_kernel.decode_body_mqa_prologue_triton import (
        mqa_prologue_megakernel as _decode_body_block_a_megakernel,
    )
except Exception:
    _decode_body_block_a_megakernel = None

# Decode-body megakernel — Block B-pre (wo_b prologue). Phase 2 wire-in stub.
# Replaces the 2-op chain (per-1x128 quant → wo_b fp8 GEMM) with one Triton
# kernel. Default OFF; gated by SGLANG_DECODE_BODY_BLOCK_B_PRE=1. Microbench
# at chi2774 GPU 0 shows G0 PASS (5 shapes, max_abs_diff <= 0.0137 bf16 ULP
# floor) and 5.41x graph-replay weighted speedup over the 2-op torch path.
# E2E activation gated by user — wo_b's downstream is AllReduce, so per-step
# kernel-time savings may be absorbed by the cross-rank rendezvous (see
# `feedback_critical_path_gate_before_fusion_wire_in.md`). Block A landed
# despite the same concern via concurrency-saturation; B-pre needs the same
# E2E proof before flipping default-ON.
_DECODE_BODY_BLOCK_B_PRE = os.environ.get("SGLANG_DECODE_BODY_BLOCK_B_PRE", "0") == "1"
try:
    from sglang.jit_kernel.decode_body_b_pre_wo_triton import (
        b_pre_wo_megakernel as _decode_body_block_b_pre_megakernel,
    )
except Exception:
    _decode_body_block_b_pre_megakernel = None
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import get_moe_a2a_backend
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.rotary_embedding import get_rope_wrapper
from sglang.srt.layers.utils import get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.compress_state import (
    CompressStatePool,
    KVAndScore,
    KVAndScoreOld,
)
from sglang.srt.mem_cache.deepseekv4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.memory_pool import RadixAttention
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_loader.utils import maybe_executor_submit, should_async_load
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.dbrx import ReplicatedLinear
from sglang.srt.models.deepseek_v2 import ParallelLMHead, _is_cuda, _is_hip, _is_npu
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    BumpAllocator,
    LazyValue,
    add_prefix,
    get_bool_env_var,
    log_info_on_rank0,
    make_layers,
    maybe_torch_compile,
)

logger = logging.getLogger(__name__)

from sglang.srt.environ import envs

MOE_BIT_WISE_EQUAL_MODE = False
ATTN_BIT_WISE_EQUAL_MODE = False
COMPRESSOR_BIT_WISE_EQUAL_MODE = False
_FP8_WO_A_GEMM = envs.SGLANG_OPT_FP8_WO_A_GEMM.get()


if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4Backend
    from sglang.srt.layers.quantization import QuantizationConfig
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding
    from sglang.srt.model_executor.forward_batch_info import (
        ForwardBatch,
        PPProxyTensors,
    )


class DeepseekRefRMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # rmsnorm in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        out = rms_normalize_triton(x, self.eps, self.weight)
        return out


@maybe_torch_compile
def rms_normalize(x: torch.Tensor, eps: float) -> torch.Tensor:
    x *= torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return x


# HC pre/post are decorated at module level so torch.compile is invoked at most
# once per process. Previously these were nested inside hc_pre/hc_post methods,
# which created a new local function object each call. With maybe_torch_compile's
# in-capture wrap (`if get_is_capture_mode(): return torch.compile(func)`),
# every cuda graph capture call recompiled the same code from scratch — 43
# layers × N captured batch sizes × 2 functions = hundreds of fresh Inductor
# compiles, all serializing on the file_baton lock. That was the cuda-graph
# capture wedge that shadowed every stacked-best run.
@maybe_torch_compile
def _hc_pre_torch_impl(x, hc_fn, rms_norm_eps: float):
    x_flat = x.flatten(1).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_norm_eps)
    mixes = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)
    return x_flat, mixes


@triton.jit
def _hc_pre_fused_kernel(
    x_ptr,           # bf16 [M, HIDDEN]  (HIDDEN = HC_MULT * HC_DIM)
    hc_fn_ptr,       # fp32 [HC_MULT, HIDDEN]
    x_flat_out_ptr,  # fp32 [M, HIDDEN]
    mixes_out_ptr,   # fp32 [M, HC_MULT]
    eps,
    M,
    HIDDEN: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused (bf16 x copy → fp32 x_flat) + (rmsnorm) + (hc_fn @ x_flat) → mixes.

    Replaces the unfused 4-kernel chain in `_hc_pre_torch_impl` for the prefill
    path (`@maybe_torch_compile` wraps decode, but eager prefill goes through
    the raw Python). At M=8192 microbench: 316 → 112 us (2.83x), cos_sim 1.000000.
    """
    pid_m = tl.program_id(0)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    sum_sq = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HC_MULT), dtype=tl.float32)

    for k_start in range(0, HIDDEN, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < HIDDEN
        mask_2d = m_mask[:, None] & k_mask[None, :]

        x_offs = m_offs[:, None] * HIDDEN + k_offs[None, :]
        x_block = tl.load(x_ptr + x_offs, mask=mask_2d, other=0.0).to(tl.float32)

        # write x_flat (downstream `pre.unsqueeze(-1) * x_flat.view(shape)` needs it)
        tl.store(x_flat_out_ptr + x_offs, x_block, mask=mask_2d)

        # accumulate per-row sum of squares
        sum_sq += tl.sum(x_block * x_block, axis=1)

        # accumulate hc_fn[n, k] @ x[m, k] for each output column n
        for n in tl.static_range(0, HC_MULT):
            hc_row = tl.load(hc_fn_ptr + n * HIDDEN + k_offs, mask=k_mask, other=0.0)
            acc_n = tl.sum(x_block * hc_row[None, :], axis=1)
            acc = tl.where(
                tl.arange(0, HC_MULT)[None, :] == n,
                acc + acc_n[:, None],
                acc,
            )

    rsqrt = tl.rsqrt(sum_sq / HIDDEN + eps)
    out = acc * rsqrt[:, None]

    out_offs = m_offs[:, None] * HC_MULT + tl.arange(0, HC_MULT)[None, :]
    out_mask = m_mask[:, None] & (tl.arange(0, HC_MULT)[None, :] < HC_MULT)
    tl.store(mixes_out_ptr + out_offs, out, mask=out_mask)


def hc_pre_fused_triton(x: torch.Tensor, hc_fn: torch.Tensor, eps: float):
    """Fused-Triton drop-in for `_hc_pre_torch_impl`.

    Returns (x_flat, mixes) matching the production signature:
      x_flat: fp32 [M, HIDDEN] (= flatten + bf16->fp32 copy)
      mixes:  fp32 [M, 1, HC_MULT] (= rmsnorm * (x_flat @ hc_fn^T))

    Only correct when x.shape == (M, HC_MULT, HC_DIM) and hc_fn.shape ==
    (HC_MULT, HC_MULT * HC_DIM). Caller must guard.
    """
    M_, HC_MULT_, HC_DIM_ = x.shape
    HIDDEN_ = HC_MULT_ * HC_DIM_
    x_flat_out = torch.empty(M_, HIDDEN_, dtype=torch.float32, device=x.device)
    mixes_out = torch.empty(M_, HC_MULT_, dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M_, 32),)
    _hc_pre_fused_kernel[grid](
        x.contiguous(), hc_fn, x_flat_out, mixes_out,
        eps, M_,
        HIDDEN=HIDDEN_, HC_MULT=HC_MULT_,
        BLOCK_M=32, BLOCK_K=256,
    )
    return x_flat_out, mixes_out.unsqueeze(1)


@maybe_torch_compile
def _hc_post_torch_impl(x, residual, post, comb):
    return (
        post.unsqueeze(-1) * x.unsqueeze(1)
        + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
    ).type_as(x)


@triton.jit
def _hc_post_fused_kernel(
    x_ptr,         # bf16 [M, HIDDEN]
    residual_ptr,  # bf16 [M, HC_MULT, HIDDEN]
    post_ptr,      # fp32 [M, HC_MULT]
    comb_ptr,      # fp32 [M, HC_MULT, HC_MULT]
    out_ptr,       # bf16 [M, HC_MULT, HIDDEN]
    M,
    HIDDEN: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused hc_post.

    Eager Python materializes a (M, HC_MULT, HC_MULT, HIDDEN) intermediate
    (3.75 GB at M=8192) before sum. This kernel keeps the per-(m, d) accumulator
    in registers — same correctness (cos_sim 1.000001 vs torch eager), 23.02x
    faster (5444 → 236 us microbench at M=8192). 82.4% of HBM bound.
    """
    pid_m = tl.program_id(0)
    pid_d = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    m_mask = m_offs < M
    d_mask = d_offs < HIDDEN
    md_mask = m_mask[:, None] & d_mask[None, :]

    # x (BLOCK_M, BLOCK_D) bf16 → fp32 (used by every hc_out)
    x_offs = m_offs[:, None] * HIDDEN + d_offs[None, :]
    x_block = tl.load(x_ptr + x_offs, mask=md_mask, other=0.0).to(tl.float32)

    # post (BLOCK_M, HC_MULT) fp32 — full row per token
    hc_offs = tl.arange(0, HC_MULT)
    post_offs = m_offs[:, None] * HC_MULT + hc_offs[None, :]
    post_block = tl.load(post_ptr + post_offs, mask=m_mask[:, None], other=0.0)

    for hc_out in tl.static_range(0, HC_MULT):
        # extract column hc_out from post (BLOCK_M,)
        post_col = tl.sum(
            post_block * (hc_offs[None, :] == hc_out).to(tl.float32),
            axis=1,
        )
        out = post_col[:, None] * x_block

        for hc_in in tl.static_range(0, HC_MULT):
            comb_offs = m_offs * (HC_MULT * HC_MULT) + hc_in * HC_MULT + hc_out
            comb_col = tl.load(comb_ptr + comb_offs, mask=m_mask, other=0.0)

            res_offs = (m_offs[:, None] * (HC_MULT * HIDDEN)
                        + hc_in * HIDDEN
                        + d_offs[None, :])
            res_block = tl.load(residual_ptr + res_offs, mask=md_mask, other=0.0).to(tl.float32)
            out += comb_col[:, None] * res_block

        out_offs = (m_offs[:, None] * (HC_MULT * HIDDEN)
                    + hc_out * HIDDEN
                    + d_offs[None, :])
        tl.store(out_ptr + out_offs, out.to(tl.bfloat16), mask=md_mask)


def hc_post_fused_triton(x, residual, post, comb):
    """Fused-Triton drop-in for `_hc_post_torch_impl`. Bf16 only.

    Caller must pre-validate shapes match (M, HIDDEN), (M, HC_MULT, HIDDEN),
    (M, HC_MULT), (M, HC_MULT, HC_MULT).
    """
    M_, HIDDEN_ = x.shape
    HC_MULT_ = residual.shape[1]
    out = torch.empty(M_, HC_MULT_, HIDDEN_, dtype=torch.bfloat16, device=x.device)
    BLOCK_M, BLOCK_D = 32, 64
    grid = (triton.cdiv(M_, BLOCK_M), triton.cdiv(HIDDEN_, BLOCK_D))
    _hc_post_fused_kernel[grid](
        x.contiguous(), residual.contiguous(),
        post.contiguous(), comb.contiguous(),
        out, M_,
        HIDDEN=HIDDEN_, HC_MULT=HC_MULT_,
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D,
    )
    return out


# ===== A2-#2: mhc_post_fused (mul + sum + cast) =====
@triton.jit
def _mhc_post_mul_sum_cast_kernel(
    pre_ptr,     # fp32 [B, HC_MULT]
    x_flat_ptr,  # fp32 [B, HC_MULT * HIDDEN] (= [B, HC_MULT, HIDDEN] flattened)
    out_ptr,     # bf16 [B, HIDDEN]
    B,
    HIDDEN: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused replacement for `(pre.unsqueeze(-1) * x_flat.view(B, HC, H)).sum(1).to(bf16)`.

    Eager chain is 3+ launches (broadcast mul, sum, cast). This kernel keeps
    the per-(b, d) accumulator in registers; HC_MULT is a small static range
    (=4 for Pro / Flash-Base). 99.6% HBM-bound on MI355X (microbench).
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    b_offs = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    b_mask = b_offs < B
    d_mask = d_offs < HIDDEN
    bd_mask = b_mask[:, None] & d_mask[None, :]

    hc_offs = tl.arange(0, HC_MULT)
    pre_offs = b_offs[:, None] * HC_MULT + hc_offs[None, :]
    pre_block = tl.load(pre_ptr + pre_offs, mask=b_mask[:, None], other=0.0)

    acc = tl.zeros((BLOCK_B, BLOCK_D), dtype=tl.float32)
    for hc in tl.static_range(0, HC_MULT):
        pre_col = tl.sum(
            pre_block * (hc_offs[None, :] == hc).to(tl.float32),
            axis=1,
        )
        x_offs = (
            b_offs[:, None] * (HC_MULT * HIDDEN)
            + hc * HIDDEN
            + d_offs[None, :]
        )
        x_block = tl.load(x_flat_ptr + x_offs, mask=bd_mask, other=0.0)
        acc += pre_col[:, None] * x_block

    out_offs = b_offs[:, None] * HIDDEN + d_offs[None, :]
    tl.store(out_ptr + out_offs, acc.to(tl.bfloat16), mask=bd_mask)


def mhc_post_fused_mul_sum_cast(pre, x_flat, shape, out_dtype):
    """Fused drop-in for `(pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(1).to(out_dtype)`.

    Args:
        pre:    (B, 1, HC_MULT) fp32
        x_flat: (B, HC_MULT*HIDDEN) fp32  (output of hc_pre_fused_triton)
        shape:  (B, HC_MULT, HIDDEN)
        out_dtype: bf16

    Returns:
        (B, HIDDEN) tensor of out_dtype.

    Microbench (B=8192 HC=4 H=7168, MI355X): 561 -> 195 us eager (2.87x),
    566 -> 212 us cuda-graph (2.68x), 99.6% HBM-bound.
    """
    B_, HC_MULT_, HIDDEN_ = shape
    out = torch.empty(B_, HIDDEN_, dtype=out_dtype, device=pre.device)
    BLOCK_B, BLOCK_D = 64, 256
    grid = (triton.cdiv(B_, BLOCK_B), triton.cdiv(HIDDEN_, BLOCK_D))
    _mhc_post_mul_sum_cast_kernel[grid](
        pre.squeeze(1).contiguous(),
        x_flat.contiguous(),
        out, B_,
        HIDDEN=HIDDEN_, HC_MULT=HC_MULT_,
        BLOCK_B=BLOCK_B, BLOCK_D=BLOCK_D,
    )
    return out


@triton.jit
def _rms_normalize_kernel(
    x_ptr,
    weight_ptr,
    eps,
    stride_row,
    dim,
    BLOCK_SIZE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < dim

    base = pid * stride_row
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # x / sqrt(mean(x^2) + eps)
    mean_sq = tl.sum(x * x, axis=0) / dim
    rms_inv = tl.rsqrt(mean_sq + eps)
    out = x * rms_inv

    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + offs, mask=mask, other=0.0)
        out = out * weight

    tl.store(x_ptr + base + offs, out, mask=mask)


def rms_normalize_triton(
    x: torch.Tensor, eps: float, weight: torch.Tensor = None
) -> torch.Tensor:
    """RMS normalize with optional weight.

    Args:
        x: Input tensor of shape (..., dim), normalizes over last dimension
        eps: Epsilon for numerical stability
        weight: Optional weight tensor of shape (dim,)
    """
    dim = x.shape[-1]
    x_flat = x.view(-1, dim)
    num_rows = x_flat.shape[0]

    BLOCK_SIZE = triton.next_power_of_2(dim)
    grid = (num_rows,)

    _rms_normalize_kernel[grid](
        x_flat,
        weight,
        eps,
        x_flat.stride(0),
        dim,
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_WEIGHT=(weight is not None),
    )
    return x


class Compressor(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        is_in_indexer: bool,
        rotary_emb: RotaryEmbedding,
        freqs_cis: torch.Tensor,  # TODO: remove it after using rotary embedding
        compress_ratio: Literal[0, 4, 128],
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_in_indexer = is_in_indexer
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        self.nope_head_dim = head_dim - self.rope_head_dim
        assert compress_ratio != 0, "compress_ratio should not be 0"
        self.ratio = compress_ratio
        self.overlap = self.ratio == 4
        self.rotate = rotate
        self.coff = coff = 1 + self.overlap

        self.ape = nn.Parameter(
            torch.empty(self.ratio, coff * self.head_dim, dtype=torch.float32)
        )
        # fuse wkv and wgate into wkv_gate, merge the last dim
        wkv_gate_dtype = torch.bfloat16
        self.wkv_gate = ReplicatedLinear(
            self.dim,
            2 * coff * self.head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wkv_gate", prefix),
            params_dtype=wkv_gate_dtype,
        )
        # NOTE: kept on DeepseekRefRMSNorm (not the fast SGLang RMSNorm) because
        # `kv_compressed` here is asserted fp32 (see compress_extend / compress_decode)
        # and HIP's fast RMSNorm path dispatches to `aiter.rmsnorm2d_fwd` which
        # only supports fp16/bf16 (verified RuntimeError on this stack). Casting
        # fp32→bf16→fp32 around the call would bypass that limit but introduces
        # ~2.5% quantization noise, which violates the asserted-fp32 contract.
        # self.norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.norm = DeepseekRefRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = rotary_emb
        self.freqs_cis = freqs_cis

        self.ape_converted = False

        # Replay-stable scratch for compress_decode / compress_extend /
        # overlap_transform. Root cause: advanced indexing / fancy alloc
        # inside `torch.cuda.graph` binds caching-allocator addresses; if a
        # later eager (bs > captured_bs) call reallocates the same scratch,
        # the captured graph's frozen pointer points to freed memory and
        # HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION (0x29) fires on the
        # next replay.
        #
        # Fix: keep TWO dicts, one for the captured graph and one for eager
        # work. Capture-time allocations land in `_scratch_captured` and
        # are NEVER reassigned (replay doesn't run Python). Eager calls
        # land in `_scratch_eager`, which is free to reallocate per shape
        # since eager has no replay-stability requirement.
        self._scratch_captured: dict = {}
        self._scratch_eager: dict = {}

    @cached_property
    def use_fused_compress(self) -> bool:
        if (
            envs.SGLANG_OPT_USE_FUSED_PAGED_COMPRESS.get()
            and envs.SGLANG_OPT_DPSK_V4_RADIX.get()
        ):
            return True
        return (
            envs.SGLANG_OPT_USE_FUSED_COMPRESS.get()
            and not envs.SGLANG_OPT_DPSK_V4_RADIX.get()
        )

    def apply_ape_hotfix(self):
        assert not self.ape_converted
        self.ape_converted = True

        # ========== copied from the hotfix in "260119-updated" of ref code ==========
        is_model_2604 = envs.SGLANG_DSV4_MODE.get() == "2604"
        if (
            self.overlap
            and not is_model_2604
            and get_bool_env_var("SGLANG_ENABLE_APE_HOTFIX", "1")
        ):
            # NOTE: We reorder the parameters here to match the layout of the provided checkpoint.
            # This is only required for compatibility with this checkpoint; the official version
            # does not need this reordering.
            ape = torch.chunk(self.ape.data, 2, dim=-1)
            if self.use_fused_compress:
                ape = torch.cat([ape[1], ape[0]], dim=0)
            else:
                ape = torch.cat([ape[1], ape[0]], dim=-1)
            self.ape.data.copy_(ape.view(self.ratio, -1))
            # ============================================================================

    # Pre-allocated cap for eager scratch (covers chunked_prefill_size + slack).
    # Setting this larger than any expected prefill batch ensures the eager
    # scratch never grows after first alloc — preventing the freed-storage
    # reuse race with captured-graph slabs (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION)
    # described at lines 280-292.
    _EAGER_SCRATCH_MAX_ROWS = int(os.environ.get("SGLANG_EAGER_SCRATCH_MAX_ROWS", "8192"))

    def _ensure_scratch(
        self, name, shape, dtype, device, grow_dim_0_only=True,
    ) -> torch.Tensor:
        """Return a contiguous tensor of `shape`, backed by per-mode scratch.

        Capture phase (`is_current_stream_capturing()`) → `_scratch_captured`,
        which is allocated lazily and never reassigned thereafter. Eager
        calls → `_scratch_eager`, **preallocated at `_EAGER_SCRATCH_MAX_ROWS`
        on first alloc** so it never grows afterward. Without this cap, the
        first eager prefill after capture could grow the scratch and free
        its old storage back into the caching allocator, where it can
        collide with addresses captured graphs are still using → IMA on
        replay (compress_extend_old:1070 fancy-index symptom).

        `grow_dim_0_only=True` (default): the backing tensor's dim 0 is the
        only one allowed to grow; trailing dims must match exactly. The
        returned slice `[:shape[0]]` is contiguous, so callers can `.view()`
        on it. Use `grow_dim_0_only=False` for tensors whose later dims
        also vary — those reallocate on any mismatch (no slicing).
        """
        scratch = (
            self._scratch_captured
            if torch.cuda.is_current_stream_capturing()
            else self._scratch_eager
        )
        cur = scratch.get(name)
        target = tuple(shape)
        if grow_dim_0_only:
            need_realloc = (
                cur is None
                or cur.dtype != dtype
                or cur.device != device
                or tuple(cur.shape[1:]) != target[1:]
                or cur.shape[0] < target[0]
            )
            if need_realloc:
                if (
                    cur is not None
                    and cur.dtype == dtype
                    and cur.device == device
                    and tuple(cur.shape[1:]) == target[1:]
                ):
                    new_rows = max(cur.shape[0], target[0])
                else:
                    new_rows = target[0]
                # Eager scratch: pre-pay the max size so we never realloc
                # after the first call. Captured scratch keeps lazy growth
                # since capture happens at fixed bs.
                if scratch is self._scratch_eager:
                    new_rows = max(new_rows, self._EAGER_SCRATCH_MAX_ROWS)
                scratch[name] = torch.empty(
                    (new_rows,) + target[1:], dtype=dtype, device=device,
                )
            return scratch[name][: target[0]]
        else:
            need_realloc = (
                cur is None
                or cur.dtype != dtype
                or cur.device != device
                or tuple(cur.shape) != target
            )
            if need_realloc:
                scratch[name] = torch.empty(target, dtype=dtype, device=device)
            return scratch[name]

    def _get_states(self, forward_batch: ForwardBatch) -> KVAndScore:
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            return token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            return token_to_kv_pool.get_attention_compress_states(self.layer_id)

    def _get_state_pool(self, forward_batch: ForwardBatch) -> CompressStatePool:
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            ret = token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        else:
            ret = token_to_kv_pool.get_attention_compress_states(self.layer_id)

        assert isinstance(ret, CompressStatePool)

        return ret

    def overlap_transform(self, tensor: torch.Tensor, fill_value: Any) -> torch.Tensor:
        # tensor: [block_num, r, 2 * d]
        assert tensor.dim() == 3
        assert tensor.shape[1:] == (self.ratio, 2 * self.head_dim)

        s, r, d = tensor.size(0), self.ratio, self.head_dim
        # Per-mode persistent scratch (`_ensure_scratch`) — captured graph
        # gets a frozen buffer, eager grows. Avoids HSA 0x29 from captured
        # replay reading freed bs=1 storage after eager bs>1 grew it.
        new_tensor = self._ensure_scratch(
            "overlap_xform", (s, 2 * r, d), tensor.dtype, tensor.device,
        )
        # Item #2 REVERTED for clean-baseline measurement. Restored 3-launch torch path.
        new_tensor.fill_(fill_value)
        new_tensor[:, r:] = tensor[:, :, d:]
        new_tensor[1:, :r] = tensor[:-1, :, :d]
        return new_tensor

    def overlap_transform_decode(self, tensor: torch.Tensor) -> torch.Tensor:
        # NOTE: the default value has been initialized when creating the states
        # tensor: [bs, 2 * r, 2 * d]
        assert tensor.dim() == 3
        assert tensor.shape[1:] == (2 * self.ratio, 2 * self.head_dim)
        r, d = self.ratio, self.head_dim
        ret = torch.cat((tensor[:, :r, :d], tensor[:, r:, d:]), dim=1)
        return ret

    @staticmethod
    def compute_state_len(seq_len: int, ratio: int):
        """Tailing length for the valid states in kv cache.
        When overlap is enabled, there is always an extra block: [extra block, compressing part]
        """
        return seq_len % ratio + (ratio == 4) * ratio

    @staticmethod
    def compute_state_len_indices(seq_len: int, ratio: int):
        state_len = seq_len % ratio + (ratio == 4) * ratio
        # NOTE: -1 here means invalid position
        return torch.arange(seq_len - state_len, seq_len).clamp(min=-1)

    def print_tensor(self, y: torch.Tensor, name: str):
        enable = int(os.environ.get("SGLANG_ENABLE_PRINT_TENSOR", 0))
        if enable:
            print(f"[sgl] {name}: shape={y.shape}, dtype={y.dtype}, device={y.device}")
            print(f"{y.flatten()[:10]}...{y.flatten()[-10:]}")

    def compress_extend_paged(
        self,
        kv_and_scores: KVAndScore,
        forward_batch: ForwardBatch,
    ):
        backend = forward_batch.attn_backend
        if TYPE_CHECKING:
            assert isinstance(backend, DeepseekV4Backend)
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        # extract some info
        state_pool = self._get_state_pool(forward_batch)
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        extend_lens = forward_batch.extend_seq_lens_cpu
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        assert not self.forward_mode.is_target_verify()

        assert extend_lens is not None and prefix_lens is not None
        device = kv_and_scores.kv.device

        # Deliberately fill w/ huge values, s.t. when misuse and access the unfilled values,
        # we have higher probability to see something very weird
        assert kv_and_scores.kv.shape[-1] == self.head_dim * self.coff
        compressed_kv_output = torch.full(
            (kv_and_scores.kv.size(0), self.head_dim),
            fill_value=10000.0,
            dtype=kv_and_scores.kv.dtype,
            device=device,
        )

        bs = forward_batch.batch_size
        pt = 0
        # Pre-host req_pool_indices once. PyTorch's advanced-indexing fast path
        # calls .item() on a 0-dim CUDA index tensor (~366 us GPU sync each)
        # when the index is `cuda_tensor[python_int]`. One D2H of bs ints up
        # front saves `bs * N_use_sites` syncs across this prefill batch.
        # Sync-reduction hygiene: even if GPU is idle now (host-bound at
        # baseline), this future-proofs against busier GPU regimes.
        req_pool_indices_cpu = req_pool_indices.tolist()
        for i in range(bs):
            kv_and_score = kv_and_scores[pt : pt + extend_lens[i]]
            pre_state_indices = self.compute_state_len_indices(
                seq_len=prefix_lens[i], ratio=self.ratio
            ).to(device)
            raw_loc = torch.where(
                pre_state_indices < 0,
                -1,
                req_to_token[req_pool_indices_cpu[i], pre_state_indices],
            )
            swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(raw_loc)
            state_loc = state_pool.translate_from_swa_loc_to_state_loc(swa_loc)
            pre_kv_state = state_pool.get_state_by_state_loc(state_loc)
            kv_and_score_buffer = KVAndScore.cat([pre_kv_state, kv_and_score], dim=0)
            valid_kv_len = kv_and_score_buffer.kv.size(0)

            post_state_indices = self.compute_state_len_indices(
                seq_len=prefix_lens[i] + extend_lens[i], ratio=self.ratio
            ).to(device)
            post_state_len = post_state_indices.size(0)

            # write to kv_and_score_states
            assert post_state_len <= valid_kv_len
            post_raw_loc = torch.where(
                post_state_indices < 0,
                -1,
                req_to_token[req_pool_indices_cpu[i], post_state_indices],
            )
            post_swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(post_raw_loc)
            post_state_loc = state_pool.translate_from_swa_loc_to_state_loc(
                post_swa_loc
            )
            post_state_to_set = kv_and_score_buffer[valid_kv_len - post_state_len :]
            state_pool.set_state_by_state_loc(post_state_loc, post_state_to_set)

            # Get the part that can be compressed (ratio-aligned)
            compress_len = valid_kv_len // self.ratio * self.ratio
            if compress_len == 0:
                # Nothing to compress yet, just update pointers
                pt += extend_lens[i]
                continue

            # kv to compress: [compressed_len, ratio, head_dim * coff]
            kv_and_score_to_compress = kv_and_score_buffer[:compress_len].view(
                compress_len // self.ratio, self.ratio, -1
            )
            # NOTE: apply ape only when compressing
            kv_and_score_to_compress.score.add_(self.ape.unsqueeze(0))

            # Apply overlap transformation if enabled
            if self.overlap:
                new_kv = self.overlap_transform(
                    kv_and_score_to_compress.kv, fill_value=0
                )
                new_score = self.overlap_transform(
                    kv_and_score_to_compress.score, fill_value=float("-inf")
                )
                kv_and_score_to_compress = KVAndScore.from_kv_score(
                    kv=new_kv, score=new_score
                )
                del new_kv, new_score
                # remove the first block before compression
                kv_and_score_to_compress = kv_and_score_to_compress[1:]

                if kv_and_score_to_compress.kv.size(0) == 0:
                    pt += extend_lens[i]
                    continue

            kv_compressed = (
                kv_and_score_to_compress.kv
                * kv_and_score_to_compress.score.softmax(dim=1)
            ).sum(dim=1)

            # NOTE: ref code requires dtype as the same as hidden states (float32)
            # the raw output of kv_compressed is float32 already
            assert kv_compressed.dtype == torch.float32
            kv_compressed = self.norm(kv_compressed)

            beg_idx = prefix_lens[i] // self.ratio * self.ratio
            end_idx = (prefix_lens[i] + extend_lens[i]) // self.ratio * self.ratio
            freqs_cis = self.freqs_cis[beg_idx : end_idx : self.ratio]
            assert freqs_cis.size(0) == kv_compressed.size(
                0
            ), f"{freqs_cis.shape=} {kv_compressed.shape=}"
            apply_rotary_emb_triton(
                kv_compressed[..., -self.rope_head_dim :], freqs_cis
            )
            del beg_idx, end_idx

            if self.rotate:
                kv_compressed = rotate_activation(kv_compressed)

            # get all the pos: ratio * n + (ratio - 1) > prefix_len - 1
            start = prefix_lens[i]
            start = start + self.ratio - 1 - start % self.ratio
            indices_in_seq = torch.arange(
                start,
                prefix_lens[i] + extend_lens[i],
                self.ratio,
                device=kv_and_scores.kv.device,
            )
            assert indices_in_seq.size(0) == kv_compressed.size(0)
            compressed_kv_output[indices_in_seq - prefix_lens[i] + pt] = kv_compressed

            pt += extend_lens[i]

        return compressed_kv_output

    def compress_decode_paged(
        self,
        kv_and_scores: KVAndScore,
        forward_batch: ForwardBatch,
    ):
        """Paged and cudagraph compatible version of compress_decode"""
        assert self.ape_converted
        state_pool = self._get_state_pool(forward_batch)
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        seq_lens = forward_batch.seq_lens

        if forward_batch.forward_mode.is_target_verify():
            draft_tokens = forward_batch.attn_backend.speculative_num_draft_tokens
            offsets = torch.arange(1, draft_tokens + 1, device=seq_lens.device)
            seq_lens_2d = seq_lens[:, None] + offsets[None, :]
            seq_lens = seq_lens_2d.view(-1)
            req_pool_indices = req_pool_indices.repeat_interleave(draft_tokens)

        raw_locs = req_to_token[req_pool_indices, seq_lens - 1]

        # Update the new decode states
        swa_locs = token_to_kv_pool.translate_loc_from_full_to_swa(raw_locs)
        state_locs = state_pool.translate_from_swa_loc_to_state_loc(swa_locs)
        state_pool.set_state_by_state_loc(state_locs, kv_and_scores)

        compress_bulk_len = self.ratio * self.coff
        compress_indices = seq_lens[:, None] + torch.arange(
            -compress_bulk_len, 0, device=seq_lens.device
        )
        compress_indices.clamp_(min=-1)
        compress_indices_raw = torch.where(
            compress_indices < 0,
            -1,
            req_to_token[req_pool_indices[:, None], compress_indices],
        )
        compress_indices_swa = token_to_kv_pool.translate_loc_from_full_to_swa(
            compress_indices_raw
        )
        compress_indices_state = state_pool.translate_from_swa_loc_to_state_loc(
            compress_indices_swa
        )
        kv_and_score_to_compress = state_pool.get_state_by_state_loc(
            compress_indices_state.view(-1)
        ).view(-1, self.ratio, self.coff * self.head_dim)
        kv_and_score_to_compress.score.add_(self.ape.unsqueeze(0))

        bs = seq_lens.size(0)
        if self.overlap:
            # shape: [bs, coff * ratio, coff * head_dim]
            kv_and_score_to_compress = kv_and_score_to_compress.view(
                bs, self.coff * self.ratio, self.coff * self.head_dim
            )
            kv_and_score_to_compress = KVAndScore.from_kv_score(
                kv=self.overlap_transform_decode(kv_and_score_to_compress.kv),
                score=self.overlap_transform_decode(kv_and_score_to_compress.score),
            )

        self.print_tensor(kv_and_score_to_compress.kv, "kv_to_compress")
        self.print_tensor(kv_and_score_to_compress.score, "score_to_compress")

        # kv_to_compress: [bs, ratio * coff, head_dim]
        kv_and_score_to_compress = kv_and_score_to_compress.view(
            bs, self.ratio * self.coff, self.head_dim
        )

        kv_compressed = (
            kv_and_score_to_compress.kv * kv_and_score_to_compress.score.softmax(dim=1)
        ).sum(dim=1)
        self.print_tensor(kv_compressed, "kv_before_norm")
        kv_compressed = self.norm(kv_compressed)
        self.print_tensor(kv_compressed, "kv_after_norm")
        freqs_cis = self.freqs_cis[(seq_lens - 1) // self.ratio * self.ratio]
        self.print_tensor(freqs_cis, "freqs_cis")
        apply_rotary_emb_triton(kv_compressed[..., -self.rope_head_dim :], freqs_cis)
        self.print_tensor(kv_compressed, "kv_after_rope")
        if self.rotate:
            kv_compressed = rotate_activation(kv_compressed)

        # `new_compressed_list` format is only used for testing
        self.print_tensor(kv_compressed, "compressed_kv_output")
        return kv_compressed

    def compress_extend(
        self,
        kv_and_scores: KVAndScore,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        assert self.ape_converted  # Please keep this assertion

        # kv_and_score_states: [max_num_reqs, compress_ratio * coff, head_dim * coff]
        kv_and_score_states = self._get_states(forward_batch)
        _, _, head_dim_times_coff = kv_and_score_states.kv.shape

        # extract some info
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        extend_lens = forward_batch.extend_seq_lens_cpu
        req_pool_indices = forward_batch.req_pool_indices
        assert extend_lens is not None and prefix_lens is not None

        # compress info — replay-stable scratch instead of fresh per-call allocs.
        # `temp_buffer.new_empty(...)` and `torch.full(...)` here used to bind the
        # caching allocator each call; under c≥8 multi-bs eager pressure those
        # addresses get churned between captures and replays → stale-pointer
        # HSA 0x29 on the next captured-graph decode. Persistent scratch with
        # grow-only resize keeps addresses stable.
        max_buffer_size = 2 * kv_and_score_states.shape[1] + kv_and_scores.shape[0]

        # Backing kv_score tensor is [N, 2 * head_dim_times_coff]: KVAndScore
        # stores kv and score concatenated on last dim. The original
        # `kv_and_scores.new_empty([..., head_dim_times_coff])` doubles last
        # dim and wraps; we replicate that.
        backing_last = 2 * head_dim_times_coff
        backing_dtype = kv_and_scores.kv_score.dtype
        backing_device = kv_and_scores.kv_score.device
        tb = self._ensure_scratch(
            "extend_temp", (max_buffer_size, backing_last), backing_dtype, backing_device,
        )
        # Wrap the slice in KVAndScore (matches the type returned by the
        # original `kv_and_scores.new_empty(...)`).
        from sglang.srt.mem_cache.compress_state import KVAndScore as _KVAndScore
        temp_buffer = _KVAndScore(tb)

        assert kv_and_scores.kv.shape[-1] == self.head_dim * self.coff
        out_rows = kv_and_scores.kv.size(0)
        compressed_kv_output = self._ensure_scratch(
            "extend_output", (out_rows, self.head_dim),
            kv_and_scores.kv.dtype, kv_and_scores.kv.device,
        )
        # A2-#2: sentinel fill is debug-only ("misuse should produce obvious
        # junk"). Default OFF — every row is overwritten by the per-request
        # scatter below, so the fill is purely a defensive invariant check.
        # Set SGLANG_COMPRESS_SENTINEL=1 to restore.
        if os.environ.get("SGLANG_COMPRESS_SENTINEL", "0") == "1":
            compressed_kv_output.fill_(10000.0)

        bs = forward_batch.batch_size
        pt = 0
        # Pre-host req_pool_indices once. PyTorch's advanced-indexing fast path
        # calls .item() on a 0-dim CUDA index tensor (~366 us GPU sync each)
        # when the index is `cuda_tensor[python_int]`. One D2H of bs ints up
        # front saves `bs * 2 * N_compressor_layers` syncs per prefill batch.
        req_pool_indices_cpu = req_pool_indices.tolist()
        for i in range(bs):
            # Definitions of variables
            #
            # kv_and_score_state: (compress_ratio * coff, head_dim * coff)
            #     only it[:old_valid_state_len] has valid data
            #
            # kv_and_score_buffer: (old_valid_state_len + valid_kv_len, head_dim * coff)
            #     content is cat(kv_and_score_state[:old_valid_state_len], kv_and_score)

            kv_and_score = kv_and_scores[pt : pt + extend_lens[i]]
            kv_and_score_state = kv_and_score_states[req_pool_indices_cpu[i]]
            if prefix_lens[i] == 0:
                # NOTE: padding with default values for overlap
                kv_and_score_state.clear()

            # Create kv_and_score_buffer
            pre_state_len = self.compute_state_len(
                seq_len=prefix_lens[i], ratio=self.ratio
            )
            valid_kv_len = pre_state_len + extend_lens[i]
            kv_and_score_buffer = temp_buffer[:valid_kv_len]
            kv_and_score_buffer[:pre_state_len] = kv_and_score_state[:pre_state_len]
            kv_and_score_buffer[pre_state_len:valid_kv_len] = kv_and_score

            # Write to kv_and_score_states
            post_state_len = self.compute_state_len(
                seq_len=valid_kv_len, ratio=self.ratio
            )
            kv_and_score_state[:post_state_len] = kv_and_score_buffer[
                valid_kv_len - post_state_len : valid_kv_len
            ]

            # Get the part that can be compressed (ratio-aligned)
            compress_len = valid_kv_len // self.ratio * self.ratio
            if compress_len == 0:
                # Nothing to compress yet, just update pointers
                pt += extend_lens[i]
                continue

            # kv to compress: [compressed_len, ratio, head_dim * coff]
            kv_and_score_to_compress = kv_and_score_buffer[:compress_len].view(
                compress_len // self.ratio, self.ratio, -1
            )
            # NOTE: apply ape only when compressing
            kv_and_score_to_compress.score.add_(self.ape.unsqueeze(0))

            # Apply overlap transformation if enabled
            if self.overlap:
                new_kv = self.overlap_transform(
                    kv_and_score_to_compress.kv, fill_value=0
                )
                new_score = self.overlap_transform(
                    kv_and_score_to_compress.score, fill_value=float("-inf")
                )
                kv_and_score_to_compress = KVAndScore.from_kv_score(
                    kv=new_kv, score=new_score
                )
                del new_kv, new_score
                # remove the first block before compression
                kv_and_score_to_compress = kv_and_score_to_compress[1:]

                if kv_and_score_to_compress.kv.size(0) == 0:
                    pt += extend_lens[i]
                    continue

            kv_compressed = (
                kv_and_score_to_compress.kv
                * kv_and_score_to_compress.score.softmax(dim=1)
            ).sum(dim=1)

            # NOTE: ref code requires dtype as the same as hidden states (float32)
            # the raw output of kv_compressed is float32 already
            assert kv_compressed.dtype == torch.float32
            kv_compressed = self.norm(kv_compressed)

            beg_idx = prefix_lens[i] // self.ratio * self.ratio
            end_idx = (prefix_lens[i] + extend_lens[i]) // self.ratio * self.ratio
            freqs_cis = self.freqs_cis[beg_idx : end_idx : self.ratio]
            assert freqs_cis.size(0) == kv_compressed.size(
                0
            ), f"{freqs_cis.shape=} {kv_compressed.shape=}"
            apply_rotary_emb_triton(
                kv_compressed[..., -self.rope_head_dim :], freqs_cis
            )
            del beg_idx, end_idx

            if self.rotate:
                kv_compressed = rotate_activation(kv_compressed)

            # get all the pos: ratio * n + (ratio - 1) > prefix_len - 1
            start = prefix_lens[i]
            start = start + self.ratio - 1 - start % self.ratio
            indices_in_seq = torch.arange(
                start,
                prefix_lens[i] + extend_lens[i],
                self.ratio,
                device=kv_and_scores.kv.device,
            )
            assert indices_in_seq.size(0) == kv_compressed.size(0)
            compressed_kv_output[indices_in_seq - prefix_lens[i] + pt] = kv_compressed

            pt += extend_lens[i]

        return compressed_kv_output

    @maybe_torch_compile
    def compress_decode(
        self,
        kv_and_scores: KVAndScore,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        assert self.ape_converted  # Please keep this assertion

        seq_lens = forward_batch.seq_lens
        kv_and_score_states_pool = self._get_states(forward_batch)
        req_pool_indices = forward_batch.req_pool_indices

        # NOTE: first, write to the states
        bs = kv_and_scores.kv.size(0)
        write_pos = (seq_lens - 1) % self.ratio + self.overlap * self.ratio
        kv_and_score_states_pool[req_pool_indices, write_pos] = kv_and_scores

        # NOTE: need to copy out before modifying overlap states.
        # Replay-stable read into pre-allocated scratch (see __init__ comment).
        # Equivalent to `kv_and_score_states_pool[req_pool_indices]` but with a
        # stable backing pointer across cuda-graph replays.
        pool_kv_score = kv_and_score_states_pool.kv_score
        decode_pool_scratch = self._ensure_scratch(
            "decode_pool", (max(bs, 1), *pool_kv_score.shape[1:]),
            pool_kv_score.dtype, pool_kv_score.device,
        )
        torch.index_select(
            pool_kv_score, 0, req_pool_indices, out=decode_pool_scratch,
        )
        kv_and_score_to_compress = KVAndScore(decode_pool_scratch)

        # Shift just compressed kv states left by ratio
        if self.overlap:
            should_shift = seq_lens % self.ratio == 0
            kv_and_score_states_pool[req_pool_indices, : self.ratio] = KVAndScore(
                kv_score=torch.where(
                    should_shift[:, None, None],
                    kv_and_score_to_compress.kv_score[:, self.ratio :],
                    kv_and_score_to_compress.kv_score[:, : self.ratio],
                )
            )

        # shape: [bs * coff, ratio, coff * head_dim]
        kv_and_score_to_compress = kv_and_score_to_compress.view(
            -1, self.ratio, self.coff * self.head_dim
        )
        kv_and_score_to_compress.score.add_(self.ape.unsqueeze(0))

        if self.overlap:
            # shape: [bs, coff * ratio, coff * head_dim]
            kv_and_score_to_compress = kv_and_score_to_compress.view(
                bs, self.coff * self.ratio, self.coff * self.head_dim
            )
            kv_and_score_to_compress = KVAndScore.from_kv_score(
                kv=self.overlap_transform_decode(kv_and_score_to_compress.kv),
                score=self.overlap_transform_decode(kv_and_score_to_compress.score),
            )

        self.print_tensor(kv_and_score_to_compress.kv, "kv_to_compress")
        self.print_tensor(kv_and_score_to_compress.score, "score_to_compress")

        # kv_to_compress: [bs, ratio * coff, head_dim]
        kv_and_score_to_compress = kv_and_score_to_compress.view(
            bs, self.ratio * self.coff, self.head_dim
        )

        kv_compressed = (
            kv_and_score_to_compress.kv * kv_and_score_to_compress.score.softmax(dim=1)
        ).sum(dim=1)
        self.print_tensor(kv_compressed, "kv_before_norm")
        kv_compressed = self.norm(kv_compressed)
        self.print_tensor(kv_compressed, "kv_after_norm")
        # Replay-stable freqs_cis read. The original
        #   freqs_cis = self.freqs_cis[(seq_lens - 1) // self.ratio * self.ratio]
        # advanced-indexes into self.freqs_cis at runtime, allocating a fresh
        # output each call. Inside cuda-graph capture the next op
        # (apply_rotary_emb_triton) records the output pointer; on replay the
        # caching allocator can re-bind to a different address → APERTURE_VIOLATION.
        freqs_idx = (seq_lens - 1) // self.ratio * self.ratio
        freqs_cis = self._ensure_scratch(
            "decode_freqs",
            (max(freqs_idx.shape[0], 1), *self.freqs_cis.shape[1:]),
            self.freqs_cis.dtype, self.freqs_cis.device,
        )
        torch.index_select(self.freqs_cis, 0, freqs_idx, out=freqs_cis)
        self.print_tensor(freqs_cis, "freqs_cis")
        apply_rotary_emb_triton(kv_compressed[..., -self.rope_head_dim :], freqs_cis)
        self.print_tensor(kv_compressed, "kv_after_rope")
        if self.rotate:
            kv_compressed = rotate_activation(kv_compressed)

        # `new_compressed_list` format is only used for testing
        self.print_tensor(kv_compressed, "compressed_kv_output")
        return kv_compressed

    def compress_fused(
        self,
        kv_score: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # TODO: this should be the final implementation after verifying correctness
        backend = forward_batch.attn_backend
        if TYPE_CHECKING:
            assert isinstance(backend, DeepseekV4Backend)
        is_paged = envs.SGLANG_OPT_DPSK_V4_RADIX.get()
        if is_paged:
            kv_score_buffer = self._get_state_pool(forward_batch)
            kv_score_buffer = kv_score_buffer.kv_score_buffer.kv_score
        else:
            kv_score_buffer = self._get_states(forward_batch).kv_score
        return backend.forward_compress(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score,
            ape=self.ape.view(-1, self.head_dim),
            head_dim=self.head_dim,
            norm=self.norm,
            freqs_cis_cache=self.freqs_cis,
            rotate=self.rotate,
            compress_ratio=self.ratio,
            forward_batch=forward_batch,
            is_paged=is_paged,
        )

    def compress_dispatch(
        self,
        kv_score: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.use_fused_compress:
            return self.compress_fused(kv_score, forward_batch)

        if envs.SGLANG_OPT_USE_OLD_COMPRESSOR.get():
            kv = kv_score[:, : self.coff * self.head_dim]
            score = kv_score[:, self.coff * self.head_dim :]
            kv_and_scores = KVAndScoreOld(kv=kv, score=score)
            self.compress_decode = self.compress_decode_old
            self.compress_extend = self.compress_extend_old
        else:
            if envs.SGLANG_OPT_DPSK_V4_RADIX.get():
                self.compress_decode = self.compress_decode_paged
                self.compress_extend = self.compress_extend_paged
            kv_and_scores = KVAndScore(kv_score)
        if TYPE_CHECKING:
            assert isinstance(kv_and_scores, KVAndScore)

        if (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        ):
            result = self.compress_decode(
                kv_and_scores=kv_and_scores,
                forward_batch=forward_batch,
            )
        elif forward_batch.forward_mode.is_extend():
            result = self.compress_extend(
                kv_and_scores=kv_and_scores,
                forward_batch=forward_batch,
            )
        else:
            msg = f"Forward mode {forward_batch.forward_mode} not supported in Compressor."
            raise NotImplementedError(msg)

        return result

    def forward(self, x: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return x.new_empty(0, self.head_dim)

        self.forward_mode = forward_batch.forward_mode

        # Cannot use aiter.fused_qk_rmsnorm here: wkv_gate emits
        # [bs, 2*coff*head_dim] which is [kv | score], NOT [q | k]. The two
        # halves go through different downstream ops (kv is mixed by the
        # softmax of score; score itself is never normed). And `self.norm`
        # only fires AFTER the score-weighted sum, on a single fp32 tensor —
        # no second tensor exists to pair-norm with. Structure does not fit.
        kv_score = linear_bf16_fp32(x, self.wkv_gate.weight)
        return self.compress_dispatch(kv_score, forward_batch)

    def compress_extend_old(
        self, kv_and_scores: KVAndScore, forward_batch: ForwardBatch
    ) -> torch.Tensor:
        assert self.ape_converted  # Please keep this assertion
        KVAndScore = KVAndScoreOld

        # kv_and_score_states: [max_num_reqs, compress_ratio * coff, head_dim * coff]
        kv_and_score_states = self._get_states(forward_batch)
        _, _, head_dim_times_coff = kv_and_score_states.kv.shape

        # extract some info
        prefix_lens = forward_batch.extend_prefix_lens_cpu
        extend_lens = forward_batch.extend_seq_lens_cpu
        req_pool_indices = forward_batch.req_pool_indices

        # compress info — replay-stable scratch (mirrors compress_extend at L693-L709).
        # Original `KVAndScore.empty_like(...)` and `torch.full(...)` here used to
        # bind fresh caching-allocator addresses each call. Under c=16 multi-bs
        # capture pressure (bs={1,2,4,8,16}) the eager allocator churns these
        # addresses against captured-graph slabs in the graph pool, creating an
        # async write-read race that surfaces as HIP IMA in this same function
        # later (see /mnt/vast/john/rocm-dynamo/A4_HIP_IMA_REPRO.md).
        max_buffer_size = 2 * kv_and_score_states.shape[1] + kv_and_scores.shape[0]

        # temp_buffer: KVAndScoreOld stores kv and score in SEPARATE tensors,
        # so we need two scratches (vs the new path which has one combined).
        temp_buffer_kv = self._ensure_scratch(
            "extend_old_temp_kv", (max_buffer_size, head_dim_times_coff),
            kv_and_scores.kv.dtype, kv_and_scores.kv.device,
        )
        temp_buffer_score = self._ensure_scratch(
            "extend_old_temp_score", (max_buffer_size, head_dim_times_coff),
            kv_and_scores.score.dtype, kv_and_scores.score.device,
        )
        temp_buffer = KVAndScore(kv=temp_buffer_kv, score=temp_buffer_score)

        assert kv_and_scores.kv.shape[-1] == self.head_dim * self.coff
        compressed_kv_output = self._ensure_scratch(
            "extend_old_output", (kv_and_scores.kv.size(0), self.head_dim),
            kv_and_scores.kv.dtype, kv_and_scores.kv.device,
        )
        # A2-#2: sentinel fill is debug-only. Default OFF (every row gets
        # overwritten by the per-request scatter below); set
        # SGLANG_COMPRESS_SENTINEL=1 to restore the invariant check.
        if os.environ.get("SGLANG_COMPRESS_SENTINEL", "0") == "1":
            compressed_kv_output.fill_(10000.0)

        bs = forward_batch.batch_size
        pt = 0
        # Pre-host req_pool_indices once. PyTorch's advanced-indexing fast path
        # calls .item() on a 0-dim CUDA index tensor (~366 us GPU sync each)
        # when the index is `cuda_tensor[python_int]`. One D2H of bs ints up
        # front saves `bs * 2 * N_compressor_layers` syncs per prefill batch.
        req_pool_indices_cpu = req_pool_indices.tolist()

        # 2026-05-04 MEGA-3' Stage 1+2 (KVAndScoreOld variant) — production wire-in.
        # Stage 1: per-request {clear-state-if-prefix-zero + cat-prefix + cat-current
        # + writeback} → 2 Triton launches (one per channel: kv fill=0, score fill=-inf).
        # Stage 2: per-request {APE-add + overlap_transform + drop-first-block} →
        # 2 Triton launches (one per channel) producing flat (total_out_blocks, 2*R, D)
        # tensors that feed the existing per-request compress_decode_full_triton chain.
        # Combined: 16 launches/request × bs × 60 layers → 4 + bs/layer launches.
        # Default-on via launch_dsv4.sh:323-333. Microbench: Stage 1 = 2.49-2.66× at
        # bs=4, Stage 2 = 6.81-7.14× at production. E2E: TPOT 21.46→21.36 ms (-0.10),
        # throughput 340.91→341.88 (+0.97 tok/s, +0.28%).
        _mega3_old_setup_on = (
            os.environ.get("SGLANG_MEGA3_PRIME_EXTEND_SETUP_OLD", "0") == "1"
            and self.ratio == 4
            and self.overlap
        )
        _mega3_stage2_on = (
            _mega3_old_setup_on
            and os.environ.get("SGLANG_MEGA3_PRIME_OVERLAP_APE_DROP", "0") == "1"
        )
        _mega3_setup_descs = None
        _mega3_stage2_out = None
        if _mega3_old_setup_on:
            from sglang.jit_kernel.extend_per_request_megakernel_triton import (
                mega3_prime_extend_setup_old_triton,
            )
            _prefix_t = torch.tensor(prefix_lens, dtype=torch.int32) if not torch.is_tensor(prefix_lens) else prefix_lens.to(torch.int32)
            _extend_t = torch.tensor(extend_lens, dtype=torch.int32) if not torch.is_tensor(extend_lens) else extend_lens.to(torch.int32)
            _mega3_setup_descs = mega3_prime_extend_setup_old_triton(
                kv_and_score_states.kv,
                kv_and_score_states.score,
                kv_and_scores.kv,
                kv_and_scores.score,
                temp_buffer.kv,
                temp_buffer.score,
                req_pool_indices,
                _prefix_t,
                _extend_t,
            )
        if _mega3_stage2_on and _mega3_setup_descs is not None:
            from sglang.jit_kernel.extend_per_request_megakernel_triton import (
                mega3_prime_overlap_ape_drop_triton,
            )
            (_pt_offsets, _buf_offsets, _pre_state_lens_t,
             _post_state_lens_t, _valid_kv_lens_t) = _mega3_setup_descs
            _n_in_blocks_cpu = (_valid_kv_lens_t // self.ratio).to(torch.int32)
            _per_req_out = torch.clamp(_n_in_blocks_cpu - 1, min=0)
            _out_block_offsets_cpu = torch.zeros_like(_per_req_out)
            _running = 0
            for _i in range(bs):
                _out_block_offsets_cpu[_i] = _running
                _running += int(_per_req_out[_i])
            _total_out = _running
            if _total_out > 0:
                _device = temp_buffer.kv.device
                _n_blocks_dev = _n_in_blocks_cpu.to(_device, non_blocking=True)
                _out_block_offsets_dev = _out_block_offsets_cpu.to(_device, non_blocking=True)
                _out_kv = torch.empty(
                    _total_out, 2 * self.ratio, self.head_dim,
                    dtype=temp_buffer.kv.dtype, device=_device,
                )
                _out_score = torch.empty(
                    _total_out, 2 * self.ratio, self.head_dim,
                    dtype=temp_buffer.score.dtype, device=_device,
                )
                _ok = mega3_prime_overlap_ape_drop_triton(
                    temp_buffer.kv, temp_buffer.score,
                    _out_kv, _out_score, self.ape,
                    _buf_offsets, _n_blocks_dev, _out_block_offsets_dev,
                    self.ratio, self.head_dim,
                )
                if _ok:
                    _mega3_stage2_out = (_out_kv, _out_score, _out_block_offsets_cpu, _n_in_blocks_cpu)

        for i in range(bs):
            # Definitions of variables
            #
            # kv_and_score_state: (compress_ratio * coff, head_dim * coff)
            #     only it[:old_valid_state_len] has valid data
            #
            # kv_and_score_buffer: (old_valid_state_len + valid_kv_len, head_dim * coff)
            #     content is cat(kv_and_score_state[:old_valid_state_len], kv_and_score)

            kv_and_score = kv_and_scores[pt : pt + extend_lens[i]]
            kv_and_score_state = kv_and_score_states[req_pool_indices_cpu[i]]

            # Stage 1 path: read pre-built per-request slice from temp_buffer
            if _mega3_setup_descs is not None:
                pt_offsets, buf_offsets, pre_state_lens_t, post_state_lens_t, valid_kv_lens_t = _mega3_setup_descs
                pre_state_len = int(pre_state_lens_t[i].item())
                post_state_len = int(post_state_lens_t[i].item())
                valid_kv_len = int(valid_kv_lens_t[i].item())
                buf_off = int(buf_offsets[i].item())
                kv_and_score_buffer = temp_buffer[buf_off:buf_off + valid_kv_len]
            else:
                if prefix_lens[i] == 0:
                    # NOTE: padding with default values for overlap
                    kv_and_score_state.clear()

                # Create kv_and_score_buffer
                pre_state_len = self.compute_state_len(
                    seq_len=prefix_lens[i], ratio=self.ratio
                )
                valid_kv_len = pre_state_len + extend_lens[i]
                kv_and_score_buffer = temp_buffer[:valid_kv_len]
                kv_and_score_buffer[:pre_state_len] = kv_and_score_state[:pre_state_len]
                kv_and_score_buffer[pre_state_len:valid_kv_len] = kv_and_score

                # Write to kv_and_score_states
                post_state_len = self.compute_state_len(
                    seq_len=valid_kv_len, ratio=self.ratio
                )
                kv_and_score_state[:post_state_len] = kv_and_score_buffer[
                    valid_kv_len - post_state_len : valid_kv_len
                ]

            # Get the part that can be compressed (ratio-aligned)
            compress_len = valid_kv_len // self.ratio * self.ratio
            if compress_len == 0:
                # Nothing to compress yet, just update pointers
                pt += extend_lens[i]
                continue

            # MEGA-3' Stage 2 path: Stage 2 kernel already produced flat (n_out, 2R, D)
            # tensors with APE+overlap_transform+drop-first applied. Read this request's
            # slice directly. Skip the per-request torch APE+overlap_transform compute.
            if _mega3_stage2_out is not None:
                _out_kv, _out_score, _out_block_offsets_cpu, _n_in_blocks_cpu = _mega3_stage2_out
                _n_out_i = max(0, int(_n_in_blocks_cpu[i].item()) - 1)
                if _n_out_i == 0:
                    pt += extend_lens[i]
                    continue
                _out_off = int(_out_block_offsets_cpu[i].item())
                from sglang.srt.mem_cache.compress_state import KVAndScore as _KVAS
                kv_and_score_to_compress = _KVAS.from_kv_score(
                    kv=_out_kv[_out_off:_out_off + _n_out_i],
                    score=_out_score[_out_off:_out_off + _n_out_i],
                )
            else:
                # kv to compress: [compressed_len, ratio, head_dim * coff]
                kv_and_score_to_compress = kv_and_score_buffer[:compress_len].view(
                    compress_len // self.ratio, self.ratio, -1
                )
                # M2-extend: APE add fuses INTO the megakernel ADD_APE branch on
                # the no-overlap path (c128 layers, S = ratio = 128). For overlap=True
                # (c4 layers), overlap_transform reorders elements between APE-add
                # and softmax, so we keep the explicit add before the transform.
                _fuse_ape_in_kernel = not self.overlap
                if not _fuse_ape_in_kernel:
                    kv_and_score_to_compress.score = (
                        kv_and_score_to_compress.score + self.ape.unsqueeze(0)
                    )

                # Apply overlap transformation if enabled
                if self.overlap:
                    kv_and_score_to_compress.kv = self.overlap_transform(
                        kv_and_score_to_compress.kv, 0
                    )
                    kv_and_score_to_compress.score = self.overlap_transform(
                        kv_and_score_to_compress.score, float("-inf")
                    )

                    # remove the first block before compression
                    kv_and_score_to_compress = kv_and_score_to_compress[1:]

                    if kv_and_score_to_compress.kv.size(0) == 0:
                        pt += extend_lens[i]
                        continue

            beg_idx = prefix_lens[i] // self.ratio * self.ratio
            end_idx = (prefix_lens[i] + extend_lens[i]) // self.ratio * self.ratio
            freqs_cis = self.freqs_cis[beg_idx : end_idx : self.ratio]

            # M2-extend megakernel: fuse softmax + mul + sum + RMSNorm + RoPE
            # (+ APE add for no-overlap = 8 ops) into one Triton launch.
            # Replaces 6 separate launches per call × ~25 calls/step in the
            # production prefill hot path. Mixed dtype: kv/score bf16 in,
            # kv_compressed fp32 out (kernel internally promotes to fp32).
            from sglang.jit_kernel.compress_decode_megakernel_triton import (
                compress_decode_full_triton,
            )
            n_blocks = kv_and_score_to_compress.kv.size(0)
            # Reshape to (n_blocks, S=ratio*coff, D=head_dim) for the megakernel.
            _kv_in = kv_and_score_to_compress.kv.contiguous().view(
                n_blocks, self.ratio * self.coff, self.head_dim
            )
            _score_in = kv_and_score_to_compress.score.contiguous().view(
                n_blocks, self.ratio * self.coff, self.head_dim
            )
            assert freqs_cis.size(0) == n_blocks, (
                f"{freqs_cis.shape=} expected first dim = {n_blocks}"
            )
            _freqs_per_bs = torch.view_as_real(freqs_cis).contiguous()  # (n_blocks, rope_dim/2, 2)
            _norm_w_fp32 = (
                self.norm.weight if self.norm.weight.dtype == torch.float32
                else self.norm.weight.to(torch.float32)
            )
            _ape_arg = (
                self.ape.view(
                    self.ratio * self.coff, self.head_dim
                ).contiguous().to(_score_in.dtype)
                if _fuse_ape_in_kernel else None
            )
            kv_compressed = torch.empty(
                (n_blocks, self.head_dim),
                dtype=torch.float32,
                device=_kv_in.device,
            )
            if not compress_decode_full_triton(
                _kv_in, _score_in, _ape_arg,
                _norm_w_fp32, _freqs_per_bs,
                self.norm.eps, self.rope_head_dim,
                kv_compressed,
            ):
                # Fallback: original 4-step torch path.
                _score_for_softmax = kv_and_score_to_compress.score
                if _fuse_ape_in_kernel:
                    _score_for_softmax = _score_for_softmax + self.ape.unsqueeze(0)
                kv_compressed = (
                    kv_and_score_to_compress.kv
                    * _score_for_softmax.softmax(dim=1)
                ).sum(dim=1)
                assert kv_compressed.dtype == torch.float32
                kv_compressed = self.norm(kv_compressed)
                apply_rotary_emb_triton(
                    kv_compressed[..., -self.rope_head_dim :], freqs_cis
                )
            del beg_idx, end_idx

            if self.rotate:
                kv_compressed = rotate_activation(kv_compressed)

            # get all the pos: ratio * n + (ratio - 1) > prefix_len - 1
            start = prefix_lens[i]
            start = start + self.ratio - 1 - start % self.ratio
            indices_in_seq = torch.arange(
                start,
                prefix_lens[i] + extend_lens[i],
                self.ratio,
                device=kv_and_scores.kv.device,
            )
            assert indices_in_seq.size(0) == kv_compressed.size(0)
            compressed_kv_output[indices_in_seq - prefix_lens[i] + pt] = kv_compressed

            pt += extend_lens[i]

        return compressed_kv_output

    def compress_decode_old(
        self,
        kv_and_scores: KVAndScore,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        assert self.ape_converted  # Please keep this assertion
        KVAndScore = KVAndScoreOld

        seq_lens = forward_batch.seq_lens
        kv_and_score_states_pool = self._get_states(forward_batch)
        req_pool_indices = forward_batch.req_pool_indices

        bs = kv_and_scores.kv.size(0)

        # 2026-05-01: optional fused kv_pool maintain + gather + APE add Triton
        # kernel (compress_decode_kv_pool_fused). Replaces 5+ launches with 1.
        # Microbench-validated 8x graph-replay speedup on c4 ratio=4 production
        # shapes. Default OFF; gated by SGLANG_FUSED_COMPRESS_DECODE_KV_POOL=1.
        # Pending E2E live smoke + TPOT bench gate (per the M1/B-pre/hc_pre
        # microbench-pass-but-E2E-fail lessons).
        _use_fused_pool = (
            os.environ.get("SGLANG_FUSED_COMPRESS_DECODE_KV_POOL", "0") == "1"
        )
        # 2026-05-01 Phase 2: V2 absorbs overlap_transform_decode into the kernel.
        # Microbench-validated 1.84-2.33x graph-replay speedup vs V1+xform on
        # c4 ratio=4 production shapes. Default OFF; gated by env knob.
        # When ON, output is already at (bs, ratio*coff, head_dim) shape and the
        # downstream `if self.overlap: ... overlap_transform_decode ...` block
        # at L1641-1651 is skipped (output shape doesn't need transform).
        _use_fused_pool_v2 = (
            os.environ.get("SGLANG_FUSED_COMPRESS_DECODE_KV_POOL_V2", "0") == "1"
        )
        # V2 implies V1 path (V2 builds on V1's kernel)
        if _use_fused_pool_v2:
            _use_fused_pool = True
        if _use_fused_pool and _use_fused_pool_v2:
            from sglang.jit_kernel.compress_decode_kv_pool_fused_triton import (
                compress_decode_kv_pool_fused_v2,
            )
            out_kv, out_score = compress_decode_kv_pool_fused_v2(
                kv_and_score_states_pool.kv,
                kv_and_score_states_pool.score,
                req_pool_indices,
                seq_lens,
                kv_and_scores.kv,
                kv_and_scores.score,
                self.ape,
                self.ratio,
                self.overlap,
                self.head_dim,
            )
            kv_and_score_to_compress = KVAndScoreOld(kv=out_kv, score=out_score)
        elif _use_fused_pool:
            from sglang.jit_kernel.compress_decode_kv_pool_fused_triton import (
                compress_decode_kv_pool_fused,
            )
            out_kv, out_score = compress_decode_kv_pool_fused(
                kv_and_score_states_pool.kv,
                kv_and_score_states_pool.score,
                req_pool_indices,
                seq_lens,
                kv_and_scores.kv,
                kv_and_scores.score,
                self.ape,
                self.ratio,
                self.overlap,
            )
            kv_and_score_to_compress = KVAndScoreOld(kv=out_kv, score=out_score)
        else:
            write_pos = (seq_lens - 1) % self.ratio + self.overlap * self.ratio
            kv_and_score_states_pool[req_pool_indices, write_pos] = kv_and_scores

            # NOTE: need to copy out before modifying overlap states
            # kv_states: [bs, coff * ratio, coff * head_dim]
            kv_and_score_to_compress = kv_and_score_states_pool[req_pool_indices]

            if self.overlap:
                # Shift just compressed kv states left by ratio
                should_shift = seq_lens % self.ratio == 0
                kv_and_score_states_pool[req_pool_indices, : self.ratio] = KVAndScore(
                    kv=torch.where(
                        should_shift[:, None, None],
                        kv_and_score_to_compress.kv[:, self.ratio :],
                        kv_and_score_to_compress.kv[:, : self.ratio],
                    ),
                    score=torch.where(
                        should_shift[:, None, None],
                        kv_and_score_to_compress.score[:, self.ratio :],
                        kv_and_score_to_compress.score[:, : self.ratio],
                    ),
                )

            # shape: [bs * coff, ratio, coff * head_dim]
            kv_and_score_to_compress = kv_and_score_to_compress.view(
                -1, self.ratio, self.coff * self.head_dim
            )
            kv_and_score_to_compress.score = (
                kv_and_score_to_compress.score + self.ape.unsqueeze(0)
            )

        if self.overlap and not _use_fused_pool_v2:
            # V2 absorbs overlap_transform_decode into the fused kernel —
            # output is already at (bs, ratio*coff, head_dim) so this block
            # is skipped when V2 is on.
            # shape: [bs, coff * ratio, coff * head_dim]
            kv_and_score_to_compress = kv_and_score_to_compress.view(
                bs, self.coff * self.ratio, self.coff * self.head_dim
            )
            kv_and_score_to_compress.kv = self.overlap_transform_decode(
                kv_and_score_to_compress.kv
            )
            kv_and_score_to_compress.score = self.overlap_transform_decode(
                kv_and_score_to_compress.score
            )

        self.print_tensor(kv_and_score_to_compress.kv, "kv_to_compress")
        self.print_tensor(kv_and_score_to_compress.score, "score_to_compress")

        # kv_to_compress: [bs, ratio * coff, head_dim]
        kv_and_score_to_compress = kv_and_score_to_compress.view(
            bs, self.ratio * self.coff, self.head_dim
        )

        # M2-Stage2 megakernel: fuse softmax + mul + sum + RMSNorm + RoPE
        # (7 ops) into one Triton launch. APE was already added above
        # (line 1523-1525) and baked into score (after overlap_transform_decode
        # if overlap=True), so pass ape=None. Falls back to torch when S > 16.
        # v2 framework verdict: SHIP (correctness PASS, eager 7.35×,
        # graph-replay 5.24× on production shape histogram).
        from sglang.jit_kernel.compress_decode_megakernel_triton import (
            compress_decode_full_triton,
        )
        _kv_in = kv_and_score_to_compress.kv.contiguous()
        _score_in = kv_and_score_to_compress.score.contiguous()
        kv_compressed = torch.empty(
            (bs, self.head_dim), dtype=_kv_in.dtype, device=_kv_in.device,
        )
        _norm_w_fp32 = (
            self.norm.weight if self.norm.weight.dtype == torch.float32
            else self.norm.weight.to(torch.float32)
        )
        # Pre-compute per-bs freqs lookup (single index_select; same launch
        # the original code does for `freqs_cis = self.freqs_cis[...]`).
        # 2026-05-01: tried per-step memoization (SGLANG_FREQS_IDX_CACHE) keyed
        # on id(forward_batch). REVERTED — Paris smoke returned "London" instead
        # of "Paris" because Python's id() reuses memory addresses after objects
        # are freed, causing stale-cache hits across decode steps. Same trap
        # as feedback_data_ptr_caching_unsafe.md. Getting cross-step caching
        # safe requires either content-hash key (sync, breaks cuda graph) or
        # precompute-in-metadata-builder (multi-file refactor).
        # 2026-05-02: kernel-level fusion (B200-style). One Triton kernel
        # replaces the 4-launch chain (sub + floor_divide + mul + index_select)
        # with a single per-batch program. Yield est: 22 layers × 3 launches
        # eliminated × 1.5 us = 0.10 ms TPOT. Gated by SGLANG_FUSED_FREQS_IDX_GATHER=1.
        if os.environ.get("SGLANG_FUSED_FREQS_IDX_GATHER", "0") == "1":
            from sglang.jit_kernel.freqs_idx_gather_triton import freqs_idx_gather_triton
            _freqs_per_bs = freqs_idx_gather_triton(seq_lens, self.freqs_cis, self.ratio)
        else:
            _freqs_idx = (seq_lens - 1) // self.ratio * self.ratio
            _freqs_per_bs = torch.view_as_real(self.freqs_cis[_freqs_idx]).contiguous()  # (bs, rope_dim//2, 2)
        if compress_decode_full_triton(
            _kv_in, _score_in, None,
            _norm_w_fp32, _freqs_per_bs,
            self.norm.eps, self.rope_head_dim,
            kv_compressed,
        ):
            # Megakernel did softmax+mul+sum + RMSNorm + RoPE in one launch.
            self.print_tensor(kv_compressed, "kv_after_rope")
        else:
            # Fallback: original 4-step torch path (Stage1 + RMSNorm + freqs_idx + RoPE)
            kv_compressed = (
                kv_and_score_to_compress.kv
                * kv_and_score_to_compress.score.softmax(dim=1)
            ).sum(dim=1)
            self.print_tensor(kv_compressed, "kv_before_norm")
            kv_compressed = self.norm(kv_compressed)
            self.print_tensor(kv_compressed, "kv_after_norm")
            freqs_cis = self.freqs_cis[(seq_lens - 1) // self.ratio * self.ratio]
            self.print_tensor(freqs_cis, "freqs_cis")
            apply_rotary_emb_triton(kv_compressed[..., -self.rope_head_dim :], freqs_cis)
            self.print_tensor(kv_compressed, "kv_after_rope")
        if self.rotate:
            kv_compressed = rotate_activation(kv_compressed)

        # `new_compressed_list` format is only used for testing
        new_compressed_list = None
        self.print_tensor(kv_compressed, "compressed_kv_output")
        return kv_compressed


class C4Indexer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        rotary_emb: RotaryEmbedding,
        freqs_cis: torch.Tensor,  # TODO: remove it after using rotary embedding
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.softmax_scale = self.head_dim**-0.5
        # TODO: do we need to support TP indexer?
        # currently, we duplicate indexer on all TP ranks
        self.n_local_heads = self.n_heads
        self.wq_b = ReplicatedLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("wq_b", prefix),
        )
        self.weights_proj = ReplicatedLinear(
            self.dim,
            self.n_heads,
            bias=False,
            quant_config=None,
            params_dtype=torch.bfloat16,
            prefix=add_prefix("weights_proj", prefix),
        )
        self.compressor = Compressor(
            config,
            self.layer_id,
            True,  # is_in_indexer
            rotary_emb,
            freqs_cis,
            compress_ratio=4,
            head_dim=self.head_dim,
            rotate=True,
            prefix=add_prefix("compressor", prefix),
        )
        self.rotary_emb = rotary_emb
        self.freqs_cis = freqs_cis
        self.weight_scale: float = self.softmax_scale * self.n_heads**-0.5
        self.alt_streams = alt_streams

    def compute_q(self, q_lora: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # [bs, n_heads, head_dim]
        #
        # NOTE: not batched with `compute_weights` below — the two GEMMs take
        # *different* inputs (q_lora, the lora-projected+normed Q vector, vs.
        # x, the un-normed hidden state). Batching via torch.bmm/aiter
        # batched_gemm would require restructuring the call chain in
        # `forward_c4_indexer` so that both projections share an input, which
        # changes the model semantics (q-norm cannot be applied to x). Skipped
        # — see fusion item 6c discussion in MQALAYER_ELEMENTWISE_ANALYSIS.md.
        q, _ = self.wq_b(q_lora)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        fused_rope(
            q[..., -self.rope_head_dim :],
            None,
            self.freqs_cis,
            positions=positions,
        )
        q = rotate_activation(q)
        return q

    def compute_weights(self, x: torch.Tensor, skip_scale=False) -> torch.Tensor:
        # See `compute_q` note: cannot batch this GEMM with `wq_b` because the
        # input tensors differ (q_lora vs x). The two are 1 launch each at the
        # measured shapes, total addressable saving is ~30-60us/token if
        # somehow batched, not worth a model-semantics-changing refactor.
        out, _ = self.weights_proj(x)
        if not skip_scale:
            out = out * self.weight_scale
        return out

    def forward(
        self,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: ForwardBatch,
        x_for_compressor: Optional[torch.Tensor] = None,
        enable_multi_stream: bool = False,
        q_lora_ready: Optional[torch.cuda.Event] = None,
    ) -> None:
        if TYPE_CHECKING:
            assert isinstance(forward_batch.attn_backend, DeepseekV4Backend)
        return forward_batch.attn_backend.forward_c4_indexer(
            x=x,
            q_lora=q_lora,
            forward_batch=forward_batch,
            c4_indexer=self,
            x_for_compressor=x_for_compressor if x_for_compressor is not None else x,
            alt_streams=self.alt_streams,
            enable_multi_stream=enable_multi_stream,
            q_lora_ready=q_lora_ready,
        )


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class MQALayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ) -> None:
        super().__init__()
        self.tp_rank = attn_tp_rank = get_attention_tp_rank()
        self.tp_size = attn_tp_size = get_attention_tp_size()
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_tp_size()
            self.tp_rank = attn_tp_rank = 0
            self.tp_size = attn_tp_size = 1
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.qk_rope_head_dim = config.qk_rope_head_dim
        if envs.SGLANG_DSV4_MODE.get() == "2604":
            self.qk_nope_head_dim = config.head_dim - config.qk_rope_head_dim
        else:
            self.qk_nope_head_dim = config.qk_nope_head_dim
        self.head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.n_heads = config.num_attention_heads
        self.n_local_heads = self.n_heads // attn_tp_size
        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // attn_tp_size
        self.rope_head_dim = config.qk_rope_head_dim
        self.softmax_scale = self.head_dim**-0.5
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.eps = config.rms_norm_eps
        compress_ratio = config.compress_ratios[layer_id]
        assert compress_ratio in [0, 4, 128]
        self.compress_ratio: Literal[0, 4, 128] = compress_ratio  # type: ignore

        if envs.SGLANG_DSV4_MODE.get() == "2604":
            assert self.head_dim == config.head_dim
        else:
            assert self.head_dim == config.v_head_dim
        assert config.num_key_value_heads == 1

        # need a indexer for compress ratio = 4
        rope_scaling = config.rope_scaling
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        # Please keep this assertion and not remove it
        # NOTE:
        # 1. 2601
        #    The `260119-updated` code changed compress_rope_theta
        # 2. 2604
        #    `official_code_0409/code/config.json` is 160000
        #    while `official_code_0409/config.json` is 40000
        #    maybe the latter is buggy? b/c dpsk's official generate.py uses `code/config.json`
        expected_compress_rope_theta = os.environ.get(
            "SGLANG_HACK_ASSERT_COMPRESS_ROPE_THETA"
        )
        if expected_compress_rope_theta is None:
            expected_compress_rope_theta = "160000"
        expected_compress_rope_theta = int(expected_compress_rope_theta)
        assert (
            config.compress_rope_theta == expected_compress_rope_theta
        ), f"{config.compress_rope_theta=} {expected_compress_rope_theta=}"
        # rope_theta may not be an attribute of the model_type-shimmed config
        # class used in newer transformers; fall back to the HF default.
        rope_base = (
            config.compress_rope_theta
            if self.compress_ratio
            else getattr(config, "rope_theta", 10000)
        )

        self.rotary_emb = get_rope_wrapper(
            head_size=self.rope_head_dim,
            rotary_dim=self.rope_head_dim,
            max_position=config.max_position_embeddings,
            base=rope_base,
            rope_scaling=rope_scaling,
            is_neox_style=False,
            device=get_global_server_args().device,
        )

        # naive impl: copy from reference code
        from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis

        if envs.SGLANG_DSV4_MODE.get() == "2604":
            assert rope_scaling["factor"] == 16
        elif envs.SGLANG_DSV4_MODE.get() == "2601":
            assert rope_scaling["factor"] == 4
        else:
            raise NotImplementedError

        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            assert self.compress_ratio in {0, 4, 128}
            if self.compress_ratio:
                original_seq_len = rope_scaling["original_max_position_embeddings"]
                assert original_seq_len == 65536
            else:
                original_seq_len = 0
        else:
            original_seq_len = rope_scaling["original_max_position_embeddings"]

        rope_scaling = config.rope_scaling
        freqs_cis = precompute_freqs_cis(
            dim=self.qk_rope_head_dim,
            seqlen=config.max_position_embeddings,
            original_seq_len=original_seq_len,
            base=rope_base,
            factor=rope_scaling["factor"],
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        self.freqs_cis: torch.Tensor

        if envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get() and alt_streams is not None:
            self.alt_streams = alt_streams[:3]  # use first 3 streams for mqa layer
            self.alt_streams_indexer = alt_streams[
                -2:
            ]  # use last 2 streams for indexer
        else:
            self.alt_streams = None
            self.alt_streams_indexer = None

        from sglang.srt.utils import is_blackwell_supported

        self._multi_stream_bs_limit = 128 if is_blackwell_supported() else 64

        self.compressor = None
        self.indexer = None
        if self.compress_ratio:
            self.compressor = Compressor(
                config,
                layer_id=self.layer_id,
                is_in_indexer=False,
                rotary_emb=self.rotary_emb,
                freqs_cis=freqs_cis,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                rotate=False,
                prefix=add_prefix("compressor", prefix),
            )
            if self.compress_ratio == 4:
                self.indexer = C4Indexer(
                    config,
                    rotary_emb=self.rotary_emb,
                    freqs_cis=freqs_cis,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix("indexer", prefix),
                    alt_streams=self.alt_streams_indexer,
                )

        # Note: attention sink should be replicated
        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.wq_a = ReplicatedLinear(
            self.hidden_size,
            self.q_lora_rank,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_a", prefix),
        )
        self.q_norm = RMSNorm(self.q_lora_rank, eps=self.eps)
        self.wq_b = ColumnParallelLinear(
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wq_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.wkv = ReplicatedLinear(
            self.hidden_size,
            self.head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("wkv", prefix),
        )
        self.kv_norm = RMSNorm(self.head_dim, eps=self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config if _FP8_WO_A_GEMM else None,
            prefix=add_prefix("wo_a", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            **({} if _FP8_WO_A_GEMM else {"params_dtype": torch.bfloat16}),
        )
        if _FP8_WO_A_GEMM:
            # fp8_einsum handles scale transform internally — skip UE8M0 conversion
            assert hasattr(
                self.wo_a, "weight_scale_inv"
            ), "FP8 quant_config must create weight_scale_inv"
            self.wo_a.weight_scale_inv.format_ue8m0 = True
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=attn_tp_size > 1,
            prefix=add_prefix("wo_b", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )

        self.attn_mqa = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )

        self.overlap_store_cache = envs.SGLANG_OPT_USE_OVERLAP_STORE_CACHE.get()

    def _compute_q_a(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # Phase 24: caller may stash a (fp8, scale) tuple for wq_a's input.
        prequant_x = getattr(self, "_prequant_x", None)
        wq_a_in = prequant_x if prequant_x is not None else x
        # [bs, q_lora_rank]
        q, _ = self.wq_a(wq_a_in)
        # [bs, q_lora_rank]
        q = self.q_norm(q)
        q_lora = q  # only used for indexer
        return q_lora

    def _compute_q_b(
        self,
        q: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # [bs, n_local_heads, head_dim]
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        q = rms_normalize_triton(q, self.eps)

        if positions is not None:
            fused_rope(
                q[..., -self.qk_rope_head_dim :],
                None,
                self.freqs_cis,
                positions=positions,
            )
        else:
            apply_rotary_emb_triton(q[..., -self.qk_rope_head_dim :], self.freqs_cis)
        return q

    def _compute_kv(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Phase 24: caller may stash a (fp8, scale) tuple for wkv's input.
        prequant_x = getattr(self, "_prequant_x", None)
        wkv_in = prequant_x if prequant_x is not None else x
        # [bs, head_dim]
        kv, _ = self.wkv(wkv_in)
        # [bs, head_dim]
        kv = self.kv_norm(kv)
        if positions is not None:
            fused_rope(
                kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
                None,
                self.freqs_cis,
                positions=positions,
            )
        else:
            apply_rotary_emb_triton(kv[..., -self.qk_rope_head_dim :], self.freqs_cis)
        return kv

    def _forward_prepare_multi_stream(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: DeepseekV4Backend,
        freqs_cis: Optional[torch.Tensor] = None,
        q_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.alt_streams is not None
        assert len(self.alt_streams) >= 3

        current_stream = torch.cuda.current_stream()
        stream_kv = self.alt_streams[0]
        stream_compressor = self.alt_streams[1]
        stream_indexer = self.alt_streams[2]

        stream_kv.wait_stream(current_stream)
        stream_compressor.wait_stream(current_stream)
        stream_indexer.wait_stream(current_stream)

        # Phase 24: snapshot prequant tuple for use across alt-streams.
        # _compute_q_a / _compute_kv read self._prequant_x; snapshot ensures
        # both see the same value even though they run on different streams.
        # Cleared after both helpers consume it (below).
        # main stream: compute q
        q_lora = self._compute_q_a(x)
        q_lora_ready = current_stream.record_event()
        q = self._compute_q_b(q_lora, positions, freqs_cis)
        if q_out is not None:
            q_out.copy_(q)

        # alt stream 2: compute indexer
        if self.indexer is not None:
            with torch.cuda.stream(stream_indexer):
                self.indexer(
                    x=x,
                    q_lora=q_lora,
                    forward_batch=forward_batch,
                    enable_multi_stream=True,
                    q_lora_ready=q_lora_ready,
                )

        # alt stream 0: compute kv
        with torch.cuda.stream(stream_kv):
            kv = self._compute_kv(x, positions, freqs_cis)
            if self.overlap_store_cache:
                attn_backend.store_cache(
                    layer_id=self.layer_id,
                    swa_k=kv,
                    forward_batch=forward_batch,
                )
        # Phase 24: clear prequant slot after both q-a and kv have been
        # dispatched (they ran on different streams; both already grabbed
        # references to the underlying tensors via wq_a/wkv input).
        self._prequant_x = None

        # alt stream 1: compute compressor
        if self.compressor is not None:
            with torch.cuda.stream(stream_compressor):
                attn_backend.forward_core_compressor(
                    x, forward_batch, self.layer_id, self.compressor
                )

        current_stream.wait_stream(stream_kv)
        current_stream.wait_stream(stream_compressor)
        current_stream.wait_stream(stream_indexer)

        return q, kv

    def _forward_prepare(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: DeepseekV4Backend,
        freqs_cis: Optional[torch.Tensor] = None,
        q_out: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Phase 24 (A2-#1 dual-output): if the parent layer pre-fused
        # input_layernorm into (fp8, scale), feed those directly into wq_a /
        # wkv. Bypasses the per-1x128 quant launches that would otherwise
        # fire inside each FP8 GEMM apply. `x` (bf16) is still used downstream
        # by indexer/compressor — it is the bf16 normed-out from the fused
        # kernel.
        prequant_x = getattr(self, "_prequant_x", None)
        self._prequant_x = None  # clear before consumption (no stale state)
        # DBM Block A consumer hook (Phase 2 stub): if the layer-level pre-pass
        # produced megakernel outputs, skip wq_a/wkv. q is the pre-q_norm bf16
        # tensor (matches what the unfused path produces post-wq_a). The next
        # branch's `q_norm` / `kv_norm` step still runs.
        dbm_outs = getattr(self, "_dbm_block_a_outputs", None)
        self._dbm_block_a_outputs = None
        if dbm_outs is not None:
            q, kv = dbm_outs
        else:
            wq_a_in = prequant_x if prequant_x is not None else x
            wkv_in = prequant_x if prequant_x is not None else x
            # [bs, q_lora_rank]
            q, _ = self.wq_a(wq_a_in)
            # [bs, head_dim]
            kv, _ = self.wkv(wkv_in)
        # A2-#1 path: env-gated. Fuses q_norm + per-1x128 fp8 quant into one
        # Triton launch, then feeds wq_b's FP8 GEMM with pre-quantized input.
        # This drops the qk_rmsnorm fusion (kv_norm goes through its own
        # launch), but eliminates the dynamic_per_group_scaled_quant launch
        # that otherwise fires inside wq_b. Net win on graph replay because
        # the per-1x128 quant launch dominates kv_norm's cost.
        # NOTE: q_lora (the indexer input) needs to be the bf16 post-q_norm
        # output. We get it from the fused kernel's residual-out slot... but
        # this path has no residual, so we materialise a separate bf16 q_lora.
        # For now, when the indexer is disabled OR not in this path, just
        # leave q_lora=None (not used). Future stacking can fold q_lora out.
        used_fused_q_quant = False
        if (
            _AITER_QK_RMSNORM_GROUP_QUANT
            and q.dtype == torch.bfloat16
            and kv.dtype == torch.bfloat16
            and q.shape[-1] % 128 == 0
            # `fused_qk_rmsnorm_group_quant` runs the kv path with `k_out=bf16`,
            # so kv stays bf16 for downstream rope+attn. Output q is fp8 +
            # per-128 scales (consumed directly by wq_b's tuple-accept path).
        ):
            n = q.size(0)
            q_dim = q.size(-1)
            kv_dim = kv.size(-1)
            # aiter's fused_qk_rmsnorm_group_quant requires fp8/fp4x2;
            # use float8_e4m3fn (matches `_fused_rmsnorm_per1x128_quant` default).
            q_fp8 = torch.empty(
                (n, q_dim), dtype=torch.float8_e4m3fn, device=q.device
            )
            q_scale = torch.empty(
                (n, q_dim // 128), dtype=torch.float32, device=q.device
            )
            kv_out = torch.empty_like(kv)
            need_q_lora_bf16 = getattr(self, "indexer", None) is not None
            q_unq = torch.empty_like(q) if need_q_lora_bf16 else None
            _aiter_fused_qk_rmsnorm_group_quant(
                q_fp8, q_scale,
                q, self.q_norm.weight, self.eps,
                q_out_unquantized=q_unq,
                k_out=kv_out,
                k=kv, k_weight=self.kv_norm.weight, k_epsilon=self.eps,
                group_size=128,
            )
            kv = kv_out
            if need_q_lora_bf16:
                q_lora = q_unq
            else:
                q_lora = None
            q, _ = self.wq_b((q_fp8, q_scale))
            used_fused_q_quant = True
        elif (
            _F4_MODE_A
            and q.dtype == torch.bfloat16
            and kv.dtype == torch.bfloat16
            and q.shape[-1] % 128 == 0
        ):
            # F4 Mode A: same shape as _FUSED_RMSNORM_QUANT_PER1x128 branch but
            # uses the F4-patched kernel (MATCH_BF16_PRODUCTION=True). The patch
            # round-trips `normed` through bf16 in registers before fp8 quant so
            # fp8 codepoints match the unfused two-launch production path
            # (within fp32 mul-order ULP). v2 microbench: 5.86x worst-mode
            # speedup on 5 production shapes; profile: 3.17 us/launch vs 6.68 us
            # unfused sum (2.11x). Drops qk_rmsnorm fusion same as the older knob.
            kv = self.kv_norm(kv)
            if getattr(self, "indexer", None) is not None:
                # Indexer needs bf16 post-q_norm q_lora; fall back this layer.
                q = self.q_norm(q)
                q_lora = q
                used_fused_q_quant = False
            else:
                q_fp8, q_scale = _fused_rmsnorm_per1x128_quant(
                    q, self.q_norm.weight, self.eps,
                    match_bf16_production=True,
                )
                q_lora = None  # only used for indexer; unused on this branch
                q, _ = self.wq_b((q_fp8, q_scale))
                used_fused_q_quant = True
        elif (
            _FUSED_RMSNORM_QUANT_PER1x128
            and q.dtype == torch.bfloat16
            and kv.dtype == torch.bfloat16
            and q.shape[-1] % 128 == 0
        ):
            # kv_norm runs separately (no fusion lever for kv_b on this path).
            kv = self.kv_norm(kv)
            # Materialise bf16 q_lora for indexer (only if compress_ratio == 4).
            # We need bf16 post-q_norm. Cheapest: do plain rmsnorm in fp32 then
            # store, while ALSO emitting the fp8 prequant. But to keep the
            # fusion benefit (single launch over q), we accept one extra
            # rmsnorm-of-q for the indexer when present.
            if getattr(self, "indexer", None) is not None:
                # Indexer present — needs bf16 post-q_norm q_lora. Fall back
                # to non-fused for safety on this layer.
                q = self.q_norm(q)
                q_lora = q
                used_fused_q_quant = False
            else:
                q_fp8, q_scale = _fused_rmsnorm_per1x128_quant(
                    q, self.q_norm.weight, self.eps,
                )
                q_lora = None  # only used for indexer; unused on this branch
                q, _ = self.wq_b((q_fp8, q_scale))
                used_fused_q_quant = True
        elif (
            _aiter_fused_qk_rmsnorm is not None
            and q.dtype == torch.bfloat16
            and kv.dtype == torch.bfloat16
            and q.size(0) == kv.size(0)
        ):
            # Fuse q_norm + kv_norm into a single launch when both are bf16
            # and share the leading dim (always true here since they read
            # the same x). Saves ~1 launch / layer × 61 = ~250-300 µs / token.
            # The aiter op falls back internally to two rmsnorm2d_fwd calls
            # at M >= 16384.
            q, kv = _aiter_fused_qk_rmsnorm(
                q, self.q_norm.weight, self.eps,
                kv, self.kv_norm.weight, self.eps,
            )
            q_lora = q  # only used for indexer (post-q_norm, pre-wq_b — unchanged)
        else:
            q = self.q_norm(q)
            kv = self.kv_norm(kv)
            q_lora = q  # only used for indexer (post-q_norm, pre-wq_b — unchanged)

        if not used_fused_q_quant:
            # [bs, n_local_heads, head_dim]
            q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        # [bs, n_local_heads, head_dim]
        # 2026-05-03 Phase A1: when q_out is set (TP>1), defer RoPE until just
        # before the q_out write — fuse rmsnorm+RoPE+strided-write into one kernel
        # via fused_rmsnorm_rope_q_triton_to_out. Eliminates the trailing
        # `q_out.copy_(q)` launch (~22 bf16_copy events/forward).
        # Gated by SGLANG_FUSED_ROPE_Q_TO_OUT=1 (default OFF until E2E validates).
        # When q_out is None (TP=1) or knob OFF, keep the existing in-place RoPE.
        _rope_to_out_active = (
            q_out is not None
            and os.environ.get("SGLANG_FUSED_ROPE_Q_TO_OUT", "1") == "1"
        )
        if not _rope_to_out_active:
            # Fused unweighted rmsnorm + RoPE on Q in one launch (saves ~25us/call,
            # microbench 1.42x vs unfused at decode bs=8). KV rope still runs separately.
            fused_rmsnorm_rope_q_triton(
                q, self.freqs_cis, positions, self.eps, self.qk_rope_head_dim,
            )
        # M1 megakernel: fuse apply_rotary_emb_triton + quant_pack + paged scatter
        # into one Triton launch. When ON, kv-cache write happens here so the
        # downstream attn_backend.forward(save_kv_cache=...) must be False.
        # Falls back to the unfused chain when the kernel import failed or the
        # knob is OFF (default).
        m1_active = (
            envs.SGLANG_M1_KV_WRITE_WITH_ROPE.get()
            and _m1_kv_write_with_rope_triton is not None
            and kv.dtype == torch.bfloat16
            and kv.shape[-1] == 512
            and self.qk_rope_head_dim == 64
        )
        if m1_active:
            tkp = forward_batch.token_to_kv_pool
            swa_loc = tkp.translate_loc_from_full_to_swa(forward_batch.out_cache_loc)
            buf = tkp.swa_kv_pool.kv_buffer[self.layer_id]
            page_size = tkp.swa_kv_pool.page_size
            _m1_kv_write_with_rope_triton(
                kv, self.freqs_cis, positions, swa_loc, buf, page_size,
            )
            self._m1_kv_written = True
        else:
            apply_rotary_emb_triton(
                kv[..., -self.qk_rope_head_dim :].unsqueeze(1),
                self.freqs_cis,
                positions=positions,
            )

        _use_cp = self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch)
        if _use_cp:
            kv = cp_all_gather_rerange_output(
                kv.contiguous(),
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )
            x_for_compressor = (
                cp_all_gather_rerange_output(
                    x.contiguous(),
                    self.cp_size,
                    forward_batch,
                    torch.cuda.current_stream(),
                )
                if self.compressor is not None
                else x
            )
        else:
            x_for_compressor = x

        if self.overlap_store_cache:
            attn_backend.store_cache(
                layer_id=self.layer_id,
                swa_k=kv,
                forward_batch=forward_batch,
            )

        if self.indexer is not None:
            self.indexer(
                x=x,
                q_lora=q_lora,
                forward_batch=forward_batch,
                x_for_compressor=x_for_compressor if _use_cp else None,
            )
        if self.compressor is not None:
            attn_backend.forward_core_compressor(
                x_for_compressor, forward_batch, self.layer_id, self.compressor
            )

        if q_out is not None:
            if _rope_to_out_active:
                # Phase A1: fuse rmsnorm+RoPE+strided-write into one kernel.
                # Replaces (in-place rope at line ~2410) + (q_out.copy_) two-launch chain.
                fused_rmsnorm_rope_q_triton_to_out(
                    q, q_out, self.freqs_cis, positions, self.eps, self.qk_rope_head_dim,
                )
            else:
                q_out.copy_(q)
        return q, kv

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        debug_return_kv: bool = False,
    ) -> torch.Tensor:
        if not get_attn_tp_context().input_scattered and x.shape[0] == 0:
            assert (
                not self.wo_b.reduce_results
            ), "short-circuiting allreduce will lead to hangs"
            return x

        attn_backend = forward_batch.attn_backend
        if TYPE_CHECKING:
            assert isinstance(attn_backend, DeepseekV4Backend)

        freqs_cis = None
        # M1 megakernel writes kv-cache inline. Reset per-call; _forward_prepare
        # sets True when the megakernel ran (single-stream path only).
        self._m1_kv_written = False

        enable_multi_stream = (
            envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
            and self.alt_streams is not None
            and get_is_capture_mode()
            and x.shape[0] <= self._multi_stream_bs_limit
            and not (self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch))
        )

        tp_slice, q_padded, q_out = slice(None), None, None
        if self.tp_size > 1:
            # pad the q to [batch_size, n_heads]
            q_padded = x.new_empty(x.shape[0], self.n_heads, self.head_dim)
            rank = self.tp_rank
            tp_slice = slice(rank * self.n_local_heads, (rank + 1) * self.n_local_heads)
            q_out = q_padded[:, tp_slice, :]

        if enable_multi_stream:
            q, kv = self._forward_prepare_multi_stream(
                x, positions, forward_batch, attn_backend, freqs_cis, q_out
            )
        else:
            q, kv = self._forward_prepare(
                x, positions, forward_batch, attn_backend, freqs_cis, q_out
            )

        # for TP attention, use the padded q, since q_out is set to the correct slice
        # When M1 ran inside _forward_prepare, kv-cache is already written.
        o = attn_backend.forward(
            q=q_padded if q_padded is not None else q,
            k=kv,
            v=kv,
            layer=self.attn_mqa,
            forward_batch=forward_batch,
            compress_ratio=self.compress_ratio,
            attn_sink=self.attn_sink,
            save_kv_cache=(not self.overlap_store_cache) and (not self._m1_kv_written),
        )
        # NOTE: no-op for pure DP-attention
        o = o[:, tp_slice, :]
        fused_rope(
            o[..., -self.qk_rope_head_dim :],
            None,
            self.freqs_cis,
            positions=positions,
            inverse=True,
        )

        o = o.view(o.shape[0], self.n_local_groups, -1)

        if _FP8_WO_A_GEMM:
            import deep_gemm

            T, G, D = o.shape
            R = self.o_lora_rank
            # `.reshape().contiguous()` was always redundant: reshape returns a
            # view if compatible (already contiguous), or a copy otherwise (also
            # contiguous). The trailing .contiguous() costs Python dispatch.
            o_fp8, o_s = sglang_per_token_group_quant_fp8(
                o.reshape(T * G, D),
                group_size=128,
            )
            output = torch.empty(T, G, R, device=o.device, dtype=torch.bfloat16)
            deep_gemm.fp8_einsum(
                "bhr,hdr->bhd",
                (o_fp8.view(T, G, D), o_s.view(T, G, -1)),
                (self.wo_a.weight.view(G, R, D), self.wo_a.weight_scale_inv.data),
                output,
                recipe=(1, 1, 128),
            )
            o = output
        else:
            wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
            o = torch.einsum("tgd,grd->tgr", o, wo_a)

        # NOTE on fusion item 7 (wo_b AllReduce + next-layer RMSNorm fusion):
        #
        # `aiter.dist.fused_allreduce_rmsnorm{,_quant}` exists and would fuse
        # AllReduce + add(residual) + RMSNorm into one launch — and
        # RowParallelLinear already supports `skip_all_reduce=True`. But
        # DSv4's decoder layer is structurally different from the standard
        # transformer this op was designed for:
        #
        #   wo_b output [n, d]  (post-AllReduce in stock SGLang)
        #     -> hc_post([n, d], residual=[n, hc, d], post, comb)  -> [n, hc, d]
        #     -> hc_pre(...)  -> [n, d] + post' + comb'
        #     -> post_attention_layernorm([n, d])     <- the next RMSNorm
        #
        # Two reasons fusion is structurally infeasible here:
        #   1. The "residual" between wo_b's AllReduce and the next RMSNorm is
        #      *not* an additive [n, d] residual — it's a [n, hc, d]
        #      hierarchical-compression tensor consumed by `hc_post`, which is
        #      itself a TileLang-fused op (mhc_post). The aiter fused op
        #      assumes `out = rmsnorm(input + residual)`; here we have
        #      `out = rmsnorm(hc_pre(hc_post(input, residual_hc, post, comb)))`
        #      which is non-linear in `input`.
        #   2. The next RMSNorm weight (`post_attention_layernorm.weight`)
        #      lives in a different module than this MQALayer; threading it
        #      across the layer boundary would require either passing it
        #      through `hc_post` + `hc_pre` (defeats the point) or restructur-
        #      ing DeepseekV4DecoderLayer.forward to inline the AllReduce-
        #      reduce -+-rmsnorm step (defeats the layer abstraction).
        #
        # Skipped per the task spec ("If cross-layer wiring is too invasive,
        # implement just the fused AllReduce + RMSNorm... If that's also too
        # messy, SKIP with a thorough comment").

        # Decode-body megakernel — Block B-pre (wo_b prologue) wire-in stub.
        # Replaces `per-1x128 quant + wo_b fp8 GEMM` (2 launches) with one
        # Triton kernel. AllReduce stays out-of-kernel (cross-rank).
        # Default OFF; gated by SGLANG_DECODE_BODY_BLOCK_B_PRE=1. Microbench
        # shows 5.41x graph-replay speedup on the 5 production shapes; E2E
        # NOT yet measured. Activation is a separate commit.
        o_in = o.flatten(1)
        if (
            _DECODE_BODY_BLOCK_B_PRE
            and _decode_body_block_b_pre_megakernel is not None
            and o_in.dtype == torch.bfloat16
            and o_in.shape[-1] % 128 == 0
            and getattr(self.wo_b, "weight_scale_inv", None) is not None
            and self.wo_b.weight.dtype == torch.float8_e4m3fn
        ):
            mk_out = _decode_body_block_b_pre_megakernel(
                o_in,
                self.wo_b.weight,
                self.wo_b.weight_scale_inv,
            )
            if mk_out is not None:
                # Megakernel produced the pre-AR (T, hidden) bf16 output. Run
                # AllReduce manually if reduce_results is on.
                if (
                    self.wo_b.reduce_results
                    and self.wo_b.tp_size > 1
                ):
                    from sglang.srt.distributed import (
                        tensor_model_parallel_all_reduce,
                    )
                    o = tensor_model_parallel_all_reduce(mk_out)
                else:
                    o = mk_out
            else:
                o, _ = self.wo_b(o_in)
        else:
            o, _ = self.wo_b(o_in)

        return o


class DeepseekV4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id
        self.is_nextn = is_nextn
        self.self_attn = MQALayer(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
            alt_streams=alt_streams,
        )
        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )
        # TODO: check whether the implementation matches
        # TODO: make necessary changes if possible
        self.mlp = deepseek_v2.DeepseekV2MoE(
            config=config,
            quant_config=moe_quant_config_override or quant_config,
            prefix=add_prefix("mlp", prefix),
            layer_id=self.layer_id,
            alt_stream=alt_streams[0] if alt_streams is not None else None,
            is_nextn=is_nextn,
            is_deepseek_v4=True,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        # self.layer_communicator = LayerCommunicator(
        #     layer_scatter_modes=self.layer_scatter_modes,
        #     input_layernorm=self.input_layernorm,
        #     post_attention_layernorm=self.post_attention_layernorm,
        #     allow_reduce_scatter=True,
        #     is_last_layer=(
        #         is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
        #     ),
        # )

        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.rms_norm_eps = config.rms_norm_eps
        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()

    def _is_layer_sparse(self, layer_id: int, is_nextn: bool) -> bool:
        if envs.SGLANG_DSV4_MODE.get() == "2604":
            first_k_dense_replace = 0
            moe_layer_freq = 1
        else:
            first_k_dense_replace = self.config.first_k_dense_replace
            moe_layer_freq = self.config.moe_layer_freq
        return is_nextn or (
            self.config.n_routed_experts is not None
            and layer_id >= first_k_dense_replace
            and layer_id % moe_layer_freq == 0
        )

    def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        # x: [n,hc,d] -> y: [n,d], where n=b*s
        shape, dtype = x.size(), x.dtype

        # Handle empty batch
        if x.shape[0] == 0:
            y = torch.empty((0, shape[-1]), dtype=dtype, device=x.device)
            post = torch.empty((0, self.hc_mult), dtype=dtype, device=x.device)
            comb = torch.empty(
                (0, self.hc_mult, self.hc_mult), dtype=dtype, device=x.device
            )
            return y, post, comb

        if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            from sglang.srt.layers.mhc import mhc_pre

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=2.0,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
            )
            # returned post should be [n, hc_mult]
            return y, post.squeeze(-1), comb

        if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            # DeepGEMM implementation
            import deep_gemm

            x_flat = x.flatten(1).bfloat16()

            m, k = x_flat.shape
            mix_hc = hc_fn.size(0)
            d_out = torch.empty((m, mix_hc), dtype=torch.float, device=x.device)
            s_out = torch.empty((m,), dtype=torch.float, device=x.device)
            # `hc_fn` is an nn.Parameter declared as `dtype=torch.float32`
            # (see L2003-L2004), so `.float()` is a no-op and `.contiguous()`
            # is redundant for a freshly-loaded parameter. Drop both — saves
            # ~10-30 us Python dispatch per layer per step.
            deep_gemm.tf32_hc_prenorm_gemm(
                x_flat, hc_fn, d_out, s_out, num_splits=None
            )
            rsqrt = torch.rsqrt(s_out / k + self.rms_norm_eps)
            mixes = (d_out * rsqrt.unsqueeze(1)).unsqueeze(1)
        else:
            # Triton fused kernel for the Pro shape (HC_MULT × HC_DIM = HIDDEN).
            # Microbench at M=8192: 316 → 112 us (2.83x), cos_sim 1.000000.
            # Falls back to torch impl for unmatched shapes.
            if (x.dtype == torch.bfloat16
                and hc_fn.dtype == torch.float32
                and x.shape[1] * x.shape[2] == hc_fn.shape[1]
                and hc_fn.shape[0] == x.shape[1]):
                x_flat, mixes = hc_pre_fused_triton(x, hc_fn, self.rms_norm_eps)
            elif (
                # 2026-05-01: decode-shape fallback. Replaces _hc_pre_torch_impl
                # with a decode-tuned RMSNorm kernel + torch GEMM + broadcast mul.
                # Per the post-fused-pool agent attribution, _hc_pre_torch_impl
                # owns ~5 ms cpu_dispatch / 602 calls in the eager profile (top
                # remaining after compress_decode_old fusion shipped).
                # Default OFF; gated by SGLANG_HC_PRE_DECODE_TRITON=1.
                # NOTE: the v1 attempt extending the prefill kernel REGRESSED +60 ms
                # at decode (kernel optimized for M=8192 prefill, slow at M=1-8).
                # This is a fundamentally different kernel (grid over M, not
                # 1-program-with-loop); see hc_pre_decode_triton.py.
                os.environ.get("SGLANG_HC_PRE_DECODE_TRITON", "0") == "1"
                and x.dtype == torch.bfloat16
                and hc_fn.dtype == torch.float32
                and x.shape[1] * x.shape[2] == hc_fn.shape[1]
                and x.shape[0] <= 32  # decode regime only
            ):
                from sglang.jit_kernel.hc_pre_decode_triton import hc_pre_decode_triton
                x_flat, mixes = hc_pre_decode_triton(x, hc_fn, self.rms_norm_eps)
            else:
                # Naive Torch implementation
                x_flat, mixes = _hc_pre_torch_impl(x, hc_fn, self.rms_norm_eps)

        from sglang.srt.layers.mhc import hc_split_sinkhorn

        B = mixes.shape[0]
        pre = mixes.new_empty((B, 1, self.hc_mult), dtype=torch.float32)
        post = mixes.new_empty((B, 1, self.hc_mult), dtype=torch.float32)
        comb = mixes.new_empty((B, 1, self.hc_mult, self.hc_mult), dtype=torch.float32)
        hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            pre,
            post,
            comb,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            eps=self.hc_eps,
        )
        # A2-#2: fuse (pre * x_flat).sum(1).to(dtype) into one kernel.
        # Default OFF (preserve Phase 13 behavior). Only valid when fused
        # hc_pre path emitted x_flat as fp32 [B, HC_MULT*HIDDEN] AND the
        # caller wants bf16 output (the only path on Flash-Base FP8).
        if (os.environ.get("SGLANG_FUSED_MHC_POST", "0") == "1"
            and dtype == torch.bfloat16
            and x_flat.dtype == torch.float32
            and pre.dtype == torch.float32
            and x_flat.dim() == 2
            and x_flat.shape == (shape[0], shape[1] * shape[2])):
            y_bf16 = mhc_post_fused_mul_sum_cast(pre, x_flat, shape, torch.bfloat16)
            return y_bf16, post.squeeze(1), comb.squeeze(1)
        y = (pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1)
        return y.to(dtype), post.squeeze(1), comb.squeeze(1)

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):

        # x: [n,d], residual: [n,hc,d] -> y: [n,hc,d]
        # post: [n,hc], comb: [n,hc,hc]

        # Handle empty batch
        if x.shape[0] == 0:
            return torch.empty(
                (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
            )

        if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get():
            from sglang.srt.layers.mhc import mhc_post

            result = mhc_post(x, residual, post, comb)
            return result

        assert residual.shape == (x.shape[0], self.hc_mult, x.shape[-1])
        assert post.shape == (x.shape[0], self.hc_mult)
        assert comb.shape == (x.shape[0], self.hc_mult, self.hc_mult)

        # Triton fused kernel: 23x over eager at prefill M=8192 (microbench).
        # Falls back to torch impl for unsupported dtype/shape combos.
        if (x.dtype == torch.bfloat16
            and residual.dtype == torch.bfloat16
            and post.dtype == torch.float32
            and comb.dtype == torch.float32):
            result = hc_post_fused_triton(x, residual, post, comb)
        else:
            result = _hc_post_torch_impl(x, residual, post, comb)
        return result

    def forward(
        self,
        positions: torch.tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: torch.Tensor,
    ) -> torch.Tensor:
        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            assert deepseek_v4_moe_code_path_checker.observed == 0

        # SGLANG_CAPTURE_TRACE=1 — per-phase stderr prints inside the captured
        # forward; last line before scheduler-dead points at the crash site.
        _trace = os.environ.get("SGLANG_CAPTURE_TRACE", "0") == "1"
        if _trace:
            import sys
            sys.stderr.write(f"[trace] L{self.layer_id} enter\n"); sys.stderr.flush()

        residual = hidden_states
        hidden_states, post, comb = self.hc_pre(
            hidden_states, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base
        )  # -> [n, d]
        # Phase 24 (A2-#1 dual-output): when the env knob is on, fuse
        # input_layernorm + per-1x128 fp8 quant into a single launch and
        # stash the (fp8, scale) tuple on self.self_attn for wq_a/wkv to
        # consume directly. The bf16 normed-out is reused as `hidden_states`
        # for indexer/compressor (non-fp8 consumers). One launch eliminates
        # the two redundant per-1x128 quants that would otherwise fire inside
        # wq_a and wkv's FP8 GEMM apply paths.
        # Decode-body megakernel — Block A (MQA prologue) wire-in stub. Phase 2.
        # Replaces input_layernorm + per-1x128 quant + wq_a + wkv with ONE kernel.
        # Stashes pre-q_norm bf16 q_lora and pre-kv_norm bf16 kv on self_attn so
        # _forward_prepare picks them up and skips its own wq_a/wkv calls.
        # Default OFF; only fires when SGLANG_DECODE_BODY_BLOCK_A=1 and the
        # indexer-bf16 path is active. E2E NOT yet enabled — baseline is broken.
        if (
            _DECODE_BODY_BLOCK_A
            and _decode_body_block_a_megakernel is not None
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.shape[-1] % 128 == 0
            and getattr(self.self_attn, "indexer", None) is not None
            and getattr(self.self_attn.wq_a, "weight_scale_inv", None) is not None
            and getattr(self.self_attn.wkv, "weight_scale_inv", None) is not None
        ):
            attn = self.self_attn
            normed_in = hidden_states
            mk_out = _decode_body_block_a_megakernel(
                normed_in,
                self.input_layernorm.weight.to(torch.float32)
                if self.input_layernorm.weight.dtype != torch.float32
                else self.input_layernorm.weight,
                attn.wq_a.weight,
                attn.wq_a.weight_scale_inv,
                attn.wkv.weight,
                attn.wkv.weight_scale_inv,
                eps=self.input_layernorm.variance_epsilon,
            )
            if mk_out is not None:
                q_lora_pre_qnorm, kv_pre_kvnorm = mk_out
                # Run the bf16 normed_in through input_layernorm anyway for the
                # downstream consumers (residual, hc_post). The megakernel only
                # short-circuits the wq_a/wkv branch.
                hidden_states = self.input_layernorm(hidden_states)
                # Stash pre-q_norm/pre-kv_norm outputs for _forward_prepare to use.
                attn._dbm_block_a_outputs = (q_lora_pre_qnorm, kv_pre_kvnorm)
                attn._prequant_x = None
            else:
                hidden_states = self.input_layernorm(hidden_states)
                attn._prequant_x = None
                attn._dbm_block_a_outputs = None
        elif (
            _FUSED_RMSNORM_QUANT_PER1x128_DUAL
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.shape[-1] % 128 == 0
        ):
            x_fp8, x_scale, hidden_states = _fused_rmsnorm_per1x128_quant_dual(
                hidden_states,
                self.input_layernorm.weight,
                self.input_layernorm.variance_epsilon,
            )
            # Stash for self_attn._forward_prepare{,_multi_stream} to consume.
            # The attribute is read+cleared inside _forward_prepare; setting it
            # here scopes its lifetime to this single self_attn call. (No
            # data_ptr-keyed cache; just a per-step transient slot.)
            self.self_attn._prequant_x = (x_fp8, x_scale)
            self.self_attn._dbm_block_a_outputs = None
        else:
            hidden_states = self.input_layernorm(hidden_states)
            self.self_attn._prequant_x = None
            self.self_attn._dbm_block_a_outputs = None

        if _trace:
            sys.stderr.write(f"[trace] L{self.layer_id} pre-attn\n"); sys.stderr.flush()

        hidden_states = self.self_attn(
            x=hidden_states,
            positions=positions,
            forward_batch=forward_batch,
        )

        if _trace:
            sys.stderr.write(f"[trace] L{self.layer_id} post-attn\n"); sys.stderr.flush()

        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        residual = hidden_states  # [n, hc, d]
        hidden_states, post, comb = self.hc_pre(
            hidden_states, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
        )  # -> [n, d]
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Communication logic (equivalent to LayerCommunicator):
        #
        # ======================== i. TP MoE ========================
        # DP attn + TP moe (moe_a2a_backend=none):
        # * mlp_mode = FULL (each-rank-has-whole-world-tokens)
        # * prepare_mlp -> _gather_hidden_states_and_residual -> dp_gather_partial
        # * postprocess_layer -> _scatter_hidden_states -> dp_scatter
        # Need Gather before MoE and Scatter after MoE.
        #
        # ======================== ii. DeepEP MoE ========================
        # DP attn + DeepEP moe (moe_a2a_backend=deepep/flashinfer/etc):
        # * mlp_mode = SCATTERED (each-rank-only-has-this-rank-tokens)
        # * prepare_mlp -> _simple (just layernorm, no gather)
        # * postprocess_layer -> _trivial (no scatter)
        # Because attn_tp_size==1 when tp==dp==ep, SCATTERED and TP_ATTN_FULL
        # have the same group_size. Token dispatch/combine is handled by
        # DeepEP inside MoE forward. No Gather/Scatter around MoE.
        _use_cp = self.nsa_enable_prefill_cp and nsa_use_prefill_cp(forward_batch)
        _use_tp_moe_gather = (
            not _use_cp
            and get_attention_dp_size() > 1
            and get_moe_a2a_backend().is_none()
        )
        # ----------------------------------- CP: fix input_ids to LOCAL ----------------
        if _use_cp:
            # CP requires DeepEP — TP MoE's all-reduce assumes identical tokens
            # across ranks, which CP violates. (Analogous to NSACPLayerCommunicator's
            # assert mlp_mode==SCATTERED when dp_size>1.)
            assert get_moe_a2a_backend().is_deepep(), (
                "CP requires DeepEP (moe_a2a_backend == deepep). "
                "Only DeepEP is tested with CP's per-rank token split."
            )
            # DeepEP handles cross-rank MoE dispatch/combine internally.
            # No gather/scatter needed — tokens stay LOCAL (SCATTERED).
            # This matches DSV3.2's mlp_mode=SCATTERED behavior with DeepEP + CP.
            #
            # Hash gating (n_hash_layers=3) needs input_ids[i] to correspond to
            # hidden_states[i]. hidden_states is LOCAL [N/cp_size] (round-robin).
            # input_ids is ORIGINAL [N] on every rank (never CP-split).
            # Slice to LOCAL to match hidden_states.
            cp_rank = get_attention_tp_rank()
            cp_size = get_attention_tp_size()
            input_ids = input_ids[cp_rank::cp_size].contiguous()
            # TODO: improve the name - it is indeed local in CP, but is only used by e.g. Hash gating
            input_ids_global = input_ids
        # ----------------------------------- DP: gather for TP MoE --------------------
        elif _use_tp_moe_gather:
            hidden_states, local_hidden_states = get_global_dp_buffer(), hidden_states
            dp_gather_partial(hidden_states, local_hidden_states, forward_batch)
        # ----------------------------------- MoE ------------------------------------
        if _trace:
            sys.stderr.write(f"[trace] L{self.layer_id} pre-mlp\n"); sys.stderr.flush()
        hidden_states = self.mlp(
            hidden_states,
            forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
        )
        if _trace:
            sys.stderr.write(f"[trace] L{self.layer_id} post-mlp\n"); sys.stderr.flush()
        # ----------------------------------- Scatter (DP only, not CP) ----------------
        if _use_tp_moe_gather:
            hidden_states, global_hidden_states = get_local_dp_buffer(), hidden_states
            dp_scatter(hidden_states, global_hidden_states, forward_batch)

        hidden_states = self.hc_post(
            hidden_states, residual, post, comb
        )  # [n, d] -> [n, hc, d]

        # if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B" and not _is_hip:
        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            assert deepseek_v4_moe_code_path_checker.observed == 1
            deepseek_v4_moe_code_path_checker.observed = 0

        return hidden_states


class DeepseekV4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.padding_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()
        self.first_k_dense_replace = config.first_k_dense_replace
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            enable_tp=not is_dp_attention_enabled(),
        )
        self.rms_norm_eps = config.rms_norm_eps
        self.alt_streams = (
            [torch.cuda.Stream() for _ in range(5)] if (_is_cuda or _is_hip) else None
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DeepseekV4DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_streams=self.alt_streams,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gemm_output_zero_allocator_size = 0
        self.layers_to_capture = []
        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            self.enable_a2a_moe = True
        else:
            self.enable_a2a_moe = False

        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        hc_dim = hc_mult * config.hidden_size
        self.hc_head_fn = nn.Parameter(
            torch.empty(hc_mult, hc_dim, dtype=torch.float32)
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_size = get_attention_tp_size()

    def hc_head(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)

    # TODO
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
        pp_proxy_tensors: Optional[PPProxyTensors],
    ) -> torch.Tensor:
        total_num_layers = self.end_layer - self.start_layer
        device = input_embeds.device if input_embeds is not None else input_ids.device
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2 * (2 if forward_batch.can_run_tbo else 1),
            dtype=torch.float32,
            device=device,
        )
        has_gemm_output_zero_allocator = hasattr(
            self, "gemm_output_zero_allocator_size"
        )
        gemm_output_zero_allocator = (
            BumpAllocator(
                buffer_size=self.gemm_output_zero_allocator_size,
                dtype=torch.float32,
                device=device,
            )
            if has_gemm_output_zero_allocator
            and self.gemm_output_zero_allocator_size > 0
            else None
        )
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)

        if get_attention_dp_size() > 1 and get_moe_a2a_backend().is_none():
            input_ids_global = torch.empty(
                (_DpGatheredBufferWrapper._global_dp_buffer_len, 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            dp_gather_partial(input_ids_global, input_ids[:, None], forward_batch)
            input_ids_global = input_ids_global.squeeze(-1)
        else:
            input_ids_global = input_ids

        if nsa_use_prefill_cp(forward_batch):
            hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)

        for i in range(self.start_layer, self.end_layer):
            # TODO: ctx?
            layer = self.layers[i]
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                input_ids=input_ids,
                input_ids_global=input_ids_global,
                # zero_allocator,
                # gemm_output_zero_allocator,
            )

        if nsa_use_prefill_cp(forward_batch):
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )

        pre_hc_head = (
            hidden_states.flatten(1)
            if envs.SGLANG_FIX_MTP_HC_HIDDEN.get()
            and envs.SGLANG_DSV4_MODE.get() == "2604"
            else None
        )

        _trace = os.environ.get("SGLANG_CAPTURE_TRACE", "0") == "1"
        if _trace:
            import sys
            sys.stderr.write("[trace] post-decoder pre-hc_head\n"); sys.stderr.flush()
        hidden_states = self.hc_head(
            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        if _trace:
            sys.stderr.write("[trace] post-hc_head pre-norm\n"); sys.stderr.flush()
        hidden_states = self.norm(hidden_states)
        if _trace:
            sys.stderr.write("[trace] post-norm return\n"); sys.stderr.flush()

        if pre_hc_head is not None:
            return hidden_states, pre_hc_head
        return hidden_states


class DeepseekV4ForCausalLM(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.model = DeepseekV4Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.pp_group = get_pp_group()
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
                use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
            )
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        # TODO: is this true that compress is kind of NSA
        get_attn_tp_context().init_context(config.q_lora_rank, is_nsa=True)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, deepseek_v2.DeepseekV2MoE)
            }
        )

        self.nsa_enable_prefill_cp = is_nsa_enable_prefill_cp()
        if self.nsa_enable_prefill_cp:
            self.cp_rank = get_attention_tp_rank()
            self.cp_size = get_attention_tp_size()

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    def determine_num_fused_shared_experts(self):
        self.num_fused_shared_experts = 0
        if get_global_server_args().disable_shared_experts_fusion:
            return

        # Only Deepseek V3/R1 can use shared experts fusion optimization now.
        disable_reason = None
        if self.config.n_routed_experts != 256 or self.config.n_shared_experts != 1:
            disable_reason = "Config not support fused shared expert(s)."
        elif (not _is_cuda or torch.cuda.get_device_capability("cuda") < (8, 0)) and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = (
                "Only Deepseek V3/R1 on NV-platform with capability >= 80 "
                "or AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization."
            )
        elif get_moe_expert_parallel_world_size() > 1 and (
            not _is_hip or torch.cuda.get_device_capability("cuda") < (9, 4)
        ):
            disable_reason = "Only Deepseek V3/R1 on AMD-platform with capability >= gfx942(MI30x) can use shared experts fusion optimization under expert parallelism."
        elif disable_reason is None and get_moe_a2a_backend().is_deepep():
            disable_reason = "Deepseek V3/R1 can not use shared experts fusion optimization under deepep expert parallelism."
        elif self.quant_config and self.quant_config.get_name() == "w4afp8":
            disable_reason = "Deepseek V3/R1 W4AFP8 model uses different quant method for routed experts and shared experts."
        elif (
            envs.SGLANG_DSV4_MODE.get() == "2604" and envs.SGLANG_DSV4_FP4_EXPERTS.get()
        ):
            disable_reason = "2604 routed experts use FP4 while shared experts remain FP8; fusion would incorrectly apply FP4 to shared experts."

        if envs.SGLANG_DSV4_2604_SUBMODE.get() == "2604B":
            disable_reason = "2604B checkpoint requires different clamping for shared and routed experts"

        if disable_reason is not None:
            get_global_server_args().disable_shared_experts_fusion = True
            self.num_fused_shared_experts = 0
            log_info_on_rank0(
                logger,
                f"{disable_reason} Shared experts fusion optimization is disabled.",
            )
            return

        self.num_fused_shared_experts = self.config.n_shared_experts

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:

        if self.nsa_enable_prefill_cp:
            if can_cp_split(len(input_ids), self.cp_size, True, forward_batch):
                forward_batch.nsa_cp_metadata = prepare_input_dp_with_cp_dsa(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                )

        _trace = os.environ.get("SGLANG_CAPTURE_TRACE", "0") == "1"
        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model.forward(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        if _trace:
            import sys
            sys.stderr.write("[trace] causallm post-model pre-logits\n"); sys.stderr.flush()
        aux_hidden_states = None
        pre_hc_head = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if (
            envs.SGLANG_FIX_MTP_HC_HIDDEN.get()
            and envs.SGLANG_DSV4_MODE.get() == "2604"
        ):
            hidden_states, pre_hc_head = hidden_states
        if _trace:
            sys.stderr.write("[trace] causallm pre-logits_processor\n"); sys.stderr.flush()
        logits = self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
            # TODO: indeed ours is "hidden_states_before_hc_head" instead of "before norm"
            #       abuse the existing field temporarily to minimize code diff
            #       should rename and generalize later, e.g. "hidden_states_for_spec"
            hidden_states_before_norm=pre_hc_head,
        )
        if _trace:
            sys.stderr.write("[trace] causallm post-logits_processor return\n"); sys.stderr.flush()
        return logits

    def _setup_fp8_wo_a_scales(self, is_nextn: bool) -> None:
        from deep_gemm import transform_sf_into_required_layout

        layers = self.model.layers
        for layer in layers:
            attn = layer.self_attn
            G = attn.n_local_groups
            R = attn.o_lora_rank
            D = attn.wo_a.weight.shape[1]

            # Pre-transform weight scale to DeepGEMM required layout (TMA-aligned / UE8M0 packed)
            # fp8_einsum('bhr,hdr->bhd') maps B=[h,d,r]=[G,R,D], so N=R, K=D for the B-side scale
            raw_scale = attn.wo_a.weight_scale_inv.data.view(G, R // 128, D // 128)
            attn.wo_a.weight_scale_inv.data = transform_sf_into_required_layout(
                raw_scale,
                mn=R,
                k=D,
                recipe=(1, 128, 128),
                num_groups=G,
                is_sfa=False,
            )

    def post_load_weights(self, is_nextn=False, weight_names=None):
        if _FP8_WO_A_GEMM:
            self._setup_fp8_wo_a_scales(is_nextn)

        # ================ apply_ape_hotfix, should not be needed for final ckpt ================
        if is_nextn:
            return
        for layer in self.model.layers:
            self_attn = layer.self_attn
            if self_attn.compress_ratio != 0 and not self_attn.compressor.ape_converted:
                self_attn.compressor.apply_ape_hotfix()
            if (
                self_attn.compress_ratio == 4
                and not self_attn.indexer.compressor.ape_converted
            ):
                self_attn.indexer.compressor.apply_ape_hotfix()

    # This is used externally, please try to keep the API mostly unchanged
    @staticmethod
    def remap_weight_name_to_dpsk_hf_format(
        name: str,
        is_nextn: bool = False,
        num_hidden_layers: Optional[int] = None,
        *,
        mlp_scale_to_weight_scale_inv: bool = True,
    ) -> str:
        if name == "embed.weight":
            return "model.embed_tokens.weight"
        if name == "head.weight":
            return "lm_head.weight"
        if name == "norm.weight":
            return "model.norm.weight"
        if name.startswith("hc_head_"):
            return "model." + name

        if is_nextn and name.startswith("mtp."):
            parts = name.split(".", 2)
            if len(parts) >= 3:
                rest = parts[2]
                nextn_spec_prefixes = [
                    "e_proj",
                    "h_proj",
                    "emb",
                    "enorm",
                    "hnorm",
                    "norm",
                    "head",
                    "hc_head",
                ]
                is_nextn_spec = any(rest.startswith(p) for p in nextn_spec_prefixes)
                if is_nextn_spec:
                    if rest.startswith("emb.tok_emb"):
                        rest = rest.replace("emb.tok_emb", "embed_tokens")
                    elif rest == "norm.weight":
                        rest = "shared_head.norm.weight"
                    elif rest.startswith("head."):
                        rest = "shared_head.head.weight"
                    elif rest == "e_proj.scale":
                        rest = "e_proj.weight_scale_inv"
                    elif rest == "h_proj.scale":
                        rest = "h_proj.weight_scale_inv"
                name = f"model.layers.{num_hidden_layers}." + rest

        if name.startswith("layers."):
            name = "model." + name
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".attn_norm.", ".input_layernorm.")
        name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

        if not ATTN_BIT_WISE_EQUAL_MODE:
            if "self_attn" in name and (
                "compressor" not in name or not COMPRESSOR_BIT_WISE_EQUAL_MODE
            ):
                name = name.replace(".scale", ".weight_scale_inv")

        if not MOE_BIT_WISE_EQUAL_MODE:
            name = name.replace(".gate.tid2eid", ".topk.tid2eid")
            name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
            name = name.replace(".w1.", ".gate_proj.")
            name = name.replace(".w2.", ".down_proj.")
            name = name.replace(".w3.", ".up_proj.")
            if "mlp" in name:
                # Hybrid mxfp4 ckpt: routed experts (`mlp.experts.{E}.*`)
                # use `.scale` (e8m0) directly; do NOT rename. Shared
                # experts and other mlp linears stay pure fp8 and need the
                # `.scale -> .weight_scale_inv` rename either way.
                if mlp_scale_to_weight_scale_inv or ".experts." not in name:
                    name = name.replace(".scale", ".weight_scale_inv")

        return name

    def _try_consume_mxfp4_expert_weight(
        self,
        name: str,
        loaded_weight: torch.Tensor,
        loaded_params: Set[str],
    ) -> bool:
        """Per-expert mxfp4 weight slot copy for hybrid checkpoints.

        Matches names of the form
            model.layers.{L}.mlp.experts.{E}.{gate|up|down}_proj.{weight|scale}
        and writes a TP-sliced copy of `loaded_weight` into the fused
        `experts.{w13_weight[_scale]|w2_weight[_scale]}` parameter slot for
        local expert `E`. Returns True if the name was consumed.

        Both the int8-packed mxfp4 weights and the float8_e8m0fnu scales are
        single-byte tensors; we reinterpret them as `uint8` (the dtype of the
        registered Mxfp4MoEMethod params) without value conversion.
        """
        import re
        m = re.match(
            r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|scale)$",
            name,
        )
        if m is None:
            return False
        layer_idx = int(m.group(1))
        global_eid = int(m.group(2))
        proj = m.group(3)
        suffix = m.group(4)

        try:
            layer = self.model.layers[layer_idx]
            experts = layer.mlp.experts
        except (AttributeError, IndexError):
            return False
        # Only intercept mxfp4 FusedMoE experts; otherwise fall through.
        if not hasattr(experts, "w13_weight") or not hasattr(experts, "w13_weight_scale"):
            return False

        local_eid = experts._map_global_expert_id_to_local_expert_id(global_eid)
        if local_eid is None or local_eid < 0:
            # This rank doesn't own this expert; treat as consumed.
            return True

        ipp = experts.intermediate_size_per_partition
        moe_tp_rank = experts.moe_tp_rank

        # Reinterpret int8/e8m0 byte storage as uint8 for the registered param.
        src = loaded_weight.view(torch.uint8) if loaded_weight.dtype != torch.uint8 else loaded_weight

        if proj in ("gate_proj", "up_proj"):
            row_start = moe_tp_rank * ipp
            row_end = row_start + ipp
            src = src.narrow(0, row_start, ipp)  # (ipp, K)
            dst = experts.w13_weight if suffix == "weight" else experts.w13_weight_scale
            slot_lo = 0 if proj == "gate_proj" else ipp
            slot_hi = slot_lo + ipp
            dst.data[local_eid, slot_lo:slot_hi, : src.shape[1]].copy_(src)
            fused_name = (
                f"model.layers.{layer_idx}.mlp.experts."
                + ("w13_weight" if suffix == "weight" else "w13_weight_scale")
            )
            loaded_params.add(fused_name)
        else:  # down_proj
            if suffix == "weight":
                col_size = ipp // 2
            else:  # scale uses mxfp4 block of 32
                col_size = ipp // 32
            col_start = moe_tp_rank * col_size
            src = src.narrow(1, col_start, col_size)  # (hidden, col_size)
            dst = experts.w2_weight if suffix == "weight" else experts.w2_weight_scale
            dst.data[local_eid, : src.shape[0], : col_size].copy_(src)
            fused_name = (
                f"model.layers.{layer_idx}.mlp.experts."
                + ("w2_weight" if suffix == "weight" else "w2_weight_scale")
            )
            loaded_params.add(fused_name)
        return True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        assert envs.SGLANG_DSV4_MODE.get() in ["2601", "2604"]
        if envs.SGLANG_DSV4_MODE.get() == "2604":
            assert envs.SGLANG_DSV4_2604_SUBMODE.get() in ["2604A", "2604B"]
        else:
            assert envs.SGLANG_DSV4_2604_SUBMODE.get() == ""

        if MOE_BIT_WISE_EQUAL_MODE:
            assert (
                self.num_fused_shared_experts == 0
            ), "use --disable-shared-experts-fusion for MoE bit-wise equal mode"

        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        # Hybrid mxfp4 checkpoints (e.g. DeepSeek-V4-Flash) ship per-expert
        # mxfp4 packed weights with `.scale` (e8m0) suffix, no biases. The
        # default mlp `.scale -> .weight_scale_inv` rename is wrong for that
        # path (registered Mxfp4 params end in `_weight_scale`, not `_inv`).
        is_mxfp4_moe_static = bool(
            self.quant_config is not None
            and self.quant_config.get_name() == "mxfp4"
            and getattr(self.quant_config, "is_checkpoint_mxfp4_serialized", False)
        )

        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                # compatible with old design
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        # Ignore this, b/c it is for nvfp4 ckpt
        # weights = self._maybe_quant_weights_to_fp8_ue8m0(
        #     weights, NVFP4_CKPT_FP8_ATTN_QUANT_MODULES, is_nextn
        # )

        if not envs.SGLANG_OPT_FP8_WO_A_GEMM.get():
            # Probe one wo_a weight to decide: if stored FP8, dequant inline;
            # if already bf16, just drop any stale scale tensor.
            weights = _maybe_dequant_fp8_wo_a(weights)

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )
        # Params for special naming rules in mixed-precision models, for example:
        # model.layers.xx.mlp.experts.xx.w1.input_scale. For details,
        # see https://huggingface.co/Barrrrry/DeepSeek-R1-W4AFP8/blob/main.

        if self.quant_config and self.quant_config.get_name() == "w4afp8":
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        # fuse compressor wkv and wgate weights into wkv_gate
        cache_compressor_weight = {}
        COMPRESSOR_PART = ".compressor.w"  # match wkv and wgate, skip ape

        # use default weight loader if module has no custom weight_loader
        def auto_weight_loader(module):
            return getattr(module, "weight_loader", default_weight_loader)

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names_out_of_layer = [
                "shared_head.norm",
                "shared_head.head",
                "embed_tokens",
                ".e_proj",  # Note that we need a . here to avoid confusion with gate_proj
                "h_proj",
                "enorm",
                "hnorm",
                "hc_head_base",
                "hc_head_fn",
                "hc_head_scale",
            ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            weight_names = []
            for name, loaded_weight in weights:
                try:
                    use_async_loading = should_async_load(loaded_weight)

                    # remap reference's temp ckpt weight -> deepseek hf format
                    name = self.remap_weight_name_to_dpsk_hf_format(
                        name,
                        is_nextn=is_nextn,
                        num_hidden_layers=self.config.num_hidden_layers,
                        mlp_scale_to_weight_scale_inv=not is_mxfp4_moe_static,
                    )

                    # mxfp4 hybrid checkpoint: per-expert experts.{E}.{proj}
                    # weight + .scale tensors land in the fused
                    # `experts.w13_weight[_scale]` / `experts.w2_weight[_scale]`
                    # params via direct slice copy. Skip the rest of the
                    # standard mapping pipeline for these keys.
                    if is_mxfp4_moe_static and self._try_consume_mxfp4_expert_weight(
                        name, loaded_weight, loaded_params
                    ):
                        continue

                    layer_id = get_layer_id(name)
                    if (
                        layer_id is not None
                        and hasattr(self.model, "start_layer")
                        and (
                            layer_id < self.model.start_layer
                            or layer_id >= self.model.end_layer
                        )
                    ):
                        continue
                    if (
                        self.num_fused_shared_experts > 0
                        and "mlp.shared_experts" in name
                    ):
                        name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts}",
                        )

                    weight_names.append(name)

                    if not is_nextn:
                        if hasattr(self.config, "num_nextn_predict_layers"):
                            num_nextn_layers = self.config.num_nextn_predict_layers
                            if num_nextn_layers > 0 and name.startswith("model.layers"):
                                name_list = name.split(".")
                                if (
                                    len(name_list) >= 3
                                    and int(name_list[2])
                                    >= self.config.num_hidden_layers
                                ):
                                    continue

                            if name.startswith("mtp"):
                                continue
                    else:
                        # Use shared head and embed weights from target model
                        if "shared_head.head" in name or "embed_tokens" in name:
                            continue

                        # Skip target model weights
                        if not name.startswith(nextn_layer_prefix):
                            continue

                        in_decoder = True
                        # For nextn specific weights (out of layer)
                        # The nextn layer prefix of these weights has been removed
                        for weight_name in nextn_spec_weight_names_out_of_layer:
                            if weight_name in name:
                                in_decoder = False
                                name = name.replace(nextn_layer_prefix, "model")
                                break

                        # For decoder layer weights
                        if in_decoder:
                            name = name.replace(nextn_layer_prefix, "model.decoder")

                    if "rotary_emb.inv_freq" in name:
                        continue
                    for param_name, weight_name, shard_id in stacked_params_mapping:
                        # Skip non-stacked layers and experts (experts handled below).
                        if weight_name not in name:
                            continue
                        if _is_npu:
                            name = name.replace("weight_packed", "weight")
                        # We have mlp.experts[0].gate_proj in the checkpoint.
                        # Since we handle the experts below in expert_params_mapping,
                        # we need to skip here BEFORE we update the name, otherwise
                        # name will be updated to mlp.experts[0].gate_up_proj, which
                        # will then be updated below in expert_params_mapping
                        # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                        if ("mlp.experts." in name) and name not in params_dict:
                            continue
                        name = name.replace(weight_name, param_name)
                        # Skip loading extra bias for GPTQ models.
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if name not in params_dict and name.startswith("mtp"):  # TODO
                            break
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(param, loaded_weight, shard_id),
                        )
                        loaded_params.add(name)
                        break
                    else:
                        for mapping in expert_params_mapping:
                            if MOE_BIT_WISE_EQUAL_MODE:
                                continue
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            if _is_npu:
                                name = name.replace("weight_packed", "weight")
                            name = name.replace(weight_name, param_name)
                            if name not in params_dict:
                                continue
                            param = params_dict[name]
                            weight_loader = param.weight_loader
                            maybe_executor_submit(
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(
                                    param,
                                    loaded_weight,
                                    name,
                                ),
                                func_kwargs={
                                    "shard_id": shard_id,
                                    "expert_id": expert_id,
                                },
                            )
                            loaded_params.add(name)
                            break
                        else:
                            # Skip loading extra bias for GPTQ models.
                            if name.endswith(".bias") and name not in params_dict:
                                continue
                            # Skip loading embed_tokens if not first rank in pipeline parallelism
                            if (
                                ".embed_tokens." in name
                                and not self.pp_group.is_first_rank
                            ):
                                continue
                            # Skip loading norm if not last rank in pipeline parallelism
                            if ".norm." in name and not self.pp_group.is_last_rank:
                                continue
                            elif COMPRESSOR_PART in name:
                                is_kv = name.endswith(".wkv.weight")
                                is_wgate = name.endswith(".wgate.weight")
                                assert is_kv != is_wgate  # exactly one is true
                                key = name.rsplit(".", 2)[0]
                                assert key.endswith(".compressor")
                                if key not in cache_compressor_weight:
                                    cache_compressor_weight[key] = (
                                        is_kv,
                                        loaded_weight,
                                    )
                                else:
                                    assert key in cache_compressor_weight
                                    cached_is_kv, cached_weight = (
                                        cache_compressor_weight[key]
                                    )
                                    assert cached_is_kv != is_kv
                                    kv = loaded_weight if is_kv else cached_weight
                                    wgate = loaded_weight if is_wgate else cached_weight
                                    fused_weight = torch.cat([kv, wgate], dim=0)
                                    param_name = key + ".wkv_gate.weight"
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_compressor_weight.pop(key)
                            else:
                                if (
                                    "k_scale" in name or "v_scale" in name
                                ) and name not in params_dict:
                                    # modelopt attn kv scale is named differently
                                    for scale in ["k_scale", "v_scale"]:
                                        if scale in name:
                                            name = name.replace(
                                                f"{scale[0]}_proj", "attn_mqa"
                                            )
                                            break
                                if name not in params_dict:
                                    # modelopt ckpt contains not needed weights for MTP module:
                                    # model.decoder.self_attn.attn_mqa.v_scale and
                                    # model.decoder.self_attn.attn_mqa.k_scale
                                    if not name.startswith("mtp"):  # TODO: mtp
                                        logger.warning(
                                            f"{name} not found in params_dict."
                                        )
                                    continue
                                param = params_dict[name]

                                # if "attn_sink" in name:
                                #     attn_tp_rank = get_attention_tp_rank()
                                #     start = attn_tp_rank * param.numel()
                                #     param.data.copy_(
                                #         loaded_weight[start : start + param.numel()]
                                #     )
                                #     loaded_params.add(name)
                                #     continue

                                weight_loader = auto_weight_loader(param)
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, loaded_weight),
                                )
                                loaded_params.add(name)
                except Exception as e:
                    e.add_note(f"{name=} {loaded_weight.shape=}")
                    raise

            # Wait for all tasks to complete and raise any exceptions.
            for future in concurrent.futures.as_completed(futures):
                future.result()

        assert len(cache_compressor_weight) == 0
        unloaded_params = params_dict.keys() - loaded_params

        skipped_checking_patterns = ["attn_mqa.k_scale", "attn_mqa.v_scale"]
        if is_nextn:
            skipped_checking_patterns.extend(["lm_head", "embed_tokens"])
        unloaded_params = {
            p
            for p in unloaded_params
            # hack to skip checking these in default ckpt. should have more rigorous check.
            if all(
                skipped_checking_pattern not in p
                for skipped_checking_pattern in skipped_checking_patterns
            )
        }
        if os.environ.get("SGLANG_SKIP_CHECKPOINT_LOAD_CHECK", "0") == "0":
            if unloaded_params:
                raise RuntimeError(
                    f"Some weights are not initialized from checkpoints: {unloaded_params}"
                )

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.n_routed_experts,
            num_groups=None,
        )


EntryClass = [DeepseekV4ForCausalLM]


def _dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequant fp8 block-quantized wo_a weight: bf16 = fp8_weight * e8m0_scale.

    Per-128x128 block scaling. Shapes seen in DSv4 2604 checkpoints:
      Flash-Base: weight [8192, 4096] fp8_e4m3fn, scale [64, 32]
      Pro-Base:   weight [16384, 4096] fp8_e4m3fn, scale [128, 32]
    Generalised to any 2D (M, K) with M % 128 == 0 and K % 128 == 0.
    """
    from einops import rearrange

    assert (
        weight.dtype == torch.float8_e4m3fn
    ), f"expected fp8_e4m3fn, got {weight.dtype}"
    assert scale.dtype in (
        torch.float8_e8m0fnu,
        torch.float32,
    ), f"expected fp8_e8m0fnu or float32, got {scale.dtype}"
    assert (
        weight.dim() == 2 and weight.shape[0] % 128 == 0 and weight.shape[1] % 128 == 0
    ), f"unexpected weight shape {weight.shape} (need 2D, both dims % 128 == 0)"
    expected_scale_shape = (weight.shape[0] // 128, weight.shape[1] // 128)
    assert (
        scale.shape == expected_scale_shape
    ), f"scale shape {scale.shape} doesn't match {expected_scale_shape} for weight {weight.shape}"

    weight_f32 = rearrange(
        weight.float(), "(sn bn) (sk bk) -> sn bn sk bk", bn=128, bk=128
    )
    result = rearrange(
        weight_f32 * scale.float()[:, None, :, None], "sn bn sk bk -> (sn bn) (sk bk)"
    )

    assert result.shape == weight.shape
    return result.to(torch.bfloat16)


def _maybe_dequant_fp8_wo_a(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterable[Tuple[str, torch.Tensor]]:
    """Auto-detect checkpoint wo_a layout and dequant inline when needed.

    - FP8 wo_a + separate scale (HF DeepSeek-V4-Flash{,-Base}): dequant to bf16.
    - BF16 wo_a + stale scale (some converted checkpoints): drop stale scale.
    - BF16 wo_a, no scale: passthrough.
    """
    weights_dict = dict(weights)
    sample_wo_a = next(
        (t for n, t in weights_dict.items() if n.endswith(".wo_a.weight")),
        None,
    )
    is_fp8_wo_a = sample_wo_a is not None and sample_wo_a.dtype == torch.float8_e4m3fn

    if is_fp8_wo_a:
        yield from _dequant_fp8_wo_a(weights_dict.items())
    else:
        for n, t in weights_dict.items():
            if n.endswith(".wo_a.scale"):
                continue
            yield n, t


def _dequant_fp8_wo_a(
    weights: Iterable[Tuple[str, torch.Tensor]],
) -> Iterable[Tuple[str, torch.Tensor]]:
    """Dequant fp8 wo_a weights inline: pair (wo_a.scale, wo_a.weight) -> bf16 wo_a.weight.

    2601 checkpoint:
      layers.0.attn.wo_a.weight  torch.bfloat16  [8192, 4096]  64.00MB  min=-0.375 max=0.3125

    2604 checkpoint:
      layers.0.attn.wo_a.scale  torch.float8_e8m0fnu  [64, 32]  0.00MB
      layers.0.attn.wo_a.weight  torch.float8_e4m3fn  [8192, 4096]  32.00MB
    """
    weights_dict = dict(weights)

    for name in list(weights_dict.keys()):
        if name not in weights_dict:
            continue
        if not name.endswith(".wo_a.weight"):
            continue
        scale_name = name.replace(".wo_a.weight", ".wo_a.scale")
        assert scale_name in weights_dict
        weight = weights_dict.pop(name)
        scale = weights_dict.pop(scale_name)
        yield name, _dequant_fp8(weight, scale)

    yield from weights_dict.items()
