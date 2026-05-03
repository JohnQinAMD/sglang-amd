import os
from typing import Optional, Tuple

import torch

from .lib import KVScope, Testcase, TestcaseForDecode, TestParam

# R5 P1-b: historically this code called torch.nan_to_num on every attention
# call to scrub NaNs produced by the pre-FP8_MAX-fix quantize_k_cache bug
# (FNUZ overflow clamped to byte 0x80). Now that the quantizer uses the actual
# FP8 max, there shouldn't be any NaN; this flag re-enables the defensive
# mask for debugging / A/B only.
_KV_NAN_DEFENSE = os.environ.get("SGLANG_SKIP_KV_NAN_TO_NUM", "1") != "1"
# torch.compile wrap for the hot sparse-attn inner loop on AMD.
# Disabled if SGLANG_DISABLE_REF_ATTN_COMPILE=1.
_USE_COMPILE = os.environ.get("SGLANG_DISABLE_REF_ATTN_COMPILE", "0") != "1"
# Opt-in Triton sparse-decode kernel (AMD MI355X / gfx950).
# Takes precedence over torch.compile when set to "1".
_USE_TRITON = os.environ.get("SGLANG_TRITON_SPARSE_DECODE", "0") == "1"
# R5 P1-e: pass bf16 Q/K directly to _sparse_attn_decode_inner instead of
# up-casting to fp32 outside, saving a ~30 ms/prefill bf16→f32 copy
# (torch.compile's BMM still accumulates in fp32 internally, so numerics
# are equivalent). Set SGLANG_FORCE_FP32_ATTN_INNER=1 to restore old path.
_FORCE_FP32_INNER = os.environ.get("SGLANG_FORCE_FP32_ATTN_INNER", "0") == "1"


def _sparse_attn_decode_inner(
    q_f32: torch.Tensor,            # [B*Sq, Hq, D]
    gathered_kv_f32: torch.Tensor,  # [B*Sq, Topk, D]
    invalid_mask: torch.Tensor,     # [B*Sq, Topk] bool
    attn_sink: Optional[torch.Tensor],  # [Hq] or None
    sm_scale: float,
    d_v: int,
    h_q: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-tensor sparse decode core: q @ K^T, masked softmax, @ V, optional sink scaling.
    Returns (output[B*Sq, Hq, dv] bf16, lse[B*Sq, Hq] f32).
    """
    attn_weight = q_f32 @ gathered_kv_f32.transpose(-1, -2)  # [B*Sq, Hq, Topk]
    attn_weight = attn_weight * sm_scale
    attn_weight = attn_weight.masked_fill(
        invalid_mask.unsqueeze(1).expand_as(attn_weight), float("-inf")
    )
    lse = attn_weight.logsumexp(dim=-1)  # [B*Sq, Hq]
    attn_weight = torch.exp(attn_weight - lse.unsqueeze(-1))
    output = attn_weight @ gathered_kv_f32[..., :d_v]  # [B*Sq, Hq, dv]
    if attn_sink is not None:
        scale = 1.0 / (1.0 + torch.exp(attn_sink.view(1, h_q) - lse))
        output = output * scale.unsqueeze(-1)
    return output.to(torch.bfloat16), lse


def _sparse_attn_decode_inner_triton(
    q_f32: torch.Tensor,            # [B*Sq, Hq, D]
    gathered_kv_f32: torch.Tensor,  # [B*Sq, Topk, D]
    invalid_mask: torch.Tensor,     # [B*Sq, Topk] bool
    attn_sink: Optional[torch.Tensor],  # [Hq] or None
    sm_scale: float,
    d_v: int,
    h_q: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton-backed sparse decode core for AMD MI355X (gfx950).

    Drop-in replacement for ``_sparse_attn_decode_inner``. Returns the same
    ``(output[B*Sq, Hq, dv] bf16, lse[B*Sq, Hq] fp32)`` tuple.

    The Triton kernel operates on bf16 Q/K; sglang callers pass fp32, so we
    cast on the boundary. The kernel itself masks invalid rows on K/V loads
    (`other=0.0`, see triton_sparse_decode_kernel.py:76-79, 99-101) and
    handles all-masked rows in the online softmax (lines 84, 88-92, 109),
    so an outer ``nan_to_num`` guard is redundant.
    """
    # Cast to bf16 (kernel contract).
    q_bf = (
        q_f32.to(torch.bfloat16).contiguous()
        if q_f32.dtype != torch.bfloat16
        else q_f32.contiguous()
    )
    kv_bf = (
        gathered_kv_f32.to(torch.bfloat16).contiguous()
        if gathered_kv_f32.dtype != torch.bfloat16
        else gathered_kv_f32.contiguous()
    )

    from .triton_sparse_decode_kernel import triton_sparse_attn_decode

    return triton_sparse_attn_decode(
        q_bf, kv_bf, invalid_mask, attn_sink, float(sm_scale), int(d_v)
    )


if _USE_TRITON:
    _sparse_attn_decode_inner = _sparse_attn_decode_inner_triton
elif _USE_COMPILE:
    _sparse_attn_decode_inner = torch.compile(
        _sparse_attn_decode_inner, dynamic=True, fullgraph=False
    )


def _merge_two_lse(
    lse0: torch.Tensor, lse1: Optional[torch.Tensor], s_q: int, h_q: int
) -> torch.Tensor:
    if lse1 is None:
        return lse0
    else:
        return torch.logsumexp(
            torch.stack([lse0.view(s_q, h_q), lse1.broadcast_to(s_q, h_q)], dim=0),
            dim=0,
        )


def ref_sparse_attn_fwd(
    p: TestParam, t: Testcase
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
    - o: [s_q, h_q, dv]
    - o_fp32: [s_q, h_q, dv]
    - max_logits: [s_q, h_q]
    - lse: [s_q, h_q]
    """
    indices = t.indices.clone().squeeze(1)
    if t.topk_length is not None:
        mask = torch.arange(p.topk, device=t.topk_length.device).unsqueeze(
            0
        ).broadcast_to(p.s_q, p.topk) >= t.topk_length.unsqueeze(
            1
        )  # [s_q, topk]
        indices[mask] = -1
    invalid_mask = (indices < 0) | (indices >= p.s_kv)  # [s_q, topk]
    indices[invalid_mask] = 0

    q = t.q.float()
    gathered_kv = (
        t.kv.index_select(dim=0, index=indices.flatten())
        .reshape(p.s_q, p.topk, p.d_qk)
        .float()
    )  # [s_q, topk, d_qk]
    P = q @ gathered_kv.transpose(1, 2)  # [s_q, h_q, topk]
    P *= t.sm_scale
    P[invalid_mask.unsqueeze(1).broadcast_to(P.shape)] = float("-inf")

    orig_lse = torch.logsumexp(P, dim=-1)  # [s_q, h_q]
    max_logits = P.max(dim=-1).values  # [s_q, h_q]

    lse_for_o = _merge_two_lse(orig_lse, t.attn_sink, p.s_q, p.h_q)
    if not torch.is_inference_mode_enabled():
        lse_for_o = lse_for_o.clone()
    lse_for_o[lse_for_o == float("-inf")] = float(
        "+inf"
    )  # So that corresponding O will be 0
    s_for_o = torch.exp(P - lse_for_o.unsqueeze(-1))
    out = s_for_o @ gathered_kv[..., : p.d_v]  # [s_q, h_q, dv]

    lonely_q_mask = orig_lse == float("-inf")  # [s_q, h_q]
    orig_lse[lonely_q_mask] = float("+inf")
    return (out.to(torch.bfloat16), out, max_logits, orig_lse)


def ref_sparse_attn_decode(
    p: TestParam, t: TestcaseForDecode
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A reference implementation of sparse decoding attention in PyTorch
    """
    assert p.h_kv == 1
    assert p.decode is not None
    b = p.decode.b

    def process_kv_scope(kv_scope: KVScope) -> Tuple[torch.Tensor, torch.Tensor]:
        assert kv_scope.indices_in_kvcache is not None
        topk = kv_scope.indices_in_kvcache.size(-1)
        indices_in_kv_cache_fixed = torch.clamp_min(
            kv_scope.indices_in_kvcache, 0
        )  # Otherwise torch.index_select will complain
        if (
            getattr(kv_scope, "blocked_k", None) is None
            and getattr(kv_scope, "blocked_k_quantized", None) is not None
        ):
            # P0-c gather-first FP8 dequant: only dequant the topk rows we
            # actually keep. Cuts per-layer HBM traffic by
            # `num_blocks*block_size / (b*s_q*topk)` (typically 30-100× on DSv4).
            from . import quant as _quant  # local import to avoid cycles
            bytes_per_token = kv_scope.blocked_k_quantized.shape[-1]
            # MODEL1_FP8Sparse (MI300): bytes_per_token = block_size*(d_nope+2*d_rope) + 8*block_size = 584
            # V32_FP8Sparse   (SM90)  : bytes_per_token = d_nope + num_tiles*4 + 2*d_rope        = 464
            fp8_layout = (
                _quant.FP8KVCacheLayout.MODEL1_FP8Sparse
                if bytes_per_token == 584
                else _quant.FP8KVCacheLayout.V32_FP8Sparse
            )
            gathered_flat = _quant.dequantize_k_cache_gather(
                kv_scope.blocked_k_quantized.view(_quant.FP8_DTYPE),
                indices_in_kv_cache_fixed.view(-1),
                fp8_layout,
            )  # [b*s_q*topk, d]
            gathered_kv = gathered_flat.view(b, p.s_q, topk, p.d_qk)
        else:
            gathered_kv = (
                kv_scope.blocked_k.view(-1, p.d_qk)
                .index_select(0, indices_in_kv_cache_fixed.view(-1))
                .view(b, p.s_q, topk, p.d_qk)
            )  # [b, s_q, topk, d]
        # 2026-05-03 Target 2: M3 publish/lookup short-circuit. The M3 indexer
        # megakernel pre-computes (idx == -1 | arange >= topk_length) and publishes
        # it under (data_ptr(indices), id(topk_length)). On hit we skip the 4 ops
        # below (Compare + arange + Compare + BitwiseOr). Trace-confirmed
        # AUnaryFunctor / BitwiseOr / arange all -1.0/lyr; aligned bench TPOT
        # 24.97→24.58 ms (-0.39 ms / +1.6% tput). Default-ON; opt out with
        # SGLANG_M3_REF_LOOKUP=0.
        import os as _os
        _m3_lookup_on = _os.environ.get("SGLANG_M3_REF_LOOKUP", "1") == "1"
        invalid_mask = None
        if _m3_lookup_on:
            try:
                from sglang.jit_kernel.m3_indexer_megakernel_triton import (
                    lookup_invalid_mask as _m3_lookup,
                )
                # M3 publishes shape (B*S_q, K). ref.py builds (b, s_q, topk) so reshape on hit.
                _m3_2d = _m3_lookup(
                    kv_scope.indices_in_kvcache,
                    kv_scope.topk_length,
                    expected_shape=(b * p.s_q, topk),
                )
                if _m3_2d is not None:
                    invalid_mask = _m3_2d.view(b, p.s_q, topk)
            except Exception:
                invalid_mask = None
        if invalid_mask is None:
            invalid_mask = kv_scope.indices_in_kvcache == -1
            if kv_scope.topk_length is not None:
                invalid_mask |= torch.arange(0, topk, device=invalid_mask.device).view(
                    1, 1, topk
                ).broadcast_to(b, p.s_q, topk) >= kv_scope.topk_length.view(b, 1, 1)
        return gathered_kv, invalid_mask

    gathered_kv, invalid_mask = process_kv_scope(t.kv_scope)
    if t.extra_kv_scope is not None:
        gathered_kv1, invalid_mask1 = process_kv_scope(t.extra_kv_scope)
        # 2026-05-03 T4: fused dual-cat (KV + mask) into pre-allocated buffers in
        # one Triton launch instead of two `torch.cat` calls. Saves 1 launch per
        # sparse-decode call. Microbench at production shape (4,1,512,2,512):
        # cuda-graph 11.99 → 9.41 us (1.27x). E2E TPOT median 23.14 → 22.82 ms
        # (−0.32 ms / −1.4%); throughput +1.2%. Default-ON; opt out with
        # SGLANG_FUSED_DUAL_CAT=0.
        import os as _os_t4
        if _os_t4.environ.get("SGLANG_FUSED_DUAL_CAT", "1") == "1":
            try:
                from sglang.jit_kernel.fused_dual_cat_triton import fused_dual_cat
                gathered_kv, invalid_mask = fused_dual_cat(
                    gathered_kv, gathered_kv1,
                    invalid_mask, invalid_mask1,
                )
            except Exception:
                gathered_kv = torch.cat([gathered_kv, gathered_kv1], dim=2)
                invalid_mask = torch.cat([invalid_mask, invalid_mask1], dim=2)
        else:
            gathered_kv = torch.cat(
                [gathered_kv, gathered_kv1], dim=2
            )  # [b, s_q, topk+extra_topk, d]
            invalid_mask = torch.cat(
                [invalid_mask, invalid_mask1], dim=2
            )  # [b, s_q, topk+extra_topk]

    # may use more advanced approach

    # R5 P1-e: drop fp32 upcast by default (saves ~30 ms/prefill in bf16→f32
    # copy + lets the inner BMM run on bf16 v_mfma which is faster on gfx950
    # than fp32 rocBLAS). torch.compile's BMM and the Triton kernel both
    # accumulate in fp32 internally so numerics are preserved.
    if _FORCE_FP32_INNER:
        gathered_kv = gathered_kv.view(b * p.s_q, -1, p.d_qk).float()
    else:
        gathered_kv = gathered_kv.view(b * p.s_q, -1, p.d_qk)
    # Triton sparse-decode kernel masks invalid rows on K/V load (`other=0.0`),
    # so the outer NaN-scrub is a redundant launch on that path. The pure-tensor
    # / torch.compile path computes `attn_weight @ gathered_kv` directly and
    # would propagate NaN, so it still needs the guard. R5 P1-b disables this
    # by default since the FP8_MAX fix in quantize_k_cache prevents the NaN
    # at the source.
    if not _USE_TRITON and _KV_NAN_DEFENSE:
        gathered_kv = torch.nan_to_num(gathered_kv, nan=0.0)
    if _FORCE_FP32_INNER:
        q = t.q.float().view(b * p.s_q, p.h_q, p.d_qk)
    else:
        q = t.q.view(b * p.s_q, p.h_q, p.d_qk)
    invalid_mask_2d = invalid_mask.view(b * p.s_q, -1)

    output, lse = _sparse_attn_decode_inner(
        q,
        gathered_kv,
        invalid_mask_2d,
        t.attn_sink,
        float(t.sm_scale),
        int(p.d_v),
        int(p.h_q),
    )

    output = output.view(b, p.s_q, p.h_q, p.d_v)
    lse = lse.view(b, p.s_q, p.h_q)

    # Correct for q tokens which has no attendable k.
    # NOTE: previously this was guarded by `if lonely_q_mask.any():` which
    # forces a GPU->CPU sync and breaks CUDA graph capture
    # (HIP error: operation not permitted when stream is capturing). We now
    # always apply the where; it's a no-op when the mask is all-False.
    lonely_q_mask = lse == float("-inf")
    output = torch.where(
        lonely_q_mask.unsqueeze(-1).expand_as(output),
        torch.zeros_like(output),
        output,
    )
    lse = torch.where(lonely_q_mask, torch.full_like(lse, float("+inf")), lse)

    return output, lse.transpose(1, 2)
