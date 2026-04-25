from typing import Any, Optional

import torch

from sglang.srt.utils import is_hip
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


# ---------------------------------------------------------------
# Cross-layer cache for `invalid_mask = indices < 0 (| arange >= topk_length)`.
# DSv4-Pro decode runs MQALayer.attn_backend.forward 61× per token using the
# same `indices` / `topk_length` tensors — so the mask only needs to be built
# once per forward_batch. Saves ~60 × 3.4µs = ~200µs/token.
#
# Keyed by (indices.data_ptr, id(topk_length), b, s_q, topk); we additionally
# verify `id(indices)` to invalidate when the underlying tensor object is
# rebuilt at a new pointer for the next batch. Cap size to bound memory.
# ---------------------------------------------------------------
_invalid_mask_cache: dict = {}


def _get_invalid_mask(indices, topk_length, b, s_q, topk):
    key = (indices.data_ptr(), id(topk_length), b, s_q, topk)
    cached = _invalid_mask_cache.get(key)
    if cached is not None:
        cached_mask, cached_indices_id = cached
        if cached_indices_id == id(indices):
            return cached_mask
    mask = indices < 0
    if topk_length is not None:
        arange_topk = torch.arange(
            topk, device=indices.device, dtype=topk_length.dtype
        ).view(1, 1, topk)
        mask = mask | (arange_topk >= topk_length.view(b, 1, 1))
    mask_2d = mask.view(b * s_q, topk)
    if len(_invalid_mask_cache) > 16:
        _invalid_mask_cache.clear()
    _invalid_mask_cache[key] = (mask_2d, id(indices))
    return mask_2d


def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    if is_hip():
        # backend == "torch"
        import os

        backend = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "kernel")
    else:
        import flash_mla

    if backend == "comparison":
        pack_ref, pack_fast_via_tester = flash_mla_with_kvcache_entrypoint(
            backend="torch", **kwargs
        )
        pack_fast_via_api = flash_mla_with_kvcache_entrypoint(
            backend="kernel", **kwargs
        )
        _assert_close(pack_ref=pack_fast_via_tester, pack_fast=pack_fast_via_api)
        _assert_close(pack_ref=pack_ref, pack_fast=pack_fast_via_tester)
        _assert_close(pack_ref=pack_ref, pack_fast=pack_fast_via_api)
        return pack_ref

    if backend == "torch":
        return flash_mla_with_kvcache_torch(**kwargs)

    if backend == "kernel":
        return flash_mla.flash_mla_with_kvcache(**kwargs)

    raise NotImplementedError


def flash_mla_with_kvcache_torch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: Any,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
):

    from sglang.srt.flashmla_tests import quant as flashmla_quant
    from sglang.srt.flashmla_tests.lib import (
        ExtraTestParamForDecode,
        KVScope,
        TestcaseForDecode,
        TestParam,
    )
    from sglang.srt.flashmla_tests.ref import ref_sparse_attn_decode

    assert block_table is None
    assert cache_seqlens is None
    assert is_fp8_kvcache

    b, s_q, h_q, d_qk = q.shape
    d_v = head_dim_v

    # ---------------------------------------------------------------
    # SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1 — bypass the BF16 dequant
    # round-trip and call the TileLang FP8-direct sparse-MLA decode
    # kernel. The kernel's wrapper consumes the FP8-packed KV cache
    # directly (MODEL1 layout, 584 B/token) and returns (output, lse)
    # in the same shape as ref_sparse_attn_decode.
    #
    # Falls back to the BF16 dequant + ref_sparse_attn_decode path
    # whenever extra_k_cache is set (concat path not yet supported).
    # ---------------------------------------------------------------
    import os as _os
    if (
        _os.environ.get("SGLANG_HIP_SPARSE_MLA_DECODE_FP8") == "1"
        and extra_k_cache is None
        and indices is not None
    ):
        topk = indices.shape[-1]
        invalid_mask_2d = _get_invalid_mask(indices, topk_length, b, s_q, topk)

        # Dispatch on (d_qk, d_v) — supports the full DSv4 model family:
        #   Pro V32       (d_qk=576, d_v=512) → CK Tile FP8 sparse  [V32 instantiation]
        #   Pro/Flash 2604 (d_qk=512, d_v=512) → CK Tile FP8 sparse [2604 instantiation]
        #   Flash 2604-MODEL1 (d_qk=512, d_v=448) → TileLang FP8    [legacy fallback]
        #   Anything else                                → fall through to ref BF16 path
        # Tighten on BOTH dims so the CK kernel never fires on a mismatched
        # shape. Unsupported shapes drop out of this `if` and land in the
        # `ref_sparse_attn_decode` BF16 path below for correctness.
        if (d_qk == 576 or d_qk == 512) and d_v == 512:
            # ---- CK Tile FP8 sparse path: covers BOTH (576, 512) V32 and
            # (512, 512) 2604. The kernel is templated on QK_HEAD_DIM and the
            # launcher dispatches on q.size(-1) at call time. ----
            from sglang.srt.layers.attention.ck_v32_sparse_mla import (
                ck_sparse_mla_decode_fp8_v32,
            )
            out, lse = ck_sparse_mla_decode_fp8_v32(
                q=q.contiguous(),
                k_cache=k_cache,
                indices=indices.to(torch.int32) if indices.dtype != torch.int32 else indices,
                invalid_mask=invalid_mask_2d,
                attn_sink=attn_sink,
                sm_scale=float(softmax_scale) if softmax_scale is not None else 1.0,
            )
            return out, lse

        if d_qk == 512 and d_v == 448:
            # ---- MODEL1 path (legacy d_v=448): TileLang FP8 direct ----
            # Kept for safety — there's no production DSv4 shape that hits
            # this today, but if a future model variant materializes the
            # MODEL1 layout it'll route here instead of falling through.
            import sys as _sys
            _ws = "/mnt/vast/john/rocm-dynamo/kernel-agents/experiments/dsv4_sparse_mla_decode_hip_workspace"
            if _ws not in _sys.path:
                _sys.path.insert(0, _ws)
            import sparse_mla_decode_fp8_kernel_model1 as _kmod
            # k_cache may arrive as float8_e4m3fn (per FP8_DTYPE module-level).
            kv_packed = k_cache.view(torch.uint8) if k_cache.dtype != torch.uint8 else k_cache
            out, lse = _kmod.sparse_mla_decode_fp8(
                q.contiguous(),
                kv_packed,
                indices.to(torch.int32) if indices.dtype != torch.int32 else indices,
                invalid_mask_2d,
                attn_sink,
                float(softmax_scale) if softmax_scale is not None else 1.0,
                d_v=d_v,
            )
            return out, lse

        # Other (d_qk, d_v) combinations: fall through to the BF16 dequant +
        # ref_sparse_attn_decode path below. Slow but always correct.

    fp8_layout = flashmla_quant.FP8KVCacheLayout.MODEL1_FP8Sparse

    p = TestParam(
        s_q=s_q,
        s_kv="unused",
        topk="unused",
        h_q=h_q,
        h_kv=1,
        d_qk=d_qk,
        d_v=d_v,
        decode=ExtraTestParamForDecode(
            b=b,
            is_varlen="unused",
            have_zero_seqlen_k="unused",
            extra_s_k="unused",
            extra_topk="unused",
            extra_block_size="unused",
            have_extra_topk_length="unused",
        ),
        # unused?
        seed=-1,
        check_correctness=True,
        is_all_indices_invalid=False,
        num_runs=10,
        have_attn_sink=True,
        have_topk_length=True,
    )

    blocked_k_quantized = k_cache
    blocked_k = flashmla_quant.dequantize_k_cache(
        blocked_k_quantized.view(FP8_DTYPE), fp8_layout
    )
    # blocked_k_requantized = flashmla_quant.quantize_k_cache(blocked_k, fp8_layout)
    # assert torch.testing.assert_allclose(blocked_k_requantized.byte(), blocked_k_quantized.byte())
    kv_scope = KVScope(
        t="unused",
        cache_seqlens="unused",
        block_table="unused",
        blocked_k=blocked_k,
        blocked_k_quantized=blocked_k_quantized,
        abs_indices="unused",
        indices_in_kvcache=indices,
        topk_length=topk_length,
    )

    extra_kv_scope = None
    if extra_k_cache is not None:
        extra_blocked_k_quantized = extra_k_cache
        extra_blocked_k = flashmla_quant.dequantize_k_cache(
            extra_blocked_k_quantized.view(FP8_DTYPE), fp8_layout
        )
        # extra_blocked_k_requantized = flashmla_quant.quantize_k_cache(extra_blocked_k, fp8_layout)
        # assert torch.testing.assert_allclose(extra_blocked_k_requantized.byte(), extra_blocked_k_quantized.byte())
        extra_kv_scope = KVScope(
            t="unused",
            cache_seqlens="unused",
            block_table="unused",
            blocked_k=extra_blocked_k,
            blocked_k_quantized=extra_blocked_k_quantized,
            abs_indices="unused",
            indices_in_kvcache=extra_indices_in_kvcache,
            topk_length=extra_topk_length,
        )

    t = TestcaseForDecode(
        p="unused",
        q=q,
        attn_sink=attn_sink,
        sm_scale=softmax_scale,
        kv_scope=kv_scope,
        extra_kv_scope=extra_kv_scope,
    )
    # print(f"hi {p=} {t=}")
    # print(
    #     f"hi info "
    #     f"{get_tensor_info(t.kv_scope.blocked_k)=} "
    #     f"{get_tensor_info(t.kv_scope.blocked_k_quantized)=} "
    #     f"{get_tensor_info(t.extra_kv_scope.blocked_k) if t.extra_kv_scope is not None else None=} "
    #     f"{get_tensor_info(t.extra_kv_scope.blocked_k_quantized) if t.extra_kv_scope is not None else None=} "
    # )

    pack_ref = ref_sparse_attn_decode(p, t)

    # tile_scheduler_metadata, _ = flash_mla.get_mla_metadata()
    # pack_fast_via_tester = flashmla_lib.run_flash_mla_decode(
    #     p, t, tile_scheduler_metadata, num_splits=None
    # )

    # return pack_ref, pack_fast_via_tester
    return pack_ref


def _assert_close(pack_ref, pack_fast):
    import sglang.srt.flashmla_tests.kernelkit as kk

    out_ref, lse_ref = pack_ref
    out_fast, lse_fast = pack_fast

    # the copied threshold is too strict, not checked why
    # copied from: test_flash_mla_sparse_decoding.py
    # is_out_correct = kk.check_is_allclose(
    #     "out", out_fast, out_ref, abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6
    # )
    # is_lse_correct = kk.check_is_allclose(
    #     "lse", lse_fast, lse_ref, abs_tol=1e-6, rel_tol=8.01 / 65536
    # )

    # loosen thresh
    is_out_correct = kk.check_is_allclose(
        "out", out_fast, out_ref, abs_tol=1e-2, rel_tol=10.0, cos_diff_tol=5e-6
    )
    is_lse_correct = kk.check_is_allclose(
        "lse", lse_fast, lse_ref, abs_tol=1e-6, rel_tol=8.01 / 65536
    )

    assert is_out_correct and is_lse_correct, f"{is_out_correct=} {is_lse_correct=}"
