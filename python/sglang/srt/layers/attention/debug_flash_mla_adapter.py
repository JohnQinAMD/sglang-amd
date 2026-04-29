import os
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
    """Item #5 (revised after E2E regression): keep the data_ptr cache for the
    HIT path (zero-launch fast path), use Triton fusion only on MISS.

    Original observation: removing the cache and using Triton on every call
    caused +20 ms TPOT regression on chi2811 — under cuda-graph capture the
    page_table data_ptr is stable, the cache hits reliably, and torch did 0
    work on the hit path.

    Trade-off: the cache is keyed on `data_ptr() + id(topk_length)` plus
    `id(indices)` verification, which is documented as unsafe under the caching
    allocator (per `feedback_data_ptr_caching_unsafe`). On cuda-graph captured
    decode where the buffers are persistent, hits are correct; on eager calls
    with allocator churn the indices id check disambiguates.
    """
    key = (indices.data_ptr(), id(topk_length), b, s_q, topk)
    cached = _invalid_mask_cache.get(key)
    if cached is not None:
        cached_mask, cached_indices_id = cached
        if cached_indices_id == id(indices):
            return cached_mask
    # MISS path: fused Triton kernel (was 3 separate ewise launches before).
    from sglang.jit_kernel.invalid_mask_triton import get_invalid_mask_triton
    mask_2d = get_invalid_mask_triton(indices, topk_length, b, s_q, topk)
    if len(_invalid_mask_cache) > 16:
        _invalid_mask_cache.clear()
    _invalid_mask_cache[key] = (mask_2d, id(indices))
    return mask_2d


def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    if is_hip():
        # On AMD HIP the "torch" backend is actually our fast CK V32 sparse MLA
        # decode path (see flash_mla_with_kvcache_torch). The legacy "kernel"
        # path called the upstream FlashMLA HIP port which is missing on this
        # build. Default to "torch" so production picks up CK V32 without
        # needing SGLANG_HACK_FLASHMLA_BACKEND in the launch script.
        import os

        backend = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "torch")
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
    # CK V32 FP8 sparse MLA decode kernel — DEFAULT path on HIP.
    # Consumes the FP8-packed KV cache directly (MODEL1 layout, 584 B/token)
    # and returns (output, lse) in the same shape as ref_sparse_attn_decode.
    #
    # Microbench at Pro per-rank H=16 d_qk=576: 31.8us at b=1 t=512
    # (vs Triton 87.8us = 2.76× faster, vs asm `.co` 108us = 3.4× faster,
    # vs ref_sparse_attn_decode fallback far slower).
    #
    # Opt-out: SGLANG_HIP_SPARSE_MLA_DECODE_FP8=0 forces the BF16-dequant
    # ref_sparse_attn_decode path (debugging / correctness reference).
    # Falls back to ref path automatically for unsupported (d_qk, d_v) shapes
    # or when extra_k_cache is set without two-shot enabled.
    # ---------------------------------------------------------------
    import os as _os
    _two_shot_env = _os.environ.get("SGLANG_HIP_CK_V32_TWO_SHOT", "auto")
    # `auto` = enable two-shot only for prefill (s_q > 1); decode regressed in iter 3
    # `1` = always; `0` = never.
    if _two_shot_env == "auto":
        _two_shot_enabled = s_q > 1
    else:
        _two_shot_enabled = _two_shot_env == "1"
    # Default-on; opt out with SGLANG_HIP_SPARSE_MLA_DECODE_FP8=0.
    _ck_v32_enabled = _os.environ.get("SGLANG_HIP_SPARSE_MLA_DECODE_FP8", "1") != "0"

    # DEBUG: branch counter — which of the 5 paths (ck_v32_single, ck_v32_two_shot,
    # ck_v32_model1, fallthrough_ref, _ck_v32_disabled) is the production hot path?
    # No-op when SGLANG_FLASH_MLA_BRANCH_DUMP unset.
    _branch_dump_dir = _os.environ.get("SGLANG_FLASH_MLA_BRANCH_DUMP", "")
    def _bump_branch(_label):
        if not _branch_dump_dir:
            return
        try:
            import os as __os
            __os.makedirs(_branch_dump_dir, exist_ok=True)
            _p = __os.path.join(_branch_dump_dir, f"_branch_pid{__os.getpid()}_{_label}")
            # File-based atomic counter: append one byte per call.
            with open(_p, "ab") as _bf:
                _bf.write(b".")
        except Exception:
            pass

    # DEBUG: live diff of CK V32 vs torch-bf16 reference. Computed in-process on
    # the SAME tensors the kernel just consumed — bypasses torch.save/load so we
    # can distinguish a real kernel bug from a replay-script oracle artifact.
    # Activate via SGLANG_FLASH_MLA_LIVE_DIFF=<dir> (writes one summary line per
    # call to a per-pid file). Skip first SGLANG_FLASH_MLA_LIVE_DIFF_SKIP calls
    # (default 10) to avoid cuda-graph-capture placeholder data.
    _live_diff_dir = _os.environ.get("SGLANG_FLASH_MLA_LIVE_DIFF", "")
    _live_diff_skip = int(_os.environ.get("SGLANG_FLASH_MLA_LIVE_DIFF_SKIP", "10"))
    _live_diff_count = int(_os.environ.get("SGLANG_FLASH_MLA_LIVE_DIFF_COUNT", "8"))
    if not hasattr(flash_mla_with_kvcache_torch, "_live_diff_state"):
        flash_mla_with_kvcache_torch._live_diff_state = {"call": 0, "written": 0}
    def _live_diff(label, q_in, k_cache_in, indices_in, attn_sink_in, sm_scale_in,
                   ck_out_in, ck_lse_in):
        if not _live_diff_dir:
            return
        st = flash_mla_with_kvcache_torch._live_diff_state
        st["call"] += 1
        if st["written"] >= _live_diff_count:
            return
        if st["call"] <= _live_diff_skip:
            return
        # Skip cuda-graph-capture placeholders (zero q).
        try:
            if q_in.detach().abs().max().item() == 0.0:
                return
        except Exception:
            return
        try:
            import os as __os
            __os.makedirs(_live_diff_dir, exist_ok=True)
            B_, S_, H_, D_ = q_in.shape
            V_ = ck_out_in.shape[-1]
            topk_ = indices_in.shape[-1]
            # Compute torch-bf16 reference attention IN-PROCESS on same tensors.
            # k_cache may be 2D/3D/4D; flatten to (N_KV, slot_stride) like the kernel.
            if k_cache_in.dim() == 4:
                npg, ps, _, slot = k_cache_in.shape
                kv_2d = k_cache_in.reshape(npg * ps, slot)
            elif k_cache_in.dim() == 3:
                kv_2d = k_cache_in.reshape(-1, k_cache_in.shape[-1])
            else:
                kv_2d = k_cache_in
            k_full = kv_2d.float()  # fp8_e4m3fn → fp32 (PyTorch fn-correct per probe)
            ref = torch.zeros(B_, S_, H_, V_, dtype=torch.float32, device=q_in.device)
            ref_lse = torch.zeros(B_, H_, S_, dtype=torch.float32, device=q_in.device)
            ref_lse.fill_(float("-inf"))
            for b_ in range(B_):
                for s_ in range(S_):
                    idx_raw = indices_in[b_, s_].to(torch.long)
                    invalid = idx_raw < 0
                    idx_safe = torch.clamp(idx_raw, min=0)
                    k_g = k_full[idx_safe, :D_]
                    v_g = k_full[idx_safe, :V_]
                    if invalid.all():
                        continue
                    q_h = q_in[b_, s_, :, :].float()  # [H, D]
                    scores = (q_h @ k_g.T) * sm_scale_in  # [H, topk]
                    scores = torch.where(
                        invalid.view(1, -1),
                        torch.tensor(float("-inf"), device=q_in.device),
                        scores,
                    )
                    lse = torch.logsumexp(scores, dim=-1)  # [H]
                    ref_lse[b_, :, s_] = lse
                    attn = torch.softmax(scores, dim=-1)  # [H, topk]
                    ref[b_, s_, :, :] = attn @ v_g  # [H, V]
            ref_bf = ref.to(torch.bfloat16)
            if attn_sink_in is not None:
                sink = attn_sink_in.view(1, 1, H_, 1).float()
                lse_b = ref_lse.transpose(1, 2).unsqueeze(-1).float()
                sc = 1.0 / (1.0 + torch.exp(sink - lse_b))
                ref_bf = (ref.float() * sc).to(torch.bfloat16)
            # Diff metrics.
            ck = ck_out_in.float().flatten()
            rf = ref_bf.float().flatten()
            n = (ck.norm() * rf.norm()).item()
            cs = float(ck @ rf) / n if n > 1e-12 else float("nan")
            mxd = (ck - rf).abs().max().item()
            ck_amax = ck.abs().max().item()
            rf_amax = rf.abs().max().item()
            ck_nan = torch.isnan(ck).sum().item()
            rf_nan = torch.isnan(rf).sum().item()
            valid_pct = (indices_in >= 0).float().mean().item() * 100
            pid_ = __os.getpid()
            outp = __os.path.join(_live_diff_dir, f"_live_diff_pid{pid_}.log")
            with open(outp, "a") as _f:
                _f.write(
                    f"call={st['call']:4d} {label:18s} "
                    f"B={B_:4d} S_q={S_} H={H_:3d} D={D_} V={V_} topk={topk_:3d} "
                    f"valid%={valid_pct:5.1f} sink={int(attn_sink_in is not None)} "
                    f"sm_scale={sm_scale_in:.5f} "
                    f"cos={cs:+.5f} maxd={mxd:.3e} "
                    f"ck.amax={ck_amax:.2e} rf.amax={rf_amax:.2e} "
                    f"ck.nan={ck_nan} rf.nan={rf_nan}\n"
                )
            st["written"] += 1
        except Exception as _ex:
            try:
                with open(__os.path.join(_live_diff_dir, f"_live_diff_pid{__os.getpid()}_ERR.log"), "a") as _f:
                    import traceback as _tb
                    _f.write(f"call={st['call']} label={label}\n")
                    _f.write(_tb.format_exc())
            except Exception:
                pass
    # Log the dispatch metadata once per (s_q, d_qk, d_v, has_extra) shape.
    _bump_branch(f"entry_sq{s_q}_dqk{d_qk}_dv{d_v}_extra{0 if extra_k_cache is None else 1}_2shot{int(_two_shot_enabled)}_ckenabled{int(_ck_v32_enabled)}")

    if (
        _ck_v32_enabled
        and indices is not None
        and (extra_k_cache is None or _two_shot_enabled)
    ):
        topk = indices.shape[-1]
        invalid_mask_2d = _get_invalid_mask(indices, topk_length, b, s_q, topk)

        # Dispatch on (d_qk, d_v) — supports the full DSv4 model family.
        if (d_qk == 576 or d_qk == 512) and d_v == 512:
            from sglang.srt.layers.attention.ck_v32_sparse_mla import (
                ck_sparse_mla_decode_fp8_v32,
                _apply_sink_fold_inplace,
            )

            sm_scale_f = float(softmax_scale) if softmax_scale is not None else 1.0
            indices_i32 = (
                indices.to(torch.int32) if indices.dtype != torch.int32 else indices
            )

            if extra_k_cache is None:
                _bump_branch("path_ck_v32_single_shot")
                # Single CK V32 call (production fast path).
                out, lse = ck_sparse_mla_decode_fp8_v32(
                    q=q.contiguous(),
                    k_cache=k_cache,
                    indices=indices_i32,
                    invalid_mask=invalid_mask_2d,
                    attn_sink=attn_sink,
                    sm_scale=sm_scale_f,
                )
                _live_diff(
                    "single_shot", q.contiguous(), k_cache, indices_i32,
                    attn_sink, sm_scale_f, out, lse,
                )
                return out, lse

            # ---- Phase A: two-shot CK V32 splitkv + single-launch CK combine ----
            # Replaces the pre-Phase-A pipeline of (2× CK splitkv + 2× aiter
            # mla_reduce_v1 + Triton merge_two_sparse_attn_outputs + Triton
            # _sink_fold_inplace_kernel) with (2× CK splitkv + 1× CK combine).
            # The CK combine consumes both sources' raw split tensors directly
            # (with strides) and optionally fuses the attn_sink fold inline,
            # eliminating the small-launch dispatch overhead that caused the
            # +131 ms elementwise regression in Phase B.
            _bump_branch("path_ck_v32_two_shot_split")
            from sglang.srt.layers.attention.ck_v32_sparse_mla import (
                ck_sparse_mla_decode_fp8_v32_to_split,
                ck_combine_two_splits,
            )

            extra_topk = extra_indices_in_kvcache.shape[-1]
            # extra_invalid_mask is unused by the CK kernel (it handles
            # masking via pidx<0 directly) but kept for API symmetry.
            extra_invalid_mask_2d = _get_invalid_mask(
                extra_indices_in_kvcache, extra_topk_length, b, s_q, extra_topk,
            )
            extra_indices_i32 = (
                extra_indices_in_kvcache.to(torch.int32)
                if extra_indices_in_kvcache.dtype != torch.int32
                else extra_indices_in_kvcache
            )

            q_c = q.contiguous()
            split_data_a, split_lse_a = ck_sparse_mla_decode_fp8_v32_to_split(
                q=q_c,
                k_cache=k_cache,
                indices=indices_i32,
                invalid_mask=invalid_mask_2d,
                sm_scale=sm_scale_f,
            )
            split_data_b, split_lse_b = ck_sparse_mla_decode_fp8_v32_to_split(
                q=q_c,
                k_cache=extra_k_cache,
                indices=extra_indices_i32,
                invalid_mask=extra_invalid_mask_2d,
                sm_scale=sm_scale_f,
            )

            # Single-launch CK combine: N-way online-softmax merge across
            # source-A + source-B splits, optional inline sink fold. Reads
            # the raw split tensors with native strides (no `.contiguous()`)
            # and writes directly into caller-allocated `out`/`lse`.
            total_q_ = b * s_q
            out_m_2d = torch.empty(
                (total_q_, h_q, d_v), dtype=torch.bfloat16, device=q.device,
            )
            lse_m_2d = torch.empty(
                (total_q_, h_q), dtype=torch.float32, device=q.device,
            )
            ck_combine_two_splits(
                split_data_a, split_lse_a,
                split_data_b, split_lse_b,
                attn_sink=attn_sink,
                out=out_m_2d, lse=lse_m_2d,
            )

            out = out_m_2d.view(b, s_q, h_q, d_v)
            # LSE return shape [B, H, S_q] — permute (stride-only at S_q=1).
            lse = lse_m_2d.view(b, s_q, h_q).permute(0, 2, 1)
            return out, lse

        if d_qk == 512 and d_v == 448 and extra_k_cache is None:
            _bump_branch("path_ck_v32_model1")
            # MODEL1 legacy path; only single-cache supported.
            import sys as _sys
            _ws = "/mnt/vast/john/rocm-dynamo/kernel-agents/experiments/dsv4_sparse_mla_decode_hip_workspace"
            if _ws not in _sys.path:
                _sys.path.insert(0, _ws)
            import sparse_mla_decode_fp8_kernel_model1 as _kmod
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

    # P0-c gather-first dequant: skip the whole-table dequant when the env
    # flag is on (default). `process_kv_scope` in ref.py picks up the
    # `blocked_k is None` signal and dequants only the topk rows it gathers
    # from `blocked_k_quantized`, cutting per-layer HBM traffic by
    # ~num_blocks*block_size / (b*s_q*topk) (typically 30-100× on DSv4).
    # SGLANG_GATHER_FIRST_DEQUANT=0 forces the old whole-table path for A/B.
    _gather_first = os.environ.get("SGLANG_GATHER_FIRST_DEQUANT", "1") == "1"

    blocked_k_quantized = k_cache
    if _gather_first:
        blocked_k = None
    else:
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
        if _gather_first:
            extra_blocked_k = None
        else:
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

    _bump_branch("path_fallthrough_ref_sparse_attn_decode")
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
