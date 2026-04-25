"""Parity tests for CK Tile FP8 sparse MLA decode (V32 / DSv4-Pro).

Validates the gfx950 CK Tile kernel at sglang/srt/layers/attention/ck_v32_sparse_mla.py
against the FP32 oracle (`_sparse_attn_decode_inner` from
sglang.srt.flashmla_tests.ref). Acceptance: cos_sim > 0.999, max_rel < 5%.

Skipped if not running on AMD ROCm gfx950, or if `aiter` is not importable
(the kernel uses `aiter.mla_reduce_v1` for the stage-2 reduce).

Run from the SGLang repo root:
    pytest test/srt/test_ck_v32_sparse_mla.py -v
"""
from __future__ import annotations

import math
import os

import pytest
import torch


def _is_gfx950() -> bool:
    """True iff CUDA is HIP and the device is gfx950."""
    if not torch.cuda.is_available():
        return False
    if not (torch.version.hip and torch.version.hip != ""):
        return False
    try:
        props = torch.cuda.get_device_properties(0)
    except Exception:
        return False
    arch = getattr(props, "gcnArchName", "")
    return arch.startswith("gfx950")


def _aiter_available() -> bool:
    try:
        import aiter  # noqa: F401
        return hasattr(aiter, "mla_reduce_v1")
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _is_gfx950(), reason="CK V32 sparse MLA requires gfx950"),
    pytest.mark.skipif(not _aiter_available(),
                       reason="aiter (with mla_reduce_v1) not available"),
]


_SHAPES = [
    # (B, H, topk)  — primary V32 decode shapes; covers low/mid/high topk × low/mid B.
    (1, 128, 256),
    (1, 128, 512),
    (1, 128, 1024),
    (1, 128, 2048),
    (4, 128, 512),
    (4, 128, 1024),
    (8, 128, 512),
]

# (d_qk, d_v) supported by the templated CK kernel:
#   (576, 512) — V32 / Pro V32 mode (d_qk = nope=512 + rope=64)
#   (512, 512) — 2604 / Flash-Base 2604 mode (no rope dim)
_DQK_DV = [
    (576, 512),
    (512, 512),
]


def _run_one(B: int, H: int, topk: int, d_qk: int, d_v: int, seed: int = 0):
    """Run CK kernel and FP32 oracle. Returns (cos_sim, max_rel_err)."""
    from sglang.srt.flashmla_tests.ref import _sparse_attn_decode_inner
    from sglang.srt.layers.attention.ck_v32_sparse_mla import (
        ck_sparse_mla_decode_fp8_v32,
    )

    torch.manual_seed(seed)
    device = torch.device("cuda")

    S_q = 1
    n_kv_pool = max(topk * 2, 1024)
    sm_scale = 1.0 / math.sqrt(d_qk)
    fp8 = torch.float8_e4m3fnuz

    q_bf16 = torch.randn((B, S_q, H, d_qk), device=device).to(torch.bfloat16)
    kv_pool_bf16 = torch.randn((n_kv_pool, d_qk), device=device).to(torch.bfloat16)
    kv_pool_fp8 = kv_pool_bf16.to(fp8).contiguous()
    k_cache = kv_pool_fp8.view(n_kv_pool, 1, 1, d_qk)

    indices = (
        torch.arange(topk, device=device, dtype=torch.int32)
        .view(1, 1, topk).expand(B, S_q, topk).contiguous()
    )
    invalid_mask = torch.zeros((B * S_q, topk), dtype=torch.bool, device=device)

    out_ck, _ = ck_sparse_mla_decode_fp8_v32(
        q=q_bf16,
        k_cache=k_cache,
        indices=indices,
        invalid_mask=invalid_mask,
        attn_sink=None,
        sm_scale=sm_scale,
    )
    torch.cuda.synchronize()

    # FP32 oracle on FP8-roundtripped KV. The oracle's d_v slices the last
    # d_v cols out of the gathered K rows for the V-projection.
    q_round = q_bf16.float().view(B * S_q, H, d_qk)
    gathered = kv_pool_fp8.float()[indices.long().view(-1)].view(B * S_q, topk, d_qk)
    out_ref, _ = _sparse_attn_decode_inner(
        q_round, gathered, invalid_mask, None,
        float(sm_scale), int(d_v), int(H),
    )

    out_ck_f = out_ck.float().reshape(-1)
    out_ref_f = out_ref.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(
        out_ref_f.unsqueeze(0), out_ck_f.unsqueeze(0)
    ).item()
    diff = (out_ref_f - out_ck_f).abs().max().item()
    ref_amx = out_ref_f.abs().max().item()
    rel = diff / max(ref_amx, 1e-9)
    return cos, rel, torch.isnan(out_ck).any().item(), torch.isinf(out_ck).any().item()


@pytest.mark.parametrize("d_qk,d_v", _DQK_DV)
@pytest.mark.parametrize("B,H,topk", _SHAPES)
def test_ck_v32_parity(B: int, H: int, topk: int, d_qk: int, d_v: int):
    cos, rel, has_nan, has_inf = _run_one(B, H, topk, d_qk, d_v)
    tag = f"B={B} H={H} topk={topk} d_qk={d_qk} d_v={d_v}"
    assert not has_nan, f"output contains NaN at {tag}"
    assert not has_inf, f"output contains Inf at {tag}"
    assert cos > 0.999, f"cos_sim={cos:.6f} < 0.999 at {tag}"
    assert rel < 0.05, f"rel_err={rel:.4e} >= 5% at {tag}"


def test_ck_v32_lonely_q_zeros():
    """Lonely-Q queries (all-invalid indices) should produce exactly zero
    output without explicit wrapper correction — the kernel zero-fills
    `pidx < 0` rows, naturally yielding Q@K=0 and P@V=0."""
    from sglang.srt.layers.attention.ck_v32_sparse_mla import (
        ck_sparse_mla_decode_fp8_v32,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, H, topk = 1, 128, 512
    D_TOTAL = 576
    fp8 = torch.float8_e4m3fnuz

    q_bf16 = torch.randn((B, 1, H, D_TOTAL), device=device).to(torch.bfloat16)
    kv_pool_fp8 = torch.randn((1024, D_TOTAL), device=device).to(torch.bfloat16).to(fp8).contiguous()
    k_cache = kv_pool_fp8.view(1024, 1, 1, D_TOTAL)

    # All -1 → all invalid
    indices = torch.full((B, 1, topk), -1, dtype=torch.int32, device=device)
    invalid_mask = torch.ones((B, topk), dtype=torch.bool, device=device)

    out, lse = ck_sparse_mla_decode_fp8_v32(
        q=q_bf16, k_cache=k_cache, indices=indices,
        invalid_mask=invalid_mask, attn_sink=None,
        sm_scale=1.0 / math.sqrt(D_TOTAL),
    )
    torch.cuda.synchronize()
    assert (out == 0).all().item(), \
        f"lonely-Q output should be all zeros, got absmax={out.float().abs().max().item():.4e}"


def test_ck_v32_attn_sink_correction():
    """Attention sink modifies output by `1 / (1 + exp(sink - lse))`. Verify
    the wrapper applies it correctly by comparing sink=0 (no-op) to sink=very
    negative (saturates to 1, no effect)."""
    from sglang.srt.layers.attention.ck_v32_sparse_mla import (
        ck_sparse_mla_decode_fp8_v32,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    B, H, topk = 1, 128, 512
    D_TOTAL = 576
    fp8 = torch.float8_e4m3fnuz

    q_bf16 = torch.randn((B, 1, H, D_TOTAL), device=device).to(torch.bfloat16)
    kv_pool_fp8 = torch.randn((1024, D_TOTAL), device=device).to(torch.bfloat16).to(fp8).contiguous()
    k_cache = kv_pool_fp8.view(1024, 1, 1, D_TOTAL)
    indices = (
        torch.arange(topk, device=device, dtype=torch.int32)
        .view(1, 1, topk).expand(B, 1, topk).contiguous()
    )
    invalid_mask = torch.zeros((B, topk), dtype=torch.bool, device=device)
    sm_scale = 1.0 / math.sqrt(D_TOTAL)

    sink_neg = torch.full((H,), -1e30, dtype=torch.float32, device=device)
    out_no_sink, _ = ck_sparse_mla_decode_fp8_v32(
        q_bf16, k_cache, indices, invalid_mask, attn_sink=None, sm_scale=sm_scale,
    )
    out_neg_sink, _ = ck_sparse_mla_decode_fp8_v32(
        q_bf16, k_cache, indices, invalid_mask, attn_sink=sink_neg, sm_scale=sm_scale,
    )
    torch.cuda.synchronize()

    # very-negative sink → 1/(1+exp(-inf)) = 1 → identity
    diff = (out_no_sink.float() - out_neg_sink.float()).abs().max().item()
    assert diff < 1e-2, f"sink=-inf should be no-op, got diff={diff:.4e}"


if __name__ == "__main__":
    # Also runnable as a script for quick repro outside pytest
    for d_qk, d_v in _DQK_DV:
        for B, H, topk in _SHAPES:
            cos, rel, has_nan, has_inf = _run_one(B, H, topk, d_qk, d_v)
            ok = (cos > 0.999 and rel < 0.05 and not has_nan and not has_inf)
            print(f"d_qk={d_qk} d_v={d_v} B={B} H={H} topk={topk}  "
                  f"cos_sim={cos:.6f} rel_err={rel:.3e} "
                  f"nan={has_nan} inf={has_inf}  {'OK' if ok else 'FAIL'}")
