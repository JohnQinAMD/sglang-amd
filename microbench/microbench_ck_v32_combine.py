"""Microbench Phase A combine kernel — numeric correctness vs Triton ref.

Validates the new ``mla_combine_fwd_ck`` (CK Tile two-source online-softmax
merge with optional inline sink fold) against the pre-Phase-A pipeline of:

    aiter.mla_reduce_v1(split_data_a, split_lse_a) -> (out_a, lse_a)
    aiter.mla_reduce_v1(split_data_b, split_lse_b) -> (out_b, lse_b)
    merge_two_sparse_attn_outputs(out_a, lse_a, out_b, lse_b) -> (out, lse)
    _apply_sink_fold_inplace(out, lse, attn_sink)  # if present

Both should produce bit-equivalent outputs (within fp32→bf16 rounding noise).
Pass criteria: cos_sim ≥ 0.9999, max_abs_diff < 1e-2.

Run on chi2774 inside the sgl-deepseek-v4-mi35x container::

    docker exec -e CUDA_VISIBLE_DEVICES=4 sgl-deepseek-v4-mi35x bash -c '
        cd /sgl-pr && PYTHONPATH=/sgl-pr/python:$PYTHONPATH \\
        python3 microbench/microbench_ck_v32_combine.py
    '
"""
from __future__ import annotations

import os
import sys
import math
from dataclasses import dataclass

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from sglang.srt.layers.attention.ck_v32_sparse_mla import (  # noqa: E402
    _ck_native_reduce,
    ck_combine_two_splits,
)
from sglang.srt.layers.attention.sparse_mla_merge import (  # noqa: E402
    merge_two_sparse_attn_outputs,
)


# ─────────────────────────── helpers ───────────────────────────
def _make_splits(total_q: int, S: int, H: int, V: int, *,
                 lse_range=(-5.0, 5.0), invalid_frac: float = 0.0,
                 device: str = "cuda", seed: int = 42):
    """Synthesize realistic CK V32 splitkv outputs.

    split_data is fp32 [total_q, S, H, V] — pre-normalized per-split (so each
    row already represents `softmax(QK)*V / rsum_split`). For testing, we
    sample randn * 0.1 (typical post-attention magnitude) and let LSE values
    encode the per-split rmax + log(rsum) signal.

    split_lse is fp32 [total_q, S, H, 1]. Sample uniform within `lse_range`,
    then optionally set a fraction to -inf to simulate invalid splits.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    sd = (torch.randn(total_q, S, H, V, generator=g, device=device,
                      dtype=torch.float32) * 0.1)
    lo, hi = lse_range
    sl = (torch.rand(total_q, S, H, 1, generator=g, device=device,
                     dtype=torch.float32) * (hi - lo) + lo)
    if invalid_frac > 0.0:
        mask = torch.rand(total_q, S, H, 1, generator=g, device=device) < invalid_frac
        sl = sl.masked_fill(mask, float("-inf"))
    return sd, sl


def _ref_two_source(split_data_a, split_lse_a, split_data_b, split_lse_b,
                    attn_sink=None):
    """Reference path: 2× aiter.mla_reduce_v1 + Triton merge + Triton sink_fold."""
    from sglang.srt.layers.attention.ck_v32_sparse_mla import (
        _apply_sink_fold_inplace,
    )
    out_a, lse_a = _ck_native_reduce(split_data_a, split_lse_a)   # (B, H, V) bf16, (B, H) fp32
    out_b, lse_b = _ck_native_reduce(split_data_b, split_lse_b)
    out_merged, lse_merged = merge_two_sparse_attn_outputs(
        out_a.contiguous(), lse_a.contiguous(),
        out_b.contiguous(), lse_b.contiguous(),
    )
    if attn_sink is not None:
        _apply_sink_fold_inplace(out_merged, lse_merged, attn_sink)
    return out_merged, lse_merged


def _ck_combine(split_data_a, split_lse_a, split_data_b, split_lse_b,
                attn_sink=None):
    return ck_combine_two_splits(
        split_data_a, split_lse_a, split_data_b, split_lse_b,
        attn_sink=attn_sink,
    )


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().flatten()
    bf = b.float().flatten()
    n_a = float(af.norm())
    n_b = float(bf.norm())
    if n_a < 1e-12 or n_b < 1e-12:
        # Both zero → degenerate; only declare success if both are exactly 0.
        return 1.0 if (n_a < 1e-12 and n_b < 1e-12) else 0.0
    return float((af * bf).sum() / (n_a * n_b))


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


# ─────────────────────────── tests ───────────────────────────
@dataclass
class TestCase:
    name: str
    total_q: int
    H: int
    V: int
    S_a: int
    S_b: int
    invalid_a: float
    invalid_b: float
    use_sink: bool
    cos_min: float = 0.9999
    max_abs_max: float = 1e-2


CASES = [
    TestCase("decode_b1_no_sink", 1, 128, 512, 8, 4, 0.0, 0.0, False),
    TestCase("decode_b4_no_sink", 4, 128, 512, 8, 4, 0.0, 0.0, False),
    TestCase("decode_b6_no_sink", 6, 128, 512, 8, 4, 0.0, 0.0, False),
    TestCase("decode_b4_with_sink", 4, 128, 512, 8, 4, 0.0, 0.0, True),
    TestCase("decode_b4_partial_invalid_a", 4, 128, 512, 8, 4, 0.25, 0.0, False),
    TestCase("decode_b4_partial_invalid_both", 4, 128, 512, 8, 4, 0.25, 0.25, True),
    TestCase("decode_b4_S_a_only", 4, 128, 512, 16, 0, 0.0, 0.0, False,
             cos_min=0.9999, max_abs_max=1e-2),  # source-B == None edge case; tested separately
]


def _run_one(tc: TestCase) -> bool:
    torch.cuda.synchronize()
    if tc.S_b == 0:
        # Single-source path — separate test below; skip here.
        return _run_single_source(tc)

    sd_a, sl_a = _make_splits(tc.total_q, tc.S_a, tc.H, tc.V,
                              invalid_frac=tc.invalid_a, seed=42)
    sd_b, sl_b = _make_splits(tc.total_q, tc.S_b, tc.H, tc.V,
                              invalid_frac=tc.invalid_b, seed=43)

    sink = None
    if tc.use_sink:
        sink = torch.randn(tc.H, device="cuda", dtype=torch.float32) * 0.5

    # Need fresh copies for the reference path because mla_reduce_v1 may
    # modify the inputs in place (it doesn't, but be safe).
    sd_a_ref = sd_a.clone()
    sl_a_ref = sl_a.clone()
    sd_b_ref = sd_b.clone()
    sl_b_ref = sl_b.clone()

    ref_out, ref_lse = _ref_two_source(sd_a_ref, sl_a_ref, sd_b_ref, sl_b_ref,
                                        attn_sink=sink)
    ck_out, ck_lse = _ck_combine(sd_a, sl_a, sd_b, sl_b, attn_sink=sink)

    cos_out = _cos(ref_out, ck_out)
    max_out = _max_abs_diff(ref_out, ck_out)
    cos_lse = _cos(ref_lse, ck_lse)
    max_lse = _max_abs_diff(ref_lse, ck_lse)

    # LSE comparison is delicate when most entries are -inf (all-invalid).
    # Replace -inf with a finite large-negative for cos_sim purposes.
    if not math.isfinite(cos_lse):
        ref_lse_f = ref_lse.clone()
        ck_lse_f = ck_lse.clone()
        ref_lse_f[~torch.isfinite(ref_lse_f)] = -1e6
        ck_lse_f[~torch.isfinite(ck_lse_f)] = -1e6
        cos_lse = _cos(ref_lse_f, ck_lse_f)
        max_lse = _max_abs_diff(ref_lse_f, ck_lse_f)

    ok_out_cos = cos_out >= tc.cos_min
    ok_out_max = max_out <= tc.max_abs_max
    ok_lse_cos = cos_lse >= tc.cos_min
    ok_lse_max = max_lse <= 1e-2  # lse is fp32, looser bound
    ok = ok_out_cos and ok_out_max and ok_lse_cos and ok_lse_max
    print(f"  [{tc.name:<32s}] "
          f"cos_out={cos_out:.6f} max_out={max_out:.4e} "
          f"cos_lse={cos_lse:.6f} max_lse={max_lse:.4e}  "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        if not ok_out_cos:
            print(f"    ! cos_out {cos_out:.6f} < {tc.cos_min}")
        if not ok_out_max:
            print(f"    ! max_out {max_out:.4e} > {tc.max_abs_max:.0e}")
        if not ok_lse_cos:
            print(f"    ! cos_lse {cos_lse:.6f} < {tc.cos_min}")
        if not ok_lse_max:
            print(f"    ! max_lse {max_lse:.4e} > 1e-2")
    return ok


def _run_single_source(tc: TestCase) -> bool:
    """For S_b=0: combine should reduce source-A only (equivalent to
    aiter.mla_reduce_v1).  We currently exercise it indirectly via the
    two-source path with synthetic source-B that's all-invalid; here we
    use the ck_combine_two_splits API with split_data_b=None."""
    sd_a, sl_a = _make_splits(tc.total_q, tc.S_a, tc.H, tc.V,
                              invalid_frac=tc.invalid_a, seed=42)
    ref_out, ref_lse = _ck_native_reduce(sd_a.clone(), sl_a.clone())
    ck_out, ck_lse = ck_combine_two_splits(sd_a, sl_a, None, None, attn_sink=None)

    cos_out = _cos(ref_out, ck_out)
    max_out = _max_abs_diff(ref_out, ck_out)
    cos_lse = _cos(ref_lse, ck_lse)
    max_lse = _max_abs_diff(ref_lse, ck_lse)
    ok = (cos_out >= tc.cos_min and max_out <= tc.max_abs_max
          and cos_lse >= tc.cos_min and max_lse <= 1e-2)
    print(f"  [{tc.name:<32s}] "
          f"cos_out={cos_out:.6f} max_out={max_out:.4e} "
          f"cos_lse={cos_lse:.6f} max_lse={max_lse:.4e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


# ─────────────────────────── perf microbench ───────────────────────────
def _timed(fn, n_warm=10, n_iter=200):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    us = sorted((s.elapsed_time(e) * 1e3) for s, e in zip(starts, ends))
    return us[n_iter // 2]


def _run_perf():
    print()
    print("=== perf microbench (production shape: total_q=4, H=128, V=512, S_a=8, S_b=4) ===")
    sd_a, sl_a = _make_splits(4, 8, 128, 512, seed=42)
    sd_b, sl_b = _make_splits(4, 4, 128, 512, seed=43)
    sink = torch.randn(128, device="cuda", dtype=torch.float32) * 0.5

    out_buf = torch.empty((4, 128, 512), dtype=torch.bfloat16, device="cuda")
    lse_buf = torch.empty((4, 128), dtype=torch.float32, device="cuda")

    def ref_path():
        _ref_two_source(sd_a, sl_a, sd_b, sl_b, attn_sink=sink)

    def ck_path():
        ck_combine_two_splits(sd_a, sl_a, sd_b, sl_b, attn_sink=sink,
                              out=out_buf, lse=lse_buf)

    us_ref = _timed(ref_path)
    us_ck  = _timed(ck_path)
    speedup = us_ref / us_ck if us_ck > 0 else float("inf")
    print(f"  reference (mla_reduce_v1 × 2 + Triton merge + sink): {us_ref:7.1f} µs/call")
    print(f"  CK combine (single launch, sink fused):              {us_ck:7.1f} µs/call")
    print(f"  speedup: {speedup:.2f}×")
    print(f"  target:  ≤15 µs  →  {'PASS' if us_ck <= 15.0 else 'check'}")


# ─────────────────────────── main ───────────────────────────
def main():
    if not torch.cuda.is_available():
        print("CUDA not available — exiting.")
        return 1
    print(f"# device: {torch.cuda.get_device_name(0)}")
    print()
    print("=== numeric correctness (CK combine vs mla_reduce_v1 + Triton merge) ===")
    fail = 0
    for tc in CASES:
        if not _run_one(tc):
            fail += 1

    _run_perf()

    print()
    if fail == 0:
        print("ALL NUMERIC PASS.")
        return 0
    print(f"{fail} test(s) FAILED.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
