"""v2 microbench: M2-Stage1 megakernel — compress_decode compression core.

Tests `(kv * softmax(score + ape, dim=1)).sum(dim=1)` fusion against the
4-launch torch baseline, under the v2 framework's full gate (correctness +
production shape distribution + eager + cuda-graph capture+replay).

Production shapes from compress_decode_old (deepseek_v4.py:1483):
    kv:    (bs, ratio*coff, head_dim)  — bs ∈ {1, 2, 4, 6, 8}, coff = 2 (overlap=True)
    score: same shape
    ape:   (ratio*coff, head_dim)
    out:   (bs, head_dim)
DSv4 Flash-Base FP8: ratio=4 (c=4 indexer), head_dim=128 → S=8, D=128.
"""
import os, sys
sys.path.insert(0, '/sgl-pr/python')
sys.path.insert(0, os.path.dirname(__file__))

import torch
from _framework import bench_v2
from sglang.jit_kernel.compress_decode_megakernel_triton import (
    compress_decode_core_triton, compress_decode_core_torch,
)

torch.manual_seed(0)
device = "cuda:0"


# ============================================================
# Production shape histogram for the compression core.
# bs values from `--cuda-graph-bs 1 2 3 4 8` in the launcher
# (sglang captures bs ∈ {1, 2, 3, 4, 8} for cuda graph).
# Decode runs at one of these bs values per step depending on
# active request count (max_running_request=6).
# Frequency proxies: bs that actually fires most under c=4
# benchmark concurrency.
# ============================================================
SHAPE_FREQ = {
    # (bs, S, D, with_ape): freq
    # The kernel gates `if S > 16: return False` (single-grid design loses at large S
    # because S=128 case is memory-bound and torch already saturates HBM). So in
    # production only the c=4 path (S=8, overlap=True, APE pre-added) hits Triton.
    # c=128 path (S=128, with_ape=True) falls back to torch (no win, no loss).
    # Histogram below reflects ONLY the path the kernel will actually fire on:
    (6, 8, 128, False): 200,    # c=4 at bs=6 (most frequent)
    (4, 8, 128, False): 80,
    (2, 8, 128, False): 30,
    (1, 8, 128, False): 50,
    (8, 8, 128, False): 20,
}


def prep(shape):
    bs, S, D, with_ape = shape
    kv = torch.randn(bs, S, D, device=device, dtype=torch.bfloat16)
    score = torch.randn(bs, S, D, device=device, dtype=torch.bfloat16)
    ape = torch.randn(S, D, device=device, dtype=torch.bfloat16) if with_ape else None
    out = torch.empty(bs, D, device=device, dtype=torch.bfloat16)
    return {"kv": kv, "score": score, "ape": ape, "out": out}


def torch_op(args):
    """Reference: 3-4 separate launches."""
    compress_decode_core_torch(args["kv"], args["score"], args["ape"], args["out"])


def triton_op(args):
    """M2-Stage1 megakernel: 1 launch (softmax + mul + sum fused; +APE if set)."""
    fired = compress_decode_core_triton(
        args["kv"], args["score"], args["ape"], args["out"]
    )
    if not fired:
        compress_decode_core_torch(args["kv"], args["score"], args["ape"], args["out"])


def get_outputs(args, op):
    """Both torch_op and triton_op write into args['out'] (in-place)."""
    # Re-use the framework's deep-copy approach to ensure both ops get fresh inputs
    args = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in args.items()}
    op(args)
    return [args["out"].detach().clone()]


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"ROCm/CUDA: {torch.version.hip or torch.version.cuda}")
    print()

    result = bench_v2(
        name="M2-Stage1 compress_decode core (APE add + softmax + mul + sum)",
        prep=prep,
        torch_op=torch_op,
        triton_op=triton_op,
        shape_freq=SHAPE_FREQ,
        hit_rate=0.0,                  # no Python-side cache
        correctness_get_outputs=get_outputs,
        correctness_atol=2e-3,         # bf16 + softmax accumulates some error
        correctness_rtol=2e-2,
        iters=2000, warmup=200,
        skip_graph=False,
    )

    print()
    print("=" * 78)
    print("LESSON / NEXT ACTION")
    print("=" * 78)
    if result["correctness"]["all_ok"] and result["speedup"] >= 1.10:
        print("  ✓ M2-Stage1 megakernel: SHIP")
        print("  → Patch compress_decode_old + default compress_decode call sites")
        print("    to call compress_decode_core_triton")
        print("  → Then run E2E aligned bench on chi2811")
        print("  → If E2E TPOT improves, proceed to M2-Stage2 (add RMSNorm + RoPE epilogue)")
    elif not result["correctness"]["all_ok"]:
        print("  ✗ M2-Stage1: CORRECTNESS FAIL — fix kernel before any further timing analysis")
    elif result["speedup"] < 0.95:
        print(f"  ✗ M2-Stage1: REGRESSION — torch is {1/result['speedup']:.2f}x faster in {result['worst_mode']}")
        print("  → 4-op fusion is not enough to amortize Triton's ~9 µs fixed overhead")
        print("  → Move to M2-Stage2 directly: add RMSNorm + RoPE in same kernel (more ops fused)")
    else:
        print(f"  ⚠ M2-Stage1: INVESTIGATE (speedup {result['speedup']:.2f}x, within noise)")
        print("  → Consider M2-Stage2 (add RMSNorm + RoPE) before deciding")
