"""v2 microbench: Triton port of mhc_post_tilelang.

Validates correctness vs torch reference + cuda-graph speedup at production
shape histogram (DSv4 Flash-Base FP8 decode batch=6).
"""
import os, sys
sys.path.insert(0, '/sgl-pr/python')
sys.path.insert(0, os.path.dirname(__file__))

import torch
from _framework import bench_v2
from sglang.jit_kernel.mhc_post_triton import (
    mhc_post_triton, mhc_post_torch,
)

torch.manual_seed(0)
device = "cuda:0"


# Production shapes for Flash-Base FP8 (TP=4, max_running_request=6, ISL=OSL=1024).
# Decode batch is `n` per layer-pass; 60 layers; chunked-prefill 8192 tokens.
# Decode hot path: n in {1,2,4,6,8} most common.
SHAPE_FREQ = {
    # (n, hc, h)
    (6, 4, 2048): 200,   # decode at concurrency=4, max_running=6 → 6 most common
    (4, 4, 2048): 80,
    (2, 4, 2048): 30,
    (1, 4, 2048): 50,
    (8, 4, 2048): 20,
}


def prep(shape):
    n, hc, h = shape
    # sglang convention: x=(n,h), residual=(n,hc,h)
    x = torch.randn(n, h, device=device, dtype=torch.bfloat16)
    residual = torch.randn(n, hc, h, device=device, dtype=torch.bfloat16)
    post_layer_mix = torch.randn(n, hc, device=device, dtype=torch.float32)
    comb_res_mix = torch.randn(n, hc, hc, device=device, dtype=torch.float32)
    out = torch.empty(n, hc, h, device=device, dtype=torch.bfloat16)
    return {
        "x": x, "residual": residual,
        "post_layer_mix": post_layer_mix, "comb_res_mix": comb_res_mix,
        "out": out,
    }


def torch_op(args):
    mhc_post_torch(args["x"], args["residual"], args["post_layer_mix"],
                    args["comb_res_mix"], args["out"])


def triton_op(args):
    fired = mhc_post_triton(args["x"], args["residual"], args["post_layer_mix"],
                             args["comb_res_mix"], args["out"])
    if not fired:
        mhc_post_torch(args["x"], args["residual"], args["post_layer_mix"],
                        args["comb_res_mix"], args["out"])


def get_outputs(args, op):
    args = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in args.items()}
    op(args)
    return [args["out"].detach().clone()]


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    result = bench_v2(
        name="mhc_post (replacing TileLang regressed kernel)",
        prep=prep, torch_op=torch_op, triton_op=triton_op,
        shape_freq=SHAPE_FREQ,
        hit_rate=0.0,
        correctness_get_outputs=get_outputs,
        correctness_atol=5e-3,
        correctness_rtol=5e-2,
        iters=2000, warmup=200,
        skip_graph=False,
    )

    print()
    print("=" * 78)
    print("LESSON / NEXT ACTION")
    print("=" * 78)
    if result["correctness"]["all_ok"] and result["speedup"] >= 1.10:
        print(f"  ✓ mhc_post Triton: SHIP (worst-mode speedup {result['speedup']:.2f}x)")
        print("  → Add SGLANG_MHC_USE_TRITON=1 to launch_dsv4.sh stacked-best preset")
        print("  → Run E2E aligned bench on chi2811")
    elif not result["correctness"]["all_ok"]:
        print("  ✗ mhc_post Triton: CORRECTNESS FAIL — DO NOT SHIP")
    else:
        print(f"  ⚠ mhc_post Triton: {result['verdict']} (speedup {result['speedup']:.2f}x)")
