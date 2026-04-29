"""v2 microbench: Triton port of mhc_pre TileLang stages.

Validates the 3-stage Triton pipeline (gemm_sqrsum + big_fuse + apply_mix)
against a torch reference, at the production decode shape histogram for
DSv4 Flash-Base FP8.
"""
import os, sys
sys.path.insert(0, '/sgl-pr/python')
sys.path.insert(0, os.path.dirname(__file__))

import torch
from _framework import bench_v2
from sglang.jit_kernel.mhc_pre_triton import (
    mhc_pre_triton, mhc_pre_torch,
)

torch.manual_seed(0)
device = "cuda:0"


SHAPE_FREQ = {
    # (n, hc, hidden) — Flash-Base FP8 hc=4, hidden=2048
    (6, 4, 2048): 200,
    (4, 4, 2048): 80,
    (2, 4, 2048): 30,
    (1, 4, 2048): 50,
    (8, 4, 2048): 20,
}

# Constants matching DSv4 production (sourced from deepseek_v4 default config).
RMS_EPS = 1e-6
HC_PRE_EPS = 1e-6
HC_SINKHORN_EPS = 1e-6
HC_POST_MULT_VALUE = 2.0
SINKHORN_REPEAT = 3


def prep(shape):
    n, hc, hidden = shape
    hc_mult3 = hc * (2 + hc)  # 24 for hc=4
    hc_hidden = hc * hidden
    residual = torch.randn(n, hc, hidden, device=device, dtype=torch.bfloat16)
    fn = torch.randn(hc_mult3, hc_hidden, device=device, dtype=torch.float32)
    hc_scale = torch.randn(3, device=device, dtype=torch.float32).abs() + 0.1
    hc_base = torch.randn(hc_mult3, device=device, dtype=torch.float32) * 0.1
    return {
        "residual": residual, "fn": fn, "hc_scale": hc_scale, "hc_base": hc_base,
    }


def _common_args(args):
    return dict(
        residual=args["residual"], fn=args["fn"],
        hc_scale=args["hc_scale"], hc_base=args["hc_base"],
        rms_eps=RMS_EPS, hc_pre_eps=HC_PRE_EPS,
        hc_sinkhorn_eps=HC_SINKHORN_EPS,
        hc_post_mult_value=HC_POST_MULT_VALUE,
        sinkhorn_repeat=SINKHORN_REPEAT,
    )


def torch_op(args):
    args["_out"] = mhc_pre_torch(**_common_args(args))


def triton_op(args):
    result = mhc_pre_triton(**_common_args(args))
    if result is None:
        result = mhc_pre_torch(**_common_args(args))
    args["_out"] = result


def get_outputs(args, op):
    args = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in args.items()}
    op(args)
    post_mix, comb_mix, layer_input = args["_out"]
    return [post_mix.detach().clone(), comb_mix.detach().clone(),
            layer_input.detach().clone()]


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    result = bench_v2(
        name="mhc_pre (3-stage Triton replacing TileLang)",
        prep=prep, torch_op=torch_op, triton_op=triton_op,
        shape_freq=SHAPE_FREQ,
        hit_rate=0.0,
        correctness_get_outputs=get_outputs,
        correctness_atol=1e-2,         # multi-stage (gemm + softmax + sinkhorn) accumulates
        correctness_rtol=1e-1,
        iters=2000, warmup=200,
        skip_graph=False,
    )

    print()
    print("=" * 78)
    print("LESSON / NEXT ACTION")
    print("=" * 78)
    if result["correctness"]["all_ok"] and result["speedup"] >= 1.10:
        print(f"  ✓ mhc_pre Triton: SHIP (worst-mode speedup {result['speedup']:.2f}x)")
        print("  → Add SGLANG_MHC_USE_TRITON=1 to launch_dsv4.sh stacked-best preset")
    elif not result["correctness"]["all_ok"]:
        print("  ✗ mhc_pre Triton: CORRECTNESS FAIL — DO NOT SHIP")
    else:
        print(f"  ⚠ mhc_pre Triton: {result['verdict']} (speedup {result['speedup']:.2f}x)")
