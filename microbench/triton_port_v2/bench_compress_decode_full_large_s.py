"""v2 microbench: M2-Stage2 megakernel — large-S variant for c128 layers.

Tests the streaming-online-softmax extension (S=128) against the torch
reference at production decode shapes for DSv4 Flash-Base FP8 c128 layers.

The small-S kernel (S ≤ 16) is already validated via bench_compress_decode_full.py.
This bench gates the LARGE-S variant which fuses APE add + softmax + sum +
RMSNorm + RoPE in one launch for c128, eliminating the explicit
`score.add_(ape)` op (one of F1's targets).
"""
import os, sys
sys.path.insert(0, '/sgl-pr/python')
sys.path.insert(0, os.path.dirname(__file__))

import torch
from _framework import bench_v2
from sglang.jit_kernel.compress_decode_megakernel_triton import (
    compress_decode_full_triton, compress_decode_full_torch,
)

torch.manual_seed(0)
device = "cuda:0"


# DSv4 Flash-Base FP8 c128 layer: ratio=128, coff=1, head_dim=128 → S=128, D=128.
# bs spans the production decode histogram (max_running_request=6, c=4).
SHAPE_FREQ = {
    # (bs, S, D, rope_dim, with_ape)
    (6, 128, 128, 64, True): 200,    # bs=6 most frequent (c=4 + max_running=6)
    (4, 128, 128, 64, True): 80,
    (2, 128, 128, 64, True): 30,
    (1, 128, 128, 64, True): 50,
    (8, 128, 128, 64, True): 20,
}


def prep(shape):
    bs, S, D, rope_dim, with_ape = shape
    kv = torch.randn(bs, S, D, device=device, dtype=torch.bfloat16)
    score = torch.randn(bs, S, D, device=device, dtype=torch.bfloat16)
    ape = (
        torch.randn(S, D, device=device, dtype=torch.bfloat16)
        if with_ape else None
    )
    norm_weight = torch.randn(D, device=device, dtype=torch.float32).abs() + 0.5
    angles = torch.randn(bs, rope_dim // 2, device=device, dtype=torch.float32)
    freqs_per_bs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    out = torch.empty(bs, D, device=device, dtype=torch.bfloat16)
    return {
        "kv": kv, "score": score, "ape": ape,
        "norm_weight": norm_weight, "freqs_per_bs": freqs_per_bs,
        "eps": 1e-6, "rope_dim": rope_dim, "out": out,
    }


def torch_op(args):
    compress_decode_full_torch(
        args["kv"], args["score"], args["ape"],
        args["norm_weight"], args["freqs_per_bs"],
        args["eps"], args["rope_dim"], args["out"],
    )


def triton_op(args):
    fired = compress_decode_full_triton(
        args["kv"], args["score"], args["ape"],
        args["norm_weight"], args["freqs_per_bs"],
        args["eps"], args["rope_dim"], args["out"],
    )
    if not fired:
        compress_decode_full_torch(
            args["kv"], args["score"], args["ape"],
            args["norm_weight"], args["freqs_per_bs"],
            args["eps"], args["rope_dim"], args["out"],
        )


def get_outputs(args, op):
    args = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in args.items()}
    op(args)
    return [args["out"].detach().clone()]


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    result = bench_v2(
        name="M2-Stage2 LARGE-S compress_decode (S=128 streaming softmax)",
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
        print(f"  ✓ M2-Stage2 LARGE-S: SHIP (worst-mode speedup {result['speedup']:.2f}x)")
        print("  → Wire c128 dispatch in compress_decode_old (deepseek_v4.py)")
        print("  → Pass ape directly into megakernel for c128 (skip explicit add)")
        print("  → Run E2E aligned bench on chi2811")
    elif not result["correctness"]["all_ok"]:
        print("  ✗ M2-Stage2 LARGE-S: CORRECTNESS FAIL — DO NOT SHIP")
    else:
        print(f"  ⚠ M2-Stage2 LARGE-S: {result['verdict']} (speedup {result['speedup']:.2f}x)")
