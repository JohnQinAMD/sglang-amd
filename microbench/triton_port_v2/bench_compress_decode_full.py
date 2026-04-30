"""v2 microbench: M2-Stage2 megakernel — Stage1 + RMSNorm + RoPE.

Tests the full compression-core + epilogue fusion against the torch
baseline (7-8 launches). Same v2 framework gates: correctness +
production shapes + eager/graph-replay.
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


# Production: head_dim=128, rope_head_dim=64 (DSv4 Flash-Base).
SHAPE_FREQ = {
    # (bs, S, D, rope_dim, with_ape): freq
    (6, 8, 128, 64, False): 200,    # c=4 path at bs=6 (most frequent)
    (4, 8, 128, 64, False): 80,
    (2, 8, 128, 64, False): 30,
    (1, 8, 128, 64, False): 50,
    (8, 8, 128, 64, False): 20,
}


def prep(shape):
    bs, S, D, rope_dim, with_ape = shape
    kv = torch.randn(bs, S, D, device=device, dtype=torch.bfloat16)
    score = torch.randn(bs, S, D, device=device, dtype=torch.bfloat16)
    ape = torch.randn(S, D, device=device, dtype=torch.bfloat16) if with_ape else None
    norm_weight = torch.randn(D, device=device, dtype=torch.float32).abs() + 0.5
    # freqs_per_bs: (bs, rope_dim//2, 2) — cos/sin pairs per row
    angles = torch.randn(bs, rope_dim // 2, device=device, dtype=torch.float32)
    freqs_per_bs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    out = torch.empty(bs, D, device=device, dtype=torch.bfloat16)
    return {
        "kv": kv, "score": score, "ape": ape,
        "norm_weight": norm_weight, "freqs_per_bs": freqs_per_bs,
        "eps": 1e-6, "rope_dim": rope_dim, "out": out,
    }


def torch_op(args):
    """Reference: Stage1 + RMSNorm + RoPE (full Stage2 reference)."""
    compress_decode_full_torch(
        args["kv"], args["score"], args["ape"],
        args["norm_weight"], args["freqs_per_bs"],
        args["eps"], args["rope_dim"], args["out"],
    )


def triton_op(args):
    """M2-Stage2 megakernel: Stage1 + RMSNorm + RoPE all fused."""
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
        name="M2-Stage2 compress_decode FULL (Stage1 + RMSNorm + RoPE)",
        prep=prep, torch_op=torch_op, triton_op=triton_op,
        shape_freq=SHAPE_FREQ,
        hit_rate=0.0,
        correctness_get_outputs=get_outputs,
        correctness_atol=5e-3,         # bf16 + RMSNorm + RoPE accumulates more error
        correctness_rtol=5e-2,
        iters=2000, warmup=200,
        skip_graph=False,
    )

    print()
    print("=" * 78)
    print("LESSON / NEXT ACTION")
    print("=" * 78)
    if result["correctness"]["all_ok"] and result["speedup"] >= 1.10:
        print(f"  ✓ M2-Stage2 megakernel: SHIP (worst-mode speedup {result['speedup']:.2f}x)")
        print("  → Patch compress_decode_old to call compress_decode_full_triton")
        print("  → Run E2E aligned bench on chi2811")
    elif not result["correctness"]["all_ok"]:
        print("  ✗ M2-Stage2: CORRECTNESS FAIL")
    else:
        print(f"  ⚠ M2-Stage2: {result['verdict']} (speedup {result['speedup']:.2f}x)")
