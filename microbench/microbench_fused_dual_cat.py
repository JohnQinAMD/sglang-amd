"""Microbench + correctness for fused_dual_cat (T4 fusion).

Tests against torch.cat reference for the production sparse-decode shapes:
  Flash-Base FP8 c=4 decode: b=4, s_q=1, topk_a=512, topk_b=2 (extra c128), d=512

Runs:
  1. Correctness: bit-exact match vs torch.cat for several shapes
  2. Eager perf: time per call vs torch.cat baseline
  3. Cuda-graph perf: capture+replay timing
"""

import torch
import triton

import sys
sys.path.insert(0, "/sgl-pr/python")

from sglang.jit_kernel.fused_dual_cat_triton import fused_dual_cat


def torch_baseline(kv_a, kv_b, mask_a, mask_b):
    kv = torch.cat([kv_a, kv_b], dim=2)
    mask = torch.cat([mask_a, mask_b], dim=2)
    return kv, mask


def correctness_check(b, s_q, topk_a, topk_b, d, dtype=torch.bfloat16, device="cuda"):
    torch.manual_seed(0)
    kv_a = torch.randn(b, s_q, topk_a, d, dtype=dtype, device=device)
    kv_b = torch.randn(b, s_q, topk_b, d, dtype=dtype, device=device)
    mask_a = torch.randint(0, 2, (b, s_q, topk_a), dtype=torch.bool, device=device)
    mask_b = torch.randint(0, 2, (b, s_q, topk_b), dtype=torch.bool, device=device)

    kv_ref, mask_ref = torch_baseline(kv_a, kv_b, mask_a, mask_b)
    kv_out, mask_out = fused_dual_cat(kv_a, kv_b, mask_a, mask_b)

    kv_match = torch.equal(kv_ref, kv_out)
    mask_match = torch.equal(mask_ref, mask_out)

    if not kv_match:
        max_diff = (kv_ref.float() - kv_out.float()).abs().max().item()
        print(f"  KV MISMATCH: max_abs_diff={max_diff}")
    if not mask_match:
        n_diff = (mask_ref != mask_out).sum().item()
        print(f"  MASK MISMATCH: {n_diff} elements differ")

    return kv_match and mask_match


def time_eager(fn, args, n_warmup=10, n_iter=100):
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter * 1000  # us


def time_cudagraph(fn, args, n_warmup=10, n_iter=100):
    """Capture cudagraph then time replays."""
    # Warmup
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter * 1000  # us


def main():
    print("=" * 70)
    print("Correctness checks:")
    print("=" * 70)
    shapes = [
        # (b, s_q, topk_a, topk_b, d) — production = (4, 1, 512, 2, 512)
        (1, 1, 512, 2, 512),
        (4, 1, 512, 2, 512),  # c=4 production
        (8, 1, 512, 2, 512),
        (4, 1, 256, 2, 512),
        (4, 1, 1024, 2, 512),
        (1, 1, 512, 64, 576),  # Pro mode
    ]
    all_pass = True
    for shape in shapes:
        ok = correctness_check(*shape)
        status = "PASS" if ok else "FAIL"
        print(f"  shape={shape}: {status}")
        all_pass &= ok
    print()
    if not all_pass:
        print("CORRECTNESS FAILED — aborting perf tests")
        return

    print("=" * 70)
    print("Eager + cuda-graph perf:")
    print("=" * 70)
    print(f"{'shape':30s}  {'eager_torch_us':>14s}  {'eager_fused_us':>14s}  {'graph_torch_us':>14s}  {'graph_fused_us':>14s}  {'graph speedup':>14s}")
    for shape in shapes:
        b, s_q, topk_a, topk_b, d = shape
        kv_a = torch.randn(b, s_q, topk_a, d, dtype=torch.bfloat16, device="cuda")
        kv_b = torch.randn(b, s_q, topk_b, d, dtype=torch.bfloat16, device="cuda")
        mask_a = torch.randint(0, 2, (b, s_q, topk_a), dtype=torch.bool, device="cuda")
        mask_b = torch.randint(0, 2, (b, s_q, topk_b), dtype=torch.bool, device="cuda")
        args = (kv_a, kv_b, mask_a, mask_b)

        t_torch_eager = time_eager(torch_baseline, args)
        t_fused_eager = time_eager(fused_dual_cat, args)
        t_torch_graph = time_cudagraph(torch_baseline, args)
        t_fused_graph = time_cudagraph(fused_dual_cat, args)
        speedup = t_torch_graph / t_fused_graph if t_fused_graph > 0 else float('inf')
        print(f"{str(shape):30s}  {t_torch_eager:>14.2f}  {t_fused_eager:>14.2f}  {t_torch_graph:>14.2f}  {t_fused_graph:>14.2f}  {speedup:>13.2f}x")


if __name__ == "__main__":
    main()
