"""v2 microbench: hc_pre_decode_triton vs _hc_pre_torch_impl on Flash decode shapes.

Decode-shape-specific replacement for the Pro-tuned hc_pre_fused_triton kernel
(which regressed +60 ms TPOT at decode shapes — see hc-pre-fallback-finding-2026-05-01.md).

GATES (per `feedback_microbench_methodology_traps.md` and `reference_microbench_v2_framework.md`):
- (G0) Correctness vs torch reference on production shapes — HARD GATE
- (G1) Eager + cuda-graph capture+replay timing — graph-replay is the deciding metric
"""
import sys
import time

sys.path.insert(0, "/sgl-pr/python")

import torch

from sglang.jit_kernel.hc_pre_decode_triton import (
    _hc_pre_decode_reference,
    hc_pre_decode_triton,
)


def make_inputs(M, HC_MULT_IN, HC_DIM, HC_MULT_OUT, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    HIDDEN = HC_MULT_IN * HC_DIM
    device = torch.device("cuda:0")
    x = torch.randn(M, HC_MULT_IN, HC_DIM, dtype=dtype, device=device) * 0.1
    hc_fn = torch.randn(HC_MULT_OUT, HIDDEN, dtype=torch.float32, device=device) * 0.05
    return x, hc_fn


def correctness(name, M, HC_MULT_IN, HC_DIM, HC_MULT_OUT):
    print(f"\n=== {name}: M={M}, HC_MULT_IN={HC_MULT_IN}, HC_DIM={HC_DIM}, HC_MULT_OUT={HC_MULT_OUT} ===")
    x, hc_fn = make_inputs(M, HC_MULT_IN, HC_DIM, HC_MULT_OUT)
    eps = 1e-6

    x_flat_ref, mixes_ref = _hc_pre_decode_reference(x, hc_fn, eps)
    x_flat_t, mixes_t = hc_pre_decode_triton(x, hc_fn, eps)

    print(f"  ref     x_flat.shape={tuple(x_flat_ref.shape)} mixes.shape={tuple(mixes_ref.shape)}")
    print(f"  triton  x_flat.shape={tuple(x_flat_t.shape)} mixes.shape={tuple(mixes_t.shape)}")

    assert x_flat_t.shape == x_flat_ref.shape
    assert mixes_t.shape == mixes_ref.shape

    diff_x = (x_flat_t - x_flat_ref).abs()
    diff_m = (mixes_t - mixes_ref).abs()
    rel_m = (diff_m / mixes_ref.abs().clamp_min(1e-6))
    cos_m = torch.nn.functional.cosine_similarity(
        mixes_t.flatten().unsqueeze(0), mixes_ref.flatten().unsqueeze(0)
    ).item()

    print(f"  max|x_flat diff| = {diff_x.max().item():.2e}")
    print(f"  max|mixes diff|  = {diff_m.max().item():.2e}")
    print(f"  max relative err = {rel_m.max().item():.2e}")
    print(f"  cos sim          = {cos_m:.6f}")

    # bf16 / fp32 reduction floor
    PASS = (
        diff_x.max().item() <= 1e-4
        and cos_m >= 0.999
        and rel_m.max().item() <= 5e-3
    )
    print(f"  RESULT: {'PASS' if PASS else 'FAIL'}")
    return PASS


def time_eager(fn, *args, iters=50, warmup=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def time_graph(fn, args, iters=50):
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for _ in range(iters):
                fn(*args)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def perf_compare(name, M, HC_MULT_IN, HC_DIM, HC_MULT_OUT):
    print(f"\n--- perf {name}: M={M} HC_OUT={HC_MULT_OUT} ---")
    x, hc_fn = make_inputs(M, HC_MULT_IN, HC_DIM, HC_MULT_OUT)
    eps = 1e-6

    us_ref_e = time_eager(_hc_pre_decode_reference, x, hc_fn, eps)
    us_trt_e = time_eager(hc_pre_decode_triton, x, hc_fn, eps)
    print(f"  EAGER: torch={us_ref_e:.2f} us  triton={us_trt_e:.2f} us  speedup={us_ref_e/us_trt_e:.2f}x")

    try:
        us_ref_g = time_graph(_hc_pre_decode_reference, (x, hc_fn, eps))
        us_trt_g = time_graph(hc_pre_decode_triton, (x, hc_fn, eps))
        print(f"  GRAPH: torch={us_ref_g:.2f} us  triton={us_trt_g:.2f} us  speedup={us_ref_g/us_trt_g:.2f}x")
    except Exception as e:
        print(f"  GRAPH: skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    # G0 — Production Flash-Base shapes: HC_MULT_IN=4, HC_DIM=4096, HC_MULT_OUT=24
    HC_MULT_IN = 4
    HC_DIM = 4096
    HC_MULT_OUT = 24

    results = []
    for M in (1, 2, 3, 4, 6, 8):
        results.append(("Flash-decode", M, correctness(f"Flash decode M={M}", M, HC_MULT_IN, HC_DIM, HC_MULT_OUT)))

    # Pro symmetric regression check
    for M in (1, 8):
        results.append(("Pro-symmetric", M, correctness(f"Pro symmetric M={M}", M, 24, 4096, 24)))

    print("\n=== correctness summary ===")
    for n, M, ok in results:
        print(f"  {n:14s} M={M:3d}  {'PASS' if ok else 'FAIL'}")
    all_pass = all(r[2] for r in results)
    print(f"OVERALL CORRECTNESS: {'PASS' if all_pass else 'FAIL'}")

    if not all_pass:
        sys.exit(1)

    # G1 — perf
    print("\n\n========== PERF (Flash decode) ==========")
    for M in (1, 4, 6, 8):
        perf_compare(f"Flash M={M}", M, HC_MULT_IN, HC_DIM, HC_MULT_OUT)
    sys.exit(0)
