"""v2 microbench: compress_decode_kv_pool_fused vs torch reference.

Validates correctness on production shapes for ratio=4 (overlap=True) and
ratio=128 (overlap=False) cases.

Per `feedback_microbench_methodology_traps.md` and `reference_microbench_v2_framework.md`:
- (G0) Correctness vs torch reference on production shapes — HARD GATE
- (G1) Eager + cuda-graph capture+replay timing on production shape histogram
- Cache HIT modeling (compile cache stays warm)
"""
import sys
import time

sys.path.insert(0, "/sgl-pr/python")

import torch

from sglang.jit_kernel.compress_decode_kv_pool_fused_triton import (
    _compress_decode_kv_pool_reference,
    compress_decode_kv_pool_fused,
)


def make_inputs(num_reqs, bs, ratio, overlap, head_dim, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    coff = 1 + (1 if overlap else 0)
    T = coff * ratio
    D = coff * head_dim
    device = torch.device("cuda:0")
    kv_pool_kv = torch.randn(num_reqs, T, D, dtype=dtype, device=device)
    kv_pool_score = torch.randn(num_reqs, T, D, dtype=dtype, device=device)
    # req_indices: pick from [0, num_reqs) without replacement (request batch)
    req_indices = torch.randperm(num_reqs, device=device)[:bs].to(torch.int32)
    seq_lens = torch.randint(1, 1024, (bs,), dtype=torch.int32, device=device)
    new_kv = torch.randn(bs, D, dtype=dtype, device=device)
    new_score = torch.randn(bs, D, dtype=dtype, device=device)
    ape_score = torch.randn(ratio, D, dtype=torch.float32, device=device)
    return kv_pool_kv, kv_pool_score, req_indices, seq_lens, new_kv, new_score, ape_score, ratio, overlap


def correctness(name, num_reqs, bs, ratio, overlap, head_dim):
    print(f"\n=== {name}: num_reqs={num_reqs} bs={bs} ratio={ratio} overlap={overlap} head_dim={head_dim} ===")
    args = make_inputs(num_reqs, bs, ratio, overlap, head_dim)

    # Clone pools so reference and triton don't share state
    pool_kv_ref = args[0].clone()
    pool_score_ref = args[1].clone()
    pool_kv_t = args[0].clone()
    pool_score_t = args[1].clone()
    req_indices, seq_lens, new_kv, new_score, ape_score, ratio_, overlap_ = args[2:]

    # Reference
    out_kv_ref, out_score_ref = _compress_decode_kv_pool_reference(
        pool_kv_ref, pool_score_ref,
        req_indices, seq_lens,
        new_kv, new_score,
        ape_score, ratio_, overlap_,
    )

    # Triton
    out_kv_t, out_score_t = compress_decode_kv_pool_fused(
        pool_kv_t, pool_score_t,
        req_indices, seq_lens,
        new_kv, new_score,
        ape_score, ratio_, overlap_,
    )

    # Validate output buffers
    diff_out_kv = (out_kv_ref.float() - out_kv_t.float()).abs()
    diff_out_score = (out_score_ref.float() - out_score_t.float()).abs()
    print(f"  out_kv: max diff={diff_out_kv.max().item():.2e} mean diff={diff_out_kv.mean().item():.2e}")
    print(f"  out_score: max diff={diff_out_score.max().item():.2e} mean diff={diff_out_score.mean().item():.2e}")

    # Validate pool side-effects
    diff_pool_kv = (pool_kv_ref.float() - pool_kv_t.float()).abs()
    diff_pool_score = (pool_score_ref.float() - pool_score_t.float()).abs()
    print(f"  pool_kv  side-effect: max diff={diff_pool_kv.max().item():.2e}")
    print(f"  pool_score side-effect: max diff={diff_pool_score.max().item():.2e}")

    # Pass criteria: bf16 ULP floor
    ULP = 1e-2  # bf16 cast tolerance
    PASS = (
        diff_out_kv.max().item() <= ULP
        and diff_out_score.max().item() <= 5e-2  # ape add adds noise
        and diff_pool_kv.max().item() <= ULP
        and diff_pool_score.max().item() <= ULP
    )
    print(f"  RESULT: {'PASS' if PASS else 'FAIL'}")
    return PASS


def time_kernel_eager(fn, *args, iters=50, warmup=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us per call


def time_kernel_graph(fn, args_template, iters=50):
    """Time via cuda-graph capture+replay (production-aligned)."""
    # Warmup
    for _ in range(3):
        fn(*args_template)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            for _ in range(iters):
                fn(*args_template)
    torch.cuda.current_stream().wait_stream(s)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g.replay()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us per call


def perf_compare(name, num_reqs, bs, ratio, overlap, head_dim):
    print(f"\n--- perf {name}: bs={bs} ratio={ratio} ---")
    args = make_inputs(num_reqs, bs, ratio, overlap, head_dim)

    # Pre-clone pools so each call has fresh state (otherwise sequential calls drift)
    # For PERF timing we just need consistent shapes; keep same seed each iter
    def _ref(*a):
        pool_kv = args[0].clone()
        pool_score = args[1].clone()
        return _compress_decode_kv_pool_reference(
            pool_kv, pool_score, args[2], args[3], args[4], args[5], args[6], args[7], args[8]
        )

    def _trt(*a):
        pool_kv = args[0].clone()
        pool_score = args[1].clone()
        return compress_decode_kv_pool_fused(
            pool_kv, pool_score, args[2], args[3], args[4], args[5], args[6], args[7], args[8]
        )

    # Eager
    us_ref = time_kernel_eager(_ref)
    us_t = time_kernel_eager(_trt)
    print(f"  EAGER: torch={us_ref:.2f} us  triton={us_t:.2f} us  speedup={us_ref/us_t:.2f}x")

    # Graph replay (production cost model)
    try:
        us_ref_g = time_kernel_graph(_ref, ())
        us_t_g = time_kernel_graph(_trt, ())
        print(f"  GRAPH: torch={us_ref_g:.2f} us  triton={us_t_g:.2f} us  speedup={us_ref_g/us_t_g:.2f}x")
    except Exception as e:
        print(f"  GRAPH: skipped ({type(e).__name__}: {e})")


if __name__ == "__main__":
    # G0 — correctness on production-aligned shapes
    head_dim = 192  # DSv4 head_dim (128 nope + 64 rope)
    results = []
    for bs in (1, 2, 4, 6, 8):
        # ratio=4 overlap=True (the c4 path — dominant)
        results.append(("c4_r4", bs, correctness(f"c4_r4 bs={bs}", 1024, bs, 4, True, head_dim)))
    # ratio=128 overlap=False (the c128 path)
    for bs in (1, 4):
        results.append(("c128", bs, correctness(f"c128 bs={bs}", 256, bs, 128, False, head_dim)))

    print("\n=== correctness summary ===")
    for n, bs, ok in results:
        print(f"  {n:8s} bs={bs:2d}  {'PASS' if ok else 'FAIL'}")
    all_pass = all(r[2] for r in results)
    print(f"OVERALL CORRECTNESS: {'PASS' if all_pass else 'FAIL'}")

    if not all_pass:
        sys.exit(1)

    # G1 — perf compare (only run if correctness PASS)
    print("\n\n========== PERF ==========")
    for bs in (1, 4, 8):
        perf_compare(f"c4_r4 bs={bs}", 1024, bs, 4, True, head_dim)
    sys.exit(0)
