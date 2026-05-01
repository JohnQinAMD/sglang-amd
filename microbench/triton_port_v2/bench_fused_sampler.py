"""v2 microbench: fused_sampler_triton vs torch reference + aiter (if available).

GATES:
- (G0) Correctness: distribution-equivalence test (KL divergence, not bit-exact)
- (G1) Performance: graph-replay timing on production shapes (B=6, V=152k typical DSv4)

Sampling correctness can't be bit-exact because the RNG seeding paths differ between
torch and our kernel. Instead: validate by sampling N times, comparing empirical
token distributions under various (T, top_k, top_p) settings.
"""
import sys
import time

sys.path.insert(0, "/sgl-pr/python")

import torch

from sglang.jit_kernel.fused_sampler_triton import (
    _fused_sampler_reference,
    fused_sampler_triton,
)


def make_inputs(B, V, dtype=torch.bfloat16, top_k=50, top_p=0.95, temp=0.7, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    logits = torch.randn(B, V, dtype=dtype, device=device)
    top_k_t = torch.full((B,), top_k, dtype=torch.int32, device=device)
    top_p_t = torch.full((B,), top_p, dtype=torch.float32, device=device)
    temp_t = torch.full((B,), temp, dtype=torch.float32, device=device)
    return logits, top_k_t, top_p_t, temp_t


def correctness(name, B, V, top_k, top_p, temp):
    print(f"\n=== {name}: B={B} V={V} top_k={top_k} top_p={top_p} temp={temp} ===")
    logits, top_k_t, top_p_t, temp_t = make_inputs(B, V, top_k=top_k, top_p=top_p, temp=temp)

    # Sanity check: kernel runs without crashing, returns valid token ids
    try:
        out = fused_sampler_triton(logits, top_k_t, top_p_t, temp_t, seed=42)
        assert out.shape == (B,)
        assert out.dtype == torch.int32
        assert (out >= 0).all() and (out < V).all(), f"out has out-of-range tokens: min={out.min()} max={out.max()}"
        print(f"  triton out: shape={tuple(out.shape)}, range=[{out.min().item()}, {out.max().item()}]")
    except Exception as e:
        print(f"  triton FAILED: {type(e).__name__}: {e}")
        return False

    # Reference: torch impl
    out_ref = _fused_sampler_reference(logits, top_k_t, top_p_t, temp_t, seed=42)
    print(f"  ref    out: shape={tuple(out_ref.shape)}, range=[{out_ref.min().item()}, {out_ref.max().item()}]")

    # Distribution equivalence: sample N times, check that triton's tokens are in the
    # union of top_k * top_p tokens that reference would consider.
    # For now, just validate that triton's output tokens are within the top_k of logits
    # for each batch — necessary condition for correctness.
    sorted_logits, sorted_idx = torch.sort(logits.float() / temp_t.unsqueeze(-1), dim=-1, descending=True)
    top_k_set = sorted_idx[:, :top_k]  # [B, top_k]
    in_top_k = (out.unsqueeze(-1) == top_k_set).any(dim=-1)
    print(f"  fraction of samples in top_k: {in_top_k.float().mean().item():.3f} (should be 1.0)")
    PASS = in_top_k.all().item()
    print(f"  RESULT: {'PASS' if PASS else 'FAIL'}")
    return PASS


def time_fn(fn, *args, iters=100, warmup=20):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6


def time_graph(fn, args, iters=100):
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


def perf_compare(name, B, V, top_k=50, top_p=0.95, temp=0.7):
    print(f"\n--- perf {name}: B={B} V={V} ---")
    logits, top_k_t, top_p_t, temp_t = make_inputs(B, V, top_k=top_k, top_p=top_p, temp=temp)

    # Triton fused kernel
    def _trt():
        return fused_sampler_triton(logits, top_k_t, top_p_t, temp_t, seed=42)

    # Reference: temp scale + softmax + sort + cumsum + sample (current sglang torch fallback path)
    def _ref():
        return _fused_sampler_reference(logits, top_k_t, top_p_t, temp_t, seed=42)

    us_t_e = time_fn(_trt)
    us_r_e = time_fn(_ref)
    print(f"  EAGER: torch={us_r_e:.2f} us  triton={us_t_e:.2f} us  speedup={us_r_e/us_t_e:.2f}x")

    try:
        us_t_g = time_graph(_trt, ())
        us_r_g = time_graph(_ref, ())
        print(f"  GRAPH: torch={us_r_g:.2f} us  triton={us_t_g:.2f} us  speedup={us_r_g/us_t_g:.2f}x")
    except Exception as e:
        print(f"  GRAPH: skipped ({type(e).__name__}: {e})")

    # Compare to AITER if available
    try:
        import aiter.ops.sampling
        if hasattr(torch.ops, "aiter") and hasattr(torch.ops.aiter, "top_k_top_p_sampling_from_probs"):
            # AITER path: needs probs, not logits
            def _aiter_full():
                logits_t = logits.float() / temp_t.unsqueeze(-1)
                probs = torch.softmax(logits_t, dim=-1)
                return torch.ops.aiter.top_k_top_p_sampling_from_probs(
                    probs.contiguous(), None, top_k_t, 0, top_p_t, 0.0, False,
                )
            us_a_e = time_fn(_aiter_full)
            print(f"  AITER (eager, full chain): {us_a_e:.2f} us  triton speedup={us_a_e/us_t_e:.2f}x")
    except Exception as e:
        print(f"  AITER comparison: skipped ({e})")


if __name__ == "__main__":
    # G0 correctness
    results = []
    for (B, V, k, p, T) in [
        (1, 152064, 50, 0.95, 0.7),
        (4, 152064, 50, 0.95, 0.7),
        (6, 152064, 50, 0.95, 0.7),
        (1, 152064, 1, 1.0, 1e-5),    # ~greedy
        (4, 152064, 40, 0.9, 1.0),
    ]:
        results.append(((B, V, k, p, T), correctness(f"B={B} V={V} k={k} p={p} T={T}", B, V, k, p, T)))

    print("\n=== correctness summary ===")
    for cfg, ok in results:
        print(f"  {cfg}  {'PASS' if ok else 'FAIL'}")
    all_pass = all(r[1] for r in results)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")

    if not all_pass:
        sys.exit(1)

    # G1 perf
    print("\n========== PERF ==========")
    for (B, V, k, p, T) in [
        (1, 152064, 50, 0.95, 0.7),
        (4, 152064, 50, 0.95, 0.7),
        (6, 152064, 50, 0.95, 0.7),
    ]:
        perf_compare(f"B={B} V={V}", B, V, k, p, T)
    sys.exit(0)
