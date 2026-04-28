"""HIP RoPE kernel — correctness vs Triton + per-call CPU latency comparison.

Validates `apply_rotary_emb_hip` against `apply_rotary_emb_triton` (the kernel
it replaces) on 8 production-realistic shape combos, and times both the GPU
exec AND the CPU launch path of each.

Goal: prove the HIP kernel is bit-equivalent OR within fp32->bf16 noise, AND
that the per-call CPU latency is reduced enough to materially help the
89 ms / window observed in trace.

Production shapes (DSv4 Flash-Base FP8 + Pro V32):
  - decode Q rope:  3D [bs=4, n_heads=128, rope=64], with positions
  - decode KV rope: 2D [bs=4, rope=64], no positions (already indexed)
  - prefill chunked Q rope:  3D [8192, 128, 64], with positions
  - prefill chunked KV rope: 2D [8192, 64], no positions
  - inverse variants for compress

Usage:
    python3 /sgl-pr/microbench/microbench_rope_hip.py
"""
from __future__ import annotations
import os
import sys
import statistics
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from sglang.srt.layers.deepseek_v4_rope import apply_rotary_emb_triton  # noqa: E402
from sglang.srt.layers.rope_hip import apply_rotary_emb_hip  # noqa: E402


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def make_inputs(B: int, n_heads: int, rope_dim: int, with_pos: bool, is_3d: bool,
                max_seqlen: int = 8192, device="cuda", seed: int = 0):
    torch.manual_seed(seed)
    if is_3d:
        x = torch.randn(B, n_heads, rope_dim, dtype=torch.bfloat16, device=device) * 0.1
    else:
        x = torch.randn(B, rope_dim, dtype=torch.bfloat16, device=device) * 0.1

    if with_pos:
        # full freqs table
        theta = torch.arange(rope_dim // 2, dtype=torch.float32, device=device)
        theta = 10000 ** (-theta * 2.0 / rope_dim)
        t = torch.arange(max_seqlen, dtype=torch.float32, device=device)
        freqs = torch.outer(t, theta)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        positions = torch.randint(0, max_seqlen, (B,), dtype=torch.int64, device=device)
    else:
        # already-indexed freqs
        theta = torch.arange(rope_dim // 2, dtype=torch.float32, device=device)
        theta = 10000 ** (-theta * 2.0 / rope_dim)
        t = torch.randn(B, dtype=torch.float32, device=device).abs() * max_seqlen
        freqs = t[:, None] * theta[None, :]
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        positions = None
    return x, freqs_cis, positions


def time_call(fn, *args, n_warmup=5, n_iter=200):
    """Time a single function call. Reports CPU-side wall + GPU exec separately."""
    import time
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()

    # CPU+GPU wall (per-call) — caller-side time including launch path
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    cpu_starts = [0.0] * n_iter
    cpu_ends = [0.0] * n_iter
    for i in range(n_iter):
        cpu_starts[i] = time.perf_counter()
        starts[i].record()
        fn(*args)
        ends[i].record()
        cpu_ends[i] = time.perf_counter()
    torch.cuda.synchronize()

    gpu_us = sorted(s.elapsed_time(e) * 1000 for s, e in zip(starts, ends))
    cpu_us = sorted((cpu_ends[i] - cpu_starts[i]) * 1e6 for i in range(n_iter))
    n = len(gpu_us)
    return {
        "cpu_p50": cpu_us[n // 2],
        "cpu_p90": cpu_us[int(n * 0.9)],
        "gpu_p50": gpu_us[n // 2],
        "gpu_p90": gpu_us[int(n * 0.9)],
    }


def run_one(label: str, B: int, n_heads: int, rope_dim: int, *,
            with_pos: bool, is_3d: bool, inverse: bool):
    x_t, freqs_cis_t, positions_t = make_inputs(B, n_heads, rope_dim, with_pos, is_3d)
    x_h, freqs_cis_h, positions_h = make_inputs(B, n_heads, rope_dim, with_pos, is_3d)

    # Run reference (Triton)
    apply_rotary_emb_triton(x_t, freqs_cis_t, positions=positions_t, inverse=inverse)
    # Run HIP
    apply_rotary_emb_hip(x_h, freqs_cis_h, positions=positions_h, inverse=inverse)

    cs = cos_sim(x_t, x_h)
    max_diff = float((x_t.float() - x_h.float()).abs().max())
    ok = (cs >= 0.99999 and max_diff < 5e-3)

    # Time both for perf comparison
    x_t2, freqs_cis_t2, positions_t2 = make_inputs(B, n_heads, rope_dim, with_pos, is_3d)
    x_h2, freqs_cis_h2, positions_h2 = make_inputs(B, n_heads, rope_dim, with_pos, is_3d)
    triton_times = time_call(
        lambda: apply_rotary_emb_triton(x_t2, freqs_cis_t2, positions=positions_t2, inverse=inverse)
    )
    hip_times = time_call(
        lambda: apply_rotary_emb_hip(x_h2, freqs_cis_h2, positions=positions_h2, inverse=inverse)
    )

    return {
        "label": label, "B": B, "n_heads": n_heads, "rope": rope_dim,
        "with_pos": with_pos, "is_3d": is_3d, "inverse": inverse,
        "cos_sim": cs, "max_diff": max_diff, "ok": ok,
        "triton_cpu_p50_us": triton_times["cpu_p50"],
        "triton_gpu_p50_us": triton_times["gpu_p50"],
        "hip_cpu_p50_us": hip_times["cpu_p50"],
        "hip_gpu_p50_us": hip_times["gpu_p50"],
    }


def main():
    configs = [
        # (label, B, n_heads, rope, with_pos, is_3d, inverse)
        ("decode-Q-pos-3d",      4,   128, 64, True,  True,  False),
        ("decode-KV-nopos-2d",   4,   1,   64, False, False, False),
        ("decode-KV-inv-2d",     4,   1,   64, False, False, True),
        ("prefill-Q-pos-3d",     8192, 128, 64, True,  True,  False),
        ("prefill-KV-nopos-2d",  8192, 1,   64, False, False, False),
        ("prefill-KV-inv-2d",    8192, 1,   64, False, False, True),
        ("decode-Q-pos-3d-d128", 4,   128, 128, True, True,  False),
    ]
    print(f"{'label':<22} {'cs':>8} {'mxd':>9} {'tCPU(us)':>9} {'hCPU(us)':>9} "
          f"{'cpu-savings':>11} {'tGPU(us)':>9} {'hGPU(us)':>9}  PASS")
    n_pass = 0
    for cfg in configs:
        label, B, H, D, p, d3, inv = cfg
        try:
            r = run_one(label, B, H, D, with_pos=p, is_3d=d3, inverse=inv)
        except Exception as e:
            print(f"{label:<22} ERROR: {type(e).__name__}: {e}")
            continue
        n_pass += int(r["ok"])
        cpu_save_pct = (r["triton_cpu_p50_us"] - r["hip_cpu_p50_us"]) / r["triton_cpu_p50_us"] * 100
        print(f"{r['label']:<22} {r['cos_sim']:8.6f} {r['max_diff']:9.2e} "
              f"{r['triton_cpu_p50_us']:9.1f} {r['hip_cpu_p50_us']:9.1f} "
              f"{cpu_save_pct:>10.1f}% "
              f"{r['triton_gpu_p50_us']:9.2f} {r['hip_gpu_p50_us']:9.2f}  "
              f"{'PASS' if r['ok'] else 'FAIL'}")
    print(f"\n  {n_pass}/{len(configs)} configs pass (cos_sim >= 0.99999)")
    return 0 if n_pass == len(configs) else 1


if __name__ == "__main__":
    sys.exit(main())
