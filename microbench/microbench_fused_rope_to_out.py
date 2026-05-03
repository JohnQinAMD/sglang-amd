"""Microbench + correctness for fused_rmsnorm_rope_q_triton_to_out (Phase A1).

Tests against current in-place RoPE + q_out.copy_(q) chain.
"""
import torch
import sys
sys.path.insert(0, "/sgl-pr/python")

from sglang.srt.layers.deepseek_v4_rope import (
    fused_rmsnorm_rope_q_triton,
    fused_rmsnorm_rope_q_triton_to_out,
)


def make_freqs_cis(seqlen, rope_dim):
    freqs = 1.0 / (10000 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim))
    t = torch.arange(seqlen, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles).cuda()


def test_correctness(bs, n_local_heads, n_heads, head_dim, rope_dim, eps=1e-6):
    """Compare fused_to_out vs (fused_inplace + q_out.copy_(q))."""
    torch.manual_seed(0)
    # Make q_padded for TP=4 case
    q = torch.randn(bs, n_local_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
    q_ref = q.clone()
    q_for_fused = q.clone()

    freqs_cis = make_freqs_cis(2048, rope_dim)
    positions = torch.randint(0, 1024, (bs,), dtype=torch.int64, device="cuda")

    # Reference: in-place fuse + copy
    q_padded_ref = torch.zeros(bs, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    tp_slice = slice(0, n_local_heads)
    fused_rmsnorm_rope_q_triton(q_ref, freqs_cis, positions, eps, rope_dim)
    q_padded_ref[:, tp_slice, :].copy_(q_ref)

    # Test: out-of-place fused
    q_padded_test = torch.zeros(bs, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
    q_out_test = q_padded_test[:, tp_slice, :]
    fused_rmsnorm_rope_q_triton_to_out(q_for_fused, q_out_test, freqs_cis, positions, eps, rope_dim)

    match = torch.equal(q_padded_ref, q_padded_test)
    if not match:
        max_diff = (q_padded_ref.float() - q_padded_test.float()).abs().max().item()
        diff_in_slice = (q_padded_ref[:, tp_slice, :].float() - q_padded_test[:, tp_slice, :].float()).abs().max().item()
        n_zero_outside = (q_padded_test[:, n_local_heads:, :] == 0).all().item()
        print(f"  max_abs_diff={max_diff:.6f}, in_slice_diff={diff_in_slice:.6f}, outside_zero={n_zero_outside}")
    return match


def time_cudagraph(fn, args, n_warmup=10, n_iter=300):
    for _ in range(n_warmup): fn(*args)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter): g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter * 1000  # us


def main():
    print("=" * 70)
    print("Correctness checks (TP=4: n_heads=64, n_local_heads=16):")
    print("=" * 70)
    shapes = [
        # (bs, n_local_heads, n_heads, head_dim, rope_dim)
        (1, 16, 64, 192, 64),
        (4, 16, 64, 192, 64),  # production c=4
        (8, 16, 64, 192, 64),
    ]
    all_pass = True
    for s in shapes:
        ok = test_correctness(*s)
        status = "PASS" if ok else "FAIL"
        print(f"  shape={s}: {status}")
        all_pass &= ok
    if not all_pass:
        print("\nCorrectness FAILED — abort")
        return

    print("\n" + "=" * 70)
    print("Cuda-graph perf:")
    print("=" * 70)
    for s in shapes:
        bs, n_local_heads, n_heads, head_dim, rope_dim = s
        eps = 1e-6
        q = torch.randn(bs, n_local_heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
        freqs_cis = make_freqs_cis(2048, rope_dim)
        positions = torch.randint(0, 1024, (bs,), dtype=torch.int64, device="cuda")
        q_padded = torch.zeros(bs, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
        q_out = q_padded[:, :n_local_heads, :]

        # Baseline: in-place fuse + copy_
        def baseline():
            q_local = q.clone()
            fused_rmsnorm_rope_q_triton(q_local, freqs_cis, positions, eps, rope_dim)
            q_out.copy_(q_local)
        # Optimized: fused-to-out
        def optimized():
            q_local = q.clone()
            fused_rmsnorm_rope_q_triton_to_out(q_local, q_out, freqs_cis, positions, eps, rope_dim)
        t_base = time_cudagraph(baseline, ())
        t_opt = time_cudagraph(optimized, ())
        speedup = t_base / t_opt if t_opt > 0 else float('inf')
        print(f"  shape={s}: baseline={t_base:.2f} us, optimized={t_opt:.2f} us, speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
