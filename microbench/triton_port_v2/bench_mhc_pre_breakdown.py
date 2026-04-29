"""Per-stage timing of the 3-kernel mhc_pre pipeline at production shape.

Identifies which of the 3 Triton kernels dominates so we tune the right one.
"""
import os, sys, time
sys.path.insert(0, '/sgl-pr/python')

import torch
import triton
from sglang.jit_kernel.mhc_pre_triton import (
    _mhc_pre_gemm_sqrsum_kernel,
    _mhc_pre_gemm_sqrsum_splitk_kernel,
    _mhc_pre_splitk_reduce_kernel,
    _mhc_pre_big_fuse_kernel,
    _mhc_pre_apply_mix_kernel,
)


device = "cuda:0"
torch.manual_seed(0)


def time_kernel(fn, iters=2000, warmup=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


def time_under_graph(fn, iters=2000, warmup=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            for _ in range(8):
                fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters // 8):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # us


def main():
    n, hc, hidden = 6, 4, 2048
    hc_mult3 = hc * (2 + hc)  # 24
    hc_hidden = hc * hidden   # 8192
    hc_mult3_pad = 32

    residual = torch.randn(n, hc, hidden, device=device, dtype=torch.bfloat16)
    fn_w = torch.randn(hc_mult3, hc_hidden, device=device, dtype=torch.float32)
    hc_scale = torch.randn(3, device=device, dtype=torch.float32).abs() + 0.1
    hc_base = torch.randn(hc_mult3, device=device, dtype=torch.float32) * 0.1

    out_partial = torch.empty(n, hc_mult3_pad, dtype=torch.float32, device=device)
    sqrsum = torch.empty(n, dtype=torch.float32, device=device)
    pre_mix = torch.empty(n, hc, dtype=torch.float32, device=device)
    post_mix = torch.empty(n, hc, dtype=torch.float32, device=device)
    comb_mix = torch.empty(n, hc * hc, dtype=torch.float32, device=device)
    layer_input = torch.empty(n, hidden, dtype=torch.bfloat16, device=device)

    # Split-K stage-1 buffers
    TOKEN_BLOCK = 8
    n_padded = ((n + TOKEN_BLOCK - 1) // TOKEN_BLOCK) * TOKEN_BLOCK
    SPLIT_K = 32
    SPLIT_SIZE = hc_hidden // SPLIT_K
    BLOCK_K_SPLITK = min(64, SPLIT_SIZE)
    out_partial_split = torch.empty(SPLIT_K, n_padded, hc_mult3_pad,
                                     dtype=torch.float32, device=device)
    sqrsum_partial = torch.empty(SPLIT_K, n_padded,
                                  dtype=torch.float32, device=device)

    print(f"Shape: n={n}, hc={hc}, hidden={hidden}, hc_hidden={hc_hidden}")
    print()

    # Stage 1: gemm + sqrsum (legacy single-CTA-per-token)
    BLOCK_K = 256
    def stage1():
        _mhc_pre_gemm_sqrsum_kernel[(n,)](
            residual.view(n, hc_hidden), fn_w, out_partial, sqrsum,
            n, hc_hidden,
            HC_MULT3=24, HC_MULT3_PAD=32, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

    # Stage 1 V2: split-K + tl.dot
    n_token_blocks = n_padded // TOKEN_BLOCK
    def stage1_v2_splitk():
        _mhc_pre_gemm_sqrsum_splitk_kernel[(SPLIT_K, n_token_blocks)](
            residual.view(n, hc_hidden), fn_w,
            out_partial_split, sqrsum_partial,
            n, hc_hidden, n_padded,
            HC_MULT3=24, HC_MULT3_PAD=32,
            TOKEN_BLOCK=TOKEN_BLOCK, SPLIT_K=SPLIT_K,
            SPLIT_SIZE=SPLIT_SIZE, BLOCK_K=BLOCK_K_SPLITK,
            num_warps=4,
        )

    # Stage 1.5 V2: split-K reduce
    def stage1_5_v2_reduce():
        _mhc_pre_splitk_reduce_kernel[(n,)](
            out_partial_split, sqrsum_partial, out_partial, sqrsum,
            n, n_padded,
            HC_MULT3=24, HC_MULT3_PAD=32, SPLIT_K=SPLIT_K,
            num_warps=1,
        )

    # Stage 2: big fuse
    def stage2():
        _mhc_pre_big_fuse_kernel[(n,)](
            out_partial, sqrsum, hc_scale, hc_base,
            pre_mix, post_mix, comb_mix,
            n, hc_hidden,
            1e-6, 1e-6, 1e-6, 2.0, 3,
            HC=4, HC_MULT3=24, HC_MULT3_PAD=32,
            num_warps=1,
        )

    # Stage 3: apply mix
    BLOCK_H = 256
    def stage3():
        _mhc_pre_apply_mix_kernel[(n, triton.cdiv(hidden, BLOCK_H))](
            layer_input, pre_mix, residual,
            n, hidden,
            HC=4, BLOCK_H=BLOCK_H,
            num_warps=4,
        )

    print(f"{'stage':<25s}  {'eager_us':>10s}  {'graph_us':>10s}")
    print("-" * 50)
    for label, fn in [
        ("S1 gemm+sqrsum (legacy)", stage1),
        ("S1 V2 splitk+tl.dot", stage1_v2_splitk),
        ("S1.5 V2 reduce", stage1_5_v2_reduce),
        ("S2 big_fuse", stage2),
        ("S3 apply_mix", stage3),
    ]:
        e = time_kernel(fn)
        g = time_under_graph(fn)
        print(f"  {label:<23s}  {e:>10.2f}  {g:>10.2f}")


if __name__ == "__main__":
    main()
