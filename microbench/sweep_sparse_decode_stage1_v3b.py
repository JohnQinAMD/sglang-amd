"""v3b stage1 sweep: zero in on BH=16 winner at production T=2048 shapes.

V3 found BH=16 wins 1.20-1.54x. Confirm best-of-best around it:
  - BH=16 + SK variants (8, 16, 32 — must be power of 2 for stage2 tl.arange)
  - BH=32 (huge tile)
  - BH=16 + ns=2 / nw=4/8 / waves variants
  - cross-shape (B=2,4,8) sanity
"""
import sys
sys.path.insert(0, "/sgl-pr/python")

import torch
from sglang.srt.flashmla_tests.triton_sparse_decode_kernel import (
    triton_sparse_attn_decode_split_k,
)


def time_cudagraph(fn, n_warmup=10, n_iter=200):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter): g.replay()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter * 1000


def make(B, Hq, Topk, D_QK, lonely_frac=0.05, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, Hq, D_QK, dtype=torch.bfloat16, device="cuda") * 0.1
    kv = torch.randn(B, Topk, D_QK, dtype=torch.bfloat16, device="cuda") * 0.1
    mask = torch.rand(B, Topk, device="cuda") < lonely_frac
    return q, kv, mask


def call(q, kv, mask, sm_scale, D_V, BH, BT, BD, SK, nw, ns, wpe, mind):
    return lambda: triton_sparse_attn_decode_split_k(
        q, kv, mask, None, sm_scale, D_V,
        BLOCK_H=BH, BLOCK_T=BT, BLOCK_D=BD, SPLIT_K=SK,
        num_warps=nw, num_stages=ns,
        waves_per_eu=wpe, matrix_instr_nonkdim=mind,
    )


def sweep(B, Topk):
    print(f"\n{'='*80}")
    print(f"=== B={B} Topk={Topk} Hq=64 D_QK=D_V=512 ===")
    print(f"{'='*80}")
    Hq, D_QK, D_V = 64, 512, 512
    sm_scale = 0.04
    q, kv, mask = make(B, Hq, Topk, D_QK)

    if B <= 2: base_BH, base_SK = 4, 16
    elif Topk >= 1024: base_BH, base_SK = 8, 16
    else: base_BH, base_SK = 8, 8
    base = (f"BH={base_BH} BT=32 BD=256 SK={base_SK} nw=4 ns=1 wpe=2 mind=16",
            base_BH, 32, 256, base_SK, 4, 1, 2, 16)
    base_us = time_cudagraph(call(q, kv, mask, sm_scale, D_V, *base[1:]))
    print(f"baseline (shipped):  {base[0]:55s}  {base_us:7.2f} us")

    cfgs = [
        # BH=16 deep dive
        ("BH=16 SK=16",          16, 32, 256, 16, 4, 1, 2, 16),
        ("BH=16 SK=8",           16, 32, 256, 8,  4, 1, 2, 16),
        ("BH=16 SK=32",          16, 32, 256, 32, 4, 1, 2, 16),
        ("BH=16 SK=16 ns=2",     16, 32, 256, 16, 4, 2, 2, 16),
        ("BH=16 SK=16 nw=8",     16, 32, 256, 16, 8, 1, 2, 16),
        ("BH=16 SK=16 wpe=1",    16, 32, 256, 16, 4, 1, 1, 16),
        ("BH=16 SK=16 wpe=4",    16, 32, 256, 16, 4, 1, 4, 16),
        ("BH=16 BT=16 SK=16",    16, 16, 256, 16, 4, 1, 2, 16),
        ("BH=16 BT=64 SK=16",    16, 64, 256, 16, 4, 1, 2, 16),
        ("BH=16 BT=64 SK=8",     16, 64, 256, 8,  4, 1, 2, 16),
        # BH=32 (huge tile)
        ("BH=32 SK=16",          32, 32, 256, 16, 4, 1, 2, 16),
        ("BH=32 SK=8",           32, 32, 256, 8,  4, 1, 2, 16),
        ("BH=32 SK=16 nw=8",     32, 32, 256, 16, 8, 1, 2, 16),
        # BH=64 (single H-tile per program)
        ("BH=64 SK=16",          64, 32, 256, 16, 4, 1, 2, 16),
        ("BH=64 SK=8",           64, 32, 256, 8,  4, 1, 2, 16),
        ("BH=64 SK=16 nw=8",     64, 32, 256, 16, 8, 1, 2, 16),
    ]

    print(f"{'config':55s}  {'us':>7s}  {'speedup':>8s}")
    best_label, best_us = "baseline", base_us
    for label, *cfg in cfgs:
        try:
            us = time_cudagraph(call(q, kv, mask, sm_scale, D_V, *cfg))
            mark = " *" if us < best_us else ""
            print(f"{label:55s}  {us:7.2f}  {base_us/us:7.3f}x{mark}")
            if us < best_us: best_label, best_us = label, us
        except Exception as ex:
            print(f"{label:55s}  FAIL: {str(ex)[:30]}")

    print(f"BEST: {best_label} = {best_us:.2f} us ({base_us/best_us:.3f}x vs shipped)")
    return base_us, best_us, best_label


def main():
    results = []
    for B, T in [(2, 2048), (4, 2048), (8, 2048), (4, 1024), (8, 1024)]:
        results.append((B, T) + sweep(B, T))
    print(f"\n\n{'='*80}\n=== SUMMARY ===\n{'='*80}")
    print(f"{'shape':18s}  {'baseline us':>11s}  {'best us':>9s}  {'speedup':>8s}  {'best config':30s}")
    for B, T, b, x, lab in results:
        print(f"B={B} T={T:<6d}      {b:11.2f}  {x:9.2f}  {b/x:7.3f}x  {lab}")


if __name__ == "__main__":
    main()
