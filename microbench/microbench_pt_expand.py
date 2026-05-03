"""Microbench + correctness for pt_expand_triton (Phase A2).

Tests against the current torch baseline used in indexer.py.
Production shape: (batch=4-8, max_pages=variable up to 128, block_size=64) for DSv4.
"""

import torch
import sys
sys.path.insert(0, "/sgl-pr/python")

from sglang.jit_kernel.pt_expand_triton import pt_expand


def torch_baseline(page_table: torch.Tensor, block_size: int, out: torch.Tensor):
    """Mirror the existing path: arange + unsqueeze*BS + add + reshape + copy_."""
    arange_buf = torch.arange(block_size, dtype=page_table.dtype, device=page_table.device)
    out.copy_(
        (page_table.unsqueeze(-1) * block_size + arange_buf.view(1, 1, -1)).view(
            page_table.shape[0], -1
        )
    )
    return out


def correctness_check(batch, max_pages, block_size, dtype=torch.int32, device="cuda"):
    torch.manual_seed(0)
    page_table = torch.randint(0, 1024, (batch, max_pages), dtype=dtype, device=device)
    expected = torch.empty((batch, max_pages * block_size), dtype=dtype, device=device)
    actual = torch.empty((batch, max_pages * block_size), dtype=dtype, device=device)

    torch_baseline(page_table, block_size, expected)
    pt_expand(page_table, block_size, out=actual)

    match = torch.equal(expected, actual)
    if not match:
        ndiff = (expected != actual).sum().item()
        print(f"  MISMATCH: {ndiff} elements differ. First diff:")
        for i in range(min(5, ndiff)):
            mask = expected != actual
            idx = mask.nonzero()[i].tolist()
            print(f"    idx={idx}: expected={expected[tuple(idx)]} actual={actual[tuple(idx)]}")
    return match


def time_eager(fn, args, n_warmup=10, n_iter=200):
    for _ in range(n_warmup):
        fn(*args)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(n_iter): fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / n_iter * 1000  # us


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
    print("Correctness checks:")
    print("=" * 70)
    shapes = [
        # (batch, max_pages, block_size)
        (1, 8, 64),
        (1, 64, 64),
        (4, 32, 64),    # production-ish
        (4, 64, 64),    # production
        (4, 128, 64),   # production max
        (8, 64, 64),    # warmup BS
    ]
    all_pass = True
    for s in shapes:
        for dt in [torch.int32, torch.int64]:
            ok = correctness_check(*s, dtype=dt)
            status = "PASS" if ok else "FAIL"
            print(f"  shape={s} dtype={dt}: {status}")
            all_pass &= ok
    if not all_pass:
        print("\nCorrectness FAILED — abort perf")
        return

    print("\n" + "=" * 70)
    print("Eager + cuda-graph perf (production shape b=4 mp=64 bs=64 int32):")
    print("=" * 70)
    print(f"{'shape':25s}  {'eager_torch_us':>14s}  {'eager_fused_us':>14s}  {'graph_torch_us':>14s}  {'graph_fused_us':>14s}  {'graph speedup':>14s}")
    for s in shapes:
        batch, max_pages, block_size = s
        page_table = torch.randint(0, 1024, (batch, max_pages), dtype=torch.int32, device="cuda")
        out_torch = torch.empty((batch, max_pages * block_size), dtype=torch.int32, device="cuda")
        out_fused = torch.empty_like(out_torch)
        t_torch_e = time_eager(torch_baseline, (page_table, block_size, out_torch))
        t_fused_e = time_eager(pt_expand, (page_table, block_size), n_warmup=10, n_iter=200)  # allocates each time
        # cuda-graph with pre-allocated out (the realistic prod pattern)
        t_torch_g = time_cudagraph(torch_baseline, (page_table, block_size, out_torch))
        # for fused: pass out= so it's pre-allocated
        def fused_with_out(pt, bs):
            return pt_expand(pt, bs, out=out_fused)
        t_fused_g = time_cudagraph(fused_with_out, (page_table, block_size))
        speedup = t_torch_g / t_fused_g if t_fused_g > 0 else float('inf')
        print(f"{str(s):25s}  {t_torch_e:>14.2f}  {t_fused_e:>14.2f}  {t_torch_g:>14.2f}  {t_fused_g:>14.2f}  {speedup:>13.2f}x")


if __name__ == "__main__":
    main()
