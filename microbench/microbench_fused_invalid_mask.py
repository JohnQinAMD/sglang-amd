"""Phase C1: correctness + perf for fused_invalid_mask_triton."""
import torch
import sys
sys.path.insert(0, "/sgl-pr/python")
from sglang.jit_kernel.fused_invalid_mask_triton import fused_invalid_mask


def torch_baseline(indices, topk_length):
    """Mirror ref.py:240-244."""
    b, s_q, topk = indices.shape
    invalid_mask = indices == -1
    if topk_length is not None:
        invalid_mask |= torch.arange(0, topk, device=indices.device).view(
            1, 1, topk
        ).broadcast_to(b, s_q, topk) >= topk_length.view(b, 1, 1)
    return invalid_mask


def correctness(b, s_q, topk, with_topk_length=True, dtype=torch.int32):
    torch.manual_seed(0)
    # Mix of valid indices (>=0) and -1 (invalid)
    indices = torch.randint(-1, 100, (b, s_q, topk), dtype=dtype, device="cuda")
    topk_length = torch.randint(0, topk + 1, (b,), dtype=torch.int32, device="cuda") if with_topk_length else None

    ref = torch_baseline(indices, topk_length)
    out = fused_invalid_mask(indices, topk_length)
    ok = torch.equal(ref, out)
    if not ok:
        ndiff = (ref != out).sum().item()
        print(f"  DIFF {ndiff} elements")
    return ok


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
    print("Correctness:")
    shapes = [
        (1, 1, 512), (4, 1, 512), (8, 1, 512), (4, 1, 256), (4, 1, 1024),
    ]
    all_ok = True
    for s in shapes:
        for tl_kind in [True, False]:
            ok = correctness(*s, with_topk_length=tl_kind)
            print(f"  shape={s} topk_len={tl_kind}: {'PASS' if ok else 'FAIL'}")
            all_ok &= ok
    if not all_ok:
        print("FAIL"); return

    print("\nCuda-graph perf:")
    for s in shapes:
        b, s_q, topk = s
        indices = torch.randint(-1, 100, s, dtype=torch.int32, device="cuda")
        topk_length = torch.randint(0, topk + 1, (b,), dtype=torch.int32, device="cuda")
        t_ref = time_cudagraph(torch_baseline, (indices, topk_length))
        t_fused = time_cudagraph(fused_invalid_mask, (indices, topk_length))
        print(f"  shape={s}: torch={t_ref:.2f} us, fused={t_fused:.2f} us, speedup={t_ref/t_fused:.2f}x")


if __name__ == "__main__":
    main()
