"""v2 microbench: fused_lonely_q_correction vs torch reference (eq + 2× where chain)."""
import sys

sys.path.insert(0, "/sgl-pr/python")

import time
import torch

from sglang.jit_kernel.fused_lonely_q_correction_triton import (
    fused_lonely_q_correction_triton,
)


# Production shapes from trace evidence (post-MEGA-3' Stage 1+2)
SHAPES = [
    # (s_q, h_q, d_v, label)
    (4, 16, 512, "decode bs=4 h_q=16"),
    (4, 1, 512, "decode bs=4 h_q=1 (indexer)"),
    (8, 16, 512, "decode bs=8 h_q=16"),
    (856, 1, 512, "prefill batch=856 h_q=1 (matches trace)"),   # the headline shape
    (2812, 1, 512, "prefill batch=2812 h_q=1 (matches trace)"),
    (856, 16, 512, "prefill batch=856 h_q=16"),
]


def torch_reference(output, lse):
    """7-launch torch chain (the unfused path)."""
    lonely_q_mask = lse == float("-inf")
    output_new = torch.where(
        lonely_q_mask.unsqueeze(-1).expand_as(output),
        torch.zeros_like(output),
        output,
    )
    lse_new = torch.where(lonely_q_mask, torch.full_like(lse, float("+inf")), lse)
    return output_new, lse_new


def make_inputs(s_q, h_q, d_v, lonely_frac=0.1, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    output = torch.randn(s_q, h_q, d_v, dtype=dtype, device=device) * 0.1
    lse = torch.randn(s_q, h_q, dtype=torch.float32, device=device) * 5.0
    # Inject lonely_q positions at lonely_frac probability
    lonely_mask = torch.rand(s_q, h_q, device=device) < lonely_frac
    lse[lonely_mask] = float("-inf")
    return output, lse


def correctness(s_q, h_q, d_v, label):
    print(f"\n=== {label} (s_q={s_q}, h_q={h_q}, d_v={d_v}) ===")
    output_ref, lse_ref = make_inputs(s_q, h_q, d_v)
    output_tri = output_ref.clone()
    lse_tri = lse_ref.clone()

    # Torch reference
    output_ref_out, lse_ref_out = torch_reference(output_ref, lse_ref)

    # Triton (in-place)
    ok = fused_lonely_q_correction_triton(output_tri, lse_tri)
    if not ok:
        print(f"  Triton returned False. FAIL.")
        return False

    # Compare
    def safe_diff(a, b):
        af, bf = a.float(), b.float()
        both_inf = (torch.isposinf(af) & torch.isposinf(bf)) | (torch.isneginf(af) & torch.isneginf(bf))
        diff = (af - bf).abs()
        diff = torch.where(both_inf, torch.zeros_like(diff), diff)
        return torch.nan_to_num(diff, nan=1e10, posinf=1e10, neginf=1e10)

    out_diff = safe_diff(output_ref_out, output_tri)
    lse_diff = safe_diff(lse_ref_out, lse_tri)
    print(f"  output diff: max={out_diff.max().item():.4e} mean={out_diff.mean().item():.4e}")
    print(f"  lse    diff: max={lse_diff.max().item():.4e} mean={lse_diff.mean().item():.4e}")

    PASS = out_diff.max().item() <= 1e-5 and lse_diff.max().item() <= 1e-5
    print(f"  RESULT: {'PASS' if PASS else 'FAIL'}")
    return PASS


def perf(s_q, h_q, d_v, label):
    print(f"\n--- perf {label} ---")
    output, lse = make_inputs(s_q, h_q, d_v)

    def call_torch():
        torch_reference(output.clone(), lse.clone())

    output_t = output.clone()
    lse_t = lse.clone()

    def call_triton():
        # Re-clone each iter so we measure consistent in-place starting state
        output_t.copy_(output)
        lse_t.copy_(lse)
        fused_lonely_q_correction_triton(output_t, lse_t)

    for _ in range(5):
        call_torch()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        call_torch()
    torch.cuda.synchronize()
    t_torch = (time.perf_counter() - t0) / 20 * 1e6

    for _ in range(5):
        call_triton()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        call_triton()
    torch.cuda.synchronize()
    t_triton = (time.perf_counter() - t0) / 20 * 1e6

    # Subtract the clone overhead from triton time
    def call_clone_only():
        output_t.copy_(output)
        lse_t.copy_(lse)

    for _ in range(5):
        call_clone_only()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        call_clone_only()
    torch.cuda.synchronize()
    t_clone = (time.perf_counter() - t0) / 20 * 1e6
    t_triton_kernel_only = t_triton - t_clone
    print(f"  eager: torch chain = {t_torch:>7.1f} us   triton (kernel only) = {t_triton_kernel_only:>7.1f} us   speedup = {t_torch/max(t_triton_kernel_only, 0.1):.2f}x")


def main():
    print("=" * 78)
    print("fused_lonely_q_correction: 7-launch torch chain → 1 Triton launch")
    print("=" * 78)
    all_pass = True
    for s_q, h_q, d_v, label in SHAPES:
        ok = correctness(s_q, h_q, d_v, label)
        if not ok:
            all_pass = False
    print(f"\nG0 SUMMARY: {'ALL PASS' if all_pass else 'FAILURES'}")
    if not all_pass:
        sys.exit(1)
    print("\n## Performance")
    for s_q, h_q, d_v, label in SHAPES:
        perf(s_q, h_q, d_v, label)


if __name__ == "__main__":
    main()
