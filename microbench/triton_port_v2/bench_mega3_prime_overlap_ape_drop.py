"""v2 microbench: MEGA-3' Stage 2 (overlap+APE+drop-first) vs torch reference."""
import sys

sys.path.insert(0, "/sgl-pr/python")

import time
import torch

from sglang.jit_kernel.extend_per_request_megakernel_triton import (
    mega3_prime_overlap_ape_drop_triton,
)


def torch_reference_overlap_ape_drop(
    in_kv,            # [n_in_blocks, R, 2*D]
    in_score,         # [n_in_blocks, R, 2*D]
    ape,              # [R, 2*D]
):
    """Mirror compress_extend_old:1146-1161 overlap_transform + APE + [1:]."""
    n_in, R, two_D = in_kv.shape
    D = two_D // 2

    # APE add to score (in-place op semantics — but operate on copy here)
    in_score = in_score + ape.unsqueeze(0)

    # overlap_transform on kv with fill_value=0
    new_kv = torch.zeros(n_in, 2 * R, D, dtype=in_kv.dtype, device=in_kv.device)
    new_kv[:, R:, :] = in_kv[:, :, D:]               # upper-half slots from upper cols
    new_kv[1:, :R, :] = in_kv[:-1, :, :D]            # lower-half slots[1:] from prev block lower cols

    # overlap_transform on score with fill_value=-inf
    new_score = torch.full((n_in, 2 * R, D), float("-inf"), dtype=in_score.dtype, device=in_score.device)
    new_score[:, R:, :] = in_score[:, :, D:]
    new_score[1:, :R, :] = in_score[:-1, :, :D]

    # drop first block
    return new_kv[1:], new_score[1:]


def make_inputs_one_request(n_in_blocks, R, D, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    in_kv = torch.randn(n_in_blocks, R, 2 * D, dtype=dtype, device=device) * 0.1
    in_score = torch.randn(n_in_blocks, R, 2 * D, dtype=dtype, device=device) * 0.1
    ape = torch.randn(R, 2 * D, dtype=torch.float32, device=device) * 0.05
    return in_kv, in_score, ape


def correctness_one_request(name, n_in_blocks, R, D):
    print(f"\n=== {name} (n_in_blocks={n_in_blocks}, R={R}, D={D}) ===")
    in_kv, in_score, ape = make_inputs_one_request(n_in_blocks, R, D)

    # Torch reference
    ref_kv, ref_score = torch_reference_overlap_ape_drop(in_kv, in_score, ape)

    # Pack into temp_buffer layout: [n_in_blocks * R, 2*D]
    device = in_kv.device
    temp_buffer_kv = in_kv.reshape(n_in_blocks * R, 2 * D).clone()
    temp_buffer_score = in_score.reshape(n_in_blocks * R, 2 * D).clone()

    # Single-request descriptors
    bs = 1
    buf_offsets = torch.zeros(bs, dtype=torch.int32, device=device)
    n_blocks = torch.tensor([n_in_blocks], dtype=torch.int32, device=device)
    out_block_offsets = torch.zeros(bs, dtype=torch.int32, device=device)

    n_out = n_in_blocks - 1
    out_kv = torch.empty(n_out, 2 * R, D, dtype=in_kv.dtype, device=device)
    out_score = torch.empty(n_out, 2 * R, D, dtype=in_score.dtype, device=device)

    fired = mega3_prime_overlap_ape_drop_triton(
        temp_buffer_kv, temp_buffer_score, out_kv, out_score, ape,
        buf_offsets, n_blocks, out_block_offsets, R, D,
    )
    if not fired:
        print(f"  Stage 2 returned False. FAIL.")
        return False

    def safe_diff(a, b):
        af, bf = a.float(), b.float()
        both_inf = (torch.isposinf(af) & torch.isposinf(bf)) | (torch.isneginf(af) & torch.isneginf(bf))
        diff = (af - bf).abs()
        diff = torch.where(both_inf, torch.zeros_like(diff), diff)
        diff = torch.nan_to_num(diff, nan=1e10, posinf=1e10, neginf=1e10)
        return diff

    kv_diff = safe_diff(ref_kv, out_kv)
    sc_diff = safe_diff(ref_score, out_score)
    print(f"  out_kv    diff: max={kv_diff.max().item():.4e} mean={kv_diff.mean().item():.4e}")
    print(f"  out_score diff: max={sc_diff.max().item():.4e} mean={sc_diff.mean().item():.4e}")

    # Tolerance: bf16 ULP for kv (no APE arithmetic), bf16 cast for score (APE add in fp32 then cast)
    PASS = kv_diff.max().item() <= 1e-3 and sc_diff.max().item() <= 5e-2
    print(f"  RESULT: {'PASS' if PASS else 'FAIL'}")
    return PASS


def correctness_multi_request(name, plans, R, D):
    """plans = list of n_in_blocks per request"""
    print(f"\n=== {name} (plans={plans}, R={R}, D={D}) ===")
    bs = len(plans)
    device = torch.device("cuda:0")

    # Build per-request temp_buffers and concat
    in_kv_list = []
    in_score_list = []
    for n_in in plans:
        in_kv, in_score, _ape = make_inputs_one_request(n_in, R, D, seed=hash(n_in) & 0xffff)
        in_kv_list.append(in_kv.reshape(n_in * R, 2 * D))
        in_score_list.append(in_score.reshape(n_in * R, 2 * D))

    ape = torch.randn(R, 2 * D, dtype=torch.float32, device=device) * 0.05

    temp_buffer_kv = torch.cat(in_kv_list, dim=0)
    temp_buffer_score = torch.cat(in_score_list, dim=0)

    # Compute offsets
    buf_offsets = []
    out_block_offsets = []
    buf_running = 0
    out_running = 0
    for n_in in plans:
        buf_offsets.append(buf_running)
        out_block_offsets.append(out_running)
        buf_running += n_in * R
        out_running += max(0, n_in - 1)
    buf_offsets = torch.tensor(buf_offsets, dtype=torch.int32, device=device)
    n_blocks = torch.tensor(plans, dtype=torch.int32, device=device)
    out_block_offsets = torch.tensor(out_block_offsets, dtype=torch.int32, device=device)

    total_out = sum(max(0, n - 1) for n in plans)
    out_kv = torch.empty(total_out, 2 * R, D, dtype=temp_buffer_kv.dtype, device=device)
    out_score = torch.empty(total_out, 2 * R, D, dtype=temp_buffer_score.dtype, device=device)

    fired = mega3_prime_overlap_ape_drop_triton(
        temp_buffer_kv, temp_buffer_score, out_kv, out_score, ape,
        buf_offsets, n_blocks, out_block_offsets, R, D,
    )
    if not fired:
        print(f"  Stage 2 returned False. FAIL.")
        return False

    # Reference per request, then concat
    ref_kv_list = []
    ref_score_list = []
    for i, n_in in enumerate(plans):
        in_kv_i = temp_buffer_kv[buf_offsets[i].item():buf_offsets[i].item() + n_in * R].reshape(n_in, R, 2 * D)
        in_score_i = temp_buffer_score[buf_offsets[i].item():buf_offsets[i].item() + n_in * R].reshape(n_in, R, 2 * D)
        rk, rs = torch_reference_overlap_ape_drop(in_kv_i, in_score_i, ape)
        ref_kv_list.append(rk)
        ref_score_list.append(rs)
    ref_kv = torch.cat(ref_kv_list, dim=0)
    ref_score = torch.cat(ref_score_list, dim=0)

    def safe_diff(a, b):
        af, bf = a.float(), b.float()
        both_inf = (torch.isposinf(af) & torch.isposinf(bf)) | (torch.isneginf(af) & torch.isneginf(bf))
        diff = (af - bf).abs()
        diff = torch.where(both_inf, torch.zeros_like(diff), diff)
        return torch.nan_to_num(diff, nan=1e10, posinf=1e10, neginf=1e10)

    kv_diff = safe_diff(ref_kv, out_kv)
    sc_diff = safe_diff(ref_score, out_score)
    print(f"  out_kv    diff: max={kv_diff.max().item():.4e} mean={kv_diff.mean().item():.4e}")
    print(f"  out_score diff: max={sc_diff.max().item():.4e} mean={sc_diff.mean().item():.4e}")
    PASS = kv_diff.max().item() <= 1e-3 and sc_diff.max().item() <= 5e-2
    print(f"  RESULT: {'PASS' if PASS else 'FAIL'}")
    return PASS


def perf(name, plans, R, D):
    print(f"\n--- perf {name} ---")
    device = torch.device("cuda:0")
    in_kv_list, in_score_list = [], []
    for n_in in plans:
        in_kv, in_score, _ = make_inputs_one_request(n_in, R, D, seed=hash(n_in) & 0xffff)
        in_kv_list.append(in_kv.reshape(n_in * R, 2 * D))
        in_score_list.append(in_score.reshape(n_in * R, 2 * D))
    ape = torch.randn(R, 2 * D, dtype=torch.float32, device=device) * 0.05
    temp_buffer_kv = torch.cat(in_kv_list, dim=0)
    temp_buffer_score = torch.cat(in_score_list, dim=0)
    buf_offsets, out_block_offsets = [], []
    buf_running, out_running = 0, 0
    for n_in in plans:
        buf_offsets.append(buf_running); out_block_offsets.append(out_running)
        buf_running += n_in * R; out_running += max(0, n_in - 1)
    buf_offsets = torch.tensor(buf_offsets, dtype=torch.int32, device=device)
    n_blocks = torch.tensor(plans, dtype=torch.int32, device=device)
    out_block_offsets = torch.tensor(out_block_offsets, dtype=torch.int32, device=device)
    total_out = sum(max(0, n - 1) for n in plans)
    out_kv = torch.empty(total_out, 2 * R, D, dtype=temp_buffer_kv.dtype, device=device)
    out_score = torch.empty(total_out, 2 * R, D, dtype=temp_buffer_score.dtype, device=device)

    def call_torch():
        for i, n_in in enumerate(plans):
            in_kv_i = temp_buffer_kv[buf_offsets[i].item():buf_offsets[i].item() + n_in * R].reshape(n_in, R, 2 * D)
            in_score_i = temp_buffer_score[buf_offsets[i].item():buf_offsets[i].item() + n_in * R].reshape(n_in, R, 2 * D)
            torch_reference_overlap_ape_drop(in_kv_i, in_score_i, ape)

    def call_triton():
        mega3_prime_overlap_ape_drop_triton(
            temp_buffer_kv, temp_buffer_score, out_kv, out_score, ape,
            buf_offsets, n_blocks, out_block_offsets, R, D,
        )

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

    print(f"  eager: torch loop = {t_torch:>8.1f} us   triton stage2 = {t_triton:>8.1f} us   speedup = {t_torch/t_triton:.2f}x")


def main():
    print("=" * 78)
    print("MEGA-3' Stage 2: overlap_transform + APE + drop-first vs torch reference")
    print("=" * 78)
    R = 4
    D = 128
    all_pass = True
    # Single-request shapes
    for n in [4, 8, 16, 64, 215]:
        ok = correctness_one_request(f"single req n_in={n}", n, R, D)
        if not ok:
            all_pass = False
    # Multi-request shapes
    for plans in [[8, 8, 8, 8], [25, 19, 32, 16], [215, 215, 215, 215]]:
        ok = correctness_multi_request(f"multi req plans={plans}", plans, R, D)
        if not ok:
            all_pass = False
    print(f"\nG0 SUMMARY: {'ALL PASS' if all_pass else 'FAILURES'}")
    if not all_pass:
        sys.exit(1)
    print("\n## Performance")
    for plans in [[4, 4, 4, 4], [25, 19, 32, 16], [215, 215, 215, 215]]:
        perf(f"plans={plans}", plans, R, D)


if __name__ == "__main__":
    main()
