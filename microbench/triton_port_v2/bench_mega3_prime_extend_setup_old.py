"""v2 microbench: MEGA-3' Stage 1 KVAndScoreOld variant — for the production
OLD compressor path (separate kv/score tensors)."""
import sys

sys.path.insert(0, "/sgl-pr/python")

import time
import torch

from sglang.jit_kernel.extend_per_request_megakernel_triton import (
    mega3_prime_extend_setup_old_triton,
)


SHAPES = [
    # (num_reqs_in_pool, bs, T_state, D_channel, label, list of (prefix_len, extend_len))
    (1024, 1, 8, 256, "single small extend",
     [(0, 256)]),
    (1024, 4, 8, 256, "decode bs=4 with prefill mix",
     [(0, 100), (32, 64), (128, 128), (0, 256)]),
    (2048, 4, 8, 256, "chunked-prefill bs=4 large extents",
     [(0, 856), (0, 768), (0, 600), (0, 588)]),
]


def _compute_state_len_overlap_ratio4(seq_len: int) -> int:
    return seq_len % 4 + 4


def torch_reference_old(
    kv_states, score_states,
    kv_input, score_input,
    temp_buffer_kv, temp_buffer_score,
    req_pool_indices, prefix_lens, extend_lens,
):
    """Mirror compress_extend_old:1102-1132 setup steps for KVAndScoreOld."""
    bs = req_pool_indices.shape[0]
    pt = 0
    buf_off = 0
    for i in range(bs):
        req = int(req_pool_indices[i])
        prefix_len = int(prefix_lens[i])
        extend_len = int(extend_lens[i])

        pre_state_len = _compute_state_len_overlap_ratio4(prefix_len)
        post_state_len = _compute_state_len_overlap_ratio4(prefix_len + extend_len)
        valid_kv_len = pre_state_len + extend_len

        if prefix_len == 0:
            kv_states[req].zero_()
            score_states[req].fill_(float("-inf"))

        # cat prefix-state
        temp_buffer_kv[buf_off:buf_off + pre_state_len] = kv_states[req, :pre_state_len]
        temp_buffer_score[buf_off:buf_off + pre_state_len] = score_states[req, :pre_state_len]

        # cat current-kv
        temp_buffer_kv[buf_off + pre_state_len:buf_off + valid_kv_len] = kv_input[pt:pt + extend_len]
        temp_buffer_score[buf_off + pre_state_len:buf_off + valid_kv_len] = score_input[pt:pt + extend_len]

        # write-back state
        kv_states[req, :post_state_len] = temp_buffer_kv[buf_off + valid_kv_len - post_state_len:buf_off + valid_kv_len]
        score_states[req, :post_state_len] = temp_buffer_score[buf_off + valid_kv_len - post_state_len:buf_off + valid_kv_len]

        pt += extend_len
        buf_off += valid_kv_len


def make_inputs(num_reqs, bs, T_state, D_channel, plans, dtype=torch.bfloat16, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    kv_states = torch.randn(num_reqs, T_state, D_channel, dtype=dtype, device=device) * 0.1
    score_states = torch.randn(num_reqs, T_state, D_channel, dtype=dtype, device=device) * 0.1
    total_tokens = sum(extend for _, extend in plans)
    kv_input = torch.randn(total_tokens, D_channel, dtype=dtype, device=device) * 0.1
    score_input = torch.randn(total_tokens, D_channel, dtype=dtype, device=device) * 0.1
    max_buf = sum(_compute_state_len_overlap_ratio4(p) + e for p, e in plans)
    temp_buffer_kv = torch.empty(max_buf + 16, D_channel, dtype=dtype, device=device)
    temp_buffer_score = torch.empty(max_buf + 16, D_channel, dtype=dtype, device=device)
    req_pool_indices = torch.randperm(num_reqs, device=device)[:bs].to(torch.int32)
    prefix_lens = torch.tensor([p for p, _ in plans], dtype=torch.int32, device=device)
    extend_lens = torch.tensor([e for _, e in plans], dtype=torch.int32, device=device)
    return (kv_states, score_states, kv_input, score_input,
            temp_buffer_kv, temp_buffer_score,
            req_pool_indices, prefix_lens, extend_lens)


def correctness(num_reqs, bs, T_state, D_channel, name, plans):
    print(f"\n=== {name} (bs={bs}, plans={plans}) ===")
    args_ref = make_inputs(num_reqs, bs, T_state, D_channel, plans)
    args_tri = tuple(a.clone() if isinstance(a, torch.Tensor) else a for a in args_ref)

    torch_reference_old(*args_ref)
    result = mega3_prime_extend_setup_old_triton(*args_tri)
    if result is None:
        print(f"  Stage 1 returned None. FAIL.")
        return False

    def safe_diff(a, b):
        af, bf = a.float(), b.float()
        both_pos_inf = torch.isposinf(af) & torch.isposinf(bf)
        both_neg_inf = torch.isneginf(af) & torch.isneginf(bf)
        both_inf = both_pos_inf | both_neg_inf
        diff = (af - bf).abs()
        diff = torch.where(both_inf, torch.zeros_like(diff), diff)
        both_nan = torch.isnan(af) & torch.isnan(bf)
        diff = torch.where(both_nan, torch.zeros_like(diff), diff)
        diff = torch.nan_to_num(diff, nan=1e10, posinf=1e10, neginf=1e10)
        return diff

    total_buf_used = sum(_compute_state_len_overlap_ratio4(p) + e for p, e in plans)

    diffs = {
        "kv_states": safe_diff(args_ref[0], args_tri[0]),
        "score_states": safe_diff(args_ref[1], args_tri[1]),
        "temp_buffer_kv": safe_diff(args_ref[4][:total_buf_used], args_tri[4][:total_buf_used]),
        "temp_buffer_score": safe_diff(args_ref[5][:total_buf_used], args_tri[5][:total_buf_used]),
    }
    PASS = True
    for name_, d in diffs.items():
        m = d.max().item()
        print(f"  {name_:<22} max={m:.4e} mean={d.mean().item():.4e}")
        if m > 1e-3:
            PASS = False
    print(f"  RESULT: {'PASS' if PASS else 'FAIL'}")
    return PASS


def perf(num_reqs, bs, T_state, D_channel, name, plans):
    print(f"\n--- perf {name} ---")
    args = make_inputs(num_reqs, bs, T_state, D_channel, plans)

    def call_torch():
        torch_reference_old(*args)

    def call_triton():
        mega3_prime_extend_setup_old_triton(*args)

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

    print(f"  eager: torch loop = {t_torch:>8.1f} us   triton stage1-old = {t_triton:>8.1f} us   speedup = {t_torch/t_triton:.2f}x")


def main():
    print("=" * 78)
    print("MEGA-3' Stage 1 KVAndScoreOld variant: vs torch loop")
    print("=" * 78)
    all_pass = True
    for shape in SHAPES:
        ok = correctness(*shape)
        if not ok:
            all_pass = False
    print(f"\nG0 SUMMARY: {'ALL PASS' if all_pass else 'FAILURES'}")
    if not all_pass:
        sys.exit(1)
    print("\n## Performance")
    for shape in SHAPES:
        perf(*shape)


if __name__ == "__main__":
    main()
