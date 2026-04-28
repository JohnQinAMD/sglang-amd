"""Microbench / correctness gate for the HIP make_swa_indices kernel.

Compares the new HIP kernel bit-equally against the CPU reference (the Python
double-loop in `make_swa_ring_buffer_indices`). Production shapes mirror the
Phase 8.2 hang scenario (which we now know was caused by the broken TileLang
kernel emitting an empty function body).

Run with PYTHONPATH set to the PR tree:
    cd /sgl-pr && PYTHONPATH=/sgl-pr/python python3 microbench/microbench_swa_indices_hip.py
"""
from __future__ import annotations

import time

import torch

from sglang.srt.layers.swa_indices_hip import hip_make_swa_prefill_indices

DEVICE = torch.device("cuda:0")
SWA_WINDOW = 128
MAX_SEQ_LEN = 2048


def cpu_reference(seq_lens_list, extend_lens_list, swa_window_size: int) -> torch.Tensor:
    """Mirror of make_swa_ring_buffer_indices CPU path in paged_prefill.py."""
    SWA_W = swa_window_size
    extend_num_tokens = sum(extend_lens_list)
    batch_size = len(seq_lens_list)
    swa_indices = torch.full(
        (extend_num_tokens, SWA_W), -1, dtype=torch.int32, device="cpu"
    )
    cum_qo_len = 0
    abs_pos_buf = torch.arange(MAX_SEQ_LEN, dtype=torch.int32)
    for seq_idx, (kv_len, qo_len) in enumerate(zip(seq_lens_list, extend_lens_list)):
        old_kv_start = seq_idx * SWA_W
        new_kv_start = batch_size * SWA_W + cum_qo_len
        prefix_len = kv_len - qo_len
        for curr_seq_qo_idx in range(qo_len):
            end_abs_pos = prefix_len + curr_seq_qo_idx + 1
            start_abs_pos = max(end_abs_pos - SWA_W, 0)
            chosen = abs_pos_buf[start_abs_pos:end_abs_pos]
            torch.where(
                chosen < prefix_len,
                old_kv_start + chosen % SWA_W,
                new_kv_start + (chosen - prefix_len),
                out=swa_indices[
                    cum_qo_len + curr_seq_qo_idx, : end_abs_pos - start_abs_pos
                ],
            )
        cum_qo_len += qo_len
    return swa_indices.to(DEVICE)


def hip_path(seq_lens_list, extend_lens_list, swa_window_size: int) -> torch.Tensor:
    extend_num_tokens = sum(extend_lens_list)
    seq_lens_k = torch.tensor(seq_lens_list, dtype=torch.int32, device=DEVICE)
    seq_lens_q = torch.tensor(extend_lens_list, dtype=torch.int32, device=DEVICE)
    # Sentinel pre-fill so we can detect un-written cells.
    swa_indices = torch.full(
        (extend_num_tokens, swa_window_size), -99, dtype=torch.int32, device=DEVICE
    )
    return hip_make_swa_prefill_indices(
        seq_lens_k=seq_lens_k,
        seq_lens_q=seq_lens_q,
        swa_indices=swa_indices,
    )


CASES = [
    ("smoke_1req_1024",                 [1024],          [1024]),
    ("prefill_5req_1024",               [1024]*5,        [1024]*5),
    ("prefill_3req_1024",               [1024]*3,        [1024]*3),
    ("prefill_mixed",                   [1024, 512, 768, 256], [1024, 512, 768, 256]),
    ("c1_single_768",                   [768],           [768]),
    ("small_qo_kv_window_overlap",      [200],           [100]),
    ("many_small_8reqs",                [128]*8,         [64]*8),
    ("kv_gt_qo_with_prefix",            [1500, 1500],    [256, 256]),
    ("qo_eq_kv_no_prefix",              [512, 512, 512], [512, 512, 512]),
    ("single_token",                    [1],             [1]),
    ("aligned_window_boundary",         [128, 128],      [128, 128]),
    ("long_kv_short_qo",                [2000],          [50]),
]


def main():
    print("=== HIP make_swa_indices microbench ===")
    print(f"SWA_WINDOW={SWA_WINDOW}, device={DEVICE}\n")
    print(f"{'case':36s} {'CPU(ms)':>9s} {'HIP(ms)':>9s} {'speedup':>8s} {'status':>10s}")
    pass_n, fail_n = 0, 0

    for label, seq_lens, extend_lens in CASES:
        # CPU reference
        torch.cuda.synchronize()
        t0 = time.time()
        ref = cpu_reference(seq_lens, extend_lens, SWA_WINDOW)
        torch.cuda.synchronize()
        cpu_t = (time.time() - t0) * 1e3

        # HIP
        torch.cuda.synchronize()
        t0 = time.time()
        try:
            out = hip_path(seq_lens, extend_lens, SWA_WINDOW)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"{label:36s} {cpu_t:9.2f} {'RAISED':>9s}    -    {'FAIL':>10s}  {e}")
            fail_n += 1
            continue
        hip_t = (time.time() - t0) * 1e3

        # Sentinel check (catches the TileLang-style empty-kernel bug)
        sentinel_remaining = (out == -99).sum().item()
        if sentinel_remaining > 0:
            print(f"{label:36s} {cpu_t:9.2f} {hip_t:9.2f} {'-':>8s} {'EMPTY':>10s}  "
                  f"({sentinel_remaining}/{out.numel()} cells unwritten)")
            fail_n += 1
            continue

        eq = torch.equal(ref, out)
        speedup = cpu_t / hip_t if hip_t > 0 else float("inf")
        if eq:
            print(f"{label:36s} {cpu_t:9.2f} {hip_t:9.2f} {speedup:7.2f}x {'PASS':>10s}")
            pass_n += 1
        else:
            diff_n = (ref != out).sum().item()
            print(f"{label:36s} {cpu_t:9.2f} {hip_t:9.2f} {speedup:7.2f}x {'FAIL':>10s}  "
                  f"({diff_n}/{ref.numel()} mismatches)")
            mask = (ref != out)
            for r, c in mask.nonzero()[:3].tolist():
                print(f"      ref[{r},{c}]={ref[r,c].item()} vs out[{r},{c}]={out[r,c].item()}")
            fail_n += 1

    print(f"\n=== {pass_n} PASS / {fail_n} FAIL ===")

    if pass_n == len(CASES):
        # Stress: repeated calls to ensure stability under the Phase 8.2 pattern
        print("\n=== STRESS: repeat Phase 8.2 hang pattern 50× ===")
        for i in range(50):
            hip_path([1024]*5, [1024]*5, SWA_WINDOW)
            hip_path([1024]*3, [1024]*3, SWA_WINDOW)
        torch.cuda.synchronize()
        print("  50 iters OK")

        # Perf profile at production shape
        print("\n=== PERF: 200-iter median latency (5 reqs, 5120 tokens) ===")
        seq_k = torch.tensor([1024]*5, dtype=torch.int32, device=DEVICE)
        seq_q = torch.tensor([1024]*5, dtype=torch.int32, device=DEVICE)
        buf = torch.empty((5120, SWA_WINDOW), dtype=torch.int32, device=DEVICE)
        # Warmup
        for _ in range(20):
            hip_make_swa_prefill_indices(seq_lens_k=seq_k, seq_lens_q=seq_q, swa_indices=buf)
        torch.cuda.synchronize()
        ts = []
        for _ in range(200):
            torch.cuda.synchronize()
            t0 = time.time()
            hip_make_swa_prefill_indices(seq_lens_k=seq_k, seq_lens_q=seq_q, swa_indices=buf)
            torch.cuda.synchronize()
            ts.append((time.time() - t0) * 1e3)
        ts.sort()
        print(f"  median: {ts[100]*1000:7.2f} us")
        print(f"  p10/p50/p90: {ts[20]*1000:.2f} / {ts[100]*1000:.2f} / {ts[180]*1000:.2f} us")
        print(f"  vs CPU reference 5req: ~65 ms — speedup ~{65000/ts[100]:.0f}×")


if __name__ == "__main__":
    main()
