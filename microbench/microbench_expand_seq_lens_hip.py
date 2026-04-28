"""Microbench / correctness gate for the HIP expand_seq_lens kernel.

Compares the new HIP kernel bit-equally against the CPU reference (the
Python double-loop in paged_prefill.expand_seq_lens).

Run with PYTHONPATH set to the PR tree:
    cd /sgl-pr && PYTHONPATH=/sgl-pr/python python3 microbench/microbench_expand_seq_lens_hip.py
"""
from __future__ import annotations

import time
from typing import List, Tuple

import torch

from sglang.srt.layers.attention.compressed.paged_prefill import expand_seq_lens
from sglang.srt.layers.expand_seq_lens_hip import hip_expand_seq_lens

DEVICE = torch.device("cuda:0")


def cpu_reference(seq_lens: List[int], extend_seq_lens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    return expand_seq_lens(
        seq_lens=seq_lens,
        extend_seq_lens=extend_seq_lens,
        device=DEVICE,
    )


def hip_path(seq_lens: List[int], extend_seq_lens: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    ext_t = torch.tensor(extend_seq_lens, dtype=torch.int32, device=DEVICE)
    extend_num_tokens = sum(extend_seq_lens)
    return hip_expand_seq_lens(seq_lens_t, ext_t, extend_num_tokens)


CASES = [
    ("smoke_1req_1024",            [1024],          [1024]),
    ("prefill_5req_1024",          [1024]*5,        [1024]*5),
    ("prefill_3req_1024",          [1024]*3,        [1024]*3),
    ("prefill_mixed",              [1024, 512, 768, 256], [1024, 512, 768, 256]),
    ("c1_single_768",              [768],           [768]),
    ("small_qo_kv_window_overlap", [200],           [100]),
    ("many_small_8reqs",           [128]*8,         [64]*8),
    ("kv_gt_qo_with_prefix",       [1500, 1500],    [256, 256]),
    ("qo_eq_kv_no_prefix",         [512, 512, 512], [512, 512, 512]),
    ("single_token",               [1],             [1]),
    ("long_kv_short_qo",           [2000],          [50]),
]


def main():
    print("=== HIP expand_seq_lens microbench ===")
    print(f"device={DEVICE}\n")
    print(f"{'case':36s} {'CPU(ms)':>9s} {'HIP(ms)':>9s} {'speedup':>8s} {'status':>10s}")
    pass_n, fail_n = 0, 0

    for label, sl, el in CASES:
        torch.cuda.synchronize()
        t0 = time.time()
        ref_a, ref_b = cpu_reference(sl, el)
        torch.cuda.synchronize()
        cpu_t = (time.time() - t0) * 1e3

        torch.cuda.synchronize()
        t0 = time.time()
        try:
            out_a, out_b = hip_path(sl, el)
            torch.cuda.synchronize()
        except Exception as e:
            print(f"{label:36s} {cpu_t:9.2f} {'RAISED':>9s}    -    {'FAIL':>10s}  {e}")
            fail_n += 1
            continue
        hip_t = (time.time() - t0) * 1e3

        ok_a = torch.equal(ref_a.to(torch.int32), out_a)
        ok_b = torch.equal(ref_b.to(torch.int32), out_b)
        speedup = cpu_t / hip_t if hip_t > 0 else float("inf")
        if ok_a and ok_b:
            print(f"{label:36s} {cpu_t:9.2f} {hip_t:9.2f} {speedup:7.2f}x {'PASS':>10s}")
            pass_n += 1
        else:
            print(f"{label:36s} {cpu_t:9.2f} {hip_t:9.2f} {speedup:7.2f}x {'FAIL':>10s}  "
                  f"(seq_lens_expanded ok={ok_a}, idx ok={ok_b})")
            for r, o, name in ((ref_a, out_a, "seq_lens_expanded"), (ref_b, out_b, "idx")):
                if not torch.equal(r.to(torch.int32), o):
                    diff = (r.to(torch.int32) != o)
                    idxs = diff.nonzero()[:3].flatten().tolist()
                    for k in idxs:
                        print(f"      {name}: ref[{k}]={r[k].item()} vs out[{k}]={o[k].item()}")
            fail_n += 1

    print(f"\n=== {pass_n} PASS / {fail_n} FAIL ===")

    if pass_n == len(CASES):
        # Stress: repeat 50× at production shape
        print("\n=== STRESS: 50× at production (5 reqs / 5120 tokens) ===")
        sl_t = torch.tensor([1024]*5, dtype=torch.int32, device=DEVICE)
        el_t = torch.tensor([1024]*5, dtype=torch.int32, device=DEVICE)
        for _ in range(50):
            hip_expand_seq_lens(sl_t, el_t, 5120)
        torch.cuda.synchronize()
        print("  50 iters OK")

        # Perf profile
        print("\n=== PERF: 200-iter median latency (5 reqs / 5120 tokens) ===")
        for _ in range(20):
            hip_expand_seq_lens(sl_t, el_t, 5120)
        torch.cuda.synchronize()
        ts = []
        for _ in range(200):
            torch.cuda.synchronize()
            t0 = time.time()
            hip_expand_seq_lens(sl_t, el_t, 5120)
            torch.cuda.synchronize()
            ts.append((time.time() - t0) * 1e3)
        ts.sort()
        print(f"  median: {ts[100]*1000:7.2f} us")
        print(f"  p10/p50/p90: {ts[20]*1000:.2f} / {ts[100]*1000:.2f} / {ts[180]*1000:.2f} us")


if __name__ == "__main__":
    main()
