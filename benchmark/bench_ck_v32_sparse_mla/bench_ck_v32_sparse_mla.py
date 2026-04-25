"""Bench: CK Tile FP8 sparse MLA decode (V32) vs asm `.co` and Triton fallback.

Sweeps the typical decode B × topk grid for DSv4-Pro and prints wall-clock
timings + ratios. Useful for nightly perf-tracking — failures here mean
either the kernel regressed or the asm baseline shifted.

Usage (chi2811 / parity_smoketest container):
    PYTHONPATH=/mnt/vast/john/sglang_v4_pr/python:/mnt/vast/john/aiter_workspace/aiter-amd \\
        python3 benchmark/bench_ck_v32_sparse_mla/bench_ck_v32_sparse_mla.py
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    if not (torch.version.hip and torch.version.hip != ""):
        return False
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950")
    except Exception:
        return False


def time_call(fn, n_warmup: int = 10, n_timed: int = 80) -> float:
    """Median wall-clock time in microseconds."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_timed):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)  # us
    times.sort()
    return times[len(times) // 2]


def run_one(B: int, H: int, topk: int, d_qk: int = 576, d_v: int = 512):
    """Returns dict with timings (µs) for ck_adapter / asm / triton, plus ratios.

    `d_qk` selects the production DSv4 shape:
      - 576 (= 512 nope + 64 rope)  Pro V32 / V32 mode
      - 512 (= 512 nope, no rope)   Flash 2604 mode
    `d_v` is the value-projection dim (currently always 512).
    """
    from sglang.srt.flashmla_tests.ref import _sparse_attn_decode_inner
    from sglang.srt.layers.attention.ck_v32_sparse_mla import (
        ck_sparse_mla_decode_fp8_v32,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    D_NOPE, D_TOTAL = d_v, d_qk
    S_q = 1
    n_kv = max(topk * 2, 1024)
    sm_scale = 1.0 / math.sqrt(D_TOTAL)
    fp8 = torch.float8_e4m3fnuz

    q_bf16 = torch.randn((B, S_q, H, D_TOTAL), device=device).to(torch.bfloat16)
    kv_pool_bf16 = torch.randn((n_kv, D_TOTAL), device=device).to(torch.bfloat16)
    kv_pool_fp8 = kv_pool_bf16.to(fp8).contiguous()
    k_cache = kv_pool_fp8.view(n_kv, 1, 1, D_TOTAL)
    indices = (
        torch.arange(topk, device=device, dtype=torch.int32)
        .view(1, 1, topk).expand(B, S_q, topk).contiguous()
    )
    invalid_mask = torch.zeros((B * S_q, topk), dtype=torch.bool, device=device)

    # ---- CK adapter ----
    def call_ck():
        ck_sparse_mla_decode_fp8_v32(
            q=q_bf16, k_cache=k_cache, indices=indices,
            invalid_mask=invalid_mask, attn_sink=None, sm_scale=sm_scale,
        )
    call_ck()
    torch.cuda.synchronize()
    ck_us = time_call(call_ck)

    # ---- Triton fallback (BF16 dequant + sparse attn) ----
    kv_pool_bf16_dq = kv_pool_fp8.to(torch.bfloat16).contiguous()
    q_f32 = q_bf16.float().view(B * S_q, H, D_TOTAL)
    kv_indices_flat = indices.view(-1).long()

    def call_tri():
        gathered = kv_pool_bf16_dq.float()[kv_indices_flat].view(
            B * S_q, topk, D_TOTAL
        )
        _sparse_attn_decode_inner(
            q_f32, gathered, invalid_mask, None,
            float(sm_scale), int(D_NOPE), int(H),
        )
    tri_us = time_call(call_tri)

    # ---- asm `.co` (optional — only if aiter is available) ----
    asm_us = None
    try:
        import aiter
        import aiter.mla as aiter_mla
        from aiter import dtypes

        fp8_asm = dtypes.fp8
        kv_pool_fp8_asm = kv_pool_bf16.to(fp8_asm).contiguous()
        kv_view_asm = kv_pool_fp8_asm.view(n_kv, 1, 1, D_TOTAL)
        q_fp8 = q_bf16.to(fp8_asm)
        qo_indptr_a = torch.tensor(
            [i * S_q for i in range(B + 1)], dtype=torch.int32, device=device,
        )
        kv_indptr_a = torch.tensor(
            [i * topk for i in range(B + 1)], dtype=torch.int32, device=device,
        )
        kv_indices_flat32 = indices.view(-1).to(torch.int32)
        kv_last_page_lens = torch.ones(B, dtype=torch.int32, device=device)
        max_split = 16
        cu_num = torch.cuda.get_device_properties(0).multi_processor_count
        max_split = min((cu_num + B - 1) // B, max_split)
        sizes = aiter.get_mla_metadata_info_v1(
            B, S_q, H, dtypes.fp8, dtypes.fp8,
            is_sparse=False, fast_mode=True, num_kv_splits=max_split,
        )
        md = [
            torch.empty(s if isinstance(s, int) else s, dtype=dt, device=device)
            if isinstance(s, int)
            else torch.empty(s, dtype=dt, device=device)
            for (s, dt) in sizes
        ]
        aiter.get_mla_metadata_v1(
            qo_indptr_a, kv_indptr_a, kv_last_page_lens, H, S_q, False,
            md[0], md[2], md[1], md[3], md[4], md[5],
            page_size=1, kv_granularity=16,
            max_seqlen_qo=S_q, uni_seqlen_qo=S_q, fast_mode=True,
            topk=-1, max_split_per_batch=max_split,
            dtype_q=dtypes.fp8, dtype_kv=dtypes.fp8,
        )
        out_asm = torch.empty((B * S_q, H, D_NOPE), dtype=torch.bfloat16, device=device)
        q_scale_t = torch.ones([1], dtype=torch.float, device=device)
        kv_scale_t = torch.ones([1], dtype=torch.float, device=device)

        def call_asm():
            aiter_mla.mla_decode_fwd(
                q_fp8, kv_view_asm, out_asm,
                qo_indptr_a, kv_indptr_a, kv_indices_flat32, kv_last_page_lens,
                S_q, 1, 1, sm_scale,
                num_kv_splits=max_split,
                q_scale=q_scale_t, kv_scale=kv_scale_t,
                work_meta_data=md[0], work_indptr=md[1], work_info_set=md[2],
                reduce_indptr=md[3], reduce_final_map=md[4], reduce_partial_map=md[5],
                intra_batch_mode=False, return_lse=False,
            )
        asm_us = time_call(call_asm)
    except Exception as exc:
        print(f"  (asm `.co` bench skipped: {type(exc).__name__}: {exc})", flush=True)

    return {
        "B": B, "H": H, "topk": topk, "d_qk": d_qk, "d_v": d_v,
        "ck_us": ck_us, "asm_us": asm_us, "tri_us": tri_us,
    }


_SHAPES_DEFAULT = [
    (B, topk) for B in [1, 2, 4, 8] for topk in [256, 512, 1024, 2048]
]

# Production DSv4 (d_qk, d_v) shapes — both go through the same templated CK kernel.
_DQK_DV_DEFAULT = [(576, 512), (512, 512)]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", default=None,
                        help="comma-separated 'B,topk' pairs; default sweeps "
                             "B∈{1,2,4,8} × topk∈{256,512,1024,2048}")
    parser.add_argument("--dqk", default=None,
                        help="comma-separated d_qk values to sweep (default: "
                             "576,512 — Pro V32 + Flash 2604)")
    parser.add_argument("--csv", default=None, help="optional CSV output path")
    args = parser.parse_args(argv)

    if not _is_gfx950():
        print("error: requires gfx950 (AMD MI355X). Skipping.", file=sys.stderr)
        return 2

    if args.shapes is None:
        shapes = _SHAPES_DEFAULT
    else:
        shapes = []
        for tok in args.shapes.split(";"):
            B, topk = (int(x) for x in tok.split(","))
            shapes.append((B, topk))

    if args.dqk is None:
        dqk_dv_list = _DQK_DV_DEFAULT
    else:
        dqk_dv_list = [(int(x), 512) for x in args.dqk.split(",")]

    rows = []
    for d_qk, d_v in dqk_dv_list:
        print(f"\n=== d_qk={d_qk} d_v={d_v} "
              f"({'Pro V32' if d_qk == 576 else 'Flash 2604'}) ===", flush=True)
        print(
            f"{'B':>3} {'topk':>5} | {'CK':>7} {'asm':>7} {'tri':>7} | "
            f"{'CK/asm':>7} {'CK/tri':>7}",
            flush=True,
        )
        for B, topk in shapes:
            r = run_one(B, 128, topk, d_qk=d_qk, d_v=d_v)
            ck_s = f"{r['ck_us']:6.1f}u"
            asm_s = f"{r['asm_us']:6.1f}u" if r["asm_us"] is not None else "      -"
            tri_s = f"{r['tri_us']:6.1f}u"
            ck_asm = f"{r['asm_us'] / r['ck_us']:6.2f}x" if r["asm_us"] is not None else "      -"
            ck_tri = f"{r['tri_us'] / r['ck_us']:6.2f}x"
            print(f"{r['B']:>3} {r['topk']:>5} | {ck_s} {asm_s} {tri_s} | {ck_asm} {ck_tri}",
                  flush=True)
            rows.append(r)

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(
                fh,
                fieldnames=["B", "H", "topk", "d_qk", "d_v", "ck_us", "asm_us", "tri_us"],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
