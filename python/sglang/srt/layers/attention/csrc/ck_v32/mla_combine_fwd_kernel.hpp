// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// CK-tile MLA Combine Forward Kernel — Phase A two-source online-softmax merge
//
// Mirrors NVIDIA `flash_fwd_mla_combine_kernel<bf16, V=512, ...>` for sparse MLA
// decode. Replaces the pre-Phase-A pipeline of (2× CK splitkv + 2× aiter
// mla_reduce_v1 + Triton merge + Triton sink_fold) with a single CK launch
// that consumes both sources' (split_data, split_lse) tensors natively (with
// strides, no `.contiguous()` calls), does N-way online-softmax merge in-place,
// and optionally fuses the attn-sink fold.
//
// Math (per (qo, head) row, across all S_a+S_b splits):
//
//     lm  = max_i(lse_i)
//     w_i = exp(lse_i - lm)                          # may be 0 if lse_i == -inf
//     sw  = sum(w_i)
//     out = sum_i(w_i * split_data_i) / sw           # split_data is pre-normalized
//     lse = lm + log(sw)
//     # if attn_sink: out *= 1 / (1 + exp(sink[h] - lse))
//
// Lonely-Q rows (all splits invalid → sw == 0): out = 0, lse = -inf.
//
// LSE units are NATURAL LOG (matches `mla_decode_fwd_kernel` line 471 which
// writes `rmax * kLn2 + log(rsum)` — kLn2 converts the kernel's softmax-base-2
// rmax to the natural-log units that all combine consumers expect).
//
// Grid:  (total_q * H, 1, 1)
// Block: BLOCK_THREADS = 256 (4 warps × 64 lanes; each thread strides V).
//
// Per-block work (V=512, default S_a+S_b ≤ 16 splits):
//   - Read S_total fp32 LSE values × 2 sources from HBM (~128 B per block)
//   - Find global max via per-thread independent recompute (no warp reduce)
//   - Compute exp() of weights and sum
//   - Stream V values: each thread processes V / BLOCK_THREADS = 2 elements
//   - Single bf16 write per V element + 1 fp32 LSE write per block

#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cmath>

namespace ck_mla_sparse_fp8 {

// Maximum total splits supported per call (S_a + S_b). cuda-graph capture
// at bs=1 with high topk hits 64+ splits (pick_num_splits returns up to
// `splits_by_total = 512 / (HEAD_GROUPS=4 × B=1)` = 128, capped by
// `splits_by_topk = topk/BLOCK_N=32`). Production observed S_a+S_b=68 at
// graph capture. 128 leaves comfortable headroom. Per-split arrays live in
// shared memory (not registers) so larger values cost LDS, not VGPRs.
static constexpr int COMBINE_MAX_SPLITS = 128;

// Block-shape constants. 256 threads × 2 V-elements per thread = 512 lane-slots
// per block, matching V_HEAD_DIM = 512. If V_HEAD_DIM ever changes, adjust
// V_PER_THREAD accordingly.
static constexpr int COMBINE_BLOCK_THREADS = 256;

struct MlaCombineArgs {
    // Source A — main KV scope splits. [total_q, S_a, H, V] fp32.
    const float*   split_data_a;
    // [total_q, S_a, H, 1] fp32.
    const float*   split_lse_a;
    int            S_a;
    int64_t        stride_sda_q, stride_sda_s, stride_sda_h;     // strides into split_data_a (V is innermost, stride 1)
    int64_t        stride_lsa_q, stride_lsa_s, stride_lsa_h;     // strides into split_lse_a

    // Source B — extra KV scope splits. May be null+S_b=0 for single-source
    // path (then this becomes a plain N-way reduce equivalent to mla_reduce_v1).
    const float*   split_data_b;
    const float*   split_lse_b;
    int            S_b;
    int64_t        stride_sdb_q, stride_sdb_s, stride_sdb_h;
    int64_t        stride_lsb_q, stride_lsb_s, stride_lsb_h;

    // Outputs.
    __bf16*        out_ptr;          // [total_q, H, V] bf16
    int64_t        stride_out_q, stride_out_h;
    float*         lse_ptr;          // [total_q, H] fp32
    int64_t        stride_lse_q, stride_lse_h;

    // Optional attn_sink fold: if non-null, out *= 1 / (1 + exp(sink[h] - lse)).
    // [H] fp32.
    const float*   attn_sink;        // null → no sink fold

    int            total_q;
    int            H;
    int            V;                // == V_HEAD_DIM (512). Compile-time constant in template.
};

// Helper: load lse value from a strided [total_q, S, H, 1] fp32 tensor.
__device__ __forceinline__ float
combine_load_lse(const float* base, int q_idx, int s_idx, int h_idx,
                 int64_t stride_q, int64_t stride_s, int64_t stride_h)
{
    return base[q_idx * stride_q + s_idx * stride_s + h_idx * stride_h];
}

// Combine kernel template. V_HEAD_DIM_T=512 for both V32 and 2604 modes.
//
// Each block handles one (q_idx, h_idx) tile. Block has BLOCK_THREADS lanes;
// V_HEAD_DIM_T / BLOCK_THREADS V elements per lane (= 2 at V=512, BT=256).
//
// Per block:
//   1. Load all S_a + S_b LSE values for this (q, h). Each thread does the
//      same loads — fully redundant, but the values are tiny (≤ 32 fp32 = 128 B
//      L1-cached after first load) and avoiding sync between LSE-reduce and
//      V-reduce keeps the kernel branch-free.
//   2. Compute lm = max(lse_i) and w_i = exp(lse_i - lm), sw = sum(w_i).
//   3. Each thread reads its V-slice, accumulates sum(w_i * split_data_i),
//      writes to out_ptr in bf16.
//   4. Lane 0 writes merged LSE.
//   5. (Optional) Apply attn_sink fold inline.
template <int V_HEAD_DIM_T>
__global__ void __launch_bounds__(COMBINE_BLOCK_THREADS, 4)
mla_combine_fwd_kernel(MlaCombineArgs args)
{
    constexpr int BT = COMBINE_BLOCK_THREADS;
    constexpr int V_PER_THREAD = V_HEAD_DIM_T / BT;
    static_assert(V_HEAD_DIM_T % BT == 0,
                  "V_HEAD_DIM must be divisible by BLOCK_THREADS");
    static_assert(V_PER_THREAD >= 1 && V_PER_THREAD <= 8,
                  "V_PER_THREAD out of expected range");

    const int qh = blockIdx.x;                     // flat (q_idx, h_idx)
    const int q_idx = qh / args.H;
    const int h_idx = qh % args.H;
    if (q_idx >= args.total_q) return;
    const int tid = threadIdx.x;
    const int S_total = args.S_a + args.S_b;

    // ───── Stage 1: load LSE → shared mem, compute weights via thread 0 ─────
    //
    // Why shared memory: at MAX_SPLITS=128, per-thread arrays of fp32 weights
    // would cost 128*4 = 512 B/thread = 128 VGPRs of register pressure on
    // top of the V accumulator + scratch — pushing total VGPR usage past
    // the 256-VGPR sweet spot and dropping occupancy. Shared memory is
    // ~1 KB/block (negligible) and the V-loop only does broadcast reads
    // of `s_w[s]` (same LDS address across all 256 threads → 1 LDS read,
    // no bank conflict).
    __shared__ float s_w[COMBINE_MAX_SPLITS];     // pre-computed weights
    __shared__ float s_inv_sw;                    // 1 / sum_w (×sink_scale if sink)
    __shared__ float s_merged_lse;                // lm + log(sw), or -inf
    __shared__ int   s_all_invalid;               // 1 iff every split is -inf

    // Cooperative LSE load + weight compute. Every thread reads its own
    // lse value (no contention) and stores the WEIGHT directly into
    // shared memory. Then thread 0 reduces to find lm/sw and back-fixes
    // the weights; finally one syncthreads gates the V-loop.
    //
    // Two-stage approach:
    //  (a) tid in [0, S_total) loads its lse into s_w (temp slot).
    //  (b) thread 0 reads back, computes lm, overwrites s_w[i] with
    //      `exp(lse_i - lm)`, accumulates sw, computes meta-fields.
    //  This keeps stage (a) parallel and contention-free.
    if (tid < S_total) {
        float lse_val;
        if (tid < args.S_a) {
            lse_val = combine_load_lse(
                args.split_lse_a, q_idx, tid, h_idx,
                args.stride_lsa_q, args.stride_lsa_s, args.stride_lsa_h);
        } else {
            const int s_b = tid - args.S_a;
            lse_val = combine_load_lse(
                args.split_lse_b, q_idx, s_b, h_idx,
                args.stride_lsb_q, args.stride_lsb_s, args.stride_lsb_h);
        }
        s_w[tid] = lse_val;
    }
    __syncthreads();

    if (tid == 0) {
        // Find global max LSE.
        float lm = -INFINITY;
        for (int i = 0; i < S_total; ++i) {
            lm = fmaxf(lm, s_w[i]);
        }
        const bool all_invalid = !isfinite(lm);
        // Compute weights, sum.
        float sw = 0.0f;
        for (int i = 0; i < S_total; ++i) {
            const float d = s_w[i] - lm;
            // d == NaN iff lse_i == -inf and lm == -inf. Use d == d to detect.
            const float wi = (d == d) ? __expf(d) : 0.0f;
            s_w[i] = wi;
            sw += wi;
        }
        const float sw_safe = (sw > 0.0f) ? sw : 1.0f;
        float inv_sw = 1.0f / sw_safe;
        // Apply attn_sink fold by folding into inv_sw so the V-loop pays a
        // single multiplication per V element.
        const float merged_lse = all_invalid ? -INFINITY : lm + __logf(sw);
        if (args.attn_sink != nullptr && !all_invalid) {
            const float sink_h = args.attn_sink[h_idx];
            inv_sw *= 1.0f / (1.0f + __expf(sink_h - merged_lse));
        }
        s_inv_sw       = inv_sw;
        s_merged_lse   = merged_lse;
        s_all_invalid  = all_invalid ? 1 : 0;
        // Lane-0 also writes merged_lse to global (single fp32 store; doing
        // it here saves a separate `if (tid == 0)` block at the end).
        args.lse_ptr[q_idx * args.stride_lse_q + h_idx * args.stride_lse_h] = merged_lse;
    }
    __syncthreads();

    // ───── Stage 2: stream V values, accumulate weighted split_data ─────
    // Each thread handles V_PER_THREAD lanes spaced by BT for coalesced loads.
    // For V=512, BT=256, V_PER_THREAD=2: thread `tid` handles V indices
    // {tid, tid+BT}.
    //
    // Source-A and source-B base offsets per (q_idx, h_idx) — computed once
    // per block. Each split is at base + s * stride_s in the V dimension.
    const int64_t base_qh_a = q_idx * args.stride_sda_q + h_idx * args.stride_sda_h;
    const int64_t base_qh_b = (args.S_b > 0)
        ? (q_idx * args.stride_sdb_q + h_idx * args.stride_sdb_h)
        : 0;

    __bf16* out_row = args.out_ptr + q_idx * args.stride_out_q + h_idx * args.stride_out_h;
    const float inv_sw   = s_inv_sw;
    const bool  all_invd = s_all_invalid != 0;

    #pragma unroll
    for (int vi = 0; vi < V_PER_THREAD; ++vi) {
        const int v = tid + vi * BT;
        // V_HEAD_DIM_T is a compile-time constant; vi*BT + tid < V_HEAD_DIM_T
        // by construction (since V % BT == 0). No mask needed.

        float acc = 0.0f;
        // Source-A splits — `s_w[s]` is broadcast-read across all 256 threads
        // (same LDS address each iteration), so no bank-conflict cost.
        #pragma unroll 1
        for (int s = 0; s < args.S_a; ++s) {
            const float xi = args.split_data_a[
                base_qh_a + s * args.stride_sda_s + v];
            acc = fmaf(s_w[s], xi, acc);
        }
        // Source-B splits (if present).
        #pragma unroll 1
        for (int s = 0; s < args.S_b; ++s) {
            const float xi = args.split_data_b[
                base_qh_b + s * args.stride_sdb_s + v];
            acc = fmaf(s_w[args.S_a + s], xi, acc);
        }
        if (all_invd) acc = 0.0f;
        const float scaled = acc * inv_sw;
        out_row[v] = __float2bfloat16(scaled);
    }
}

} // namespace ck_mla_sparse_fp8
