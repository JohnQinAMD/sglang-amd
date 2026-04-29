// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// CK-tile MLA Decode Forward Kernel — FP8 sparse port (P0/P1)
//
// Derived from `ck_mla_decode/mla_decode_fwd_kernel.hpp` (bf16 dense). For P0
// we keep the LDS layout in bf16 and the existing bf16 MFMA path. The only
// dtype change is on the global KV side: the KV cache is now stored as
// fp8_e4m3fnuz (gfx950 native), and the KV-tile loader decodes fp8 -> bf16
// while staging from HBM into LDS. This halves HBM bandwidth (the dominant
// cost for memory-bound MLA decode) while keeping the validated MFMA output
// layout. Q remains bf16 for P0; we'll quantize Q to fp8 in P2 alongside the
// switch to native fp8 MFMA.
//
// Sparse: the existing args.kv_indices indirect gather (kv_base[pidx*stride])
// already implements per-row sparse access. The wrapper feeds the
// page_table/topk indices through kv_indices and sets kv_indptr=[0,topk,...]
// to express "this batch has `topk` valid KV rows from the indirect index
// array". Zero kernel changes for sparse beyond what the dense kernel
// already supports.

#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>

namespace ck_mla_sparse_fp8 {

static constexpr int WARP_SIZE   = 64;
// V_HEAD_DIM is shared by both production DSv4 attention shapes (V32 / 2604).
// QK_HEAD_DIM is templated below — see `mla_decode_fwd_kernel<QK_HEAD_DIM_T>`.
//   * V32  / Pro V32 mode  : QK_HEAD_DIM = 576 (d_qk = nope=512 + rope=64)
//   * 2604 / Flash-Base    : QK_HEAD_DIM = 512 (d_qk = nope=512, no rope)
static constexpr int V_HEAD_DIM  = 512;

typedef __attribute__((ext_vector_type(8))) __bf16 bf16x8_native;
typedef __attribute__((ext_vector_type(4))) float  float4_native;
typedef __attribute__((ext_vector_type(2))) float  float2_native;

struct bf16x8_t { unsigned int v[4]; };
struct float4_t {
    float v[4];
    __device__ float& operator[](int i) { return v[i]; }
    __device__ float  operator[](int i) const { return v[i]; }
};

__device__ __forceinline__ void
mfma_bf16_16x16x32(float4_t& c, bf16x8_t a, bf16x8_t b)
{
    float4_native cn = *reinterpret_cast<float4_native*>(&c);
    cn = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
        *reinterpret_cast<bf16x8_native*>(&a),
        *reinterpret_cast<bf16x8_native*>(&b),
        cn, 0, 0, 0);
    asm volatile("" : "+v"(cn));
    *reinterpret_cast<float4_native*>(&c) = cn;
}

__device__ __forceinline__ bf16x8_t
lds_load_bf16x8(const __bf16* p)
{
    bf16x8_t r;
    *reinterpret_cast<uint4*>(&r) = *reinterpret_cast<const uint4*>(p);
    return r;
}

// fp16 variants for the higher-precision PV path. fp16 has 10-bit mantissa
// (vs bf16's 7-bit) at the same total width — gain ~3 mantissa bits in P,
// eliminating the 60-layer-compounding precision drift that produced
// garbage tokens under SGLANG_HIP_CK_V32_SINGLESHOT=1. mfma_f32_16x16x32_f16
// has identical compute throughput to mfma_f32_16x16x32_bf16 on gfx950.
typedef __attribute__((ext_vector_type(8))) _Float16 f16x8_native;
struct f16x8_t { unsigned int v[4]; };  // same 16 bytes as bf16x8

__device__ __forceinline__ void
mfma_f16_16x16x32(float4_t& c, f16x8_t a, f16x8_t b)
{
    float4_native cn = *reinterpret_cast<float4_native*>(&c);
    cn = __builtin_amdgcn_mfma_f32_16x16x32_f16(
        *reinterpret_cast<f16x8_native*>(&a),
        *reinterpret_cast<f16x8_native*>(&b),
        cn, 0, 0, 0);
    asm volatile("" : "+v"(cn));
    *reinterpret_cast<float4_native*>(&c) = cn;
}

__device__ __forceinline__ f16x8_t
lds_load_f16x8(const _Float16* p)
{
    f16x8_t r;
    *reinterpret_cast<uint4*>(&r) = *reinterpret_cast<const uint4*>(p);
    return r;
}

// Convert bf16x8 → f16x8 in registers (lossless for fp8-decoded values:
// fp8 e4m3 has only 4 mantissa bits, so it fits in fp16's 10 bits with
// margin). Used to feed bf16 V from LDS into the fp16 PV mfma.
// fp32 mfma for the highest-precision PV path. Used when
// SGLANG_CK_V32_FP32_PV is defined at compile time. K=4 per call (vs fp16's
// K=32), so 8 calls per (qp, vl) tuple — ~8x compute increase on PV.
// Eliminates the bf16/fp16 mantissa loss that caused the 60-layer-compounded
// drift under SGLANG_HIP_CK_V32_SINGLESHOT=1 (production residual stream
// expects fp32-equivalent per-layer attention output).
//
// Per-lane operand layout for mfma_f32_16x16x4_f32:
//   A: 1 fp32 per lane, lane = M + K*16 (M=lane%16, K=lane/16)
//   B: 1 fp32 per lane, lane = N + K*16 (N=lane%16, K=lane/16)
//   C: 4 fp32 per lane, c=M%4, lane=N+(M/4)*16 (same as bf16/fp16 mfma)
__device__ __forceinline__ void
mfma_f32_16x16x4(float4_t& c, float a, float b)
{
    float4_native cn = *reinterpret_cast<float4_native*>(&c);
    cn = __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, cn, 0, 0, 0);
    asm volatile("" : "+v"(cn));
    *reinterpret_cast<float4_native*>(&c) = cn;
}

__device__ __forceinline__ f16x8_t
bf16x8_to_f16x8(bf16x8_t a)
{
    bf16x8_native an = *reinterpret_cast<bf16x8_native*>(&a);
    f16x8_t out;
    _Float16* op = reinterpret_cast<_Float16*>(&out);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        // bf16 → fp32 (cast) → fp16 (RTNE).
        float v = (float)an[i];
        op[i] = (_Float16)v;
    }
    return out;
}

// Decode 8 packed fp8 bytes (in two uint32_t lanes) to 8 bf16 values.
// Uses the gfx950 native fp8->f32 hardware converter (`v_cvt_pk_f32_fp8`).
//
// CALIBRATION (verified by `_fp8_decode_probe2.py` on gfx950):
// `cvt_pk_f32_fp8` reads bytes with the **e4m3fn exponent bias (7)**,
// matching `torch.float8_e4m3fn` exactly (0/256 mismatches). Bytes
// representing `2^N` decode as `2^(N+1)` from the perspective of
// `torch.float8_e4m3fnuz` (bias=8), i.e. 2× the fnuz value (252/256
// mismatches).
//
// Caller-supplied `decode_scale` reconciles the storage format vs the
// HW interpretation:
//   * KV stored as `torch.float8_e4m3fn` (MI355X gfx950 default,
//     `is_fp8_fnuz()=False`)         → decode_scale = **1.0** (no fold).
//   * KV stored as `torch.float8_e4m3fnuz` (MI300X gfx942 default, or
//     legacy testing setup that quantized as fnuz)
//                                       → decode_scale = **0.5** (legacy
//     value of `kFnuzBiasFix`; HW gives 2× fnuz, fold halves it).
//
// Symptom of using the wrong scale: kernel output has uniform magnitude
// error (cos_sim still high but absolute scale 0.5× or 2× off). On
// Flash 2604 (qk_head_dim=512) this surfaced as "London" instead of
// "Paris" greedy probes — the residual addition + downstream RMSnorm
// is sensitive to the absolute scale even though softmax is invariant.
struct fp8x8_t { uint32_t v[2]; };  // 8 fp8 bytes
__device__ __forceinline__ bf16x8_t
fp8x8_decode_to_bf16x8(fp8x8_t in, float decode_scale)
{
    float2_native lo0 = __builtin_amdgcn_cvt_pk_f32_fp8(in.v[0], false);
    float2_native hi0 = __builtin_amdgcn_cvt_pk_f32_fp8(in.v[0], true);
    float2_native lo1 = __builtin_amdgcn_cvt_pk_f32_fp8(in.v[1], false);
    float2_native hi1 = __builtin_amdgcn_cvt_pk_f32_fp8(in.v[1], true);

    float f[8];
    f[0] = lo0[0] * decode_scale; f[1] = lo0[1] * decode_scale;
    f[2] = hi0[0] * decode_scale; f[3] = hi0[1] * decode_scale;
    f[4] = lo1[0] * decode_scale; f[5] = lo1[1] * decode_scale;
    f[6] = hi1[0] * decode_scale; f[7] = hi1[1] * decode_scale;

    bf16x8_t out;
    __bf16* outp = reinterpret_cast<__bf16*>(&out);
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        outp[i] = __float2bfloat16(f[i]);
    }
    return out;
}

struct MlaDecodeArgs {
    const void* q_ptr; const void* kv_ptr;          // q: bf16, kv: fp8_e4m3fn (gfx950) or fp8_e4m3fnuz (gfx942)
    float* split_data_ptr; float* split_lse_ptr;
    const int32_t* qo_indptr; const int32_t* kv_indptr; const int32_t* kv_indices;
    float sm_scale; int nhead; int num_kv_splits;
    int64_t stride_q_s, stride_q_h, stride_kv;       // stride_kv: bytes between consecutive pages within a pool row
    int64_t stride_sd_s, stride_sd_split, stride_sd_h;
    int64_t stride_lse_s, stride_lse_split, stride_lse_h;
    // FP8 decode scale: 1.0 if KV stored as torch.float8_e4m3fn (HW intrinsic
    // matches storage, no fold needed); 0.5 if stored as fnuz (HW gives 2×
    // fnuz value, fold halves it). See `fp8x8_decode_to_bf16x8` comment.
    float fp8_decode_scale;
    // Phase B+ : 4D padded pool support (e.g., c4/c128 caches).
    // pool_outer_stride: bytes between rows of the pool (>= pages_per_row*stride_kv)
    // pages_per_row:     valid pages per row (= 1 for 2D linear pools, = P for 4D).
    // For 2D linear mode set pool_outer_stride=stride_kv and pages_per_row=1; the
    // kernel formula degenerates to `addr = base + idx*stride_kv` (legacy).
    int64_t pool_outer_stride;
    int32_t pages_per_row;
};

static constexpr int MFMA_HEADS  = 16;
// Default Q_PASSES (compile-time constant for legacy launcher dispatch). Phase
// C2a templates this — the kernel takes Q_PASSES_T as a non-type template arg
// and Q_PASSES is shadowed inside the kernel scope. The default below is
// retained for places that compute LDS sizes at the .hip launch site for the
// LEGACY 2-pass path. New callers should use MlaSizes<QK_HEAD_DIM_T, Q_PASSES_T>.
static constexpr int Q_PASSES    = 2;
static constexpr int TILE_HEADS  = MFMA_HEADS * Q_PASSES;
static constexpr int BLOCK_N     = 32;
static constexpr int NUM_WARPS   = 4;
static constexpr int BLOCK_SIZE  = NUM_WARPS * WARP_SIZE;
static constexpr int MFMA_K      = 32;
static constexpr int MFMA_N      = 16;
static constexpr int QK_N_TILES  = BLOCK_N / MFMA_N;
static constexpr int PV_N_TILES  = V_HEAD_DIM / MFMA_N;
static constexpr int PV_PER_WARP = PV_N_TILES / NUM_WARPS;

static constexpr int LDS_PAD     = 8;
// LDS for the P (softmax) tile is independent of QK_HEAD_DIM — only depends
// on MFMA_HEADS x BLOCK_N x bf16 size, shared across both V32 and 2604.
static constexpr int LDS_P_ONE   = MFMA_HEADS * BLOCK_N * 2;
static constexpr int LDS_P_SIZE  = LDS_P_ONE * Q_PASSES;

// Per-shape derived sizes. The kernel template instantiates these and the
// launcher reads `lds_total<QK_HEAD_DIM_T, Q_PASSES_T>()` to size the dynamic
// LDS request.
//
// Phase C2a (2026-04-28): added Q_PASSES_T template parameter. With Q_PASSES_T=1
// the LDS P-tile shrinks by half AND the kernel runs ~half the QK MFMA work
// (the second pass is wasted at production H=16, which fits in MFMA_HEADS=16).
// Default Q_PASSES_T=2 preserves legacy behavior for H>16 (Pro V32 / Flash
// non-TP-sharded).
//
// Math sanity per shape (verified at compile time below via static_asserts
// inside the templated loader):
//   QK_HEAD_DIM=576: QK_K_ITERS=18 (576/32), VECS_PER_ROW=72 (576/8), 72/8=9
//   QK_HEAD_DIM=512: QK_K_ITERS=16 (512/32), VECS_PER_ROW=64 (512/8), 64/8=8
template <int QK_HEAD_DIM_T, int Q_PASSES_T = 2>
struct MlaSizes {
    static constexpr int QK_HEAD_DIM = QK_HEAD_DIM_T;
    static constexpr int Q_PASSES    = Q_PASSES_T;
    static constexpr int TILE_HEADS  = MFMA_HEADS * Q_PASSES_T;
    static constexpr int QK_K_ITERS  = QK_HEAD_DIM_T / MFMA_K;
    static constexpr int LDS_STRIDE  = QK_HEAD_DIM_T + LDS_PAD;
    static constexpr int LDS_KV_ONE  = BLOCK_N * LDS_STRIDE * 2;
    static constexpr int LDS_KV_SIZE = LDS_KV_ONE * 2;
    static constexpr int LDS_P_ONE_T = MFMA_HEADS * BLOCK_N * 2;
    static constexpr int LDS_P_SIZE  = LDS_P_ONE_T * Q_PASSES_T;
    static constexpr int LDS_TOTAL   = LDS_KV_SIZE + LDS_P_SIZE;
};

template <int QK_HEAD_DIM_T, int Q_PASSES_T = 2>
__host__ __device__ constexpr int lds_total() {
    return MlaSizes<QK_HEAD_DIM_T, Q_PASSES_T>::LDS_TOTAL;
}

// FP8 KV-tile loader: read QK_HEAD_DIM_T fp8 bytes per row from HBM via
// indirect gather, decode to bf16 in registers, store VECS_PER_ROW uint4 (=
// VECS_PER_ROW*8 bf16) per row to LDS. Each thread handles 1 of 8 column
// groups in a row.
//   QK_HEAD_DIM=576 -> 9 uint4 chunks per thread
//   QK_HEAD_DIM=512 -> 8 uint4 chunks per thread
template <int QK_HEAD_DIM_T>
__device__ __forceinline__ void
load_kv_tile_fp8_to_lds(const MlaDecodeArgs& args, int tid, int kv_start, int s_end,
                        int n_start, const uint8_t* __restrict__ kv_base_fp8,
                        __bf16* __restrict__ dst_buf)
{
    int n_valid = min(BLOCK_N, s_end - n_start);
    constexpr int VEC_BF16        = 8;
    constexpr int VECS_PER_ROW    = QK_HEAD_DIM_T / VEC_BF16;
    constexpr int THREADS_PER_ROW = 8;
    constexpr int VECS_PER_THREAD = VECS_PER_ROW / THREADS_PER_ROW;
    constexpr int LDS_STRIDE_T    = MlaSizes<QK_HEAD_DIM_T>::LDS_STRIDE;
    static_assert(BLOCK_N * THREADS_PER_ROW == BLOCK_SIZE, "schedule mismatch");
    static_assert(VECS_PER_ROW == THREADS_PER_ROW * VECS_PER_THREAD, "vec count mismatch");
    static_assert(QK_HEAD_DIM_T % VEC_BF16 == 0,
                  "QK_HEAD_DIM_T must be a multiple of 8 (bf16 vec width)");

    int ld_row       = tid / THREADS_PER_ROW;
    int ld_col_group = tid & (THREADS_PER_ROW - 1);

    uint4* dst_row = reinterpret_cast<uint4*>(dst_buf + ld_row * LDS_STRIDE_T);
    // Mask for SGLang-style invalid indices (pidx < 0). Treating those as
    // zero K rows is mathematically equivalent to fully masking them out:
    // QK product is 0 for that row → softmax weight cancels through, and
    // V contribution is 0. For an all-invalid query (lonely-Q), this also
    // gives a correct zero output without a separate wrapper correction.
    int pidx = (ld_row < n_valid)
                   ? args.kv_indices[kv_start + n_start + ld_row]
                   : -1;
    if (pidx >= 0) {
        // 4D padded-pool addressing: each pool row contains `pages_per_row`
        // consecutive pages then padding. Linear/2D mode: pages_per_row=1 and
        // pool_outer_stride=stride_kv → degenerates to `pidx*stride_kv`.
        const int outer = pidx / args.pages_per_row;
        const int inner = pidx - outer * args.pages_per_row;
        const uint32_t* src_row_fp8 = reinterpret_cast<const uint32_t*>(
            kv_base_fp8 + outer * args.pool_outer_stride + inner * args.stride_kv);
        const float decode_scale = args.fp8_decode_scale;
        #pragma unroll
        for (int k = 0; k < VECS_PER_THREAD; ++k) {
            int vc = ld_col_group + k * THREADS_PER_ROW;  // 0..71 bf16x8 column index
            // Each bf16x8 (8 bf16 = uint4) corresponds to 8 fp8 = 2 uint32.
            fp8x8_t fp8_in;
            fp8_in.v[0] = src_row_fp8[vc * 2 + 0];
            fp8_in.v[1] = src_row_fp8[vc * 2 + 1];
            bf16x8_t bf = fp8x8_decode_to_bf16x8(fp8_in, decode_scale);
            dst_row[vc] = *reinterpret_cast<uint4*>(&bf);
        }
    } else {
        #pragma unroll
        for (int k = 0; k < VECS_PER_THREAD; ++k) {
            int vc = ld_col_group + k * THREADS_PER_ROW;
            dst_row[vc] = make_uint4(0, 0, 0, 0);
        }
    }
}

template <int QK_HEAD_DIM_T, int Q_PASSES_T = 2>
__global__ void __launch_bounds__(BLOCK_SIZE, 2)
mla_decode_fwd_kernel(MlaDecodeArgs args)
{
    // Phase C2a: Q_PASSES is now per-instantiation. Default 2 preserves legacy
    // behavior; Q_PASSES_T=1 cuts compute ~2× when H ≤ MFMA_HEADS=16 (Flash-Base
    // FP8 TP=4 production case). All references to Q_PASSES below resolve to the
    // template parameter via this constexpr shadow.
    constexpr int Q_PASSES    = Q_PASSES_T;
    constexpr int TILE_HEADS  = MFMA_HEADS * Q_PASSES_T;
    constexpr int QK_K_ITERS  = MlaSizes<QK_HEAD_DIM_T, Q_PASSES_T>::QK_K_ITERS;
    constexpr int LDS_STRIDE  = MlaSizes<QK_HEAD_DIM_T, Q_PASSES_T>::LDS_STRIDE;
    constexpr int LDS_KV_ONE  = MlaSizes<QK_HEAD_DIM_T, Q_PASSES_T>::LDS_KV_ONE;
    constexpr int LDS_KV_SIZE = MlaSizes<QK_HEAD_DIM_T, Q_PASSES_T>::LDS_KV_SIZE;

    const int batch_id   = blockIdx.x;
    const int head_group = blockIdx.y;
    const int split_id   = blockIdx.z;
    const int head_start = head_group * TILE_HEADS;
    const int tid        = threadIdx.x;
    const int warp_id    = tid / WARP_SIZE;
    const int lane_id    = tid % WARP_SIZE;
    const int kgrp       = lane_id / 16;

    const int kv_start = args.kv_indptr[batch_id];
    const int kv_len   = args.kv_indptr[batch_id + 1] - kv_start;
    const int qo_start = args.qo_indptr[batch_id];
    const int split_len = (kv_len + args.num_kv_splits - 1) / args.num_kv_splits;
    const int s_start   = split_len * split_id;
    const int s_end     = min(s_start + split_len, kv_len);

    // Empty-split tile: skip the QK/softmax loop but STILL run the epilogue
    // so split_data/split_lse get safe zeros + -1e30 sentinel. Without this,
    // the empty tile leaves the cached split_data/split_lse buffers (allocated
    // via torch.empty in the Python wrapper and reused across calls) at
    // whatever stale memory pattern a prior call left there — including NaN.
    // The combine kernel then reads NaN and propagates it to the model output,
    // poisoning residual + RMSnorm + lm_head and producing garbage tokens
    // (verified root cause for the Flash mxfp4 e2e regression). Sanitization
    // below (row_ok = isfinite(rmax) && rsum > 0) handles this case because
    // rmax stays -1e30 and rsum stays 0 if the QK/softmax loop never runs.
    const bool empty_split = (s_end <= s_start);

    extern __shared__ char smem[];
    // Avoid initializer-list of LDS-derived pointers (triggers an
    // "addrspacecast in static initializer" compiler error on ROCm 7 hipcc).
    // Materialize per-buffer pointers separately and select via ternary at use.
    __bf16* lds_kv_buf0 = reinterpret_cast<__bf16*>(smem);
    __bf16* lds_kv_buf1 = reinterpret_cast<__bf16*>(smem + LDS_KV_ONE);

    const uint8_t* __restrict__ kv_base = reinterpret_cast<const uint8_t*>(args.kv_ptr);
    const __bf16*  __restrict__ q_base  = reinterpret_cast<const __bf16*>(args.q_ptr);
    _Float16* __restrict__ lds_p_r =
        reinterpret_cast<_Float16*>(smem + LDS_KV_SIZE);

    constexpr int O_FLAT = Q_PASSES * PV_PER_WARP;
    float4_t o_acc[O_FLAT];
    float rmax[Q_PASSES * 4];
    float rsum[Q_PASSES * 4];
    #pragma unroll
    for (int i = 0; i < O_FLAT; ++i)
        o_acc[i] = {{0,0,0,0}};
    #pragma unroll
    for (int i = 0; i < Q_PASSES * 4; ++i) {
        rmax[i] = -1e30f;
        rsum[i] = 0;
    }

    const __bf16* q_row_ptrs[Q_PASSES];
    bool qp_active[Q_PASSES];
    const int q_head_lane = lane_id % 16;
    const int64_t q_row_base_off = (int64_t)qo_start * args.stride_q_s;
    #pragma unroll
    for (int qp = 0; qp < Q_PASSES; ++qp) {
        int qp_head_start = head_start + qp * MFMA_HEADS;
        int q_row_head    = qp_head_start + q_head_lane;
        qp_active[qp]     = (qp_head_start < args.nhead);
        int q_row_safe    = (q_row_head < args.nhead) ? q_row_head : 0;
        q_row_ptrs[qp]    = q_base + q_row_base_off + q_row_safe * args.stride_q_h;
    }

    if (s_start < s_end) {
        load_kv_tile_fp8_to_lds<QK_HEAD_DIM_T>(args, tid, kv_start, s_end, s_start,
                                kv_base, lds_kv_buf0);
    }
    __syncthreads();

    int buf_idx = 0;
    for (int n_start = s_start; n_start < s_end; n_start += BLOCK_N)
    {
        int n_valid = min(BLOCK_N, s_end - n_start);

        int next_n = n_start + BLOCK_N;
        if (next_n < s_end) {
            __bf16* next_buf = (buf_idx == 0) ? lds_kv_buf1 : lds_kv_buf0;
            load_kv_tile_fp8_to_lds<QK_HEAD_DIM_T>(args, tid, kv_start, s_end, next_n,
                                    kv_base, next_buf);
        }

        __bf16* __restrict__ lds_kv = (buf_idx == 0) ? lds_kv_buf0 : lds_kv_buf1;

        float4_t s_acc_all[Q_PASSES][QK_N_TILES];
        #pragma unroll
        for (int qp = 0; qp < Q_PASSES; ++qp)
            #pragma unroll
            for (int nt = 0; nt < QK_N_TILES; ++nt)
                s_acc_all[qp][nt] = {{0,0,0,0}};

        #pragma unroll 3
        for (int ki = 0; ki < QK_K_ITERS; ++ki)
        {
            bf16x8_t qa[Q_PASSES];
            #pragma unroll
            for (int qp = 0; qp < Q_PASSES; ++qp)
                qa[qp] = *reinterpret_cast<const bf16x8_t*>(
                    q_row_ptrs[qp] + ki * MFMA_K + kgrp * 8);
            #pragma unroll
            for (int nt = 0; nt < QK_N_TILES; ++nt)
            {
                int krow = (lane_id % 16) + nt * 16;
                bf16x8_t kb = lds_load_bf16x8(
                    &lds_kv[krow * LDS_STRIDE + ki * MFMA_K + kgrp * 8]);
                #pragma unroll
                for (int qp = 0; qp < Q_PASSES; ++qp)
                    mfma_bf16_16x16x32(s_acc_all[qp][nt], qa[qp], kb);
            }
        }

        // pa_all switched to fp16 for the higher-precision PV mfma (Layer-3 fix).
        f16x8_t pa_all[Q_PASSES];
        for (int qp = 0; qp < Q_PASSES; ++qp)
        {
            if (!qp_active[qp]) break;
            float4_t (&s_acc)[QK_N_TILES] = s_acc_all[qp];

            constexpr float kLog2e = 1.4426950408889634f;
            const float sm_scale_log2e = args.sm_scale * kLog2e;
            #pragma unroll
            for (int nt = 0; nt < QK_N_TILES; ++nt)
                #pragma unroll
                for (int c = 0; c < 4; ++c)
                    s_acc[nt][c] *= sm_scale_log2e;
            {
                int kv_col = lane_id % 16;
                #pragma unroll
                for (int nt = 0; nt < QK_N_TILES; ++nt) {
                    int row_in_tile = kv_col + nt * 16;
                    bool beyond = (row_in_tile >= n_valid);
                    // Layer-2 fix: also mask rows with pidx < 0 (real invalid
                    // indices, not just beyond-last-tile padding). Without this,
                    // q @ k_zero_filled = 0 produces score 0, which becomes a
                    // valid (non-`-inf`) entry in the softmax. When subsequent
                    // tiles produce small max values, the bf16-rounded P[invalid]
                    // is non-zero, inflating rsum and scaling output down. Oracle
                    // treats pidx<0 as -inf so its softmax excludes them entirely.
                    // This was the asymmetry causing the FP8-saturation microbench
                    // failures at invalid_frac=0.95.
                    bool invalid_idx = false;
                    if (!beyond) {
                        int pidx_check = args.kv_indices[
                            kv_start + n_start + row_in_tile];
                        invalid_idx = (pidx_check < 0);
                    }
                    if (beyond || invalid_idx) {
                        #pragma unroll
                        for (int c = 0; c < 4; ++c)
                            s_acc[nt][c] = -1e30f;
                    }
                }
            }
#ifdef SGLANG_CK_V32_DEBUG_DUMP
            // Per-(lane, c, nt) s_acc dump for the smallest failing config
            // (B=0, head_group=0, split_id=0, qp=0, first KV tile). Used by
            // microbench/microbench_ck_v32_score_dump.py to verify QK math.
            // Confirmed correct on 2026-04-29: scores match oracle exactly.
            if (batch_id == 0 && head_group == 0 && split_id == 0 &&
                qp == 0 && n_start == s_start && lane_id < 32) {
                for (int nt = 0; nt < QK_N_TILES; ++nt) {
                    for (int c = 0; c < 4; ++c) {
                        int head = (lane_id / 16) * 4 + c;
                        int kv_col = (lane_id % 16) + nt * 16;
                        printf("[CK_DBG] lane=%2d c=%d nt=%d head=%2d kv_col=%2d "
                               "s_acc=%+.4f (post-scale,post-mask)\n",
                               lane_id, c, nt, head, kv_col, s_acc[nt][c]);
                    }
                }
            }
#endif

            float lmax[4];
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                lmax[c] = -1e30f;
                #pragma unroll
                for (int nt = 0; nt < QK_N_TILES; ++nt)
                    lmax[c] = fmaxf(lmax[c], s_acc[nt][c]);
            }
            #pragma unroll
            for (int off = 1; off < 16; off *= 2) {
                #pragma unroll
                for (int c = 0; c < 4; ++c)
                    lmax[c] = fmaxf(lmax[c], __shfl_xor(lmax[c], off));
            }

            float new_max_arr[4];
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                new_max_arr[c] = fmaxf(rmax[qp*4+c], lmax[c]);
                float rsc = exp2f(rmax[qp*4+c] - new_max_arr[c]);
                #pragma unroll
                for (int i = 0; i < PV_PER_WARP; ++i)
                    o_acc[qp*PV_PER_WARP+i][c] *= rsc;
                rsum[qp*4+c] *= rsc;
            }

            float lsum[4];
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                lsum[c] = 0.0f;
                #pragma unroll
                for (int nt = 0; nt < QK_N_TILES; ++nt) {
                    float pv = exp2f(s_acc[nt][c] - new_max_arr[c]);
                    s_acc[nt][c] = pv;
                    lsum[c] += pv;
                }
            }
            #pragma unroll
            for (int off = 1; off < 16; off *= 2) {
                #pragma unroll
                for (int c = 0; c < 4; ++c)
                    lsum[c] += __shfl_xor(lsum[c], off);
            }
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                rsum[qp*4+c] += lsum[c];
                rmax[qp*4+c] = new_max_arr[c];
            }

            _Float16* __restrict__ lds_p_qp = lds_p_r + qp * (MFMA_HEADS * BLOCK_N);
            {
                int kv_col = lane_id % 16;
                for (int c = 0; c < 4; ++c) {
                    int head = kgrp * 4 + c;
                    for (int nt = 0; nt < QK_N_TILES; ++nt) {
                        // Layer-3 precision fix: store P as fp16 (10 mantissa
                        // bits) instead of bf16 (7 mantissa bits). Same byte
                        // size — drops cleanly into the LDS layout. Used by
                        // the fp16 PV mfma below at zero compute cost vs bf16.
                        // P values are bounded in [0, 1] so fp16's smaller
                        // exponent range (5-bit, max ~6.5e4) is more than
                        // sufficient. The 3 extra mantissa bits eliminate
                        // the 60-layer compounding drift that produced
                        // garbage tokens under greedy + sampling decoding.
                        // (_Float16) cast is RTNE on gfx950 hipcc.
                        lds_p_qp[head * BLOCK_N + kv_col + nt * 16] =
                            (_Float16)s_acc[nt][c];
                    }
                }
            }

            pa_all[qp] = lds_load_f16x8(
                &lds_p_qp[(lane_id % 16) * BLOCK_N + kgrp * 8]);

#ifdef SGLANG_CK_V32_DEBUG_DUMP
            // Stage-B: dump pa_all after the LDS write→read round-trip.
            // The PV mfma_bf16_16x16x32 expects pa_all[qp] for lane L holds
            // P[head=L%16, kv_col=kgrp*8..kgrp*8+7] in a bf16x8.
            // For B=1 with 2 valid kv_idx (X=0, X=1), the EXPECTED P at
            // (head=h, kv_col=0..7) is:
            //   kv_col=0: softmax weight on idx[0]=128 = exp2(score_0-max)/rsum
            //   kv_col=1: softmax weight on idx[1]=129 = exp2(score_1-max)/rsum
            //                                            actually rsum NOT applied here yet
            //                                            (rsum applied in epilogue via inv)
            //   kv_col=2..7: 0 (kernel sets s_acc=-1e30 for these → exp=0)
            // Compare to the s_acc dump in Stage-A: that printed POST-MASK
            // s_acc (still in score-space). Stage-B prints what made it into
            // pa_all (= softmax-output P, post-bf16-roundtrip).
            if (batch_id == 0 && head_group == 0 && split_id == 0 &&
                qp == 0 && n_start == s_start && lane_id < 16) {
                _Float16* p = reinterpret_cast<_Float16*>(&pa_all[qp]);
                printf("[CK_DBG_PA] lane=%2d head=%d kgrp=%d kv_cols=%d..%d "
                       "P=[%+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f %+.4f] "
                       "rmax=%+.4f rsum=%+.4f\n",
                       lane_id, lane_id % 16, kgrp,
                       kgrp * 8, kgrp * 8 + 7,
                       (float)p[0], (float)p[1], (float)p[2], (float)p[3],
                       (float)p[4], (float)p[5], (float)p[6], (float)p[7],
                       rmax[qp*4+0], rsum[qp*4+0]);
            }
#endif

        } // Q passes

        #pragma unroll 2
        for (int vl = 0; vl < PV_PER_WARP; ++vl)
        {
            int vt = warp_id * PV_PER_WARP + vl;
            int v_col = (lane_id % 16) + vt * 16;
            int kv_bt = kgrp * 8;

#ifdef SGLANG_CK_V32_FP32_PV
            // Highest-precision PV path: fp32 inputs to mfma_f32_16x16x4_f32.
            // 8x more MFMA calls per (qp, vl) than mfma_f16_16x16x32 (K=4 vs
            // K=32) but matches the bf16 ref path's fp32 softmax-output @ V
            // precision. Required to ship Lever 1's perf gain — fp16's
            // 10-bit mantissa wasn't enough; the model's residual stream
            // expects fp32-equivalent attention output across 60 layers.
            //
            // Per-call layout (fp32 mfma 16x16x4):
            //   A[M=lane%16, K=lane/16]: 1 fp32/lane → P value at this kgrp
            //   B[K=lane/16, N=lane%16]: 1 fp32/lane → V value at this kgrp
            // 8 iterations × K=4 = K=32 BLOCK_N. Each iteration the lane
            // reads its specific (M, K) and (K, N) operand from LDS.
            //
            // For this lane (kgrp = lane_id/16):
            //   k_off in 0..7: A reads P[head=lane%16, kv_col=k_off*4+kgrp]
            //                   B reads V[kv_row=k_off*4+kgrp, v_col]
            // (kgrp varies across lanes 0..63 as 0,0..15→0; 16..31→1; etc.)
            #pragma unroll
            for (int qp = 0; qp < Q_PASSES; ++qp) {
                if (!qp_active[qp]) continue;
                _Float16* __restrict__ lds_p_qp_x =
                    lds_p_r + qp * (MFMA_HEADS * BLOCK_N);
                int head = lane_id % 16;
                int kgrp_lcl = lane_id / 16;
                #pragma unroll
                for (int k_off = 0; k_off < 8; ++k_off) {
                    int k_global = k_off * 4 + kgrp_lcl;
                    // P value for this lane's (M=head, K=k_global)
                    _Float16 p_h = lds_p_qp_x[head * BLOCK_N + k_global];
                    // V value for this lane's (K=k_global, N=v_col)
                    __bf16 v_b = lds_kv[k_global * LDS_STRIDE + v_col];
                    float p_f = (float)p_h;
                    float v_f = (float)v_b;
                    mfma_f32_16x16x4(
                        o_acc[qp*PV_PER_WARP+vl], p_f, v_f);
                }
            }
#else
            __bf16 vb[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                vb[j] = lds_kv[(kv_bt + j) * LDS_STRIDE + v_col];
            }
            bf16x8_t vb_v_bf = *reinterpret_cast<bf16x8_t*>(vb);
            // Layer-3 precision fix (default fp16 path): convert V from bf16
            // (LDS storage) to fp16 in registers for the higher-precision PV
            // mfma. fp8-decoded V values fit losslessly in fp16 (fp8 has 4
            // mantissa bits, fp16 has 10).
            f16x8_t vb_v = bf16x8_to_f16x8(vb_v_bf);
            #pragma unroll
            for (int qp = 0; qp < Q_PASSES; ++qp) {
                if (!qp_active[qp]) continue;
                mfma_f16_16x16x32(
                    o_acc[qp*PV_PER_WARP+vl], pa_all[qp], vb_v);
            }
#endif
        }
        __syncthreads();
        buf_idx ^= 1;
    } // KV tiles

    // Epilogue: write split_data + split_lse with NaN/Inf sanitization.
    //
    // Production Flash mxfp4 captures with FP8-saturated KV (kv.abs.max=255)
    // and B>=6 trip a NaN-producing path inside the online softmax accumulator
    // (verified by microbench/microbench_ck_v32_nan_diff.py — 99% of rows
    // yield NaN even with 92.6% of indices valid). Suspected internal cause:
    // an inf score from a max-reduce edge case turning subsequent (s - max)
    // into NaN. Until the root-cause kernel change lands, sanitize at the
    // boundary so the broken split doesn't poison downstream combine /
    // residual / lm_head and produce garbage tokens. Sanitized rows
    // contribute LSE = -1e30 to the combine, which weights them to zero by
    // the online-softmax merge across splits — same behavior as the
    // all-masked case.
    int v_col_base = lane_id % 16;
    for (int qp = 0; qp < Q_PASSES; ++qp) {
        for (int c = 0; c < 4; ++c) {
            int cur_head = head_start + qp * MFMA_HEADS + kgrp * 4 + c;
            if (cur_head >= args.nhead) continue;
            float rmax_c = rmax[qp*4+c];
            float rsum_c = rsum[qp*4+c];
            bool row_ok = isfinite(rmax_c) && isfinite(rsum_c) && (rsum_c > 0.0f);
            float inv = row_ok ? (1.0f / rsum_c) : 0.0f;
            int sd = qo_start * args.stride_sd_s +
                     split_id * args.stride_sd_split +
                     cur_head * args.stride_sd_h;
            for (int vl = 0; vl < PV_PER_WARP; ++vl) {
                int vc = v_col_base + (warp_id * PV_PER_WARP + vl) * 16;
                if (vc < V_HEAD_DIM) {
                    float o_val = o_acc[qp*PV_PER_WARP+vl][c] * inv;
                    if (!isfinite(o_val)) o_val = 0.0f;
                    args.split_data_ptr[sd + vc] = o_val;
                }
            }
            if (v_col_base == 0) {
                int lb = qo_start * args.stride_lse_s +
                         split_id * args.stride_lse_split +
                         cur_head * args.stride_lse_h;
                constexpr float kLn2 = 0.6931471805599453f;
                float lse_val = row_ok
                    ? (rmax_c * kLn2 + logf(rsum_c))
                    : -1e30f;
                args.split_lse_ptr[lb] = lse_val;
            }
        }
    }
}

} // namespace ck_mla_sparse_fp8
