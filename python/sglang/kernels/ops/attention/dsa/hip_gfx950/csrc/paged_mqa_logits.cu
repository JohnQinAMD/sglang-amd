// -----------------------------------------------------------------------------
// GLM-5.2 DSA indexer, the logits kernel: paged MQA FP8 logits.  Raw HIP for gfx950.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

typedef __attribute__((__vector_size__(8 * sizeof(int)))) int i32x8;
typedef __attribute__((__vector_size__(4 * sizeof(int)))) int i32x4;
typedef __attribute__((__vector_size__(4 * sizeof(float)))) float f32x4;

#define PAGE_TOK 64
#define HD 128
#define TOK_STRIDE 132
#define PAGE_BYTES (PAGE_TOK * TOK_STRIDE)  // 8448
#define K_BYTES (PAGE_TOK * HD)             // 8192

union V8 {
  i32x8 v;
  i32x4 h[2];
};

__device__ __forceinline__ f32x4 mfma128(i32x8 a, i32x8 b, f32x4 c) {
  return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c, 0, 0, 0, 127, 0, 127);
}

// relu in exactly one v_max_f32.  fmaxf() lowers to TWO v_max_f32 (LLVM
// canonicalises the operand for NaN semantics); the MFMA output here can never
// be NaN, so the canonicalisation is pure waste -- 32 instructions per page.
__device__ __forceinline__ float relu0(float x) {
  float r;
  asm("v_max_f32 %0, %1, 0" : "=v"(r) : "v"(x));
  return r;
}

// wave64 lane exchanges on the VALU instead of through LDS:
//   xor32 <- v_permlane64_b32 (lane l reads lane l^32)
//   xor16 <- v_permlanex16_b32 with identity selectors (lane l reads lane l^16)
typedef __attribute__((__vector_size__(2 * sizeof(int)))) int i32x2;
__device__ __forceinline__ float xor32(float x, int lane) {
  int v = __builtin_bit_cast(int, x);
  i32x2 r = __builtin_amdgcn_permlane32_swap(v, v, false, false);
  return __builtin_bit_cast(float, (lane & 32) ? r[0] : r[1]);
}
__device__ __forceinline__ float xor16(float x, int lane) {
  int v = __builtin_bit_cast(int, x);
  i32x2 r = __builtin_amdgcn_permlane16_swap(v, v, false, false);
  return __builtin_bit_cast(float, (lane & 16) ? r[0] : r[1]);
}

template <int NT>
__device__ __forceinline__ i32x4 ldv(const void* p) {
  return NT ? __builtin_nontemporal_load((const i32x4*)p) : *(const i32x4*)p;
}
template <int NT>
__device__ __forceinline__ float lds1(const float* p) {
  return NT ? __builtin_nontemporal_load(p) : *p;
}
template <int NT>
__device__ __forceinline__ void stf(float v, float* p) {
  if (NT)
    __builtin_nontemporal_store(v, p);
  else
    *p = v;
}

// -----------------------------------------------------------------------------
// The top-k stage's histogram, built here.
// This kernel's issue slots are idle waiting on KV loads, so binning a logit it
// already holds in a register is close to free, and it deletes a whole 1.49 MB
// read pass plus a launch from the top-k stage.
// THREE THINGS THAT MUST BE EXACTLY RIGHT:
//  1. bin on the SAME key the top-k stage's threshold derivation uses,
//     order_key16(v) >> (16-HB), so that derivation stays bit-identical;
//  2. positions >= seqlen must NOT be binned.  `out` is [B, 62016] and this
//     kernel writes -INFINITY past the end of the row; order_key16(-inf) >> 4
//     is bin 63 and would inflate `above` and under-fill the output with -1;
//  3. rows with seqlen <= topk must not be binned AT ALL.  The stage that
//     re-zeroes the histogram early-returns on them, so binning such a row
//     would leave the histogram permanently dirty for every later call.  Those
//     rows are written in full by k_scatter.
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t order_key16(float x) {
  __half h = __float2half_rn(x);
  unsigned short bits = __half_as_ushort(h);
  unsigned short key = (bits & 0x8000) ? (unsigned short)(~bits) : (unsigned short)(bits | 0x8000);
  return (uint32_t)key;
}

// -----------------------------------------------------------------------------
// Phase K / C1: CO==1 additionally maintains a 64-bin COARSE summary of the same
// histogram, so that the top-k stage's 192 k_scatter blocks can derive the threshold
#define LG_CBITS 6
#define LG_CBINS (1 << LG_CBITS)

#define CAND_GBMAX 64
#define CAND_NBUCKET 256
#define PAGE_BITS 6

// =============================================================================
// Fused paged-MQA FP8 logits + in-loop fine histogram + coarse summary.  The
// histogram is what lets the top-k stage skip a separate scoring pass.

// order_key16(x) >> LOWB in 4 VALU ops after the cvt, with no branch and no
// v_cndmask.  BIT-IDENTICAL to (order_key16(x) >> LOWB) for every finite x and
// for +/-0:  writing m = (int)float_bits(x) >> 31 (all-ones iff x is negative,
template <int LOWB>
__device__ __forceinline__ uint32_t order_bin_fast(float x) {
  const uint32_t h = (uint32_t)__half_as_ushort(__float2half_rn(x));
  const uint32_t m = (uint32_t)((int32_t)__float_as_int(x) >> 31);
  return (h >> LOWB) ^ (0x8000u >> LOWB) ^ (m & (0x7fffu >> LOWB));
}

// -----------------------------------------------------------------------------
template <int WARPS, int HB>
__global__ __launch_bounds__(WARPS * 64) void logits_hist_m(
    const uint8_t* __restrict__ q,
    const uint8_t* __restrict__ kv,
    const float* __restrict__ wgt,
    const int* __restrict__ seqlens,
    const int* __restrict__ ptable,
    float* __restrict__ out,
    unsigned int* __restrict__ ghist,
    int heads,
    int max_pages,
    int out_stride,
    int topk) {
  constexpr int NBIN = 1 << HB;
  constexpr int LOWB = 16 - HB;
  constexpr int GHS = NBIN + LG_CBINS;
  constexpr int NTH = WARPS * 64;
  constexpr int PERT = NBIN / NTH;  // GLOBAL bins per thread (coarse map)
  constexpr int CSH = HB - LG_CBITS;
  constexpr int GRP = NTH / LG_CBINS;
  static_assert(NBIN >= 2 && (NBIN % 2) == 0, "paired flush needs even bins");
  static_assert(NBIN % NTH == 0 && NTH >= LG_CBINS, "");
  static_assert(PERT <= LG_CBINS && (LG_CBINS % PERT) == 0, "a thread's contiguous run must sit inside one coarse bin");
  __shared__ unsigned int s_hist[NBIN];

  const int row = blockIdx.y;
  const int lane = threadIdx.x & 63;
  const int warp = threadIdx.x >> 6;
  const int seqlen = seqlens[row];
  const int npages = (seqlen + PAGE_TOK - 1) / PAGE_TOK;
  const int g = lane >> 4, c = lane & 15;
  // Nothing to rank when the row is shorter than topk: everything is selected,
  // so the histogram, its two barriers and the epilogue all compile out.
  const bool do_hist = (seqlen > topk);

  if (do_hist) {
    for (int i = threadIdx.x; i < NBIN; i += NTH)
      s_hist[i] = 0u;
  }

  const uint8_t* qp = q + (size_t)row * (size_t)(heads * HD) + (size_t)c * HD + g * 32;
  V8 qa0, qa1;
  qa0.h[0] = *(const i32x4*)(qp);
  qa0.h[1] = *(const i32x4*)(qp + 16);
  qa1.h[0] = *(const i32x4*)(qp + 16 * HD);
  qa1.h[1] = *(const i32x4*)(qp + 16 * HD + 16);
  const float* wp = wgt + (size_t)row * heads + g * 4;
  f32x4 w0 = *(const f32x4*)(wp);
  f32x4 w1 = *(const f32x4*)(wp + 16);

  const int nwaves = gridDim.x * WARPS;
  const int wid = blockIdx.x * WARPS + warp;
  const int koff = g * 512 + c * 16;

  if (do_hist) __syncthreads();

  unsigned int* __restrict__ gh = ghist + (size_t)row * GHS;

  const int* __restrict__ pt = ptable + (size_t)row * max_pages;

  for (int p = wid; p < npages; p += nwaves) {
    const int phys = pt[p];
    const uint8_t* base = kv + (size_t)phys * PAGE_BYTES;
    const uint8_t* kb = base + koff;
    V8 b[4];
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      b[tt].h[0] = *(const i32x4*)(kb + tt * 2048);
      b[tt].h[1] = *(const i32x4*)(kb + tt * 2048 + 256);
    }
    const float ks = ((const float*)(base + K_BYTES))[lane];
    float s[4];
#pragma unroll
    for (int tt = 0; tt < 4; ++tt) {
      const f32x4 z = {0.f, 0.f, 0.f, 0.f};
      f32x4 a0 = mfma128(qa0.v, b[tt].v, z);
      f32x4 a1 = mfma128(qa1.v, b[tt].v, z);
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc = fmaf(fmaxf(a0[i], 0.f), w0[i], acc);
#pragma unroll
      for (int i = 0; i < 4; ++i)
        acc = fmaf(fmaxf(a1[i], 0.f), w1[i], acc);
      s[tt] = acc;
    }
    const bool hi = (g & 2) != 0;
    float A0 = (hi ? s[2] : s[0]) + xor32(hi ? s[0] : s[2], lane);
    float A1 = (hi ? s[3] : s[1]) + xor32(hi ? s[1] : s[3], lane);
    const bool odd = (g & 1) != 0;
    float T = (odd ? A1 : A0) + xor16(odd ? A0 : A1, lane);
    const int pos = p * PAGE_TOK + lane;
    const bool live = pos < seqlen;
    const float v = T * ks;
    out[(size_t)row * out_stride + pos] = live ? v : -INFINITY;
    if (do_hist && live) {
      const uint32_t bb = order_bin_fast<LOWB>(v);
      atomicAdd(&s_hist[bb], 1u);
    }
  }

  if (do_hist) {
    __syncthreads();

    // ---- coarse summary, mapped over the GLOBAL bin index ----
    // Thread tx owns global bins [tx*PERT, tx*PERT+PERT), which lies inside
    // coarse bin (tx*PERT)>>CSH.  No barrier of its own, and emitted BEFORE
    const int g0 = (int)threadIdx.x * PERT;
    unsigned int acc = 0u;
#pragma unroll
    for (int j = 0; j < PERT; ++j)
      acc += s_hist[g0 + j];
#pragma unroll
    for (int o = GRP >> 1; o >= 1; o >>= 1)
      acc += __shfl_down(acc, o, 64);
    if ((threadIdx.x & (GRP - 1)) == 0 && acc) atomicAdd(&gh[NBIN + (g0 >> CSH)], acc);

    for (int i = threadIdx.x; i < NBIN / 2; i += NTH) {
      const unsigned long long vv = (unsigned long long)s_hist[2 * i] | ((unsigned long long)s_hist[2 * i + 1] << 32);
      if (vv) atomicAdd((unsigned long long*)(gh + 2 * i), vv);
    }
  }
}

// -----------------------------------------------------------------------------
extern "C" {

// One kernel ships: logits_hist_m<8, 12, ...>, the accepted hip_histM_w8b48.
// The Phase M search space -- 7 kernel families, 80 variants, the histogram-
// narrowing, flush-mode and warp-count crosses -- is in the campaign log.
int launch_logits(
    const void* q,
    const void* kv,
    const void* wgt,
    const void* seqlens,
    const void* ptable,
    void* out,
    int batch,
    int heads,
    int max_pages,
    int out_stride,
    int blocks_per_row,
    int warps,
    void* stream,
    void* ghist,
    int hist_bits,
    int topk) {
  if (hist_bits != 12) return -3;
  if (warps != 8) return -6;
  hipLaunchKernelGGL(
      (logits_hist_m<8, 12>),
      dim3(blocks_per_row, batch),
      dim3(8 * 64),
      0,
      (hipStream_t)stream,
      (const uint8_t*)q,
      (const uint8_t*)kv,
      (const float*)wgt,
      (const int*)seqlens,
      (const int*)ptable,
      (float*)out,
      (unsigned int*)ghist,
      heads,
      max_pages,
      out_stride,
      topk);
  return (int)hipGetLastError();
}
}
