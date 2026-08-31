// Fused DUAL bf16 GEMV for the GLM-5.2 DSA indexer front end.
//
// Replaces the first two launches of section A with ONE kernel:
//
//   Q  [M, 4096] = q_lora [M, 2048] @ w_q_b [4096, 2048]^T     (K = 2048)
//   KW [M,  160] = x      [M, 6144] @ w_kw  [ 160, 6144]^T     (K = 6144)
//
// The two problems are independent (different source, different K), so one
// grid can carry both: blocks below ``nblk_k`` do the skinny 160-column
// projection, the rest do the 4096-column one.  Motivation:
//
//   * the 160-column GEMV alone launches only 160 workgroups on 256 CUs and
//     measures 0.39 TB/s -- it is latency-bound, not bandwidth-bound.  Merged
//     with the 4096-column one its latency hides inside the big kernel.
//   * 3 -> 2 launches in section A; the in-graph launch floor on this machine
//     is 1.45 us per ordinary kernel.
//
// Decomposition (all knobs are template parameters, selected by a runtime
// config id so the whole space can be swept from one build):
//
//   A block has THREADS threads, split into ``THREADS/TPC`` *column groups*.
//   A group owns NCOL output columns and its TPC threads split the K range,
//   VPT = (K/8)/TPC sixteen-byte vectors each.  X is loaded into registers
//   ONCE per thread and reused across all NCOL columns, so the steady-state
//   VMEM stream is pure weight traffic.  The dot product uses
//   v_dot2c_f32_bf16 (2 bf16 MACs / instruction, fp32 accumulate) which
//   removes every bf16 -> f32 conversion from the inner loop.
//
// Layout contract: X is [M,K] row-major bf16, W is [N,K] row-major bf16
// (torch F.linear weight), O is [M,N] row-major bf16.  K % (8*TPC) == 0 and
// all pointers 16-byte aligned.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

namespace {

typedef unsigned short ushort_t;
typedef __bf16 bf16_t;
typedef bf16_t bf16x2_t __attribute__((ext_vector_type(2)));
typedef bf16_t bf16x8_t __attribute__((ext_vector_type(8)));
typedef float float4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float dot2_bf16(unsigned int x, unsigned int w,
                                           float acc) {
  return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(bf16x2_t, x),
                                         __builtin_bit_cast(bf16x2_t, w), acc,
                                         false);
}

// round-to-nearest-even, matching torch's fp32 -> bf16 cast
__device__ __forceinline__ ushort_t f32_to_bf16_rne(float f) {
  unsigned int u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) {  // NaN -> quiet NaN
    return static_cast<ushort_t>((u >> 16) | 0x0040u);
  }
  unsigned int lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<ushort_t>(u >> 16);
}

typedef unsigned int u32x4_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ uint4 load_w(const uint4* p, bool nt) {
  if (nt) {
    u32x4_t v = __builtin_nontemporal_load(
        reinterpret_cast<const u32x4_t*>(p));
    return make_uint4(v.x, v.y, v.z, v.w);
  }
  return *p;
}

// gfx950 exact-Q path.  This is the reduction topology used by the production
// hipBLASLt MT16x16x512 kernel: four local-split-U accumulators, each owning a
// contiguous 128-K slice of every DepthU=512 tile, reduced in wave order.
// experiments/phaseK_mfma/test_lsu4_exact.py proves byte identity for all
// 24,576 primary-shape BF16 outputs (the serial one-wave MFMA differs in 8).
template <int M>
__device__ __forceinline__ void load_q_mfma_operands(
    const bf16_t* __restrict__ x, const bf16_t* __restrict__ w,
    int col0, int N, int K, int wave, int lane, int step,
    bf16x8_t& a, bf16x8_t& b) {
  const int operand_row = lane & 15;
  const int kg = lane >> 4;
  const int base = (step >> 2) * 512;
  const int off = (step & 3) * 32;
  const int kk = base + wave * 128 + off;
  a = {};
  b = {};
  if (operand_row < M) {
    a = *reinterpret_cast<const bf16x8_t*>(
        x + (long)operand_row * K + kk + kg * 8);
  }
  const int wc = col0 + operand_row;
  if (wc < N) {
    b = *reinterpret_cast<const bf16x8_t*>(
        w + (long)wc * K + kk + kg * 8);
  }
}

// PHASE Q / SHAPE ARM.  ``THREADS`` was added as a template parameter so this
// body can run as ``THREADS/256`` INDEPENDENT 256-thread column groups inside
// one workgroup.  Each group keeps its own 4 waves, its own 16 columns, its own
// 64*M-float slice of LDS and its own four-part LSU reduction, so every output
// column is still produced by exactly the same 16 MFMA steps summed in exactly
// the same order.  BIT-EXACT by construction: nothing but the (block, group)
// that owns a column moves.  For THREADS == 256, kQGroups == 1 and every
// expression below folds back to the original, so cfg61's codegen is untouched.
template <int M, int THREADS, int PIPE = 0>
__device__ __forceinline__ void gemv_q_mfma_lsu4(
    const uint4* __restrict__ Xv, const uint4* __restrict__ Wv,
    ushort_t* __restrict__ O, long ldo, int col0, int N, int K,
    float* __restrict__ smem) {
  static_assert(THREADS % 256 == 0, "MFMA Q path needs a multiple of 256");
  constexpr int kQGroups = THREADS / 256;
  const int gid = (kQGroups == 1) ? 0 : (int)(threadIdx.x >> 8);
  const int tid = (kQGroups == 1) ? (int)threadIdx.x : (int)(threadIdx.x & 255);
  if constexpr (kQGroups > 1) {
    smem += gid * (64 * M);   // 4 waves x 16 columns x M floats per group
    col0 += gid * 16;
  }
  const int wave = tid >> 6;
  const int lane = tid & 63;
  const auto* x = reinterpret_cast<const bf16_t*>(Xv);
  const auto* w = reinterpret_cast<const bf16_t*>(Wv);
  const int operand_row = lane & 15;
  const int kg = lane >> 4;
  float4_t acc = {};
  if constexpr (PIPE > 0) {
    // Shift the next global-load pair ahead of the current dependent MFMA.
    // The accumulator visits the same 16 K fragments in the same order; only
    // the operand lifetime changes.  This is therefore bit-exact to LSU4.
    constexpr int kStages = PIPE + 1;
    bf16x8_t abuf[kStages];
    bf16x8_t bbuf[kStages];
#pragma unroll
    for (int preload = 0; preload < kStages - 1; ++preload) {
      load_q_mfma_operands<M>(x, w, col0, N, K, wave, lane, preload,
                              abuf[preload], bbuf[preload]);
    }
#pragma unroll
    for (int step = 0; step < 16; ++step) {
      const int cur = step % kStages;
      const int ahead = step + kStages - 1;
      if (ahead < 16) {
        const int next = ahead % kStages;
        load_q_mfma_operands<M>(x, w, col0, N, K, wave, lane, ahead,
                                abuf[next], bbuf[next]);
      }
      acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
          abuf[cur], bbuf[cur], acc, 0, 0, 0);
      asm volatile("" : "+v"(acc));
    }
  } else {
    const bf16x8_t zero = {};
    for (int base = 0; base < K; base += 512) {
#pragma unroll
      for (int off = 0; off < 128; off += 32) {
        const int kk = base + wave * 128 + off;
        bf16x8_t a = zero;
        bf16x8_t b = zero;
        if (operand_row < M) {
          a = *reinterpret_cast<const bf16x8_t*>(
              x + (long)operand_row * K + kk + kg * 8);
        }
        const int wc = col0 + operand_row;
        if (wc < N) {
          b = *reinterpret_cast<const bf16x8_t*>(
              w + (long)wc * K + kk + kg * 8);
        }
        acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(
            a, b, acc, 0, 0, 0);
        asm volatile("" : "+v"(acc));
      }
    }
  }
  const int out_col_local = lane & 15;
  const int out_row0 = (lane >> 4) * 4;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    if (out_row0 + j < M) {
      smem[(wave * 16 + out_col_local) * M + out_row0 + j] = acc[j];
    }
  }
  __syncthreads();

  if (tid < 16 * M) {
    const int c = tid / M;
    const int r = tid - c * M;
    float sum = smem[c * M + r] + smem[(16 + c) * M + r];
    sum += smem[(32 + c) * M + r];
    sum += smem[(48 + c) * M + r];
    if (col0 + c < N) O[(long)r * ldo + col0 + c] = f32_to_bf16_rne(sum);
  }
}

// ---------------------------------------------------------------------------
// one column-group body.  smem is [THREADS/64][NCOL][M] floats.
// ---------------------------------------------------------------------------
//
// THIS IS NOT THE REFUTED NCOLQ EXPERIMENT.  TIMELINE G-N1-4 refuted ONE
// MECHANISM for amortising X -- raising NCOLQ, which pays in registers
// (VGPR 143 -> 174 -> 238, monotonically worse) -- and generalised that
// mechanism's failure into a failure of the hypothesis.  The `noX` bypass tests
// the hypothesis directly and it holds.  LDS staging amortises X at ZERO
// register cost, which the NCOLQ axis structurally cannot.
//
// BIT-EXACT BY CONSTRUCTION: this changes only WHERE the operands are read
// from.  The bytes are the same bytes, `xr` holds the same values, and the dot2
// accumulation order is untouched.  No reduction order moves.
//
// Applied to the Q side ONLY.  The K side has TPCK == THREADS, i.e. a single
// column group, so it has no X redundancy to amortise -- and its X would be
// M * Kk/8 = 4608 uint4 = 72 KB, far past a sane LDS budget.
template <int M, int THREADS, int TPC, int NCOL, int VPT, bool NT>
__device__ __forceinline__ void gemv_group(
    const uint4* __restrict__ Xv, long ldxv,
    const uint4* __restrict__ Wv, long ldwv,
    ushort_t* __restrict__ O, long ldo,
    int blk_col0, int N, float* smem) {
  constexpr int kGroups = THREADS / TPC;
  constexpr int kWaves = THREADS / 64;
  constexpr int kWPG = TPC / 64;  // waves per group (>=1 by construction)

  const int tid = threadIdx.x;
  const int lane = tid % TPC;  // K-slot inside the group
  const int gid = tid / TPC;   // which column group
  const int cbase = blk_col0 + gid * NCOL;

  // ---- X into registers, packed as 4 x bf16x2 per 16-byte vector ----------
  constexpr int kXVec = M * VPT * TPC;      // uint4 of X the whole block needs
  unsigned int xr[M][VPT][4];
#pragma unroll
  for (int m = 0; m < M; ++m) {
#pragma unroll
    for (int p = 0; p < VPT; ++p) {
      uint4 t;
      t = Xv[m * ldxv + lane + p * TPC];
      xr[m][p][0] = t.x;
      xr[m][p][1] = t.y;
      xr[m][p][2] = t.z;
      xr[m][p][3] = t.w;
    }
  }

  // ---- all NCOL x VPT weight vectors in flight ---------------------------
  uint4 wr[NCOL][VPT];
#pragma unroll
  for (int c = 0; c < NCOL; ++c) {
    int col = cbase + c;
    // clamp instead of branch: out-of-range columns are computed and dropped
    const uint4* wp = Wv + (long)(col < N ? col : 0) * ldwv + lane;
#pragma unroll
    for (int p = 0; p < VPT; ++p) {
      wr[c][p] = load_w(wp + p * TPC, NT);
    }
  }

  float acc[NCOL][M];
#pragma unroll
  for (int c = 0; c < NCOL; ++c)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[c][m] = 0.0f;

#pragma unroll
  for (int c = 0; c < NCOL; ++c) {
#pragma unroll
    for (int p = 0; p < VPT; ++p) {
      unsigned int w4[4] = {wr[c][p].x, wr[c][p].y, wr[c][p].z, wr[c][p].w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
#pragma unroll
        for (int m = 0; m < M; ++m)
          acc[c][m] = dot2_bf16(xr[m][p][e], w4[e], acc[c][m]);
      }
    }
  }

  // ---- reduce: 64-lane butterfly, then cross-wave through LDS -------------
  const int wave = tid >> 6;
  const int wlane = tid & 63;
#pragma unroll
  for (int c = 0; c < NCOL; ++c) {
#pragma unroll
    for (int m = 0; m < M; ++m) {
      float v = acc[c][m];
#pragma unroll
      for (int off = 32; off > 0; off >>= 1) v += __shfl_down(v, off, 64);
      if (wlane == 0) smem[(wave * NCOL + c) * M + m] = v;
    }
  }
  __syncthreads();

  constexpr int kOut = kGroups * NCOL * M;
  if (tid < kOut) {
    const int g = tid / (NCOL * M);
    const int r = tid - g * (NCOL * M);
    const int c = r / M;
    const int m = r - c * M;
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < kWPG; ++w)
      s += smem[((g * kWPG + w) * NCOL + c) * M + m];
    const int col = blk_col0 + g * NCOL + c;
    if (col < N) {
      const ushort_t v = f32_to_bf16_rne(s);
      O[(long)m * ldo + col] = v;
    }
  }
  (void)kWaves;
}

template <int M, int THREADS,
          int TPCK, int NCOLK, int VPTK,
          int TPCQ, int NCOLQ, int VPTQ,
          bool NT, int ORDER,
          bool QMFMA = false, int QPIPE = 0>
__global__ __launch_bounds__(THREADS) void dual_gemv_kernel(
    const uint4* __restrict__ Xq, long ldxq,
    const uint4* __restrict__ Wq, long ldwq,
    ushort_t* __restrict__ Oq, long ldoq, int Nq,
    const uint4* __restrict__ Xk, long ldxk,
    const uint4* __restrict__ Wk, long ldwk,
    ushort_t* __restrict__ Ok, long ldok, int Nk,
    int nblk_k, int stride_k) {
  constexpr int kColsPerBlkK = (THREADS / TPCK) * NCOLK;
  constexpr int kColsPerBlkQ = (THREADS / TPCQ) * NCOLQ;
  constexpr int kSmemK = (THREADS / 64) * NCOLK * M;
  constexpr int kSmemQ = (THREADS / 64) * NCOLQ * M;
  __shared__ float smem[kSmemK > kSmemQ ? kSmemK : kSmemQ];

  const int bid = blockIdx.x;
  bool do_k;
  int b;
  if (ORDER == 0) {
    do_k = bid < nblk_k;
    b = do_k ? bid : bid - nblk_k;
  } else if (ORDER == 1) {
    const int first_k = gridDim.x - nblk_k;
    do_k = bid >= first_k;
    b = do_k ? bid - first_k : bid;
  } else {
    const int q = bid / stride_k;
    do_k = (bid - q * stride_k) == 0 && q < nblk_k;
    if (do_k) {
      b = q;
    } else {
      int before = (bid + stride_k - 1) / stride_k;
      if (before > nblk_k) before = nblk_k;
      b = bid - before;
    }
  }

  if (do_k) {
    gemv_group<M, THREADS, TPCK, NCOLK, VPTK, NT>(
        Xk, ldxk, Wk, ldwk, Ok, ldok, b * kColsPerBlkK, Nk, smem);
  } else {
    if constexpr (QMFMA) {
      gemv_q_mfma_lsu4<M, THREADS, QPIPE>(
          Xq, Wq, Oq, ldoq, b * kColsPerBlkQ, Nq, VPTQ * TPCQ * 8, smem);
    } else {
      gemv_group<M, THREADS, TPCQ, NCOLQ, VPTQ, NT>(
          Xq, ldxq, Wq, ldwq, Oq, ldoq, b * kColsPerBlkQ, Nq, smem);
    }
  }

}

// ---------------------------------------------------------------------------
// PHASE K / R9 PROBE -- pure segmented streaming read of the SAME 18.75 MB.
//
// The open question is why `dual_gemv` takes 8.90 us to move 18.75 MB when a
// pure read of the same bytes was measured at 6.125 us in a different context.
// This kernel reproduces that endpoint INSIDE the graph-replay harness, with
// the identical grid decomposition (512 blocks x 32 KB of w_q_b + 80 blocks x
// 24 KB of w_kw, 256 threads), so the two are directly comparable: same bytes,
// same block count, same per-block footprint, same launch path.  It differs
// from `dual_gemv` in exactly one respect -- it does no arithmetic and holds no
// weight residency, so it has a handful of VGPRs instead of 143.
//
// `lds_bytes` is a dynamic-LDS occupancy CLAMP, the idiom already used by
// bench/impl_logits.py's bw_floor_occ4/occ2.  gfx950 has 160 KB of LDS per CU
// and these are 4-wave (256-thread) blocks, so blocks/CU == waves/SIMD and
// floor(163840 / lds_bytes) is the achievable waves/SIMD.  Clamping the pure
// read down to dual_gemv's 3 waves/SIMD is the direct test of the occupancy
// hypothesis: if it slows to ~8.9 us the cause is occupancy, if it stays at
// ~6.1 us it is not.
//
// WRONG ANSWER by construction (it writes nothing anyone reads); timing only,
// gated behind FORGE_R9_PROBE=1 on the python side.
// ---------------------------------------------------------------------------
template <int THREADS>
__global__ __launch_bounds__(THREADS) void stream_read_kernel(
    const uint4* __restrict__ Wq, const uint4* __restrict__ Wk,
    unsigned int* __restrict__ sink, int nblk_q, int per_q, int per_k) {
  HIP_DYNAMIC_SHARED(char, _lds);
  if (threadIdx.x == THREADS - 1) _lds[0] = 1;   // keep the LDS clamp live

  const int bid = blockIdx.x;
  const uint4* p;
  int cnt;
  if (bid < nblk_q) {
    p = Wq + (long)bid * per_q;
    cnt = per_q;
  } else {
    p = Wk + (long)(bid - nblk_q) * per_k;
    cnt = per_k;
  }
  uint4 acc = make_uint4(0u, 0u, 0u, 0u);
  for (int i = threadIdx.x; i < cnt; i += THREADS) {
    const uint4 v = p[i];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0xDEADBEEFu) sink[0] = 1u;
}

// ---------------------------------------------------------------------------
// PHASE O / A-MLP PROBE -- pure segmented streaming read with a RUNTIME grid.
//
// The shipped cfg53 kernel launches 256 Q blocks (16 columns, 64 KB each) plus
// 80 KW blocks (24 KB each) = 336 blocks x 256 threads = 1344 waves on 1024
// SIMDs = 1.31 waves/SIMD, while its register footprint (VGPR 124 + 4 AGPR,
// LDS 1536 B) allows 4 waves/SIMD.  Occupancy here is set by the GRID, not by
// the register file.  The R9 probe could not test that: it was hard-wired to
// cfg26's 592-block decomposition and its dynamic-LDS knob only clamps
// occupancy DOWNWARD from an already grid-limited 2.31.
//
// This probe reads exactly the same 18.75 MB with a runtime (nblk_q, nblk_k),
// a compile-time THREADS and a compile-time UNROLL (independent 16-byte loads
// in flight per thread), so block count, waves per block and per-lane
// memory-level parallelism are three separate knobs on one instrument.
//
// VALIDITY CONTROL: block 0 lane 0 records gridDim/blockDim/UNROLL into
// sink[1..3] and the python side asserts the values it asked for -- this
// campaign has twice shipped a fast path that never executed.
//
// WRONG ANSWER by construction (it writes nothing anyone reads); timing only.
// ---------------------------------------------------------------------------
template <int THREADS, int UNROLL>
__global__ __launch_bounds__(THREADS) void stream_read2_kernel(
    const uint4* __restrict__ Wq, const uint4* __restrict__ Wk,
    unsigned int* __restrict__ sink, int nblk_q, int per_q, int per_k,
    int limit) {
  HIP_DYNAMIC_SHARED(char, _lds);
  if (threadIdx.x == THREADS - 1) _lds[0] = 1;  // keep the LDS clamp live
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    sink[1] = gridDim.x;
    sink[2] = THREADS;
    sink[3] = UNROLL;
  }

  const int bid = blockIdx.x;
  const uint4* p;
  int cnt;
  if (bid < nblk_q) {
    p = Wq + (long)bid * per_q;
    cnt = per_q;
  } else {
    p = Wk + (long)(bid - nblk_q) * per_k;
    cnt = per_k;
  }
  // ``limit`` caps the per-block read.  limit == 0 turns this into a NULL
  // kernel at an arbitrary grid, which is how the dispatch-and-drain pedestal
  // is separated from the streaming read on the same instrument.
  if (limit >= 0 && limit < cnt) cnt = limit;
  uint4 acc = make_uint4(0u, 0u, 0u, 0u);
  constexpr int kStep = THREADS * UNROLL;
  int i = threadIdx.x;
  for (; i + (UNROLL - 1) * THREADS < cnt; i += kStep) {
    uint4 v[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) v[u] = p[i + u * THREADS];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      acc.x ^= v[u].x; acc.y ^= v[u].y; acc.z ^= v[u].z; acc.w ^= v[u].w;
    }
  }
  for (; i < cnt; i += THREADS) {
    const uint4 v = p[i];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0xDEADBEEFu) sink[0] = 1u;
}

struct Args {
  const uint4* Xq; long ldxq;
  const uint4* Wq; long ldwq;
  ushort_t* Oq;    long ldoq; int Nq; int Kq;
  const uint4* Xk; long ldxk;
  const uint4* Wk; long ldwk;
  ushort_t* Ok;    long ldok; int Nk; int Kk;
  hipStream_t stream;
};

template <int M, int THREADS, int TPCK, int NCOLK, int TPCQ, int NCOLQ,
          bool NT, int ORDER, bool QMFMA = false, int QPIPE = 0>
void launch(const Args& a) {
  constexpr int kColsK = (THREADS / TPCK) * NCOLK;
  constexpr int kColsQ = (THREADS / TPCQ) * NCOLQ;
  const int VPTK = (a.Kk / 8) / TPCK;
  const int VPTQ = (a.Kq / 8) / TPCQ;
  TORCH_CHECK(VPTK * TPCK * 8 == a.Kk, "Kk not divisible by 8*TPCK");
  TORCH_CHECK(VPTQ * TPCQ * 8 == a.Kq, "Kq not divisible by 8*TPCQ");
  const int nblk_k = (a.Nk + kColsK - 1) / kColsK;
  const int nblk_q = (a.Nq + kColsQ - 1) / kColsQ;
  const int stride_k = (nblk_k + nblk_q) / (nblk_k > 0 ? nblk_k : 1);

#define DG_CALL(VK, VQ)                                                        \
  do {                                                                         \
    dual_gemv_kernel<M, THREADS, TPCK, NCOLK, VK, TPCQ, NCOLQ, VQ, NT, ORDER,  \
                     QMFMA, QPIPE>                                             \
        <<<dim3(nblk_k + nblk_q), dim3(THREADS), 0, a.stream>>>(               \
            a.Xq, a.ldxq, a.Wq, a.ldwq, a.Oq, a.ldoq, a.Nq,                    \
            a.Xk, a.ldxk, a.Wk, a.ldwk, a.Ok, a.ldok, a.Nk, nblk_k, stride_k); \
  } while (0)

  // Kq = 2048 -> 256 vectors; Kk = 6144 -> 768 vectors.  Only the (VPTK,VPTQ)
  // pairs that those two shapes can produce for TPC in {64,128,256} are
  // instantiated.
  if (VPTK == 3 && VPTQ == 1) { DG_CALL(3, 1); }
  else if (VPTK == 3 && VPTQ == 2) { DG_CALL(3, 2); }
  else if (VPTK == 3 && VPTQ == 4) { DG_CALL(3, 4); }
  else if (VPTK == 6 && VPTQ == 1) { DG_CALL(6, 1); }
  else if (VPTK == 6 && VPTQ == 2) { DG_CALL(6, 2); }
  else if (VPTK == 6 && VPTQ == 4) { DG_CALL(6, 4); }
  else if (VPTK == 12 && VPTQ == 4) { DG_CALL(12, 4); }
  else { TORCH_CHECK(false, "unsupported (VPTK,VPTQ)=(", VPTK, ",", VPTQ, ")"); }
#undef DG_CALL
}

// ---------------------------------------------------------------------------
// config table.  cfg id -> (THREADS, TPCK, NCOLK, TPCQ, NCOLQ, NT, KFIRST)
// ---------------------------------------------------------------------------
template <int M>
void dispatch_cfg(int cfg, const Args& a) {
  switch (cfg) {
    // The accepted configuration, kept alone.  The 67-cfg sweep that chose it is
    // in the campaign log; each retired case cost 8 more template expansions.
    //             THR TPCK NK TPCQ NQ  NT     ORDER MFMA PIPE
    case 61: launch<M, 256, 256, 2, 256, 16, false, 1, true, 2>(a); break;
    default: TORCH_CHECK(false, "unknown dual_gemv cfg ", cfg);
  }
}

}  // namespace

void dual_gemv_bf16(at::Tensor Xq, at::Tensor Wq, at::Tensor Oq,
                    at::Tensor Xk, at::Tensor Wk, at::Tensor Ok,
                    int64_t cfg) {
  for (auto* t : {&Xq, &Wq, &Oq, &Xk, &Wk, &Ok})
    TORCH_CHECK(t->scalar_type() == at::kBFloat16, "all tensors must be bf16");
  TORCH_CHECK(Xq.dim() == 2 && Wq.dim() == 2 && Oq.dim() == 2, "2-D");
  TORCH_CHECK(Xk.dim() == 2 && Wk.dim() == 2 && Ok.dim() == 2, "2-D");
  const int M = static_cast<int>(Xq.size(0));
  TORCH_CHECK(Xk.size(0) == M && Oq.size(0) == M && Ok.size(0) == M,
              "row count must match");
  TORCH_CHECK(M >= 1 && M <= 8, "specialised for M in [1,8]");

  Args a;
  a.Xq = reinterpret_cast<const uint4*>(Xq.data_ptr());
  a.Wq = reinterpret_cast<const uint4*>(Wq.data_ptr());
  a.Oq = reinterpret_cast<ushort_t*>(Oq.data_ptr());
  a.Kq = static_cast<int>(Xq.size(1));
  a.Nq = static_cast<int>(Wq.size(0));
  a.ldxq = Xq.stride(0) / 8;
  a.ldwq = Wq.stride(0) / 8;
  a.ldoq = Oq.stride(0);
  a.Xk = reinterpret_cast<const uint4*>(Xk.data_ptr());
  a.Wk = reinterpret_cast<const uint4*>(Wk.data_ptr());
  a.Ok = reinterpret_cast<ushort_t*>(Ok.data_ptr());
  a.Kk = static_cast<int>(Xk.size(1));
  a.Nk = static_cast<int>(Wk.size(0));
  a.ldxk = Xk.stride(0) / 8;
  a.ldwk = Wk.stride(0) / 8;
  a.ldok = Ok.stride(0);
  a.stream = c10::cuda::getCurrentCUDAStream();

  TORCH_CHECK(Wq.size(1) == a.Kq && Wk.size(1) == a.Kk, "K mismatch");
  TORCH_CHECK(Oq.size(1) == a.Nq && Ok.size(1) == a.Nk, "N mismatch");
  TORCH_CHECK(Xq.stride(0) % 8 == 0 && Wq.stride(0) % 8 == 0, "align Q");
  TORCH_CHECK(Xk.stride(0) % 8 == 0 && Wk.stride(0) % 8 == 0, "align K");
  TORCH_CHECK(Xq.stride(1) == 1 && Wq.stride(1) == 1 && Oq.stride(1) == 1, "contig");
  TORCH_CHECK(Xk.stride(1) == 1 && Wk.stride(1) == 1 && Ok.stride(1) == 1, "contig");

  switch (M) {
    case 1: dispatch_cfg<1>(static_cast<int>(cfg), a); break;
    case 2: dispatch_cfg<2>(static_cast<int>(cfg), a); break;
    case 3: dispatch_cfg<3>(static_cast<int>(cfg), a); break;
    case 4: dispatch_cfg<4>(static_cast<int>(cfg), a); break;
    case 5: dispatch_cfg<5>(static_cast<int>(cfg), a); break;
    case 6: dispatch_cfg<6>(static_cast<int>(cfg), a); break;
    case 7: dispatch_cfg<7>(static_cast<int>(cfg), a); break;
    case 8: dispatch_cfg<8>(static_cast<int>(cfg), a); break;
    default: TORCH_CHECK(false, "unreachable");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dual_gemv_bf16", &dual_gemv_bf16,
        "fused dual bf16 GEMV: Oq = Xq @ Wq^T and Ok = Xk @ Wk^T in one launch",
        pybind11::arg("Xq"), pybind11::arg("Wq"), pybind11::arg("Oq"),
        pybind11::arg("Xk"), pybind11::arg("Wk"), pybind11::arg("Ok"),
        pybind11::arg("cfg") = 0);
}
