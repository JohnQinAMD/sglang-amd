// A copy of AITER's `indexer_qk_rope_quant_and_cache_kernel`
// (source/aiter/csrc/kernels/cache_kernels.cu:1405) with an OPTIONAL inline
// 128-point Hadamard, so the GLM-5.2 front end can keep `rotate_activation`
// AND the fusion at the same time.
//
// This is a *copy*: source/aiter is untouched, so no shared JIT module is
// rebuilt and no other fellow's tree moves.  It includes AITER's own headers
// (opus/opus.hpp, hip_reduce.h), so `opus::cast<fp8>`, `opus::finfo` and
// `block_reduce` are literally the same code as the original -- the only
// difference in the HADAMARD=false path is that it is compiled here.
//
// Why an inline Hadamard is possible at all
// -----------------------------------------
// The op launches grid(num_tokens, n_heads) x block(head_dim=128): one whole
// 128-dim head lives in ONE workgroup, one element per thread, and the kernel
// already stages that head in LDS (`q_vals` / `normed`) for the RoPE pair
// exchange.  A 128-point Hadamard is 7 butterfly rounds over exactly that LDS
// array -- no extra global traffic at all, on a kernel that is latency-bound.
//
// Why it is bit-identical to `fast_hadamard_transform`
// ----------------------------------------------------
// The input is a bf16 value (8-bit mantissa) and the butterfly is a strict
// binary sum/difference tree of depth 7, so every intermediate needs at most
// 8 + 7 = 15 mantissa bits and is therefore EXACT in fp32 regardless of the
// stage order.  experiments/phaseB/hadamard_order.py confirms this empirically:
// ascending and descending stage orders both reproduce
// `hadamard_transform(x, scale=128**-0.5)` bit-for-bit over 524288 elements,
// with `scale` applied once at the end in fp32 before the bf16 round.
//
// Placement.  The baseline applies the Hadamard AFTER the RoPE write-back and
// BEFORE act_quant, over the full head_dim=128.  RoPE only touches dims 0..63,
// and H_128 mixes all 128 dims, so H and RoPE do NOT commute -- H must sit
// between them.  (This is also why H cannot be folded into wq_b's weights at
// load time: that would place it before RoPE.)  The K side is the same:
// LayerNorm -> RoPE -> Hadamard -> ue8m0 quant -> cache store.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include "hip_reduce.h"
#include "opus/opus.hpp"

namespace {

using cache_t = opus::fp8_t;
using scalar_t = opus::bf16_t;

constexpr int HEAD_DIM = 128;
constexpr int ROPE_DIM = 64;
constexpr int LOG2_HEAD_DIM = 7;

// 7-round Hadamard butterfly over a head held one element per thread.
// HEAD_DIM=128 is exactly two 64-lane wavefronts, so strides 1..32 are
// intra-wave and ride `__shfl_xor` with NO barrier; only the stride-64 stage
// crosses wavefronts and needs a single LDS exchange.  That is 2 barriers
// instead of the 14 a pure-LDS butterfly would cost.
//
// Ascending stage order (1, 2, 4, ... 64); see the header comment for why the
// order does not affect the result bit-for-bit.
__device__ __forceinline__ float hadamard128(float* __restrict__ buf, int dim,
                                             float v, float scale) {
#pragma unroll
  for (int s = 0; s < LOG2_HEAD_DIM - 1; ++s) {   // strides 1, 2, 4, 8, 16, 32
    const int d = 1 << s;
    const float partner = __shfl_xor(v, d, 64);
    // (dim & d) == 0 -> this lane holds the "a" of the pair -> a + b
    v = ((dim & d) == 0) ? (v + partner) : (partner - v);
  }
  __syncthreads();                                // stride 64: cross-wave
  buf[dim] = v;
  __syncthreads();
  const float partner = buf[dim ^ (HEAD_DIM / 2)];
  v = ((dim & (HEAD_DIM / 2)) == 0) ? (v + partner) : (partner - v);
  return v * scale;
}

// ---------------------------------------------------------------------------
// K1: the same kernel folded onto ONE 64-lane wavefront.
//
// The straightforward 128-thread form costs ~3.2 us of wall over an empty launch
// of the same grid on a ~100 KB working set.  That is not bandwidth, it is a
// dependent chain: HEAD_DIM=128 spans two wavefronts, so each of its five
// `block_reduce` calls and the Hadamard's stride-64 stage has to round-trip
// through LDS behind a `__syncthreads()`.
//
// Fix: give each thread TWO dims -- lane t owns dim t and dim t+64 -- and run a
// single wavefront per (token, head).  Then
//   * both halves of every reduction are an independent 64-lane DPP tree, and
//   * the Hadamard's stride-64 butterfly is a register-local add/sub,
// so the kernel has ZERO __syncthreads and ZERO LDS.
//
// BIT-EXACTNESS vs that 128-thread form (the reason for the odd "two trees then
// combine" shape).  `block_reduce<float, F, 128>` computes, exactly:
//     T0 = wave_reduce<64,false>(dims 0..63)      (in lane 63 of wave 0)
//     T1 = wave_reduce<64,false>(dims 64..127)    (in lane 63 of wave 1)
//     result = reduce_op(T1, T0)
// Lane t here holds dim t in the "lo" slot and dim t+64 in the "hi" slot, so
// wave_reduce over the lo slot reproduces T0 and over the hi slot reproduces T1
// *operand for operand*, and the combine is written in the same argument order.
// `threadBroadcast=true` only appends a readlane, which cannot change a value.
// The Hadamard keeps the identical ascending stage order and identical operand
// order, and the rope/LayerNorm arithmetic is unchanged.  So this kernel is
// bit-identical to the one above, not merely close -- and that is checked
// directly in experiments/phaseK_a/equiv.py rather than asserted here.
// ---------------------------------------------------------------------------

constexpr int HALF_DIM = HEAD_DIM / 2;  // 64 == one wavefront

template <typename F>
__device__ __forceinline__ float wred64(float v, F op) {
  return wave_reduce<float, F, 64, true>(v, op);
}

// rope on the low 64 dims, held one-per-lane in `v`.  `partner` comes from a
// cross-lane shuffle instead of LDS; the arithmetic is byte-for-byte the same.
__device__ __forceinline__ float rope_lo(float v, int t, bool is_neox,
                                         const scalar_t* __restrict__ cos_ptr,
                                         const scalar_t* __restrict__ sin_ptr) {
  float pair_val;
  int cos_idx;
  bool first;
  if (is_neox) {
    constexpr int HALF = ROPE_DIM / 2;
    pair_val = __shfl_xor(v, HALF, 64);
    cos_idx = t < HALF ? t : t - HALF;
    first = t < HALF;
  } else {
    pair_val = __shfl_xor(v, 1, 64);
    cos_idx = t / 2;
    first = (t % 2 == 0);
  }
  float cos_v, sin_v;
  cos_v = static_cast<float>(cos_ptr[cos_idx]);
  sin_v = static_cast<float>(sin_ptr[cos_idx]);
  v = first ? (v * cos_v - pair_val * sin_v) : (v * cos_v + pair_val * sin_v);
  return static_cast<float>(static_cast<scalar_t>(v));
}

// 128-point Hadamard on a pair (lo = dim t, hi = dim t+64) held in one lane.
__device__ __forceinline__ void hadamard128_wave(float& lo, float& hi, int t,
                                                 float scale) {
#pragma unroll
  for (int s = 0; s < LOG2_HEAD_DIM - 1; ++s) {  // strides 1, 2, 4, 8, 16, 32
    const int d = 1 << s;
    const float plo = __shfl_xor(lo, d, 64);
    const float phi = __shfl_xor(hi, d, 64);
    const bool a = ((t & d) == 0);   // (dim & d) is the same for dim and dim+64
    lo = a ? (lo + plo) : (plo - lo);
    hi = a ? (hi + phi) : (phi - hi);
  }
  // stride 64: partner of dim t is dim t+64 -- both already in this lane.
  const float nlo = lo + hi;
  const float nhi = lo - hi;        // (dim & 64) != 0 -> partner - v
  lo = nlo * scale;
  hi = nhi * scale;
}

// KSPLIT: the k-side epilogue (LayerNorm -> rope -> Hadamard -> fp8 -> paged
// store) is 6 tokens' worth of work, and in the AITER structure it runs at the
// END of the head_idx==0 blocks -- i.e. strictly serialised behind those
// blocks' q-side chain, on the kernel's critical path, while the other 186
// blocks sit idle having finished.  With KSPLIT the grid is (T, n_heads+1) and
// column n_heads does ONLY the k side, so the two chains run concurrently and
// the kernel costs max(q, k) instead of q + k.  Same arithmetic, same operand
// order, so still bit-exact; 6 extra 1-wave blocks on a 256-CU part.
// SPEC: pin the four runtime mode flags to the values the GLM-5.2 decode path
// always passes (is_neox=false / ue8m0 / preshuffle / compute_all_q_rope) so the
// branches fold and the preshuffled cache address math becomes constant.  The
// host asserts the runtime flags actually match before selecting this
// instantiation, so it cannot silently compute the wrong thing.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
struct QKArgs {
  const scalar_t* __restrict__ q;
  cache_t* __restrict__ q_out;
  const scalar_t* __restrict__ weights;
  float* __restrict__ weights_out;
  const scalar_t* __restrict__ k;
  cache_t* __restrict__ kv_cache;
  const int64_t* __restrict__ slot_mapping;
  const float* __restrict__ norm_weight;
  const float* __restrict__ norm_bias;
  const int64_t* __restrict__ positions;
  const scalar_t* __restrict__ cos_cache;
  const scalar_t* __restrict__ sin_cache;
  int n_heads, cache_block_size, cache_stride, max_position;
  int q_stride_t, q_stride_h, q_out_stride_t, q_out_stride_h;
  int weights_stride_t, weights_out_stride_t, k_stride_t;
  int cos_stride0, sin_stride0, quant_block_size;
  float epsilon, weights_scale, hadamard_scale;
  bool use_ue8m0_rt, preshuffle_rt, is_neox_rt, compute_all_q_rope_rt;
};

template <bool HADAMARD, bool KSPLIT, bool SPEC>
__device__ __forceinline__ void qk_body(const QKArgs& A) {
  const bool use_ue8m0 = SPEC ? true : A.use_ue8m0_rt;
  const bool preshuffle = SPEC ? true : A.preshuffle_rt;
  const bool is_neox = SPEC ? false : A.is_neox_rt;
  const bool compute_all_q_rope = SPEC ? true : A.compute_all_q_rope_rt;
  const int qblk = SPEC ? HEAD_DIM : A.quant_block_size;
  const int n_heads = A.n_heads;
  const int cache_block_size = A.cache_block_size;
  const int cache_stride = A.cache_stride;
  const float epsilon = A.epsilon;
  const float weights_scale = A.weights_scale;
  const float hadamard_scale = A.hadamard_scale;
  const int64_t token_idx = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int t = threadIdx.x;                 // owns dim t and dim t + 64
  const bool do_q = KSPLIT ? (head_idx < n_heads) : true;
  const bool do_k = KSPLIT ? (head_idx == n_heads) : (head_idx == 0);
  // num_tokens == gridDim.x by construction, so the bound check is free
  if (head_idx >= n_heads + (KSPLIT ? 1 : 0)) return;

  const int64_t slot_idx = A.slot_mapping[token_idx];
  if (!compute_all_q_rope && slot_idx < 0) return;
  int64_t pos = A.positions[token_idx];
  if (slot_idx < 0)
    pos = pos < 0 ? 0 : (pos >= A.max_position ? A.max_position - 1 : pos);
  const scalar_t* cos_ptr = A.cos_cache + pos * A.cos_stride0;
  const scalar_t* sin_ptr = A.sin_cache + pos * A.sin_stride0;

  auto max_func = [](float a, float b) { return fmaxf(a, b); };
  auto sum_func = [](float a, float b) { return a + b; };
  const float q_fp8_max = static_cast<float>(opus::finfo<cache_t>::max());

  // ------------------------------- q side ---------------------------------
  if (do_q) {
    const scalar_t* q_row =
        A.q + token_idx * A.q_stride_t + head_idx * A.q_stride_h;
    float lo = static_cast<float>(q_row[t]);
    float hi = static_cast<float>(q_row[t + HALF_DIM]);

    float q_amax;
    lo = rope_lo(lo, t, is_neox, cos_ptr, sin_ptr);  // dims >=64 untouched

    if constexpr (HADAMARD) {
      hadamard128_wave(lo, hi, t, hadamard_scale);
      lo = static_cast<float>(static_cast<scalar_t>(lo));
      hi = static_cast<float>(static_cast<scalar_t>(hi));
    }

    const float a0 = wred64(fabsf(lo), max_func);
    const float a1 = wred64(fabsf(hi), max_func);
    q_amax = max_func(a1, a0);                     // block_reduce's order

    const float q_inv_fp8_max = 1.0f / q_fp8_max;
    float q_scale = fmaxf(q_amax, 1e-10f) * q_inv_fp8_max;
    if (use_ue8m0) q_scale = exp2f(ceilf(log2f(q_scale)));
    const float q_inv_scale = 1.0f / q_scale;
    cache_t* qo =
        A.q_out + token_idx * A.q_out_stride_t + head_idx * A.q_out_stride_h;
    qo[t] = opus::cast<cache_t>(lo * q_inv_scale);
    qo[t + HALF_DIM] = opus::cast<cache_t>(hi * q_inv_scale);
    if (t == 0) {
      const float w = static_cast<float>(
          A.weights[token_idx * A.weights_stride_t + head_idx]);
      const float head_scale = rsqrtf(static_cast<float>(n_heads));
      const scalar_t w_head = static_cast<scalar_t>(w * head_scale);
      const float softmax_scale = weights_scale / head_scale;
      A.weights_out[token_idx * A.weights_out_stride_t + head_idx] =
          static_cast<float>(w_head) * q_scale * softmax_scale;
    }
  }

  if (!do_k || slot_idx < 0) return;

  // ------------------------------- k side ---------------------------------
  const scalar_t* k_row = A.k + token_idx * A.k_stride_t;
  const float* __restrict__ norm_weight = A.norm_weight;
  const float* __restrict__ norm_bias = A.norm_bias;
  float xlo = static_cast<float>(k_row[t]);
  float xhi = static_cast<float>(k_row[t + HALF_DIM]);

  float klo, khi, k_amax;
  const float s0 = wred64(xlo, sum_func);
  const float s1 = wred64(xhi, sum_func);
  const float mean = sum_func(s1, s0) / static_cast<float>(HEAD_DIM);

  float clo = xlo - mean;
  float chi = xhi - mean;
  const float q0 = wred64(clo * clo, sum_func);
  const float q1 = wred64(chi * chi, sum_func);
  const float ss = sum_func(q1, q0);
  const float inv_std = rsqrtf(ss / static_cast<float>(HEAD_DIM) + epsilon);

  klo = clo * inv_std * norm_weight[t] + norm_bias[t];
  khi = chi * inv_std * norm_weight[t + HALF_DIM] + norm_bias[t + HALF_DIM];
  klo = static_cast<float>(static_cast<scalar_t>(klo));
  khi = static_cast<float>(static_cast<scalar_t>(khi));

  klo = rope_lo(klo, t, is_neox, cos_ptr, sin_ptr);

  if constexpr (HADAMARD) {
    hadamard128_wave(klo, khi, t, hadamard_scale);
    klo = static_cast<float>(static_cast<scalar_t>(klo));
    khi = static_cast<float>(static_cast<scalar_t>(khi));
  }

  const float m0 = wred64(fabsf(klo), max_func);
  const float m1 = wred64(fabsf(khi), max_func);
  k_amax = max_func(m1, m0);
  float k_scale = fmaxf(k_amax, 1e-4f) / q_fp8_max;
  if (use_ue8m0) k_scale = exp2f(ceilf(log2f(k_scale)));

  const int64_t block_idx = slot_idx / cache_block_size;
  const int64_t block_offset = slot_idx % cache_block_size;
  const int64_t page_base = block_idx * cache_block_size * cache_stride;
  if (t == 0) {
    const int64_t dst_scale_idx = page_base + cache_block_size * HEAD_DIM +
                                  block_offset * HEAD_DIM * 4 / qblk;
    reinterpret_cast<float*>(A.kv_cache)[dst_scale_idx / 4] = k_scale;
  }
  const float k_inv_scale = 1.0f / k_scale;

#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int dim = t + h * HALF_DIM;
    int64_t dst_offset;
    if (preshuffle) {
      constexpr int TILE = 16;
      const int token_tile_id = block_offset / TILE;
      const int token_in_tile = block_offset % TILE;
      const int col_tile_id = dim / TILE;
      const int col_in_tile = dim % TILE;
      dst_offset = page_base + token_tile_id * (TILE * HEAD_DIM) +
                   col_tile_id * (TILE * TILE) + token_in_tile * TILE + col_in_tile;
    } else {
      dst_offset = page_base + block_offset * HEAD_DIM + dim;
    }
    A.kv_cache[dst_offset] =
        opus::cast<cache_t>((h == 0 ? klo : khi) * k_inv_scale);
  }
}

// FAT signature -- 35 arguments, ~232 B of kernarg.  Control for K4.
template <bool HADAMARD, bool KSPLIT = false, bool SPEC = false>
__global__ __launch_bounds__(HALF_DIM) void indexer_qk_had_wave_kernel(
    const scalar_t* __restrict__ q, cache_t* __restrict__ q_out,
    const scalar_t* __restrict__ weights, float* __restrict__ weights_out,
    const scalar_t* __restrict__ k, cache_t* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping,
    const float* __restrict__ norm_weight, const float* __restrict__ norm_bias,
    const int64_t* __restrict__ positions,
    const scalar_t* __restrict__ cos_cache, const scalar_t* __restrict__ sin_cache,
    const int num_tokens, const int n_heads, const int quant_block_size,
    const int cache_block_size, const int cache_stride,
    const int64_t q_stride_t, const int64_t q_stride_h, const int64_t q_stride_d,
    const int64_t q_out_stride_t, const int64_t q_out_stride_h,
    const int64_t q_out_stride_d,
    const int64_t weights_stride_t, const int64_t weights_stride_h,
    const int64_t weights_out_stride_t, const int64_t weights_out_stride_h,
    const int64_t k_stride_t, const int64_t k_stride_d,
    const int64_t cos_stride0, const int64_t sin_stride0,
    const float epsilon, const float weights_scale, const float hadamard_scale,
    const bool use_ue8m0, const bool preshuffle, const bool is_neox,
    const int max_position, const bool compute_all_q_rope) {
  QKArgs A;
  A.q = q; A.q_out = q_out; A.weights = weights; A.weights_out = weights_out;
  A.k = k; A.kv_cache = kv_cache; A.slot_mapping = slot_mapping;
  A.norm_weight = norm_weight; A.norm_bias = norm_bias; A.positions = positions;
  A.cos_cache = cos_cache; A.sin_cache = sin_cache;
  A.n_heads = n_heads; A.cache_block_size = cache_block_size;
  A.cache_stride = cache_stride; A.max_position = max_position;
  A.q_stride_t = static_cast<int>(q_stride_t);
  A.q_stride_h = static_cast<int>(q_stride_h);
  A.q_out_stride_t = static_cast<int>(q_out_stride_t);
  A.q_out_stride_h = static_cast<int>(q_out_stride_h);
  A.weights_stride_t = static_cast<int>(weights_stride_t);
  A.weights_out_stride_t = static_cast<int>(weights_out_stride_t);
  A.k_stride_t = static_cast<int>(k_stride_t);
  A.cos_stride0 = static_cast<int>(cos_stride0);
  A.sin_stride0 = static_cast<int>(sin_stride0);
  A.quant_block_size = quant_block_size;
  A.epsilon = epsilon; A.weights_scale = weights_scale;
  A.hadamard_scale = hadamard_scale;
  A.use_ue8m0_rt = use_ue8m0; A.preshuffle_rt = preshuffle;
  A.is_neox_rt = is_neox; A.compute_all_q_rope_rt = compute_all_q_rope;
  (void)num_tokens; (void)q_stride_d; (void)q_out_stride_d; (void)k_stride_d;
  (void)weights_stride_h; (void)weights_out_stride_h;
  qk_body<HADAMARD, KSPLIT, SPEC>(A);
}

}  // namespace

void indexer_qk_rope_hadamard_quant_and_cache(
    at::Tensor q, at::Tensor q_out, at::Tensor weights, at::Tensor weights_out,
    at::Tensor k, at::Tensor kv_cache, at::Tensor slot_mapping,
    at::Tensor norm_weight, at::Tensor norm_bias, at::Tensor positions,
    at::Tensor cos_cache, at::Tensor sin_cache, double epsilon,
    int64_t quant_block_size, const std::string& scale_fmt, double weights_scale,
    bool preshuffle, bool is_neox, bool compute_all_q_rope, bool hadamard) {
  const int num_tokens = std::min<int>(k.size(0), slot_mapping.size(0));
  const int head_dim = k.size(1);
  const int n_heads = q.size(1);
  const int rope_dim = cos_cache.size(-1) * 2;
  const int cache_block_size = kv_cache.size(1);
  const int cache_stride = kv_cache.size(2);
  const int max_position = cos_cache.size(0);
  const bool use_ue8m0 = scale_fmt == "ue8m0";

  TORCH_CHECK(head_dim == HEAD_DIM, "head_dim must be 128");
  TORCH_CHECK(rope_dim == ROPE_DIM, "rope_dim must be 64");
  TORCH_CHECK(quant_block_size == head_dim, "quant_block_size must equal head_dim");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16,
              "q/k must be bf16");
  TORCH_CHECK(weights.scalar_type() == at::kBFloat16, "weights must be bf16");
  TORCH_CHECK(weights_out.scalar_type() == at::kFloat, "weights_out must be fp32");
  TORCH_CHECK(norm_weight.scalar_type() == at::kFloat &&
                  norm_bias.scalar_type() == at::kFloat,
              "norm params must be fp32");
  TORCH_CHECK(cos_cache.dim() == 2 && sin_cache.dim() == 2, "cos/sin must be 2-D");
  TORCH_CHECK(cos_cache.stride(1) == 1 && sin_cache.stride(1) == 1,
              "cos/sin last dim must be contiguous");
  TORCH_CHECK(slot_mapping.scalar_type() == at::kLong, "slot_mapping must be int64");
  TORCH_CHECK(positions.scalar_type() == at::kLong, "positions must be int64");

  const auto* qp = reinterpret_cast<const scalar_t*>(q.data_ptr());
  auto* qop = reinterpret_cast<cache_t*>(q_out.data_ptr());
  const auto* wp = reinterpret_cast<const scalar_t*>(weights.data_ptr());
  auto* wop = weights_out.data_ptr<float>();
  const auto* kp = reinterpret_cast<const scalar_t*>(k.data_ptr());
  auto* cp = reinterpret_cast<cache_t*>(kv_cache.data_ptr());
  const auto* sp = slot_mapping.data_ptr<int64_t>();
  const auto* nwp = norm_weight.data_ptr<float>();
  const auto* nbp = norm_bias.data_ptr<float>();
  const auto* pp = positions.data_ptr<int64_t>();
  const auto* cosp = reinterpret_cast<const scalar_t*>(cos_cache.data_ptr());
  const auto* sinp = reinterpret_cast<const scalar_t*>(sin_cache.data_ptr());

  // rotate_activation(x) == hadamard_transform(x, scale=x.size(-1) ** -0.5)
  const float had_scale = 1.0f / sqrtf(static_cast<float>(HEAD_DIM));
  auto stream = c10::cuda::getCurrentCUDAStream();

  // The wave path indexes the head dimension directly, so the layout invariants
  // it relies on are asserted rather than carried as arguments.
  TORCH_CHECK(q.stride(2) == 1 && q_out.stride(2) == 1 && k.stride(1) == 1,
              "wave kernel needs contiguous head dim");
  TORCH_CHECK(weights.stride(1) == 1 && weights_out.stride(1) == 1,
              "wave kernel needs contiguous head axis on weights");
  // SPEC and the Hadamard are folded into the instantiation; refuse it if the
  // caller is not actually on that path.
  TORCH_CHECK(hadamard, "the fused indexer requires the inline Hadamard");
  TORCH_CHECK(use_ue8m0 && preshuffle && !is_neox && compute_all_q_rope &&
                  quant_block_size == HEAD_DIM,
              "specialised for ue8m0 + preshuffle + interleaved rope + "
              "compute_all_q_rope + quant_block == 128");
  dim3 wgrid(num_tokens, n_heads + 1);
  dim3 wblock(HALF_DIM);
  indexer_qk_had_wave_kernel<true, true, true><<<wgrid, wblock, 0, stream>>>(
      qp, qop, wp, wop, kp, cp, sp, nwp, nbp, pp, cosp, sinp, num_tokens,
      n_heads, static_cast<int>(quant_block_size), cache_block_size,
      cache_stride, q.stride(0), q.stride(1), q.stride(2), q_out.stride(0),
      q_out.stride(1), q_out.stride(2), weights.stride(0),
      weights.stride(1), weights_out.stride(0), weights_out.stride(1),
      k.stride(0), k.stride(1), cos_cache.stride(0), sin_cache.stride(0),
      static_cast<float>(epsilon), static_cast<float>(weights_scale),
      had_scale, use_ue8m0, preshuffle, is_neox, max_position,
      compute_all_q_rope);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("indexer_qk_rope_hadamard_quant_and_cache",
        &indexer_qk_rope_hadamard_quant_and_cache,
        "AITER indexer_qk_rope_quant_and_cache + optional inline 128-pt Hadamard",
        pybind11::arg("q"), pybind11::arg("q_out"), pybind11::arg("weights"),
        pybind11::arg("weights_out"), pybind11::arg("k"),
        pybind11::arg("kv_cache"), pybind11::arg("slot_mapping"),
        pybind11::arg("norm_weight"), pybind11::arg("norm_bias"),
        pybind11::arg("positions"), pybind11::arg("cos_cache"),
        pybind11::arg("sin_cache"), pybind11::arg("epsilon"),
        pybind11::arg("quant_block_size"), pybind11::arg("scale_fmt"),
        pybind11::arg("weights_scale"), pybind11::arg("preshuffle"),
        pybind11::arg("is_neox"), pybind11::arg("compute_all_q_rope"),
        pybind11::arg("hadamard"));
}
