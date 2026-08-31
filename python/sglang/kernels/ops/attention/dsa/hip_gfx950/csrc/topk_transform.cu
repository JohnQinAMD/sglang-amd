// Adapted from: https://github.com/ROCm/aiter  aiter/csrc/include/coop_topk.cuh
// (MIT)
//
// Top-k(2048) + page transform for the GLM-5.2 DSA indexer on gfx950 / MI355X.
// aiter's coop_topk runs one block per row, which at the decode shape occupies
// 6 blocks of a 256-CU GPU; this splits a row across G blocks instead.  Blocks
// agree without communicating: each derives the same threshold from the same
// per-row histogram -- already built by the logits kernel -- then rescans its
// own slice, writing winners straight out through the page table.  Ties are
// resolved exactly in the last-arriving block, which also pads -1 and zeroes
// the histogram, so no memset launch is needed.
//
// No grid-wide sync, no cooperative launch, no allocation, no host sync: HIP
// graph capturable.  All workspace is owned and kept alive by the caller.

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace dsa_topk {

constexpr uint32_t TOPK = 2048u;
// Coarse bin width, in bits of the fp16 ordered key: trades candidate-set size
// against global-histogram traffic.
#ifndef DSA_TOPK_HIST_BITS
#define DSA_TOPK_HIST_BITS 12
#endif
constexpr uint32_t HIST_BITS = (uint32_t)DSA_TOPK_HIST_BITS;
constexpr uint32_t HIST_BINS = 1u << HIST_BITS;
constexpr uint32_t LOW_BITS = 16u - HIST_BITS;
// -----------------------------------------------------------------------------
// Hierarchical (two-level) threshold.
constexpr uint32_t CBITS = 6u;
constexpr uint32_t CBINS = 1u << CBITS;               // 64, one wave wide
constexpr uint32_t FINE_PER_CRS = HIST_BINS >> CBITS; // 64 at HIST_BITS=12
static_assert(HIST_BITS >= CBITS,
              "coarse must be no wider than the fine histogram");
static_assert(FINE_PER_CRS <= 64u, "the fine group must fit in one wave");
// Row stride of the ghist workspace: the fine histogram followed by the coarse
// summary.  Both halves are zeroed by whoever owns the reset.
constexpr uint32_t GH_STRIDE = HIST_BINS + CBINS;
constexpr uint32_t GH_CRS_OFF = HIST_BINS;
// Threads per block.  Build-time tunable (Phase K): it is the only knob that
// reshapes the scan, which at G=64/BS=256 issues exactly ONE dwordx4 per thread
// (14 of 256 threads issue none) and is therefore one dependent round trip with
#ifndef DSA_TOPK_BS
#define DSA_TOPK_BS 256
#endif
constexpr uint32_t BS = (uint32_t)DSA_TOPK_BS; // threads per block
constexpr uint32_t RADIX = 256u;               // refinement radix
static_assert(BS % 64u == 0u, "BS must be a whole number of wave64 waves");
static_assert(BS >= 64u && BS <= 1024u, "HIP workgroup bound");
// Per-block LDS staging for the scatter's two output streams.  See k_scatter.
#ifndef DSA_TOPK_STAGE
#define DSA_TOPK_STAGE 512
#endif
constexpr uint32_t STAGE = (uint32_t)DSA_TOPK_STAGE;
// Candidates held in LDS by the refinement.  Above this it streams from global,
// which only happens on inputs where one coarse bin holds > REF_CAP elements.
constexpr uint32_t REF_CAP = 2048u;
// Candidate count up to which the refinement uses the O(n^2) rank method
// instead of the radix rounds.  See refine_row.
#ifndef DSA_TOPK_RANK_CAP
#define DSA_TOPK_RANK_CAP 256
#endif
constexpr uint32_t RANK_CAP = (uint32_t)DSA_TOPK_RANK_CAP;
// Phase O / PRANK: the parallel rank puts ONE THREAD ON ONE j, so it needs
// n <= BS as well as n <= RANK_CAP.  BS is a build knob (64..1024), so the
// bound is the min and the serial rank remains the fallback above it -- the
constexpr uint32_t PRANK_CAP = RANK_CAP < BS ? RANK_CAP : BS;
// Per-row counters are padded to their own cache line, and the two counters a
// block updates live in one 8-byte word, so a block reserves output space with
// ONE returning global atomic instead of two and rows do not false-share.
constexpr uint32_t RC_STRIDE = 32u;
constexpr uint32_t RC_WIN = 0u;  // winners emitted so far
constexpr uint32_t RC_CAND = 1u; // candidates appended so far
constexpr uint32_t RC_ARR = 4u;  // Phase H2 arrival counter (own dword)
// Phase O / PRANK: the DEPARTURE half of the two-phase row barrier.  The
// arrival counter alone is not a barrier that more than one block may pass:
// whoever resets it would race the blocks still spinning on it.  A separate
constexpr uint32_t RC_DEP = 5u;
// Phase O / PRANK observability.  A parallel rank that silently ranks nothing
// in a row returns the SAME answer whenever the missing blocks owned only
// non-selected candidates, so participation is counted rather than assumed:
constexpr uint32_t RC_DBG = 6u;
// Phase O: a SECOND observability dword, bumped once by every block that
// enters the PRANK tail, whatever refine branch it then takes.  RC_DBG counts
// blocks that RANKED, which is zero on any input that overflows RANK_CAP and
constexpr uint32_t RC_DBG2 = 7u;

// ---------------------------------------------------------------------------
// Phase G: page-table entries staged in LDS by k_scatter.  The page_size=64
// table for the production shape is 969 int32 = 3.9 KB per row, so one block
// can
constexpr uint32_t PT_LDS = 1024u;
// ... and the *window* form of the same idea.  A block's slice is CONTIGUOUS,
// so every position it can emit lies in one span of the page table: at G=32 and
// row_len 62000 the slice is ~1938 positions = 31 pages = 124 bytes.  Staging
constexpr uint32_t PT_WIN = 256u;

// Ordered 32-bit key: unsigned compare on the result matches float compare.
__device__ __forceinline__ uint32_t order_key32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Ordered 16-bit key of x rounded to fp16.  __float2half_rn is monotonic, so
// key16(a) <= key16(b) whenever a <= b, which is what makes the three-way
// split in K2 sound.
__device__ __forceinline__ uint32_t order_key16(float x) {
  __half h = __float2half_rn(x);
  unsigned short bits = __half_as_ushort(h);
  unsigned short key = (bits & 0x8000) ? (unsigned short)(~bits)
                                       : (unsigned short)(bits | 0x8000);
  return (uint32_t)key;
}

// Physical KV slot of a row-relative position: pt64[row][p >> 6] * 64 + (p &
// 63) for page_size 64, which is the definition of page_table_1.  PB/PM carry
// the shift and mask so page_size 1 collapses to the identity.
__device__ __forceinline__ int32_t slot_of(const int32_t *__restrict__ pt,
                                           uint32_t pos, uint32_t page_bits,
                                           uint32_t page_mask) {
  return (pt[pos >> page_bits] << page_bits) | (int32_t)(pos & page_mask);
}

struct Slice {
  uint32_t start, len;
};

// This block's slice, cut on float4 boundaries so every load stays 16B aligned.
__device__ __forceinline__ Slice slice_of(uint32_t row_len, uint32_t g,
                                          uint32_t G) {
  const uint32_t units = (row_len + 3u) / 4u;
  const uint32_t base = units / G;
  const uint32_t extra = units % G;
  const uint32_t my_u = base + (g < extra ? 1u : 0u);
  const uint32_t off_u = g * base + (g < extra ? g : extra);
  Slice s{};
  s.start = off_u * 4u;
  s.len = s.start >= row_len ? 0u : min(my_u * 4u, row_len - s.start);
  return s;
}

// Walk the slice applying op(value, row_relative_index).
template <typename Op>
__device__ __forceinline__ void scan_slice(const float *__restrict__ in,
                                           Slice sl, Op op) {
  const uint32_t tx = threadIdx.x;
  const uint32_t vec_len = sl.len & ~3u;
  const float4 *in4 = reinterpret_cast<const float4 *>(in + sl.start);
  const uint32_t n4 = vec_len >> 2;

  uint32_t i = tx;
  for (; i + 3u * BS < n4; i += 4u * BS) {
    const float4 v0 = in4[i];
    const float4 v1 = in4[i + BS];
    const float4 v2 = in4[i + 2u * BS];
    const float4 v3 = in4[i + 3u * BS];
    const float4 vv[4] = {v0, v1, v2, v3};
#pragma unroll
    for (uint32_t u = 0; u < 4u; ++u) {
      const float4 v = vv[u];
      const float vals[4] = {v.x, v.y, v.z, v.w};
      const uint32_t bpos = sl.start + ((i + u * BS) << 2);
#pragma unroll
      for (uint32_t j = 0; j < 4u; ++j) {
        op(vals[j], bpos + j);
      }
    }
  }
  for (; i < n4; i += BS) {
    const float4 v = in4[i];
    const float vals[4] = {v.x, v.y, v.z, v.w};
    const uint32_t bpos = sl.start + (i << 2);
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
      op(vals[j], bpos + j);
    }
  }
  for (uint32_t t = vec_len + tx; t < sl.len; t += BS) {
    op(in[sl.start + t], sl.start + t);
  }
}

constexpr uint32_t WAVE = 64u; // gfx950
constexpr uint32_t NWAVE = BS / WAVE;

// LDS histogram increment that collapses a whole wave into one atomic when
// every active lane agrees on the bin.
__device__ __forceinline__ void hist_add_agg(uint32_t *hist, uint32_t bin) {
  const uint64_t active = __ballot(1);
  const int leader = __ffsll((unsigned long long)active) - 1;
  const uint32_t lead_bin = __shfl(bin, leader, WAVE);
  if (__all(bin == lead_bin)) {
    if ((int)(threadIdx.x % WAVE) == leader) {
      atomicAdd(&hist[lead_bin], (uint32_t)__popcll(active));
    }
  } else {
    atomicAdd(&hist[bin], 1u);
  }
}

// Threshold search over NB counts: find bin b with above(b) < want <=
// above(b)+hist[b], where above(b) is the sum of all bins strictly greater.

template <uint32_t NB>
__device__ __forceinline__ void
find_thr(const uint32_t *__restrict__ hist, uint32_t *__restrict__ wtot,
         uint32_t want, uint32_t *out_thr, uint32_t *out_above) {
  // Phase N: NB < BS is legal.  PER is then 1 and the threads past NB read
  // nothing and hold a zero count.  A zero bin cannot satisfy the bracket
  // predicate below (it needs acc < want && acc + 0 >= want), and it adds
  constexpr uint32_t PER = (NB >= BS) ? (NB / BS) : 1u;
  const uint32_t tx = threadIdx.x;
  const uint32_t lane = tx % WAVE;
  const uint32_t wv = tx / WAVE;

  uint32_t local[PER];
  uint32_t mine = 0;
#pragma unroll
  for (uint32_t j = 0; j < PER; ++j) {
    if constexpr (NB >= BS) {
      local[j] = hist[tx * PER + j];
    } else {
      const uint32_t idx = tx * PER + j;
      local[j] = (idx < NB) ? hist[idx] : 0u;
    }
    mine += local[j];
  }

  uint32_t incl = mine;
#pragma unroll
  for (uint32_t o = 1; o < WAVE; o <<= 1) {
    const uint32_t nv = __shfl_up(incl, o, WAVE);
    if (lane >= o) {
      incl += nv;
    }
  }
  if (lane == WAVE - 1u) {
    wtot[wv] = incl;
  }
  __syncthreads();

  uint32_t base = 0, total = 0;
#pragma unroll
  for (uint32_t w = 0; w < NWAVE; ++w) {
    const uint32_t t = wtot[w];
    if (w < wv) {
      base += t;
    }
    total += t;
  }
  uint32_t acc = total - (base + incl); // strictly above this thread's group

#pragma unroll
  for (int j = (int)PER - 1; j >= 0; --j) {
    const uint32_t c = local[j];
    if (acc < want && acc + c >= want) {
      *out_thr = tx * PER + (uint32_t)j;
      *out_above = acc;
    }
    acc += c;
  }
  __syncthreads(); // wtot[] is scratch and out_thr/out_above must be visible
}

// Inclusive SUFFIX sum across one wave: lane l receives sum_{k>=l} v_k.
__device__ __forceinline__ uint32_t wave_suffix_sum(uint32_t v) {
  const uint32_t lane = threadIdx.x & (WAVE - 1u);
#pragma unroll
  for (uint32_t o = 1; o < WAVE; o <<= 1) {
    const uint32_t t = __shfl_down(v, o, WAVE);
    v += (lane + o < WAVE) ? t : 0u;
  }
  return v;
}

// The SAME threshold as find_thr<HIST_BINS>, derived from the two-level
// histogram.  A coarse bin is exactly the sum of FINE_PER_CRS consecutive fine
// bins, so SC(j) == S(j*FINE_PER_CRS) identically and the bracketed b is the
// same -- bit-identical, not merely equivalent.
//
// PAIRING GUARD.  Reading a histogram whose producer did not write the coarse
// summary fails SILENTLY: level 1's ballot is empty, thr comes out 0, every
// element classifies as an outright winner and the kernel returns an ARBITRARY
// 2048 of the row.  The check is free -- after the level-1 suffix sum lane 0
// holds SC(0), which for any row reaching here is exactly row_len.  On mismatch
// return HIER_BAD and let the caller fall back to the flat find_thr over the
// same histogram, so this path can never introduce a failure mode the flat one
// did not already have.
constexpr uint32_t HIER_BAD = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Phase L: the compact-candidate contract's constants.  A BUCKET is the top 8
// bits of the SAME order_key16 key the fine histogram uses, i.e. exactly
constexpr uint32_t CAND_NBUCKET = 256u;
constexpr uint32_t CAND_GBMAX = 64u; // max section-B blocks per row
static_assert(HIST_BITS >= 8u, "bucket key is the top 8 bits of the fp16 key");
constexpr uint32_t BKSH = HIST_BITS - 8u; // fine_bin >> BKSH == bucket

// WANT_SBK additionally returns S_fine(thr & ~(2^BKSH - 1)), i.e. how many of
// the row's elements the HISTOGRAM says have bucket >= thr>>BKSH.  That number
// is the Phase L interlock: it is what the logits kernel's per-block count
// table must
template <bool WANT_SBK, bool NOCHK = false>
__device__ __forceinline__ uint32_t find_thr_hier_impl(
    const uint32_t *__restrict__ gh, uint32_t want, uint32_t crs,
    uint32_t total_must_be, uint32_t *out_sbk, uint32_t *out_above = nullptr) {
  const uint32_t lane = threadIdx.x & (WAVE - 1u);

  // level 1 -- 64 coarse bins, one per lane
  const uint32_t c = crs;
  const uint32_t s = wave_suffix_sum(c);
  if constexpr (NOCHK) {
    if (__shfl(s, 0, WAVE) == 0xFFFFFFFFu) {
      return HIER_BAD;
    }
  } else {
    if (__shfl(s, 0, WAVE) != total_must_be) {
      return HIER_BAD;
    }
  }
  uint32_t sn = __shfl_down(s, 1, WAVE);
  if (lane == WAVE - 1u) {
    sn = 0u;
  }
  const uint64_t b1 = __ballot(s >= want && sn < want);
  // Cannot be empty (total == row_len > want and lane 63 sees sn == 0), but a
  // zero ballot would index gh at -1, so it degrades to bin 0 -- which is also
  // what find_thr leaves in *out_thr when it finds nothing.
  const uint32_t j = b1 ? (uint32_t)(__ffsll((unsigned long long)b1) - 1) : 0u;
  const uint32_t abv = b1 ? __shfl(sn, (int)j, WAVE) : 0u;

  // level 2 -- the FINE_PER_CRS fine bins under coarse bin j.  Lanes past the
  // group contribute 0, so they can never satisfy the predicate (want2 >= 1).
  const uint32_t f = (lane < FINE_PER_CRS) ? gh[j * FINE_PER_CRS + lane] : 0u;
  const uint32_t sf = wave_suffix_sum(f);
  uint32_t sfn = __shfl_down(sf, 1, WAVE);
  if (lane == WAVE - 1u) {
    sfn = 0u;
  }
  const uint32_t want2 = want - abv;
  const uint64_t b2 = __ballot(sf >= want2 && sfn < want2);
  const uint32_t l = b2 ? (uint32_t)(__ffsll((unsigned long long)b2) - 1) : 0u;
  if constexpr (WANT_SBK) {
    // The bucket boundary below thr is (thr & ~(2^BKSH-1)).  FINE_PER_CRS
    // (64) is a multiple of 2^BKSH (16), so that boundary lies in the SAME
    // coarse group and its suffix is abv + sf[l & ~(2^BKSH-1)] -- one
    constexpr uint32_t BM = (1u << BKSH) - 1u;
    *out_sbk = abv + __shfl(sf, (int)(l & ~BM), WAVE);
    // sfn[x] is by construction sf[x+1], i.e. the count in this coarse
    // group STRICTLY above fine bin x, so above(thr) = abv + sfn[l].
    *out_above = abv + __shfl(sfn, (int)l, WAVE);
  }
  return j * FINE_PER_CRS + l;
}

__device__ __forceinline__ uint32_t
find_thr_hier(const uint32_t *__restrict__ gh, uint32_t want, uint32_t crs,
              uint32_t total_must_be) {
  return find_thr_hier_impl<false>(gh, want, crs, total_must_be, nullptr);
}

// ---------------------------------------------------------------------------
// Exact refinement of the threshold bin + padding.

// Radix-256 on the full fp32 ordered key, four rounds = all 32 bits.  The
// candidate set is enumerated eight times (histogram + emit, per round), and
// membership in the still-undecided set is re-derived from the prefix resolved
__device__ __forceinline__ float ld_val(const float *p, bool agl) {
  return agl ? __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
             : *p;
}
__device__ __forceinline__ int32_t ld_idx(const int32_t *p, bool agl) {
  return agl ? __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
             : *p;
}

template <bool IN_LDS, bool AGL = false>
__device__ __forceinline__ void
refine_core(uint32_t n, uint32_t above, uint32_t remain,
            const uint32_t *__restrict__ s_key,
            const int32_t *__restrict__ s_slot, const float *__restrict__ cv,
            const int32_t *__restrict__ ci, int32_t *__restrict__ o,
            uint32_t *__restrict__ s_hist, uint32_t *__restrict__ s_grp,
            uint32_t *s_thr, uint32_t *s_above, uint32_t *s_emit) {
  const uint32_t tx = threadIdx.x;

  if (remain == 0 || n <= remain) {
    // Defensive: the threshold guarantees n >= remain, so this is the
    // degenerate path only.  Emit everything we have, then pad.
    for (uint32_t i = tx; i < n; i += BS) {
      const uint32_t p = above + i;
      if (p < TOPK) {
        o[p] = IN_LDS ? s_slot[i] : ld_idx(&ci[i], AGL);
      }
    }
    __syncthreads();
    for (uint32_t i = above + n + tx; i < TOPK; i += BS) {
      o[i] = -1;
    }
    return;
  }

  uint32_t prefix = 0;
  for (int r = 0; r < 4; ++r) {
    const uint32_t sh = 24u - (uint32_t)r * 8u;

    for (uint32_t i = tx; i < RADIX; i += BS) {
      s_hist[i] = 0u;
    }
    if (tx == 0) {
      *s_thr = 0;
      *s_above = 0;
    }
    __syncthreads();

    for (uint32_t i = tx; i < n; i += BS) {
      const uint32_t key = IN_LDS ? s_key[i] : order_key32(ld_val(&cv[i], AGL));
      const bool in_play =
          (r == 0) || (((key >> (sh + 8u)) << (sh + 8u)) == prefix);
      if (in_play) {
        hist_add_agg(s_hist, (key >> sh) & 0xFFu);
      }
    }
    __syncthreads();

    find_thr<RADIX>(s_hist, s_grp, remain, s_thr, s_above);

    const uint32_t thr = *s_thr;
    const uint32_t abv = *s_above;

    // Nothing above the threshold bin settles no winner, so the emit scan
    // would find nothing.  Skipping it matters on clustered rows, where the
    // early rounds are exactly this case.
    if (abv == 0 && r != 3) {
      prefix |= (thr << sh);
      continue;
    }

    for (uint32_t i = tx; i < n; i += BS) {
      const uint32_t key = IN_LDS ? s_key[i] : order_key32(ld_val(&cv[i], AGL));
      const bool in_play =
          (r == 0) || (((key >> (sh + 8u)) << (sh + 8u)) == prefix);
      if (!in_play) {
        continue;
      }
      const uint32_t bin = (key >> sh) & 0xFFu;
      // Last round: survivors in the threshold bin are numerically equal,
      // so any of them is a correct answer.  The TopK bound on the cursor
      // enforces the quota.
      if (bin > thr || (bin == thr && r == 3)) {
        const uint32_t p = atomicAdd(s_emit, 1u);
        if (p < TOPK) {
          o[p] = IN_LDS ? s_slot[i] : ld_idx(&ci[i], AGL);
        }
      }
    }
    __syncthreads();

    prefix |= (thr << sh);
    remain -= abv;
    if (remain == 0) {
      break;
    }
  }

  __syncthreads();
  // The emit loop above already wrote o[above .. *s_emit) directly, so all
  // that is left is the -1 padding.
  const uint32_t filled = *s_emit < TOPK ? *s_emit : TOPK;
  for (uint32_t i = filled + tx; i < TOPK; i += BS) {
    o[i] = -1;
  }
}

// The exact rank of the boundary candidates.  PRANK is the parallel form, and
// it is bit-identical to the serial rank element by element.
//
// One thread per j, one block per i-set: thread tx evaluates the order
// predicate for j = tx (needs n <= BS), the block reduces those BS booleans to
// rank_i with one __ballot per wave plus an NWAVE-entry LDS sum, and block pb
// owns i == pb (mod PB).  Partitioning the i axis instead would not work: at
// n = 55..120 each block would get two active threads still walking all n j's,
// leaving the serial depth unchanged.
//
// EXACTNESS.  (key_j, j) is a strict total order on [0, n) -- keys are
// order_key32 of the values, ties broken by the unique array index -- so rank
// is a bijection onto [0, n).  rank_i is summed over ALL j in EVERY block (only
// the i loop is partitioned), so a candidate on an i-partition boundary is
// ranked against the full set exactly as in the serial path; this is the one
// thing a partitioned rank can get wrong and it is structurally impossible
// here.  i == pb (mod PB) partitions [0, n) exactly, so every candidate is
// ranked by exactly one block.  Hence exactly `remain` candidates have
// rank < remain, o[above .. above+remain) is covered once and o[0..above) is
// untouched, and the value at o[above + r] does not depend on block scheduling.
//
// The barrier pair inside the i loop is safe because the loop's trip count is
// block-uniform.
template <bool AGL = false, bool PRANK = false>
__device__ __forceinline__ void
refine_row(uint32_t row, uint32_t above, uint32_t n_raw, uint32_t cap,
           const int32_t *__restrict__ cand_idx,
           const float *__restrict__ cand_val, int32_t *__restrict__ o,
           uint32_t *__restrict__ s_hist, uint32_t *__restrict__ s_grp,
           uint32_t *__restrict__ s_key, int32_t *__restrict__ s_slot,
           uint32_t *s_thr, uint32_t *s_above, uint32_t *s_emit,
           uint32_t dbg = 7u, bool have_pre = false, int32_t pre_slot = 0,
           float pre_val = 0.f, uint32_t pb = 0u, uint32_t PB = 1u,
           uint32_t *out_ranked = nullptr) {
  const uint32_t tx = threadIdx.x;
  const int32_t *__restrict__ ci = cand_idx + (size_t)row * cap;
  const float *__restrict__ cv = cand_val + (size_t)row * cap;

  const uint32_t n = n_raw > cap ? cap : n_raw;
  const uint32_t remain = above < TOPK ? TOPK - above : 0u;

  if (tx == 0) {
    *s_emit = above;
  }

  const bool in_lds = (n <= REF_CAP);
  if (in_lds && (dbg & 1u)) {
    // R6a: the caller may already have issued cand_idx[tx] / cand_val[tx]
    // WITHOUT waiting for `n`, which is what breaks the dependent chain
    // row_ends -> cursor -> candidates into a single round trip.  Those two
    const uint32_t i0 = have_pre ? tx + BS : tx;
    if (have_pre && tx < n) {
      s_key[tx] = order_key32(pre_val);
      s_slot[tx] = pre_slot;
    }
    for (uint32_t i = i0; i < n; i += BS) {
      s_key[i] = order_key32(ld_val(&cv[i], AGL));
      s_slot[i] = ld_idx(&ci[i], AGL); // k_scatter already translated it
    }
  }
  __syncthreads();

  if (!(dbg & 2u)) { /* timing probe: skip the refinement */
  } else if (in_lds && n <= RANK_CAP && remain != 0 && n > remain) {
    // Exact selection by rank -- which is what the four radix rounds were
    // computing the hard way.
    if constexpr (PRANK) {
      if (n <= PRANK_CAP) {
        const uint32_t lane = tx & (WAVE - 1u);
        const uint32_t wv = tx / WAVE;
        uint32_t nranked = 0u;
        // this thread's j, read ONCE and held in a register
        const uint32_t kj = (tx < n) ? s_key[tx] : 0u;
        for (uint32_t i = pb; i < n; i += PB) {
          const uint32_t ki = s_key[i]; // block-uniform read
          const bool p = (tx < n) && ((kj > ki) || (kj == ki && tx < i));
          const uint64_t m = __ballot(p);
          if (lane == 0) {
            s_grp[wv] = (uint32_t)__popcll((unsigned long long)m);
          }
          __syncthreads();
          uint32_t rank = 0;
#pragma unroll
          for (uint32_t q = 0; q < NWAVE; ++q) {
            rank += s_grp[q];
          }
          if (tx == 0 && rank < remain) {
            o[above + rank] = s_slot[i];
          }
          __syncthreads(); // s_grp is reused by the next i
          ++nranked;
        }
        if (out_ranked) {
          *out_ranked = nranked;
        }
        return;
      }
      // n > PRANK_CAP: one thread per j is not available.  Fall through
      // to the serial rank, in ONE block only -- never silently skipped.
      if (pb != 0u) {
        return;
      }
    }
    for (uint32_t i = tx; i < n; i += BS) {
      const uint32_t ki = s_key[i];
      uint32_t rank = 0;
      uint32_t j = 0;
      // Unrolled by 8: the naive loop issues one LDS read and then waits
      // on it, so it runs at LDS latency rather than LDS throughput, and
      // at n=320 that alone cost ~13 us.  Eight independent reads in
      for (; j + 8u <= n; j += 8u) {
        const uint32_t k0 = s_key[j + 0], k1 = s_key[j + 1];
        const uint32_t k2 = s_key[j + 2], k3 = s_key[j + 3];
        const uint32_t k4 = s_key[j + 4], k5 = s_key[j + 5];
        const uint32_t k6 = s_key[j + 6], k7 = s_key[j + 7];
        rank += (k0 > ki) || (k0 == ki && (j + 0u) < i);
        rank += (k1 > ki) || (k1 == ki && (j + 1u) < i);
        rank += (k2 > ki) || (k2 == ki && (j + 2u) < i);
        rank += (k3 > ki) || (k3 == ki && (j + 3u) < i);
        rank += (k4 > ki) || (k4 == ki && (j + 4u) < i);
        rank += (k5 > ki) || (k5 == ki && (j + 5u) < i);
        rank += (k6 > ki) || (k6 == ki && (j + 6u) < i);
        rank += (k7 > ki) || (k7 == ki && (j + 7u) < i);
      }
      for (; j < n; ++j) {
        const uint32_t kj = s_key[j];
        rank += (kj > ki) || (kj == ki && j < i);
      }
      if (rank < remain) {
        o[above + rank] = s_slot[i];
      }
    }
  } else if (PRANK && pb != 0u) {
    // The radix fallback (and the short/degenerate cases it absorbs) emits
    // through a shared cursor and pads o[..TOPK) -- one block's work by
    // construction.  b1_ctx2049 takes it (n = 1, remain = 1, so n > remain
  } else if (in_lds) {
    refine_core<true, AGL>(n, above, remain, s_key, s_slot, cv, ci, o, s_hist,
                           s_grp, s_thr, s_above, s_emit);
  } else {
    refine_core<false, AGL>(n, above, remain, s_key, s_slot, cv, ci, o, s_hist,
                            s_grp, s_thr, s_above, s_emit);
  }
}

// ---------------------------------------------------------------------------
// K2: threshold + scatter (page gather folded into the emit)
// ---------------------------------------------------------------------------

// PTMODE 0: page table read from global (the original path).
template <int PTMODE, bool FUSEREF, bool HIER = false, bool NACQ = false,
          bool PRANK = false, int PBLK = 0>
__global__ __launch_bounds__(BS) void k_scatter(
    const float *__restrict__ logits, const int32_t *__restrict__ row_ends,
    const int32_t *__restrict__ page_table, int32_t *__restrict__ out,
    uint32_t *__restrict__ ghist, int32_t *__restrict__ cursor,
    int32_t *__restrict__ cand_cnt, int32_t *__restrict__ cand_idx,
    float *__restrict__ cand_val, int64_t lg_stride, int64_t pt_stride,
    uint32_t page_bits, uint32_t page_mask, uint32_t cap, uint32_t G,
    uint32_t emit_on, uint32_t own_short, uint32_t dbg, uint32_t selfclean,
    const uint2 *__restrict__ cand2, const unsigned short *__restrict__ sfxt,
    const uint32_t *__restrict__ cmeta) {
  // With HIER the flat histogram never enters LDS at all: all that is left is
  // the RADIX-sized scratch the fused refinement needs.  That is 16 KB off
  // every block's footprint, which is what lets the FUSEREF form keep its
  static_assert(HIST_BINS >= RADIX, "refinement scratch must fit in s_hist");
  // Sized for the flat fallback even under HIER (see find_thr_hier): the
  // hierarchical path never touches it, and LDS cannot limit occupancy at
  // 192 blocks on 256 CUs.
  __shared__ uint32_t s_hist[HIST_BINS];
  __shared__ uint32_t s_grp[NWAVE]; // cross-wave scan fixup (see find_thr)
  __shared__ uint32_t s_thr, s_above;
  // Per-block staging.  One global atomicAdd per emitted element on a single
  // per-row cursor would be 2048 winners x 6 rows = 12288 returning atomics to
  // 6 addresses, serialising at ~69 cycles each.  Staging in LDS turns that
  __shared__ uint32_t s_wcnt, s_ccnt;
  __shared__ int32_t s_wbase, s_cbase;
  __shared__ int32_t s_wbuf[STAGE];
  __shared__ int32_t s_cidx[STAGE];
  __shared__ float s_cval[STAGE];
  // Phase G: the whole row's page table, when it fits.  LDSPT is a *template*
  // parameter and not a runtime flag for the same reason IN_LDS is in
  // refine_core: with a runtime select the compiler emits BOTH the LDS read
  __shared__ int32_t s_pt[PTMODE == 1 ? PT_LDS : (PTMODE == 2 ? PT_WIN : 1)];
  // Phase H2: the refinement's working set, only allocated when the
  // refinement is actually fused in (it is 24 KB, and reserving it
  // unconditionally would change this kernel's LDS footprint for every
  __shared__ uint32_t s_key[FUSEREF ? REF_CAP : 1];
  __shared__ int32_t s_slot[FUSEREF ? REF_CAP : 1];
  __shared__ uint32_t s_emit;
  __shared__ uint32_t s_last;
  // Phase L: per-section-B-block candidate counts, so the compact read loop
  // does not re-issue a uniform global load per element.
  __shared__ uint32_t s_bn;

  const uint32_t row = blockIdx.y;
  const uint32_t g = blockIdx.x;
  const uint32_t tx = threadIdx.x;

  const int32_t rl_s = row_ends[row];
  // R6a: the 64-bin coarse summary is issued HERE, before row_ends has even
  // come back, because its address depends only on blockIdx and it is always
  // in bounds.  Otherwise the chain is row_ends -> coarse bins -> fine group:
  uint32_t pre_crs = 0u;
  if constexpr (HIER) {
    pre_crs = ((const uint32_t *)ghist)[(size_t)row * GH_STRIDE + GH_CRS_OFF +
                                        (threadIdx.x & (WAVE - 1u))];
  }
  const uint32_t row_len = rl_s > 0 ? (uint32_t)rl_s : 0u;
  if (row_len <= TOPK) {
    // Phase H1.  When the histogram is built inside the logits kernel, k_hist
    // is gone -- and k_hist was the ONLY writer of the whole output row for
    // rows that need no selection (this is the ctx=0 and ctx=2048 case; at
    if (own_short) {
      const int32_t *__restrict__ pt0 = page_table + (int64_t)row * pt_stride;
      int32_t *__restrict__ o0 = out + (size_t)row * TOPK;
      for (uint32_t i = g * BS + tx; i < TOPK; i += G * BS) {
        o0[i] = i < row_len ? slot_of(pt0, i, page_bits, page_mask) : -1;
      }
    }
    return;
  }

  int32_t *__restrict__ rc = cursor + (size_t)row * RC_STRIDE;

  // Issued FIRST: these are independent of everything else in the kernel, and
  // the barrier below already waits for the histogram staging, so the page
  // table rides along for free.  969 int32 over 256 threads = four coalesced
  const Slice sl = slice_of(row_len, g, G);
  // First page-table entry this block can possibly touch.  Every position it
  // emits is inside its own slice, so the window is [pt_base, pt_base+npt).
  const uint32_t pt_base = (PTMODE == 2) ? (sl.start >> page_bits) : 0u;
  if (PTMODE != 0) {
    const int32_t *__restrict__ pt0 = page_table + (int64_t)row * pt_stride;
    uint32_t npt;
    if (PTMODE == 1) {
      npt = (uint32_t)pt_stride < PT_LDS ? (uint32_t)pt_stride : PT_LDS;
    } else {
      npt = sl.len ? (((sl.start + sl.len - 1u) >> page_bits) - pt_base + 1u)
                   : 0u;
      npt = npt < PT_WIN ? npt
                         : PT_WIN; // host guarantees it fits; belt-and-braces
    }
    for (uint32_t i = tx; i < npt; i += BS) {
      s_pt[i] = pt0[pt_base + i];
    }
  }

  // Phase K/C2 attribution probes.  emit_on==5 skips BOTH the 16 KB histogram
  // load and find_thr (thr forced to 0); emit_on==6 keeps the load but skips
  // find_thr.  Both give a WRONG answer and are timing-only, and both leave
  const uint32_t *__restrict__ gh =
      (const uint32_t *)ghist + (size_t)row * GH_STRIDE;
  // Every block derives the same threshold from the same global histogram, so
  // no communication is needed beyond the kernel boundary itself.
  uint32_t thr_h = 0u;
  bool need_flat = !HIER;
  uint32_t sbk = 0u;
  uint32_t hier_above = 0u;
  if constexpr (HIER) {
    // Issued BEFORE the barrier so its two (dependent) 256 B loads overlap
    // the page-table window staging above.  Needs neither LDS nor a barrier.
    if (emit_on < 5u) {
      thr_h = find_thr_hier_impl<false>(gh, TOPK, pre_crs, row_len, &sbk,
                                        &hier_above);
      need_flat = (thr_h == HIER_BAD); // block-uniform
    }
  }

  // -----------------------------------------------------------------------
  // Phase L: decide whether the compact candidate stream may be used.
  bool use_cand = false;
  uint32_t cap_b = 0u;
  uint32_t gb_all = 0u; // section-B blocks per row, read from cmeta
  uint32_t run_sh = 0u; // log2 of the padded per-segment boundary run
  uint32_t lo_b = 0u, hi_b = 0u;
  if (tx == 0) {
    s_bn = 0u;
  }
  if (need_flat && emit_on != 5u) {
    for (uint32_t i = tx; i < HIST_BINS; i += BS) {
      s_hist[i] = gh[i];
    }
  }
  if (tx == 0) {
    s_thr = 0;
    s_above = 0;
    s_wcnt = 0;
    s_ccnt = 0;
  }
  __syncthreads();

  if (need_flat && emit_on < 5u) {
    find_thr<HIST_BINS>(s_hist, s_grp, TOPK, &s_thr, &s_above);
    __syncthreads();
  }

  const uint32_t thr = need_flat ? s_thr : thr_h;

  // -----------------------------------------------------------------------
  // Phase K / R6b: clear this block's stripe of the FINE histogram HERE,
  // rather than leaving the whole 4160-dword-per-row reset to k_refine.
  if (HIER && selfclean && !need_flat && emit_on == 1u) {
    uint32_t *__restrict__ ghw = ghist + (size_t)row * GH_STRIDE;
    const uint32_t gj0 = (thr / FINE_PER_CRS) * FINE_PER_CRS;
    const uint32_t per = (HIST_BINS + G - 1u) / G;
    const uint32_t lo = g * per;
    const uint32_t hi = (lo + per) < HIST_BINS ? (lo + per) : HIST_BINS;
    for (uint32_t i = lo + tx; i < hi; i += BS) {
      if (i < gj0 || i >= gj0 + FINE_PER_CRS) {
        ghw[i] = 0u;
      }
    }
  }

  const float *__restrict__ in = logits + (int64_t)row * lg_stride;
  const int32_t *__restrict__ pt = page_table + (int64_t)row * pt_stride;
  int32_t *__restrict__ o = out + (size_t)row * TOPK;
  int32_t *__restrict__ ci = cand_idx + (size_t)row * cap;
  float *__restrict__ cv = cand_val + (size_t)row * cap;

  // The Phase G lookup.  `if constexpr` so exactly one of the two forms is
  // emitted (see the s_pt declaration).
  auto SLOT = [&](uint32_t p) -> int32_t {
    if constexpr (PTMODE == 1) {
      return (s_pt[p >> page_bits] << page_bits) | (int32_t)(p & page_mask);
    } else if constexpr (PTMODE == 2) {
      return (s_pt[(p >> page_bits) - pt_base] << page_bits) |
             (int32_t)(p & page_mask);
    } else {
      return slot_of(pt, p, page_bits, page_mask);
    }
  };

  if (emit_on == 0u || emit_on >= 5u) {
    // Timing probe: scan + threshold only, no emit.  Leaves the cursors at
    // zero, which is a valid state (the refinement then takes its
    // short-row/degenerate path), so repeated invocation stays consistent.
    uint32_t acc = 0;
    scan_slice(in, sl, [&](float v, uint32_t gi) {
      const uint32_t kb = order_key16(v) >> LOW_BITS;
      acc += (kb > thr) ? gi : 1u; // opaque to the optimiser
    });
    if (acc == 0xFFFFFFFFu) {
      o[0] = (int32_t)acc;
    }
    return;
  }
  // -----------------------------------------------------------------------
  // Phase L fast path: consume the logits kernel's compact, bucket-descending
  // candidate stream instead of re-reading the flat logits array.
  auto classify = [&](float v, uint32_t gi) {
    const uint32_t kb = order_key16(v) >> LOW_BITS;
    if (kb > thr) {
      const uint32_t p = atomicAdd(&s_wcnt, 1u);
      if (p < STAGE) {
        s_wbuf[p] = (int32_t)gi;
      } else {
        const unsigned long long old =
            atomicAdd((unsigned long long *)rc, 1ull);
        const uint32_t q = (uint32_t)old;
        if (q < TOPK) {
          o[q] = (int32_t)gi;
        }
      }
    } else if (kb == thr) {
      const uint32_t p = atomicAdd(&s_ccnt, 1u);
      if (p < STAGE) {
        s_cidx[p] = (int32_t)gi;
        s_cval[p] = v;
      } else {
        const unsigned long long old =
            atomicAdd((unsigned long long *)rc, 1ull << 32);
        const uint32_t q = (uint32_t)(old >> 32);
        if (q < cap) {
          if constexpr (FUSEREF) {
            __hip_atomic_store(&ci[q], (int32_t)gi, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&cv[q], v, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
          } else {
            ci[q] = (int32_t)gi;
            cv[q] = v;
          }
        }
      }
    }
  };

  scan_slice(in, sl, [&](float v, uint32_t gi) {
    const uint32_t kb = order_key16(v) >> LOW_BITS;
    if (kb > thr) {
      // Winner outright: monotonicity of order_key16 guarantees it beats
      // every element of the threshold bin.
      const uint32_t p = atomicAdd(&s_wcnt, 1u);
      if (p < STAGE) {
        s_wbuf[p] = (int32_t)gi;
      } else {
        const unsigned long long old =
            atomicAdd((unsigned long long *)rc, 1ull);
        const uint32_t q = (uint32_t)old;
        // The threshold guarantees q < TopK; the bound is belt-and-braces
        // so that no reachable state can scribble past the output row.
        if (q < TOPK) {
          o[q] = SLOT(gi);
        }
      }
    } else if (kb == thr) {
      const uint32_t p = atomicAdd(&s_ccnt, 1u);
      if (p < STAGE) {
        s_cidx[p] = (int32_t)gi;
        s_cval[p] = v;
      } else {
        const unsigned long long old =
            atomicAdd((unsigned long long *)rc, 1ull << 32);
        const uint32_t q = (uint32_t)(old >> 32);
        if (q < cap) {
          if constexpr (FUSEREF) {
            __hip_atomic_store(&ci[q], SLOT(gi), __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&cv[q], v, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
          } else {
            ci[q] = SLOT(gi);
            cv[q] = v;
          }
        }
      }
    }
  });

  // ---- FROM HERE ON THE TWO PATHS SHARE EVERYTHING ----------------------
  // Both fill the same s_wbuf/s_cidx/s_cval staging with the same
  // classification, so the flush, the single reserving atomic, the emit and
  __syncthreads();
  if (emit_on == 2u) {
    return;
  } // timing probe: classify+stage, no flush

  const uint32_t wn = s_wcnt < STAGE ? s_wcnt : STAGE;
  const uint32_t cn = s_ccnt < STAGE ? s_ccnt : STAGE;

  // Issue the page-table gathers FIRST.  They do not depend on the base this
  // block is about to reserve, so they overlap the atomic's round trip
  // instead of queueing behind it.  wn <= STAGE and cn <= STAGE, so with
  constexpr uint32_t PERT = (STAGE + BS - 1u) / BS;
  int32_t wslot[PERT], cslot[PERT];
  float cvalr[PERT];
  // Phase L: on the compact path the staged value is ALREADY the physical
  // slot -- the logits kernel computed it from the page-table entry it had
  // loaded for its own KV read -- so the gather is the identity and the page
  // table is
  const bool pre_slotted = false;
#pragma unroll
  for (uint32_t u = 0; u < PERT; ++u) {
    const uint32_t i = tx + u * BS;
    // emit_on==4 is a timing probe: same flush, no page-table gather.
    if (i < wn) {
      wslot[u] = (emit_on == 4u || pre_slotted) ? s_wbuf[i]
                                                : SLOT((uint32_t)s_wbuf[i]);
    }
    if (i < cn) {
      cslot[u] = (emit_on == 4u || pre_slotted) ? s_cidx[i]
                                                : SLOT((uint32_t)s_cidx[i]);
      cvalr[u] = s_cval[i];
    }
  }

  // path it takes for the full kernel and still restores both zero
  // invariants.
  if (tx == 0) {
    const unsigned long long pack =
        ((unsigned long long)cn << 32) | (unsigned long long)wn;
    const unsigned long long old = atomicAdd((unsigned long long *)rc, pack);
    s_wbase = (int32_t)(uint32_t)old;
    s_cbase = (int32_t)(uint32_t)(old >> 32);
  }
  __syncthreads();

#pragma unroll
  for (uint32_t u = 0; u < PERT; ++u) {
    const uint32_t i = tx + u * BS;
    if (i < wn) {
      const uint32_t p = (uint32_t)s_wbase + i;
      if (p < TOPK) {
        o[p] = wslot[u];
      }
    }
    if (i < cn) {
      const uint32_t q = (uint32_t)s_cbase + i;
      if (q < cap) {
        if constexpr (FUSEREF) {
          // Phase H2 handshake, cheap half.  These are the ONLY bytes
          // another block reads inside this kernel, and there are ~65
          // of them per block, so making just these stores agent-scope
          __hip_atomic_store(&ci[q], cslot[u], __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
          __hip_atomic_store(&cv[q], cvalr[u], __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
        } else {
          ci[q] = cslot[u];
          cv[q] = cvalr[u];
        }
      }
    }
  }

  if constexpr (FUSEREF) {
    // Release, the cheap way: the candidate stores above are already
    // write-through to a coherent point, so all that is needed is to wait
    // for them to retire.  No __threadfence() here -- see the comment on
    __builtin_amdgcn_s_waitcnt(/*vmcnt(0)*/ 0x0f70);
    __syncthreads();
    if (tx == 0) {
      const uint32_t old =
          __hip_atomic_fetch_add((uint32_t *)&rc[RC_ARR], 1u, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
      // PRANK needs this block's ARRIVAL RANK, the non-PRANK form needs
      // only the is-last flag, and no form needs both.  They share
      // s_last rather than adding a slot: a separate __shared__ dword
      s_last = PRANK ? old : ((old + 1u == G) ? 1u : 0u);
    }
    __syncthreads();
    // PBLK: how many of the row's blocks take part in the parallel rank.
    const uint32_t PB_N =
        (PBLK == 0 || (uint32_t)PBLK > G) ? G : (uint32_t)PBLK;
    const uint32_t pfirst = G - PB_N;
    const uint32_t pidx = s_last - pfirst; // valid only if s_last>=pfirst
    // -------------------------------------------------------------------
    // Phase O / PRANK, the FUSED harness.  The arrival counter stops being
    // an ELECTION (one block continues, G-1 return) and becomes the arrival
    if constexpr (PRANK) {
      if (s_last < pfirst) {
        return;
      }
      if (tx == 0) {
        uint32_t v =
            __hip_atomic_load((uint32_t *)&rc[RC_ARR], __ATOMIC_RELAXED,
                              __HIP_MEMORY_SCOPE_AGENT);
        while (v < G) {
          __builtin_amdgcn_s_sleep(2);
          v = __hip_atomic_load((uint32_t *)&rc[RC_ARR], __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT);
        }
      }
      __syncthreads();
    } else if (!s_last) {
      return;
    }

    // ACQUIRE side of the handshake.
    if (!NACQ) {
      __threadfence();
    }

    // The reset is now ONE block's work (the row's histogram is dead, see
    // above), which is the case the original k_refine comment measured at
    // 2 us of pure latency for a strided dword loop.  Four dwordx4 stores
    static_assert(GH_STRIDE % 4u == 0u, "vectorised reset needs 4 | stride");
    // Under PRANK the histogram reset is spread over the row's blocks
    // instead of being one block's 4160-dword store loop; every block has
    // long since finished reading the two live regions (the barrier above
    if (PRANK && !(selfclean && HIER && !need_flat)) {
      uint4 *__restrict__ ghw4 = (uint4 *)(ghist + (size_t)row * GH_STRIDE);
      const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
      for (uint32_t i = pidx * BS + tx; i < GH_STRIDE / 4u; i += PB_N * BS) {
        ghw4[i] = z4;
      }
    } else if (selfclean && HIER && !need_flat) {
      // Phase K/R6b, fused form.  Every block already cleared its stripe
      // of the FINE histogram on the way in (see the R6b block above),
      // excluding the threshold group -- which is the only fine region any
      uint32_t *__restrict__ ghw = ghist + (size_t)row * GH_STRIDE;
      const uint32_t gj0 = (thr / FINE_PER_CRS) * FINE_PER_CRS;
      for (uint32_t i = tx; i < FINE_PER_CRS; i += BS) {
        ghw[gj0 + i] = 0u;
      }
      for (uint32_t i = tx; i < CBINS; i += BS) {
        ghw[GH_CRS_OFF + i] = 0u;
      }
    } else {
      uint4 *__restrict__ ghw4 = (uint4 *)(ghist + (size_t)row * GH_STRIDE);
      const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
      for (uint32_t i = tx; i < GH_STRIDE / 4u; i += BS) {
        ghw4[i] = z4;
      }
    }

    const uint32_t above =
        NACQ ? (uint32_t)__hip_atomic_load(&rc[RC_WIN], __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT)
             : (uint32_t)rc[RC_WIN];
    const uint32_t nraw =
        NACQ ? (uint32_t)__hip_atomic_load(&rc[RC_CAND], __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT)
             : (uint32_t)rc[RC_CAND];
    // R6a, fused form: the candidate prefetch is issued alongside the two
    // cursor reads instead of behind them.  Unconditionally in bounds
    // (cap >= TOPK >= BS); entries at or past n are discarded.
    const int32_t pre_slot = ld_idx(&cand_idx[(size_t)row * cap + tx], NACQ);
    const float pre_val = ld_val(&cand_val[(size_t)row * cap + tx], NACQ);
    __syncthreads(); // every thread has read them before they are cleared
    if (!PRANK && tx == 0) {
      rc[RC_WIN] = 0;
      rc[RC_CAND] = 0;
      rc[RC_ARR] = 0;
    }
    uint32_t nranked = 0u;
    refine_row<NACQ, PRANK>(row, above, nraw, cap, cand_idx, cand_val, o,
                            s_hist, s_grp, s_key, s_slot, &s_thr, &s_above,
                            &s_emit, dbg, true, pre_slot, pre_val, pidx, PB_N,
                            &nranked);
    if (PRANK && tx == 0 && nranked != 0u) {
      __hip_atomic_fetch_add((uint32_t *)&rc[RC_DBG], 1u, __ATOMIC_RELAXED,
                             __HIP_MEMORY_SCOPE_AGENT);
    }
    // Departure half of the row barrier.  The counters cannot be reset by
    // the block that happens to finish first -- the other G-1 are still
    // spinning on RC_ARR and still reading RC_WIN/RC_CAND -- so the last
    if constexpr (PRANK) {
      __syncthreads();
      if (tx == 0) {
        // Observability, issued HERE and not before the spin.
        __hip_atomic_fetch_add((uint32_t *)&rc[RC_DBG2], 1u, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
        const uint32_t d =
            __hip_atomic_fetch_add((uint32_t *)&rc[RC_DEP], 1u,
                                   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (d + 1u == PB_N) {
          rc[RC_WIN] = 0;
          rc[RC_CAND] = 0;
          rc[RC_ARR] = 0;
          rc[RC_DEP] = 0;
        }
      }
    }
  }
}

} // namespace dsa_topk

// ---------------------------------------------------------------------------
// Phase O / PRANK liveness guard.
static bool prank_resident_ok(const void *fn, int blocks_per_row, int rows) {
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess) {
    return false;
  }
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess) {
    return false;
  }
  int per_cu = 0;
  if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
          &per_cu, fn, (int)dsa_topk::BS, 0) != hipSuccess) {
    return false;
  }
  if (per_cu <= 0) {
    return false;
  }
  const long long slots =
      (long long)per_cu * (long long)prop.multiProcessorCount;
  return (long long)blocks_per_row * (long long)rows <= slots;
}

// ---------------------------------------------------------------------------
// host entry
// ---------------------------------------------------------------------------

// The accepted Phase D configuration, baked in rather than selected:
//   PTMODE 2   per-block window of the page table staged in LDS
//   FUSEREF    the refinement runs in k_scatter's last-arriving block
void topk_transform(torch::Tensor logits, torch::Tensor row_ends,
                    torch::Tensor page_table, torch::Tensor out,
                    torch::Tensor ghist, torch::Tensor cursor,
                    torch::Tensor cand_cnt, torch::Tensor cand_idx,
                    torch::Tensor cand_val, int64_t g_per_row,
                    int64_t page_size) {
  const int64_t R = logits.size(0);
  const int64_t L = logits.stride(0);
  const int64_t PTS = page_table.stride(0);
  const int64_t cap = cand_idx.size(1);
  const uint32_t G = (uint32_t)g_per_row;
  // page_size must be 1 or a power of two; both give the identical mapping.
  uint32_t PB = 0, PM = 0;
  for (int64_t ps = page_size; ps > 1; ps >>= 1) {
    ++PB;
  }
  PM = (page_size > 1) ? (uint32_t)(page_size - 1) : 0u;
  TORCH_CHECK((page_size & (page_size - 1)) == 0 && page_size >= 1,
              "page_size must be a power of two, got ", page_size);

  // The histogram width is a compile-time property of the kernel.  Getting it
  // wrong on the host silently walks off the end of ghist and corrupts the
  // buffers next to it, which is exactly what happened once already.
  TORCH_CHECK(ghist.numel() == R * (int64_t)dsa_topk::GH_STRIDE,
              "ghist must be [rows, ", dsa_topk::GH_STRIDE, "], got ",
              ghist.numel());
  TORCH_CHECK(cap >= (int64_t)dsa_topk::TOPK,
              "candidate capacity must be >= TopK");
  TORCH_CHECK(logits.scalar_type() == at::kFloat &&
                  out.scalar_type() == at::kInt,
              "dtype");
  // PTMODE 2 is baked into the instantiation, so the window it stages has to
  // fit -- checked here rather than dispatched around.  page_size 64, G 64 and
  // a 62016-token context need 18 of the 256 entries.
  TORCH_CHECK(
      (PTS + (int64_t)G - 1) / (int64_t)G + 2 <= (int64_t)dsa_topk::PT_WIN,
      "page-table window needs <= ", dsa_topk::PT_WIN, " entries per block");

  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((unsigned)G, (unsigned)R, 1);
  dim3 blk(dsa_topk::BS, 1, 1);
  // The four trailing scalars are emit_on=1 (emit the selection), own_short=1
  // (the logits kernel built the histogram, so k_scatter owns the short-row
  // path), dbg=7 and selfclean=0.  The candidate-stream pointers are unused
  // here.
#define SCATTER_ARGS                                                           \
  grid, blk, 0, stream, logits.data_ptr<float>(),                              \
      row_ends.data_ptr<int32_t>(), page_table.data_ptr<int32_t>(),            \
      out.data_ptr<int32_t>(), (uint32_t *)ghist.data_ptr<int32_t>(),          \
      cursor.data_ptr<int32_t>(), cand_cnt.data_ptr<int32_t>(),                \
      cand_idx.data_ptr<int32_t>(), cand_val.data_ptr<float>(), L, PTS, PB,    \
      PM, (uint32_t)cap, G, 1u, 1u, 7u, 0u, nullptr, nullptr, nullptr

  // The persistent-rank tail finishes the exact rank behind a row barrier, so
  // its blocks must be co-resident.  When they would not be, the ordinary
  // kernel-boundary form is launched instead and the selector cannot hang.
  if (prank_resident_ok(
          (const void *)dsa_topk::k_scatter<2, true, true, true, true>, (int)G,
          (int)R)) {
    hipLaunchKernelGGL((dsa_topk::k_scatter<2, true, true, true, true, 16>),
                       SCATTER_ARGS);
  } else {
    hipLaunchKernelGGL((dsa_topk::k_scatter<2, true, true, true>),
                       SCATTER_ARGS);
  }
#undef SCATTER_ARGS
}

// Row stride of the ghist workspace = fine bins + the Phase K coarse summary.
// The workspace MUST be sized on this, not on hist_bins(): getting it wrong
// walks off the end of ghist into whatever is allocated next.
int64_t hist_stride() { return (int64_t)dsa_topk::GH_STRIDE; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_transform", &topk_transform, "phase D top-k + page transform",
        py::arg("logits"), py::arg("row_ends"), py::arg("page_table"),
        py::arg("out"), py::arg("ghist"), py::arg("cursor"),
        py::arg("cand_cnt"), py::arg("cand_idx"), py::arg("cand_val"),
        py::arg("g_per_row"), py::arg("page_size"));
  m.def("hist_stride", &hist_stride,
        "ghist row stride (fine bins + coarse summary)");
}
