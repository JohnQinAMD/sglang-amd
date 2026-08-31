// SPDX-License-Identifier: MIT
// Phase D section-C top-k(2048) + page transform for gfx950 / MI355X.
//
// WHY THIS EXISTS
// ---------------
// aiter's coop_topk_kernel<2048,12,4096,1024> runs ONE block per row.  At the
// production decode shape (batch 6, ctx 62000) that is 6 blocks on a 256-CU
// GPU.  Measured: 25.2 us to move 2 x 1.49 MB = 2.98 MB, i.e. 118 GB/s, i.e.
// ~2% of MI355X HBM bandwidth and ~20 GB/s per active CU.  A probe of the real
// logits (experiments/phaseD/probe_logits.py) shows the kernel is NOT stuck on
// LDS-atomic contention -- 60.5 of 64 lanes in a wave hit distinct histogram
// bins, and round 1 never fires (threshold bin holds 57-120 of 62000, against
// TieCap 1024).  It takes the intended two-pass fast path and is simply
// starved of memory-level parallelism: 6 blocks x 16 waves cannot keep enough
// loads in flight to cover HBM latency.
//
// So the fix is parallelism *within* a row, done with ordinary launches and no
// grid-wide synchronisation: agreement between blocks happens only at kernel
// boundaries, exactly as aiter's own row-split path does, but with three
// kernels instead of five and with the page-table gather folded into the emit.
//
//   K1 hist    : grid (G, R).  Each block histograms its slice of the row into
//                LDS (1024 bins on the top 10 bits of the fp16 ordered key) and
//                flushes non-zero bins into a per-row global histogram with
//                global atomics.  Also zeroes the row cursors.  Short rows
//                (len <= TopK) are finished here: they need no selection.
//   K2 scatter : grid (G, R).  Every block re-derives the same threshold bin
//                from the same global histogram (no communication needed), then
//                re-scans its slice.  Keys above the threshold bin are winners
//                outright and are written straight to the output *already
//                translated through the page table*; keys equal to it are
//                appended to a per-row candidate list.
//   K3 refine  : grid (R).  One block per row resolves the candidate list
//                exactly by radix-256 on the full fp32 ordered key, four rounds,
//                so ties are broken on the untruncated value and the answer is
//                bit-exact with a float64 reference.  Then it pads -1 and zeroes
//                the global histogram for the next invocation.
//
// The fp16 binning key and the exact-refinement structure are taken from
// aiter's coop_topk.cuh (MIT), whose header explains why fp16 beats fp32 for
// the coarse bin on clustered logits.  What is new here is the block
// decomposition, the atomic-free inter-kernel agreement, the folded page
// gather, and the self-cleaning histogram (K3 zeroes it, so no memset launch).
//
// NO grid-wide sync, NO cooperative launch, NO allocation, NO host sync:
// HIP-graph capturable.  All workspace is owned and kept alive by the caller.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

namespace phased {

constexpr uint32_t TOPK      = 2048u;
// Coarse bin width, in bits of the fp16 ordered key.  Build-time tunable: it
// trades candidate-set size (refinement work) against global-histogram traffic
// (flush atomics in K1, the dependent per-block read in K2, and the reset in K3).
#ifndef PHASED_HIST_BITS
#define PHASED_HIST_BITS 12
#endif
constexpr uint32_t HIST_BITS = (uint32_t)PHASED_HIST_BITS;
constexpr uint32_t HIST_BINS = 1u << HIST_BITS;
constexpr uint32_t LOW_BITS  = 16u - HIST_BITS;
// -----------------------------------------------------------------------------
// Phase K / C1: hierarchical (two-level) threshold.
//
// Every one of the 192 k_scatter blocks derives the SAME threshold from the SAME
// per-row histogram, and each of them was reading all 4096 bins (16 KB) and
// running a full 4096-element block-wide prefix scan to do it -- 3.1 MB of
// redundant traffic and 192 redundant scans, against a data scan of only ~7.6
// logits per thread.
//
// So section B also maintains a 64-bin COARSE summary of the same histogram
// (coarse bin = top CBITS bits of the same order_key16 key = fine_bin >>
// (HIST_BITS - CBITS), i.e. the sum of FINE_PER_COARSE consecutive fine bins).
// k_scatter then scans 64 coarse bins to find the coarse threshold bin, and only
// the FINE_PER_COARSE fine bins underneath it.  Two 256 B loads and two 64-lane
// wave scans instead of 16 KB and one 4096-element block scan.
//
// The derived threshold is BIT-IDENTICAL, not merely equivalent: it is the same
// integer histogram summed in a different association order, and integer
// addition is associative.  probe_hier.py checks the equality directly.
// -----------------------------------------------------------------------------
constexpr uint32_t CBITS         = 6u;
constexpr uint32_t CBINS         = 1u << CBITS;             // 64, one wave wide
constexpr uint32_t FINE_PER_CRS  = HIST_BINS >> CBITS;      // 64 at HIST_BITS=12
static_assert(HIST_BITS >= CBITS, "coarse must be no wider than the fine histogram");
static_assert(FINE_PER_CRS <= 64u, "the fine group must fit in one wave");
// Row stride of the ghist workspace: the fine histogram followed by the coarse
// summary.  Both halves are zeroed by whoever owns the reset.
constexpr uint32_t GH_STRIDE     = HIST_BINS + CBINS;
constexpr uint32_t GH_CRS_OFF    = HIST_BINS;
// Threads per block.  Build-time tunable (Phase K): it is the only knob that
// reshapes the scan, which at G=64/BS=256 issues exactly ONE dwordx4 per thread
// (14 of 256 threads issue none) and is therefore one dependent round trip with
// nothing to pipeline.  Halving BS doubles the loads per thread at the same
// total wave count.
//
// Phase N: the old upper bound "find_thr<RADIX> needs RADIX >= BS, so BS <= 256"
// was a property of find_thr's indexing, not of the algorithm.  find_thr now
// handles NB < BS by giving the surplus threads an all-zero bin (see the
// `if constexpr(NB >= BS)` there); a zero bin can never satisfy the bracket
// predicate `acc < want && acc + c >= want`, so it contributes nothing and the
// answer is bit-identical.  The NB >= BS arm is textually the old code, so the
// BS = 256 codegen is unchanged -- verified by disassembly diff against the
// accepted .so, see experiments/phaseN/.
// Everything else (STAGE/BS, NWAVE, the flush loops, the coarse reduction's
// HIST_BINS/BS run per thread) was already expressed in BS.
#ifndef PHASED_BS
#define PHASED_BS 256
#endif
constexpr uint32_t BS        = (uint32_t)PHASED_BS;  // threads per block
constexpr uint32_t RADIX     = 256u;                 // refinement radix
static_assert(BS % 64u == 0u, "BS must be a whole number of wave64 waves");
static_assert(BS >= 64u && BS <= 1024u, "HIP workgroup bound");
// Per-block LDS staging for the scatter's two output streams.  See k_scatter.
// Phase N: build-time tunable so that the STAGE/BS ratio (PERT, the per-thread
// register run in the flush) can be swept jointly with BS.  Default unchanged.
#ifndef PHASED_STAGE
#define PHASED_STAGE 512
#endif
constexpr uint32_t STAGE     = (uint32_t)PHASED_STAGE;
// Candidates held in LDS by the refinement.  Above this it streams from global,
// which only happens on inputs where one coarse bin holds > REF_CAP elements.
constexpr uint32_t REF_CAP   = 2048u;
// Candidate count up to which the refinement uses the O(n^2) rank method
// instead of the radix rounds.  See refine_row.
//
// Phase K TRIED AND REVERTED: raising this to 512 to rescue the h10 width.  The
// reasoning was that the O(n^2) rank loop is only n LDS broadcasts per thread,
// so n = 472 should cost ~0.25 us against four dependent radix rounds -- and
// that the Phase H figure "h10 makes k_refine 5.84 -> 9.18" was a cliff in this
// constant rather than a property of h10.  MEASURED: at h10 the narrower
// histogram really does buy 1.70 us in section B (logits_hist 12.66 -> 10.96),
// but k_refine went to 32.00 us and the stack to 68.20.  So the rank path is NOT
// cheap at n in the hundreds, the cliff is real, and 256 stands.  Left as a
// build knob with the measurement attached so nobody re-derives the same idea.
#ifndef PHASED_RANK_CAP
#define PHASED_RANK_CAP 256
#endif
constexpr uint32_t RANK_CAP  = (uint32_t)PHASED_RANK_CAP;
// Phase O / PRANK: the parallel rank puts ONE THREAD ON ONE j, so it needs
// n <= BS as well as n <= RANK_CAP.  BS is a build knob (64..1024), so the
// bound is the min and the serial rank remains the fallback above it -- the
// parallel path is never silently skipped, it is explicitly not entered.
constexpr uint32_t PRANK_CAP = RANK_CAP < BS ? RANK_CAP : BS;
// Per-row counters are padded to their own cache line and the two counters a
// block updates live in one 8-byte word, so a block reserves output space with
// ONE returning global atomic instead of two, and rows do not false-share.
// Before this, `cursor[6]` and `cand_cnt[6]` were each a single 24-byte array:
// every one of the 192 blocks hit the SAME line twice, serially, and the flush
// cost 8.7 us of the kernel's 12.9.
// Phase O, MEASURED AND REVERTED.  This was widened to 96 dwords (three 128 B
// lines) so RcMap<true> could give the PRANK spin dword a line of its own.
// Line-isolating it moved the composed stack by -0.080 and -0.120 us over two
// independent processes -- i.e. nothing, and if anything slightly worse -- so
// the wide stride is not carried.  See experiments/phaseO_C/pad_run{1,2}.json.
//
// The false-sharing HAZARD is real and was separately quantified
// (attrib_run{1,2}.json): the identical agent-scope atomic issued just before
// the spin costs +0.520 / +0.520 us when it lands in the spin line and
// -0.040 / +0.120 us when it does not.  What the null says is only that the
// shipped kernel does not write that line concurrently enough for isolation to
// pay -- not that writing it is safe.  Anyone adding a per-row counter written
// during the spin window should expect to pay ~0.5 us for it.
constexpr uint32_t RC_STRIDE = 32u;
constexpr uint32_t RC_WIN    = 0u;    // winners emitted so far
constexpr uint32_t RC_CAND   = 1u;    // candidates appended so far
constexpr uint32_t RC_ARR    = 4u;    // Phase H2 arrival counter (own dword)
// Phase O / PRANK: the DEPARTURE half of the two-phase row barrier.  The
// arrival counter alone is not a barrier that more than one block may pass:
// whoever resets it would race the blocks still spinning on it.  A separate
// departure counter lets the last block OUT do the reset, after every block has
// provably read rc[RC_WIN]/rc[RC_CAND].  Dword 5 of a 32-dword row line.
constexpr uint32_t RC_DEP    = 5u;
// Phase O / PRANK observability.  A parallel rank that silently ranks nothing
// in a row returns the SAME answer whenever the missing blocks owned only
// non-selected candidates, so participation is counted rather than assumed:
// every block that ranks at least one candidate bumps this dword, and the
// prediction -- written before the measurement -- is
//     participation[row] == min(G, n[row])
// i.e. 64/64/64/55/64/64 at the primary shape's n = 88/90/120/55/66/74
// (experiments/phaseO_C/refine_shape.json).  Dword 6 of a 32-dword row line,
// provably dead, and NOT reset, so one eager call reads back exactly the count.
constexpr uint32_t RC_DBG    = 6u;
// Phase O: a SECOND observability dword, bumped once by every block that
// enters the PRANK tail, whatever refine branch it then takes.  RC_DBG counts
// blocks that RANKED, which is zero on any input that overflows RANK_CAP and
// falls to the radix path -- e.g. the tie-heavy fixture, where the whole row
// shares one bin and n ~ 7750.  So RC_DBG cannot answer "did the parallel-rank
// kernel run at all", and that is exactly the question a tie-heavy audit has to
// answer.  This one can.  Dword 7 of a 32-dword row line, provably dead.
constexpr uint32_t RC_DBG2   = 7u;

// ---------------------------------------------------------------------------
// Phase O: THE ROW COUNTER LAYOUT, as a compile-time map.
//
// RC_WIN/RC_CAND are pinned at 0/1 in BOTH layouts and are not negotiable: the
// flush reserves both with ONE 64-bit atomicAdd on &rc[0], so they must stay
// adjacent and 8 B aligned.
//
// PADRC == false  reproduces the shipped layout exactly: ARR/DEP/DBG/DBG2 sit
//                 at dwords 4..7, i.e. in the SAME 128 B line as the two
//                 cursors.  This is the control for the false-sharing question,
//                 not a straw man -- the sharing pattern inside the line is
//                 identical to what ships today.
// PADRC == true   gives the spin dword a 128 B line to itself and puts the
//                 departure/debug counters in a third line:
//                     line 0  dwords  0.. 31   RC_WIN, RC_CAND
//                     line 1  dwords 32.. 63   ARR      <-- spun on, ALONE
//                     line 2  dwords 64.. 95   DEP, DBG, DBG2
//
// WHY THIS IS THE EXPERIMENT.  Every block of a row reserves output with a
// 64-bit atomic on rc[0:1] and then bumps rc[ARR]; under PRANK the blocks that
// have already arrived are SPINNING on rc[ARR] while later blocks are still
// issuing their reserving atomics into the same line.  Measured with an
// injected write, that costs 0.6-1.7 us and scales with the spinner count, so
// the question is what the kernel already pays for the writes it makes anyway.
//
// A 128 B exclusive span is also correct if the line is 64 B: an exclusive
// 128 B region contains two 64 B lines and nothing else lives in either.
// ---------------------------------------------------------------------------
template <bool PADRC>
struct RcMap
{
    static constexpr uint32_t ARR  = PADRC ? 32u : RC_ARR;
    static constexpr uint32_t DEP  = PADRC ? 64u : RC_DEP;
    static constexpr uint32_t DBG  = PADRC ? 65u : RC_DBG;
    static constexpr uint32_t DBG2 = PADRC ? 66u : RC_DBG2;
    static_assert(!PADRC || RC_STRIDE >= 96u,
                  "RcMap<true> places counters at dwords 32/64, which is OUTSIDE "
                  "a 32-dword row stride -- it would silently write the NEXT "
                  "row's counters.  Widen RC_STRIDE (and the cursor tensor in "
                  "bench/impl_topk.py) before re-enabling padding.");
    static_assert(ARR / 32u != 0u || !PADRC, "spin dword must leave line 0");
    static_assert(DEP / 32u != ARR / 32u || !PADRC,
                  "departure counter must not share the spin line");
};
// Phase G: page-table entries staged in LDS by k_scatter.  The page_size=64
// table for the production shape is 969 int32 = 3.9 KB per row, so one block can
// hold the WHOLE row's table.  The lookup then never leaves the CU.  See the
// comment above slot_of: v3 proved the gather is dependent-load-latency bound
// and not footprint bound, so shrinking the table did nothing -- moving it out
// of HBM entirely is the untried lever.  1024 entries = 4 KB.
constexpr uint32_t PT_LDS    = 1024u;
// ... and the *window* form of the same idea.  A block's slice is CONTIGUOUS,
// so every position it can emit lies in one span of the page table: at G=32 and
// row_len 62000 the slice is ~1938 positions = 31 pages = 124 bytes.  Staging
// the whole row's table instead (PT_LDS above) makes every one of the 192
// blocks load 969 entries to serve ~64 lookups -- a 15x amplification, which is
// what the measurement of that variant showed.
constexpr uint32_t PT_WIN    = 256u;

// Ordered 32-bit key: unsigned compare on the result matches float compare.
__device__ __forceinline__ uint32_t order_key32(float x)
{
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Ordered 16-bit key of x rounded to fp16.  __float2half_rn is monotonic, so
// key16(a) <= key16(b) whenever a <= b, which is what makes the three-way
// split in K2 sound.
__device__ __forceinline__ uint32_t order_key16(float x)
{
    __half h            = __float2half_rn(x);
    unsigned short bits = __half_as_ushort(h);
    unsigned short key  = (bits & 0x8000) ? (unsigned short)(~bits)
                                          : (unsigned short)(bits | 0x8000);
    return (uint32_t)key;
}

// Physical KV slot of a row-relative position.
//
//   page_bits = 0  -> page_table is the page_size=1 table: slot = pt[pos].
//   page_bits = 6  -> page_table is the page_size=64 table:
//                     slot = (pt[pos>>6] << 6) | (pos & 63).
//
// Both are the same mapping; the second is enormously cheaper to gather.  The
// selection emits 2048 *random* positions per row, so with the page_size=1
// table each lookup is a random 4-byte read into a 248 KB array -- 2048 distinct
// cache lines per row, measured at 5.85 us for the six rows, which was a third
// of this kernel's entire cost.  The page_size=64 table for the same row is 969
// int32 = 3.9 KB, so the identical 2048 lookups land in a cache-resident working
// set.  (page_table_1 is itself *derived* from page_table_64 by the caller, so
// nothing is lost.)
__device__ __forceinline__ int32_t
slot_of(const int32_t* __restrict__ pt, uint32_t pos, uint32_t page_bits, uint32_t page_mask)
{
    return (pt[pos >> page_bits] << page_bits) | (int32_t)(pos & page_mask);
}

struct Slice
{
    uint32_t start, len;
};

// This block's slice, cut on float4 boundaries so every load stays 16B aligned.
__device__ __forceinline__ Slice slice_of(uint32_t row_len, uint32_t g, uint32_t G)
{
    const uint32_t units = (row_len + 3u) / 4u;
    const uint32_t base  = units / G;
    const uint32_t extra = units % G;
    const uint32_t my_u  = base + (g < extra ? 1u : 0u);
    const uint32_t off_u = g * base + (g < extra ? g : extra);
    Slice s{};
    s.start = off_u * 4u;
    s.len   = s.start >= row_len ? 0u : min(my_u * 4u, row_len - s.start);
    return s;
}

// Walk the slice applying op(value, row_relative_index).
//
// Four float4 loads are issued back to back before any of them is consumed.
// That is the whole point of this kernel: at G=32 a block only owns ~485
// float4, so without manual unrolling each thread would have exactly one load
// in flight and the grid would be latency-bound all over again.
template <typename Op>
__device__ __forceinline__ void scan_slice(const float* __restrict__ in, Slice sl, Op op)
{
    const uint32_t tx      = threadIdx.x;
    const uint32_t vec_len = sl.len & ~3u;
    const float4* in4      = reinterpret_cast<const float4*>(in + sl.start);
    const uint32_t n4      = vec_len >> 2;

    uint32_t i = tx;
    for(; i + 3u * BS < n4; i += 4u * BS)
    {
        const float4 v0 = in4[i];
        const float4 v1 = in4[i + BS];
        const float4 v2 = in4[i + 2u * BS];
        const float4 v3 = in4[i + 3u * BS];
        const float4 vv[4] = {v0, v1, v2, v3};
#pragma unroll
        for(uint32_t u = 0; u < 4u; ++u)
        {
            const float4 v       = vv[u];
            const float vals[4]  = {v.x, v.y, v.z, v.w};
            const uint32_t bpos  = sl.start + ((i + u * BS) << 2);
#pragma unroll
            for(uint32_t j = 0; j < 4u; ++j) { op(vals[j], bpos + j); }
        }
    }
    for(; i < n4; i += BS)
    {
        const float4 v      = in4[i];
        const float vals[4] = {v.x, v.y, v.z, v.w};
        const uint32_t bpos = sl.start + (i << 2);
#pragma unroll
        for(uint32_t j = 0; j < 4u; ++j) { op(vals[j], bpos + j); }
    }
    for(uint32_t t = vec_len + tx; t < sl.len; t += BS) { op(in[sl.start + t], sl.start + t); }
}

constexpr uint32_t WAVE  = 64u;   // gfx950
constexpr uint32_t NWAVE = BS / WAVE;

// LDS histogram increment that collapses a whole wave into one atomic when
// every active lane agrees on the bin.
//
// The refinement's candidate set is, by construction, the members of ONE coarse
// bin, so in the early radix rounds every candidate lands in the same 8-bit
// bucket.  Plain per-lane LDS atomics then serialise 64 deep per wave, and with
// a few hundred candidates over four rounds that was measurable.  When the lanes
// disagree this falls through to a per-lane atomic, so the only cost in the
// spread-out case is the ballot.  (Same trick, and the same reasoning, as
// aiter's hist_add_aggregated.)
__device__ __forceinline__ void hist_add_agg(uint32_t* hist, uint32_t bin)
{
    const uint64_t active   = __ballot(1);
    const int leader        = __ffsll((unsigned long long)active) - 1;
    const uint32_t lead_bin = __shfl(bin, leader, WAVE);
    if(__all(bin == lead_bin))
    {
        if((int)(threadIdx.x % WAVE) == leader)
        {
            atomicAdd(&hist[lead_bin], (uint32_t)__popcll(active));
        }
    }
    else
    {
        atomicAdd(&hist[bin], 1u);
    }
}

// Threshold search over NB counts: find bin b with above(b) < want <=
// above(b)+hist[b], where above(b) is the sum of all bins strictly greater.
//
// The cross-thread prefix sum is done with wave shuffles plus a single
// cross-wave fixup: TWO barriers per call.  The obvious alternatives are both
// expensive here.  A Hillis-Steele LDS scan needs 2*log2(BS) = 16 barriers, and
// on CDNA __syncthreads() is s_barrier *plus* s_waitcnt vmcnt(0)/lgkmcnt(0), so
// each one also drains outstanding global traffic; called four times inside the
// refinement that was worth ~7 us at six blocks.  Having every thread sum
// grp[tx+1..BS) is barrier-free but costs BS serial LDS reads per thread.

template <uint32_t NB>
__device__ __forceinline__ void find_thr(const uint32_t* __restrict__ hist,
                                         uint32_t* __restrict__ wtot,
                                         uint32_t want,
                                         uint32_t* out_thr,
                                         uint32_t* out_above)
{
    // Phase N: NB < BS is legal.  PER is then 1 and the threads past NB read
    // nothing and hold a zero count.  A zero bin cannot satisfy the bracket
    // predicate below (it needs acc < want && acc + 0 >= want), and it adds
    // nothing to any scan, so the threshold is bit-identical to the NB == BS
    // case.  The NB >= BS arm is character-for-character the pre-Phase-N code
    // so that no existing build changes a single instruction.
    constexpr uint32_t PER = (NB >= BS) ? (NB / BS) : 1u;
    const uint32_t tx   = threadIdx.x;
    const uint32_t lane = tx % WAVE;
    const uint32_t wv   = tx / WAVE;

    uint32_t local[PER];
    uint32_t mine = 0;
#pragma unroll
    for(uint32_t j = 0; j < PER; ++j)
    {
        if constexpr(NB >= BS) { local[j] = hist[tx * PER + j]; }
        else
        {
            const uint32_t idx = tx * PER + j;
            local[j] = (idx < NB) ? hist[idx] : 0u;
        }
        mine += local[j];
    }

    uint32_t incl = mine;
#pragma unroll
    for(uint32_t o = 1; o < WAVE; o <<= 1)
    {
        const uint32_t nv = __shfl_up(incl, o, WAVE);
        if(lane >= o) { incl += nv; }
    }
    if(lane == WAVE - 1u) { wtot[wv] = incl; }
    __syncthreads();

    uint32_t base = 0, total = 0;
#pragma unroll
    for(uint32_t w = 0; w < NWAVE; ++w)
    {
        const uint32_t t = wtot[w];
        if(w < wv) { base += t; }
        total += t;
    }
    uint32_t acc = total - (base + incl);  // strictly above this thread's group

#pragma unroll
    for(int j = (int)PER - 1; j >= 0; --j)
    {
        const uint32_t c = local[j];
        if(acc < want && acc + c >= want)
        {
            *out_thr   = tx * PER + (uint32_t)j;
            *out_above = acc;
        }
        acc += c;
    }
    __syncthreads();  // wtot[] is scratch and out_thr/out_above must be visible
}

// Inclusive SUFFIX sum across one wave: lane l receives sum_{k>=l} v_k.
__device__ __forceinline__ uint32_t wave_suffix_sum(uint32_t v)
{
    const uint32_t lane = threadIdx.x & (WAVE - 1u);
#pragma unroll
    for(uint32_t o = 1; o < WAVE; o <<= 1)
    {
        const uint32_t t = __shfl_down(v, o, WAVE);
        v += (lane + o < WAVE) ? t : 0u;
    }
    return v;
}

// Phase K / C1: the SAME threshold as find_thr<HIST_BINS>, derived from the
// two-level histogram instead of the flat one.
//
// find_thr's contract is "the bin b with above(b) < want <= above(b)+hist[b]",
// i.e. the LARGEST b whose suffix sum S(b) = sum_{i>=b} hist[i] is still >=
// want.  Because the coarse bin is exactly the sum of FINE_PER_CRS consecutive
// fine bins, SC(j) == S(j*FINE_PER_CRS) identically -- integer addition, no
// rounding, no reassociation error.  So the largest j with SC(j) >= want brackets
// b into [j*FPC, (j+1)*FPC), and within the group the same predicate against
// want - SC(j+1) picks exactly the same b.  Bit-identical by construction, and
// checked directly by experiments/phaseK_bc/probe_hier.py.
//
// Every WAVE runs this redundantly on its own registers: 2 x 256 B coalesced
// loads and 2 x 6 shuffle steps, NO LDS and NO barrier, against 16 KB + a
// 4096-element block scan + 2 barriers.  Redundancy across the 4 waves of a
// block is cheaper than the LDS round trip plus barrier it would take to share.
// `crs` is the caller's already-issued load of gh[GH_CRS_OFF + lane] (see the
// speculative-prefetch comment in k_scatter).  `total_must_be` is the number of
// positions section B is contractually required to have binned for this row.
//
// PAIRING GUARD.  A hierarchical C composed with a NON-coarse B reads an
// all-zero coarse summary, and that failure is silent, not loud: level 1's
// ballot is empty so j = 0 and abv = 0, level 2 then runs on the 64 LOWEST fine
// bins and returns thr = 0, every element classifies as an outright winner, the
// `q < TOPK` bound stops anything from being corrupted -- and the kernel
// cheerfully returns an ARBITRARY 2048 of the row.  A comment in the bench file
// is not a defence against that.
//
// The check is free because the value is already in a register: after the level-1
// suffix sum lane 0 holds SC(0), the total of every binned position, which for
// any row reaching this code is exactly row_len.  One v_cmp, block-uniform (every
// wave computes the same suffix sum from the same 64 dwords).
//
// On mismatch it returns HIER_BAD and the caller falls back to the ORIGINAL flat
// find_thr<HIST_BINS> over the same global histogram.  Falling back rather than
// poisoning is deliberate: it makes the hierarchical kernel behave EXACTLY like
// the flat one whenever its precondition does not hold, so C1 cannot introduce a
// failure mode the flat path did not already have.  (It also keeps a real
// existing gate green: reference_f64.tie_heavy_check invokes section C standalone
// on a synthetic logits buffer WITHOUT running section B, so the histogram does
// not describe C's input at all -- a poisoning guard turns that into an empty
// selection and an exception, while the fallback reproduces the flat kernel's
// behaviour bit for bit.)  The fallback costs one branch and the 16 KB s_hist
// reservation; at 192 blocks on 256 CUs at most one block is ever resident per
// CU, so that LDS is provably free here -- measured, not assumed.
// Sentinel: no fine bin index can be this, and no key can equal or exceed it.
constexpr uint32_t HIER_BAD = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Phase L: the compact-candidate contract's constants.  A BUCKET is the top 8
// bits of the SAME order_key16 key the fine histogram uses, i.e. exactly
// 2^BKSH consecutive fine bins, so nothing about the threshold changes -- the
// bucket is only the granularity at which section B is able to pre-sort.
// ---------------------------------------------------------------------------
constexpr uint32_t CAND_NBUCKET = 256u;
constexpr uint32_t CAND_GBMAX   = 64u;    // max section-B blocks per row
static_assert(HIST_BITS >= 8u, "bucket key is the top 8 bits of the fp16 key");
constexpr uint32_t BKSH = HIST_BITS - 8u;   // fine_bin >> BKSH == bucket

// WANT_SBK additionally returns S_fine(thr & ~(2^BKSH - 1)), i.e. how many of
// the row's elements the HISTOGRAM says have bucket >= thr>>BKSH.  That number
// is the Phase L interlock: it is what section B's per-block count table must
// sum to if the table describes the same logits the threshold was derived from.
// Templated rather than given an optional out-parameter so that the existing
// callers' code generation is provably unchanged.
// It also returns ABOVE, the exact number of the row's elements strictly greater
// than the threshold bin.  That number is the hinge of Phase L2: today section C
// learns it from rc[RC_WIN] AFTER an arrival handshake has serialised every
// block of the row, but it is an exact suffix sum of the very histogram the
// threshold came from, so every block can compute it independently, at t=0, with
// no communication whatsoever.  find_thr<HIST_BINS> already computes the same
// quantity into *out_above on the flat path; this is the hierarchical form of
// exactly that value, and it costs one extra __shfl on a register the level-2
// scan already holds.
//
// NOCHK (Phase N, PROBE ONLY): keep the interlock's shuffle and compare but make
// the predicate unsatisfiable instead of comparing against total_must_be.  The
// launch-floor probe below runs inside a replayed graph in which section B
// ACCUMULATES into ghist and nothing re-zeroes it, so by round 2 the real
// interlock would fail, take its early return and skip level 2 -- the probe
// would then silently measure half of the threshold it claims to price.  The
// instruction count is identical (one __shfl, one compare, one branch); only
// the constant differs, and 0xFFFFFFFF is not reachable by a suffix sum of
// int32 bin counts.  NOCHK defaults false, so every production call site's
// codegen is unchanged.
template <bool WANT_SBK, bool NOCHK = false>
__device__ __forceinline__ uint32_t
find_thr_hier_impl(const uint32_t* __restrict__ gh, uint32_t want, uint32_t crs,
                   uint32_t total_must_be, uint32_t* out_sbk,
                   uint32_t* out_above = nullptr)
{
    const uint32_t lane = threadIdx.x & (WAVE - 1u);

    // level 1 -- 64 coarse bins, one per lane
    const uint32_t c = crs;
    const uint32_t s = wave_suffix_sum(c);
    if constexpr(NOCHK)
    {
        if(__shfl(s, 0, WAVE) == 0xFFFFFFFFu) { return HIER_BAD; }
    }
    else
    {
        if(__shfl(s, 0, WAVE) != total_must_be) { return HIER_BAD; }
    }
    uint32_t sn      = __shfl_down(s, 1, WAVE);
    if(lane == WAVE - 1u) { sn = 0u; }
    const uint64_t b1 = __ballot(s >= want && sn < want);
    // Cannot be empty (total == row_len > want and lane 63 sees sn == 0), but a
    // zero ballot would index gh at -1, so it degrades to bin 0 -- which is also
    // what find_thr leaves in *out_thr when it finds nothing.
    const uint32_t j   = b1 ? (uint32_t)(__ffsll((unsigned long long)b1) - 1) : 0u;
    const uint32_t abv = b1 ? __shfl(sn, (int)j, WAVE) : 0u;

    // level 2 -- the FINE_PER_CRS fine bins under coarse bin j.  Lanes past the
    // group contribute 0, so they can never satisfy the predicate (want2 >= 1).
    const uint32_t f  = (lane < FINE_PER_CRS) ? gh[j * FINE_PER_CRS + lane] : 0u;
    const uint32_t sf = wave_suffix_sum(f);
    uint32_t sfn      = __shfl_down(sf, 1, WAVE);
    if(lane == WAVE - 1u) { sfn = 0u; }
    const uint32_t want2 = want - abv;
    const uint64_t b2    = __ballot(sf >= want2 && sfn < want2);
    const uint32_t l     = b2 ? (uint32_t)(__ffsll((unsigned long long)b2) - 1) : 0u;
    if constexpr(WANT_SBK)
    {
        // The bucket boundary below thr is (thr & ~(2^BKSH-1)).  FINE_PER_CRS
        // (64) is a multiple of 2^BKSH (16), so that boundary lies in the SAME
        // coarse group and its suffix is abv + sf[l & ~(2^BKSH-1)] -- one
        // shuffle on a value already in a register.
        constexpr uint32_t BM = (1u << BKSH) - 1u;
        *out_sbk = abv + __shfl(sf, (int)(l & ~BM), WAVE);
        // sfn[x] is by construction sf[x+1], i.e. the count in this coarse
        // group STRICTLY above fine bin x, so above(thr) = abv + sfn[l].
        *out_above = abv + __shfl(sfn, (int)l, WAVE);
    }
    return j * FINE_PER_CRS + l;
}

__device__ __forceinline__ uint32_t
find_thr_hier(const uint32_t* __restrict__ gh, uint32_t want, uint32_t crs,
              uint32_t total_must_be)
{
    return find_thr_hier_impl<false>(gh, want, crs, total_must_be, nullptr);
}

// ---------------------------------------------------------------------------
// Exact refinement of the threshold bin + padding.
//
// Defined ahead of k_scatter because Phase H2 calls refine_row from
// k_scatter's last-arriving block as well as from the standalone k_refine.
// ---------------------------------------------------------------------------

// Radix-256 on the full fp32 ordered key, four rounds = all 32 bits.  The
// candidate set is enumerated eight times (histogram + emit, per round), and
// membership in the still-undecided set is re-derived from the prefix resolved
// so far rather than tracked per element.
//
// IN_LDS is a template parameter, not a runtime flag, on purpose: with a
// runtime `in_lds ? s_key[i] : order_key32(cv[i])` the compiler emitted BOTH
// the LDS read and the global load and selected between them, so every round
// paid a global round trip that six blocks cannot hide.  That single select was
// worth ~5 us.
// AGL (Phase K / R10): read the candidate arrays with AGENT-scope relaxed loads
// (`global_load ... sc0 sc1`, i.e. bypass the non-coherent per-CU vector L1)
// instead of relying on a blanket __threadfence() to invalidate the whole L1.
// Only meaningful in the FUSED tail, where the producer is a different block; in
// the standalone k_refine a kernel boundary has already made them visible.
__device__ __forceinline__ float ld_val(const float* p, bool agl)
{
    return agl ? __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
               : *p;
}
__device__ __forceinline__ int32_t ld_idx(const int32_t* p, bool agl)
{
    return agl ? __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
               : *p;
}

template <bool IN_LDS, bool AGL = false>
__device__ __forceinline__ void
refine_core(uint32_t n, uint32_t above, uint32_t remain,
            const uint32_t* __restrict__ s_key, const int32_t* __restrict__ s_slot,
            const float* __restrict__ cv, const int32_t* __restrict__ ci,
            int32_t* __restrict__ o,
            uint32_t* __restrict__ s_hist, uint32_t* __restrict__ s_grp,
            uint32_t* s_thr, uint32_t* s_above, uint32_t* s_emit)
{
    const uint32_t tx = threadIdx.x;

    if(remain == 0 || n <= remain)
    {
        // Defensive: the threshold guarantees n >= remain, so this is the
        // degenerate path only.  Emit everything we have, then pad.
        for(uint32_t i = tx; i < n; i += BS)
        {
            const uint32_t p = above + i;
            if(p < TOPK) { o[p] = IN_LDS ? s_slot[i] : ld_idx(&ci[i], AGL); }
        }
        __syncthreads();
        for(uint32_t i = above + n + tx; i < TOPK; i += BS) { o[i] = -1; }
        return;
    }

    uint32_t prefix = 0;
    for(int r = 0; r < 4; ++r)
    {
        const uint32_t sh = 24u - (uint32_t)r * 8u;

        for(uint32_t i = tx; i < RADIX; i += BS) { s_hist[i] = 0u; }
        if(tx == 0)
        {
            *s_thr   = 0;
            *s_above = 0;
        }
        __syncthreads();

        for(uint32_t i = tx; i < n; i += BS)
        {
            const uint32_t key =
                IN_LDS ? s_key[i] : order_key32(ld_val(&cv[i], AGL));
            const bool in_play = (r == 0) || (((key >> (sh + 8u)) << (sh + 8u)) == prefix);
            if(in_play) { hist_add_agg(s_hist, (key >> sh) & 0xFFu); }
        }
        __syncthreads();

        find_thr<RADIX>(s_hist, s_grp, remain, s_thr, s_above);

        const uint32_t thr = *s_thr;
        const uint32_t abv = *s_above;

        // Nothing above the threshold bin settles no winner, so the emit scan
        // would find nothing.  Skipping it matters on clustered rows, where the
        // early rounds are exactly this case.
        if(abv == 0 && r != 3)
        {
            prefix |= (thr << sh);
            continue;
        }

        for(uint32_t i = tx; i < n; i += BS)
        {
            const uint32_t key =
                IN_LDS ? s_key[i] : order_key32(ld_val(&cv[i], AGL));
            const bool in_play = (r == 0) || (((key >> (sh + 8u)) << (sh + 8u)) == prefix);
            if(!in_play) { continue; }
            const uint32_t bin = (key >> sh) & 0xFFu;
            // Last round: survivors in the threshold bin are numerically equal,
            // so any of them is a correct answer.  The TopK bound on the cursor
            // enforces the quota.
            if(bin > thr || (bin == thr && r == 3))
            {
                const uint32_t p = atomicAdd(s_emit, 1u);
                if(p < TOPK) { o[p] = IN_LDS ? s_slot[i] : ld_idx(&ci[i], AGL); }
            }
        }
        __syncthreads();

        prefix |= (thr << sh);
        remain -= abv;
        if(remain == 0) { break; }
    }

    __syncthreads();
    // The emit loop above already wrote o[above .. *s_emit) directly, so all
    // that is left is the -1 padding.
    //
    // There used to be a third LDS array, `s_win[REF_CAP]`, copied over
    // o[above .. filled) here -- and NOTHING in the file ever wrote it.  It was
    // harmless only by accident: an LDS array that is read and never written is
    // undef, so LLVM deleted both the array and the copy (k_refine's LDS is
    // 17436 B = s_hist + s_grp + 3 scalars + s_key + s_slot EXACTLY, with zero
    // bytes for s_win).  Relying on a DCE of undefined behaviour to stay
    // correct is not a contract, so the array and the copy are gone.  Removing
    // them changes no measured byte of LDS and no emitted instruction; it
    // removes a landmine that a future compiler could arm.
    const uint32_t filled = *s_emit < TOPK ? *s_emit : TOPK;
    for(uint32_t i = filled + tx; i < TOPK; i += BS) { o[i] = -1; }
}

// Shared by the standalone K3 and by the fused tail inside k_scatter.
// FRANK (Phase L2): the exact-rank selection, with the n inner keys broadcast
// from a REGISTER instead of re-read from LDS.
//
// The rank loop is O(n^2) LDS *broadcasts* -- j is uniform across the wave -- and
// at n = 88..120 that is only ~120 reads per thread, which should be well under
// 0.1 us.  Measured, it is 3.09 us (Phase L2 bisection: k_scatter 10.11 with the
// refinement, 7.02 with dbg bit 1 clear), because the whole phase runs in SIX
// blocks on a 256-CU part and every ds_read pays its full latency against an
// otherwise empty machine.  Staging a 64-key chunk into one register per lane
// and broadcasting it with __shfl at a UNIFORM lane index lowers to
// v_readlane_b32 -- SALU, no LDS at all -- so the inner loop stops touching the
// LDS pipe entirely: 2 ds_read per thread instead of n.
//
// RNOCMP (Phase O): DIAGNOSTIC, WRONG ANSWER, TIMING ONLY.  The rank branch is
// run with the O(n^2) comparison loop DELETED and `rank := i`.  Since i ranges
// over [0, n) exactly as a true rank does, EXACTLY `remain` threads still pass
// `rank < remain` and exactly `remain` stores still land in
// o[above .. above+remain) -- the same count, the same address range, a
// different permutation.  So it holds the store tail, the barriers, the
// staging, the cursor round trip and the 6-block geometry FIXED and removes
// only the comparisons.
//
//     (full - RNOCMP)  = the comparison work        -> 64 blocks remove ~63/64
//     (RNOCMP - dbg&2 off) = the fixed tail         -> 64 blocks remove none
//
// It is a TEMPLATE parameter, not a dbg bit, for the reason the RESV probe
// established the hard way: a runtime predicate lets the optimiser keep the
// very instructions the probe exists to delete.  Defaulted false, so every
// shipping instantiation is textually unchanged.
//
// PRANK (Phase O): THE PARALLEL RANK.  Exact, deterministic, bit-identical to
// the serial rank -- see the argument below, which is the whole point of this
// block comment.
//
// WHY THE OBVIOUS PARTITION IS WRONG.  "Give each of the row's 64 blocks 1/64
// of the candidates" does NOT work and the arithmetic says so before any code
// is written: there are only n = 55..120 candidates
// (experiments/phaseO_C/refine_shape.json), so slicing i over 64 blocks leaves
// each block TWO active threads, each still walking all n j's.  The serial
// DEPTH is unchanged and the measurement would have been a null.  The measured
// split (experiments/phaseO_C/split_cmp_tail.json) says the 3.64 us selection
// is 3.22 comparisons + 0.42 fixed tail, and the comparisons are being run by
// 12 waves (6 blocks x 2 populated waves) on a 1024-SIMD machine.  So the
// parallelism has to come from the j axis, not the i axis.
//
// THE PARTITION THAT WORKS.  One thread per j, one block per i-set:
//   * thread tx evaluates the order predicate for j = tx  (needs n <= BS)
//   * the block reduces those BS booleans to rank_i with one __ballot per wave
//     plus an NWAVE-entry LDS sum -- s_grp, which already exists
//   * block pb owns i == pb (mod PB), so at n = 120, PB = 64 each block owns
//     one or two i, and the DEPTH per block is 2 x (one compare + one ballot +
//     two barriers) instead of 120 dependent LDS-fed compares.
//
// EXACTNESS, which is what PRECISION_BLOCKER.md gates on:
//   1. (key_j, j) is a STRICT TOTAL ORDER on [0, n): keys are order_key32 of
//      the candidate values and ties are broken by the array index, which is
//      unique.  Hence rank : [0,n) -> [0,n) is a BIJECTION.
//   2. rank_i is summed over ALL j in [0, n) in EVERY block -- the j loop is
//      not partitioned by pb, only the i loop is.  So a candidate on an
//      i-partition boundary is ranked against the FULL set, exactly as in the
//      serial path.  This is the one thing a partitioned rank can get wrong and
//      it is structurally impossible here.
//   3. i == pb (mod PB) with pb in [0, PB) partitions [0, n) exactly, so every
//      candidate is ranked by exactly one block: none twice, none dropped, for
//      every n and every PB, including PB > n (blocks with no i simply run zero
//      iterations, uniformly).
//   4. By (1) exactly `remain` candidates have rank < remain, so
//      o[above .. above+remain) is covered exactly once, and o[0..above) is
//      untouched.  Independent of how above (1990..2045) and n (55..120) vary
//      per row, because both are block-uniform reads of the same two dwords.
//   5. DETERMINISM IS PRESERVED, not merely correctness: the value written to
//      o[above + r] is the slot of the candidate whose rank is r, and r does
//      not depend on block scheduling.  The output is bit-identical to the
//      serial rank ELEMENT BY ELEMENT, not just as a set -- which is what the
//      documented "deterministic rather than arrival-ordered" property of
//      o[above..TOPK) requires.
//
// The barrier pair inside the i loop is safe because the loop's trip count
// (pb, n, PB) is block-uniform, so every thread of the block executes exactly
// the same number of barriers.
template <bool AGL = false, bool FRANK = false, bool RNOCMP = false,
          bool PRANK = false>
__device__ __forceinline__ void
refine_row(uint32_t row, uint32_t above, uint32_t n_raw, uint32_t cap,
           const int32_t* __restrict__ cand_idx, const float* __restrict__ cand_val,
           int32_t* __restrict__ o,
           uint32_t* __restrict__ s_hist, uint32_t* __restrict__ s_grp,
           uint32_t* __restrict__ s_key, int32_t* __restrict__ s_slot,
           uint32_t* s_thr, uint32_t* s_above, uint32_t* s_emit,
           uint32_t dbg = 7u, bool have_pre = false,
           int32_t pre_slot = 0, float pre_val = 0.f,
           uint32_t pb = 0u, uint32_t PB = 1u,
           uint32_t* out_ranked = nullptr)
{
    const uint32_t tx = threadIdx.x;
    const int32_t* __restrict__ ci = cand_idx + (size_t)row * cap;
    const float* __restrict__ cv   = cand_val + (size_t)row * cap;

    const uint32_t n      = n_raw > cap ? cap : n_raw;
    const uint32_t remain = above < TOPK ? TOPK - above : 0u;

    if(tx == 0) { *s_emit = above; }

    const bool in_lds = (n <= REF_CAP);
    if(in_lds && (dbg & 1u))
    {
        // R6a: the caller may already have issued cand_idx[tx] / cand_val[tx]
        // WITHOUT waiting for `n`, which is what breaks the dependent chain
        // row_ends -> cursor -> candidates into a single round trip.  Those two
        // loads are unconditionally in bounds (cap >= TOPK >= BS), and entries
        // at or past n are simply never stored, so the staged contents are
        // identical either way.
        const uint32_t i0 = have_pre ? tx + BS : tx;
        if(have_pre && tx < n)
        {
            s_key[tx]  = order_key32(pre_val);
            s_slot[tx] = pre_slot;
        }
        for(uint32_t i = i0; i < n; i += BS)
        {
            s_key[i]  = order_key32(ld_val(&cv[i], AGL));
            s_slot[i] = ld_idx(&ci[i], AGL);   // k_scatter already translated it
        }
    }
    __syncthreads();

    if(!(dbg & 2u)) { /* timing probe: skip the refinement */ }
    else if(in_lds && n <= RANK_CAP && remain != 0 && n > remain)
    {
        // Exact selection by rank -- which is what the four radix rounds were
        // computing the hard way.
        //
        // The candidate set is the members of ONE coarse bin: 57 to 120 of the
        // row's 62000 elements at the production shape.  So O(n^2) comparisons
        // is under 120 LDS reads per thread, and every one of them is a wave
        // broadcast because j is uniform across the wave.  The radix path costs
        // four *dependent* rounds instead, each with a histogram, a block scan
        // and barriers, and at one block per row there is nothing to overlap
        // them with.
        //
        // (key, candidate index) is a strict total order, so exactly `remain`
        // candidates have rank < remain, and slots [above, TopK) are each
        // written exactly once: no atomics, no emit cursor, no padding pass,
        // and the result is deterministic rather than arrival-ordered.
        if constexpr(PRANK)
        {
            if(n <= PRANK_CAP)
            {
                const uint32_t lane = tx & (WAVE - 1u);
                const uint32_t wv   = tx / WAVE;
                uint32_t nranked    = 0u;
                // this thread's j, read ONCE and held in a register
                const uint32_t kj   = (tx < n) ? s_key[tx] : 0u;
                for(uint32_t i = pb; i < n; i += PB)
                {
                    const uint32_t ki = s_key[i];          // block-uniform read
                    const bool p = (tx < n) &&
                                   ((kj > ki) || (kj == ki && tx < i));
                    const uint64_t m = __ballot(p);
                    if(lane == 0)
                    {
                        s_grp[wv] = (uint32_t)__popcll((unsigned long long)m);
                    }
                    __syncthreads();
                    uint32_t rank = 0;
#pragma unroll
                    for(uint32_t q = 0; q < NWAVE; ++q) { rank += s_grp[q]; }
                    if(tx == 0 && rank < remain) { o[above + rank] = s_slot[i]; }
                    __syncthreads();   // s_grp is reused by the next i
                    ++nranked;
                }
                if(out_ranked) { *out_ranked = nranked; }
                return;
            }
            // n > PRANK_CAP: one thread per j is not available.  Fall through
            // to the serial rank, in ONE block only -- never silently skipped.
            if(pb != 0u) { return; }
        }
        if(FRANK && n <= BS)
        {
            const uint32_t lane = tx & (WAVE - 1u);
            const uint32_t ki   = (tx < n) ? s_key[tx] : 0u;
            uint32_t rank       = 0;
            for(uint32_t c = 0; c < n; c += WAVE)
            {
                const uint32_t kc  = (c + lane < n) ? s_key[c + lane] : 0u;
                const uint32_t lim = (n - c) < WAVE ? (n - c) : WAVE;
                for(uint32_t jj = 0; jj < lim; ++jj)
                {
                    const uint32_t kj = __shfl(kc, (int)jj, WAVE);
                    const uint32_t j  = c + jj;
                    rank += (kj > ki) || (kj == ki && j < tx);
                }
            }
            if(tx < n && rank < remain) { o[above + rank] = s_slot[tx]; }
        }
        else if constexpr(RNOCMP)
        {
            // Phase O probe: same store tail, no comparisons.  See the header.
            for(uint32_t i = tx; i < n; i += BS)
            {
                if(i < remain) { o[above + i] = s_slot[i]; }
            }
        }
        else
        for(uint32_t i = tx; i < n; i += BS)
        {
            const uint32_t ki = s_key[i];
            uint32_t rank     = 0;
            uint32_t j        = 0;
            // Unrolled by 8: the naive loop issues one LDS read and then waits
            // on it, so it runs at LDS latency rather than LDS throughput, and
            // at n=320 that alone cost ~13 us.  Eight independent reads in
            // flight turns it back into a throughput loop.
            for(; j + 8u <= n; j += 8u)
            {
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
            for(; j < n; ++j)
            {
                const uint32_t kj = s_key[j];
                rank += (kj > ki) || (kj == ki && j < i);
            }
            if(rank < remain) { o[above + rank] = s_slot[i]; }
        }
    }
    else if(PRANK && pb != 0u)
    {
        // The radix fallback (and the short/degenerate cases it absorbs) emits
        // through a shared cursor and pads o[..TOPK) -- one block's work by
        // construction.  b1_ctx2049 takes it (n = 1, remain = 1, so n > remain
        // is false), so this is a live path and not a theoretical one.
    }
    else if(in_lds)
    {
        refine_core<true, AGL>(n, above, remain, s_key, s_slot, cv, ci, o,
                          s_hist, s_grp, s_thr, s_above, s_emit);
    }
    else
    {
        refine_core<false, AGL>(n, above, remain, s_key, s_slot, cv, ci, o,
                           s_hist, s_grp, s_thr, s_above, s_emit);
    }

}


// ---------------------------------------------------------------------------
// K2: threshold + scatter (page gather folded into the emit)
// ---------------------------------------------------------------------------

// PTMODE 0: page table read from global (the original path).
//        1: the whole row's table staged in LDS at block entry.
//        2: only the block's own slice window staged in LDS.
// HIER (Phase K / C1): derive the threshold from the two-level histogram, so
// the 16 KB flat-histogram load, the 4096-element block scan and -- crucially
// for the fused form -- the 16 KB s_hist LDS block all disappear.  Requires a
// section-B variant that maintains the coarse summary (hip_hist*c).
// NACQ (Phase K / R10): narrow the fused handshake's ACQUIRE side.  See the
// comment at the __threadfence() below.
// CANDIN (Phase L): take the compact candidate stream section B now emits
// instead of re-scanning the flat logits array.  See the contract comment in
// experiments/phaseC/logits_kernel.hip.  The flat `scan_slice` path stays
// compiled in and is taken whenever the interlock does not hold.
// NOHS (Phase L2): delete the arrival handshake from the DEPENDENCY CHAIN.
//
// The fused tail exists because refine_row needs three things it believes only
// the other blocks can give it: `above` (how many strict winners were emitted),
// the boundary elements, and the guarantee that they have all landed.  Under
// CANDIN none of that is true any more:
//   * `above` is an exact suffix sum of ghist, which find_thr_hier now returns.
//     Every block computes the identical value at t=0 with zero communication.
//   * the boundary elements (fine_bin == thr) all live in bucket thr>>BKSH, and
//     section B's count table says exactly where that bucket sits inside each
//     segment's compact prefix: [sfx[b][thr_b+1], sfx[b][thr_b]).  They come
//     from B's output, not from C's other blocks.
//   * the two output regions are disjoint BY CONSTRUCTION -- strict winners are
//     exactly o[0..above) and boundary winners exactly o[above..TOPK) -- so the
//     refiner never has to know when the emitters finished.
// So block 0 resolves the boundary CONCURRENTLY with all 95 other blocks
// emitting strict winners, instead of after them.  The arrival counter survives
// but now guards only the ghist/cursor reset, which is fire-and-forget stores
// with no dependent read: no release s_waitcnt, no acquire, no dependent
// rc[RC_WIN]/rc[RC_CAND] round trip, no candidate arrays at all.
template <int PTMODE, bool FUSEREF, bool HIER = false, bool NACQ = false,
          bool CANDIN = false, bool NOHS = false, bool FRANK = false,
          int RESV = 0, bool PRANK = false, int PBLK = 0, bool PADRC = false,
          bool DBGPRE = false>
__global__ __launch_bounds__(BS) void k_scatter(const float* __restrict__ logits,
                                                const int32_t* __restrict__ row_ends,
                                                const int32_t* __restrict__ page_table,
                                                int32_t* __restrict__ out,
                                                uint32_t* __restrict__ ghist,
                                                int32_t* __restrict__ cursor,
                                                int32_t* __restrict__ cand_cnt,
                                                int32_t* __restrict__ cand_idx,
                                                float* __restrict__ cand_val,
                                                int64_t lg_stride,
                                                int64_t pt_stride,
                                                uint32_t page_bits,
                                                uint32_t page_mask,
                                                uint32_t cap,
                                                uint32_t G,
                                                uint32_t emit_on,
                                                uint32_t own_short,
                                                uint32_t dbg,
                                                uint32_t selfclean,
                                                const uint2* __restrict__ cand2,
                                                const unsigned short* __restrict__ sfxt,
                                                const uint32_t* __restrict__ cmeta)
{
    // With HIER the flat histogram never enters LDS at all: all that is left is
    // the RADIX-sized scratch the fused refinement needs.  That is 16 KB off
    // every block's footprint, which is what lets the FUSEREF form keep its
    // occupancy (see the s_key/s_slot comment below).
    static_assert(HIST_BINS >= RADIX, "refinement scratch must fit in s_hist");
    // Sized for the flat fallback even under HIER (see find_thr_hier): the
    // hierarchical path never touches it, and LDS cannot limit occupancy at
    // 192 blocks on 256 CUs.
    __shared__ uint32_t s_hist[HIST_BINS];
    __shared__ uint32_t s_grp[NWAVE];  // cross-wave scan fixup (see find_thr)
    __shared__ uint32_t s_thr, s_above;
    // Per-block staging.  The first version of this kernel did one *global*
    // atomicAdd per emitted element on a single per-row cursor: 2048 winners x
    // 6 rows = 12288 returning atomics to 6 addresses, which serialise at ~69
    // cycles each and cost 59 of the kernel's 63 us.  Staging in LDS turns that
    // into two global atomics per block (384 total) and is the single largest
    // effect in this file.
    __shared__ uint32_t s_wcnt, s_ccnt;
    __shared__ int32_t s_wbase, s_cbase;
    __shared__ int32_t s_wbuf[STAGE];
    __shared__ int32_t s_cidx[STAGE];
    __shared__ float s_cval[STAGE];
    // Phase G: the whole row's page table, when it fits.  LDSPT is a *template*
    // parameter and not a runtime flag for the same reason IN_LDS is in
    // refine_core: with a runtime select the compiler emits BOTH the LDS read
    // and the global load and picks between them, so every lookup still pays a
    // memory round trip.  The host decides from pt_stride, which is uniform.
    __shared__ int32_t s_pt[PTMODE == 1 ? PT_LDS : (PTMODE == 2 ? PT_WIN : 1)];
    // Phase H2: the refinement's working set, only allocated when the
    // refinement is actually fused in (it is 24 KB, and reserving it
    // unconditionally would change this kernel's LDS footprint for every
    // variant, including the control).
    __shared__ uint32_t s_key[FUSEREF ? REF_CAP : 1];
    __shared__ int32_t s_slot[FUSEREF ? REF_CAP : 1];
    __shared__ uint32_t s_emit;
    __shared__ uint32_t s_last;
    // Phase L: per-section-B-block candidate counts, so the compact read loop
    // does not re-issue a uniform global load per element.
    __shared__ uint32_t s_need[CANDIN ? CAND_GBMAX : 1];
    // Phase L2: the upper edge of the threshold bucket inside each segment's
    // compact prefix, so the refiner can address the boundary run directly.
    __shared__ uint32_t s_needhi[(CANDIN && NOHS) ? CAND_GBMAX : 1];
    __shared__ uint32_t s_bn;

    const uint32_t row = blockIdx.y;
    const uint32_t g   = blockIdx.x;
    const uint32_t tx  = threadIdx.x;

    const int32_t rl_s = row_ends[row];
    // R6a: the 64-bin coarse summary is issued HERE, before row_ends has even
    // come back, because its address depends only on blockIdx and it is always
    // in bounds.  Otherwise the chain is row_ends -> coarse bins -> fine group:
    // three serial HBM latencies before the first logit can be classified.  On
    // the short-row path the value is simply discarded.
    uint32_t pre_crs = 0u;
    if constexpr(HIER)
    {
        pre_crs = ((const uint32_t*)ghist)[(size_t)row * GH_STRIDE + GH_CRS_OFF
                                           + (threadIdx.x & (WAVE - 1u))];
    }
    const uint32_t row_len = rl_s > 0 ? (uint32_t)rl_s : 0u;
    if(row_len <= TOPK)
    {
        // Phase H1.  When the histogram is built inside section B, k_hist is
        // gone -- and k_hist was the ONLY writer of the whole output row for
        // rows that need no selection (this is the ctx=0 and ctx=2048 case; at
        // ctx=0 section B launches zero pages for the row and writes nothing at
        // all).  So the duty moves here, to the kernel whose (G, rows) grid
        // always launches.  Without this the reused output buffer would keep
        // stale contents, which PHASED_POISON=1 is there to prove.
        if(own_short)
        {
            const int32_t* __restrict__ pt0 = page_table + (int64_t)row * pt_stride;
            int32_t* __restrict__ o0        = out + (size_t)row * TOPK;
            for(uint32_t i = g * BS + tx; i < TOPK; i += G * BS)
            {
                o0[i] = i < row_len ? slot_of(pt0, i, page_bits, page_mask) : -1;
            }
        }
        return;
    }

    int32_t* __restrict__ rc = cursor + (size_t)row * RC_STRIDE;

    // Issued FIRST: these are independent of everything else in the kernel, and
    // the barrier below already waits for the histogram staging, so the page
    // table rides along for free.  969 int32 over 256 threads = four coalesced
    // dwords each.
    const Slice sl = slice_of(row_len, g, G);
    // First page-table entry this block can possibly touch.  Every position it
    // emits is inside its own slice, so the window is [pt_base, pt_base+npt).
    const uint32_t pt_base = (PTMODE == 2) ? (sl.start >> page_bits) : 0u;
    if(PTMODE != 0)
    {
        const int32_t* __restrict__ pt0 = page_table + (int64_t)row * pt_stride;
        uint32_t npt;
        if(PTMODE == 1)
        {
            npt = (uint32_t)pt_stride < PT_LDS ? (uint32_t)pt_stride : PT_LDS;
        }
        else
        {
            npt = sl.len ? (((sl.start + sl.len - 1u) >> page_bits) - pt_base + 1u) : 0u;
            npt = npt < PT_WIN ? npt : PT_WIN;   // host guarantees it fits; belt-and-braces
        }
        for(uint32_t i = tx; i < npt; i += BS) { s_pt[i] = pt0[pt_base + i]; }
    }

    // Phase K/C2 attribution probes.  emit_on==5 skips BOTH the 16 KB histogram
    // load and find_thr (thr forced to 0); emit_on==6 keeps the load but skips
    // find_thr.  Both give a WRONG answer and are timing-only, and both leave
    // the cursors in the same valid zero state emit_on==0 does (they fall into
    // the no-emit branch below).  The branch is on a kernel-uniform scalar, so
    // the production path costs one s_cmp.
    const uint32_t* __restrict__ gh = (const uint32_t*)ghist + (size_t)row * GH_STRIDE;
    // Every block derives the same threshold from the same global histogram, so
    // no communication is needed beyond the kernel boundary itself.
    uint32_t thr_h  = 0u;
    bool need_flat  = !HIER;
    uint32_t sbk    = 0u;
    uint32_t hier_above = 0u;
    if constexpr(HIER)
    {
        // Issued BEFORE the barrier so its two (dependent) 256 B loads overlap
        // the page-table window staging above.  Needs neither LDS nor a barrier.
        if(emit_on < 5u)
        {
            thr_h     = find_thr_hier_impl<CANDIN>(gh, TOPK, pre_crs, row_len,
                                                   &sbk, &hier_above);
            need_flat = (thr_h == HIER_BAD);   // block-uniform
        }
    }

    // -----------------------------------------------------------------------
    // Phase L: decide whether the compact candidate stream may be used.
    //
    // THIS IS THE INTERLOCK, AND IT IS NOT A COMMENT.  The campaign already
    // learned from find_thr_hier's mispairing hazard that a consumer must be
    // able to detect a bad pairing from data it already holds, so three
    // independent facts are checked against values that are already in
    // registers, and ANY failure falls back to the flat scan_slice path which
    // stays compiled in:
    //
    //  1. cmeta[row] is self-describing.  Section B writes (CAP << 16) | G_B
    //     itself, so C is never *told* the layout of the array it is about to
    //     index -- it reads it.  A producer that did not write the contract
    //     leaves cmeta == 0 (the buffer is zero-initialised) and C falls back.
    //  2. sum_b sfx[b][0] == row_len.  Every live position of the row must be
    //     accounted for exactly once by the count table -- the same property
    //     find_thr_hier's SC(0) == row_len guard relies on.
    //  3. sum_b sfx[b][thr>>BKSH] == S_fine(thr & ~mask), i.e. the count table
    //     and the HISTOGRAM THE THRESHOLD CAME FROM agree, at the threshold,
    //     on how many elements are in play.  This is the check that catches a
    //     stale candidate array paired with a fresh histogram -- exactly what
    //     reference_f64.tie_heavy_check constructs when it feeds section C a
    //     synthetic logits buffer with its own k_hist while the candidate
    //     arrays still hold the previous real-input run.
    //
    // Plus the capacity precondition max_b sfx[b][thr>>BKSH] <= CAP, without
    // which the bucket-descending prefix would not be guaranteed to contain
    // every element at or above the threshold.  Never a silent arbitrary 2048.
    // -----------------------------------------------------------------------
    bool use_cand   = false;
    uint32_t cap_b  = 0u;
    uint32_t gb_all = 0u;          // section-B blocks per row, read from cmeta
    uint32_t run_sh = 0u;          // log2 of the padded per-segment boundary run
    uint32_t lo_b = 0u, hi_b = 0u;
    if(tx == 0) { s_bn = 0u; }
    if constexpr(CANDIN)
    {
        static_assert(PTMODE != 2,
                      "the compact stream is not slice-contiguous, so the "
                      "page-table WINDOW form cannot serve it");
        if(HIER && !need_flat && emit_on == 1u)
        {
            const uint32_t ln    = tx & (WAVE - 1u);
            const uint32_t meta  = cmeta[row];
            const uint32_t gb    = meta & 0xFFFFu;
            gb_all               = gb;
            const uint32_t thr_b = thr_h >> BKSH;
            cap_b                = meta >> 16;
            uint32_t sum_t = 0u, sum_n = 0u, max_n = 0u, sum_in = 0u, max_r = 0u;
            if(gb != 0u && gb <= CAND_GBMAX && cap_b != 0u)
            {
                for(uint32_t base = 0; base < gb; base += WAVE)
                {
                    const uint32_t b = base + ln;
                    uint32_t t0 = 0u, tk = 0u;
                    if(b < gb)
                    {
                        // BUCKET-MAJOR: the gb entries of one bucket are
                        // consecutive, so each of these is one 128 B coalesced
                        // load per wave.  The block-major layout this replaced
                        // put them 512 B apart -- 64 separate lines per load,
                        // 12.6 MB of L2 line fetches over the grid, on the
                        // dependent path right behind the threshold.
                        const unsigned short* __restrict__ r =
                            sfxt + (size_t)row * CAND_NBUCKET * CAND_GBMAX;
                        t0 = (uint32_t)r[b];
                        tk = (uint32_t)r[(size_t)thr_b * CAND_GBMAX + b];
                        s_need[b] = tk;   // every wave writes the same value
                        if constexpr(NOHS)
                        {
                            // Same 128 B coalesced shape, adjacent bucket row,
                            // so it rides in the same round trip as s_need.
                            const uint32_t th = (thr_b + 1u < CAND_NBUCKET)
                                ? (uint32_t)r[(size_t)(thr_b + 1u) * CAND_GBMAX + b]
                                : 0u;
                            s_needhi[b] = th;
                            sum_in += tk - th;
                            max_r = (tk - th) > max_r ? (tk - th) : max_r;
                        }
                    }
                    sum_t += t0;
                    sum_n += tk;
                    max_n = tk > max_n ? tk : max_n;
                }
#pragma unroll
                for(uint32_t sh = 1; sh < WAVE; sh <<= 1)
                {
                    sum_t += __shfl_xor(sum_t, sh, WAVE);
                    sum_n += __shfl_xor(sum_n, sh, WAVE);
                    sum_in += __shfl_xor(sum_in, sh, WAVE);
                    const uint32_t m = __shfl_xor(max_n, sh, WAVE);
                    max_n = m > max_n ? m : max_n;
                    const uint32_t mr = __shfl_xor(max_r, sh, WAVE);
                    max_r = mr > max_r ? mr : max_r;
                }
                run_sh = (max_r <= 1u) ? 0u : (32u - (uint32_t)__clz(max_r - 1u));
                use_cand = (sum_t == row_len) && (sum_n == sbk) && (max_n <= cap_b);
                // Phase L2 adds a fourth precondition, from the same table: the
                // whole row's threshold-BUCKET run has to fit the refiner's LDS,
                // because block 0 now resolves the boundary alone and has no
                // second chance to spill.  Measured 1018-1787 against
                // REF_CAP = 2048; over that, fall back to the flat path exactly
                // as for any other interlock failure.
                if constexpr(NOHS) { use_cand = use_cand && (sum_in <= REF_CAP); }
                // Observability, not control flow.  cand_cnt is otherwise
                // unused by this kernel, so recording the decision here costs
                // one store per block and lets a bench run PROVE the fast path
                // engaged instead of inferring it -- a fallback returns the
                // same answer, so no correctness gate can tell the two apart.
                if(tx == 0) { cand_cnt[row] = use_cand ? 1 : 0; }
                // Every wave ran the identical reduction over the identical
                // dwords, so this is block-uniform without a barrier.
                if(NOHS && G > 1u)
                {
                    // DEDICATED REFINER.  Block 0 emits no strict winners at
                    // all; blocks 1..G-1 partition the segments between them.
                    //
                    // This is what makes removing the handshake pay.  With
                    // block 0 doing both duties the kernel's span is
                    // emit + collect + rank in ONE block, which is strictly
                    // worse than the handshake it replaced (measured: 15.58 us
                    // against 10.25).  With block 0 doing only collect + rank,
                    // its span runs entirely underneath the other blocks' emit
                    // and the rank stops being on the critical path at all.
                    const uint32_t ge = g - 1u, Ge = G - 1u;
                    lo_b = (g == 0u) ? 0u : (ge * gb) / Ge;
                    hi_b = (g == 0u) ? 0u : ((ge + 1u) * gb) / Ge;
                }
                else
                {
                    lo_b = (g * gb) / G;
                    hi_b = ((g + 1u) * gb) / G;
                }
            }
        }
    }
    if(need_flat && emit_on != 5u)
    {
        for(uint32_t i = tx; i < HIST_BINS; i += BS) { s_hist[i] = gh[i]; }
    }
    if(tx == 0)
    {
        s_thr   = 0;
        s_above = 0;
        s_wcnt  = 0;
        s_ccnt  = 0;
    }
    __syncthreads();

    if(need_flat && emit_on < 5u)
    {
        find_thr<HIST_BINS>(s_hist, s_grp, TOPK, &s_thr, &s_above);
        __syncthreads();
    }

    const uint32_t thr = need_flat ? s_thr : thr_h;

    // -----------------------------------------------------------------------
    // Phase K / R6b: clear this block's stripe of the FINE histogram HERE,
    // rather than leaving the whole 4160-dword-per-row reset to k_refine.
    //
    // This is only sound because of HIER, and it is sound outright rather than
    // "probably": with the hierarchical threshold the ONLY dwords any block
    // ever READS are the 64 coarse bins and the 64 fine bins of the threshold
    // group.  Every other fine bin is written by section B and read by nobody,
    // so it is dead the instant the kernel starts and clearing it needs no
    // ordering with respect to any other block.  The two live regions are
    // excluded: the coarse region is outside the loop, and the block whose
    // stripe contains the threshold group skips exactly those 64 bins (`thr` is
    // block-uniform and identical in every block, so every block agrees on
    // which they are).  k_refine then clears only those 128 dwords.
    //
    // Excluded on the fallback path, where find_thr<HIST_BINS> really does read
    // all 4096 bins.  need_flat depends only on (coarse total vs row_len), which
    // is the same value in every block of the row, so all blocks agree.
    if(HIER && selfclean && !need_flat && emit_on == 1u)
    {
        uint32_t* __restrict__ ghw = ghist + (size_t)row * GH_STRIDE;
        const uint32_t gj0 = (thr / FINE_PER_CRS) * FINE_PER_CRS;
        const uint32_t per = (HIST_BINS + G - 1u) / G;
        const uint32_t lo  = g * per;
        const uint32_t hi  = (lo + per) < HIST_BINS ? (lo + per) : HIST_BINS;
        for(uint32_t i = lo + tx; i < hi; i += BS)
        {
            if(i < gj0 || i >= gj0 + FINE_PER_CRS) { ghw[i] = 0u; }
        }
    }

    const float* __restrict__ in   = logits + (int64_t)row * lg_stride;
    const int32_t* __restrict__ pt = page_table + (int64_t)row * pt_stride;
    int32_t* __restrict__ o        = out + (size_t)row * TOPK;
    int32_t* __restrict__ ci       = cand_idx + (size_t)row * cap;
    float* __restrict__ cv         = cand_val + (size_t)row * cap;

    // The Phase G lookup.  `if constexpr` so exactly one of the two forms is
    // emitted (see the s_pt declaration).
    auto SLOT = [&](uint32_t p) -> int32_t {
        if constexpr(PTMODE == 1)
        {
            return (s_pt[p >> page_bits] << page_bits) | (int32_t)(p & page_mask);
        }
        else if constexpr(PTMODE == 2)
        {
            return (s_pt[(p >> page_bits) - pt_base] << page_bits) | (int32_t)(p & page_mask);
        }
        else
        {
            return slot_of(pt, p, page_bits, page_mask);
        }
    };

    if(emit_on == 0u || emit_on >= 5u)
    {
        // Timing probe: scan + threshold only, no emit.  Leaves the cursors at
        // zero, which is a valid state (the refinement then takes its
        // short-row/degenerate path), so repeated invocation stays consistent.
        uint32_t acc = 0;
        scan_slice(in, sl, [&](float v, uint32_t gi) {
            const uint32_t kb = order_key16(v) >> LOW_BITS;
            acc += (kb > thr) ? gi : 1u;   // opaque to the optimiser
        });
        if(acc == 0xFFFFFFFFu) { o[0] = (int32_t)acc; }
        return;
    }
    // -----------------------------------------------------------------------
    // Phase L fast path: consume section B's compact, bucket-descending
    // candidate stream instead of re-reading the flat logits array.
    //
    // Expected traffic per row: sum_b sfx[b][thr>>BKSH] ~= 2900 pairs = 23 KB,
    // against 62000 float4-loaded floats = 248 KB.
    //
    // THE CLASSIFICATION BODY IS THE SAME LAMBDA the flat path uses, and it
    // feeds the SAME LDS staging and the SAME flush.  The first version of this
    // replaced the staging with a count-then-emit two-pass and a ballot-ranked
    // emit; measured, that was 1.3 us WORSE than the scan it deleted, because
    // it added two dependent global rounds and two barriers to a kernel whose
    // whole remaining cost is dependent latency.  `gi` here is already the
    // physical KV slot (section B emitted it), so SLOT() is the identity and no
    // page table is touched at all on this path.
    // -----------------------------------------------------------------------
    auto classify = [&](float v, uint32_t gi) {
        const uint32_t kb = order_key16(v) >> LOW_BITS;
        if(kb > thr)
        {
            const uint32_t p = atomicAdd(&s_wcnt, 1u);
            if(p < STAGE) { s_wbuf[p] = (int32_t)gi; }
            else
            {
                const unsigned long long old =
                    atomicAdd((unsigned long long*)rc, 1ull);
                const uint32_t q = (uint32_t)old;
                if(q < TOPK) { o[q] = (int32_t)gi; }
            }
        }
        else if(NOHS && CANDIN)
        {
            // Phase L2: boundary elements are NOT published here.  Block 0
            // reads them straight out of section B's stream, so there is no
            // candidate array, no agent-scope write-through store, and nothing
            // for another block to wait on.
        }
        else if(kb == thr)
        {
            const uint32_t p = atomicAdd(&s_ccnt, 1u);
            if(p < STAGE)
            {
                s_cidx[p] = (int32_t)gi;
                s_cval[p] = v;
            }
            else
            {
                const unsigned long long old =
                    atomicAdd((unsigned long long*)rc, 1ull << 32);
                const uint32_t q = (uint32_t)(old >> 32);
                if(q < cap)
                {
                    if constexpr(FUSEREF)
                    {
                        __hip_atomic_store(&ci[q], (int32_t)gi, __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT);
                        __hip_atomic_store(&cv[q], v, __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT);
                    }
                    else
                    {
                        ci[q] = (int32_t)gi;
                        cv[q] = v;
                    }
                }
            }
        }
    };

    if(CANDIN && use_cand)
    {
        const uint32_t ln = tx & (WAVE - 1u);
        const uint32_t wv = tx / WAVE;
        const uint32_t nb = hi_b - lo_b;
        // Two traversal shapes for the same work: one section-B block per WAVE
        // when a C block owns at least NWAVE of them, otherwise the whole C
        // block on one section-B block at a time.  Either way a wave's lanes
        // read 64 consecutive 8 B pairs, i.e. one 512 B coalesced burst.
        if(nb >= NWAVE)
        {
            for(uint32_t b = lo_b + wv; b < hi_b; b += NWAVE)
            {
                const uint32_t n_ = s_need[b];
                const uint2* __restrict__ src =
                    cand2 + ((size_t)row * CAND_GBMAX + b) * cap_b;
                for(uint32_t i_ = ln; i_ < n_; i_ += WAVE)
                {
                    const uint2 pr = src[i_];
                    classify(__uint_as_float(pr.x), pr.y);
                }
            }
        }
        else
        {
            for(uint32_t b = lo_b; b < hi_b; ++b)
            {
                const uint32_t n_ = s_need[b];
                const uint2* __restrict__ src =
                    cand2 + ((size_t)row * CAND_GBMAX + b) * cap_b;
                for(uint32_t i_ = tx; i_ < n_; i_ += BS)
                {
                    const uint2 pr = src[i_];
                    classify(__uint_as_float(pr.x), pr.y);
                }
            }
        }
    }
    else
    {
    scan_slice(in, sl, [&](float v, uint32_t gi) {
        const uint32_t kb = order_key16(v) >> LOW_BITS;
        if(kb > thr)
        {
            // Winner outright: monotonicity of order_key16 guarantees it beats
            // every element of the threshold bin.
            //
            // Only the POSITION is staged here.  Doing `s_wbuf[p] = pt[gi]`
            // inline puts a dependent scattered global load inside the scan, and
            // since roughly every unrolled iteration has some lane hitting it,
            // the whole wave stalls on a full memory round trip each time: worth
            // ~3 us.  The gather is done in the flush below, where all of the
            // block's lookups are independent and issue together.
            const uint32_t p = atomicAdd(&s_wcnt, 1u);
            if(p < STAGE) { s_wbuf[p] = (int32_t)gi; }
            else
            {
                const unsigned long long old =
                    atomicAdd((unsigned long long*)rc, 1ull);
                const uint32_t q = (uint32_t)old;
                // The threshold guarantees q < TopK; the bound is belt-and-braces
                // so that no reachable state can scribble past the output row.
                if(q < TOPK) { o[q] = SLOT(gi); }
            }
        }
        else if(kb == thr)
        {
            const uint32_t p = atomicAdd(&s_ccnt, 1u);
            if(p < STAGE)
            {
                s_cidx[p] = (int32_t)gi;
                s_cval[p] = v;
            }
            else
            {
                const unsigned long long old =
                    atomicAdd((unsigned long long*)rc, 1ull << 32);
                const uint32_t q = (uint32_t)(old >> 32);
                if(q < cap)
                {
                    if constexpr(FUSEREF)
                    {
                        __hip_atomic_store(&ci[q], SLOT(gi), __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT);
                        __hip_atomic_store(&cv[q], v, __ATOMIC_RELAXED,
                                           __HIP_MEMORY_SCOPE_AGENT);
                    }
                    else
                    {
                        ci[q] = SLOT(gi);
                        cv[q] = v;
                    }
                }
            }
        }
    });
    }   // end of the flat scan_slice path (Phase L: else-branch of use_cand)

    // -----------------------------------------------------------------------
    // Phase L2: the boundary resolution, in block 0, CONCURRENTLY with every
    // other block's strict-winner emit.  Issued BEFORE this block's own flush
    // so its two loads are in flight across the flush's atomic round trip.
    //
    // The run [s_needhi[b], s_need[b]) of segment b's compact prefix is exactly
    // its elements whose BUCKET equals thr>>BKSH -- 1018-1787 per row measured,
    // of which 57-120 have fine_bin == thr exactly.  Nothing here depends on
    // any other block having done anything.
    // -----------------------------------------------------------------------
    const bool nohs = NOHS && CANDIN && use_cand;
    if(nohs && g == 0u && (dbg & 1u))
    {
        // FLAT, RECTANGULAR ITERATION SPACE.  The obvious `for(b) for(i)` nest
        // over the 64 segments measured 36.78 us: 64 SERIAL round trips of ~28
        // elements each, the same serialisation trap as the first Phase L emit.
        // Padding each segment's run up to a power of two turns the whole thing
        // into one flat index with a shift/mask mapping, so 8 independent loads
        // are in flight per thread and the entire 1018-1787-element boundary
        // run is ~1 memory round trip.  The padding costs no global traffic:
        // out-of-run items are skipped on an LDS compare, before any load.
        const uint32_t nit  = gb_all << run_sh;
        const uint32_t rmsk = (1u << run_sh) - 1u;
        for(uint32_t it0 = tx; it0 < nit; it0 += 8u * BS)
        {
            uint2 pr[8];
            bool ok[8];
#pragma unroll
            for(uint32_t u = 0; u < 8u; ++u)
            {
                const uint32_t it = it0 + u * BS;
                ok[u] = false;
                if(it < nit)
                {
                    const uint32_t b = it >> run_sh;
                    const uint32_t k = it & rmsk;
                    const uint32_t e0 = s_needhi[b];
                    if(e0 + k < s_need[b])
                    {
                        pr[u] = cand2[((size_t)row * CAND_GBMAX + b) * cap_b
                                      + e0 + k];
                        ok[u] = true;
                    }
                }
            }
#pragma unroll
            for(uint32_t u = 0; u < 8u; ++u)
            {
                if(!ok[u]) { continue; }
                const float v = __uint_as_float(pr[u].x);
                if((order_key16(v) >> LOW_BITS) != thr) { continue; }
                const uint32_t p = atomicAdd(&s_bn, 1u);
                if(p < REF_CAP)
                {
                    s_key[p]  = order_key32(v);
                    s_slot[p] = (int32_t)pr[u].y;
                }
            }
        }
    }

    // ---- FROM HERE ON THE TWO PATHS SHARE EVERYTHING ----------------------
    // Both fill the same s_wbuf/s_cidx/s_cval staging with the same
    // classification, so the flush, the single reserving atomic, the emit and
    // the fused refinement tail below are literally the same instructions for
    // both.  (The first version of this closed the else-brace AFTER the flush
    // instead of here; the fast path then staged into LDS and emitted nothing,
    // which made k_scatter look 3.4 us faster and produced an all -1 output.
    // The all-shape gate caught it; no timing measurement could have.)
    __syncthreads();
    if(emit_on == 2u) { return; }   // timing probe: classify+stage, no flush

    const uint32_t wn = s_wcnt < STAGE ? s_wcnt : STAGE;
    const uint32_t cn = s_ccnt < STAGE ? s_ccnt : STAGE;

    // Issue the page-table gathers FIRST.  They do not depend on the base this
    // block is about to reserve, so they overlap the atomic's round trip
    // instead of queueing behind it.  wn <= STAGE and cn <= STAGE, so with
    // BS=256 threads this is at most STAGE/BS registers per thread.
    constexpr uint32_t PERT = (STAGE + BS - 1u) / BS;
    int32_t wslot[PERT], cslot[PERT];
    float cvalr[PERT];
    // Phase L: on the compact path the staged value is ALREADY the physical
    // slot -- section B computed it from the page-table entry it had loaded for
    // its own KV read -- so the gather is the identity and the page table is
    // never touched.  Block-uniform, so this is one s_cmp.
    const bool pre_slotted = CANDIN && use_cand;
#pragma unroll
    for(uint32_t u = 0; u < PERT; ++u)
    {
        const uint32_t i = tx + u * BS;
        // emit_on==4 is a timing probe: same flush, no page-table gather.
        if(i < wn)
        {
            wslot[u] = (emit_on == 4u || pre_slotted) ? s_wbuf[i]
                                                      : SLOT((uint32_t)s_wbuf[i]);
        }
        if(i < cn)
        {
            cslot[u] = (emit_on == 4u || pre_slotted) ? s_cidx[i]
                                                      : SLOT((uint32_t)s_cidx[i]);
            cvalr[u] = s_cval[i];
        }
    }

    // -----------------------------------------------------------------------
    // Phase O probe, RESV == 1.  WRONG ANSWER BY CONSTRUCTION, TIMING ONLY.
    //
    // The emit ladder (experiments/phaseO_C/ladder_emit_on.json) prices the
    // whole flush at 4.40 us of WALL, of which the page-table LDS gather is
    // 0.24 (e1 - e4).  The remaining 4.16 us is {reserving atomic, emit store,
    // drain} and BUDGET.md's k=3 model attributes ~2.2 of it to the atomic's
    // round trip sitting on the store's critical path:
    //     last classify -> atomicAdd(rc) -> s_wbase -> store address -> drain.
    //
    // RESV == 1 isolates EXACTLY that dependency and nothing else.  The atomic
    // is STILL ISSUED, same address, same 8-byte packing, same 384-way
    // contention -- only its RETURN VALUE is dropped, so it lowers to a
    // non-returning global_atomic_add_x2, the `s_waitcnt vmcnt(0)` in front of
    // the emit disappears, and the base comes from blockIdx instead.
    //
    // WHY A TEMPLATE PARAMETER AND NOT AN `emit_on` VALUE.  It was first
    // written as a runtime `if(emit_on == 3u)` branch.  The compiler CSE'd the
    // two atomicAdd call sites into ONE returning atomic and selected the base
    // with a v_cndmask -- verified in the disassembly:
    //     global_atomic_add_x2 v[6:7], v13, v[6:7], s[28:29] sc0
    //     s_waitcnt vmcnt(0)          <-- STILL THERE
    //     s_cmp_eq_u32 s38, 3 ; v_cndmask_b32 v6, v6, v7, vcc
    // i.e. the probe kept the very dependency it was built to remove and would
    // have reported a null for a reason that had nothing to do with the
    // hardware.  `if constexpr` makes the returning atomic non-existent in this
    // instantiation, which is the only form the optimiser cannot undo.
    //
    // Read it as the CEILING of every reservation-overlap rewrite (per-wave,
    // chunked, speculative): none can beat "the round trip is not on the
    // critical path at all".  Flat here => the whole family is dead unbuilt.
    //
    // The bases collide across blocks, so the output is garbage.  The cursors
    // still advance (the atomic is issued), so k_refine takes exactly the same
    // path it takes for the full kernel and still restores both zero
    // invariants.
    if(tx == 0)
    {
        const unsigned long long pack =
            ((unsigned long long)cn << 32) | (unsigned long long)wn;
        if constexpr(RESV == 1)
        {
            atomicAdd((unsigned long long*)rc, pack);   // return value dropped
            s_wbase = (int32_t)((g * 32u) & (TOPK - 1u));
            s_cbase = (int32_t)(g * 64u);
        }
        else
        {
            const unsigned long long old = atomicAdd((unsigned long long*)rc, pack);
            s_wbase = (int32_t)(uint32_t)old;
            s_cbase = (int32_t)(uint32_t)(old >> 32);
        }
    }
    __syncthreads();

#pragma unroll
    for(uint32_t u = 0; u < PERT; ++u)
    {
        const uint32_t i = tx + u * BS;
        if(i < wn)
        {
            const uint32_t p = (uint32_t)s_wbase + i;
            if(p < TOPK) { o[p] = wslot[u]; }
        }
        if(i < cn)
        {
            const uint32_t q = (uint32_t)s_cbase + i;
            if(q < cap)
            {
                if constexpr(FUSEREF)
                {
                    // Phase H2 handshake, cheap half.  These are the ONLY bytes
                    // another block reads inside this kernel, and there are ~65
                    // of them per block, so making just these stores agent-scope
                    // (sc0 sc1 = write through, bypassing the non-coherent
                    // per-CU vector L1) is far cheaper than a __threadfence()
                    // in every one of the 192 blocks -- on an 8-XCD part that
                    // lowers to a full L2 writeback, and it measured as a cost
                    // that scaled linearly with the block count.
                    __hip_atomic_store(&ci[q], cslot[u], __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
                    __hip_atomic_store(&cv[q], cvalr[u], __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
                }
                else
                {
                    ci[q] = cslot[u];
                    cv[q] = cvalr[u];
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Phase H2: the refinement, in the LAST-ARRIVING block of the row.
    //
    // Refinement is one block's work, so a __threadfence() plus an arrival
    // counter is enough -- no grid sync, and emphatically no
    // hipLaunchCooperativeKernel (measured 21.1 us in-graph and it silently
    // loses the cooperative attribute under capture).
    //
    // THE HISTOGRAM RESET.  k_refine used to spread it over all (G, rows)
    // blocks because at six blocks a 24 KB store loop cost 2 us of pure
    // latency, and the review's objection to moving it into k_scatter was that
    // every block reads the WHOLE row histogram at the top to derive its
    // threshold, so no stripe is provably dead mid-kernel.  That objection is
    // about clearing it *early*.  By the time the LAST block arrives, every
    // block has already finished that read -- the read happens before find_thr,
    // which happens before the scan, which happens before the flush, which
    // happens before the arrival increment.  So the row's histogram is provably
    // dead here, and this block can clear it without any ping-pong buffer.
    // -----------------------------------------------------------------------
    if(nohs)
    {
        // -------------------------------------------------------------------
        // Phase L2 tail.  Two things happen here and NEITHER is a dependency
        // chain any more.
        //
        // (a) Block 0 finishes the boundary it started before the flush: rank
        //     the fine_bin == thr elements and write them into o[above, TOPK).
        //     `above` came from ghist at t=0, so this block never waited for a
        //     single other block, and the region it writes is disjoint from
        //     o[0..above) by construction.
        // (b) The arrival counter survives, but it now guards ONLY the ghist
        //     and cursor reset.  That reset is pure fire-and-forget stores with
        //     no dependent read, so it needs no release s_waitcnt (nothing is
        //     published), no __threadfence acquire (nothing is consumed), and
        //     no rc[RC_WIN]/rc[RC_CAND] round trip.  All it needs is that every
        //     block has finished READING the two live histogram regions, and a
        //     block cannot reach its arrival increment before its own threshold
        //     loads have returned -- the value is a data dependency of thr.
        // -------------------------------------------------------------------
        if(g == 0u)
        {
            __syncthreads();
            const uint32_t n      = s_bn < REF_CAP ? s_bn : REF_CAP;
            const uint32_t above  = hier_above;
            // Observability: RC_STRIDE is 32 dwords and only 0/1/4 are live, so
            // dwords 8-11 are free scratch for "what did the refiner actually
            // see".  Guessing at these numbers is how the first two Phase L
            // designs were built on a wrong model.
            if(tx == 0)
            {
                rc[8]  = (int32_t)n;
                rc[9]  = (int32_t)above;
                rc[10] = (int32_t)run_sh;
                rc[11] = (int32_t)(gb_all << run_sh);
            }
            const uint32_t remain = above < TOPK ? TOPK - above : 0u;
            if(!(dbg & 2u)) { /* timing probe: skip the rank */ }
            else if(remain != 0u && n > remain)
            {
                // Same exact-selection-by-rank the fused refinement used, on
                // the same (key32, index) strict total order.
                for(uint32_t i = tx; i < n; i += BS)
                {
                    const uint32_t ki = s_key[i];
                    uint32_t rank = 0, j = 0;
                    for(; j + 8u <= n; j += 8u)
                    {
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
                    for(; j < n; ++j)
                    {
                        const uint32_t kj = s_key[j];
                        rank += (kj > ki) || (kj == ki && j < i);
                    }
                    if(rank < remain) { o[above + rank] = s_slot[i]; }
                }
            }
            else
            {
                // Degenerate (cannot happen when the threshold is honoured):
                // emit what there is and leave a contiguous -1 tail.
                for(uint32_t i = tx; i < n; i += BS)
                {
                    const uint32_t p = above + i;
                    if(p < TOPK) { o[p] = s_slot[i]; }
                }
                __syncthreads();
                for(uint32_t i = above + n + tx; i < TOPK; i += BS) { o[i] = -1; }
            }
        }
        __syncthreads();
        if(tx == 0)
        {
            const uint32_t old = __hip_atomic_fetch_add(
                (uint32_t*)&rc[RcMap<PADRC>::ARR], 1u, __ATOMIC_RELAXED,
                __HIP_MEMORY_SCOPE_AGENT);
            // PRANK needs this block's ARRIVAL RANK, the non-PRANK form needs
            // only the is-last flag, and no form needs both.  They share
            // s_last rather than adding a slot: a separate __shared__ dword
            // costs 4 B of LDS in EVERY instantiation including the shipped
            // one, whose footprint must stay at exactly 39988 B.
            s_last = PRANK ? old : ((old + 1u == G) ? 1u : 0u);
        }
        __syncthreads();
        // PBLK: how many of the row's blocks take part in the parallel rank.
        // PBLK == 0 means all G.  The participants are the LAST PBLK TO ARRIVE,
        // which is the whole point: a block learns its arrival rank from the
        // counter it just incremented, so the G - PBLK early blocks return
        // exactly where they return today and never enter the spin.  Measured:
        // with all 64 spinning, prank4 was 0.540 us SLOWER than the control
        // even though the rank itself got 1.9 us faster -- 384 blocks polling
        // one coherent dword contend with the emit stores of the very blocks
        // they are waiting for.  Fewer, later spinners is the direct fix and
        // the only variable changed.
        const uint32_t PB_N   = (PBLK == 0 || (uint32_t)PBLK > G)
                                    ? G : (uint32_t)PBLK;
        const uint32_t pfirst = G - PB_N;
        const uint32_t pidx   = s_last - pfirst;  // valid only if s_last>=pfirst
        // -------------------------------------------------------------------
        // Phase O / PRANK, the FUSED harness.  The arrival counter stops being
        // an ELECTION (one block continues, G-1 return) and becomes the arrival
        // half of a two-phase ROW barrier that all G blocks pass, so the
        // parallel rank can use them.  This buys the node the k_refine harness
        // spends, and costs a spin.
        //
        // THE SPIN NEEDS THE ROW'S BLOCKS TO BE CO-RESIDENT, which is not a
        // property of this file -- it is a property of (G, rows, LDS, CU count)
        // and is checked ON THE HOST before this instantiation is ever
        // launched.  If the check fails the non-PRANK kernel is launched
        // instead.  A spin whose liveness rests on an unchecked occupancy
        // assumption is a hang waiting for someone to raise G.
        //
        // The acquire side is unchanged and still sound: producers store the
        // handful of shared values write-through (sc0 sc1) and s_waitcnt them
        // before their arrival increment, the counter is an agent-scope atomic,
        // and every consumer reads the data with agent-scope loads that never
        // consult the non-coherent L1.  That argument never depended on there
        // being exactly one consumer.
        if constexpr(PRANK)
        {
            if(s_last < pfirst) { return; }
            // ATTRIBUTION ARM (Phase O).  Adding an agent-scope atomic RMW here
            // -- immediately before the spin -- cost 0.64 us at PB=16 and
            // 1.66 us at PB=32, and I attributed that to FALSE SHARING because
            // the counter sat in the line being spun on.  Line-isolating the
            // spin dword then measured -0.10 us, which does not support that
            // attribution.  The competing explanation is that the cost is the
            // extra RMW on the critical path, wherever it lands.  DBGPRE puts
            // the bump back in this position so it can be run WITH and WITHOUT
            // line isolation and the two explanations separated.
            if(DBGPRE && tx == 0)
            {
                __hip_atomic_fetch_add((uint32_t*)&rc[RcMap<PADRC>::DBG2], 1u,
                                       __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
            }
            if(tx == 0)
            {
                uint32_t v = __hip_atomic_load((uint32_t*)&rc[RcMap<PADRC>::ARR],
                                               __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
                while(v < G)
                {
                    __builtin_amdgcn_s_sleep(2);
                    v = __hip_atomic_load((uint32_t*)&rc[RcMap<PADRC>::ARR],
                                          __ATOMIC_RELAXED,
                                          __HIP_MEMORY_SCOPE_AGENT);
                }
            }
            __syncthreads();
        }
        else if(!s_last) { return; }
        static_assert(GH_STRIDE % 4u == 0u, "vectorised reset needs 4 | stride");
        uint4* __restrict__ ghw4 = (uint4*)(ghist + (size_t)row * GH_STRIDE);
        const uint4 z4           = make_uint4(0u, 0u, 0u, 0u);
        for(uint32_t i = tx; i < GH_STRIDE / 4u; i += BS) { ghw4[i] = z4; }
        if(tx == 0)
        {
            rc[RC_WIN]  = 0;
            rc[RC_CAND] = 0;
            rc[RcMap<PADRC>::ARR]  = 0;
        }
        return;
    }

    if constexpr(FUSEREF)
    {
        // Release, the cheap way: the candidate stores above are already
        // write-through to a coherent point, so all that is needed is to wait
        // for them to retire.  No __threadfence() here -- see the comment on
        // the stores.
        __builtin_amdgcn_s_waitcnt(/*vmcnt(0)*/ 0x0f70);
        __syncthreads();
        if(tx == 0)
        {
            const uint32_t old = __hip_atomic_fetch_add(
                (uint32_t*)&rc[RcMap<PADRC>::ARR], 1u, __ATOMIC_RELAXED,
                __HIP_MEMORY_SCOPE_AGENT);
            // PRANK needs this block's ARRIVAL RANK, the non-PRANK form needs
            // only the is-last flag, and no form needs both.  They share
            // s_last rather than adding a slot: a separate __shared__ dword
            // costs 4 B of LDS in EVERY instantiation including the shipped
            // one, whose footprint must stay at exactly 39988 B.
            s_last = PRANK ? old : ((old + 1u == G) ? 1u : 0u);
        }
        __syncthreads();
        // PBLK: how many of the row's blocks take part in the parallel rank.
        // PBLK == 0 means all G.  The participants are the LAST PBLK TO ARRIVE,
        // which is the whole point: a block learns its arrival rank from the
        // counter it just incremented, so the G - PBLK early blocks return
        // exactly where they return today and never enter the spin.  Measured:
        // with all 64 spinning, prank4 was 0.540 us SLOWER than the control
        // even though the rank itself got 1.9 us faster -- 384 blocks polling
        // one coherent dword contend with the emit stores of the very blocks
        // they are waiting for.  Fewer, later spinners is the direct fix and
        // the only variable changed.
        const uint32_t PB_N   = (PBLK == 0 || (uint32_t)PBLK > G)
                                    ? G : (uint32_t)PBLK;
        const uint32_t pfirst = G - PB_N;
        const uint32_t pidx   = s_last - pfirst;  // valid only if s_last>=pfirst
        // -------------------------------------------------------------------
        // Phase O / PRANK, the FUSED harness.  The arrival counter stops being
        // an ELECTION (one block continues, G-1 return) and becomes the arrival
        // half of a two-phase ROW barrier that all G blocks pass, so the
        // parallel rank can use them.  This buys the node the k_refine harness
        // spends, and costs a spin.
        //
        // THE SPIN NEEDS THE ROW'S BLOCKS TO BE CO-RESIDENT, which is not a
        // property of this file -- it is a property of (G, rows, LDS, CU count)
        // and is checked ON THE HOST before this instantiation is ever
        // launched.  If the check fails the non-PRANK kernel is launched
        // instead.  A spin whose liveness rests on an unchecked occupancy
        // assumption is a hang waiting for someone to raise G.
        //
        // The acquire side is unchanged and still sound: producers store the
        // handful of shared values write-through (sc0 sc1) and s_waitcnt them
        // before their arrival increment, the counter is an agent-scope atomic,
        // and every consumer reads the data with agent-scope loads that never
        // consult the non-coherent L1.  That argument never depended on there
        // being exactly one consumer.
        if constexpr(PRANK)
        {
            if(s_last < pfirst) { return; }
            // ATTRIBUTION ARM (Phase O).  Adding an agent-scope atomic RMW here
            // -- immediately before the spin -- cost 0.64 us at PB=16 and
            // 1.66 us at PB=32, and I attributed that to FALSE SHARING because
            // the counter sat in the line being spun on.  Line-isolating the
            // spin dword then measured -0.10 us, which does not support that
            // attribution.  The competing explanation is that the cost is the
            // extra RMW on the critical path, wherever it lands.  DBGPRE puts
            // the bump back in this position so it can be run WITH and WITHOUT
            // line isolation and the two explanations separated.
            if(DBGPRE && tx == 0)
            {
                __hip_atomic_fetch_add((uint32_t*)&rc[RcMap<PADRC>::DBG2], 1u,
                                       __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
            }
            if(tx == 0)
            {
                uint32_t v = __hip_atomic_load((uint32_t*)&rc[RcMap<PADRC>::ARR],
                                               __ATOMIC_RELAXED,
                                               __HIP_MEMORY_SCOPE_AGENT);
                while(v < G)
                {
                    __builtin_amdgcn_s_sleep(2);
                    v = __hip_atomic_load((uint32_t*)&rc[RcMap<PADRC>::ARR],
                                          __ATOMIC_RELAXED,
                                          __HIP_MEMORY_SCOPE_AGENT);
                }
            }
            __syncthreads();
        }
        else if(!s_last) { return; }

        // ACQUIRE side of the handshake.
        //
        // The release side was already narrowed: the producers do NOT
        // __threadfence(), they use write-through `sc0 sc1` stores on only the
        // ~65 values another block reads, plus a wait for them to retire.  That
        // was worth 11 us.  This side never got the same treatment -- it is a
        // blanket __threadfence(), a DEVICE-scope acquire that invalidates the
        // whole per-CU vector L1, so every subsequent load in this block misses,
        // including the 16 KB histogram reset's write-allocate and the page
        // table window.  All it actually needs to see is ~120 candidate entries
        // and two cursor dwords.
        //
        // NACQ replaces it with per-access AGENT-scope relaxed loads on exactly
        // those values (`global_load ... sc0 sc1` = bypass L1, read at the
        // coherent point).  Ordering is still sound: the producers' stores are
        // write-through and retired before their arrival increment, the arrival
        // counter is an AGENT-scope atomic, and this block only reaches the code
        // below through a control dependency on that atomic plus a
        // __syncthreads().  A load that never consults L1 cannot be stale.
        if(!NACQ) { __threadfence(); }

        // The reset is now ONE block's work (the row's histogram is dead, see
        // above), which is the case the original k_refine comment measured at
        // 2 us of pure latency for a strided dword loop.  Four dwordx4 stores
        // per thread instead of sixteen dwords keeps it off the critical path.
        static_assert(GH_STRIDE % 4u == 0u, "vectorised reset needs 4 | stride");
        // Under PRANK the histogram reset is spread over the row's blocks
        // instead of being one block's 4160-dword store loop; every block has
        // long since finished reading the two live regions (the barrier above
        // is strictly later than every block's threshold derivation).
        if(PRANK && !(selfclean && HIER && !need_flat))
        {
            uint4* __restrict__ ghw4 = (uint4*)(ghist + (size_t)row * GH_STRIDE);
            const uint4 z4           = make_uint4(0u, 0u, 0u, 0u);
            for(uint32_t i = pidx * BS + tx; i < GH_STRIDE / 4u;
                i += PB_N * BS)
            {
                ghw4[i] = z4;
            }
        }
        else if(selfclean && HIER && !need_flat)
        {
            // Phase K/R6b, fused form.  Every block already cleared its stripe
            // of the FINE histogram on the way in (see the R6b block above),
            // excluding the threshold group -- which is the only fine region any
            // block reads.  So the last block has 64 fine bins plus the 64
            // coarse bins left, not 4160 dwords, and the reset stops being a
            // 100 KB store loop issued by six blocks after the handshake.
            uint32_t* __restrict__ ghw = ghist + (size_t)row * GH_STRIDE;
            const uint32_t gj0 = (thr / FINE_PER_CRS) * FINE_PER_CRS;
            for(uint32_t i = tx; i < FINE_PER_CRS; i += BS) { ghw[gj0 + i] = 0u; }
            for(uint32_t i = tx; i < CBINS; i += BS) { ghw[GH_CRS_OFF + i] = 0u; }
        }
        else
        {
            uint4* __restrict__ ghw4 = (uint4*)(ghist + (size_t)row * GH_STRIDE);
            const uint4 z4           = make_uint4(0u, 0u, 0u, 0u);
            for(uint32_t i = tx; i < GH_STRIDE / 4u; i += BS) { ghw4[i] = z4; }
        }

        const uint32_t above = NACQ
            ? (uint32_t)__hip_atomic_load(&rc[RC_WIN], __ATOMIC_RELAXED,
                                          __HIP_MEMORY_SCOPE_AGENT)
            : (uint32_t)rc[RC_WIN];
        const uint32_t nraw = NACQ
            ? (uint32_t)__hip_atomic_load(&rc[RC_CAND], __ATOMIC_RELAXED,
                                          __HIP_MEMORY_SCOPE_AGENT)
            : (uint32_t)rc[RC_CAND];
        // R6a, fused form: the candidate prefetch is issued alongside the two
        // cursor reads instead of behind them.  Unconditionally in bounds
        // (cap >= TOPK >= BS); entries at or past n are discarded.
        const int32_t pre_slot = ld_idx(&cand_idx[(size_t)row * cap + tx], NACQ);
        const float pre_val    = ld_val(&cand_val[(size_t)row * cap + tx], NACQ);
        __syncthreads();   // every thread has read them before they are cleared
        if(!PRANK && tx == 0)
        {
            rc[RC_WIN]  = 0;
            rc[RC_CAND] = 0;
            rc[RcMap<PADRC>::ARR]  = 0;
        }
        uint32_t nranked = 0u;
        refine_row<NACQ, FRANK, false, PRANK>(row, above, nraw, cap, cand_idx,
                         cand_val, o,
                         s_hist, s_grp, s_key, s_slot, &s_thr, &s_above, &s_emit,
                         dbg, true, pre_slot, pre_val, pidx, PB_N, &nranked);
        if(PRANK && tx == 0 && nranked != 0u)
        {
            __hip_atomic_fetch_add((uint32_t*)&rc[RcMap<PADRC>::DBG], 1u, __ATOMIC_RELAXED,
                                   __HIP_MEMORY_SCOPE_AGENT);
        }
        // Departure half of the row barrier.  The counters cannot be reset by
        // the block that happens to finish first -- the other G-1 are still
        // spinning on RC_ARR and still reading RC_WIN/RC_CAND -- so the last
        // block OUT does it.  RC_ARR is only cleared here, after every block
        // has provably left the spin.
        if constexpr(PRANK)
        {
            __syncthreads();
            if(tx == 0)
            {
                // Observability, issued HERE and not before the spin.
                // RC_STRIDE is 32 dwords = ONE 128 B line, so every counter in
                // the row shares a line with RC_ARR -- and RC_ARR is the dword
                // the barrier spins on.  Bumping a debug counter just before
                // the spin therefore injected a write into the exact line 16-32
                // blocks were polling, and it cost real time: prank4_p32 went
                // from +1.26 to -1.39 us and p16 from +1.24 to +0.40 when this
                // counter was added.  In the departure window the line is being
                // written anyway, so it adds no new contention.
                __hip_atomic_fetch_add((uint32_t*)&rc[RcMap<PADRC>::DBG2], 1u,
                                       __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
                const uint32_t d = __hip_atomic_fetch_add(
                    (uint32_t*)&rc[RcMap<PADRC>::DEP], 1u, __ATOMIC_RELAXED,
                    __HIP_MEMORY_SCOPE_AGENT);
                if(d + 1u == PB_N)
                {
                    rc[RC_WIN]  = 0;
                    rc[RC_CAND] = 0;
                    rc[RcMap<PADRC>::ARR]  = 0;
                    rc[RcMap<PADRC>::DEP]  = 0;
                }
            }
        }
    }
}

}  // namespace phased

// ---------------------------------------------------------------------------
// Phase O / PRANK liveness guard.
//
// The FUSED parallel rank spins on the row's arrival counter, so every block of
// a row must be resident at the same time or the spin never completes.  That is
// a property of (grid, block size, LDS, VGPR, CU count) and NOT of this file --
// it is true today at G=64, rows=6, LDS 39988 (4 blocks/CU x 256 CU = 1024 slots
// against 384 blocks) and false the moment anyone raises G or the LDS budget.
//
// So it is asked of the runtime, about the exact kernel that will be launched,
// rather than asserted in a comment.  If it does not hold, the ordinary
// single-block-refine kernel is launched instead: slower, never wrong, never
// hung.  Note the k_refine harness (PRANK without FUSEREF) needs none of this --
// its barrier is the kernel boundary.
static bool prank_resident_ok(const void* fn, int blocks_per_row, int rows)
{
    int dev = 0;
    if(hipGetDevice(&dev) != hipSuccess) { return false; }
    hipDeviceProp_t prop;
    if(hipGetDeviceProperties(&prop, dev) != hipSuccess) { return false; }
    int per_cu = 0;
    if(hipOccupancyMaxActiveBlocksPerMultiprocessor(
           &per_cu, fn, (int)phased::BS, 0) != hipSuccess) { return false; }
    if(per_cu <= 0) { return false; }
    const long long slots = (long long)per_cu * (long long)prop.multiProcessorCount;
    return (long long)blocks_per_row * (long long)rows <= slots;
}

// ---------------------------------------------------------------------------
// host entry
// ---------------------------------------------------------------------------

// The accepted Phase D configuration, baked in rather than selected:
//   PTMODE 2   per-block window of the page table staged in LDS
//   FUSEREF    the refinement runs in k_scatter's last-arriving block
//   HIER       hierarchical threshold, valid because section B also wrote the
//              coarse summary -- which is why no separate k_hist pass is needed
//   NACQ       narrow acquire in the fused tail
//   PRANK/16   16 of a row's blocks stay resident for the exact rank
void topk_transform(torch::Tensor logits,
                    torch::Tensor row_ends,
                    torch::Tensor page_table,
                    torch::Tensor out,
                    torch::Tensor ghist,
                    torch::Tensor cursor,
                    torch::Tensor cand_cnt,
                    torch::Tensor cand_idx,
                    torch::Tensor cand_val,
                    int64_t g_per_row,
                    int64_t page_size)
{
    const int64_t R   = logits.size(0);
    const int64_t L   = logits.stride(0);
    const int64_t PTS = page_table.stride(0);
    const int64_t cap = cand_idx.size(1);
    const uint32_t G  = (uint32_t)g_per_row;
    // page_size must be 1 or a power of two; both give the identical mapping.
    uint32_t PB = 0, PM = 0;
    for(int64_t ps = page_size; ps > 1; ps >>= 1) { ++PB; }
    PM = (page_size > 1) ? (uint32_t)(page_size - 1) : 0u;
    TORCH_CHECK((page_size & (page_size - 1)) == 0 && page_size >= 1,
                "page_size must be a power of two, got ", page_size);

    // The histogram width is a compile-time property of the kernel.  Getting it
    // wrong on the host silently walks off the end of ghist and corrupts the
    // buffers next to it, which is exactly what happened once already.
    TORCH_CHECK(ghist.numel() == R * (int64_t)phased::GH_STRIDE,
                "ghist must be [rows, ", phased::GH_STRIDE, "], got ", ghist.numel());
    TORCH_CHECK(cap >= (int64_t)phased::TOPK,
                "candidate capacity must be >= TopK");
    TORCH_CHECK(logits.scalar_type() == at::kFloat && out.scalar_type() == at::kInt,
                "dtype");
    // PTMODE 2 is baked into the instantiation, so the window it stages has to
    // fit -- checked here rather than dispatched around.  page_size 64, G 64 and
    // a 62016-token context need 18 of the 256 entries.
    TORCH_CHECK((PTS + (int64_t)G - 1) / (int64_t)G + 2 <= (int64_t)phased::PT_WIN,
                "page-table window needs <= ", phased::PT_WIN, " entries per block");

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid((unsigned)G, (unsigned)R, 1);
    dim3 blk(phased::BS, 1, 1);
    // The four trailing scalars are emit_on=1 (emit the selection), own_short=1
    // (section B built the histogram, so k_scatter owns the short-row path),
    // dbg=7 and selfclean=0.  The candidate-stream pointers are unused here.
#define SCATTER_ARGS                                                  \
    grid, blk, 0, stream,                                             \
    logits.data_ptr<float>(),                                         \
    row_ends.data_ptr<int32_t>(),                                     \
    page_table.data_ptr<int32_t>(),                                   \
    out.data_ptr<int32_t>(),                                          \
    (uint32_t*)ghist.data_ptr<int32_t>(),                             \
    cursor.data_ptr<int32_t>(),                                       \
    cand_cnt.data_ptr<int32_t>(),                                     \
    cand_idx.data_ptr<int32_t>(),                                     \
    cand_val.data_ptr<float>(),                                       \
    L, PTS, PB, PM, (uint32_t)cap, G, 1u, 1u, 7u, 0u,                 \
    nullptr, nullptr, nullptr

    // The persistent-rank tail finishes the exact rank behind a row barrier, so
    // its blocks must be co-resident.  When they would not be, the ordinary
    // kernel-boundary form is launched instead and the selector cannot hang.
    if(prank_resident_ok(
           (const void*)phased::k_scatter<2, true, true, true, false,
                                          false, false, 0, true>,
           (int)G, (int)R))
    {
        hipLaunchKernelGGL(
            (phased::k_scatter<2, true, true, true, false, false, false, 0, true, 16>),
            SCATTER_ARGS);
    }
    else
    {
        hipLaunchKernelGGL((phased::k_scatter<2, true, true, true>), SCATTER_ARGS);
    }
#undef SCATTER_ARGS
}

// Row stride of the ghist workspace = fine bins + the Phase K coarse summary.
// The workspace MUST be sized on this, not on hist_bins(): getting it wrong
// walks off the end of ghist into whatever is allocated next.
int64_t hist_stride() { return (int64_t)phased::GH_STRIDE; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("topk_transform", &topk_transform, "phase D top-k + page transform",
          py::arg("logits"), py::arg("row_ends"), py::arg("page_table"),
          py::arg("out"), py::arg("ghist"), py::arg("cursor"),
          py::arg("cand_cnt"), py::arg("cand_idx"), py::arg("cand_val"),
          py::arg("g_per_row"), py::arg("page_size"));
    m.def("hist_stride", &hist_stride, "ghist row stride (fine bins + coarse summary)");
}
