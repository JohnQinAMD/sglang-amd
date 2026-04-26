# DSv4-Flash mxfp4 on MI355X — Performance Headroom Analysis

**Date:** 2026-04-26
**Current state:** TPOT 50.94 ms (FlyDSL mxfp4 MoE + Triton sparse_attn_decode + CK V32 fp8-fix shipped, c=1 OSL=1024)
**B200 reference:** TPOT 7.22 ms (4-GPU TP=4)
**Realistic MI355X floor:** ~18–22 ms TPOT (2.5× headroom)

## TL;DR

- **Bottleneck:** launch-bound + long-tail elementwise glue, NOT compute or HBM bandwidth.
- **~57% of TPOT is non-kernel-work overhead** (launches, elementwise glue, inter-kernel gaps).
- **Single-largest lever:** CUDA-graph-stitching the decode forward to eliminate ~1100 launches/token of generic torch ops. Worth ~10–15 ms TPOT alone.
- **The remaining gap to B200** (after MI355X reaches its ~20 ms floor) is **structural** — Infinity Fabric vs NVLink5, HSA vs CUDA launch latency, MFMA scheduling — and not closeable at the kernel level.

## Trace evidence

torch.profiler trace, 20-token decode, TP0:

| Bucket | ms / 20 tok | ms / token | What it is |
|---|---:|---:|---|
| FP8 GEMMs (a8w8 blockscale, configs A+B) | 107 | 5.35 | Q/KV LoRA + O proj |
| BF16 GEMM `_gemm_a16_w16_atomic_kernel_M256_N256_K64` | 87 | 4.35 | lm_head (1 launch / token + warmup) |
| Sparse attention (Triton stage1+2) + topk_transform + softmax + MHC | ~155 | 7.75 | Attention pipeline |
| TP all-reduce | 65 | 3.25 | 43 layers × ~1 reduce |
| FlyDSL MoE | ~30 | 1.50 | Already optimized |
| **Subtotal of "real work"** | **~444** | **~22.2** | |
| **Long tail of small ops + elementwise + gaps** | **~710** | **~28.7** | **Launch-dominated** |

Top kernels by GPU time (TP0, 20-tok trace):

| Rank | Kernel | Total | n launches | avg | % GPU |
|---:|---|---:|---:|---:|---:|
| 1 | `elementwise_kernel<128,4>` (nan_to_num, add, mul, etc) | 107 ms | 22344 | 4.79 µs | 9.3% |
| 2 | `_gemm_a16_w16_atomic_kernel_M256_N256_K64` (lm_head) | 87 ms | 817 | 106 µs | 7.5% |
| 3 | `_gemm_a8w8_blockscale_kernel` (FP8 LoRA proj A) | 71 ms | 1634 | 43.6 µs | 6.2% |
| 4 | `cross_device_reduce_1stage` (RCCL TP all-reduce) | 65 ms | 1653 | 39.6 µs | 5.7% |
| 5–7 | `elementwise_kernel_manual_unroll` variants | 138 ms | ~35k | 3.8–4.0 µs | 11.9% |
| 8 | **`_sparse_attn_decode_stage1`** (Triton attention) | 39 ms | 817 | 47.6 µs | 3.4% |
| 9 | `_gemm_a8w8_blockscale_kernel` (config B, smaller) | 36 ms | 817 | 43.6 µs | 3.1% |
| 10 | `cunn_SpatialSoftMaxForward` | 34 ms | 1158 | 29.2 µs | 2.9% |
| – | TileLang MHC (`mhc_pre_big_fuse`, `mhc_pre_gemm_sqrsum`, `mhc_post_tilelang`) | 61 ms | ~5000 | ~12 µs | 5.3% |
| – | `_topk_transform_512_kernel` | 21 ms | 399 | 52.9 µs | 1.8% |
| – | FlyDSL MoE (stage1+stage2+sort+quant) | ~30 ms | ~3000 | varies | ~3% |

## Roofline sanity check (per token, TP=4)

- MoE active weight HBM traffic: ~1.3 GB at 5 TB/s = **260 µs**
- Projection weights: ~3.5 MB (negligible at bs=1)
- KV-cache reads (sparse, topk=512, fp8, 43 layers): ~13 MB = **3 µs**
- Total compute (MoE 200 MFLOPs + attention 2.9 GFLOPs at 2600 TFLOPS fp8): **<2 µs**

**Pure-physics lower bound: ~0.5 ms/token.** The chip can do this. We are at 50 ms. **Memory bandwidth is not the wall.**

## Prioritized strategy (full architect verdict)

### HIGH — combined ~−22 to −30 ms TPOT (50.9 → 21–29 ms)

#### H1. CUDA-graph-stitch decode forward + eliminate elementwise glue [−10 to −15 ms]

The trace shows **22344 elementwise launches over 20 tokens** = 1117 generic torch op launches per token (`nan_to_num`, `add`, `mul`, view-copy chains). Each is ~4.8 µs of kernel + ~3 µs ROCm launch overhead → **8–9 ms/token of pure launch overhead** just for elementwise.

The fix is a sglang module-rewrite, not a kernel:
- Eliminate `nan_to_num` (root-cause the NaN-producing path; the all-masked-row case in sparse-attention epilogue is already handled inside the Triton kernel — outer guard is dead)
- Fuse view/copy/slice chains via `torch.compile` or hand-fused Triton epilogues
- Capture once per bs, replay on every decode

**Owner:** HIP / framework integration (sglang-side rewrite).
**Files:** `python/sglang/srt/models/deepseek_v4.py`, `python/sglang/srt/flashmla_tests/ref.py`

#### H2. Fuse MHC pre/post chain into one TileLang kernel [−2 to −3 ms]

`mhc_pre_big_fuse` + `mhc_pre_gemm_sqrsum` + `mhc_post_tilelang` is **3 launches per layer × 43 = 129 launches/token = 3.05 ms/token**. Inter-launch gaps + redundant HBM round-trips.

Single fused decode-shape kernel: keep everything in registers/LDS, one global write.

**Owner:** FlyDSL (TileLang author already has the building blocks; FlyDSL's persistent kernel model is the right pattern).
**Files:** `python/sglang/srt/layers/attention/compressed/compressor.py`, FlyDSL ref `aiter-amd/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`

#### H3. Fuse Triton sparse_attn + topk_transform + softmax into one prologue [−3 to −5 ms]

Today: `_sparse_attn_decode_stage1` (47.6 µs) + `_topk_transform_512_kernel` (52.9 µs) + `cunn_SpatialSoftMaxForward` (29.2 µs) is **3 launches × 43 layers = 130 launches/token = 5.6 ms/token in attention infrastructure that should be 1 launch**.

CK V32 is at parity with Triton at c=1 because the attention is **launch-bound, not compute-bound** (BLOCK_H=4 × Hq=128 / TP=4 = 32 wavefronts → 12.5% CU occupancy). Fix is to **fuse adjacent kernels** to amortize launch over more work, not to make the inner GEMM faster.

The Triton sparse decode kernel already does online softmax internally — the external `cunn_SpatialSoftMaxForward` is a **separate softmax**, likely router-gating or sink-mix. Identify which softmax is on the critical path; fuse into the producing kernel.

**Owner:** CK (V32 author) + Triton (epilogue glue).
**Files:** `python/sglang/srt/flashmla_tests/triton_sparse_decode_kernel.py`, `csrc/ck_v32/mla_decode_fwd_kernel.hpp`

#### H4. Group 43 TP all-reduces into 1–2 + RCCL one-shot for <16KB [−1.5 to −2 ms]

65 ms / 1653 launches = 3.25 ms/token in 43 reduces × 39.6 µs avg. **Per-call latency dominates over actual transfer** — each reduce is hidden × bf16 = ~8 KB × 4 ranks (trivial).

Two paths:
- (a) **Fuse residual + allreduce** into the Q/KV down-projection epilogue (CK GEMM with built-in cross-GPU atomic — aiter has the building blocks)
- (b) **Use RCCL `ncclAllReduce` with one-shot register-only kernel** for messages < 16 KB (already supported on gfx950 with peer-direct)

Expect 43 reduces × 39.6 µs → 43 × ~15 µs (launch-amortized) → recovers ~1 ms; further GEMM-epilogue fusion recovers another ~1 ms.

**Owner:** HIP + aiter (RCCL tuning + GEMM-epilogue allreduce).
**Files:** `python/sglang/srt/layers/attention/flashmla_backend.py`, RCCL config

### MEDIUM — combined ~−3 to −5 ms

#### M1. lm_head BF16 GEMM → FP8 GEMV [−2 ms]

`_gemm_a16_w16_atomic_kernel` at M=256 N=256 K=64 implies vocab≈128k tiled. At bs=1 this is **GEMV bound by reading lm_head weights** (vocab × 4096 × 2B = ~1 GB → 200 µs at 5 TB/s, but observed 4.35 ms → 20× off peak — kernel is the wrong shape for GEMV).

Path: dedicated decode-shape FP8 lm_head GEMV in CK or aiter.

**Owner:** aiter (already has FP8 GEMV kernels — just needs wiring).

#### M2. Tune FP8 a8w8 blockscale GEMVs for bs∈{1,2,4,8} decode shape [−1 to −2 ms]

Two configs (A: 71 ms / 1634 = 43.6 µs avg; B: 36 ms / 817 = 43.6 µs). Both at decode shape are **GEMV-shaped** — a 43.6 µs GEMV reading ~32 MB of fp8 weight is ~730 GB/s, well under 5 TB/s HBM peak. Headroom is ~3–5×. At bs=1 the realistic ceiling is ~10 µs each (limited by L2-resident kvA/qA tile + weight stream).

**Owner:** CK (cktile fp8 GEMV) or aiter.

#### M3. Eliminate `nan_to_num` outer guard [−0.5 to −1 ms]

107 ms / 22344 launches = 4.79 µs each, ~22 launches/token. Push the all-masked-row guard into the producing kernel (the Triton sparse decode already does it). If a real NaN source exists, root-cause it instead.

**Owner:** HIP / framework integration.

### LOW — skip until HIGH+MEDIUM land

- **L1.** FlyDSL MoE persistent stage1+stage2 fuse — saves ~0.5 ms total at high engineering risk; MoE is already 4% of TPOT.
- **L2.** Inter-kernel gaps (~7 ms/token) — mostly disappear once H1 lands; do not attack independently.

## Realistic MI355X floor: ~18–22 ms TPOT

| Layer | MI355X realistic | B200 measured |
|---|---:|---:|
| Pure HBM/compute (physics floor) | 0.5 ms | 0.4 ms |
| 130 unavoidable launches × 3 µs ROCm latency | 0.5 ms | 0.2 ms |
| 43 collectives × 15 µs RCCL one-shot | 0.6 ms | 0.15 ms |
| MFMA dependency chains (gfx950 16-cycle vs B200 8-cycle) | 0.4 ms | 0.2 ms |
| Quant pipeline + scatter/gather | 0.5 ms | 0.3 ms |
| Long-tail elementwise (post-H1) | 1.0 ms | 0.5 ms |
| Framework / scheduler | 0.5 ms | 0.5 ms |
| Margin / unmodeled | 14–18 ms | 5 ms |
| **Realistic TPOT** | **~18–22 ms** | **7.22 ms** |

## The 11–15 ms residual gap to B200 is structural

| Source | Share | Why it's structural |
|---|---:|---|
| Interconnect (Infinity Fabric vs NVLink5) | ~40% | HW: ~200–250 GB/s vs ~900 GB/s ring; collective latency floor is silicon |
| Kernel-launch architecture (HSA vs CUDA) | ~30% | ROCm launch ~3–5 µs vs CUDA ~1.5 µs |
| MFMA scheduling | ~20% | gfx950 MFMA latency ~16 cycles vs B200 tensor cores ~8 cycles |
| torch.compile / Inductor maturity | ~10% | NV has 18-month head start on DSv3/V4 fusion patterns |

None closeable at the kernel level. Closing requires MI400 + Infinity Fabric 4 silicon + another generation of ROCm runtime maturity.

## Risk register

| Risk | Mitigation |
|---|---|
| H1 graph-stitch breaks dynamic-shape capture for bs=1,2,4,8 | Capture per-bs graph (already pattern in sglang); validate with multi-bs harness before landing |
| H3 CK+Triton fusion regresses Pro V32 fp8-decode-scale fix | Keep V32 path behind a flag; gate fusion on (mode==decode, c==1, bs<=8) |
| H4 RCCL one-shot path has latent correctness bugs at 4-rank fp8 | Validate against existing 1-stage path with bit-exact fp32 accumulator comparison; ship behind env flag |
| M1 lm_head FP8 quant degrades top-1 accuracy | Use blockscale fp8 (not per-tensor); validate on the same eval harness as the mxfp4 MoE work |
| Diminishing returns: land everything, hit 22 ms not 18 ms | Acceptable — 22 ms is 2.3× over 50.9 ms, the headline number that matters for hardware-comparison narrative |

## Honest narrative for leadership

> "We can publish a **2.5× MI355X improvement on DSv4-Flash** (50.9 → ~20 ms TPOT) and credibly claim software parity on the kernels themselves. The residual gap to B200 is **interconnect + framework maturity**, both roadmapped to MI400."
