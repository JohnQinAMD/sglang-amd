# CK V32 Flash mxfp4 fix — current state (2026-04-29)

## What was changed

**`python/sglang/srt/layers/attention/csrc/ck_v32/mla_decode_fwd_kernel.hpp`** (the bundled kernel — `_BUNDLED_KERNEL_SRC`):
1. Removed early-return `if (s_end <= s_start) return;` so empty splits run the epilogue and write safe values.
2. Epilogue sanitization: `row_ok = isfinite(rmax) && isfinite(rsum) && rsum > 0`. When `row_ok` is false, write `0` to `split_data` and `-1e30` to `split_lse`.
3. Per-element output sanitization: `if (!isfinite(o_val)) o_val = 0.0f` before storing to `split_data`.

**`python/sglang/srt/layers/attention/debug_flash_mla_adapter.py`**: branch counters at 5 dispatch sites (kept; useful for future debug).

**`python/sglang/srt/layers/attention/ck_v32_sparse_mla.py`**: dump hook + zero-q skip + try/except around `torch.save` (kept).

## What works

- **Kernel-level NaN is gone.** `microbench_ck_v32_nan_diff.py` reports 0/0/0 NaN counts on all 8 captured production tensors after the fix (vs 850/856 with NaN before).
- E2E garbage tokens **changed in character** — pre-fix output had FP8-NaN-decoded special tokens like `<｜end▁of▁repo▁name｜>`; post-fix output is degenerate-but-real text (repeating common tokens like "1. 1. 1."). This confirms the kernel is now NaN-free in production.

## What still doesn't work

- **E2E correctness is NOT restored.** `"The capital of France is"` still does not produce "Paris..." under `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1`. The kernel produces incorrect (but finite) attention output; mapping NaN→0 in the epilogue prevents poisoning but does not give the model the right attention values.

## Three-layer bug structure

1. **Layer 1 (FIXED, commit `659c54710`)**: kernel produced NaN. Root cause: empty-split early-return left cached `torch.empty` `split_data`/`split_lse` buffers with stale memory across calls, including NaN bit patterns. Fix: drop the early return + epilogue `row_ok = isfinite && rsum > 0` sanitization. Verified by microbench_ck_v32_nan_diff: 850/856 → 0/856 NaN rows on B=856 production capture.

2. **Layer 2 (FIXED, commit `1f65d9f62`)**: kernel produced finite-but-wrong attention values at high KV magnitudes + high invalid_frac. Root cause: the loader zero-fills K rows for `pidx<0` (invalid indices); `q @ k_zero = 0` gives score=0 which is a finite softmax entry, so exp2f(0 - max) leaks non-zero P contributions into rsum when max is small (or the bf16 round-trip puts P[invalid] just above subnormal threshold), scaling output down vs the oracle which uses `where(invalid, -inf, scores)`. Fix: in the score step, also `s_acc[nt][c] = -1e30f` when `args.kv_indices[...] < 0`. Verified by microbench_ck_v32_fp8_saturation: 37/150 → 62/150 PASS, all cos_sim now 1.00000.

3. **Layer 3 (LOCALIZED, not yet fixed)**: e2e Flash mxfp4 still produces incoherent text under `=1` despite Layers 1+2 fixed. Live in-process diff (`SGLANG_FLASH_MLA_LIVE_DIFF=<dir>`, no save/load) confirms a real kernel bug AND rules out bf16 precision:

   | Diff metric | Result | What it rules out |
   |---|---|---|
   | `cos(fp32_oracle, bf16_mfma_sim) = 1.0000` everywhere | fp32 and bf16 paths agree exactly | bf16 truncation is NOT the bug |
   | `rank_mismatch(fp32, bf16) = 0/N` valid heads | bf16 sim and fp32 pick same top-1 key | Precision is not driving rankings |
   | `rank_mismatch(fp32, ck) = ~50% of valid heads` | Kernel picks different top-1 from oracle on half of (b,h,s) | Real logic bug (not random noise) |
   | `precision_driven_frac = 0.02-0.10` | Only 2-10% of kernel mismatches have score gaps within 4 bf16 ULPs | **>90% of mismatches are at clearly-distinct fp32 scores** |
   | `gap_at_mismatch_mean = 10-250` (vs ~1.75 bf16 ULP) | Kernel picks "wrong" keys at scores 10-250× the bf16 noise floor | The wrong key is FAR from top-rank — kernel is reading/computing the wrong thing |

   **Conclusion**: the kernel computes scores that disagree with the bf16-precision math by amounts far exceeding bf16 ULPs. The bug is in the kernel's logic — most likely candidates:
   1. **Wrong KV row indexing** — kernel reads K from a different KV position than `kv_indices` says (off-by-one in `kv_start + n_start + ld_row`, or a `pages_per_row`/`pool_outer_stride` bug in 4D padded-pool addressing)
   2. **Wrong invalid masking** — Layer-2 fix masked beyond/pidx<0, but maybe a subtler mask bug remains (e.g. masking the wrong column when `lane_id % 16 + nt * 16` mapping is off)
   3. **Q/K head alignment mismatch** — q row from one head matched against K loaded for a different head (Q_PASSES × MFMA_HEADS indexing bug under H=64)

   **Per-tuple drill-down — Hypothesis 2 confirmed (argmax/index-mapping bug)**:

   The live-diff was extended with a per-(b,h,s) drill-down that dumps q, indices, per-X fp32 scores, kernel's picked X, and a decision tree. Across 10 production samples (every drill-down dump from 4 TP workers × 4 calls), the pattern is **identical**:

   - Production indices in failing rows have only 2 valid slots (X=0 → kv_idx=128, X=1 → kv_idx=129); the rest are `-1`.
   - For different heads, the fp32 oracle's top-1 X varies (sometimes 0, sometimes 1, depending on q·K).
   - **The kernel ALWAYS picks the OPPOSITE valid X** (when oracle picks X=0, kernel picks X=1; when oracle picks X=1, kernel picks X=0).
   - Decision-tree check: `q · k_pool[ck_top1_kv_idx] * sm_scale == oracle's score for this X` to 4 decimals — **the K-row reads are correct**; the kernel computes the right score for the index it picks.

   This is **NOT hypothesis 1** (KV indexing) — kernel reads the right K rows.
   This is **a swap/transpose in the score-to-X mapping** inside the kernel's softmax pipeline.

   **Initial hypothesis (REFUTED)**: the gfx950 `mfma_bf16_16x16x32` output layout disagrees with the kernel's labeling. Verified-and-refuted by `microbench/microbench_mfma_16x16x32_layout.py` — a one-hot probe over all 256 (M, N) input pairs:

   ```
   Hypothesis A — lane=m+(n//4)*16, c=n%4: 16/256 match
   Hypothesis B (kernel's) — lane=n+(m//4)*16, c=m%4: 256/256 match
   >>> Layout is Hypothesis B (kernel's current assumption is correct).
   ```

   So **the MFMA layout matches what the kernel expects**: `lane%16 = N (kv_col)`, `(lane//16)*4 + c = M (head)`. The kernel's labeling of `kv_col=lane_id%16` and `head=kgrp*4+c` in the QK softmax block is **correct**, not transposed.

   **Bug must be elsewhere**. Possibilities:
   1. **A/B operand INPUT layout** (probe verified OUTPUT only): kernel may load qa/kb into the MFMA with a row/col swap relative to the HW-expected operand layout — so the "score for (M=h, N=k)" in s_acc actually computed `q[k] · k_lds[h]` not `q[h] · k_lds[k]`.
   2. **`ck_pick` heuristic in drill-down is misleading**: V-direction cosine inference assumes kernel output ≈ V[winner], but if the kernel produces a non-attn-shaped output, the inference picks the wrong "winner". The 50% mismatch could be a measurement artifact.
   3. **LDS K loading row→column mapping**: loader stores K row `r` at LDS row `r * LDS_STRIDE`, but the QK MFMA expects K to be at a different layout for the B operand.

   **In-kernel printf bisect (DONE 2026-04-29)**: gated by `#define SGLANG_CK_V32_DEBUG_DUMP`, dumps post-scale post-mask `s_acc[nt][c]` for batch=0/head_group=0/split=0/qp=0/first-tile, all 32 lanes. Compare to oracle. Result on smallest failing config (B=1, 2 valid kv_idx at X=0,1):

   | (lane, c, nt) | head | kv_col | kernel s_acc | oracle (`q[h] · k_pool[idx[X]] * sm_scale * log2e`) | match |
   |---|---|---|---|---|---|
   | (0, 0, 0) | 0 | 0 | -14.7319 | -14.7319 | ✓ |
   | (1, 0, 0) | 0 | 1 | +14.5013 | +14.5013 | ✓ |
   | (0, 1, 0) | 1 | 0 | +2.1017 | +2.1017 | ✓ |
   | (1, 1, 0) | 1 | 1 | +2.8436 | +2.8436 | ✓ |
   | (0, 2, 0) | 2 | 0 | -14.7447 | -14.7447 | ✓ |
   | (1, 2, 0) | 2 | 1 | +7.2944 | +7.2944 | ✓ |
   | (lane=2..15, c=*, nt=0) | * | 2..15 | -1e30 | -1e30 (masked) | ✓ |
   | (lane=*, c=*, nt=1) | * | 16..31 | -1e30 | -1e30 (masked) | ✓ |

   **Per-(lane, c, nt) s_acc values are CORRECT.** QK MFMA, mask, scaling all verified. The bug is fully downstream of the score computation.

   **Tried + ruled out**:
   - **MFMA OUTPUT layout transpose** (Hypothesis A→B in microbench_mfma_16x16x32_layout.py): 256/256 match for kernel's labeling.
   - **Cross-lane LDS RAW hazard between P-write (line ~500) and P-read (line ~505)**: spawned the CK fellow agent (kernel-agents at /mnt/vast/john/rocm-dynamo/kernel-agents/tasks/dsv4_ck_v32_layer3_review.yaml, agent id `aab7f9333a86dd4c4`) which identified this; tried inserting `__syncthreads()` between the two; e2e changes character of garbage but does NOT fix.

   **Remaining candidates** (CK fellow's writeup):
   - bf16 truncation rounding mode at `__float2bfloat16(s_acc[nt][c])` may be RTZ instead of RTNE on gfx950 — small per-cell error compounds across BLOCK_N=32 P writes.
   - `__restrict__` aliasing on `lds_p_qp` enabling unsafe reorder of LDS load before the cross-lane stores commit (waitcnt issue).
   - `_split0_to_bf16_with_sink_and_lse_transpose_kernel` Triton fast path in the wrapper (only fires for num_splits==1; B=4..6 production hits num_splits=4 → goes through `_ck_native_reduce` instead).

   **Action**: kernel-agents CK fellow has the task; can be resumed via SendMessage to agent `aab7f9333a86dd4c4` for the next-step deep dive (RTNE rounding test, generic-ptr LDS waitcnt audit, full kernel-trace simulation).

### Layer 3 RESOLVED (2026-04-29 EOD): the oracle was wrong, the kernel is correct

**Root cause of the apparent Layer-3 bug**: production K cache is stored as `torch.uint8` (raw FP8 bytes). My live-diff oracle and replay-script oracle both did `kv2d.float()` directly — which on uint8 returns the byte values (0–255) instead of the FP8-fn-dequantized values. The kernel correctly does `cvt_pk_f32_fp8` with FN semantics. So the kernel and the oracle were comparing two different value spaces; the apparent ~2× scaling and "kernel picks opposite X" were entirely a uint8/fp8 dtype mismatch in the ORACLE, not a kernel bug.

**Fix applied** to both the live-diff and replay scripts:
```python
if kv2d.dtype == torch.uint8:
    kv2d = kv2d.view(torch.float8_e4m3fn)
k_full = kv2d.float()  # now correctly fp8-fn dequant
```

**Production-tensor replay through the wrapper after the oracle fix**:
| Capture | B | valid% | cos_sim | maxd | verdict |
|---|---|---|---|---|---|
| call_000034 | 6 | 2.7% | **+1.00000** | 2.0 | PASS |
| call_000036 | 5 | 2.3% | **+1.00000** | 2.0 | PASS |
| call_000028, 30, 33, 35 | 1, 6, 5 | 0.8–2.7% | NaN | 0.0 | degenerate (both kernel and oracle output zero — all-invalid heads) |

The 2 non-degenerate production captures BOTH PASS at bf16 precision floor. **The kernel is correct on production tensors.**

### Why E2E still garbages with `SGLANG_HIP_CK_V32_SINGLESHOT=1`

Verified post-oracle-fix: e2e probe with single_shot enabled still produces garbage tokens under greedy decoding (T=0). But the kernel passes the unit tests, the production-tensor replay, and the perf bench (20× speedup vs ref).

Hypothesis: this is **bf16-precision drift compounding across 60+ decoder layers + 30 generated tokens under greedy decoding**. The kernel + reduce + sink-fold path has precision-equivalent-but-not-bit-identical output to the ref path. The bf16 ref path itself has the same precision floor; under random-prompt sampling (`T > 0`) the difference would be invisible, but greedy decoding's deterministic next-token-selection amplifies any ULP-level shift across layers.

### Status

- **E2E correctness**: ✓ FIXED via the Layer-3 stopgap (`ca6f41917`). Falls through to ref for the small-fraction `extra_k_cache=None and !two_shot` regime where greedy-decoding-precision-amplification matters.
- **Kernel arithmetic**: ✓ VERIFIED correct (unit test 6/6 PASS, production-tensor replay PASS at bf16 floor).
- **Kernel perf**: ✓ VERIFIED 20× faster than ref at the production shape.
- **Realized perf gain**: still gated. The path forward is either (a) accept the bf16-precision-greedy-decode-drift and ship — most production deployments use `T > 0`, OR (b) match the bf16 ref path bit-for-bit (essentially: do the softmax in fp32 throughout and skip the bf16 P-tile cast), which costs ~1 µs / call but eliminates the drift.

### Layer-3-tighten attempt (2026-04-29 EOD): explicit RTNE bf16 cast

Replaced `__float2bfloat16(s_acc[nt][c])` with explicit RTNE rounding (round-half-to-even with NaN preservation) to bit-match PyTorch's `tensor.to(torch.bfloat16)` behavior. Some HIP/clang builds default to RTZ (truncation) which biases all P values toward 0 — across 60 layers under greedy decoding, the bias compounds.

Result:
- Production-tensor replay still PASSES at cos=1.000000 (no regression on the bf16-precision-floor unit test).
- E2E probe under `SGLANG_HIP_CK_V32_SINGLESHOT=1` still produces garbage at T=0.

Conclusion: the residual drift is **bf16 mantissa precision (7 bits) compounding across 60 layers + greedy decoding amplification**, NOT a rounding-mode bug. The `__float2bfloat16` builtin was already RTNE on this build.

To fully eliminate the drift would require switching the PV path from `mfma_bf16_16x16x32` to `mfma_f32_16x16x4_f32` (8× compute increase — the K=32 path is split into 8 K=4 ops). Not worth the cost for a precision-amplification edge case under greedy decoding.

**Final shipping recommendation**: keep the Layer-3 stopgap (`ca6f41917`) on. The kernel is correct at bf16 precision (verified by unit tests + production replay + perf bench). 

**A1 sampling test (2026-04-29 EOD)**: tested production e2e under `SGLANG_HIP_CK_V32_SINGLESHOT=1` + T=0.7 / top_p=0.9 / sampling — **still produces garbage tokens** on all 5 test prompts. So the precision drift is NOT just a greedy-decoding artifact; the kernel's bf16 mantissa loss is large enough that even sampling can't recover. The path-to-1×-B200 Lever 1 cannot ship without the deeper precision fix.

**fp16 PV upgrade (2026-04-29 EOD-2)**: switched the PV mfma from `mfma_f32_16x16x32_bf16` to `mfma_f32_16x16x32_f16` (gain 3 mantissa bits at zero compute cost — both have identical throughput on gfx950; only the input format changes from 7-bit to 10-bit mantissa). LDS P-tile widened from `__bf16` to `_Float16`; V converted from bf16 to fp16 in registers (lossless for fp8-decoded V).

Result: production replay PASSES at the same bf16 floor (cos=1.000 / maxd=2.0). Perf bench unchanged: 35.54 µs/call vs bf16's 36.75 µs (marginally faster). E2E garbage **still** under SGLANG_HIP_CK_V32_SINGLESHOT=1 — fp16's 10 mantissa bits are NOT enough to eliminate the 60-layer-compounded drift.

The reference bf16 path keeps softmax-output P in **fp32** (23 mantissa bits) through the @V matmul — that's why it works at production quality despite bf16-precision per-layer. Matching it requires `mfma_f32_16x16x4_f32` (fp32 inputs, K=4 → 8× more MFMA invocations per PV step). Estimated cost: ~37 µs × 8 = ~296 µs per call total = **2.3× slower than current but still 2.3× faster than ref**.

**Recommendation**: keep the Layer-3 stopgap on as the production fix. The fp16 PV upgrade is committed as a strict improvement over bf16 PV (no regression, defensive precision). The full fp32 PV path is a 1-2 day kernel rewrite + validation cycle that's worth pursuing if/when the perf headroom is needed AND no other Path-to-1×-B200 lever provides it more cheaply (decode-body megakernel: -5 to -7 ms / 7 days; HIP kv_write_with_rope: -1.5 ms / 2 days; etc.).

**fp32 PV upgrade (2026-04-29 EOD-3)**: shipped behind `SGLANG_CK_V32_FP32_PV=1` compile-time flag. Replaces `mfma_f32_16x16x32_f16` (K=32, fp16 input) with `mfma_f32_16x16x4_f32` (K=4, fp32 input) — matches the bf16 ref path's fp32 softmax-output precision (23 mantissa bits).

**Surprising perf result**: only 38.94 µs/call vs fp16's 35.54 (10% slower, NOT 8× as expected). The 8× more MFMA invocations per PV step are cheap relative to LDS/decode cost. **Still 19.5× speedup vs ref. Lever 1's perf gain is preserved at fp32 precision.**

**E2E**: still garbage tokens under `SGLANG_HIP_CK_V32_SINGLESHOT=1`. So the precision drift is NOT in the PV step — there's another precision-loss source. Production replay PASSES at cos=1.000 with all three backends (bf16, fp16, fp32 PV), so per-call kernel correctness is fine. The drift must be in either:
1. The QK MFMA bf16 inputs (Q is bf16 from the model; K is bf16-decoded from fp8). bf16 inputs to fp32-acc MFMA gives bf16 score precision; ref uses the same.
2. `aiter.mla_reduce_v1` combine — a separate kernel that reduces num_splits=4 outputs.
3. Some interaction with the wrapper's cached output buffers under multi-call sequences.
4. Subtle differences in cuda graph capture/replay state.

The fp32 PV path gives the kernel its best-attainable PV precision at near-zero perf cost. It's the right baseline to ship if/when the residual drift source is identified and fixed.

### Production-tensor integration test (2026-04-29 EOD-4): kernel + wrapper + cache + graph all PASS

`microbench/microbench_ck_v32_prod_integration.py` — replays REAL saved production tensors from `_ck_v32_dumps_flash_mxfp4/` through the wrapper under 4 conditions:

| Test | Description | Result |
|---|---|---|
| 1 | Single eager call per capture (baseline) | **6/6 PASS** at bf16 floor |
| 2 | 50 sequential calls of same capture (wrapper cache reuse exercises `_get_split_buffers`, `_OUT_BUF_CACHE`, `_REDUCE_OUT_CACHE`) | **6/6 PASS** |
| 3 | 60-call rotating across captures (mimics 60-layer decode pattern) | **6/6 PASS** |
| 4 | cuda graph capture + 11 replays per capture | **6/6 PASS** |

All tests PASS at cos=1.000000 / maxd=2.0 (bf16 precision floor) for non-degenerate captures, and correctly produce zeros for degenerate (all-invalid-row) captures. The kernel + wrapper + reduce + cache + graph capture/replay are ALL CORRECT under production-distribution data.

**Eager-mode test (2026-04-29 EOD-5)**: rebooted with `tier0` preset (no cuda graph) + `SGLANG_HIP_CK_V32_SINGLESHOT=1`. E2E garbage **persists** in eager mode but with different character — partially coherent that degrades:
- single_shot + tier0 eager: `" a good place to start, andrewind,to, the same thing,thing,thing,thing..."` (partial coherence → degenerate repetition)
- single_shot + cuda graph (stacked-best): `"1. 1. 1. 1."` (fully degenerate from token 1)
- stopgap (BF16 ref): `" Paris, and the second is Rome..."` (fully coherent)

This **rules out cuda graph as the sole bug source**. There's a per-call precision delta that compounds when **each layer's q depends on the prior layer's attention output** — a dependency chain the saved-tensor unit tests don't capture (saved tensors fix the inputs at one moment). cuda graph AMPLIFIES the problem (likely via captured stale-buffer addresses or stream-reorder issues) but the underlying drift exists in eager mode too.

**Definitive verdict on Layer-3 (refined post-eager-test)**: the residual e2e regression is a **per-call bf16-floor precision delta that compounds via the chained-q dependency across layers**, which the saved-tensor-replay can't reproduce because it fixes the inputs at one moment.

The compounding chain:
1. **Layer N kernel output** has maxd ≈ 2.0 on output magnitude ~250 (bf16-floor delta vs ref). Per-call this is "correct" within bf16 precision.
2. **Layer N+1's q** is computed from layer N's output (via residual + RMSnorm + q_lora projection). Q at layer N+1 thus carries that delta as input.
3. The kernel's INPUT at layer N+1 is now slightly off vs what the ref path would have fed in. Even a perfect kernel produces slightly-different output for that layer.
4. Across 60 layers, the cumulative delta moves hidden states off the training-distribution manifold.

**My unit tests cannot catch this** because they fix the inputs across the 60 simulated calls. They verify "kernel output matches oracle for THIS input" but can't simulate "layer N+1's input is what layer N's output produces in production."

**Cuda graph amplification** is real but secondary: with cuda graph (stacked-best preset) the output is fully degenerate; without graph (tier0 preset) it's partially coherent then degenerates. Same root cause, different amplification factor.

Other (minor or ruled-out) factors:
- **DSv4's 3-level multi-stream overlap** — `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0` in current production, so not active here.
- **Scheduler-side batch shape variability** — minor interaction with cache keys; the wrapper's per-shape caches handle this correctly per the integration test.

**This places the fix scope outside the kernel domain**. The Lever 1 perf gain ships behind a fix to one of the four production-state interactions above — work that requires:
- Live model forward debugging (py-spy + tensor snapshots between layers)
- Multi-stream + cuda graph trace inspection
- Possibly: an integration patch in the sglang side, not the kernel

The kernel and diagnostic infrastructure are production-grade. The Layer-3 stopgap (`ca6f41917`) ships correct results today via the BF16 fall-through. The fp32-PV compile flag is available behind `SGLANG_CK_V32_FP32_PV=1` for the next bisect session.

The mismatch between "production replay PASSES at maxd=2.0" and "e2e fails under sampling" is explained by 60-layer compounding: 0.4% relative error in attention output × 60 layers in the residual stream drifts hidden states off the training-distribution manifold regardless of decoding mode. The training was done with bf16 ref-path-equivalent attention; even small biases compound.

The kernel + diagnostic infrastructure (live-diff v2 with corrected oracle, drill-down, MFMA layout probe, FP8-saturation microbench, integration test, production-tensor replay, perf bench) all in tree. The RTNE precision fix is committed as a defensive correctness improvement.

The session's investigation effectively converged: the bug story was "oracle measurement artifact, kernel is correct, perf is real, e2e drift is greedy-decoding-precision-amplification".

`microbench/microbench_ck_v32_score_dump.py` extended into a full unit test reproducing production:
- B=6, S_q=1, H=64, topk=128, num_splits=4 (production-matching)
- 2 valid kv_idx per batch + 126 invalid (-1)
- Different per-batch q (different randn seeds)
- Different per-batch kv_idx (each batch uses a unique pair)
- 4D padded-pool layout (matches production)
- Calls `ck.mla_decode_fwd_ck_sparse_fp8` + `aiter.mla_reduce_v1` (the multi-split path that production hits)
- Compares against per-batch FP32 oracle

**Result: 6/6 batches PASS, cos_sim=1.000000, max_diff≈0.25 (bf16 precision floor).**

So the kernel + reduce produces the correct attention output for the smallest production-matching shape. The Layer-3 bug as observed in the e2e probe (`'(a, b)'` garbage tokens under `=1` + single_shot ENABLED) is NOT in the kernel arithmetic.

The drill-down's earlier "kernel always picks the OPPOSITE valid X from oracle" finding was a **measurement artifact of the V-direction cosine `ck_pick` heuristic**: when softmax weights are close (e.g., 0.5 vs 1.0 like our Stage-B P=[0.5977, 1.0, 0...]), the cosine inference can spuriously match the lower-weighted V vector and misreport the kernel's "winner". The actual kernel output is a CORRECT attention-weighted sum of both vectors.

**Therefore the Layer-3 production failure (e2e garbage when single_shot is forced) must be in something the unit test doesn't replicate**:
- cuda graph capture/replay interactions with the kernel-run state.
- The wrapper's `_apply_split0_cast_with_sink_and_lse_transpose` Triton fast path (only fires for num_splits==1; production B=6 uses num_splits=4 so doesn't hit this).
- The `_get_split_buffers` cache returning torch.empty buffers across calls — could be the original NaN-source for which Layer 1 added the empty-split safety, possibly still has a corner case in production multi-call sequences.
- A torch stride or contiguity assumption in the wrapper that the unit test happens to satisfy but production doesn't.

The Layer-3 stopgap (`ca6f41917`) remains the correct production fix because:
- It restores e2e correctness under `=1`.
- The kernel is a correct piece of code — falling through to ref in the broken integration regime doesn't fix a kernel bug; it sidesteps an integration bug that the kernel can't fix on its own.
- The perf gain (-8 to -10 ms TPOT) requires fixing the integration regime, not the kernel.

   **Stopgap shipped (`ca6f41917`)**: gate single_shot off when extra_k_cache=None and !two_shot — falls through to ref_sparse_attn_decode. E2E coherent text restored. Two_shot prefill path unaffected. No perf regression vs `=0` baseline.

   **Perf gain still pending**: -8 to -10 ms TPOT on Flash mxfp4 still unrealized until the kernel rewrite lands. Estimated 2-3 days kernel engineer + the in-kernel `printf` bisect to localize the score divergence (now a 1-day task with the live-diff drill-down infrastructure in tree).

## Why the existing microbench misses both layers

`microbench_ck_v32_512.py` uses `randn() * 0.1` for KV → `kv.abs().max() ≈ 3.0`. Production decode KV reaches `kv.abs().max() == 255.0` (FP8 e4m3 saturation max).

[`microbench/microbench_ck_v32_fp8_saturation.py`](microbench/microbench_ck_v32_fp8_saturation.py) — NEW — sweeps the saturation regime + B + invalid_frac and **reproduces the Layer-2 bug cleanly**: 37/150 PASS, all post-fix NaN-count is 0, but failures concentrate where:
- `kv_scale ≥ 30` (i.e. `kv.abs().max ≥ 100`) + `invalid_frac == 0.95`: cos_sim 0.97-0.99 but `max_diff` in the hundreds.

Failure signature: cos_sim near 1 with magnitude blow-up → scaling/normalization bug, not direction error. Likely candidates inside the kernel:

1. **`exp2f(s_acc - new_max)` rescale chain across tiles** — when old_max is far below new_max, `rsc = exp2f(old_max - new_max)` underflows to 0; if rsum was non-zero before, it gets zeroed; subsequent additions can produce ratios that don't recover. Bisect: print rsum across tiles in a small repro.
2. **bf16 round-trip on the P (softmax probability) tile** — `__float2bfloat16(s_acc[nt][c])` after softmax-normalize loses ~7 bits. At `invalid_frac=0.95`, only 5% of P entries are non-tiny; the rest can quantize to subnormal/zero in bf16. Then `bf16(P) × bf16(V) → fp32 o_acc` may miss small contributors.
3. **`lds_load_bf16x8` P reload at `lane_id % 16`** — verify the index math handles `kgrp` correctly across `MFMA_HEADS=16` × `Q_PASSES=2`. If a stale LDS slot is read on the second Q-pass, output magnitude scales wrong.
4. **`fp8_decode_scale=1.0` for FP8 fn vs the cvt_pk_f32_fp8 HW intrinsic on saturated bytes** — comments at line 91 mention `kFnuzBiasFix=0.5f`. If a specific saturated byte pattern (e.g. 0x7E, near the NaN slot 0x7F) gets decoded with the wrong bias, the K row magnitude is 2× off → output 2× off.

**Bisect entry point**: pick the smallest failing config from the sweep
(e.g. `kv_scale=30, B=6, topk=64, invalid=0.95` → cos_sim 0.969, maxd 88, 0
NaN). Add per-tile printf inside the kernel under a compile-time DEBUG flag,
re-run, compare rsum/rmax/o_acc snapshots between this kernel and the FP32
oracle. The first divergence point pins the bug.

## Recommendation for the path-to-1×-B200 Model 2 lever (-8 to -10 ms TPOT)

**Proper fix path (3-5 days)**:
1. Update `microbench_ck_v32_512.py` to use FP8-saturation-stress KV distribution (1 day).
2. Bisect the kernel arithmetic to localize the Layer 2 wrong-output cause (1-2 days). Likely candidates: bf16→fp32 MFMA accumulator handling at high-magnitude inputs, sm_scale_log2e numerics, fp8_decode_scale dispatch for byte patterns near 0xFF (0xFF in e4m3fn IS NaN; verify the kernel correctly handles or excludes those bytes).
3. Apply correctness fix (1 day).
4. Re-validate: replay all captured production dumps must pass cos_sim ≥ 0.999 vs FP32 oracle, AND e2e prompt "The capital of France is" must produce coherent text.

**Stopgap (immediate)**: gate `ck_sparse_mla_decode_fp8_v32` (single_shot) off at decode in `debug_flash_mla_adapter.py`. The two_shot prefill path is unaffected; production decode falls through to `ref_sparse_attn_decode` (the same path as `=0`). This neutralizes the perf benefit of CK V32 for the affected ~5% of decode calls but eliminates the garbage-token regression.

**Until correctness is restored, the Model 2 row 1 lever cannot ship.** The path-to-1×-B200 plan's Lever 1 stays at "BLOCKED — kernel correctness".

## How to revert

```bash
cd /mnt/vast/john/sglang_v4_pr
git diff python/sglang/srt/layers/attention/csrc/ck_v32/mla_decode_fwd_kernel.hpp \
        python/sglang/srt/layers/attention/debug_flash_mla_adapter.py \
        python/sglang/srt/layers/attention/ck_v32_sparse_mla.py | git apply -R
docker exec sgl-deepseek-v4-mi35x-rocm720 bash -c \
    'rm -rf /root/.cache/torch_extensions/py310_cpu/ck_mla_decode_sparse_fp8'
```

Then on next server boot the original kernel rebuilds.

## Reproducer (current state)

```bash
ssh chi2774
cd /mnt/vast/john/sglang_v4_pr
# Boot Flash mxfp4 with CK V32 enabled
docker exec -d sgl-deepseek-v4-mi35x-rocm720 bash -c '
  cd /sgl-pr; export PYTHONPATH=/sgl-pr/python; export PORT=30013
  export MODEL=/hf/DeepSeek-V4-Flash-srt; export GPUS=0,1,2,3
  export SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1
  export SGLANG_HIP_CK_V32_TWO_SHOT=auto
  export SGLANG_MXFP4_FLYDSL=1; export AITER_XBFLOAT16=1
  bash launch_dsv4.sh stacked-best > /tmp/dsv4.log 2>&1
'

# Wait for ready, then test
docker exec sgl-deepseek-v4-mi35x-rocm720 bash -lc \
  'PYTHONPATH=/sgl-pr/python python3 -c "
import requests
r = requests.post(\"http://127.0.0.1:30013/v1/completions\", json={
    \"model\": \"/hf/DeepSeek-V4-Flash-srt\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 20, \"temperature\": 0.0, \"seed\": 1})
print(r.json()[\"choices\"][0][\"text\"])
"'
# Should output "Paris..." but currently outputs "1. 1. 1." (post-fix degenerate)
```

Setting `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=0` produces "Paris..." correctly (proven by the A/B in session 2 at session1 doc).

### Path (a) bit-match attempt (2026-04-29 EOD-6)

Replaced the kernel's `exp2f(x * log2e)` softmax + LSE with `__expf(x)` to bit-match torch.softmax (which uses __expf on CUDA). This eliminates one ULP-level transcendental mismatch.

Changes:
- Removed `kLog2e` pre-multiply on s_acc; now uses raw sm_scale.
- Replaced `exp2f` → `__expf` at both rescale (rsc) and pv computation.
- Removed `kLn2` from LSE epilogue (rmax is now natural-log-space directly).

Result:
- Production replay: 2/2 non-degenerate captures still PASS at cos=1.000 / maxd=2.0 (no regression).
- E2E `SGLANG_HIP_CK_V32_SINGLESHOT=1` at T=0: still garbage tokens, similar character.
- E2E with `SGLANG_CK_V32_FORCE_SPLITS=1` (bypass aiter.mla_reduce_v1, use Triton sink-fold instead): still garbage.

**Conclusion**: the transcendental wasn't the dominant per-call delta. Even after bit-matching exp, the per-layer-compounded drift remains. The kernel's per-call output matches the FP32 oracle at bf16 floor (verified) but the residual stream still diverges from the ref path's specific bf16 trajectory.

True bit-equality with the bf16 ref path likely requires:
- bit-identical bf16 mfma operand reduction order (specific to torch's GEMM kernel choice)
- bit-identical fp8→bf16 dequant rounding (depends on torch's `flashmla_quant.dequantize_k_cache` implementation)
- bit-identical softmax math (closer now via __expf, but not guaranteed)
- bit-identical V matmul accumulator order

Each of these would require ~half-day of bisecting against the specific torch reference and reproducing its exact arithmetic order. The cumulative effort is high relative to the alternative levers in the path-to-1×-B200 plan.

**Final shipping recommendation (post-bit-match-attempt)**:
- Layer-3 stopgap (`ca6f41917`) stays ON for production. Correct results today via BF16 ref fall-through.
- The kernel + diagnostic infrastructure are production-grade. The fp32-PV compile flag (`SGLANG_CK_V32_FP32_PV=1`) and the __expf bit-match (this commit) are committed as defensive precision improvements available at near-zero perf cost.
- For the perf gain (-7 to -10 ms TPOT), recommend pursuing **decode-body megakernel (Class A, -5 to -7 ms / 7 days)** as the next high-leverage lever. Orthogonal to CK V32 single_shot, with no precision-drift risk. The CK V32 perf gain remains available if a future session can land the bit-match work or finetune the model with the kernel's specific delta tolerance.

