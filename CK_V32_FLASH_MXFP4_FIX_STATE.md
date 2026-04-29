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

3. **Layer 3 (NOT FIXED)**: e2e Flash mxfp4 still produces incoherent text under `=1` despite Layers 1 and 2 fixed. Production-tensor replay shows cos_sim ≈ -0.06 (direction mismatch, NOT scaling) on B=6 and B=856 captures with mostly-invalid index distributions. Microbench at the same shapes shows cos_sim = 1.000, so either (a) the synthetic data doesn't capture a production-specific correlation between q-heads + indexer-selected KV pages, or (b) the replay script's oracle treats the production tensors differently than the kernel does (e.g. dtype roundtrip in torch.save/load).

   **Next-step bisect for Layer 3**: instrument the wrapper to compare kernel output vs ref_sparse_attn_decode output on the SAME production tensor (no torch.save/load detour). If they agree, the replay's oracle is bugged. If they disagree, instrument layer-by-layer in production to find the divergence point.

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
