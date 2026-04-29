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

## Two-layer bug structure

1. **Layer 1 (FIXED)**: kernel produced NaN for production captures with `kv.abs().max == 255.0` (FP8 saturation) and B≥6 or large prefill batches. Root cause: most likely an inf score from a max-reduce edge case turning subsequent `(s - max)` into NaN inside the online-softmax accumulator.
2. **Layer 2 (NOT fixed)**: even when the kernel produces finite output, the attention values are arithmetically wrong for FP8-saturated KV — possibly a per-tile range mismatch, wrong `fp8_decode_scale` for some byte patterns, or an MFMA bf16 truncation issue that the microbench's `randn() * 0.1` synthetic data does not exercise.

## Why the existing microbench misses both layers

`microbench_ck_v32_512.py` uses `randn() * 0.1` for KV → `kv.abs().max() ≈ 3.0`. Production decode KV reaches `kv.abs().max() == 255.0` (FP8 e4m3 saturation max).

Recommended microbench addition (Layer 2 bug should reproduce):
```python
# Replace
k_cache_bf16 = torch.randn(N_KV, SLOT, dtype=torch.bfloat16, device=device) * 0.1
# With
k_cache_bf16 = torch.randn(N_KV, SLOT, dtype=torch.bfloat16, device=device) * 80.0
k_cache_bf16 = torch.clamp(k_cache_bf16, -240, 240)  # near FP8 saturation
```

If this PASSES the cos_sim ≥ 0.999 check, the bug is data-distribution-specific in another way (e.g. correlations between q heads + indexer-selected KV pages that random data doesn't capture). If it FAILS, you have a localized microbench reproducer.

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
