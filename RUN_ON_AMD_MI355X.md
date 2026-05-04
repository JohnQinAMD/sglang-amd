# Running SGLang PR #23608 (DeepSeek V4) on AMD MI355X

Self-contained recipe to bring up DeepSeek-V4-Flash-Base (FP8, ~285B params) on a single MI355X node using SGL PR #23608. Tested on chi2761 (8× MI355X gfx950, 192 GB/GPU).

PR: https://github.com/sgl-project/sglang/pull/23608 (status: open, NOT merged as of 2026-04-24).

## TL;DR — current state

- **Correctness FIXED** locally — see "Step 3c" below. With the wo_a FP8 dequant patch, "The capital of France is" → " Paris. The capital of Germany is Berlin. The capital of Italy is Rome..." (Base model, but coherent).
- All attention is the **pytorch reference path** (`ref_sparse_attn_decode`) — no real kernel. Performance is roughly 1/15 to 1/30 of what the same hardware will reach once HIP kernels exist.
- A single bench (`isl=8192 osl=1024 c=16`) yielded **~48 tok/s aggregate output, 342 s mean E2E** before any kernel work — that's the lower-bound floor.

## Hardware / paths assumed

| Thing | Value |
|---|---|
| Node | chi2761 |
| GPUs | 8× MI355X (gfx950) |
| Models on disk | `/mnt/vast/john/huggingface/DeepSeek-V4-Flash{,-Base}` |
| PR clone (host) | `/mnt/vast/john/sglang_v4_pr` (this dir) |
| JIT cache (host) | `/mnt/vast/john/sglang_v4_pr_jitcache` |
| Patched checkpoint | `/mnt/vast/john/huggingface/DeepSeek-V4-Flash-Base-srt` |

## Choose the right checkpoint

| Variant | Total size | Routed experts | Use when |
|---|---|---|---|
| `DeepSeek-V4-Flash-Base` | 295 GB | **FP8** e4m3, block 128×128, fp32 scales | This recipe (PR's tested path) |
| `DeepSeek-V4-Flash` | 160 GB | **MXFP4** int8-packed, ue8m0 scales | NOT supported by this recipe — extra weight-loader bug |

The two checkpoints have **identical `config.json`** — there's no auto-detect. You must pick correctly. If you load Flash with `SGLANG_DSV4_FP4_EXPERTS=false` (or Flash-Base with `=true`), weight loading dies with a (4096) vs (2048) shape mismatch.

## Step 1 — Pull the docker image

```bash
docker pull docker.io/rocm/sgl-dev:v0.5.8-rocm700-mi35x-20260129
```

This is the PR's stated base image. It already contains ROCm 7.0.0, AITER, sglang main (pre-V4). The PR source gets overlaid via PYTHONPATH below.

(There's also `lmsysorg/sglang:deepseek-v4-mi350` which has the PR baked in. It works for the model code but bypasses the `BUILD_AITER_ALL` fix path — prefer the base image + PYTHONPATH overlay so you can edit PR source.)

## Step 2 — Clone the PR branch

```bash
git clone --depth 1 -b amd/deepseek_v4 \
  https://github.com/AgainstEntropy/sglang.git /mnt/vast/john/sglang_v4_pr
# Verify HEAD is `a0b59e3 Add AMD support for DeepSeek V4`
```

## Step 3 — Apply 3 source patches in the PR clone

These are mandatory; without them the server can't boot.

### 3c. **Fix wo_a FP8 dequant** (THE accuracy fix — without this, output is gibberish)

The PR creates `wo_a` as a bf16 ColumnParallelLinear when `SGLANG_OPT_FP8_WO_A_GEMM=False` (default), but the HF DeepSeek-V4-Flash{,-Base} checkpoints store wo_a as `fp8_e4m3fn` weight + `fp32` block scale `[64, 32]`. The PR's existing dequant codepath only fires for `SGLANG_DSV4_MODE="2604" AND SGLANG_DSV4_FP4_EXPERTS=true` and assumes `fp8_e8m0fnu` scale dtype — neither matches the HF release. Without the fix, fp8 bytes get copied into bf16 params unscaled → wo_a is off by 1-10000× per block → gibberish.

Two-line patch in [python/sglang/srt/models/deepseek_v4.py](python/sglang/srt/models/deepseek_v4.py):

1. Relax `_dequant_fp8` scale dtype check (around line 2763) to accept `float32` in addition to `fp8_e8m0fnu`.
2. Replace the `SGLANG_DSV4_MODE=="2604"` branch (around line 2417) with a layout-detecting helper that auto-dequants when wo_a is FP8 in the checkpoint, drops stale scales otherwise.

This is already applied to `/mnt/vast/john/sglang_v4_pr` — diff with `git diff` to view. Should be filed upstream.

### 3a. Stub `bench.py` (PR ships an `__init__.py` that imports a file that doesn't exist)

```bash
cat > /mnt/vast/john/sglang_v4_pr/python/sglang/srt/flashmla_tests/kernelkit/bench.py <<'PY'
def bench_by_cuda_events(*args, **kwargs):
    raise NotImplementedError("bench stub - benchmarking path not used in production")
def bench_kineto(*args, **kwargs):
    raise NotImplementedError("bench stub - benchmarking path not used in production")
PY
```

Without this: `ImportError: cannot import name 'bench' from partially initialized module
'sglang.srt.flashmla_tests.kernelkit'` when the torch attention fallback is invoked → schedulers go zombie, server hangs at "fired up and ready to roll" but never serves.

### 3b. (Optional) Extend the model-type fallback to also match `deepseek_v4`

Edit [hf_transformers_utils.py:281](python/sglang/srt/utils/hf_transformers_utils.py#L281):

```python
# was:  if "deepseek_ref" in str(e):
       if "deepseek_ref" in str(e) or "deepseek_v4" in str(e):
```

We didn't apply this because we use the patched-checkpoint approach in step 4 instead. Either works.

## Step 4 — Build a patched sibling checkpoint

The HF checkpoint has `model_type: "deepseek_v4"` which HF's AutoConfig doesn't know. The PR's fallback mechanism is buggy (writes config to a file path then calls `from_pretrained` on it). Workaround: build a sibling dir with relative symlinks + a patched `config.json` and `tokenizer_config.json`.

```bash
SRC=DeepSeek-V4-Flash-Base
DST=/mnt/vast/john/huggingface/DeepSeek-V4-Flash-Base-srt
rm -rf "$DST" && mkdir -p "$DST"
cd "$DST"
for f in ../$SRC/*; do ln -s "$f" .; done   # RELATIVE symlinks (must resolve inside container)
rm -f config.json tokenizer_config.json
python3 -c "
import json
c = json.load(open('../$SRC/config.json'))
c['model_type'] = 'deepseek_v3'                  # so HF AutoConfig recognises it
c['architectures'] = ['DeepseekV4ForCausalLM']   # so sglang dispatches to V4 model class
json.dump(c, open('config.json','w'), indent=2)

t = json.load(open('../$SRC/tokenizer_config.json'))
t['tokenizer_class'] = 'PreTrainedTokenizerFast'  # bypass LlamaTokenizer slow-path SPM crash
json.dump(t, open('tokenizer_config.json','w'), indent=2)
"
```

Two failure modes if you skip this:
- Without the config patch → `KeyError: 'deepseek_v4'` from AutoConfig.
- Without the `tokenizer_class` patch → `TypeError: not a string` from sentencepiece (LlamaTokenizer's slow class wants a `tokenizer.model` SPM file the checkpoint doesn't ship).
- With absolute symlinks → `No such file or directory` inside container (host paths don't resolve in container).

## Step 5 — Seed the persistent JIT cache

aiter's first-run JIT compiles 1361 cpp variants for `module_rmsnorm`. At the default 8-way parallel that's 30+ minutes. We mount a persistent cache on vast and crank `MAX_JOBS=128`. Cold start: ~10 min. Warm start: <1 min.

```bash
mkdir -p /mnt/vast/john/sglang_v4_pr_jitcache
docker run --rm -v /mnt/vast/john/sglang_v4_pr_jitcache:/cache \
  rocm/sgl-dev:v0.5.8-rocm700-mi35x-20260129 \
  bash -c 'cp -a /sgl-workspace/aiter/aiter/jit/. /cache/ && rm -rf /cache/build/lock_*'
```

The cache will fill up (~5-15 GB) the first time you actually run the model.

## Step 6 — Launch the server

All env knobs and CLI flags are baked into the unified launcher
[`launch_dsv4.sh`](launch_dsv4.sh) at the repo root. Pick one preset; env
overrides win against preset defaults.

| Preset | When to use |
|---|---|
| **`stacked-best`** (default) | **Recommended for production.** Multi-bs cuda graph + Top-K JIT + TileLang mHC + indexer cap. ISL=256/OSL=16 c=8 → **27.95 tok/s** on MI355X TP=4. |
| `tier0` | Sanity-check baseline (no patches, no cuda graph). Slower but most-conservative path. |
| `testP` / `testT` / `testV` | Per-feature isolation (bs=1 graph, multi-bs graph, TopK JIT only). |
| `stacked-aiter-moe` | Same as stacked-best but routes MoE through aiter CK kernels. |
| `stacked-widebs` | stacked-best + cuda-graph-bs includes 16 (lift at c≥16). |

```bash
docker run -d --name sglang_v4_flash \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --network=host \
  --shm-size 64g \
  --security-opt seccomp=unconfined \
  --group-add video --group-add render \
  --cap-add SYS_PTRACE \
  -v /mnt/vast/john/huggingface:/models:ro \
  -v /mnt/vast/john/sglang_v4_pr:/sgl-pr:ro \
  -v /mnt/vast/john/sglang_v4_pr_jitcache:/sgl-workspace/aiter/aiter/jit \
  rocm/sgl-dev:v0.5.8-rocm700-mi35x-20260129 \
  bash /sgl-pr/launch_dsv4.sh stacked-best
```

The launcher prints the resolved knobs at startup so it's obvious which
config actually fired:

```
===================================================================
Preset: stacked-best    Port: 30010    GPUs: 0,1,2,3
TOPK_TORCH=0  FORCE_TRITON_MOE=1  CAP=4096
MHC_PRE=1  MHC_POST=1
MULTI_STREAM=0  DISABLE_COMPILE=1
CUDA_GRAPH: max_bs=8 bs=[1 2 4 8]
===================================================================
```

### Env overrides

Any preset can be tweaked via env vars without editing the script:

```bash
# Alt port / GPU set
PORT=30011 GPUS=4,5,6,7 bash /sgl-pr/launch_dsv4.sh stacked-best

# A/B isolate one knob
SGLANG_FORCE_TRITON_MOE_FP8=0 bash /sgl-pr/launch_dsv4.sh stacked-best

# Run eager only (sanity)
DISABLE_CUDA_GRAPH=1 bash /sgl-pr/launch_dsv4.sh stacked-best
```

### `--skip-server-warmup` for first boot

`tier0` preset enables it; `stacked-*` presets do NOT (cuda graph capture
covers the warmup path safely). On a cold-cache first boot, the aiter JIT
pass takes 5-15 min — that's normal. The persistent
`/mnt/vast/john/sglang_v4_pr_jitcache` mount means subsequent boots skip it.

## Step 7 — Verify

Server logs should end with:

```
The server is fired up and ready to roll!
INFO: Uvicorn running on http://0.0.0.0:30010
```

(Health endpoint will return 503 until something exercises the detokenizer; this is expected with `--skip-server-warmup`.)

First /generate triggers the cold JIT path. With the persistent cache + MAX_JOBS=128, expect **~50 s for the first request, ~25 s for subsequent** (single request, 16 output tokens):

```bash
curl --max-time 600 http://localhost:30010/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 16, "temperature": 0}}'
```

You will get back valid JSON with **gibberish text** (e.g. Chinese characters + repeated tokens). This is a real PR bug, not a setup error. Your forward pass is working — the bug is in the V4 weight-load mapping or the compressed-attention numerics.

## Step 8 — Benchmark

```bash
docker exec sglang_v4_flash python3 -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 30010 \
  --dataset-name random \
  --random-input-len 8192 --random-output-len 1024 --random-range-ratio 1.0 \
  --num-prompts 32 --max-concurrency 16 \
  --warmup-requests 2 \
  --disable-tqdm
```

Reference numbers from chi2761 (drop `--disable-stream` to get TTFT/TPOT split):

| Metric | Value |
|---|---|
| Total throughput | 431 tok/s |
| Decode throughput | 47.9 tok/s |
| Prefill throughput | 383 tok/s |
| Mean E2E | 342 s/req |
| Duration (32 reqs) | 684 s |

These are the floor — every attention op runs in eager pytorch. Real kernel work moves them by 10-30×.

## Known issues we hit (file upstream if not already)

1. `_load_deepseek_temp_model` only matches `"deepseek_ref"` in error string; HF checkpoints carry `"deepseek_v4"`. Fallback never fires.
2. `_load_deepseek_temp_model` writes config to a file path then calls `AutoConfig.from_pretrained` on it — HF expects a directory.
3. `flashmla_tests/kernelkit/__init__.py` imports `bench` and `bench.{bench_by_cuda_events,bench_kineto}` but `bench.py` is missing from the PR (likely .gitignored during dev).
4. `assert intermediate_size_per_partition == 2048` in `fp8.py:749` fails for any `tp_size > 1` that splits the intermediate (only matters if you also enable FP4 experts).
5. `SGLANG_DSV4_FP4_EXPERTS` is a manual switch — PR doesn't auto-detect MXFP4 vs FP8 from `config.json`.
6. Internal warmup uses hardcoded 600s timeout; AITER's first-run JIT (especially `module_rmsnorm`'s 1361 variants at default 8-way parallel) can't fit. Workaround: `--skip-server-warmup`.
7. Tokenizer trap: setting `model_type=deepseek_v3` (the only way to get past AutoConfig) routes `AutoTokenizer` to `LlamaTokenizerFast`, which then unconditionally instantiates `LlamaTokenizer` (slow) → wants `tokenizer.model` SPM that DSv4 doesn't ship → `TypeError: not a string` from sentencepiece.
8. Output is gibberish — structural numerics bug (see TL;DR).
9. No MoE Triton tuning configs for AMD MI355X — only `device_name=AMD_Instinct_MI300X_VF` is shipped, AMD MI3xx falls back to default block sizes (logged as "Performance might be sub-optimal").
10. Apex fused RoPE prints "Aiter backend is selected for fused RoPE. This has lower precision." — `USE_ROCM_AITER_ROPE_BACKEND=0` disables it but does NOT change the gibberish output (so RoPE precision isn't the bug).

## What's actually running on AMD (kernel inventory)

The PR's `SGLANG_HACK_FLASHMLA_BACKEND=torch` and 5 other `*_torch=1` / `*_USE_FUSED*=false` env vars route the V4 attention path through pytorch reference implementations:

| Op | Implementation on AMD |
|---|---|
| Sparse attention compute (decode + prefill) | `ref_sparse_attn_decode` — **pytorch eager** `q@K.T → exp → @V`, no fused kernel |
| KV quant + RoPE pack | Triton (`quant_to_nope_fp8_rope_bf16_pack_triton`) |
| SWA prefill index prep | pytorch (TileLang variant disabled) |
| History compressor (c4/c128 cache writes) | pytorch (CUDA `.cuh` exists, no HIP port) |
| Indexer top-512 sparse selection | pytorch (`SGLANG_TOPK_TRANSFORM_512_TORCH=1`) |
| FP8 paged MQA logits (indexer scoring) | pytorch (`SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1`) |
| RMSNorm + FP8 quant | aiter CK-tile `module_rmsnorm` (real kernel) |

Optimization plan when correctness lands: replace the pytorch ref attention with a Triton sparse-decode kernel (5-8× alone), HIP-port the c4/c128 compressors (~2× more), then write a Triton FP8 paged MQA kernel for the indexer (~1.5-2× more). Re-enable CUDA graph after that for another 1.5-2×.

## Tear down

```bash
docker rm -f sglang_v4_flash
# Patched checkpoint dir is just symlinks (~2 KB), safe to leave
# JIT cache is ~5-15 GB on vast — keep it for next run
```
