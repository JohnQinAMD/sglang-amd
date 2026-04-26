# Running DeepSeek V4 models on AMD MI355X

Single launcher per model. All assume:

- Container: `rocm/sgl-dev:v0.5.8-rocm700-mi35x-20260129`
- PR tree: `/mnt/vast/john/sglang_v4_pr` (branch `rocm-deepseek-v4`)
- HF root: `/mnt/vast/john/huggingface`
- JIT cache: `/mnt/vast/john/sglang_v4_pr_jitcache`
- Patched sibling checkpoints (`-srt` suffix) hold the symlinks + `model_type=deepseek_v3` config patch

## Quick reference

| Model | Params | Quant scheme | TP / EP | Launcher | Per-GPU best |
|---|---:|---|---:|---|---:|
| Model | Params | Quant scheme | TP / EP | Launcher | Per-GPU best (measured) |
|---|---:|---|---:|---|---:|
| DeepSeek-V4-Flash-Base | ~285 B | FP8 attn + FP8 experts | TP=4 | [`launch_dsv4.sh`](launch_dsv4.sh) | see RUN_ON_AMD_MI355X.md |
| DeepSeek-V4-Flash (mxfp4) | ~285 B | FP8 attn + mxfp4 experts | TP=4 | [`launch_dsv4.sh`](launch_dsv4.sh) + 2 env knobs | 1.00 tok/s/GPU (c=1 OSL=1024) † |
| DeepSeek-V4-Pro-Base | ~671 B | FP8 attn + FP8 experts | TP=8 | [`launch_dsv4_pro_base.sh`](launch_dsv4_pro_base.sh) | **6.51 tok/s/GPU** (c=8 OSL=1024) |
| DeepSeek-V4-Pro (mxfp4) | ~671 B | FP8 attn + mxfp4 experts | TP=8 EP=8 | [`launch_dsv4_pro_mxfp4.sh`](launch_dsv4_pro_mxfp4.sh) | 4.45 tok/s/GPU (c=8 OSL=1024) |

† Flash mxfp4 c=1 only. c≥4 prefill batches (M ≥ 2048 tokens to MoE) hit a pre-existing HIP IMA in `compress_state.__getitem__` (compressor metadata path) — orthogonal to mxfp4. See "Flash mxfp4" section below.

Recommended production targets:
- 4-GPU node: **Flash-Base** (fastest, smallest footprint, no compressor IMA)
- 8-GPU node: **Pro-Base** (Pro-mxfp4 is functional but slower; use Pro-Base unless you specifically need the mxfp4 ckpt)

## Common docker run prefix

All examples below assume this docker invocation. Set `GPUS` to the device list you want.

```bash
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
docker run -d --name sglang \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --network=host --shm-size 64g \
  --security-opt seccomp=unconfined \
  --group-add video --group-add render --cap-add SYS_PTRACE \
  -v /mnt/vast/john/huggingface:/hf \
  -v /mnt/vast/john/sglang_v4_pr:/sgl-pr \
  -v /mnt/vast/john/sglang_v4_pr_jitcache:/sgl-workspace/aiter/aiter/jit \
  -e CUDA_VISIBLE_DEVICES="$GPUS" \
  rocm/sgl-dev:v0.5.8-rocm700-mi35x-20260129 \
  bash /sgl-pr/<launcher>.sh
```

## DeepSeek-V4-Flash-Base (FP8 + FP8)

Smallest, fastest, default. 4-GPU node. See [RUN_ON_AMD_MI355X.md](RUN_ON_AMD_MI355X.md) for the full bring-up (3 source patches, sibling checkpoint, JIT cache seeding).

```bash
GPUS=0,1,2,3 ... bash /sgl-pr/launch_dsv4.sh stacked-best
```

Greedy probe:

```bash
curl -s http://127.0.0.1:30010/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V4-Flash-Base","prompt":"The capital of France is","max_tokens":15,"temperature":0}'
# → " Paris. The capital of Germany is Berlin..."
```

## DeepSeek-V4-Flash (mxfp4)

Tested 2026-04-26: the default `Mxfp4MoEMethod` BF16-upcast path on this PR tree produces **garbage tokens** (` No! The? The? The?`) on Flash mxfp4. The fix is the same `SGLANG_MXFP4_AITER=1` packed-runtime path used for Pro mxfp4 — Flash's per-rank shapes (E=256, ipp=512, K=16) satisfy the cktile a16w4 shuffle constraints at TP=4 with no EP needed.

Required env:
- `SGLANG_MXFP4_AITER=1` — skip BF16 upcast, use aiter cktile a16w4 path with weight + scale shuffle and Swiglu activation. See [PRO_MXFP4_BRINGUP.md](PRO_MXFP4_BRINGUP.md) for the underlying mechanism.
- `SGLANG_OPT_USE_OLD_COMPRESSOR=false` — the old compressor's `compress_extend_old` path has a known HIP IMA at `kv_and_score_states[req_pool_indices[i]]` (line 1070 of `models/deepseek_v4.py`).
- `MODEL=/hf/DeepSeek-V4-Flash-srt` — `launch_dsv4.sh` accepts a `MODEL` env override (defaults to `/model` for backwards compat).

Mount the parent HF dir so the relative symlinks in the `-srt` sibling resolve:

```bash
docker run -d --name sglang \
  --device=/dev/kfd --device=/dev/dri --ipc=host --network=host --shm-size 64g \
  --security-opt seccomp=unconfined --group-add video --group-add render --cap-add SYS_PTRACE \
  -v /mnt/vast/john/huggingface:/hf \
  -v /mnt/vast/john/sglang_v4_pr:/sgl-pr \
  -v /mnt/vast/john/sglang_v4_pr_jitcache:/sgl-workspace/aiter/aiter/jit \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e MODEL=/hf/DeepSeek-V4-Flash-srt \
  -e SGLANG_MXFP4_AITER=1 -e SGLANG_OPT_USE_OLD_COMPRESSOR=false \
  rocm/sgl-dev:v0.5.8-rocm700-mi35x-20260129 \
  bash /sgl-pr/launch_dsv4.sh
```

Greedy probe:

```bash
curl -s http://127.0.0.1:30010/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V4-Flash-srt","prompt":"Q: What is the capital of France? A:","max_tokens":15,"temperature":0}'
# → " Paris. Q: What is the capital of Germany? A: Berlin."
```

Bench (c=1 OSL=1024, 4 prompts, greedy):

| Metric | Value |
|---|---:|
| Output throughput | 4.02 tok/s |
| Per-GPU (TP=4) | **1.00 tok/s/GPU** |
| TPOT | 247.6 ms |
| TTFT | 1208 ms |

**Known limitation**: c≥4 (or any prefill batch with M ≥ 2048 tokens reaching MoE) crashes with HIP IMA at `compress_state.py:32` (`KVAndScoreOld.__getitem__`). The IMA fires inside the compressor's metadata gather, not the MoE — a separate, pre-existing PR-tree issue. Until that's fixed, run Flash mxfp4 at c=1 only or use Flash-Base for higher concurrency.

## DeepSeek-V4-Pro-Base (FP8 + FP8)

8-GPU node. Loads ~140 GB / rank. Uses the in-tree Triton MoE (no aiter dispatch) plus a CK MoE override for prefill. Best measured: **6.51 tok/s/GPU at c=8 OSL=1024, 11.26 at c=16 OSL=1024**.

```bash
docker run ... bash /sgl-pr/launch_dsv4_pro_base.sh
```

Greedy probe:

```bash
curl -s http://127.0.0.1:30010/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V4-Pro-Base-srt","prompt":"Q: What is the capital of France? A:","max_tokens":15,"temperature":0}'
# → " Paris\n..."
```

Bench:

```bash
docker exec sglang python3 -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 30010 \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --random-range-ratio 1.0 --num-prompts 32 --max-concurrency 8 \
  --warmup-requests 0 --disable-tqdm
```

Pro-Base uses the same source patches as Flash-Base plus three Pro-specific ones (wo_a fp8 dequant for the larger 16384×4096 shape, `c4_sparse_topk in (512, 1024)` relaxation, generalized Triton TOPK). The launcher sets `SGLANG_FP8_PAGED_MQA_LOGITS_AITER=1` (R3 aiter MQA) and `SGLANG_FORCE_TRITON_MOE_FP8=0` (A1 aiter CK MoE for prefill).

Open levers (not yet shipped): higher concurrency (c=24/c=32), `--enable-piecewise-cuda-graph`, real flash_mla backend instead of `SGLANG_HACK_FLASHMLA_BACKEND=torch`. See `PRO_BASE_PERF_RESULTS.md` in the dynamo tree for the marathon comparison.

## DeepSeek-V4-Pro (mxfp4)

8-GPU node, **EP=8** required. Loads ~97 GB / rank packed. Uses `aiter.fused_moe` cktile a16w4 path with weight + scale shuffle. Best measured: **4.45 tok/s/GPU at c=8 OSL=1024**.

```bash
docker run ... bash /sgl-pr/launch_dsv4_pro_mxfp4.sh
```

Greedy probe:

```bash
curl -s http://127.0.0.1:30010/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V4-Pro-srt","prompt":"Q: What is the capital of France? A:","max_tokens":15,"temperature":0}'
# → " Paris\n..."
```

See [PRO_MXFP4_BRINGUP.md](PRO_MXFP4_BRINGUP.md) for the full bring-up walk-through, microbench results, and code-change rationale. The packed-runtime path is gated on `SGLANG_MXFP4_AITER=1` — without it, the loader OOMs trying to upcast 386 GB / rank of mxfp4 experts to BF16.

Why slower than Pro-Base: the cktile a16w4 kernel has no tuned config in `tuned_fmoe.csv` for Pro shapes (E=48, hidden=7168, ipp=3072) and falls back to a default heuristic. Tuning the cktile config is a path to close the gap, not pursued in this session.

## Sibling checkpoint setup

Each `-srt` directory is a sibling of the original HF checkpoint with relative symlinks to all files (assets, weights, tokenizer, etc.) and a patched `config.json`. The patches that matter for sglang:

- `model_type` → `deepseek_v3` (sglang's loader recognizes V3, V4 reuses)
- `architectures` → `["DeepseekV4ForCausalLM"]`
- `tokenizer_config.json` `tokenizer_class` → `PreTrainedTokenizerFast` (Pro variants only)
- For mxfp4 variants only: `quantization_config.quant_method` → `mxfp4`

Build a sibling dir from scratch:

```bash
mkdir -p /mnt/vast/john/huggingface/<MODEL>-srt
cd /mnt/vast/john/huggingface/<MODEL>-srt
for f in ../<MODEL>/*; do ln -sf "$f" .; done
rm config.json
# Edit and write the patched config.json explicitly (don't ln config.json).
```

## Verifying the build

```bash
docker exec sglang python3 -c "
import sglang
print('sglang', sglang.__version__)
import aiter
print('aiter', aiter.__file__)
"
# Expected:
#   sglang 0.5.x
#   aiter /sgl-workspace/aiter/aiter/__init__.py
```

If the launcher hangs in JIT compile on first run (5-15 min for aiter modules on a cold cache), let it finish — subsequent launches reuse `/mnt/vast/john/sglang_v4_pr_jitcache`. Restarting mid-compile leaves stale `.so.tmp` files; clean with `find $JIT_DIR -name '*.tmp' -delete`.

## Tear down

```bash
docker rm -f sglang
# Sibling dirs (~2 KB) and JIT cache (5-15 GB) are safe to keep between runs.
```
