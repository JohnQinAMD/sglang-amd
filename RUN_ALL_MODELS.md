# Running DeepSeek V4 models on AMD MI355X

Single launcher per model. All assume:

- Container: `rocm/sgl-dev:v0.5.8-rocm700-mi35x-20260129`
- PR tree: `/mnt/vast/john/sglang_v4_pr` (branch `rocm-deepseek-v4`)
- HF root: `/mnt/vast/john/huggingface`
- JIT cache: `/mnt/vast/john/sglang_v4_pr_jitcache`
- Patched sibling checkpoints (`-srt` suffix) hold the symlinks + `model_type=deepseek_v3` config patch

## Quick reference

| Model | Params | Quant scheme | TP / EP | Launcher | Per-GPU best (measured) |
|---|---:|---|---:|---|---:|
| DeepSeek-V4-Flash-Base | ~285 B | FP8 attn + FP8 experts | TP=4 | [`launch_dsv4.sh`](launch_dsv4.sh) | 2.15 tok/s/GPU (c=1 OSL=1024) |
| DeepSeek-V4-Flash (mxfp4 cktile) | ~285 B | FP8 attn + mxfp4 experts | TP=4 | [`launch_dsv4.sh`](launch_dsv4.sh) + `SGLANG_MXFP4_AITER=1` | 1.00 tok/s/GPU (c=1 OSL=1024) |
| **DeepSeek-V4-Flash (mxfp4 FlyDSL)** | ~285 B | FP8 attn + mxfp4 experts | TP=4 | + `SGLANG_MXFP4_FLYDSL=1` (needs FlyDSL wheel) | **4.85 tok/s/GPU** (c=1 OSL=1024) |
| DeepSeek-V4-Pro-Base | ~671 B | FP8 attn + FP8 experts | TP=8 | [`launch_dsv4_pro_base.sh`](launch_dsv4_pro_base.sh) | **6.51 tok/s/GPU** (c=8 OSL=1024) |
| DeepSeek-V4-Pro (mxfp4) | ~671 B | FP8 attn + mxfp4 experts | TP=8 EP=8 | [`launch_dsv4_pro_mxfp4.sh`](launch_dsv4_pro_mxfp4.sh) | 4.45 tok/s/GPU (c=8 OSL=1024) |

c=1 OSL=1024 num_prompts=10 warmup=2, ISL=1024 random, greedy.
**Flash mxfp4 FlyDSL is currently the fastest per-GPU AMD path for the Flash family** — 2.27× faster than the FP8 baseline on the same hardware, because mxfp4's 4-bit weights halve the HBM read at decode time and FlyDSL is hand-tuned for gfx950.

Recommended production targets:
- 4-GPU node, low concurrency: **Flash mxfp4 FlyDSL** (fastest path, beats FP8 at c=1)
- 4-GPU node, high concurrency: **Flash-Base FP8** (mxfp4 c≥4 prefill hits a pre-existing compressor IMA, orthogonal to mxfp4)
- 8-GPU node: **Pro-Base** (Pro-mxfp4 is functional but slower; use Pro-Base unless you specifically need the mxfp4 ckpt)

## Sparse MLA decode kernel — IMPORTANT for accuracy

The CK V32 sparse MLA decode kernel (enabled via `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1`)
had a silent **2× scaling bug** before commit `eb373f796`:

- The kernel was designed assuming KV cache is stored as `torch.float8_e4m3fnuz`
  (MI300X / gfx942 default, exponent bias=8).
- On MI355X (gfx950) sglang stores KV as `torch.float8_e4m3fn` (OCP standard, bias=7).
- The hardcoded `kFnuzBiasFix = 0.5f` halved every KV value → attention output was
  silently 0.5× the reference (microbench-confirmed: ratio=0.5000 at both
  qk_head_dim=512 and 576).
- Cosine similarity stayed high (0.9997 — softmax is invariant under uniform
  scaling) but the absolute scale shift broke residual addition + downstream
  RMSnorm.
- **Symptom**: Flash mxfp4 + CK V32 returned "London" instead of "Paris".
  Pro V32 produced sensible-but-not-quite-right text (less sensitive).

**Fix**: the kernel now takes a runtime `fp8_decode_scale` parameter, and the
Python wrapper picks the right value from `k_cache.dtype`:
- `torch.float8_e4m3fn` → 1.0  (MI355X / OCP standard)
- `torch.float8_e4m3fnuz` → 0.5 (MI300X / legacy)
- `torch.uint8` view → 1.0 (assume MI355X-style fn storage)

After the fix:
- CK V32 numerically matches reference (max_diff 1.34e-3 vs 1.82e-2 before).
- **Pro V32 attention is now correct** — was running with halved output.
- **Flash mxfp4 + CK V32 is now selectable** (was unusable due to the bug).

CK V32 is **roughly at parity** with the existing Triton sparse_attn_decode at c=1
(both ~50 ms TPOT). Use whichever your model needs:
- Pro V32 (qk_head_dim=576): use CK V32 (`SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1`)
- Flash 2604 (qk_head_dim=512): either CK V32 OR Triton sparse — pick by env

Microbench: [`microbench_ck_v32_512.py`](microbench_ck_v32_512.py) — A/B test for
fn vs fnuz storage at D=512 and D=576. Also [`_fp8_decode_probe2.py`](_fp8_decode_probe2.py)
proves the gfx950 HW intrinsic decodes with fn semantics (0/256 byte mismatches
vs 252/256 for fnuz).

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

### FlyDSL (much faster — 4.86× over cktile, 2.27× over FP8)

`SGLANG_MXFP4_FLYDSL=1` replaces the cktile `aiter.fused_moe` wrapper chain with a direct call to `flydsl_moe_stage1` + `flydsl_moe_stage2` (gfx950-tuned FP4 GEMM kernels). Bypasses the `fused_moe_2stages` Python overhead entirely. Fused-quant in stage1 outputs FP4 + sorted scale ready for stage2, so no intermediate dequant.

Setup (one-time):

```bash
# 1. Install FlyDSL wheel (must be exactly 0.1.3.1 — version check is strict)
docker exec sglang bash -c '
  curl -fSL -o /tmp/flydsl-0.1.3.1+20260418.68f5725-cp310-cp310-manylinux_2_35_x86_64.whl \
    https://rocm.frameworks-nightlies.amd.com/whl/gfx942-gfx950/flydsl-0.1.3.1%2B20260418.68f5725-cp310-cp310-manylinux_2_35_x86_64.whl
  pip install /tmp/flydsl-0.1.3.1+20260418.68f5725-cp310-cp310-manylinux_2_35_x86_64.whl
'

# 2. Drop in aiter.ops.flydsl module from the rocm-dynamo aiter-amd workspace
docker exec sglang cp -r /flydsl_src /sgl-workspace/aiter/aiter/ops/flydsl
# (mount /mnt/vast/john/rocm-dynamo/aiter-amd/aiter/ops/flydsl as /flydsl_src in docker run)

# 3. Verify
docker exec sglang python3 -c "
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2
print('available:', is_flydsl_available())
"
```

Run with `SGLANG_MXFP4_FLYDSL=1`:

```bash
docker run -d --name sglang \
  --device=/dev/kfd --device=/dev/dri --ipc=host --network=host --shm-size 64g \
  --security-opt seccomp=unconfined --group-add video --group-add render --cap-add SYS_PTRACE \
  -v /mnt/vast/john/huggingface:/hf \
  -v /mnt/vast/john/sglang_v4_pr:/sgl-pr \
  -v /mnt/vast/john/sglang_v4_pr_jitcache:/sgl-workspace/aiter/aiter/jit \
  -v /mnt/vast/john/rocm-dynamo/aiter-amd/aiter/ops/flydsl:/flydsl_src \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e MODEL=/hf/DeepSeek-V4-Flash-srt \
  -e SGLANG_MXFP4_FLYDSL=1 -e SGLANG_OPT_USE_OLD_COMPRESSOR=false \
  rocm/sgl-dev:v0.5.8-rocm700-mi35x-20260129 \
  bash /flydsl_setup/launch_with_flydsl.sh   # see flydsl_setup/launch_with_flydsl.sh in this tree
```

Greedy probe:

```bash
# → " Paris. Q: What is the capital of Germany? A: Berlin."
```

Bench (c=1 OSL=1024 num=10 warmup=2, greedy):

| Metric | cktile (old) | **FlyDSL (new)** | improvement |
|---|---:|---:|---:|
| Output throughput | 3.95 tok/s | **19.40 tok/s** | **4.91×** |
| Per-GPU (TP=4) | 1.00 | **4.85 tok/s/GPU** | 4.85× |
| TPOT | 250.7 ms | **50.94 ms** | 4.92× |
| TTFT | 712 ms | 427 ms | 1.67× |

**FlyDSL Flash mxfp4 (50.94 ms TPOT) is 2.27× faster than Flash-Base FP8 (115.4 ms TPOT)** on identical hardware — finally realizes the bandwidth advantage of 4-bit weights at c=1 decode.

Implementation notes (see [`mxfp4.py:_use_flydsl_mxfp4`](python/sglang/srt/layers/quantization/mxfp4.py)):
- Pre-shuffle weights at load: `shuffle_weight((16,16))` for w13, `shuffle_weight_a16w4(_,16,False)` for w2; `e8m0_shuffle` for w13 scale, `shuffle_scale_a16w4` for w2 scale.
- A-side mxfp4 quant per call: `aiter.ops.triton.quant.dynamic_mxfp4_quant` (capture-safe; the `aiter.get_torch_quant` `.f32_to_e8m0` path uses an indexed assign that fails under cuda graph capture).
- Microbench-validated cos=0.96 vs BF16 reference at M=1.

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
