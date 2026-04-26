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
| **DeepSeek-V4-Flash (mxfp4 FlyDSL + R5 stack)** | ~285 B | FP8 attn + mxfp4 experts | TP=4 | + `SGLANG_MXFP4_FLYDSL=1` (needs FlyDSL wheel) | **6.42 tok/s/GPU** (c=1 OSL=1024) |
| DeepSeek-V4-Pro-Base | ~671 B | FP8 attn + FP8 experts | TP=8 | [`launch_dsv4_pro_base.sh`](launch_dsv4_pro_base.sh) | **6.51 tok/s/GPU** (c=8 OSL=1024) |
| DeepSeek-V4-Pro (mxfp4) | ~671 B | FP8 attn + mxfp4 experts | TP=8 EP=8 | [`launch_dsv4_pro_mxfp4.sh`](launch_dsv4_pro_mxfp4.sh) | 4.45 tok/s/GPU (c=8 OSL=1024) |

c=1 OSL=1024 num_prompts=10 warmup=2, ISL=1024 random, greedy.
**Flash mxfp4 FlyDSL + R5 stack is currently the fastest per-GPU AMD path for the Flash family** — ~3× faster than the FP8 baseline on the same hardware (38.96 ms TPOT vs 115.4 ms), because mxfp4's 4-bit weights halve the HBM read at decode time, FlyDSL is hand-tuned for gfx950, and the R5 stack ports MI300's 5-patch dequant fusion (gather-first + Triton kernel + bf16 inner BMM + skip nan_to_num + fused index gather).

> ⚠️ **CK V32 sparse MLA decode is currently broken for end-to-end Flash mxfp4 inference** at production TOPK ≥ 256 with the multi-split reduce path. `microbench_ck_v32_512.py` reports `cos_sim 0.999998` because it uses TOPK=64 / num_splits=1, which doesn't exercise the broken path. **Do NOT set `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1` for Flash mxfp4** until the kernel is fixed. The launchers below leave it unset.

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
- CK V32 numerically matches reference at TOPK=64 / num_splits=1 (max_diff 1.34e-3 vs 1.82e-2 before).
- **Pro V32 attention is now correct** at small TOPK — was running with halved output.

> ⚠️ **Open regression (2026-04-26):** CK V32 still produces silent garbage on
> end-to-end Flash mxfp4 inference at production TOPK ≥ 256 + the multi-split
> reduce path. `microbench_ck_v32_512.py` (TOPK=64) does NOT catch this — its
> cos-sim is 0.999998. Bisected by toggling `SGLANG_HIP_SPARSE_MLA_DECODE_FP8`:
>
> | mode | greedy "The capital of France is" |
> |---|---|
> | `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1` (CK V32) | garbage tokens |
> | `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=0` (torch ref + R5) | ✓ " Paris.\nThe capital of France is Paris…" |
>
> Use **`SGLANG_HIP_SPARSE_MLA_DECODE_FP8=0`** (or unset, which is the
> default) for Flash mxfp4 until the kernel multi-split path is fixed.
> The R5 stack on the torch ref path delivers SOTA Flash mxfp4 perf
> (38.96 ms TPOT, see Flash mxfp4 section below). For Pro V32 the impact
> is unverified — use at your own risk and validate against an FP32 oracle
> at production shapes.

Recommended action items (ordered by ROI):
- Extend [`microbench_ck_v32_512.py`](microbench_ck_v32_512.py) to TOPK ∈ {256, 512, 1024} and explicitly cover the multi-split reduce path (`pick_num_splits` returns >1) — should reproduce the inference garbage in a few seconds instead of needing a 2-min server restart.
- Once reproducible, fix the multi-split reduce path or the wrapper's `pick_num_splits` heuristic.

Microbench artifacts: [`microbench_ck_v32_512.py`](microbench_ck_v32_512.py) and [`_fp8_decode_probe2.py`](_fp8_decode_probe2.py) (proves the gfx950 HW intrinsic decodes with fn semantics — 0/256 byte mismatches vs 252/256 for fnuz).

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

### FlyDSL + R5 stack (SOTA — 6.50× over cktile, ~3× over FP8)

Two independent components must be in the tree for SOTA Flash mxfp4:

1. **`SGLANG_MXFP4_FLYDSL=1`** replaces the cktile `aiter.fused_moe` wrapper chain with a direct call to `flydsl_moe_stage1` + `flydsl_moe_stage2` (gfx950-tuned FP4 GEMM kernels). Bypasses the `fused_moe_2stages` Python overhead entirely. Fused-quant in stage1 outputs FP4 + sorted scale ready for stage2, so no intermediate dequant.

2. **R5 attention stack + HIP routing** (default-on once the patch is in tree, no env knob needed beyond CK V32 OFF):
   - **P0-c gather-first FP8 dequant** — only dequant the topk rows actually used, not the whole `[num_blocks*block_size, 512]` table
   - **P1-a 14-tile dequant → 1 Triton kernel** in `flashmla_tests/quant.py` (`dequantize_k_cache_gather`)
   - **P1-b skip `nan_to_num`** — `quantize_k_cache` now uses `FP8_MAX = finfo(dtype).max` so the defensive scrub becomes a redundant launch
   - **P1-c fused index gather** — one launch reads packed FP8 + indices → bf16
   - **P1-e bf16 inner BMM** — drops `.float()` upcast, gfx950 v_mfma is faster on bf16
   - **HIP routing for compress / fused_norm_rope / hash_topk / topk_transform_512** — replaces JIT C++ kernels with Triton/torch versions tuned for gfx950 (`jit_kernel/{compress_torch, compress_c4_c128_torch, fused_norm_rope_triton, hash_topk_triton}.py`)

Setup (one-time, host side — wheel + matching aiter glue must coexist on disk):

```bash
# 1. Pre-stage the wheel + this tree's launcher into /mnt/vast/john/flydsl_setup
#    (already in place on chi2761; replicate on other hosts):
mkdir -p /mnt/vast/john/flydsl_setup
curl -fSL -o /mnt/vast/john/flydsl_setup/flydsl-0.1.3.1+20260418.68f5725-cp310-cp310-manylinux_2_35_x86_64.whl \
  https://rocm.frameworks-nightlies.amd.com/whl/gfx942-gfx950/flydsl-0.1.3.1%2B20260418.68f5725-cp310-cp310-manylinux_2_35_x86_64.whl
cp $REPO_ROOT/flydsl_setup/launch_with_flydsl.sh /mnt/vast/john/flydsl_setup/

# 2. Pin the matching aiter-amd checkout (its `_REQUIRED_FLYDSL_VERSION` MUST
#    equal the wheel version `0.1.3.1`, otherwise sglang crashes at first MoE
#    call with `ImportError: Unsupported flydsl version`).
#    Verify with:
grep _REQUIRED_FLYDSL_VERSION /mnt/vast/john/rocm-dynamo/aiter-amd/aiter/ops/flydsl/__init__.py
#    Expected: _REQUIRED_FLYDSL_VERSION = "0.1.3.1"
#    DO NOT mount /mnt/vast/john/aiter_workspace/...   (expects 0.1.1.dev409)
#    DO NOT mount /mnt/vast/john/tmp-sla-*/aiter-amd/  (expects 0.1.1+20260401)
```

The launch script (`flydsl_setup/launch_with_flydsl.sh`) handles wheel install +
aiter glue copy + version validation inside the container — no manual
`docker exec ... pip install` step needed.

Run with `SGLANG_MXFP4_FLYDSL=1`:

```bash
docker run -d --name sglang_flash \
  --device=/dev/kfd --device=/dev/dri --ipc=host --network=host --shm-size 64g \
  --security-opt seccomp=unconfined --group-add video --group-add render --cap-add SYS_PTRACE \
  -v /mnt/vast/john/huggingface:/hf \
  -v /mnt/vast/john/sglang_v4_pr:/sgl-pr \
  -v /mnt/vast/john/sglang_v4_pr_jitcache:/sgl-workspace/aiter/aiter/jit \
  -v /mnt/vast/john/flydsl_setup:/flydsl_setup \
  -v /mnt/vast/john/rocm-dynamo/aiter-amd/aiter/ops/flydsl:/flydsl_src \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e MODEL=/hf/DeepSeek-V4-Flash-srt -e PORT=30010 \
  -e SGLANG_MXFP4_FLYDSL=1 -e SGLANG_OPT_USE_OLD_COMPRESSOR=false \
  -e SGLANG_TRITON_SPARSE_DECODE=0 \
  -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  rocm/sgl-dev:v0.5.8-rocm700-mi35x-20260129 \
  bash /flydsl_setup/launch_with_flydsl.sh
```

Note: **`SGLANG_HIP_SPARSE_MLA_DECODE_FP8` is intentionally NOT set** — see the Sparse MLA section above for why CK V32 is currently broken on Flash mxfp4 e2e inference.

The launcher pre-flight checks (in `flydsl_setup/launch_with_flydsl.sh`):
- `/flydsl_setup` mount has the wheel (else: clear error)
- `/flydsl_src` mount exists (else: clear error)
- Installed wheel version == `_REQUIRED_FLYDSL_VERSION` in `/flydsl_src/__init__.py`
  (else: clear error before model load — saves the 1-2 min boot-then-crash cycle)

Greedy probe:

```bash
# → " Paris. Q: What is the capital of Germany? A: Berlin."
```

Bench (c=1 OSL=1024 num=10 warmup=2, greedy):

| Metric | cktile (old) | FlyDSL (initial) | **FlyDSL + R5 stack (current)** | improvement |
|---|---:|---:|---:|---:|
| Output throughput | 3.95 tok/s | 19.40 tok/s | **25.66 tok/s** | **6.50×** vs cktile |
| Per-GPU (TP=4) | 1.00 | 4.85 tok/s/GPU | **6.42 tok/s/GPU** | 6.42× |
| TPOT | 250.7 ms | 50.94 ms* | **38.96 ms** | 6.43× |

\*The 50.94 ms FlyDSL number was measured with `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1`
(CK V32 sparse MLA decode), but that path actually returned garbage tokens for
e2e inference (the microbench at `microbench_ck_v32_512.py` reports 0.999998
cos-sim vs FP32 oracle, but uses TOPK=64 / num_splits=1 — production decode
hits TOPK ≥ 256 and the multi-split reduce path, which produces wrong output).
**Disable CK V32 with `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=0`** until that path
is fixed. With CK V32 off + R5 stack (gather-first FP8 dequant P0-c, Triton
dequant kernel P1-a, fused index gather P1-c, bf16 inner BMM P1-e, skip
nan_to_num P1-b — all default-on once the patch is in tree), TPOT lands at
38.96 ms with correct output.

**FlyDSL Flash mxfp4 + R5 (38.96 ms TPOT) is ~3× faster than Flash-Base FP8 (115.4 ms TPOT)** on identical hardware — finally realizes the bandwidth advantage of 4-bit weights at c=1 decode.

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
