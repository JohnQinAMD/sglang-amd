# DSv4-Pro mxfp4 bring-up on AMD MI355X

Date: 2026-04-26 · Container `rocm/sgl-dev:v0.5.8-rocm700-mi35x-20260129` · branch `rocm-deepseek-v4`

## TL;DR

DSv4-Pro **mxfp4** (FP8 attention + mxfp4-packed routed experts, 384 experts × 61 layers, hidden 7168) brought up on a single MI355X TP=8 node. Greedy probe `Q: What is the capital of France? A:` returns ` Paris\n` correctly.

**Perf is below Pro-Base** at c=8 OSL=1024: 4.45 vs 6.51 tok/s/GPU. Pro-Base ([PRO_BASE_PERF_RESULTS.md](../rocm-dynamo/PRO_BASE_PERF_RESULTS.md) in the dynamo tree) remains the recommended production target.

| | Pro-Base R3+A1 | Pro mxfp4 (this work) |
|---|---:|---:|
| Memory at TP=8 | ~140 GB / rank (FP8) | ~97 GB / rank packed |
| Correctness | GREEN | GREEN ("Paris" probe) |
| c=8 OSL=1024 output tok/s | 52.04 | **35.57** |
| c=8 OSL=1024 tok/s/GPU | **6.51** | 4.45 |
| TPOT (ms) | 147.84 | 219.6 |
| TTFT (ms) | 6083 | 5583 |

## Why this work was needed

The default `Mxfp4MoEMethod.process_weights_after_loading` upcasts every routed expert to BF16 for the Triton MoE kernel (no AMD packed-mxfp4 runtime in tree). For Pro:

- BF16 routed-expert footprint: 384 × 61 × hidden(7168) × 2*ipp(6144) × 2 bytes ≈ **386 GB / rank** → exceeds 288 GB / GPU.
- The loader OOMs with `out of memory: 280.81 GB allocated` regardless of `--mem-fraction-static` (the upcast happens before KV-cache allocation).

Packed footprint is ~97 GB / rank (4 bits per weight + e8m0 scales). The fix is to skip the upcast and forward to a packed-mxfp4 MoE runtime — aiter's `cktile_moe_stage1/2` handles a16w4 (BF16 input × FP4×2 weight) on gfx950.

## Code changes

### 1. `python/sglang/srt/layers/quantization/mxfp4.py` (+97 lines)

New code path gated on `SGLANG_MXFP4_AITER=1`:

**`process_weights_after_loading`** — when the env knob is set, skip the BF16 upcast and instead pre-shuffle for the cktile a16w4 kernel:

- `shuffle_weight_a16w4(w13, NLane=16, gate_up=True)` for fused gate/up.
- `shuffle_weight_a16w4(w2, NLane=16, gate_up=False)` for down.
- `shuffle_scale_a16w4(w13_scale.view(E*N, K_blocks), E, gate_up=True)` then view back.
- Same for `w2_scale`.
- Final tensors viewed as `aiter.utility.dtypes.fp4x2` / `fp8_e8m0`.

**`apply()`** — when the env knob is set, route to `aiter.fused_moe(...)` with:

- `activation=ActivationType.Swiglu` (cktile dispatch — Silu fell into `ck_moe_stage2_fwd` which has no kernel for Pro shapes and aborts during cuda graph capture).
- `quant_type=QuantType.per_1x32`, `doweight_stage1=False`.
- `topk_weights = topk_output.topk_weights.to(torch.float32)` (aiter requires fp32).
- **EP topk_ids remap.** `StandardDispatcher.dispatch()` skips `topk_ids = local_expert_mapping[topk_ids]` when `_use_aiter` ([standard.py:170](python/sglang/srt/layers/moe/token_dispatcher/standard.py#L170) `not _use_aiter`), so apply() receives global IDs in `[0, num_experts)`. The aiter cktile path expects local IDs in `[0, num_local_experts)`. Remap with safe `-1 → 0` and zero out the corresponding `topk_weights` so non-local routings contribute nothing from this rank (the all-reduce sums real contributions from other ranks).
- `routed_scaling_factor` is **not** applied here — on HIP the model layer at [deepseek_v2.py:654](python/sglang/srt/models/deepseek_v2.py#L654) multiplies `final_hidden_states *= routed_scaling_factor` after the MoE call.

### 2. `python/sglang/srt/flashmla_tests/kernelkit/__init__.py` (-2)

Drop `from . import bench` and `from .bench import bench_by_cuda_events, bench_kineto` — `bench.py` is missing from the tree. Any path that imports `kernelkit` (e.g. `SGLANG_HACK_FLASHMLA_BACKEND=torch`) raises `ImportError` otherwise. The two helpers were unused outside the package.

### 3. `launch_dsv4_pro_mxfp4.sh` (new)

Self-contained launcher. Sets `SGLANG_MXFP4_AITER=1`, requires `--ep-size 8`. Sibling settings match Pro-Base (compressed attn backend, paged compressor, aiter MQA logits, page_size 256, chunked-prefill 8192, multi-bs cuda graph capture).

## Microbench-validated config

`/tmp/microbench_mxfp4_moe.py` (host-side, throwaway) builds a 48-expert / 7168-hidden / 3072-ipp MoE problem, dequantizes mxfp4 weights to BF16 reference, and probes 14 combinations of {weight shuffle, scale shuffle, activation, is_shuffled flag} against `aiter.fused_moe`. Sweep across M ∈ {1, 8, 32, 128}:

| Probe | Activation | Cosine vs BF16 ref |
|---|---|---:|
| raw weights, raw 3D scales | Swiglu | 0.01 |
| raw weights, raw 3D scales | Silu | 0.84–0.87 |
| raw weights, e8m0 scales (2D or 3D) | Silu | 0.95–0.98 |
| `shuffle_weight((16,16))` + e8m0 scales, `is_shuffled=True` | Silu | 0.95–1.00 |
| **`shuffle_weight_a16w4` + `shuffle_scale_a16w4`** | **Swiglu** | **1.0000** ✓ |
| `shuffle_weight_a16w4` + `shuffle_scale_a16w4` | Silu | 0.00 |

The `shuffle_weight_a16w4` + `shuffle_scale_a16w4` + Swiglu combination is the only one that matches the BF16 reference within bf16 precision (max_diff ~2e-3 across all batch sizes).

The microbench cut iteration cost from ~2.5 min (full server restart + cuda graph capture) to ~1 s — without it, identifying the right config from this combinatorial space would have taken hours.

## Why Pro mxfp4 is slower than Pro-Base

aiter's per_1x32 cktile path has no tuned configs in `tuned_fmoe.csv` for Pro's exact shape (E=48, hidden=7168, ipp=3072, topk=6) — the kernel falls back to the default heuristic (`[fused_moe] using 2stage default for ...` in the server log). Pro-Base uses the in-tree Triton MoE which has had targeted optimization for these shapes.

Two paths to close the gap (not pursued in this session):

1. **Tune the cktile a16w4 config for Pro shapes.** Run `aiter`'s `gemm_moe_tune.py` with Pro's exact M/N/K/E. Could plausibly recover most of the gap.
2. **Wire the lower-level `aiter.ops.triton.moe.fused_moe_mxfp4_silu_fused` + `fused_moe_mxfp4` directly,** bypassing the high-level cktile dispatch. The triton kernels have broader shape coverage but require manual moe_align_block_size + per-stage A-quant glue. Not present in the container's aiter package; would need a workspace bump.

## Reproduce

```bash
docker run -d --name sglang_pro \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --network=host --shm-size 64g \
  --security-opt seccomp=unconfined \
  --group-add video --group-add render --cap-add SYS_PTRACE \
  -v /mnt/vast/john/huggingface:/hf \
  -v /mnt/vast/john/sglang_v4_pr:/sgl-pr \
  -v /mnt/vast/john/sglang_v4_pr_jitcache:/sgl-workspace/aiter/aiter/jit \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  rocm/sgl-dev:v0.5.8-rocm700-mi35x-20260129 \
  bash /sgl-pr/launch_dsv4_pro_mxfp4.sh

# Greedy probe
curl -s http://127.0.0.1:30010/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"DeepSeek-V4-Pro-srt","prompt":"Q: What is the capital of France? A:","max_tokens":15,"temperature":0}'

# Bench
docker exec sglang_pro python3 -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port 30010 \
  --dataset-name random --random-input-len 1024 --random-output-len 1024 \
  --random-range-ratio 1.0 --num-prompts 32 --max-concurrency 8 \
  --warmup-requests 0 --disable-tqdm
```

The model dir `DeepSeek-V4-Pro-srt` is a sibling of `DeepSeek-V4-Pro/` with relative symlinks to the safetensor shards; only `config.json` is patched (`model_type=deepseek_v3`, `architectures=DeepseekV4ForCausalLM`, `quantization_config.quant_method=mxfp4`).
