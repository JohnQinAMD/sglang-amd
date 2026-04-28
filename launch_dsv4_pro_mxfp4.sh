#!/bin/bash
# DSv4-Pro mxfp4 launcher for AMD MI355X (TP=8, EP=8).
#
# Pro mxfp4 packs FP4 routed-expert weights with E8M0 microscales (block_k=32)
# alongside FP8 attention weights. Without the packed-runtime path, the loader
# upcasts every routed expert to BF16 — for Pro that's 384 experts × 61 layers
# × hidden=7168 × moe_inter=3072, ~386 GB / rank, exceeds 288 GB / GPU.
#
# This launcher enables SGLANG_MXFP4_AITER=1 which:
#   * keeps weights packed during load (~97 GB / rank — fits on MI355X)
#   * routes the MoE forward to aiter.fused_moe (cktile a16w4 path)
#   * applies shuffle_weight_a16w4 + shuffle_scale_a16w4 + Swiglu activation
#     (microbench-validated cos=1.0 vs bf16 reference)
#   * remaps EP global expert IDs → local in apply() with safe -1 → 0
#     (the StandardDispatcher skips this remap when _use_aiter)
#
# Required: --ep-size 8. With moe_tp_size=8 (default), ipp=384 → w2 scale K=12,
# fails the K_Pack*K_Lane=8 divisibility constraint of shuffle_scale_a16w4. EP
# gives ipp=3072 (full intermediate per rank) → K=96, divisible by 8.
#
# Usage:
#   docker run --device=/dev/kfd --device=/dev/dri --ipc=host --network=host \
#     --shm-size 64g --security-opt seccomp=unconfined --group-add video --group-add render \
#     --cap-add SYS_PTRACE -v /path/to/hf:/hf -v /path/to/sglang_v4_pr:/sgl-pr \
#     -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     rocm/sgl-dev:rocm720-deepseek-v4-mi35x \
#     bash /sgl-pr/launch_dsv4_pro_mxfp4.sh
#
# Bench (chi2761, 2026-04-26, c=8 OSL=1024):
#   output throughput 35.57 tok/s (4.45 tok/s/GPU)
#   TPOT 219.6 ms, TTFT 5583 ms
# Slower than Pro-Base R3+A1 (6.51 tok/s/GPU, TPOT 147.84 ms) — cktile a16w4
# has no tuned configs for Pro shapes (E=48, hidden=7168, ipp=3072), so it
# falls back to a default heuristic.
set -euo pipefail

PORT="${PORT:-30010}"
MODEL="${MODEL:-/hf/DeepSeek-V4-Pro-srt}"
MEM_FRACTION="${MEM_FRACTION_OVERRIDE:-0.85}"   # Hardcoded 0.85 (override with MEM_FRACTION_OVERRIDE)
MAX_RUNNING_REQ="${MAX_RUNNING_REQ:-16}"
CONTEXT_LEN="${CONTEXT_LEN:-1048576}"
INDEXER_CAP="${INDEXER_CAP:-4096}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-1 2 4 8 16 32}"

export PYTHONPATH=/sgl-pr/python:${PYTHONPATH:-}
export MAX_JOBS=128
# Packed mxfp4 path — required for Pro to fit on 288 GB / GPU.
export SGLANG_MXFP4_AITER=1
# Compressor + indexer (matches Flash-Base / Pro-Base setup).
export SGLANG_OPT_USE_FUSED_COMPRESS=false
export SGLANG_OPT_USE_OLD_COMPRESSOR=false
export SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=false
export SGLANG_HACK_FLASHMLA_BACKEND=torch
# CK V32 sparse MLA decode is default-on via source change.
# export SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1

# Phase E (2026-04-28) — same combine-kernel gates as Flash-Base / Pro-Base.
# Pro-mxfp4 routes through the same sparse MLA combine path. See
# phase_e/STATUS.md for the work-score gate logic + measured numbers.
export SGLANG_CK_V32_TRITON_COMBINE="${SGLANG_CK_V32_TRITON_COMBINE:-1}"
export SGLANG_CK_V32_TRITON_COMBINE_NWAY="${SGLANG_CK_V32_TRITON_COMBINE_NWAY:-1}"
export SGLANG_CK_V32_TRITON_COMBINE_MIN_WORK="${SGLANG_CK_V32_TRITON_COMBINE_MIN_WORK:-8192}"
export SGLANG_CK_V32_TRITON_COMBINE_BLOCK_V="${SGLANG_CK_V32_TRITON_COMBINE_BLOCK_V:-256}"
export SGLANG_CK_V32_TRITON_COMBINE_NUM_WARPS="${SGLANG_CK_V32_TRITON_COMBINE_NUM_WARPS:-4}"
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_ENABLE_THINKING=1
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_TOPK_TRANSFORM_512_TORCH=0
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0
export SGLANG_FP8_PAGED_MQA_LOGITS_AITER=1
# Opt out of the fused-triton default — on Pro shapes the aiter
# `_deepgemm_fp8_paged_mqa_logits` (240 ms / call) beats fused-triton
# (876 ms / call). See dsv4-bottleneck-analysis.md.
export SGLANG_FP8_PAGED_MQA_LOGITS_FUSED_TRITON=0
export SGLANG_DSV4_FP4_EXPERTS=false
export SGLANG_OPT_DPSK_V4_RADIX=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
# mxfp4 routed experts use the aiter path; FORCE_TRITON_MOE_FP8 doesn't apply.
export SGLANG_FORCE_TRITON_MOE_FP8=0
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export SGLANG_INDEXER_MAX_SEQ_LEN="$INDEXER_CAP"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHINDUCTOR_DISABLE="${TORCHINDUCTOR_DISABLE:-1}"

echo "==================================================================="
echo "Pro-mxfp4 launcher  (TP=8 EP=8 packed-mxfp4 path)"
echo "  MODEL=$MODEL  PORT=$PORT  MEM_FRACTION=$MEM_FRACTION"
echo "  CUDA_GRAPH_BS=[$CUDA_GRAPH_BS]  MAX_RUNNING_REQ=$MAX_RUNNING_REQ"
echo "==================================================================="

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" --trust-remote-code \
  --disable-radix-cache --attention-backend compressed \
  --max-running-request "$MAX_RUNNING_REQ" --page-size 256 --chunked-prefill-size 8192 \
  --mem-fraction-static "$MEM_FRACTION" \
  --host 0.0.0.0 --disable-shared-experts-fusion \
  --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
  --skip-server-warmup --watchdog-timeout 1800 \
  --tp 8 --ep-size 1 --cuda-graph-bs $CUDA_GRAPH_BS \
  --num-continuous-decode-steps "${NUM_DECODE_STEPS:-1}" \
  ${ENABLE_PIECEWISE_CG:+--enable-piecewise-cuda-graph} \
  ${PIECEWISE_TOKENS:+--piecewise-cuda-graph-tokens $PIECEWISE_TOKENS} \
  ${DISABLE_CUDA_GRAPH:+--disable-cuda-graph} \
  ${LOAD_FORMAT:+--load-format $LOAD_FORMAT} \
  --context-length "$CONTEXT_LEN" --port "$PORT"
