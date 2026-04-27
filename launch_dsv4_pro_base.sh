#!/bin/bash
# DSv4-Pro-Base launcher for AMD MI355X (TP=8).
#
# Pro-Base is the FP8 variant of Pro (vs the mxfp4 variant in
# launch_dsv4_pro_mxfp4.sh). 1.5 TB FP8 weights load to ~140 GB / rank
# at TP=8 — fits in 288 GB / GPU.
#
# Pro-Base differences vs Flash-Base that drove patches in the source tree:
#   - 1.5 TB FP8 weights (vs Flash 295 GB) → needs lower mem_fraction_static
#   - index_topk=1024 (Flash: 512) → Triton TOPK port + metadata/backend patches
#   - wo_a is FP8 with separate per-layer scale (vs Flash converted-to-bf16) →
#     base-model dequant path required (mirrors upstream dc2b50758)
#   - 384 routed experts (Flash: 256), 61 layers (Flash: 43), hidden 7168
#
# This is the proven R3+A1 stack as of 2026-04-25 (cuda graph multi-bs +
# Triton TOPK + paged compressor + aiter MQA + aiter CK MoE).
#
# Best measured: c=8 OSL=1024 6.51 tok/s/GPU, c=16 OSL=1024 11.26 tok/s/GPU.
#
# Usage (host runs `docker run` separately, mounting volumes):
#   docker run --device=/dev/kfd --device=/dev/dri --ipc=host --network=host \
#     --shm-size 64g --security-opt seccomp=unconfined \
#     --group-add video --group-add render --cap-add SYS_PTRACE \
#     -v /path/to/hf:/hf -v /path/to/sglang_v4_pr:/sgl-pr \
#     -v /path/to/jitcache:/sgl-workspace/aiter/aiter/jit \
#     -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     rocm/sgl-dev:v0.5.8-rocm700-mi35x-20260129 \
#     bash /sgl-pr/launch_dsv4_pro_base.sh
set -euo pipefail

PORT="${PORT:-30010}"
MODEL="${MODEL:-/hf/DeepSeek-V4-Pro-Base-srt}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
MAX_RUNNING_REQ="${MAX_RUNNING_REQ:-64}"
CONTEXT_LEN="${CONTEXT_LEN:-1048576}"
INDEXER_CAP="${INDEXER_CAP:-4096}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-1 2 4 8 16 32}"

# A1 lever: =0 routes routed experts to aiter CK MoE (+2-4% over Triton MoE).
# Set to =1 if you want Triton MoE (e.g. for debugging).
SGLANG_FORCE_TRITON_MOE_FP8="${SGLANG_FORCE_TRITON_MOE_FP8:-0}"

export PYTHONPATH=/sgl-pr/python:${PYTHONPATH:-}
export MAX_JOBS=128
export SGLANG_OPT_USE_FUSED_COMPRESS=false
export SGLANG_OPT_USE_OLD_COMPRESSOR=false
export SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=false
export SGLANG_HACK_FLASHMLA_BACKEND=torch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
export SGLANG_OPT_USE_TILELANG_MHC_POST=false
export SGLANG_ENABLE_THINKING=1
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
# Triton TOPK port avoids a host sync in the pytorch fallback (capture-safe).
export SGLANG_TOPK_TRANSFORM_512_TORCH=0
# R3 aiter MQA logits wrapper.
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0
export SGLANG_FP8_PAGED_MQA_LOGITS_AITER=1
# Opt out of the fused-triton default — on Pro-Base shapes the aiter
# `_deepgemm_fp8_paged_mqa_logits` (240 ms / call) beats fused-triton
# (876 ms / call). See profile compare in dsv4-bottleneck-analysis.md.
export SGLANG_FP8_PAGED_MQA_LOGITS_FUSED_TRITON=0
export SGLANG_DSV4_FP4_EXPERTS=false
export SGLANG_OPT_DPSK_V4_RADIX=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
export SGLANG_FORCE_TRITON_MOE_FP8
export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0
export SGLANG_INDEXER_MAX_SEQ_LEN="$INDEXER_CAP"
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE=1

echo "==================================================================="
echo "Pro-Base launcher  (TP=8, R3+A1 stack)"
echo "  MODEL=$MODEL  PORT=$PORT  MEM_FRACTION=$MEM_FRACTION"
echo "  CUDA_GRAPH_BS=[$CUDA_GRAPH_BS]  MAX_RUNNING_REQ=$MAX_RUNNING_REQ"
echo "  FORCE_TRITON_MOE_FP8=$SGLANG_FORCE_TRITON_MOE_FP8 (=0 → A1 aiter CK MoE)"
echo "==================================================================="

exec python3 -m sglang.launch_server \
  --model-path "$MODEL" --trust-remote-code \
  --disable-radix-cache --attention-backend compressed \
  --max-running-request "$MAX_RUNNING_REQ" --page-size 256 --chunked-prefill-size 8192 \
  --mem-fraction-static "$MEM_FRACTION" \
  --host 0.0.0.0 --disable-shared-experts-fusion \
  --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
  --skip-server-warmup --watchdog-timeout 1800 \
  --tp 8 ${EP_SIZE:+--ep-size $EP_SIZE} --cuda-graph-bs $CUDA_GRAPH_BS \
  --num-continuous-decode-steps "${NUM_DECODE_STEPS:-1}" \
  --context-length "$CONTEXT_LEN" --port "$PORT"
