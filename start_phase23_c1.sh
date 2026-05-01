#!/bin/bash
# Phase 23: C1 fix on top of Phase 22 stack. Switch
# SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 -> AITER=1, routing through the existing
# `fp8_paged_mqa_logits_aiter` single-kernel Triton path. Pro variants already
# use this; Flash-Base launch script never flipped. Predicted ~10 launches/call
# saved x ~30 c4 layers/step = ~300 launches/step (-0.44 ms graphed).
set -e
echo "[$(date)] killing old sglang..."
pkill -9 -f 'python3 -m sglang' 2>/dev/null || true
sleep 4
pkill -9 -f sglang 2>/dev/null || true
sleep 2

cd /sgl-pr
export PYTHONPATH=/sgl-pr/python:${PYTHONPATH:-}
export PORT=30012
export MODEL=/hf/DeepSeek-V4-Flash-Base-srt
# Phase 13 baseline (less the TORCH knob)
# Use "auto" (default) — =1 exposes the s_q=1 labeled-transpose bug for
# Flash-Base FP8 (no extra_k_cache) → garbage tokens at greedy decode.
# See handoff-baseline-garbage-token-FOUND-followup.md.
export SGLANG_HIP_CK_V32_TWO_SHOT=auto
export SGLANG_FP8_PAGED_MQA_LOGITS_FUSED_TRITON=0
export AITER_XBFLOAT16=1

# Phase 22 stack
export SGLANG_FUSED_ACT_QUANT_GATE=1
export SGLANG_FUSED_MHC_POST=1
# A2-#1 dual-output wiring REVERTED: TPOT 53.60 ms (+22 ms regression at Phase 25).
# Same degenerate-sampling pattern as Phase 24 F1. Likely correctness bug in
# dual-output kernel or wq_a/wkv tuple-consume path. Default OFF preserves
# Phase 23 baseline (TPOT 31.13 ms / 124.80 tok/s).
export SGLANG_FUSED_RMSNORM_QUANT_PER1x128=0

# Phase 23 C1: switch from torch fallback to aiter single-kernel
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0
export SGLANG_FP8_PAGED_MQA_LOGITS_AITER=1

echo "[$(date)] starting Phase 23 (C1: AITER paged-mqa-logits)"
exec bash launch_dsv4.sh stacked-best
