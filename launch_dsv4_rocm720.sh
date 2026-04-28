#!/bin/bash
# DSv4 Flash-Base FP8 — Production launch script for ROCm 7.2 + aiter HEAD.
#
# Migrates from `rocm/sgl-dev:deepseek-v4-mi35x` (ROCm 7.0/7.1) to
# `rocm/sgl-dev:rocm720-deepseek-v4-mi35x` (ROCm 7.2). The HSA dispatch latency
# improvement in 7.2 (14.4 us -> 3.92 us per launch, 3.7x better) plus newer
# AITER fused kernels (rmsnorm_quant, fused_qk_norm_rope_cache_quant_shuffle,
# CK 2-stage MoE, asm bf16 fmoe) deliver a major E2E win:
#
#   metric         old container (Phase 12)   new container (Phase 13)   delta
#   TPOT (ms)            41.07                    32.28                -8.79 (-21.4%)
#   Output tok/s         91.19                    112.12                +22.97%
#   Total tok/s         183.61                    225.75                +22.97%
#   Bench duration       399.40 s                 324.84 s              -18.7%
#
# vs B200 reference 9.49 ms = 3.40x gap (was 4.34x).
#
# This script (a) creates the new container with the right mounts, (b) does
# one-time setup (flydsl upgrade, shadow jit dir for libstdc++ ABI mismatch),
# (c) starts the sglang server.
#
# Usage:
#   bash launch_dsv4_rocm720.sh [preset]    # default: stacked-best
#
# Run from inside the chi2774 host (NOT inside an existing container).

set -e

PRESET="${1:-stacked-best}"
CONTAINER="${CONTAINER:-sgl-deepseek-v4-mi35x-rocm720}"
IMAGE="${IMAGE:-rocm/sgl-dev:rocm720-deepseek-v4-mi35x}"
SGLANG_PR_TREE="${SGLANG_PR_TREE:-/mnt/vast/john/sglang_v4_pr}"
HF_DIR="${HF_DIR:-/mnt/vast/john/huggingface}"
AITER_HEAD="${AITER_HEAD:-/mnt/vast/john/rocm-dynamo/aiter-amd}"
JIT_SHADOW="${JIT_SHADOW:-/mnt/vast/john/aiter_jit_rocm720_shadow}"

# 1. Bootstrap the shadow JIT dir (libstdc++ ABI between aiter-amd's prebuilt
# .so files and ROCm 7.2's libstdc++ doesn't match — let aiter rebuild fresh).
if [ ! -d "$JIT_SHADOW" ] || [ ! -f "$JIT_SHADOW/core.py" ]; then
  echo "[$(date)] Bootstrapping shadow JIT dir at $JIT_SHADOW ..."
  rm -rf "$JIT_SHADOW"
  cp -r "$AITER_HEAD/aiter/jit" "$JIT_SHADOW"
  rm -f "$JIT_SHADOW"/*.so
  rm -rf "$JIT_SHADOW/build" "$JIT_SHADOW/__pycache__"
fi

# 2. Create the container if it doesn't exist.
if ! docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "[$(date)] Creating container $CONTAINER from $IMAGE ..."
  docker run -d --name "$CONTAINER" --network=host --shm-size=128G --ipc=host \
    --group-add video --device=/dev/kfd --device=/dev/dri \
    -v "$SGLANG_PR_TREE":/sgl-pr \
    -v "$HF_DIR":/hf \
    -v "$AITER_HEAD":/sgl-workspace/aiter \
    -v "$JIT_SHADOW":/sgl-workspace/aiter/aiter/jit \
    "$IMAGE" sleep infinity
fi

# 3. One-time setup inside the container.
docker exec "$CONTAINER" bash -c '
  set -e
  git config --global --add safe.directory /sgl-workspace/aiter || true
  # flydsl 0.1.2 in image is too old for aiter HEAD (needs >= 0.1.3).
  pip show flydsl 2>/dev/null | grep -q "Version: 0.1.[34]" || pip install --upgrade flydsl 2>&1 | tail -3
'

# 4. Start the server.
docker exec -d "$CONTAINER" bash -c "
  pkill -9 -f 'python3 -m sglang' 2>/dev/null || true
  sleep 4
  pkill -9 -f sglang 2>/dev/null || true
  sleep 2
  cd /sgl-pr
  export PYTHONPATH=/sgl-pr/python:\${PYTHONPATH:-}
  export PORT=30012
  export MODEL=/hf/DeepSeek-V4-Flash-Base-srt
  export SGLANG_HIP_CK_V32_TWO_SHOT=1
  export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
  export SGLANG_FP8_PAGED_MQA_LOGITS_FUSED_TRITON=0
  echo \"[\$(date)] starting sglang ($PRESET) in $CONTAINER\"
  exec bash launch_dsv4.sh $PRESET > /tmp/dsv4_server.log 2>&1
"

echo "[$(date)] Server starting in container $CONTAINER. Tail with:"
echo "  docker exec $CONTAINER tail -F /tmp/dsv4_server.log"
echo "First boot rebuilds ~12 aiter JIT modules (~6-8 min). Subsequent boots reuse the cache (~50s)."
echo ""
echo "Health: docker exec $CONTAINER curl -sf http://127.0.0.1:30012/health"
