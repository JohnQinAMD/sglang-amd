#!/bin/bash
# DSv4 Flash-Base FP8 — AITER sampler A/B bench launcher.
#
# Mirrors launch_dsv4_rocm720.sh but uses a separate container, port, and
# GPU set so the AITER-sampler ON/OFF runs can be benched in parallel against
# the production server (port 30012, GPUs 0-3) if needed.
#
# Toggle via SGLANG_SAMPLING_BACKEND_AITER=1|0 (default 1 once code is in).
#
# Usage:
#   SAMPLER_AITER=1 bash launch_dsv4_rocm720_aiter_sampler.sh   # AITER sampler ON
#   SAMPLER_AITER=0 bash launch_dsv4_rocm720_aiter_sampler.sh   # pytorch sampler

set -e

PRESET="${1:-stacked-best}"
CONTAINER="${CONTAINER:-sgl-aiter-sampler-bench}"
IMAGE="${IMAGE:-rocm/sgl-dev:rocm720-deepseek-v4-mi35x}"
SGLANG_PR_TREE="${SGLANG_PR_TREE:-/mnt/vast/john/sglang_v4_pr}"
HF_DIR="${HF_DIR:-/mnt/vast/john/huggingface}"
AITER_HEAD="${AITER_HEAD:-/mnt/vast/john/rocm-dynamo/aiter-amd}"
JIT_SHADOW="${JIT_SHADOW:-/mnt/vast/john/aiter_jit_rocm720_shadow}"
PORT="${PORT:-30022}"
GPUS="${GPUS:-4,5,6,7}"
SAMPLER_AITER="${SAMPLER_AITER:-1}"

# 1. Bootstrap the shadow JIT dir (shared with launch_dsv4_rocm720.sh).
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
  export PORT=$PORT
  export GPUS=$GPUS
  export MODEL=/hf/DeepSeek-V4-Flash-Base-srt
  # See launch_dsv4_rocm720.sh + handoff-baseline-garbage-token-FOUND-followup.md:
  # =1 bypasses the singleshot gate and exposes the s_q=1 labeled-transpose bug
  # → garbage tokens at greedy decode on Flash-Base FP8. "auto" enables two-shot
  # only at prefill (s_q>1) where it's safe.
  export SGLANG_HIP_CK_V32_TWO_SHOT=auto
  export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=0
  export SGLANG_FP8_PAGED_MQA_LOGITS_AITER=1
  export SGLANG_FP8_PAGED_MQA_LOGITS_FUSED_TRITON=0
  export AITER_XBFLOAT16=1
  # AITER sampler A/B feature gate.
  export SGLANG_SAMPLING_BACKEND_AITER=$SAMPLER_AITER
  echo \"[\$(date)] starting sglang ($PRESET) in $CONTAINER (port=$PORT GPUs=$GPUS sampler_aiter=$SAMPLER_AITER)\"
  exec bash launch_dsv4.sh $PRESET > /tmp/dsv4_server.log 2>&1
"

echo "[$(date)] Server starting in container $CONTAINER (port $PORT). Tail with:"
echo "  docker exec $CONTAINER tail -F /tmp/dsv4_server.log"
echo "Health: docker exec $CONTAINER curl -sf http://127.0.0.1:$PORT/health"
