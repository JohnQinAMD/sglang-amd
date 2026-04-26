#!/bin/bash
# Unified DSv4-Flash-Base launcher for AMD MI355X (TP=4).
#
# Replaces the 12+ launch_*.sh scripts that diverged on /tmp during the
# Tier-1 campaign. All shared settings are hardcoded; per-preset deltas
# live in the case statement.
#
# Usage:
#   bash launch_dsv4.sh [preset]            # picks preset (default: stacked-best)
#   PORT=30011 bash launch_dsv4.sh tier0    # override port
#   SGLANG_FORCE_TRITON_MOE_FP8=0 bash launch_dsv4.sh stacked-best    # ad-hoc env override
#
# Presets:
#   tier0             Tier-0 baseline. No cuda graph, --skip-server-warmup.
#                     Sanity-check reference; matches RUN_ON_AMD_MI355X.md.
#   testP             Test P. bs=1 cuda graph + indexer patch + multi-stream off.
#   testT             Test T. multi-bs cuda graph (1,2,4,8) + cap=32768.
#   testV             Test V. Top-K JIT HIP Triton port + bs=1 cuda graph + cap=4096.
#   stacked-best      Tier-1 stacked best (testT + testV + A6 TileLang mHC).
#                     DEFAULT. ISL=256 OSL=16 c=8 → 27.95 tok/s (post-A1+A6).
#   stacked-aiter-moe stacked-best + SGLANG_FORCE_TRITON_MOE_FP8=0 (A2).
#   stacked-widebs    stacked-best + cuda-graph-bs 1 2 4 8 16 (A4).
#
# Env overrides (apply on top of any preset):
#   PORT                            default 30010
#   GPUS                            default 0,1,2,3
#   SGLANG_TOPK_TRANSFORM_512_TORCH preset-dependent
#   SGLANG_FORCE_TRITON_MOE_FP8     preset-dependent
#   SGLANG_INDEXER_MAX_SEQ_LEN      preset-dependent
#   SGLANG_OPT_USE_TILELANG_MHC_PRE  preset-dependent (1 for stacked-*)
#   SGLANG_OPT_USE_TILELANG_MHC_POST preset-dependent (1 for stacked-*)
#   CUDA_GRAPH_BS                   preset-dependent (e.g. "1 2 4 8" or "1")
#   CUDA_GRAPH_MAX_BS               preset-dependent
#   DISABLE_CUDA_GRAPH              preset-dependent (1 to pass --disable-cuda-graph)
#   SKIP_SERVER_WARMUP              preset-dependent (1 to pass --skip-server-warmup)

set -euo pipefail
ulimit -c 0

PRESET="${1:-stacked-best}"
PORT="${PORT:-30010}"
GPUS="${GPUS:-0,1,2,3}"
MODEL="${MODEL:-/model}"

# ───── Common (never changes) ────────────────────────────────────────────────
export PYTHONPATH=/sgl-pr/python:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES="$GPUS"
export MAX_JOBS=128 NINJA_MAX_JOBS=128

export SGLANG_OPT_USE_FUSED_COMPRESS=false
export SGLANG_OPT_USE_OLD_COMPRESSOR=true
export SGLANG_OPT_USE_TILELANG_SWA_PREPARE=false
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=false
export SGLANG_OPT_USE_FUSED_HASH_TOPK=false
export SGLANG_HACK_FLASHMLA_BACKEND=torch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
# SGLANG_OPT_USE_TILELANG_MHC_PRE / _POST default per preset (see case block).
# Env override wins; otherwise preset default (1 for stacked-* on MI355X — Xiaobo
# measured +9% prefill on MI300X, replicated as +2-4% throughput / -6 to -7% TTFT
# on MI355X; A6 in TIER2_OPTIMIZATION_PLAN.md).
export SGLANG_ENABLE_THINKING=1
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1
export SGLANG_DSV4_FP4_EXPERTS=false
export SGLANG_OPT_DPSK_V4_RADIX=0
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
# Sparse MLA decode kernel.
#   DSv4-Flash-Base (qk_head_dim=512): use Triton fallback in flashmla_tests/triton_sparse_decode_kernel.py
#   DSv4-Pro V32 (qk_head_dim=576): use CK Tile FP8 (SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1, commit 6e86d00f2)
# The CK V32 kernel hardcodes QK_HEAD_DIM=576 → asserts at runtime on Flash.
export SGLANG_TRITON_SPARSE_DECODE=1

export TORCHINDUCTOR_CACHE_DIR=/mnt/vast/john/rocm-dynamo/sglang/.inductor_cache_dsv4
export TRITON_CACHE_DIR=/mnt/vast/john/rocm-dynamo/sglang/.triton_cache_dsv4
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# ───── Preset deltas ─────────────────────────────────────────────────────────
# Each preset sets these knobs (env-var overrides win against these defaults):
#   _TOPK_TORCH       → SGLANG_TOPK_TRANSFORM_512_TORCH
#   _FORCE_TRITON_MOE → SGLANG_FORCE_TRITON_MOE_FP8
#   _INDEXER_CAP      → SGLANG_INDEXER_MAX_SEQ_LEN
#   _MULTI_STREAM     → SGLANG_OPT_USE_MULTI_STREAM_OVERLAP
#   _DISABLE_COMPILE  → SGLANG_DISABLE_CAPTURE_COMPILE
#   _CG_BS            → --cuda-graph-bs values (space-separated)
#   _CG_MAX_BS        → --cuda-graph-max-bs
#   _DISABLE_CG       → 1 to pass --disable-cuda-graph (omits both _CG_*)
#   _SKIP_WARMUP      → 1 to pass --skip-server-warmup

case "$PRESET" in
  tier0)
    _TOPK_TORCH=1; _FORCE_TRITON_MOE=1; _INDEXER_CAP=0
    _MULTI_STREAM=""; _DISABLE_COMPILE=""
    _MHC_PRE=false; _MHC_POST=false
    _DISABLE_CG=1; _SKIP_WARMUP=1
    ;;
  testP)
    _TOPK_TORCH=1; _FORCE_TRITON_MOE=1; _INDEXER_CAP=0
    _MULTI_STREAM=0; _DISABLE_COMPILE=1
    _MHC_PRE=false; _MHC_POST=false
    _CG_BS="1"; _CG_MAX_BS=1
    ;;
  testT)
    _TOPK_TORCH=1; _FORCE_TRITON_MOE=1; _INDEXER_CAP=32768
    _MULTI_STREAM=0; _DISABLE_COMPILE=1
    _MHC_PRE=false; _MHC_POST=false
    _CG_BS="1 2 4 8"; _CG_MAX_BS=8
    ;;
  testV)
    _TOPK_TORCH=0; _FORCE_TRITON_MOE=1; _INDEXER_CAP=4096
    _MULTI_STREAM=0; _DISABLE_COMPILE=1
    _MHC_PRE=false; _MHC_POST=false
    _CG_BS="1"; _CG_MAX_BS=1
    ;;
  stacked-best)
    # A6 baked-in (2026-04-25): TileLang mHC PRE/POST=1 on gfx950.
    # Measured +2-4% throughput, -6 to -7% TTFT vs MHC=off; replicates
    # Xiaobo's MI300X +9% prefill (xiaobo/optim_p0a_mhc.md).
    _TOPK_TORCH=0; _FORCE_TRITON_MOE=1; _INDEXER_CAP=4096
    _MULTI_STREAM=0; _DISABLE_COMPILE=1
    _MHC_PRE=1; _MHC_POST=1
    _CG_BS="1 2 4 8"; _CG_MAX_BS=8
    ;;
  stacked-aiter-moe)
    _TOPK_TORCH=0; _FORCE_TRITON_MOE=0; _INDEXER_CAP=4096
    _MULTI_STREAM=0; _DISABLE_COMPILE=1
    _MHC_PRE=1; _MHC_POST=1
    _CG_BS="1 2 4 8"; _CG_MAX_BS=8
    ;;
  stacked-widebs)
    # bs=[1,2,4,8,16] full set, c=16 = 46.55 tok/s.
    # Earlier bisection (2026-04-25 A4-debug) found bs=[1,2,4,8,16] crashed
    # at c=16 with HIP IMA in compress_extend_old:1035. Root cause: fresh
    # per-call allocations of `temp_buffer` and `compressed_kv_output` in
    # compress_extend_old churned the caching allocator, creating an
    # alias window with the captured-graph slabs in the graph pool —
    # async race surfaced as IMA later. Fix: scratch-stabilized those two
    # allocations via _ensure_scratch (mirrors the working pattern in
    # compress_extend at L693-L709). See models/deepseek_v4.py:1010-1031.
    _TOPK_TORCH=0; _FORCE_TRITON_MOE=1; _INDEXER_CAP=4096
    _MULTI_STREAM=0; _DISABLE_COMPILE=1
    _MHC_PRE=1; _MHC_POST=1
    _CG_BS="1 2 4 8 16"; _CG_MAX_BS=16
    ;;
  *)
    echo "Unknown preset: $PRESET" >&2
    echo "Valid: tier0 testP testT testV stacked-best stacked-aiter-moe stacked-widebs" >&2
    exit 2
    ;;
esac

# ───── Apply preset (allow env override) ─────────────────────────────────────
export SGLANG_TOPK_TRANSFORM_512_TORCH="${SGLANG_TOPK_TRANSFORM_512_TORCH:-$_TOPK_TORCH}"
export SGLANG_FORCE_TRITON_MOE_FP8="${SGLANG_FORCE_TRITON_MOE_FP8:-$_FORCE_TRITON_MOE}"
export SGLANG_INDEXER_MAX_SEQ_LEN="${SGLANG_INDEXER_MAX_SEQ_LEN:-$_INDEXER_CAP}"
export SGLANG_OPT_USE_TILELANG_MHC_PRE="${SGLANG_OPT_USE_TILELANG_MHC_PRE:-$_MHC_PRE}"
export SGLANG_OPT_USE_TILELANG_MHC_POST="${SGLANG_OPT_USE_TILELANG_MHC_POST:-$_MHC_POST}"
[ -n "$_MULTI_STREAM" ] && export SGLANG_OPT_USE_MULTI_STREAM_OVERLAP="${SGLANG_OPT_USE_MULTI_STREAM_OVERLAP:-$_MULTI_STREAM}"
[ -n "$_DISABLE_COMPILE" ] && export SGLANG_DISABLE_CAPTURE_COMPILE="${SGLANG_DISABLE_CAPTURE_COMPILE:-$_DISABLE_COMPILE}"

# ───── Build CLI ─────────────────────────────────────────────────────────────
CLI=(
  python3 -m sglang.launch_server
  --model-path "$MODEL"
  --trust-remote-code
  --tp 4
  --disable-radix-cache
  --attention-backend compressed
  --max-running-request 256
  --page-size 256
  --chunked-prefill-size 8192
  --disable-shared-experts-fusion
  --watchdog-timeout 1800
  --host 0.0.0.0
  --port "$PORT"
)

if [ "${DISABLE_CUDA_GRAPH:-${_DISABLE_CG:-}}" = "1" ]; then
  CLI+=(--disable-cuda-graph)
else
  CLI+=(--cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-$_CG_MAX_BS}")
  # --cuda-graph-bs takes nargs='+' — pass all values as separate args after the flag.
  CLI+=(--cuda-graph-bs)
  for bs in ${CUDA_GRAPH_BS:-$_CG_BS}; do
    CLI+=("$bs")
  done
fi

[ "${SKIP_SERVER_WARMUP:-${_SKIP_WARMUP:-}}" = "1" ] && CLI+=(--skip-server-warmup)

echo "==================================================================="
echo "Preset: $PRESET    Port: $PORT    GPUs: $GPUS"
echo "TOPK_TORCH=$SGLANG_TOPK_TRANSFORM_512_TORCH  FORCE_TRITON_MOE=$SGLANG_FORCE_TRITON_MOE_FP8  CAP=$SGLANG_INDEXER_MAX_SEQ_LEN"
echo "MHC_PRE=$SGLANG_OPT_USE_TILELANG_MHC_PRE  MHC_POST=$SGLANG_OPT_USE_TILELANG_MHC_POST"
echo "MULTI_STREAM=${SGLANG_OPT_USE_MULTI_STREAM_OVERLAP:-unset}  DISABLE_COMPILE=${SGLANG_DISABLE_CAPTURE_COMPILE:-unset}"
echo "CUDA_GRAPH: $([ "${DISABLE_CUDA_GRAPH:-${_DISABLE_CG:-}}" = "1" ] && echo disabled || echo "max_bs=${CUDA_GRAPH_MAX_BS:-$_CG_MAX_BS} bs=[${CUDA_GRAPH_BS:-$_CG_BS}]")"
echo "==================================================================="

exec "${CLI[@]}"
