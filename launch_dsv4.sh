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

# Allow caller env overrides (use `${X:-default}` instead of plain `=default`).
# Phase 4 host-side optimization needs to flip these via env without editing the script.
export SGLANG_OPT_USE_FUSED_COMPRESS="${SGLANG_OPT_USE_FUSED_COMPRESS:-false}"
export SGLANG_OPT_USE_OLD_COMPRESSOR="${SGLANG_OPT_USE_OLD_COMPRESSOR:-true}"
export SGLANG_OPT_USE_TILELANG_SWA_PREPARE="${SGLANG_OPT_USE_TILELANG_SWA_PREPARE:-false}"
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK="${SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK:-false}"
export SGLANG_OPT_USE_FUSED_HASH_TOPK="${SGLANG_OPT_USE_FUSED_HASH_TOPK:-false}"
export SGLANG_HACK_FLASHMLA_BACKEND="${SGLANG_HACK_FLASHMLA_BACKEND:-torch}"
export SGLANG_OPT_DEEPGEMM_HC_PRENORM="${SGLANG_OPT_DEEPGEMM_HC_PRENORM:-false}"
# Allow Phase 4 / B200-aligned envs to be set via caller. Defaults match prior behavior.
export SGLANG_OPT_DPSK_V4_RADIX="${SGLANG_OPT_DPSK_V4_RADIX:-0}"
# SGLANG_OPT_USE_TILELANG_MHC_PRE / _POST default per preset (see case block).
# Env override wins; otherwise preset default (1 for stacked-* on MI355X — Xiaobo
# measured +9% prefill on MI300X, replicated as +2-4% throughput / -6 to -7% TTFT
# on MI355X; A6 in TIER2_OPTIMIZATION_PLAN.md).
export SGLANG_ENABLE_THINKING=1
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=1
# L1 fp8_paged_mqa_logits dispatch knobs. Default keeps the proven torch path
# (45.23 ms TPOT). Two opt-in alternatives:
#   SGLANG_FP8_PAGED_MQA_LOGITS_FUSED_TRITON=1 — Triton fused (32 us microbench
#     but cuda-graph regression on Flash-Base, see L1 history).
#   SGLANG_FP8_PAGED_MQA_LOGITS_HIP=1 — gfx950 HIP kernel, 4-30 us microbench
#     across b=1..6 (vs torch 132, B200 sm100 8.7); E2E A/B in progress. JIT
#     compiles via cpp_extension on first server boot (~30 s extra startup).
export SGLANG_FP8_PAGED_MQA_LOGITS_TORCH="${SGLANG_FP8_PAGED_MQA_LOGITS_TORCH:-1}"
export SGLANG_FP8_PAGED_MQA_LOGITS_FUSED_TRITON="${SGLANG_FP8_PAGED_MQA_LOGITS_FUSED_TRITON:-0}"
export SGLANG_FP8_PAGED_MQA_LOGITS_HIP="${SGLANG_FP8_PAGED_MQA_LOGITS_HIP:-0}"
# E2E A/B (chi2811 c=4 max=6, ISL=OSL=1024 num=16, 16/16 successful):
#   torch (default):       TPOT 45.25 ms   throughput 85.95 tok/s
#   FUSED_TRITON:          TPOT 49.81 ms   throughput 78.25 tok/s   (+4.6 ms)
#   HIP v1 (no scratch):   TPOT 46.44 ms   throughput 84.06 tok/s   (+1.2 ms)
#   HIP v2 (+persistent scratch): TPOT 45.46 / 45.53 ms   throughput 85.52 / 87.03 tok/s
#                          (parity with torch — within bench noise)
# Persistent output scratch in the HIP loader (mirrors the torch path's
# _FP8_PAGED_SCRATCH_CAPTURED pattern) closed ~0.95 ms vs naive HIP. The
# remaining torch-vs-HIP delta is at the noise floor; either path is shippable.
# Microbench (eager) numbers:  HIP 4-30 us  /  fused 32 us  /  torch 132 us.
# Microbench wins do not always predict cuda-graph E2E wins because graph
# replay amortizes torch's 14 small launches.
export SGLANG_DSV4_FP4_EXPERTS=false
export SGLANG_OPT_DPSK_V4_RADIX="${SGLANG_OPT_DPSK_V4_RADIX:-0}"
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false
export SGLANG_OPT_USE_FUSED_STORE_CACHE=false
# Sparse MLA decode kernel.
#   DSv4-Flash-Base (qk_head_dim=512): use torch ref + R5 stack (gather-first
#     dequant + Triton fusion + bf16 inner BMM); enable
#     `SGLANG_TRITON_SPARSE_DECODE=1` to route through the Triton sparse-decode
#     kernel, otherwise the torch-compile path runs.
#   DSv4-Pro V32 (qk_head_dim=576): use CK Tile FP8 via
#     `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1` (commit 6e86d00f2).
#
# WARNING: as of 2026-04-26 the CK V32 kernel produces garbage on end-to-end
# Flash mxfp4 inference at production TOPK ≥ 256 + multi-split reduce path
# (microbench_ck_v32_512.py uses TOPK=64 / num_splits=1 and reports cos-sim
# 0.999998, which masks the bug). Until that's fixed, DO NOT set
# `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1` for Flash mxfp4 — the launcher leaves
# it unset by default; the working SOTA path (38.96 ms TPOT) is
# `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=0` (or unset) + the R5 + HIP-routing
# patches in this commit.
export SGLANG_TRITON_SPARSE_DECODE=1

# Flash-Base FP8 c=8+ stability: at chunked prefill s_q=8192 the Triton
# sparse-decode kernel allocates ~4.5 GB temp gathered_kv per layer × 43
# layers, which churns the caching allocator and produces server crashes /
# graph-pool aperture violations. Routing prefill (s_q > 1) to the CK V32
# kernel via `SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1` + the two-shot patch
# avoids this — measured stable at c=8 / 40 prompts in iter19_c8_bench.
# (The Flash mxfp4 CK V32 bug noted above does NOT affect Flash-Base FP8;
# the bug is at TOPK≥256 + multi-split reduce specifically for the mxfp4
# attention shape. Flash-Base FP8 uses CK V32 successfully in iter19.)
# At decode (s_q=1) the auto gate falls back to the Triton path so the
# decode kernel still benefits from R5 stack.
if [ "${SGLANG_HIP_SPARSE_MLA_DECODE_FP8:-}" = "" ]; then
    # default-on for Flash-Base FP8 to fix c≥8 stability; opt-out by setting
    # SGLANG_HIP_SPARSE_MLA_DECODE_FP8=0 explicitly.
    export SGLANG_HIP_SPARSE_MLA_DECODE_FP8=1
fi
if [ "${SGLANG_HIP_CK_V32_TWO_SHOT:-}" = "" ]; then
    # `auto` = CK V32 only at prefill (s_q > 1). Decode (s_q=1) keeps the
    # Triton path because CK V32 underperforms at the small extra-KV shape
    # (page_size=2 for compress_ratio=128). See iter3 / iter14 measurements.
    export SGLANG_HIP_CK_V32_TWO_SHOT=auto
fi

# Phase E (2026-04-28) — sparse MLA combine kernel gates for any DSv4 path
# routed through ck_v32_sparse_mla.py (Flash-Base FP8 + any future variant).
# Code defaults already match these — exported explicitly for discoverability.
# E2E A/B chi2811 (shipping config, num=40 c=4): TPOT neutral, P99 TTFT -33%,
# throughput +0.5%. Microbench: BLOCK_V=256/num_warps=4 hits HBM roofline
# (1.6-1.8x faster than CK at PREFILL q≥4096); below work_score 8192 the gate
# routes to CK to avoid Triton's ~40us launch overhead. See phase_e/STATUS.md.
export SGLANG_CK_V32_TRITON_COMBINE="${SGLANG_CK_V32_TRITON_COMBINE:-1}"
export SGLANG_CK_V32_TRITON_COMBINE_NWAY="${SGLANG_CK_V32_TRITON_COMBINE_NWAY:-1}"
export SGLANG_CK_V32_TRITON_COMBINE_MIN_WORK="${SGLANG_CK_V32_TRITON_COMBINE_MIN_WORK:-8192}"
export SGLANG_CK_V32_TRITON_COMBINE_BLOCK_V="${SGLANG_CK_V32_TRITON_COMBINE_BLOCK_V:-256}"
export SGLANG_CK_V32_TRITON_COMBINE_NUM_WARPS="${SGLANG_CK_V32_TRITON_COMBINE_NUM_WARPS:-4}"

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
    #
    # Phase 7 (2026-04-28): _FORCE_TRITON_MOE flipped 1→0. AITER's
    # `fmoe_bf16_blockscaleFp8_g1u1` (gfx950-native FP8 [128,128] blockwise)
    # produces 4× fewer launches per layer than Triton fused_moe_kernel
    # (86 vs 344) for similar GPU exec time. Aligned bench result on chi2774
    # (Flash-Base FP8, 40 prompts c=4 max=6):
    #     metric       Triton MoE  AITER MoE   delta
    #     TPOT (ms)      42.68       41.39    -1.29 (-3.0%)
    #     TTFT (ms)      230.57      219.17  -11.40 (-4.9%)
    #     output tok/s    86.67       89.91   +3.74%
    #     duration (s)   420.22      405.09   -3.6%
    # Net first measurable TPOT win after Phase B+ shipped.
    _TOPK_TORCH=0; _FORCE_TRITON_MOE=0; _INDEXER_CAP=4096
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
  --max-running-request 6
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
