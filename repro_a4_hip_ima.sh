#!/bin/bash
# Automated repro for the HIP IMA at c=16 with bs=[1,2,4,8,16] cuda graph capture.
# Run inside dsv4_proven container on chi2762 (or any node with the model loaded).
#
# Demonstrates:
#   1. Broken combo crashes c=16 within first ~80 prompts (HIP IMA)
#   2. AMD_SERIALIZE_KERNEL=3 makes the crash go away (proves it's an async race)
#   3. Dropping bs=2 from the captured set also makes the crash go away
#
# Usage:
#   bash repro_a4_hip_ima.sh           # full sequence (broken → serialize → drop-bs2)
#   bash repro_a4_hip_ima.sh broken    # just the broken case
#   bash repro_a4_hip_ima.sh serialize # broken + AMD_SERIALIZE_KERNEL=3
#   bash repro_a4_hip_ima.sh dropbs2   # bs=[1,4,8,16] workaround
#
# Output:
#   /tmp/repro_a4_<case>.log — full server log for each case
#   stdout: bench results + crash status

set -uo pipefail
LAUNCHER=/sgl-pr/launch_dsv4.sh
PORT=30010
CASE="${1:-all}"

bench_c16() {
    local label="$1"
    echo
    echo "=== Bench c=16 [$label] ==="
    python3 -m sglang.bench_serving \
        --backend sglang --base-url "http://127.0.0.1:$PORT" \
        --dataset-name random --random-input-len 256 --random-output-len 16 \
        --random-range-ratio 1.0 --max-concurrency 16 --num-prompts 160 \
        --warmup-requests 32 2>&1 | grep -E "Output token|Median TTFT|Median TPOT|Successful|exception|refused|Error|TransferEncoding" \
        | head -8
    echo
}

wait_ready() {
    local timeout=$((SECONDS + 240))
    while [ $SECONDS -lt $timeout ]; do
        if curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200; then
            return 0
        fi
        sleep 5
    done
    echo "[ERROR] server did not reach /health 200 in 240s" >&2
    return 1
}

cleanup() {
    pkill -9 -f sglang.launch_server 2>/dev/null || true
    sleep 3
}

run_case() {
    local case_name="$1"; shift
    local logfile="/tmp/repro_a4_${case_name}.log"

    cleanup
    rm -f "$logfile"

    echo "════════════════════════════════════════════════════════════════════"
    echo " CASE: $case_name"
    echo " ENV: $@"
    echo " LOG: $logfile"
    echo "════════════════════════════════════════════════════════════════════"

    env "$@" nohup bash "$LAUNCHER" stacked-best > "$logfile" 2>&1 &
    sleep 3

    if ! wait_ready; then
        echo "[$case_name] server failed to come up — see $logfile"
        return 1
    fi

    bench_c16 "$case_name"

    if curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q 200; then
        echo "[$case_name] server still alive after bench — no crash"
    else
        echo "[$case_name] *** server CRASHED during bench ***"
        echo "Last traceback:"
        awk '/Scheduler hit an exception/{found=1} found{print; if(/AcceleratorError|^[A-Z][a-zA-Z]+Error/) {exit}}' "$logfile" | tail -25
    fi
}

case "$CASE" in
  broken)
    run_case "broken" CUDA_GRAPH_BS="1 2 4 8 16" CUDA_GRAPH_MAX_BS=16
    ;;
  serialize)
    run_case "serialize" AMD_SERIALIZE_KERNEL=3 CUDA_GRAPH_BS="1 2 4 8 16" CUDA_GRAPH_MAX_BS=16
    ;;
  dropbs2)
    run_case "dropbs2" CUDA_GRAPH_BS="1 4 8 16" CUDA_GRAPH_MAX_BS=16
    ;;
  all|"")
    run_case "broken" CUDA_GRAPH_BS="1 2 4 8 16" CUDA_GRAPH_MAX_BS=16
    run_case "serialize" AMD_SERIALIZE_KERNEL=3 CUDA_GRAPH_BS="1 2 4 8 16" CUDA_GRAPH_MAX_BS=16
    run_case "dropbs2" CUDA_GRAPH_BS="1 4 8 16" CUDA_GRAPH_MAX_BS=16
    ;;
  *)
    echo "Usage: $0 [broken|serialize|dropbs2|all]" >&2
    exit 2
    ;;
esac

cleanup

cat <<'SUMMARY'
════════════════════════════════════════════════════════════════════
 EXPECTED RESULTS:
   broken    → c=16 crashes mid-bench (HIP IMA in compress_extend_old:1035)
   serialize → c=16 succeeds at ~38 tok/s (AMD_SERIALIZE_KERNEL=3 disables
               async dispatch, eliminates the race)
   dropbs2   → c=16 succeeds at ~46 tok/s (workaround: drop bs=2 from
               captured set so {2,4,8,16}-all-captured trigger isn't met)

 ROOT CAUSE: async race in HIP graph pool / caching allocator
 SHIPPED FIX: launcher preset stacked-widebs uses bs=[1,4,8,16]
 OPEN: full root-cause kernel-side fix
 DETAILS: /mnt/vast/john/rocm-dynamo/A4_HIP_IMA_REPRO.md
════════════════════════════════════════════════════════════════════
SUMMARY
