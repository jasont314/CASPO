#!/usr/bin/env bash
# Sidecar watcher that runs scripts/eval.py on each new step_N/ checkpoint
# as it appears under one or more run output dirs. Decoupled from the
# trainer — no code changes required, just run alongside training.
#
# Usage:
#   ./scripts/eval_periodic.sh \
#       --gpu 7 \
#       --config configs/caspo_rho1b_math.yaml \
#       --k 8 --limit 100 --temperature 0.7 \
#       /mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_paper_seed0 \
#       /mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_frozen_rm_paper_seed0
#
# Each step_N/ dir gets evaluated exactly once; result lands at
# step_N/eval.json. Uses --gpu-memory-utilization=0.30 to share with
# concurrent training rollouts. Skips already-evaluated checkpoints
# on restart (idempotent).
set -uo pipefail

EVAL_GPU="${EVAL_GPU:-7}"
CONFIG="${CONFIG:-configs/caspo_rho1b_math.yaml}"
K="${K:-8}"
LIMIT="${LIMIT:-100}"
TEMP="${TEMP:-0.7}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"

declare -a RUN_DIRS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) EVAL_GPU="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --temperature) TEMP="$2"; shift 2 ;;
        --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
        --) shift; RUN_DIRS+=("$@"); break ;;
        -*) echo "[eval_watcher] unknown flag: $1" >&2; exit 2 ;;
        *) RUN_DIRS+=("$1"); shift ;;
    esac
done

if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
    echo "[eval_watcher] usage: $0 [--gpu N] [--config Y] [--k K] [--limit N] [--temperature T] <run_dir> [run_dir ...]" >&2
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/scalable/bin/python}"
echo "[eval_watcher] watching ${#RUN_DIRS[@]} run dirs on GPU $EVAL_GPU"
echo "[eval_watcher] config=$CONFIG k=$K limit=$LIMIT temp=$TEMP poll=${POLL_INTERVAL}s"

run_eval() {
    local ckpt_dir="$1"
    local out="$ckpt_dir/eval.json"
    local logf="$ckpt_dir/eval.log"
    if [[ -f "$out" ]]; then return 0; fi
    echo "[eval_watcher] evaluating $ckpt_dir"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
        "$PYTHON_BIN" -u scripts/eval.py \
            --config "$CONFIG" \
            --override "model_name_or_path=$ckpt_dir" \
            --benchmarks math500 \
            --k "$K" --limit "$LIMIT" \
            --temperature "$TEMP" --top-p 0.9 \
            --backend vllm \
            --gpu-memory-utilization 0.30 \
            --output "$out" \
            > "$logf" 2>&1
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "[eval_watcher] OK $ckpt_dir → $out"
    else
        echo "[eval_watcher] FAIL $ckpt_dir (rc=$rc); see $logf"
    fi
    return $rc
}

while true; do
    for run_dir in "${RUN_DIRS[@]}"; do
        if [[ ! -d "$run_dir" ]]; then continue; fi
        # Glob step_N dirs in numeric order, oldest first (so eval lags
        # training but always processes earliest pending checkpoint next).
        for ckpt in $(ls -d "$run_dir"/step_* 2>/dev/null | sort -t_ -k2 -n); do
            [[ -d "$ckpt" ]] || continue
            run_eval "$ckpt" || true
        done
    done
    sleep "$POLL_INTERVAL"
done
