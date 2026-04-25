#!/usr/bin/env bash
# Phase 5 — eval all method checkpoints on the full Rho-1B-MATH eval suite.
# Runs: math500 (subset), math (full 5K), collegemath (500), olympiadbench (674).
# k=16 / temp=0.35 / top_p=0.9 / max_tokens=1024 (matches VinePPO eval protocol).
#
# Usage:
#   ./scripts/launch_eval_all.sh
#
# Each method gets one GPU; runs sequentially per method but parallel across methods.

set -eo pipefail
# Don't use 'set -u' — conda activate scripts have unbound vars.
# Run from the repo root so that --config configs/*.yaml resolves regardless
# of where this launcher was invoked from (cron, parent shell, absolute path).
cd "$(dirname "$0")/.."
source /opt/conda/etc/profile.d/conda.sh
conda activate scalable

# Cache env (must match trainer launchers for shared HF cache).
export HF_HOME=/mnt/nvme_tmp/jason_caspo/hf_cache
export HF_HUB_CACHE=/mnt/nvme_tmp/jason_caspo/hf_cache
export TRANSFORMERS_CACHE=/mnt/nvme_tmp/jason_caspo/hf_cache

# Centralized performance/env config: CUDA allocator, NCCL knobs,
# tokenizer + vLLM + HF noise reduction, CPU thread caps. See
# scripts/perf_env.sh for what's exported and why.
source ./scripts/perf_env.sh

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/scalable/bin/python}"
ROOT=/mnt/nvme_tmp/jason_caspo
LOGDIR="$ROOT/caspo_rho1b_math/logs"
mkdir -p "$LOGDIR"

# Track child PIDs so the parent waits for them — without `wait`, the shell
# exits immediately after backgrounding nohup and orphans logs mid-flush.
PIDS=()

cleanup() {
    local rc=$?
    if (( ${#PIDS[@]} > 0 )); then
        echo "[eval] cleanup (rc=$rc) — surviving children: ${PIDS[*]}"
    fi
    exit $rc
}
trap cleanup EXIT
trap 'echo "[eval] ERR at line $LINENO (rc=$?)"' ERR

# Headline benchmarks. Skip aime24 / aime25 / amc23 here (too small to be
# headline numbers); add them for the OOD section in the paper.
BENCHMARKS="math500,math,collegemath,olympiadbench"

eval_method() {
    local method=$1
    local gpu=$2
    local ckpt="$ROOT/caspo_rho1b_math_${method}/final"
    local out_dir="$ROOT/caspo_rho1b_math_${method}"
    local log="$LOGDIR/phase5_eval_${method}.log"
    if [[ ! -d "$ckpt" ]]; then
        echo "[eval] SKIP ${method} — no checkpoint at ${ckpt}"
        return
    fi
    echo "[eval] ${method} → GPU ${gpu} → ckpt=${ckpt}"
    # python -u for unbuffered output (belt-and-suspenders with PYTHONUNBUFFERED).
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON_BIN" -u -m scripts.eval \
        --config configs/caspo_rho1b_math.yaml \
        --override "model_name_or_path=${ckpt}" \
        --override "output_dir=${out_dir}" \
        --override wandb_enabled=false \
        --override rollout_backend=vllm \
        --benchmarks "$BENCHMARKS" \
        --k 16 \
        --temperature 0.35 \
        --top-p 0.9 \
        --max-new-tokens 1024 \
        > "$log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    echo "  pid=$pid log=$log"
}

eval_method caspo   0
eval_method grpo    1
eval_method vineppo 2
eval_method ppo     3

echo "[eval] launched ${#PIDS[@]} job(s); logs in $LOGDIR/"
echo "[eval] aggregate results live at $ROOT/caspo_rho1b_math_<method>/eval_results.json"

# Wait for each child and report any non-zero exits without aborting the
# remaining evals.
fail=0
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        :
    else
        rc=$?
        echo "[eval] pid=$pid exited with rc=$rc"
        fail=$((fail + 1))
    fi
done
if (( fail > 0 )); then
    echo "[eval] DONE — $fail/${#PIDS[@]} eval job(s) failed; check logs"
    exit 1
fi
echo "[eval] DONE — all ${#PIDS[@]} eval job(s) completed cleanly"
