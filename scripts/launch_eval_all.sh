#!/usr/bin/env bash
# Phase 5 — eval all method checkpoints on the full Rho-1B-MATH eval suite.
# Runs: math500 (subset), math (full 5K), collegemath (500), olympiadbench (674).
# k=16 / temp=0.35 / top_p=0.9 (matches VinePPO eval protocol).
#
# Defaults are tuned for Rho-1B (max-new-tokens=1024, [MATH_TASK] template).
# For Qwen2.5-Math-1.5B + dsr_sub evals, override the relevant env vars:
#   EVAL_MAX_NEW_TOKENS=2048 (matches Qwen training cap)
#   EVAL_PROMPT_TEMPLATE="{query}\nLet's think step by step and output the final answer within \\boxed{}."
# Otherwise the launcher will silently truncate Qwen responses at 1024 and
# fall through to the [MATH_TASK] template, hiding 30+pp of Qwen gains.
#
# Usage:
#   ./scripts/launch_eval_all.sh
#   EVAL_GPU_LIST="4 5 6 7" ./scripts/launch_eval_all.sh
#   RUN_TAG=paper512_seed0 EVAL_GPU_LIST="4 5 6 7" ./scripts/launch_eval_all.sh
#   RUN_TAG=paper512_seed0 CKPT_SUBDIR=step_250 EVAL_BENCHMARKS=math500 EVAL_LIMIT=100 EVAL_K=8 ./scripts/launch_eval_all.sh
#   METHODS="caspo_prob caspo_logprob" RUN_TAG=paper512_seed0 ./scripts/launch_eval_all.sh
#
# Each method gets one GPU; benchmarks run sequentially inside each method job
# while methods run in parallel across GPUs.

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
ROOT="${ROOT:-/mnt/nvme_tmp2/jason_caspo}"
RUN_TAG="${RUN_TAG:-}"
RUN_SUFFIX=""
if [[ -n "$RUN_TAG" ]]; then
    RUN_SUFFIX="_${RUN_TAG}"
fi
LOGDIR="$ROOT/caspo_rho1b_math${RUN_SUFFIX}/logs"
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
# headline numbers); add them for the OOD section in the paper. Override with
# EVAL_BENCHMARKS=math500 and EVAL_LIMIT=100 for cheap periodic sample eval.
BENCHMARKS="${EVAL_BENCHMARKS:-math500,math,collegemath,olympiadbench}"
EVAL_K="${EVAL_K:-16}"
CKPT_SUBDIR="${CKPT_SUBDIR:-final}"
EVAL_VLLM_GPU_MEMORY_UTILIZATION="${EVAL_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1024}"     # 2048 for Qwen2.5-Math-1.5B
EVAL_PROMPT_TEMPLATE="${EVAL_PROMPT_TEMPLATE:-}"        # empty = use cfg.prompt_template
EVAL_LIMIT_ARGS=()
LIMIT_TAG="full"
if [[ -n "${EVAL_LIMIT:-}" ]]; then
    EVAL_LIMIT_ARGS=(--limit "$EVAL_LIMIT")
    LIMIT_TAG="limit${EVAL_LIMIT}"
fi
BENCH_TAG="${BENCHMARKS//,/+}"
CKPT_TAG="${CKPT_SUBDIR//\//_}"
read -r -a METHODS <<< "${METHODS:-grpo ppo_critic vineppo_ddp2 caspo caspo_prob caspo_logprob caspo_frozen_rm}"
IFS=' ' read -r -a EVAL_GPUS <<< "${EVAL_GPU_LIST:-0 1 2 3 4 5 6}"
if (( ${#EVAL_GPUS[@]} < ${#METHODS[@]} )); then
    echo "[eval] EVAL_GPU_LIST must provide at least ${#METHODS[@]} GPUs; got: ${EVAL_GPU_LIST:-0 1 2 3 4 5 6}" >&2
    exit 1
fi

eval_method() {
    local method=$1
    local gpu=$2
    local ckpt="$ROOT/caspo_rho1b_math_${method}${RUN_SUFFIX}/${CKPT_SUBDIR}"
    local out_dir="$ROOT/caspo_rho1b_math_${method}${RUN_SUFFIX}"
    local log="$LOGDIR/phase5_eval_${method}_${CKPT_TAG}_${BENCH_TAG}_k${EVAL_K}_${LIMIT_TAG}.log"
    local output_json="$out_dir/eval_results_${CKPT_TAG}_${BENCH_TAG}_k${EVAL_K}_${LIMIT_TAG}.json"
    if [[ ! -d "$ckpt" ]]; then
        echo "[eval] SKIP ${method} — no checkpoint at ${ckpt}"
        return
    fi
    echo "[eval] ${method} -> GPU ${gpu} -> ckpt=${ckpt}"
    local template_override=()
    if [[ -n "$EVAL_PROMPT_TEMPLATE" ]]; then
        template_override=(--override "prompt_template=${EVAL_PROMPT_TEMPLATE}")
    fi
    # python -u for unbuffered output (belt-and-suspenders with PYTHONUNBUFFERED).
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON_BIN" -u -m scripts.eval \
        --config configs/caspo_rho1b_math.yaml \
        --override "model_name_or_path=${ckpt}" \
        --override "output_dir=${out_dir}" \
        --override wandb_enabled=false \
        --override rollout_backend=vllm \
        "${template_override[@]}" \
        --benchmarks "$BENCHMARKS" \
        --k "$EVAL_K" \
        --temperature 0.35 \
        --top-p 0.9 \
        --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
        --gpu-memory-utilization "$EVAL_VLLM_GPU_MEMORY_UTILIZATION" \
        "${EVAL_LIMIT_ARGS[@]}" \
        --output "$output_json" \
        > "$log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    echo "  pid=$pid log=$log"
}

for i in "${!METHODS[@]}"; do
    eval_method "${METHODS[$i]}" "${EVAL_GPUS[$i]}"
done

echo "[eval] launched ${#PIDS[@]} job(s); logs in $LOGDIR/"
echo "[eval] aggregate results live under $ROOT/caspo_rho1b_math_<method>${RUN_SUFFIX}/"

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
