#!/usr/bin/env bash
# Phase 4b — launch PPO + CASPO + GRPO + VinePPO baselines in parallel on Rho-1B-MATH.
# Each method gets one GPU (trainer + vLLM share). Override GPU_LIST to choose
# physical GPU IDs, e.g. GPU_LIST="4 5 6 7" ./scripts/launch_rho1b_parallel.sh
#
# Usage:
#   ./scripts/launch_rho1b_parallel.sh
#   RUN_TAG=paper512_seed0 GPU_LIST="4 5 6 7" ./scripts/launch_rho1b_parallel.sh
#   RUN_TAG=paper512_seed0 SAVE_EVERY=100 WANDB_MODE=offline ./scripts/launch_rho1b_parallel.sh
#
# Logs land in /mnt/nvme_tmp/jason_caspo/caspo_rho1b_math${RUN_SUFFIX}/logs.
# Outputs go to /mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_<method>${RUN_SUFFIX}/.
# Default checkpoint cadence is save_every=250, yielding step_250, step_500,
# step_750, and final for a 1000-step run.
#
set -eo pipefail
# Don't use 'set -u' — conda activate scripts have unbound vars.
source /opt/conda/etc/profile.d/conda.sh
conda activate scalable

export HF_HOME=/mnt/nvme_tmp/jason_caspo/hf_cache
export HF_HUB_CACHE=/mnt/nvme_tmp/jason_caspo/hf_cache
export TRANSFORMERS_CACHE=/mnt/nvme_tmp/jason_caspo/hf_cache

cd "$(dirname "$0")/.."
source ./scripts/perf_env.sh

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/scalable/bin/python}"
ROOT=/mnt/nvme_tmp/jason_caspo
BASE_CONFIG=configs/caspo_rho1b_math.yaml
RUN_TAG="${RUN_TAG:-}"
RUN_SUFFIX=""
if [[ -n "$RUN_TAG" ]]; then
    RUN_SUFFIX="_${RUN_TAG}"
fi
LOGDIR="$ROOT/caspo_rho1b_math${RUN_SUFFIX}/logs"
mkdir -p "$LOGDIR"
PIDS=()
read -r -a GPUS <<< "${GPU_LIST:-4 5 6 7}"
if (( ${#GPUS[@]} < 4 )); then
    echo "[launch] ERROR: GPU_LIST must contain at least 4 GPU ids; got: ${GPU_LIST:-}"
    exit 2
fi
echo "[launch] using GPUs: ${GPUS[*]} (ppo caspo grpo vineppo)"

cleanup() {
    local rc=$?
    if (( rc != 0 )); then
        echo "[launch] exiting with rc=$rc; launched pids: ${PIDS[*]:-none}"
    fi
    exit $rc
}
trap cleanup EXIT
trap 'echo "[launch] ERR at line $LINENO (rc=$?)"' ERR

# Trainer batching defaults from the Apr 2026 Pareto sweep (see README and
# scripts/_launch_rho1b_one_gpu.sh): mb=8/accum=8/ckpt=false redistributes the
# 64-response global PPO minibatch and runs ~45-47% faster per step than the
# YAML defaults. vllm_gpu_memory_utilization=0.30 leaves ~3-4 GB trainer
# headroom for CASPO's value+adam states (vLLM is not the bottleneck above
# ~0.30 in this layout).
COMMON_OVERRIDES=(
    --override vllm_gpu_memory_utilization=0.30
    --override vllm_enforce_eager=false
    --override vllm_weight_sync_backend=ipc
    --override "micro_batch_size=${MICRO_BATCH_SIZE:-8}"
    --override "grad_accum_steps=${GRAD_ACCUM_STEPS:-8}"
    --override "use_gradient_checkpointing=${USE_GRADIENT_CHECKPOINTING:-false}"
    --override "save_every=${SAVE_EVERY:-250}"
    --override "wandb_mode=${WANDB_MODE:-offline}"
    --override "wandb_project=${WANDB_PROJECT:-caspo-rho1b-math}"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
    COMMON_OVERRIDES+=(--override "max_steps=${MAX_STEPS}")
fi

launch_method() {
    local method=$1
    local gpu=$2
    local extra=$3   # extra overrides as a single string (or empty)
    local outdir="$ROOT/caspo_rho1b_math_${method}${RUN_SUFFIX}"
    local log="$LOGDIR/phase2_${method}.log"
    echo "[launch] ${method} → GPU ${gpu} → ${outdir}"
    # shellcheck disable=SC2086  # $extra is intentionally word-split into multiple --override flags
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON_BIN" -u -m scripts.train_caspo \
        --config "$BASE_CONFIG" \
        --override "method=${method}" \
        --override "output_dir=${outdir}" \
        --override "wandb_run_name=rho1b_math_${method}_seed0${RUN_SUFFIX}" \
        "${COMMON_OVERRIDES[@]}" \
        ${extra} \
        > "$log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    echo "  pid=$pid log=$log"
}

# PPO — sequence-level terminal reward advantage baseline.
launch_method ppo "${GPUS[0]}" "--override update_value_during_policy=false"

# CASPO with online IPVRM + ADB+DLW (default)
launch_method caspo "${GPUS[1]}" ""

# GRPO — no V_φ; the trainer only requires prefix_value_path for method=caspo.
launch_method grpo "${GPUS[2]}" "--override update_value_during_policy=false"

# VinePPO — no V_φ; uses sample_with_prefix MC values at each prefix.
launch_method vineppo "${GPUS[3]}" "--override update_value_during_policy=false --override vineppo_mc_rollouts=9"

echo "[launch] all ${#PIDS[@]} methods started; logs in $LOGDIR/"

fail=0
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        :
    else
        rc=$?
        echo "[launch] pid=$pid exited with rc=$rc"
        fail=$((fail + 1))
    fi
done
if (( fail > 0 )); then
    echo "[launch] DONE — $fail/${#PIDS[@]} training job(s) failed; check logs"
    exit 1
fi
echo "[launch] DONE — all ${#PIDS[@]} training jobs completed cleanly"
