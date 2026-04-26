#!/usr/bin/env bash
# Launch the Rho-1B MATH CASPO frozen-RM ablation.
#
# This keeps CASPO's IPVRM prefix-value scoring, but disables online updates
# to phi. It is the speed/stability ablation for:
#   update_value_during_policy=false
#
# Usage:
#   RUN_TAG=paper512_seed0 GPU_LIST="4" WANDB_MODE=offline \
#     ./scripts/launch_rho1b_caspo_frozen_rm.sh
#
set -eo pipefail
# Don't use 'set -u' - conda activate scripts have unbound vars.
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

read -r -a GPUS <<< "${GPU_LIST:-4}"
GPU="${GPUS[0]:-4}"
TAG="${RUN_METHOD_TAG:-caspo_frozen_rm}"
OUTDIR="$ROOT/caspo_rho1b_math_${TAG}${RUN_SUFFIX}"
LOG="$LOGDIR/phase2_${TAG}.log"

COMMON_OVERRIDES=(
    --override method=caspo
    --override update_value_during_policy=false
    --override caspo_advantage_transform=value
    --override vllm_gpu_memory_utilization=0.45
    --override vllm_enforce_eager=false
    --override "save_every=${SAVE_EVERY:-250}"
    --override "wandb_mode=${WANDB_MODE:-online}"
    --override "wandb_project=${WANDB_PROJECT:-caspo-rho1b-math}"
    --override "output_dir=${OUTDIR}"
    --override "wandb_run_name=rho1b_math_${TAG}_seed0${RUN_SUFFIX}"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
    COMMON_OVERRIDES+=(--override "max_steps=${MAX_STEPS}")
fi

echo "[frozen-rm] caspo frozen RM -> GPU ${GPU} -> ${OUTDIR}"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u -m scripts.train_caspo \
    --config "$BASE_CONFIG" \
    "${COMMON_OVERRIDES[@]}" \
    > "$LOG" 2>&1
echo "[frozen-rm] DONE - log=$LOG"
