#!/usr/bin/env bash
# Launch CASPO advantage-transform ablations on Rho-1B MATH.
#
# Default launches only the two extra experiments because the direct-value
# variant is the existing CASPO run:
#   prob    -> TD on sigmoid(V)
#   logprob -> TD on log sigmoid(V)
#
# Usage:
#   RUN_TAG=paper512_seed0 GPU_LIST="4 5" WANDB_MODE=offline \
#     ./scripts/launch_rho1b_caspo_ablations.sh
#
# Optional:
#   ADV_VARIANTS="value prob logprob" ./scripts/launch_rho1b_caspo_ablations.sh
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

read -r -a GPUS <<< "${GPU_LIST:-4 5}"
read -r -a VARIANTS <<< "${ADV_VARIANTS:-prob logprob}"
if (( ${#GPUS[@]} < ${#VARIANTS[@]} )); then
    echo "[ablate] ERROR: GPU_LIST must contain at least ${#VARIANTS[@]} GPU ids; got: ${GPU_LIST:-}"
    exit 2
fi

COMMON_OVERRIDES=(
    --override vllm_gpu_memory_utilization=0.30
    --override vllm_enforce_eager=false
    --override vllm_weight_sync_backend=ipc
    --override "micro_batch_size=${MICRO_BATCH_SIZE:-8}"
    --override "grad_accum_steps=${GRAD_ACCUM_STEPS:-8}"
    --override "use_gradient_checkpointing=${USE_GRADIENT_CHECKPOINTING:-false}"
    --override "save_every=${SAVE_EVERY:-250}"
    --override "eval_every=${EVAL_EVERY:-${SAVE_EVERY:-250}}"
    --override "wandb_mode=${WANDB_MODE:-offline}"
    --override "wandb_project=${WANDB_PROJECT:-caspo-rho1b-math}"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
    COMMON_OVERRIDES+=(--override "max_steps=${MAX_STEPS}")
fi

PIDS=()

cleanup() {
    local rc=$?
    if (( rc != 0 )); then
        echo "[ablate] exiting with rc=$rc; launched pids: ${PIDS[*]:-none}"
    fi
    exit $rc
}
trap cleanup EXIT
trap 'echo "[ablate] ERR at line $LINENO (rc=$?)"' ERR

resolve_variant() {
    local variant=$1
    case "$variant" in
        value|direct|td)
            echo "caspo value"
            ;;
        prob|probability|sigmoid)
            echo "caspo_prob prob"
            ;;
        logprob|log_probability|logsigmoid)
            echo "caspo_logprob logprob"
            ;;
        *)
            echo "[ablate] ERROR: unknown ADV_VARIANTS entry '$variant' (use value, prob, logprob)" >&2
            return 2
            ;;
    esac
}

launch_variant() {
    local variant=$1
    local gpu=$2
    local resolved
    resolved=$(resolve_variant "$variant")
    read -r tag transform <<< "$resolved"

    local outdir="$ROOT/caspo_rho1b_math_${tag}${RUN_SUFFIX}"
    local log="$LOGDIR/phase2_${tag}.log"
    echo "[ablate] ${tag} (${transform}) -> GPU ${gpu} -> ${outdir}"
    CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON_BIN" -u -m scripts.train_caspo \
        --config "$BASE_CONFIG" \
        --override method=caspo \
        --override "caspo_advantage_transform=${transform}" \
        --override "output_dir=${outdir}" \
        --override "wandb_run_name=rho1b_math_${tag}_seed0${RUN_SUFFIX}" \
        "${COMMON_OVERRIDES[@]}" \
        > "$log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    echo "  pid=$pid log=$log"
}

for i in "${!VARIANTS[@]}"; do
    launch_variant "${VARIANTS[$i]}" "${GPUS[$i]}"
done

echo "[ablate] launched ${#PIDS[@]} CASPO ablation job(s); logs in $LOGDIR/"

fail=0
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        :
    else
        rc=$?
        echo "[ablate] pid=$pid exited with rc=$rc"
        fail=$((fail + 1))
    fi
done
if (( fail > 0 )); then
    echo "[ablate] DONE - $fail/${#PIDS[@]} ablation job(s) failed; check logs"
    exit 1
fi
echo "[ablate] DONE - all ${#PIDS[@]} ablation jobs completed cleanly"
