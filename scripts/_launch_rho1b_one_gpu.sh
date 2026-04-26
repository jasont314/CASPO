# Shared one-GPU Rho-1B MATH launcher body.
#
# Source this from a thin wrapper after setting:
#   METHOD=ppo|grpo|caspo
#   RUN_METHOD_TAG=<output/log tag>
#   GPU_DEFAULT=<default physical GPU id>
#   EXTRA_OVERRIDES=(--override key=value ...)
#
# Do not execute this file directly.
set -eo pipefail
# Don't use 'set -u' - conda activate scripts have unbound vars.

if [[ -z "${METHOD:-}" || -z "${RUN_METHOD_TAG:-}" ]]; then
    echo "[rho1b-onegpu] ERROR: wrapper must set METHOD and RUN_METHOD_TAG"
    exit 2
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate scalable

export HF_HOME="${HF_HOME:-/mnt/nvme_tmp/jason_caspo/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/nvme_tmp/jason_caspo/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/nvme_tmp/jason_caspo/hf_cache}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./scripts/perf_env.sh

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/scalable/bin/python}"
ROOT="${ROOT:-/mnt/nvme_tmp/jason_caspo}"
BASE_CONFIG="${BASE_CONFIG:-configs/caspo_rho1b_math.yaml}"

GPU_DEFAULT="${GPU_DEFAULT:-0}"
if [[ -n "${GPU:-}" ]]; then
    SELECTED_GPU="$GPU"
else
    read -r -a GPUS <<< "${GPU_LIST:-$GPU_DEFAULT}"
    SELECTED_GPU="${GPUS[0]:-$GPU_DEFAULT}"
fi

RUN_TAG="${RUN_TAG:-}"
RUN_SUFFIX=""
if [[ -n "$RUN_TAG" ]]; then
    RUN_SUFFIX="_${RUN_TAG}"
fi
LOGDIR="$ROOT/caspo_rho1b_math${RUN_SUFFIX}/logs"
mkdir -p "$LOGDIR"

OUTDIR="$ROOT/caspo_rho1b_math_${RUN_METHOD_TAG}${RUN_SUFFIX}"
LOG="$LOGDIR/phase2_${RUN_METHOD_TAG}.log"

CASPO_VLLM_GPU_MEMORY_UTILIZATION="${CASPO_VLLM_GPU_MEMORY_UTILIZATION:-${VLLM_GPU_MEMORY_UTILIZATION:-0.45}}"
CASPO_VLLM_MULTI_SAMPLE_MODE="${CASPO_VLLM_MULTI_SAMPLE_MODE:-${VLLM_MULTI_SAMPLE_MODE:-auto}}"
CASPO_VLLM_MAX_NUM_SEQS="${CASPO_VLLM_MAX_NUM_SEQS:-${VLLM_MAX_NUM_SEQS:-256}}"
CASPO_VLLM_MAX_NUM_BATCHED_TOKENS="${CASPO_VLLM_MAX_NUM_BATCHED_TOKENS:-${VLLM_MAX_NUM_BATCHED_TOKENS:-}}"

# These four names are launcher aliases, not native vLLM environment variables.
unset VLLM_GPU_MEMORY_UTILIZATION
unset VLLM_MULTI_SAMPLE_MODE
unset VLLM_MAX_NUM_SEQS
unset VLLM_MAX_NUM_BATCHED_TOKENS

COMMON_OVERRIDES=(
    --override "method=${METHOD}"
    --override rollout_backend=vllm
    --override vllm_weight_sync_backend=ipc
    --override "vllm_gpu_memory_utilization=${CASPO_VLLM_GPU_MEMORY_UTILIZATION}"
    --override vllm_enforce_eager=false
    --override "vllm_multi_sample_mode=${CASPO_VLLM_MULTI_SAMPLE_MODE}"
    --override "vllm_max_num_seqs=${CASPO_VLLM_MAX_NUM_SEQS}"
    --override "save_every=${SAVE_EVERY:-250}"
    --override "wandb_mode=${WANDB_MODE:-online}"
    --override "wandb_project=${WANDB_PROJECT:-caspo-rho1b-math}"
    --override "output_dir=${OUTDIR}"
    --override "wandb_run_name=rho1b_math_${RUN_METHOD_TAG}_seed0${RUN_SUFFIX}"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
    COMMON_OVERRIDES+=(--override "max_steps=${MAX_STEPS}")
fi
if [[ -n "${CASPO_VLLM_MAX_NUM_BATCHED_TOKENS:-}" ]]; then
    COMMON_OVERRIDES+=(--override "vllm_max_num_batched_tokens=${CASPO_VLLM_MAX_NUM_BATCHED_TOKENS}")
fi

echo "[rho1b-onegpu] ${RUN_METHOD_TAG} method=${METHOD} gpu=${SELECTED_GPU} out=${OUTDIR}"
echo "[rho1b-onegpu] vllm_util=${CASPO_VLLM_GPU_MEMORY_UTILIZATION} vllm_mode=${CASPO_VLLM_MULTI_SAMPLE_MODE} max_seqs=${CASPO_VLLM_MAX_NUM_SEQS} max_batched_tokens=${CASPO_VLLM_MAX_NUM_BATCHED_TOKENS:-auto}"
echo "[rho1b-onegpu] log=${LOG}"

CUDA_VISIBLE_DEVICES="$SELECTED_GPU" "$PYTHON_BIN" -u -m scripts.train_caspo \
    --config "$BASE_CONFIG" \
    "${COMMON_OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}" \
    > "$LOG" 2>&1

echo "[rho1b-onegpu] DONE ${RUN_METHOD_TAG} - log=${LOG}"
