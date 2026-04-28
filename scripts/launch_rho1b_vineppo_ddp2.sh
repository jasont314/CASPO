#!/usr/bin/env bash
# Two-GPU Rho-1B MATH VinePPO launcher.
#
# This is the fast Rho-scale multi-GPU path:
#   - 2 replicated DDP ranks, launched as one Python process per GPU
#   - one rank-local vLLM engine per process/GPU
#   - CUDA-IPC weight sync from each DDP replica into its local vLLM
#   - different prompts per rank
#
# We intentionally do not use torchrun here. Each process gets a one-GPU
# CUDA_VISIBLE_DEVICES value so vLLM's EngineCore sees exactly the same
# rank-local device as the trainer, while torch.distributed still connects
# both ranks through the shared MASTER_ADDR/MASTER_PORT rendezvous.
#
# Defaults preserve the paper-faithful global batch. The trainer interprets
# prompts_per_step as a GLOBAL count, so PROMPTS_PER_STEP=64 becomes
# 32 prompts/rank at world_size=2:
#   64 global prompts x G=8 = 512 responses/global outer step
#   2 ranks x grad_accum_steps=8 x micro_batch_size=4 = 64-response global PPO minibatch
#
# Usage:
#   RUN_TAG=paper512_seed0 GPU_LIST="2 3" WANDB_MODE=offline \
#     ./scripts/launch_rho1b_vineppo_ddp2.sh
#
# vLLM tuning knobs are accepted with a CASPO_ prefix so they are not
# mistaken for native vLLM environment variables by the child runtime.
# The older VLLM_* aliases are still accepted by this launcher.
#
# Smoke:
#   MAX_STEPS=1 SAVE_EVERY=0 PROMPTS_PER_STEP=1 GROUP_SIZE=1 \
#   GRAD_ACCUM_STEPS=1 VINEPPO_MC_ROLLOUTS=1 RUN_TAG=ddp2_smoke \
#   GPU_LIST="2 3" WANDB_MODE=disabled ./scripts/launch_rho1b_vineppo_ddp2.sh
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
ROOT="${ROOT:-/mnt/nvme_tmp2/jason_caspo}"
BASE_CONFIG="${BASE_CONFIG:-configs/caspo_rho1b_math.yaml}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29577}"

read -r -a GPUS <<< "${GPU_LIST:-2 3}"
if (( ${#GPUS[@]} != 2 )); then
    echo "[vineppo-ddp2] ERROR: GPU_LIST must contain exactly 2 GPU ids; got: ${GPU_LIST:-2 3}"
    exit 2
fi

RUN_TAG="${RUN_TAG:-}"
RUN_SUFFIX=""
if [[ -n "$RUN_TAG" ]]; then
    RUN_SUFFIX="_${RUN_TAG}"
fi
LOGDIR="$ROOT/caspo_rho1b_math${RUN_SUFFIX}/logs"
mkdir -p "$LOGDIR"

METHOD_TAG="${RUN_METHOD_TAG:-vineppo_ddp2}"
OUTDIR="$ROOT/caspo_rho1b_math_${METHOD_TAG}${RUN_SUFFIX}"
LOG0="$LOGDIR/phase2_${METHOD_TAG}_rank0.log"
LOG1="$LOGDIR/phase2_${METHOD_TAG}_rank1.log"

PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-64}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
VINEPPO_MC_ROLLOUTS="${VINEPPO_MC_ROLLOUTS:-9}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-false}"
LOGPROB_MICRO_BATCH_SIZE="${LOGPROB_MICRO_BATCH_SIZE:-16}"
CASPO_REWARD_WORKERS="${CASPO_REWARD_WORKERS:-${REWARD_WORKERS:-4}}"
CASPO_COMPILE="${CASPO_COMPILE:-${COMPILE:-false}}"
CASPO_VLLM_GPU_MEMORY_UTILIZATION="${CASPO_VLLM_GPU_MEMORY_UTILIZATION:-${VLLM_GPU_MEMORY_UTILIZATION:-0.45}}"
CASPO_VLLM_MULTI_SAMPLE_MODE="${CASPO_VLLM_MULTI_SAMPLE_MODE:-${VLLM_MULTI_SAMPLE_MODE:-auto}}"
CASPO_VLLM_MAX_NUM_SEQS="${CASPO_VLLM_MAX_NUM_SEQS:-${VLLM_MAX_NUM_SEQS:-256}}"
CASPO_VLLM_MAX_NUM_BATCHED_TOKENS="${CASPO_VLLM_MAX_NUM_BATCHED_TOKENS:-${VLLM_MAX_NUM_BATCHED_TOKENS:-}}"

# These four names are launcher aliases, not native vLLM environment variables.
# Leaving them exported causes noisy "unknown vLLM environment variable" warnings
# inside vLLM workers. Keep the resolved CASPO_* values above and drop aliases.
unset VLLM_GPU_MEMORY_UTILIZATION
unset VLLM_MULTI_SAMPLE_MODE
unset VLLM_MAX_NUM_SEQS
unset VLLM_MAX_NUM_BATCHED_TOKENS

OVERRIDES=(
    --override method=vineppo
    --override distributed_backend=ddp
    --override rollout_backend=vllm
    --override vllm_tensor_parallel_size=1
    --override vllm_weight_sync_backend=ipc
    --override "vllm_gpu_memory_utilization=${CASPO_VLLM_GPU_MEMORY_UTILIZATION}"
    --override vllm_enforce_eager=false
    --override "vllm_multi_sample_mode=${CASPO_VLLM_MULTI_SAMPLE_MODE}"
    --override "vllm_max_num_seqs=${CASPO_VLLM_MAX_NUM_SEQS}"
    --override update_value_during_policy=false
    --override "use_gradient_checkpointing=${USE_GRADIENT_CHECKPOINTING}"
    --override "prompts_per_step=${PROMPTS_PER_STEP}"
    --override "group_size=${GROUP_SIZE}"
    --override "micro_batch_size=${MICRO_BATCH_SIZE}"
    --override "grad_accum_steps=${GRAD_ACCUM_STEPS}"
    --override "vineppo_mc_rollouts=${VINEPPO_MC_ROLLOUTS}"
    --override "reward_workers=${CASPO_REWARD_WORKERS}"
    --override "compile=${CASPO_COMPILE}"
    --override "save_every=${SAVE_EVERY:-200}"
    --override "eval_every=${EVAL_EVERY:-${SAVE_EVERY:-200}}"
    --override "wandb_mode=${WANDB_MODE:-offline}"
    --override "wandb_project=${WANDB_PROJECT:-caspo-rho1b-math}"
    --override "output_dir=${OUTDIR}"
    --override "wandb_run_name=rho1b_math_${METHOD_TAG}_seed0${RUN_SUFFIX}"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
    OVERRIDES+=(--override "max_steps=${MAX_STEPS}")
fi
if [[ -n "${LOGPROB_MICRO_BATCH_SIZE:-}" ]]; then
    OVERRIDES+=(--override "logprob_micro_batch_size=${LOGPROB_MICRO_BATCH_SIZE}")
fi
if [[ -n "${CASPO_VLLM_MAX_NUM_BATCHED_TOKENS:-}" ]]; then
    OVERRIDES+=(--override "vllm_max_num_batched_tokens=${CASPO_VLLM_MAX_NUM_BATCHED_TOKENS}")
fi

echo "[vineppo-ddp2] GPUs=${GPUS[*]} out=${OUTDIR}"
echo "[vineppo-ddp2] rendezvous=${MASTER_ADDR}:${MASTER_PORT}"
echo "[vineppo-ddp2] prompts/rank=${PROMPTS_PER_STEP} group=${GROUP_SIZE} micro=${MICRO_BATCH_SIZE} grad_accum=${GRAD_ACCUM_STEPS} K=${VINEPPO_MC_ROLLOUTS}"
echo "[vineppo-ddp2] grad_ckpt=${USE_GRADIENT_CHECKPOINTING} logprob_micro=${LOGPROB_MICRO_BATCH_SIZE:-auto}"
echo "[vineppo-ddp2] vllm_util=${CASPO_VLLM_GPU_MEMORY_UTILIZATION} vllm_mode=${CASPO_VLLM_MULTI_SAMPLE_MODE} max_seqs=${CASPO_VLLM_MAX_NUM_SEQS} max_batched_tokens=${CASPO_VLLM_MAX_NUM_BATCHED_TOKENS:-auto}"
echo "[vineppo-ddp2] logs:"
echo "  rank0: ${LOG0}"
echo "  rank1: ${LOG1}"

PIDS=()

cleanup_children() {
    local code=$?
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    exit "$code"
}
trap cleanup_children INT TERM

launch_rank() {
    local rank="$1"
    local gpu="$2"
    local log="$3"
    echo "[vineppo-ddp2] launch rank=${rank} physical_gpu=${gpu} log=${log}"
    CUDA_VISIBLE_DEVICES="$gpu" \
    RANK="$rank" \
    LOCAL_RANK=0 \
    WORLD_SIZE=2 \
    LOCAL_WORLD_SIZE=1 \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u scripts/train_caspo.py \
        --config "$BASE_CONFIG" \
        "${OVERRIDES[@]}" \
        > "$log" 2>&1 &
    PIDS+=("$!")
}

launch_rank 0 "${GPUS[0]}" "$LOG0"
launch_rank 1 "${GPUS[1]}" "$LOG1"

remaining=${#PIDS[@]}
while (( remaining > 0 )); do
    if wait -n; then
        remaining=$((remaining - 1))
    else
        status=$?
        echo "[vineppo-ddp2] ERROR: rank process failed with status ${status}"
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done
        wait || true
        echo "[vineppo-ddp2] rank logs:"
        echo "  ${LOG0}"
        echo "  ${LOG1}"
        exit "$status"
    fi
done

trap - INT TERM

echo "[vineppo-ddp2] DONE"
