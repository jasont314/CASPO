# Shared FSDP-based 7B launcher body. Source this from a thin per-method wrapper.
#
# Required wrapper-set vars:
#   METHOD=ppo|grpo|caspo|vineppo
#   RUN_METHOD_TAG=<output/log tag>
#   GPU_DEFAULT_LIST="0 1 2 3"   # default GPU set (4 or 8 depending on method)
#   EXTRA_OVERRIDES=(--override key=value ...)
#
# Optional env knobs:
#   GPU_LIST              override default GPU set (space-separated ids)
#   RUN_TAG               output dir + log suffix (default empty)
#   MAX_STEPS             override cfg.max_steps
#   SAVE_EVERY            override save cadence (default 250)
#   WANDB_MODE            online | offline | disabled (default offline)
#   WANDB_PROJECT
#   BASE_CONFIG           path to YAML (default configs/caspo_deepseekmath7b_math.yaml)
#   CASPO_VLLM_GPU_MEMORY_UTILIZATION   default 0.30 (vLLM rank-local; 7B trainer is tight)
#   CASPO_VLLM_MAX_NUM_SEQS              default 256
#   CASPO_VLLM_ENFORCE_EAGER             default false (CUDA graphs on)
#   CASPO_MICRO_BATCH_SIZE               default 1
#   CASPO_GRAD_ACCUM_STEPS               default 16 for 4-GPU, 8 for 8-GPU (override via env)
#   CASPO_USE_GRADIENT_CHECKPOINTING     default true (7B activation memory)
#   CASPO_FSDP_CPU_OFFLOAD               default false (flip to true if OOM)
#   CASPO_REWARD_WORKERS                 default 4
#   PROMPTS_PER_STEP                     default 8 (paper-faithful global)
#
# Do not execute this file directly.
set -eo pipefail

if [[ -z "${METHOD:-}" || -z "${RUN_METHOD_TAG:-}" ]]; then
    echo "[7b-fsdp] ERROR: wrapper must set METHOD and RUN_METHOD_TAG"
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
BASE_CONFIG="${BASE_CONFIG:-configs/caspo_deepseekmath7b_math.yaml}"

# Resolve GPU set: GPU_LIST env wins, else GPU_DEFAULT_LIST from wrapper.
read -r -a GPUS <<< "${GPU_LIST:-${GPU_DEFAULT_LIST:-0 1 2 3}}"
NRANK=${#GPUS[@]}
if (( NRANK < 1 )); then
    echo "[7b-fsdp] ERROR: GPU_LIST must contain at least one GPU id"
    exit 2
fi

RUN_TAG="${RUN_TAG:-}"
RUN_SUFFIX=""
if [[ -n "$RUN_TAG" ]]; then
    RUN_SUFFIX="_${RUN_TAG}"
fi

LOGDIR="$ROOT/deepseekmath7b_math${RUN_SUFFIX}/logs"
mkdir -p "$LOGDIR"

OUTDIR="$ROOT/deepseekmath7b_math_${RUN_METHOD_TAG}${RUN_SUFFIX}"
LOG="$LOGDIR/phase2_${RUN_METHOD_TAG}.log"

# vLLM and trainer knobs (overridable via env).
CASPO_VLLM_GPU_MEMORY_UTILIZATION="${CASPO_VLLM_GPU_MEMORY_UTILIZATION:-${VLLM_GPU_MEMORY_UTILIZATION:-0.30}}"
CASPO_VLLM_MAX_NUM_SEQS="${CASPO_VLLM_MAX_NUM_SEQS:-${VLLM_MAX_NUM_SEQS:-256}}"
CASPO_VLLM_ENFORCE_EAGER="${CASPO_VLLM_ENFORCE_EAGER:-false}"

CASPO_MICRO_BATCH_SIZE="${CASPO_MICRO_BATCH_SIZE:-${MICRO_BATCH_SIZE:-1}}"
# Default accum: keep global PPO minibatch = 64 across world_size.
# 4 ranks × 1 mb × 16 accum = 64; 8 ranks × 1 mb × 8 accum = 64.
DEFAULT_ACCUM=$(( 64 / NRANK / CASPO_MICRO_BATCH_SIZE ))
if (( DEFAULT_ACCUM < 1 )); then DEFAULT_ACCUM=1; fi
CASPO_GRAD_ACCUM_STEPS="${CASPO_GRAD_ACCUM_STEPS:-${GRAD_ACCUM_STEPS:-$DEFAULT_ACCUM}}"

CASPO_USE_GRADIENT_CHECKPOINTING="${CASPO_USE_GRADIENT_CHECKPOINTING:-${USE_GRADIENT_CHECKPOINTING:-true}}"
CASPO_FSDP_CPU_OFFLOAD="${CASPO_FSDP_CPU_OFFLOAD:-${FSDP_CPU_OFFLOAD:-false}}"
CASPO_REWARD_WORKERS="${CASPO_REWARD_WORKERS:-${REWARD_WORKERS:-4}}"
PROMPTS_PER_STEP_VAL="${PROMPTS_PER_STEP:-8}"

# Drop alias env names so vLLM doesn't warn.
unset VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_NUM_SEQS
unset MICRO_BATCH_SIZE GRAD_ACCUM_STEPS USE_GRADIENT_CHECKPOINTING
unset FSDP_CPU_OFFLOAD REWARD_WORKERS PROMPTS_PER_STEP

COMMON_OVERRIDES=(
    --override "method=${METHOD}"
    --override rollout_backend=vllm
    --override vllm_weight_sync_backend=ipc
    --override "vllm_gpu_memory_utilization=${CASPO_VLLM_GPU_MEMORY_UTILIZATION}"
    --override "vllm_enforce_eager=${CASPO_VLLM_ENFORCE_EAGER}"
    --override "vllm_max_num_seqs=${CASPO_VLLM_MAX_NUM_SEQS}"
    --override distributed_backend=fsdp
    --override fsdp_sharding_strategy=full_shard
    --override fsdp_use_orig_params=true
    --override fsdp_auto_wrap=true
    --override "fsdp_cpu_offload=${CASPO_FSDP_CPU_OFFLOAD}"
    --override "micro_batch_size=${CASPO_MICRO_BATCH_SIZE}"
    --override "grad_accum_steps=${CASPO_GRAD_ACCUM_STEPS}"
    --override "use_gradient_checkpointing=${CASPO_USE_GRADIENT_CHECKPOINTING}"
    --override "prompts_per_step=${PROMPTS_PER_STEP_VAL}"
    --override "reward_workers=${CASPO_REWARD_WORKERS}"
    --override "save_every=${SAVE_EVERY:-250}"
    --override "wandb_mode=${WANDB_MODE:-offline}"
    --override "wandb_project=${WANDB_PROJECT:-caspo-7b-math}"
    --override "output_dir=${OUTDIR}"
    --override "wandb_run_name=7b_math_${RUN_METHOD_TAG}_seed0${RUN_SUFFIX}"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
    COMMON_OVERRIDES+=(--override "max_steps=${MAX_STEPS}")
fi

echo "[7b-fsdp] ${RUN_METHOD_TAG} method=${METHOD} gpus=${GPUS[*]} (world=${NRANK}) out=${OUTDIR}"
echo "[7b-fsdp] mb=${CASPO_MICRO_BATCH_SIZE} accum=${CASPO_GRAD_ACCUM_STEPS} grad_ckpt=${CASPO_USE_GRADIENT_CHECKPOINTING} prompts/step=${PROMPTS_PER_STEP_VAL}"
echo "[7b-fsdp] vllm_util=${CASPO_VLLM_GPU_MEMORY_UTILIZATION} fsdp_cpu_offload=${CASPO_FSDP_CPU_OFFLOAD}"
echo "[7b-fsdp] log=${LOG}"

# Pick a free MASTER_PORT.
MASTER_PORT="${MASTER_PORT:-$(( ((RANDOM<<15) | RANDOM) % 24000 + 30000 ))}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

GPU_CSV=$(IFS=,; echo "${GPUS[*]}")

CUDA_VISIBLE_DEVICES="$GPU_CSV" "$PYTHON_BIN" -u -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NRANK" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m scripts.train_caspo \
    --config "$BASE_CONFIG" \
    "${COMMON_OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}" \
    > "$LOG" 2>&1

echo "[7b-fsdp] DONE ${RUN_METHOD_TAG} - log=${LOG}"
