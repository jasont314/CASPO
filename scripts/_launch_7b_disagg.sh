# Disaggregated 7B launcher body (FSDP trainer on one GPU set,
# vLLM AsyncLLM with TP=N on a disjoint set). Source from a thin
# per-method wrapper.
#
# Required wrapper-set vars:
#   METHOD=ppo|grpo|caspo|vineppo
#   RUN_METHOD_TAG=<output/log tag>
#   TRAIN_GPU_DEFAULT_LIST="0 1 2 3"     # FSDP trainer GPUs
#   ROLLOUT_GPU_DEFAULT_LIST="4 5 6 7"   # vLLM TP GPUs (dedicated)
#   EXTRA_OVERRIDES=(--override key=value ...)
#
# Topology rationale: see docs/disaggregated_topology_plan.md.
# Why disaggregated: VinePPO K=9 MC fan-out (~1296 concurrent
# generations/rank under colocated TP=1) bottlenecks on per-rank KV
# cache. Pooling onto one TP=N vLLM with dedicated rollout GPUs
# raises usable KV cache from ~25 GB/rank to ~270 GB total, removing
# scheduler queueing.
set -eo pipefail

if [[ -z "${METHOD:-}" || -z "${RUN_METHOD_TAG:-}" ]]; then
    echo "[7b-disagg] ERROR: wrapper must set METHOD and RUN_METHOD_TAG"
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

# Trainer (FSDP) GPU set
read -r -a TRAIN_GPUS <<< "${TRAIN_GPU_LIST:-${TRAIN_GPU_DEFAULT_LIST:-0 1 2 3}}"
NRANK=${#TRAIN_GPUS[@]}
if (( NRANK < 1 )); then
    echo "[7b-disagg] ERROR: TRAIN_GPU_LIST must contain at least one GPU id"
    exit 2
fi

# Rollout (vLLM TP) GPU set — dedicated, must be disjoint from TRAIN_GPUS
read -r -a ROLLOUT_GPUS <<< "${ROLLOUT_GPU_LIST:-${ROLLOUT_GPU_DEFAULT_LIST:-4 5 6 7}}"
NROLLOUT=${#ROLLOUT_GPUS[@]}
if (( NROLLOUT < 1 )); then
    echo "[7b-disagg] ERROR: ROLLOUT_GPU_LIST must contain at least one GPU id"
    exit 2
fi

# Disjointness guard
for tg in "${TRAIN_GPUS[@]}"; do
    for rg in "${ROLLOUT_GPUS[@]}"; do
        if [[ "$tg" == "$rg" ]]; then
            echo "[7b-disagg] ERROR: GPU ${tg} appears in both TRAIN and ROLLOUT lists"
            exit 2
        fi
    done
done

RUN_TAG="${RUN_TAG:-}"
RUN_SUFFIX=""
if [[ -n "$RUN_TAG" ]]; then
    RUN_SUFFIX="_${RUN_TAG}"
fi

LOGDIR="$ROOT/deepseekmath7b_math${RUN_SUFFIX}/logs"
mkdir -p "$LOGDIR"

OUTDIR="$ROOT/deepseekmath7b_math_${RUN_METHOD_TAG}${RUN_SUFFIX}"

# vLLM and trainer knobs (env-overridable). Note: under disaggregation
# the rollout GPUs are dedicated, so vllm_gpu_memory_utilization can
# safely run hot. Default 0.85.
CASPO_VLLM_GPU_MEMORY_UTILIZATION="${CASPO_VLLM_GPU_MEMORY_UTILIZATION:-${VLLM_GPU_MEMORY_UTILIZATION:-0.85}}"
# max_num_seqs=2048 default (was 1024): tiny but consistent win on
# disagg TP=4 VinePPO MC. Smoke 2026-04-27:
#   1024 step 2: 245 s (t_value 197)
#   2048 step 2: 242 s (t_value 192)
# We're concurrency-bound at the scheduler layer; doubling the
# pending-decode queue lets vLLM keep more of the K=9 MC fan-out
# in flight per scheduler step. Safe at vllm_util=0.85 fp8 KV
# (no OOM observed); the cost is ~few hundred MB of scheduler
# bookkeeping.
CASPO_VLLM_MAX_NUM_SEQS="${CASPO_VLLM_MAX_NUM_SEQS:-${VLLM_MAX_NUM_SEQS:-2048}}"
CASPO_VLLM_MAX_NUM_BATCHED_TOKENS="${CASPO_VLLM_MAX_NUM_BATCHED_TOKENS:-${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}}"
CASPO_VLLM_ENFORCE_EAGER="${CASPO_VLLM_ENFORCE_EAGER:-false}"

CASPO_MICRO_BATCH_SIZE="${CASPO_MICRO_BATCH_SIZE:-${MICRO_BATCH_SIZE:-2}}"
DEFAULT_ACCUM=$(( 64 / NRANK / CASPO_MICRO_BATCH_SIZE ))
if (( DEFAULT_ACCUM < 1 )); then DEFAULT_ACCUM=1; fi
CASPO_GRAD_ACCUM_STEPS="${CASPO_GRAD_ACCUM_STEPS:-${GRAD_ACCUM_STEPS:-$DEFAULT_ACCUM}}"

CASPO_USE_GRADIENT_CHECKPOINTING="${CASPO_USE_GRADIENT_CHECKPOINTING:-${USE_GRADIENT_CHECKPOINTING:-true}}"
CASPO_FSDP_CPU_OFFLOAD="${CASPO_FSDP_CPU_OFFLOAD:-${FSDP_CPU_OFFLOAD:-false}}"
CASPO_FSDP_SHARDING_STRATEGY="${CASPO_FSDP_SHARDING_STRATEGY:-${FSDP_SHARDING_STRATEGY:-full_shard}}"
CASPO_FSDP_WRAP_BLOCK_GROUP_SIZE="${CASPO_FSDP_WRAP_BLOCK_GROUP_SIZE:-${FSDP_WRAP_BLOCK_GROUP_SIZE:-1}}"
CASPO_ACTIVATION_CHECKPOINTING_MODE="${CASPO_ACTIVATION_CHECKPOINTING_MODE:-${ACTIVATION_CHECKPOINTING_MODE:-off}}"
# reward_workers=16 default (was 4): VinePPO MC fan-out generates
# ~5184 verification calls per outer step, and the math verifier
# (sympy/regex) is CPU-bound. Smoke 2026-04-27 measured t_value
# 222 s → 197 s when bumping rw from 4 to 16 (saves ~25 s/step).
# The host has 40+ cores so 16 workers don't oversubscribe.
CASPO_REWARD_WORKERS="${CASPO_REWARD_WORKERS:-${REWARD_WORKERS:-16}}"
CASPO_LOGPROB_MICRO_BATCH_SIZE="${CASPO_LOGPROB_MICRO_BATCH_SIZE:-${LOGPROB_MICRO_BATCH_SIZE:-8}}"
PROMPTS_PER_STEP_VAL="${PROMPTS_PER_STEP:-64}"

unset VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS
unset MICRO_BATCH_SIZE GRAD_ACCUM_STEPS USE_GRADIENT_CHECKPOINTING
unset FSDP_CPU_OFFLOAD REWARD_WORKERS PROMPTS_PER_STEP LOGPROB_MICRO_BATCH_SIZE
unset FSDP_SHARDING_STRATEGY FSDP_WRAP_BLOCK_GROUP_SIZE
unset ACTIVATION_CHECKPOINTING_MODE

# The rollout-GPU list is communicated to the trainer rank 0 via a
# dedicated env var (CASPO_ROLLOUT_GPU_PHYSICAL_IDS). Rank 0 reads
# this and tells VLLMRolloutEngine which physical GPUs to bind for
# its TP=N workers. The trainer's CUDA_VISIBLE_DEVICES on rank 0 is
# extended to include both its trainer GPU and the rollout GPUs (the
# rest of the world doesn't need them visible).
ROLLOUT_GPUS_CSV=$(IFS=,; echo "${ROLLOUT_GPUS[*]}")

# Weight-sync backend default. 'nccl' is the production path (Phase
# 4b): trainer rank 0 + N vLLM workers join a side
# PyNcclCommunicator and broadcast packed weight buffers over
# NVLink, ~0.2 s/sync at 7B vs ~28 s/sync for the checkpoint
# (save_pretrained + reload_weights) path. 'checkpoint' is kept as
# an opt-in fallback for environments where the side NCCL group
# can't form.
DISAGG_WEIGHT_SYNC_BACKEND="${DISAGG_WEIGHT_SYNC_BACKEND:-${WEIGHT_SYNC_BACKEND:-nccl}}"

OVERRIDES=(
    --override "method=${METHOD}"
    --override rollout_backend=vllm
    --override "vllm_weight_sync_backend=${DISAGG_WEIGHT_SYNC_BACKEND}"
    --override vllm_disaggregated=true
    --override "vllm_disaggregated_tp=${NROLLOUT}"
    --override "vllm_gpu_memory_utilization=${CASPO_VLLM_GPU_MEMORY_UTILIZATION}"
    --override "vllm_enforce_eager=${CASPO_VLLM_ENFORCE_EAGER}"
    --override "vllm_max_num_seqs=${CASPO_VLLM_MAX_NUM_SEQS}"
    --override "vllm_max_num_batched_tokens=${CASPO_VLLM_MAX_NUM_BATCHED_TOKENS}"
    --override distributed_backend=fsdp
    --override "fsdp_sharding_strategy=${CASPO_FSDP_SHARDING_STRATEGY}"
    --override "fsdp_wrap_block_group_size=${CASPO_FSDP_WRAP_BLOCK_GROUP_SIZE}"
    --override fsdp_use_orig_params=true
    --override fsdp_auto_wrap=true
    --override "fsdp_cpu_offload=${CASPO_FSDP_CPU_OFFLOAD}"
    --override "micro_batch_size=${CASPO_MICRO_BATCH_SIZE}"
    --override "grad_accum_steps=${CASPO_GRAD_ACCUM_STEPS}"
    --override "use_gradient_checkpointing=${CASPO_USE_GRADIENT_CHECKPOINTING}"
    --override "activation_checkpointing_mode=${CASPO_ACTIVATION_CHECKPOINTING_MODE}"
    --override "logprob_micro_batch_size=${CASPO_LOGPROB_MICRO_BATCH_SIZE}"
    --override "prompts_per_step=${PROMPTS_PER_STEP_VAL}"
    --override "reward_workers=${CASPO_REWARD_WORKERS}"
    --override "save_every=${SAVE_EVERY:-250}"
    --override "wandb_mode=${WANDB_MODE:-offline}"
    --override "wandb_project=${WANDB_PROJECT:-caspo-7b-math}"
    --override "output_dir=${OUTDIR}"
    --override "wandb_run_name=7b_math_${RUN_METHOD_TAG}_seed0${RUN_SUFFIX}_disagg"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
    OVERRIDES+=(--override "max_steps=${MAX_STEPS}")
fi

MASTER_PORT="${MASTER_PORT:-$(( ((RANDOM<<15) | RANDOM) % 24000 + 30000 ))}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

# NCCL weight-sync side-channel port. Distinct from MASTER_PORT (used
# by the FSDP process group) so the two PGs don't collide.
WEIGHT_SYNC_PORT="${WEIGHT_SYNC_PORT:-$(( MASTER_PORT + 1 ))}"

echo "[7b-disagg] ${RUN_METHOD_TAG} method=${METHOD}"
echo "[7b-disagg] trainer FSDP=${NRANK} on GPUs ${TRAIN_GPUS[*]}"
echo "[7b-disagg] rollout vLLM TP=${NROLLOUT} on GPUs ${ROLLOUT_GPUS[*]}"
echo "[7b-disagg] mb=${CASPO_MICRO_BATCH_SIZE} accum=${CASPO_GRAD_ACCUM_STEPS} grad_ckpt=${CASPO_USE_GRADIENT_CHECKPOINTING} prompts/step=${PROMPTS_PER_STEP_VAL}"
echo "[7b-disagg] vllm_util=${CASPO_VLLM_GPU_MEMORY_UTILIZATION} max_num_seqs=${CASPO_VLLM_MAX_NUM_SEQS} max_num_batched_tokens=${CASPO_VLLM_MAX_NUM_BATCHED_TOKENS}"
echo "[7b-disagg] rdzv=${MASTER_ADDR}:${MASTER_PORT}  weight_sync_port=${WEIGHT_SYNC_PORT}"
echo "[7b-disagg] out=${OUTDIR}"

PIDS=()
cleanup_children() {
    local code=$?
    echo "[7b-disagg] caught signal/exit; killing rank PIDs: ${PIDS[*]:-none}"
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
    local trainer_gpu="$2"
    local log="$3"
    # Rank 0 also sees the rollout GPUs so AsyncLLM(tp=N) can bind to
    # them. Ranks 1..N-1 see only their trainer GPU.
    local visible_csv
    if (( rank == 0 )); then
        visible_csv="${trainer_gpu},${ROLLOUT_GPUS_CSV}"
    else
        visible_csv="${trainer_gpu}"
    fi
    echo "[7b-disagg] launch rank=${rank} trainer_gpu=${trainer_gpu} CUDA_VISIBLE_DEVICES=${visible_csv} log=${log}"
    CUDA_VISIBLE_DEVICES="$visible_csv" \
    RANK="$rank" \
    LOCAL_RANK=0 \
    WORLD_SIZE="$NRANK" \
    LOCAL_WORLD_SIZE=1 \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
    CASPO_ROLLOUT_GPU_PHYSICAL_IDS="$ROLLOUT_GPUS_CSV" \
    CASPO_WEIGHT_SYNC_MASTER_ADDR="$MASTER_ADDR" \
    CASPO_WEIGHT_SYNC_MASTER_PORT="$WEIGHT_SYNC_PORT" \
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u -m scripts.train_caspo \
        --config "$BASE_CONFIG" \
        "${OVERRIDES[@]}" \
        "${EXTRA_OVERRIDES[@]}" \
        > "$log" 2>&1 &
    PIDS+=("$!")
}

for ((i=0; i<NRANK; i++)); do
    LOG_RANK="$LOGDIR/phase2_${RUN_METHOD_TAG}_rank${i}.log"
    launch_rank "$i" "${TRAIN_GPUS[$i]}" "$LOG_RANK"
done

remaining=${#PIDS[@]}
while (( remaining > 0 )); do
    if wait -n; then
        remaining=$((remaining - 1))
    else
        status=$?
        echo "[7b-disagg] ERROR: rank process failed with status ${status}"
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done
        wait || true
        echo "[7b-disagg] rank logs in: $LOGDIR"
        exit "$status"
    fi
done

trap - INT TERM
echo "[7b-disagg] DONE ${RUN_METHOD_TAG}"
