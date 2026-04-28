# Colocated TP=8 launcher body. Every GPU runs BOTH a trainer
# FSDP-rank AND a vLLM TP-rank. Source from a thin per-method
# wrapper.
#
# Required wrapper-set vars:
#   METHOD=ppo|grpo|caspo|vineppo
#   RUN_METHOD_TAG=<output/log tag>
#   GPU_DEFAULT_LIST="0 1 2 3 4 5 6 7"  # all 8
#   EXTRA_OVERRIDES=(--override key=value ...)
#
# Topology rationale: see docs/disaggregated_topology_plan.md (Phase
# 5b notes). Phase 5a's disagg FSDP=4 + TP=4 was 12-20% faster than
# colocated TP=1, but t_value was still 220 s/step — pooled TP=4 KV
# does not beat 4× TP=1 engines on the K=9 MC fan-out due to
# per-layer NCCL all-reduce overhead. TP=8 single engine should
# (i) double the KV pool, (ii) halve the parallel all-reduce ranks
# walk (8-way ring is slower per call than 4-way but amortizes
# better over the bigger KV concurrency), and (iii) preserve
# paper-faithful global minibatch with world=8 / mb=2 / accum=4.
#
# Memory math (per GPU, 80 GB H100):
#   trainer FSDP=8 policy shard:     1.75 GB
#   trainer FSDP=8 ref shard:        1.75 GB
#   Adam state (m+v+master):        10.50 GB
#   trainer activations (peak):    ~10.00 GB  (mb=2, with grad ckpt)
#   vLLM TP=8 weight shard:          1.75 GB
#   vLLM KV cache (util=0.45):     ~34.00 GB
#   subtotal trainer side:         ~26.00 GB
#   subtotal vLLM side:            ~36.00 GB
#   peak total per GPU:            ~62.00 GB  (fits 80 GB with head)
set -eo pipefail

if [[ -z "${METHOD:-}" || -z "${RUN_METHOD_TAG:-}" ]]; then
    echo "[7b-tp8] ERROR: wrapper must set METHOD and RUN_METHOD_TAG"
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
ROOT="${ROOT:-/mnt/nvme_tmp2/jason_caspo}"
BASE_CONFIG="${BASE_CONFIG:-configs/caspo_deepseekmath7b_math.yaml}"

read -r -a GPUS <<< "${GPU_LIST:-${GPU_DEFAULT_LIST:-0 1 2 3 4 5 6 7}}"
NRANK=${#GPUS[@]}
if (( NRANK < 2 )); then
    echo "[7b-tp8] ERROR: TP=8 colocated requires at least 2 GPUs (got ${NRANK})"
    exit 2
fi
GPUS_CSV=$(IFS=,; echo "${GPUS[*]}")

# Trainer FSDP world == TP world. mb=2 + accum=(64 / NRANK / 2)
# preserves paper-faithful global = 64. At NRANK=8: mb=2, accum=4.
CASPO_VLLM_GPU_MEMORY_UTILIZATION="${CASPO_VLLM_GPU_MEMORY_UTILIZATION:-${VLLM_GPU_MEMORY_UTILIZATION:-0.45}}"
CASPO_VLLM_MAX_NUM_SEQS="${CASPO_VLLM_MAX_NUM_SEQS:-${VLLM_MAX_NUM_SEQS:-1024}}"
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
CASPO_REWARD_WORKERS="${CASPO_REWARD_WORKERS:-${REWARD_WORKERS:-4}}"
CASPO_LOGPROB_MICRO_BATCH_SIZE="${CASPO_LOGPROB_MICRO_BATCH_SIZE:-${LOGPROB_MICRO_BATCH_SIZE:-8}}"
PROMPTS_PER_STEP_VAL="${PROMPTS_PER_STEP:-64}"

unset VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS
unset MICRO_BATCH_SIZE GRAD_ACCUM_STEPS USE_GRADIENT_CHECKPOINTING
unset FSDP_CPU_OFFLOAD REWARD_WORKERS PROMPTS_PER_STEP LOGPROB_MICRO_BATCH_SIZE
unset FSDP_SHARDING_STRATEGY FSDP_WRAP_BLOCK_GROUP_SIZE
unset ACTIVATION_CHECKPOINTING_MODE

# Weight sync backend. NCCL is the only sane choice at TP=8 (the
# checkpoint backend's 14 GB save_pretrained → reload at every step
# costs ~28 s/step vs ~1-2 s for NCCL bcast over NVLink).
TP8_WEIGHT_SYNC_BACKEND="${TP8_WEIGHT_SYNC_BACKEND:-${WEIGHT_SYNC_BACKEND:-nccl}}"

OVERRIDES=(
    --override "method=${METHOD}"
    --override rollout_backend=vllm
    --override "vllm_weight_sync_backend=${TP8_WEIGHT_SYNC_BACKEND}"
    --override vllm_disaggregated=true
    --override "vllm_disaggregated_tp=${NRANK}"
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
    --override "eval_every=${EVAL_EVERY:-${SAVE_EVERY:-250}}"
    --override "wandb_mode=${WANDB_MODE:-offline}"
    --override "wandb_project=${WANDB_PROJECT:-caspo-7b-math}"
    --override "output_dir=${ROOT}/deepseekmath7b_math_${RUN_METHOD_TAG}${RUN_TAG:+_${RUN_TAG}}"
    --override "wandb_run_name=7b_math_${RUN_METHOD_TAG}_seed0${RUN_TAG:+_${RUN_TAG}}_tp8"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
    OVERRIDES+=(--override "max_steps=${MAX_STEPS}")
fi

LOGDIR="$ROOT/deepseekmath7b_math${RUN_TAG:+_${RUN_TAG}}/logs"
mkdir -p "$LOGDIR"

MASTER_PORT="${MASTER_PORT:-$(( ((RANDOM<<15) | RANDOM) % 24000 + 30000 ))}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
WEIGHT_SYNC_PORT="${WEIGHT_SYNC_PORT:-$(( MASTER_PORT + 1 ))}"

echo "[7b-tp8] ${RUN_METHOD_TAG} method=${METHOD}"
echo "[7b-tp8] FSDP=${NRANK} + vLLM TP=${NRANK} colocated on GPUs ${GPUS[*]}"
echo "[7b-tp8] mb=${CASPO_MICRO_BATCH_SIZE} accum=${CASPO_GRAD_ACCUM_STEPS} grad_ckpt=${CASPO_USE_GRADIENT_CHECKPOINTING}"
echo "[7b-tp8] vllm_util=${CASPO_VLLM_GPU_MEMORY_UTILIZATION} max_num_seqs=${CASPO_VLLM_MAX_NUM_SEQS} sync=${TP8_WEIGHT_SYNC_BACKEND}"
echo "[7b-tp8] rdzv=${MASTER_ADDR}:${MASTER_PORT}  weight_sync_port=${WEIGHT_SYNC_PORT}"

PIDS=()
cleanup_children() {
    local code=$?
    echo "[7b-tp8] caught signal/exit; killing rank PIDs: ${PIDS[*]:-none}"
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
    local log="$2"
    # Every rank sees all GPUs (TP=8 colocated requires it). Each
    # trainer rank pins to cuda:LOCAL_RANK = its own GPU position
    # in CUDA_VISIBLE_DEVICES. Rank 0 also drives the vLLM AsyncLLM,
    # whose worker subprocesses spawn pinned to cuda:0..cuda:N-1.
    echo "[7b-tp8] launch rank=${rank} CUDA_VISIBLE_DEVICES=${GPUS_CSV} log=${log}"
    CUDA_VISIBLE_DEVICES="$GPUS_CSV" \
    RANK="$rank" \
    LOCAL_RANK="$rank" \
    WORLD_SIZE="$NRANK" \
    LOCAL_WORLD_SIZE="$NRANK" \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
    CASPO_ROLLOUT_GPU_PHYSICAL_IDS="$GPUS_CSV" \
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
    launch_rank "$i" "$LOG_RANK"
done

remaining=${#PIDS[@]}
while (( remaining > 0 )); do
    if wait -n; then
        remaining=$((remaining - 1))
    else
        status=$?
        echo "[7b-tp8] ERROR: rank process failed with status ${status}"
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done
        wait || true
        echo "[7b-tp8] rank logs in: $LOGDIR"
        exit "$status"
    fi
done

trap - INT TERM
echo "[7b-tp8] DONE ${RUN_METHOD_TAG}"
