# Shared FSDP-based 7B launcher body. Source this from a thin per-method wrapper.
#
# Per-rank manual bash spawn (no torchrun) so each Python process has fully
# independent CUDA / torch.distributed / multiprocessing state. This matches
# the rho-1B DDP-2 launcher pattern that's known to work with rank-local
# vLLM EngineCore subprocesses.
#
# Required wrapper-set vars:
#   METHOD=ppo|grpo|caspo|vineppo
#   RUN_METHOD_TAG=<output/log tag>
#   GPU_DEFAULT_LIST="0 1 2 3"   # default GPU set (4 or 8 depending on method)
#   EXTRA_OVERRIDES=(--override key=value ...)
#
# Optional env knobs: see /home/jason/experiment/CASPO/configs/caspo_deepseekmath7b_math.yaml
# and the env var block below.
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
ROOT="${ROOT:-/mnt/nvme_tmp2/jason_caspo}"
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

# vLLM and trainer knobs (overridable via env).
CASPO_VLLM_GPU_MEMORY_UTILIZATION="${CASPO_VLLM_GPU_MEMORY_UTILIZATION:-${VLLM_GPU_MEMORY_UTILIZATION:-0.30}}"
CASPO_VLLM_MAX_NUM_SEQS="${CASPO_VLLM_MAX_NUM_SEQS:-${VLLM_MAX_NUM_SEQS:-256}}"
CASPO_VLLM_ENFORCE_EAGER="${CASPO_VLLM_ENFORCE_EAGER:-false}"

CASPO_MICRO_BATCH_SIZE="${CASPO_MICRO_BATCH_SIZE:-${MICRO_BATCH_SIZE:-2}}"
DEFAULT_ACCUM=$(( 64 / NRANK / CASPO_MICRO_BATCH_SIZE ))
if (( DEFAULT_ACCUM < 1 )); then DEFAULT_ACCUM=1; fi
CASPO_GRAD_ACCUM_STEPS="${CASPO_GRAD_ACCUM_STEPS:-${GRAD_ACCUM_STEPS:-$DEFAULT_ACCUM}}"

CASPO_USE_GRADIENT_CHECKPOINTING="${CASPO_USE_GRADIENT_CHECKPOINTING:-${USE_GRADIENT_CHECKPOINTING:-true}}"
CASPO_FSDP_CPU_OFFLOAD="${CASPO_FSDP_CPU_OFFLOAD:-${FSDP_CPU_OFFLOAD:-false}}"
# Hybrid-shard (HSDP) is the better default for 7B at world>=4: shards
# within a 2-rank pair and replicates across, ~33% fewer AG/RS hops vs
# full_shard. Override via CASPO_FSDP_SHARDING_STRATEGY=full_shard if you
# need the legacy fully-sharded layout (e.g. tight-VRAM single-node runs).
CASPO_FSDP_SHARDING_STRATEGY="${CASPO_FSDP_SHARDING_STRATEGY:-${FSDP_SHARDING_STRATEGY:-hybrid_shard}}"
# Coarser FSDP wrap: group every N transformer blocks into one FSDP unit.
# Default 1 = per-block wrap (legacy behavior). At 7B (32 blocks) setting
# this to 4 cuts backward reduce-scatter calls 32->8 and roughly doubles
# per-collective payload from ~440 MB to ~1.7 GB for better NVLink BW.
CASPO_FSDP_WRAP_BLOCK_GROUP_SIZE="${CASPO_FSDP_WRAP_BLOCK_GROUP_SIZE:-${FSDP_WRAP_BLOCK_GROUP_SIZE:-1}}"
# Activation checkpointing mode: "off" / "full" / "selective". When non-"off"
# overrides cfg.use_gradient_checkpointing. "selective" recomputes only the
# attention block (cheap with FA3, ~10% layer FLOPs) and keeps MLP
# activations live — saves recompute cost vs "full" while still freeing
# enough activation memory at 7B mb=2.
CASPO_ACTIVATION_CHECKPOINTING_MODE="${CASPO_ACTIVATION_CHECKPOINTING_MODE:-${ACTIVATION_CHECKPOINTING_MODE:-off}}"
CASPO_REWARD_WORKERS="${CASPO_REWARD_WORKERS:-${REWARD_WORKERS:-4}}"
# Logprob micro-batch for the no-grad rescore + ref-logprob passes.
# Default 8: bigger than mb=2 (the trainable forward) because there's no
# autograd graph to keep, so larger forwards amortize python/launch
# overhead. Verified at rho-1B DDP-2 (=16 there). Override via env.
CASPO_LOGPROB_MICRO_BATCH_SIZE="${CASPO_LOGPROB_MICRO_BATCH_SIZE:-${LOGPROB_MICRO_BATCH_SIZE:-4}}"
PROMPTS_PER_STEP_VAL="${PROMPTS_PER_STEP:-64}"

unset VLLM_GPU_MEMORY_UTILIZATION VLLM_MAX_NUM_SEQS
unset MICRO_BATCH_SIZE GRAD_ACCUM_STEPS USE_GRADIENT_CHECKPOINTING
unset FSDP_CPU_OFFLOAD REWARD_WORKERS PROMPTS_PER_STEP LOGPROB_MICRO_BATCH_SIZE
unset FSDP_SHARDING_STRATEGY
unset ACTIVATION_CHECKPOINTING_MODE
unset FSDP_WRAP_BLOCK_GROUP_SIZE

OVERRIDES=(
    --override "method=${METHOD}"
    --override rollout_backend=vllm
    --override vllm_weight_sync_backend=ipc
    --override "vllm_gpu_memory_utilization=${CASPO_VLLM_GPU_MEMORY_UTILIZATION}"
    --override "vllm_enforce_eager=${CASPO_VLLM_ENFORCE_EAGER}"
    --override "vllm_max_num_seqs=${CASPO_VLLM_MAX_NUM_SEQS}"
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
    --override "save_every=${SAVE_EVERY:-200}"
    --override "eval_every=${EVAL_EVERY:-${SAVE_EVERY:-200}}"
    --override "wandb_mode=${WANDB_MODE:-offline}"
    --override "wandb_project=${WANDB_PROJECT:-caspo-7b-math}"
    --override "output_dir=${OUTDIR}"
    --override "wandb_run_name=7b_math_${RUN_METHOD_TAG}_seed0${RUN_SUFFIX}"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
    OVERRIDES+=(--override "max_steps=${MAX_STEPS}")
fi

# Pick a fixed MASTER_PORT for this run. All ranks share this rdzv endpoint.
MASTER_PORT="${MASTER_PORT:-$(( ((RANDOM<<15) | RANDOM) % 24000 + 30000 ))}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

echo "[7b-fsdp] ${RUN_METHOD_TAG} method=${METHOD} gpus=${GPUS[*]} (world=${NRANK}) out=${OUTDIR}"
echo "[7b-fsdp] mb=${CASPO_MICRO_BATCH_SIZE} accum=${CASPO_GRAD_ACCUM_STEPS} grad_ckpt=${CASPO_USE_GRADIENT_CHECKPOINTING} prompts/step=${PROMPTS_PER_STEP_VAL}"
echo "[7b-fsdp] vllm_util=${CASPO_VLLM_GPU_MEMORY_UTILIZATION} fsdp_cpu_offload=${CASPO_FSDP_CPU_OFFLOAD} fsdp_shard=${CASPO_FSDP_SHARDING_STRATEGY} fsdp_wrap_group=${CASPO_FSDP_WRAP_BLOCK_GROUP_SIZE}"
echo "[7b-fsdp] rdzv=${MASTER_ADDR}:${MASTER_PORT}"

PIDS=()
cleanup_children() {
    local code=$?
    echo "[7b-fsdp] caught signal/exit; killing rank PIDs: ${PIDS[*]:-none}"
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
    echo "[7b-fsdp] launch rank=${rank} physical_gpu=${gpu} log=${log}"
    CUDA_VISIBLE_DEVICES="$gpu" \
    RANK="$rank" \
    LOCAL_RANK=0 \
    WORLD_SIZE="$NRANK" \
    LOCAL_WORLD_SIZE=1 \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
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
    launch_rank "$i" "${GPUS[$i]}" "$LOG_RANK"
done

remaining=${#PIDS[@]}
while (( remaining > 0 )); do
    if wait -n; then
        remaining=$((remaining - 1))
    else
        status=$?
        echo "[7b-fsdp] ERROR: rank process failed with status ${status}"
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done
        wait || true
        echo "[7b-fsdp] rank logs in: $LOGDIR"
        exit "$status"
    fi
done

trap - INT TERM
echo "[7b-fsdp] DONE ${RUN_METHOD_TAG}"
