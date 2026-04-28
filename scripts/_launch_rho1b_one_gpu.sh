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
ROOT="${ROOT:-/mnt/nvme_tmp2/jason_caspo}"
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

CASPO_VLLM_GPU_MEMORY_UTILIZATION="${CASPO_VLLM_GPU_MEMORY_UTILIZATION:-${VLLM_GPU_MEMORY_UTILIZATION:-0.30}}"
CASPO_VLLM_MULTI_SAMPLE_MODE="${CASPO_VLLM_MULTI_SAMPLE_MODE:-${VLLM_MULTI_SAMPLE_MODE:-auto}}"
CASPO_VLLM_MAX_NUM_SEQS="${CASPO_VLLM_MAX_NUM_SEQS:-${VLLM_MAX_NUM_SEQS:-256}}"
CASPO_VLLM_MAX_NUM_BATCHED_TOKENS="${CASPO_VLLM_MAX_NUM_BATCHED_TOKENS:-${VLLM_MAX_NUM_BATCHED_TOKENS:-}}"

# Trainer batching: defaults updated 2026-04-28 for fp32 master path.
# Original Pareto sweep (bf16, Apr 2026) found mb=8, accum=8 optimal —
# but fp32 master adds ~6 GB persistent at 1B (policy fp32 + AdamW
# fp32). With vLLM at u=0.30 (24 GB) and ~50 GB trainer working set
# at mb=8, multi-step runs hit allocator-fragmentation OOM by step
# 3-4 even though step 1 fits. Dropping to mb=4 (accum=16 to keep
# the 64-response PPO minibatch) costs only ~5-10% step time vs
# mb=8 but is stable across long runs at fp32 master.
# Verified: 1B GRPO/CASPO/CASPO-frozen/PPO+critic all complete 4-step
# smokes at mb=4. PPO+critic still shows residual cumulative
# slowdown (~+6s/step past step 2) — known limitation of
# single-GPU + colocated vLLM at fp32 master, not OOM.
# Override env: ``MICRO_BATCH_SIZE=8 GRAD_ACCUM_STEPS=8`` to revert
# (only safe if you also disable fp32 master).
CASPO_MICRO_BATCH_SIZE="${CASPO_MICRO_BATCH_SIZE:-${MICRO_BATCH_SIZE:-4}}"
CASPO_GRAD_ACCUM_STEPS="${CASPO_GRAD_ACCUM_STEPS:-${GRAD_ACCUM_STEPS:-16}}"
CASPO_USE_GRADIENT_CHECKPOINTING="${CASPO_USE_GRADIENT_CHECKPOINTING:-${USE_GRADIENT_CHECKPOINTING:-false}}"

# Round 2 knobs (Apr 2026):
#   reward_workers — ProcessPoolExecutor for SymPy verifier; default 4 mirrors
#     the cfg dataclass default. Lower to 1 for deterministic/serial scoring.
#   compile — torch.compile on policy + value_model.phi. Currently OFF by
#     default: with mode="default" it runs (no CUDA-graph crash) but variable
#     seq lengths trigger ~8 recompiles before dynamo gives up; net win
#     unclear at our shape distribution. Try `CASPO_COMPILE=true` to opt in.
CASPO_REWARD_WORKERS="${CASPO_REWARD_WORKERS:-${REWARD_WORKERS:-4}}"
CASPO_COMPILE="${CASPO_COMPILE:-${COMPILE:-false}}"

# These names are launcher aliases, not native vLLM environment variables.
unset VLLM_GPU_MEMORY_UTILIZATION
unset VLLM_MULTI_SAMPLE_MODE
unset VLLM_MAX_NUM_SEQS
unset VLLM_MAX_NUM_BATCHED_TOKENS
unset MICRO_BATCH_SIZE
unset GRAD_ACCUM_STEPS
unset USE_GRADIENT_CHECKPOINTING
unset REWARD_WORKERS
unset COMPILE

COMMON_OVERRIDES=(
    --override "method=${METHOD}"
    --override rollout_backend=vllm
    --override vllm_weight_sync_backend=ipc
    --override "vllm_gpu_memory_utilization=${CASPO_VLLM_GPU_MEMORY_UTILIZATION}"
    --override vllm_enforce_eager=false
    --override "vllm_multi_sample_mode=${CASPO_VLLM_MULTI_SAMPLE_MODE}"
    --override "vllm_max_num_seqs=${CASPO_VLLM_MAX_NUM_SEQS}"
    --override "micro_batch_size=${CASPO_MICRO_BATCH_SIZE}"
    --override "grad_accum_steps=${CASPO_GRAD_ACCUM_STEPS}"
    --override "use_gradient_checkpointing=${CASPO_USE_GRADIENT_CHECKPOINTING}"
    --override "reward_workers=${CASPO_REWARD_WORKERS}"
    --override "compile=${CASPO_COMPILE}"
    --override "save_every=${SAVE_EVERY:-250}"
    --override "eval_every=${EVAL_EVERY:-${SAVE_EVERY:-250}}"
    --override "wandb_mode=${WANDB_MODE:-offline}"
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
echo "[rho1b-onegpu] mb=${CASPO_MICRO_BATCH_SIZE} accum=${CASPO_GRAD_ACCUM_STEPS} grad_ckpt=${CASPO_USE_GRADIENT_CHECKPOINTING} reward_workers=${CASPO_REWARD_WORKERS} compile=${CASPO_COMPILE}"
echo "[rho1b-onegpu] log=${LOG}"

CUDA_VISIBLE_DEVICES="$SELECTED_GPU" "$PYTHON_BIN" -u -m scripts.train_caspo \
    --config "$BASE_CONFIG" \
    "${COMMON_OVERRIDES[@]}" \
    "${EXTRA_OVERRIDES[@]}" \
    > "$LOG" 2>&1

echo "[rho1b-onegpu] DONE ${RUN_METHOD_TAG} - log=${LOG}"
