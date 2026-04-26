#!/usr/bin/env bash
# Per-rank manual bash launcher for 7B IPVRM value-model training (Phase 1).
#
# Mirrors scripts/_launch_7b_fsdp.sh but for `scripts/train_value.py`. Value
# training does NOT use vLLM — it's a supervised BCE-margin pass on the
# ``value_data.pt`` blob produced by ``collect_value_data.py``. The 7B value
# model (phi trainable + ref frozen + AdamW fp32 m+v) does not fit on a
# single H100, so we FSDP-shard phi across 4 GPUs by default.
#
# Memory back-of-envelope (4-GPU full_shard, no offload, bf16 phi/ref):
#   phi   bf16, sharded: 14 GB / 4 = 3.5 GB / rank
#   AdamW fp32 m+v, sharded: 56 GB / 4 = 14 GB / rank
#   ref   bf16, full-resident (frozen, NOT FSDP-wrapped): 14 GB / rank
#   activations + value_data blob + headroom: ~30 GB / rank
#   total / rank ≈ 60 GB << H100 80 GB. Drops further on 8 GPUs.
#
# Usage:
#   bash scripts/_launch_7b_value_train.sh
#
# Env knobs (all optional):
#   GPU_LIST="0 1 2 3"           # default 4 GPUs
#   BASE_CONFIG=...              # path to config; defaults to 7b math
#   OUTDIR=...                   # output dir override
#   VALUE_DATA_PATH=...          # value_data.pt path (cfg default used otherwise)
#   VALUE_MICRO_BATCH_SIZE=1
#   VALUE_GRAD_ACCUM_STEPS=8
#   VALUE_MAX_EPOCHS=3
#   VALUE_LR=5e-7
#   FSDP_CPU_OFFLOAD=false
#   USE_GRADIENT_CHECKPOINTING=true
#   MASTER_ADDR=127.0.0.1
#   MASTER_PORT=<random>
#   RUN_TAG=<suffix>             # adds _${RUN_TAG} to logdir / outdir
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate scalable

export HF_HOME="${HF_HOME:-/mnt/nvme_tmp/jason_caspo/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/nvme_tmp/jason_caspo/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/nvme_tmp/jason_caspo/hf_cache}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Sources VLLM_WORKER_MULTIPROC_METHOD=spawn etc. — value training doesn't use
# vLLM, but sourcing keeps env parity (NCCL knobs, OMP caps, allocator config).
source ./scripts/perf_env.sh

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/scalable/bin/python}"
ROOT="${ROOT:-/mnt/nvme_tmp/jason_caspo}"
BASE_CONFIG="${BASE_CONFIG:-configs/caspo_deepseekmath7b_math.yaml}"

read -r -a GPUS <<< "${GPU_LIST:-0 1 2 3}"
NRANK=${#GPUS[@]}
if (( NRANK < 1 )); then
    echo "[7b-value] ERROR: GPU_LIST must contain at least one GPU id"
    exit 2
fi

RUN_TAG="${RUN_TAG:-}"
RUN_SUFFIX=""
if [[ -n "$RUN_TAG" ]]; then
    RUN_SUFFIX="_${RUN_TAG}"
fi

LOGDIR="$ROOT/deepseekmath7b_math${RUN_SUFFIX}/logs"
mkdir -p "$LOGDIR"

OUTDIR="${OUTDIR:-$ROOT/deepseekmath7b_math${RUN_SUFFIX}}"

VALUE_MICRO_BATCH_SIZE="${VALUE_MICRO_BATCH_SIZE:-1}"
VALUE_GRAD_ACCUM_STEPS="${VALUE_GRAD_ACCUM_STEPS:-8}"
VALUE_MAX_EPOCHS="${VALUE_MAX_EPOCHS:-3}"
VALUE_LR="${VALUE_LR:-5e-7}"
FSDP_CPU_OFFLOAD="${FSDP_CPU_OFFLOAD:-false}"
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-true}"
# VALUE_EVAL_EVERY=50 (matches cfg default). Validation is now FSDP-collective:
# every rank evaluates its own val-row shard, the per-batch losses are
# all-reduced, and best-step / early-stop decisions advance identically on
# every rank. No more rank-0-only deadlock.
VALUE_EVAL_EVERY="${VALUE_EVAL_EVERY:-50}"

OVERRIDES=(
    --override distributed_backend=fsdp
    --override fsdp_sharding_strategy=full_shard
    --override fsdp_use_orig_params=true
    --override fsdp_auto_wrap=true
    --override "fsdp_cpu_offload=${FSDP_CPU_OFFLOAD}"
    --override "value_micro_batch_size=${VALUE_MICRO_BATCH_SIZE}"
    --override "value_grad_accum_steps=${VALUE_GRAD_ACCUM_STEPS}"
    --override "value_max_epochs=${VALUE_MAX_EPOCHS}"
    --override "value_lr=${VALUE_LR}"
    --override "value_eval_every=${VALUE_EVAL_EVERY}"
    --override "use_gradient_checkpointing=${USE_GRADIENT_CHECKPOINTING}"
    --override "output_dir=${OUTDIR}"
)
if [[ -n "${VALUE_DATA_PATH:-}" ]]; then
    OVERRIDES+=(--override "value_data_path=${VALUE_DATA_PATH}")
fi
if [[ -n "${WANDB_MODE:-}" ]]; then
    OVERRIDES+=(--override "wandb_mode=${WANDB_MODE}")
fi

MASTER_PORT="${MASTER_PORT:-$(( ((RANDOM<<15) | RANDOM) % 24000 + 30000 ))}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

echo "[7b-value] gpus=${GPUS[*]} (world=${NRANK}) out=${OUTDIR}"
echo "[7b-value] mb=${VALUE_MICRO_BATCH_SIZE} accum=${VALUE_GRAD_ACCUM_STEPS} epochs=${VALUE_MAX_EPOCHS} lr=${VALUE_LR}"
echo "[7b-value] grad_ckpt=${USE_GRADIENT_CHECKPOINTING} fsdp_cpu_offload=${FSDP_CPU_OFFLOAD}"
echo "[7b-value] rdzv=${MASTER_ADDR}:${MASTER_PORT}"

PIDS=()
cleanup_children() {
    local code=$?
    echo "[7b-value] caught signal/exit; killing rank PIDs: ${PIDS[*]:-none}"
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
    echo "[7b-value] launch rank=${rank} physical_gpu=${gpu} log=${log}"
    CUDA_VISIBLE_DEVICES="$gpu" \
    RANK="$rank" \
    LOCAL_RANK=0 \
    WORLD_SIZE="$NRANK" \
    LOCAL_WORLD_SIZE=1 \
    MASTER_ADDR="$MASTER_ADDR" \
    MASTER_PORT="$MASTER_PORT" \
    PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -u -m scripts.train_value \
        --config "$BASE_CONFIG" \
        "${OVERRIDES[@]}" \
        > "$log" 2>&1 &
    PIDS+=("$!")
}

for ((i=0; i<NRANK; i++)); do
    LOG_RANK="$LOGDIR/value_train_rank${i}.log"
    launch_rank "$i" "${GPUS[$i]}" "$LOG_RANK"
done

remaining=${#PIDS[@]}
while (( remaining > 0 )); do
    if wait -n; then
        remaining=$((remaining - 1))
    else
        status=$?
        echo "[7b-value] ERROR: rank process failed with status ${status}"
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done
        wait || true
        echo "[7b-value] rank logs in: $LOGDIR"
        exit "$status"
    fi
done

trap - INT TERM
echo "[7b-value] DONE value train"
