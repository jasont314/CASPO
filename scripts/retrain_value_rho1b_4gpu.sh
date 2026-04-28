#!/usr/bin/env bash
# 4-GPU V_φ retrain pipeline for Rho-1B-MATH:
#   1) shard-parallel collect_value_data on 4 GPUs (current verifier + BOS)
#   2) merge per-shard .pt files into one
#   3) FSDP=4 train_value
#   4) sanity-check the new V_φ on a 1-step rollout
#
# Writes to a SEPARATE output dir so the existing Apr 25 V_φ baseline at
# /mnt/nvme_tmp/jason_caspo/caspo_rho1b_math/ stays intact for comparison.
#
# Env knobs (all optional):
#   GPU_LIST="4 5 6 7"
#   OUT_ROOT=/mnt/nvme_tmp4/jason_caspo/caspo_rho1b_math_v2
#   BASE_CONFIG=configs/caspo_rho1b_math.yaml
#   VALUE_LR=5e-7  VALUE_MAX_EPOCHS=3
#   SKIP_COLLECT=true / SKIP_MERGE=true / SKIP_TRAIN=true / SKIP_VALIDATE=true
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate scalable
export HF_HOME="${HF_HOME:-/mnt/nvme_tmp/jason_caspo/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/nvme_tmp/jason_caspo/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/nvme_tmp/jason_caspo/hf_cache}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source ./scripts/perf_env.sh

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/scalable/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-configs/caspo_rho1b_math.yaml}"

read -r -a GPUS <<< "${GPU_LIST:-4 5 6 7}"
N=${#GPUS[@]}
if (( N != 4 )); then
    echo "[retrain] ERROR: this script is hard-coded for N=4; got GPU_LIST='${GPU_LIST:-4 5 6 7}'"
    exit 2
fi

OUT_ROOT="${OUT_ROOT:-/mnt/nvme_tmp4/jason_caspo/caspo_rho1b_math_v2}"
SHARD_DIR="$OUT_ROOT/shards"
MERGED_DATA="$OUT_ROOT/value_data.pt"
TRAIN_OUT="$OUT_ROOT"
LOGDIR="$OUT_ROOT/logs"
mkdir -p "$SHARD_DIR" "$LOGDIR"

VALUE_MICRO_BATCH_SIZE="${VALUE_MICRO_BATCH_SIZE:-1}"
VALUE_GRAD_ACCUM_STEPS="${VALUE_GRAD_ACCUM_STEPS:-16}"
VALUE_MAX_EPOCHS="${VALUE_MAX_EPOCHS:-3}"
VALUE_LR="${VALUE_LR:-5e-7}"
VALUE_SAVE_EVERY="${VALUE_SAVE_EVERY:-0}"  # 0 = use cfg default (500); set to control ckpt cadence
USE_GRADIENT_CHECKPOINTING="${USE_GRADIENT_CHECKPOINTING:-false}"

echo "[retrain] OUT_ROOT=$OUT_ROOT"
echo "[retrain] GPUs=${GPUS[*]} (N=$N)"

# =========================================================
# 1) SHARDED COLLECTION
# =========================================================
if [[ "${SKIP_COLLECT:-false}" != "true" ]]; then
    echo "[retrain] === Phase 1: 4-shard collect_value_data on GPUs ${GPUS[*]} ==="
    PIDS=()
    PAPER_PAIRING_FLAG=""
    if [[ "${PAPER_PAIRING:-false}" == "true" ]]; then
        PAPER_PAIRING_FLAG="--paper-pairing"
        echo "[retrain] paper-faithful 1-pair-per-prompt pairing enabled"
    fi
    for i in 0 1 2 3; do
        gpu="${GPUS[$i]}"
        out="$SHARD_DIR/value_data_shard${i}.pt"
        log="$LOGDIR/collect_shard${i}.log"
        echo "[retrain] launching shard $i/4 on GPU $gpu -> $out"
        CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON_BIN" -u -m scripts.collect_value_data \
            --config "$BASE_CONFIG" \
            --shard "$i/4" \
            --output "$out" \
            $PAPER_PAIRING_FLAG \
            > "$log" 2>&1 &
        PIDS+=("$!")
    done
    fail=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            rc=$?
            echo "[retrain] ERROR: collect pid=$pid failed rc=$rc"
            fail=$((fail + 1))
        fi
    done
    if (( fail > 0 )); then
        echo "[retrain] $fail/4 shards failed; aborting"
        exit 1
    fi
    echo "[retrain] all 4 shards collected"
fi

# =========================================================
# 2) MERGE
# =========================================================
if [[ "${SKIP_MERGE:-false}" != "true" ]]; then
    echo "[retrain] === Phase 2: merge shards -> $MERGED_DATA ==="
    "$PYTHON_BIN" -u -m scripts.merge_value_data_shards \
        --inputs "$SHARD_DIR"/value_data_shard*.pt \
        --output "$MERGED_DATA" \
        | tee "$LOGDIR/merge.log"
fi

# =========================================================
# 3) FSDP=4 TRAIN_VALUE
# =========================================================
if [[ "${SKIP_TRAIN:-false}" != "true" ]]; then
    echo "[retrain] === Phase 3: FSDP=4 train_value ==="
    MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
    MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
    PIDS=()
    for r in 0 1 2 3; do
        gpu="${GPUS[$r]}"
        log="$LOGDIR/train_value_rank${r}.log"
        echo "[retrain] launching rank $r/$N on GPU $gpu (MASTER_PORT=$MASTER_PORT)"
        CUDA_VISIBLE_DEVICES="$gpu" \
        WORLD_SIZE="$N" RANK="$r" LOCAL_RANK=0 \
        MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
            nohup "$PYTHON_BIN" -u -m scripts.train_value \
            --config "$BASE_CONFIG" \
            --data "$MERGED_DATA" \
            --override "output_dir=$TRAIN_OUT" \
            --override "distributed_backend=fsdp" \
            --override "value_micro_batch_size=$VALUE_MICRO_BATCH_SIZE" \
            --override "value_grad_accum_steps=$VALUE_GRAD_ACCUM_STEPS" \
            --override "value_max_epochs=$VALUE_MAX_EPOCHS" \
            --override "value_lr=$VALUE_LR" \
            $( (( VALUE_SAVE_EVERY > 0 )) && echo "--override value_save_every=$VALUE_SAVE_EVERY" ) \
            --override "use_gradient_checkpointing=$USE_GRADIENT_CHECKPOINTING" \
            > "$log" 2>&1 &
        PIDS+=("$!")
    done
    fail=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            rc=$?
            echo "[retrain] ERROR: train_value rank pid=$pid failed rc=$rc"
            fail=$((fail + 1))
        fi
    done
    if (( fail > 0 )); then
        echo "[retrain] $fail/$N FSDP ranks failed; aborting"
        exit 1
    fi
    echo "[retrain] FSDP train_value done"
    if [[ ! -L "$TRAIN_OUT/value_final" && ! -d "$TRAIN_OUT/value_final" ]]; then
        ln -s "final" "$TRAIN_OUT/value_final"
        echo "[retrain] created value_final -> final symlink"
    fi
fi

# =========================================================
# 4) VALIDATE — quick v_acc check on the new V_φ
# =========================================================
if [[ "${SKIP_VALIDATE:-false}" != "true" ]]; then
    echo "[retrain] === Phase 4: smoke-validate new V_φ (1-step CASPO rollout) ==="
    log="$LOGDIR/validate_smoke.log"
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" "$PYTHON_BIN" -u -m scripts.train_caspo \
        --config "$BASE_CONFIG" \
        --override method=caspo \
        --override "prefix_value_path=$TRAIN_OUT/value_final" \
        --override max_steps=1 \
        --override save_every=999999 \
        --override eval_every=999999 \
        --override wandb_mode=disabled \
        --override "output_dir=$OUT_ROOT/_smoke_validate" \
        --override micro_batch_size=4 \
        --override grad_accum_steps=16 \
        > "$log" 2>&1
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[retrain] WARN: smoke-validate exited rc=$rc — check $log"
    fi
    vacc=$(grep -oE "v_acc=[0-9.]+" "$log" | head -1 | sed 's/v_acc=//')
    echo "[retrain] new V_φ step-1 v_acc = ${vacc:-(not found)}"
    if [[ -n "$vacc" ]]; then
        # awk — compare against 0.7 threshold
        if awk -v v="$vacc" 'BEGIN { exit !(v >= 0.7) }'; then
            echo "[retrain] PASS: v_acc >= 0.7 (close to offline-trained 0.96, ahead of broken 0.32)"
        else
            echo "[retrain] WARN: v_acc < 0.7; BOS+verifier fix did not fully restore calibration"
        fi
    fi
fi

echo "[retrain] === DONE ==="
echo "[retrain] new V_φ at: $TRAIN_OUT/value_final"
echo "[retrain] new value_data.pt at: $MERGED_DATA"
echo "[retrain] to use: --override prefix_value_path=$TRAIN_OUT/value_final"
