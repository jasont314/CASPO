#!/usr/bin/env bash
# MC PRM training pipeline for Qwen2.5-Math-1.5B + dsr_sub.
#
# Two phases:
#   1. Phase A — multi-shard data collection via mc_step_label.py
#   2. Phase B — FSDP=4 training via train_value_mc.py
#
# Supports both:
#   - INITIAL PRM: collect from base SFT, train V_φ from base SFT
#       POLICY=Qwen/Qwen2.5-Math-1.5B (default)
#       PHI_INIT=Qwen/Qwen2.5-Math-1.5B (default)
#   - REFRESH PRM: collect from a CASPO/RL ckpt, train V_φ from base SFT (scratch)
#       POLICY=/path/to/caspo_step_150
#       PHI_INIT=Qwen/Qwen2.5-Math-1.5B (recommended; from-scratch is empirically better)
#
# ---- Configurable env vars ----
#   CONDA_ENV=scalable
#   PYBIN=...python
#   GPU_LIST="0 1 2 3"          # collection: one GPU per shard; training: FSDP across all
#   POLICY=Qwen/Qwen2.5-Math-1.5B   # model used to GENERATE rollouts during collection
#   DSR_SUB=/path/to/dsr_sub.jsonl
#   OUT_DIR=/path/to/PRM_output
#   LOG_DIR=/tmp/mc_prm_$(date +%H%M)
#
#   # Phase A (collection):
#   K=16                        # base rollouts per prompt
#   J=16                        # MC continuations per labeled prefix
#   STEPS_PER_RESPONSE=5
#   NUM_PROMPTS=                # empty = use all (1209 for dsr_sub)
#   MAX_RESPONSE_LEN=2048       # matches RL deployment cap
#   MAX_TRAIN_PREFIX_LEN=0      # 0 = match collection cap (no decoupling)
#
#   # Phase B (training):
#   PHI_INIT=Qwen/Qwen2.5-Math-1.5B   # PRM backbone init (use base SFT for from-scratch)
#   REF_PATH=                          # reference model for cumulative-log-ratio
#                                      # architecture (V_φ = log π_φ/π_ref). Empty
#                                      # = same as PHI_INIT.
#   LR=5e-6
#   TRAIN_MB=4
#   GRAD_ACCUM=2                # eff_batch = N_GPUS × TRAIN_MB × GRAD_ACCUM = 32
#   EPOCHS=2
#   VAL_FRACTION=0.1
#   EARLY_STOP_PATIENCE=999     # 999 = effectively no early stop
#   BETA=10.0
#   SEED=0
#
# ---- ETA ----
#   Initial PRM (base SFT, full dsr_sub N=1209):
#     ~41 min collection + ~50 min training on 4 GPUs ≈ 91 min total
#   Refresh PRM (CASPO ckpt, N=300):
#     ~30 min collection + ~50 min training ≈ 80 min total
#
set -o pipefail

CONDA_ENV="${CONDA_ENV:-scalable}"
if [[ -z "${PYBIN:-}" ]]; then
    if [[ -d "/opt/conda" ]]; then
        source /opt/conda/etc/profile.d/conda.sh
    elif [[ -d "$HOME/miniconda3" ]]; then
        source $HOME/miniconda3/etc/profile.d/conda.sh
    elif [[ -d "$HOME/anaconda3" ]]; then
        source $HOME/anaconda3/etc/profile.d/conda.sh
    else
        echo "ERROR: cannot find conda. Set PYBIN explicitly to override."
        exit 1
    fi
    conda activate "$CONDA_ENV"
    PYBIN="$(which python)"
fi

set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
[[ -f ./scripts/perf_env.sh ]] && source ./scripts/perf_env.sh

GPU_LIST="${GPU_LIST:-0 1 2 3}"
read -r -a GPUS <<< "$GPU_LIST"
N_GPUS=${#GPUS[@]}
[[ "$N_GPUS" -ge 4 ]] || { echo "ERROR: need >= 4 GPUs; got $N_GPUS"; exit 1; }

POLICY="${POLICY:-Qwen/Qwen2.5-Math-1.5B}"
DSR_SUB="${DSR_SUB:-/path/to/dsr_sub.jsonl}"
[[ -f "$DSR_SUB" ]] || { echo "ERROR: dataset not found: $DSR_SUB"; exit 1; }

OUT_DIR="${OUT_DIR:?OUT_DIR must be set}"
LOG_DIR="${LOG_DIR:-/tmp/mc_prm_$(date +%Y%m%d_%H%M)}"

K="${K:-16}"
J="${J:-16}"
STEPS_PER_RESPONSE="${STEPS_PER_RESPONSE:-5}"
NUM_PROMPTS="${NUM_PROMPTS:-}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-2048}"
MAX_TRAIN_PREFIX_LEN="${MAX_TRAIN_PREFIX_LEN:-0}"

PHI_INIT="${PHI_INIT:-Qwen/Qwen2.5-Math-1.5B}"
REF_PATH="${REF_PATH:-}"
LR="${LR:-5e-6}"
# Reverted to validated v3 config (mb=4 grad_accum=2). Earlier mb=8 grad_accum=1
# was untested at seq=3072 and ran ~2× slower per step than mb=4 (per
# feedback_grpo_mb_pareto.md: "mb=4 best (65s/step), mb=8 slower (72s)").
TRAIN_MB="${TRAIN_MB:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
EPOCHS="${EPOCHS:-2}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-999}"
BETA="${BETA:-10.0}"
SEED="${SEED:-0}"

mkdir -p "$OUT_DIR/shards" "$LOG_DIR"

echo "[mc-prm] $(date +%H:%M:%S) === START ==="
echo "[mc-prm] POLICY (rollout source):    $POLICY"
echo "[mc-prm] PHI_INIT (PRM backbone):    $PHI_INIT"
echo "[mc-prm] dataset:                    $DSR_SUB"
echo "[mc-prm] N=${NUM_PROMPTS:-all} K=$K J=$J steps=$STEPS_PER_RESPONSE"
echo "[mc-prm] max_response_len=$MAX_RESPONSE_LEN  max_train_prefix_len=$MAX_TRAIN_PREFIX_LEN"
echo "[mc-prm] training: lr=$LR mb=$TRAIN_MB×$GRAD_ACCUM eff_batch=$((N_GPUS * TRAIN_MB * GRAD_ACCUM)) ep=$EPOCHS"
echo "[mc-prm] output: $OUT_DIR  logs: $LOG_DIR"

# ----------------------------------------------------------
# Phase A — N_GPUS-shard data collection
# ----------------------------------------------------------
echo "[mc-prm] $(date +%H:%M:%S) Phase A: $N_GPUS-shard collection"
PIDS=()
for shard_i in $(seq 0 $((N_GPUS - 1))); do
  gpu="${GPUS[$shard_i]}"
  log="$LOG_DIR/collect_shard${shard_i}.log"
  np_arg=()
  [[ -n "$NUM_PROMPTS" ]] && np_arg=(--num_prompts "$NUM_PROMPTS")
  CUDA_VISIBLE_DEVICES="$gpu" \
    nohup "$PYBIN" -u scripts/mc_step_label.py \
    --model "$POLICY" \
    --dataset_name "$DSR_SUB" \
    --prompt_template "{query}\nLet's think step by step and output the final answer within \\boxed{}." \
    --output "$OUT_DIR/shards/shard${shard_i}.pt" \
    --shard "${shard_i}/${N_GPUS}" \
    --K "$K" --J "$J" --steps_per_response "$STEPS_PER_RESPONSE" \
    --temperature 1.0 --top_p 1.0 \
    --max_prompt_len 1024 --max_response_len "$MAX_RESPONSE_LEN" \
    --max_train_prefix_len "$MAX_TRAIN_PREFIX_LEN" \
    --gpu_memory_utilization 0.85 \
    --seed "$SEED" \
    "${np_arg[@]}" \
    > "$log" 2>&1 &
  PIDS+=("$!")
done
echo "[mc-prm] collect PIDs: ${PIDS[*]}"

fail=0
for pid in "${PIDS[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "[mc-prm] $(date +%H:%M:%S) Phase A done ($fail shard failures)"
(( fail > 0 )) && { echo "[mc-prm] ABORTING — see $LOG_DIR/collect_shard*.log"; exit 1; }

# ----------------------------------------------------------
# Merge shards
# ----------------------------------------------------------
echo "[mc-prm] $(date +%H:%M:%S) merging $N_GPUS shards"
SHARD_FILES=()
for shard_i in $(seq 0 $((N_GPUS - 1))); do
  SHARD_FILES+=("$OUT_DIR/shards/shard${shard_i}.pt")
done
"$PYBIN" -u scripts/merge_mc_shards.py \
  --inputs "${SHARD_FILES[@]}" \
  --output "$OUT_DIR/mc_labels.pt" \
  > "$LOG_DIR/merge.log" 2>&1
[[ -s "$OUT_DIR/mc_labels.pt" ]] || { echo "[mc-prm] ABORTING — merge failed"; cat "$LOG_DIR/merge.log"; exit 1; }
echo "[mc-prm] $(date +%H:%M:%S) merge done"

# ----------------------------------------------------------
# Phase B — FSDP=4 training
# ----------------------------------------------------------
echo "[mc-prm] $(date +%H:%M:%S) Phase B: FSDP=4 training"
TRAIN_PORT=$((30000 + RANDOM % 5000))
TRAIN_PIDS=()
ref_arg=()
[[ -n "$REF_PATH" ]] && ref_arg=(--ref_path "$REF_PATH")
for r in 0 1 2 3; do
  gpu="${GPUS[$r]}"
  log="$LOG_DIR/train_rank${r}.log"
  CUDA_VISIBLE_DEVICES="$gpu" \
  WORLD_SIZE=4 RANK="$r" LOCAL_RANK=0 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT="$TRAIN_PORT" \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
    nohup "$PYBIN" -u scripts/train_value_mc.py \
    --config configs/caspo_rho1b_math.yaml \
    --data "$OUT_DIR/mc_labels.pt" \
    --output_dir "$OUT_DIR" \
    --phi_init_path "$PHI_INIT" \
    "${ref_arg[@]}" \
    --lr "$LR" --mb "$TRAIN_MB" --grad_accum "$GRAD_ACCUM" \
    --epochs "$EPOCHS" --val_fraction "$VAL_FRACTION" \
    --early_stop_patience "$EARLY_STOP_PATIENCE" \
    --beta "$BETA" --seed "$SEED" \
    --save_every 500 --eval_every 100 \
    > "$log" 2>&1 &
  TRAIN_PIDS+=("$!")
done
echo "[mc-prm] train PIDs: ${TRAIN_PIDS[*]}"

fail=0
for pid in "${TRAIN_PIDS[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "[mc-prm] $(date +%H:%M:%S) Phase B done ($fail rank failures)"
echo "[mc-prm] $(date +%H:%M:%S) === ALL DONE === final=$OUT_DIR/final  best=$OUT_DIR/best"
