#!/usr/bin/env bash
# Collect mc_labels.pt on the HELD-OUT subset of dsr_sub for true OOD-prompt
# PRM evaluation. Skips prompts already used in the primary training collection
# by index range, then runs the same K base × J MC continuations × verifier
# pipeline.
#
# Usage:
#   POLICY=Qwen/Qwen2.5-Math-1.5B \
#   DSR_SUB=/tmp/rlvr_replication/dsr_sub.jsonl \
#   PRIMARY_NUM_PROMPTS=300 \
#   HOLDOUT_NUM_PROMPTS=300 \
#   OUT_DIR=/mnt/nvme_tmp4/jason_caspo/qwen_mc_prm_holdout \
#   bash scripts/launch_qwen_mc_prm_holdout.sh
#
# This collects starting from index PRIMARY_NUM_PROMPTS (default 300) so the
# held-out prompts are disjoint from the primary collection's first-N-prompts
# slice. dsr_sub has 1209 prompts → up to 909 available for holdout.

set -o pipefail
CONDA_ENV="${CONDA_ENV:-scalable}"
if [[ -z "${PYBIN:-}" ]]; then
    if [[ -d "/opt/conda" ]]; then
        source /opt/conda/etc/profile.d/conda.sh
    elif [[ -d "$HOME/miniconda3" ]]; then
        source $HOME/miniconda3/etc/profile.d/conda.sh
    elif [[ -d "$HOME/anaconda3" ]]; then
        source $HOME/anaconda3/etc/profile.d/conda.sh
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

POLICY="${POLICY:-Qwen/Qwen2.5-Math-1.5B}"
DSR_SUB="${DSR_SUB:-/tmp/rlvr_replication/dsr_sub.jsonl}"
[[ -f "$DSR_SUB" ]] || { echo "ERROR: dataset not found: $DSR_SUB"; exit 1; }

PRIMARY_NUM_PROMPTS="${PRIMARY_NUM_PROMPTS:-300}"
HOLDOUT_NUM_PROMPTS="${HOLDOUT_NUM_PROMPTS:-300}"
OUT_DIR="${OUT_DIR:?OUT_DIR must be set}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs_$(date +%Y%m%d_%H%M)}"

K="${K:-16}"
J="${J:-8}"
STEPS_PER_RESPONSE="${STEPS_PER_RESPONSE:-5}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-2048}"
SEED="${SEED:-1}"  # different seed from primary so MC randomness is independent

mkdir -p "$OUT_DIR/shards" "$LOG_DIR"

# We want indices [PRIMARY_NUM_PROMPTS, PRIMARY_NUM_PROMPTS + HOLDOUT_NUM_PROMPTS).
# mc_step_label.py applies --num_prompts (head cap) BEFORE sharding, and there's
# no --skip_first flag. Cleanest path: write a tiny one-shot prep script that
# slices the JSONL and points mc_step_label at the slice.
HOLDOUT_DSR="$OUT_DIR/dsr_sub_holdout.jsonl"
"$PYBIN" - <<PYEOF
import json
src = "$DSR_SUB"
dst = "$HOLDOUT_DSR"
skip = $PRIMARY_NUM_PROMPTS
take = $HOLDOUT_NUM_PROMPTS
with open(src) as f, open(dst, "w") as g:
    rows = [next(f) for _ in range(skip)]
    n = 0
    for line in f:
        if n >= take:
            break
        g.write(line)
        n += 1
print(f"[holdout] wrote {n} prompts to {dst} (skipped first {skip})")
PYEOF

echo "[holdout] $(date +%H:%M:%S) === START === collecting on $HOLDOUT_DSR"
PIDS=()
for shard_i in $(seq 0 $((N_GPUS - 1))); do
  gpu="${GPUS[$shard_i]}"
  log="$LOG_DIR/collect_shard${shard_i}.log"
  CUDA_VISIBLE_DEVICES="$gpu" \
    nohup "$PYBIN" -u scripts/mc_step_label.py \
    --model "$POLICY" \
    --dataset_name "$HOLDOUT_DSR" \
    --prompt_template "{query}\nLet's think step by step and output the final answer within \\boxed{}." \
    --output "$OUT_DIR/shards/shard${shard_i}.pt" \
    --shard "${shard_i}/${N_GPUS}" \
    --K "$K" --J "$J" --steps_per_response "$STEPS_PER_RESPONSE" \
    --temperature 1.0 --top_p 1.0 \
    --max_prompt_len 1024 --max_response_len "$MAX_RESPONSE_LEN" \
    --gpu_memory_utilization 0.85 \
    --seed "$SEED" \
    > "$log" 2>&1 &
  PIDS+=("$!")
done
echo "[holdout] collect PIDs: ${PIDS[*]}"
fail=0
for pid in "${PIDS[@]}"; do wait "$pid" || fail=$((fail+1)); done
(( fail > 0 )) && { echo "[holdout] ABORTING — see $LOG_DIR/collect_shard*.log"; exit 1; }

SHARD_FILES=()
for shard_i in $(seq 0 $((N_GPUS - 1))); do
  SHARD_FILES+=("$OUT_DIR/shards/shard${shard_i}.pt")
done
"$PYBIN" -u scripts/merge_mc_shards.py \
  --inputs "${SHARD_FILES[@]}" \
  --output "$OUT_DIR/mc_labels_holdout.pt" \
  > "$LOG_DIR/merge.log" 2>&1
[[ -s "$OUT_DIR/mc_labels_holdout.pt" ]] || { echo "[holdout] merge failed"; exit 1; }
echo "[holdout] $(date +%H:%M:%S) === DONE === $OUT_DIR/mc_labels_holdout.pt"
