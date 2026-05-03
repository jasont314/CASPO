#!/usr/bin/env bash
# Two-phase ALTERNATING CASPO: cycles of (RL → refresh PRM → RL → refresh PRM → ...).
#
# Each cycle:
#   1. RL train for REFRESH_EVERY steps from current ckpt, with current PRM
#   2. Collect fresh MC labels at the new policy (mc_step_label.py, sharded 4-way)
#   3. Train a fresh PRM on those labels (train_value_mc.py)
#   4. Repeat until TOTAL_STEPS reached
#
# Resume between cycles preserves: optimizer.pt, lr_scheduler, ref_policy=base SFT
# ONLY thing that changes per cycle: prefix_value_path → fresh PRM
#
# ---- Required env vars ----
#   INITIAL_CKPT   : starting policy ckpt (e.g., orig CASPO step_0 or base SFT path)
#   INITIAL_PRM    : starting PRM ckpt (e.g., orig PRM trained at base SFT)
#   OUT_ROOT       : output root dir
#
# ---- Optional env vars ----
#   GPU_LIST       : default "0 1 2 3"
#   DSR_SUB        : default /tmp/rlvr_replication/dsr_sub.jsonl
#   REF_MODEL      : default Qwen/Qwen2.5-Math-1.5B (base SFT — held fixed across cycles)
#   TOTAL_STEPS    : default 500 (total RL steps across all cycles)
#   REFRESH_EVERY  : default 150 (RL steps between refreshes)
#   METHOD         : default "caspo"
#   ADV_TRANSFORM  : default "prob"  (or "logprob" for Δlogp)
#   PRM_TRAIN_K    : default 16 (mc_step_label K)
#   PRM_TRAIN_J    : default 16
#   PRM_TRAIN_S    : default 5
#   PRM_TRAIN_MAX_RESP : default 2048 (matches RL training cap; capture all RL-policy responses)
#   PRM_TRAIN_PREFIX_CAP : default 0 (= match collection cap; no decoupling, train/deploy aligned)
#   PRM_TRAIN_EPOCHS : default 2
#   ALL Phase-1 hparams (LR, KL_COEF, etc.) inherit defaults from launch_caspo_refresh_resume.sh
#
# ---- Method-specific REFRESH_EVERY recommendations ----
# Δp / orig CASPO  : REFRESH_EVERY=150 (faster policy drift → refresh sooner)
# Δlogp            : REFRESH_EVERY=200 (slower drift, "rescue amplification")
# unfamiliar       : start at 150, raise if PRM ρ holds >0.40 at refresh point
#
# Examples:
#   # Δp (orig CASPO style), refresh every 150
#   INITIAL_CKPT=/path/base_ckpt INITIAL_PRM=/path/orig_prm OUT_ROOT=/out \
#     scripts/launch_caspo_alternating.sh
#
#   # Δlogp variant, refresh every 200 (sparser)
#   ADV_TRANSFORM=logprob METHOD=caspo REFRESH_EVERY=200 \
#     INITIAL_CKPT=... INITIAL_PRM=... OUT_ROOT=... \
#     scripts/launch_caspo_alternating.sh
#
#   # Aggressive refresh every 75 steps (more compute on PRM)
#   REFRESH_EVERY=75 ... scripts/launch_caspo_alternating.sh
#
# ---- ETA ----
# 500 RL steps × ~80s/step = 11h
# + N refresh cycles × (~30 min collect + ~50 min train)
# 500 / 150 = 3 cycles → 11h + 4h = ~15h on 4 GPUs
# 500 / 200 = 2 cycles → 11h + 2.7h = ~14h on 4 GPUs (Δlogp)
# 500 / 75  = 6 cycles → 11h + 8h = ~19h on 4 GPUs (aggressive)

set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-scalable}"
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYBIN="${PYBIN:-$(which python)}"

: "${INITIAL_CKPT:?INITIAL_CKPT must be set}"
: "${INITIAL_PRM:?INITIAL_PRM must be set}"
: "${OUT_ROOT:?OUT_ROOT must be set}"

GPU_LIST="${GPU_LIST:-0 1 2 3}"
DSR_SUB="${DSR_SUB:-/tmp/rlvr_replication/dsr_sub.jsonl}"
REF_MODEL="${REF_MODEL:-Qwen/Qwen2.5-Math-1.5B}"
TOTAL_STEPS="${TOTAL_STEPS:-500}"
REFRESH_EVERY="${REFRESH_EVERY:-150}"
METHOD="${METHOD:-caspo}"
ADV_TRANSFORM="${ADV_TRANSFORM:-prob}"
PRM_TRAIN_K="${PRM_TRAIN_K:-16}"
PRM_TRAIN_J="${PRM_TRAIN_J:-16}"
PRM_TRAIN_S="${PRM_TRAIN_S:-5}"
PRM_TRAIN_MAX_RESP="${PRM_TRAIN_MAX_RESP:-2048}"
# refresh: matches RL max_response_len; captures ~98% of correct chains (p98 ≈ 1613)
PRM_TRAIN_PREFIX_CAP="${PRM_TRAIN_PREFIX_CAP:-0}"
# 0 = match collection cap; no decoupling. Validated by 1536-prefix v3 refresh (ρ=0.630).
PRM_TRAIN_EPOCHS="${PRM_TRAIN_EPOCHS:-2}"
PROMPT_TPL='{query}\nLet'\''s think step by step and output the final answer within \boxed{}.'

mkdir -p "$OUT_ROOT"

current_ckpt="$INITIAL_CKPT"
current_prm="$INITIAL_PRM"
current_step=0
cycle=0

echo "[alt] $(date +%H:%M:%S) === START ALTERNATING ==="
echo "[alt] init ckpt:     $INITIAL_CKPT"
echo "[alt] init PRM:      $INITIAL_PRM"
echo "[alt] total steps:   $TOTAL_STEPS"
echo "[alt] refresh every: $REFRESH_EVERY"

while (( current_step < TOTAL_STEPS )); do
  cycle=$((cycle + 1))
  next_target=$((current_step + REFRESH_EVERY))
  (( next_target > TOTAL_STEPS )) && next_target="$TOTAL_STEPS"

  # ---- Phase: RL from current_ckpt + current_prm until step=next_target ----
  RL_OUT="$OUT_ROOT/cycle_${cycle}_rl"
  echo ""
  echo "[alt] $(date +%H:%M:%S) === CYCLE $cycle: RL [$current_step → $next_target] ==="
  echo "[alt] policy: $current_ckpt"
  echo "[alt] PRM:    $current_prm"
  echo "[alt] out:    $RL_OUT"

  POLICY_CKPT="$current_ckpt" \
  NEW_PRM="$current_prm" \
  OUT_DIR="$RL_OUT" \
  REF_MODEL="$REF_MODEL" \
  GPU_LIST="$GPU_LIST" \
  DSR_SUB="$DSR_SUB" \
  MAX_STEPS="$next_target" \
  SAVE_EVERY=50 \
  METHOD="$METHOD" \
  ADV_TRANSFORM="$ADV_TRANSFORM" \
  LOG_DIR="$OUT_ROOT/logs/cycle_${cycle}_rl" \
    bash scripts/launch_caspo_refresh_resume.sh

  # Find the ckpt we just saved at next_target (or final if hit TOTAL_STEPS)
  if [[ -d "$RL_OUT/step_${next_target}" ]]; then
    current_ckpt="$RL_OUT/step_${next_target}"
  elif [[ -d "$RL_OUT/final" ]]; then
    current_ckpt="$RL_OUT/final"
  else
    echo "[alt] ERROR: cannot find ckpt at $RL_OUT/step_${next_target} or /final"
    exit 1
  fi
  current_step="$next_target"
  echo "[alt] $(date +%H:%M:%S) cycle $cycle RL done. current_ckpt=$current_ckpt"

  # If we've hit TOTAL_STEPS, stop — no need to refresh PRM further
  if (( current_step >= TOTAL_STEPS )); then
    echo "[alt] reached TOTAL_STEPS=$TOTAL_STEPS — stopping (no further refresh needed)"
    break
  fi

  # ---- Refresh PRM at current_ckpt ----
  PRM_OUT="$OUT_ROOT/cycle_${cycle}_prm"
  PRM_LOGS="$OUT_ROOT/logs/cycle_${cycle}_prm"
  mkdir -p "$PRM_OUT/shards" "$PRM_LOGS"
  echo ""
  echo "[alt] $(date +%H:%M:%S) === CYCLE $cycle: REFRESH PRM at $current_ckpt ==="

  # Phase 1 of refresh: collect MC labels (4-shard parallel)
  read -r -a GPUS_ARR <<< "$GPU_LIST"
  PIDS=()
  for i in 0 1 2 3; do
    gpu="${GPUS_ARR[$i]}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -u scripts/mc_step_label.py \
      --model "$current_ckpt" \
      --dataset_name "$DSR_SUB" \
      --prompt_template "$PROMPT_TPL" \
      --output "$PRM_OUT/shards/shard${i}.pt" \
      --shard "${i}/4" \
      --K "$PRM_TRAIN_K" --J "$PRM_TRAIN_J" --steps_per_response "$PRM_TRAIN_S" \
      --temperature 1.0 --top_p 1.0 \
      --max_prompt_len 1024 --max_response_len "$PRM_TRAIN_MAX_RESP" \
      --max_train_prefix_len "$PRM_TRAIN_PREFIX_CAP" \
      --gpu_memory_utilization 0.85 --seed "$cycle" \
      > "$PRM_LOGS/collect_shard${i}.log" 2>&1 &
    PIDS+=("$!")
  done
  fail=0
  for pid in "${PIDS[@]}"; do wait "$pid" || fail=$((fail+1)); done
  echo "[alt] $(date +%H:%M:%S) collect done ($fail failures)"
  (( fail > 0 )) && { echo "[alt] ABORT: refresh collect failed"; exit 1; }

  # Merge shards
  "$PYBIN" -u scripts/merge_mc_shards.py \
    --inputs "$PRM_OUT/shards/shard0.pt" "$PRM_OUT/shards/shard1.pt" "$PRM_OUT/shards/shard2.pt" "$PRM_OUT/shards/shard3.pt" \
    --output "$PRM_OUT/mc_labels.pt" > "$PRM_LOGS/merge.log" 2>&1

  # Phase 2 of refresh: train PRM (FSDP=4)
  TRAIN_PORT=$((35000 + RANDOM % 100))
  TRAIN_PIDS=()
  for r in 0 1 2 3; do
    gpu="${GPUS_ARR[$r]}"
    CUDA_VISIBLE_DEVICES="$gpu" \
    WORLD_SIZE=4 RANK="$r" LOCAL_RANK=0 \
    MASTER_ADDR=127.0.0.1 MASTER_PORT="$TRAIN_PORT" \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
      "$PYBIN" -u scripts/train_value_mc.py \
      --config configs/caspo_rho1b_math.yaml \
      --data "$PRM_OUT/mc_labels.pt" \
      --output_dir "$PRM_OUT" \
      --phi_init_path "$REF_MODEL" --ref_path "$REF_MODEL" \
      --lr 5e-6 --mb 4 --grad_accum 2 --epochs "$PRM_TRAIN_EPOCHS" \
      --save_every 500 --eval_every 100 --early_stop_patience 999 \
      --val_fraction 0.1 --beta 10.0 --seed 0 \
      > "$PRM_LOGS/train_rank${r}.log" 2>&1 &
    TRAIN_PIDS+=("$!")
  done
  fail=0
  for pid in "${TRAIN_PIDS[@]}"; do wait "$pid" || fail=$((fail+1)); done
  echo "[alt] $(date +%H:%M:%S) PRM train done ($fail failures)"
  (( fail > 0 )) && { echo "[alt] ABORT: refresh train failed"; exit 1; }

  current_prm="$PRM_OUT/best"
  [[ -d "$current_prm" ]] || { echo "[alt] ERROR: $current_prm not found after train"; exit 1; }
  echo "[alt] $(date +%H:%M:%S) refresh complete. new PRM=$current_prm"
done

echo ""
echo "[alt] $(date +%H:%M:%S) === ALL DONE: $cycle cycles, $current_step total RL steps ==="
echo "[alt] final policy: $current_ckpt"
echo "[alt] final PRM:    $current_prm"
