#!/usr/bin/env bash
# End-to-end ALTERNATING CASPO: PRM → RL → PRM → RL → ... → PRM → RL
#
# Self-contained pipeline. Trains the initial PRM internally (Phase 0), then
# alternates RL training and PRM refresh until TOTAL_STEPS reached.
#
# Each cycle after Phase 0:
#   1. RL train for REFRESH_EVERY steps from current ckpt with current PRM
#   2. Collect fresh MC labels at the new policy + train a new PRM from scratch
#   3. Repeat until TOTAL_STEPS reached (no PRM refresh after the final RL)
#
# Resume between RL phases preserves: optimizer.pt, lr_scheduler, ref_policy=
# REF_MODEL (held fixed across all cycles). Only prefix_value_path changes.
#
# ---- Required env vars ----
#   INITIAL_CKPT   : starting policy (default: Qwen/Qwen2.5-Math-1.5B base SFT)
#   OUT_ROOT       : output root dir
#   DSR_SUB        : dataset path (default: /tmp/rlvr_replication/dsr_sub.jsonl)
#
# ---- Optional env vars ----
#   INITIAL_PRM    : pre-trained initial PRM (skips Phase 0 if set). Default
#                    empty → Phase 0 trains the initial PRM at REF_MODEL on
#                    rollouts from INITIAL_CKPT.
#   GPU_LIST       : default "0 1 2 3" (4 GPUs)
#   REF_MODEL      : default Qwen/Qwen2.5-Math-1.5B (base SFT — held fixed
#                    across cycles for KL anchor + PRM init from-scratch)
#   TOTAL_STEPS    : default 600 (total RL steps across all cycles)
#   REFRESH_EVERY  : default 150 (RL steps between refreshes)
#   METHOD         : default "caspo"
#   ADV_TRANSFORM  : default "prob"  (or "logprob" for Δlogp)
#   PRM_TRAIN_K, PRM_TRAIN_J, PRM_TRAIN_S        : default 16, 16, 5
#   PRM_TRAIN_NUM_PROMPTS                        : default empty (= use all)
#   PRM_TRAIN_MAX_RESP                           : default 2048 (= RL cap)
#   PRM_TRAIN_PREFIX_CAP                         : default 0 (= match collection)
#   PRM_TRAIN_EPOCHS                             : default 2
#   ALL Phase-1 hparams (LR, KL_COEF, etc.) inherit defaults from
#     scripts/launch_caspo_refresh_resume.sh
#
# ---- Method-specific REFRESH_EVERY recommendations ----
# Δp / orig CASPO  : REFRESH_EVERY=150 (faster policy drift → refresh sooner)
# Δlogp            : REFRESH_EVERY=200 (slower drift, "rescue amplification")
# unfamiliar       : start at 150, raise if PRM ρ holds >0.40 at refresh point
#
# ---- Examples ----
#
#   # Δp full pipeline, refresh every 150 (Phase 0 + alternating)
#   INITIAL_CKPT=Qwen/Qwen2.5-Math-1.5B \
#   OUT_ROOT=/mnt/data/caspo_alt \
#   DSR_SUB=/tmp/rlvr_replication/dsr_sub.jsonl \
#     bash scripts/launch_caspo_alternating.sh
#
#   # Δlogp variant, refresh every 200 (sparser)
#   ADV_TRANSFORM=logprob REFRESH_EVERY=200 \
#   INITIAL_CKPT=... OUT_ROOT=... DSR_SUB=... \
#     bash scripts/launch_caspo_alternating.sh
#
#   # Skip Phase 0 by providing a pre-trained PRM
#   INITIAL_PRM=/path/to/prm/best \
#   INITIAL_CKPT=... OUT_ROOT=... DSR_SUB=... \
#     bash scripts/launch_caspo_alternating.sh
#
# ---- ETA ----
# Phase 0:                  ~91 min (collect ~41 + train ~50)
# 600 RL steps × ~80s/step: ~13.3 h
# Refresh cycles:           ~91 min each
# 600 / 150 = 3 refreshes     → 13.3 h + 1.5 h (Phase 0) + 4.5 h (refreshes) = ~19 h
# 600 / 200 = 2 refreshes     → 13.3 h + 1.5 h (Phase 0) + 3 h (refreshes) = ~17.8 h (Δlogp)
# Skip Phase 0 (INITIAL_PRM provided): subtract 1.5 h.

set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-scalable}"
# preserve -e from above; only ADD -u (treat unset vars as errors).
# `set -uo pipefail` (without -e) would silently drop -e and let
# failing nested launchers continue the alternating loop.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYBIN="${PYBIN:-$(which python)}"

: "${OUT_ROOT:?OUT_ROOT must be set}"

INITIAL_CKPT="${INITIAL_CKPT:-Qwen/Qwen2.5-Math-1.5B}"
INITIAL_PRM="${INITIAL_PRM:-}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
DSR_SUB="${DSR_SUB:-/tmp/rlvr_replication/dsr_sub.jsonl}"
REF_MODEL="${REF_MODEL:-Qwen/Qwen2.5-Math-1.5B}"
TOTAL_STEPS="${TOTAL_STEPS:-600}"
REFRESH_EVERY="${REFRESH_EVERY:-150}"
METHOD="${METHOD:-caspo}"
ADV_TRANSFORM="${ADV_TRANSFORM:-prob}"
PRM_TRAIN_K="${PRM_TRAIN_K:-16}"
PRM_TRAIN_J="${PRM_TRAIN_J:-16}"
PRM_TRAIN_S="${PRM_TRAIN_S:-5}"
PRM_TRAIN_NUM_PROMPTS="${PRM_TRAIN_NUM_PROMPTS:-}"
PRM_TRAIN_MAX_RESP="${PRM_TRAIN_MAX_RESP:-2048}"
# refresh: matches RL max_response_len; captures ~98% of correct chains (p98 ≈ 1613)
PRM_TRAIN_PREFIX_CAP="${PRM_TRAIN_PREFIX_CAP:-0}"
# 0 = match collection cap; no decoupling. Validated by 1536-prefix v3 refresh (ρ=0.630).
PRM_TRAIN_EPOCHS="${PRM_TRAIN_EPOCHS:-2}"

mkdir -p "$OUT_ROOT/logs"

# ----------------------------------------------------------
# Helper: train a PRM via scripts/launch_qwen_mc_prm.sh
# Args:
#   $1: rollout source POLICY (path or HF id)
#   $2: output dir for the PRM
#   $3: log dir for this PRM training
#   $4: seed (so refresh PRMs use varying seeds across cycles)
# ----------------------------------------------------------
train_prm() {
  local policy="$1"
  local prm_out="$2"
  local prm_log="$3"
  local seed="$4"
  local np_arg=()
  [[ -n "$PRM_TRAIN_NUM_PROMPTS" ]] && np_arg=(NUM_PROMPTS="$PRM_TRAIN_NUM_PROMPTS")
  POLICY="$policy" \
  PHI_INIT="$REF_MODEL" \
  REF_PATH="$REF_MODEL" \
  OUT_DIR="$prm_out" \
  LOG_DIR="$prm_log" \
  GPU_LIST="$GPU_LIST" \
  DSR_SUB="$DSR_SUB" \
  K="$PRM_TRAIN_K" \
  J="$PRM_TRAIN_J" \
  STEPS_PER_RESPONSE="$PRM_TRAIN_S" \
  MAX_RESPONSE_LEN="$PRM_TRAIN_MAX_RESP" \
  MAX_TRAIN_PREFIX_LEN="$PRM_TRAIN_PREFIX_CAP" \
  EPOCHS="$PRM_TRAIN_EPOCHS" \
  SEED="$seed" \
  "${np_arg[@]}" \
    bash scripts/launch_qwen_mc_prm.sh
}

current_ckpt="$INITIAL_CKPT"
current_prm="$INITIAL_PRM"
current_step=0
cycle=0

echo "[alt] $(date +%H:%M:%S) === START ALTERNATING ==="
echo "[alt] init ckpt:     $INITIAL_CKPT"
echo "[alt] init PRM:      ${INITIAL_PRM:-<unset — Phase 0 will train it>}"
echo "[alt] ref model:     $REF_MODEL"
echo "[alt] total steps:   $TOTAL_STEPS"
echo "[alt] refresh every: $REFRESH_EVERY"
echo "[alt] method:        $METHOD ($ADV_TRANSFORM)"
echo "[alt] out root:      $OUT_ROOT"

# ----------------------------------------------------------
# Phase 0: train initial PRM if not provided
# ----------------------------------------------------------
if [[ -z "$current_prm" ]]; then
  PHASE0_OUT="$OUT_ROOT/phase_0_prm"
  PHASE0_LOG="$OUT_ROOT/logs/phase_0_prm"
  echo ""
  echo "[alt] $(date +%H:%M:%S) === PHASE 0: TRAIN INITIAL PRM ==="
  echo "[alt] rollout source: $current_ckpt (= INITIAL_CKPT)"
  echo "[alt] PRM init:       $REF_MODEL (= REF_MODEL, from-scratch)"
  echo "[alt] out:            $PHASE0_OUT"
  train_prm "$current_ckpt" "$PHASE0_OUT" "$PHASE0_LOG" 0
  current_prm="$PHASE0_OUT/best"
  [[ -d "$current_prm" ]] || { echo "[alt] ERROR: $current_prm not found after Phase 0"; exit 1; }
  echo "[alt] $(date +%H:%M:%S) Phase 0 done. initial PRM=$current_prm"
fi

# ----------------------------------------------------------
# Alternating cycles: RL → refresh PRM → RL → ... → final RL
# ----------------------------------------------------------
while (( current_step < TOTAL_STEPS )); do
  cycle=$((cycle + 1))
  next_target=$((current_step + REFRESH_EVERY))
  (( next_target > TOTAL_STEPS )) && next_target="$TOTAL_STEPS"

  # ---- RL phase: from current_ckpt + current_prm until step=next_target ----
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

  # Locate the saved ckpt for the next phase
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
  PRM_LOG="$OUT_ROOT/logs/cycle_${cycle}_prm"
  echo ""
  echo "[alt] $(date +%H:%M:%S) === CYCLE $cycle: REFRESH PRM at $current_ckpt ==="
  echo "[alt] rollout source: $current_ckpt"
  echo "[alt] out:            $PRM_OUT"
  train_prm "$current_ckpt" "$PRM_OUT" "$PRM_LOG" "$cycle"
  current_prm="$PRM_OUT/best"
  [[ -d "$current_prm" ]] || { echo "[alt] ERROR: $current_prm not found after train"; exit 1; }
  echo "[alt] $(date +%H:%M:%S) refresh complete. new PRM=$current_prm"
done

echo ""
echo "[alt] $(date +%H:%M:%S) === ALL DONE: $cycle cycles, $current_step total RL steps ==="
echo "[alt] final policy: $current_ckpt"
echo "[alt] final PRM:    $current_prm"
