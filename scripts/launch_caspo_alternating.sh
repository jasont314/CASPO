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
#   PRM_TRAIN_K                                  : default 16
#   PRM_TRAIN_J (initial)                        : default 16  (orig recipe)
#   PRM_TRAIN_J_REFRESH                          : default 8   (v3 refresh recipe)
#   PRM_TRAIN_S                                  : default 5
#   PRM_TRAIN_NUM_PROMPTS                        : default empty (= use all dsr_sub)
#   PRM_TRAIN_MAX_RESP (initial)                 : default 1024 (orig recipe — base SFT rollouts compact)
#   PRM_TRAIN_MAX_RESP_REFRESH                   : default 1536 (v3 refresh recipe — captures p99 step_150 correct chains)
#   PRM_TRAIN_PREFIX_CAP                         : default 0 (= match collection)
#   PRM_TRAIN_EPOCHS                             : default 3 (every metric-B-best PRM was 3-epoch)
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
# Initial PRM uses J=16 (orig recipe, deployed production). Refresh PRMs use J=8
# (faster, refresh v3 recipe — won metric-B comparison).
PRM_TRAIN_J="${PRM_TRAIN_J:-16}"
PRM_TRAIN_J_REFRESH="${PRM_TRAIN_J_REFRESH:-8}"
PRM_TRAIN_S="${PRM_TRAIN_S:-5}"
PRM_TRAIN_NUM_PROMPTS="${PRM_TRAIN_NUM_PROMPTS:-}"
# Per-phase response cap — initial trains on base SFT (compact rollouts, cap=1024
# matches deployed orig). Refresh trains on policy-current rollouts (longer; cap=1536
# captures ~99% of step_150-policy correct chains — validated by v3 refresh ρ=0.364).
PRM_TRAIN_MAX_RESP="${PRM_TRAIN_MAX_RESP:-1024}"
PRM_TRAIN_MAX_RESP_REFRESH="${PRM_TRAIN_MAX_RESP_REFRESH:-1536}"
PRM_TRAIN_PREFIX_CAP="${PRM_TRAIN_PREFIX_CAP:-0}"
# 0 = match collection cap; no decoupling. Validated by 1536-prefix v3 refresh (ρ=0.364).
# 3 epochs: every metric-B-best PRM today (orig, v3 refresh, refresh_3ep) trained 3 ep.
# 2-epoch was undertrained per intermediate-checkpoint sweep (best-step at 95%+ of 2ep).
PRM_TRAIN_EPOCHS="${PRM_TRAIN_EPOCHS:-3}"

mkdir -p "$OUT_ROOT/logs"

# ----------------------------------------------------------
# Helper: train a PRM via scripts/launch_qwen_mc_prm.sh
# Args:
#   $1: rollout source POLICY (path or HF id)
#   $2: output dir for the PRM
#   $3: log dir for this PRM training
#   $4: seed (so refresh PRMs use varying seeds across cycles)
#   $5: phase ("initial" or "refresh") — selects J / cap defaults
# ----------------------------------------------------------
train_prm() {
  local policy="$1"
  local prm_out="$2"
  local prm_log="$3"
  local seed="$4"
  local phase="${5:-refresh}"   # default refresh for backward-compat
  local np_arg=()
  [[ -n "$PRM_TRAIN_NUM_PROMPTS" ]] && np_arg=(NUM_PROMPTS="$PRM_TRAIN_NUM_PROMPTS")
  # Per-phase J / cap selection
  local j_phase max_resp_phase
  if [[ "$phase" == "initial" ]]; then
    j_phase="$PRM_TRAIN_J"
    max_resp_phase="$PRM_TRAIN_MAX_RESP"
  else
    j_phase="$PRM_TRAIN_J_REFRESH"
    max_resp_phase="$PRM_TRAIN_MAX_RESP_REFRESH"
  fi
  echo "[alt] train_prm phase=$phase  J=$j_phase  max_resp=$max_resp_phase  ep=$PRM_TRAIN_EPOCHS"
  POLICY="$policy" \
  PHI_INIT="$REF_MODEL" \
  REF_PATH="$REF_MODEL" \
  OUT_DIR="$prm_out" \
  LOG_DIR="$prm_log" \
  GPU_LIST="$GPU_LIST" \
  DSR_SUB="$DSR_SUB" \
  K="$PRM_TRAIN_K" \
  J="$j_phase" \
  STEPS_PER_RESPONSE="$PRM_TRAIN_S" \
  MAX_RESPONSE_LEN="$max_resp_phase" \
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
  train_prm "$current_ckpt" "$PHASE0_OUT" "$PHASE0_LOG" 0 initial
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
  # Cycle 1 from base SFT (no optimizer.pt) → launch_qwen_caspo.sh (from-scratch).
  # Cycles 2+ from RL ckpt (has optimizer.pt) → launch_caspo_refresh_resume.sh
  # (preserves AdamW + lr scheduler state across the refresh boundary).
  RL_OUT="$OUT_ROOT/cycle_${cycle}_rl"
  echo ""
  echo "[alt] $(date +%H:%M:%S) === CYCLE $cycle: RL [$current_step → $next_target] ==="
  echo "[alt] policy: $current_ckpt"
  echo "[alt] PRM:    $current_prm"
  echo "[alt] out:    $RL_OUT"

  if [[ -f "$current_ckpt/optimizer.pt" ]]; then
    echo "[alt] resume mode: warm (optimizer.pt + lr_scheduler preserved)"
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
  else
    echo "[alt] resume mode: from-scratch (no optimizer.pt at $current_ckpt; cycle 1 path)"
    PRM_PATH="$current_prm" \
    OUT_DIR="$RL_OUT" \
    GPU_LIST="$GPU_LIST" \
    DSR_SUB="$DSR_SUB" \
    MAX_STEPS="$next_target" \
    SAVE_EVERY=50 \
    SAVE_OPTIMIZER_STATE=true \
    ADV_TRANSFORM="$ADV_TRANSFORM" \
    LOG_DIR="$OUT_ROOT/logs/cycle_${cycle}_rl" \
    RUN_EVAL=false \
      bash scripts/launch_qwen_caspo.sh
  fi

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
  train_prm "$current_ckpt" "$PRM_OUT" "$PRM_LOG" "$cycle" refresh
  current_prm="$PRM_OUT/best"
  [[ -d "$current_prm" ]] || { echo "[alt] ERROR: $current_prm not found after train"; exit 1; }
  echo "[alt] $(date +%H:%M:%S) refresh complete. new PRM=$current_prm"
done

echo ""
echo "[alt] $(date +%H:%M:%S) === RL+REFRESH DONE: $cycle cycles, $current_step total RL steps ==="
echo "[alt] final policy: $current_ckpt"
echo "[alt] final PRM:    $current_prm"

# ---- Post-run greedy eval on every saved RL ckpt ----
# Mirrors launch_qwen_caspo.sh's built-in eval block but scoped across
# every cycle_*_rl/step_* dir (typically save_every=50 → 12 ckpts at
# 600 total steps), so we get the full pass@1 trajectory across cycles.
# Suppressed by RUN_EVAL=false. Parallel across GPU_LIST.
RUN_EVAL="${RUN_EVAL:-true}"
if [[ "$RUN_EVAL" == "true" ]]; then
  read -r -a GPUS <<< "$GPU_LIST"
  EVAL_DIR="$OUT_ROOT/eval"
  mkdir -p "$EVAL_DIR"
  echo ""
  echo "[alt] $(date +%H:%M:%S) === POST-RUN EVAL ==="

  # Collect all step_* ckpts across cycles, numerically sorted by step.
  mapfile -t CKPTS < <(
    for cy_dir in "$OUT_ROOT"/cycle_*_rl; do
      [[ -d "$cy_dir" ]] || continue
      for ck_dir in "$cy_dir"/step_*; do
        [[ -d "$ck_dir" ]] && echo "$ck_dir"
      done
    done | awk -F'/step_' '{print $2 "\t" $0}' | sort -n | cut -f2-
  )
  echo "[alt] eval ckpts (${#CKPTS[@]}):"
  printf '[alt]   %s\n' "${CKPTS[@]}"

  eval_one() {
    local gpu="$1" ckpt="$2"
    local tag="$(basename "$(dirname "$ckpt")")_$(basename "$ckpt")"
    local out="$EVAL_DIR/${tag}.json"
    local elog="$EVAL_DIR/${tag}.log"
    [[ -f "$out" ]] && { echo "[$tag] skip — exists"; return; }
    CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -u scripts/eval.py \
      --config configs/caspo_rho1b_math.yaml \
      --override "model_name_or_path=$ckpt" \
      --override "prompt_template={query}\nLet's think step by step and output the final answer within \\boxed{}." \
      --override "max_response_len=2048" \
      --benchmarks "math500,gsm8k,olympiadbench" \
      --k 1 --temperature 0.0 --top-p 1.0 \
      --max-new-tokens 3072 \
      --backend vllm --gpu-memory-utilization 0.85 \
      --output "$out" \
      > "$elog" 2>&1 \
    && echo "[$tag] $(date +%H:%M:%S) done" || echo "[$tag] FAIL"
  }

  i=0
  while (( i < ${#CKPTS[@]} )); do
    EPIDS=()
    for ((j=0; j<${#GPUS[@]} && i+j<${#CKPTS[@]}; j++)); do
      ( eval_one "${GPUS[$j]}" "${CKPTS[$((i+j))]}" ) &
      EPIDS+=("$!")
    done
    for pid in "${EPIDS[@]}"; do wait "$pid"; done
    i=$((i + ${#EPIDS[@]}))
  done
  echo "[alt] $(date +%H:%M:%S) === EVAL DONE: $EVAL_DIR ==="
fi

echo ""
echo "[alt] $(date +%H:%M:%S) === ALL DONE ==="
