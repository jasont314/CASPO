#!/usr/bin/env bash
# Auto-chain CASPO phases:
#   1. wait for phase 1a (collect) to finish — signal: value_data.pt exists
#   2. launch phase 1b (V_φ training) on GPU 0
#   3. wait for phase 1b to finish — signal: final/ exists with caspo_value_meta.json
#   4. launch CASPO RL (phase 2) on GPU 3
#
# Ignores GRPO/VinePPO which were launched separately on GPUs 1-2.
#
set -eo pipefail
# Don't use 'set -u' — conda activate scripts have unbound vars.
source /opt/conda/etc/profile.d/conda.sh
conda activate scalable
export HF_HOME=/mnt/nvme_tmp/jason_caspo/hf_cache
export HF_HUB_CACHE=/mnt/nvme_tmp/jason_caspo/hf_cache
export TRANSFORMERS_CACHE=/mnt/nvme_tmp/jason_caspo/hf_cache

cd "$(dirname "$0")/.."
source ./scripts/perf_env.sh

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/scalable/bin/python}"
trap 'echo "[chain] ERR at line $LINENO (rc=$?)"' ERR

ROOT=/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

VALUE_DATA="$ROOT/value_data.pt"
# train_value.py saves to <output_dir>/final/ and creates a value_final/ symlink.
# Either is acceptable — accept whichever appears first.
VALUE_FINAL_OPTS=(
    "$ROOT/value_final/caspo_value_meta.json"
    "$ROOT/final/caspo_value_meta.json"
)

echo "[chain] $(date) waiting for phase 1a → $VALUE_DATA"
until [[ -f "$VALUE_DATA" ]]; do
    if ! pgrep -f "scripts.collect_value_data.*caspo_rho1b_math" >/dev/null; then
        # Race: process may have just exited cleanly after writing the file.
        # Re-check existence before declaring failure.
        sleep 2
        if [[ -f "$VALUE_DATA" ]]; then
            break
        fi
        echo "[chain] $(date) ERROR: collect process is GONE but value_data.pt missing — aborting"
        exit 1
    fi
    sleep 60
done
echo "[chain] $(date) phase 1a DONE — value_data.pt ready"

echo "[chain] $(date) launching phase 1b (V_φ training) on GPU 0"
CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" -u -m scripts.train_value \
    --config configs/caspo_rho1b_math.yaml \
    > "$LOGDIR/phase1b_train_value.log" 2>&1
RC=$?
if [[ $RC -ne 0 ]]; then
    echo "[chain] $(date) ERROR: phase 1b exited with $RC — aborting"
    exit $RC
fi
FOUND=""
for c in "${VALUE_FINAL_OPTS[@]}"; do
    if [[ -f "$c" ]]; then FOUND="$c"; break; fi
done
if [[ -z "$FOUND" ]]; then
    echo "[chain] $(date) ERROR: phase 1b finished but no caspo_value_meta.json in either location — aborting"
    exit 1
fi
# Ensure the value_final symlink exists for the trainer config to find it.
if [[ ! -L "$ROOT/value_final" && ! -d "$ROOT/value_final" ]]; then
    ln -s "final" "$ROOT/value_final"
fi
echo "[chain] $(date) phase 1b DONE — V_φ trained ($FOUND)"

echo "[chain] $(date) launching CASPO RL on GPU 3"
CUDA_VISIBLE_DEVICES=3 nohup "$PYTHON_BIN" -u -m scripts.train_caspo \
    --config configs/caspo_rho1b_math.yaml \
    --override method=caspo \
    --override output_dir=/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo \
    --override wandb_run_name=rho1b_math_caspo_seed0 \
    --override wandb_project=caspo-rho1b-math \
    --override update_value_during_policy=true \
    --override use_adb=true \
    --override use_dlw=true \
    --override vllm_gpu_memory_utilization=0.45 \
    --override vllm_enforce_eager=false \
    > "$LOGDIR/phase2_caspo.log" 2>&1 &
CASPO_PID=$!
echo "[chain] $(date) CASPO RL launched pid=$CASPO_PID, log=$LOGDIR/phase2_caspo.log"
if wait "$CASPO_PID"; then
    echo "[chain] $(date) CASPO RL completed cleanly"
else
    RC=$?
    echo "[chain] $(date) ERROR: CASPO RL exited with $RC"
    exit "$RC"
fi
