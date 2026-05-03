#!/usr/bin/env bash
# Launch GRPO + PPO+Critic + CASPO + CASPO-Δp on Rho-1B-MATH in parallel,
# one method per tmp drive so 10 ckpts per run (save_every=100 over
# 1000 steps) all fit comfortably with no cross-method contention.
#
# GPU + drive layout (override with GPU_LIST and ROOT_{1..4}):
#   GPU 4 → GRPO        → /mnt/nvme_tmp2  (~153 GB; policy only)
#   GPU 5 → PPO+Critic  → /mnt/nvme_tmp4  (~270 GB; +critic_optimizer.pt
#                                          ~12 GB/ckpt, LM head stripped)
#   GPU 6 → CASPO       → /mnt/nvme_tmp3  (~285 GB; +value_optimizer.pt
#                                          ~13 GB/ckpt, full Rho-1B-size phi)
#   GPU 7 → CASPO Δp    → /mnt/nvme_tmp5  (~285 GB; same as CASPO,
#                                          only the advantage transform differs)
#
# Per-drive footprint @ save_every=100 (10 ckpts at steps 100, 200, …, 1000):
#   nvme_tmp2 (GRPO):       ~153 GB / 350 GB free  (~197 GB headroom)
#   nvme_tmp3 (CASPO):      ~285 GB / 350 GB free  (~65 GB headroom)
#   nvme_tmp4 (PPO+critic): ~270 GB / 350 GB free  (~80 GB headroom)
#   nvme_tmp5 (CASPO Δp):   ~285 GB / 350 GB free  (~65 GB headroom)
#
# Override env vars:
#   ROOT_1     output root for GRPO        (default /mnt/nvme_tmp2/jason_caspo)
#   ROOT_2     output root for PPO+critic  (default /mnt/nvme_tmp4/jason_caspo)
#   ROOT_3     output root for CASPO       (default /mnt/nvme_tmp3/jason_caspo)
#   ROOT_4     output root for CASPO Δp    (default /mnt/nvme_tmp5/jason_caspo)
#   GPU_LIST   four physical GPU ids       (default "4 5 6 7")
#   RUN_TAG    suffix appended to run dirs (default "paper_seed0")
#   SAVE_EVERY ckpt cadence                (default 100; YAML default is 200)
#   MAX_STEPS  total outer steps           (default 1000, from YAML)
#
# Usage:
#   bash scripts/launch_rho1b_4method_split.sh
#   RUN_TAG=run2 GPU_LIST="0 1 2 3" bash scripts/launch_rho1b_4method_split.sh
set -eo pipefail
# Don't use 'set -u' — conda activate scripts have unbound vars.

cd "$(dirname "$0")/.."
SCRIPTS_DIR="$(pwd)/scripts"

ROOT_1="${ROOT_1:-/mnt/nvme_tmp2/jason_caspo}"  # GRPO
ROOT_2="${ROOT_2:-/mnt/nvme_tmp4/jason_caspo}"  # PPO+critic
ROOT_3="${ROOT_3:-/mnt/nvme_tmp3/jason_caspo}"  # CASPO
ROOT_4="${ROOT_4:-/mnt/nvme_tmp5/jason_caspo}"  # CASPO Δp
RUN_TAG="${RUN_TAG:-paper_seed0}"
SAVE_EVERY="${SAVE_EVERY:-100}"

read -r -a GPUS <<< "${GPU_LIST:-4 5 6 7}"
if (( ${#GPUS[@]} < 4 )); then
    echo "[4method] ERROR: GPU_LIST must contain at least 4 GPU ids; got: ${GPU_LIST:-}"
    exit 2
fi

# Verify all 4 drives exist + writable before launching any trainers.
for root in "$ROOT_1" "$ROOT_2" "$ROOT_3" "$ROOT_4"; do
    parent="$(dirname "$root")"
    if [[ ! -d "$parent" ]]; then
        echo "[4method] ERROR: parent dir does not exist: $parent (drive not mounted?)"
        exit 2
    fi
    mkdir -p "$root"
    if [[ ! -w "$root" ]]; then
        echo "[4method] ERROR: not writable: $root"
        exit 2
    fi
done

# Top-level launcher logs live with the GRPO drive (lightest).
PARENT_LOGDIR="$ROOT_1/caspo_rho1b_math_${RUN_TAG}/launcher_logs"
mkdir -p "$PARENT_LOGDIR"

PIDS=()
declare -A PID_TO_METHOD

launch() {
    local method="$1" gpu="$2" root="$3" wrapper="$4"
    local out="$PARENT_LOGDIR/${method}_launcher.out"
    echo "[4method] ${method} → GPU ${gpu} → ROOT=${root}"
    ROOT="$root" GPU="$gpu" RUN_TAG="$RUN_TAG" SAVE_EVERY="$SAVE_EVERY" \
        nohup bash "$SCRIPTS_DIR/$wrapper" > "$out" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    PID_TO_METHOD[$pid]="$method"
    echo "  pid=$pid launcher_log=$out"
}

launch grpo        "${GPUS[0]}" "$ROOT_1" launch_rho1b_grpo.sh
launch ppo_critic  "${GPUS[1]}" "$ROOT_2" launch_rho1b_ppo_critic.sh
launch caspo       "${GPUS[2]}" "$ROOT_3" launch_rho1b_caspo.sh
launch caspo_prob  "${GPUS[3]}" "$ROOT_4" launch_rho1b_caspo_delta_prob.sh

echo "[4method] all 4 launched (save_every=${SAVE_EVERY}). trainer logs:"
echo "  $ROOT_1/caspo_rho1b_math_${RUN_TAG}/logs/phase2_grpo.log"
echo "  $ROOT_2/caspo_rho1b_math_${RUN_TAG}/logs/phase2_ppo_critic.log"
echo "  $ROOT_3/caspo_rho1b_math_${RUN_TAG}/logs/phase2_caspo.log"
echo "  $ROOT_4/caspo_rho1b_math_${RUN_TAG}/logs/phase2_caspo_prob.log"

cleanup() {
    local rc=$?
    if (( rc != 0 )); then
        echo "[4method] exiting rc=$rc; launched pids: ${PIDS[*]:-none}"
    fi
    exit $rc
}
trap cleanup EXIT
trap 'echo "[4method] ERR at line $LINENO (rc=$?)"' ERR

fail=0
for pid in "${PIDS[@]}"; do
    method="${PID_TO_METHOD[$pid]}"
    if wait "$pid"; then
        echo "[4method] ${method} (pid=$pid) ✓"
    else
        rc=$?
        echo "[4method] ${method} (pid=$pid) FAILED rc=$rc"
        fail=$((fail + 1))
    fi
done

if (( fail > 0 )); then
    echo "[4method] DONE — $fail/${#PIDS[@]} jobs failed"
    exit 1
fi
echo "[4method] DONE — all ${#PIDS[@]} jobs complete"
