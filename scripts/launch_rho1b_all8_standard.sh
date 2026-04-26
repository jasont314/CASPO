#!/usr/bin/env bash
# Launch the standard seven-experiment Rho-1B MATH suite on 8 GPUs.
#
# Default GPU map:
#   GPU 0: GRPO
#   GPU 1: PPO
#   GPU 2-3: VinePPO DDP2
#   GPU 4: CASPO
#   GPU 5: CASPO delta-prob
#   GPU 6: CASPO delta-log-prob
#   GPU 7: CASPO frozen RM
#
# Usage:
#   RUN_TAG=paper512_seed0 GPU_LIST="0 1 2 3 4 5 6 7" WANDB_MODE=offline \
#     ./scripts/launch_rho1b_all8_standard.sh
#
set -eo pipefail
# Don't use 'set -u' - launch scripts source conda activation.

cd "$(dirname "$0")/.."

read -r -a GPUS <<< "${GPU_LIST:-0 1 2 3 4 5 6 7}"
if (( ${#GPUS[@]} != 8 )); then
    echo "[all8] ERROR: GPU_LIST must contain exactly 8 GPU ids; got: ${GPU_LIST:-0 1 2 3 4 5 6 7}"
    exit 2
fi

ROOT="${ROOT:-/mnt/nvme_tmp/jason_caspo}"
RUN_TAG="${RUN_TAG:-paper512_seed0}"
WANDB_MODE="${WANDB_MODE:-offline}"
RUN_SUFFIX="_${RUN_TAG}"
LOGDIR="$ROOT/caspo_rho1b_math${RUN_SUFFIX}/logs"
mkdir -p "$LOGDIR"

echo "[all8] RUN_TAG=${RUN_TAG} WANDB_MODE=${WANDB_MODE}"
echo "[all8] GPU map: grpo=${GPUS[0]} ppo=${GPUS[1]} vineppo=${GPUS[2]},${GPUS[3]} caspo=${GPUS[4]} prob=${GPUS[5]} logprob=${GPUS[6]} frozen=${GPUS[7]}"
echo "[all8] logs=${LOGDIR}"

PIDS=()

cleanup_children() {
    local code=$?
    if (( code != 0 )); then
        echo "[all8] exiting with rc=${code}; launched pids: ${PIDS[*]:-none}"
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done
    fi
    exit "$code"
}
trap cleanup_children INT TERM

launch_job() {
    local name="$1"
    shift
    local launcher_log="$LOGDIR/launcher_${name}.log"
    echo "[all8] launch ${name}; launcher_log=${launcher_log}"
    "$@" > "$launcher_log" 2>&1 &
    PIDS+=("$!")
}

COMMON_ENV=(
    "RUN_TAG=${RUN_TAG}"
    "WANDB_MODE=${WANDB_MODE}"
    "ROOT=${ROOT}"
)
if [[ -n "${MAX_STEPS:-}" ]]; then
    COMMON_ENV+=("MAX_STEPS=${MAX_STEPS}")
fi
if [[ -n "${SAVE_EVERY:-}" ]]; then
    COMMON_ENV+=("SAVE_EVERY=${SAVE_EVERY}")
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    COMMON_ENV+=("WANDB_PROJECT=${WANDB_PROJECT}")
fi
if [[ -n "${PYTHON_BIN:-}" ]]; then
    COMMON_ENV+=("PYTHON_BIN=${PYTHON_BIN}")
fi

launch_job grpo env "${COMMON_ENV[@]}" "GPU=${GPUS[0]}" ./scripts/launch_rho1b_grpo.sh
launch_job ppo env "${COMMON_ENV[@]}" "GPU=${GPUS[1]}" ./scripts/launch_rho1b_ppo.sh
launch_job vineppo_ddp2 env "${COMMON_ENV[@]}" "GPU_LIST=${GPUS[2]} ${GPUS[3]}" ./scripts/launch_rho1b_vineppo_ddp2.sh
launch_job caspo env "${COMMON_ENV[@]}" "GPU=${GPUS[4]}" ./scripts/launch_rho1b_caspo.sh
launch_job caspo_prob env "${COMMON_ENV[@]}" "GPU=${GPUS[5]}" ./scripts/launch_rho1b_caspo_delta_prob.sh
launch_job caspo_logprob env "${COMMON_ENV[@]}" "GPU=${GPUS[6]}" ./scripts/launch_rho1b_caspo_delta_log_prob.sh
launch_job caspo_frozen_rm env "${COMMON_ENV[@]}" "GPU_LIST=${GPUS[7]}" ./scripts/launch_rho1b_caspo_frozen_rm.sh

fail=0
for pid in "${PIDS[@]}"; do
    if wait "$pid"; then
        :
    else
        rc=$?
        echo "[all8] pid=${pid} exited with rc=${rc}"
        fail=$((fail + 1))
    fi
done

trap - INT TERM

if (( fail > 0 )); then
    echo "[all8] DONE - ${fail}/${#PIDS[@]} jobs failed; check ${LOGDIR}"
    exit 1
fi

echo "[all8] DONE - all ${#PIDS[@]} jobs completed cleanly"
