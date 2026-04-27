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
# Env knobs:
#   AUTO_EVAL_ON_FINISH (default 1) - when a training job exits with rc=0, kick
#     off scripts/launch_eval_all.sh on that method using the freed GPU. Has no
#     effect when WAIT_FOR_CHILDREN=0 (we don't wait, so we can't dispatch).
#   WATCHDOG (default 1) - spawn a sidecar bash loop that polls
#     scripts.health_check on each method's log every 60s and warns into
#     ${LOGDIR}/launcher_dispatcher.log on two consecutive STALE polls.
#     The watchdog only warns; it does not kill or restart anything.
#   WAIT_FOR_CHILDREN (default 1) - if 0, write the PID/GPU JSON map and exit
#     immediately without waiting (fire-and-forget detached suite). With 0,
#     AUTO_EVAL_ON_FINISH and WATCHDOG are forced off because there is no
#     parent process left to dispatch from.
#
# After launch, ${LOGDIR}/launcher_pids.json contains:
#   {"pids": {"grpo": 12345, ...}, "gpus": {"grpo": "0", ...}}
# for jq-friendly status queries.
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
# Watchdog needs a Python interpreter to run scripts.health_check; resolve it
# here so the watchdog subshell does not have to source conda init.
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/scalable/bin/python}"
RUN_SUFFIX="_${RUN_TAG}"
LOGDIR="$ROOT/caspo_rho1b_math${RUN_SUFFIX}/logs"
mkdir -p "$LOGDIR"

AUTO_EVAL_ON_FINISH="${AUTO_EVAL_ON_FINISH:-1}"
WATCHDOG="${WATCHDOG:-1}"
WAIT_FOR_CHILDREN="${WAIT_FOR_CHILDREN:-1}"
DISPATCHER_LOG="$LOGDIR/launcher_dispatcher.log"

# Detached mode forces the dispatcher pieces off — there is no parent shell
# alive to wait for child PIDs or run the watchdog after we exit.
if [[ "$WAIT_FOR_CHILDREN" == "0" ]]; then
    AUTO_EVAL_ON_FINISH=0
    WATCHDOG=0
fi

echo "[all8] RUN_TAG=${RUN_TAG} WANDB_MODE=${WANDB_MODE}"
echo "[all8] GPU map: grpo=${GPUS[0]} ppo=${GPUS[1]} vineppo=${GPUS[2]},${GPUS[3]} caspo=${GPUS[4]} prob=${GPUS[5]} logprob=${GPUS[6]} frozen=${GPUS[7]}"
echo "[all8] logs=${LOGDIR}"
echo "[all8] flags: AUTO_EVAL_ON_FINISH=${AUTO_EVAL_ON_FINISH} WATCHDOG=${WATCHDOG} WAIT_FOR_CHILDREN=${WAIT_FOR_CHILDREN}"

PIDS=()
EVAL_PIDS=()
WATCHDOG_PID=""

# PID -> method, PID -> gpu maps. Stored as parallel arrays of "key=value"
# strings because bash 4 associative-array support is uneven across environments
# we run on; lookups are O(N) but N=7.
declare -a PID_METHOD_KV=()
declare -a PID_GPU_KV=()
declare -a METHOD_LOG_KV=()
declare -a METHOD_NAMES=()

_kv_get() {
    # _kv_get <array_name> <key>
    local arr_name="$1"
    local key="$2"
    local -n arr="$arr_name"
    local entry
    for entry in "${arr[@]}"; do
        if [[ "${entry%%=*}" == "$key" ]]; then
            printf '%s' "${entry#*=}"
            return 0
        fi
    done
    return 1
}

cleanup_children() {
    local code=$?
    if [[ -n "$WATCHDOG_PID" ]] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
        kill "$WATCHDOG_PID" 2>/dev/null || true
    fi
    if (( code != 0 )); then
        echo "[all8] exiting with rc=${code}; launched pids: ${PIDS[*]:-none} eval pids: ${EVAL_PIDS[*]:-none}"
        local pid
        for pid in "${PIDS[@]}" "${EVAL_PIDS[@]}"; do
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
    local gpu="$2"
    shift 2
    local launcher_log="$LOGDIR/launcher_${name}.log"
    echo "[all8] launch ${name} on GPU(s) ${gpu}; launcher_log=${launcher_log}"
    "$@" > "$launcher_log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    PID_METHOD_KV+=("${pid}=${name}")
    PID_GPU_KV+=("${pid}=${gpu}")
    METHOD_LOG_KV+=("${name}=${launcher_log}")
    METHOD_NAMES+=("$name")
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

launch_job grpo            "${GPUS[0]}"                 env "${COMMON_ENV[@]}" "GPU=${GPUS[0]}"                       ./scripts/launch_rho1b_grpo.sh
launch_job ppo_critic      "${GPUS[1]}"                 env "${COMMON_ENV[@]}" "GPU=${GPUS[1]}"                       ./scripts/launch_rho1b_ppo_critic.sh
launch_job vineppo_ddp2    "${GPUS[2]},${GPUS[3]}"      env "${COMMON_ENV[@]}" "GPU_LIST=${GPUS[2]} ${GPUS[3]}"       ./scripts/launch_rho1b_vineppo_ddp2.sh
launch_job caspo           "${GPUS[4]}"                 env "${COMMON_ENV[@]}" "GPU=${GPUS[4]}"                       ./scripts/launch_rho1b_caspo.sh
launch_job caspo_prob      "${GPUS[5]}"                 env "${COMMON_ENV[@]}" "GPU=${GPUS[5]}"                       ./scripts/launch_rho1b_caspo_delta_prob.sh
launch_job caspo_logprob   "${GPUS[6]}"                 env "${COMMON_ENV[@]}" "GPU=${GPUS[6]}"                       ./scripts/launch_rho1b_caspo_delta_log_prob.sh
launch_job caspo_frozen_rm "${GPUS[7]}"                 env "${COMMON_ENV[@]}" "GPU_LIST=${GPUS[7]}"                  ./scripts/launch_rho1b_caspo_frozen_rm.sh

# Write a jq-friendly snapshot of {method: pid, method: gpu} so detached suites
# can be monitored without reading the bash process tree.
write_pids_json() {
    local out="$LOGDIR/launcher_pids.json"
    local tmp="${out}.tmp"
    {
        printf '{\n'
        printf '  "pids": {\n'
        local first=1 entry
        for entry in "${PID_METHOD_KV[@]}"; do
            local pid="${entry%%=*}"
            local method="${entry#*=}"
            if (( first )); then first=0; else printf ',\n'; fi
            printf '    "%s": %s' "$method" "$pid"
        done
        printf '\n  },\n'
        printf '  "gpus": {\n'
        first=1
        for entry in "${PID_GPU_KV[@]}"; do
            local pid="${entry%%=*}"
            local gpu="${entry#*=}"
            local method
            method="$(_kv_get PID_METHOD_KV "$pid")"
            if (( first )); then first=0; else printf ',\n'; fi
            printf '    "%s": "%s"' "$method" "$gpu"
        done
        printf '\n  }\n'
        printf '}\n'
    } > "$tmp"
    mv "$tmp" "$out"
    echo "[all8] wrote ${out}"
}
write_pids_json

# Map a launched method to its trainer's stdout log file. The all8 launcher
# captures the *wrapper's* output in launcher_<name>.log, but the trainer
# writes structured "[method step ...]" lines to phase2_*.log via the shared
# one-GPU body. health_check.py needs the trainer log for step parsing.
trainer_log_for() {
    local method="$1"
    case "$method" in
        grpo|ppo_critic|caspo|caspo_prob|caspo_logprob|caspo_frozen_rm)
            local tag
            case "$method" in
                grpo)             tag="grpo" ;;
                ppo_critic)       tag="ppo_critic" ;;
                caspo)            tag="caspo" ;;
                caspo_prob)       tag="caspo_delta_prob" ;;
                caspo_logprob)    tag="caspo_delta_log_prob" ;;
                caspo_frozen_rm)  tag="caspo_frozen_rm" ;;
            esac
            printf '%s/phase2_%s.log' "$LOGDIR" "$tag"
            ;;
        vineppo_ddp2)
            # Pick rank0 — both ranks emit step lines, parsing one is sufficient.
            printf '%s/phase2_vineppo_ddp2_rank0.log' "$LOGDIR"
            ;;
        *)
            printf '%s/launcher_%s.log' "$LOGDIR" "$method"
            ;;
    esac
}

# --- Watchdog sidecar -------------------------------------------------------
#
# Polls health_check on each method's trainer log every 60s. If a method shows
# STATUS: STALE on two consecutive polls AND its training PID is still alive,
# write a single warning line to ${DISPATCHER_LOG}. The watchdog never kills
# or restarts anything in this version — we just want visibility.
#
# Race notes:
#  * The watchdog reads PID liveness with `kill -0`; a PID that exits between
#    the health_check and the kill -0 will simply not warn this round.
#  * We don't share state with the dispatcher loop; a method that finishes
#    cleanly will stop appearing in the watchdog's PID set on the next poll
#    because `kill -0` returns nonzero.
#  * Trap forwarding: parent INT/TERM goes through cleanup_children which
#    kills the watchdog by PID. The watchdog itself traps INT/TERM to its
#    own exit so a stray `kill <watchdog>` propagates cleanly.
start_watchdog() {
    if [[ "$WATCHDOG" != "1" ]]; then
        return 0
    fi
    local pids_csv="${PIDS[*]}"
    # Build "method:log" pairs to feed the subshell — we pass them via env
    # because bash subshells lose array context cleanly across `&`.
    local methods_csv=""
    local logs_csv=""
    local m
    for m in "${METHOD_NAMES[@]}"; do
        local tlog
        tlog="$(trainer_log_for "$m")"
        methods_csv+="${m}|"
        logs_csv+="${tlog}|"
    done
    # Build per-pid -> method index for the subshell.
    local pid_method_csv=""
    local entry
    for entry in "${PID_METHOD_KV[@]}"; do
        pid_method_csv+="${entry}|"
    done

    (
        trap 'exit 0' INT TERM
        # Per-method consecutive STALE counter, parallel to METHOD_NAMES.
        declare -A stale_count=()
        IFS='|' read -r -a _methods <<< "${methods_csv%|}"
        IFS='|' read -r -a _logs <<< "${logs_csv%|}"
        IFS='|' read -r -a _pid_method_pairs <<< "${pid_method_csv%|}"
        # Build method -> pid map inside the subshell.
        declare -A method_pid=()
        local pair
        for pair in "${_pid_method_pairs[@]}"; do
            method_pid["${pair#*=}"]="${pair%%=*}"
        done

        local i
        while true; do
            local any_alive=0
            for i in "${!_methods[@]}"; do
                local meth="${_methods[$i]}"
                local mlog="${_logs[$i]}"
                local mpid="${method_pid[$meth]:-}"
                if [[ -z "$mpid" ]]; then
                    continue
                fi
                if ! kill -0 "$mpid" 2>/dev/null; then
                    # Trainer exited; reset its stale counter and skip.
                    stale_count["$meth"]=0
                    continue
                fi
                any_alive=1
                if [[ ! -f "$mlog" ]]; then
                    # Trainer log not yet created (warmup); don't count as stale.
                    continue
                fi
                local out rc
                # health_check exits 1 on STALE, 2 on missing log, 0 on healthy.
                # We grep the STATUS line to disambiguate STALE from CRASHED.
                out="$("$PYTHON_BIN" -m scripts.health_check --log "$mlog" --stale-secs 1200 2>&1 || true)"
                if printf '%s\n' "$out" | grep -q '^STATUS: STALE'; then
                    stale_count["$meth"]=$(( ${stale_count["$meth"]:-0} + 1 ))
                    if (( ${stale_count["$meth"]} >= 2 )); then
                        printf '[%s] WATCHDOG WARN method=%s pid=%s: STALE for %d consecutive polls; log=%s\n' \
                            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$meth" "$mpid" "${stale_count[$meth]}" "$mlog" \
                            >> "$DISPATCHER_LOG"
                    fi
                else
                    stale_count["$meth"]=0
                fi
            done
            if (( any_alive == 0 )); then
                printf '[%s] WATCHDOG: no live training PIDs; exiting\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$DISPATCHER_LOG"
                exit 0
            fi
            sleep 60
        done
    ) >> "$DISPATCHER_LOG" 2>&1 &
    WATCHDOG_PID=$!
    echo "[all8] watchdog pid=${WATCHDOG_PID} log=${DISPATCHER_LOG}"
}

# --- Auto-eval dispatch -----------------------------------------------------
#
# Kick off scripts/launch_eval_all.sh on a freed GPU when a method exits with
# rc=0. Failed methods are NOT auto-evaluated; their dispatcher line will
# record the non-zero rc instead.
dispatch_eval() {
    local method="$1"
    local gpu="$2"
    if [[ "$AUTO_EVAL_ON_FINISH" != "1" ]]; then
        return 0
    fi
    # vineppo_ddp2 occupies two GPUs; pick the first for eval (eval is 1-GPU).
    local eval_gpu="${gpu%%,*}"
    printf '[%s] DISPATCH eval method=%s on GPU %s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$method" "$eval_gpu" >> "$DISPATCHER_LOG"
    METHODS="$method" EVAL_GPU_LIST="$eval_gpu" RUN_TAG="$RUN_TAG" ROOT="$ROOT" \
        ./scripts/launch_eval_all.sh \
        >> "$LOGDIR/launcher_eval_${method}.log" 2>&1 &
    local epid=$!
    EVAL_PIDS+=("$epid")
    printf '[%s] DISPATCH eval method=%s pid=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$method" "$epid" >> "$DISPATCHER_LOG"
}

if [[ "$WAIT_FOR_CHILDREN" != "1" ]]; then
    echo "[all8] WAIT_FOR_CHILDREN=0 - detaching; PIDs in ${LOGDIR}/launcher_pids.json"
    trap - INT TERM
    exit 0
fi

start_watchdog

# Initialize the dispatcher log up-front so tail -f starts working immediately.
printf '[%s] DISPATCH start - tracking %d training pids\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${#PIDS[@]}" >> "$DISPATCHER_LOG"

# wait -n returns when ANY child exits. We use "$pid" so $? is that child's rc.
# Note: wait -n returns 127 if there are no children to wait for; we guard
# with a counter so that cannot happen here.
fail=0
remaining_train=${#PIDS[@]}
declare -A pid_done=()
while (( remaining_train > 0 )); do
    # `wait -n -p done_pid` is bash >= 5.1; fall back to plain `wait -n` and
    # determine the freed pid by polling kill -0 on each tracked pid. We use
    # the modern form when available.
    done_pid=""
    if wait -n -p done_pid "${PIDS[@]}" 2>/dev/null; then
        rc=0
    else
        rc=$?
        # If `wait -n -p` is not supported, rc may be 2 (usage error). Detect
        # by trying the older form.
        if [[ -z "$done_pid" ]] && (( rc == 2 )); then
            # Fallback path: use plain wait -n and discover which pid exited.
            if wait -n; then rc=0; else rc=$?; fi
            local_pid=""
            for p in "${PIDS[@]}"; do
                if [[ -n "${pid_done[$p]:-}" ]]; then continue; fi
                if ! kill -0 "$p" 2>/dev/null; then
                    local_pid="$p"
                    break
                fi
            done
            done_pid="$local_pid"
        fi
    fi

    if [[ -z "$done_pid" ]]; then
        # Could not identify which child finished; decrement and continue.
        remaining_train=$((remaining_train - 1))
        continue
    fi
    if [[ -n "${pid_done[$done_pid]:-}" ]]; then
        continue
    fi
    pid_done[$done_pid]=1
    remaining_train=$((remaining_train - 1))

    method="$(_kv_get PID_METHOD_KV "$done_pid" || echo unknown)"
    gpu="$(_kv_get PID_GPU_KV "$done_pid" || echo unknown)"

    if (( rc == 0 )); then
        printf '[%s] FINISH method=%s pid=%s gpu=%s rc=0\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$method" "$done_pid" "$gpu" \
            >> "$DISPATCHER_LOG"
        dispatch_eval "$method" "$gpu"
    else
        fail=$((fail + 1))
        echo "[all8] pid=${done_pid} (${method}) exited with rc=${rc}"
        printf '[%s] FINISH method=%s pid=%s gpu=%s rc=%d (NO AUTO-EVAL)\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$method" "$done_pid" "$gpu" "$rc" \
            >> "$DISPATCHER_LOG"
    fi
done

# Wait for any in-flight eval jobs we kicked off.
if (( ${#EVAL_PIDS[@]} > 0 )); then
    echo "[all8] waiting on ${#EVAL_PIDS[@]} dispatched eval job(s)"
    for epid in "${EVAL_PIDS[@]}"; do
        if wait "$epid"; then
            :
        else
            erc=$?
            echo "[all8] eval pid=${epid} exited with rc=${erc}"
        fi
    done
fi

# Watchdog should exit on its own once no training PIDs remain, but make sure.
if [[ -n "$WATCHDOG_PID" ]] && kill -0 "$WATCHDOG_PID" 2>/dev/null; then
    kill "$WATCHDOG_PID" 2>/dev/null || true
    wait "$WATCHDOG_PID" 2>/dev/null || true
fi

trap - INT TERM

if (( fail > 0 )); then
    echo "[all8] DONE - ${fail}/${#PIDS[@]} jobs failed; check ${LOGDIR}"
    exit 1
fi

echo "[all8] DONE - all ${#PIDS[@]} jobs completed cleanly"
