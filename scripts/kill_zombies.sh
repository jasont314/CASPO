#!/usr/bin/env bash
# Kill orphaned vLLM EngineCore subprocesses.
# When the parent train_caspo process gets killed, the EngineCore can survive
# and hold ~37 GB of GPU memory hostage. This finds and kills any EngineCore
# whose parent PID is no longer alive.
#
# Exit codes:
#   0 - no zombies found, or all zombies killed and GPU memory freed
#   1 - zombies found but some failed to die after SIGKILL
#   2 - zombies killed but GPU memory not freed (driver/other process holding it)
#   3 - usage / unexpected error
set -uo pipefail

# Use a regex that won't match this script itself (the [V] trick prevents
# the grep pattern from matching its own command line).
PATTERN='[V]LLM::EngineCore'
SELF_PID=$$

is_numeric() { [[ "$1" =~ ^[0-9]+$ ]]; }

# Re-verifies a candidate is still a zombie at kill time (race-safe).
# A process is a zombie if:
#   - it still exists, AND
#   - its current parent is init (PID 1) OR its parent is dead, AND
#   - its cmdline still matches the EngineCore pattern.
is_still_zombie() {
    local pid="$1"
    is_numeric "$pid" || return 1
    [[ -d "/proc/$pid" ]] || return 1
    # Confirm cmdline still matches (defends against PID reuse between scan and kill).
    if ! tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "VLLM::EngineCore"; then
        return 1
    fi
    local ppid
    ppid=$(awk '/^PPid:/ { print $2 }' "/proc/$pid/status" 2>/dev/null)
    if [[ -z "$ppid" ]]; then
        return 0
    fi
    if [[ "$ppid" == "1" ]]; then
        return 0
    fi
    if ! kill -0 "$ppid" 2>/dev/null; then
        return 0
    fi
    return 1
}

# Capture pre-kill GPU memory to detect whether kills actually freed memory.
gpu_mem_total_used() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk '{ sum += $1 } END { print sum+0 }'
}

mem_before=$(gpu_mem_total_used)

# Collect candidate PIDs. pgrep -f matches against full cmdline; safer than ps|grep.
mapfile -t candidates < <(pgrep -f "VLLM::EngineCore" 2>/dev/null || true)

zombies=()
for pid in "${candidates[@]}"; do
    is_numeric "$pid" || continue
    [[ "$pid" == "$SELF_PID" ]] && continue
    if is_still_zombie "$pid"; then
        zombies+=("$pid")
    fi
done

if (( ${#zombies[@]} == 0 )); then
    echo "no zombie EngineCore processes found"
    exit 0
fi

echo "found zombie EngineCore pids: ${zombies[*]}"

# Try SIGTERM first for graceful shutdown.
to_kill=()
for pid in "${zombies[@]}"; do
    # Re-verify just before sending the signal (race-safe).
    if is_still_zombie "$pid"; then
        to_kill+=("$pid")
    else
        echo "  pid $pid no longer a zombie, skipping"
    fi
done

if (( ${#to_kill[@]} == 0 )); then
    echo "all candidates resolved themselves before kill"
    exit 0
fi

echo "sending SIGTERM to: ${to_kill[*]}"
kill -TERM "${to_kill[@]}" 2>/dev/null || true

# Wait up to 5s for graceful exit.
for _ in 1 2 3 4 5; do
    sleep 1
    still_alive=()
    for pid in "${to_kill[@]}"; do
        if [[ -d "/proc/$pid" ]] && is_still_zombie "$pid"; then
            still_alive+=("$pid")
        fi
    done
    (( ${#still_alive[@]} == 0 )) && break
done

# SIGKILL anything that survived.
survivors=()
for pid in "${to_kill[@]}"; do
    if [[ -d "/proc/$pid" ]] && is_still_zombie "$pid"; then
        survivors+=("$pid")
    fi
done

if (( ${#survivors[@]} > 0 )); then
    echo "sending SIGKILL to: ${survivors[*]}"
    kill -KILL "${survivors[@]}" 2>/dev/null || true
    sleep 2
fi

# Final liveness check.
final_alive=()
for pid in "${to_kill[@]}"; do
    if [[ -d "/proc/$pid" ]] && is_still_zombie "$pid"; then
        final_alive+=("$pid")
    fi
done

# GPU memory verification (driver may take a few seconds to release).
mem_after=$(gpu_mem_total_used)
echo "gpu memory used: ${mem_before} MiB -> ${mem_after} MiB (across all GPUs)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -8

if (( ${#final_alive[@]} > 0 )); then
    echo "ERROR: pids still alive after SIGKILL: ${final_alive[*]}"
    exit 1
fi

# If we killed something but GPU memory didn't drop appreciably, warn.
# Threshold: 100 MiB tolerance for normal driver bookkeeping noise.
if (( mem_before - mem_after < 100 )) && (( mem_before > 1000 )); then
    echo "WARN: killed ${#to_kill[@]} process(es) but GPU memory did not drop noticeably"
    echo "      (some other process may be holding GPU memory)"
    exit 2
fi

echo "killed ${#to_kill[@]} zombie EngineCore process(es); GPU memory freed"
exit 0
