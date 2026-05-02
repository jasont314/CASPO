#!/usr/bin/env bash
# One-shot status snapshot for the 4-method parallel Rho-1B run.
# For a refreshing dashboard, wrap with `watch`:
#   watch -n 10 bash scripts/watch_4method.sh
#
# Columns:
#   step       current_step / max_steps
#   t_step     wall-clock time of the most recent step (seconds)
#   t_avg(K)   mean t_step over the last $WATCH_AVG_K step lines (default 10)
#   ETA        (max_steps - current_step) × t_avg(K), formatted Hh Mm
#   mem_alloc  torch allocator (trainer working set) at end of last step
#   GPU_MiB    nvidia-smi current memory.used (includes vLLM KV cache)
#   GPU%       nvidia-smi current utilization.gpu (instantaneous)
#   elapsed    cumulative seconds since training began (from the log)
set -eo pipefail

RUN_TAG="${RUN_TAG:-paper_seed0}"
WATCH_AVG_K="${WATCH_AVG_K:-10}"

declare -A METHOD_LOG
METHOD_LOG[grpo]="/mnt/nvme_tmp2/jason_caspo/caspo_rho1b_math_${RUN_TAG}/logs/phase2_grpo.log"
METHOD_LOG[ppo_critic]="/mnt/nvme_tmp4/jason_caspo/caspo_rho1b_math_${RUN_TAG}/logs/phase2_ppo_critic.log"
METHOD_LOG[caspo]="/mnt/nvme_tmp3/jason_caspo/caspo_rho1b_math_${RUN_TAG}/logs/phase2_caspo.log"
METHOD_LOG[caspo_prob]="/mnt/nvme_tmp5/jason_caspo/caspo_rho1b_math_${RUN_TAG}/logs/phase2_caspo_prob.log"

declare -A METHOD_GPU
METHOD_GPU[grpo]=4
METHOD_GPU[ppo_critic]=5
METHOD_GPU[caspo]=6
METHOD_GPU[caspo_prob]=7

# Snapshot nvidia-smi once per invocation.
declare -A GPU_USED GPU_UTIL
while IFS=, read -r idx used util; do
    idx=$(echo "$idx" | xargs)
    used=$(echo "$used" | xargs | sed 's/ MiB//')
    util=$(echo "$util" | xargs | sed 's/ %//')
    GPU_USED[$idx]=$used
    GPU_UTIL[$idx]=$util
done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader)

# Format seconds → "Hh Mm" (or "Mm" if < 1 h).
fmt_eta() {
    local s="${1%.*}"
    if [[ -z "$s" || "$s" == "—" || "$s" == 0 ]]; then echo "—"; return; fi
    local h=$(( s / 3600 ))
    local m=$(( (s % 3600) / 60 ))
    if (( h > 0 )); then
        printf "%dh%02dm" "$h" "$m"
    else
        printf "%dm" "$m"
    fi
}

now="$(date +%H:%M:%S)"
echo "[watch_4method @ ${now}]   t_avg over last ${WATCH_AVG_K} steps"
printf "%-12s %3s %10s %8s %9s %9s %9s %8s %7s %8s\n" \
    "method" "GPU" "step" "t_step" "t_avg${WATCH_AVG_K}" "ETA" \
    "mem_alloc" "GPU_MiB" "GPU%" "elapsed"
printf "%-12s %3s %10s %8s %9s %9s %9s %8s %7s %8s\n" \
    "------" "---" "----" "------" "------" "---" \
    "---------" "-------" "----" "-------"

for method in grpo ppo_critic caspo caspo_prob; do
    log="${METHOD_LOG[$method]}"
    gpu="${METHOD_GPU[$method]}"

    if [[ ! -f "$log" ]]; then
        printf "%-12s %3s %10s %8s %9s %9s %9s %7sM %6s%% %8s\n" \
            "$method" "$gpu" "(no log)" "—" "—" "—" "—" \
            "${GPU_USED[$gpu]:-?}" "${GPU_UTIL[$gpu]:-?}" "—"
        continue
    fi

    # Most-recent step line (anchored on "step N/M" — both caspo and caspo_prob
    # log under METHOD=caspo, so we don't anchor on the method tag).
    step_line="$(grep -E "step [0-9]+/[0-9]+\]" "$log" 2>/dev/null | tail -1 || true)"
    if [[ -z "$step_line" ]]; then
        printf "%-12s %3s %10s %8s %9s %9s %9s %7sM %6s%% %8s\n" \
            "$method" "$gpu" "loading" "—" "—" "—" "—" \
            "${GPU_USED[$gpu]:-?}" "${GPU_UTIL[$gpu]:-?}" "—"
        continue
    fi

    step="$(echo "$step_line" | grep -oE "step [0-9]+/[0-9]+" | head -1 | sed 's/step //')"
    t_step_v="$(echo "$step_line" | grep -oE "t_step=[0-9.]+s" | sed 's/t_step=//;s/s$//' | head -1)"
    mem="$(echo "$step_line" | grep -oE "mem_alloc=[0-9.]+G" | sed 's/mem_alloc=//' | head -1)"
    elapsed="$(echo "$step_line" | grep -oE "elapsed=[0-9.]+s" | sed 's/elapsed=//' | head -1)"

    cur_step="${step%%/*}"
    max_step="${step##*/}"

    # Rolling mean over last K t_step values.
    t_avg="$(grep -oE "t_step=[0-9.]+s" "$log" 2>/dev/null \
        | tail -n "$WATCH_AVG_K" \
        | sed 's/t_step=//;s/s$//' \
        | awk 'BEGIN{n=0;s=0} {s+=$1; n++} END{ if (n>0) printf "%.1f", s/n; else print "—" }')"

    if [[ "$t_avg" == "—" || -z "$cur_step" || -z "$max_step" ]]; then
        eta_fmt="—"
        t_avg_fmt="—"
    else
        steps_left=$(( max_step - cur_step ))
        eta_sec="$(awk -v k="$steps_left" -v t="$t_avg" 'BEGIN{ printf "%.0f", k*t }')"
        eta_fmt="$(fmt_eta "$eta_sec")"
        t_avg_fmt="${t_avg}s"
    fi

    printf "%-12s %3s %10s %7ss %9s %9s %9s %7sM %6s%% %8s\n" \
        "$method" "$gpu" "$step" "${t_step_v:-—}" "$t_avg_fmt" "$eta_fmt" \
        "${mem:-—}" "${GPU_USED[$gpu]:-?}" "${GPU_UTIL[$gpu]:-?}" "${elapsed:-—}"
done

# Per-drive disk usage (informational; ckpt cadence is save_every=100).
echo
printf "%-22s %8s %8s\n" "drive" "used" "avail"
for d in /mnt/nvme_tmp2 /mnt/nvme_tmp4 /mnt/nvme_tmp3 /mnt/nvme_tmp5; do
    line="$(df -h --output=used,avail "$d" 2>/dev/null | tail -1)"
    printf "%-22s %s\n" "$d" "$line"
done
