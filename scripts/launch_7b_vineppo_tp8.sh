#!/usr/bin/env bash
set -eo pipefail
# Colocated TP=8 VinePPO K=9 launcher: trainer FSDP=8 + vLLM TP=8
# share all 8 GPUs. See docs/disaggregated_topology_plan.md (Phase 6+).
METHOD=vineppo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-vineppo_tp8}"
GPU_DEFAULT_LIST="${GPU_DEFAULT_LIST:-0 1 2 3 4 5 6 7}"
EXTRA_OVERRIDES=(
    --override update_value_during_policy=false
    --override "vineppo_mc_rollouts=${VINEPPO_MC_ROLLOUTS:-9}"
)
source "$(dirname "$0")/_launch_7b_tp8_colocated.sh"
