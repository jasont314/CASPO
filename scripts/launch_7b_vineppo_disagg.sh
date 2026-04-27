#!/usr/bin/env bash
set -eo pipefail
# Disaggregated VinePPO K=9 launcher: trainer FSDP=4 on GPUs 0-3,
# vLLM TP=4 on dedicated GPUs 4-7. See docs/disaggregated_topology_plan.md.
METHOD=vineppo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-vineppo_disagg}"
TRAIN_GPU_DEFAULT_LIST="${TRAIN_GPU_DEFAULT_LIST:-0 1 2 3}"
ROLLOUT_GPU_DEFAULT_LIST="${ROLLOUT_GPU_DEFAULT_LIST:-4 5 6 7}"
EXTRA_OVERRIDES=(
    --override update_value_during_policy=false
    --override "vineppo_mc_rollouts=${VINEPPO_MC_ROLLOUTS:-9}"
)
source "$(dirname "$0")/_launch_7b_disagg.sh"
