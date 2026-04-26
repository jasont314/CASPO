#!/usr/bin/env bash
set -eo pipefail
# 8-GPU FSDP VinePPO K=9 on DeepSeekMath-7B-MATH. K=9 MC continuations per
# step boundary on a 7B model is the heaviest method — 8 GPUs default.
# Override GPU_LIST to 4 GPUs if you want to compare throughput-per-GPU.
METHOD=vineppo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-vineppo}"
GPU_DEFAULT_LIST="${GPU_DEFAULT_LIST:-0 1 2 3 4 5 6 7}"
EXTRA_OVERRIDES=(
    --override update_value_during_policy=false
    --override "vineppo_mc_rollouts=${VINEPPO_MC_ROLLOUTS:-9}"
)
source "$(dirname "$0")/_launch_7b_fsdp.sh"
