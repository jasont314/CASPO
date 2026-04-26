#!/usr/bin/env bash
# 4-GPU FSDP PPO (terminal-reward, sequence-level advantages) on DeepSeekMath-7B-MATH.
METHOD=ppo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-ppo}"
GPU_DEFAULT_LIST="${GPU_DEFAULT_LIST:-0 1 2 3}"
EXTRA_OVERRIDES=(
    --override update_value_during_policy=false
)
source "$(dirname "$0")/_launch_7b_fsdp.sh"
