#!/usr/bin/env bash
set -eo pipefail
# 4-GPU FSDP CASPO with FROZEN reward model on DeepSeekMath-7B-MATH.
# Same as launch_7b_caspo.sh but disables online value updates.
METHOD=caspo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-caspo_frozen_rm}"
GPU_DEFAULT_LIST="${GPU_DEFAULT_LIST:-0 1 2 3}"
EXTRA_OVERRIDES=(
    --override caspo_advantage_transform=value
    --override update_value_during_policy=false
    --override use_adb=true
    --override use_dlw=true
    --override standardize_advantage_scope=batch
    --override advantage_clip=3.0
    --override kl_coef=1.0e-4
)
source "$(dirname "$0")/_launch_7b_fsdp.sh"
