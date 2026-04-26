#!/usr/bin/env bash
# 4-GPU FSDP CASPO with online IPVRM updates on DeepSeekMath-7B-MATH.
# Requires a trained 7B value model at cfg.prefix_value_path.
METHOD=caspo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-caspo}"
GPU_DEFAULT_LIST="${GPU_DEFAULT_LIST:-0 1 2 3}"
EXTRA_OVERRIDES=(
    --override caspo_advantage_transform=value
    --override update_value_during_policy=true
    --override use_adb=true
    --override use_dlw=true
    --override standardize_advantage_scope=batch
    --override advantage_clip=3.0
    --override kl_coef=1.0e-4
)
source "$(dirname "$0")/_launch_7b_fsdp.sh"
