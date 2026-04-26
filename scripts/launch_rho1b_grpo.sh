#!/usr/bin/env bash
# Standard one-GPU Rho-1B MATH GRPO run.
#
# Default 8-GPU-suite placement: GPU 0.
# Override with GPU=<id> or GPU_LIST="<id>".
METHOD=grpo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-grpo}"
GPU_DEFAULT="${GPU_DEFAULT:-0}"
EXTRA_OVERRIDES=(
    --override update_value_during_policy=false
)
source "$(dirname "$0")/_launch_rho1b_one_gpu.sh"
