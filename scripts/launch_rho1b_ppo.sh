#!/usr/bin/env bash
set -eo pipefail
# Standard one-GPU Rho-1B MATH PPO run.
#
# Default 8-GPU-suite placement: GPU 1.
# Override with GPU=<id> or GPU_LIST="<id>".
METHOD=ppo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-ppo}"
GPU_DEFAULT="${GPU_DEFAULT:-1}"
EXTRA_OVERRIDES=(
    --override update_value_during_policy=false
)
source "$(dirname "$0")/_launch_rho1b_one_gpu.sh"
