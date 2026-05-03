#!/usr/bin/env bash
set -eo pipefail
# Standard one-GPU Rho-1B MATH CASPO run with online IPVRM updates.
#
# Default 8-GPU-suite placement: GPU 4.
# Override with GPU=<id> or GPU_LIST="<id>".
METHOD=caspo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-caspo}"
GPU_DEFAULT="${GPU_DEFAULT:-4}"
EXTRA_OVERRIDES=(
    --override caspo_advantage_transform=value
    --override update_value_during_policy=true
)
source "$(dirname "$0")/_launch_rho1b_one_gpu.sh"
