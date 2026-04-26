#!/usr/bin/env bash
set -eo pipefail
# Standard one-GPU Rho-1B MATH CASPO frozen-RM ablation.
#
# This keeps CASPO's IPVRM prefix-value scoring, but disables online value-model
# updates. Default 8-GPU-suite placement: GPU 7.
# Override with GPU=<id> or GPU_LIST="<id>".
METHOD=caspo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-caspo_frozen_rm}"
GPU_DEFAULT="${GPU_DEFAULT:-7}"
EXTRA_OVERRIDES=(
    --override caspo_advantage_transform=value
    --override update_value_during_policy=false
)
source "$(dirname "$0")/_launch_rho1b_one_gpu.sh"
