#!/usr/bin/env bash
set -eo pipefail
# Standard one-GPU Rho-1B MATH CASPO delta-probability ablation.
#
# This computes CASPO step TD on sigmoid(V) before step-advantage normalization.
# Default 8-GPU-suite placement: GPU 5.
# Override with GPU=<id> or GPU_LIST="<id>".
METHOD=caspo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-caspo_prob}"
GPU_DEFAULT="${GPU_DEFAULT:-5}"
EXTRA_OVERRIDES=(
    --override caspo_advantage_transform=prob
    --override update_value_during_policy=true
)
source "$(dirname "$0")/_launch_rho1b_one_gpu.sh"
