#!/usr/bin/env bash
# Standard one-GPU Rho-1B MATH CASPO delta-log-probability ablation.
#
# This computes CASPO step TD on log sigmoid(V) before step-advantage normalization.
# Default 8-GPU-suite placement: GPU 6.
# Override with GPU=<id> or GPU_LIST="<id>".
METHOD=caspo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-caspo_logprob}"
GPU_DEFAULT="${GPU_DEFAULT:-6}"
EXTRA_OVERRIDES=(
    --override caspo_advantage_transform=logprob
    --override update_value_during_policy=true
)
source "$(dirname "$0")/_launch_rho1b_one_gpu.sh"
