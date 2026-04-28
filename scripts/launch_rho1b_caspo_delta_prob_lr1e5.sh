#!/usr/bin/env bash
set -eo pipefail
# CASPO Δp (sigmoid-transform step-TD advantage) with online_value_lr=1e-5.
#
# Identical to launch_rho1b_caspo_delta_prob.sh except the V_φ online
# update LR is bumped 10× from the default 1e-6 (which is conservative
# for full-FT) to 1e-5 (closer to the IPVRM paper's 1e-4 LoRA setting,
# but still 10× below the paper value to stay safely full-FT).
#
# Default 8-GPU-suite placement: GPU 5.
# Override with GPU=<id> or GPU_LIST="<id>".
METHOD=caspo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-caspo_prob_lr1e5}"
GPU_DEFAULT="${GPU_DEFAULT:-5}"
ONLINE_VALUE_LR="${ONLINE_VALUE_LR:-1.0e-5}"
export ONLINE_VALUE_LR
EXTRA_OVERRIDES=(
    --override caspo_advantage_transform=prob
    --override update_value_during_policy=true
)
source "$(dirname "$0")/_launch_rho1b_one_gpu.sh"
