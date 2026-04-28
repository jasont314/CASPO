#!/usr/bin/env bash
set -eo pipefail
# CASPO Δp (sigmoid-transform step-TD advantage) with online_value_lr=1e-4.
#
# Paper-faithful LR for V_φ online update (IPVRM Appendix C). The IPVRM
# paper uses 1e-4 with LoRA adapters; this script applies that LR to our
# full-fine-tune V_φ. Be aware: a prior memory entry
# (project_caspo_value_lr.md) records that an earlier full-FT run at
# 1e-4 collapsed the value model. With the post-Apr-28 stack
# (kl_coef=1e-2 strong KL anchor, advantage_clip=3.0, retrained V_φ at
# v2 path), it MAY now be stable — but treat this as an ablation, not
# a recommended default.
#
# Default 8-GPU-suite placement: GPU 5.
# Override with GPU=<id> or GPU_LIST="<id>".
METHOD=caspo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-caspo_prob_lr1e4}"
GPU_DEFAULT="${GPU_DEFAULT:-5}"
ONLINE_VALUE_LR="${ONLINE_VALUE_LR:-1.0e-4}"
export ONLINE_VALUE_LR
EXTRA_OVERRIDES=(
    --override caspo_advantage_transform=prob
    --override update_value_during_policy=true
)
source "$(dirname "$0")/_launch_rho1b_one_gpu.sh"
