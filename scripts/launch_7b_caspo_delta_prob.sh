#!/usr/bin/env bash
set -eo pipefail
# 4-GPU FSDP CASPO delta-probability ablation on DeepSeekMath-7B-MATH.
# Step TD computed on sigmoid(V) before step-advantage normalization.
# Same memory/topology profile as the standard CASPO online launcher.
METHOD=caspo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-caspo_prob}"
GPU_DEFAULT_LIST="${GPU_DEFAULT_LIST:-0 1 2 3}"
CASPO_VLLM_GPU_MEMORY_UTILIZATION="${CASPO_VLLM_GPU_MEMORY_UTILIZATION:-0.20}"
export CASPO_VLLM_GPU_MEMORY_UTILIZATION
EXTRA_OVERRIDES=(
    --override caspo_advantage_transform=prob
    --override update_value_during_policy=true
    --override use_adb=true
    --override use_dlw=true
    --override standardize_advantage_scope=batch
    --override advantage_clip=3.0
    --override kl_coef=1.0e-4
)
source "$(dirname "$0")/_launch_7b_fsdp.sh"
