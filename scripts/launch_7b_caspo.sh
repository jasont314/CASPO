#!/usr/bin/env bash
set -eo pipefail
# 4-GPU FSDP CASPO with online IPVRM updates on DeepSeekMath-7B-MATH.
# Requires a trained 7B value model at cfg.prefix_value_path.
#
# CASPO online keeps a trainable copy of phi (~14 GB params + 56 GB Adam
# state per rank at FSDP=4) IN ADDITION to the policy + ref + policy Adam.
# Verified empirically (Apr 2026): u=0.30 OOMs with 726 MB free, u=0.20
# fits comfortably. Steady state: ~45 s/step. PPO/GRPO/caspo-frozen-rm
# stay at u=0.30 since they don't carry value Adam.
METHOD=caspo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-caspo}"
GPU_DEFAULT_LIST="${GPU_DEFAULT_LIST:-0 1 2 3}"
# Lower vLLM util default for CASPO online; trainer needs the headroom.
CASPO_VLLM_GPU_MEMORY_UTILIZATION="${CASPO_VLLM_GPU_MEMORY_UTILIZATION:-0.20}"
export CASPO_VLLM_GPU_MEMORY_UTILIZATION
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
