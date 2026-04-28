#!/usr/bin/env bash
set -eo pipefail
# Standard one-GPU Rho-1B MATH PPO+critic run (Schulman 2017 with a
# learned value network). Mirrors the 7B ppo_critic config; uses
# CASPO's 1B launcher defaults (mb=8, accum=8, ckpt=false,
# vllm_util=0.30) so the comparison is config-matched.
#
# Default 8-GPU-suite placement: GPU 5 (alongside CASPO on GPU 4).
# Override with GPU=<id> or GPU_LIST="<id>".
METHOD=ppo_critic
RUN_METHOD_TAG="${RUN_METHOD_TAG:-ppo_critic}"
GPU_DEFAULT="${GPU_DEFAULT:-5}"
EXTRA_OVERRIDES=(
    --override "kl_estimator=${KL_ESTIMATOR:-k3}"
    --override "value_loss_coef=${VALUE_LOSS_COEF:-1.0}"
    --override "cliprange_value=${CLIPRANGE_VALUE:-0.2}"
    --override "ppo_gae_lambda=${PPO_GAE_LAMBDA:-1.0}"
    --override "critic_lr=${CRITIC_LR:-1.0e-7}"
    --override "critic_weight_decay=${CRITIC_WEIGHT_DECAY:-0.0}"
    --override "critic_grad_clip=${CRITIC_GRAD_CLIP:-1.0}"
    --override "clip_eps_low=${CLIP_EPS_LOW:-0.2}"
    --override "clip_eps_high=${CLIP_EPS_HIGH:-0.2}"
    --override "epochs_per_rollout=${EPOCHS_PER_ROLLOUT:-2}"
)
# Note: kl_coef is NOT overridden here — let it inherit from the YAML
# (currently 1e-2). Previously hardcoded to 1e-4 which silently bypassed
# the YAML's stronger ref-anchor and caused PPO+critic to diverge while
# CASPO/CASPO-Δp/GRPO (which don't override) trained stably.
[[ -n "${KL_COEF:-}" ]] && EXTRA_OVERRIDES+=(--override "kl_coef=${KL_COEF}")
source "$(dirname "$0")/_launch_rho1b_one_gpu.sh"
