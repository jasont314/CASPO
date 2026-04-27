#!/usr/bin/env bash
set -eo pipefail
# Disaggregated PPO+critic launcher for direct head-to-head against
# VinePPO. Schulman 2017 PPO with a separate ~14 GB value network
# (see caspo/critic/) trained jointly via clipped-MSE on GAE returns.
#
# Memory budget at 7B FSDP=4 disagg:
#   policy   shard +Adam: ~24.5 GB/rank
#   ref      shard:        ~3.5 GB/rank
#   critic   shard +Adam: ~24.5 GB/rank   ← NEW vs critic-free PPO
#   activations (peak):  ~10-15 GB/rank
#   total per rank:      ~62-67 GB on 80 GB H100 (tight; full AC mandatory)
METHOD=ppo_critic
RUN_METHOD_TAG="${RUN_METHOD_TAG:-ppo_critic}"
TRAIN_GPU_DEFAULT_LIST="${TRAIN_GPU_DEFAULT_LIST:-0 1 2 3}"
ROLLOUT_GPU_DEFAULT_LIST="${ROLLOUT_GPU_DEFAULT_LIST:-4 5 6 7}"
EXTRA_OVERRIDES=(
    --override "kl_coef=${KL_COEF:-1.0e-4}"
    --override "value_loss_coef=${VALUE_LOSS_COEF:-0.1}"
    --override "cliprange_value=${CLIPRANGE_VALUE:-0.2}"
    --override "ppo_gae_lambda=${PPO_GAE_LAMBDA:-0.95}"
    --override "critic_lr=${CRITIC_LR:-1.0e-5}"
)
# Critic memory pressure forces full activation checkpointing.
export CASPO_USE_GRADIENT_CHECKPOINTING="${CASPO_USE_GRADIENT_CHECKPOINTING:-true}"
source "$(dirname "$0")/_launch_7b_disagg.sh"
