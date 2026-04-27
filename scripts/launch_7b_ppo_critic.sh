#!/usr/bin/env bash
set -eo pipefail
# 4-GPU FSDP PPO+critic (Schulman 2017) on DeepSeekMath-7B-MATH.
# Matches launch_7b_caspo.sh's topology: colocated FSDP trainer +
# vLLM on the same 4 GPUs at vllm_util=0.20, so PPO+critic and CASPO
# are head-to-head on identical hardware budget.
#
# VinePPO upstream config that this implements:
#   policIter_deepseekSft2_ppo_MATH.jsonnet
#     ← imports trainers/ppo_MATH.jsonnet
#     ← imports trainers/lam1.jsonnet            (lam=1.0)
#     ← imports trainers/refKl0.0001.jsonnet     (init_kl_coef=1e-4)
#     ← imports trainers/klLoss.jsonnet          (KL as loss term)
#
# Effective hyperparameters (verified 2026-04-27):
#   critic_lr=1e-6, value_loss_coef=1.0, cliprange_value=0.2
#   ppo_gae_lambda=1.0, kl_coef=1e-4
#   clip_eps_low=0.2, clip_eps_high=0.2
#   epochs_per_rollout=2, target_train_batch_size=64 (mb=2 × accum=8)
#
# Memory at 7B FSDP=4: critic adds a separate ~7B value model
# (~3.5 GB params + ~14 GB Adam per rank, sharded). Combined with
# policy + ref + activations + colocated vLLM at u=0.20, fits at
# mb=2 because:
#   - Phase G.F decoupled critic backward halves activation peak
#   - Phase G.H FSDP-wraps the critic per-LlamaDecoderLayer
# For an 8-GPU disaggregated variant (matches VinePPO upstream's PPO
# baseline topology), use ``launch_7b_ppo_critic_disagg.sh``.
METHOD=ppo_critic
RUN_METHOD_TAG="${RUN_METHOD_TAG:-ppo_critic}"
GPU_DEFAULT_LIST="${GPU_DEFAULT_LIST:-0 1 2 3}"
# Lower vLLM util default (matches CASPO's 0.20) so the trainer has
# headroom for the extra critic + Adam.
CASPO_VLLM_GPU_MEMORY_UTILIZATION="${CASPO_VLLM_GPU_MEMORY_UTILIZATION:-0.20}"
export CASPO_VLLM_GPU_MEMORY_UTILIZATION
EXTRA_OVERRIDES=(
    --override "kl_coef=${KL_COEF:-1.0e-4}"
    --override "kl_estimator=${KL_ESTIMATOR:-k3}"
    --override "value_loss_coef=${VALUE_LOSS_COEF:-1.0}"
    --override "cliprange_value=${CLIPRANGE_VALUE:-0.2}"
    --override "ppo_gae_lambda=${PPO_GAE_LAMBDA:-1.0}"
    --override "critic_lr=${CRITIC_LR:-1.0e-6}"
    --override "critic_weight_decay=${CRITIC_WEIGHT_DECAY:-0.0}"
    --override "critic_grad_clip=${CRITIC_GRAD_CLIP:-1.0}"
    --override "clip_eps_low=${CLIP_EPS_LOW:-0.2}"
    --override "clip_eps_high=${CLIP_EPS_HIGH:-0.2}"
    --override "epochs_per_rollout=${EPOCHS_PER_ROLLOUT:-2}"
)
source "$(dirname "$0")/_launch_7b_fsdp.sh"
