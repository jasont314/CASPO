#!/usr/bin/env bash
set -eo pipefail
# Disaggregated PPO+critic launcher (Schulman 2017) for direct head-
# to-head against VinePPO. Implements VinePPO upstream's canonical
# DeepSeekMath SFT2 PPO baseline:
#
#   policIter_deepseekSft2_ppo_MATH.jsonnet
#     ← imports trainers/ppo_MATH.jsonnet
#     ← imports trainers/lam1.jsonnet            (lam=1.0)
#     ← imports trainers/refKl0.0001.jsonnet     (init_kl_coef=1e-4)
#     ← imports trainers/klLoss.jsonnet          (KL as loss term)
#
# Effective hyperparameters (verified 2026-04-27):
#   learning_rate          = 1e-6     (policy AND critic; VinePPO
#                                      uses the same scalar for both)
#   weight_decay           = 0.0
#   warmup_ratio           = 0.03
#   max_grad_norm          = 1.0
#   cliprange (policy)     = 0.2
#   cliprange_value        = 0.2
#   gamma                  = 1.0
#   lam (GAE λ)            = 1.0      (lam1.jsonnet override)
#   init_kl_coef           = 1e-4     (refKl0.0001.jsonnet override)
#   num_epochs_per_iteration = 2
#   target_train_batch_size  = 64    (matches our paper-faithful global)
#   whiten_advantages      = true
#
# Memory at 7B FSDP=4: critic adds a SEPARATE ~7B value model
# (~14 GB params + ~84 GB Adam state, sharded → ~24.5 GB/rank).
# Combined with policy + ref + activations + colocated vLLM, joint
# fwd+bwd OOMs at default mb=2. We ship with mb=1 + accum=16 to keep
# global=64 paper-faithful while halving the activation peak. If
# that still OOMs, ``CRITIC_FSDP_CPU_OFFLOAD=true`` enables PyTorch
# FSDP's CPU param-offload (slows step time ~30% but unblocks the
# topology — VinePPO upstream uses DeepSpeed Stage 2 + CPU offload
# for the same reason).
METHOD=ppo_critic
RUN_METHOD_TAG="${RUN_METHOD_TAG:-ppo_critic}"
TRAIN_GPU_DEFAULT_LIST="${TRAIN_GPU_DEFAULT_LIST:-0 1 2 3}"
ROLLOUT_GPU_DEFAULT_LIST="${ROLLOUT_GPU_DEFAULT_LIST:-4 5 6 7}"

# Empirical 7B PPO+critic optimum (2026-04-27, v11 smoke): mb=4,
# accum=4 keeps global=64 paper-faithful and runs ~76.7 s/step
# steady-state — 15% faster than mb=2/accum=8 (89.9 s) thanks to
# better GPU saturation in policy backward AND in the decoupled
# critic train. Earlier revisions forced mb=1/accum=16 because the
# joint policy+critic backward OOMed; Phase G.F decoupled the critic
# backward and Phase G.H wrapped the critic per-LlamaDecoderLayer
# (frees ~7 GB/rank) — together they make mb=4 fit on 4×H100/80GB.
export CASPO_MICRO_BATCH_SIZE="${CASPO_MICRO_BATCH_SIZE:-4}"
export CASPO_GRAD_ACCUM_STEPS="${CASPO_GRAD_ACCUM_STEPS:-4}"
# Optional escape hatch retained: PyTorch FSDP CPU param offload.
export CASPO_FSDP_CPU_OFFLOAD="${CASPO_FSDP_CPU_OFFLOAD:-false}"

EXTRA_OVERRIDES=(
    --override "kl_coef=${KL_COEF:-1.0e-4}"
    --override "kl_estimator=${KL_ESTIMATOR:-k3}"
    --override "value_loss_coef=${VALUE_LOSS_COEF:-1.0}"
    --override "cliprange_value=${CLIPRANGE_VALUE:-0.2}"
    --override "ppo_gae_lambda=${PPO_GAE_LAMBDA:-1.0}"
    --override "critic_lr=${CRITIC_LR:-1.0e-6}"
    --override "critic_weight_decay=${CRITIC_WEIGHT_DECAY:-0.0}"
    --override "critic_grad_clip=${CRITIC_GRAD_CLIP:-1.0}"
    # Match VinePPO's policy clip too (paper-faithful):
    --override "clip_eps_low=${CLIP_EPS_LOW:-0.2}"
    --override "clip_eps_high=${CLIP_EPS_HIGH:-0.2}"
    --override "epochs_per_rollout=${EPOCHS_PER_ROLLOUT:-2}"
)
source "$(dirname "$0")/_launch_7b_disagg.sh"
