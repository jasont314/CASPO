#!/usr/bin/env bash
set -eo pipefail
# 8-GPU FSDP VinePPO K=9 on DeepSeekMath-7B-MATH. K=9 MC continuations per
# step boundary on a 7B model is the heaviest method — 8 GPUs default.
# Override GPU_LIST to 4 GPUs if you want to compare throughput-per-GPU.
METHOD=vineppo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-vineppo}"
GPU_DEFAULT_LIST="${GPU_DEFAULT_LIST:-0 1 2 3 4 5 6 7}"
# VinePPO-specific vLLM tuning. With K=9 MC continuations × ~9 step
# boundaries × ~16 prompts/rank, each rank dispatches up to 1296
# concurrent generation requests during the MC value pass — far more
# than the 128 the YAML default sizes for. Bump max_num_seqs and the
# prefill budget so vLLM can schedule the MC fan-out without
# serializing into many small decode batches. Bump
# vllm_gpu_memory_utilization to 0.50 for the same reason: VinePPO
# does not load a separate value model (CASPO does), so the trainer
# memory budget has more headroom on this method than CASPO has.
EXTRA_OVERRIDES=(
    --override update_value_during_policy=false
    --override "vineppo_mc_rollouts=${VINEPPO_MC_ROLLOUTS:-9}"
    --override "vllm_max_num_seqs=${VINEPPO_VLLM_MAX_NUM_SEQS:-512}"
    --override "vllm_max_num_batched_tokens=${VINEPPO_VLLM_MAX_NUM_BATCHED_TOKENS:-32768}"
)
# Override gpu_memory_utilization specifically for VinePPO (no value
# model). Honors the same CASPO_VLLM_GPU_MEMORY_UTILIZATION env knob
# but has its own default = 0.50 (vs 0.30 for CASPO/GRPO/PPO).
export CASPO_VLLM_GPU_MEMORY_UTILIZATION="${CASPO_VLLM_GPU_MEMORY_UTILIZATION:-0.50}"
source "$(dirname "$0")/_launch_7b_fsdp.sh"
