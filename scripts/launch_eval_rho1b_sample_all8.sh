#!/usr/bin/env bash
# Cheap checkpoint eval for the standard seven-run Rho-1B suite.
#
# Default is MATH-500 limited to 100 problems with k=8. Use this at saved
# checkpoints (step_250, step_500, step_750, final) to get a trend without
# paying for the full benchmark suite.
#
# Usage:
#   RUN_TAG=paper512_seed0 CKPT_SUBDIR=step_250 \
#     EVAL_GPU_LIST="0 1 2 3 4 5 6" ./scripts/launch_eval_rho1b_sample_all8.sh
#
set -eo pipefail

cd "$(dirname "$0")/.."

export METHODS="${METHODS:-grpo ppo vineppo_ddp2 caspo caspo_prob caspo_logprob caspo_frozen_rm}"
export CKPT_SUBDIR="${CKPT_SUBDIR:-step_250}"
export EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-math500}"
export EVAL_LIMIT="${EVAL_LIMIT:-100}"
export EVAL_K="${EVAL_K:-8}"
export EVAL_GPU_LIST="${EVAL_GPU_LIST:-0 1 2 3 4 5 6}"
export EVAL_VLLM_GPU_MEMORY_UTILIZATION="${EVAL_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

exec ./scripts/launch_eval_all.sh
