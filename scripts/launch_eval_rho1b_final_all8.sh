#!/usr/bin/env bash
# Full final eval for the standard seven-run Rho-1B suite.
#
# Runs math500, full MATH test, CollegeMath, and OlympiadBench with k=16.
# Methods run in parallel across 7 eval GPUs; each method reuses one vLLM
# engine across all benchmarks.
#
# Usage:
#   RUN_TAG=paper512_seed0 EVAL_GPU_LIST="0 1 2 3 4 5 6" \
#     ./scripts/launch_eval_rho1b_final_all8.sh
#
set -eo pipefail

cd "$(dirname "$0")/.."

export METHODS="${METHODS:-grpo ppo vineppo_ddp2 caspo caspo_prob caspo_logprob caspo_frozen_rm}"
export CKPT_SUBDIR="${CKPT_SUBDIR:-final}"
export EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-math500,math,collegemath,olympiadbench}"
export EVAL_K="${EVAL_K:-16}"
export EVAL_GPU_LIST="${EVAL_GPU_LIST:-0 1 2 3 4 5 6}"
export EVAL_VLLM_GPU_MEMORY_UTILIZATION="${EVAL_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"

exec ./scripts/launch_eval_all.sh
