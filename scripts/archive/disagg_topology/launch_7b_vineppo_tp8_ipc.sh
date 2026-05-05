#!/usr/bin/env bash
set -eo pipefail
# Colocated TP=8 VinePPO K=9 launcher with hybrid per-GPU IPC sync
# (Phase F). Trainer FSDP=8 + vLLM TP=8 share all 8 GPUs; weights
# go through CUDA IPC handles, one per (rank, GPU) pair, gathered
# on rank 0 and submitted as a single multi-UUID update_weights
# RPC. See docs/disaggregated_topology_plan.md (Phase F).
METHOD=vineppo
RUN_METHOD_TAG="${RUN_METHOD_TAG:-vineppo_tp8_ipc}"
GPU_DEFAULT_LIST="${GPU_DEFAULT_LIST:-0 1 2 3 4 5 6 7}"

# Force IPC sync — Phase F's whole point. The shared body's default
# is 'nccl', which fails at colocated TP because two NCCL ranks on
# one device aren't allowed.
export TP8_WEIGHT_SYNC_BACKEND="${TP8_WEIGHT_SYNC_BACKEND:-${WEIGHT_SYNC_BACKEND:-ipc}}"

# Memory headroom: trainer FSDP shard + ref shard + Adam + activations
# (~25 GB) plus vLLM TP=8 shard + KV. With util=0.40 vLLM gets ~32 GB
# per GPU, leaving ~23 GB headroom for the +14 GB summon_full_params
# transient. Tune up if rollout is decode-bound on KV.
export CASPO_VLLM_GPU_MEMORY_UTILIZATION="${CASPO_VLLM_GPU_MEMORY_UTILIZATION:-0.40}"

EXTRA_OVERRIDES=(
    --override update_value_during_policy=false
    --override "vineppo_mc_rollouts=${VINEPPO_MC_ROLLOUTS:-9}"
)
source "$(dirname "$0")/_launch_7b_tp8_colocated.sh"
