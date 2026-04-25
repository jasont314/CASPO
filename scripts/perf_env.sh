# Sourceable performance/env-config snippet for CASPO launchers.
#
# Usage (from any launcher EXCEPT in-flight chain_caspo_phases.sh /
# launch_rho1b_parallel.sh — those must not be modified mid-run):
#
#   source "$(dirname "$0")/perf_env.sh"
#
# Centralizes CUDA allocator config, NCCL knobs, tokenizer/vLLM/HF noise
# reduction, and CPU thread caps so behaviour is consistent across phases
# (collect_value_data, train_value, train_caspo, eval). Safe to source
# multiple times — every line is an idempotent `export`.

# CUDA caching allocator: expandable_segments dramatically reduces
# fragmentation in long PPO runs that grow/shrink activations across
# rollout / forward / backward phases.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL: surface hangs as actionable errors instead of silent stalls, and
# extend the collective timeout (default 30s is far too short for slow
# weight-sync ops between trainer and vLLM worker).
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

# Only matters for HPC tensor-parallel setups; harmless at TP=1.
export CUDA_DEVICE_MAX_CONNECTIONS=1

# HF tokenizers: silence the fork-after-parallelism warning we hit on every
# DataLoader spawn. We pay the (negligible) single-thread tokenization cost.
export TOKENIZERS_PARALLELISM=false

# Unbuffered stdout/stderr so `tail -f` shows live progress.
export PYTHONUNBUFFERED=1

# vLLM: skip the usage telemetry ping and quiet the per-request INFO logs.
export VLLM_NO_USAGE_STATS=1
export VLLM_LOGGING_LEVEL=WARNING

# Suppress HF "you may want to upgrade" / config advisory noise.
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Offline HF mode — only safe when the cache at $HF_HOME is fully
# populated. Default OFF; opt in by exporting CASPO_HF_OFFLINE=1 before
# sourcing this file.
if [[ "${CASPO_HF_OFFLINE:-0}" == "1" ]]; then
    export HF_HUB_OFFLINE=1
fi

# CPU thread caps: prevent BLAS oversubscription when several GPU jobs
# share a host (4 trainer ranks × default OMP threads = host stall).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
