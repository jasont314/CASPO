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
#
# Earlier we layered on ``garbage_collection_threshold:0.6`` and
# ``max_split_size_mb:512`` to chase a ~100 MB OOM gap at mb=4
# colocated. Both proactively re-fragment / re-allocate during step
# compute and slow the policy forward by 3-4× on memory-tight regimes
# (v15 GRPO mb=4 colocated: t_pol=100 s vs mb=2 baseline ~32 s). Net
# loss vs simply running mb=2. Reverted to the minimal config.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL: surface hangs as actionable errors instead of silent stalls, and
# extend the collective timeout (default 30s is far too short for slow
# weight-sync ops between trainer and vLLM worker).
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

# NOTE: CUDA_DEVICE_MAX_CONNECTIONS=1 was previously set here. Removed
# because it serializes CUDA work-issue across streams and demonstrably
# hurts FSDP's compute/comm overlap on H100 (PyTorch issue #110155 and
# follow-ups). The "harmless at TP=1" justification was wrong for FSDP —
# FSDP relies on multiple streams to overlap reduce-scatter / all-gather
# with backward compute. Leave the variable unset so PyTorch picks the
# multi-connection default (typically 8).
#
# NCCL knobs targeting H100 / NVLink-SHARP topology. NVLink SHARP (NVLS)
# offloads reduce-scatter / all-reduce reductions to the NVSwitch ASIC,
# cutting per-collective wire bytes roughly in half on hopper-class boxes;
# safe to enable when the switch supports it (NCCL silently falls back if
# it does not). NCHANNELS / NTHREADS / BUFFSIZE bumps give NCCL more
# parallelism and bigger pipelined chunks, which empirically helps the
# many-small-collective profile FSDP produces. None of these are
# correctness-affecting; remove if you observe regressions on a non-H100
# host.
export NCCL_NVLS_ENABLE=1
export NCCL_MIN_NCHANNELS=8
export NCCL_NTHREADS=512
export NCCL_BUFFSIZE=8388608  # 8 MiB transport buffer

# HF tokenizers: silence the fork-after-parallelism warning we hit on every
# DataLoader spawn. We pay the (negligible) single-thread tokenization cost.
export TOKENIZERS_PARALLELISM=false

# Unbuffered stdout/stderr so `tail -f` shows live progress.
export PYTHONUNBUFFERED=1

# vLLM: skip the usage telemetry ping and quiet the per-request INFO logs.
export VLLM_NO_USAGE_STATS=1
export VLLM_LOGGING_LEVEL=WARNING

# vLLM EngineCore subprocess start method. Default is "fork", which makes the
# child inherit torch.distributed state (TCPStore connections, NCCL groups)
# from the parent. When the parent is FSDP/DDP-wrapped, the child's
# init_distributed_environment for its rank-local TP=1 process group
# collides with the inherited state and hangs on TCPStore client validation
# (observed: 600 s timeout on 127.0.0.1:<port>). "spawn" gives the
# EngineCore a clean Python+libtorch state. Required for FSDP+vLLM
# colocated on the same rank.
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# vLLM IPC handle serialization through the EngineCore control channel.
# Required for our IPC weight-sync path; without it vLLM rejects the
# pickle-based collective_rpc with a security error.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

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
