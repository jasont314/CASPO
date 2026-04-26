#!/usr/bin/env bash
set -euo pipefail

# Optimized full-model RL path:
#   - torchrun multi-process trainer
#   - PyTorch FSDP full-shard policy/ref/value models
#   - one rank-local vLLM rollout engine per GPU
#
# Usage:
#   CONFIG=configs/caspo_qwen25_math_7b.yaml \
#   PREFIX_VALUE_PATH=out/value/final \
#   NUM_GPUS=8 \
#   scripts/launch_fsdp_vllm.sh
# Restrict physical GPUs with GPU_LIST, e.g. GPU_LIST="4 5 6 7" NUM_GPUS=4.
#
# Extra train_caspo.py args can be appended after the script name.

cd "$(dirname "$0")/.."
source scripts/perf_env.sh

TORCHRUN_BIN="${TORCHRUN_BIN:-/opt/conda/envs/scalable/bin/torchrun}"
CONFIG="${CONFIG:-configs/caspo_qwen25_math_7b.yaml}"
NUM_GPUS="${NUM_GPUS:-}"
if [[ -n "${GPU_LIST:-}" ]]; then
    read -r -a _GPU_ARRAY <<< "${GPU_LIST}"
    export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${_GPU_ARRAY[*]}")"
    if [[ -z "${NUM_GPUS}" ]]; then
        NUM_GPUS="${#_GPU_ARRAY[@]}"
    fi
    echo "[fsdp] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} NUM_GPUS=${NUM_GPUS}"
fi
if [[ -z "${NUM_GPUS}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        NUM_GPUS="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
    else
        NUM_GPUS=1
    fi
fi

OVERRIDES=(
    --override distributed_backend=fsdp
    --override rollout_backend=vllm
    --override vllm_tensor_parallel_size=1
)

if [[ -n "${PREFIX_VALUE_PATH:-}" ]]; then
    OVERRIDES+=(--override "prefix_value_path=${PREFIX_VALUE_PATH}")
fi
if [[ -n "${OUTPUT_DIR:-}" ]]; then
    OVERRIDES+=(--override "output_dir=${OUTPUT_DIR}")
fi
if [[ -n "${METHOD:-}" ]]; then
    OVERRIDES+=(--override "method=${METHOD}")
fi

exec "${TORCHRUN_BIN}" \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    scripts/train_caspo.py \
    --config "${CONFIG}" \
    "${OVERRIDES[@]}" \
    "$@"
