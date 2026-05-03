#!/usr/bin/env bash
# CASPO for Qwen2.5-Math-1.5B on dsr_sub (One-Shot-RLVR replication
# subset). Step-TD policy optimization over a frozen prefix value model.
# This is the v1 / "no refresh" CASPO baseline; for iterated PRM refresh
# use scripts/launch_caspo_alternating.sh; for resume-from-ckpt with
# refreshed PRM use scripts/launch_caspo_refresh_resume.sh.
#
# ---- Configurable env vars ----
#   CONDA_ENV=scalable
#   PYBIN=...python
#   GPU_LIST="0 1 2 3"          (4 GPUs; FSDP=4)
#   DSR_SUB=/path/to/dsr_sub.jsonl
#   PRM_PATH=/path/to/prefix_value_model
#                               # default: qwen_mc_prm_15b_dsr_sub/best (initial PRM)
#   OUT_DIR=/path/to/outputs
#   LOG_DIR=/tmp/caspo_$(date +%H%M)
#   ADV_TRANSFORM=prob          # 'prob' for Δp, 'logprob' for Δlogp
#   MAX_STEPS=600
#   SAVE_EVERY=50
#   KL_COEF=0.001
#   EPOCHS_PER_ROLLOUT=2
#   RUN_EVAL=true
#
# ---- ETA ----
#   ~15-17h on 4×H100 80GB (~95s/step × 600)
#
set -o pipefail

CONDA_ENV="${CONDA_ENV:-scalable}"
if [[ -z "${PYBIN:-}" ]]; then
    if [[ -d "/opt/conda" ]]; then
        source /opt/conda/etc/profile.d/conda.sh
    elif [[ -d "$HOME/miniconda3" ]]; then
        source $HOME/miniconda3/etc/profile.d/conda.sh
    elif [[ -d "$HOME/anaconda3" ]]; then
        source $HOME/anaconda3/etc/profile.d/conda.sh
    else
        echo "ERROR: cannot find conda. Set PYBIN explicitly to override."
        exit 1
    fi
    conda activate "$CONDA_ENV"
    PYBIN="$(which python)"
fi

set -uo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"
[[ -f ./scripts/perf_env.sh ]] && source ./scripts/perf_env.sh

GPU_LIST="${GPU_LIST:-0 1 2 3}"
read -r -a GPUS <<< "$GPU_LIST"
N_GPUS=${#GPUS[@]}
[[ "$N_GPUS" -eq 4 ]] || { echo "ERROR: need 4 GPUs (FSDP=4); got $N_GPUS: $GPU_LIST"; exit 1; }

DSR_SUB="${DSR_SUB:-/path/to/dsr_sub.jsonl}"
[[ -f "$DSR_SUB" ]] || { echo "ERROR: dataset not found: $DSR_SUB"; exit 1; }

PRM_PATH="${PRM_PATH:-/mnt/nvme_tmp4/jason_caspo/qwen_mc_prm_15b_dsr_sub/best}"
[[ -d "$PRM_PATH" ]] || { echo "ERROR: PRM not found: $PRM_PATH"; exit 1; }

OUT_DIR="${OUT_DIR:-./caspo_qwen25math15b_dsr_sub}"
LOG_DIR="${LOG_DIR:-/tmp/caspo_$(date +%Y%m%d_%H%M)}"
ADV_TRANSFORM="${ADV_TRANSFORM:-prob}"
MAX_STEPS="${MAX_STEPS:-600}"
SAVE_EVERY="${SAVE_EVERY:-50}"
KL_COEF="${KL_COEF:-0.001}"
EPOCHS_PER_ROLLOUT="${EPOCHS_PER_ROLLOUT:-2}"
SAVE_OPTIMIZER_STATE="${SAVE_OPTIMIZER_STATE:-true}"

mkdir -p "$OUT_DIR/logs" "$LOG_DIR"

echo "[caspo] $(date +%H:%M:%S) === START ==="
echo "[caspo] model:           Qwen/Qwen2.5-Math-1.5B"
echo "[caspo] dataset:         $DSR_SUB"
echo "[caspo] PRM:             $PRM_PATH"
echo "[caspo] adv_transform:   $ADV_TRANSFORM ($([[ "$ADV_TRANSFORM" == "prob" ]] && echo "Δp" || echo "Δlogp"))"
echo "[caspo] GPUs:            ${GPUS[*]}"
echo "[caspo] output:          $OUT_DIR"
echo "[caspo] max_steps:       $MAX_STEPS"

PORT=$((30000 + RANDOM % 5000))
PIDS=()
for r in 0 1 2 3; do
  gpu="${GPUS[$r]}"
  log="$LOG_DIR/caspo_rank${r}.log"
  echo "[caspo] launching rank $r on GPU $gpu (log: $log)"
  CUDA_VISIBLE_DEVICES="$gpu" \
  WORLD_SIZE=4 RANK="$r" LOCAL_RANK=0 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT="$PORT" \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
    nohup "$PYBIN" -u -m scripts.train_caspo \
    --config configs/caspo_rho1b_math.yaml \
    --override "method=caspo" \
    --override "caspo_advantage_transform=$ADV_TRANSFORM" \
    --override "prefix_value_path=$PRM_PATH" \
    --override "online_value_lr=0.0" \
    --override "update_value_during_policy=false" \
    --override "model_name_or_path=Qwen/Qwen2.5-Math-1.5B" \
    --override "tokenizer_name_or_path=Qwen/Qwen2.5-Math-1.5B" \
    --override "trust_remote_code=true" \
    --override "torch_dtype=bfloat16" \
    --override "attn_implementation=sdpa" \
    --override "dataset_name=$DSR_SUB" \
    --override "dataset_split=train" \
    --override "filter_eval_leakage=false" \
    --override "prompt_template={query}\nLet's think step by step and output the final answer within \\boxed{}." \
    --override "system_prompt=null" \
    --override "max_prompt_len=1024" \
    --override "max_response_len=2048" \
    --override "max_sequence_len=3072" \
    --override "rollout_temperature=0.6" \
    --override "rollout_top_p=1.0" \
    --override "group_size=8" \
    --override "prompts_per_step=128" \
    --override "micro_batch_size=4" \
    --override "grad_accum_steps=8" \
    --override "use_gradient_checkpointing=true" \
    --override "lr=1.0e-6" \
    --override "kl_coef=$KL_COEF" \
    --override "kl_estimator=k3" \
    --override "max_steps=$MAX_STEPS" \
    --override "save_every=$SAVE_EVERY" \
    --override "save_optimizer_state=$SAVE_OPTIMIZER_STATE" \
    --override "eval_every=999999" \
    --override "epochs_per_rollout=$EPOCHS_PER_ROLLOUT" \
    --override "rollout_backend=vllm" \
    --override "vllm_weight_sync_backend=ipc" \
    --override "vllm_gpu_memory_utilization=0.45" \
    --override "vllm_kv_cache_dtype=fp8" \
    --override "vllm_enforce_eager=false" \
    --override "vllm_multi_sample_mode=auto" \
    --override "vllm_max_num_seqs=128" \
    --override "reward_workers=4" \
    --override "compile=false" \
    --override "wandb_mode=disabled" \
    --override "output_dir=$OUT_DIR" \
    --override "wandb_run_name=caspo_${ADV_TRANSFORM}_qwen25math15b_dsr" \
    --override "distributed_backend=fsdp" \
    > "$log" 2>&1 &
  PIDS+=("$!")
done
echo "[caspo] PIDs: ${PIDS[*]}"
echo "${PIDS[*]}" > "$LOG_DIR/pids.txt"

fail=0
for pid in "${PIDS[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "[caspo] $(date +%H:%M:%S) TRAINING DONE ($fail failures) — ckpts at $OUT_DIR/"

# ---- Post-training greedy eval ----
RUN_EVAL="${RUN_EVAL:-true}"
if [[ "$RUN_EVAL" == "true" && "$fail" -eq 0 ]]; then
  EVAL_DIR="$OUT_DIR/eval"
  mkdir -p "$EVAL_DIR"
  echo "[caspo] $(date +%H:%M:%S) === POST-TRAIN EVAL ==="

  CKPTS=()
  for d in "$OUT_DIR"/step_* "$OUT_DIR/final"; do
    [[ -d "$d" ]] && CKPTS+=("$(basename "$d")")
  done
  echo "[caspo] eval ckpts: ${CKPTS[*]}"

  eval_one() {
    local gpu="$1" ckpt="$2"
    local out="$EVAL_DIR/${ckpt}.json"
    local elog="$EVAL_DIR/${ckpt}.log"
    if [[ -f "$out" ]]; then echo "[$ckpt] skip — exists"; return; fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYBIN" -u scripts/eval.py \
      --config configs/caspo_rho1b_math.yaml \
      --override "model_name_or_path=$OUT_DIR/$ckpt" \
      --override "prompt_template={query}\nLet's think step by step and output the final answer within \\boxed{}." \
      --override "max_response_len=2048" \
      --benchmarks "math500,gsm8k,olympiadbench" \
      --k 1 --temperature 0.0 --top-p 1.0 \
      --max-new-tokens 3072 \
      --backend vllm --gpu-memory-utilization 0.85 \
      --output "$out" \
      > "$elog" 2>&1 \
    && echo "[$ckpt] $(date +%H:%M:%S) done" || echo "[$ckpt] FAIL"
  }

  i=0
  while (( i < ${#CKPTS[@]} )); do
    EPIDS=()
    for ((j=0; j<4 && i+j<${#CKPTS[@]}; j++)); do
      ( eval_one "${GPUS[$j]}" "${CKPTS[$((i+j))]}" ) &
      EPIDS+=("$!")
    done
    for pid in "${EPIDS[@]}"; do wait "$pid"; done
    i=$((i + ${#EPIDS[@]}))
  done
  echo "[caspo] $(date +%H:%M:%S) === EVAL DONE ==="
fi

echo "[caspo] $(date +%H:%M:%S) === ALL DONE ==="
