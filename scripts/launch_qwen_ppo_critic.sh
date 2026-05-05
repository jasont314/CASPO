#!/usr/bin/env bash
# PPO+critic baseline for Qwen2.5-Math-1.5B on dsr_sub (One-Shot-RLVR
# replication subset). Matches VinePPO upstream's PPO baseline config:
#   - num_epochs_per_iteration = 2
#   - ppo_gae_lambda = 1.0 (terminal verifier reward only)
#   - value_loss_coef = 1.0
#   - cliprange_value = 0.2
#   - clip_eps = 0.2
#   - critic_lr = policy_lr = 1e-6
#
# This is the canonical PPO+critic baseline against which CASPO Δp /
# Δlogp / VinePPO are compared in the paper.
#
# ---- Configurable env vars (override before running) ----
#
#   CONDA_ENV=scalable          # conda env with torch, transformers, vllm, flash_attn
#   PYBIN=...python             # auto-detected from conda; override if needed
#   GPU_LIST="0 1 2 3"          # 4 GPUs; FSDP requires exactly 4 ranks here
#   DSR_SUB=/path/to/dsr_sub.jsonl  # the 1209-prompt One-Shot-RLVR JSONL
#   OUT_DIR=/path/to/outputs
#   LOG_DIR=/tmp/ppo_critic_$(date +%H%M)
#   MAX_STEPS=600               # 600 RL outer iterations (~25h on 4×H100 80GB)
#   SAVE_EVERY=50               # ckpt every 50 steps
#   KL_COEF=0.01                # 1B-stable; VinePPO upstream uses 1e-4 at 7B
#                               # (1e-4 caused divergence at 1B in our runs)
#   RUN_EVAL=true               # auto-eval all saved ckpts after training
#                               # on math500/gsm8k/olympiadbench. Set false
#                               # to skip eval and just train.
#
# ---- Prerequisites ----
#
#   1. conda env "scalable" with: torch>=2.4, transformers>=4.45,
#      vllm>=0.7, flash_attn>=2.5, peft>=0.10
#   2. Source code: clone https://github.com/<your-org>/CASPO.git and
#      run from repo root
#   3. Dataset: dsr_sub.jsonl from One-Shot-RLVR replication
#      (1209 DeepScaleR prompts; SHA256 verifiable from paper repo)
#   4. Disk: ~80 GB free for ckpts (10 saves × ~7 GB each at
#      save_optimizer_state=false; ~190 GB if save_optim=true)
#   5. Hardware: 4× H100 80GB (FSDP=4, colocated vLLM at
#      gpu_memory_utilization=0.35)
#
# ---- ETA ----
#   ~25h on 4×H100 80GB (~150s/step × 600 steps)
#
set -o pipefail

# ---- env activation ----
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

# ---- locate repo root ----
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Source perf env if present (NCCL/CUDA tuning)
[[ -f ./scripts/perf_env.sh ]] && source ./scripts/perf_env.sh

# ---- configurable parameters ----
GPU_LIST="${GPU_LIST:-0 1 2 3}"
read -r -a GPUS <<< "$GPU_LIST"
N_GPUS=${#GPUS[@]}
[[ "$N_GPUS" -eq 4 ]] || { echo "ERROR: need 4 GPUs (FSDP=4); got $N_GPUS: $GPU_LIST"; exit 1; }

DSR_SUB="${DSR_SUB:-/tmp/rlvr_replication/dsr_sub.jsonl}"
[[ -f "$DSR_SUB" ]] || { echo "ERROR: dataset not found: $DSR_SUB"; echo "  Set DSR_SUB env var to dsr_sub.jsonl path"; exit 1; }

OUT_DIR="${OUT_DIR:-./ppo_critic_qwen25math15b_dsr_sub}"
LOG_DIR="${LOG_DIR:-/tmp/ppo_critic_$(date +%Y%m%d_%H%M)}"
MAX_STEPS="${MAX_STEPS:-600}"
SAVE_EVERY="${SAVE_EVERY:-50}"
KL_COEF="${KL_COEF:-0.01}"
SAVE_OPTIMIZER_STATE="${SAVE_OPTIMIZER_STATE:-false}"

mkdir -p "$OUT_DIR/logs" "$LOG_DIR"

echo "[ppo-critic] $(date +%H:%M:%S) === START ==="
echo "[ppo-critic] model:       Qwen/Qwen2.5-Math-1.5B"
echo "[ppo-critic] dataset:     $DSR_SUB"
echo "[ppo-critic] GPUs:        ${GPUS[*]}"
echo "[ppo-critic] output:      $OUT_DIR"
echo "[ppo-critic] logs:        $LOG_DIR"
echo "[ppo-critic] max_steps:   $MAX_STEPS"
echo "[ppo-critic] save_every:  $SAVE_EVERY"
echo "[ppo-critic] kl_coef:     $KL_COEF"
echo "[ppo-critic] config: matches VinePPO PPO baseline (epochs=2, lambda=1.0)"
echo "[ppo-critic] ETA: ~150s/step × $MAX_STEPS = $((MAX_STEPS * 150 / 3600))h"

PORT=$((30000 + RANDOM % 5000))
PIDS=()
for r in 0 1 2 3; do
  gpu="${GPUS[$r]}"
  log="$LOG_DIR/ppo_critic_rank${r}.log"
  echo "[ppo-critic] launching rank $r on GPU $gpu (log: $log)"
  CUDA_VISIBLE_DEVICES="$gpu" \
  WORLD_SIZE=4 RANK="$r" LOCAL_RANK=0 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT="$PORT" \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
    nohup "$PYBIN" -u -m scripts.train_caspo \
    --config configs/caspo_rho1b_math.yaml \
    --override "method=ppo_critic" \
    --override "model_name_or_path=Qwen/Qwen2.5-Math-1.5B" \
    --override "tokenizer_name_or_path=Qwen/Qwen2.5-Math-1.5B" \
    --override "trust_remote_code=true" \
    --override "torch_dtype=bfloat16" \
    --override "attn_implementation=flash_attention_2" \
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
    --override "micro_batch_size=8" \
    --override "grad_accum_steps=4" \
    --override "use_gradient_checkpointing=true" \
    --override "lr=1.0e-6" \
    --override "kl_coef=$KL_COEF" \
    --override "kl_estimator=k3" \
    --override "value_loss_coef=1.0" \
    --override "cliprange_value=0.2" \
    --override "ppo_gae_lambda=1.0" \
    --override "critic_lr=1.0e-6" \
    --override "critic_weight_decay=0.0" \
    --override "critic_grad_clip=1.0" \
    --override "clip_eps_low=0.2" \
    --override "clip_eps_high=0.2" \
    --override "max_steps=$MAX_STEPS" \
    --override "save_every=$SAVE_EVERY" \
    --override "save_optimizer_state=$SAVE_OPTIMIZER_STATE" \
    --override "eval_every=999999" \
    --override "epochs_per_rollout=2" \
    --override "rollout_backend=vllm" \
    --override "vllm_weight_sync_backend=ipc" \
    --override "vllm_gpu_memory_utilization=0.35" \
    --override "vllm_enforce_eager=false" \
    --override "vllm_multi_sample_mode=auto" \
    --override "vllm_max_num_seqs=128" \
    --override "reward_workers=4" \
    --override "compile=false" \
    --override "wandb_mode=disabled" \
    --override "output_dir=$OUT_DIR" \
    --override "wandb_run_name=ppo_critic_qwen25math15b_dsr" \
    --override "distributed_backend=fsdp" \
    > "$log" 2>&1 &
  PIDS+=("$!")
done
echo "[ppo-critic] PIDs: ${PIDS[*]}"
echo "${PIDS[*]}" > "$LOG_DIR/pids.txt"

fail=0
for pid in "${PIDS[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "[ppo-critic] $(date +%H:%M:%S) TRAINING DONE ($fail failures) — ckpts at $OUT_DIR/"

# ---- Post-training greedy eval (math500/gsm8k/olympiadbench) ----
#
# Default: ON. Set RUN_EVAL=false to skip.
# Evals each saved step_* ckpt + final/ on all 3 math benchmarks at
# greedy decoding (k=1, T=0). Output: $OUT_DIR/eval/${ckpt}.json.
# Uses up to 4 GPUs in parallel for speed.
RUN_EVAL="${RUN_EVAL:-true}"
if [[ "$RUN_EVAL" == "true" && "$fail" -eq 0 ]]; then
  EVAL_DIR="$OUT_DIR/eval"
  mkdir -p "$EVAL_DIR"
  echo "[ppo-critic] $(date +%H:%M:%S) === POST-TRAIN EVAL ==="
  echo "[ppo-critic] eval out: $EVAL_DIR"

  CKPTS=()
  for d in "$OUT_DIR"/step_* "$OUT_DIR/final"; do
    [[ -d "$d" ]] && CKPTS+=("$(basename "$d")")
  done
  echo "[ppo-critic] eval ckpts: ${CKPTS[*]}"

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
    && echo "[$ckpt] $(date +%H:%M:%S) done" \
    || echo "[$ckpt] FAILED — see $elog"
  }

  i=0
  while (( i < ${#CKPTS[@]} )); do
    EPIDS=()
    for ((j=0; j<N_GPUS && i+j<${#CKPTS[@]}; j++)); do
      gpu="${GPUS[$j]}"
      ( eval_one "$gpu" "${CKPTS[$((i+j))]}" ) &
      EPIDS+=("$!")
    done
    for ep in "${EPIDS[@]}"; do wait "$ep"; done
    i=$((i + ${#EPIDS[@]}))
  done

  echo ""
  echo "=== EVAL SUMMARY (greedy pass@1) ==="
  printf "%-12s %-10s %-10s %-13s\n" "ckpt" "math500" "gsm8k" "olympiad"
  for ckpt in "${CKPTS[@]}"; do
    f="$EVAL_DIR/${ckpt}.json"
    [[ -f "$f" ]] || continue
    m=$("$PYBIN" -c "import json; d=json.load(open('$f'))['results']; print(f\"{d['math500']['pass@k']:.3f}\")" 2>/dev/null || echo "?")
    g=$("$PYBIN" -c "import json; d=json.load(open('$f'))['results']; print(f\"{d.get('gsm8k',{}).get('pass@k',0):.3f}\")" 2>/dev/null || echo "?")
    o=$("$PYBIN" -c "import json; d=json.load(open('$f'))['results']; print(f\"{d.get('olympiadbench',{}).get('pass@k',0):.3f}\")" 2>/dev/null || echo "?")
    printf "%-12s %-10s %-10s %-13s\n" "$ckpt" "$m" "$g" "$o"
  done
  echo ""
  echo "[ppo-critic] $(date +%H:%M:%S) === ALL DONE ==="
else
  if [[ "$fail" -gt 0 ]]; then
    echo "[ppo-critic] skipping eval due to training failures"
  else
    echo "[ppo-critic] eval skipped (RUN_EVAL=false)"
  fi
fi
