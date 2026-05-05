#!/usr/bin/env bash
# Two-phase CASPO with PRM refresh: PHASE 2 RESUME launcher.
#
# Resumes CASPO RL training from a Phase-1 checkpoint with a NEWLY-trained
# (refreshed) PRM. EVERYTHING ELSE stays IDENTICAL to Phase 1:
#   - policy weights         loaded from POLICY_CKPT
#   - optimizer state        loaded from POLICY_CKPT/optimizer.pt
#   - lr_scheduler state     loaded from POLICY_CKPT (continues lr decay)
#   - global_step counter    loaded from POLICY_CKPT
#   - ref policy             still the ORIGINAL BASE (NOT the resumed ckpt)
#                            -- critical: ref drift breaks KL anchor (see project_caspo_refresh_validates)
#   - dataset, prompt template, lr, mb, kl_coef, group_size, prompts_per_step,
#     epochs_per_rollout, rollout_temperature, top_p — all match Phase 1
#
# The ONLY thing that changes vs Phase 1 is:
#   - prefix_value_path  →  NEW_PRM (the freshly-trained refresh PRM)
#
# ---- Required env vars ----
#   POLICY_CKPT         : Phase-1 ckpt directory (e.g., step_150)
#                         Must contain: model.safetensors, optimizer.pt, lr_scheduler.pt
#   NEW_PRM             : path to refreshed PRM (e.g., output of mc_step_label.py + train_value_mc.py)
#   OUT_DIR             : output directory for Phase-2 ckpts
#
# ---- Optional env vars (defaults match Phase-1 defaults) ----
#   GPU_LIST            : default "0 1 2 3" (FSDP=4)
#   DSR_SUB             : default /tmp/rlvr_replication/dsr_sub.jsonl
#   REF_MODEL           : default Qwen/Qwen2.5-Math-1.5B (base SFT)
#   MAX_STEPS           : default 600 (so lr_scheduler picks up correctly)
#   SAVE_EVERY          : default 50
#   METHOD              : default "caspo" (use "caspo_logprob" for Δlogp, etc.)
#   ADV_TRANSFORM       : default "prob" (Δp); use "logprob" for Δlogp
#   LR                  : default 1e-6
#   KL_COEF             : default 0.001 (1B-stable for CASPO)
#   GROUP_SIZE          : default 8
#   PROMPTS_PER_STEP    : default 128
#   MB                  : default 4
#   ACCUM               : default 8
#   EPOCHS_PER_ROLLOUT  : default 2 (matches PPO+critic/VinePPO upstream)
#   KL_ESTIMATOR        : default "k3"

set -o pipefail
source /opt/conda/etc/profile.d/conda.sh
CONDA_ENV="${CONDA_ENV:-scalable}"
conda activate "$CONDA_ENV"
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -f ./scripts/perf_env.sh ]] && source ./scripts/perf_env.sh
PYBIN="${PYBIN:-$(which python)}"

# ---- required ----
: "${POLICY_CKPT:?POLICY_CKPT must be set (Phase-1 ckpt dir with model + optimizer.pt)}"
: "${NEW_PRM:?NEW_PRM must be set (path to refreshed PRM)}"
: "${OUT_DIR:?OUT_DIR must be set}"

[[ -d "$POLICY_CKPT" ]] || { echo "ERROR: POLICY_CKPT not found: $POLICY_CKPT"; exit 1; }
[[ -f "$POLICY_CKPT/optimizer.pt" ]] || { echo "ERROR: optimizer.pt not found in $POLICY_CKPT"; echo "  Phase 1 must have used save_optimizer_state=true"; exit 1; }
[[ -d "$NEW_PRM" ]] || { echo "ERROR: NEW_PRM not found: $NEW_PRM"; exit 1; }

# ---- optional (defaults match Phase 1) ----
GPU_LIST="${GPU_LIST:-0 1 2 3}"
read -r -a GPUS <<< "$GPU_LIST"
N_GPUS=${#GPUS[@]}
[[ "$N_GPUS" -eq 4 ]] || { echo "ERROR: need 4 GPUs (FSDP=4); got $N_GPUS"; exit 1; }

DSR_SUB="${DSR_SUB:-/tmp/rlvr_replication/dsr_sub.jsonl}"
REF_MODEL="${REF_MODEL:-Qwen/Qwen2.5-Math-1.5B}"
MAX_STEPS="${MAX_STEPS:-600}"
SAVE_EVERY="${SAVE_EVERY:-50}"
METHOD="${METHOD:-caspo}"
ADV_TRANSFORM="${ADV_TRANSFORM:-prob}"
LR="${LR:-1.0e-6}"
KL_COEF="${KL_COEF:-0.001}"
GROUP_SIZE="${GROUP_SIZE:-8}"
PROMPTS_PER_STEP="${PROMPTS_PER_STEP:-128}"
MB="${MB:-8}"
ACCUM="${ACCUM:-4}"
EPOCHS_PER_ROLLOUT="${EPOCHS_PER_ROLLOUT:-2}"
KL_ESTIMATOR="${KL_ESTIMATOR:-k3}"

LOG_DIR="${LOG_DIR:-/tmp/caspo_resume_$(date +%Y%m%d_%H%M)}"
mkdir -p "$OUT_DIR/logs" "$LOG_DIR"

echo "[refresh-resume] $(date +%H:%M:%S) === Phase 2 START ==="
echo "[refresh-resume] policy ckpt:    $POLICY_CKPT"
echo "[refresh-resume] new PRM:        $NEW_PRM"
echo "[refresh-resume] ref model:      $REF_MODEL  (base SFT — NOT the resume ckpt)"
echo "[refresh-resume] output:         $OUT_DIR"
echo "[refresh-resume] method:         $METHOD (adv_transform=$ADV_TRANSFORM)"
echo "[refresh-resume] max_steps:      $MAX_STEPS  (preserves lr_scheduler horizon)"
echo "[refresh-resume] save_every:     $SAVE_EVERY  (with optimizer state for chained refresh)"
echo "[refresh-resume] hparams: lr=$LR kl=$KL_COEF groups=$GROUP_SIZE pps=$PROMPTS_PER_STEP mb=$MB×$ACCUM ep=$EPOCHS_PER_ROLLOUT"

PORT=$((34000 + RANDOM % 100))
PIDS=()
for r in 0 1 2 3; do
  gpu="${GPUS[$r]}"
  log="$LOG_DIR/refresh_resume_rank${r}.log"
  CUDA_VISIBLE_DEVICES="$gpu" \
  WORLD_SIZE=4 RANK="$r" LOCAL_RANK=0 \
  MASTER_ADDR=127.0.0.1 MASTER_PORT="$PORT" \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
    nohup "$PYBIN" -u -m scripts.train_caspo \
    --config configs/caspo_rho1b_math.yaml \
    --override "method=$METHOD" \
    --override "caspo_advantage_transform=$ADV_TRANSFORM" \
    --override "prefix_value_path=$NEW_PRM" \
    --override "online_value_lr=0.0" \
    --override "update_value_during_policy=false" \
    --override "model_name_or_path=$POLICY_CKPT" \
    --override "tokenizer_name_or_path=$POLICY_CKPT" \
    --override "resume_from=$POLICY_CKPT" \
    --override "ref_model_path=$REF_MODEL" \
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
    --override "group_size=$GROUP_SIZE" \
    --override "prompts_per_step=$PROMPTS_PER_STEP" \
    --override "micro_batch_size=$MB" \
    --override "grad_accum_steps=$ACCUM" \
    --override "use_gradient_checkpointing=true" \
    --override "lr=$LR" \
    --override "kl_coef=$KL_COEF" \
    --override "kl_estimator=$KL_ESTIMATOR" \
    --override "clip_eps_low=0.2" \
    --override "clip_eps_high=0.2" \
    --override "max_steps=$MAX_STEPS" \
    --override "save_every=$SAVE_EVERY" \
    --override "save_optimizer_state=true" \
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
    --override "wandb_run_name=caspo_refresh_resume" \
    --override "distributed_backend=fsdp" \
    > "$log" 2>&1 &
  PIDS+=("$!")
done
echo "[refresh-resume] PIDs: ${PIDS[*]}"
echo "${PIDS[*]}" > "$LOG_DIR/pids.txt"

fail=0
for pid in "${PIDS[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "[refresh-resume] $(date +%H:%M:%S) DONE ($fail failures) — ckpts at $OUT_DIR/"
