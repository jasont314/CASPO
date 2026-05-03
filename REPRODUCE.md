# Reproducing Current CASPO Runs

The active research target is **Qwen2.5-Math-1.5B + dsr_sub** (1209
DeepScaleR prompts from One-Shot-RLVR), 4-GPU FSDP. See the
"Qwen2.5-Math-1.5B + dsr_sub track" section below for the full reproduction
recipe.

The earlier Rho-1B-MATH track is **archived** — sections labeled
"Rho-1B-MATH (ARCHIVED)" below describe the historical paper-faithful
VinePPO replication setup. Launchers and configs remain in the tree but
are no longer the active comparison.

DeepSeekMath-7B on MATH-lighteval is retained as a paper-faithful 7B
reference site; it runs on the same trainer with model+data+template
overrides.

## Rho-1B-MATH track (ARCHIVED)

> ⚠️ Historical reproduction notes for the original VinePPO-replication
> target. No longer active; preserved for reference.

## Environment

Use the scalable environment for all current scripts:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate scalable
cd /home/jason/experiment/CASPO
```

Launchers set Hugging Face caches and outputs under:

```text
/mnt/nvme_tmp/jason_caspo
```

All production launchers source `scripts/perf_env.sh` for CUDA allocator,
NCCL, tokenizer, vLLM, and CPU-thread settings.

## Main Config

Current paper-faithful Rho-1B MATH config:

```text
configs/caspo_rho1b_math.yaml
```

Matched settings:

- Base model: `realtreetune/rho-1b-sft-MATH`
- Train data: `DigitalLearningGmbH/MATH-lighteval`
- Eval data: `HuggingFaceH4/MATH-500`
- Prompt: VinePPO MATH task template
- Rollout shape: `64 prompts x 8 responses = 512 responses` per outer step
- PPO minibatch: 64 responses
- PPO epochs per rollout: 2
- Policy LR: `1e-6`
- Warmup: 480 optimizer updates
- KL coefficient: `1e-4`
- CASPO advantage transform: `value`
- Sampling: temperature `0.6`, top-p `0.9`, max response length 1024
- Training length: 1000 outer steps

The YAML still contains the original `save_every: 40`, but the production
launcher overrides it to `250`, giving `step_250`, `step_500`, `step_750`, and
`final`.

## Value Model

CASPO reuses the trained IPVRM-style prefix value checkpoint:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math/value_final
```

The current summary for that checkpoint is:

```text
n_train_rollouts = 5960
n_val_rollouts   = 664
value_max_epochs = 3
best_val_loss    = 0.8037
val_acc_at_last  = 0.9484
```

Retrain this value model only if the base model, prompt, dataset, verifier,
segmentation, or rollout sampling changes materially.

CASPO online value learning is enabled with ADB/DLW:

```yaml
update_value_during_policy: true
online_value_lr: 1.0e-6
use_adb: true
use_dlw: true
```

## Four-Method Training Run

Run all methods in parallel on GPUs 4-7:

```bash
RUN_TAG=paper512_seed0 GPU_LIST="4 5 6 7" WANDB_MODE=offline \
  ./scripts/archive/rho1b/launch_rho1b_parallel.sh
```

Default mapping:

| Method | GPU | Notes |
|---|---:|---|
| PPO | 4 | terminal-reward PPO |
| CASPO | 5 | IPVRM prefix value model, online updates |
| GRPO | 6 | grouped terminal-reward advantages |
| VinePPO | 7 | `vineppo_mc_rollouts=9` |

With `RUN_TAG=paper512_seed0`, outputs land at:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_ppo_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_grpo_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_vineppo_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_paper512_seed0/logs
```

Useful overrides:

```bash
MAX_STEPS=20 RUN_TAG=smoke ./scripts/archive/rho1b/launch_rho1b_parallel.sh
SAVE_EVERY=100 RUN_TAG=short ./scripts/archive/rho1b/launch_rho1b_parallel.sh
WANDB_MODE=offline RUN_TAG=paper512_seed1 ./scripts/archive/rho1b/launch_rho1b_parallel.sh
```

## CASPO Advantage Ablations

The default CASPO run computes step TD on the direct IPVRM value:

```text
A_t = r_t + gamma * V_{t+1} - V_t
```

Two additional ablations keep the same data, value checkpoint, online value
updates, normalization, clipping, PPO loop, and vLLM infrastructure, but
transform `V` before the TD difference:

- `prob`: `sigmoid(V)`
- `logprob`: `log sigmoid(V)`

Launch only the two extra experiments:

```bash
RUN_TAG=paper512_seed0 GPU_LIST="4 5" WANDB_MODE=offline \
  ./scripts/archive/rho1b/launch_rho1b_caspo_ablations.sh
```

Outputs:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_prob_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_logprob_paper512_seed0
```

To run all three CASPO variants in one sweep:

```bash
ADV_VARIANTS="value prob logprob" GPU_LIST="4 5 6" \
  ./scripts/archive/rho1b/launch_rho1b_caspo_ablations.sh
```

Frozen-RM CASPO disables online value updates while keeping prefix-value
scoring:

```bash
RUN_TAG=paper512_seed0 GPU_LIST="4" WANDB_MODE=offline \
  ./scripts/archive/rho1b/launch_rho1b_caspo_frozen_rm.sh
```

Output:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_frozen_rm_paper512_seed0
```

## Two-GPU VinePPO DDP

Rho-1B VinePPO has a dedicated two-GPU DDP launcher. It uses one replicated
trainer and one rank-local vLLM engine per GPU, with IPC weight sync on each
rank. The launcher starts one Python process per physical GPU so each rank and
its local vLLM engine see exactly one CUDA device. By default it preserves the
512-response global outer step:

```bash
RUN_TAG=paper512_seed0 GPU_LIST="6 7" WANDB_MODE=offline \
  ./scripts/archive/rho1b/launch_rho1b_vineppo_ddp2.sh
```

Default shape:

```text
2 ranks x 32 prompts/rank x 8 rollouts = 512 responses/global step
2 ranks x 32 grad-accum micros x 1 response = 64-response global PPO minibatch
```

Output:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_vineppo_ddp2_paper512_seed0
```

Smoke:

```bash
MAX_STEPS=1 SAVE_EVERY=0 PROMPTS_PER_STEP=1 GROUP_SIZE=1 \
GRAD_ACCUM_STEPS=1 VINEPPO_MC_ROLLOUTS=1 RUN_TAG=ddp2_smoke \
GPU_LIST="6 7" WANDB_MODE=disabled ./scripts/archive/rho1b/launch_rho1b_vineppo_ddp2.sh
```

## Evaluation

Use cheap sample evals at intermediate checkpoints:

```bash
RUN_TAG=paper512_seed0 CKPT_SUBDIR=step_250 \
EVAL_BENCHMARKS=math500 EVAL_LIMIT=100 EVAL_K=8 \
EVAL_GPU_LIST="4 5 6 7" ./scripts/launch_eval_all.sh
```

Run full eval at final:

```bash
RUN_TAG=paper512_seed0 EVAL_GPU_LIST="4 5 6 7" ./scripts/launch_eval_all.sh
```

Eval defaults:

- Methods: CASPO, GRPO, VinePPO, PPO
- Override with `METHODS="caspo_prob caspo_logprob caspo_frozen_rm vineppo_ddp2"` for ablation/DDP evals
- Benchmarks: `math500,math,collegemath,olympiadbench`
- `EVAL_K=16`
- temperature `0.35`
- top-p `0.9`
- max new tokens `1024`

## Current Speed Probe

Latest paper-faithful probe: Rho-1B MATH, one H100 80GB per method, 512
responses per outer step, vLLM IPC sync, `save_every=0`, `max_steps=3`.

| Method | Mean step time | Rollout | Value/MC phase | Policy phase |
|---|---:|---:|---:|---:|
| PPO | ~90s | ~4s | 0s | ~74s |
| GRPO | ~92s | ~4s | 0s | ~76s |
| CASPO | ~141s | ~4s | ~55s | ~69s |
| VinePPO K=9 | ~237s | ~4s | ~152s | ~69s |

Approximate 1000-step ETAs:

- PPO: ~25 hours
- GRPO: ~26 hours
- CASPO: ~39 hours
- VinePPO K=9: ~66 hours

If the four methods run in parallel on GPUs 4-7, wall-clock is gated by
VinePPO: roughly 66-72 hours plus checkpoint and eval overhead.

## Validation

Config sanity:

```bash
/opt/conda/envs/scalable/bin/python scripts/validate_configs.py --diff
```

Targeted trainer/vLLM tests:

```bash
/opt/conda/envs/scalable/bin/python -m pytest -q \
  tests/test_vllm_engine.py \
  tests/test_trainer_integration.py \
  tests/test_method_dispatch.py
```

---

## Qwen2.5-Math-1.5B + dsr_sub track (current paper setup)

### Setup summary

| Field | Value |
|---|---|
| Base policy | `Qwen/Qwen2.5-Math-1.5B` |
| Dataset | `/tmp/rlvr_replication/dsr_sub.jsonl` (1209 DeepScaleR prompts; One-Shot-RLVR subset) |
| Eval | `math500`, `gsm8k`, `olympiadbench` (greedy, k=1, T=0) |
| Prompt template | `{query}\nLet's think step by step and output the final answer within \boxed{}.` |
| Max prompt | 1024 tokens |
| **Max response** | **2048 tokens** |
| **Max sequence** | **3072 tokens** (max_prompt + max_response; well within Qwen `max_position_embeddings=4096`) |
| Group size | 8 |
| Prompts per step | 128 (= 1024 responses per outer step at G=8) |
| PPO minibatch | mb=4, grad_accum=8 (= 32 effective per rank, 128 global) |
| Topology | FSDP=4 + colocated vLLM, 4× H100/A100 80GB |
| Policy LR | 1e-6 |
| KL coef | 0.001 (CASPO/GRPO/VinePPO), 0.01 (PPO+critic — needs stronger anchor at 1B) |
| Steps | 500 |

### Why response budget = 2048

Empirical measurement (n=800 uncapped Qwen2.5-Math-1.5B base rollouts on
dsr_sub at max=3000, T=1.0, May 2026):

| pct | tokens (all chains) | tokens (correct chains only) |
|---|---|---|
| p50 | 684 | 558 |
| p75 | 1142 | 778 |
| p90 | 1906 | 1076 |
| p95 | 2902 | 1377 |
| **p98** | (right-censored) | **1613** |
| p99 | (right-censored) | 2187 |

At cap=2048: 8% truncation overall, but only **2% of correct chains** get
truncated. The 6pp gap (8% truncated total vs 2% correct truncated) is
entirely failed/rambling chains — exactly what the seq-len penalty signal is
designed to discourage. Going to cap=3000 catches one additional correct
chain in 150 (~0.7% gain) at the cost of risking the Qwen RoPE position-
embedding ceiling at prompt+response=4096.

### Method launchers (4-GPU FSDP)

| Method | Launcher | `epochs_per_rollout` | `kl_coef` |
|---|---|---|---|
| GRPO | `scripts/launch_qwen_grpo.sh` | 1 | 0.001 |
| PPO+critic | `scripts/launch_qwen_ppo_critic.sh` | 2 | 0.01 |
| VinePPO | `scripts/launch_qwen_vineppo.sh` | 2 | 0.01 |
| CASPO Δp | `scripts/launch_qwen_caspo.sh` (default `ADV_TRANSFORM=prob`) | 2 | 0.001 |
| CASPO Δlogp | `scripts/launch_qwen_caspo.sh` with `ADV_TRANSFORM=logprob` | 2 | 0.001 |
| CASPO + alternating refresh | `scripts/launch_caspo_alternating.sh` | 2 | 0.001 |
| CASPO refresh resume (Phase 2 only) | `scripts/launch_caspo_refresh_resume.sh` | 2 | 0.001 |

All Qwen launchers source the base config `configs/caspo_rho1b_math.yaml`
and override the relevant fields (template, response length, model). The
Rho-1B YAML is the trainer-config schema; no separate Qwen YAML required.

### PRM training recipe (gap-closed, unified at 2048)

Initial PRM and refresh PRM use the same recipe — no decoupling between
collection and training prefix budgets. Use `scripts/launch_qwen_mc_prm.sh`
for both:

```bash
# Initial PRM (rollouts from base SFT, V_φ from base SFT)
OUT_DIR=/mnt/nvme_tmp4/jason_caspo/qwen_mc_prm_initial \
DSR_SUB=/tmp/rlvr_replication/dsr_sub.jsonl \
GPU_LIST="0 1 2 3" \
  bash scripts/launch_qwen_mc_prm.sh

# Refresh PRM (rollouts from current RL ckpt, V_φ trained from-scratch)
OUT_DIR=/mnt/nvme_tmp7/jason_caspo/qwen_mc_prm_refresh_step150 \
POLICY=/path/to/caspo/step_150 \
NUM_PROMPTS=300 \
DSR_SUB=/tmp/rlvr_replication/dsr_sub.jsonl \
GPU_LIST="0 1 2 3" \
  bash scripts/launch_qwen_mc_prm.sh
```

The launcher consolidates both phases (mc_step_label.py 4-shard collection
+ merge + train_value_mc.py FSDP=4 training) with unified-2048 defaults:
K=16, J=16, steps=5, max_response_len=2048, max_train_prefix_len=0,
lr=5e-6, mb=4, grad_accum=2, epochs=2, beta=10.0. All knobs are env-var
overridable.

ETA: ~91 min total for the initial PRM (~41 min collection + ~50 min
training on 4 GPUs). Refresh PRM at N=300 is ~30 min collection + ~50 min
training.

`--max_train_prefix_len 0` (default) means "use whatever was collected" —
NO prefix decoupling. This matches RL deployment cap so train/deploy
distributions are aligned.

(The earlier "Option C" decoupling — collect long, train short — was a
hedge against an interpretation of the iter_max1792 sweep finding that
turned out to be a probe-cap mismatch artifact. The v3 refresh PRM at
cap=1536 trained without decoupling achieved ρ=0.630 in-distribution,
directly contradicting the "long prefix training is noisy" theory. See
`docs/RM_TRAINING.md` for the full re-interpretation.)

### Alternating refresh (PRM → RL → PRM → RL → ...)

End-to-end pipeline. Trains the initial PRM internally as Phase 0, then
alternates RL training with PRM refresh. Self-contained — no external
PRM training step required.

```bash
# Minimal invocation: Phase 0 + alternating cycles
INITIAL_CKPT=Qwen/Qwen2.5-Math-1.5B \
OUT_ROOT=/mnt/data/caspo_alternating \
DSR_SUB=/tmp/rlvr_replication/dsr_sub.jsonl \
GPU_LIST="0 1 2 3" \
  bash scripts/launch_caspo_alternating.sh

# Optional knobs
ADV_TRANSFORM=logprob        # 'prob' (Δp, default) or 'logprob' (Δlogp)
REFRESH_EVERY=200            # 150 for Δp (default), 200 for Δlogp
TOTAL_STEPS=500
METHOD=caspo                 # default
PRM_TRAIN_K=16 PRM_TRAIN_J=16 PRM_TRAIN_S=5
PRM_TRAIN_MAX_RESP=2048      # matches RL cap
PRM_TRAIN_PREFIX_CAP=0       # no decoupling

# Skip Phase 0 by providing a pre-trained PRM
INITIAL_PRM=/path/to/prm/best \
INITIAL_CKPT=Qwen/Qwen2.5-Math-1.5B \
OUT_ROOT=... \
DSR_SUB=... \
  bash scripts/launch_caspo_alternating.sh
```

Or, for Phase-2-only resume (RL ckpt + new PRM, preserving optimizer +
lr_scheduler + ref_policy from Phase 1):

```bash
POLICY_CKPT=/path/to/phase1/step_150 \
NEW_PRM=/path/to/refreshed_PRM \
OUT_DIR=/path/to/phase2 \
  bash scripts/launch_caspo_refresh_resume.sh
```

### Evaluation (Qwen)

Use the post-train auto-eval blocks built into the per-method launchers
(set `RUN_EVAL=true`, default). For ad-hoc periodic eval during training,
use `scripts/eval_periodic.sh` with the matching template + length:

```bash
./scripts/eval_periodic.sh --gpu 7 --max-new-tokens 2048 \
    --prompt-template '{query}\nLet'\''s think step by step and output the final answer within \boxed{}.' \
    /mnt/nvme_tmp4/jason_caspo/your_qwen_run_dir
```

Or `scripts/launch_eval_all.sh` with overrides:

```bash
EVAL_MAX_NEW_TOKENS=2048 \
EVAL_PROMPT_TEMPLATE='{query}\nLet'\''s think step by step and output the final answer within \boxed{}.' \
  ./scripts/launch_eval_all.sh
```

Without these overrides, the launcher silently falls through to the Rho-1B
`[MATH_TASK]` template + 1024-token cap, which empirically hides 30+pp of
Qwen gains.

Paper build:

```bash
make -C paper
```

## 7B Note

The 7B configs remain available, but they are not the current single-node
four-method production target. Full-model 7B training should use FSDP, and the
current 7B vLLM sync path is still checkpoint-based until an exact-runtime
NCCL/in-memory sync path is implemented.
