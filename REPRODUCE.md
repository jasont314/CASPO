# Reproducing Current CASPO Runs

This runbook tracks the current repository setup for the Rho-1B MATH
comparison. It is intentionally narrower than the older cross-repo playbook:
the active production target is one shared trainer/inference stack that can run
PPO, GRPO, CASPO, and VinePPO under the same config.

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
  ./scripts/launch_rho1b_parallel.sh
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
MAX_STEPS=20 RUN_TAG=smoke ./scripts/launch_rho1b_parallel.sh
SAVE_EVERY=100 RUN_TAG=short ./scripts/launch_rho1b_parallel.sh
WANDB_MODE=offline RUN_TAG=paper512_seed1 ./scripts/launch_rho1b_parallel.sh
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
  ./scripts/launch_rho1b_caspo_ablations.sh
```

Outputs:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_prob_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_logprob_paper512_seed0
```

To run all three CASPO variants in one sweep:

```bash
ADV_VARIANTS="value prob logprob" GPU_LIST="4 5 6" \
  ./scripts/launch_rho1b_caspo_ablations.sh
```

Frozen-RM CASPO disables online value updates while keeping prefix-value
scoring:

```bash
RUN_TAG=paper512_seed0 GPU_LIST="4" WANDB_MODE=offline \
  ./scripts/launch_rho1b_caspo_frozen_rm.sh
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
  ./scripts/launch_rho1b_vineppo_ddp2.sh
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
GPU_LIST="6 7" WANDB_MODE=disabled ./scripts/launch_rho1b_vineppo_ddp2.sh
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

Paper build:

```bash
make -C paper
```

## 7B Note

The 7B configs remain available, but they are not the current single-node
four-method production target. Full-model 7B training should use FSDP, and the
current 7B vLLM sync path is still checkpoint-based until an exact-runtime
NCCL/in-memory sync path is implemented.
