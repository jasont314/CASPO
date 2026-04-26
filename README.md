# CASPO - Credit Assignment Step Policy Optimization

CASPO is a math-RL training stack for comparing:

- `ppo`: terminal-reward PPO with a sequence-level advantage.
- `grpo`: group-relative terminal reward advantages.
- `vineppo`: VinePPO-style step TD using Monte Carlo prefix rollouts.
- `caspo`: step TD using a learned IPVRM prefix value model instead of Monte Carlo prefix rollouts.

The main current target is a paper-faithful Rho-1B MATH run that matches the
VinePPO Rho-1B MATH setup where possible, while running all four methods in the
same trainer and vLLM infrastructure.

## Current Rho-1B MATH Setup

Main config: `configs/caspo_rho1b_math.yaml`

| Field | Current value |
|---|---|
| Base policy | `realtreetune/rho-1b-sft-MATH` |
| Dataset | `DigitalLearningGmbH/MATH-lighteval`, train split |
| Eval base | `HuggingFaceH4/MATH-500`, test split |
| Prompt template | `[MATH_TASK] Problem:\n{query}\n\nSolution:` |
| Response budget | 1024 tokens |
| Rollout group | `group_size=8` |
| Prompts per step | `64` |
| Responses per PPO outer step | `64 x 8 = 512` |
| PPO minibatch | `micro_batch_size=1`, `grad_accum_steps=64` |
| PPO epochs per rollout | `2` |
| Policy LR | `1e-6` |
| Warmup | `480` optimizer updates |
| PPO clip | `0.2` |
| KL coefficient | `1e-4` with `k3` estimator |
| Rollout sampling | temperature `0.6`, top-p `0.9` |
| Steps | `1000` |
| Launch checkpoint cadence | `step_250`, `step_500`, `step_750`, `final` |

The YAML still documents VinePPO's original save cadence, but the production
launcher overrides `save_every=250` so each method writes four checkpoints total
for a 1000-step run.

## Value Model

CASPO uses an IPVRM-style prefix value model:

```text
V_phi(prefix_t) = beta * sum_{i < t} log [pi_phi(y_i | prefix_i) / pi_ref(y_i | prefix_i)]
```

The current Rho-1B MATH value checkpoint is:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math/value_final
```

It is a symlink to the trained `final/` value checkpoint. The existing training
summary shows:

```text
n_train_rollouts = 5960
n_val_rollouts   = 664
value_max_epochs = 3
best_val_loss    = 0.8037
val_acc_at_last  = 0.9484
```

Reuse this value model for the current Rho-1B MATH experiments. Retrain it only
if the base model, prompt template, dataset, verifier, segmentation, or rollout
temperature/top-p changes substantially.

For online CASPO updates, the config uses:

```yaml
update_value_during_policy: true
online_value_lr: 1.0e-6
use_adb: true
use_dlw: true
```

IPVRM reports larger online learning rates in a LoRA setting. This codebase is
currently doing full-model value updates, so `1e-6` is the safer setting after
earlier reward-model drift/collapse concerns.

## Advantage Construction

PPO and GRPO do not segment responses:

- PPO standardizes terminal-reward sequence advantages according to
  `standardize_advantage_scope`.
- GRPO uses per-prompt group-relative normalization over each prompt's `G=8`
  terminal rewards.

CASPO and VinePPO segment responses using the LaTeX-aware step splitter ported
from VinePPO. They compute step TD advantages:

```text
A_t = r_t + gamma * V_{t+1} - V_t
```

CASPO obtains `V_t` from the IPVRM prefix value model. VinePPO obtains `V_t`
from `K=9` Monte Carlo continuations per nonterminal prefix. CASPO currently
normalizes valid step advantages over the whole rollout batch:

```yaml
standardize_step_advantage: true
standardize_advantage_scope: batch
```

That means CASPO whitens step advantages across all valid steps from all 512
responses in the outer PPO step, not separately within each prompt group.

## Training/Inference Infrastructure

All launchers use the `scalable` conda environment:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate scalable
```

Launchers set Hugging Face caches under:

```text
/mnt/nvme_tmp/jason_caspo/hf_cache
```

Outputs also live under:

```text
/mnt/nvme_tmp/jason_caspo
```

`scripts/perf_env.sh` centralizes CUDA allocator, NCCL, tokenizer, vLLM, and CPU
thread settings. Notable defaults:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `TORCH_NCCL_BLOCKING_WAIT=1`
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
- `NCCL_TIMEOUT=1800`
- `TOKENIZERS_PARALLELISM=false`
- `PYTHONUNBUFFERED=1`
- `VLLM_NO_USAGE_STATS=1`
- `VLLM_LOGGING_LEVEL=WARNING`
- `OMP_NUM_THREADS=4`
- `MKL_NUM_THREADS=4`

For Rho-1B single-GPU-per-method runs, vLLM weight sync uses CUDA IPC:

```yaml
rollout_backend: vllm
vllm_weight_sync_backend: ipc
vllm_gpu_memory_utilization: 0.45
vllm_enforce_eager: false
```

The trainer also primes vLLM's AsyncLLM frontend loop before generation. This
avoids a vLLM V1 embedded-engine stall where metadata requests wait for
EngineCore output before the output handler is draining it.

## Launching the Four-Method Run

Use GPUs 4-7. Each method gets one H100:

```bash
cd /home/jason/experiment/CASPO
RUN_TAG=paper512_seed0 GPU_LIST="4 5 6 7" WANDB_MODE=offline \
  ./scripts/launch_rho1b_parallel.sh
```

Default mapping:

| Method | GPU |
|---|---|
| PPO | first GPU in `GPU_LIST` |
| CASPO | second GPU |
| GRPO | third GPU |
| VinePPO K=9 | fourth GPU |

With `RUN_TAG=paper512_seed0`, outputs are:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_ppo_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_grpo_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_vineppo_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_paper512_seed0/logs
```

You can override the default cadence and run length:

```bash
SAVE_EVERY=100 MAX_STEPS=300 RUN_TAG=debug ./scripts/launch_rho1b_parallel.sh
```

## Evaluation

Do cheap sample evals at saved checkpoints and full eval only at the end.

Sample eval example:

```bash
RUN_TAG=paper512_seed0 CKPT_SUBDIR=step_250 \
EVAL_BENCHMARKS=math500 EVAL_LIMIT=100 EVAL_K=8 \
EVAL_GPU_LIST="4 5 6 7" ./scripts/launch_eval_all.sh
```

Full final eval:

```bash
RUN_TAG=paper512_seed0 EVAL_GPU_LIST="4 5 6 7" ./scripts/launch_eval_all.sh
```

The eval launcher supports:

- `CKPT_SUBDIR`: checkpoint under each method output dir, e.g. `step_250` or `final`.
- `EVAL_BENCHMARKS`: comma-separated list, default `math500,math,collegemath,olympiadbench`.
- `EVAL_LIMIT`: optional per-benchmark problem cap for sample eval.
- `EVAL_K`: samples per problem, default `16`.

## Latest Paper-Faithful Speed Probe

Hardware: one H100 80GB per method, GPUs 4-7. Config: Rho-1B MATH, 512
responses per PPO outer step, vLLM IPC sync, `save_every=0`, `max_steps=3`.

| Method | Mean step time | Rollout | Value/MC phase | Policy phase | Notes |
|---|---:|---:|---:|---:|---|
| PPO | ~90s | ~4s | 0s | ~74s | Terminal reward PPO |
| GRPO | ~92s | ~4s | 0s | ~76s | Group-relative terminal reward |
| CASPO | ~141s | ~4s | ~55s | ~69s | IPVRM value forward + online update |
| VinePPO K=9 | ~237s | ~4s | ~152s | ~69s | MC prefix rollouts dominate |

Approximate 1000-step ETAs:

- PPO: ~25 hours.
- GRPO: ~26 hours.
- CASPO: ~39 hours.
- VinePPO K=9: ~66 hours.

If all four methods run in parallel on GPUs 4-7, wall-clock is gated by
VinePPO: roughly 66-72 hours plus checkpoint/eval overhead.

## Validation Commands

Config sanity:

```bash
/opt/conda/envs/scalable/bin/python scripts/validate_configs.py --diff
```

Targeted tests:

```bash
/opt/conda/envs/scalable/bin/python -m pytest -q \
  tests/test_vllm_engine.py \
  tests/test_trainer_integration.py \
  tests/test_method_dispatch.py
```

IPC weight-sync probe:

```bash
CUDA_VISIBLE_DEVICES=4 /opt/conda/envs/scalable/bin/python \
  -m scripts.probe_vllm_ipc_sync \
  --config configs/caspo_rho1b_math.yaml \
  --output-dir /tmp/caspo_vllm_ipc_probe
```

## 7B Notes

The 7B configs remain available, but they are not the current four-method
single-node production target. Full-model 7B training should use FSDP; current
7B vLLM weight sync remains checkpoint-based until NCCL/in-memory sync is added
for the exact vLLM runtime. Do not assume the Rho-1B one-GPU-per-method plan
transfers to 7B.
