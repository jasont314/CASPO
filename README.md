# CASPO - Credit Assignment Step Policy Optimization

CASPO is a math-RL training stack for comparing:

- `ppo`: terminal-reward PPO with a sequence-level advantage.
- `grpo`: group-relative terminal reward advantages.
- `vineppo`: VinePPO-style step TD using Monte Carlo prefix rollouts.
- `caspo`: step TD using a learned IPVRM prefix value model instead of Monte Carlo prefix rollouts.

The main current target is a paper-faithful Rho-1B MATH run that matches the
VinePPO Rho-1B MATH setup where possible, while running all four methods in the
same trainer and vLLM infrastructure.

## Current Project State

This repo is a working research codebase, not a packaged library. The current
goal is to run controlled Rho-1B MATH RL experiments comparing terminal-reward
PPO/GRPO, Monte-Carlo-step VinePPO, and learned-prefix-value CASPO under one
trainer, one verifier, one dataset pipeline, and the same vLLM rollout stack.

The current production target is:

- Full-model RL fine-tuning, not LoRA.
- Rho-1B SFT base policy on MATH.
- 512 sampled responses per outer PPO step.
- Four saved checkpoints per full run: `step_250`, `step_500`, `step_750`,
  and `final`.
- Periodic cheap sample evals; full eval only at final unless curves are
  needed.
- CASPO uses the already-trained offline Rho-1B IPVRM checkpoint and can update
  that value model online during RL.
- VinePPO has both a one-GPU launcher and a faster two-GPU DDP launcher.

Important caveat: the Rho-1B one-GPU-per-method launchers are ready for the
main comparison. The 7B configs exist, but the fast Rho path should not be
blindly extrapolated to 7B because full-model 7B training needs sharding and
vLLM sync becomes a different systems problem.

## Repository Map

Core package:

```text
caspo/
  algo/                 PPO loss, step TD, advantage transforms/normalization
  config.py             Single dataclass config contract and validation
  data/                 MATH/GSM-style dataset loading and prompt formatting
  reward/               Math final-answer verifier wrapper
  rollout/
    sampler.py          HF rollout sampler
    vllm_engine.py      Embedded vLLM AsyncLLM rollout engine + IPC sync
  segmentation/
    steps.py            Token/newline and LaTeX-aware step segmentation
    latex_splitter.py   VinePPO-derived LaTeX-aware text splitter
  trainer/
    caspo_trainer.py    Main phase-2 trainer for PPO/GRPO/VinePPO/CASPO
  utils/                Distributed/runtime helpers, seeds, misc utilities
  value/                IPVRM prefix value model, ADB/DLW, value loss
```

Configs:

```text
configs/caspo_rho1b_math.yaml          Main Rho-1B MATH config
configs/caspo_rho1b_gsm8k.yaml         Rho-1B GSM8K variant
configs/caspo_deepseekmath7b_*.yaml    7B configs, not current production path
configs/caspo_smoke*.yaml              Small smoke configs
configs/value_smoke.yaml               Tiny phase-1 value smoke
```

Launch/eval scripts:

```text
scripts/launch_rho1b_parallel.sh       PPO/CASPO/GRPO/VinePPO, one GPU each
scripts/launch_rho1b_all8_standard.sh  Full seven-run, eight-GPU suite
scripts/launch_rho1b_{grpo,ppo,caspo}.sh
scripts/launch_rho1b_caspo_delta_{prob,log_prob}.sh
scripts/launch_rho1b_vineppo_ddp2.sh   Fast two-GPU VinePPO DDP path
scripts/launch_rho1b_caspo_ablations.sh
scripts/launch_rho1b_caspo_frozen_rm.sh
scripts/launch_eval_all.sh
scripts/launch_eval_rho1b_{sample,final}_all8.sh
scripts/train_value.py                 Phase-1 IPVRM training
scripts/train_caspo.py                 Phase-2 RL entrypoint
scripts/collect_value_data.py          Phase-1 rollout data collection
scripts/validate_configs.py            Config sanity and batch-shape checks
scripts/perf_env.sh                    Shared CUDA/NCCL/vLLM env settings
scripts/kill_zombies.sh                Cleanup helper for stale vLLM processes
```

Paper draft:

```text
paper/main.tex
paper/main.pdf
paper/references.bib
```

Tests:

```text
tests/test_distributed_runtime.py
tests/test_vllm_engine.py
tests/test_method_dispatch.py
tests/test_trainer_integration.py
tests/test_latex_splitter.py
...
```

## Method Summary

All phase-2 methods share the same rollout engine, verifier, tokenizer, prompt
template, PPO clipped objective, checkpointing, and eval path.

| Method | Credit signal | Extra model/compute | Main purpose |
|---|---|---|---|
| `ppo` | Terminal reward standardized over batch/group/off | No step model | Terminal-reward baseline |
| `grpo` | Per-prompt group-relative terminal reward | No value model | DeepSeekMath-style grouped baseline |
| `vineppo` | Step TD from MC prefix values | K continuations per nonterminal prefix | Paper-faithful VinePPO comparison |
| `caspo` | Step TD from learned IPVRM prefix values | Prefix value model, optional online update | Replace MC prefixes with learned value |

The key experiment is whether CASPO's learned prefix values recover enough
step-level credit assignment to be competitive with VinePPO while avoiding
VinePPO's expensive MC continuation phase.

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

CASPO also has a pre-normalization advantage ablation knob:

```yaml
caspo_advantage_transform: value
```

Supported values:

- `value`: direct IPVRM value TD difference, the current/default implementation.
- `prob`: TD difference after transforming prefix values with `sigmoid(V)`.
- `logprob`: TD difference after transforming prefix values with `log sigmoid(V)`.

The transform is applied before step-advantage normalization and clipping.
The terminal verifier reward term stays unchanged.

Separator sanity check: manual Rho-1B generations on MATH examples produced
coherent step boundaries for ordinary prose, equations, factorization, and
line-by-line aligned derivations. The splitter is a structural heuristic, not a
correctness judge. If a model keeps writing after a final boxed answer, those
post-answer fragments are currently still segmented as later steps; this is a
known analysis/cleanup point rather than a blocker for the current runs.

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

## Teammate Run Checklist

Before launching on lab GPUs:

```bash
cd /home/jason/experiment/CASPO
source /opt/conda/etc/profile.d/conda.sh
conda activate scalable
/opt/conda/envs/scalable/bin/python scripts/validate_configs.py --diff
nvidia-smi
```

Confirm these paths exist or update the YAML/env vars:

```text
/mnt/nvme_tmp/jason_caspo/hf_cache
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math/value_final
```

The reward/value model path in `configs/caspo_rho1b_math.yaml` is:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math/value_final
```

That path is a symlink to:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math/final
```

For SCP/rsync, prefer the real directory:

```bash
rsync -a user@HOST:/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math/final/ \
  ./rho1b_math_ipvrm/
```

If a run is interrupted, check for stale trainer/vLLM processes:

```bash
pgrep -af "train_caspo|VLLM::EngineCore|vllm"
nvidia-smi
```

Then clean only stale CASPO/vLLM jobs:

```bash
./scripts/kill_zombies.sh
```

Do not delete `/mnt/nvme_tmp/jason_caspo/hf_cache`; it avoids repeated model
downloads and startup delays.

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

## Standard 8-GPU Suite

For the full current experiment set, use seven launch scripts across eight
GPUs. The default map is:

| Experiment | GPUs | Script | Output tag |
|---|---:|---|---|
| GRPO | 0 | `scripts/launch_rho1b_grpo.sh` | `grpo` |
| PPO | 1 | `scripts/launch_rho1b_ppo.sh` | `ppo` |
| VinePPO K=9 DDP2 | 2,3 | `scripts/launch_rho1b_vineppo_ddp2.sh` | `vineppo_ddp2` |
| CASPO online RM | 4 | `scripts/launch_rho1b_caspo.sh` | `caspo` |
| CASPO delta-prob | 5 | `scripts/launch_rho1b_caspo_delta_prob.sh` | `caspo_prob` |
| CASPO delta-log-prob | 6 | `scripts/launch_rho1b_caspo_delta_log_prob.sh` | `caspo_logprob` |
| CASPO frozen RM | 7 | `scripts/launch_rho1b_caspo_frozen_rm.sh` | `caspo_frozen_rm` |

Launch all seven jobs at once:

```bash
cd /home/jason/experiment/CASPO
RUN_TAG=paper512_seed0 GPU_LIST="0 1 2 3 4 5 6 7" WANDB_MODE=offline \
  ./scripts/launch_rho1b_all8_standard.sh
```

Or launch a single job by overriding its GPU:

```bash
RUN_TAG=paper512_seed0 GPU=0 WANDB_MODE=offline ./scripts/launch_rho1b_grpo.sh
RUN_TAG=paper512_seed0 GPU=1 WANDB_MODE=offline ./scripts/launch_rho1b_ppo.sh
RUN_TAG=paper512_seed0 GPU_LIST="2 3" WANDB_MODE=offline ./scripts/launch_rho1b_vineppo_ddp2.sh
RUN_TAG=paper512_seed0 GPU=4 WANDB_MODE=offline ./scripts/launch_rho1b_caspo.sh
RUN_TAG=paper512_seed0 GPU=5 WANDB_MODE=offline ./scripts/launch_rho1b_caspo_delta_prob.sh
RUN_TAG=paper512_seed0 GPU=6 WANDB_MODE=offline ./scripts/launch_rho1b_caspo_delta_log_prob.sh
RUN_TAG=paper512_seed0 GPU=7 WANDB_MODE=offline ./scripts/launch_rho1b_caspo_frozen_rm.sh
```

All seven scripts use `configs/caspo_rho1b_math.yaml`, vLLM IPC sync,
`save_every=250`, and the current 1000-step standard unless `MAX_STEPS` or
`SAVE_EVERY` is overridden.

## CASPO Advantage Ablations

The direct-value CASPO variant is the normal `caspo` run above. Launch the two
additional CASPO ablations with:

```bash
RUN_TAG=paper512_seed0 GPU_LIST="4 5" WANDB_MODE=offline \
  ./scripts/launch_rho1b_caspo_ablations.sh
```

Default ablation outputs:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_prob_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_logprob_paper512_seed0
```

To include the direct-value run in an ablation-only sweep:

```bash
ADV_VARIANTS="value prob logprob" GPU_LIST="4 5 6" \
  ./scripts/launch_rho1b_caspo_ablations.sh
```

Frozen-RM CASPO keeps IPVRM prefix scoring but disables online value-model
updates:

```bash
RUN_TAG=paper512_seed0 GPU_LIST="4" WANDB_MODE=offline \
  ./scripts/launch_rho1b_caspo_frozen_rm.sh
```

Output:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_frozen_rm_paper512_seed0
```

## Two-GPU VinePPO

For Rho-1B VinePPO, the fastest multi-GPU path is replicated DDP with one
rank-local vLLM engine per GPU. The dedicated launcher uses GPUs 2 and 3 by
default, starts one trainer process per physical GPU, and preserves the
current 512-response global outer step:

```bash
RUN_TAG=paper512_seed0 GPU_LIST="2 3" WANDB_MODE=offline \
  ./scripts/launch_rho1b_vineppo_ddp2.sh
```

Default DDP shape:

```text
2 ranks x 32 prompts/rank x 8 rollouts = 512 responses/global step
2 ranks x 8 grad-accum micros x 4 responses = 64-response global PPO minibatch
```

Learning/effective-batch note: this two-GPU launcher is configured to match the
one-GPU VinePPO global batch. The one-GPU path uses `64 prompts x G=8 = 512`
responses per outer step; the two-GPU path uses `2 ranks x 32 prompts/rank x
G=8 = 512`. PyTorch DDP averages gradients across ranks, so the effective
learning rate is unchanged when `2 x micro_batch_size x grad_accum_steps = 64`.
The run is not bit-identical because prompt sharding, generation RNG,
all-reduce order, and vLLM scheduling differ.

Recommended fastest tested Rho-1B DDP settings:

```bash
# Preserves the same 64-response global PPO minibatch:
# 2 ranks x micro_batch_size=4 x grad_accum_steps=8 = 64 responses.
MICRO_BATCH_SIZE=4 GRAD_ACCUM_STEPS=8 USE_GRADIENT_CHECKPOINTING=false \
LOGPROB_MICRO_BATCH_SIZE=16 CASPO_VLLM_GPU_MEMORY_UTILIZATION=0.55 \
RUN_TAG=paper512_seed0 GPU_LIST="2 3" WANDB_MODE=offline \
  ./scripts/launch_rho1b_vineppo_ddp2.sh
```

Useful optional knobs:

```bash
# vLLM aliases use CASPO_ prefixes so they do not leak into vLLM as unknown
# native environment variables. The older VLLM_* aliases are accepted by the
# launcher, then unset before child processes start.
CASPO_VLLM_GPU_MEMORY_UTILIZATION=0.55
CASPO_VLLM_MAX_NUM_SEQS=512
CASPO_VLLM_MAX_NUM_BATCHED_TOKENS=32768
CASPO_VLLM_MULTI_SAMPLE_MODE=batched
```

Do not force the aggressive vLLM scheduler knobs by default. In the latest
probe, `CASPO_VLLM_MULTI_SAMPLE_MODE=batched` with `max_num_seqs=512` and
`max_num_batched_tokens=32768` was slower than the default/auto scheduler.

The launcher prints the resolved knobs before starting rank processes and
writes separate rank logs:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_<RUN_TAG>/logs/phase2_vineppo_ddp2_rank0.log
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_<RUN_TAG>/logs/phase2_vineppo_ddp2_rank1.log
```

Rank-zero step logs include:

```text
t_roll   rollout generation time
t_old    trainer-policy old-logprob rescore
t_value  CASPO value or VinePPO MC prefix-value phase
t_ref    frozen-reference logprob precompute
t_pol    PPO forward/backward/optimizer phase
t_sync   trainer-to-vLLM weight sync
t_step   full outer-step wall-clock
```

Output:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_vineppo_ddp2_paper512_seed0
```

For an infrastructure smoke:

```bash
MAX_STEPS=1 SAVE_EVERY=0 PROMPTS_PER_STEP=1 GROUP_SIZE=1 \
GRAD_ACCUM_STEPS=1 VINEPPO_MC_ROLLOUTS=1 RUN_TAG=ddp2_smoke \
GPU_LIST="2 3" WANDB_MODE=disabled ./scripts/launch_rho1b_vineppo_ddp2.sh
```

## Evaluation

Do cheap sample evals at saved checkpoints and full eval only at the end. The
training loop does not run eval in-process; eval is launched from saved
checkpoints with vLLM. Keep the sample cadence aligned with checkpointing:
`step_250`, `step_500`, `step_750`, and `final`.

Standard seven-method sample eval:

```bash
RUN_TAG=paper512_seed0 CKPT_SUBDIR=step_250 \
EVAL_GPU_LIST="0 1 2 3 4 5 6" ./scripts/launch_eval_rho1b_sample_all8.sh
```

This defaults to `math500`, `EVAL_LIMIT=100`, and `EVAL_K=8`. On Rho-1B, the
old full MATH-500 k=16 eval took about 1-2 minutes per model including vLLM
startup; the 100-problem k=8 sample is expected to be well under that. If all
seven methods are evaluated in parallel, sample eval wall-clock should usually
be a couple of minutes, but it needs free eval GPUs.

Standard seven-method full final eval:

```bash
RUN_TAG=paper512_seed0 EVAL_GPU_LIST="0 1 2 3 4 5 6" \
  ./scripts/launch_eval_rho1b_final_all8.sh
```

This runs `math500,math,collegemath,olympiadbench` at `k=16`. The prior
Rho-1B MATH-500 k=16 generation time was about 56 seconds per model after vLLM
startup. Full final eval is dominated by full MATH test, so budget roughly
15-25 minutes wall-clock when the seven methods run in parallel on seven free
H100s.

The eval launcher supports:

- `METHODS`: space-separated checkpoint directory tags, default `grpo ppo vineppo_ddp2 caspo caspo_prob caspo_logprob caspo_frozen_rm`.
- `CKPT_SUBDIR`: checkpoint under each method output dir, e.g. `step_250` or `final`.
- `EVAL_BENCHMARKS`: comma-separated list, default `math500,math,collegemath,olympiadbench`.
- `EVAL_LIMIT`: optional per-benchmark problem cap for sample eval.
- `EVAL_K`: samples per problem, default `16`.
- `EVAL_VLLM_GPU_MEMORY_UTILIZATION`: defaults to `0.85` because eval does not
  share the GPU with a trainer.

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

## Latest Two-GPU VinePPO Probe

Hardware: two H100 80GB GPUs, replicated DDP, one rank-local vLLM engine per
GPU, global shape still 512 responses per outer step. Each row is a one-step
probe, so treat small differences as noise; the large policy-path gains were
repeatable enough to keep.

| DDP setting | Step time | Old logprobs | MC/value | Ref logprobs | Policy | Notes |
|---|---:|---:|---:|---:|---:|---|
| `micro=1, accum=32, grad_ckpt=true` | 155.2s | 5.5s | 79.9s | 5.2s | 50.6s | Original DDP shape |
| `micro=2, accum=16, grad_ckpt=false` | 134.1s | 3.6s | 80.0s | 3.3s | 23.0s | Same global PPO minibatch |
| `micro=4, accum=8, grad_ckpt=false` | 118.2s | 2.9s | 78.2s | 2.7s | 19.2s | Main policy win |
| `micro=4, accum=8, logprob_micro=16` | 115.7s | 2.4s | 72.8s | 2.2s | 18.6s | Best conservative setting |
| Same plus `CASPO_VLLM_GPU_MEMORY_UTILIZATION=0.55` | 115.5s | 2.4s | 80.5s | 2.1s | 18.7s | Tie within noise |
| Same plus forced batched vLLM, `512` seqs, `32768` tokens | 131.1s | 2.4s | 92.4s | 2.2s | 18.6s | Slower; do not use by default |

The current recommended two-GPU VinePPO command is the optimized
`micro=4/accum=8/grad_ckpt=false/logprob_micro=16` launcher above. Compared
with the earlier one-GPU VinePPO probe average of about 232.5s/step, the
optimized two-GPU path is roughly 2x faster while preserving the same global
outer-step batch and PPO minibatch. A 1000-step two-GPU VinePPO run is roughly
32 hours before checkpoint/eval overhead. On a four-GPU machine, using the
two-GPU VinePPO path means the four-method comparison should be run in waves or
with PPO/GRPO/CASPO on other available GPUs rather than all four methods sharing
only GPUs 4-7 at once.

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
