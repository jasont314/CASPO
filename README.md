# CASPO - Credit Assignment Step Policy Optimization

CASPO is a math-RL training stack for comparing:

- `grpo`: group-relative terminal reward advantages.
- `ppo_critic`: PPO with a learned scalar value head (Schulman-2017).
- `vineppo`: VinePPO-style step TD using Monte Carlo prefix rollouts.
- `caspo`: step TD using a learned IPVRM prefix value model instead of
  Monte Carlo prefix rollouts. Supports `Δp`, `Δlogp`, `frozen-RM`, and
  iterated-refresh variants.

**Current target (May 2026): Qwen2.5-Math-1.5B + dsr_sub** (1209-prompt
DeepScaleR subset from One-Shot-RLVR), 4-GPU FSDP. DeepSeekMath-7B on
MATH-lighteval is retained as a paper-faithful 7B reference site.

The earlier Rho-1B-MATH replication track (paper-faithful VinePPO setup
on `realtreetune/rho-1b-sft-MATH`) is **archived**: launchers and configs
remain in the tree for reproduction but are no longer the active research
target. Sections describing the Rho-1B setup below are kept for historical
context only.

## Upstream VinePPO Attribution

This project is an adaptation and extension of the VinePPO experimental setup,
not an independent reimplementation from only the paper text. We use the
VinePPO paper and public codebase as the reference point for PPO/VinePPO
hyperparameters, rollout shape, prompt template, evaluation protocol, and
LaTeX-aware step segmentation behavior. The current Qwen2.5-Math-1.5B +
dsr_sub setup adapts these to the model's native template and the
One-Shot-RLVR `dsr_sub` subset; the archived Rho-1B-MATH track is the
verbatim VinePPO replication.

Upstream VinePPO resources:

```text
paper: https://arxiv.org/abs/2410.01679
code:  https://github.com/McGill-NLP/VinePPO
```

Concrete places adapted or matched from VinePPO:

- `configs/caspo_rho1b_math.yaml` mirrors the Rho-1B MATH PPO/VinePPO setup,
  including 64 prompts, group size 8, 512 responses per outer step,
  1024-token response budget, PPO epochs, policy LR, KL coefficient, and
  warmup shape.
- `caspo/segmentation/latex_splitter.py` is a VinePPO-derived LaTeX-aware step
  splitter used by both CASPO and VinePPO runs.
- `method=vineppo` implements the VinePPO-style Monte Carlo prefix-value
  baseline inside this repo's shared trainer/vLLM stack.
- CASPO is the project-specific extension: it replaces VinePPO's online Monte
  Carlo prefix rollouts with an IPVRM-style learned prefix value model while
  keeping the comparison stack matched where possible.

## Related Work

This repo is built around a direct comparison among several math-RL credit
assignment approaches:

- PPO: the clipped policy-gradient objective used as the shared policy update
  backbone. Paper: `https://arxiv.org/abs/1707.06347`.
- VinePPO: the main upstream experimental reference. VinePPO estimates
  intermediate reasoning values with Monte Carlo continuations from prefixes.
  This repo keeps a VinePPO baseline for direct comparison and adapts the
  upstream PPO recipe to Qwen2.5-Math-1.5B + dsr_sub. Paper:
  `https://arxiv.org/abs/2410.01679`; code:
  `https://github.com/McGill-NLP/VinePPO`.
- IPVRM: the learned prefix-value model used for CASPO's reward/value signal.
  CASPO replaces VinePPO's online MC prefix values with IPVRM-style learned
  prefix values and optional online value updates. Paper:
  `https://arxiv.org/abs/2604.13197`.

  Differences between CASPO's PRM and the original IPVRM paper:

  | aspect | IPVRM (paper) | CASPO PRM (this repo) |
  |---|---|---|
  | architecture | V_φ = β · Σ log(π_φ/π_ref) | **same** (cumulative log-ratio) |
  | training data | trajectory-level outcome labels | **per-prefix Monte Carlo p̂** (J=8 continuations per labeled prefix, K=16 base rollouts, S=5 step boundaries per response, mixed-outcome filter on K) |
  | loss | BCE-with-margin on prefix-vs-trajectory pairs | **BCE on continuous targets** σ(V_φ/β) vs p̂ ∈ [0,1] (no margin term) |
  | refresh strategy | inline online updates during RL | **frozen during RL by default**; iterated-refresh variant retrains from scratch on current-policy rollouts |
  | step-TD plug-in | V_φ as generalized value | σ(V_φ/β) plugged into vanilla step-TD: A_t = σ(V_φ(s_{t+1})/β) − σ(V_φ(s_t)/β) |
  | β | learned/tuned | fixed β = 10 |
  | indexing fix | — | V read at `step_end+1` (cumsum-through-token correction; off-by-one repro guard) |

  We empirically confirmed (2026-05-04) that the cumulative-log-ratio
  parameterization is essential: at matched hparams, an alternative
  single-sigmoid-head V_φ = σ(W·h_φ) collapsed val Spearman ρ from ~0.78 to
  ~0.43 because the log-ratio's per-token decomposition gives every response
  token a direct gradient signal (~80× supervision density vs the head's
  single end-of-prefix gradient).
- DeepSeekMath/GRPO: the grouped relative policy optimization baseline. In this
  repo, GRPO uses grouped terminal rewards over `G=8` responses per prompt and
  the same clipped PPO loss implementation as the other methods. Paper:
  `https://arxiv.org/abs/2402.03300`.
- vLLM: the rollout/eval generation engine used for high-throughput sampling
  and CUDA-IPC trainer-to-vLLM weight sync. Paper:
  `https://arxiv.org/abs/2309.06180`.

## Current Project State

This repo is a working research codebase, not a packaged library. The
current goal is to run controlled RL experiments on Qwen2.5-Math-1.5B
with the One-Shot-RLVR `dsr_sub` subset (1209 DeepScaleR prompts),
comparing GRPO, PPO+critic, VinePPO, and CASPO (Δp / Δlogp / frozen-RM /
iterated-refresh) under one trainer, one verifier, one dataset pipeline,
and the same vLLM rollout stack.

The current production target is:

- Full-model RL fine-tuning, not LoRA.
- Qwen2.5-Math-1.5B SFT base policy on `dsr_sub` (1209 DeepScaleR prompts).
- 1024 sampled responses per outer PPO step (128 prompts × G=8).
- Saved checkpoints every 50 outer steps for trajectory-aware eval.
- 4-GPU FSDP + colocated rank-local vLLM, full-model fine-tuning.
- CASPO uses an MC-trained Qwen-1.5B PRM and can update online during RL,
  or refresh from scratch periodically against current-policy rollouts
  (alternating launcher).
- DeepSeekMath-7B on MATH-lighteval is retained as a paper-faithful 7B
  reference; runs on the same trainer with model+data+template overrides.

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
scripts/archive/rho1b/launch_rho1b_parallel.sh       PPO/CASPO/GRPO/VinePPO, one GPU each
scripts/archive/rho1b/launch_rho1b_all8_standard.sh  Full seven-run, eight-GPU suite
scripts/archive/rho1b/launch_rho1b_{grpo,ppo,caspo}.sh
scripts/archive/rho1b/launch_rho1b_caspo_delta_{prob,log_prob}.sh
scripts/archive/rho1b/launch_rho1b_vineppo_ddp2.sh   Fast two-GPU VinePPO DDP path
scripts/archive/rho1b/launch_rho1b_caspo_ablations.sh
scripts/archive/rho1b/launch_rho1b_caspo_frozen_rm.sh
scripts/launch_eval_all.sh
scripts/archive/rho1b/launch_eval_rho1b_{sample,final}_all8.sh
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

## Qwen2.5-Math-1.5B + dsr_sub Setup (May 2026)

The **current paper setup uses Qwen2.5-Math-1.5B + dsr_sub** (the
1209-prompt DeepScaleR subset from One-Shot-RLVR), 4-GPU FSDP, and is the
active research target. Same trainer code, same config file
(`configs/caspo_rho1b_math.yaml`, reused as the trainer-config schema),
but different `model_name_or_path`, `dataset_name`, `prompts_per_step`,
`max_response_len`, `epochs_per_rollout`. The archived Rho-1B-MATH section
near the bottom of this document retains the historical numbers for
reference.

| Field | Qwen2.5-Math-1.5B value |
|---|---|
| Base policy | `Qwen/Qwen2.5-Math-1.5B` |
| Dataset | `dsr_sub.jsonl` (1209 DeepScaleR prompts; One-Shot-RLVR subset) |
| Eval | `math500`, `gsm8k`, `olympiadbench` (greedy, k=1, T=0) |
| Prompt template | `{query}\nLet's think step by step and output the final answer within \boxed{}.` |
| Response budget | **2048 tokens** (catches p98 of CORRECT Qwen2.5-Math-1.5B chains on dsr_sub, n=800 empirical; 2026-05-03 update). Truncates ~8% overall but only ~2% of correct chains — the truncated tail is overwhelmingly failed/rambling chains, which is the seq_len penalty signal. |
| Sequence budget | **3072 tokens** (max_prompt=1024 + max_response=2048; well within Qwen `max_position_embeddings=4096`) |
| Rollout group | `group_size=8` |
| Prompts per step | `128` (= 1024 responses per outer step at G=8) |
| Topology | **FSDP=4 + colocated vLLM**, `vllm_gpu_memory_utilization=0.45` (CASPO) or `0.35` (PPO+critic, leaves room for critic) |
| PPO minibatch | `micro_batch_size=4`, `grad_accum_steps=8` (= 32 effective batch / rank) |
| Policy LR | `1e-6` |
| KL coefficient | `0.001` for CASPO/GRPO, `0.01` for PPO+critic (1B-stable; 1e-4 diverges) |
| Steps | `600` (default for all methods); resume runs add 250-350 more |
| Save cadence | `save_every=50` |

### Method launchers (Qwen2.5-Math-1.5B + dsr_sub, 4-GPU FSDP)

| Method | Launcher | `epochs_per_rollout` | `kl_coef` | Other notable |
|---|---|---|---|---|
| **GRPO** | **`scripts/launch_qwen_grpo.sh`** | **1 AND 2** (run both) | 0.001 | value-free baseline; group-relative terminal advantages. Reported at both μ=1 (DeepSeekMath / TRL canonical) and μ=2 (iso-budget vs PPO+critic / VinePPO / CASPO). Toggle via `EPOCHS_PER_ROLLOUT=2` |
| **PPO+critic** | **`scripts/launch_qwen_ppo_critic.sh`** | **2** (fixed) | 0.01 | VinePPO PPO baseline config: `lambda=1.0`, `value_loss_coef=1.0`, `cliprange_value=0.2`, `critic_lr=1e-6` |
| **VinePPO** | **`scripts/launch_qwen_vineppo.sh`** | **2** (fixed) | 0.01 | upstream MATH config K_MC=9; ⚠ ~33 min/step at K=9, drop to `VINEPPO_MC_ROLLOUTS=5` and lower `MAX_STEPS` for tractable wall-clock |
| **CASPO** | **`scripts/launch_qwen_caspo.sh`** | **2** (fixed) | 0.001 | step-TD over frozen PRM; `ADV_TRANSFORM=prob` (Δp, default) or `logprob` (Δlogp); needs `PRM_PATH` |
| **CASPO + alternating refresh** | **`scripts/launch_caspo_alternating.sh`** | 2 | 0.001 | self-contained PRM → RL → PRM → RL pipeline. Trains initial PRM as Phase 0, then alternates RL with PRM refresh. `REFRESH_EVERY=150` (Δp) or `200` (Δlogp) |
| **CASPO refresh (Phase 2 only)** | **`scripts/launch_caspo_refresh_resume.sh`** | 2 | 0.001 | resume from any Phase-1 ckpt with new PRM. Preserves optimizer/lr_scheduler/ref_policy from Phase 1 |
| **MC PRM training** | **`scripts/launch_qwen_mc_prm.sh`** | – | – | unified pipeline (mc_step_label.py 4-shard collect + train_value_mc.py FSDP=4 train). Use for initial PRM (`POLICY=base`) or ad-hoc refresh |

For copy-paste teammate quickstart commands, see
[REPRODUCE.md → Quickstart](REPRODUCE.md#quickstart-baselines-for-a-teammate).

### Quickstart (copy-paste)

Prerequisites: clone the repo, activate the `scalable` conda env (or set
`CONDA_ENV` / `PYBIN`), have `dsr_sub.jsonl` available locally, and 4×
H100 80GB GPUs (FSDP=4). All launchers default to `MAX_STEPS=600`,
auto-eval all saved checkpoints on `math500/gsm8k/olympiadbench` at
greedy (k=1, T=0) after training, and write eval JSON to
`$OUT_DIR/eval/${ckpt}.json`.

```bash
# === GRPO (canonical, μ=1) — DeepSeekMath / TRL default ===
DSR_SUB=/path/to/dsr_sub.jsonl \
GPU_LIST="0 1 2 3" \
OUT_DIR=/mnt/data/runs/grpo_mu1_qwen25math15b_dsr \
LOG_DIR=/tmp/grpo_mu1_$(date +%Y%m%d_%H%M) \
  bash scripts/launch_qwen_grpo.sh
# ETA: ~10-14 h on 4× H100 80GB

# === GRPO (iso-budget, μ=2) — VinePPO-paper-style ===
DSR_SUB=/path/to/dsr_sub.jsonl \
GPU_LIST="0 1 2 3" \
OUT_DIR=/mnt/data/runs/grpo_mu2_qwen25math15b_dsr \
LOG_DIR=/tmp/grpo_mu2_$(date +%Y%m%d_%H%M) \
EPOCHS_PER_ROLLOUT=2 \
  bash scripts/launch_qwen_grpo.sh
# ETA: ~18-26 h on 4× H100 80GB

# === PPO + critic baseline ===
DSR_SUB=/path/to/dsr_sub.jsonl \
GPU_LIST="0 1 2 3" \
OUT_DIR=/mnt/data/runs/ppo_critic_qwen25math15b_dsr \
LOG_DIR=/tmp/ppo_critic_$(date +%Y%m%d_%H%M) \
  bash scripts/launch_qwen_ppo_critic.sh
# ETA: ~25 h on 4× H100 80GB
```

**On GRPO μ=1 vs μ=2:** μ=1 is the GRPO-canonical choice (DeepSeekMath
origin paper, TRL/verl defaults). μ=2 is what the VinePPO paper used for
its GRPO baseline to match PPO+critic compute. We report both — the
canonical-μ=1 number is the GRPO-faithful comparison and the μ=2 number
is the iso-compute comparison (so a CASPO-vs-GRPO win can't be
attributed to GRPO doing fewer policy updates per rollout).

**On VinePPO**: a baseline `scripts/launch_qwen_vineppo.sh` exists, but at
the upstream-faithful K_MC=9 config its step time is ~33 min/step on this
dataset (because the LaTeX-aware splitter produces ~39 step boundaries
per response on Qwen chains, dwarfing the upstream MATH config's ~10-20).
600 steps would take ~14 days. For a teammate run, drop K_MC to 5 and
steps to 300:

```bash
DSR_SUB=/path/to/dsr_sub.jsonl \
GPU_LIST="0 1 2 3" \
OUT_DIR=/mnt/data/runs/vineppo_qwen25math15b_dsr \
VINEPPO_MC_ROLLOUTS=5 \
MAX_STEPS=300 \
  bash scripts/launch_qwen_vineppo.sh
# ETA: ~12 min/step × 300 = ~60 h (~2.5 days) on 4× H100 80GB
```

If 8 GPUs are available, runs can be parallelized: GRPO μ=1 on GPUs 0-3,
PPO+critic on GPUs 4-7, etc. Total wall-clock = max of individual ETAs
instead of sum.

Useful overrides per launcher: `MAX_STEPS=200` (smoke test),
`RUN_EVAL=false` (skip auto-eval), `SAVE_EVERY=100` (fewer ckpts).

### PPO+critic recipe — exact config

The portable launcher at `scripts/launch_qwen_ppo_critic.sh` matches
VinePPO upstream's PPO baseline (`lam1.jsonnet` + `ppo_MATH.jsonnet`).
Configurable via env vars (`CONDA_ENV`, `GPU_LIST`, `DSR_SUB`,
`OUT_DIR`, `MAX_STEPS`, `KL_COEF`).

```
method=ppo_critic
epochs_per_rollout=2          # VinePPO upstream uses 2 (NOT 1)
ppo_gae_lambda=1.0            # VinePPO lam1.jsonnet — terminal-only verifier reward
value_loss_coef=1.0           # VinePPO + textbook
cliprange_value=0.2           # VinePPO + textbook
clip_eps_low/high=0.2         # VinePPO + textbook
critic_lr=1e-6                # = policy_lr; matches VinePPO
critic_weight_decay=0.0       # VinePPO
critic_grad_clip=1.0          # VinePPO
kl_coef=0.01                  # 1B-stable. VinePPO uses 1e-4 at 7B; at 1B with our config, 1e-4 diverges. See "F5" lesson below.
```

### F5 lesson — kl_coef at 1B

`launch_rho1b_ppo_critic.sh` previously hardcoded `kl_coef=1e-4`,
which silently bypassed the YAML's `1e-2` default. PPO+critic
diverged at 1B while CASPO/Δp/GRPO trained stably (because they
inherit from YAML and don't override). Fix: do NOT hardcode
`kl_coef` in PPO+critic launchers; inherit YAML's stronger
ref-anchor or set explicitly to ≥0.01.

ETA on 4×H100 80GB: ~150-180s/step × 500 steps ≈ **~22-25h** for
PPO+critic. Slower than CASPO Δp (~80s/step × 500 ≈ 11h) primarily
because of `epochs_per_rollout=2`.

### CASPO refresh (May 2026 result)

The "v3" experiment (resume CASPO Δp from step_150 with a freshly-
trained PRM, with proper Adam/lr_scheduler/base-ref resume) lifted
peak math500 from 0.664 → 0.680 (+1.6pp) and peak gsm8k from
0.770 → 0.829 (+5.9pp). See `docs/RM_TRAINING.md` for the selected
recipe + the full 14-axis experimental matrix (decay-curve probing,
LoRA vs full-FT, refresh interval sweep, etc.).

Resume requires loading optimizer state + lr_scheduler state +
global_step (added in `caspo/trainer/caspo_trainer.py:_load_optimizer_state`)
AND setting `cfg.ref_model_path` to the original SFT base (so KL
doesn't anchor to the rolling policy). Without these, refresh
collapses (math500 −28pp). See
`feedback_resume_optimizer_state.md` in session memory.

### Two-phase / alternating refresh (2026-05-03)

Two new scripts encapsulate the refresh pattern:

- **`scripts/launch_caspo_refresh_resume.sh`** — Phase 2 only.
  Resume from any Phase-1 ckpt with a NEW PRM. Required env:
  `POLICY_CKPT`, `NEW_PRM`, `OUT_DIR`. All other hparams (lr, kl,
  ep, optimizer, lr_scheduler, ref_policy=base SFT) preserved
  from Phase 1; ONLY `prefix_value_path` changes.

- **`scripts/launch_caspo_alternating.sh`** — End-to-end
  PRM → RL → PRM → RL → ... pipeline. Trains the initial PRM as
  Phase 0, then alternates RL with PRM refresh until `TOTAL_STEPS`
  reached. Required env: `INITIAL_CKPT`, `OUT_ROOT`, `DSR_SUB`.
  Configurable `REFRESH_EVERY` (default 150 for Δp; 200 for Δlogp due
  to slower drift), `TOTAL_STEPS`, `ADV_TRANSFORM` (`prob`/`logprob`),
  `METHOD`. Pass `INITIAL_PRM=/path/...` to skip Phase 0 if a
  pre-trained PRM is already available.

Each refresh cycle uses unified `max_response_len=2048` (matches RL
deployment cap) without prefix decoupling — train/deploy distributions
stay aligned. Empirically: 2048 catches p98 of correct Qwen2.5-Math-1.5B
chains (1613 tokens), and the ~8% of all chains it truncates are
overwhelmingly failed/rambling — exactly what the seq_len penalty is
designed to discourage.

### PRM training recipe (unified at 2048, 2026-05-03)

Best PRM config (ρ=0.456 vs orig PRM 0.443) on 4-GPU FSDP:

```
COLLECTION (mc_step_label.py — 4-shard parallel):
  --K 16 --J 16 --steps_per_response 5
  --max_prompt_len 1024 --max_response_len 2048
  --max_train_prefix_len 0           # = match collection cap (default)
  --temperature 1.0 --top_p 1.0 --seed 0

TRAINING (train_value_mc.py — FSDP=4):
  --lr 5e-6 --mb 4 --grad_accum 2     # eff_batch = 4×4×2 = 32
  --eval_mb 16                        # eval has no backward → 4× larger fits
  --epochs 2                          # ρ saturates ~step_3500 ≈ 2 epochs
  --val_fraction 0.1                  # matches orig PRM
  --early_stop_patience 999           # no early stop; let val select best
  --beta 10.0 --seed 0

VAL SPLIT MODES (train_value_mc.py):
  default                              # row-level shuffle. LEAKY — same prompt
                                       # may have prefixes in both train and val.
                                       # val ρ overstates OOD generalization.
  --split_by_prompt                    # hash prompt_ids, hold out val_fraction
                                       # of UNIQUE PROMPTS (all their prefixes
                                       # → val). Removes same-prompt leakage
                                       # within the same N=300 collection.
  --held_out_data path/to/holdout.pt   # use a separate mc_labels-format .pt
                                       # as the val set (true OOD-prompt eval).
                                       # Generate with launch_qwen_mc_prm_holdout.sh
                                       # on the remaining 909 dsr_sub prompts.
```

Key lessons from sweep (2026-05-02 to 2026-05-03):
- `eff_batch` matters: mb=1 (eff=4) → ρ=0.35; mb=4 acc=2 (eff=32) → ρ=0.45
- ep=1 ≈ ep=2 ≈ ep=3 (within 0.01); ep=2 sweet spot
- K=4 (eff=16) < K=8 (eff=32) only because of compute, not K diversity per se
- J subsampling not supported by current data format (J=16 always)
- The earlier "1024 ≫ 1792 (-0.10 ρ)" finding was confounded with
  probe-cap mismatch: probe was at cap=1024, so 1792-trained PRMs got
  scored OOD on their long-prefix half. v3 refresh at cap=1536 trained
  without prefix decoupling → ρ=0.630 in-dist, confirming longer-cap
  training is fine when train/probe distributions match.

---

## Rho-1B MATH Setup (ARCHIVED)

> ⚠️ **This section is archived.** The Rho-1B-MATH replication track is no
> longer the active research target. Sections below describe the historical
> setup; launchers and configs remain in the tree for reproducibility but
> have been superseded by the Qwen2.5-Math-1.5B + dsr_sub setup above.

Main config: `configs/caspo_rho1b_math.yaml` (note: this YAML is still the
trainer-config schema reused by Qwen launchers via `--override`; the
defaults below are for the Rho-1B run, not the Qwen run)

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
| PPO minibatch (YAML default) | `micro_batch_size=1`, `grad_accum_steps=64` |
| PPO minibatch (launcher override, recommended) | `micro_batch_size=8`, `grad_accum_steps=8`, `use_gradient_checkpointing=false` |
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

## Exact Models And Data

All current standard runs use the same base policy, tokenizer, RL prompt set,
prompt format, verifier, and eval suite unless explicitly overridden.

Training/RL policy model:

```text
realtreetune/rho-1b-sft-MATH
```

The tokenizer defaults to the same path as the policy model/checkpoint unless a
checkpoint directory supplies its own tokenizer files.

RL prompt data:

```text
DigitalLearningGmbH/MATH-lighteval
split: train
question field: problem
prompt template: [MATH_TASK] Problem:\n{query}\n\nSolution:
```

During RL, each outer step samples 64 prompts and generates 8 responses per
prompt, for 512 responses per step. Rollout sampling uses temperature `0.6`,
top-p `0.9`, top-k disabled, and max response length `1024`.

Phase-1 IPVRM value data artifact:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math/value_data.pt
```

Current IPVRM prefix value checkpoint used by CASPO:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math/value_final
```

This is a symlink to the trained value checkpoint directory:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math/final
```

Standard final eval suite:

| Eval name | Dataset | Split/config | Default size |
|---|---|---|---:|
| `math500` | `HuggingFaceH4/MATH-500` | `test` | 500 |
| `math` | `DigitalLearningGmbH/MATH-lighteval` | `test` | full test |
| `collegemath` | `realtreetune/college_math` | `test` | 500-problem default limit |
| `olympiadbench` | `Hothan/OlympiadBench` | `OE_TO_maths_en_COMP`, `train` | 674 |

Standard method output roots for `RUN_TAG=paper512_seed0`:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_grpo_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_ppo_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_vineppo_ddp2_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_prob_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_logprob_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_frozen_rm_paper512_seed0
```

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

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (we previously layered on
  `garbage_collection_threshold` and `max_split_size_mb`; reverted because both
  added 3-4× step-time cost on memory-tight regimes)
- `TORCH_NCCL_BLOCKING_WAIT=1`
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
- `NCCL_TIMEOUT=1800`
- `TOKENIZERS_PARALLELISM=false`
- `PYTHONUNBUFFERED=1`
- `VLLM_NO_USAGE_STATS=1`
- `VLLM_LOGGING_LEVEL=WARNING`
- `OMP_NUM_THREADS=4`
- `MKL_NUM_THREADS=4`

**vLLM CUDA graph trim** (default ON, gate via
`CASPO_VLLM_CUDAGRAPH_TRIM=0`): the engine constructor passes
`compilation_config={"cudagraph_capture_sizes":[…], "cudagraph_mode":
"FULL_DECODE_ONLY"}` to vLLM v1 — drops piecewise prefill graphs and
caps the captured shape set to a sparse log-spaced list up through
`max_num_seqs`. Frees ~150-300 MB / rank at near-zero speed cost on
RL rollout patterns (decode-bound; prefill is short and infrequent).

**Trainer-side `empty_cache`**: `caspo_trainer.py:step()` calls
`torch.cuda.empty_cache()` at the very start of each step (releases
fragmentation from prior vLLM weight sync) and immediately before the
next sync (releases policy/critic activation peaks). Both are unconditional;
overhead is <0.15% of step time, helps memory-tight regimes (mb=4 colocated)
fit without hurting the relaxed regimes.

For Rho-1B single-GPU-per-method runs, vLLM weight sync uses CUDA IPC:

```yaml
rollout_backend: vllm
vllm_weight_sync_backend: ipc
vllm_gpu_memory_utilization: 0.30   # set by _launch_rho1b_one_gpu.sh; YAML still says 0.45
vllm_enforce_eager: false
```

The launcher's `vllm_gpu_memory_utilization=0.30` default came out of an
April 2026 Pareto sweep on Rho-1B; vLLM rollout already runs in ~4 s out of a
50 s step, so KV-cache budget above ~0.30 buys nothing for step time but eats
trainer headroom. At `u=0.30` the trainer keeps ~3-4 GB margin even for CASPO
(peak ~77 GB at `mb=8, accum=8, ckpt=false`). Override with
`CASPO_VLLM_GPU_MEMORY_UTILIZATION=...` if a method needs different headroom.

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
df -h /mnt/nvme_tmp                       # at least ~50 GB free for 7-method × 4-checkpoint suite
```

A full 8-GPU suite writes 7 methods × 4 checkpoints × ~2.1 GB ≈ 60 GB of
saved weights, plus per-method logs and wandb buffers. If `/mnt/nvme_tmp` is
above ~95% full before launch, free space first — checkpoint writes that
ENOSPC mid-run can corrupt the active save and the trainer will crash on the
next save attempt.

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
  ./scripts/archive/rho1b/launch_rho1b_parallel.sh
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
SAVE_EVERY=100 MAX_STEPS=300 RUN_TAG=debug ./scripts/archive/rho1b/launch_rho1b_parallel.sh
```

## Standard 8-GPU Suite

For the full current experiment set, use seven launch scripts across eight
GPUs. The default map is:

| Experiment | GPUs | Script | Output tag |
|---|---:|---|---|
| GRPO | 0 | `scripts/archive/rho1b/launch_rho1b_grpo.sh` | `grpo` |
| PPO+critic | 1 | `scripts/archive/rho1b/launch_rho1b_ppo_critic.sh` | `ppo_critic` |
| VinePPO K=9 DDP2 | 2,3 | `scripts/archive/rho1b/launch_rho1b_vineppo_ddp2.sh` | `vineppo_ddp2` |
| CASPO online RM | 4 | `scripts/archive/rho1b/launch_rho1b_caspo.sh` | `caspo` |
| CASPO delta-prob | 5 | `scripts/archive/rho1b/launch_rho1b_caspo_delta_prob.sh` | `caspo_prob` |
| CASPO delta-log-prob | 6 | `scripts/archive/rho1b/launch_rho1b_caspo_delta_log_prob.sh` | `caspo_logprob` |
| CASPO frozen RM | 7 | `scripts/archive/rho1b/launch_rho1b_caspo_frozen_rm.sh` | `caspo_frozen_rm` |

(The legacy critic-free `launch_rho1b_ppo.sh` is preserved but **not** part
of the standard suite — "PPO" in the head-to-head means PPO+critic, the
proper Schulman 2017 baseline against CASPO's pretrained V_φ.)

Launch all seven jobs at once:

```bash
cd /home/jason/experiment/CASPO
RUN_TAG=paper512_seed0 GPU_LIST="0 1 2 3 4 5 6 7" WANDB_MODE=offline \
  ./scripts/archive/rho1b/launch_rho1b_all8_standard.sh
```

Or launch a single job by overriding its GPU:

```bash
RUN_TAG=paper512_seed0 GPU=0 WANDB_MODE=offline ./scripts/archive/rho1b/launch_rho1b_grpo.sh
RUN_TAG=paper512_seed0 GPU=1 WANDB_MODE=offline ./scripts/archive/rho1b/launch_rho1b_ppo_critic.sh
RUN_TAG=paper512_seed0 GPU_LIST="2 3" WANDB_MODE=offline ./scripts/archive/rho1b/launch_rho1b_vineppo_ddp2.sh
RUN_TAG=paper512_seed0 GPU=4 WANDB_MODE=offline ./scripts/archive/rho1b/launch_rho1b_caspo.sh
RUN_TAG=paper512_seed0 GPU=5 WANDB_MODE=offline ./scripts/archive/rho1b/launch_rho1b_caspo_delta_prob.sh
RUN_TAG=paper512_seed0 GPU=6 WANDB_MODE=offline ./scripts/archive/rho1b/launch_rho1b_caspo_delta_log_prob.sh
RUN_TAG=paper512_seed0 GPU=7 WANDB_MODE=offline ./scripts/archive/rho1b/launch_rho1b_caspo_frozen_rm.sh
```

All seven scripts use `configs/caspo_rho1b_math.yaml`, vLLM IPC sync,
`save_every=250`, and the current 1000-step standard unless `MAX_STEPS` or
`SAVE_EVERY` is overridden.

The 8-GPU launcher accepts these orchestration env vars:

| Var | Default | Effect |
|---|---|---|
| `AUTO_EVAL_ON_FINISH` | `1` | When a training job exits rc=0, immediately dispatch its eval on the freed GPU instead of waiting for everyone — saves ~7-12 h per suite. Auto-disabled when `WAIT_FOR_CHILDREN=0`. |
| `WATCHDOG` | `1` | Spawn a sidecar polling each method's log every 60 s; warn on `STATUS: STALE` for ≥2 polls (no auto-kill in v1). |
| `WAIT_FOR_CHILDREN` | `1` | Set to `0` to write `${LOGDIR}/launcher_pids.json` and exit without blocking — detachable suite. |
| `WANDB_MODE` | `offline` | Inherited by all per-method launchers. |

## CASPO Advantage Ablations

The direct-value CASPO variant is the normal `caspo` run above. Launch the two
additional CASPO ablations with:

```bash
RUN_TAG=paper512_seed0 GPU_LIST="4 5" WANDB_MODE=offline \
  ./scripts/archive/rho1b/launch_rho1b_caspo_ablations.sh
```

Default ablation outputs:

```text
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_prob_paper512_seed0
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_caspo_logprob_paper512_seed0
```

To include the direct-value run in an ablation-only sweep:

```bash
ADV_VARIANTS="value prob logprob" GPU_LIST="4 5 6" \
  ./scripts/archive/rho1b/launch_rho1b_caspo_ablations.sh
```

Frozen-RM CASPO keeps IPVRM prefix scoring but disables online value-model
updates:

```bash
RUN_TAG=paper512_seed0 GPU_LIST="4" WANDB_MODE=offline \
  ./scripts/archive/rho1b/launch_rho1b_caspo_frozen_rm.sh
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
  ./scripts/archive/rho1b/launch_rho1b_vineppo_ddp2.sh
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
  ./scripts/archive/rho1b/launch_rho1b_vineppo_ddp2.sh
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

# Round 2 trainer knobs (apply to both 1-GPU and DDP-2 launchers):
CASPO_REWARD_WORKERS=4   # ProcessPoolExecutor for SymPy verifier (default 4)
CASPO_COMPILE=false      # torch.compile — leave false (see Round 2 caveats)
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
GPU_LIST="2 3" WANDB_MODE=disabled ./scripts/archive/rho1b/launch_rho1b_vineppo_ddp2.sh
```

## Evaluation

Do cheap sample evals at saved checkpoints and full eval only at the end. The
training loop does not run eval in-process; eval is launched from saved
checkpoints with vLLM. Keep the sample cadence aligned with intermediate
checkpoints: `step_250`, `step_500`, and `step_750`. At `final`, run the full
benchmark suite instead of a separate sample eval because the full suite already
includes MATH-500.

Standard seven-method sample eval:

```bash
RUN_TAG=paper512_seed0 CKPT_SUBDIR=step_250 \
EVAL_GPU_LIST="0 1 2 3 4 5 6" ./scripts/archive/rho1b/launch_eval_rho1b_sample_all8.sh
```

This defaults to `math500`, `EVAL_LIMIT=100`, and `EVAL_K=8`. On Rho-1B, the
old full MATH-500 k=16 eval took about 1-2 minutes per model including vLLM
startup; the 100-problem k=8 sample is expected to be well under that. If all
seven methods are evaluated in parallel, sample eval wall-clock should usually
be a couple of minutes, but it needs free eval GPUs.

Standard seven-method full final eval:

```bash
RUN_TAG=paper512_seed0 EVAL_GPU_LIST="0 1 2 3 4 5 6" \
  ./scripts/archive/rho1b/launch_eval_rho1b_final_all8.sh
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
- `EVAL_VLLM_GPU_MEMORY_UTILIZATION`: defaults to `0.92` because eval does
  not share the GPU with a trainer; combined with `kv_cache_dtype="fp8"`
  this lets the engine hold ~2× more concurrent KV blocks for the K=16
  multi-sample fanout.

## Latest Paper-Faithful Speed Probe

Hardware: one H100 80GB per method. Config: Rho-1B MATH, 512 responses per PPO
outer step, vLLM IPC sync, `save_every=0`, `max_steps=3`.

### Pre-optimization defaults (`mb=1, accum=64, ckpt=true, vllm_util=0.45`)

| Method | Mean step time | Rollout | Value/MC phase | Policy phase | Notes |
|---|---:|---:|---:|---:|---|
| PPO | ~90s | ~4s | 0s | ~74s | Terminal reward PPO |
| GRPO | ~92s | ~4s | 0s | ~76s | Group-relative terminal reward |
| CASPO | ~141s | ~4s | ~55s | ~69s | IPVRM value forward + online update |
| VinePPO K=9 | ~237s | ~4s | ~152s | ~69s | MC prefix rollouts dominate |

### Post-optimization (`mb=8, accum=8, ckpt=false, vllm_util=0.30`, IPC sync)

April 2026 sweep across 18 (mb, accum, ckpt, util) configurations. The
Pareto-optimal point matches `mb × accum = 64` (same global PPO minibatch as
the paper) but redistributes to `mb=8 × accum=8` and disables gradient
checkpointing (zero measurable cost on Rho-1B because activation recompute is
already free). vLLM utilization drops to 0.30 because rollout is not the
bottleneck at this batch shape.

| Method | Mean step time | Speedup | Peak GPU mem | Notes |
|---|---:|---:|---:|---|
| PPO | ~50s | 1.80x | ~74 GB | 10-step long smoke, mem flat |
| GRPO | ~48s | 1.92x | ~76 GB | 10-step long smoke, mem flat |
| CASPO (online RM) | ~75s | 1.88x | ~77 GB | 10 steps at u=0.30, t_sync stable after step 5 |
| CASPO frozen RM | ~63s | 2.24x | ~64 GB | No value Adam states, ~13 GB headroom |
| CASPO delta-prob ablation | ~75s | n/a | ~77 GB | Same as standard CASPO |
| CASPO delta-log-prob ablation | ~75s | n/a | ~77 GB | Same as standard CASPO |
| VinePPO K=9 (1-GPU) | ~191s | 1.24x | ~73 GB | Gated by K=9 MC rollouts (constant-cost) |
| VinePPO K=9 (DDP-2) | ~115s | 2.06x | ~70 GB | `mb=4, accum=8, logprob_micro=16` |

Approximate 1000-step ETAs (post-optimization):

- PPO: ~14 hours.
- GRPO: ~13 hours.
- CASPO: ~21 hours.
- CASPO frozen RM: ~18 hours.
- VinePPO K=9 1-GPU: ~53 hours.
- VinePPO K=9 DDP-2: ~32 hours.

If all four primary methods run in parallel on H100s, wall-clock is gated by
CASPO at ~21 hours (single-GPU layout) or VinePPO DDP-2 at ~32 hours, both
plus checkpoint/eval overhead.

### Why these defaults

- `mb=8 × accum=8 = 64` preserves the global PPO minibatch (same as paper).
- The trajectory is mathematically identical to `mb=1, accum=64` modulo
  ~1e-3 bf16 reduction-order noise — well below seed-level variance.
- `gradient_checkpointing=false` has no measurable step-time cost on Rho-1B
  but frees ~10 GB of activation memory, which lets `mb=8` fit alongside vLLM.
- `vllm_util=0.30` leaves ~3-4 GB trainer margin even for the CASPO online-RM
  path (which loads policy + ref + value + value Adam states all on one GPU).
- IPC weight sync stabilizes to <0.5 s/step after a one-time spike at
  steps 3-4 (~24 s + 9 s); total amortized overhead at 1000 steps is ~5 min.

### Override / revert

Each knob is overridable from the launcher CLI for safety:

```bash
MICRO_BATCH_SIZE=1 GRAD_ACCUM_STEPS=64 USE_GRADIENT_CHECKPOINTING=true \
CASPO_VLLM_GPU_MEMORY_UTILIZATION=0.45 \
RUN_TAG=conservative ./scripts/archive/rho1b/launch_rho1b_caspo.sh
```

### Round 2 optimizations (Apr 2026)

A second optimization pass landed on top of the Pareto sweep. Each item is
a pure speed-up — no effective-learning change beyond ~1e-3 bf16 noise.

| Optimization | Mechanism | Gain |
|---|---|---|
| FlashAttention 3 (HF + vLLM) | `attn_implementation: flash_attention_3` in YAML; FA3 Hopper backend installed in `scalable` env | Faster attention forward+backward on H100 |
| Reuse epoch-0 forward as `old_logprobs` | Skip dedicated `_rescore_old_logprobs` pass; capture `new_logprobs.detach()` from the first PPO epoch | One full forward per step eliminated; `t_old` drops from ~6 s to 0 s |
| Share `ref_logprobs` between trainer KL and value model | `value_model.forward(ref_logprobs=...)` accepts a precomputed tensor | One full ref forward eliminated for CASPO; ~10 GB activation peak reduction |
| Drop `.float()` upcast inside `cross_entropy` | bf16 cross_entropy with internal fp32 reduction is numerically equivalent | ~2 GB activation per microbatch saved; CE kernel ~5-10% faster |
| `fused=True` AdamW | Single fused CUDA kernel instead of foreach + 3 launches | Bit-identical, 1-2% step time |
| Stack scalars in microbatch loop | Replace per-microbatch `.item()` syncs with on-device accumulator + single `.tolist()` | Removes CPU-GPU sync per microbatch |
| `torch.compile` (opt-in via `cfg.compile=true`) | `mode="default", dynamic=True` on policy and `value_model.phi` | **CURRENTLY UNUSABLE** — see compile incompatibility note below |
| Parallel SymPy reward verifier | `ProcessPoolExecutor` over chunked predictions when `cfg.reward_workers > 1`, gated on batch size | Hides up to 2-10 s/step on hard problem batches |
| Persistent ground-truth cache | Per-trainer `OrderedDict` keyed on raw GT string with FIFO eviction at `cfg.gt_cache_max_size` | Eliminates per-call GT renormalization across the ~7.5K-prompt cycle |
| Cached tokenized dataset | First call writes `${HF_HOME}/caspo_dataset_cache/<hash>.pt`; subsequent processes load directly | Faster cold-start; `CASPO_DATASET_CACHE_DISABLE=1` opts out |
| Shuffled training cycle | `random.Random(cfg.seed + epoch).shuffle(...)` per cycle | Same gradients per step, different order — quality lever, not speed |
| Pre-tokenize prompt cache in vLLM rollout | LRU cache (max=1024) keyed by prompt string | Removes per-step Python tokenize loop |
| Eval `kv_cache_dtype="fp8"` + `gpu_memory_utilization=0.92` | Halves KV bytes; doubles concurrent KV blocks for K=16 fanout | ~30-50% faster eval |
| Eval `enable_chunked_prefill=True` + `max_num_batched_tokens=8192` | Interleaves prefills with decodes when many K-fanout requests share prompt prefixes | Stacks with above |
| Eval `top_k=50` for SamplingParams | Avoids full-vocab sort under top_p sampling | ~2-3% eval |
| Default `wandb_mode=offline` in single-method launchers | Avoids intermittent network stalls during training | Risk reduction |
| `wait -n` dispatcher in 8-GPU launcher | When a method finishes, immediately dispatch its eval on the freed GPU instead of waiting for everyone | ~7-12 h saved per full suite |
| Health-check sidecar watchdog | Polls each method's log every 60 s, warns on `STATUS: STALE` for 2 consecutive polls (no auto-kill in v1) | Avoids overnight idle on hangs |
| `WAIT_FOR_CHILDREN=0` detached mode | Writes `launcher_pids.json` and exits without blocking | Detachable suite |

Per-method post-Round-2 step times (single H100, validated by 6-step smokes
on `mb=8/accum=8/ckpt=false/vllm_util=0.30` plus all Round 2 patches):

| Method | Round 1 step time | Round 2 step time | Round 1+2 vs original |
|---|---:|---:|---:|
| PPO | ~50s | **~43s** | 90s → 43s (~2.1×) |
| GRPO | ~48s | **~43s** | 92s → 43s (~2.1×) |
| CASPO (online RM) | ~75s | **~60s** | 141s → 60s (~2.4×) |
| caspo-frozen-rm | ~63s | (untested at R2; should track CASPO ratio) | 141s → ~50s |
| VinePPO K=9 (1-GPU) | ~191s | **per-MC throughput unchanged**; per-step varies with `steps/r` (191-308s observed across batches) | net ~unchanged at fixed `steps/r` |

VinePPO's step time scales as `steps_per_response × K=9` because each
non-terminal step boundary triggers K MC continuations. Per-MC sampling
throughput on Round 2 measured at 0.0066 s/sample, identical to Round 1.
Variance across smokes (steps/r 6.6-10.2) is a function of the policy's
response-length distribution at SFT init, not an optimization regression.
GRPO/PPO are unaffected (`steps/r=1` always); CASPO's per-step value is one
V_φ forward per response and does not multiply with `steps/r`.

Notes:
- Rollout's `enable_chunked_prefill` is OFF by default (verified that VinePPO's K=9 MC pattern regressed by ~70% with chunked-prefill on, since mixed prefill-decode CUDA graphs penalize many short prefixes). Eval keeps it on.
- `cfg.compile=True` is wired but currently unusable on this stack. Validated empirically (Apr 2026): with `mode="reduce-overhead"` the trainer crashes with `Error: accessing tensor output of CUDAGraphs that has been overwritten` because the trainer reuses output tensors across optimizer micro-steps. With `mode="default"` (no CUDA graphs) the policy forward hits two HF/transformers limitations at once: (1) graph break inside `_get_unpad_data` from `seqlens_in_batch.max().item()` on every call, and (2) per-layer recompiles because `module.layer_idx` is a static integer that differs across the 22 transformer layers — dynamo exhausts the recompile budget at layer 8 and falls back to eager. Until HF wires a compile-friendly attention path or we patch around `_get_unpad_data`, leave `cfg.compile=false`.
- `EnsureLR / no schedule changes`: nothing in this round affects learning rate, KL coef, or PPO clip; comparison with the original VinePPO setup remains apples-to-apples up to bf16 reduction noise.

### FA3 install

Installed via:

```bash
/opt/conda/envs/scalable/bin/pip install --no-build-isolation \
  "flash-attn-3 @ git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=hopper"
```

Verify:

```bash
/opt/conda/envs/scalable/bin/python -c "
import flash_attn_interface
import torch
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained('realtreetune/rho-1b-sft-MATH',
    torch_dtype=torch.bfloat16, attn_implementation='flash_attention_3').cuda()
ids = torch.tensor([[1,2,3,4,5,6,7,8]], device='cuda')
print(m(ids).logits.shape)
"
```

If FA3 is unavailable at runtime, HF transformers falls back to FA2 with a
warning — non-fatal.

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

## fp32 Master Weights + Memory Fit (Apr 28, 2026)

VinePPO upstream uses DeepSpeed BF16_Optimizer which keeps **fp32
master weights + fp32 m/v** for AdamW even when params compute in
bf16. Without this, AdamW updates at lr=1e-6 round to zero in bf16's
7-bit mantissa and the policy drifts back toward init (silent
training failure). To match, CASPO now defaults `fp32_master_weights:
true` (in `caspo/config.py`).

Two FSDP+vLLM infrastructure changes were required to make fp32
master fit in 4×H100 80 GB:

1. **CPU-load + `sync_module_states=True`**: skip per-rank
   `.to(device)` for policy/ref/value/critic when FSDP will wrap
   them. FSDP places only the sharded slice on GPU and broadcasts
   from rank 0's CPU copy. Avoids ~56 GiB of transient peak (full
   policy fp32 + ref bf16 + value bf16 on each rank's GPU before
   sharding).

2. **AdamW CPU-offload during sync**: temporarily move policy/critic/
   value AdamW state (m, v) to pinned host memory before each
   `_sync_vllm_weights` call, restore after. FSDP's `summon_full_
   params` materializes ~28 GiB unsharded fp32 on each rank during
   sync; without freeing the 14 GiB AdamW slot first, sync OOMs at
   the step 1→2 boundary on 7B CASPO. ~400-500 ms / sync at PCIe
   Gen5 (~10% sync-step slowdown). Default off; on for 7B configs.

3. **Single-RPC IPC sync + per-param bf16 cast** (replaced earlier
   "streaming" version): cast each param from fp32 → bf16 on the
   source side (vLLM expects bf16) and submit ALL params in one
   `update_weights` RPC. We initially tried chunking the RPC into
   8-param batches, but vLLM's `_reload_weights_in_place` calls
   `initialize_layerwise_reload` + `load_weights` + `finalize_
   layerwise_reload` PER RPC when `is_checkpoint_format=True`, so
   each chunk re-initialized the layerwise-reload state and only a
   small fraction of params actually persisted. Symptom: thousands
   of "<Module>: Failed to load weights" warnings per sync, and
   step 2+ rollouts producing garbage (no `\boxed{...}`), reward=0
   instant collapse. Reverted to single-RPC in commit `b34dd80`
   (2026-04-28). For 7B fp32 master memory pressure, AdamW
   CPU-offload (item 2) carries the load instead.

4. **Preallocate AdamW state at trainer init**: run a one-time
   zero-grad dummy `optim.step()` after each optimizer is built, so
   the lazy AdamW state (`m`, `v`) is allocated BEFORE vLLM grabs
   its KV cache. Avoids spurious step-3 OOMs on tight 1B configs
   from allocator fragmentation racing vLLM's resident weights.
   ~50-200 ms one-time at init.

5. **PPO+critic uniform R in critic train**: dropped per-microbatch
   `R_eff` trim in `_ppo_critic_train_critic` so all microbatches
   use the full `[mb, P+R]` shape. Variable R_eff was minting fresh
   allocator segments per shape that `empty_cache()` released back
   to the driver each step → linear `t_value` growth. Trades ~5-15%
   extra padding FLOPs for constant-time per-step.

### Verified configurations

**1B Rho-1B-SFT-MATH (single GPU, no FSDP, fp32 master via autocast)**

| Method | mb / accum | vllm_util | Multi-step | Mean t_step |
|---|---|---:|---|---:|
| GRPO            | 4 / 16 | 0.30 | ✓ | ~50 s |
| CASPO frozen-RM | 4 / 16 | 0.30 | ✓ | ~55 s |
| CASPO online    | 4 / 16 | 0.30 | ✓ | ~75 s |
| PPO+critic      | 4 / 16 | 0.30 | ✓ | ~100 s |

8-step PPO+critic probe with per-step `torch.cuda.memory_stats()`
shows `mem_alloc=21.7 GiB`, `mem_reserved=22.0 GiB`, `n_alloc=1818`
flat across all steps — no leak, no fragmentation accumulation. Step
time oscillates 75-121 s around a ~102 s mean (rollout-content
variance), with the warmup-tail contributing 4-step samples that
look monotonic but aren't.

`scripts/_launch_rho1b_one_gpu.sh` defaults: `mb=4, accum=16` (was
`mb=8, accum=8` before fp32 master). The mb drop is required for
PPO+critic — at mb=8, AdamW state init triggers a ~8.8 GiB burst
allocation in step 3 that races vLLM's resident KV cache and OOMs.
Preallocation + GAE microbatching + uniform R together close the
gap at mb=4.

**7B DeepSeekMath-7B-MATH (4-GPU FSDP, hybrid_shard, colocated vLLM)**

| Method | mb / accum | vllm_util | step 1 | step 2 | step 3 | Pattern |
|---|---|---:|---:|---:|---:|---|
| GRPO             | 2 / 8  | 0.30 | 42 s | 38 s | 33 s | flat |
| PPO+critic       | 2 / 8  | 0.20 | 65 s | 75 s | 69 s | flat |
| CASPO frozen-RM  | 2 / 8  | 0.20 | 44 s | 41 s | 36 s | flat |
| CASPO online     | 2 / 8  | 0.20 | 57 s | 68 s | 66 s | flat |

Sync overhead 4-5 s/step (AdamW CPU-offload + streaming IPC). All
within 80 GB H100 budget at peak ~66 GB / rank.

7B configs inherit `offload_optim_during_sync: true` from
`configs/caspo_deepseekmath7b_math.yaml`. 1B doesn't need it.

### Config knobs (defaults shown)

```yaml
fp32_master_weights: true        # fp32 AdamW state for noise floor
fsdp_reduce_dtype: float32       # fp32 grad accumulator
preallocate_optim_state: true    # eager AdamW state init
save_optimizer_state: true       # save m/v alongside model.safetensors
offload_optim_during_sync: false # 7B configs override to true
vllm_extra_stop_strings: ["\n\n\nProblem:"]   # paper-faithful stop
max_sequence_len: 2048           # post-rollout unfinished penalty
advantage_clip: 3.0              # ±3σ clip post-whitening (outlier safety)
kl_coef: 1.0e-2                  # 100× upstream's 1e-4 — see stabilization below
online_value_lr: 1.0e-6          # CASPO V_φ online update LR (full-FT)
critic_lr: 1.0e-6                # PPO+critic V_ψ LR (matches policy LR)
```

GPU 0 reserved for teammate's evaluation experiment — all CASPO runs
default to GPUs 1-7 (override with `GPU=` or `GPU_LIST=`).

### 4-method stabilization recipe (Apr 28, 2026)

A bisection on Apr 28 surfaced **two compounding bugs** that made the
4-method 1B parallel run (GRPO + PPO+Critic + CASPO + CASPO Δp)
collapse to reward=0 within ~20 outer steps. Documented here so
future-you doesn't recreate the trap.

**Bug 1 — Streaming IPC sync corrupted vLLM** (commit `a28099d`,
reverted in `b34dd80`). Detail above (Single-RPC IPC sync). Headline
symptom: instant collapse — step 2 rollouts produce garbage, reward
drops to 0 across all 4 methods because vLLM's policy is half-loaded
after the first weight sync.

**Bug 2 — Online-value drift collapsed CASPO/PPO+Critic** (multiple
commits `f149966 → 21cbd41`). After fixing Bug 1, CASPO still
collapsed by step 17-20 with KL → 100s and `steps/r 8 → 45` (mode
collapse to long degenerate responses). Root cause: VinePPO upstream
absorbs V_φ noise via K=9 MC continuations per step boundary; we
have a single learned V_φ forward (no MC averaging), so the policy
update sees noisier advantages and drifts faster than the KL
penalty can pull it back at upstream's `kl_coef=1e-4`. Fix stack:

| # | Knob | Was | Now | Why |
|---|---|---|---|---|
| F1 | `advantage_clip` | 0 (Patch H removed) | 3.0 | outlier safety net |
| F2 | `kl_coef` | 1e-4 (upstream) | 1e-2 | 100× anchor against learned-value drift |
| F3 | `online_value_lr` | 1e-6 → 1e-7 → **1e-6** | 1e-6 | dropped during weak-KL phase, restored when KL anchor took over |
| F4 | `critic_lr` | 1e-6 → 1e-7 → **1e-6** | 1e-6 | same path as F3 |
| F5 | `launch_rho1b_ppo_critic.sh` | hardcoded `kl_coef=1e-4` override | inherits YAML | silent launcher override was bypassing F2 for PPO+Critic only |

Verification: `fixed_v6` (F1+F2+F3=1e-7+F4=1e-7+F5) reached step 100
with reward stable ~0.20 and KL bounded < 2 across all 4 methods.
Held-out eval at step_100 vs SFT init on MATH-500 (k=8, n=100) shows
GRPO `avg@k 0.199 → 0.236` (+19%) — first clean signal that learning
actually works post-stabilization.

**Detection signals** for future debugging:
- vLLM "Failed to load weights" warnings count: healthy ~hundreds at
  startup, broken ~tens of thousands per sync (Bug 1 signature).
- `steps/r` (mean response length / step count) rising sharply over
  ~5 outer steps from init values to ~max-cap → policy is mode-
  collapsing to long degenerate responses (Bug 2 signature).
- `reward=0.000, pass@G=0.000` for >5 consecutive steps from any
  early step (≤ step 20) → kill and bisect immediately, don't wait.

### V_φ retrain after verifier + BOS changes (Apr 28, 2026)

A **third** issue surfaced after fixing Bugs 1 & 2: CASPO's `v_acc` was
running at 0.32 at step 1 of RL, vs the offline-trained V_φ baseline of
~0.96 (from `value_train_log.jsonl`'s `acc_at_last`). Investigation
showed V_φ is **mis-calibrated to runtime distribution**, not lost or
broken:

* **Stale outcomes**: V_φ was trained on outcomes from the pre-Minerva
  verifier (Apr 25). Adding the Minerva-style extractor cascade in
  `caspo/reward/math_verifier.py` flipped many rolled-out responses
  from "wrong" to "right". V_φ trained against the old labels then
  signs-disagrees with the new labels.
* **Stale prompt tokenization**: V_φ's training data (`value_data.pt`)
  was collected before Patch A added explicit BOS prepending in
  `VLLMRolloutEngine`. Inspection: `prompt_ids[i, first_nonpad] == 518`
  ('[' from `[MATH_TASK]`), no BOS token id 1 anywhere. Runtime now
  prepends BOS, shifting per-token log-ratios.

**Fix**: re-collect + retrain V_φ with current code (no patches needed
since `collect_value_data.py` uses the live `VLLMRolloutEngine` and
`MathRewardFn` — both Patch A + Minerva are inherited automatically):

```bash
# Default: keep all G rollouts per mixed-outcome prompt (~7-8k rows,
#          ~25% positive rate)
GPU_LIST="4 5 6 7" bash scripts/retrain_value_rho1b_4gpu.sh

# IPVRM-faithful 1-pair-per-prompt (PAPER_PAIRING=true, ~2.4k rows, 50/50)
PAPER_PAIRING=true GPU_LIST="4 5 6 7" \
  bash scripts/retrain_value_rho1b_4gpu.sh

# IPVRM generalization: min(n_pos, n_neg) DISJOINT pairs per prompt
# (PAPER_PAIRING_MULTI=true, ~4-6k rows, 50/50, no rollout reused)
PAPER_PAIRING_MULTI=true GPU_LIST="4 5 6 7" \
  bash scripts/retrain_value_rho1b_4gpu.sh
```

The orchestrator does:
1. **4-shard collect** in parallel (one shard per GPU; merged after) —
   uses `--shard i/N` in `collect_value_data.py` (interleaved slicing).
2. **Merge** the 4 `.pt` files via `scripts/merge_value_data_shards.py`
   (left-pads prompts and right-pads responses to global max-len before
   concatenating along dim 0).
3. **FSDP=4 train_value** with `value_micro_batch_size=16,
   value_grad_accum_steps=1` (same effective batch as paper-faithful
   `mb=1, accum=16` but ~3.7× faster wall-clock at 1B because kernel
   launches amortize). Honors `cfg.value_save_every` (env
   `VALUE_SAVE_EVERY`) to save per-epoch checkpoints into `step_<N>/`
   subdirs for AUC-trajectory analysis.
4. **Smoke-validate** new V_φ on a 1-step CASPO rollout; passes if
   `v_acc >= 0.7`.

End-to-end ~15-30 min on 4 H100s (multi-pair / paper-pairing) or
~30-45 min (default).

**LR caveat — paper's 5e-7 doesn't learn full-FT at 1B**. The IPVRM
paper uses 5e-7 with LoRA adapters; full fine-tuning the base SFT
model needs more gradient signal. An LR=5e-7 retrain on the multi-pair
dataset early-stopped at step 300 with val_loss = 5.0058 = initial
margin (V_φ never moved off init). Bump to **`VALUE_LR=5e-6`** (10×)
for full-FT runs. Pass via env to the orchestrator. Worked example:
```bash
VALUE_LR=5e-6 PAPER_PAIRING_MULTI=true \
  GPU_LIST="4 5 6 7" bash scripts/retrain_value_rho1b_4gpu.sh
```

**Output paths** (current Apr 28 retrain set):
* `caspo_rho1b_math_v6_multi/value_final/` — **current live RM** (YAML default), multi-pair 50/50 collect + LR=5e-6 + 5 epochs. **ROC-AUC = 0.633** on prompt-level held-out val.
* `caspo_rho1b_math_v2/value_final/` — earlier same-day retrain (keep-all-rollouts, LR=5e-7). AUC = 0.517 on the same val set. Still used by live fixed_v6/v7 RL runs to keep their experimental condition consistent.
* `caspo_rho1b_math_v5_multi_lr5e7_failed/` — multi-pair retrain at paper's 5e-7 (failed — kept for diagnostic).

**`v_acc` is class-imbalance-sensitive**: at runtime the rollout
class balance is ~80% incorrect / ~20% correct, so the *naive
predict-incorrect* baseline already scores `v_acc ≈ 0.80`. A V_φ at
`v_acc = 0.77` is therefore *below* the naive baseline by ~3pp; the
useful signal lives in `ROC-AUC` (~0.55-0.60) and `score-conditioned
margins`, not in raw accuracy. Run AUC eval over the trainer's
held-out val split via
`python scripts/eval_vphi_auc.py --vphi <path> --label <tag>`.

To deploy in a CASPO RL run, set
`PREFIX_VALUE_PATH=/mnt/nvme_tmp4/jason_caspo/caspo_rho1b_math_v6_multi/value_final`
(YAML default; explicit env only needed to override). The 1B per-GPU
launcher in `scripts/_launch_rho1b_one_gpu.sh` accepts this env
override and forwards it as `--override prefix_value_path=...`.

### Teammate handoff: CASPO ablation runs (Rho-1B, single GPU each)

To distribute CASPO ablations across collaborators' machines, each
ablation is one self-contained launcher that takes one H100 and runs
1000 outer steps in ~16 hours. Each writes ~150-285 GB of checkpoints
across 10 saves (`save_every=100`).

**Required inputs** (one-time copy):
- SFT model: HuggingFace `realtreetune/rho-1b-sft-MATH` (~2 GB; auto-downloaded)
- V_φ checkpoint: rsync from
  `/mnt/nvme_tmp4/jason_caspo/caspo_rho1b_math_v2/value_final/` (~4 GB).
  All CASPO methods use the same V_φ; only `caspo_frozen_rm` keeps it
  fixed during RL, the others fine-tune online.

**Four CASPO ablations** (each its own thin launcher in `scripts/`).
The default `caspo_prob` at `online_value_lr=1e-6` is owned by the host
machine; teammates run the four below as ablations:

| Variant | Launcher | Ablation knob |
|---|---|---|
| CASPO frozen-RM | `launch_rho1b_caspo_frozen_rm.sh` | `update_value_during_policy=false` |
| CASPO logprob | `launch_rho1b_caspo_delta_log_prob.sh` | `caspo_advantage_transform=logprob` |
| CASPO Δp (LR=1e-5) | `launch_rho1b_caspo_delta_prob_lr1e5.sh` | `caspo_advantage_transform=prob`, `online_value_lr=1e-5` |
| CASPO Δp (LR=1e-4) | `launch_rho1b_caspo_delta_prob_lr1e4.sh` | as above + `online_value_lr=1e-4` (paper-faithful for LoRA; ablation at full-FT — was unstable pre-Apr-28-stack, may now be safe with strong KL anchor) |

**Standard invocation** (override any defaults via env):

```bash
# Required env: ROOT (output drive root), GPU (single id)
# Optional env: PREFIX_VALUE_PATH (default: YAML's path = v2 retrained V_φ)
#               ONLINE_VALUE_LR (default: YAML's value_lr = 1e-6 except 1e5/1e4 launchers)
#               KL_COEF (default: YAML's kl_coef = 1e-2)
#               RUN_TAG (suffix appended to output dir)

# Example: friend's box, GPU 0, run tag "ablation_seed0"
ROOT=/path/to/output GPU=0 RUN_TAG=ablation_seed0 \
  bash scripts/archive/rho1b/launch_rho1b_caspo_frozen_rm.sh

# If V_φ checkpoint isn't at the YAML default path on friend's box:
ROOT=/path/to/output GPU=0 RUN_TAG=ablation_seed0 \
PREFIX_VALUE_PATH=/path/to/local/value_final \
  bash scripts/archive/rho1b/launch_rho1b_caspo_delta_log_prob.sh
```

**Required pre-flight**: the YAML's `prefix_value_path` defaults to a
local NVMe path (`/mnt/nvme_tmp4/jason_caspo/caspo_rho1b_math_v2/value_final`)
that won't exist on a teammate's machine. Either:
1. Copy V_φ to the same absolute path on their box, OR
2. Pass `PREFIX_VALUE_PATH=<their_path>` via env (recommended).

Per-method disk footprint at `save_every=100` × 1000 steps × ckpts
(model.safetensors + AdamW optimizer.pt for policy + value):

| Variant | Per-ckpt | × 10 ckpts |
|---|---|---|
| frozen-RM | ~17 GB | ~170 GB (no value optim → smaller) |
| logprob / Δp (any LR) | ~28 GB | ~285 GB (full V_φ + AdamW) |

Each variant takes ~16 hours on a single H100 80GB.

### 4-method parallel launcher (Rho-1B, four NVMe drives)

`scripts/archive/rho1b/launch_rho1b_4method_split.sh` runs GRPO + PPO+Critic +
CASPO + CASPO Δp concurrently on GPUs 4-7 (or the user's
`GPU_LIST`). Each method writes its checkpoints to its own NVMe
drive so 10 ckpts/run × 4 runs (save_every=100 × 1000 steps) all
fit without disk contention:

| GPU | Method | Drive | ckpts | Footprint |
|---|---|---|---|---|
| 4 | GRPO | `/mnt/nvme_tmp2` | 10 | ~153 GB |
| 5 | PPO+Critic | `/mnt/nvme_tmp4` | 10 | ~270 GB |
| 6 | CASPO | `/mnt/nvme_tmp3` | 10 | ~285 GB |
| 7 | CASPO Δp | `/mnt/nvme_tmp5` | 10 | ~285 GB |

Pre-launch validates all 4 drives mounted + writable. Live status
via `RUN_TAG=<tag> watch -n 10 bash scripts/watch_4method.sh` —
shows per-method step / t_step / t_avg10 / ETA / mem_alloc.

## 7B Infrastructure (Apr 2026)

Full FSDP + vLLM-IPC weight-sync stack for DeepSeekMath-7B-MATH. Validated
end-to-end on all 5 methods. Per-rank manual bash launch (matches the
rho-1B DDP-2 pattern); torchrun's `--standalone` rdzv conflicts with
vLLM's EngineCore distributed init.

### Layout

| Method | World size | Default util | Steady step time |
|---|---:|---:|---:|
| PPO | 4 (FSDP) | 0.30 | ~30 s |
| GRPO | 4 (FSDP) | 0.30 | ~30 s |
| CASPO frozen-RM | 4 (FSDP) | 0.30 | ~35 s |
| CASPO online | 4 (FSDP) | **0.20** | ~45 s |
| VinePPO K=9 | 8 (FSDP) | 0.30 | ~570 s |
| VinePPO K=9 (u=0.45) | 8 (FSDP) | 0.45 | ~310 s (step 2) — ~33% per-MC-sample throughput gain |

CASPO online needs the lower vLLM util because it carries a full
trainable phi (params + Adam fp32 m+v ≈ 70 GB / 4 ranks = ~17 GB
sharded) on top of the policy. PPO/GRPO/frozen-RM stay at u=0.30.

VinePPO 8-GPU is dominated by `t_value` (~315 s) — K=9 MC continuations
at 7B is the heaviest workload in the stack.

### Pareto sweep (4-GPU FSDP, GRPO smoke)

| Config | Step time | Notes |
|---|---:|---|
| mb=1, accum=16, ckpt=true, util=0.30 | ~40 s | Old default |
| **mb=2, accum=8, ckpt=true, util=0.30** | **~30 s** | Winning default |
| mb=4, accum=4, ckpt=true, util=0.30 | hangs | Silent OOM on rank 0 |
| mb=4, accum=4, ckpt=true, util=0.30 + fsdp_cpu_offload=true | ~76 s | CPU↔GPU dominates |
| mb=2, accum=8, ckpt=false, util=0.30 | OOM at vLLM init | Activations claim trainer headroom; vLLM can't claim 24 GB for KV |
| mb=2, accum=8, ckpt=true, util=0.40 | ~33 s | vLLM not the bottleneck above ~0.30 |

mb=2 is 25-27% faster than mb=1 with no memory or stability issues.
Global PPO minibatch stays 64 (FSDP=4 × mb=2 × accum=8).

### Critical fixes for FSDP + vLLM colocation

1. `VLLM_WORKER_MULTIPROC_METHOD=spawn` (perf_env.sh): vLLM's default
   `fork` makes EngineCore inherit the trainer's torch.distributed
   state, causing the child's TP=1 init to collide and hang on
   TCPStore client validation.
2. `VLLM_HOST_IP=127.0.0.1` (set per-rank in `caspo/rollout/vllm_engine.py`):
   vLLM's get_ip() falls back to the cluster-external interface IP;
   force loopback so each rank's EngineCore can bind a private TCPStore.
3. Manual per-rank bash spawn (no `torch.distributed.run --standalone`):
   torchrun's rdzv backend can't be cleanly inherited by vLLM's
   EngineCore subprocess.
4. FSDP→vLLM IPC sync (`caspo/trainer/caspo_trainer.py:_sync_vllm_weights_fsdp`):
   `summon_full_params(writeback=False)` collectively, then push
   directly through `sampler.sync_weights_from_model` while the
   context is alive; strip `_fsdp_wrapped_module.` prefix from the
   FSDP-injected parameter names so vLLM's HF loader matches them.

### Launchers

```bash
# Canonical 7-method 7B suite (4-GPU FSDP colocated for everything
# except VinePPO, which uses 8-GPU disagg). Same RUN_TAG ties them
# into one logical seed:
GPU_LIST="0 1 2 3" RUN_TAG=paper7b_seed0 ./scripts/launch_7b_grpo.sh
GPU_LIST="0 1 2 3" RUN_TAG=paper7b_seed0 ./scripts/launch_7b_ppo_critic.sh
GPU_LIST="0 1 2 3" RUN_TAG=paper7b_seed0 ./scripts/launch_7b_caspo.sh
GPU_LIST="0 1 2 3" RUN_TAG=paper7b_seed0 ./scripts/launch_7b_caspo_frozen_rm.sh
GPU_LIST="0 1 2 3" RUN_TAG=paper7b_seed0 ./scripts/launch_7b_caspo_delta_prob.sh
GPU_LIST="0 1 2 3" RUN_TAG=paper7b_seed0 ./scripts/launch_7b_caspo_delta_log_prob.sh

# 8-GPU disaggregated topology for VinePPO (FSDP=4 trainer + vLLM TP=4
# dedicated). NCCL+packed weight sync at 0.3 s/step. Production path
# for VinePPO 7B; see docs/disaggregated_topology_plan.md:
TRAIN_GPU_LIST="0 1 2 3" ROLLOUT_GPU_LIST="4 5 6 7" \
    RUN_TAG=paper7b_seed0 ./scripts/launch_7b_vineppo_disagg.sh

# Optional alternates:
#   launch_7b_ppo.sh             — legacy critic-free PPO (sequence-level
#                                   advantage; not in the canonical suite)
#   launch_7b_ppo_critic_disagg.sh — 8-GPU disagg PPO+critic (matches
#                                     VinePPO upstream topology)
#   launch_7b_vineppo.sh          — legacy 8-GPU rank-local TP=1 VinePPO
#                                   (slower; kept for ablation)
#   launch_7b_vineppo_tp8.sh      — colocated TP=8 VinePPO
```

Env knobs: `CASPO_VLLM_GPU_MEMORY_UTILIZATION`, `CASPO_MICRO_BATCH_SIZE`,
`CASPO_GRAD_ACCUM_STEPS`, `CASPO_USE_GRADIENT_CHECKPOINTING`,
`CASPO_FSDP_CPU_OFFLOAD`, `CASPO_REWARD_WORKERS`, `MAX_STEPS`,
`SAVE_EVERY`, `WANDB_MODE`. Defaults computed from `world_size` so
the global PPO minibatch stays 64 across configurations.

### Per-method step times — full canonical suite (measured 2026-04-27)

The canonical comparison sweep is **7 methods at each model size**:
GRPO, PPO+critic, VinePPO, CASPO, CASPO-frozen, CASPO-delta-p,
CASPO-delta-logp. Step times are paper-faithful (`global=64`
minibatch, bf16 mixed precision, fp8 KV cache, IPC weight sync).

#### Rho-1B-MATH (1 H100 / method, 2 H100s for VinePPO DDP-2)

`mb=8, accum=8` (post-Pareto-sweep optimum), `vllm_util=0.30`,
`ckpt=false`, `max_steps=1000`.

| Method | Script | GPUs | Steady step | 1000-step wall |
|---|---|---:|---:|---:|
| GRPO  | `launch_rho1b_grpo.sh` | 1 | ~48 s | ~13 h |
| PPO+critic | `launch_rho1b_ppo_critic.sh` | 1 | **45.4 s** | ~13 h |
| VinePPO K=9 (DDP-2) | `launch_rho1b_vineppo_ddp2.sh` | 2 | ~115 s | ~32 h |
| CASPO (online V_φ) | `launch_rho1b_caspo.sh` | 1 | ~75 s | ~21 h |
| CASPO frozen | `launch_rho1b_caspo_frozen_rm.sh` | 1 | ~63 s | ~18 h |
| CASPO delta-p | `launch_rho1b_caspo_delta_prob.sh` | 1 | ~75 s | ~21 h |
| CASPO delta-logp | `launch_rho1b_caspo_delta_log_prob.sh` | 1 | ~75 s | ~21 h |

PPO+critic 1B steady-state measured 2026-04-27 at t_pol=17.0s,
t_value=20.3s, t_roll=5.0s, t_ref=3.1s → t_step=45.4s. Faster than
CASPO online at 1B because the 16-step Schulman critic cadence on
a 1B value head is cheap (small backward + Adam step), while CASPO
runs a full IPVRM forward+backward path that's heavier than the
clipped-MSE on a scalar value head. The 7B ratio flips because the
critic backward dominates t_value at scale.

Total 8-GPU suite wall time: gated by VinePPO DDP-2 at ~32 h
(parallel across the 8 GPUs).

#### DeepSeekMath-7B-MATH (4 H100s / method, 8 for VinePPO disagg)

`mb=2, accum=8` (mb=4/accum=4 for GRPO, which has no critic Adam),
`vllm_util=0.20` (CASPO/PPO+critic) or `0.30` (GRPO), `ckpt=true`,
`max_steps=1000`. PPO+critic and CASPO online share the same
4-GPU FSDP colocated topology so the head-to-head is honest on
identical hardware.

| Method | Script | GPUs | Steady step | 1000-step wall |
|---|---|---:|---:|---:|
| GRPO | `launch_7b_grpo.sh` | 4 | 48.8 s | ~14 h |
| PPO+critic | `launch_7b_ppo_critic.sh` | 4 | 92.0 s | ~26 h |
| VinePPO K=9 (disagg) | `launch_7b_vineppo_disagg.sh` | 8 | 197.6 s | ~55 h |
| CASPO (online V_φ) | `launch_7b_caspo.sh` | 4 | 47.7 s | ~13 h |
| CASPO frozen | `launch_7b_caspo_frozen_rm.sh` | 4 | 57.6 s | ~16 h |
| CASPO delta-p | `launch_7b_caspo_delta_prob.sh` | 4 | ~50 s* | ~14 h* |
| CASPO delta-logp | `launch_7b_caspo_delta_log_prob.sh` | 4 | ~50 s* | ~14 h* |

*7B delta-p/delta-logp are ablations of the standard CASPO online
trainer; same memory and compute as CASPO online with a different
``caspo_advantage_transform``. Step time inherits from CASPO.

The 7B suite **cannot run all methods in parallel on 8 H100s**
(each method needs 4-8 GPUs). Reference wall budget for the full
7-method 7B sweep, sequential: ~152 GPU-hours per seed (single
H100×8 box: ~19 days wall, or 7 days with 4 H100×8 boxes).

#### PPO+critic vs CASPO online: same compute envelope, different V₍φ₎

PPO+critic at 7B (92 s/step) is 1.93× slower than CASPO online
(47.7 s/step) **not because of a topology gap** — both run on the
same 4-GPU FSDP colocated config — but because PPO+critic uses the
**Schulman 2017 critic cadence** (`epochs_per_rollout=2 × accum=8 =
16 critic optim steps/outer iteration`) while CASPO online does
**1 V_φ optim step per outer** (the IPVRM-style update; 16× cheaper).
This is intentional asymmetry: CASPO's pretrained V_φ enters RL
already-good and only needs sparse online updates, while PPO's
from-scratch critic must catch up on a moving advantage target.

Disagg PPO+critic (8 GPUs, mb=4/accum=4) at 76.7 s/step is
available via `launch_7b_ppo_critic_disagg.sh` — matches VinePPO
upstream's PPO baseline topology for paper-faithful comparison.

**VinePPO / CASPO-online ≈ 4.14× per-iteration overhead** at 7B
(197.6 s vs 47.7 s). Matches upstream VinePPO paper Section 6.2.

### VinePPO speed iteration (2026-04-27)

Cumulative wins on disagg topology, all paper-faithful (no algorithm
deviations):

| Step | Result |
|---|---:|
| Original 8-GPU colocated TP=1 | 338s |
| Disagg + checkpoint sync | 298s |
| + NCCL packed weight sync | 277s |
| + fp8 KV cache | 268s |
| + reward_workers=16 | 245s |
| + max_num_seqs=2048 | **242s** (steady), **197s** (best step) |

28% faster than original; 17 hours saved on a 1000-step run.

### Phase-1 IPVRM value model (CASPO prerequisite)

Two scripts:

```bash
# (a) Single-GPU vLLM rollout collection (~25 min on 1 H100 for 4000 prompts)
./mnt/nvme_tmp/jason_caspo/smoke/run_collect_7b.sh

# (b) FSDP value-model training (4 GPUs)
GPU_LIST="0 1 2 3" RUN_TAG=value_v1 \
VALUE_DATA_PATH=/mnt/nvme_tmp/jason_caspo/deepseekmath7b_math/value_data.pt \
./scripts/_launch_7b_value_train.sh
```

Phase-1 collection produced ~17.5K mixed-outcome rollouts from 4000 MATH
prompts (54.5% mixed-outcome rate at G=8, temp=1.0 on DeepSeekMath-7B-SFT).

Known follow-ups: scripts/train_value.py runs validation eval on rank 0
only, which deadlocks under FSDP (the all-gather requires every rank).
Default `VALUE_EVAL_EVERY=0` in the launcher disables periodic val and
early stopping; the trainer just runs to `value_max_epochs`. Re-enable
once val eval is rewritten as a collective forward.

### Throughput-per-GPU note

For maximum throughput per GPU-hour on a single 7B run, 4 GPUs is the
sweet spot (vs 1/2/8). On an 8-GPU host, run 2 methods in parallel
(GPUs 0-3 + 4-7) — the suite finishes ~37% faster end-to-end than
serial 8-GPU runs and uses ~25% fewer GPU-hours.
