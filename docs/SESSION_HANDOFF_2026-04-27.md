# Session Handoff — 2026-04-27

Context dump for handing this conversation off to a fresh assistant (GPT 5.5
or similar). Captures what was attempted, what landed, what's still in flight,
and what remains open.

---

## 1. Project context

**Repo**: `/home/jason/experiment/CASPO` — research codebase for a
**CASPO** method (CAuSal Prefix-value Online — author's working name).
CASPO = a research idea combining:

- **VinePPO** (Kazemnejad et al. 2024, arXiv:2410.01679) for step-TD
  advantage construction with K MC continuations OR a learned prefix
  value, on math reasoning.
- **IPVRM** (arXiv:2604.13197 "Unleashing Implicit Rewards: Prefix-Value
  Learning for Distribution-Level Optimization") for the value model V_φ
  trained with a BCE-with-margin objective on cumulative log-ratios.

Concrete: standard PPO policy update, but the per-step advantage is
computed as TD on a learned prefix-value V_φ (an HF causal LM with a
margin-BCE objective), rather than via MC continuations.

**Models**:
- 1B: `realtreetune/rho-1b-sft-MATH` (VinePPO authors' SFT init)
- 7B: `realtreetune/deepseekmath-7b-sft-MATH` (DSMath-7B SFT2)

**Train data**: `DigitalLearningGmbH/MATH-lighteval` (full MATH train, 12.5K)
**Eval**: `HuggingFaceH4/MATH-500` (Lightman et al. 500-problem split)
**Prompt template**: `[MATH_TASK] Problem:\n{q}\n\nSolution:`
**Sampling at eval**: T=0.35, top_p=0.9, max_tokens=1024 (paper-faithful)

**Hardware**: 8× H100 80GB on a single Debian 11 box. See
`docs/INFRA_SETUP.md` for full env / pinned package versions / GOTCHAs.

---

## 2. The 7 canonical methods × 2 model sizes

| Method | Description | 1B launcher | 7B launcher |
|---|---|---|---|
| GRPO | Group-relative reward, no critic | `launch_rho1b_grpo.sh` | `launch_7b_grpo.sh` |
| **PPO+critic** | Schulman 2017 PPO with separate critic | `launch_rho1b_ppo_critic.sh` | `launch_7b_ppo_critic.sh` |
| VinePPO | K=9 MC continuations as prefix value | `launch_rho1b_vineppo_ddp2.sh` | `launch_7b_vineppo_disagg.sh` |
| CASPO online | Pretrained V_φ with online IPVRM update | `launch_rho1b_caspo.sh` | `launch_7b_caspo.sh` |
| CASPO frozen | Pretrained V_φ, no online update | `launch_rho1b_caspo_frozen_rm.sh` | `launch_7b_caspo_frozen_rm.sh` |
| CASPO delta-p | Adv on `σ(V)` instead of raw V | `launch_rho1b_caspo_delta_prob.sh` | `launch_7b_caspo_delta_prob.sh` |
| CASPO delta-logp | Adv on `log σ(V)` | `launch_rho1b_caspo_delta_log_prob.sh` | `launch_7b_caspo_delta_log_prob.sh` |

**Plus** `launch_*_ppo.sh` (legacy critic-free PPO with sequence-level
advantages, kept for ablation but NOT in the canonical suite). See
README.md "Standard 8-GPU Suite" and "Per-method step times" tables.

### 7B step times (measured 2026-04-27, mb=2 colocated, 4 GPUs)

| Method | Step time | 1000-step wall |
|---|---:|---:|
| GRPO | 48.8 s (mb=4 also fits) | ~14 h |
| PPO+critic | 92.0 s | ~26 h |
| VinePPO disagg (8 GPU) | 197.6 s | ~55 h |
| CASPO online | 47.7 s | ~13 h |
| CASPO frozen | 57.6 s | ~16 h |

### 1B step times

| Method | Step time | 1000-step wall |
|---|---:|---:|
| GRPO | ~29 s (just measured) | ~8-13 h |
| PPO+critic | 45.4 s | ~13 h |
| VinePPO DDP-2 | ~115 s (2 GPUs) | ~32 h |
| CASPO online | ~75 s | ~21 h |
| CASPO frozen | ~63 s | ~18 h |

---

## 3. What was done this session (2026-04-27)

### Phase G — PPO+critic baseline (committed)

Built a proper Schulman 2017 PPO+critic baseline at 7B + 1B. Key fixes:

1. **`@torch.no_grad()` decorator removed from `_ppo_critic_train_critic`**
   (caspo_trainer.py:1186) — was silently disabling grad through the entire
   critic forward, making `v_loss.backward()` raise "element 0 of tensors
   does not require grad". Spent multiple iterations chasing GC / FSDP red
   herrings before finding it. Saved as `feedback_no_grad_decorator_bug.md`.
2. **`_infer_fsdp_layer_classes`** (caspo_trainer.py:747-770) now recurses
   into `module.modules()` to find `_no_split_modules` on inner backbones
   of wrappers like `CriticModel(backbone=Llama)`. Without this the
   critic was top-level FSDP-wrapped (full unsharded ~14 GB on every
   rank, no compute/comm overlap).
3. **`CriticModel.forward`** pre-embeds tokens and flips
   `requires_grad_(True)` (belt-and-suspenders for use_reentrant=False
   gradient checkpointing — `enable_input_require_grads` hooks get
   dropped during FSDP auto_wrap).
4. **Decoupled critic backward** from policy mb loop (Phase G.F) —
   halves activation peak so PPO+critic fits at mb=2 colocated.
5. **t_value reporting** — was always 0.0s for ppo_critic; now stores
   `t_value_forward_s` in stats.
6. **Two empty_cache calls** in `step()`: one at start, one before vLLM
   sync. Releases ~300 MB of fragmented allocator pool. Cost <0.15% of
   step time.
7. **Earlier alloc-config tuning was REVERTED**: tried adding
   `garbage_collection_threshold:0.6` and `max_split_size_mb:512` to
   `PYTORCH_CUDA_ALLOC_CONF` — both slowed step times 3-4× on tight
   regimes. Reverted to just `expandable_segments:True`.

### vLLM CUDA-graph trim (committed)

`caspo/rollout/vllm_engine.py` now passes
`compilation_config={"cudagraph_capture_sizes": [1,2,4,...,max_num_seqs],
"cudagraph_mode": "FULL_DECODE_ONLY"}` to the AsyncEngine. Default
captures ~256 shapes + piecewise prefill graphs (~150-300 MB / rank);
trim cuts to a sparse log-spaced set + drops piecewise prefill at near-zero
speed cost on RL rollout patterns. Gated by `CASPO_VLLM_CUDAGRAPH_TRIM=0`.

### Eval metric fix (committed)

`mean_response_len` was `len(decoded_text)` (CHARACTERS) — looked
alarming when CASPO frozen RM reported 1800 vs the 1024-token cap.
After unit fix:
- vLLM path uses `len(c.token_ids)` from `CompletionOutput`
- HF path uses `(new_tokens != pad_id).sum()`
- `tokens_per_sec_estimate` no longer divides by 4

Now reports actual generated-token count. Frozen RM goes from
"1800 (alarming)" → "1024.0 EXACTLY (every sample at the cap, length
collapsed)".

### VinePPO-strict reward (committed, just now)

`caspo/rollout/vllm_engine.py:1004-1014` now zeros out reward when
vLLM `finish_reason == "length"` (rollout halted by max_tokens
regardless of whether `\boxed{...}` happens to be in the truncated
text). Matches VinePPO upstream's `MathEpisodeGenerator`'s
`unfinished_response_penalty=0.0` post-processing exactly.

Diff: when a model emits `\boxed{42}` at token 50 then keeps rambling
to token 1024, our old path graded it (returned 1.0 if correct);
strict path returns 0.0. Rare in practice but paper-faithful.

### Infrastructure docs (committed)

`docs/INFRA_SETUP.md` — comprehensive setup guide for new contributors
(teammate had dependency issues). Covers exact pinned versions, env
recreation steps, GLIBCXX gotcha, host-RAM OOM during parallel 7B loads,
disk-full mitigations, etc.

### PPO suite consolidation (partially reverted)

Initially renamed `launch_*_ppo.sh` to mean PPO+critic. User reverted —
"keep both, have ppo just be ppo w/out critic". So:
- `launch_*_ppo.sh` = legacy critic-free PPO (preserved)
- `launch_*_ppo_critic.sh` = canonical Schulman 2017 PPO+critic
- `launch_7b_ppo_critic_disagg.sh` = 8-GPU disagg variant of PPO+critic

Suite map (`launch_rho1b_all8_standard.sh`) updated to use ppo_critic on
GPU 1.

---

## 4. What's currently running on this box

```
GPU 3:  GRPO 1B at paper_seed0     step ~1+/1000   strict-reward (NEW)
GPU 4:  CASPO online 1B            step 510/1000   collapsed at ~step 400, dragging
GPU 5:  CASPO frozen RM 1B         step 606/1000   length-collapsed (1024 cap on every sample)
GPU 6:  CASPO delta-p 1B           step 643/1000   mid-pack
GPU 7:  CASPO delta-logp 1B        step 631/1000   mid-pack
GPUs 0-2: free
```

**RUN_TAG=`paper_seed0`** for all 5. Output dirs:
`/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_<method>_paper_seed0/`

Logs: `/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_paper_seed0/logs/phase2_<method>.log`

**Important**: the 4 CASPO runs use the OLD reward path (started before
the VinePPO-strict reward commit `733785b`). The new GRPO run uses the
strict path. We'll keep them as-is to avoid losing 5-6h of training.

---

## 5. Key empirical findings this session

### a) The 4-method 1B comparison (step_500 sample evals, math500, k=8, n=100)

| Method | avg@8 | pass@8 | mean tokens | Δ avg vs SFT |
|---|---:|---:|---:|---:|
| SFT init | 0.205 | 0.48 | 108 | — |
| CASPO frozen | 0.160 | 0.43 | **1024 (capped!)** | -22% |
| CASPO delta-p | 0.224 | 0.49 | 213 | **+9%** |
| CASPO delta-logp | 0.171 | 0.42 | 212 | -17% |
| CASPO online | (collapsed at training step ~400; no step_500 ckpt) | | | |

**Full MATH-500 (n=500, k=16) at step_500**:
| Method | avg@16 | pass@16 | mean tokens |
|---|---:|---:|---:|
| SFT init | 0.150 | 0.476 | 232 |
| CASPO frozen | 0.112 | 0.454 | 1024 |
| CASPO delta-p | 0.157 | 0.452 | 442 |
| CASPO delta-logp | 0.151 | 0.456 | 429 |

**None of the CASPO variants are convincingly beating SFT on full MATH-500
at step 500**:
- delta-p: +0.7 pp avg, -2.4 pp pass (statistically marginal)
- delta-logp: ~equal (+0.1 pp avg, -2.0 pp pass)
- frozen: -3.8 pp avg, -2.2 pp pass — actively regressed
- online: collapsed (no checkpoint)

The literature search noted **"distribution sharpening"** as the
expected behavior — pass@K drops as RL concentrates probability mass
toward already-likely correct trajectories (Yue et al. 2504.13837).
But here we're seeing avg@k AND pass@k both drop for some variants —
that's worse than pure sharpening.

### b) Length-collapse failure mode (frozen RM)

At step_500, **every single one of frozen RM's 1600 samples (100 problems
× 16) hit the 1024-token cap**. Mean = exactly 1024.0. The model has
lost the ability to emit EOS within 1024 tokens — vLLM stops generation
at the cap, the cut-off response usually lacks `\boxed{...}`, grader
returns 0, but the gradient still finds enough signal in *some* longer
responses that succeeded — over training, length grows monotonically
with no force pulling it back. Once the model "forgets" how to be
concise, even truncation can't help (cut-off response doesn't include
the answer marker).

CASPO online had the same failure pattern at training step 430+
(reward=0.000 stuck at the floor). delta-p and delta-logp avoid this
because the bounded `prob`/`logprob` advantage transforms cap how much
any single step can push the policy toward verbosity.

### c) CASPO online's training-step collapse trajectory

```
step 1-10   reward ≈ 0.22  pass@G ≈ 0.50  KL=0
step 51     reward ≈ 0.27  pass@G ≈ 0.56  KL=0.018
step 200    reward ≈ 0.19  pass@G ≈ 0.47  KL=0.04
step 350    reward ≈ 0.17  pass@G ≈ 0.44  KL=0.05
step 400    reward = 0.000  pass@G = 0.000  KL=0.21  v_loss=0
step 410+   stuck at floor
```

v_loss=0 at step 400 means V_φ converged to "always predict negative"
(works for the all-negative batch). KL drift to 0.21 confirms the
policy moved meaningfully from ref before collapse.

Saved as memory `project_caspo_value_lr.md` — the original IPVRM paper
uses `online_value_lr=1e-4` but that's in a LoRA-VeRL pipeline; for
full-FT (our setup) we run at 1e-6 because 1e-4 collapsed V_φ in an
earlier session. So the current 1e-6 is intentional, not a bug.

---

## 6. Open questions to resolve

### Diagnostic

1. **Does vanilla GRPO improve over SFT?** (in flight on GPU 3 — that's
   the headline diagnostic). If yes → pipeline works, CASPO has issues.
   If no → something upstream (reward, KL, hyperparams) is broken.
   Wall ETA ~8 h (29 s/step × 1000).

2. **Has CASPO-delta-p's apparent step_250→step_500 climb (+0.038 avg@k
   on first-100) held up to step_750/step_1000?** Wait for those
   checkpoints to land then re-eval.

3. **Should we run vanilla VinePPO at 1B?** We have
   `launch_rho1b_vineppo_ddp2.sh` (uses 2 GPUs). If GRPO turns out fine
   but CASPO fails, VinePPO is the bigger piece of the puzzle (it's the
   paper baseline CASPO is trying to improve on).

### Engineering

4. **HF rollout path** (`caspo/rollout/sampler.py`) doesn't have
   finish_reason plumbing — the strict-reward fix only applies to vLLM.
   If we ever need the HF path, would need to detect truncation by
   `response_mask.sum(dim=1) == max_response_len AND no EOS`.

5. **Periodic eval during training** is NOT wired into the trainer.
   `eval_every: 40` is in YAMLs but ignored (`config.py:25` flags it
   as unused). Sample evals must be invoked via
   `launch_eval_rho1b_sample_all8.sh` after each `step_N` checkpoint
   lands. A sidecar polling script exists at
   `scripts/eval_periodic.sh` (untested).

6. **mb=4 colocated 7B** — not yet shipping for any method. GRPO mb=4
   works (48.8s vs mb=2 ~47s — marginal gain). PPO+critic mb=4 OOMs
   at colocated u=0.20 by ~100 MB. CASPO mb=4 OOMs by ~280 MB. Tried
   alloc-config tuning + cudagraph trim; the alloc-config slowed
   things 3-4× while saving memory; cudagraph trim is the only
   no-cost-no-OOM win we kept. Verdict: stick with mb=2 for colocated
   trainings; mb=4 is for disagg topology only (PPO+critic 8-GPU
   disagg fits at 76.7 s/step).

### Research

7. **CASPO collapse / regression vs SFT** — likely root causes (in
   priority order):
   - KL coef (1e-4) too low; policy drifts unconstrained
   - No length penalty in reward → length-collapse pathway open
   - Online V_φ updates can destabilize when V_φ is full-FT (saved
     memory: at lr=1e-4 V_φ collapsed; at 1e-6 it doesn't but doesn't
     track policy either)
   - `value` advantage transform produces unbounded advantages →
     larger policy steps → more drift; `prob`/`logprob` are stable
8. **Does CASPO need a length penalty in reward** even though VinePPO
   doesn't? Their paper-faithful PPO/VinePPO must somehow avoid
   length collapse — possibly because their policy gradient is
   bounded differently, OR because they train on shorter
   `max_response_len`, OR because their SFT init is more terminating.
   Worth running their vanilla PPO baseline as control.

---

## 7. Files of interest (all with absolute paths)

### Core trainer
- `/home/jason/experiment/CASPO/caspo/trainer/caspo_trainer.py` — main
  CASPOTrainer with method dispatch (caspo / ppo / grpo / vineppo /
  ppo_critic). Memory-related guards at lines 1919-1925, 2388-2400.
  Method branches around lines 2003-2150.
- `/home/jason/experiment/CASPO/caspo/critic/critic_model.py` —
  CriticModel for PPO+critic (`from_pretrained` calls
  `enable_input_require_grads` post gradient_checkpointing_enable).
- `/home/jason/experiment/CASPO/caspo/value/prefix_value.py` — IPVRM
  PrefixValueModel implementing Eq. 5 of arXiv:2604.13197
- `/home/jason/experiment/CASPO/caspo/value/train_value.py` — IPVRM
  offline trainer (Eq. 9 BCE-with-margin)
- `/home/jason/experiment/CASPO/caspo/algo/advantages.py` —
  step-TD advantage construction; `transform_step_values_for_advantage`
  at line 275 implements value/prob/logprob transforms.

### Rollout
- `/home/jason/experiment/CASPO/caspo/rollout/vllm_engine.py` — vLLM
  AsyncLLM wrapper. CUDA-graph trim at lines 320-340 (gated by
  `CASPO_VLLM_CUDAGRAPH_TRIM`). Strict-reward zero-out at lines
  1004-1014 (NEW).
- `/home/jason/experiment/CASPO/caspo/rollout/disagg.py` — proxy that
  slices a vLLM-produced RolloutBatch for the disagg trainer ranks.
- `/home/jason/experiment/CASPO/caspo/rollout/sampler.py` — HF fallback
  rollout (NOT used in production; doesn't have strict-reward).

### Eval
- `/home/jason/experiment/CASPO/scripts/eval.py` — entrypoint
- `/home/jason/experiment/CASPO/caspo/eval/benchmarks.py` — math500
  driver. `mean_response_len` now in TOKENS at line 535 (vLLM path)
  and 691 (HF path).
- `/home/jason/experiment/CASPO/caspo/reward/math_verifier.py` —
  binary 0/1 + format_bonus=0.0 default. `grade_math` at line 272.

### Configs
- `/home/jason/experiment/CASPO/configs/caspo_rho1b_math.yaml` — 1B
  base config; `max_response_len=1024`, `max_steps=1000`,
  `online_value_lr=1e-6`, `value_max_epochs=3`.
- `/home/jason/experiment/CASPO/configs/caspo_deepseekmath7b_math.yaml`
  — 7B counterpart; same structure.

### Launchers
- `/home/jason/experiment/CASPO/scripts/perf_env.sh` — central env vars
- `/home/jason/experiment/CASPO/scripts/_launch_7b_fsdp.sh` — colocated
  4-GPU base launcher for 7B
- `/home/jason/experiment/CASPO/scripts/_launch_7b_disagg.sh` — 8-GPU
  disagg base launcher (FSDP=4 trainer + vLLM TP=4 dedicated)
- `/home/jason/experiment/CASPO/scripts/_launch_rho1b_one_gpu.sh` — 1B
  base launcher
- `/home/jason/experiment/CASPO/scripts/launch_rho1b_all8_standard.sh`
  — orchestrator for the 8-GPU 7-method 1B suite
- `/home/jason/experiment/CASPO/scripts/launch_eval_rho1b_sample_all8.sh`
  — cheap sample eval (math500, limit=100, k=8)
- `/home/jason/experiment/CASPO/scripts/launch_eval_rho1b_final_all8.sh`
  — full final eval (math, math500, collegemath, olympiadbench, k=16)

### Docs
- `/home/jason/experiment/CASPO/README.md` — main README; per-method
  step-time tables (1B and 7B)
- `/home/jason/experiment/CASPO/docs/INFRA_SETUP.md` — env setup
  guide for new contributors (NEW this session)
- `/home/jason/experiment/CASPO/docs/disaggregated_topology_plan.md`
  — context on the FSDP+vLLM disagg work
- `/home/jason/experiment/CASPO/docs/SESSION_HANDOFF_2026-04-27.md`
  — THIS FILE

---

## 8. Recent commits (chronological, this session only)

```
733785b rollout: zero reward when finish_reason==length (VinePPO-strict)
c55f5ee docs: comprehensive infrastructure setup guide for new contributors
39e48a5 eval: count mean_response_len in tokens, not chars
a5cb24c 7B canonical-suite parity: delta launchers, README per-method tables, vLLM CUDA-graph trim
fa8440e Phase G.M perf env tuning: alloc config + logprob_mb=4 for mb=4 colocated  [later partly reverted]
4bbcdf2 Phase G.G + G.H + G.M trainer fixes: make PPO+critic actually work
1c8b987 Phase G.F: decouple critic backward from policy mb loop  [pre-session]
```

---

## 9. Saved memories that might be relevant

In `/home/jason/.claude/projects/-home-jason-experiment/memory/`:

- `project_caspo.md` — overall CASPO status (last updated 2026-04-25)
- `project_caspo_optim_sweep.md` — Rho-1B 1-GPU optimum config
- `project_caspo_7b_speedup.md` — 47.7s/step baseline for 7B
- `project_caspo_value_lr.md` — **online_value_lr=1e-6 not 1e-4
  because full-FT collapses at the paper's value** (recurring gotcha)
- `feedback_no_grad_decorator_bug.md` — the @torch.no_grad bug we hit
- `feedback_storage.md` — disk-full pre-launch check (we hit this twice)

Read these into context before reasoning about CASPO; they've already
been hard-won through prior debugging.

---

## 10. Suggested first action for the next assistant

1. **Check the GRPO run on GPU 3** — that's the headline diagnostic
   for whether the pipeline works at all on Rho-1B with paper-faithful
   reward. Output dir
   `/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_grpo_paper_seed0/`,
   log `/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_paper_seed0/logs/phase2_grpo.log`.
   First step_250 checkpoint ETA ~2h from now.

2. **Then run sample eval** at step_250 once it lands:
   ```bash
   RUN_TAG=paper_seed0 CKPT_SUBDIR=step_250 \
     METHODS="grpo" EVAL_GPU_LIST="0 1 2" \
     EVAL_BENCHMARKS=math500 EVAL_K=16 \
     EVAL_VLLM_GPU_MEMORY_UTILIZATION=0.85 \
     ./scripts/launch_eval_all.sh   # NB: NOT the _sample_all8 wrapper if you want full MATH-500
   ```

3. **Decide based on result**:
   - If GRPO at step_250 hits ≥0.18 avg@16 (above SFT's 0.150) → pipeline works,
     CASPO algorithm has issues, focus there.
   - If GRPO also flatlines at SFT → reward / KL / hyperparam tuning is
     needed. Try KL coef ↑ (1e-3 instead of 1e-4) on a fresh seed.
