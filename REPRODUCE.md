# Reproducing CASPO vs. VinePPO baselines

This is the playbook for running CASPO and the VinePPO/PPO baselines on the
**same setup** (Rho-1B / DeepSeekMath-7B SFT init, MATH or GSM8K, identical
sample budget) so the headline table is apples-to-apples.

---

## Prereqs

- VinePPO repo cloned alongside CASPO at `/home/jason/experiment/VinePPO/` (already done).
- CASPO env: `source /opt/conda/etc/profile.d/conda.sh && conda activate /home/jason/experiment/.conda/envs/llm-research`
- VinePPO env: their README's instructions (separate conda env recommended — they pin DeepSpeed and vLLM).

## Models / data

VinePPO hosts their SFT checkpoints publicly. Just point the config at them.

| Setup | Policy SFT init | Train data | Eval | n |
|---|---|---|---|---|
| MATH (small) | `realtreetune/rho-1b-sft-MATH` | `lighteval/MATH` train | `HuggingFaceH4/MATH-500` | 500 |
| MATH (large) | `realtreetune/deepseekmath-7b-sft-MATH` | `lighteval/MATH` train | `HuggingFaceH4/MATH-500` | 500 |
| GSM8K (small) | `realtreetune/rho-1b-sft-GSM8K` | `openai/gsm8k` train | `openai/gsm8k` test | 1319 |
| GSM8K (large) | `realtreetune/deepseekmath-7b-sft-GSM8K` | `openai/gsm8k` train | `openai/gsm8k` test | 1319 |

Prompt format (verbatim from VinePPO's `MATH_step_by_step_sft.jsonnet`):
```
[MATH_TASK] Problem:
{query}

Solution:
```

This is wired into `caspo/config.py:prompt_template` and used in `caspo/data/math_data.py:format_prompt`.

## CASPO runs (this repo)

For each setup, run all three phases:

```bash
cd /home/jason/experiment/CASPO

# Phase 1a: collect (prompt, response, outcome) data with the SFT model.
# Override --num-prompts to control scale (3K is a reasonable floor; paper used ~480K).
python -m scripts.collect_value_data \
    --config configs/caspo_rho1b_math.yaml \
    --num-prompts 3000

# Phase 1b: train V_phi via IPVRM BCE-with-margin (Eq. 9). β=10, m=5, lr=5e-7.
# Trains for value_max_epochs=3 epochs over the filtered data with held-out
# per-prompt val split (10%); early-stops on val loss (patience=5 evals);
# saves the best-val-loss checkpoint as final/.
python -m scripts.train_value \
    --config configs/caspo_rho1b_math.yaml

# Phase 2: CASPO policy training. Online IPVRM (Eq. 15 with ADB+DLW) is on by default.
python -m scripts.train_caspo \
    --config configs/caspo_rho1b_math.yaml

# Eval
python -m scripts.eval \
    --config configs/caspo_rho1b_math.yaml \
    --override model_name_or_path=out/caspo_rho1b_math/final \
    --benchmarks math500 \
    --k 4
```

For DeepSeekMath-7B, GSM8K, etc. swap the config name. Memory: 7B + value-7B
+ ref-7B + (optional) policy-ref-7B → ~3-4× a single 7B forward.
Use `update_value_during_policy=false` to free V_φ optimizer state if memory-bound.

## VinePPO / PPO baseline runs (their repo)

VinePPO uses jsonnet configs and DeepSpeed. Their README has the exact
launch invocation.

```bash
cd /home/jason/experiment/VinePPO
# their setup; see their README for env activation and exact command
APP_DIRECTORY=runs/ppo_rho1b_math \
APP_SEED=0 \
deepspeed --no_local_rank --num_gpus=8 src/treetune/main.py \
    --configs configs/polIter_rho1bSft2_ppo_MATH.jsonnet \
    run_iteration_loop

# VinePPO baseline
APP_DIRECTORY=runs/vineppo_rho1b_math \
APP_SEED=0 \
deepspeed --no_local_rank --num_gpus=8 src/treetune/main.py \
    --configs configs/polIter_rho1bSft2_vineppo_MATH.jsonnet \
    run_iteration_loop
```

## What's matched, what's not

**Matched** (verbatim from VinePPO configs):
- SFT init checkpoints
- Dataset + prompt format
- group_size = 8, temperature = 0.6, top_p = 0.9, max_tokens = 1024 (MATH) / 512 (GSM8K)
- Rho-1B MATH rollout shape = 512 episodes/iteration = 64 prompts × 8 rollouts
- LR = 1e-6, weight_decay = 0, warmup ~ 0.03, max_grad_norm = 1.0
- target_train_batch_size = 64 (mapped to 64-response PPO minibatches)
- num_epochs_per_iteration = 2, total_num_iterations = 1000
- KL: init_kl_coef = 1e-4 (control_variate KL ≈ k3 estimator)
- γ = 1.0, λ = 1.0 (VinePPO match), PPO clip ε = 0.2

**Documented deviations** (call these out in the writeup):
1. **Value model**: VinePPO uses K=9 MC rollouts at every step boundary;
   CASPO uses an IPVRM prefix value model trained offline + online (Eq. 15).
   This is the actual contribution; not a deviation.
2. **Step segmentation**: matched. We ported VinePPO's
   ``math_extract_steps_inplace.py`` verbatim into
   ``caspo/segmentation/latex_splitter.py`` (only change: ``md5_hash``
   inlined). The LaTeX-aware splitter is wired in via
   ``segmentation_mode: latex_aware`` (default in the four reproduction
   configs). Token-level mapping uses per-token decode spans
   (``segment_responses_batch_latex_aware``) — equivalent to VinePPO's
   ``return_offsets_mapping`` pipeline for the BPE tokenizers we care about.
3. **No-ops** (numerically equivalent): CASPO's PPO loss uses the k3 KL
   estimator; VinePPO uses ``control_variate`` KL — both from Schulman 2020,
   same form modulo clipping bounds.

## Suggested table layout

| Method | Rho-1B MATH-500 | DSMath-7B MATH-500 | Rho-1B GSM8K | DSMath-7B GSM8K | Compute (avg gen / step) |
|---|---|---|---|---|---|
| SFT | (eval their ckpt) | (eval their ckpt) | … | … | 0 |
| PPO (their config) | from their numbers | from their numbers | … | … | 1× |
| VinePPO (their config) | from their numbers | from their numbers | … | … | ~10× |
| **CASPO (ours)** | run | run | run | run | ~1.1× |

If CASPO matches or beats VinePPO at ~1× compute, that's the headline.

## Smoke before the real run

```bash
# Tiny SmolLM smoke to verify the pipeline still works after the new prompt
# template + Rho config additions:
python -m pytest tests/ -q
python -m scripts.collect_value_data --config configs/value_smoke.yaml --num-prompts 4
python -m scripts.train_value --config configs/value_smoke.yaml
python -m scripts.train_caspo --config configs/caspo_smoke.yaml \
    --override update_value_during_policy=true \
    --override use_adb=true \
    --override use_dlw=true
```
