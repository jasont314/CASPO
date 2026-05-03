# IPVRM Reward-Model (V_φ) Training Guide

This document covers training the **prefix value model V_φ** used by CASPO.
V_φ is what distinguishes CASPO from VinePPO: instead of running fresh
Monte-Carlo rollouts at every RL step to estimate prefix values, CASPO
uses a learned model that scores prefixes in one forward pass.

**Why this matters for the paper:** the V_φ quality directly bounds how
useful CASPO's step-TD signal is. AUC = 0.50 means V_φ adds nothing over
random; AUC ≈ 0.7+ is "useful." Our current Rho-1B V_φ sits at **AUC =
0.633** on held-out, which is "weakly useful." Improving it via larger
datasets (this doc) is the main lever before downstream RL eval.

---

## Pipeline overview

`scripts/retrain_value_rho1b_4gpu.sh` orchestrates four phases on 4 GPUs:

1. **Collect** — 4-shard parallel rollout via `scripts/collect_value_data.py`.
   Each shard uses interleaved slicing (`prompts[i::4]`) so the difficulty
   distribution stays roughly uniform across shards.
2. **Merge** — concatenate the 4 `.pt` files via `scripts/merge_value_data_shards.py`.
3. **Train** — FSDP=4 `scripts/train_value.py` with BCE-with-margin loss
   on (correct, incorrect) pairs.
4. **Validate** — 1-step CASPO smoke rollout with the new V_φ; reports
   runtime `v_acc`.

---

## Recommended dataset per model

**Plan: train V_φ AND policy on the same dataset (per model size), then
eval on MATH-500 + GSM8K + AIME-2025.** This avoids cross-distribution
shift between V_φ training and RL deployment, follows standard
RL-for-LLM paper practice (PRM800K, DeepSeek-R1, Math-Shepherd), and
lets MATH-500 serve as a true OOD generalization eval.

| Model | RM training | RL training | Eval | Why |
|---|---|---|---|---|
| **Rho-1B-SFT** | **`open-r1/Big-Math-RL-Verified-Processed`** config `level_1` (~40K prompts) | Same (subsampled if needed) | MATH-500, GSM8K, AIME-2025-I, OlympiadBench | Empirically: 62.5% mixed-outcome yield at level_1, drops to 40% at level_2, 16% at level_4. Use `level_1` for max efficiency. |
| **DeepSeekMath-7B** | **`open-r1/Big-Math-RL-Verified-Processed`** config `level_2` (~45K prompts) | Same (subsampled if needed) | MATH-500, GSM8K, AIME-2025-I, OlympiadBench | Empirically: 81% mixed yield at level_2 (sweet spot — 7B-SFT saturates to all-correct on level_1 problems). Combined `level_1+level_2` (~85K, 82% avg yield) is also fine. **Note: DeepScaleR is TOO HARD** for the SFT base — only 12% mixed yield, 96% failures. |
| (paper-faithful baseline) | `DigitalLearningGmbH/MATH-lighteval` (7,500 rows) | Same | MATH-500 only (matches VinePPO paper) | Reproduction reference. Too small for V_φ to reach paper-grade AUC (we plateau at ~0.63 here vs Big-Math's projected ~0.70). Run as the "VinePPO-faithful" arm only. |

### Empirical mixed-outcome yield per model + Big-Math level

Sampled 32 prompts × K=8 rollouts at temperature=1.0 to measure how
many problems produce useful mixed-outcome (correct+incorrect) groups.
**Saturated all-correct or all-incorrect prompts contribute zero
training rows after the BCE-margin filter** — the mixed-outcome
percentage is the effective compute efficiency.

**Rho-1B-SFT-MATH** (`realtreetune/rho-1b-sft-MATH`, 32 prompts/level):

| Level | Mixed | all-correct | all-incorrect | avg pos rate |
|---|---|---|---|---|
| `level_1` | **62.5%** (20/32) | 0 | 12 | 0.195 |
| `level_2` | 40.6% (13/32) | 0 | 19 | 0.059 |
| `level_3` | 31.2% (10/32) | 0 | 22 | 0.066 |
| `level_4` | 15.6% (5/32) | 0 | 27 | 0.023 |

Rho-1B never saturates to all-correct on any level — wasted compute is
all on the all-incorrect side. **Pick `level_1`** → ~40K prompts ×
62.5% mixed = ~25K mixed prompts (16× current v6_multi's 1,575).

**DeepSeekMath-7B-SFT-MATH-v2** (`realtreetune/deepseekmath-7b-sft-MATH-v2`,
32 prompts/level + DeepScaleR):

| Pool | Mixed | all-correct | all-incorrect | avg pos rate |
|---|---|---|---|---|
| BigMath `level_1` | 84% (27/32) | 5 | 0 | 0.617 (saturating easy) |
| BigMath `level_2` | **81%** (26/32) | 0 | 6 | 0.270 (sweet spot) |
| BigMath `level_3` | 59% (19/32) | 0 | 13 | 0.168 |
| BigMath `level_4` | 47% (15/32) | 0 | 17 | 0.113 |
| BigMath `level_5` | 22% (7/32) | 0 | 25 | 0.035 |
| **DeepScaleR** | **12.5%** (4/32) ✗ | 0 | 28 | 0.039 |

**DeepScaleR is TOO HARD for DeepSeekMath-7B-SFT** — only 12% mixed
yield because 96% of rollouts fail. DeepScaleR was likely curated for
stronger reasoning baselines (R1-Distill / RL-finetuned). **Pick
BigMath `level_2`** (or `level_1+level_2` combined for max data) → ~80K
mixed prompts × multi-pair = ~250K rows.

**Net dataset choice (replaces earlier per-model recommendation):**
- Rho-1B → `open-r1/Big-Math-RL-Verified-Processed` config `level_1`
- DeepSeekMath-7B-SFT → same dataset, config `level_2` (or `level_1+level_2`)
- DeepScaleR is unused (too hard for both base models)

Verification: rerun `/tmp/bigmath_sample.py` (Rho) or `/tmp/dsmath_sample.py`
(DeepSeek) after any base model change to re-measure yield.

### Eval-set leakage (verified 2026-04-28)

Cross-checked all candidate train datasets against the eval suite:

| Train | MATH-500 | GSM8K | AIME-2025-I | OlympiadBench | AIME-2024 | AMC-val | AIME-val (AI-MO) |
|---|---|---|---|---|---|---|---|
| Big-Math (215K) | 0/500 ✓ | 0/1319 ✓ | 0/15 ✓ | 2/674 (0.3%) ✓ | 17/30 (57%) ✗ | 47/83 (57%) ✗ | 62/90 (69%) ✗ |
| DeepScaleR (40K) | **3/500 (0.6%)** ✗ | 0/1319 ✓ | 0/15 ✓ | 0/674 ✓ | 0/30 ✓ | 0/83 ✓ | 46/90 (51%) ✗ |
| MATH-lighteval (7.5K) | 0/500 ✓ | 0/1319 ✓ | 0/15 ✓ | 0/674 ✓ | 0/30 ✓ | 0/83 ✓ | 0/90 ✓ |

**Filter applied automatically.** `caspo.data.eval_leak` hashes all
problems in MATH-500 + GSM8K + AIME-2025-I + OlympiadBench at training
data load time; `_build_from_rows` drops any matching train row.
Filtering is on by default (`cfg.filter_eval_leakage = True`); set
False for paper-faithful baseline reproductions where the original
dataset author's filtering should be respected verbatim.

**Eval-set restrictions for paper writeup:**
- **PRIMARY (clean for both train sets):** MATH-500, GSM8K, AIME-2025-I, OlympiadBench. Use these.
- **DO NOT eval on with Big-Math train**: AIME-2024, AMC-val (AI-MO).
- **DO NOT eval on with EITHER train**: AIME-val (AI-MO).

**Rationale for difficulty matching:** the BCE-with-margin loss only
trains on **mixed-outcome prompts** (some rollouts correct, some
incorrect). Saturated prompts (all 8 rollouts agree) are dropped. With
K=8 rollouts:

* If true policy success on a prompt is 0% or 100% → 100% saturated.
* If true success is 50% → ~92% mixed (best yield).
* If true success is 20% → ~83% mixed (good yield).
* If true success is 5% → ~33% mixed (poor yield — most rollouts
  unanimously wrong).

So we want most prompts to have true success in [0.2, 0.8] for the
specific model. A dataset that's too hard wastes 50%+ of compute on
all-incorrect rollouts; too easy wastes it on all-correct.

---

## Current Rho-1B V_φ checkpoints

| Path | Dataset | Pairing | LR | AUC | Status |
|---|---|---|---|---|---|
| `caspo_rho1b_math_v6_multi/value_final/` | MATH-lighteval (1,575 mixed) | multi-pair | 5e-6 | **0.633** | live (YAML default) |
| `caspo_rho1b_math_v2/value_final/` | MATH-lighteval | keep-all-G | 5e-7 | 0.517 | deprecated, fixed_v6/v7 RL runs use this |
| `caspo_rho1b_math_v5_multi_lr5e7_failed/` | MATH-lighteval | multi-pair | 5e-7 | – | failed (LR too low for full-FT) |

---

## Knobs that matter

### Pairing protocol (`--paper-pairing` / `--paper-pairing-multi`)

* **Default (no flag)**: keep all G rollouts of each mixed-outcome
  prompt → ~G× rows but `pos_rate ≈ 0.20-0.30` (imbalanced).
* **`--paper-pairing`** (IPVRM §4.1 faithful): exactly 1 (correct,
  incorrect) pair per mixed prompt → 2 rows/prompt, 50/50 balanced.
* **`--paper-pairing-multi`** (our generalization): `min(n_pos, n_neg)`
  disjoint pairs per prompt → up to G rows/prompt, 50/50 balanced, no
  rollout reused. **Recommended.** Wins on row count (~5× over single
  pair) without violating the paper's pairing assumption.

### K (rollouts per prompt) — `cfg.group_size`

* Paper uses K=5. We use **K=8** (matches our RL `group_size`). At K=8
  we observe more mixed-outcome prompts and more pairs per mixed prompt
  vs K=5. Bumping further to K=16 marginally increases mixed-outcome
  yield (saturated prompts at K=8 may become mixed at K=16) at 2× the
  rollout cost.

### Learning rate — `VALUE_LR`

* **Paper's value: 5e-7.** This is for **LoRA** adapters. Don't use it
  for full-FT — V_φ won't move from initialization. (Confirmed: our
  v5_multi at 5e-7 early-stopped at step 300 with val_loss = init.)
* **Full-FT 1B: use `VALUE_LR=5e-6`** (10× paper). Validated on v6_multi.
* Going higher (1e-5, 5e-5) might add another ~0.02-0.05 AUC but risks
  divergence on small datasets. Try only after the data lever is exhausted.

### Epochs — `VALUE_MAX_EPOCHS`

* On v6_multi (12.6K rows): AUC plateaus at ~5 epochs. Going to 10
  adds ~0.01 AUC and starts to overfit (val_loss bottoms then rises).
* On a 5-10× larger dataset (DeepScaleR + Big-Math): probably 3-4
  epochs is enough.

### Save interval — `VALUE_SAVE_EVERY`

Set to roughly `n_steps_per_epoch / 4` to get ~4 ckpts per epoch for AUC
trajectory evaluation. For v6_multi with 177 steps/epoch, we used 50.

---

## Quick recipes

The orchestrator and one-GPU launcher both accept `DATASET_NAME` and
`DATASET_CONFIG` env vars (forwarded as `--override` to
`collect_value_data` / `train_caspo`). No YAML edits needed.

### Rho-1B retrain on Big-Math `level_1` (recommended, paper-grade)

```bash
GPU_LIST="4 5 6 7" \
OUT_ROOT=/mnt/nvme_tmp4/jason_caspo/caspo_rho1b_math_v7_bigmath_l1 \
DATASET_NAME=open-r1/Big-Math-RL-Verified-Processed \
DATASET_CONFIG=level_1 \
PAPER_PAIRING_MULTI=true \
VALUE_LR=5e-6 \
VALUE_MAX_EPOCHS=3 \
VALUE_SAVE_EVERY=500 \
  bash scripts/retrain_value_rho1b_4gpu.sh
```

ETA on 4× H100: **~2.5 hours** (collect ~75 min on ~40K prompts ×
K=8, train ~70 min on ~100K rows × 3 epochs, smoke ~5 min). Yields
~25K mixed prompts at 62.5% level_1 mixed-outcome rate (16× v6_multi's
1,575). Projected AUC: **0.68-0.71** (vs v6_multi's 0.633 on
MATH-lighteval).

### Rho-1B retrain on MATH-lighteval (paper-faithful baseline, fast)

```bash
GPU_LIST="4 5 6 7" \
OUT_ROOT=/mnt/nvme_tmp4/jason_caspo/caspo_rho1b_math_v6_multi \
PAPER_PAIRING_MULTI=true \
VALUE_LR=5e-6 \
VALUE_MAX_EPOCHS=5 \
VALUE_SAVE_EVERY=50 \
  bash scripts/retrain_value_rho1b_4gpu.sh
```

ETA: ~20 min total. AUC plateau: 0.633.

### DeepSeekMath-7B retrain on Big-Math `level_2` (recommended)

Use the 7B equivalent of the orchestrator (`scripts/_launch_7b_value_train.sh`)
with the same env-var pattern:

```bash
GPU_LIST="4 5 6 7" \
OUT_ROOT=/mnt/nvme_tmp/jason_caspo/value_model_dsmath7b_bigmath_l2 \
DATASET_NAME=open-r1/Big-Math-RL-Verified-Processed \
DATASET_CONFIG=level_2 \
PAPER_PAIRING_MULTI=true \
VALUE_LR=5e-6 \
VALUE_MAX_EPOCHS=3 \
  bash scripts/_launch_7b_value_train.sh
```

ETA: ~6-8 hours on 4× H100. **Do NOT use DeepScaleR** — empirically too
hard (only 12% mixed-outcome yield for 7B-SFT; 96% failures).

---

## Validation: did the new V_φ actually improve?

Run AUC eval over the trainer's held-out prompt-level val split:

```bash
python scripts/eval_vphi_auc.py \
    --vphi /path/to/value_final \
    --label v7_bigmath \
    --data /path/to/value_data.pt
```

Reports ROC-AUC, sign-acc, mean V on positive/negative classes, and
score margin. **AUC is the only metric that matters** — sign-acc is
threshold-sensitive and not what CASPO uses (CASPO uses ΔV between
adjacent steps, which is threshold-independent).

Reference baselines on v6_multi held-out:
* Random predictor: 0.500
* v2 (MATH-lighteval, keep-all-G, LR=5e-7): 0.517
* v6_multi (MATH-lighteval, multi-pair, LR=5e-6): 0.633
* IPVRM paper claim: ~0.65-0.75 (LoRA, larger K, larger dataset)
* Strong PRMs (PRM800K, MathShepherd): 0.80+ (require step-labeled data)

---

## Known TODOs

1. ~~**Difficulty filter for Big-Math**~~ — **resolved**: Big-Math's
   own `level_1`..`level_5` configs (and `quintile_1`..`quintile_5`)
   provide pre-computed difficulty buckets. Use `DATASET_CONFIG=level_1`
   for Rho-1B and `level_2` for DeepSeekMath-7B (validated empirically
   above).
2. **Multi-dataset concat**: loader currently takes one `dataset_name`.
   For "MATH + Big-Math" or "Big-Math level_1 + level_2" we'd need to
   extend `caspo/data/__init__.py:load_train_dataset` to accept a list.
   Workaround for now: pick a single config (level_1 alone is enough
   for Rho-1B; level_2 alone suffices for 7B).
3. **Per-step V_φ AUC during RL**: currently only offline AUC. Adding
   periodic AUC eval inside the RL loop would let us detect online
   drift in real time (relevant when bumping `online_value_lr`).

---

## Why we don't train V_φ on Monte-Carlo step labels

The conceptually cleaner approach — at each prefix s_t, run K rollouts
to estimate p̂_t = P(eventual correct | s_t), train V_φ to predict
p̂_t — is what **VinePPO does inline at every RL step**. Pulling it
out into a learned V_φ would cost K× more rollouts during V_φ
training (multiplied by every prefix, not just terminal), and the
resulting V_φ wouldn't have the free per-step decomposition that the
IPVRM cumulative-log-ratio parameterization gives us. IPVRM trades
classification accuracy for cheap inference + free credit assignment.

If we wanted to push AUC above ~0.7 cheaply, the closest we'd get is
**MC-pretraining a small portion of V_φ on step labels, then
fine-tuning on the IPVRM objective**. We haven't tried this; would
cost a few hundred GPU-h.

---

## Selected Refresh Recipe (v1 paper, 2026-05-01) — **SUPERSEDED**

> ⚠️ **Superseded by "Updated Recipe (gap-closed)" below (2026-05-03).**
> Kept here as historical record of how the recipe evolved; do **not**
> follow this section as current guidance. Specifically: response cap,
> training epochs, mb, accumulation, and ETAs all changed in the
> updated recipe. The reasoning around scratch-vs-warmstart and
> val_loss/ρ disagreement still stands.

After axis A (recipe), axis C (warmstart vs scratch), axis D (LoRA vs
full-FT), and the val_loss-vs-ρ disagreement findings, the recommended
refresh recipe **at the time** was:

**For each refresh trigger** (e.g., Spearman ρ drops 25% from peak):

1. **Snapshot** current policy checkpoint.
2. **Collect** MC labels on the current policy:
   - N=300 prompts (held out from any prior probe sets)
   - K=16 base rollouts × J=8 MC continuations × 5 step-boundary samples
   - max_response_len=1536, temp=1.0, top_p=1.0
   - 8-shard parallel: ~17 min on 8×A100
3. **Train V_φ from scratch** (NOT from previous PRM):
   - Init: `Qwen/Qwen2.5-Math-1.5B` (base SFT)
   - 3 epochs, lr=5e-6, mb=8 per rank, FSDP=4
   - β=10, BCE on continuous p_hat
   - ~136 min on 4×A100
4. **Pick ckpt by Spearman ρ on a fixed held-out 500-prompt probe** (NOT
   val_loss — see "val_loss/ρ disagreement" below). For full-FT, the
   `final/` ckpt is usually right; for LoRA (not recommended) it's
   step_1000.
5. **Resume CASPO** Δp from the policy snapshot with the new V_φ.

**Refresh budget:** ~2.5h wall-clock total per refresh, ~13 GPU-hours
on 8 GPUs (collection) + 4 GPUs (training). Well under VinePPO K=9's
~22 GPU-hour MC overhead per 150 RL steps.

## Updated Recipe (gap-closed, 2026-05-03; revised 2026-05-03)

After investigating why our sweep PRMs underperformed orig PRM, we
identified the gap as data design (max_response_len) and effective
batch size. The gap-closed recipe matches/exceeds orig PRM (ρ=0.456 vs
0.443) and is the new default.

**Recipe (initial PRM and all refreshes, unified at 2048):**
```
Collection (mc_step_label.py 4-shard):
  --K 16 --J 16 --steps_per_response 5
  --max_prompt_len 1024 --max_response_len 2048
  --max_train_prefix_len 0     # default — match collection cap
  --temperature 1.0 --top_p 1.0 --seed 0

Training (train_value_mc.py FSDP=4):
  --lr 5e-6 --mb 4 --grad_accum 2     # eff_batch = 4 × 4 × 2 = 32
  --epochs 2 --val_fraction 0.1
  --early_stop_patience 999            # no early stop
  --beta 10.0 --seed 0
```
ETA: ~41 min collection + ~50 min training on 4 GPUs.

**Why 2048?** Empirical Qwen2.5-Math-1.5B base rollouts on dsr_sub
(800 chains, max=3000) show **p98 of CORRECT chains at 1613 tokens**:
2048 catches 98% of correct chains, while truncating ~8% overall —
mostly failed/rambling chains, which is the seq_len penalty signal we
want. `max_response_len=2048` is the right ceiling for both initial PRM
collection and refresh PRM collection (the policy distribution shifts
across RL training but stays within this cap).

**Why no `--max_train_prefix_len` decoupling?** RL deploys at cap=2048,
so PRM training at cap=2048 keeps train/deploy distributions aligned.
The original "Option C" hedge (collect at 2048, train at 1024) was
motivated by the iter_max1792 result (-0.10 ρ vs iter_max1024). That
result is now **understood as a probe-cap mismatch artifact**: the
sweep evaluated all PRMs against a probe at cap=1024, so 1792-trained
PRMs were scoring out-of-distribution. The v3 refresh at cap=1536
trained without prefix decoupling and achieved ρ=0.630 in-distribution,
directly validating that training prefix cap >1024 is fine.

**When to use `--max_train_prefix_len`:** only as a hedge if you
specifically observe rambling-tail noise hurting downstream RL.
Default 0 (= match collection) is the correct production choice.

### Sweep findings (2026-05-02 to 2026-05-03)

Single-axis ablations on Qwen2.5-Math-1.5B / dsr_sub at gap-closed config:

| axis | range tested | best | notes |
|---|---|---|---|
| max_response_len | 512, 768, 1024, 1792 | 768 (≈1024) | ⚠️ probe-cap confound: probe was at cap=1024; 1792-trained PRMs were scored OOD. v3 refresh at cap=1536 (in-dist) got ρ=0.630, contradicting "longer hurts". 512 → NaN unrelated. |
| eff_batch | 4, 16, 32, 64 | 32 | ★★★★ doubling 4→32 → +0.10ρ |
| ep_per_run | 1, 2, 3, 5 | 2 (≈ 1) | ★★ marginal past ep=2; ep=1 already 99% of value |
| K | 4, 8, 16 | 16 (= 8 at matched compute) | ★ K is artifact of training data size, not diversity |
| S | 3, 5, 10 | 5 | minor — saturated |
| N | 100, 300, 500, 1209 | 1209 | monotone, diminishing past ~500 |
| J | 16 (only) | — | not subsamplable from current data |
| lr | 1e-6, 3e-6, 5e-6, 1e-5, 3e-5 | 5e-6 | ★★★ very narrow window |

The "gap-closing" findings:
- `max_response_len=1792` looked bad (-0.10ρ) but was scored against a
  probe at cap=1024 — half its training prefixes (the >1024 ones) were
  out-of-distribution at probe time. Likely measurement artifact, not
  intrinsic. Confirmed by v3 refresh at cap=1536 trained without
  prefix decoupling → ρ=0.630 in-dist.
- `eff_batch=4` (our prior default mb=1 FSDP=4) was the second-largest miss.
  Bumping to 32 (mb=4 acc=2) closed the rest.
- val_fraction=0.05 vs 0.1 was a minor but real factor (smaller val → noisier
  best-ckpt selection with patience=999).

### Why scratch (not warmstart, not LoRA)

Measured on a step_150 holdout 500-prompt probe (cross-policy):

| recipe | ρ | within-AUC | trajectory |
|---|---|---|---|
| pre_refresh (no refresh) | 0.540 | 0.724 | (baseline) |
| **full-FT scratch** | **0.630** | **0.805** | improves with steps |
| full-FT warmstart | 0.627 | 0.719 | improves with steps |
| LoRA-on-step_150 (peak) | 0.523 | 0.729 | peaks step_1000, regresses |
| LoRA-on-base (peak) | 0.383 | 0.680 | peaks step_1000, collapses |

- **Scratch beats warmstart** on within-AUC by 8.6pp; the val_loss
  best of warmstart slightly tied scratch on ρ but lost on within-AUC.
- **Scratch is operationally simpler** — no state to carry across
  refreshes; bias does not compound across iterated refreshes.
- **LoRA loses by 0.11 ρ at peak** and by 0.17 ρ at final
  (regression). LoRA's val_loss decreases throughout training but ρ
  drops past step_1000 — the cumulative-log-ratio architecture (β=10)
  amplifies LoRA's small delta into V swings that overfit
  in-distribution but don't transfer cross-policy.

### val_loss / ρ disagreement (critical for selection)

Both for full-FT and LoRA, the val_loss-best ckpt is NOT the ρ-best:

| | full-FT scratch | full-FT warmstart | LoRA-step150 | LoRA-base |
|---|---|---|---|---|
| best (val_loss) ρ | 0.585 | 0.604 | 0.450 | 0.190 |
| **final** ρ | **0.630** | **0.627** | 0.461 | 0.133 |
| **ρ-peak** ρ | 0.630 (final) | 0.627 (final) | **0.523 (step_1000)** | **0.383 (step_1000)** |

For full-FT, taking `final/` is safe (val_loss-best is conservative,
final usually has higher ρ). For LoRA, ρ peaks at step_1000 and val_loss
keeps improving past that — selecting by val_loss gives systematically
worse PRMs.

**Engineering action**: add an in-loop Spearman-ρ-on-probe eval to
`train_value_mc.py` (axis M of the matrix). Trigger best-ckpt save on
ρ improvement, not val_loss. This generalizes the selection criterion
to be correct for both full-FT and LoRA.

### Why "from base SFT" not "from current policy backbone"

Tested LoRA-on-step_150 (backbone init = step_150 ckpt) vs LoRA-on-base
(backbone init = base SFT). The step_150 backbone gives slightly better
LoRA quality (ρ=0.52 vs 0.38) — but full-FT from base wins both at 0.63.

For full-FT scratch on step_150 data with backbone=base, the model has
to learn step quality from scratch — but the BCE-on-p_hat loss
provides the right signal regardless of starting point. Using base SFT
keeps the recipe stateless across refreshes.

---

## PRM Refresh Experimental Matrix (2026-04-30)

Empirical observation (CASPO Δp on Qwen2.5-Math-1.5B, dsr_sub):
**step_150 is the dual-peak — best math500 (66.4%) and best PRM
Spearman ρ (0.54). Both collapse together by step_250 (math500 61.4%,
ρ=0.27).** This motivates the refresh experimental program: at the
peak step, refresh the PRM on fresh policy rollouts and continue.

### Per-refresh budget envelope (iso-VinePPO-K=9)

VinePPO at K=9 adds ~21.7 GPU-hours of MC overhead per 150 RL steps.
This is the ceiling for one CASPO refresh cycle. Current N=300, K=16,
J=8, 3-epoch refresh measured at **~8.6 GPU-hours** (40% of envelope).
Headroom of ~13 GPU-hours per refresh.

Per-component costs (Qwen2.5-Math-1.5B, 8×A100 collection, 4-rank FSDP train):
- 1 GPU-h Phase B ≈ 54K MC continuations
- 1 GPU-h training ≈ 600 train-steps over 40K prefixes (≈ 0.5 epochs)
- Phase B is ~95% of collection cost; scales linearly in **N × K × J**.

### Primary axes to sweep

**A. Refresh data recipe (N, K, J, epochs, budget) — optimize useful-window, NOT single-point ρ**

The metric of interest is **NOT** "ρ on a fixed probe set" but the
**decay curve of ρ as the policy drifts post-refresh**. Define:

```
useful_window = ∫ max(0, ρ(policy_at_step_t) − τ) dt over RL steps t
                from refresh_step until ρ first drops below τ
```

Where τ ≈ 0.4 (empirically: original PRM crossed 0.4 at the same step
that policy collapse started in original CASPO trajectory). The
quantity to maximize at fixed GPU-hour budget is the integrated
useful-window, **not** single-point initial ρ.

The four data-recipe knobs likely affect initial ρ vs decay rate
asymmetrically:

| axis | initial ρ effect | decay-rate effect | rationale |
|---|---|---|---|
| **N** (prompts) | mild | **strong slowdown** | More prompt diversity → better OOD generalization to drifted policy |
| **K** (base rollouts/prompt) | mild | mild slowdown | More rollout diversity per prompt |
| **J** (MC continuations/prefix) | **strong (lower noise)** | mild | Lifts initial ρ; doesn't help OOD generalization much |
| **epochs** | strong up then plateau | possibly accelerates decay | More epochs = better in-dist fit but may overfit → faster OOD decay |

**Empirical from 2026-05-01 v3 measurement** (refreshed PRM ρ vs RL
steps post-refresh, on (300, 16, 8, 3) recipe):

| RL steps post-refresh | refreshed PRM ρ |
|---|---|
| 0 (training dist) | 0.630 |
| 50 (step_200) | 0.613 |
| 100 (step_250) | 0.518 |
| 150 (step_300) | 0.527 |
| 200 (step_350) | 0.357 ← τ=0.4 cliff |
| 250 (step_400) | 0.443 |
| 300 (step_450) | 0.287 |

Useful window for current recipe ≈ **150-200 RL steps** (until ρ first
drops below 0.4 reliably).

**Sweep design — iso-budget, full decay-curve probe:**

| budget | candidate recipes (N, K, J, epochs) | hypothesis |
|---|---|---|
| ~5 GPU-h | (200, 16, 8, 3) · (300, 16, 4, 3) · (300, 16, 8, 1) | minimum-viable refresh |
| ~10 GPU-h | **(300, 16, 8, 3) — current** · (400, 16, 8, 3) · (300, 16, 16, 3) · (300, 16, 8, 6) | decay-rate sensitivity |
| ~10 GPU-h NEW | **(600, 16, 4, 3) — N-heavy** · (200, 16, 12, 3) — J-heavy | iso-budget N vs J shootout |
| ~17 GPU-h | (600, 16, 8, 3) · (300, 16, 24, 3) · (300, 16, 8, 9) · (400, 16, 16, 3) | scaling each knob |
| ~22 GPU-h | (600, 16, 16, 6) — iso-VinePPO ceiling | best PRM money can buy |

**Probe protocol — two-tier (cheap shortlist + expensive validation):**

Different PRMs produce **different policy trajectories** (empirically:
v3's refreshed-PRM trajectory diverges from orig's pre_refresh PRM
trajectory after ~50 RL steps even from same starting weights). So a
PRM's "decay" is path-dependent: PRM A scoring its own trajectory ≠
PRM A scoring some canonical trajectory. The cheap protocol below
factors this away to enable scalable ranking; the expensive validation
catches path-dependent effects for top candidates.

**Tier 1 — cheap protocol** (intrinsic OOD generalization, ~3 min/recipe):

Probe each candidate PRM against a **fixed canonical policy
trajectory** (orig CASPO step_150, 200, 250, 300, 350, 400 ckpts —
already saved at `caspo_dsr_full/`). Holds policy distribution
constant; isolates PRM OOD generalization.

**MC labels already collected** at
`/mnt/nvme_tmp7/jason_caspo/caspo_prm_probe_hires/step_{50,100,150,200,250,300,350,400}_probe_labels.pt`.
Generated 2026-04-30 with K=4, J=8 on 500-prompt holdout (same policy
ckpts as orig CASPO trajectory, same seed). Reusable across all
future recipe sweeps. **No additional collection needed.**

```
PER RECIPE (~3 min on 6 GPUs):
  for ckpt in {step_150, step_200, step_250, step_300, step_350, step_400}:
    eval_mcprm_auc_v2.py recipe_PRM/best labels_$ckpt.pt
    → cross-AUC, within-AUC, ρ
  Plot ρ(t) decay curve. Compute integrated useful-window.
```

Caveat: prefix count decreases at later ckpts (step_50 has 93 MB of
prefixes; step_400 only 24 MB) because more prompts saturate as
all-correct after RL training, leaving fewer mixed-outcome rollouts.
ρ measurement at step_350-400 is noisier (~960 prefixes vs ~3000 at
step_150). Consider adding step_450/500 labels later if those
trajectory points become important.

Use Tier 1 to **rank all candidate recipes** by useful-window. Cheap
because the policy MC labels are collected once and reused.

**Tier 2 — expensive validation** (downstream RL for top-2 winners,
~4-5h/recipe):

```
for top-2 winners from Tier 1:
  Run continuation CASPO from step_150 with that PRM, 250 RL steps
    (use existing v3 launcher, swap prefix_value_path)
  Save policy ckpts every 50 steps
  Probe the same PRM against ITS OWN policy ckpts (NOT canonical)
    → recipe-specific decay curve
  Greedy eval each policy ckpt on math500/gsm8k/olympiad
  Report: peak math500, useful-window from PRM perspective, math500 trajectory
```

Tier 2 catches path-dependent effects: a PRM with great Tier-1 score
might still drive policy in a counterproductive direction (or vice
versa). **Both numbers go in the paper.**

**Per-recipe cost:**
- Tier 1: recipe_GPU-h training + ~3 min probing
- Tier 2 (if top-2): recipe_GPU-h + ~5h downstream RL + eval

**5-recipe sweep total:** ~100 min one-time + 5 × recipe_GPU-h + 2 × 5h
≈ 14-17h depending on recipe budgets. Vs 5 × full RL = 25-30h.

Hypotheses to test:
- **At iso-budget, more N (with smaller J or fewer epochs to compensate)
  yields longer useful-window** — predicts (600, 16, 4, 3) >
  (300, 16, 8, 3) on the integrated metric, even if initial ρ is lower.
- **More epochs accelerate decay** — predicts (300, 16, 8, 6) starts
  with higher initial ρ but decays faster than (300, 16, 8, 3).
- **J has diminishing returns past J=8** — most of the noise reduction
  is already captured; J=16 may not justify 2× cost.

This is the right framing for the paper — single-point ρ is a metric
proxy, useful-window is what determines refresh frequency in
production deployment.

**B. Steps-per-response (boundary count per rollout)**

Currently 5. Sweep {3, 5, 10, 15}. More boundaries = more prefixes
per Phase A generation (cheap), but Phase B grows linearly. Test
whether RL benefits more from boundary-density or prompt-diversity.

**C. Initialization: warm-start vs from-scratch — IN FLIGHT 2026-04-30**

Currently running side-by-side, 4 GPUs each:
- warm-start: GPUs 0-3, init from `qwen_mc_prm_15b_dsr_sub/best`
- from-scratch: GPUs 4-7, init from `Qwen/Qwen2.5-Math-1.5B`
- shared collected data (N=300, K=16, J=8, step_150 policy rollouts)
- 3 epochs each, output at `mc_prm_refresh_step150/{warmstart,scratch}`

Compare:
- val_loss curve shape (does warm-start descend faster?)
- step-to-best (warm-start should converge in fewer steps)
- final Spearman ρ on probe (does warm-start inherit bias?)
- after multiple refresh iterations: does warm-start drift compound?

**D. LoRA vs full FT**

LoRA on PRM backbone (rank 16-32, attn+mlp targets) for 3-5× cheaper
training. Open question: does LoRA-PRM hit same ρ as full-FT, or does
the value-head calibration require dense backbone updates?

**E. Advantage transform: prob vs logprob**

- `prob`: A_t = sigmoid(V_{t+1}) − sigmoid(V_t) (probability
  difference; current default)
- `logprob`: A_t = logsigmoid(V_{t+1}) − logsigmoid(V_t)
  (log-probability difference)

`prob` bounds A_t to [-1, 1] but saturates near V≈±∞ (advantage
collapses for very-confident-correct or very-confident-wrong
prefixes). `logprob` doesn't saturate at the failure tail but is
unbounded — long failure chains accumulate large negative advantages.

Compare on: gradient magnitude stability, sign coherence with
ground-truth credit, downstream eval impact.

### Refresh-cadence axes

**F. Refresh interval (RL steps between refreshes)**

Sweep {50, 100, 150, 200}. Trades: tight = less drift + more
amortization cost; loose = more drift (we already saw collapse at
250). The Spearman drop is the natural trigger. Could measure
**ρ-vs-step** in a single long run to identify the inflection.

**G. Number of refreshes (drift compounding)**

Single refresh → multiple-refresh comparison. After 3+ iterations:
does PRM stay calibrated, or does small bias compound?

**H. Refresh-from-which-checkpoint**

Default: from step_150 (peak). Alternatives:
- earlier (step_100 — preempt peak)
- collapse point (step_250 — see if PRM recovers)
- delayed (step_400 — late catch-up)

### Architecture / objective axes

**I. PRM backbone size**

Current 1.5B (matches policy). Sweep {0.5B, 1.5B, 7B}. Smaller =
faster + cheaper; bigger = better ρ ceiling. Open: does PRM benefit
from being LARGER than the policy (richer features)?

**J. Cumulative log-ratio architecture vs direct scalar value head**

Current: V is the cumulative log-ratio between policy and a frozen
reference (architectural inheritance from IPVRM, even though training
is now MC-BCE). Inference requires 2× forward pass (π_φ AND π_ref).

Alternative: scalar value head on frozen backbone with V predicting
p_hat directly via sigmoid output. 1× forward, simpler. Compare
inference cost, training cost, ρ ceiling. If ρ matches, drop
log-ratio architecture for simpler/faster scoring.

**K. Training objective**

Current: BCE on continuous p_hat (with `--phi_init_path`-warmstart or
from-scratch). Alternatives: MSE on p_hat; asymmetric loss weighting
near-0/near-1 prefixes (where Δp matters most).

### Evaluation axes

**L. Probe set design**

Need a fixed test set across all sweeps for apples-to-apples. Today:
step_150 rollouts on held-out 500 dsr_sub prompts. Decisions:
- Same 500 every refresh? (controls difficulty)
- Re-sample each time? (controls memorization)
- Add OOD (Big-Math level_3)?
- Multiple policies (step_50, step_150, step_250) → cross-policy AUC
  matrix as primary metric

**M. Early-stop criterion in PRM training**

Current: val_loss + patience. Better: Spearman ρ on held-out probe
during training (the metric we actually care about). Needs new
in-loop probe in `train_value_mc.py`.

**N. Downstream RL eval per recipe**

For each PRM, run 50 RL steps from step_150 and report:
- math500 pass@1 at step_200
- KL drift, Δp magnitude stats
- whether eval keeps climbing past step_200

This is the only ground-truth metric. AUC/ρ are proxies.

### Suggested ordering (cheapest first)

1. **L** — probe set design (lock first, before any sweep, so all
   sweeps use same eval target)
2. **A** — recipe sweep at fixed budget, pick winner triple (~20 GPU-h)
3. **C** — warm-start vs scratch (in flight, ~13 GPU-h)
4. **E** — value/prob/logprob advantage (~8 GPU-h, needs short downstream RL)
5. **D** — LoRA vs full FT on winner from A (~10 GPU-h)
6. **F** — refresh interval sweep (~40 GPU-h, full RL trajectories)
7. **G** — multiple refreshes (~30 GPU-h, after F picks interval)
8. **B, K, M** — secondary, time permitting
9. **J** — log-ratio vs scalar value head (architectural simplification)
10. **I** — backbone size, only if 1.5B saturates badly

### Explicitly out of scope

- PRM ensembles (M models combined) — orthogonal direction
- Per-token PRM (vs per-step) — paper scope is step-level
- Off-policy MC labeling (use prior CASPO ckpts as MC source) —
  plausible but adds confounds; revisit if budget permits
