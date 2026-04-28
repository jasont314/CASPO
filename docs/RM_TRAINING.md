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
