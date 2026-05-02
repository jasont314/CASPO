# Paper Plan: Iteratively-Refreshed MC PRM as PPO Critic for RLVR

**Status**: drafting plan, not yet executed
**Last updated**: 2026-04-30

## TL;DR

The paper's core claim: **An explicit Math-Shepherd-style Monte Carlo PRM, when periodically refreshed against the current RL policy, becomes a useful per-step critic in PPO and recovers (or beats) vanilla GRPO at 1.5B-7B math scale — addressing the long-standing "frozen PRM goes stale during RL" failure mode.**

This positions the work as the explicit-PRM analog to PRIME (Cui et al., Feb 2025), which made the same argument for *implicit* (DPO-log-ratio) PRMs. The contribution is the explicit-MC variant + clean ablations.

## Architectural framing (course-corrected 2026-04-30)

PPO with the PRM serving as the critic is the natural framework — NOT a GRPO + Δp hybrid as I'd briefly suggested earlier:

```
PPO advantage (TD residual, sparse-reward setting):
  A_t = r_t + V(s_{t+1}) - V(s_t)
      = V(s_{t+1}) - V(s_t)         (since r_t = 0 except terminal)
      = Δp_t                         (after sigmoid)
```

So **CASPO Δp IS PPO with the PRM as critic** under one framing. There's no "GRPO + Δp hybrid" to invent — that was a confusion. The clean axis is:
- Vanilla GRPO (no critic)
- PPO with various critics (frozen/refreshed × IPVRM/MC-explicit/implicit)

GRPO doesn't natively support per-step credit (its advantage is shared across all tokens in a trajectory). Bolting Δp onto GRPO is ad-hoc; PPO is principled.

## Why prior CASPO Δp lost (and why this run won't)

Existing eval matrix (the v7/v8 runs):
| Method | math500 pass@k peak |
|--------|---------------------|
| GRPO (vanilla) | **51.0** |
| PPO+Critic | 46.4 (-4.6pp) |
| CASPO | 46.8 (-4.2pp) |
| CASPO Δp | 45.0 (-6.0pp) |

All PPO-critic-based methods lost by 4-6pp. But this v8 had:
1. 0.73 within-AUC IPVRM PRM (vs 0.81-0.84 we have now)
2. **No refresh** — PRM was static throughout training
3. No measurement of distribution drift during RL

The transfer-matrix experiment we ran (2026-04-30) showed:
- In-distribution AUC: 0.92
- Cross-distribution AUC: 0.50 (chance)

Implication: a static PRM trained at base-Qwen labeler distribution becomes useless once RLVR drifts the policy. **Refreshing the PRM closes that gap.**

## Method specification — math

### Per-step Δp credit assignment

For a trajectory with steps $\{1, 2, \ldots, K\}$, where step $k$ spans tokens $[t_k^{\text{start}}, t_k^{\text{end}}]$, define:

$$V_k \triangleq V(s_{t_k^{\text{start}}}) \quad \text{(PRM forward at start of step } k\text{)}$$

**Step-level advantage** (one value per step, for non-terminal steps $k < K$):
$$\tilde{A}_k = \sigma(V_{k+1}/\beta) - \sigma(V_k/\beta)$$

**Terminal step** absorbs the verifier reward:
$$\tilde{A}_K = R_T - \sigma(V_K/\beta)$$

where $R_T \in \{0, 1\}$ is the math verifier outcome and $\beta = 10$ (training-time temperature).

**Token-level advantage** (uniform within step):
$$A_t = \tilde{A}_k \quad \forall t \in [t_k^{\text{start}}, t_k^{\text{end}}]$$

**Batch normalization** (across all step-advantages in a batch of size $B$ rollouts):
$$A_t^{\text{norm}} = \frac{\tilde{A}_k - \mu_{\text{batch}}(\tilde{A})}{\sigma_{\text{batch}}(\tilde{A}) + \varepsilon}$$

This whitened $A_t^{\text{norm}}$ enters the standard PPO clipped surrogate loss.

### Telescoping property — total trajectory credit is preserved

The sum of all step-advantages telescopes:
$$\sum_{k=1}^{K} \tilde{A}_k = [\sigma(V_2) - \sigma(V_1)] + [\sigma(V_3) - \sigma(V_2)] + \cdots + [R_T - \sigma(V_K)]$$
$$= R_T - \sigma(V_1)$$

So **total trajectory credit equals the terminal verifier reward minus the PRM's initial prediction**, regardless of the intermediate $V_k$ values. This is the same total signal vanilla PPO/GRPO would assign at the trajectory level — Δp credit just *redistributes* it across steps according to the PRM's read of where progress was made or lost.

### Behavior on unsuccessful trajectories

For $R_T = 0$ trajectories (failures), with $\sigma(V_1) = 0.4$:
- Total trajectory advantage = $0 - 0.4 = -0.4$
- Sum of all $\tilde{A}_k = -0.4$ (forced by telescoping)
- Individual step advantages can be positive or negative depending on whether that step was estimated to advance or harm the trajectory's success probability

Worked example with K=4 plus a terminal step:

| Step | $\sigma(V_k)$ | $\sigma(V_{k+1})$ | $\tilde{A}_k$ | Interpretation |
|------|---|---|---|---|
| 1 | 0.40 | 0.60 | +0.20 | promising |
| 2 | 0.60 | 0.75 | +0.15 | progressing |
| 3 | 0.75 | 0.50 | -0.25 | mistake step |
| 4 | 0.50 | 0.55 | +0.05 | small recovery |
| K (terminal) | 0.55 | $R_T = 0$ | -0.55 | trajectory failed |

Sum: $0.20 + 0.15 - 0.25 + 0.05 - 0.55 = -0.40 = R_T - \sigma(V_1)$ ✓

The policy gradient receives:
- Step 3's tokens: $A = -0.25$ → push DOWN (correct: that's where the mistake happened)
- Step 1's tokens: $A = +0.20$ → push UP (don't punish steps the PRM saw as progress, even on failed trajectories)
- Terminal: $A = -0.55$ → strong push DOWN on the final answer

Compared to vanilla PPO/GRPO (terminal-only), where every token in the failed trajectory gets the same $\approx -0.4$ advantage uniformly. **Δp credit identifies which step screwed up; vanilla credit punishes the whole rollout uniformly.**

### Behavior on successful trajectories

Same math, opposite sign. Example with $R_T = 1$, $\sigma(V_1) = 0.4$:

| Step | $\sigma(V_k)$ | $\sigma(V_{k+1})$ | $\tilde{A}_k$ |
|------|---|---|---|
| 1 | 0.40 | 0.60 | +0.20 |
| 2 | 0.60 | 0.45 | -0.15 (lucky misstep) |
| 3 | 0.45 | 0.85 | +0.40 (decisive insight) |
| 4 | 0.85 | 0.90 | +0.05 |
| K | 0.90 | $R_T = 1$ | +0.10 |

Sum: $+0.60 = 1 - 0.40$ ✓

Step 3 receives the largest credit (+0.40) — the decisive moment. Step 2 receives *negative* credit despite the trajectory ultimately succeeding (it was a misstep that the model recovered from). Vanilla GRPO would give $+0.6$ to all steps uniformly, reinforcing step 2's mistake.

### Robustness to PRM noise (the elegant property)

**Telescoping holds regardless of PRM quality**, because the terminal $\tilde{A}_K = R_T - \sigma(V_K)$ explicitly closes the gap. Consequence:

- Pure-noise PRM (within-AUC = 0.5, $\rho$ = 0): individual $\tilde{A}_k$ are random, but $\sum_k \tilde{A}_k = R_T - \sigma(V_1)$ still holds. Per-step decomposition is uninformative noise on top of the trajectory-level signal. Variance increases, mean preserved.

- Calibrated PRM (within-AUC ≥ 0.8, $\rho$ ≥ 0.7): $\tilde{A}_k$ tracks real difficulty changes. Decomposition is informative. Variance decreases.

So Δp credit can never be net-worse than trajectory-level credit *on average* — at worst it adds variance with the same mean. **The risk axis is variance, not bias.**

This makes the v8 CASPO Δp $-6$pp loss vs GRPO somewhat puzzling under this analysis. Three possible explanations:
1. The IPVRM v7 PRM was *actively miscalibrated* post-drift (worse than noise — biased), not just noisy.
2. Variance overwhelmed the small per-step credit signal at 1B scale with low per-prompt rollout budgets.
3. Implementation bug — the v8 code path may have failed to properly include the terminal $\tilde{A}_K$ correction.

(3) is testable. Mid-training drift (1) is what refresh fixes. (2) implies the technique helps more at larger scale or longer trajectories.

### PRM querying granularity

The PRM is queried at step boundaries only — once per step, not once per token. This is a deliberate efficiency choice:
- Matches the label granularity of MC training (we trained on step-boundary prefixes)
- ~30× fewer PRM forwards than per-token querying
- Per-token querying = VinePPO-style alternative; would give finer credit but at much higher inference cost
- For math reasoning, step boundaries are semantically meaningful and the PRM was *trained* to discriminate at those boundaries

### Normalization scope

Two natural variants:
1. **Per-prompt-group normalization** (closer to GRPO): standardize $\tilde{A}_k$ within each prompt's $G$ rollouts. Inherits the group baseline.
2. **Full-batch normalization** (CASPO default): standardize across all step-advantages in the entire RL step. Lower-variance estimates of $\mu, \sigma$ at the cost of mixing prompt difficulties.

The full-batch variant is what current CASPO Δp uses. The per-prompt-group variant is a reasonable ablation — could close the v8 gap if prompt-difficulty leakage was a factor.

## Method specification

### Iterative refresh schedule

```
Phase 0 (one-time):
  Train initial MC PRM at base policy (~50 min, AUC=0.90)

For phase t = 1, 2, ..., T (every k=100 RL steps):
  1. Snapshot current policy π_t
  2. Generate ~2K rollouts from π_t on RL prompt set      (~5 min)
  3. MC-label at random token positions, K=8, J=8         (~10 min)
  4. APPEND to cumulative replay buffer (don't replace)
  5. Fine-tune PRM_{t-1} on cumulative buffer:
     - target within-AUC ≈ 0.83 (~step 1500 from cold)
     - or warm-start from PRM_{t-1} for 1000 steps
                                                          (~10-15 min)
  6. Use PRM_t as critic for next k RL steps
```

**Refresh cost per phase**: ~25-30 min.
**Total over 500-step RL run**: 5 phases × ~30 min = **2.5 hours additional compute**.
**Wall-clock impact**: zero if PRM refresh runs on idle GPUs in parallel with RL.

### Hyperparameters
- k (refresh interval): 100 RL steps (start) — ablate over {50, 100, 200}
- Target AUC per refresh: within-AUC=0.83 (Spearman ρ ≈ 0.7)
- Replay mix: 70% new phase / 30% sampled from cumulative buffer
- Refresh fine-tune LR: 1e-6 (1/5 of original 5e-6)
- Fine-tune steps per refresh: 1000

## Experimental design

### Methods to compare
| ID | Description | Critic | Refresh? |
|----|-------------|--------|----------|
| A | Vanilla GRPO | None | N/A |
| B | PPO + frozen IPVRM critic | IPVRM v7 | No |
| C | PPO + frozen MC-explicit critic | MC PRM | No |
| D | **PPO + refreshed MC-explicit critic** (proposed) | MC PRM (refreshed) | Yes (every 100 steps) |
| E | PRIME (implicit PRM, online update) | Log-ratio | Yes (online) |

### Benchmarks
- math500 (in-domain, easy)
- gsm8k (in-domain, easy)
- olympiadbench (harder, where step credit may matter more)
- aime25 (hardest, sparse signal)

### Decision rules
- If D beats A on math500 by ≥3pp: strong positive result
- If D beats A on aime/olympiad by ≥5pp: even stronger (where step credit theoretically helps most)
- If D ties A on math500 but beats on aime/olympiad: still publishable
- If D underperforms A on all: pivot to negative-result paper centered on transfer matrix

### Existing assets we can leverage
- 4-method comparison (A, B, C-frozen, CASPO-Δp-frozen) already evaluated → A and B are done
- dsr_sub PRM trained to AUC=0.92 (cross), within=0.84, Spearman=0.72
- AUC saturation curve measured (10 ckpts at step 250-2500)
- RLVR vanilla GRPO endpoint at step ~500 (06:30 UTC, in progress)

What's missing for the paper:
- D: the actual refresh-PRM run (~9-12h compute)
- E: PRIME baseline reproduction (~1 day to set up + 1 day to run)
- (Optional): cross-family multi-labeler PRM ablation (not needed for first paper)

## Methodological contributions (in priority order)

1. **Transfer matrix** showing MC PRMs are distribution-specific to chance level (within-domain to within-domain). Cleanly motivates iterative refresh.

2. **AUC-vs-step saturation curve** — within and cross AUC saturate at different rates; within (the metric that matters for credit) saturates ~2× slower. PRM training budget should be allocated based on use case.

3. **Spearman ρ as the credit-assignment metric** — within-prompt Spearman matters more than within-prompt AUC for Δp credit because it captures gradation, not just sign. Most prior PRM papers report cross-prompt accuracy/AUC; we argue for within-prompt Spearman.

4. **Iterative MC-PRM refresh during RLVR** — the explicit-PRM analog to PRIME's implicit-PRM online update. Shows the technique extends beyond log-ratio formulations.

5. **Pareto allocation** — "train shallow, refresh often" beats "train deep, refresh rarely" at fixed compute. Within-AUC = 0.83 at 30 min × 5 refreshes > Within-AUC = 0.92 at 90 min × 1 refresh.

## Reviewer attack vectors and responses

**"Why not just PRIME's implicit PRM?"**
- PRIME's PRM is a single log-ratio of LMs. Many existing released PRMs (Math-Shepherd, OmegaPRM, Skywork-PRM, Qwen2.5-Math-PRM) are MC-explicit; our refresh recipe applies directly to those without re-architecting.
- Empirical: planned ablation comparing D (explicit MC + refresh) vs E (PRIME) on the same prompts.

**"GRPO already works, why bother with PPO+PRM?"**
- Per-step credit is theoretically faster on long-horizon problems; expected to show up on aime/olympiadbench specifically.
- GRPO has limited per-step granularity; PPO+PRM enables curriculum-style intermediate-signal learning.

**"Refresh frequency is a hyperparameter — how do you choose?"**
- Ablation over k ∈ {50, 100, 200} planned.
- Saturation curve gives a budget-based answer: pick k such that per-phase compute ≤ baseline RL compute / 5.

**"Did you measure distribution drift during RL directly?"**
- Yes — KL(policy_t || policy_0) per-token tracked, plus a planned AUC-vs-RL-step ablation evaluating frozen PRM on policy_t rollouts at multiple t. This is the smoking-gun figure for "PRM goes stale."

## What's already running tonight

- **Vanilla GRPO** RLVR replication (One-Shot-RLVR's dsr_sub recipe), GPUs 4-7, ETA ~05:34 UTC.
  - This is method A in the comparison.
  - Also gives us policy ckpts at step_200, step_400, final for the AUC-vs-RL-step ablation.

- **MC PRM dense retrain** (dsr_sub, K=16/J=16, 60K prefixes), GPUs 0-3, killed at step 2500 by watcher.
  - Provides AUC saturation curve (already measured).
  - 10 ckpts on /mnt/nvme_tmp7/jason_caspo/mc_prm_15b_dsr_sub_dense.

## What to build next (in priority order)

1. **Implement refresh-phase orchestrator**: bash + Python that runs RL for k steps, snapshots policy, kicks off Phase-B labeling on free GPUs, fine-tunes PRM, swaps PRM into RL.
2. **Run method D**: PPO + refreshed MC PRM on dsr_sub, same 500-step budget as vanilla GRPO. Expect ~11-12h on 8 GPUs (RL on 4-7, refresh on 0-3 in parallel).
3. **Run methods C, E**: frozen MC PRM (drop refresh), PRIME (re-implementation).
4. **AUC-vs-RL-step ablation**: take frozen dsr_sub PRM, evaluate AUC on rollouts from RLVR's step_0, step_100, step_200, step_300, step_400, step_500 policy ckpts. Plot AUC vs RL-step → smoking gun for staleness.
5. **Final eval**: run full eval suite on all method endpoints + step_200 / step_400 ckpts.

## Compute budget estimate

| Item | Compute | Wall clock |
|------|---------|------------|
| Vanilla GRPO (running) | 9 h × 4 GPU | done by 05:34 UTC |
| Method D (PPO + refresh) | 12 h × 4 GPU + 2.5 h × 4 GPU refresh (parallelized) | ~12 h |
| Method C (frozen MC PRM) | 9 h × 4 GPU | ~9 h |
| Method E (PRIME) | 1 day setup + 9 h × 4 GPU | ~2 days |
| AUC-vs-RL-step ablation | 30 min × 1 GPU | quick |
| Full eval suite all endpoints | 2-3 h × 7 GPU (parallel like prior) | ~3 h |

Total ~3-4 days of work to fully execute, most of it compute on GPUs that are otherwise idle overnight.

---

*Plan written 2026-04-30 after the literature search confirming this is novel territory (PRIME is closest prior art, focuses on implicit PRMs).*
