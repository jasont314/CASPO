"""Synthetic-tensor tests for :func:`caspo.value.train_value.ipvrm_loss`.

These exercise Eq. 9 of arXiv:2604.13197 directly, no model load needed.
"""

from __future__ import annotations

import math

import pytest
import torch

from caspo.value.train_value import compute_adb_dlw_factors, ipvrm_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _full_mask(B: int, R: int) -> torch.Tensor:
    return torch.ones(B, R, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_loss_zero_when_perfect_prediction() -> None:
    B, R = 4, 8
    margin = 5.0
    # Per-token signal so v_bar(t) is far inside the correct half-plane.
    # For positive rows: log_ratio = +20  → v_bar = +20 ≫ +m
    # For negative rows: log_ratio = -20  → v_bar = -20 ≪ -m
    log_ratio = torch.zeros(B, R, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        log_ratio[0:2] = 20.0
        log_ratio[2:4] = -20.0
    outcomes = torch.tensor([1.0, 1.0, 0.0, 0.0])
    mask = _full_mask(B, R)
    loss, stats = ipvrm_loss(log_ratio, mask, outcomes, margin)
    assert loss.item() < 1e-4, f"expected ~0 loss, got {loss.item()}"
    assert stats["acc_at_last"] == 1.0


def test_loss_increases_when_outcomes_flipped() -> None:
    B, R = 4, 8
    margin = 5.0
    log_ratio = torch.zeros(B, R, dtype=torch.float32)
    log_ratio[0:2] = 20.0
    log_ratio[2:4] = -20.0
    mask = _full_mask(B, R)

    correct = torch.tensor([1.0, 1.0, 0.0, 0.0])
    flipped = 1.0 - correct

    loss_correct, _ = ipvrm_loss(log_ratio.clone(), mask, correct, margin)
    loss_flipped, _ = ipvrm_loss(log_ratio.clone(), mask, flipped, margin)
    assert loss_flipped.item() > loss_correct.item() + 1.0


def test_loss_grad_flows_to_log_ratio() -> None:
    B, R = 3, 5
    log_ratio = torch.randn(B, R, requires_grad=True)
    mask = _full_mask(B, R)
    outcomes = torch.tensor([1.0, 0.0, 1.0])
    loss, _ = ipvrm_loss(log_ratio, mask, outcomes, margin=1.0)
    assert loss.requires_grad
    loss.backward()
    assert log_ratio.grad is not None
    assert torch.isfinite(log_ratio.grad).all()
    # Some entries must have nonzero gradient.
    assert log_ratio.grad.abs().sum().item() > 0.0


def test_padding_does_not_contribute() -> None:
    B, R_short = 2, 4
    margin = 2.0
    base = torch.tensor(
        [[0.1, -0.2, 0.3, -0.4], [-0.5, 0.6, -0.7, 0.8]],
        dtype=torch.float32,
    )
    mask_short = _full_mask(B, R_short)
    outcomes = torch.tensor([1.0, 0.0])
    loss_short, _ = ipvrm_loss(base.clone(), mask_short, outcomes, margin)

    # Same data padded with zeros + mask=0 on the extra positions.
    R_long = 7
    padded = torch.zeros(B, R_long, dtype=torch.float32)
    padded[:, :R_short] = base
    mask_long = torch.zeros(B, R_long, dtype=torch.float32)
    mask_long[:, :R_short] = 1.0
    loss_long, _ = ipvrm_loss(padded, mask_long, outcomes, margin)

    assert torch.allclose(loss_short, loss_long, atol=1e-6), (
        f"padding leaked: short={loss_short.item()} long={loss_long.item()}"
    )


def test_masked_nan_padding_does_not_poison_loss() -> None:
    log_ratio = torch.tensor([[0.2, float("nan")]], requires_grad=True)
    mask = torch.tensor([[1.0, 0.0]])
    outcomes = torch.tensor([1.0])

    loss, stats = ipvrm_loss(log_ratio, mask, outcomes, margin=1.0)

    assert torch.isfinite(loss).item()
    assert all(math.isfinite(v) for v in stats.values())
    loss.backward()
    assert torch.isfinite(log_ratio.grad).all().item()
    assert log_ratio.grad[0, 1].item() == 0.0


def test_all_padded_row_safe() -> None:
    B, R = 3, 5
    log_ratio = torch.randn(B, R, requires_grad=True)
    mask = torch.ones(B, R, dtype=torch.float32)
    mask[2] = 0.0  # row 2 is fully padded
    outcomes = torch.tensor([1.0, 0.0, 1.0])
    loss, stats = ipvrm_loss(log_ratio, mask, outcomes, margin=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert log_ratio.grad is not None
    # Row 2 should accumulate exactly zero gradient (it had no valid prefixes).
    assert torch.allclose(log_ratio.grad[2], torch.zeros(R), atol=1e-6)


def test_length_normalization() -> None:
    B = 2
    margin = 1.0
    # Per-token signal +0.5 / -0.5.  v_bar(T) = mean of per-token log_ratio = ±0.5
    # regardless of T because cumsum / t = signal.  So loss is approximately
    # invariant to T.
    R_short = 6
    short = torch.full((B, R_short), 0.5)
    short[1] = -0.5
    mask_short = _full_mask(B, R_short)
    outcomes = torch.tensor([1.0, 0.0])
    loss_short, _ = ipvrm_loss(short, mask_short, outcomes, margin)

    R_long = 12
    long = torch.full((B, R_long), 0.5)
    long[1] = -0.5
    mask_long = _full_mask(B, R_long)
    loss_long, _ = ipvrm_loss(long, mask_long, outcomes, margin)

    # Per-row averages of identical per-token logsigmoid values are equal,
    # so the total loss matches up to floating-point noise.
    assert math.isclose(loss_short.item(), loss_long.item(), rel_tol=1e-5, abs_tol=1e-6)


def test_adb_dlw_factors_shapes_and_logit() -> None:
    """ADB V_x = logit(μ); DLW w = (1-μ) for positives, μ for negatives."""
    # Two prompts, G=4 each. prompt 0: 3/4 correct → μ=0.75. prompt 1: 1/4 → μ=0.25.
    outcomes = torch.tensor([1.0, 1.0, 1.0, 0.0,    # prompt 0: μ=0.75
                              0.0, 1.0, 0.0, 0.0])  # prompt 1: μ=0.25
    prompt_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    V_x, w = compute_adb_dlw_factors(outcomes, prompt_index, eps=0.05)

    # V_x is per-prompt logit, broadcast to per-rollout.
    expected_logit_p0 = math.log(0.75) - math.log(0.25)   # ≈ +1.0986
    expected_logit_p1 = math.log(0.25) - math.log(0.75)   # ≈ -1.0986
    assert torch.allclose(V_x[:4], torch.tensor([expected_logit_p0] * 4), atol=1e-5)
    assert torch.allclose(V_x[4:], torch.tensor([expected_logit_p1] * 4), atol=1e-5)

    # DLW: prompt 0 positives (3 of them) → w = 1-0.75 = 0.25
    #      prompt 0 negative (1)         → w = 0.75
    #      prompt 1 negatives (3)        → w = 0.25
    #      prompt 1 positive (1)         → w = 0.75
    assert torch.allclose(w[:3], torch.tensor([0.25, 0.25, 0.25]), atol=1e-6)
    assert w[3].item() == pytest.approx(0.75)
    assert w[5].item() == pytest.approx(0.75)  # prompt 1's lone positive
    assert w[4].item() == pytest.approx(0.25)
    assert w[6].item() == pytest.approx(0.25)
    assert w[7].item() == pytest.approx(0.25)


def test_adb_dlw_eps_clamp_avoids_inf() -> None:
    """μ=0 or μ=1 must not produce ±inf logit; eps clamp handles it."""
    outcomes = torch.tensor([1.0, 1.0, 1.0, 1.0,    # μ=1.0 → clip to 0.95
                              0.0, 0.0, 0.0, 0.0])  # μ=0.0 → clip to 0.05
    prompt_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    V_x, w = compute_adb_dlw_factors(outcomes, prompt_index, eps=0.05)
    assert torch.isfinite(V_x).all()
    # logit(0.95) ≈ +2.944
    assert torch.allclose(V_x[:4], torch.full((4,), math.log(0.95) - math.log(0.05)), atol=1e-5)
    assert torch.allclose(V_x[4:], torch.full((4,), math.log(0.05) - math.log(0.95)), atol=1e-5)


def test_adb_shifts_loss_per_prompt_difficulty() -> None:
    """For an "easy" prompt (high μ), the loss for the rare wrong response
    should be larger than the same wrong response on a hard prompt — ADB's
    explicit goal."""
    margin = 1.0
    R = 6
    log_ratio = torch.full((1, R), -0.3)         # negative-leaning prefix
    mask = torch.ones(1, R)
    outcomes = torch.tensor([0.0])               # this rollout is wrong

    # Easy prompt: μ=0.8 → V_x = logit(0.8) ≈ +1.386
    V_x_easy = torch.tensor([math.log(0.8) - math.log(0.2)])
    loss_easy, _ = ipvrm_loss(
        log_ratio.clone().requires_grad_(False), mask, outcomes, margin,
        prompt_value_baseline=V_x_easy,
    )

    # Hard prompt: μ=0.2 → V_x = logit(0.2) ≈ -1.386
    V_x_hard = torch.tensor([math.log(0.2) - math.log(0.8)])
    loss_hard, _ = ipvrm_loss(
        log_ratio.clone().requires_grad_(False), mask, outcomes, margin,
        prompt_value_baseline=V_x_hard,
    )

    # On the EASY prompt, a wrong rollout is rare → bigger loss → bigger gradient.
    assert loss_easy.item() > loss_hard.item(), (
        f"expected easy-prompt wrong rollout to incur larger loss; "
        f"easy={loss_easy.item()}, hard={loss_hard.item()}"
    )


def test_dlw_downweights_majority_class() -> None:
    """DLW with w=(1-μ) on common-correct rollouts should shrink their loss
    contribution relative to uniform weighting."""
    margin = 1.0
    R = 6
    # μ=0.9 prompt — 9/10 correct, 1/10 wrong. Look at one of the common (correct) rollouts.
    log_ratio = torch.full((1, R), 0.2, requires_grad=False)
    mask = torch.ones(1, R)
    outcomes = torch.tensor([1.0])               # common-correct rollout

    loss_uniform, _ = ipvrm_loss(log_ratio.clone(), mask, outcomes, margin)
    loss_dlw, _ = ipvrm_loss(
        log_ratio.clone(), mask, outcomes, margin,
        loss_weights=torch.tensor([0.1]),        # w = 1 - 0.9 = 0.1
    )

    # DLW should shrink the loss magnitude by roughly 10x (since w=0.1).
    assert abs(loss_dlw.item()) < abs(loss_uniform.item()) * 0.5
    assert torch.isfinite(loss_dlw)


def test_eq15_recovers_eq9_when_baseline_zero_and_weights_one() -> None:
    """Eq. 15 with V_x=0, w=1 must equal Eq. 9 exactly."""
    B, R = 4, 6
    margin = 2.0
    log_ratio = torch.randn(B, R)
    mask = torch.ones(B, R)
    outcomes = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss_eq9, _ = ipvrm_loss(log_ratio.clone(), mask, outcomes, margin)
    loss_eq15, _ = ipvrm_loss(
        log_ratio.clone(), mask, outcomes, margin,
        prompt_value_baseline=torch.zeros(B),
        loss_weights=torch.ones(B),
    )
    assert torch.allclose(loss_eq9, loss_eq15, atol=1e-6)


def test_stats_signs() -> None:
    B, R = 6, 4
    log_ratio = torch.zeros(B, R)
    log_ratio[:3] = 0.7   # positive rows → v_bar > 0
    log_ratio[3:] = -0.7  # negative rows → v_bar < 0
    outcomes = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    mask = _full_mask(B, R)
    _, stats = ipvrm_loss(log_ratio, mask, outcomes, margin=0.1)
    assert stats["mean_v_bar_pos"] > stats["mean_v_bar_neg"]
    assert stats["mean_v_bar_pos"] > 0.0
    assert stats["mean_v_bar_neg"] < 0.0
    assert stats["acc_at_last"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Edge cases: extreme hyper-parameters and degenerate batches.
# ---------------------------------------------------------------------------

def test_extreme_beta_does_not_produce_nan() -> None:
    """β is baked into log_ratio = β·(log π_φ - log π_ref); huge magnitudes
    must not overflow logsigmoid or break the gradient."""
    B, R = 4, 6
    margin = 1.0
    # Simulate β=100 with a moderate (log π_φ - log π_ref) ≈ ±0.5 → log_ratio ≈ ±50,
    # and worst-case ±1.0 → log_ratio = ±100.
    base = torch.tensor(
        [
            [0.5, -0.4, 0.6, -0.5, 0.7, -0.3],
            [-0.5, 0.4, -0.6, 0.5, -0.7, 0.3],
            [1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    log_ratio = (100.0 * base).clone().requires_grad_(True)
    mask = _full_mask(B, R)
    outcomes = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss, stats = ipvrm_loss(log_ratio, mask, outcomes, margin)
    assert torch.isfinite(loss), f"loss not finite at β=100: {loss.item()}"
    assert all(math.isfinite(v) for v in stats.values())
    loss.backward()
    assert log_ratio.grad is not None
    assert torch.isfinite(log_ratio.grad).all(), "grad has nan/inf at β=100"


def test_zero_margin_still_produces_sensible_loss() -> None:
    """m=0 collapses Eq. 9 to plain BCE on v_bar; still must be finite,
    nonneg, monotonically smaller for correct rollouts than for flipped ones."""
    B, R = 4, 6
    margin = 0.0
    log_ratio = torch.zeros(B, R, dtype=torch.float32)
    log_ratio[0:2] = 1.0    # positive-leaning
    log_ratio[2:4] = -1.0   # negative-leaning
    mask = _full_mask(B, R)
    correct = torch.tensor([1.0, 1.0, 0.0, 0.0])
    flipped = 1.0 - correct

    loss_correct, stats = ipvrm_loss(log_ratio.clone(), mask, correct, margin)
    loss_flipped, _ = ipvrm_loss(log_ratio.clone(), mask, flipped, margin)
    assert torch.isfinite(loss_correct)
    assert torch.isfinite(loss_flipped)
    # log sigma(v_bar) is always negative, so loss = -mean(...) is positive.
    assert loss_correct.item() > 0.0
    # Correct prediction with v_bar=±1, m=0  → loss ≈ -log σ(1) ≈ 0.31.
    # Flipped: same magnitude on the wrong side → loss ≈ -log σ(-1) ≈ 1.31.
    assert loss_flipped.item() > loss_correct.item() + 0.5
    assert stats["acc_at_last"] == pytest.approx(1.0)
    # Gradient still flows (no dead-zone collapse to zero).
    log_ratio = log_ratio.clone().requires_grad_(True)
    loss, _ = ipvrm_loss(log_ratio, mask, correct, margin)
    loss.backward()
    assert log_ratio.grad is not None
    assert log_ratio.grad.abs().sum().item() > 0.0


def test_adb_dlw_all_zero_outcomes() -> None:
    """μ ≡ 0 across every prompt: V_x must clamp to logit(eps) (negative),
    DLW weights are all μ ≈ 0 → near-zero loss contribution; nothing inf."""
    outcomes = torch.zeros(8)
    prompt_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    eps = 0.05
    V_x, w = compute_adb_dlw_factors(outcomes, prompt_index, eps=eps)

    expected_logit = math.log(eps) - math.log(1.0 - eps)  # ≈ -2.944
    assert torch.isfinite(V_x).all()
    assert torch.allclose(V_x, torch.full((8,), expected_logit), atol=1e-5)
    # Every rollout is a negative ⇒ w = μ = 0 (before clip; the unclipped μ
    # is what feeds DLW per the source).
    assert torch.allclose(w, torch.zeros(8), atol=1e-6)

    # Plug into the loss: w=0 should annihilate the per-row contribution.
    B, R = 8, 5
    log_ratio = torch.randn(B, R)
    mask = _full_mask(B, R)
    loss, _ = ipvrm_loss(
        log_ratio, mask, outcomes, margin=1.0,
        prompt_value_baseline=V_x, loss_weights=w,
    )
    assert torch.isfinite(loss)
    assert abs(loss.item()) < 1e-6, f"all-w=0 should give ~0 loss, got {loss.item()}"


def test_adb_dlw_all_one_outcomes() -> None:
    """μ ≡ 1 across every prompt: V_x clamps to logit(1-eps) (positive),
    DLW weights are all (1-μ) ≈ 0; loss must remain finite and tiny."""
    outcomes = torch.ones(8)
    prompt_index = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    eps = 0.05
    V_x, w = compute_adb_dlw_factors(outcomes, prompt_index, eps=eps)

    expected_logit = math.log(1.0 - eps) - math.log(eps)  # ≈ +2.944
    assert torch.isfinite(V_x).all()
    assert torch.allclose(V_x, torch.full((8,), expected_logit), atol=1e-5)
    # Every rollout is a positive ⇒ w = 1 - μ = 0.
    assert torch.allclose(w, torch.zeros(8), atol=1e-6)

    B, R = 8, 5
    log_ratio = torch.randn(B, R)
    mask = _full_mask(B, R)
    loss, stats = ipvrm_loss(
        log_ratio, mask, outcomes, margin=1.0,
        prompt_value_baseline=V_x, loss_weights=w,
    )
    assert torch.isfinite(loss)
    assert all(math.isfinite(v) for v in stats.values())
    assert abs(loss.item()) < 1e-6, f"all-w=0 should give ~0 loss, got {loss.item()}"


def test_ipvrm_loss_single_prompt_G1() -> None:
    """G=1 edge case: a single rollout for a single prompt. μ is either 0
    or 1, both must be handled by the eps clamp. The loss itself with B=1
    must still produce a scalar with grad."""
    # ---- G=1 with a correct rollout (μ=1.0 → clipped) ----
    outcomes = torch.tensor([1.0])
    prompt_index = torch.tensor([0], dtype=torch.long)
    eps = 0.05
    V_x, w = compute_adb_dlw_factors(outcomes, prompt_index, eps=eps)
    assert V_x.shape == (1,)
    assert w.shape == (1,)
    assert torch.isfinite(V_x).all()
    assert V_x.item() == pytest.approx(math.log(1.0 - eps) - math.log(eps), abs=1e-5)
    # w = 1 - μ = 0 for a correct lone rollout.
    assert w.item() == pytest.approx(0.0, abs=1e-6)

    # ---- G=1 with a wrong rollout (μ=0.0 → clipped) ----
    outcomes_wrong = torch.tensor([0.0])
    V_x_w, w_w = compute_adb_dlw_factors(outcomes_wrong, prompt_index, eps=eps)
    assert torch.isfinite(V_x_w).all()
    assert V_x_w.item() == pytest.approx(math.log(eps) - math.log(1.0 - eps), abs=1e-5)
    # w = μ = 0 for a wrong lone rollout.
    assert w_w.item() == pytest.approx(0.0, abs=1e-6)

    # ---- B=1 loss path: scalar shape, grad flows, no nans ----
    R = 5
    log_ratio = torch.randn(1, R, requires_grad=True)
    mask = _full_mask(1, R)
    loss, stats = ipvrm_loss(log_ratio, mask, outcomes, margin=1.0)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert all(math.isfinite(v) for v in stats.values())
    loss.backward()
    assert log_ratio.grad is not None
    assert torch.isfinite(log_ratio.grad).all()
    assert log_ratio.grad.abs().sum().item() > 0.0

    # ---- Same B=1 with the (degenerate) ADB+DLW factors plugged in. ----
    # w=0 zeroes the row; loss must be ~0 but still scalar/finite.
    log_ratio2 = torch.randn(1, R, requires_grad=True)
    loss_adb, _ = ipvrm_loss(
        log_ratio2, mask, outcomes, margin=1.0,
        prompt_value_baseline=V_x, loss_weights=w,
    )
    assert torch.isfinite(loss_adb)
    assert abs(loss_adb.item()) < 1e-6
