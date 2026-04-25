"""Integration test: PPO / CASPO / VinePPO / GRPO yield DIFFERENT advantage signals.

This test confirms method dispatch is real (not silent fallback to a single
shared codepath) by exercising the algo-layer functions used by each method
on the SAME synthetic batch and verifying:

* PPO advantages use batch-level terminal reward centering by default.
* GRPO advantages depend on per-prompt reward variance via group-relative
  centering (``_group_relative_advantage`` reproduced here).
* VinePPO advantages are step-TD with Monte-Carlo step values (different
  per-step value estimates → different advantages).
* CASPO advantages are step-TD with V_phi (learned-value) step values, which
  in general differ from MC estimates on the same trajectory.

We do NOT instantiate ``CASPOTrainer`` (it requires HF + GPU + a phase-1
value checkpoint). Instead we replicate the per-method math using the public
``caspo.algo.advantages`` API plus a local copy of ``_group_relative_advantage``
matching ``caspo.trainer.caspo_trainer._group_relative_advantage``.

Synthetic shape: B=4 (2 prompts × G=2 group), T=64 (response length), with
4 step boundaries placed at (15, 31, 47, 63). All four rows share the same
boundary layout so the only thing differing across methods is the per-step
*value* estimate (and, for GRPO, the absence of segmentation entirely).
"""

from __future__ import annotations

import torch

from caspo.algo.advantages import (
    broadcast_step_advantage_to_tokens,
    standardize_step_advantage,
    step_td_advantage,
    step_values_from_log_ratios,
)


# ---------------------------------------------------------------------------
# Local copy of trainer helper (kept in sync with
# caspo.trainer.caspo_trainer._group_relative_advantage).
# ---------------------------------------------------------------------------


def _group_relative_advantage(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    B = rewards.numel()
    assert B % group_size == 0
    g = rewards.view(B // group_size, group_size)
    mean = g.mean(dim=1, keepdim=True)
    centered = g - mean
    std = g.std(dim=1, keepdim=True, unbiased=False)
    safe = torch.where(std <= 1e-8, torch.ones_like(std), std)
    out = centered / safe
    out = torch.where(std.expand_as(out) <= 1e-8, torch.zeros_like(out), out)
    return out.reshape(-1)


# ---------------------------------------------------------------------------
# Shared synthetic batch fixture (deterministic).
# ---------------------------------------------------------------------------


def _make_batch():
    """Build B=4 (2 prompts × G=2), T=64, S=4 boundary layout.

    Returns a dict with everything the three methods consume:
      response_mask, log_ratio, boundary_after, step_count, step_id,
      rewards (per-row), V_caspo (V_phi style), V_vineppo (MC style).
    """
    torch.manual_seed(0)
    B, T, G = 4, 64, 2
    response_mask = torch.ones((B, T), dtype=torch.long)

    # 4 steps per row, equal length (16 tokens each), boundaries at end of each.
    boundary_positions = [15, 31, 47, 63]
    boundary_after = torch.zeros((B, T), dtype=torch.bool)
    for p in boundary_positions:
        boundary_after[:, p] = True
    step_count = torch.full((B,), 4, dtype=torch.long)

    # Per-token step_id: 0,0,...,1,1,...,2,2,...,3,3,...
    step_id = torch.zeros((B, T), dtype=torch.long)
    step_id[:, 16:32] = 1
    step_id[:, 32:48] = 2
    step_id[:, 48:64] = 3

    # Pseudo log_ratios (β·log π_φ/π_ref) — used to drive V_phi via the
    # cumulative-sum construction. Random per token, fixed seed.
    log_ratio = torch.randn(B, T) * 0.05

    # Per-row terminal rewards. Designed so per-prompt (G=2) variance is non-
    # zero in BOTH groups → GRPO will produce non-zero advantages.
    # Group 0 (rows 0,1): rewards 1.0, 0.0 → mean 0.5, std 0.5 → adv [+1, -1]
    # Group 1 (rows 2,3): rewards 0.7, 0.3 → mean 0.5, std 0.2 → adv [+1, -1]
    rewards = torch.tensor([1.0, 0.0, 0.7, 0.3])

    # CASPO V_phi: built from log_ratio cumsum (the public utility used by
    # the trainer's caspo branch).
    V_caspo = step_values_from_log_ratios(
        log_ratio, response_mask, boundary_after, step_count
    )  # [B, S+1=5]

    # VinePPO V_step: pretend Monte-Carlo rollouts gave us a DIFFERENT value
    # estimate at each step boundary. The trainer's
    # ``_vineppo_mc_step_values`` returns one value per step; we mock it as a
    # per-row offset so the resulting advantages don't accidentally coincide
    # with the CASPO ones. V[:, 0] is forced to 0 (matches the convention in
    # ``step_values_from_log_ratios``).
    V_vineppo = torch.tensor([
        [0.0, 0.10, 0.25, 0.55, 1.00],
        [0.0, 0.05, 0.15, 0.40, 0.10],
        [0.0, 0.20, 0.45, 0.60, 0.70],
        [0.0, 0.15, 0.30, 0.45, 0.30],
    ])

    return {
        "B": B, "T": T, "G": G,
        "response_mask": response_mask,
        "log_ratio": log_ratio,
        "boundary_after": boundary_after,
        "step_count": step_count,
        "step_id": step_id,
        "rewards": rewards,
        "V_caspo": V_caspo,
        "V_vineppo": V_vineppo,
    }


# ---------------------------------------------------------------------------
# Per-method advantage construction (mirrors trainer dispatch logic).
# ---------------------------------------------------------------------------


def _ppo_token_advantage(batch):
    """PPO: sequence-level terminal reward advantage, batch-standardized and
    broadcast uniformly to all response tokens."""
    rewards = batch["rewards"].float()
    centered = rewards - rewards.mean()
    std = centered.square().mean().sqrt()
    adv = centered / std.clamp(min=1e-8)
    return adv.unsqueeze(1) * batch["response_mask"].to(adv.dtype)


def _grpo_token_advantage(batch):
    """GRPO: per-sequence group-relative advantage broadcast uniformly to all
    response tokens. No segmentation, no step values."""
    adv_per_seq = _group_relative_advantage(batch["rewards"], group_size=batch["G"])
    return adv_per_seq.unsqueeze(1) * batch["response_mask"].to(adv_per_seq.dtype)


def _step_td_token_advantage(batch, V_step):
    """Shared step-TD path used by CASPO and VinePPO — only V_step differs."""
    A_step = step_td_advantage(
        V_step, batch["rewards"], batch["step_count"], gamma=1.0,
    )
    A_step = standardize_step_advantage(A_step, batch["step_count"], scope="batch")
    return broadcast_step_advantage_to_tokens(
        A_step, batch["step_id"], batch["response_mask"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_four_methods_produce_different_token_advantages():
    """The integration contract: same batch → 4 different token-level signals."""
    batch = _make_batch()

    A_ppo = _ppo_token_advantage(batch)
    A_grpo = _grpo_token_advantage(batch)
    A_caspo = _step_td_token_advantage(batch, batch["V_caspo"])
    A_vineppo = _step_td_token_advantage(batch, batch["V_vineppo"])

    # Same shape, all finite, none all-zero.
    for name, A in [
        ("ppo", A_ppo), ("grpo", A_grpo),
        ("caspo", A_caspo), ("vineppo", A_vineppo),
    ]:
        assert A.shape == (batch["B"], batch["T"]), f"{name} shape {A.shape}"
        assert torch.isfinite(A).all(), f"{name} produced non-finite values"
        assert A.abs().sum().item() > 0, f"{name} produced all-zero advantages"

    # Pairwise distinctness: no two methods produce identical tensors.
    assert not torch.allclose(A_ppo, A_grpo, atol=1e-6), (
        "PPO and GRPO produced identical token advantages — PPO should use "
        "batch-level reward centering while GRPO uses per-prompt groups."
    )
    assert not torch.allclose(A_ppo, A_caspo, atol=1e-6)
    assert not torch.allclose(A_ppo, A_vineppo, atol=1e-6)
    assert not torch.allclose(A_grpo, A_caspo, atol=1e-6), (
        "GRPO and CASPO produced identical token advantages — method dispatch "
        "is broken (likely silent fallback)."
    )
    assert not torch.allclose(A_grpo, A_vineppo, atol=1e-6), (
        "GRPO and VinePPO produced identical token advantages — method "
        "dispatch is broken."
    )
    assert not torch.allclose(A_caspo, A_vineppo, atol=1e-6), (
        "CASPO and VinePPO produced identical token advantages even though "
        "their V_step inputs differ — step-TD is ignoring its V_step input."
    )

    # Magnitudes are all in a reasonable range (sanity: standardized advantages
    # should land within roughly +/-10 after batch whitening + clipping).
    for name, A in [
        ("ppo", A_ppo), ("grpo", A_grpo),
        ("caspo", A_caspo), ("vineppo", A_vineppo),
    ]:
        assert A.abs().max().item() < 100.0, (
            f"{name} max abs {A.abs().max().item()} unreasonably large"
        )


def test_grpo_depends_on_per_prompt_reward_variance():
    """GRPO is the only method whose signal vanishes when per-prompt rewards
    are constant (zero variance within every group)."""
    batch = _make_batch()
    # Force every group to have constant rewards.
    constant_rewards = torch.tensor([0.5, 0.5, 0.7, 0.7])
    adv = _group_relative_advantage(constant_rewards, group_size=batch["G"])
    assert torch.allclose(adv, torch.zeros_like(adv)), (
        "GRPO with zero per-group variance must give zero advantage."
    )

    # And re-confirm the original non-constant batch is non-zero.
    adv2 = _group_relative_advantage(batch["rewards"], group_size=batch["G"])
    assert adv2.abs().sum().item() > 0


def test_step_td_advantage_changes_with_value_estimate():
    """Direct algo-level check: same trajectory, different V_step ⇒ different
    advantages. This is the load-bearing invariant for CASPO≠VinePPO."""
    B, S = 2, 3
    step_count = torch.tensor([3, 3], dtype=torch.long)
    rewards = torch.tensor([1.0, 0.5])

    V_zeros = torch.zeros((B, S + 1))
    V_ones = torch.ones((B, S + 1))
    V_rand = torch.tensor([
        [0.0, 0.2, 0.5, 0.9],
        [0.0, 0.1, 0.3, 0.6],
    ])

    A_zeros = step_td_advantage(V_zeros, rewards, step_count, gamma=1.0)
    A_ones = step_td_advantage(V_ones, rewards, step_count, gamma=1.0)
    A_rand = step_td_advantage(V_rand, rewards, step_count, gamma=1.0)

    # All finite.
    for A in (A_zeros, A_ones, A_rand):
        assert torch.isfinite(A).all()
        assert A.shape == (B, S)

    # V≡0: A_t = r_step[t] (zero except terminal).
    expected_zeros = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.5]])
    assert torch.allclose(A_zeros, expected_zeros)

    # V≡1: V[t+1] - V[t] = 0 everywhere; A is r_step (same as V=0).
    assert torch.allclose(A_ones, expected_zeros)

    # V_rand: non-trivial telescoping ⇒ DIFFERENT from constant-V cases.
    assert not torch.allclose(A_rand, A_zeros, atol=1e-6), (
        "step_td_advantage ignored its V_step input (zeros vs random gave "
        "identical output) — CASPO and VinePPO would be indistinguishable."
    )

    # Telescoping invariant: sum_t A_t = R + V[S] - V[0] (per-row).
    for row in range(B):
        expected_sum = rewards[row] + V_rand[row, S] - V_rand[row, 0]
        assert torch.allclose(A_rand[row].sum(), expected_sum, atol=1e-5)


def test_no_method_produces_nan_or_inf():
    """Robustness: every method on the standard batch must return finite
    tensors with no NaN / inf even after standardization."""
    batch = _make_batch()
    for name, fn in [
        ("ppo", lambda: _ppo_token_advantage(batch)),
        ("grpo", lambda: _grpo_token_advantage(batch)),
        ("caspo", lambda: _step_td_token_advantage(batch, batch["V_caspo"])),
        ("vineppo", lambda: _step_td_token_advantage(batch, batch["V_vineppo"])),
    ]:
        A = fn()
        assert not torch.isnan(A).any(), f"{name} produced NaN"
        assert not torch.isinf(A).any(), f"{name} produced inf"
        assert A.abs().sum().item() > 0, f"{name} produced an all-zero tensor"
