"""Unit tests for caspo.algo.ppo_loss."""

from __future__ import annotations

import math

import torch

from caspo.algo.ppo_loss import ppo_clipped_loss, k3_kl_estimator


def test_zero_advantage_zero_loss():
    B, R = 2, 4
    logp = torch.zeros(B, R, requires_grad=True)
    old = torch.zeros(B, R)
    A = torch.zeros(B, R)
    mask = torch.ones(B, R)
    loss, stats = ppo_clipped_loss(logp, old, A, mask)
    assert loss.item() == 0.0
    # Gradients should flow even though loss is zero (no error).
    loss.backward()
    assert logp.grad is not None


def test_clip_lower_bound():
    # ratio = 2.0 (logp - old = ln 2), clip_eps_high=0.2 → clipped = 1.2.
    # A > 0 → unclipped*A = 2A, clipped*A = 1.2A; min = 1.2A; loss = -1.2A.
    B, R = 1, 1
    logp = torch.full((B, R), math.log(2.0), requires_grad=True)
    old = torch.zeros(B, R)
    A_val = 3.0
    A = torch.full((B, R), A_val)
    mask = torch.ones(B, R)
    loss, stats = ppo_clipped_loss(
        logp, old, A, mask, clip_eps_low=0.2, clip_eps_high=0.2
    )
    expected = -1.2 * A_val
    assert abs(loss.item() - expected) < 1e-5
    # Clip frac should be 1 (the only valid token was clipped).
    assert abs(stats["clip_frac"].item() - 1.0) < 1e-5


def test_clip_upper_bound():
    # ratio = 0.5, A = -1.
    # unclipped = 0.5 * -1 = -0.5
    # clipped = clamp(0.5, 0.8, 1.2) * -1 = 0.8 * -1 = -0.8
    # min = -0.8; loss = -min = 0.8.
    B, R = 1, 1
    logp = torch.full((B, R), math.log(0.5), requires_grad=True)
    old = torch.zeros(B, R)
    A = torch.full((B, R), -1.0)
    mask = torch.ones(B, R)
    loss, stats = ppo_clipped_loss(
        logp, old, A, mask, clip_eps_low=0.2, clip_eps_high=0.2
    )
    assert abs(loss.item() - 0.8) < 1e-5
    assert abs(stats["clip_frac"].item() - 1.0) < 1e-5


def test_mask_excludes_padding():
    # Build a batch where masked positions have wildly different logprobs;
    # loss should not change when those positions are perturbed.
    B, R = 2, 5
    logp = torch.zeros(B, R, requires_grad=True)
    old = torch.zeros(B, R)
    A = torch.full((B, R), 0.5)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.float)
    loss_a, _ = ppo_clipped_loss(logp, old, A, mask)

    # Now perturb only masked positions in a fresh tensor.
    logp2 = torch.zeros(B, R, requires_grad=True)
    with torch.no_grad():
        logp2[0, 3:] = 5.0
        logp2[1, 2:] = -3.0
    loss_b, _ = ppo_clipped_loss(logp2, old, A, mask)
    assert abs(loss_a.item() - loss_b.item()) < 1e-6


def test_clip_frac_stat():
    # Build a batch where exactly half the valid positions are clipped.
    # 4 tokens, 2 with ratio outside [0.8, 1.2], 2 inside.
    B, R = 1, 4
    logp = torch.tensor([[math.log(2.0), math.log(1.0), math.log(0.5), math.log(1.0)]],
                        requires_grad=True)
    old = torch.zeros(B, R)
    A = torch.ones(B, R)  # A>0 to ensure clipping is "active" on the deviating ratios
    mask = torch.ones(B, R)
    loss, stats = ppo_clipped_loss(logp, old, A, mask)
    assert abs(stats["clip_frac"].item() - 0.5) < 1e-5


def test_grad_flows():
    B, R = 2, 3
    logp = torch.randn(B, R, requires_grad=True)
    old = torch.randn(B, R)
    A = torch.randn(B, R)
    mask = torch.ones(B, R)
    loss, _ = ppo_clipped_loss(logp, old, A, mask)
    loss.backward()
    assert logp.grad is not None
    assert torch.any(logp.grad != 0)


def test_kl_k3_nonneg():
    torch.manual_seed(0)
    logp = torch.randn(8, 16) * 2.0
    logp_ref = torch.randn(8, 16) * 2.0
    kl = k3_kl_estimator(logp, logp_ref)
    assert torch.all(kl >= 0)


def test_kl_term_added():
    # With kl_coef=1 and ref_logprobs given, loss = pg_loss + mean(kl).
    B, R = 1, 2
    logp = torch.zeros(B, R, requires_grad=True)
    old = torch.zeros(B, R)
    A = torch.zeros(B, R)
    mask = torch.ones(B, R)
    ref = torch.tensor([[1.0, -1.0]])
    loss, stats = ppo_clipped_loss(
        logp, old, A, mask,
        ref_logprobs=ref, kl_coef=1.0, kl_estimator="k3",
    )
    # pg_loss should be 0 since A=0.
    assert abs(stats["pg_loss"].item()) < 1e-6
    # k3 KL element-wise: exp(ref - logp) - (ref - logp) - 1
    kl = torch.exp(ref - logp.detach()) - (ref - logp.detach()) - 1.0
    expected_mean_kl = kl.mean().item()
    assert abs(stats["mean_kl"].item() - expected_mean_kl) < 1e-5
    assert abs(loss.item() - expected_mean_kl) < 1e-5


def test_extreme_ratio_no_nan():
    # Massive policy drift in both directions: log-ratio of ±ln(1e10) is
    # well past the ±20 internal clamp. Loss/stats must remain finite —
    # an unclamped exp here would overflow to inf and poison the batch.
    B, R = 2, 3
    big = math.log(1e10)
    small = math.log(1e-10)
    # Mix: some catastrophic upward, some catastrophic downward, some sane.
    logp = torch.tensor(
        [[big, small, 0.0], [small, big, 0.0]], requires_grad=True
    )
    old = torch.zeros(B, R)
    A = torch.tensor([[1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    mask = torch.ones(B, R)
    loss, stats = ppo_clipped_loss(logp, old, A, mask)
    assert torch.isfinite(loss).item(), f"loss not finite: {loss.item()}"
    for name, v in stats.items():
        assert torch.isfinite(v).all().item(), f"stat {name} not finite: {v}"
    # And with KL on top: exp of huge negative log-ratio in k3 must not blow up.
    ref = torch.zeros(B, R)
    loss2, stats2 = ppo_clipped_loss(
        logp, old, A, mask,
        ref_logprobs=ref, kl_coef=0.1, kl_estimator="k3",
    )
    assert torch.isfinite(loss2).item()
    assert torch.isfinite(stats2["mean_kl"]).item()
    # Gradients must flow and be finite for tokens whose log-ratio is in-clamp.
    loss2.backward()
    assert logp.grad is not None
    assert torch.isfinite(logp.grad).all().item()


def test_very_negative_advantage_clipping():
    # ratio = 1.5 (above the 1+eps_high=1.2 band), A very negative.
    # unclipped = 1.5 * A; clipped = 1.2 * A.
    # For A < 0: 1.5*A < 1.2*A, so min picks unclipped → loss = -1.5*A = 1.5*|A|.
    # i.e. when ratio is *above* the band and A<0, the clip does NOT bind
    # (PPO punishes the model fully — this is the pessimistic direction).
    B, R = 1, 1
    logp = torch.full((B, R), math.log(1.5), requires_grad=True)
    old = torch.zeros(B, R)
    A_val = -1e6
    A = torch.full((B, R), A_val)
    mask = torch.ones(B, R)
    loss, stats = ppo_clipped_loss(
        logp, old, A, mask, clip_eps_low=0.2, clip_eps_high=0.2
    )
    expected = -1.5 * A_val  # = 1.5e6
    assert abs(loss.item() - expected) / abs(expected) < 1e-5
    assert torch.isfinite(loss).item()
    # clip_frac in this loss reports "ratio left the clip band", not
    # "clip was binding under min" — so it's 1 here even though min picked
    # the unclipped surrogate (PPO's pessimistic-update direction).
    assert abs(stats["clip_frac"].item() - 1.0) < 1e-5

    # Now flip: ratio = 0.5 (below 1-eps_low=0.8) with very negative A.
    # unclipped = 0.5*A; clipped = 0.8*A. For A<0: 0.5*A > 0.8*A, so min
    # picks clipped → loss = -0.8*A = 0.8*|A|. Clip *is* binding.
    logp2 = torch.full((B, R), math.log(0.5), requires_grad=True)
    loss2, stats2 = ppo_clipped_loss(
        logp2, old, A, mask, clip_eps_low=0.2, clip_eps_high=0.2
    )
    expected2 = -0.8 * A_val  # = 8e5
    assert abs(loss2.item() - expected2) / abs(expected2) < 1e-5
    assert abs(stats2["clip_frac"].item() - 1.0) < 1e-5


def test_kl_scales_with_coef():
    # With pg_loss=0, total loss = kl_coef * mean_kl. Scan kl_coef and
    # verify the total tracks coefficient linearly and mean_kl is invariant.
    B, R = 2, 4
    torch.manual_seed(7)
    logp = (torch.randn(B, R) * 0.5).requires_grad_(True)
    old = logp.detach().clone()  # ratio = 1 → pg_loss = 0 for any A
    A = torch.zeros(B, R)
    mask = torch.ones(B, R)
    ref = torch.randn(B, R) * 0.5

    # Baseline mean_kl from kl_coef=0 (KL still computed since ref is given).
    _, stats0 = ppo_clipped_loss(
        logp, old, A, mask, ref_logprobs=ref, kl_coef=0.0, kl_estimator="k3",
    )
    base_kl = stats0["mean_kl"].item()
    assert "kl_term" not in stats0  # kl_term only emitted when coef != 0
    assert abs(stats0["pg_loss"].item()) < 1e-6

    for coef in (0.01, 0.1, 1.0, 10.0):
        loss, stats = ppo_clipped_loss(
            logp, old, A, mask,
            ref_logprobs=ref, kl_coef=coef, kl_estimator="k3",
        )
        # mean_kl is the KL itself; should not depend on coef.
        assert abs(stats["mean_kl"].item() - base_kl) < 1e-6
        # kl_term reflects the scaling.
        assert abs(stats["kl_term"].item() - coef * base_kl) < 1e-5
        # Total loss = 0 (pg) + coef * mean_kl.
        assert abs(loss.item() - coef * base_kl) < 1e-5


def test_per_token_mean_mostly_padded():
    # Token-mean reduction must divide by the count of *valid* tokens, not
    # by B*R. Build a 4x16 batch with only 3 valid tokens total: the
    # reported pg_loss should equal the mean of those 3 per-token losses.
    B, R = 4, 16
    logp = torch.zeros(B, R, requires_grad=True)
    old = torch.zeros(B, R)
    # Distinct advantages on the three valid positions; rest is padding garbage.
    A = torch.zeros(B, R)
    A[0, 0] = 2.0
    A[1, 5] = -3.0
    A[3, 15] = 4.0
    mask = torch.zeros(B, R)
    mask[0, 0] = 1.0
    mask[1, 5] = 1.0
    mask[3, 15] = 1.0

    # Ratio = 1 → unclipped = clipped = A → per_token_loss = -A.
    expected = -(2.0 + -3.0 + 4.0) / 3.0  # = -1.0
    loss, stats = ppo_clipped_loss(logp, old, A, mask)
    assert abs(loss.item() - expected) < 1e-6
    assert abs(stats["pg_loss"].item() - expected) < 1e-6
    # mean_advantage and mean_ratio also use the valid-token denominator.
    assert abs(stats["mean_advantage"].item() - (2.0 - 3.0 + 4.0) / 3.0) < 1e-6
    assert abs(stats["mean_ratio"].item() - 1.0) < 1e-6
    # No NaN/Inf despite 61 of 64 entries being masked-out garbage.
    assert torch.isfinite(loss).item()


def test_kl_estimators_k1_and_k3_sane():
    # Both estimators should produce finite numbers under modest drift,
    # and k3 should be non-negative element-wise. k1 mean is signed
    # log-ratio mean; for our setup it should match logp-ref mean.
    torch.manual_seed(13)
    B, R = 3, 8
    logp = (torch.randn(B, R) * 0.3).requires_grad_(True)
    old = logp.detach().clone()
    A = torch.zeros(B, R)
    mask = torch.ones(B, R)
    ref = torch.randn(B, R) * 0.3

    # k3 path
    loss_k3, stats_k3 = ppo_clipped_loss(
        logp, old, A, mask,
        ref_logprobs=ref, kl_coef=0.5, kl_estimator="k3",
    )
    assert torch.isfinite(loss_k3).item()
    assert torch.isfinite(stats_k3["mean_kl"]).item()
    # k3 element-wise non-negativity is enforced by k3_kl_estimator;
    # mean is therefore non-negative too.
    assert stats_k3["mean_kl"].item() >= 0.0
    expected_k3 = (
        torch.exp(ref - logp.detach()) - (ref - logp.detach()) - 1.0
    ).mean().item()
    assert abs(stats_k3["mean_kl"].item() - expected_k3) < 1e-5

    # k1 path — fresh tensor to avoid grad accumulation across invocations.
    logp2 = logp.detach().clone().requires_grad_(True)
    loss_k1, stats_k1 = ppo_clipped_loss(
        logp2, old, A, mask,
        ref_logprobs=ref, kl_coef=0.5, kl_estimator="k1",
    )
    assert torch.isfinite(loss_k1).item()
    assert torch.isfinite(stats_k1["mean_kl"]).item()
    expected_k1 = (logp2.detach() - ref).mean().item()
    assert abs(stats_k1["mean_kl"].item() - expected_k1) < 1e-5
    # Both should backprop cleanly.
    loss_k3.backward()
    loss_k1.backward()
    assert torch.isfinite(logp.grad).all().item()
    assert torch.isfinite(logp2.grad).all().item()
