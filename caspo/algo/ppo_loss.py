"""PPO clipped surrogate loss for CASPO.

Pure PPO with step-level TD advantage broadcast to tokens (no entropy
shaping; no DAPO quadrant masking). Decoupled clip ε_low / ε_high allowed.
Token-mean reduction over all valid tokens in the batch.

KL term to π_ref is optional. The "k3" estimator (Schulman 2020,
"Approximating KL Divergence", http://joschu.net/blog/kl-approx.html) is
unbiased and always non-negative; "k1" is the signed log-ratio.
"""

from __future__ import annotations

from typing import Optional

import torch


# Clamp range for log-ratios fed into ``exp``. fp32 ``exp`` overflows past
# ~88; we use a tighter bound so squared / multiplied downstream values
# stay well below fp32 inf and bf16 stays safe too. At |diff| = 20 the
# k3 estimator is already ~5e8, which dominates any sane KL budget.
_LOG_RATIO_CLAMP = 20.0


def k3_kl_estimator(
    logp: torch.Tensor,
    logp_ref: torch.Tensor,
) -> torch.Tensor:
    """Schulman's k3 unbiased low-variance KL estimator.

    .. math::
        \\widehat{KL}_t = \\exp(\\log\\pi_{ref} - \\log\\pi_\\theta)
            - (\\log\\pi_{ref} - \\log\\pi_\\theta) - 1

    Element-wise non-negative; in expectation under π_θ this estimates
    KL(π_θ || π_ref). The log-ratio is clamped to ±20 before ``exp`` so
    that catastrophic policy drift cannot produce ``inf``/``nan`` (fp32
    ``exp`` overflows above ~88, bf16 above ~11 — clamping at 20 is
    safe in fp32 and the caller is expected to cast bf16 inputs to fp32
    before calling, which trainer code does).

    Args:
        logp: ``[B, R]`` log π_θ on the sampled tokens.
        logp_ref: ``[B, R]`` log π_ref on the same tokens.

    Returns:
        ``[B, R]`` non-negative KL estimate, same dtype/device as ``logp``.
    """
    if logp.shape != logp_ref.shape:
        raise ValueError(
            f"shape mismatch: logp {tuple(logp.shape)} vs logp_ref {tuple(logp_ref.shape)}"
        )
    if logp.device != logp_ref.device:
        raise ValueError(
            f"device mismatch: logp on {logp.device} vs logp_ref on {logp_ref.device}"
        )
    diff = logp_ref - logp  # log(ref/θ)
    # Clamp to avoid exp overflow. Gradients past the boundary are zero,
    # which is the desired behavior under runaway drift (don't propagate
    # an exploding KL gradient).
    diff_safe = torch.clamp(diff, min=-_LOG_RATIO_CLAMP, max=_LOG_RATIO_CLAMP)
    out = torch.exp(diff_safe) - diff_safe - 1.0
    # Algebraically non-negative; clamp to 0 to scrub fp rounding noise so
    # downstream "KL >= 0" assertions hold exactly.
    return out.clamp_min(0.0)


def _k1_kl(logp: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """k1 KL estimator: signed log-ratio ``logπ_θ - logπ_ref``.

    In expectation under π_θ this equals ``KL(π_θ || π_ref)``, but the
    per-token estimate is signed (high variance). Use k3 unless you need
    the unbiased signed form (e.g. for diagnostics).
    """
    if logp.shape != logp_ref.shape:
        raise ValueError(
            f"shape mismatch: logp {tuple(logp.shape)} vs logp_ref {tuple(logp_ref.shape)}"
        )
    if logp.device != logp_ref.device:
        raise ValueError(
            f"device mismatch: logp on {logp.device} vs logp_ref on {logp_ref.device}"
        )
    return logp - logp_ref


def ppo_clipped_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantage: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    clip_eps_low: float = 0.2,
    clip_eps_high: float = 0.2,
    ref_logprobs: Optional[torch.Tensor] = None,
    kl_coef: float = 0.0,
    kl_estimator: str = "k3",
    ratio_threshold: Optional[float] = 10.0,
) -> tuple[torch.Tensor, dict]:
    """PPO clipped surrogate, token-mean reduction.

    .. math::
        \\rho_t = \\exp(\\log\\pi_\\theta - \\log\\pi_{old})
        L_t   = -\\min(\\rho_t \\hat A_t, \\mathrm{clip}(\\rho_t, 1 - \\varepsilon_{lo},
                                                  1 + \\varepsilon_{hi}) \\hat A_t)
        L     = \\frac{\\sum_t L_t m_t}{\\sum_t m_t}
              + \\beta_{KL} \\frac{\\sum_t \\hat{KL}_t m_t}{\\sum_t m_t}\\;\\;
                (\\text{if } \\beta_{KL} > 0)

    Args:
        logprobs: ``[B, R]`` log π_θ on the sampled response tokens (REQUIRES grad).
        old_logprobs: ``[B, R]`` log π_old on the same tokens (no grad).
        advantage: ``[B, R]`` per-token advantage.
        response_mask: ``[B, R]`` 1 on valid response positions, 0 on padding.
        clip_eps_low: lower clip half-width.
        clip_eps_high: upper clip half-width.
        ref_logprobs: optional ``[B, R]`` log π_ref for the KL term.
        kl_coef: coefficient on the KL term (added to loss when > 0).
        kl_estimator: ``"k1"`` or ``"k3"``.

    Returns:
        ``(loss, stats)``.
    """
    if logprobs.shape != old_logprobs.shape \
            or logprobs.shape != advantage.shape \
            or logprobs.shape != response_mask.shape:
        raise ValueError(
            "logprobs/old_logprobs/advantage/response_mask must share shape; got "
            f"{tuple(logprobs.shape)}, {tuple(old_logprobs.shape)}, "
            f"{tuple(advantage.shape)}, {tuple(response_mask.shape)}"
        )
    # Device parity: a silent CPU/GPU mix here would fail deep inside an
    # autograd op with a confusing message. Catch it up front.
    for name, t in (
        ("old_logprobs", old_logprobs),
        ("advantage", advantage),
        ("response_mask", response_mask),
    ):
        if t.device != logprobs.device:
            raise ValueError(
                f"{name} on {t.device} but logprobs on {logprobs.device}"
            )
    if clip_eps_low < 0.0 or clip_eps_high < 0.0:
        raise ValueError(
            f"clip_eps_low/high must be non-negative; got {clip_eps_low}, {clip_eps_high}"
        )
    if clip_eps_low >= 1.0:
        raise ValueError(
            f"clip_eps_low must be < 1.0 (else 1-eps_low <= 0); got {clip_eps_low}"
        )
    if kl_coef != 0.0 and ref_logprobs is None:
        raise ValueError(
            "kl_coef != 0 but ref_logprobs is None; KL term cannot be computed"
        )

    mask_bool = response_mask.to(torch.bool)
    fmask = mask_bool.to(logprobs.dtype)
    zeros = torch.zeros((), device=logprobs.device, dtype=logprobs.dtype)
    # Important: ``nan * 0 == nan``. Use where-based masking before any
    # arithmetic that touches padded positions — see
    # test_masked_nan_padding_does_not_poison_loss_or_kl which asserts a
    # NaN at a masked slot is fully scrubbed from both loss and grad.
    log_ratio_raw = torch.where(mask_bool, logprobs - old_logprobs, zeros)
    advantage_safe = torch.where(mask_bool, advantage, zeros)

    # Clamp the log-ratio before exp to avoid fp overflow under heavy
    # policy drift. ratio itself is unclipped within the clip band; the
    # outer ratio clamp at ±20 only kicks in for catastrophic divergence
    # (ratio ~5e8) and stops gradients there, which is the desired
    # behavior under runaway drift.
    # ``torch.clamp`` does NOT scrub NaN — a NaN logprob at a padded
    # position survives clamp, propagates through exp, and poisons the
    # final ``(per_token_loss * fmask).sum()`` because ``nan * 0 = nan``.
    # Zero the log-ratio at masked positions so any garbage there
    # cannot taint the loss; gradients at those positions are dropped
    # anyway (multiplied by ``fmask`` later), so this is purely defensive.
    # Fused: subtract, mask, and clamp in one expression chain so the
    # intermediate ``logprobs - old_logprobs`` and unmasked log_ratio are
    # not held as separate live tensors past the clamp.
    log_ratio_safe = torch.clamp(
        log_ratio_raw,
        min=-_LOG_RATIO_CLAMP,
        max=_LOG_RATIO_CLAMP,
    )
    ratio = torch.exp(log_ratio_safe)
    unclipped = ratio * advantage_safe
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps_low, 1.0 + clip_eps_high)
    clipped = clipped_ratio * advantage_safe
    # Pre-mask per-token loss inline with the minimum so we don't allocate
    # a separate ``per_token_loss`` tensor only to multiply it by fmask.
    per_token_loss = -torch.minimum(unclipped, clipped)

    denom = fmask.sum().clamp(min=1.0)
    # Multiply by mask before sum so padded positions contribute zero
    # gradient (otherwise nan logprobs at padding could poison the grad).
    pg_loss = (per_token_loss * fmask).sum() / denom

    # PPO ratio safety net (matches VinePPO ratio_threshold=10.0,
    # treetune/trainers/ppo_trainer.py:927-934). When the masked-mean
    # ratio across this microbatch exceeds the threshold, the policy has
    # drifted catastrophically — zero out the policy gradient on this mb
    # rather than taking a destabilizing step. KL term still contributes.
    is_skipped = torch.zeros((), device=logprobs.device, dtype=logprobs.dtype)
    if ratio_threshold is not None and ratio_threshold > 0.0:
        mean_ratio_for_skip = (ratio.detach() * fmask).sum() / denom
        skip_flag = (mean_ratio_for_skip > float(ratio_threshold)).to(logprobs.dtype)
        pg_loss = pg_loss * (1.0 - skip_flag)
        is_skipped = skip_flag

    loss = pg_loss
    mean_kl_val = None
    kl_term_val = None
    if ref_logprobs is not None:
        if ref_logprobs.shape != logprobs.shape:
            raise ValueError(
                f"ref_logprobs shape {tuple(ref_logprobs.shape)} != "
                f"logprobs {tuple(logprobs.shape)}"
            )
        if ref_logprobs.device != logprobs.device:
            raise ValueError(
                f"ref_logprobs on {ref_logprobs.device} but logprobs on {logprobs.device}"
            )
        # Sanitize garbage at padded positions before KL math: a NaN
        # ``logprobs`` at a masked slot survives clamp (and clamp_min),
        # propagates through exp, and ``nan * 0 == nan`` taints the
        # masked sum below. We mask the *difference* once (instead of
        # masking both inputs) — algebraically equivalent for binary
        # masks and saves one mul + one tensor allocation.
        if kl_estimator == "k3":
            diff = torch.where(mask_bool, ref_logprobs - logprobs, zeros)
            diff_safe = torch.clamp(
                diff, min=-_LOG_RATIO_CLAMP, max=_LOG_RATIO_CLAMP
            )
            kl = (torch.exp(diff_safe) - diff_safe - 1.0).clamp_min(0.0)
        elif kl_estimator == "k1":
            kl = torch.where(mask_bool, logprobs - ref_logprobs, zeros)
        else:
            raise ValueError(
                f"unknown kl_estimator {kl_estimator!r}; must be 'k1' or 'k3'"
            )
        # Per-token KL clamp [0, 10]. Matches VinePPO's
        # ``kl_penalty_loss_clip_min=0, kl_penalty_loss_clip_max=10`` in
        # ``configs/trainers/klLoss.jsonnet``. Without this, a single
        # high-drift token (log-ratio ~3 → k3 KL ~10; clamp boundary at
        # ±20 → k3 ~5e8) can dominate the per-sequence sum and inject a
        # gradient spike disproportionate to the policy update.
        kl = kl.clamp(min=0.0, max=10.0)
        # KL reduction: per-sequence sum, then batch mean. Matches VinePPO
        # upstream's ``ref_kl.sum(dim=1).mean()`` (treetune ppo_trainer.py:920).
        # The earlier per-token mean (``(kl * fmask).sum() / denom``) made the
        # effective KL coefficient ~response_length × smaller (e.g. ~200× at
        # mean=200 tokens), so kl_coef=1e-4 acted like ~5e-7 — policy drifted
        # unconstrained from SFT and regressed below baseline on eval. Switching
        # to per-sequence sum-then-mean restores paper-faithful KL strength.
        kl_per_seq = (kl * fmask).sum(dim=1)              # [B]
        mean_kl = kl_per_seq.mean()
        mean_kl_val = mean_kl.detach()
        if kl_coef != 0.0:
            loss = loss + kl_coef * mean_kl
            kl_term_val = (kl_coef * mean_kl).detach()

    with torch.no_grad():
        # ``denom`` is already clamp(min=1) so divisions are safe even when
        # the batch is fully masked; in that case the masked sums are 0
        # and we yield exact zeros without an extra ``.any()`` host sync.
        # Removing the host-side ``mask_bool.any()`` short-circuit avoids
        # a GPU→CPU round-trip on every step.
        mean_ratio = (ratio * fmask).sum() / denom
        mean_adv = (advantage_safe * fmask).sum() / denom
        mean_logp = torch.where(mask_bool, logprobs, zeros).sum() / denom
        # A token is "actively clipped" iff (a) it lies outside the
        # clip band AND (b) the ``min`` selected the clipped surrogate
        # (i.e. clip is binding for this advantage sign). Comparing
        # ``unclipped != clipped`` captures both: when ratio is in-band
        # they're identical; when out-of-band but A=0 they're both 0.
        # AND with the float mask directly (cast once) — equivalent to
        # ``& (fmask > 0)`` for {0, 1} masks and saves a bool tensor.
        clipped_active = (unclipped != clipped).to(logprobs.dtype) * fmask
        clip_frac = clipped_active.sum() / denom

    stats = {
        "pg_loss": pg_loss.detach(),
        "mean_ratio": mean_ratio,
        "clip_frac": clip_frac,
        "mean_advantage": mean_adv,
        "mean_logp": mean_logp,
        "ratio_skip": is_skipped.detach(),
    }
    if mean_kl_val is not None:
        stats["mean_kl"] = mean_kl_val
    if kl_term_val is not None:
        stats["kl_term"] = kl_term_val
    return loss, stats
