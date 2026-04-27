"""GAE (Generalized Advantage Estimation, Schulman 2016) +
clipped-value MSE loss (Schulman 2017 §6.1).

The verifier-RL setting puts a single terminal reward at the EOS
token; intermediate-token rewards are zero. GAE rolls that backward
through the value predictions to produce per-token advantages and
returns.
"""

from __future__ import annotations

from typing import Tuple

import torch


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token advantages + returns for the response span.

    Args:
        rewards: ``[B, R]`` per-token rewards. Verifier-RL convention:
            terminal-only — last valid token has the verifier reward,
            zeros elsewhere. Caller is responsible for setting this up.
        values: ``[B, R]`` critic value predictions for each response
            position. ``values[:, t]`` is V(s_t), where s_t is the
            state BEFORE token t (i.e., the prefix ending at t-1).
        response_mask: ``[B, R]`` 1 on response tokens, 0 on padding.
        gamma: discount factor. 1.0 for our undiscounted setting.
        gae_lambda: GAE λ. 0.95 is the upstream VinePPO/RLHF default.

    Returns:
        ``(advantages, returns)`` both ``[B, R]``. Advantages are NOT
        standardized — caller should whiten if desired (the trainer's
        ``standardize_advantage_scope`` knob handles this in step()).
        Returns is the GAE target for value training:
        ``returns[t] = advantages[t] + values[t]``.

    Math (per row, t = R-1 down to 0):
        next_v = values[t+1] if t+1 < R else 0
        delta_t = rewards[t] + gamma * next_v - values[t]
        advantages[t] = delta_t + gamma * lambda * advantages[t+1]
    """
    if rewards.shape != values.shape:
        raise ValueError(
            f"rewards {tuple(rewards.shape)} != values {tuple(values.shape)}"
        )
    if response_mask.shape != rewards.shape:
        raise ValueError(
            f"response_mask {tuple(response_mask.shape)} != rewards "
            f"{tuple(rewards.shape)}"
        )
    R = rewards.shape[1]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[:, 0])
    mask_f = response_mask.to(rewards.dtype)
    for t in reversed(range(R)):
        next_value = values[:, t + 1] if t + 1 < R else torch.zeros_like(values[:, 0])
        # Mask out beyond-response tokens — their reward/value contributions
        # are spurious. Per-row response lengths are baked into the mask.
        next_mask = mask_f[:, t + 1] if t + 1 < R else torch.zeros_like(mask_f[:, 0])
        next_value = next_value * next_mask
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        last_gae = delta + gamma * gae_lambda * last_gae * next_mask
        advantages[:, t] = last_gae * mask_f[:, t]
    returns = advantages + values * mask_f
    return advantages, returns


def clipped_value_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    cliprange: float = 0.2,
) -> torch.Tensor:
    """Schulman 2017 §6.1 clipped-value MSE loss.

    The clip prevents large value drift between PPO epochs: each
    prediction is clipped to ``[old_v - cliprange, old_v + cliprange]``
    before the MSE. The realized loss is the max of clipped vs
    unclipped (so we always penalize whichever is larger), masked
    to response tokens.

    Args:
        values:    ``[B, R]`` current critic predictions.
        old_values:``[B, R]`` predictions at the start of the PPO
                   epoch (frozen, .detach()'d).
        returns:   ``[B, R]`` GAE returns (target).
        response_mask: ``[B, R]`` 1 on response tokens, 0 on pad.
        cliprange: scalar clip width.

    Returns:
        Scalar tensor (mean over response tokens), times 0.5 (the
        canonical 1/2 in MSE) to match Schulman's expression.
    """
    values_clipped = torch.clamp(
        values, old_values - cliprange, old_values + cliprange
    )
    vf_losses1 = (values - returns) ** 2
    vf_losses2 = (values_clipped - returns) ** 2
    vf_losses = torch.maximum(vf_losses1, vf_losses2)
    mask_f = response_mask.to(vf_losses.dtype)
    denom = mask_f.sum().clamp(min=1.0)
    return 0.5 * (vf_losses * mask_f).sum() / denom
