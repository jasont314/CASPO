"""PPO+critic baseline for direct comparison against VinePPO.

The upstream VinePPO paper compares against PPO with a learned value
network ("critic") — a separate PreTrainedModel that emits a scalar
value per response token. This module implements that baseline:

* :class:`CriticModel` — HuggingFace LM with the LM head replaced by
  a 1-D linear value head; trained with clipped-value MSE loss
  against GAE returns.
* :func:`compute_gae` — Generalized Advantage Estimation (Schulman
  2016) given per-token rewards + critic value predictions.
* :func:`clipped_value_loss` — Schulman 2017 §6.1 clipped-value MSE
  for stable value-function training.

Usage::

    from caspo.critic import CriticModel, compute_gae, clipped_value_loss
    critic = CriticModel.from_pretrained(cfg, cfg.critic_model_name_or_path)
    # ... in trainer step():
    values = critic(input_ids, attention_mask)
    advantages, returns = compute_gae(rewards, values, ...)
    v_loss = clipped_value_loss(values, old_values, returns, mask, cliprange)

The trainer wires this into the ``method="ppo_critic"`` branch of
``CASPOTrainer.step()``.
"""

from caspo.critic.critic_model import CriticModel
from caspo.critic.gae import compute_gae, clipped_value_loss

__all__ = ["CriticModel", "compute_gae", "clipped_value_loss"]
