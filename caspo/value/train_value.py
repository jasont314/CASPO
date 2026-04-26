"""IPVRM training loss + thin trainer wrapper.

Implements Eq. 9 of arXiv:2604.13197:

    v_bar(t) = V_phi(s_t) / t                       # length-normalised value
    L = -E[ (1/T) * sum_{t=1..T} (
              r_o     * log sigma( v_bar(t) - m )
            + (1-r_o) * log( 1 - sigma( v_bar(t) + m ) )
        ) ]

where ``r_o`` is the eventual binary outcome and ``m`` is the BCE-with-margin
dead-zone.  The trainer here is a *thin* wrapper around the loss; the outer
training script drives the data iterator and the checkpointing.
"""

from __future__ import annotations

import functools
import warnings
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from caspo.config import CASPOConfig
from caspo.utils.distributed import distributed_env, is_dist_initialized
from caspo.value.prefix_value import PrefixValueModel


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def ipvrm_loss(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    outcomes: torch.Tensor,
    margin: float,
    *,
    prompt_value_baseline: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """IPVRM loss. Default: Eq. 9 (offline). With ADB/DLW kwargs: Eq. 15 (online).

    Args:
        log_ratio:     ``[B, R]``  beta * (log pi_phi - log pi_ref); requires grad.
        response_mask: ``[B, R]``  1 on response tokens, 0 on padding.
        outcomes:      ``[B]``     in ``{0., 1.}``.
        margin:        scalar ``m`` in Eq. 9 / 15.
        prompt_value_baseline:
            optional ``[B]`` tensor of per-row ``V(x) = logit(μ(x))`` from ADB.
            Added inside the sigmoid arguments to shift the BCE boundary per
            prompt difficulty (paper §3.3). ``None`` = standard Eq. 9.
        loss_weights:
            optional ``[B]`` tensor of per-row DLW weights (``1-μ`` for
            correct, ``μ`` for incorrect). Multiplies each row's loss
            contribution. ``None`` = uniform weighting (Eq. 9).

    Returns:
        ``(loss, stats)`` where ``loss`` is a scalar tensor (with grad) and
        ``stats`` carries detached scalars.
    """
    B, R = log_ratio.shape
    device = log_ratio.device
    # Promote to fp32 for cumsum stability on long prefixes (bf16 cumsum
    # accumulates round-off catastrophically beyond ~T=512).
    log_ratio_f = log_ratio.float()
    mask_bool = response_mask.to(torch.bool)
    mask_f = mask_bool.to(log_ratio_f.dtype)
    outcomes_f = outcomes.to(log_ratio_f.dtype).view(B)

    # cumulative sum over t = 1..R (so cumsum[:, t-1] is V at prefix-length t).
    log_ratio_safe = torch.where(mask_bool, log_ratio_f, torch.zeros_like(log_ratio_f))
    cumsum = torch.cumsum(log_ratio_safe, dim=1)  # [B, R]
    t_idx = torch.arange(1, R + 1, device=device, dtype=log_ratio_f.dtype).unsqueeze(0)  # [1, R]
    v_bar = cumsum / t_idx  # [B, R]

    # T_b = number of valid response tokens in row b.
    T_b = response_mask.sum(dim=1).to(log_ratio_f.dtype)  # [B]
    valid_rows = T_b > 0
    # Broadcast guard: positions with t > T_b contribute 0.
    t_mask = (t_idx <= T_b.unsqueeze(1)).to(log_ratio_f.dtype)  # [B, R]

    # ADB: shift BCE boundary by per-prompt V(x) (Eq. 15).
    if prompt_value_baseline is not None:
        if prompt_value_baseline.shape != (B,):
            raise ValueError(
                f"prompt_value_baseline must be shape ({B},), got "
                f"{tuple(prompt_value_baseline.shape)}"
            )
        V_x = prompt_value_baseline.to(log_ratio_f.dtype).view(B, 1)
    else:
        V_x = torch.zeros(B, 1, device=device, dtype=log_ratio_f.dtype)

    # BCE-with-margin terms (with optional ADB shift):
    #   pos: log sigma( v_bar - m + V_x )
    #   neg: log( 1 - sigma( v_bar + m + V_x ) ) = log sigma( -(v_bar + m + V_x) )
    pos_term = F.logsigmoid(v_bar - margin + V_x)
    neg_term = F.logsigmoid(-(v_bar + margin + V_x))
    r = outcomes_f.unsqueeze(1)  # [B, 1]
    per_t = r * pos_term + (1.0 - r) * neg_term  # [B, R]
    per_t = per_t * t_mask  # zero out beyond T_b

    # Per-row average over the T_b valid prefix lengths. ``per_t`` is
    # already zeroed past T_b via t_mask, so we only need to clamp the
    # divisor. Empty rows are also masked to zero by valid_rows below.
    safe_T = T_b.clamp(min=1.0)
    per_row = per_t.sum(dim=1) / safe_T  # [B]
    per_row = per_row * valid_rows.to(log_ratio_f.dtype)

    # DLW: rebalance the gradient by outcome rarity per prompt.
    if loss_weights is not None:
        if loss_weights.shape != (B,):
            raise ValueError(
                f"loss_weights must be shape ({B},), got {tuple(loss_weights.shape)}"
            )
        w = loss_weights.to(log_ratio_f.dtype).view(B)
        per_row = per_row * w

    n_valid = valid_rows.sum().clamp(min=1).to(log_ratio_f.dtype)
    loss = -(per_row.sum() / n_valid)

    # ---- stats ----
    # All scalars are computed on-device then transferred in a single
    # 4-element .tolist() call to amortise the GPU→CPU sync cost.
    with torch.no_grad():
        if R == 0:
            last_v = torch.zeros(B, device=device, dtype=log_ratio_f.dtype)
        else:
            # ``last_idx`` already needs to be safe for invalid rows; clamp
            # to 0 (gather is then a no-op since we mask the result below).
            last_idx = (T_b.long() - 1).clamp_(min=0)
            last_v = v_bar.gather(1, last_idx.unsqueeze(1)).squeeze(1)
            last_v = last_v * valid_rows.to(log_ratio_f.dtype)
        valid_f = valid_rows.to(log_ratio_f.dtype)
        pos_w = (outcomes_f > 0.5).to(log_ratio_f.dtype) * valid_f
        neg_w = valid_f - pos_w
        pos_n = pos_w.sum().clamp(min=1.0)
        neg_n = neg_w.sum().clamp(min=1.0)
        valid_n = valid_f.sum().clamp(min=1.0)
        mean_v_pos = (last_v * pos_w).sum() / pos_n
        mean_v_neg = (last_v * neg_w).sum() / neg_n
        match = ((last_v > 0).to(outcomes_f.dtype) == outcomes_f).to(log_ratio_f.dtype)
        acc_at_last = (match * valid_f).sum() / valid_n
        # Single sync for all four scalars.
        scalars = torch.stack([loss.detach(), mean_v_pos, mean_v_neg, acc_at_last]).tolist()
        stats = {
            "loss": float(scalars[0]),
            "mean_v_bar_pos": float(scalars[1]),
            "mean_v_bar_neg": float(scalars[2]),
            "acc_at_last": float(scalars[3]),
        }
    return loss, stats


def compute_adb_dlw_factors(
    outcomes: torch.Tensor,
    prompt_index: torch.Tensor,
    *,
    eps: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-rollout ADB baseline V(x) and DLW weight w (paper §3.3, Eq. 15).

    For each prompt p, ``μ(p) = mean(outcomes over rollouts of prompt p)``.
    The per-rollout factors are:

        V_x[b] = logit(clip(μ[prompt_index[b]], eps, 1-eps))
        w[b]   = (1 - μ[prompt_index[b]])  if outcomes[b] == 1
                  μ[prompt_index[b]]        otherwise

    Args:
        outcomes: ``[B]`` binary {0., 1.} per rollout.
        prompt_index: ``[B]`` long; the prompt id of each rollout. Rollouts
            sharing a prompt id are pooled to estimate μ.
        eps: smoothing for the logit so V_x is finite when a prompt has 0/G
            or G/G correct.

    Returns:
        ``(V_x: [B], w: [B])`` both float tensors on the same device as
        ``outcomes``.
    """
    if outcomes.dim() != 1 or prompt_index.dim() != 1:
        raise ValueError("outcomes and prompt_index must both be 1D")
    if outcomes.shape != prompt_index.shape:
        raise ValueError(
            f"shape mismatch: outcomes {tuple(outcomes.shape)} vs "
            f"prompt_index {tuple(prompt_index.shape)}"
        )

    device = outcomes.device
    o = outcomes.to(torch.float32)
    # Colocate prompt_index onto outcomes' device — otherwise index_add_
    # below silently mismatches when callers pass CPU/GPU mixed tensors.
    pidx_raw = prompt_index.to(device=device, dtype=torch.long)
    if pidx_raw.numel() == 0:
        return torch.zeros(0, device=device), torch.ones(0, device=device)

    # Densify the prompt-id space via torch.unique so we allocate
    # exactly num_unique buckets instead of (max_id + 1). Avoids the
    # GPU↔CPU sync from `.item()` and works with sparse pidx.
    unique_ids, pidx = torch.unique(pidx_raw, return_inverse=True)
    num_prompts = unique_ids.numel()

    counts = torch.zeros(num_prompts, device=device)
    sums = torch.zeros(num_prompts, device=device)
    counts.index_add_(0, pidx, torch.ones_like(o))
    sums.index_add_(0, pidx, o)
    mu_per_prompt = sums / counts.clamp(min=1.0)  # [num_prompts]

    mu = mu_per_prompt[pidx]  # [B]
    mu_clipped = mu.clamp(min=eps, max=1.0 - eps)
    V_x = torch.log(mu_clipped) - torch.log1p(-mu_clipped)  # logit(μ)

    # DLW: w = (1-μ) for positives, μ for negatives. Vectorised via
    # arithmetic blend so we don't pay torch.where's cond-tensor cost.
    w = mu + o - 2.0 * o * mu  # = o*(1-μ) + (1-o)*μ

    return V_x, w


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def _infer_fsdp_layer_classes(module: torch.nn.Module) -> set:
    """Mirror of CASPOTrainer._infer_fsdp_layer_classes for the value phi."""
    names = set(getattr(module, "_no_split_modules", None) or [])
    config = getattr(module, "config", None)
    names.update(getattr(config, "_no_split_modules", None) or [])
    if not names:
        return set()
    return {
        m.__class__
        for m in module.modules()
        if m.__class__.__name__ in names
    }


def _wrap_phi_fsdp(
    phi: torch.nn.Module,
    cfg: CASPOConfig,
    *,
    local_rank: int,
    world_size: int,
) -> torch.nn.Module:
    """FSDP-wrap the trainable ``phi`` LM.

    Mirrors ``caspo/trainer/caspo_trainer.py:_wrap_fsdp_if_enabled``. The
    frozen ``ref`` LM is left un-sharded (no opt state, just 14 GB resident
    per rank for a 7B bf16 model) so all-gathers during forward are not
    needed for the ref pass.
    """
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        CPUOffload,
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    backward_prefetch_map = {
        "backward_pre": BackwardPrefetch.BACKWARD_PRE,
        "backward_post": BackwardPrefetch.BACKWARD_POST,
        "none": None,
    }

    kwargs = {
        "sharding_strategy": strategy_map[cfg.fsdp_sharding_strategy],
        "cpu_offload": CPUOffload(offload_params=cfg.fsdp_cpu_offload),
        "use_orig_params": cfg.fsdp_use_orig_params,
        "forward_prefetch": cfg.fsdp_forward_prefetch,
        "limit_all_gathers": cfg.fsdp_limit_all_gathers,
    }
    if torch.cuda.is_available():
        kwargs["device_id"] = torch.device("cuda", local_rank)
    backward_prefetch = backward_prefetch_map[cfg.fsdp_backward_prefetch]
    if backward_prefetch is not None:
        kwargs["backward_prefetch"] = backward_prefetch

    if cfg.fsdp_auto_wrap:
        layer_classes = _infer_fsdp_layer_classes(phi)
        if layer_classes:
            kwargs["auto_wrap_policy"] = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=layer_classes,
            )
        else:
            warnings.warn(
                "FSDP auto-wrap found no transformer block classes for value phi; "
                "wrapping the top-level module only."
            )

    wrapped = FSDP(phi, **kwargs)
    return wrapped


def _build_lr_schedule(optimizer, warmup_steps: int) -> LambdaLR:
    """Linear warmup → constant LR (matches LatEntRL's helper)."""
    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


class PrefixValueTrainer:
    """Phase-1 trainer for :class:`PrefixValueModel`.

    Provides a single :meth:`step` that consumes a pre-tokenised batch dict
    and runs one optimiser update with gradient accumulation.  The outer
    training script (``scripts/train_value.py``) handles data loading,
    checkpointing, and the global step counter.
    """

    def __init__(self, cfg: CASPOConfig, model: PrefixValueModel) -> None:
        self.cfg = cfg
        self.model = model

        # FSDP-wrap phi BEFORE constructing the optimizer so AdamW sees the
        # sharded FlatParameters (cuts opt-state from 56 GB → 56/world_size GB
        # per rank for a 7B bf16 model). The frozen ref is intentionally NOT
        # wrapped: it has no opt state and frequent all-gathers during forward
        # would dominate runtime.
        env = distributed_env()
        wrap_fsdp = (
            cfg.distributed_backend == "fsdp"
            and env.is_distributed
            and is_dist_initialized()
        )
        if wrap_fsdp:
            self.model.phi = _wrap_phi_fsdp(
                self.model.phi, cfg,
                local_rank=env.local_rank, world_size=env.world_size,
            )
            if env.rank == 0:
                print(
                    f"[value] FSDP wrapped phi "
                    f"(strategy={cfg.fsdp_sharding_strategy}, "
                    f"world_size={env.world_size})",
                    flush=True,
                )

        # ``fused=True`` is bit-identical to non-fused on CUDA and ~10-20%
        # faster; silently fall back on CPU where the kwarg is rejected.
        device = self.model.device
        use_fused = device.type == "cuda" and torch.cuda.is_available()
        self.optimizer = AdamW(
            (p for p in self.model.phi.parameters() if p.requires_grad),
            lr=cfg.value_lr,
            weight_decay=cfg.value_weight_decay,
            fused=use_fused,
        )
        self.lr_scheduler = _build_lr_schedule(self.optimizer, cfg.value_warmup_steps)
        self.global_step = 0

    def step(self, batch: dict) -> dict:
        """Run one optimiser step over a batch with grad accumulation.

        Expected batch keys:
            ``prompt_ids``, ``prompt_mask``, ``response_ids``,
            ``response_mask`` (all ``[B, *]`` long/bool tensors)
            ``outcomes`` (``[B]`` float in ``{0., 1.}``)
        """
        cfg = self.cfg
        device = self.model.device
        prompt_ids = batch["prompt_ids"].to(device)
        prompt_mask = batch["prompt_mask"].to(device)
        response_ids = batch["response_ids"].to(device)
        response_mask = batch["response_mask"].to(device)
        outcomes = batch["outcomes"].to(device).float()

        B = prompt_ids.shape[0]
        mb = max(1, int(cfg.value_micro_batch_size))
        accum = max(1, int(cfg.value_grad_accum_steps))
        self.optimizer.zero_grad(set_to_none=True)

        agg = {"loss": 0.0, "mean_v_bar_pos": 0.0, "mean_v_bar_neg": 0.0, "acc_at_last": 0.0}
        micro_ranges = [(start, min(start + mb, B)) for start in range(0, B, mb)]
        total_micros = len(micro_ranges)
        valid_rows = (response_mask.sum(dim=1) > 0).detach().cpu()
        pos_rows = ((outcomes >= 0.5) & (response_mask.sum(dim=1) > 0)).detach().cpu()
        neg_rows = ((outcomes < 0.5) & (response_mask.sum(dim=1) > 0)).detach().cpu()
        micro_row_counts = [
            float(valid_rows[start:end].sum().item())
            for start, end in micro_ranges
        ]
        micro_pos_counts = [
            float(pos_rows[start:end].sum().item())
            for start, end in micro_ranges
        ]
        micro_neg_counts = [
            float(neg_rows[start:end].sum().item())
            for start, end in micro_ranges
        ]
        group_row_counts = [
            sum(micro_row_counts[i : min(i + accum, total_micros)])
            for i in range(0, total_micros, accum)
        ]
        total_rows = max(1.0, sum(micro_row_counts))
        total_pos_rows = max(1.0, sum(micro_pos_counts))
        total_neg_rows = max(1.0, sum(micro_neg_counts))
        n_micro_in_block = 0
        for micro_idx, (start, end) in enumerate(micro_ranges):
            out = self.model(
                prompt_ids[start:end], prompt_mask[start:end],
                response_ids[start:end], response_mask[start:end],
            )
            loss, stats = ipvrm_loss(
                log_ratio=out["log_ratio"],
                response_mask=response_mask[start:end],
                outcomes=outcomes[start:end],
                margin=self.model.margin,
            )
            group_idx = micro_idx // accum
            group_rows = group_row_counts[group_idx] if group_row_counts else 0.0
            micro_rows = micro_row_counts[micro_idx]
            grad_weight = micro_rows / group_rows if group_rows > 0.0 else 0.0
            (loss * grad_weight).backward()
            agg["loss"] += stats["loss"] * micro_rows
            agg["acc_at_last"] += stats["acc_at_last"] * micro_rows
            agg["mean_v_bar_pos"] += (
                stats["mean_v_bar_pos"] * micro_pos_counts[micro_idx]
            )
            agg["mean_v_bar_neg"] += (
                stats["mean_v_bar_neg"] * micro_neg_counts[micro_idx]
            )
            n_micro_in_block += 1
            if n_micro_in_block >= accum or end == B:
                if cfg.value_grad_clip and cfg.value_grad_clip > 0:
                    phi = self.model.phi
                    is_fsdp = phi.__class__.__name__ == "FullyShardedDataParallel"
                    if is_fsdp and hasattr(phi, "clip_grad_norm_"):
                        # FSDP's own clip_grad_norm_ does the right cross-rank
                        # gather of squared norms before clipping.
                        phi.clip_grad_norm_(cfg.value_grad_clip)
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            phi.parameters(), max_norm=cfg.value_grad_clip,
                        )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                n_micro_in_block = 0

        merged = {
            "loss": agg["loss"] / total_rows,
            "acc_at_last": agg["acc_at_last"] / total_rows,
            "mean_v_bar_pos": agg["mean_v_bar_pos"] / total_pos_rows,
            "mean_v_bar_neg": agg["mean_v_bar_neg"] / total_neg_rows,
        }
        merged["lr"] = self.optimizer.param_groups[0]["lr"]
        self.global_step += 1
        return merged
