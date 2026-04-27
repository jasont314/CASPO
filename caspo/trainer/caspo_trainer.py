"""CASPO policy trainer (phase 2) with method dispatch (ppo / caspo / grpo / vineppo).

Selectable via ``cfg.method``:

* ``ppo``: sequence-level terminal reward advantage, optionally standardized
  over the batch/group, broadcast to every response token, then PPO clip.
* ``caspo``: step segmentation + IPVRM V_φ forward + step TD + token broadcast.
  Optional online IPVRM update (Eq. 15 with ADB+DLW) when
  ``cfg.update_value_during_policy=True``.
* ``grpo``: no segmentation; per-sequence advantage from
  ``group_relative_advantage`` (DeepSeekMath / Shao et al.) broadcast to tokens.
* ``vineppo``: step segmentation + K MC rollouts at each step boundary via the
  vLLM rollout engine, average reward = V at boundary, then step TD.

All methods share the rollout (HF or vLLM), reward grading, and PPO clipped
surrogate. The forward pass for the policy loss is separate from the rollout's
sampling-time logprobs (importance ratio = π_θ / π_old).

vLLM weight sync: after each optimizer step, the trainer either pushes weights
directly with vLLM's CUDA-IPC update API (Rho-scale single-process/DDP
replicated runs) or saves a checkpoint to ``cfg.vllm_sync_dir`` and calls
``engine.sync_weights_from_path(...)`` (FSDP / larger-model fallback). NCCL
upgrade is a follow-up for sharded 7B-scale runs.

Wandb: enabled by default. Logs policy/reward/segmentation/value/timing/eval
metrics per step. Set ``cfg.wandb_enabled=False`` to disable.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import random
import subprocess
import sys
import time
import warnings
from dataclasses import asdict
from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from caspo.algo import (
    broadcast_step_advantage_to_tokens,
    ppo_clipped_loss,
    standardize_step_advantage,
    step_td_advantage,
    step_values_from_log_ratios,
    transform_step_values_for_advantage,
)
from caspo.algo.advantages import standardize_step_advantage as _stdadv  # noqa
from caspo.config import CASPOConfig
from caspo.data import load_eval_dataset, load_train_dataset
from caspo.reward import MathRewardFn
from caspo.rollout import HFRolloutSampler, RolloutBatch, build_rollout_engine
from caspo.segmentation import (
    segment_responses_batch,
    segment_responses_batch_latex_aware,
)
from caspo.utils.distributed import (
    DistributedInfo,
    barrier as dist_barrier,
    distributed_env,
    init_distributed_if_needed,
    rank0_print,
    reduce_numeric_stats,
    resolve_device,
)
from caspo.utils.seed import set_seed
from caspo.value import PrefixValueModel, compute_adb_dlw_factors, ipvrm_loss


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"unknown torch_dtype {name!r}")
    return _DTYPE_MAP[name]


def _fused_adamw_supported(device: torch.device) -> bool:
    """Return True iff fused AdamW is safe on this device.

    Fused AdamW is bit-identical to non-fused (CUDA-only fused-mul-add path),
    ~10-20% faster on modern GPUs. Falls back silently on CPU / older
    PyTorch builds where the kwarg is unsupported.
    """
    if not isinstance(device, torch.device):
        device = torch.device(device)
    return device.type == "cuda" and torch.cuda.is_available()


def _build_lr_schedule(
    optimizer,
    warmup_steps: int,
    total_steps: Optional[int] = None,
) -> LambdaLR:
    """Linear-warmup → linear-decay-to-zero.

    Matches VinePPO upstream's HF ``get_linear_schedule_with_warmup``
    (lr_scheduler_type='linear' default). The earlier constant-tail
    schedule kept LR at the peak value forever — at step 800/1000 we
    were training at lr=1e-6 while upstream had decayed to ~2e-7,
    contributing to late-training drift below SFT. ``total_steps`` is
    the total number of optimizer.step() calls expected over the run;
    when omitted we fall back to constant-tail (legacy callers like
    the V_φ trainer that don't expose total).
    """
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps is None or total_steps <= warmup_steps:
            return 1.0
        # Linear decay from 1.0 at step=warmup_steps down to 0.0 at total_steps.
        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.0, 1.0 - progress)
    return LambdaLR(optimizer, lr_lambda)


def _cycle(iterable: Iterable):
    while True:
        any_yielded = False
        for x in iterable:
            any_yielded = True
            yield x
        if not any_yielded:
            raise RuntimeError("data iterable yielded no examples")


def _reshuffling_cycle(examples: list, seed: int):
    """Infinite iterator over ``examples`` that reshuffles between passes.

    Each pass uses a fresh ``random.Random(seed + epoch)`` so the order is
    deterministic given ``seed`` and reproducible across reruns, but doesn't
    repeat the same fixed cycle that a plain ``itertools.cycle`` would. The
    list is shuffled in place via a per-epoch local copy so callers can keep
    a stable ``examples`` reference for diagnostics if needed.
    """
    if not examples:
        raise RuntimeError("data iterable yielded no examples")
    epoch = 0
    while True:
        rng = random.Random(int(seed) + epoch)
        order = list(range(len(examples)))
        rng.shuffle(order)
        for i in order:
            yield examples[i]
        epoch += 1


def _configured_logprob_micro_batch_size(cfg: CASPOConfig) -> int:
    return max(1, int(cfg.logprob_micro_batch_size or cfg.micro_batch_size))


def _scalar_tensor_stats(values: torch.Tensor) -> dict[str, float]:
    """Return cheap scalar diagnostics for a 1D-ish tensor."""
    if values.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    x = values.detach().float().reshape(-1)
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "min": float(x.min().item()),
        "max": float(x.max().item()),
    }


def _tokenize_delimiter(tokenizer, delimiter: str) -> List[int]:
    if not delimiter:
        raise ValueError("step_delimiter must be a non-empty string")
    ids = tokenizer(delimiter, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(
            f"tokenizer returned empty token-id list for delimiter {delimiter!r}"
        )
    return [int(t) for t in ids]


def _group_relative_advantage(rewards: torch.Tensor, group_size: int) -> torch.Tensor:
    """GRPO advantage (Shao et al., DeepSeekMath): center+std-normalize within
    each group of G samples sharing a prompt. Zero-variance groups → 0."""
    B = rewards.numel()
    if B % group_size != 0:
        raise ValueError(f"rewards.numel()={B} not divisible by G={group_size}")
    g = rewards.view(B // group_size, group_size)
    mean = g.mean(dim=1, keepdim=True)
    centered = g - mean
    std = g.std(dim=1, keepdim=True, unbiased=False)
    safe = torch.where(std <= 1e-8, torch.ones_like(std), std)
    out = centered / safe
    out = torch.where(std.expand_as(out) <= 1e-8, torch.zeros_like(out), out)
    return out.reshape(-1)


def _sequence_reward_advantage(
    rewards: torch.Tensor,
    group_size: int,
    *,
    scope: str = "batch",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Sequence-level PPO baseline for terminal rewards.

    ``scope='batch'`` gives the usual reward-centering baseline over the full
    rollout batch. ``scope='group'`` intentionally matches GRPO's per-prompt
    normalization, and ``scope='off'`` uses raw rewards.
    """
    if rewards.dim() != 1:
        raise ValueError(f"rewards must be 1D, got {tuple(rewards.shape)}")
    if scope == "off":
        return rewards.clone()
    if rewards.numel() == 0:
        return rewards.clone()
    if scope == "group":
        return _group_relative_advantage(rewards, group_size=group_size)
    if scope != "batch":
        raise ValueError(f"unknown scope {scope!r}; must be 'batch', 'group', or 'off'")

    r = rewards.float()
    mean = r.mean()
    centered = r - mean
    std = centered.square().mean().sqrt()
    zero_var = std <= eps
    safe_std = torch.where(zero_var, torch.ones_like(std), std)
    out = centered / safe_std
    return torch.where(zero_var.expand_as(out), torch.zeros_like(out), out)


def _is_fsdp_module(module: torch.nn.Module) -> bool:
    return module.__class__.__name__ == "FullyShardedDataParallel"


def _is_ddp_module(module: torch.nn.Module) -> bool:
    return module.__class__.__name__ == "DistributedDataParallel"


def _unwrap_parallel(module: torch.nn.Module) -> torch.nn.Module:
    if _is_fsdp_module(module) or _is_ddp_module(module):
        return getattr(module, "module", module)
    return module


def _resolve_ac_mode(cfg: CASPOConfig) -> str:
    """Pick between cfg.activation_checkpointing_mode and the legacy bool.

    ``activation_checkpointing_mode`` defaults to ``"off"``; when the user
    leaves it at the default, fall back to the bool ``use_gradient_checkpointing``
    (which has been the legacy knob). Any explicit non-"off" value of the new
    field overrides the bool. This preserves every existing config's behaviour
    while letting selective AC opt-in via the new literal.
    """
    mode = getattr(cfg, "activation_checkpointing_mode", "off")
    if mode != "off":
        return mode
    return "full" if cfg.use_gradient_checkpointing else "off"


def _apply_activation_checkpointing(
    model: torch.nn.Module, cfg: CASPOConfig,
) -> None:
    """Apply activation checkpointing to ``model`` per cfg.

    * ``"off"`` — leave the model alone (default).
    * ``"full"`` — call HF's ``gradient_checkpointing_enable()`` (every block
      recomputes its forward on backward; the legacy default when
      ``cfg.use_gradient_checkpointing=True``).
    * ``"selective"`` — wrap only attention submodules with PyTorch's
      ``apply_activation_checkpointing``. Attention activations are the bulk
      of the per-layer activation memory but are also FLOPS-cheap to recompute,
      so this is usually the better point on the memory-vs-throughput curve
      than ``"full"`` (Megatron's ``recompute_granularity="selective"``).
    """
    mode = _resolve_ac_mode(cfg)
    if mode == "off":
        return
    if mode == "full":
        try:
            model.gradient_checkpointing_enable()
        except Exception as e:
            warnings.warn(f"could not enable gradient checkpointing: {e}")
        return
    if mode == "selective":
        try:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                apply_activation_checkpointing,
                checkpoint_wrapper,
            )
        except ImportError as e:  # pragma: no cover
            warnings.warn(
                f"selective activation checkpointing unavailable on this "
                f"torch build ({e}); falling back to full gradient checkpointing."
            )
            try:
                model.gradient_checkpointing_enable()
            except Exception as e2:
                warnings.warn(f"could not enable gradient checkpointing: {e2}")
            return

        def _is_attention(m: torch.nn.Module) -> bool:
            return m.__class__.__name__.endswith("Attention")

        non_reentrant = checkpoint_wrapper
        try:
            non_reentrant = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
        except Exception:
            pass

        try:
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=non_reentrant,
                check_fn=_is_attention,
            )
        except Exception as e:
            warnings.warn(
                f"selective AC apply failed ({e}); falling back to full "
                f"gradient checkpointing."
            )
            try:
                model.gradient_checkpointing_enable()
            except Exception as e2:
                warnings.warn(f"could not enable gradient checkpointing: {e2}")
        return
    warnings.warn(f"unknown activation_checkpointing_mode={mode!r}; ignoring")


class CASPOTrainer:
    """End-to-end RL trainer with ``cfg.method`` dispatch."""

    def __init__(self, cfg: CASPOConfig):
        self.cfg = cfg
        env_dist = distributed_env()
        if cfg.distributed_backend == "none":
            if env_dist.is_distributed:
                raise ValueError(
                    "torchrun launched WORLD_SIZE>1 but "
                    "distributed_backend='none'. Use "
                    "distributed_backend='fsdp' or 'ddp' for full-model distributed "
                    "training, or launch a single process."
                )
            self.dist = DistributedInfo()
        else:
            self.dist = init_distributed_if_needed(
                backend=cfg.dist_backend,
                timeout_s=cfg.dist_timeout_s,
            )
        self._fsdp_enabled = (
            cfg.distributed_backend == "fsdp" and self.dist.is_distributed
        )
        self._ddp_enabled = (
            cfg.distributed_backend == "ddp" and self.dist.is_distributed
        )
        if cfg.distributed_backend == "fsdp" and not self.dist.is_distributed:
            warnings.warn(
                "distributed_backend='fsdp' requested but WORLD_SIZE=1; "
                "continuing in single-process mode. Launch with torchrun for "
                "sharded full-model training."
            )
        if cfg.distributed_backend == "ddp" and not self.dist.is_distributed:
            warnings.warn(
                "distributed_backend='ddp' requested but WORLD_SIZE=1; "
                "continuing in single-process mode. Launch multiple ranks for "
                "replicated data-parallel training."
            )
        self.device = resolve_device(cfg.device, self.dist)
        set_seed(int(cfg.seed) + int(self.dist.rank))

        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

        # ---- tokenizer ----
        tok_path = cfg.tokenizer_name_or_path or cfg.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_path, trust_remote_code=cfg.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- policy model (π_θ, trainable) ----
        torch_dtype = _resolve_dtype(cfg.torch_dtype)
        model_kwargs = dict(
            torch_dtype=torch_dtype,
            trust_remote_code=cfg.trust_remote_code,
        )
        if cfg.attn_implementation:
            model_kwargs["attn_implementation"] = cfg.attn_implementation

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path, **model_kwargs,
        )
        try:
            self.model.config.use_cache = False
        except AttributeError:
            pass
        _apply_activation_checkpointing(self.model, cfg)

        self.model.to(self.device)

        # ---- prefix value model (V_φ) — only for method="caspo" ----
        self.value_model: Optional[PrefixValueModel] = None
        self.value_optimizer: Optional[AdamW] = None
        if cfg.method == "caspo":
            if not cfg.prefix_value_path:
                raise ValueError(
                    "cfg.prefix_value_path must point to a phase-1 PrefixValueModel "
                    "checkpoint when method=caspo. Run scripts/train_value.py first."
                )
            self.value_model = PrefixValueModel.from_pretrained(cfg, cfg.prefix_value_path)
            self.value_model.to(self.device)
            if cfg.update_value_during_policy:
                self.value_model.phi.train()
            else:
                for p in self.value_model.phi.parameters():
                    p.requires_grad_(False)
                self.value_model.phi.eval()

        # ---- learned critic — only for method="ppo_critic" ----
        # Schulman 2017 PPO with a separate value network. Built from
        # cfg.critic_model_name_or_path (defaults to cfg.model_name_or_path
        # so the critic shares the policy's SFT init — VinePPO upstream
        # pattern). Trained jointly with the policy via clipped-MSE loss
        # against GAE returns. FSDP-wrapped on the same path as the policy.
        self.critic_model = None
        self.critic_optimizer: Optional[AdamW] = None
        self.critic_lr_scheduler = None
        if cfg.method == "ppo_critic":
            from caspo.critic import CriticModel

            critic_path = cfg.critic_model_name_or_path or cfg.model_name_or_path
            self.critic_model = CriticModel.from_pretrained(cfg, critic_path)
            self.critic_model.to(self.device)
            self.critic_model.train()
            self.critic_model = self._wrap_fsdp_if_enabled(
                self.critic_model, module_name="critic",
            )
            self.critic_model = self._wrap_ddp_if_enabled(
                self.critic_model, module_name="critic",
            )
            self.critic_optimizer = AdamW(
                (p for p in self.critic_model.parameters() if p.requires_grad),
                lr=cfg.critic_lr,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=cfg.critic_weight_decay,
                fused=_fused_adamw_supported(self.device),
            )
            self.critic_lr_scheduler = _build_lr_schedule(
                self.critic_optimizer, cfg.critic_warmup_steps,
            )

        # ---- reference policy for PPO KL term (optional) ----
        self.ref_policy = None
        if cfg.kl_coef and cfg.kl_coef > 0:
            self.ref_policy = AutoModelForCausalLM.from_pretrained(
                cfg.model_name_or_path, **model_kwargs,
            )
            # Mirror policy: never cache KV (we always run a fresh
            # full-sequence forward) and never train (frozen reference).
            try:
                self.ref_policy.config.use_cache = False
            except AttributeError:
                pass
            for p in self.ref_policy.parameters():
                p.requires_grad_(False)
            self.ref_policy.eval()
            self.ref_policy.to(self.device)

            self.ref_policy = self._wrap_fsdp_if_enabled(
                self.ref_policy, module_name="ref_policy",
            )

            # ---- share frozen ref between trainer and value model ----
            # When method=caspo and the value model's reference LM was loaded
            # from the SAME checkpoint with the SAME dtype/attn_impl/
            # trust_remote_code, point both consumers at a single underlying
            # module instead of holding two identical copies. Saves ~1× base-
            # model GPU RAM (e.g. ~2 GB at 1B-bf16, ~14 GB at 7B-bf16).
            #
            # Conservative gate: only share when the value model's
            # ref_model_path equals cfg.model_name_or_path. If a future config
            # points the value model at a different ref (e.g. a custom SFT
            # init), the constructor signature is unchanged and share_ref is
            # not invoked - both modules retain their independent loads.
            # dtype/attn_impl/trust_remote_code are guaranteed equal here
            # because both refs were built from the same ``model_kwargs``.
            if (
                self.value_model is not None
                and getattr(self.value_model, "_ref_model_path", None)
                    == cfg.model_name_or_path
            ):
                try:
                    self.value_model.share_ref(self.ref_policy)
                except Exception as e:  # pragma: no cover
                    warnings.warn(
                        f"share_ref failed (continuing with duplicate refs): {e}"
                    )

        self.model = self._wrap_fsdp_if_enabled(self.model, module_name="policy")
        self.model = self._wrap_ddp_if_enabled(
            self.model, module_name="policy", require_trainable=True,
        )
        # ---- torch.compile (opt-in, applied AFTER DDP/FSDP wrapping) ----
        # ``mode="reduce-overhead"`` reuses CUDA graphs across steps and is
        # the right knob for steady-shape workloads like RL rollouts.
        # ``dynamic=True`` accepts the variable response_len + microbatch
        # padding without recompiling on each shape; ``fullgraph=False``
        # tolerates HF's Python-side preprocessing (rotary cache, etc.).
        # Frozen modules (ref_policy, value_model.ref) are intentionally
        # not compiled — no autograd, no gain. The value-model phi is
        # compiled below when present and trainable.
        if cfg.compile:
            self.model = torch.compile(
                self.model,
                mode="default",
                dynamic=True,
                fullgraph=False,
            )
        if self.value_model is not None:
            if self.ref_policy is None:
                self.value_model.ref = self._wrap_fsdp_if_enabled(
                    self.value_model.ref, module_name="value_ref",
                )
            self.value_model.phi = self._wrap_fsdp_if_enabled(
                self.value_model.phi, module_name="value_phi",
            )
            self.value_model.phi = self._wrap_ddp_if_enabled(
                self.value_model.phi,
                module_name="value_phi",
                require_trainable=True,
            )
            if cfg.compile:
                self.value_model.phi = torch.compile(
                    self.value_model.phi,
                    mode="reduce-overhead",
                    dynamic=True,
                    fullgraph=False,
                )
            if cfg.update_value_during_policy:
                self.value_optimizer = AdamW(
                    (p for p in self.value_model.phi.parameters() if p.requires_grad),
                    lr=cfg.online_value_lr,
                    weight_decay=cfg.value_weight_decay,
                    fused=_fused_adamw_supported(self.device),
                )

        # ---- optimizer + schedule ----
        self.optimizer = AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=_fused_adamw_supported(self.device),
        )
        # Total optimizer.step() calls over the run for linear-decay-to-zero
        # schedule. ``cfg.max_steps`` outer iterations × ``cfg.epochs_per_rollout``
        # epochs over each rollout × (rollout_size / (mb × accum)) optim steps
        # per epoch. Rollout size = prompts_per_step × group_size. Falls back
        # to a conservative estimate if any term is missing.
        _ppr = max(1, int(getattr(cfg, "prompts_per_step", 64)))
        _G = max(1, int(getattr(cfg, "group_size", 8)))
        _mb = max(1, int(getattr(cfg, "micro_batch_size", 8)))
        _acc = max(1, int(getattr(cfg, "grad_accum_steps", 8)))
        _epr = max(1, int(getattr(cfg, "epochs_per_rollout", 2)))
        _per_outer_optim_steps = max(1, (_ppr * _G) // (_mb * _acc) * _epr)
        _total_optim_steps = int(cfg.max_steps) * _per_outer_optim_steps
        self.lr_scheduler = _build_lr_schedule(
            self.optimizer, cfg.warmup_steps, total_steps=_total_optim_steps,
        )

        # ---- delimiter token ids (for token_delimiter segmentation only) ----
        self.delimiter_token_ids = _tokenize_delimiter(self.tokenizer, cfg.step_delimiter)

        # ---- reward + data ----
        self.reward_fn = MathRewardFn(
            num_workers=cfg.reward_workers,
            gt_cache_max_size=cfg.gt_cache_max_size,
        )
        self.train_examples = list(load_train_dataset(cfg, tokenizer=self.tokenizer))
        if self.dist.is_distributed:
            self.train_examples = self.train_examples[
                self.dist.rank::self.dist.world_size
            ]
        if not self.train_examples:
            raise RuntimeError(
                "training dataset is empty after filtering/sharding on "
                f"rank {self.dist.rank}/{self.dist.world_size}"
            )
        # Reshuffle between passes so we don't lock into the same fixed cycle
        # order across epochs. Seed = cfg.seed bumped per-epoch internally; on
        # distributed runs each rank already has its own shard, so the
        # rank-local shuffle is independent — no extra rank stride needed.
        self._train_iter = _reshuffling_cycle(self.train_examples, cfg.seed)

        # ---- rollout backend (HF or vLLM) ----
        backend = cfg.rollout_backend
        if backend == "vllm":
            if (
                self._ddp_enabled
                and cfg.vllm_weight_sync_backend == "ipc"
                and torch.cuda.is_available()
            ):
                visible_cuda_devices = torch.cuda.device_count()
                if visible_cuda_devices != 1:
                    raise RuntimeError(
                        "distributed_backend='ddp' with "
                        "vllm_weight_sync_backend='ipc' expects each rank "
                        "process to see exactly one CUDA device. Launch with "
                        "one physical GPU in CUDA_VISIBLE_DEVICES per rank "
                        "(for example scripts/launch_rho1b_vineppo_ddp2.sh); "
                        f"torch.cuda.device_count()={visible_cuda_devices}."
                    )
            # Disaggregated topology: only trainer rank 0 holds the real
            # VLLMRolloutEngine. The engine spawns vllm_disaggregated_tp
            # worker subprocesses pinned to the rollout GPUs (whose
            # physical IDs are passed in via CASPO_ROLLOUT_GPU_PHYSICAL_IDS
            # by the disaggregated launcher). Other ranks wrap a
            # ``DisaggregatedSamplerProxy`` so generation calls go through
            # rank 0 via gather/scatter.
            if cfg.vllm_disaggregated:
                from caspo.rollout.disagg import DisaggregatedSamplerProxy

                if self.dist.is_main:
                    rollout_ids_env = os.environ.get(
                        "CASPO_ROLLOUT_GPU_PHYSICAL_IDS", ""
                    ).strip()
                    if not rollout_ids_env:
                        raise RuntimeError(
                            "vllm_disaggregated=True requires the launcher to "
                            "set CASPO_ROLLOUT_GPU_PHYSICAL_IDS=<csv list> for "
                            "rank 0 (see scripts/_launch_7b_disagg.sh)."
                        )
                    rollout_ids = [int(s) for s in rollout_ids_env.split(",") if s.strip()]
                    if len(rollout_ids) != int(cfg.vllm_disaggregated_tp):
                        raise RuntimeError(
                            f"CASPO_ROLLOUT_GPU_PHYSICAL_IDS has "
                            f"{len(rollout_ids)} GPU(s) but "
                            f"vllm_disaggregated_tp={cfg.vllm_disaggregated_tp}; "
                            f"the launcher must keep them in sync."
                        )
                    inner_kwargs = dict(
                        gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
                        tensor_parallel_size=int(cfg.vllm_disaggregated_tp),
                        enforce_eager=cfg.vllm_enforce_eager,
                        seed=cfg.seed,
                        max_num_seqs=cfg.vllm_max_num_seqs,
                        max_num_batched_tokens=cfg.vllm_max_num_batched_tokens,
                        max_inflight_requests=cfg.vllm_max_inflight_requests,
                        enable_chunked_prefill=bool(cfg.vllm_enable_chunked_prefill),
                        # Tell the engine: bind to these PHYSICAL GPU IDs as
                        # a contiguous TP group. The engine handles the
                        # CUDA_VISIBLE_DEVICES gymnastics so that AsyncLLM
                        # workers see them as cuda:0..cuda:N-1.
                        gpu_ids=rollout_ids,
                    )
                    inner = build_rollout_engine(
                        cfg, self.reward_fn, **inner_kwargs,
                    )
                else:
                    inner = None
                # Phase F: when ``vllm_disaggregated=True`` AND
                # ``vllm_weight_sync_backend=='ipc'``, every trainer rank
                # is expected to colocate with its same-index vLLM
                # worker on the same physical GPU. The proxy uses the
                # multirank-IPC sync path (gather per-rank IPC handles
                # → merge → submit). Sanity-check the topology
                # invariants the launcher promises.
                multirank_ipc = (
                    cfg.vllm_weight_sync_backend == "ipc"
                    and cfg.vllm_disaggregated
                )
                if multirank_ipc:
                    # Every trainer rank must occupy the corresponding
                    # rollout-GPU slot. Specifically: the launcher must
                    # set the trainer's world_size == rollout-GPU count
                    # and CUDA_VISIBLE_DEVICES so trainer rank N pins
                    # to physical GPU N. The trainer-side
                    # ``CASPO_ROLLOUT_GPU_PHYSICAL_IDS`` env var
                    # already lists those physical GPUs in launch
                    # order (set by _launch_7b_tp8_ipc.sh).
                    rollout_ids_env = os.environ.get(
                        "CASPO_ROLLOUT_GPU_PHYSICAL_IDS", ""
                    ).strip()
                    if rollout_ids_env:
                        n_rollout = len(
                            [s for s in rollout_ids_env.split(",") if s.strip()]
                        )
                        if n_rollout != int(self.dist.world_size):
                            raise RuntimeError(
                                "multirank-IPC sync (vllm_disaggregated=True + "
                                "vllm_weight_sync_backend=ipc) requires the "
                                "trainer world_size to equal the rollout-GPU "
                                f"count, but world_size={self.dist.world_size} "
                                f"and len(CASPO_ROLLOUT_GPU_PHYSICAL_IDS)={n_rollout}. "
                                "Use scripts/_launch_7b_tp8_ipc.sh which sets up "
                                "the colocated topology correctly."
                            )
                self.sampler = DisaggregatedSamplerProxy(
                    inner, self.dist, multirank_ipc=multirank_ipc,
                )
            else:
                engine_kwargs = dict(
                    gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
                    tensor_parallel_size=cfg.vllm_tensor_parallel_size,
                    enforce_eager=cfg.vllm_enforce_eager,
                    seed=cfg.seed,
                    max_num_seqs=cfg.vllm_max_num_seqs,
                    max_num_batched_tokens=cfg.vllm_max_num_batched_tokens,
                    max_inflight_requests=cfg.vllm_max_inflight_requests,
                    enable_chunked_prefill=bool(cfg.vllm_enable_chunked_prefill),
                    gpu_id=(
                        self.dist.local_rank
                        if self.dist.is_distributed
                        and cfg.vllm_tensor_parallel_size == 1
                        else None
                    ),
                )
                self.sampler = build_rollout_engine(cfg, self.reward_fn, **engine_kwargs)
            self._sync_dir = cfg.vllm_sync_dir or os.path.join(cfg.output_dir, "_vllm_sync")
            os.makedirs(self._sync_dir, exist_ok=True)
            self._sync_tokenizer_saved = False
            rank0_print(
                self.dist,
                "[trainer] vLLM rollout backend ON — "
                f"weight_sync={cfg.vllm_weight_sync_backend} "
                f"sync_dir={self._sync_dir}",
                flush=True,
            )
        else:
            self.sampler = build_rollout_engine(
                cfg, self.reward_fn, model=self.model, tokenizer=self.tokenizer,
            )
            self._sync_dir = None
            self._sync_tokenizer_saved = False

        # ---- bookkeeping ----
        self.global_step = 0
        os.makedirs(cfg.output_dir, exist_ok=True)

        # ---- wandb ----
        self._wandb = None
        if self.dist.is_main and cfg.wandb_enabled and cfg.wandb_mode != "disabled":
            try:
                import wandb
                tags = (
                    [t.strip() for t in cfg.wandb_tags.split(",") if t.strip()]
                    if cfg.wandb_tags else []
                )
                run_name = cfg.wandb_run_name or self._auto_run_name()
                self._wandb = wandb.init(
                    project=cfg.wandb_project,
                    entity=cfg.wandb_entity,
                    name=run_name,
                    mode=cfg.wandb_mode,
                    tags=tags,
                    config=asdict(cfg),
                )
                print(
                    f"[trainer] wandb run: {run_name} "
                    f"(project={cfg.wandb_project}, mode={cfg.wandb_mode})",
                    flush=True,
                )
            except Exception as e:
                warnings.warn(f"wandb init failed (continuing without): {e}")
                self._wandb = None

    # ------------------------------------------------------------------ helpers

    def _infer_fsdp_layer_classes(self, module: torch.nn.Module) -> set[type]:
        """Infer transformer block classes for FSDP auto-wrapping.

        For wrappers like ``CriticModel(backbone=LlamaForCausalLM)`` the
        ``_no_split_modules`` attribute lives on the inner backbone, not
        on the top-level module. Walk the module tree until we find one
        that exposes it (or its ``config``). Without this, the critic
        would FSDP-wrap as a single flat 7B param with no compute/comm
        overlap and full unsharded peak on every rank.
        """
        names: set[str] = set()
        for m in [module, *module.modules()]:
            names.update(getattr(m, "_no_split_modules", None) or [])
            cfg = getattr(m, "config", None)
            names.update(getattr(cfg, "_no_split_modules", None) or [])
            if names:
                break
        if not names:
            return set()
        return {
            m.__class__
            for m in module.modules()
            if m.__class__.__name__ in names
        }

    def _make_block_group_wrap_policy(
        self,
        root_module: torch.nn.Module,
        layer_classes: set[type],
        group_size: int,
    ):
        """Build an FSDP auto-wrap policy that groups every ``group_size``
        transformer blocks into one FSDP unit.

        At 7B (32 blocks) ``group_size=4`` cuts the backward reduce-scatter
        call count from 32 to 8, doubling each collective's payload from
        ~440 MB to ~1.7 GB and lifting NVLink BW utilization. Embedding /
        LM head stay in the root flat-param (not separately wrapped), which
        is the desired layout — leaving contiguous unwrapped layers between
        wrap points lets PyTorch FSDP collapse them into the parent unit.

        Implementation: pre-walk the module tree to enumerate transformer
        blocks in module-tree (post-order via ``modules()``) order, mark
        every Nth block (the last of each group) as a wrap point by
        ``id(module)``, then return a closure that consults the membership
        set. We use ``id`` (not the module object) for the set key to avoid
        relying on ``__hash__`` semantics for nn.Module subclasses.
        """
        layer_set = tuple(layer_classes)
        blocks = [m for m in root_module.modules() if isinstance(m, layer_set)]
        # Mark the last block of each contiguous group of ``group_size``
        # as the FSDP wrap point. The earlier blocks in the group remain
        # unwrapped and get folded into the parent flat-param, producing
        # one FSDP unit per N consecutive blocks end-to-end.
        wrap_ids = {
            id(b) for i, b in enumerate(blocks)
            if (i % group_size) == (group_size - 1)
        }
        # Always wrap the trailing partial group (if total_blocks % N != 0)
        # so the final block(s) don't get pulled into the root unit.
        if blocks and id(blocks[-1]) not in wrap_ids:
            wrap_ids.add(id(blocks[-1]))

        def policy(module, recurse, nonwrapped_numel, **kwargs):
            if recurse:
                # Always descend into children so we visit every block.
                return True
            if isinstance(module, layer_set):
                return id(module) in wrap_ids
            return False

        return policy

    def _wrap_fsdp_if_enabled(
        self,
        module: torch.nn.Module,
        *,
        module_name: str,
    ) -> torch.nn.Module:
        if not self._fsdp_enabled:
            return module

        from torch.distributed.fsdp import (
            BackwardPrefetch,
            CPUOffload,
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        strategy_map = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
            "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
        }
        backward_prefetch_map = {
            "backward_pre": BackwardPrefetch.BACKWARD_PRE,
            "backward_post": BackwardPrefetch.BACKWARD_POST,
            "none": None,
        }

        kwargs: dict[str, Any] = {
            "sharding_strategy": strategy_map[self.cfg.fsdp_sharding_strategy],
            "cpu_offload": CPUOffload(offload_params=self.cfg.fsdp_cpu_offload),
            "use_orig_params": self.cfg.fsdp_use_orig_params,
            "forward_prefetch": self.cfg.fsdp_forward_prefetch,
            "limit_all_gathers": self.cfg.fsdp_limit_all_gathers,
        }
        if self.device.type == "cuda":
            kwargs["device_id"] = torch.device("cuda", self.dist.local_rank)
        backward_prefetch = backward_prefetch_map[self.cfg.fsdp_backward_prefetch]
        if backward_prefetch is not None:
            kwargs["backward_prefetch"] = backward_prefetch

        # ---- MixedPrecision policy ----
        # Default FSDP keeps reduce_dtype in fp32 even under bf16 params, which
        # silently doubles the wire bytes for grad reduce-scatter on every
        # backward — typically the dominant comm cost at 7B/H100 FSDP. Build
        # an explicit MixedPrecision so param/reduce/buffer dtype all match
        # the chosen training dtype (or the user's override).
        compute_dtype = _resolve_dtype(self.cfg.torch_dtype)
        reduce_name = self.cfg.fsdp_reduce_dtype or self.cfg.torch_dtype
        reduce_dtype = _resolve_dtype(reduce_name)
        if compute_dtype != torch.float32:
            kwargs["mixed_precision"] = MixedPrecision(
                param_dtype=compute_dtype,
                reduce_dtype=reduce_dtype,
                buffer_dtype=compute_dtype,
            )

        # ---- HYBRID_SHARD device mesh ----
        # HSDP shards within a small intra-node group and replicates across
        # nodes/groups. With a (world_size//2, 2) mesh the reduce-scatter and
        # all-gather hops only cross 2 ranks instead of all-world, cutting
        # comms ~33% at world=4 and more at larger world sizes. Falls back to
        # plain FSDP semantics if the world is too small or odd; the validator
        # in __post_init__ allows the literal but we guard at runtime so a
        # 2-GPU smoke test still works.
        if self.cfg.fsdp_sharding_strategy == "hybrid_shard":
            world_size = int(self.dist.world_size)
            if world_size >= 4 and world_size % 2 == 0:
                # init_device_mesh requires every rank to see ALL physical
                # devices in the mesh: a (world_size//2, 2) mesh on "cuda"
                # binds rank N to local cuda:(N % visible_count). With our
                # rank-local launcher (one CUDA_VISIBLE_DEVICES gpu per rank,
                # local_rank=0 always) cuda:0 *is* the only visible device on
                # every rank, but PyTorch still validates that local_rank <
                # device_count and silently fails on the higher ranks. The
                # fix is twofold:
                #   1. try-catch the init so a hostile environment falls back
                #      to plain HYBRID_SHARD (no mesh, PyTorch picks default
                #      replicate groups) instead of deadlocking the run.
                #   2. only attempt device_mesh when every rank actually
                #      sees ``world_size`` cuda devices — that is the regime
                #      where init_device_mesh is sound.
                try:
                    from torch.distributed.device_mesh import init_device_mesh
                except ImportError:  # pragma: no cover
                    from torch.distributed._tensor import (  # type: ignore
                        init_device_mesh,
                    )
                visible = (
                    torch.cuda.device_count() if self.device.type == "cuda" else 1
                )
                # Plain HYBRID_SHARD without a device_mesh interprets
                # LOCAL_WORLD_SIZE as the intra-node shard group size. Our
                # rank-local launcher sets LOCAL_WORLD_SIZE=1 (one GPU per
                # process), which collapses HSDP to "no shard, full replicate"
                # — every rank holds the full 7B param + opt-state (~98 GB)
                # and OOMs. So if we cannot get a real mesh, downgrade the
                # strategy to FULL_SHARD instead of pretending HSDP works.
                if self.device.type == "cuda" and visible < world_size:
                    if self.dist.is_main:
                        warnings.warn(
                            f"hybrid_shard device_mesh requires every rank to "
                            f"see all {world_size} GPUs, but each rank sees only "
                            f"{visible}. Downgrading to FULL_SHARD. For HSDP "
                            f"perf launch with CUDA_VISIBLE_DEVICES exposing all "
                            f"ranks' GPUs and LOCAL_RANK / LOCAL_WORLD_SIZE set "
                            f"per-rank."
                        )
                    kwargs["sharding_strategy"] = ShardingStrategy.FULL_SHARD
                else:
                    try:
                        device_mesh = init_device_mesh(
                            "cuda" if self.device.type == "cuda" else "cpu",
                            (world_size // 2, 2),
                        )
                        kwargs["device_mesh"] = device_mesh
                    except (RuntimeError, ValueError, IndexError) as e:
                        if self.dist.is_main:
                            warnings.warn(
                                f"hybrid_shard init_device_mesh failed: {e}. "
                                f"Downgrading to FULL_SHARD."
                            )
                        kwargs["sharding_strategy"] = ShardingStrategy.FULL_SHARD
            elif self.dist.is_main:
                warnings.warn(
                    f"fsdp_sharding_strategy='hybrid_shard' but "
                    f"world_size={world_size} is < 4 or not divisible by 2; "
                    f"PyTorch will pick a default replicate-group layout."
                )

        if self.cfg.fsdp_auto_wrap:
            layer_classes = self._infer_fsdp_layer_classes(module)
            if layer_classes:
                if self.cfg.fsdp_wrap_block_group_size > 1:
                    # Coarser wrap: group every N transformer blocks into
                    # one FSDP unit. Cuts reduce-scatter call count by N at
                    # backward, giving each collective a bigger payload
                    # (better NVLink BW utilization).
                    kwargs["auto_wrap_policy"] = self._make_block_group_wrap_policy(
                        module, layer_classes, self.cfg.fsdp_wrap_block_group_size,
                    )
                    if self.dist.is_main:
                        n_blocks = sum(
                            1 for m in module.modules()
                            if isinstance(m, tuple(layer_classes))
                        )
                        print(
                            f"[trainer] FSDP coarser wrap: "
                            f"group_size={self.cfg.fsdp_wrap_block_group_size}, "
                            f"transformer_blocks={n_blocks} for {module_name}",
                            flush=True,
                        )
                else:
                    kwargs["auto_wrap_policy"] = functools.partial(
                        transformer_auto_wrap_policy,
                        transformer_layer_cls=layer_classes,
                    )
            elif self.dist.is_main:
                warnings.warn(
                    f"FSDP auto-wrap found no transformer block classes for "
                    f"{module_name}; wrapping the top-level module only."
                )

        wrapped = FSDP(module, **kwargs)
        if self.dist.is_main:
            print(
                f"[trainer] FSDP wrapped {module_name} "
                f"(strategy={self.cfg.fsdp_sharding_strategy}, "
                f"world_size={self.dist.world_size}, "
                f"reduce_dtype={reduce_name})",
                flush=True,
            )
        return wrapped

    def _wrap_ddp_if_enabled(
        self,
        module: torch.nn.Module,
        *,
        module_name: str,
        require_trainable: bool = True,
    ) -> torch.nn.Module:
        if not self._ddp_enabled:
            return module
        if _is_ddp_module(module):
            return module
        if require_trainable and not any(p.requires_grad for p in module.parameters()):
            rank0_print(
                self.dist,
                f"[trainer] DDP skip {module_name} (no trainable parameters)",
                flush=True,
            )
            return module

        from torch.nn.parallel import DistributedDataParallel as DDP

        kwargs: dict[str, Any] = {
            "broadcast_buffers": False,
            "gradient_as_bucket_view": True,
        }
        if self.device.type == "cuda":
            kwargs["device_ids"] = [self.dist.local_rank]
            kwargs["output_device"] = self.dist.local_rank
        wrapped = DDP(module, **kwargs)
        rank0_print(
            self.dist,
            f"[trainer] DDP wrapped {module_name} (world_size={self.dist.world_size})",
            flush=True,
        )
        return wrapped

    def _clip_grad_norm(
        self,
        module: torch.nn.Module,
        parameters: Iterable[torch.nn.Parameter],
        max_norm: float,
    ) -> torch.Tensor:
        if _is_fsdp_module(module) and hasattr(module, "clip_grad_norm_"):
            return module.clip_grad_norm_(max_norm)
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)

    @staticmethod
    def _maybe_no_sync(module: torch.nn.Module, enabled: bool):
        if enabled and _is_ddp_module(module):
            return module.no_sync()
        return contextlib.nullcontext()

    def _sequence_reward_advantage(
        self,
        rewards: torch.Tensor,
        *,
        group_size: int,
        scope: str,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if not self.dist.is_distributed or scope != "batch":
            return _sequence_reward_advantage(
                rewards, group_size=group_size, scope=scope, eps=eps,
            )

        import torch.distributed as dist

        r = rewards.float()
        count = torch.tensor(float(r.numel()), device=r.device)
        sums = torch.stack([r.sum(), (r * r).sum(), count])
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        denom = sums[2].clamp(min=1.0)
        mean = sums[0] / denom
        var = (sums[1] / denom - mean * mean).clamp(min=0.0)
        std = var.sqrt()
        if (std <= eps).item():
            return torch.zeros_like(r)
        return (r - mean) / std

    def _standardize_step_advantage(
        self,
        A_step: torch.Tensor,
        step_count: torch.Tensor,
        *,
        scope: str,
        group_size: int,
    ) -> torch.Tensor:
        if not self.dist.is_distributed or scope != "batch":
            return standardize_step_advantage(
                A_step, step_count, scope=scope, group_size=group_size,
            )

        import torch.distributed as dist

        if A_step.dim() != 2:
            raise ValueError(f"A_step must be 2D, got {tuple(A_step.shape)}")
        B, S_max = A_step.shape
        if step_count.device != A_step.device:
            step_count = step_count.to(A_step.device)
        arange_S = torch.arange(S_max, device=A_step.device).unsqueeze(0)
        valid = arange_S < step_count.unsqueeze(1)
        fmask = valid.to(torch.float32)
        a = A_step.float()
        local_count = fmask.sum()
        masked = a * fmask
        sums = torch.stack([
            masked.sum(),
            (masked * masked).sum(),
            local_count,
        ])
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        denom = sums[2].clamp(min=1.0)
        mean = sums[0] / denom
        var = (sums[1] / denom - mean * mean).clamp(min=0.0)
        std = var.sqrt()
        if (std <= 1e-8).item():
            return A_step.clone()
        out = (a - mean) / std * fmask
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.to(A_step.dtype) if out.dtype != A_step.dtype else out

    def _standardize_token_advantage(
        self,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        *,
        scope: str,
        group_size: int,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Standardize token-level advantages over valid response tokens."""
        if advantages.dim() != 2:
            raise ValueError(
                f"advantages must be 2D, got {tuple(advantages.shape)}"
            )
        if mask.shape != advantages.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match advantages "
                f"{tuple(advantages.shape)}"
            )

        mask_f = mask.to(torch.float32)
        a = advantages.float()
        if scope == "off":
            out = a * mask_f
            return out.to(advantages.dtype) if out.dtype != advantages.dtype else out

        if scope == "group":
            B, R = a.shape
            if B % group_size != 0:
                raise ValueError(f"B={B} not divisible by group_size={group_size}")
            g = a.view(B // group_size, group_size, R)
            m = mask_f.view(B // group_size, group_size, R)
            denom = m.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
            mean = (g * m).sum(dim=(1, 2), keepdim=True) / denom
            var = ((g - mean).square() * m).sum(dim=(1, 2), keepdim=True) / denom
            std = var.clamp(min=0.0).sqrt()
            out = (g - mean) / std.clamp(min=eps) * m
            out = torch.where(std <= eps, torch.zeros_like(out), out)
            out = out.reshape_as(a)
        elif scope == "batch":
            if self.dist.is_distributed:
                import torch.distributed as dist

                local_count = mask_f.sum()
                masked = a * mask_f
                sums = torch.stack([
                    masked.sum(),
                    (masked * masked).sum(),
                    local_count,
                ])
                dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                denom = sums[2].clamp(min=1.0)
                mean = sums[0] / denom
                var = (sums[1] / denom - mean * mean).clamp(min=0.0)
            else:
                denom = mask_f.sum().clamp(min=1.0)
                mean = (a * mask_f).sum() / denom
                var = ((a - mean).square() * mask_f).sum() / denom
            std = var.sqrt()
            if (std <= eps).item():
                out = torch.zeros_like(a)
            else:
                out = (a - mean) / std * mask_f
        else:
            raise ValueError(
                f"unknown scope {scope!r}; must be 'batch', 'group', or 'off'"
            )

        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.to(advantages.dtype) if out.dtype != advantages.dtype else out

    def _save_policy_pretrained(
        self,
        path: str,
        *,
        safe_serialization: bool = True,
        save_tokenizer: bool = True,
    ) -> None:
        """Save policy weights, handling FSDP full-state gathering when needed."""

        def _raise_if_save_failed(error_msg: Optional[str]) -> None:
            if self.dist.is_distributed:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    payload: list[Optional[str]] = [error_msg if self.dist.is_main else None]
                    dist.broadcast_object_list(payload, src=0)
                    error_msg = payload[0]
            if error_msg:
                raise RuntimeError(f"save_pretrained failed at {path}: {error_msg}")

        if _is_fsdp_module(self.model):
            from torch.distributed.fsdp import (
                FullStateDictConfig,
                FullyShardedDataParallel as FSDP,
                StateDictType,
            )

            full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.model, StateDictType.FULL_STATE_DICT, full_cfg,
            ):
                state_dict = self.model.state_dict()

            if self.dist.is_main:
                error_msg = None
                try:
                    os.makedirs(path, exist_ok=True)
                    _unwrap_parallel(self.model).save_pretrained(
                        path,
                        state_dict=state_dict,
                        safe_serialization=safe_serialization,
                    )
                    if save_tokenizer:
                        self.tokenizer.save_pretrained(path)
                except Exception as e:  # noqa: BLE001
                    error_msg = f"{type(e).__name__}: {e}"
            else:
                error_msg = None
            _raise_if_save_failed(error_msg)
            return

        if self.dist.is_main:
            error_msg = None
            try:
                os.makedirs(path, exist_ok=True)
                _unwrap_parallel(self.model).save_pretrained(
                    path, safe_serialization=safe_serialization,
                )
                if save_tokenizer:
                    self.tokenizer.save_pretrained(path)
            except Exception as e:  # noqa: BLE001
                error_msg = f"{type(e).__name__}: {e}"
        else:
            error_msg = None
        _raise_if_save_failed(error_msg)

    def _auto_run_name(self) -> str:
        cfg = self.cfg
        model_short = cfg.model_name_or_path.rsplit("/", 1)[-1]
        ds_short = cfg.dataset_name.rsplit("/", 1)[-1]
        return f"{cfg.method}_{model_short}_{ds_short}_seed{cfg.seed}"

    def _wandb_log(self, payload: dict, step: Optional[int] = None) -> None:
        if self._wandb is None:
            return
        try:
            self._wandb.log(payload, step=step)
        except Exception as e:
            warnings.warn(f"wandb.log failed: {e}")

    def _ppo_critic_train_critic(
        self,
        tiled_prompt_ids: torch.Tensor,
        tiled_prompt_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        n_epochs: int = 1,
    ) -> Tuple[float, float]:
        """Decoupled critic training pass for ``method='ppo_critic'``.

        Runs ``n_epochs`` over the rollout batch in
        ``cfg.micro_batch_size`` chunks, computing
        ``clipped_value_loss(values, old_values, returns, mask,
        cliprange=cfg.cliprange_value)`` per chunk and accumulating
        gradients across ``cfg.grad_accum_steps`` before stepping
        ``self.critic_optimizer``.

        This mirrors what a joint policy+critic backward in the
        Schulman 2017 PPO loop would compute in expectation; because
        the policy and critic parameter sets are disjoint, the
        gradient with respect to the critic params is identical
        regardless of whether the policy and critic backwards run
        together or separately. The decoupling is purely a memory
        optimization: it halves the activation peak (only one
        network's grad graph alive at a time) and is what TRL,
        OpenRLHF, and ColossalAI all do at scale.

        Returns ``(t_critic_seconds, mean_v_loss_scalar)``.
        """
        from caspo.critic import clipped_value_loss

        cfg = self.cfg
        if self.critic_model is None or self.critic_optimizer is None:
            return 0.0, 0.0
        t0 = time.time()
        B = response_ids.shape[0]
        mb = max(1, int(cfg.micro_batch_size))
        accum = max(1, int(cfg.grad_accum_steps))
        coef = float(cfg.value_loss_coef)
        cliprange = float(cfg.cliprange_value)
        micro_ranges = [(start, min(start + mb, B)) for start in range(0, B, mb)]
        n_micros_total = len(micro_ranges)
        row_token_counts = response_mask.sum(dim=1).detach().cpu()
        micro_token_counts = [
            float(row_token_counts[start:end].sum().item())
            for start, end in micro_ranges
        ]
        group_token_counts = [
            sum(micro_token_counts[i : min(i + accum, n_micros_total)])
            for i in range(0, n_micros_total, accum)
        ]
        global_group_token_counts = list(group_token_counts)
        world_size = max(1, int(self.dist.world_size))
        if self.dist.is_distributed:
            import torch.distributed as dist

            group_tokens_t = torch.tensor(
                global_group_token_counts,
                device=response_mask.device,
                dtype=torch.float32,
            )
            dist.all_reduce(group_tokens_t, op=dist.ReduceOp.SUM)
            global_group_token_counts = [
                float(x) for x in group_tokens_t.detach().cpu().tolist()
            ]

        self.critic_model.train()
        self.critic_optimizer.zero_grad(set_to_none=True)

        v_loss_terms: list[torch.Tensor] = []
        v_loss_weight_denom = 0.0
        for _ in range(int(n_epochs)):
            n_micro_in_epoch = 0
            for micro_idx, (start, end) in enumerate(micro_ranges):
                group_idx = micro_idx // accum
                global_group_tokens = (
                    global_group_token_counts[group_idx]
                    if global_group_token_counts else 0.0
                )
                micro_tokens = micro_token_counts[micro_idx]
                grad_weight = (
                    coef * micro_tokens * world_size / global_group_tokens
                    if global_group_tokens > 0.0 else 0.0
                )
                will_step = (
                    ((n_micro_in_epoch + 1) % accum == 0)
                    or (micro_idx == n_micros_total - 1)
                )
                mb_resp_mask = response_mask[start:end]
                # Per-microbatch padding trim: shrink R to the longest
                # actual response in this slice. Mirrors the policy mb
                # loop's per-mb trim and saves attention compute on
                # short responses.
                R_eff = int(mb_resp_mask.sum(dim=1).max().item())
                if R_eff == 0:
                    n_micro_in_epoch += 1
                    if will_step:
                        if cfg.critic_grad_clip and cfg.critic_grad_clip > 0:
                            try:
                                self._clip_grad_norm(
                                    self.critic_model,
                                    self.critic_model.parameters(),
                                    max_norm=cfg.critic_grad_clip,
                                )
                            except Exception:
                                pass
                        self.critic_optimizer.step()
                        if self.critic_lr_scheduler is not None:
                            self.critic_lr_scheduler.step()
                        self.critic_optimizer.zero_grad(set_to_none=True)
                    continue
                mb_resp_mask_eff = mb_resp_mask[:, :R_eff]
                full_ids = torch.cat(
                    [
                        tiled_prompt_ids[start:end],
                        response_ids[start:end, :R_eff],
                    ],
                    dim=1,
                )
                full_mask = torch.cat(
                    [tiled_prompt_mask[start:end], mb_resp_mask_eff],
                    dim=1,
                )
                crit_full = self.critic_model(
                    input_ids=full_ids, attention_mask=full_mask,
                )
                P = tiled_prompt_ids[start:end].shape[1]
                crit_values = crit_full[:, P - 1 : P - 1 + R_eff]
                crit_values = crit_values * mb_resp_mask_eff.to(
                    crit_values.dtype,
                )
                old_v_mb = old_values[start:end, :R_eff]
                ret_mb = returns[start:end, :R_eff]
                v_loss = clipped_value_loss(
                    crit_values, old_v_mb, ret_mb, mb_resp_mask_eff,
                    cliprange=cliprange,
                )
                # Weight microbatch token means by their valid-token count
                # inside each accumulation group. This matches the policy
                # objective and avoids overweighting short responses.
                (v_loss * grad_weight).backward()
                v_loss_terms.append(v_loss.detach() * float(micro_tokens))
                v_loss_weight_denom += float(micro_tokens)
                n_micro_in_epoch += 1
                if will_step:
                    if cfg.critic_grad_clip and cfg.critic_grad_clip > 0:
                        try:
                            self._clip_grad_norm(
                                self.critic_model,
                                self.critic_model.parameters(),
                                max_norm=cfg.critic_grad_clip,
                            )
                        except Exception:
                            pass
                    self.critic_optimizer.step()
                    if self.critic_lr_scheduler is not None:
                        self.critic_lr_scheduler.step()
                    self.critic_optimizer.zero_grad(set_to_none=True)

        v_loss_mean = (
            float(torch.stack(v_loss_terms).sum().item() / max(v_loss_weight_denom, 1.0))
            if v_loss_terms
            else 0.0
        )
        return time.time() - t0, v_loss_mean

    def _sync_vllm_weights_fsdp(self) -> float:
        """IPC sync from an FSDP-wrapped policy to the rank-local vLLM engine.

        Collective requirement: ``FSDP.summon_full_params`` is a collective op,
        so EVERY rank must enter this function (do NOT gate by rank). With
        ``rank0_only=False`` each rank temporarily materializes the unsharded
        parameters on its own GPU; this is necessary because each rank also
        runs its own vLLM EngineCore that consumes the full weights.
        Memory cost is roughly +param_bytes per rank for the duration of the
        ``with`` block (~14 GB for a 7B bf16 model), so callers must ensure no
        other large allocations are live at the same time.

        The IPC kernel pushes one parameter at a time and the EngineCore
        opens, copies, and closes each cuda IPC handle before the next push,
        so the source pointer only needs to be valid during the synchronous
        push. We therefore iterate ``named_parameters()`` *inside* the
        ``summon_full_params`` context — the unsharded view is alive for the
        entire push, then immediately re-sharded on context exit. No clone
        needed, which saves +param_bytes (~14 GB on 7B bf16) per rank.

        Memory cost during summon is +param_bytes/rank (the unsharded
        materialization). On Rho-1B that's negligible; on 7B it's ~14 GB
        and combined with vLLM's KV-cache budget can be tight. If OOM hits
        here, drop ``vllm_gpu_memory_utilization`` or enable
        ``fsdp_cpu_offload=true`` to push the optimizer state to CPU.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        t0 = time.time()
        with FSDP.summon_full_params(self.model, writeback=False, with_grads=False):
            base = _unwrap_parallel(self.model)
            # FSDP auto_wrap injects `_fsdp_wrapped_module.` into the qualified
            # name of every leaf parameter (e.g. `layers.0._fsdp_wrapped_module.
            # self_attn.qkv_proj.weight`). vLLM's checkpoint loader keys on the
            # bare HF name and raises KeyError otherwise. Strip the marker on
            # the way out via a tiny named-params view; the underlying tensors
            # are unmodified so IPC handles still point into FSDP's
            # summon-full storage.
            class _FsdpStrippedParamsView(torch.nn.Module):
                # Both FSDP auto_wrap and `apply_activation_checkpointing`
                # inject prefixes into the qualified parameter name. We strip
                # both. Order matters only for cosmetics — `replace` is
                # idempotent if the marker isn't present.
                _MARKERS = (
                    "._fsdp_wrapped_module.",
                    "_fsdp_wrapped_module.",
                    "._checkpoint_wrapped_module.",
                    "_checkpoint_wrapped_module.",
                )

                def __init__(self, src: torch.nn.Module):
                    super().__init__()
                    self._src = src

                @staticmethod
                def _clean(name: str) -> str:
                    for m in _FsdpStrippedParamsView._MARKERS:
                        name = name.replace(m, "." if m.startswith(".") else "")
                    return name

                def named_parameters(self, *_, **__):  # type: ignore[override]
                    for raw_name, p in self._src.named_parameters():
                        yield self._clean(raw_name), p

            view = _FsdpStrippedParamsView(base)
            # Push directly through the IPC kernel — it iterates
            # view.named_parameters() once and synchronously calls
            # collective_rpc("update_weights", ...). By the time this
            # returns, all weights have been copied into vLLM's worker
            # tensors and the source pointers are no longer referenced.
            self.sampler.sync_weights_from_model(view)
        return time.time() - t0

    @torch.no_grad()
    def _sync_vllm_weights(self) -> float:
        """Save the current policy and push it to the vLLM engine. Returns wall time."""
        if self._sync_dir is None or not hasattr(self.sampler, "sync_weights_from_path"):
            return 0.0
        # Both 'ipc' and 'nccl' backends push weights through
        # ``sync_weights_from_model``; the engine's own dispatch
        # picks the right transport. The file-based fallback below
        # is only for the legacy 'checkpoint' backend.
        if (
            self.cfg.vllm_weight_sync_backend in ("ipc", "nccl")
            and hasattr(self.sampler, "sync_weights_from_model")
        ):
            if _is_fsdp_module(self.model):
                return self._sync_vllm_weights_fsdp()
            return self.sampler.sync_weights_from_model(_unwrap_parallel(self.model))
        t0 = time.time()
        save_tokenizer = not self._sync_tokenizer_saved
        self._save_policy_pretrained(
            self._sync_dir,
            safe_serialization=True,
            save_tokenizer=save_tokenizer,
        )
        if save_tokenizer:
            self._sync_tokenizer_saved = True
        save_t = time.time() - t0
        sync_t = self.sampler.sync_weights_from_path(self._sync_dir)
        return save_t + sync_t

    def _value_forward_with_optional_update(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
        outcomes: torch.Tensor,
        prompt_index: Optional[torch.Tensor] = None,
        *,
        ref_logprobs_full: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        cfg = self.cfg
        assert self.value_model is not None, "value_model must be set for caspo method"
        mb = max(1, int(cfg.micro_batch_size))
        B = prompt_ids.shape[0]
        # Sanity: when sharing ref logprobs from the trainer, the precomputed
        # tensor must cover the full batch. Mismatched shape would silently
        # corrupt the value-model log-ratio for every microbatch slice.
        if ref_logprobs_full is not None:
            if ref_logprobs_full.shape[0] != B:
                raise ValueError(
                    f"shared ref_logprobs_full has B={ref_logprobs_full.shape[0]} "
                    f"but value-model batch is B={B}"
                )
            if ref_logprobs_full.shape[1] != response_ids.shape[1]:
                raise ValueError(
                    f"shared ref_logprobs_full has R="
                    f"{ref_logprobs_full.shape[1]} but response_ids has R="
                    f"{response_ids.shape[1]}"
                )

        def _ref_slice(start: int, end: int) -> Optional[torch.Tensor]:
            return (
                ref_logprobs_full[start:end] if ref_logprobs_full is not None else None
            )

        if self.value_optimizer is None:
            out_chunks: List[torch.Tensor] = []
            for start in range(0, B, mb):
                end = min(start + mb, B)
                with torch.no_grad():
                    out = self.value_model(
                        prompt_ids[start:end], prompt_mask[start:end],
                        response_ids[start:end], response_mask[start:end],
                        ref_logprobs=_ref_slice(start, end),
                    )
                out_chunks.append(out["log_ratio"].detach())
            return torch.cat(out_chunks, dim=0), {}

        V_x_full = None
        w_full = None
        if prompt_index is not None and (cfg.use_adb or cfg.use_dlw):
            V_x_full, w_full = compute_adb_dlw_factors(
                outcomes, prompt_index, eps=cfg.adb_dlw_eps,
            )

        self.value_optimizer.zero_grad(set_to_none=True)
        micro_ranges = [(start, min(start + mb, B)) for start in range(0, B, mb)]
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
        total_rows = max(1.0, sum(micro_row_counts))
        total_pos_rows = max(1.0, sum(micro_pos_counts))
        total_neg_rows = max(1.0, sum(micro_neg_counts))
        global_total_rows = total_rows
        world_size = max(1, int(self.dist.world_size))
        if self.dist.is_distributed:
            import torch.distributed as dist

            total_rows_t = torch.tensor(
                total_rows, device=self.device, dtype=torch.float32,
            )
            dist.all_reduce(total_rows_t, op=dist.ReduceOp.SUM)
            global_total_rows = float(total_rows_t.item())
        out_chunks = []
        agg = {"value_loss": 0.0, "value_acc": 0.0,
               "v_bar_pos": 0.0, "v_bar_neg": 0.0}
        for micro_idx, (start, end) in enumerate(micro_ranges):
            # DDP all-reduces once for the accumulated online value update.
            sync_value = micro_idx == len(micro_ranges) - 1
            with self._maybe_no_sync(self.value_model.phi, enabled=not sync_value):
                out = self.value_model(
                    prompt_ids[start:end], prompt_mask[start:end],
                    response_ids[start:end], response_mask[start:end],
                    ref_logprobs=_ref_slice(start, end),
                )
                V_x_slice = V_x_full[start:end] if (V_x_full is not None and cfg.use_adb) else None
                w_slice = w_full[start:end] if (w_full is not None and cfg.use_dlw) else None
                v_loss, v_stats = ipvrm_loss(
                    log_ratio=out["log_ratio"],
                    response_mask=response_mask[start:end],
                    outcomes=outcomes[start:end],
                    margin=self.value_model.margin,
                    prompt_value_baseline=V_x_slice,
                    loss_weights=w_slice,
                )
                micro_rows = micro_row_counts[micro_idx]
                # Distributed backends average gradients across ranks. Scale
                # by world_size/global_rows so the effective gradient is a
                # true row-weighted mean, not an equal average of rank-local
                # means when response counts differ by rank.
                grad_weight = (
                    micro_rows * world_size / global_total_rows
                    if global_total_rows > 0.0 else 0.0
                )
                (v_loss * grad_weight).backward()
            stat_weight = micro_rows / total_rows if total_rows > 0.0 else 0.0
            agg["value_loss"] += v_stats["loss"] * stat_weight
            agg["value_acc"] += v_stats["acc_at_last"] * stat_weight
            agg["v_bar_pos"] += (
                v_stats["mean_v_bar_pos"]
                * (micro_pos_counts[micro_idx] / total_pos_rows)
            )
            agg["v_bar_neg"] += (
                v_stats["mean_v_bar_neg"]
                * (micro_neg_counts[micro_idx] / total_neg_rows)
            )
            out_chunks.append(out["log_ratio"].detach())

        if cfg.value_grad_clip and cfg.value_grad_clip > 0:
            self._clip_grad_norm(
                self.value_model.phi,
                self.value_model.phi.parameters(),
                max_norm=cfg.value_grad_clip,
            )
        self.value_optimizer.step()
        self.value_optimizer.zero_grad(set_to_none=True)
        if V_x_full is not None:
            agg["adb_v_x_mean"] = float(V_x_full.mean().item())
            agg["adb_v_x_std"] = float(V_x_full.std().item())
        if w_full is not None:
            agg["dlw_w_mean"] = float(w_full.mean().item())
            agg["dlw_w_std"] = float(w_full.std().item())
        return torch.cat(out_chunks, dim=0), agg

    @staticmethod
    def _gather_response_logprobs(
        logits: torch.Tensor, response_ids: torch.Tensor, P: int, R: int,
    ) -> torch.Tensor:
        """Compute per-token logprobs for the response tokens without
        materializing the full ``[B, R, V]`` log-softmax. Uses fused
        ``cross_entropy`` (which calls ``log_softmax_lastdim`` under the
        hood) on a flattened ``[B*R, V]`` view — equivalent to
        ``log_softmax(.float()).gather(...)`` but ~2× faster and avoids the
        extra ``[B, R, V]`` fp32 activation.

        ``cross_entropy`` already accumulates softmax in fp32 internally
        when fed bf16/fp16 logits, so we feed the original dtype directly
        instead of materializing a ``[B*R, V]`` fp32 copy of the logits
        (~2 GB at vocab=32k, B*R=64k saved per microbatch).
        """
        sliced = logits[:, P - 1 : P - 1 + R, :]
        B, R_, V = sliced.shape
        # cross_entropy returns -logp; negate to get logp.
        neg_logp = F.cross_entropy(
            sliced.reshape(B * R_, V),
            response_ids.reshape(B * R_),
            reduction="none",
        )
        return (-neg_logp).reshape(B, R_)

    @staticmethod
    def _forward_response_logits(
        model,
        full_ids: torch.Tensor,
        full_mask: torch.Tensor,
        P: int,
        R: int,
    ) -> torch.Tensor:
        """Forward ``model`` and return only logits that score response tokens.

        Modern Transformers causal LMs accept ``logits_to_keep`` as either an
        int or a tensor of sequence positions. Passing the exact positions
        ``P-1 .. P+R-2`` avoids running the LM head over prompt positions and
        avoids materializing their ``[B, prompt, vocab]`` logits. Older/custom
        models fall back to a normal forward once and then cache that mode on
        the model object so we do not pay exception overhead every microbatch.
        """
        mode = getattr(model, "_caspo_logits_to_keep_mode", None)
        if mode != "full" and P > 0 and R > 0:
            idx = torch.arange(P - 1, P - 1 + R, device=full_ids.device)
            try:
                out = model(
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    logits_to_keep=idx,
                )
                logits = out.logits
                if logits.shape[1] == R:
                    setattr(model, "_caspo_logits_to_keep_mode", "tensor")
                    return logits
            except (TypeError, ValueError, IndexError):
                setattr(model, "_caspo_logits_to_keep_mode", "full")

        out = model(input_ids=full_ids, attention_mask=full_mask)
        return out.logits[:, P - 1 : P - 1 + R, :]

    def _forward_policy_logprobs(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        full_mask = torch.cat([prompt_mask, response_mask], dim=1)
        P = prompt_ids.shape[1]
        R = response_ids.shape[1]
        logits = self._forward_response_logits(self.model, full_ids, full_mask, P, R)
        return self._gather_response_logprobs(logits, response_ids, 1, R)

    @torch.no_grad()
    def _rescore_old_logprobs(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute "old" logprobs with the trainer model (pre-update).

        Why: vLLM and the HF trainer compute logprobs with different
        attention impls + softmax precisions, so the sampling-time logprobs
        coming back from vLLM are systematically different from what the
        trainer's ``model.forward`` produces — even at identical weights.
        That leaves PPO's importance ratio off by a non-trivial constant
        (we observed mean_ratio ≈ 0.1 on Rho-1B) and breaks the surrogate.
        Re-scoring with the trainer's own forward at the pre-update weights
        gives the true π_θ_old, ratio ≈ 1 at iter start, drifting cleanly
        with each PPO epoch.
        """
        cfg = self.cfg
        mb = _configured_logprob_micro_batch_size(cfg)
        B, R_max = response_ids.shape[0], response_ids.shape[1]
        # Per-microbatch padding trim: skip computing logits for trailing
        # padding tokens nobody scores. Each chunk is padded back to R_max
        # before cat so the [B, R_max] contract for callers is unchanged.
        out_chunks: List[torch.Tensor] = []
        was_training = self.model.training
        self.model.eval()
        try:
            for start in range(0, B, mb):
                end = min(start + mb, B)
                mb_mask_full = response_mask[start:end]
                R_eff = (
                    int(mb_mask_full.sum(dim=1).max().item())
                    if mb_mask_full.numel() else 0
                )
                if R_eff <= 0:
                    R_eff = R_max
                lp = self._forward_policy_logprobs(
                    prompt_ids[start:end], prompt_mask[start:end],
                    response_ids[start:end, :R_eff], mb_mask_full[:, :R_eff],
                )
                if R_eff < R_max:
                    pad = torch.zeros(
                        (lp.shape[0], R_max - R_eff),
                        dtype=lp.dtype, device=lp.device,
                    )
                    lp = torch.cat([lp, pad], dim=1)
                out_chunks.append(lp.detach())
        finally:
            if was_training:
                self.model.train()
        return torch.cat(out_chunks, dim=0)

    @torch.no_grad()
    def _forward_ref_logprobs(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        full_mask = torch.cat([prompt_mask, response_mask], dim=1)
        P = prompt_ids.shape[1]
        R = response_ids.shape[1]
        logits = self._forward_response_logits(
            self.ref_policy, full_ids, full_mask, P, R,
        )
        return self._gather_response_logprobs(
            logits, response_ids, 1, R,
        ).detach()

    @torch.no_grad()
    def _precompute_ref_logprobs(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute frozen-reference logprobs once per rollout.

        The reference policy does not change across PPO epochs, so doing this
        inside every microbatch/epoch repeats identical work when
        ``epochs_per_rollout > 1``. Caching the full ``[B, R]`` tensor costs
        little memory and saves one reference-model forward per extra epoch.
        """
        if self.ref_policy is None:
            return None
        cfg = self.cfg
        mb = _configured_logprob_micro_batch_size(cfg)
        B, R_max = response_ids.shape[0], response_ids.shape[1]
        # Per-microbatch padding trim, mirroring _rescore_old_logprobs. The
        # caller (step()) consumes ref_logprobs_full as a [B, R_max] tensor
        # sliced by [start:end] in the policy loop, so we re-pad each
        # microbatch's trimmed chunk back to R_max before cat.
        out_chunks: List[torch.Tensor] = []
        for start in range(0, B, mb):
            end = min(start + mb, B)
            mb_mask_full = response_mask[start:end]
            R_eff = (
                int(mb_mask_full.sum(dim=1).max().item())
                if mb_mask_full.numel() else 0
            )
            if R_eff <= 0:
                R_eff = R_max
            lp = self._forward_ref_logprobs(
                prompt_ids[start:end], prompt_mask[start:end],
                response_ids[start:end, :R_eff], mb_mask_full[:, :R_eff],
            )
            if R_eff < R_max:
                pad = torch.zeros(
                    (lp.shape[0], R_max - R_eff),
                    dtype=lp.dtype, device=lp.device,
                )
                lp = torch.cat([lp, pad], dim=1)
            out_chunks.append(lp.detach())
        return torch.cat(out_chunks, dim=0)

    # ------------------------------------------------------------------ vineppo MC

    @torch.no_grad()
    def _vineppo_mc_step_values(
        self,
        rollout: RolloutBatch,
        seg,
    ) -> torch.Tensor:
        """For each rollout, build (prefix, K=mc) requests at every step boundary
        and average rewards into V at each boundary. Returns
        ``V_step: [B, S_max + 1]`` matching :func:`step_values_from_log_ratios`'s
        output contract.

        Uses ``self.sampler.sample_with_prefix`` if the engine supports it
        (vLLM); otherwise falls back to per-prefix HF generation (slow).
        """
        cfg = self.cfg
        K = int(cfg.vineppo_mc_rollouts)
        # rollout is the on-CPU RolloutBatch; trainer downstream tensors live
        # on self.device. V_step must end up on self.device to match.
        device = self.device

        # Build prefix requests for the initial prompt state plus every
        # non-terminal step boundary. VinePPO advantages compare adjacent state
        # values, so V(s0) must be estimated too; setting it to zero biases the
        # first step by the prompt-level solve probability.
        boundary_after_cpu = seg.boundary_after.detach().cpu()
        step_count_cpu = seg.step_count.detach().cpu().tolist()
        response_ids_cpu = rollout.response_ids.detach().cpu()
        response_mask_cpu = rollout.response_mask.detach().cpu()
        prompt_ids_cpu = rollout.prompt_ids.detach().cpu()
        prompt_mask_cpu = rollout.prompt_mask.detach().cpu()
        prompt_index_cpu = rollout.prompt_index.detach().cpu().tolist()
        ground_truths = rollout.ground_truths

        B, R = response_ids_cpu.shape
        S_max = max(step_count_cpu) if step_count_cpu else 1

        prefix_token_lists: List[List[int]] = []
        prefix_max_tokens: List[int] = []
        # Per-prefix metadata so we can reduce after generation.
        # kind == "prompt": payload is the unique prompt index p.
        # kind == "step":   payload is (row b, step boundary index t_next).
        # last tuple entry is the already-generated response prefix text that
        # must be prepended before grading the sampled continuation.
        prefix_meta: List[Tuple[str, int, int, str]] = []

        prompt_token_lists: List[List[int]] = []
        for p_idx in range(prompt_ids_cpu.shape[0]):
            pmask = prompt_mask_cpu[p_idx]
            prompt_token_lists.append(prompt_ids_cpu[p_idx][pmask.bool()].tolist())
            prefix_token_lists.append(prompt_token_lists[-1])
            initial_budget = int(cfg.max_response_len)
            if int(cfg.vineppo_mc_max_tokens) > 0:
                initial_budget = min(initial_budget, int(cfg.vineppo_mc_max_tokens))
            prefix_max_tokens.append(initial_budget)
            prefix_meta.append(("prompt", p_idx, 0, ""))

        prompt_to_rows: dict[int, List[int]] = {}
        for b, p_idx in enumerate(prompt_index_cpu):
            prompt_to_rows.setdefault(int(p_idx), []).append(b)

        # Pre-extract per-row prompt token ids (left-padded; strip pads).
        # NB: "boundary t" in our segmentation is the LAST token of step t, so the
        # prefix = prompt + response[:boundary_idx_of_step_t + 1]. That gives V at
        # the START of step t+1.
        for b in range(B):
            vl = int(response_mask_cpu[b].sum().item())
            if vl <= 0:
                continue
            # prompt_ids/mask are [num_prompts, P_max] (one per UNIQUE prompt);
            # response b's prompt is at index prompt_index_cpu[b].
            p_idx = int(prompt_index_cpu[b])
            p_ids = prompt_token_lists[p_idx]
            r_ids = response_ids_cpu[b, :vl].tolist()
            # boundary positions where boundary_after[b, k] = True
            boundaries = (boundary_after_cpu[b, :vl].nonzero(as_tuple=True)[0].tolist())
            # Each boundary marks end of step t (0-indexed); we want V at the
            # START of step t+1, which for the LAST step (terminal) should be 0.
            # We skip the terminal boundary because V(s_terminal)=0 by convention
            # for VinePPO. Equivalently: only sample MC values for steps 0..S-2,
            # set V[S]=0 manually.
            for t, k in enumerate(boundaries):
                if t == len(boundaries) - 1:
                    # terminal — V(s_T) = 0 in VinePPO convention; skip MC.
                    continue
                prefix = p_ids + r_ids[: k + 1]
                prefix_response_text = self.tokenizer.decode(
                    r_ids[: k + 1],
                    skip_special_tokens=True,
                )
                prefix_token_lists.append(prefix)
                # The continuation only needs the remaining response budget.
                remaining = max(1, int(cfg.max_response_len) - (k + 1))
                if int(cfg.vineppo_mc_max_tokens) > 0:
                    remaining = min(remaining, int(cfg.vineppo_mc_max_tokens))
                prefix_max_tokens.append(remaining)
                prefix_meta.append(
                    ("step", b, t + 1, prefix_response_text)
                )  # V at start of step t+1

        # Generate K rollouts from each prefix.
        if not hasattr(self.sampler, "sample_with_prefix"):
            raise NotImplementedError(
                "VinePPO MC requires a rollout backend with sample_with_prefix() "
                "(use cfg.rollout_backend='vllm')."
            )
        if not prefix_token_lists:
            # Edge case: no non-terminal boundaries in the entire batch.
            V_step = torch.zeros(B, S_max + 1, device=device, dtype=torch.float32)
            return V_step

        # Keep per-prefix max-token budgets exact, but submit them as one
        # mixed-SamplingParams batch so vLLM can schedule all prefixes
        # together. The engine falls back internally if this vLLM build cannot
        # return K completions from one SamplingParams(n=K) request.
        gens = self.sampler.sample_with_prefix(
            prefix_token_lists,
            K=K,
            max_tokens=prefix_max_tokens,
            temperature=cfg.rollout_temperature,
            top_p=cfg.rollout_top_p,
        )

        # For each generated rollout, the MC value estimate = binary correctness.
        # Decode the COMPLETE trajectory (prefix's response part + new tokens)
        # against the ground truth via reward_fn. We re-decode just the new
        # tokens; the verifier looks for \boxed{...} which lives at the end.
        # Reduce: V_at_start_of_step[t] = mean over K of binary_correct.
        V_step = torch.zeros(B, S_max + 1, device="cpu", dtype=torch.float32)
        # Defensive: also build a presence mask so we can fill missing slots
        # with the row's "last known" value (constant tail).
        for (kind, first, second, prefix_text), per_prefix_gens in zip(prefix_meta, gens):
            # Grade the complete partial trajectory, not only the continuation:
            # prefixes can already contain \boxed{...} or an unfinished boxed
            # marker that the continuation completes.
            if kind == "prompt":
                p_idx = first
                gt = ground_truths[p_idx]
            else:
                b = first
                gt = ground_truths[prompt_index_cpu[b]]
            texts = [prefix_text + g.text for g in per_prefix_gens]
            scores = self.reward_fn(texts, [gt] * len(texts))
            value = float(sum(scores) / max(len(scores), 1))
            if kind == "prompt":
                for row_b in prompt_to_rows.get(first, []):
                    V_step[row_b, 0] = value
            else:
                b = first
                t_next = second
                V_step[b, t_next] = value

        # Constant-tail fill so V at terminal step = 0 explicitly.
        # V_step[b, t > step_count[b]] should also be 0 by construction.
        return V_step.to(device, non_blocking=True)

    # ------------------------------------------------------------------ step

    def step(self, examples: List[dict], *, sync_vllm: bool = True) -> dict:
        cfg = self.cfg
        G = int(cfg.group_size)
        t_step_start = time.time()

        # Release reserved-but-unallocated blocks from the prior step's
        # vLLM sync before this step grows its forward activations. Without
        # this, the allocator carries 200–500 MB of reserved fragments into
        # step 2's policy forward, OOMing at mb=4 colocated by ~80 MB even
        # after the end-of-step empty_cache (which fires *before* sync, so
        # vLLM's KV-cache grow re-fragments the pool).
        torch.cuda.empty_cache()

        # 1. Rollout (timed)
        t_rollout_start = time.time()
        rollout: RolloutBatch = self.sampler.sample(examples)
        t_rollout = time.time() - t_rollout_start

        num_prompts = len(rollout.raw_prompts)
        B = num_prompts * G
        assert rollout.response_ids.shape[0] == B

        # 2. Move to device + tile prompts
        t_device_start = time.time()
        prompt_ids = rollout.prompt_ids.to(self.device, non_blocking=True)
        prompt_mask = rollout.prompt_mask.to(self.device, non_blocking=True)
        response_ids = rollout.response_ids.to(self.device, non_blocking=True)
        response_mask = rollout.response_mask.to(self.device, non_blocking=True)
        # When rolling out via vLLM, sampling_logprobs come from a different
        # softmax/attention path than the trainer's. PPO's "old logprobs"
        # must come from the trainer's own forward at the pre-update weights
        # (otherwise ratio is biased by the rollout backend). Compute the
        # frozen π_old before any optimizer step: epoch 0 still contains
        # multiple optimizer steps for real configs, so caching old logprobs
        # from later epoch-0 microbatches would silently use already-updated
        # policy weights as π_old.
        rewards = rollout.rewards.to(self.device, non_blocking=True).float()
        prompt_index = rollout.prompt_index.to(self.device, non_blocking=True)
        tiled_prompt_ids = prompt_ids[prompt_index]
        tiled_prompt_mask = prompt_mask[prompt_index]
        prompt_token_counts = prompt_mask.sum(dim=1).float()
        response_token_counts = response_mask.sum(dim=1).float()
        prompt_len_stats = _scalar_tensor_stats(prompt_token_counts)
        response_len_stats = _scalar_tensor_stats(response_token_counts)
        response_tokens_total = float(response_token_counts.sum().item())
        t_device = time.time() - t_device_start
        t_old_logprobs_start = time.time()
        old_logprobs_full = self._rescore_old_logprobs(
            tiled_prompt_ids, tiled_prompt_mask, response_ids, response_mask,
        )
        t_old_logprobs = time.time() - t_old_logprobs_start

        # Precompute frozen reference logprobs once per rollout. We hoist
        # this *before* the value-model forward (when method=caspo) so the
        # value model can reuse the same ``[B, R]`` tensor instead of
        # running its own ``self.ref`` forward — saving one base-model
        # forward per step. ``ref_logprobs_full`` is also reused inside
        # every PPO epoch's microbatch loop, replacing the previous
        # per-step ref-precompute that ran after the advantage block.
        t_ref_logprobs_start = time.time()
        ref_logprobs_full = self._precompute_ref_logprobs(
            tiled_prompt_ids, tiled_prompt_mask, response_ids, response_mask,
        )
        t_ref_logprobs = time.time() - t_ref_logprobs_start

        method = cfg.method

        # ---- Method-specific advantage construction ----
        t_advantage_start = time.time()
        seg = None
        value_stats: dict = {}
        t_value = 0.0
        V_step = None
        A_step = None
        token_advantage: torch.Tensor
        adv_values_for_stats: torch.Tensor

        if method in {"ppo", "grpo"}:
            # No segmentation. PPO uses sequence terminal rewards with a
            # configurable baseline; GRPO uses per-prompt group-relative
            # normalized rewards. Both are broadcast uniformly to response
            # tokens and trained with the same PPO clipped surrogate.
            if method == "grpo":
                adv_per_seq = _group_relative_advantage(rewards, group_size=G)
            elif cfg.standardize_step_advantage:
                adv_per_seq = self._sequence_reward_advantage(
                    rewards, group_size=G, scope=cfg.standardize_advantage_scope,
                )
            else:
                adv_per_seq = rewards.clone()
            if cfg.advantage_clip and cfg.advantage_clip > 0:
                adv_per_seq = adv_per_seq.clamp(
                    min=-float(cfg.advantage_clip),
                    max=float(cfg.advantage_clip),
                )
            token_advantage = (
                adv_per_seq.unsqueeze(1).expand_as(response_mask).to(torch.float32)
                * response_mask.to(torch.float32)
            )
            mean_step_advantage = float(adv_per_seq.abs().mean().item())
            mean_step_count = 1.0
            adv_values_for_stats = adv_per_seq
        elif method == "ppo_critic":
            # Schulman 2017 PPO with learned critic. Per-token rewards
            # are terminal-only (verifier-RL convention): the last
            # valid response token gets the verifier reward, all
            # others zero. GAE rolls the reward backward through the
            # critic's per-token value predictions.
            from caspo.critic import compute_gae

            t_value_start = time.time()
            B_local = response_mask.shape[0]
            R_local = response_mask.shape[1]
            response_lens = response_mask.sum(dim=1).long()  # [B]
            last_idx = (response_lens - 1).clamp(min=0)
            per_token_rewards = torch.zeros(
                B_local, R_local, device=self.device, dtype=torch.float32,
            )
            # Scatter terminal reward to last response token of each row.
            row_idx = torch.arange(B_local, device=self.device)
            per_token_rewards[row_idx, last_idx] = rewards.float()
            # Mask out rows that have zero response length (shouldn't
            # happen but defensive).
            valid = (response_lens > 0).float()
            per_token_rewards = per_token_rewards * valid.unsqueeze(1)

            # Critic forward over the full (prompt + response) input.
            # ``critic_full[:, t]`` = V of the state with input[:t+1]
            # consumed, i.e., V after position t.
            # Position P-1 is end-of-prompt → V(s_0) = V(prompt).
            # Position P-1+t is end-of-(prompt + response[:t]) → V(s_t).
            # So the value window for response tokens is
            # ``critic_full[:, P-1:P-1+R]`` (length R), aligning V(s_t)
            # with response token t.
            with torch.no_grad():
                full_ids = torch.cat([tiled_prompt_ids, response_ids], dim=1)
                full_mask = torch.cat([tiled_prompt_mask, response_mask], dim=1)
                P = tiled_prompt_ids.shape[1]
                critic_full = self.critic_model(
                    input_ids=full_ids, attention_mask=full_mask,
                )
                critic_values = critic_full[:, P - 1 : P - 1 + R_local].contiguous()
            # Mask values past the response end so GAE doesn't propagate
            # through padding.
            critic_values = critic_values * response_mask.to(critic_values.dtype)

            advantages_per_token, returns_per_token = compute_gae(
                per_token_rewards, critic_values, response_mask,
                gamma=cfg.gamma, gae_lambda=cfg.ppo_gae_lambda,
            )

            # Standardize over the configured scope, respecting only valid
            # response tokens. ``group`` means the G completions for each
            # prompt are normalized together, matching the sequence-level
            # PPO/GRPO paths.
            if cfg.standardize_step_advantage:
                advantages_per_token = self._standardize_token_advantage(
                    advantages_per_token, response_mask,
                    scope=cfg.standardize_advantage_scope,
                    group_size=G,
                )
            if cfg.advantage_clip and cfg.advantage_clip > 0:
                advantages_per_token = advantages_per_token.clamp(
                    min=-float(cfg.advantage_clip),
                    max=float(cfg.advantage_clip),
                )
            token_advantage = (
                advantages_per_token.to(torch.float32)
                * response_mask.to(torch.float32)
            )

            # Stash for the policy mb loop's clipped-value-loss path.
            self._ppo_critic_returns = returns_per_token.detach()
            self._ppo_critic_old_values = critic_values.detach()

            valid_adv = advantages_per_token[response_mask.bool()]
            mean_step_advantage = (
                float(valid_adv.abs().mean().item()) if valid_adv.numel() else 0.0
            )
            mean_step_count = float(response_lens.float().mean().item())
            adv_values_for_stats = valid_adv
            t_value = time.time() - t_value_start
        else:
            # Segment for caspo + vineppo
            if cfg.segmentation_mode == "latex_aware":
                seg = segment_responses_batch_latex_aware(
                    response_ids, response_mask, self.tokenizer,
                    min_step_tokens=cfg.min_step_tokens,
                    max_steps=cfg.max_steps_per_response,
                )
            else:
                seg = segment_responses_batch(
                    response_ids, response_mask, self.delimiter_token_ids,
                    min_step_tokens=cfg.min_step_tokens,
                    max_steps=cfg.max_steps_per_response,
                )

            t_value_start = time.time()
            if method == "caspo":
                binary_outcomes = (rewards >= 0.5).float()
                # Share the trainer's already-computed ref_logprobs_full so
                # the value model skips its own ``self.ref`` forward — saves
                # one base-model forward over the rollout per step. The
                # value model raises if shapes don't match the response
                # tensor (which would indicate a wiring bug).
                log_ratio, value_stats = self._value_forward_with_optional_update(
                    tiled_prompt_ids, tiled_prompt_mask, response_ids, response_mask,
                    binary_outcomes, prompt_index=prompt_index,
                    ref_logprobs_full=ref_logprobs_full,
                )
                V_step = step_values_from_log_ratios(
                    log_ratio, response_mask, seg.boundary_after, seg.step_count,
                )
                V_step_for_advantage = transform_step_values_for_advantage(
                    V_step, cfg.caspo_advantage_transform,
                )
                value_stats["caspo_advantage_transform_id"] = {
                    "value": 0.0,
                    "prob": 1.0,
                    "logprob": 2.0,
                }[cfg.caspo_advantage_transform]
            else:  # vineppo
                V_step = self._vineppo_mc_step_values(rollout, seg)
                V_step_for_advantage = V_step
                value_stats["vineppo_mc_K"] = int(cfg.vineppo_mc_rollouts)
            t_value = time.time() - t_value_start

            A_step = step_td_advantage(
                V_step_for_advantage, rewards, seg.step_count, gamma=cfg.gamma,
            )
            if cfg.standardize_step_advantage:
                A_step = self._standardize_step_advantage(
                    A_step, seg.step_count,
                    scope=cfg.standardize_advantage_scope,
                    group_size=G,
                )
            # Clip advantage magnitude to prevent rare large values from
            # dominating the policy gradient (rare in well-mixed batches,
            # common when V_φ produces a few large values among many zeros).
            if cfg.advantage_clip and cfg.advantage_clip > 0:
                A_step = A_step.clamp(min=-float(cfg.advantage_clip),
                                       max=float(cfg.advantage_clip))
            token_advantage = broadcast_step_advantage_to_tokens(
                A_step, seg.step_id, response_mask,
            )
            valid_step_mask = (
                torch.arange(A_step.shape[1], device=A_step.device).unsqueeze(0)
                < seg.step_count.to(A_step.device).unsqueeze(1)
            )
            valid_A = A_step[valid_step_mask]
            mean_step_advantage = float(valid_A.abs().mean().item()) if valid_A.numel() else 0.0
            mean_step_count = float(seg.step_count.float().mean().item())
            value_stats["t_value_forward_s"] = t_value
            adv_values_for_stats = valid_A
        t_advantage = time.time() - t_advantage_start - t_value
        adv_stats = _scalar_tensor_stats(adv_values_for_stats)

        # Reward stats
        rewards_grouped = rewards.view(num_prompts, G)
        pass_at_g = float((rewards_grouped > 0.5).any(dim=1).float().mean().item())
        reward_stats = _scalar_tensor_stats(rewards)
        mean_reward = reward_stats["mean"]
        positive_frac = float((rewards >= 0.5).float().mean().item())

        # 7. Micro-batched policy forward + PPO loss
        # Reset peak-memory counter so the gpu_mem_peak_gb reported below
        # reflects this step's peak only, not a cumulative high-water mark.
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        t_policy_start = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        mb = max(1, int(cfg.micro_batch_size))
        accum = max(1, int(cfg.grad_accum_steps))
        n_epochs = max(1, int(cfg.epochs_per_rollout))

        # On-device accumulators: each microbatch appends one stacked
        # tensor (per-stat scalar weighted by micro_tokens). We materialize
        # all of them in a single ``torch.stack(...).tolist()`` at the end
        # of the policy loop instead of paying a CUDA→host sync per
        # ``.item()`` per microbatch (~5 syncs per micro). Reference:
        # caspo/value/train_value.py:144 uses the same pattern.
        accum_device = self.device
        accum_dtype = torch.float32
        zero_scalar = torch.zeros((), device=accum_device, dtype=accum_dtype)
        loss_terms: List[torch.Tensor] = []
        pg_terms: List[torch.Tensor] = []
        logp_terms: List[torch.Tensor] = []
        clip_terms: List[torch.Tensor] = []
        ratio_terms: List[torch.Tensor] = []
        kl_terms: List[torch.Tensor] = []
        total_kl_seen = 0   # only counts micros that produced a KL estimate
        n_micro = 0
        token_weight_denom = 0.0
        n_optim_steps = 0
        grad_norm_sum = 0.0
        grad_norm_max = 0.0
        grad_norm_count = 0

        # ``ref_logprobs_full`` was already precomputed at the top of step()
        # so the value model could share it (see hoist comment above). It
        # stays valid for every PPO epoch since π_ref is frozen.

        # Iterate the SAME rollout n_epochs times. π_old / advantages stay
        # frozen (computed before this loop); only π_θ updates each epoch.
        # This is standard PPO behavior — the importance ratio drifts away
        # from 1 over epochs, and the clipped surrogate guards against it.
        # ppo_clipped_loss returns a valid-token mean for each microbatch.
        # Weight each micro by its token count inside the accumulation group
        # so the optimizer sees the same objective as one large token-mean
        # batch. Equal averaging of micro means overweights short responses.
        micro_ranges = [(start, min(start + mb, B)) for start in range(0, B, mb)]
        n_micros_total = len(micro_ranges)
        row_token_counts = response_mask.sum(dim=1).detach().cpu()
        micro_token_counts = [
            float(row_token_counts[start:end].sum().item())
            for start, end in micro_ranges
        ]
        group_token_counts = [
            sum(micro_token_counts[i : min(i + accum, n_micros_total)])
            for i in range(0, n_micros_total, accum)
        ]
        global_group_token_counts = list(group_token_counts)
        world_size = max(1, int(self.dist.world_size))
        if self.dist.is_distributed:
            import torch.distributed as dist

            group_tokens_t = torch.tensor(
                global_group_token_counts, device=self.device, dtype=torch.float32,
            )
            dist.all_reduce(group_tokens_t, op=dist.ReduceOp.SUM)
            global_group_token_counts = [
                float(x) for x in group_tokens_t.detach().cpu().tolist()
            ]
        for epoch in range(n_epochs):
            # Reset accumulator counter so the optimizer step boundary
            # ``n_micro_in_epoch % accum == 0`` is computed within the epoch.
            n_micro_in_epoch = 0
            for micro_idx, (start, end) in enumerate(micro_ranges):
                group_idx = micro_idx // accum
                global_group_tokens = (
                    global_group_token_counts[group_idx]
                    if global_group_token_counts else 0.0
                )
                micro_tokens = micro_token_counts[micro_idx]
                grad_weight = (
                    micro_tokens * world_size / global_group_tokens
                    if global_group_tokens > 0.0 else 0.0
                )
                will_step = ((n_micro_in_epoch + 1) % accum == 0) or (end == B)
                # Per-microbatch padding trim: MATH responses average ~600
                # tokens but max_response_len pads to 1024+. Slice the R-axis
                # down to the longest live response in this microbatch so the
                # policy forward (the dominant FLOPS term) doesn't compute
                # logits for padding it'll just multiply out of the loss.
                # ppo_clipped_loss requires logprobs/old_logprobs/adv/mask to
                # all share shape, so trim every R-keyed tensor consistently.
                mb_mask_full = response_mask[start:end]
                R_eff = int(mb_mask_full.sum(dim=1).max().item()) if mb_mask_full.numel() else 0
                if R_eff <= 0:
                    R_eff = mb_mask_full.shape[1]
                mb_resp_ids = response_ids[start:end, :R_eff]
                mb_resp_mask = mb_mask_full[:, :R_eff]
                with self._maybe_no_sync(self.model, enabled=not will_step):
                    new_logprobs = self._forward_policy_logprobs(
                        tiled_prompt_ids[start:end], tiled_prompt_mask[start:end],
                        mb_resp_ids, mb_resp_mask,
                    )
                    old_lp = old_logprobs_full[start:end, :R_eff]
                    adv = token_advantage[start:end, :R_eff]

                    ref_lp = (
                        ref_logprobs_full[start:end, :R_eff]
                        if ref_logprobs_full is not None else None
                    )

                    loss, stats = ppo_clipped_loss(
                        logprobs=new_logprobs, old_logprobs=old_lp, advantage=adv,
                        response_mask=mb_resp_mask,
                        clip_eps_low=cfg.clip_eps_low, clip_eps_high=cfg.clip_eps_high,
                        ref_logprobs=ref_lp, kl_coef=cfg.kl_coef,
                        kl_estimator=cfg.kl_estimator,
                    )
                    # PPO+critic: critic forward+backward is run in a
                    # SEPARATE pass after the policy mb loop completes
                    # (see ``_ppo_critic_train_critic`` below). Doing
                    # joint forward+backward inside the policy mb loop
                    # doubles the activation peak (~120 GB at 7B) and
                    # OOMs even at mb=1; decoupling halves it and
                    # produces identical gradients (the critic and
                    # policy parameter sets are disjoint, so a joint
                    # backward over the same graph is not numerically
                    # different from two separate backwards).
                    (loss * grad_weight).backward()

                with torch.no_grad():
                    # Stage on-device contributions; we ``.tolist()`` the
                    # whole stack once after the policy loop. ``micro_tokens``
                    # is a Python float — multiplying by it avoids creating a
                    # tensor scalar for the weight.
                    w = float(micro_tokens)
                    loss_terms.append(loss.detach().to(accum_dtype) * w)
                    pg_terms.append(stats["pg_loss"].to(accum_dtype) * w)
                    logp_terms.append(stats["mean_logp"].to(accum_dtype) * w)
                    clip_terms.append(stats["clip_frac"].to(accum_dtype) * w)
                    ratio_terms.append(stats["mean_ratio"].to(accum_dtype) * w)
                    token_weight_denom += w
                    if "mean_kl" in stats:
                        kl_terms.append(stats["mean_kl"].to(accum_dtype) * w)
                        total_kl_seen += 1
                    n_micro += 1
                    n_micro_in_epoch += 1

                if will_step:
                    grad_norm = None
                    if cfg.grad_clip and cfg.grad_clip > 0:
                        grad_norm = float(
                            self._clip_grad_norm(
                                self.model,
                                self.model.parameters(),
                                max_norm=cfg.grad_clip,
                            )
                        )
                    if grad_norm is not None:
                        grad_norm_sum += float(grad_norm)
                        grad_norm_max = max(grad_norm_max, float(grad_norm))
                        grad_norm_count += 1
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    # Critic update is decoupled from the policy mb
                    # loop (see ``_ppo_critic_train_critic`` invoked
                    # AFTER this loop completes). Here we only step
                    # the policy.
                    n_optim_steps += 1
        # Single sync: materialize all per-stat token-weighted sums in one
        # CUDA→host round-trip. ``torch.stack`` over zero-dim tensors costs
        # virtually nothing; the saving is ~5 syncs/microbatch * n_micro
        # eliminated.
        with torch.no_grad():
            stat_stack = torch.stack(
                [
                    torch.stack(loss_terms).sum() if loss_terms else zero_scalar,
                    torch.stack(pg_terms).sum() if pg_terms else zero_scalar,
                    torch.stack(logp_terms).sum() if logp_terms else zero_scalar,
                    torch.stack(clip_terms).sum() if clip_terms else zero_scalar,
                    torch.stack(ratio_terms).sum() if ratio_terms else zero_scalar,
                    torch.stack(kl_terms).sum() if kl_terms else zero_scalar,
                ]
            )
            stat_vals = stat_stack.tolist()
        total_loss, total_pg, total_logp, total_clip_frac, total_ratio, total_kl = (
            stat_vals[0], stat_vals[1], stat_vals[2],
            stat_vals[3], stat_vals[4], stat_vals[5],
        )
        t_policy = time.time() - t_policy_start

        # PPO+critic: critic update runs AFTER the policy mb loop
        # completes. Same gradient algebra as a joint backward (the
        # parameter sets are disjoint) but half the activation peak
        # because we only ever materialize one network's grad graph
        # at a time. Reuses cfg.micro_batch_size + cfg.grad_accum_steps
        # for granularity; runs cfg.epochs_per_rollout passes over the
        # batch (matching the policy loop's effective compute count).
        if method == "ppo_critic" and self.critic_model is not None:
            t_value_extra, v_loss_avg = self._ppo_critic_train_critic(
                tiled_prompt_ids, tiled_prompt_mask,
                response_ids, response_mask,
                self._ppo_critic_old_values, self._ppo_critic_returns,
                n_epochs=int(cfg.epochs_per_rollout),
            )
            t_value = t_value + t_value_extra
            value_stats = {"v_loss": v_loss_avg, "t_value_forward_s": t_value}
            # Drop the stashed tensors so they don't leak across steps.
            self._ppo_critic_old_values = None
            self._ppo_critic_returns = None

        # Free fragmented allocations from the policy mb loop and decoupled
        # critic backward before vLLM sync. ``summon_full_params`` inside
        # ``_sync_vllm_weights_fsdp`` materializes ~14 GB of unsharded bf16
        # params; without first releasing the now-unused stat tensors and
        # ref/old logprob caches, the allocator pool is fragmented and the
        # weight sync OOMs colocated vLLM. Saves 200–800 MB / rank in our
        # measurements (caspo trainer wave 4, 2026-04-27).
        del loss_terms, pg_terms, logp_terms, clip_terms, ratio_terms, kl_terms
        # ref_logprobs_full / old_logprobs_full may not exist on every
        # branch (no-ref-policy GRPO, first-epoch caspo). Use locals()
        # rather than dir() — dir() in a method returns instance attrs,
        # not the function's locals.
        _locals = locals()
        if "old_logprobs_full" in _locals:
            del old_logprobs_full
        if "ref_logprobs_full" in _locals:
            del ref_logprobs_full
        del _locals
        torch.cuda.empty_cache()

        # Sync vLLM weights for the next rollout (if applicable). The train
        # loop passes ``sync_vllm=False`` for its final step because no later
        # rollout will consume the engine; direct callers keep the historical
        # default of syncing after every step.
        t_sync = self._sync_vllm_weights() if self._sync_dir and sync_vllm else 0.0

        denom = max(1.0, token_weight_denom)
        # Average KL only over the micros that actually produced one
        # (i.e. when ref_policy is enabled). When no KL was produced
        # (no ref_policy, kl_coef=0), omit ``mean_kl`` from the result so
        # downstream logging treats it as missing rather than reporting a
        # misleading 0.0 (which would suggest π_θ ≈ π_ref).
        result = {
            "loss": total_loss / denom,
            "pg_loss": total_pg / denom,
            "mean_logp": total_logp / denom,
            "mean_ratio": total_ratio / denom,
            "clip_frac": total_clip_frac / denom,
            "mean_reward": mean_reward,
            "reward_std": reward_stats["std"],
            "reward_min": reward_stats["min"],
            "reward_max": reward_stats["max"],
            "pass_at_g": pass_at_g,
            "positive_frac": positive_frac,
            "mean_step_count": mean_step_count,
            "mean_step_advantage": mean_step_advantage,
            "adv_mean": adv_stats["mean"],
            "adv_std": adv_stats["std"],
            "adv_min": adv_stats["min"],
            "adv_max": adv_stats["max"],
            "num_prompts": num_prompts,
            "num_responses": B,
            "prompt_tokens_mean": prompt_len_stats["mean"],
            "prompt_tokens_max": prompt_len_stats["max"],
            "response_tokens_mean": response_len_stats["mean"],
            "response_tokens_max": response_len_stats["max"],
            "response_tokens_total": response_tokens_total,
            "lr": self.optimizer.param_groups[0]["lr"],
            "method": method,
            "epochs_per_rollout": n_epochs,
            "n_microbatches": n_micro,
            "n_optim_steps": n_optim_steps,
            "t_rollout_s": t_rollout,
            "t_device_s": t_device,
            "t_old_logprobs_s": t_old_logprobs,
            "t_advantage_s": t_advantage,
            "t_ref_logprobs_s": t_ref_logprobs,
            "t_policy_s": t_policy,
            "t_sync_s": t_sync,
            "t_step_s": time.time() - t_step_start,
        }
        if total_kl_seen > 0:
            result["mean_kl"] = total_kl / denom
            result["kl_term"] = float(cfg.kl_coef) * result["mean_kl"]
        if grad_norm_count > 0:
            result["grad_norm_mean"] = grad_norm_sum / float(grad_norm_count)
            result["grad_norm_max"] = grad_norm_max
        if value_stats:
            result.update(value_stats)
            if self.value_optimizer is not None:
                result["value_lr"] = self.value_optimizer.param_groups[0]["lr"]

        # GPU memory snapshot
        try:
            result["gpu_mem_alloc_gb"] = (
                torch.cuda.memory_allocated() / (1024 ** 3)
            )
            result["gpu_mem_peak_gb"] = (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            )
        except Exception:
            pass

        if self.dist.is_distributed:
            result = reduce_numeric_stats(
                result,
                sum_keys=("num_prompts", "num_responses", "response_tokens_total"),
            )
            result["world_size"] = self.dist.world_size

        return result

    # ------------------------------------------------------------------ loop

    def _next_examples(self, n: int) -> List[dict]:
        return [next(self._train_iter) for _ in range(n)]

    def _prompts_per_rank(self) -> int:
        """Resolve cfg.prompts_per_step (a *global* count, paper-faithful) into
        a per-rank prompt count.

        cfg.prompts_per_step is the number of unique prompts that participate
        in one outer PPO step *globally* — i.e. the count that, multiplied by
        ``group_size``, yields the global episodes-per-step the paper reports
        ("64 prompts × G=8 = 512 episodes"; see configs/*.yaml comments and
        VinePPO Table 2).

        Each rank holds an independent ``[rank::world_size]`` shard of the
        dataset, so to roll out N prompts globally, each rank pulls
        ``ceil(N / world_size)`` from its own shard. We round up so the
        global count is *at least* the requested N (avoids dropping prompts
        when N is not divisible by world_size).

        Returns the per-rank prompt count.
        """
        cfg = self.cfg
        world = max(1, int(self.dist.world_size))
        # ceil-divide so total >= cfg.prompts_per_step. The trainer used
        # to call _next_examples(cfg.prompts_per_step) directly, which made
        # the *per-rank* count = global count and silently inflated the
        # actual global count by world_size. That changed the effective
        # episodes-per-step (and thus optimizer step count) by world_size,
        # making 4-GPU runs do 4x the work the paper specifies.
        per_rank = (int(cfg.prompts_per_step) + world - 1) // world
        return max(1, per_rank)

    def train(self) -> None:
        cfg = self.cfg
        t0 = time.time()

        # Opt-in torch.profiler. Zero overhead when profile_steps=0 — we
        # never enter the with-block or call prof.step(). When enabled,
        # schedule(warmup=2, active=profile_steps, repeat=1) skips the first
        # 2 steps (cold-start dataloader / kernel autotune), records the
        # next ``profile_steps`` steps, then stays idle.
        if cfg.profile_steps > 0:
            profile_dir = os.path.join(cfg.output_dir, "profile")
            os.makedirs(profile_dir, exist_ok=True)
            print(
                f"[trainer] torch.profiler ON — warmup=2 active={cfg.profile_steps} "
                f"repeat=1 → traces at {profile_dir}",
                flush=True,
            )
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(
                    wait=0, warmup=2, active=int(cfg.profile_steps), repeat=1,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
                record_shapes=True,
                with_stack=False,
            ) as prof:
                while self.global_step < cfg.max_steps:
                    examples = self._next_examples(self._prompts_per_rank())
                    is_final_step = (self.global_step + 1) >= int(cfg.max_steps)
                    stats = self.step(examples, sync_vllm=not is_final_step)
                    self.global_step += 1
                    if cfg.log_every and (self.global_step % cfg.log_every == 0):
                        self._log(stats, t0)
                    if (
                        cfg.save_every
                        and self.global_step < int(cfg.max_steps)
                        and self.global_step % int(cfg.save_every) == 0
                    ):
                        self.save_checkpoint(final=False)
                    prof.step()
        else:
            while self.global_step < cfg.max_steps:
                examples = self._next_examples(self._prompts_per_rank())
                is_final_step = (self.global_step + 1) >= int(cfg.max_steps)
                stats = self.step(examples, sync_vllm=not is_final_step)
                self.global_step += 1
                if cfg.log_every and (self.global_step % cfg.log_every == 0):
                    self._log(stats, t0)
                if (
                    cfg.save_every
                    and self.global_step < int(cfg.max_steps)
                    and self.global_step % int(cfg.save_every) == 0
                ):
                    self.save_checkpoint(final=False)
                    if (
                        int(cfg.eval_every) > 0
                        and self.global_step % int(cfg.eval_every) == 0
                    ):
                        self._dispatch_periodic_eval(
                            os.path.join(cfg.output_dir, f"step_{self.global_step}")
                        )

        # Honor an env-level skip switch for smoke runs (each 7B final save
        # is ~13 GB and fills the scratch disk during rapid-iteration
        # benchmarking). Set CASPO_SKIP_FINAL_SAVE=1 in the launcher to
        # skip the post-training save_pretrained(final=True). Default is
        # to save (production behavior unchanged).
        if os.environ.get("CASPO_SKIP_FINAL_SAVE", "0") not in ("1", "true", "True"):
            self.save_checkpoint(final=True)
        elif self.dist.is_main:
            print(
                "[trainer] CASPO_SKIP_FINAL_SAVE=1 → skipping final "
                "save_checkpoint (smoke mode)",
                flush=True,
            )
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass

    def _log(self, stats: dict, t0: float) -> None:
        if not self.dist.is_main:
            return
        elapsed = time.time() - t0
        method = stats.get("method", "?")
        msg = (
            f"[{method} step {self.global_step}/{self.cfg.max_steps}] "
            f"loss={stats['loss']:.4f} pg={stats['pg_loss']:.4f} "
            f"reward={stats['mean_reward']:.3f} pass@G={stats['pass_at_g']:.3f} "
            f"|A|={stats['mean_step_advantage']:.3f} "
            f"steps/r={stats['mean_step_count']:.1f} "
            f"clip_frac={stats['clip_frac']:.3f} ratio={stats['mean_ratio']:.3f} "
            f"lr={stats['lr']:.2e}"
        )
        if "mean_kl" in stats:
            msg += f" kl={stats['mean_kl']:.4f}"
        if "value_loss" in stats:
            msg += f" v_loss={stats['value_loss']:.4f} v_acc={stats['value_acc']:.3f}"
        t_value = float(stats.get("t_value_forward_s", 0.0))
        t_step = float(
            stats.get(
                "t_step_s",
                stats["t_rollout_s"] + t_value + stats["t_policy_s"] + stats["t_sync_s"],
            )
        )
        msg += (
            f" t_roll={stats['t_rollout_s']:.1f}s "
            f"t_old={float(stats.get('t_old_logprobs_s', 0.0)):.1f}s "
            f"t_value={t_value:.1f}s "
            f"t_ref={float(stats.get('t_ref_logprobs_s', 0.0)):.1f}s "
            f"t_pol={stats['t_policy_s']:.1f}s "
            f"t_sync={stats['t_sync_s']:.1f}s "
            f"t_step={t_step:.1f}s "
            f"elapsed={elapsed:.1f}s"
        )
        print(msg, flush=True)

        # Wandb log: namespace metrics by category
        log_payload = {
            "policy/loss": stats["loss"],
            "policy/pg_loss": stats["pg_loss"],
            "policy/mean_logp": float(stats.get("mean_logp", 0.0)),
            "policy/mean_ratio": stats["mean_ratio"],
            "policy/clip_frac": stats["clip_frac"],
            "policy/lr": stats["lr"],
            "reward/mean": stats["mean_reward"],
            "reward/std": float(stats.get("reward_std", 0.0)),
            "reward/min": float(stats.get("reward_min", 0.0)),
            "reward/max": float(stats.get("reward_max", 0.0)),
            "reward/pass_at_g": stats["pass_at_g"],
            "reward/positive_frac": stats["positive_frac"],
            "seg/mean_step_count": stats["mean_step_count"],
            "adv/abs_mean": stats["mean_step_advantage"],
            "adv/mean": float(stats.get("adv_mean", 0.0)),
            "adv/std": float(stats.get("adv_std", 0.0)),
            "adv/min": float(stats.get("adv_min", 0.0)),
            "adv/max": float(stats.get("adv_max", 0.0)),
            "rollout/num_prompts": int(stats.get("num_prompts", 0)),
            "rollout/num_responses": int(stats.get("num_responses", 0)),
            "rollout/prompt_tokens_mean": float(stats.get("prompt_tokens_mean", 0.0)),
            "rollout/prompt_tokens_max": float(stats.get("prompt_tokens_max", 0.0)),
            "rollout/response_tokens_mean": float(stats.get("response_tokens_mean", 0.0)),
            "rollout/response_tokens_max": float(stats.get("response_tokens_max", 0.0)),
            "rollout/response_tokens_total": float(stats.get("response_tokens_total", 0.0)),
            "optim/epochs_per_rollout": int(stats.get("epochs_per_rollout", 0)),
            "optim/n_microbatches": int(stats.get("n_microbatches", 0)),
            "optim/n_steps": int(stats.get("n_optim_steps", 0)),
            "time/rollout_s": stats["t_rollout_s"],
            "time/device_s": float(stats.get("t_device_s", 0.0)),
            "time/old_logprobs_s": float(stats.get("t_old_logprobs_s", 0.0)),
            "time/advantage_s": float(stats.get("t_advantage_s", 0.0)),
            "time/value_s": float(stats.get("t_value_forward_s", 0.0)),
            "time/ref_logprobs_s": float(stats.get("t_ref_logprobs_s", 0.0)),
            "time/policy_s": stats["t_policy_s"],
            "time/sync_s": stats["t_sync_s"],
            "time/step_s": float(stats.get("t_step_s", 0.0)),
            "time/total_elapsed_s": elapsed,
            "step/global": self.global_step,
        }
        if "mean_kl" in stats:
            log_payload["policy/mean_kl"] = stats["mean_kl"]
        if "kl_term" in stats:
            log_payload["policy/kl_term"] = stats["kl_term"]
        if "grad_norm_mean" in stats:
            log_payload["optim/grad_norm_mean"] = stats["grad_norm_mean"]
            log_payload["optim/grad_norm_max"] = stats.get("grad_norm_max", 0.0)
        for k in ("value_loss", "value_acc", "v_bar_pos", "v_bar_neg",
                  "adb_v_x_mean", "adb_v_x_std", "dlw_w_mean", "dlw_w_std",
                  "value_lr", "vineppo_mc_K", "caspo_advantage_transform_id",
                  "t_value_forward_s",
                  "gpu_mem_alloc_gb", "gpu_mem_peak_gb"):
            if k in stats:
                log_payload[f"value/{k}" if k.startswith(("value", "v_bar", "adb", "dlw"))
                            else f"misc/{k}"] = stats[k]
        self._wandb_log(log_payload, step=self.global_step)

    def _dispatch_periodic_eval(self, ckpt_path: str) -> None:
        """Fire-and-forget ``scripts/eval.py`` on a checkpoint dir.

        Gated by ``cfg.eval_during_training_gpu >= 0`` and rank-0-only.
        Writes result JSON beside the checkpoint at
        ``<ckpt>/eval_results_<bench>_k<K>_limit<N>.json`` (the eval
        script's own output convention) plus a stdout/stderr log at
        ``<ckpt>/periodic_eval.log``.

        Subprocess uses ``CUDA_VISIBLE_DEVICES=<eval_gpu>`` and inherits
        the trainer's environment (conda env, perf_env.sh exports). Does
        NOT block the training loop — process is started detached.
        """
        cfg = self.cfg
        if not getattr(self, "dist", None) or not self.dist.is_main:
            return
        # Cfg field is the canonical knob; ``CASPO_EVAL_GPU`` env override
        # is a shortcut so users can flip on periodic eval without editing
        # the YAML / passing a --override flag.
        env_gpu = os.environ.get("CASPO_EVAL_GPU")
        eval_gpu = int(env_gpu) if env_gpu not in (None, "") else int(
            getattr(cfg, "eval_during_training_gpu", -1)
        )
        if eval_gpu < 0:
            return
        if not os.path.isdir(ckpt_path):
            warnings.warn(
                f"[periodic_eval] ckpt path missing: {ckpt_path} — skipping"
            )
            return

        # Resolve repo root from this file's location so the subprocess
        # finds ``scripts/eval.py`` regardless of cwd.
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        eval_script = os.path.join(repo_root, "scripts", "eval.py")
        config_path = os.environ.get(
            "CASPO_PERIODIC_EVAL_CONFIG",
            os.path.join(repo_root, "configs", "caspo_rho1b_math.yaml")
            if "rho1b" in cfg.output_dir.lower()
            else os.path.join(repo_root, "configs", "caspo_deepseekmath7b_math.yaml"),
        )

        log_path = os.path.join(ckpt_path, "periodic_eval.log")
        cmd = [
            sys.executable, "-u", eval_script,
            "--config", config_path,
            "--override", f"model_name_or_path={ckpt_path}",
            "--benchmarks", str(cfg.eval_during_training_benchmarks),
            "--k", str(int(cfg.eval_during_training_k)),
            "--temperature", str(float(cfg.eval_during_training_temperature)),
            "--top-p", "0.9",
            "--backend", "vllm",
            "--gpu-memory-utilization", str(float(cfg.eval_during_training_vllm_util)),
        ]
        if int(cfg.eval_during_training_limit) > 0:
            cmd += ["--limit", str(int(cfg.eval_during_training_limit))]

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(eval_gpu)

        try:
            log_fh = open(log_path, "w")
            proc = subprocess.Popen(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # detach; trainer signals don't reach it
            )
            print(
                f"[periodic_eval] dispatched eval pid={proc.pid} on GPU "
                f"{eval_gpu} for {ckpt_path} (log: {log_path})",
                flush=True,
            )
        except Exception as e:
            warnings.warn(f"[periodic_eval] failed to dispatch: {e}")

    def save_checkpoint(self, final: bool = False) -> str:
        cfg = self.cfg
        sub = "final" if final else f"step_{self.global_step}"
        path = os.path.join(cfg.output_dir, sub)
        self._save_policy_pretrained(
            path, safe_serialization=True, save_tokenizer=True,
        )
        meta_error: Optional[str] = None
        if self.dist.is_main:
            try:
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, "caspo_run_config.json"), "w") as f:
                    json.dump(asdict(cfg), f, indent=2, default=str)
                with open(os.path.join(path, "step.json"), "w") as f:
                    json.dump({"global_step": self.global_step}, f)
                print(f"[checkpoint] saved to {path}", flush=True)
            except Exception as e:  # noqa: BLE001
                meta_error = f"{type(e).__name__}: {e}"
        if self.dist.is_distributed:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                payload: list[Optional[str]] = [
                    meta_error if self.dist.is_main else None
                ]
                dist.broadcast_object_list(payload, src=0)
                meta_error = payload[0]
        if meta_error:
            raise RuntimeError(f"checkpoint metadata write failed at {path}: {meta_error}")
        dist_barrier()
        return path
