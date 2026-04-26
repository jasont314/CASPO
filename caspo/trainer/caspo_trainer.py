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


def _build_lr_schedule(optimizer, warmup_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0
    return LambdaLR(optimizer, lr_lambda)


def _cycle(iterable: Iterable):
    while True:
        any_yielded = False
        for x in iterable:
            any_yielded = True
            yield x
        if not any_yielded:
            raise RuntimeError("data iterable yielded no examples")


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
        if cfg.use_gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
            except Exception as e:
                warnings.warn(f"could not enable gradient checkpointing: {e}")

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
            if cfg.update_value_during_policy:
                self.value_optimizer = AdamW(
                    (p for p in self.value_model.phi.parameters() if p.requires_grad),
                    lr=cfg.online_value_lr,
                    weight_decay=cfg.value_weight_decay,
                )

        # ---- optimizer + schedule ----
        self.optimizer = AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        self.lr_scheduler = _build_lr_schedule(self.optimizer, cfg.warmup_steps)

        # ---- delimiter token ids (for token_delimiter segmentation only) ----
        self.delimiter_token_ids = _tokenize_delimiter(self.tokenizer, cfg.step_delimiter)

        # ---- reward + data ----
        self.reward_fn = MathRewardFn()
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
        self._train_iter = _cycle(self.train_examples)

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
            engine_kwargs = dict(
                gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
                tensor_parallel_size=cfg.vllm_tensor_parallel_size,
                enforce_eager=cfg.vllm_enforce_eager,
                seed=cfg.seed,
                max_num_seqs=cfg.vllm_max_num_seqs,
                max_num_batched_tokens=cfg.vllm_max_num_batched_tokens,
                max_inflight_requests=cfg.vllm_max_inflight_requests,
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
        """Infer transformer block classes for FSDP auto-wrapping."""

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

        if self.cfg.fsdp_auto_wrap:
            layer_classes = self._infer_fsdp_layer_classes(module)
            if layer_classes:
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
                f"world_size={self.dist.world_size})",
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

    @torch.no_grad()
    def _sync_vllm_weights(self) -> float:
        """Save the current policy and push it to the vLLM engine. Returns wall time."""
        if self._sync_dir is None or not hasattr(self.sampler, "sync_weights_from_path"):
            return 0.0
        if (
            self.cfg.vllm_weight_sync_backend == "ipc"
            and hasattr(self.sampler, "sync_weights_from_model")
        ):
            if _is_fsdp_module(self.model):
                raise RuntimeError(
                    "vllm_weight_sync_backend='ipc' is only implemented for "
                    "single-process, unsharded trainers. Use checkpoint sync "
                    "or implement NCCL sync for FSDP."
                )
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
    ) -> Tuple[torch.Tensor, dict]:
        cfg = self.cfg
        assert self.value_model is not None, "value_model must be set for caspo method"
        mb = max(1, int(cfg.micro_batch_size))
        B = prompt_ids.shape[0]

        if self.value_optimizer is None:
            out_chunks: List[torch.Tensor] = []
            for start in range(0, B, mb):
                end = min(start + mb, B)
                with torch.no_grad():
                    out = self.value_model(
                        prompt_ids[start:end], prompt_mask[start:end],
                        response_ids[start:end], response_mask[start:end],
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
                weight = micro_rows / total_rows if total_rows > 0.0 else 0.0
                (v_loss * weight).backward()
            agg["value_loss"] += v_stats["loss"] * weight
            agg["value_acc"] += v_stats["acc_at_last"] * weight
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
        hood) on a flattened ``[B*R, V]`` view in fp32 — equivalent to
        ``log_softmax(.float()).gather(...)`` but ~2× faster and avoids the
        extra ``[B, R, V]`` fp32 activation. ``logits`` slice is fed
        directly; we keep the original dtype on input and only upcast to
        fp32 inside ``cross_entropy``."""
        sliced = logits[:, P - 1 : P - 1 + R, :]
        B, R_, V = sliced.shape
        # cross_entropy returns -logp; negate to get logp.
        neg_logp = F.cross_entropy(
            sliced.reshape(B * R_, V).float(),
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
        mb = max(1, int(cfg.micro_batch_size))
        B = prompt_ids.shape[0]
        out_chunks: List[torch.Tensor] = []
        was_training = self.model.training
        self.model.eval()
        try:
            for start in range(0, B, mb):
                end = min(start + mb, B)
                lp = self._forward_policy_logprobs(
                    prompt_ids[start:end], prompt_mask[start:end],
                    response_ids[start:end], response_mask[start:end],
                )
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
        mb = max(1, int(cfg.micro_batch_size))
        B = prompt_ids.shape[0]
        out_chunks: List[torch.Tensor] = []
        for start in range(0, B, mb):
            end = min(start + mb, B)
            lp = self._forward_ref_logprobs(
                prompt_ids[start:end], prompt_mask[start:end],
                response_ids[start:end], response_mask[start:end],
            )
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

        # 1. Rollout (timed)
        t_rollout_start = time.time()
        rollout: RolloutBatch = self.sampler.sample(examples)
        t_rollout = time.time() - t_rollout_start

        num_prompts = len(rollout.raw_prompts)
        B = num_prompts * G
        assert rollout.response_ids.shape[0] == B

        # 2. Move to device + tile prompts
        prompt_ids = rollout.prompt_ids.to(self.device, non_blocking=True)
        prompt_mask = rollout.prompt_mask.to(self.device, non_blocking=True)
        response_ids = rollout.response_ids.to(self.device, non_blocking=True)
        response_mask = rollout.response_mask.to(self.device, non_blocking=True)
        # When rolling out via vLLM, sampling_logprobs come from a different
        # softmax/attention path than the trainer's. Re-score with the
        # trainer model so PPO's ratio is meaningful (otherwise mean_ratio
        # is biased by ~10×). For HF rollout the two paths are identical and
        # this re-score is a no-op modulo numerics — but cheap, so always do it.
        rewards = rollout.rewards.to(self.device, non_blocking=True).float()
        prompt_index = rollout.prompt_index.to(self.device, non_blocking=True)
        tiled_prompt_ids = prompt_ids[prompt_index]
        tiled_prompt_mask = prompt_mask[prompt_index]
        # Re-score old logprobs with trainer's forward at the pre-update weights.
        old_logprobs_full = self._rescore_old_logprobs(
            tiled_prompt_ids, tiled_prompt_mask, response_ids, response_mask,
        )

        method = cfg.method

        # ---- Method-specific advantage construction ----
        seg = None
        value_stats: dict = {}
        V_step = None
        A_step = None
        token_advantage: torch.Tensor

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
                log_ratio, value_stats = self._value_forward_with_optional_update(
                    tiled_prompt_ids, tiled_prompt_mask, response_ids, response_mask,
                    binary_outcomes, prompt_index=prompt_index,
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
            valid_A = A_step[A_step != 0]
            mean_step_advantage = float(valid_A.abs().mean().item()) if valid_A.numel() else 0.0
            mean_step_count = float(seg.step_count.float().mean().item())
            value_stats["t_value_forward_s"] = t_value

        # Reward stats
        rewards_grouped = rewards.view(num_prompts, G)
        pass_at_g = float((rewards_grouped > 0.5).any(dim=1).float().mean().item())
        mean_reward = float(rewards.mean().item())
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

        total_loss = 0.0
        total_pg = 0.0
        total_kl = 0.0
        total_clip_frac = 0.0
        total_ratio = 0.0
        total_kl_seen = 0   # only counts micros that produced a KL estimate
        n_micro = 0
        token_weight_denom = 0.0
        n_optim_steps = 0

        ref_logprobs_full = self._precompute_ref_logprobs(
            tiled_prompt_ids, tiled_prompt_mask, response_ids, response_mask,
        )

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
        for epoch in range(n_epochs):
            # Reset accumulator counter so the optimizer step boundary
            # ``n_micro_in_epoch % accum == 0`` is computed within the epoch.
            n_micro_in_epoch = 0
            for micro_idx, (start, end) in enumerate(micro_ranges):
                group_idx = micro_idx // accum
                group_tokens = group_token_counts[group_idx] if group_token_counts else 0.0
                micro_tokens = micro_token_counts[micro_idx]
                grad_weight = (
                    micro_tokens / group_tokens
                    if group_tokens > 0.0 else 0.0
                )
                will_step = ((n_micro_in_epoch + 1) % accum == 0) or (end == B)
                with self._maybe_no_sync(self.model, enabled=not will_step):
                    new_logprobs = self._forward_policy_logprobs(
                        tiled_prompt_ids[start:end], tiled_prompt_mask[start:end],
                        response_ids[start:end], response_mask[start:end],
                    )
                    old_lp = old_logprobs_full[start:end]
                    adv = token_advantage[start:end]
                    mask = response_mask[start:end]

                    ref_lp = (
                        ref_logprobs_full[start:end]
                        if ref_logprobs_full is not None else None
                    )

                    loss, stats = ppo_clipped_loss(
                        logprobs=new_logprobs, old_logprobs=old_lp, advantage=adv,
                        response_mask=mask,
                        clip_eps_low=cfg.clip_eps_low, clip_eps_high=cfg.clip_eps_high,
                        ref_logprobs=ref_lp, kl_coef=cfg.kl_coef,
                        kl_estimator=cfg.kl_estimator,
                    )
                    (loss * grad_weight).backward()

                with torch.no_grad():
                    total_loss += float(loss.item()) * micro_tokens
                    total_pg += float(stats["pg_loss"].item()) * micro_tokens
                    total_clip_frac += float(stats["clip_frac"].item()) * micro_tokens
                    total_ratio += float(stats["mean_ratio"].item()) * micro_tokens
                    token_weight_denom += micro_tokens
                    if "mean_kl" in stats:
                        total_kl += float(stats["mean_kl"].item()) * micro_tokens
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
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    n_optim_steps += 1
        t_policy = time.time() - t_policy_start

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
            "mean_ratio": total_ratio / denom,
            "clip_frac": total_clip_frac / denom,
            "mean_reward": mean_reward,
            "pass_at_g": pass_at_g,
            "positive_frac": positive_frac,
            "mean_step_count": mean_step_count,
            "mean_step_advantage": mean_step_advantage,
            "num_prompts": num_prompts,
            "num_responses": B,
            "lr": self.optimizer.param_groups[0]["lr"],
            "method": method,
            "epochs_per_rollout": n_epochs,
            "n_optim_steps": n_optim_steps,
            "t_rollout_s": t_rollout,
            "t_policy_s": t_policy,
            "t_sync_s": t_sync,
            "t_step_s": time.time() - t_step_start,
        }
        if total_kl_seen > 0:
            result["mean_kl"] = total_kl / denom
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
                sum_keys=("num_prompts", "num_responses"),
            )
            result["world_size"] = self.dist.world_size

        return result

    # ------------------------------------------------------------------ loop

    def _next_examples(self, n: int) -> List[dict]:
        return [next(self._train_iter) for _ in range(n)]

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
                    examples = self._next_examples(cfg.prompts_per_step)
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
                examples = self._next_examples(cfg.prompts_per_step)
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

        self.save_checkpoint(final=True)
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
            f"t_value={t_value:.1f}s "
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
            "policy/mean_ratio": stats["mean_ratio"],
            "policy/clip_frac": stats["clip_frac"],
            "policy/lr": stats["lr"],
            "reward/mean": stats["mean_reward"],
            "reward/pass_at_g": stats["pass_at_g"],
            "reward/positive_frac": stats["positive_frac"],
            "seg/mean_step_count": stats["mean_step_count"],
            "adv/abs_mean": stats["mean_step_advantage"],
            "time/rollout_s": stats["t_rollout_s"],
            "time/value_s": float(stats.get("t_value_forward_s", 0.0)),
            "time/policy_s": stats["t_policy_s"],
            "time/sync_s": stats["t_sync_s"],
            "time/step_s": float(stats.get("t_step_s", 0.0)),
            "time/total_elapsed_s": elapsed,
            "step/global": self.global_step,
        }
        if "mean_kl" in stats:
            log_payload["policy/mean_kl"] = stats["mean_kl"]
        for k in ("value_loss", "value_acc", "v_bar_pos", "v_bar_neg",
                  "adb_v_x_mean", "adb_v_x_std", "dlw_w_mean", "dlw_w_std",
                  "value_lr", "vineppo_mc_K", "caspo_advantage_transform_id",
                  "t_value_forward_s",
                  "gpu_mem_alloc_gb", "gpu_mem_peak_gb"):
            if k in stats:
                log_payload[f"value/{k}" if k.startswith(("value", "v_bar", "adb", "dlw"))
                            else f"misc/{k}"] = stats[k]
        self._wandb_log(log_payload, step=self.global_step)

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
