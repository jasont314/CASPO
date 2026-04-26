"""Single source of truth for every configurable knob.

Every module in `caspo/` imports `CASPOConfig` from here. Keeping this file
small and dataclass-only lets agents and scripts share the same contract.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict, fields
from typing import Optional, Literal

import yaml


# Fields that are accepted (for backwards-compatible YAML loading) but not
# currently consumed by any module in the codebase. Loading a YAML that sets
# one of these will emit a warning rather than silently no-op.
#
# Kept as a dict (key -> reason) for the warning text. The loader uses ``in``
# membership which is already O(1) on a dict — converting to frozenset would
# lose the reason mapping. We freeze the message lookup at module load.
_DEPRECATED_OR_UNUSED: dict[str, str] = {
    "compile": "torch.compile is not wired up in CASPOTrainer; ignored at runtime.",
    "eval_every": "in-loop eval is not wired up in CASPOTrainer; run scripts/eval.py separately.",
    "vllm_skip_initial_sync": "vLLM engine always inits from cfg.model_name_or_path; flag not consumed.",
    "value_save_every": "scripts/train_value.py only writes a single 'final' checkpoint; periodic save not consumed.",
}


# Module-level constants used by __post_init__. Defined once at import time so
# every CASPOConfig instantiation reuses the same tuples/dicts instead of
# re-allocating per-call. Validation runs on every YAML load, so this matters.
_FLOAT_FIELDS: tuple[str, ...] = (
    "value_beta", "value_margin", "value_lr", "value_weight_decay",
    "value_val_fraction", "online_value_lr", "adb_dlw_eps",
    "value_grad_clip", "value_data_temperature",
    "rollout_temperature", "rollout_top_p",
    "gamma", "advantage_clip",
    "clip_eps_low", "clip_eps_high", "kl_coef",
    "lr", "weight_decay", "grad_clip",
    "vllm_gpu_memory_utilization",
)

# Curated Literal validation table. With ``from __future__ import annotations``
# active, dataclass field type annotations are strings, so reflective
# get_origin/get_args inspection of ``f.type`` always returns None for
# Literal[...]. We rely on this curated map instead.
_LITERAL_CHECKS: dict[str, tuple[str, ...]] = {
    "torch_dtype": ("bfloat16", "float16", "float32"),
    "rollout_backend": ("hf", "vllm"),
    "segmentation_mode": ("token_delimiter", "latex_aware"),
    "method": ("ppo", "caspo", "grpo", "vineppo"),
    "caspo_advantage_transform": ("value", "prob", "logprob"),
    "standardize_advantage_scope": ("batch", "group", "off"),
    "kl_estimator": ("k1", "k3"),
    "wandb_mode": ("online", "offline", "disabled"),
    "distributed_backend": ("none", "fsdp", "ddp"),
    "fsdp_sharding_strategy": ("full_shard", "shard_grad_op", "no_shard"),
    "fsdp_backward_prefetch": ("backward_pre", "backward_post", "none"),
    "vllm_multi_sample_mode": ("auto", "expanded", "batched"),
    "vllm_weight_sync_backend": ("checkpoint", "ipc"),
}

# Bool fields that may arrive as the strings "true"/"false" from YAML when
# someone writes ``trust_remote_code: "false"`` (quoted). yaml.safe_load
# normally handles unquoted booleans, but quoted strings slip through and
# would fail Literal/range checks downstream with confusing errors.
_BOOL_FIELDS: tuple[str, ...] = (
    "trust_remote_code",
    "update_value_during_policy", "use_adb", "use_dlw",
    "standardize_step_advantage",
    "vllm_enforce_eager", "vllm_skip_initial_sync",
    "vllm_return_logprobs",
    "wandb_enabled",
    "compile", "use_gradient_checkpointing",
    "fsdp_auto_wrap", "fsdp_use_orig_params", "fsdp_cpu_offload",
    "fsdp_forward_prefetch", "fsdp_limit_all_gathers",
)

_BOOL_TRUE = frozenset({"true", "True", "TRUE", "yes", "1"})
_BOOL_FALSE = frozenset({"false", "False", "FALSE", "no", "0"})


@dataclass
class CASPOConfig:
    # ---- model ----
    # Policy + value backbone are usually the same SFT init; the value model
    # is cloned and frozen-ref-pinned at the start of phase 1.
    model_name_or_path: str = "Qwen/Qwen2.5-Math-7B"
    tokenizer_name_or_path: Optional[str] = None  # defaults to model
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    trust_remote_code: bool = True

    # ---- data ----
    dataset_name: str = "agentica-org/DeepScaleR-Preview-Dataset"
    dataset_split: str = "train"
    eval_dataset_name: str = "HuggingFaceH4/MATH-500"
    eval_split: str = "test"
    system_prompt: Optional[str] = None
    # Optional explicit prompt template, used instead of the tokenizer's chat
    # template. Use ``"{query}"`` for the problem text. For paper-faithful
    # reproductions, e.g. VinePPO's MATH format:
    #   "[MATH_TASK] Problem:\n{query}\n\nSolution:"
    prompt_template: Optional[str] = None
    max_prompt_len: int = 1024
    max_response_len: int = 4096

    # ---- rollout ----
    group_size: int = 8
    rollout_temperature: float = 1.0          # used by phase-2 RL rollouts
    rollout_top_p: float = 1.0
    rollout_top_k: int = -1
    rollout_backend: Literal["hf", "vllm"] = "hf"
    # IPVRM phase-1 data collection uses higher temperature for diversity
    # (paper §4.1 / Appendix C: temp 1.0). We keep this separate from
    # rollout_temperature so phase-2 RL stays at its lower-temp setting.
    value_data_temperature: float = 1.0

    # ---- step segmentation ----
    # ``token_delimiter``: split by token-id sequence of ``step_delimiter``
    #     (e.g. tokens of "\n\n" or "\n"). Fast, simple, no LaTeX awareness.
    # ``latex_aware``:     port of VinePPO's ``split_solution_inplace``
    #     (math_extract_steps_inplace.py). Decodes the response, masks
    #     LaTeX/asymptote/tabular environments with placeholders, splits on
    #     sentence-final periods + newlines outside placeholders, then maps
    #     the char-level boundaries back to tokens. Required for faithful
    #     reproduction of VinePPO's MATH numbers.
    segmentation_mode: Literal["token_delimiter", "latex_aware"] = "token_delimiter"
    step_delimiter: str = "\n\n"
    min_step_tokens: int = 4         # merge fragments shorter than this into the previous step
    max_steps_per_response: int = 64 # safety cap; longer rollouts collapse extra steps into the last

    # ---- prefix value model (IPVRM) ----
    # Eq. 5 / Eq. 9 of arXiv:2604.13197.
    # V_phi(s_t) = beta * sum_{i<t} log(pi_phi(y_i|s_i) / pi_ref(y_i|s_i))
    prefix_value_path: Optional[str] = None  # set in phase 2 to a phase-1 ckpt dir
    value_beta: float = 10.0          # β in IPVRM; rescales log-ratios
    value_margin: float = 5.0         # m in IPVRM; BCE dead-zone
    # Phase 1 training:
    value_lr: float = 5e-7
    value_weight_decay: float = 0.0
    value_warmup_steps: int = 0
    value_micro_batch_size: int = 1
    value_grad_accum_steps: int = 4
    # Step budget: ``value_max_epochs`` * train_size / batch overrides
    # ``value_max_steps`` when > 0. Set ``value_max_epochs=0`` to use the
    # raw step count instead.
    value_max_epochs: int = 3
    value_max_steps: int = 2000           # fallback when value_max_epochs == 0
    # NOTE: value_save_every is NOT consumed by scripts/train_value.py —
    # only the final checkpoint is written. Kept for YAML compat.
    value_save_every: int = 500
    value_log_every: int = 1
    value_grad_clip: float = 1.0
    # Per-prompt train/val split (val prompts are entirely held out from
    # training). Used for early-stopping on val loss.
    value_val_fraction: float = 0.10
    value_eval_every: int = 50            # eval val loss every N optimizer steps
    value_early_stop_patience: int = 5    # stop if no val-loss improvement for N evals
    value_early_stop_min_steps: int = 100  # never early-stop before this many steps
    # Where the (prompt, response, outcome) data lives for phase 1.
    value_data_path: Optional[str] = None  # .pt produced by collect_value_data.py
    # Online updating during phase 2 (off by default — keep V_phi frozen).
    update_value_during_policy: bool = False
    # IPVRM reports 1e-4 for LoRA/adapters. This implementation updates the
    # full trainable value phi unless the model is externally PEFT-wrapped, so
    # the safer default is the same order as policy/value full fine-tuning.
    online_value_lr: float = 1e-6
    # Online stabilization (IPVRM §3.3, Eq. 15) — only meaningful when
    # update_value_during_policy=True. ADB shifts the BCE boundary by the
    # per-prompt logit μ(x); DLW weights each example by outcome rarity.
    # Together they fix the label-imbalance pathology of online RM updates.
    use_adb: bool = True
    use_dlw: bool = True
    adb_dlw_eps: float = 0.05      # μ ∈ [eps, 1-eps] before logit (keeps V(x) finite)

    # ---- method dispatch ----
    # Picks the credit-assignment recipe in CASPOTrainer.step():
    # * "ppo"     — sequence-level terminal reward advantage + PPO clip
    # * "caspo"   — step segmentation + V_φ + step TD (this paper's contribution)
    # * "grpo"    — no segmentation, group-relative reward as advantage
    # * "vineppo" — step segmentation + K MC rollouts at each boundary + step TD
    method: Literal["ppo", "caspo", "grpo", "vineppo"] = "caspo"
    vineppo_mc_rollouts: int = 9       # K in VinePPO Eq. 5
    # Optional speed/quality tradeoff for VinePPO MC value estimates. 0 keeps
    # the paper-faithful remaining-response budget; positive values cap each
    # prefix continuation to this many new tokens.
    vineppo_mc_max_tokens: int = 0
    # CASPO ablation knob for the value signal used in step TD advantages.
    # The raw prefix value V is still produced by the IPVRM log-ratio model;
    # this transform is applied to V_step before ``r_t + gamma V_{t+1} - V_t``
    # and before advantage normalization/clipping.
    # * "value":   direct IPVRM value TD difference (current/default CASPO)
    # * "prob":    sigmoid(V) TD difference
    # * "logprob": log sigmoid(V) TD difference
    caspo_advantage_transform: Literal["value", "prob", "logprob"] = "value"

    # ---- step-level TD ----
    gamma: float = 1.0                # discount; VinePPO uses 1.0
    standardize_step_advantage: bool = True  # whiten advantages within batch
    standardize_advantage_scope: Literal["batch", "group", "off"] = "batch"
    # Clip standardized advantages to ±N sigma to prevent rare-event spikes
    # from blowing up the policy gradient (e.g., when V_φ produces a few large
    # values among many zeros, the std normalization can map them to ±10+
    # sigma). 3.0 = mild guardrail; 0 = disable.
    advantage_clip: float = 3.0

    # ---- PPO ----
    clip_eps_low: float = 0.20
    clip_eps_high: float = 0.20       # symmetric by default; can DAPO-style decouple
    kl_coef: float = 0.0              # KL to π_ref as per-token bonus, 0 disables
    kl_estimator: Literal["k1", "k3"] = "k3"  # Schulman k3 unbiased estimator

    # ---- training ----
    output_dir: str = "out/caspo"
    lr: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 0
    grad_clip: float = 1.0
    prompts_per_step: int = 8
    micro_batch_size: int = 1
    grad_accum_steps: int = 1
    # Number of inner passes the trainer makes over each rollout (PPO epochs).
    # The same rollout's π_old and advantages are frozen across epochs; only
    # π_θ updates each epoch, with the clipped surrogate guarding ratio drift.
    epochs_per_rollout: int = 1
    max_steps: int = 1000
    seed: int = 0
    log_every: int = 1
    # ``save_every`` writes periodic policy checkpoints; ``eval_every`` is
    # currently metadata only, with eval invoked via scripts/eval.py.
    save_every: int = 200
    eval_every: int = 200

    # ---- vLLM backend (when rollout_backend = "vllm") ----
    vllm_gpu_memory_utilization: float = 0.85
    vllm_tensor_parallel_size: int = 1
    vllm_enforce_eager: bool = False
    # Optional vLLM scheduler caps. Leave unset to use vLLM defaults; increase
    # for high-throughput rollout jobs after confirming KV cache headroom.
    vllm_max_num_seqs: Optional[int] = None
    vllm_max_num_batched_tokens: Optional[int] = None
    # "expanded": one vLLM request per sample (safe on all observed vLLM V1
    # builds, but Python-heavy for GRPO/VinePPO).
    # "batched": one request with SamplingParams(n=K); fail fast if the vLLM
    # runtime does not return K completions.
    # "auto": try batched once, then fall back to expanded if this runtime has
    # the vLLM V1 n>1 regression.
    vllm_multi_sample_mode: Literal["auto", "expanded", "batched"] = "auto"
    # Trainer recomputes π_old logprobs with the policy model before PPO, so
    # vLLM logprobs are optional diagnostics. Keeping them off trims rollout
    # payload size and decode-side logprob work.
    vllm_return_logprobs: bool = False
    # Client-side cap on concurrently submitted AsyncLLM requests. This does
    # not change vLLM's scheduler caps; it prevents VinePPO MC batches from
    # creating thousands of live Python async generators at once.
    vllm_max_inflight_requests: Optional[int] = 1024
    # Where to dump the policy snapshot for vLLM weight sync each iter.
    # Defaults to {output_dir}/_vllm_sync if None.
    vllm_sync_dir: Optional[str] = None
    # "checkpoint": save_pretrained() to disk, then vLLM reload_weights().
    # "ipc": use vLLM's CUDA-IPC RL weight-transfer API. IPC is single-node and
    # requires trainer and vLLM engine to live on the same physical GPU.
    vllm_weight_sync_backend: Literal["checkpoint", "ipc"] = "checkpoint"
    # If True, skip the initial sync from disk (use SFT init, faster startup
    # for the very first iter).
    # NOTE: not currently consumed — VLLMRolloutEngine always inits from
    # cfg.model_name_or_path at construction. Kept for YAML compat.
    vllm_skip_initial_sync: bool = True

    # ---- wandb ----
    wandb_enabled: bool = True
    wandb_project: str = "caspo"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[str] = None       # comma-separated; parsed in trainer
    wandb_mode: Literal["online", "offline", "disabled"] = "online"

    # ---- runtime ----
    device: str = "cuda"
    # NOTE: torch.compile is not wired into CASPOTrainer; setting this has no
    # effect. Kept for YAML compat.
    compile: bool = False
    use_gradient_checkpointing: bool = False
    # ---- distributed full-model training ----
    # ``fsdp`` is the sharded full-finetune path for memory-bound jobs.
    # ``ddp`` is the replicated full-finetune path for Rho-scale jobs where
    # rank-local vLLM + IPC sync is faster than checkpoint-based sync.
    distributed_backend: Literal["none", "fsdp", "ddp"] = "none"
    dist_backend: str = "nccl"
    dist_timeout_s: int = 1800
    fsdp_sharding_strategy: Literal[
        "full_shard", "shard_grad_op", "no_shard"
    ] = "full_shard"
    fsdp_auto_wrap: bool = True
    fsdp_use_orig_params: bool = True
    fsdp_cpu_offload: bool = False
    fsdp_forward_prefetch: bool = False
    fsdp_limit_all_gathers: bool = True
    fsdp_backward_prefetch: Literal[
        "backward_pre", "backward_post", "none"
    ] = "backward_pre"
    # ---- profiling ----
    # Opt-in torch.profiler trace dump. When > 0, the trainer wraps its main
    # loop with torch.profiler.profile(...) using schedule(warmup=2,
    # active=profile_steps, repeat=1) and emits TensorBoard traces to
    # {output_dir}/profile/. Default 0 = profiler off (zero overhead, just an
    # if-check before the loop).
    profile_steps: int = 0

    def __post_init__(self) -> None:
        # ---- str → bool coercion ----
        # YAML quirk: ``trust_remote_code: "false"`` (quoted) becomes the str
        # ``"false"`` which is truthy. Coerce common spellings before any
        # downstream check. Tolerate already-bool values.
        for name in _BOOL_FIELDS:
            val = getattr(self, name)
            if isinstance(val, str):
                if val in _BOOL_TRUE:
                    setattr(self, name, True)
                elif val in _BOOL_FALSE:
                    setattr(self, name, False)
                else:
                    raise ValueError(
                        f"{name}={val!r} must be a bool; got string not in "
                        f"true/false set"
                    )

        # ---- Type coercion: int → float for float-annotated fields ----
        # YAML parses ``lr: 1`` as int, but downstream code (AdamW, math
        # ops, isinstance checks) may break or silently produce surprising
        # behavior. Coerce known float fields here so YAML callers can
        # write either ``1.0`` or ``1`` interchangeably.
        for name in _FLOAT_FIELDS:
            val = getattr(self, name)
            if type(val) is bool:
                # bool is a subclass of int but coercing it to float would
                # silently turn ``True`` into 1.0; refuse instead. Use
                # ``type(val) is bool`` (not isinstance) for a single-op
                # check that beats isinstance() in CPython micro-bench.
                raise ValueError(
                    f"{name}={val!r} must be a float, got bool"
                )
            if type(val) is int:
                setattr(self, name, float(val))

        # ---- Literal enforcement ----
        # Validate every Literal field via the curated table. Faster than
        # the previous fields()+get_origin() loop, which always missed under
        # PEP 563 string annotations and was effectively dead code.
        for name, allowed in _LITERAL_CHECKS.items():
            val = getattr(self, name)
            if val not in allowed:
                raise ValueError(
                    f"{name}={val!r} not in allowed values {allowed}"
                )

        # ---- Range / sanity checks for fields with silent failure modes ----
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if self.group_size < 2 and self.method == "grpo":
            raise ValueError(
                "GRPO requires group_size >= 2 for group-relative advantage"
            )
        if self.max_prompt_len <= 0:
            raise ValueError(f"max_prompt_len must be > 0, got {self.max_prompt_len}")
        if self.max_response_len <= 0:
            raise ValueError(f"max_response_len must be > 0, got {self.max_response_len}")
        if self.min_step_tokens < 1:
            raise ValueError(f"min_step_tokens must be >= 1, got {self.min_step_tokens}")
        if self.max_steps_per_response < 1:
            raise ValueError(
                f"max_steps_per_response must be >= 1, got {self.max_steps_per_response}"
            )
        if not (0.0 < self.value_val_fraction < 1.0):
            raise ValueError(
                f"value_val_fraction must be in (0, 1), got {self.value_val_fraction}"
            )
        # adb_dlw_eps shifts μ into [eps, 1-eps] before logit; eps>=0.5 collapses
        # the interval and produces inf/NaN. Only relevant if ADB or DLW is on.
        if (self.use_adb or self.use_dlw):
            if not (0.0 < self.adb_dlw_eps < 0.5):
                raise ValueError(
                    f"adb_dlw_eps must be in (0, 0.5), got {self.adb_dlw_eps}"
                )
        if self.gamma <= 0.0 or self.gamma > 1.0:
            raise ValueError(
                f"gamma must be in (0, 1], got {self.gamma}"
            )
        if self.clip_eps_low < 0 or self.clip_eps_high < 0:
            raise ValueError(
                "PPO clip eps must be non-negative; "
                f"got low={self.clip_eps_low}, high={self.clip_eps_high}"
            )
        if self.kl_coef < 0:
            raise ValueError(f"kl_coef must be >= 0, got {self.kl_coef}")
        if self.advantage_clip < 0:
            raise ValueError(
                f"advantage_clip must be >= 0 (0 disables), got {self.advantage_clip}"
            )
        if self.vineppo_mc_rollouts < 1 and self.method == "vineppo":
            raise ValueError(
                f"vineppo_mc_rollouts must be >= 1 for method='vineppo', "
                f"got {self.vineppo_mc_rollouts}"
            )
        if self.method == "vineppo" and self.rollout_backend != "vllm":
            raise ValueError(
                "method='vineppo' requires rollout_backend='vllm' because "
                "MC prefix value estimation needs sample_with_prefix()."
            )
        if self.vineppo_mc_max_tokens < 0:
            raise ValueError(
                f"vineppo_mc_max_tokens must be >= 0 (0 disables), "
                f"got {self.vineppo_mc_max_tokens}"
            )
        if self.epochs_per_rollout < 1:
            raise ValueError(
                f"epochs_per_rollout must be >= 1, got {self.epochs_per_rollout}"
            )
        if self.save_every < 0:
            raise ValueError(
                f"save_every must be >= 0 (0 disables periodic save), got {self.save_every}"
            )
        if self.eval_every < 0:
            raise ValueError(
                f"eval_every must be >= 0 (0 disables), got {self.eval_every}"
            )
        if self.grad_accum_steps < 1:
            raise ValueError(
                f"grad_accum_steps must be >= 1, got {self.grad_accum_steps}"
            )
        if self.micro_batch_size < 1:
            raise ValueError(
                f"micro_batch_size must be >= 1, got {self.micro_batch_size}"
            )
        if self.dist_timeout_s < 1:
            raise ValueError(
                f"dist_timeout_s must be >= 1, got {self.dist_timeout_s}"
            )
        if self.distributed_backend == "fsdp" and self.rollout_backend == "vllm":
            if self.vllm_tensor_parallel_size != 1:
                raise ValueError(
                    "distributed_backend='fsdp' with rollout_backend='vllm' "
                    "currently expects one rank-local vLLM engine per process "
                    "(vllm_tensor_parallel_size=1). Use a separate rollout "
                    "topology for tensor-parallel vLLM."
                )
        if self.distributed_backend == "ddp" and self.rollout_backend != "vllm":
            raise ValueError(
                "distributed_backend='ddp' currently requires "
                "rollout_backend='vllm' so each rank can use a rank-local "
                "rollout engine. Use distributed_backend='none' for HF rollout."
            )
        if self.distributed_backend == "ddp" and self.vllm_tensor_parallel_size != 1:
            raise ValueError(
                "distributed_backend='ddp' expects one rank-local vLLM engine "
                "per process (vllm_tensor_parallel_size=1)."
            )
        if self.vllm_weight_sync_backend == "ipc":
            if self.distributed_backend == "fsdp":
                raise ValueError(
                    "vllm_weight_sync_backend='ipc' is only supported for the "
                    "single-process or DDP replicated trainer. Use checkpoint "
                    "sync for FSDP until NCCL weight sync is implemented."
                )
            if self.vllm_tensor_parallel_size != 1:
                raise ValueError(
                    "vllm_weight_sync_backend='ipc' requires "
                    "vllm_tensor_parallel_size=1 so trainer and vLLM share one "
                    "physical GPU."
                )
        if self.profile_steps < 0:
            raise ValueError(
                f"profile_steps must be >= 0 (0 disables profiling), got {self.profile_steps}"
            )
        if self.vllm_max_num_seqs is not None and self.vllm_max_num_seqs < 1:
            raise ValueError(
                f"vllm_max_num_seqs must be >= 1 when set, got {self.vllm_max_num_seqs}"
            )
        if self.vllm_tensor_parallel_size < 1:
            raise ValueError(
                "vllm_tensor_parallel_size must be >= 1, "
                f"got {self.vllm_tensor_parallel_size}"
            )
        if (
            self.vllm_max_num_batched_tokens is not None
            and self.vllm_max_num_batched_tokens < 1
        ):
            raise ValueError(
                "vllm_max_num_batched_tokens must be >= 1 when set, "
                f"got {self.vllm_max_num_batched_tokens}"
            )
        if (
            self.vllm_max_inflight_requests is not None
            and self.vllm_max_inflight_requests < 1
        ):
            raise ValueError(
                "vllm_max_inflight_requests must be >= 1 when set, "
                f"got {self.vllm_max_inflight_requests}"
            )
        if self.prompt_template is not None and "{query}" not in self.prompt_template:
            # Silent fallback to chat template would otherwise hide the typo.
            warnings.warn(
                f"prompt_template={self.prompt_template!r} has no '{{query}}' "
                f"placeholder; it will be ignored by data loaders.",
                stacklevel=2,
            )

    @classmethod
    def from_yaml(cls, path: str) -> "CASPOConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(
                f"YAML at {path} must be a mapping at the top level, "
                f"got {type(data).__name__}"
            )
        # Cache the field-name set on the class for repeated YAML loads
        # (e.g. when scripts iterate over many configs at once).
        known = getattr(cls, "_field_names", None)
        if known is None:
            known = frozenset(f.name for f in fields(cls))
            cls._field_names = known  # type: ignore[attr-defined]
        unknown = [k for k in data if k not in known]
        if unknown:
            raise ValueError(
                f"Unknown config keys in {path}: {sorted(unknown)}. "
                f"Did you mean a renamed field? Known fields: {sorted(known)}"
            )
        # Warn on accepted-but-unused keys so users don't think these knobs do
        # anything. We do NOT drop them — re-saving via to_yaml will round-trip.
        for k in data:
            if k in _DEPRECATED_OR_UNUSED:
                warnings.warn(
                    f"config key {k!r} is set in {path} but is currently "
                    f"unused: {_DEPRECATED_OR_UNUSED[k]}",
                    stacklevel=2,
                )
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, sort_keys=False)
