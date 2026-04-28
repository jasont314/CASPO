"""Single source of truth for every configurable knob.

Every module in `caspo/` imports `CASPOConfig` from here. Keeping this file
small and dataclass-only lets agents and scripts share the same contract.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, asdict, field, fields
from typing import List, Optional, Literal

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
    "vllm_skip_initial_sync": "vLLM engine always inits from cfg.model_name_or_path; flag not consumed.",
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
    "critic_lr", "critic_weight_decay", "critic_grad_clip",
    "value_loss_coef", "cliprange_value", "ppo_gae_lambda",
    "lr", "weight_decay", "grad_clip",
    "eval_during_training_temperature", "eval_during_training_vllm_util",
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
    "method": ("ppo", "caspo", "grpo", "vineppo", "ppo_critic"),
    "caspo_advantage_transform": ("value", "prob", "logprob"),
    "standardize_advantage_scope": ("batch", "group", "off"),
    "kl_estimator": ("k1", "k3"),
    "wandb_mode": ("online", "offline", "disabled"),
    "distributed_backend": ("none", "fsdp", "ddp"),
    "fsdp_sharding_strategy": (
        "full_shard", "shard_grad_op", "no_shard", "hybrid_shard",
    ),
    "fsdp_backward_prefetch": ("backward_pre", "backward_post", "none"),
    "activation_checkpointing_mode": ("off", "full", "selective"),
    "vllm_multi_sample_mode": ("auto", "expanded", "batched"),
    "vllm_weight_sync_backend": ("checkpoint", "ipc", "nccl"),
    # vllm_kv_cache_dtype is Optional — None means "auto" (vLLM default).
    # Validated below in __post_init__ rather than via the curated table so
    # we keep None passthrough.
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
    "vllm_return_logprobs", "vllm_enable_chunked_prefill",
    "wandb_enabled",
    "compile", "use_gradient_checkpointing",
    "critic_share_fsdp_policy",
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
    # When True, the training-data loader drops any row whose normalized
    # problem text matches a problem in our eval suite (MATH-500, GSM8K,
    # AIME-2025, OlympiadBench). Verified necessary for DeepScaleR
    # (3/500 MATH-500 overlap) and Big-Math (2/674 OlympiadBench
    # incidental). See ``caspo/data/eval_leak.py``.
    filter_eval_leakage: bool = True
    # Optional HF dataset config name. Required by multi-config datasets
    # like ``open-r1/Big-Math-RL-Verified-Processed`` which expose
    # `level_1`, `level_2`, ..., `quintile_5`, `all`. Forwarded as the
    # ``name`` arg to ``datasets.load_dataset(repo, name, split=...)``.
    # None => single-config dataset (e.g. MATH-lighteval, DeepScaleR).
    dataset_config: Optional[str] = None
    # VinePPO ``max_sequence_length=2048`` unfinished-response penalty.
    # When prompt+response token total exceeds this, the rollout is treated
    # as unfinished and reward is zeroed (matches MathEpisodeGenerator's
    # post-hoc seq_len penalty in treetune episode_generator). 0 disables.
    max_sequence_len: int = 2048

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
    # Periodic save in scripts/train_value.py: every value_save_every
    # optimizer steps, dump the current V_φ to <output_dir>/step_<N>/.
    # Independent of value_eval_every / best/. Set to 0 to disable.
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
    # * "ppo"        — sequence-level terminal reward advantage + PPO clip
    #                  (critic-free; "standard practice for verifier-RL")
    # * "caspo"      — step segmentation + V_φ + step TD (this paper's contribution)
    # * "grpo"       — no segmentation, group-relative reward as advantage
    # * "vineppo"    — step segmentation + K MC rollouts at each boundary + step TD
    # * "ppo_critic" — Schulman 2017 PPO with a learned value network (separate
    #                  PreTrainedModel, ~14 GB at 7B) + GAE + clipped-value MSE.
    #                  Provided as a fair head-to-head baseline against VinePPO
    #                  per the upstream paper's framing (PPO+critic is the
    #                  reference VinePPO compares against).
    method: Literal["ppo", "caspo", "grpo", "vineppo", "ppo_critic"] = "caspo"
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
    # Clip standardized advantages to ±N sigma. VinePPO upstream and
    # standard PPO impls do NOT clip; truncating ±3 silently nukes terminal-
    # success advantages (the strongest learning signal) after whitening.
    # 0 = disable (paper-faithful default). Kept as a knob for ablation.
    advantage_clip: float = 0.0

    # ---- PPO ----
    clip_eps_low: float = 0.20
    clip_eps_high: float = 0.20       # symmetric by default; can DAPO-style decouple
    kl_coef: float = 0.0              # KL to π_ref as per-token bonus, 0 disables
    kl_estimator: Literal["k1", "k3"] = "k3"  # Schulman k3 unbiased estimator

    # ---- PPO+critic (method="ppo_critic") ----
    # Schulman 2017's classic PPO with a learned value network. The critic is
    # a separate ``PreTrainedModel`` (~14 GB at 7B bf16) initialized from
    # ``critic_model_name_or_path`` (defaults to the policy SFT path so the
    # value head sees a pretrained backbone). It is trained jointly with the
    # policy via clipped-value MSE loss against GAE returns; both models live
    # on the trainer GPUs and share the FSDP wrap pipeline. Provided as a
    # paper-faithful baseline for VinePPO comparison (the upstream paper's
    # "PPO" reference is critic-based).
    critic_model_name_or_path: Optional[str] = None  # None = use cfg.model_name_or_path
    # Critic optimizer hyperparameters. Aligned with VinePPO upstream's
    # ``configs/trainers/ppo_MATH.jsonnet`` which uses ``learning_rate=1e-6``
    # for both policy and critic (single ``learning_rate`` field shared by
    # actor + critic in their DeepSpeed setup; verified 2026-04-27).
    critic_lr: float = 1e-6
    critic_weight_decay: float = 0.0
    # VinePPO upstream warms actor and critic with the same ratio
    # (warmup_ratio=0.03 of total optimizer steps). Default 0 here was
    # a cold start; set to 480 to match the actor's 1000×2×8=16000-step
    # × 3% schedule. PPO+critic launchers can override per-config.
    critic_warmup_steps: int = 480
    critic_grad_clip: float = 1.0
    # Value-loss coefficient in the joint objective. The full PPO loss is
    # ``policy_pg_loss + value_loss_coef * value_loss``. VinePPO upstream's
    # ppo_trainer scales by 0.5 inside the loss function and applies a
    # separate vf_coef multiplier; we fold the 0.5 into ``clipped_value_loss``
    # already and default ``value_loss_coef=1.0`` to match the effective
    # policy:value ratio (1:1).
    value_loss_coef: float = 1.0
    # Clipped value loss range (Schulman 2017 §6.1). Each value prediction
    # is clipped to [old_v - cliprange, old_v + cliprange] before the MSE.
    # Stabilizes value training when V is far from the target return.
    cliprange_value: float = 0.2
    # GAE (Schulman 2016) discount and λ. VinePPO upstream's effective
    # config for DeepSeekMath SFT2 (``configs/trainers/lam1.jsonnet``
    # overrides the base ``ppo_MATH.jsonnet`` lam=0.96 to lam=1.0)
    # uses gamma=1.0, lambda=1.0 (Monte-Carlo-like advantage with no
    # bootstrap-decay).
    ppo_gae_lambda: float = 1.0
    # Whether the critic shares its FSDP wrap policy with the main policy.
    # True (default) keeps the critic under the same fsdp_wrap_block_group_size
    # / fsdp_sharding_strategy as the policy. False is a future hook for
    # asymmetric sharding (e.g., critic CPU-offloaded).
    critic_share_fsdp_policy: bool = True

    # ---- training ----
    output_dir: str = "out/caspo"
    lr: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 0
    grad_clip: float = 1.0
    prompts_per_step: int = 8
    micro_batch_size: int = 1
    # No-grad old-policy/ref logprob passes can usually use a larger batch than
    # PPO backward. None means reuse micro_batch_size.
    logprob_micro_batch_size: Optional[int] = None
    grad_accum_steps: int = 1
    # Number of inner passes the trainer makes over each rollout (PPO epochs).
    # The same rollout's π_old and advantages are frozen across epochs; only
    # π_θ updates each epoch, with the clipped surrogate guarding ratio drift.
    epochs_per_rollout: int = 1
    max_steps: int = 1000
    seed: int = 0
    log_every: int = 1
    # ``save_every`` writes periodic policy checkpoints AND drives
    # periodic in-loop eval cadence. Eval fires at every save when
    # ``eval_during_training_gpu >= 0`` (or ``CASPO_EVAL_GPU`` env is
    # set). ``eval_every`` is metadata only — setting it != save_every
    # was a footgun (would mis-align and only fire at LCM cadence).
    save_every: int = 200
    eval_every: int = 200
    # When True, also save AdamW state (m/v/step) + LR-scheduler state +
    # critic/value-model optimizer state (when present) at every save_every
    # boundary. Required to resume mid-training without losing optimizer
    # momentum after a crash. fp32 master adds ~3× model-bytes per save
    # (params + m + v); disable on tight-disk runs and pay the resume cost.
    save_optimizer_state: bool = True
    # When True, temporarily move policy/critic/value AdamW state (m, v) to
    # pinned host memory before each ``_sync_vllm_weights`` call and restore
    # afterward. Required for 7B fp32-master + 4-rank FSDP + colocated vLLM:
    # without offload, sync's ``summon_full_params`` materializes ~28 GiB
    # unsharded fp32 on top of ~14 GiB AdamW state and OOMs at step 1→2
    # transition. With offload, the AdamW slot is freed for the duration of
    # the summon, the IPC push runs, then state is moved back. Cost:
    # ~14 GiB × 2 (down + up) over PCIe per sync ≈ 400-500 ms/sync at PCIe
    # Gen5 (~1% step-time slowdown at 7B). Default False — only flip on
    # for memory-tight configs (7B fp32-master + 4-GPU).
    offload_optim_during_sync: bool = False
    # When True, run a one-time dummy ``optimizer.step()`` with zero grads
    # right after each optimizer is constructed at trainer init, to force
    # AdamW state (m, v) allocation BEFORE vLLM grabs its KV cache. With
    # lazy init, the first real step has to find a contiguous ~param_bytes
    # × 2 block on a GPU already 30%+ filled with vLLM — at fp32 master +
    # tight configs (1B single-GPU) this races allocator fragmentation and
    # OOMs. Eager allocation gets a clean address space. One-time cost
    # ~50-200 ms (kernel launches + zero-fill); zero per-step overhead
    # since state is already populated by step 1. Default True (small
    # cost, real benefit on tight configs).
    preallocate_optim_state: bool = True
    # Set to a free GPU index (e.g. CASPO_EVAL_GPU=2 in the launcher
    # env) to dispatch ``scripts/eval.py`` as a fire-and-forget
    # subprocess after each ``eval_every`` checkpoint. -1 disables.
    # Subprocess uses CUDA_VISIBLE_DEVICES=<this gpu>; result JSON
    # lands at ``<ckpt>/eval.json``. Rank-0 only.
    eval_during_training_gpu: int = -1
    # Periodic-eval sample params (cheap MATH-500 sample by default;
    # override via launcher env if you want different settings).
    eval_during_training_benchmarks: str = "math500"
    eval_during_training_k: int = 8
    eval_during_training_limit: int = 100
    eval_during_training_temperature: float = 0.35
    eval_during_training_vllm_util: float = 0.30

    # ---- vLLM backend (when rollout_backend = "vllm") ----
    vllm_gpu_memory_utilization: float = 0.85
    vllm_tensor_parallel_size: int = 1
    vllm_enforce_eager: bool = False
    # Optional vLLM scheduler caps. Leave unset to use vLLM defaults; increase
    # for high-throughput rollout jobs after confirming KV cache headroom.
    vllm_max_num_seqs: Optional[int] = None
    vllm_max_num_batched_tokens: Optional[int] = None
    # Chunked prefill: interleaves prefill chunks with decode steps so a
    # newly-arriving long prompt does not stall in-flight decode. Default
    # OFF for the rollout engine because we verified empirically (Apr 2026)
    # that VinePPO K=9 MC rollouts regress ~70% (191s -> 321s/step) when
    # this is forced on — the mixed prefill-decode CUDA graphs penalize the
    # MC pattern of many short prefixes followed by decodes. For
    # prefill-heavy workloads (G=8 same-prompt fan-out, eval-time
    # generation) flip to True via override.
    vllm_enable_chunked_prefill: bool = False
    # KV-cache dtype. ``None`` (auto) picks vLLM's default (fp16/bf16 to match
    # the model dtype); ``"fp8"`` halves KV memory and is used by the eval
    # pipeline (inference-only, accuracy hit is below seed noise on avg@k).
    # Rollout side defaults to None so RL training keeps full-precision KV.
    vllm_kv_cache_dtype: Optional[Literal["auto", "fp8"]] = None
    # "expanded": one vLLM request per sample (safe on all observed vLLM V1
    # builds, but Python-heavy for GRPO/VinePPO).
    # "batched": one request with SamplingParams(n=K); fail fast if the vLLM
    # runtime does not return K completions.
    # "auto": try batched once, then fall back to expanded if this runtime has
    # the vLLM V1 n>1 regression.
    vllm_multi_sample_mode: Literal["auto", "expanded", "batched"] = "auto"
    # Textual stop strings passed to vLLM SamplingParams. VinePPO uses
    # "\n\n\nProblem:" so the model can't auto-regressively start a NEW
    # MATH problem after finishing the current one (which the
    # `[MATH_TASK] Problem: ...` template invites). Without this, otherwise-
    # correct rollouts run to length cap and get reward 0 under the
    # finish_reason=='length' rule. Empty list disables.
    vllm_extra_stop_strings: List[str] = field(
        default_factory=lambda: ["\n\n\nProblem:"]
    )
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
    # "nccl": vLLM's NCCL weight-transfer backend. Required for the
    # disaggregated topology where trainer and vLLM live on disjoint GPU
    # sets — IPC handles can't span physical GPUs. The trainer side opens a
    # side PyNcclCommunicator group with the vLLM workers and broadcasts
    # weights through it once per outer step.
    vllm_weight_sync_backend: Literal["checkpoint", "ipc", "nccl"] = "checkpoint"
    # Disaggregated rollout: trainer ranks live on one GPU set, vLLM
    # AsyncLLM(tensor_parallel_size=N) lives on a disjoint set. Only rank
    # 0 of the trainer instantiates the engine; other ranks gather their
    # examples to rank 0 before sample(), and rank 0 scatters back.
    # Memory layout: trainer GPUs hold FSDP-sharded params + activations
    # only (no colocated vLLM); rollout GPUs hold TP-sharded vLLM only
    # (so vllm_gpu_memory_utilization can rise to ~0.85 vs 0.30 for
    # colocated). See docs/disaggregated_topology_plan.md.
    vllm_disaggregated: bool = False
    # vLLM tensor-parallel size when disaggregated. Validated against
    # rollout-GPU count by the launcher (each TP rank pins one GPU).
    # Has no effect when vllm_disaggregated=False.
    vllm_disaggregated_tp: int = 1
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
    # Activation-checkpoint policy refinement. ``"off"`` matches
    # ``use_gradient_checkpointing=False`` (no AC); ``"full"`` matches
    # ``use_gradient_checkpointing=True`` (HF .gradient_checkpointing_enable
    # path: every transformer block recomputes its forward on backward);
    # ``"selective"`` wraps only attention submodules with PyTorch's
    # apply_activation_checkpointing — recomputes attention (the FLOPS-cheap
    # but memory-heavy op) and keeps MLP activations live, matching the
    # Megatron/recompute_granularity="selective" pattern. When this field is
    # set to anything other than ``"off"``, it OVERRIDES
    # ``use_gradient_checkpointing``. Kept default ``"off"`` so existing
    # configs keep their current behaviour driven by
    # ``use_gradient_checkpointing``.
    activation_checkpointing_mode: Literal["off", "full", "selective"] = "off"
    # ---- distributed full-model training ----
    # ``fsdp`` is the sharded full-finetune path for memory-bound jobs.
    # ``ddp`` is the replicated full-finetune path for Rho-scale jobs where
    # rank-local vLLM + IPC sync is faster than checkpoint-based sync.
    distributed_backend: Literal["none", "fsdp", "ddp"] = "none"
    dist_backend: str = "nccl"
    dist_timeout_s: int = 1800
    fsdp_sharding_strategy: Literal[
        "full_shard", "shard_grad_op", "no_shard", "hybrid_shard"
    ] = "full_shard"
    fsdp_auto_wrap: bool = True
    fsdp_use_orig_params: bool = True
    fsdp_cpu_offload: bool = False
    fsdp_forward_prefetch: bool = False
    fsdp_limit_all_gathers: bool = True
    fsdp_backward_prefetch: Literal[
        "backward_pre", "backward_post", "none"
    ] = "backward_pre"
    # When >1, group every Nth transformer block into a single FSDP unit
    # instead of wrapping each block individually. At 7B (32 blocks) the
    # default per-block wrap issues 32 reduce-scatter calls per backward at
    # ~440 MB each; group_size=4 cuts that to 8 calls at ~1.7 GB, lifting
    # NVLink BW utilization (target step-time win 10-18%). Default 1 keeps
    # the existing per-block wrap behaviour byte-for-byte.
    fsdp_wrap_block_group_size: int = 1
    # FSDP MixedPrecision reduce_dtype. fp32 reduce matches DeepSpeed
    # BF16_Optimizer's fp32 grad accumulator behavior used by VinePPO
    # upstream — at PPO's lr=1e-6 with grad_accum_steps=8, accumulating
    # 8× bf16 reductions into a bf16 .grad buffer puts the effective
    # update inside the bf16 noise floor and the run drifts back toward
    # init. fp32 reduce + FSDP no_sync() during inner micros (so only
    # the LAST mb's reduction is paid in fp32) restores update fidelity
    # at minor wire-byte cost (2× bytes on accum-boundary mb only).
    # Set to "bfloat16" or "float16" to revert to the older lower-precision
    # path if memory or wire-bytes are tight.
    fsdp_reduce_dtype: Optional[Literal["bfloat16", "float16", "float32"]] = "float32"
    # When True, load policy weights in fp32 and rely on FSDP MixedPrecision
    # ``param_dtype=bf16`` to cast at compute time. This preserves an fp32
    # master copy on the shard so AdamW's ``exp_avg``/``exp_avg_sq`` and the
    # ``param.add_(lr * m_hat / (sqrt(v_hat)+eps))`` update happen in fp32.
    # Without this, the optimizer state and update both live in bf16 (7-bit
    # mantissa) and at lr=1e-6 the per-step delta rounds to zero — the same
    # symptom DeepSpeed's BF16_Optimizer was built to fix. Cost: ~2× param
    # memory per shard, ~2× optimizer-state memory. Disable for tight
    # 7B/4-GPU configs and pay the regression risk.
    fp32_master_weights: bool = True
    # ---- profiling ----
    # Opt-in torch.profiler trace dump. When > 0, the trainer wraps its main
    # loop with torch.profiler.profile(...) using schedule(warmup=2,
    # active=profile_steps, repeat=1) and emits TensorBoard traces to
    # {output_dir}/profile/. Default 0 = profiler off (zero overhead, just an
    # if-check before the loop).
    profile_steps: int = 0

    # ---- reward grading ----
    # Number of worker processes for parallel SymPy/math_verify grading. The
    # SymPy fallback uses SIGALRM for adversarial-expression timeouts, which
    # only works in a process (not a thread). A persistent ProcessPoolExecutor
    # is created lazily on the first grading call and reused across PPO outer
    # steps. ``1`` keeps the legacy serial path (deterministic, no fork cost).
    reward_workers: int = 4
    # Max size of the per-MathRewardFn ground-truth normalization cache. The
    # same prompts cycle every ~7.5K examples, so persisting normalized GT
    # across PPO outer steps avoids redoing the LaTeX peel + replace-chain.
    # When the cache exceeds this many entries, the oldest 1024 are evicted.
    gt_cache_max_size: int = 8192

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
        if (
            self.logprob_micro_batch_size is not None
            and self.logprob_micro_batch_size < 1
        ):
            raise ValueError(
                "logprob_micro_batch_size must be >= 1 when set, "
                f"got {self.logprob_micro_batch_size}"
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
            if self.vllm_tensor_parallel_size != 1:
                raise ValueError(
                    "vllm_weight_sync_backend='ipc' requires "
                    "vllm_tensor_parallel_size=1 so trainer and vLLM share one "
                    "physical GPU."
                )
            # Note: ``vllm_disaggregated=True`` + ``vllm_weight_sync_backend=ipc``
            # is the colocated-TP path (Phase F). That mode requires every
            # trainer rank N to share GPU N with vLLM Worker_TP_N (so each
            # rank's local IPC handle is openable by its same-GPU worker).
            # We can't fully verify the GPU-sharing invariant from the cfg
            # alone (it's a launcher topology fact), so the trainer asserts
            # at runtime that LOCAL_RANK / world_size matches
            # vllm_disaggregated_tp before building the multirank-IPC proxy.
        # Disaggregated topology validations
        if self.vllm_disaggregated:
            if self.vllm_disaggregated_tp < 1:
                raise ValueError(
                    f"vllm_disaggregated_tp must be >= 1, "
                    f"got {self.vllm_disaggregated_tp}"
                )
            if self.rollout_backend != "vllm":
                raise ValueError(
                    "vllm_disaggregated=True requires rollout_backend='vllm'"
                )
            if self.distributed_backend != "fsdp":
                raise ValueError(
                    "vllm_disaggregated=True is only validated against "
                    f"distributed_backend='fsdp', got "
                    f"{self.distributed_backend!r}"
                )
            if self.vllm_weight_sync_backend not in ("nccl", "checkpoint", "ipc"):
                raise ValueError(
                    f"vllm_disaggregated=True requires "
                    f"vllm_weight_sync_backend in {{'nccl','checkpoint','ipc'}}, "
                    f"got {self.vllm_weight_sync_backend!r}."
                )
            # ``ipc`` under disaggregated is the Phase F colocated-TP path:
            # every trainer rank shares its physical GPU with the matching
            # vLLM TP-worker, so per-rank IPC handles can be opened by the
            # same-GPU worker. The trainer's
            # ``DisaggregatedSamplerProxy._sync_weights_multirank_ipc`` does
            # the cross-rank gather + per-param multi-UUID merge before
            # submitting one ``AsyncLLM.update_weights``. The launcher must
            # set CUDA_VISIBLE_DEVICES so trainer rank N pins to the same
            # physical GPU as vLLM Worker_TP_N.
            # vllm_tensor_parallel_size on the trainer-side cfg is ignored
            # under disaggregation — TP belongs to the rollout engine and is
            # carried by vllm_disaggregated_tp instead. Reject the redundant
            # combo to keep launcher logic unambiguous.
            if self.vllm_tensor_parallel_size != 1:
                raise ValueError(
                    "Under vllm_disaggregated=True the trainer-side cfg "
                    "must keep vllm_tensor_parallel_size=1; the rollout "
                    "engine's TP size is set by vllm_disaggregated_tp "
                    f"(here: {self.vllm_disaggregated_tp})."
                )
        elif self.vllm_disaggregated_tp != 1:
            raise ValueError(
                "vllm_disaggregated_tp != 1 only makes sense with "
                "vllm_disaggregated=True; set vllm_disaggregated=True or "
                "leave vllm_disaggregated_tp=1 (default)."
            )
        # Optional Literal["bfloat16", "float16", "float32"] — None passthrough
        # means "match torch_dtype". Anything else is rejected.
        if self.fsdp_reduce_dtype is not None and self.fsdp_reduce_dtype not in (
            "bfloat16", "float16", "float32",
        ):
            raise ValueError(
                "fsdp_reduce_dtype must be one of None, 'bfloat16', 'float16', "
                f"'float32'; got {self.fsdp_reduce_dtype!r}"
            )
        if self.profile_steps < 0:
            raise ValueError(
                f"profile_steps must be >= 0 (0 disables profiling), got {self.profile_steps}"
            )
        if self.fsdp_wrap_block_group_size < 1:
            raise ValueError(
                "fsdp_wrap_block_group_size must be >= 1 (1 = per-block wrap), "
                f"got {self.fsdp_wrap_block_group_size}"
            )
        if self.reward_workers < 1:
            raise ValueError(
                f"reward_workers must be >= 1 (1 disables pool), got {self.reward_workers}"
            )
        if self.gt_cache_max_size < 1:
            raise ValueError(
                f"gt_cache_max_size must be >= 1, got {self.gt_cache_max_size}"
            )
        # ---- additional sanity checks for fields with silent failure modes ----
        if self.prompts_per_step < 1:
            raise ValueError(
                f"prompts_per_step must be >= 1, got {self.prompts_per_step}"
            )
        if self.max_steps < 1:
            raise ValueError(
                f"max_steps must be >= 1, got {self.max_steps}"
            )
        if self.value_micro_batch_size < 1:
            raise ValueError(
                f"value_micro_batch_size must be >= 1, "
                f"got {self.value_micro_batch_size}"
            )
        if self.value_grad_accum_steps < 1:
            raise ValueError(
                f"value_grad_accum_steps must be >= 1, "
                f"got {self.value_grad_accum_steps}"
            )
        if self.value_warmup_steps < 0:
            raise ValueError(
                f"value_warmup_steps must be >= 0, got {self.value_warmup_steps}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be >= 0, got {self.warmup_steps}"
            )
        # vLLM GPU memory utilization is a fraction in (0, 1); 0 deadlocks the
        # KV allocator and >1 segfaults the engine. vLLM itself rejects these
        # but the error there is opaque.
        if not (0.0 < self.vllm_gpu_memory_utilization <= 1.0):
            raise ValueError(
                f"vllm_gpu_memory_utilization must be in (0, 1], "
                f"got {self.vllm_gpu_memory_utilization}"
            )
        # Sampling temperature: 0 is "greedy" but vLLM/HF treat <=0 as a special
        # path; negative temperature has no defined meaning. Reject negatives.
        if self.rollout_temperature < 0.0:
            raise ValueError(
                f"rollout_temperature must be >= 0, got {self.rollout_temperature}"
            )
        if self.value_data_temperature < 0.0:
            raise ValueError(
                f"value_data_temperature must be >= 0, "
                f"got {self.value_data_temperature}"
            )
        # top_p must be in (0, 1]; 0 disables sampling entirely (no candidates).
        if not (0.0 < self.rollout_top_p <= 1.0):
            raise ValueError(
                f"rollout_top_p must be in (0, 1], got {self.rollout_top_p}"
            )
        # Learning rates: negative values flip the optimizer direction silently.
        if self.lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {self.lr}")
        if self.value_lr < 0.0:
            raise ValueError(f"value_lr must be >= 0, got {self.value_lr}")
        if self.online_value_lr < 0.0:
            raise ValueError(
                f"online_value_lr must be >= 0, got {self.online_value_lr}"
            )
        if self.critic_lr < 0.0:
            raise ValueError(f"critic_lr must be >= 0, got {self.critic_lr}")
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )
        if self.value_weight_decay < 0.0:
            raise ValueError(
                f"value_weight_decay must be >= 0, "
                f"got {self.value_weight_decay}"
            )
        if self.critic_weight_decay < 0.0:
            raise ValueError(
                f"critic_weight_decay must be >= 0, "
                f"got {self.critic_weight_decay}"
            )
        # IPVRM β rescales log-ratios into the BCE; 0 collapses V to 0
        # everywhere, negative β sign-inverts the value head.
        if self.value_beta <= 0.0:
            raise ValueError(
                f"value_beta must be > 0, got {self.value_beta}"
            )
        # value_margin is the BCE dead-zone; negative margins invert the loss.
        if self.value_margin < 0.0:
            raise ValueError(
                f"value_margin must be >= 0, got {self.value_margin}"
            )
        # Grad-clip <= 0 silently disables gradient clipping in most code
        # paths. Permit 0 (disabled) but reject negative.
        if self.grad_clip < 0.0:
            raise ValueError(
                f"grad_clip must be >= 0 (0 disables), got {self.grad_clip}"
            )
        if self.value_grad_clip < 0.0:
            raise ValueError(
                f"value_grad_clip must be >= 0 (0 disables), "
                f"got {self.value_grad_clip}"
            )
        if self.critic_grad_clip < 0.0:
            raise ValueError(
                f"critic_grad_clip must be >= 0 (0 disables), "
                f"got {self.critic_grad_clip}"
            )
        if self.value_loss_coef < 0.0:
            raise ValueError(
                f"value_loss_coef must be >= 0, got {self.value_loss_coef}"
            )
        if self.cliprange_value < 0.0:
            raise ValueError(
                f"cliprange_value must be >= 0, got {self.cliprange_value}"
            )
        if not (0.0 <= self.ppo_gae_lambda <= 1.0):
            raise ValueError(
                f"ppo_gae_lambda must be in [0, 1], got {self.ppo_gae_lambda}"
            )
        if self.eval_during_training_k < 1:
            raise ValueError(
                f"eval_during_training_k must be >= 1, got {self.eval_during_training_k}"
            )
        if self.eval_during_training_limit < 0:
            raise ValueError(
                "eval_during_training_limit must be >= 0 "
                f"(0 means no explicit cap), got {self.eval_during_training_limit}"
            )
        if self.eval_during_training_temperature < 0.0:
            raise ValueError(
                "eval_during_training_temperature must be >= 0, got "
                f"{self.eval_during_training_temperature}"
            )
        if not (0.0 < self.eval_during_training_vllm_util <= 1.0):
            raise ValueError(
                "eval_during_training_vllm_util must be in (0, 1], got "
                f"{self.eval_during_training_vllm_util}"
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
        # Optional Literal["auto", "fp8"] — validated here so None remains
        # a legal passthrough (= vLLM default). Any other string is rejected.
        if self.vllm_kv_cache_dtype is not None and self.vllm_kv_cache_dtype not in (
            "auto", "fp8",
        ):
            raise ValueError(
                "vllm_kv_cache_dtype must be one of None, 'auto', 'fp8'; "
                f"got {self.vllm_kv_cache_dtype!r}"
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
