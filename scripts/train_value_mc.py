"""Train V_phi via Math-Shepherd-style MC step labels (BCE on continuous p_hat targets).

Uses the same PrefixValueModel architecture (cumulative-log-ratio, β·log(π_φ/π_ref)) but
replaces the BCE-with-margin loss with standard BCE between σ(V_at_step_end / β) and the
MC-estimated p_hat. Naturally calibrated to P(success | prefix).

Input data: produced by scripts/mc_step_label.py
  - prompt_ids:    [N, P]
  - prompt_mask:   [N, P]
  - response_ids:  [N, R]
  - response_mask: [N, R]
  - step_end_idx:  [N]    int    # response-token idx to read V at
  - p_hat:         [N]    float  # MC-estimated P(success | prefix)
  - outcomes:      [N]    float  # base rollout outcome (for diagnostic)

Mostly mirrors scripts/train_value.py for FSDP / save / eval setup.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

# Apply Liger Kernel fused triton ops to Qwen2 BEFORE the model loads —
# RMSNorm, RoPE, SwiGLU get fused triton kernels for ~15-25% throughput.
# Skipping fused_linear_cross_entropy because we use logits_to_keep slicing.
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2
    apply_liger_kernel_to_qwen2(
        rope=True, swiglu=True, rms_norm=True,
        fused_linear_cross_entropy=False,
    )
except ImportError:
    pass  # liger_kernel optional; fall through to vanilla Qwen2

sys.path.insert(0, "/home/jason/experiment/CASPO")


class SigmoidHeadValueModel(torch.nn.Module):
    """Single-forward sigmoid-head V_φ(s) = σ(W·h_φ(s)).

    Architecture: phi backbone (Qwen2 etc.) + Linear(hidden_size, 1).
    Trained with BCE(σ(logit), p_hat). No ref model — single forward per step.
    """
    def __init__(self, model_name_or_path: str, attn_impl=None):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kwargs = dict(torch_dtype=torch.bfloat16)
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.phi = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        hidden = self.phi.config.hidden_size
        self.value_head = torch.nn.Linear(hidden, 1, bias=True, dtype=torch.bfloat16)
        # Zero-init: σ(0)=0.5 ≈ p_hat prior; stable starting point.
        torch.nn.init.zeros_(self.value_head.weight)
        torch.nn.init.zeros_(self.value_head.bias)
        # For compat with the IPVRM path that names attribute self.ref:
        self.ref = None
        self.cfg_model_name = model_name_or_path

    def forward(self, prompt_ids, prompt_mask, response_ids, response_mask):
        input_ids = torch.cat([prompt_ids, response_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        # Call phi (the FSDP-wrapped outer module) so flat-sharded params are
        # properly all-gathered. Calling self.phi.model directly trips FSDP's
        # 1-D flat param view → "weight must be 2-D" inside torch.embedding.
        # We pay an unused lm_head matmul (tied embeds → no extra storage, ~1ms/step).
        out = self.phi(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False,
        )
        hidden = out.hidden_states[-1]  # last-layer hidden [B, P+R, H]
        # Slice response-side hidden states only — caller indexes by step_end_idx
        # which is already a response-token offset (0..R-1).
        P = prompt_ids.shape[1]
        resp_hidden = hidden[:, P:, :]  # [B, R, H]
        logits = self.value_head(resp_hidden).squeeze(-1).float()  # [B, R] in fp32
        return {"logits": logits}

    def save_pretrained(self, path: str):
        # FSDP-aware: gather to rank-0 then write. Mirrors
        # PrefixValueModel.save_pretrained.
        phi = self.phi
        is_fsdp = phi.__class__.__name__ == "FullyShardedDataParallel"
        try:
            _dist_init = dist.is_available() and dist.is_initialized()
            rank = dist.get_rank() if _dist_init else 0
        except Exception:
            _dist_init = False
            rank = 0
        if is_fsdp:
            from torch.distributed.fsdp import (
                FullStateDictConfig,
                FullyShardedDataParallel as _FSDP,
                StateDictType,
            )
            full_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with _FSDP.state_dict_type(phi, StateDictType.FULL_STATE_DICT, full_cfg):
                state_dict = phi.state_dict()
            if rank == 0:
                os.makedirs(path, exist_ok=True)
                inner = getattr(phi, "module", phi)
                inner.save_pretrained(path, state_dict=state_dict)
                try:
                    self.tokenizer.save_pretrained(path)
                except Exception:
                    pass
                torch.save({
                    "value_head_state": self.value_head.state_dict(),
                    "architecture": "sigmoid_head",
                    "phi_init": self.cfg_model_name,
                }, os.path.join(path, "value_head.pt"))
            return
        if rank != 0:
            return
        os.makedirs(path, exist_ok=True)
        phi.save_pretrained(path)
        try:
            self.tokenizer.save_pretrained(path)
        except Exception:
            pass
        torch.save({
            "value_head_state": self.value_head.state_dict(),
            "architecture": "sigmoid_head",
            "phi_init": self.cfg_model_name,
        }, os.path.join(path, "value_head.pt"))


def is_dist_initialized() -> bool:
    try:
        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def _rprint(msg, flush=True):
    rank = dist.get_rank() if is_dist_initialized() else 0
    if rank == 0:
        print(msg, flush=flush)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="path to MC-labeled .pt from mc_step_label.py")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--eval_mb", type=int, default=0,
                    help="eval batch size (0 = use 4×mb). Eval has no backward, "
                         "so the activation footprint is much smaller than train; "
                         "a 4× larger eval batch typically halves total eval wall.")
    ap.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps; effective batch = mb*FSDP_size*grad_accum")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--save_every", type=int, default=200)
    # eval_every: 50 was over-evaluating. eval_val is collective FSDP across
    # ~1606 val rows / mb=4 ≈ ~400 forward steps per eval. At eval_every=50 +
    # 2712 train steps that's 54 evals × 400 ≈ 21.6k val forwards, dominating
    # the 2712 train forwards. eval_every=200 is the new default; launchers
    # may still pass eval_every=100 to keep early-stop responsiveness.
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--early_stop_patience", type=int, default=8)
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, default=10.0,
                    help="V scaling for sigmoid (loss = BCE(σ(V/β), p_hat))")
    ap.add_argument("--ref_path", default=None,
                    help="ref model path (default: cfg.model_name_or_path)")
    ap.add_argument("--phi_init_path", default=None,
                    help="phi init path (default: cfg.model_name_or_path; same as ref)")
    ap.add_argument("--lora_r", type=int, default=0,
                    help="LoRA rank; 0 = full FT, >0 = LoRA on phi backbone")
    ap.add_argument("--lora_alpha", type=int, default=64,
                    help="LoRA alpha (typical 2*r)")
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--architecture", choices=["ipvrm", "sigmoid_head"], default="ipvrm",
                    help="ipvrm: V=β·Σlog(π_φ/π_ref) (2 fwds, 2 backbones); "
                         "sigmoid_head: V=σ(W·h_φ) (1 fwd, 1 backbone + linear head)")
    ap.add_argument("--split_by_prompt", action="store_true",
                    help="Hold out --val_fraction of unique PROMPTS (all their prefixes "
                         "go to val) rather than a random prefix-level shuffle. Eliminates "
                         "same-prompt leakage in the val set within the same collection.")
    ap.add_argument("--held_out_data", default=None,
                    help="Path to a separate mc_labels-format .pt file to use as the val "
                         "set (replacing the in-data split). Use this for true OOD-prompt "
                         "evaluation against a held-out dsr_sub subset.")
    ap.add_argument("--held_out_max_rows", type=int, default=0,
                    help="If >0, truncate held-out val to this many rows. ~12k rows "
                         "(≈200 unique prompts) gives ρ CI half-width ±0.07 — plenty "
                         "for an architecture-comparison signal — and cuts eval wall by "
                         "~4× vs evaluating the full 909-prompt collection.")
    args = ap.parse_args()

    from caspo.config import CASPOConfig
    from caspo.value.prefix_value import PrefixValueModel

    # Init distributed
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not is_dist_initialized():
        dist.init_process_group(backend="nccl")
    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    cfg = CASPOConfig.from_yaml(args.config)
    # If --phi_init_path is given, set cfg.model_name_or_path so PrefixValueModel
    # initializes phi+ref from there. (Both initialized from same checkpoint to
    # match IPVRM cumulative-log-ratio architecture: V starts at 0 since π_φ ≈ π_ref.)
    if args.phi_init_path is not None:
        cfg.model_name_or_path = args.phi_init_path
    cfg.value_beta = args.beta
    # Disable gradient checkpointing — combination of FSDP + grad-ckpt + Qwen2
    # produces "setStorage out of bounds for storage of size 0" in backward.
    cfg.use_gradient_checkpointing = False
    # FA3 sometimes incompatible with Qwen2; use SDPA for safety
    cfg.attn_implementation = None

    _rprint(f"[mc-train] loading data from {args.data}")
    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    # Pin source tensors so per-batch .to(device, non_blocking=True) copies
    # can overlap with the previous step's compute. Cheap (just registers
    # the host buffer with CUDA); pays off every step since each batch
    # touches 5 tensors with .to(device).
    for _k in ("prompt_ids", "prompt_mask", "response_ids",
               "response_mask", "step_end_idx", "p_hat"):
        if _k in blob and torch.is_tensor(blob[_k]) and not blob[_k].is_pinned():
            try:
                blob[_k] = blob[_k].pin_memory()
            except Exception:
                pass  # pin_memory can OOM on host RAM-tight boxes
    N = int(blob["prompt_ids"].shape[0])
    _rprint(f"[mc-train] N={N} labeled prefixes")
    _rprint(f"[mc-train] p_hat: mean={float(blob['p_hat'].mean()):.3f} std={float(blob['p_hat'].std()):.3f}")

    # ----------------------------------------------------------
    # Train/val split. Three modes:
    # (a) --held_out_data <path>: load a separate mc_labels .pt as val. True
    #     OOD-prompt eval. Train uses ALL of the primary blob.
    # (b) --split_by_prompt: hash prompt_ids per row, group by prompt, hold
    #     out --val_fraction of unique prompts (ALL their prefixes go to val).
    #     Removes same-prompt prefix leakage within the same collection.
    # (c) default (legacy): random row-level shuffle. Leaks same-prompt
    #     prefixes between train and val; val ρ overstates OOD generalization.
    # ----------------------------------------------------------
    val_blob = blob  # default: val drawn from the same blob as train
    if args.held_out_data is not None:
        _rprint(f"[mc-train] loading HELD-OUT val data from {args.held_out_data}")
        val_blob = torch.load(args.held_out_data, map_location="cpu", weights_only=False)
        # Optional cap to keep eval wall manageable. Held-out files are
        # typically deterministically ordered by shard concat, so a
        # head-slice is a representative sample (no need to shuffle).
        if args.held_out_max_rows > 0 and val_blob["prompt_ids"].shape[0] > args.held_out_max_rows:
            cap = args.held_out_max_rows
            for _k in ("prompt_ids", "prompt_mask", "response_ids",
                       "response_mask", "step_end_idx", "p_hat", "outcomes"):
                if _k in val_blob and torch.is_tensor(val_blob[_k]):
                    val_blob[_k] = val_blob[_k][:cap].contiguous()
            _rprint(f"[mc-train] held-out truncated to {cap} rows (--held_out_max_rows)")
        for _k in ("prompt_ids", "prompt_mask", "response_ids",
                   "response_mask", "step_end_idx", "p_hat"):
            if _k in val_blob and torch.is_tensor(val_blob[_k]) and not val_blob[_k].is_pinned():
                try:
                    val_blob[_k] = val_blob[_k].pin_memory()
                except Exception:
                    pass
        N_val = int(val_blob["prompt_ids"].shape[0])
        train_idxs = list(range(N))
        val_idxs = list(range(N_val))
        _rprint(f"[mc-train] mode=held_out_file  train={len(train_idxs)} val={len(val_idxs)}")
    elif args.split_by_prompt:
        # Hash each row's prompt_ids (× prompt_mask) to identify unique prompts.
        rng = random.Random(args.seed)
        pid_t = blob["prompt_ids"]
        pmask_t = blob["prompt_mask"]
        # Mask out padding before hashing so identical prompts with different
        # padding align. Convert to bytes once.
        prompt_to_rows: dict[bytes, list[int]] = {}
        for i in range(N):
            real = (pid_t[i] * pmask_t[i]).numpy().tobytes()
            prompt_to_rows.setdefault(real, []).append(i)
        unique_prompts = list(prompt_to_rows.keys())
        rng.shuffle(unique_prompts)
        n_val_prompts = max(1, int(len(unique_prompts) * args.val_fraction))
        val_prompts = set(unique_prompts[:n_val_prompts])
        val_idxs = [r for p in unique_prompts[:n_val_prompts] for r in prompt_to_rows[p]]
        train_idxs = [r for p in unique_prompts[n_val_prompts:] for r in prompt_to_rows[p]]
        _rprint(f"[mc-train] mode=split_by_prompt  unique_prompts={len(unique_prompts)} "
                f"(val={n_val_prompts}, train={len(unique_prompts)-n_val_prompts})  "
                f"train_rows={len(train_idxs)} val_rows={len(val_idxs)}")
    else:
        rng = random.Random(args.seed)
        perm = list(range(N))
        rng.shuffle(perm)
        n_val = max(1, int(N * args.val_fraction))
        val_idxs = perm[:n_val]
        train_idxs = perm[n_val:]
        _rprint(f"[mc-train] mode=row_shuffle (LEAKY — same prompt may appear in "
                f"both train and val)  train={len(train_idxs)} val={len(val_idxs)}")

    # Per-rank shard of train rows
    if world_size > 1:
        train_idxs = train_idxs[rank::world_size]
        _rprint(f"[mc-train] rank {rank}: {len(train_idxs)} local train rows")

    # Build model — branch on architecture
    _rprint(f"[mc-train] architecture={args.architecture}")
    if args.architecture == "ipvrm":
        _rprint(f"[mc-train] init phi+ref from cfg.model_name_or_path={cfg.model_name_or_path}")
        pv = PrefixValueModel(cfg)
        pv.phi = pv.phi.to(device)
        pv.ref = pv.ref.to(device)
    else:
        _rprint(f"[mc-train] init sigmoid-head phi from {cfg.model_name_or_path}")
        pv = SigmoidHeadValueModel(cfg.model_name_or_path)
        pv.phi = pv.phi.to(device)
        pv.value_head = pv.value_head.to(device)

    use_lora = args.lora_r > 0
    if use_lora:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj","k_proj","v_proj","o_proj",
                            "gate_proj","up_proj","down_proj"],
        )
        pv.phi = get_peft_model(pv.phi, lora_cfg)
        if is_main:
            pv.phi.print_trainable_parameters()
        # LoRA: use DDP (trainable params are tiny → cheap all-reduce)
        if world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            # convert frozen base to bf16 to mirror FSDP MixedPrecision savings
            pv.phi = pv.phi.to(torch.bfloat16)
            pv.phi = DDP(pv.phi, device_ids=[local_rank],
                         find_unused_parameters=False,
                         broadcast_buffers=False)
    else:
        # FSDP-wrap phi only (ref is frozen) — full FT path
        if world_size > 1:
            # Use per-block auto_wrap_policy (mirrors caspo/value/train_value.py).
            # Without this, the entire 1.5B model is one FSDP unit → no
            # compute/comm overlap → ~2-3× wallclock vs per-block wrapping.
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            import functools
            # Auto-detect transformer block classes (Qwen2DecoderLayer etc.)
            layer_classes = set()
            for module in pv.phi.modules():
                cls_name = type(module).__name__
                if cls_name.endswith("DecoderLayer") or cls_name.endswith("Block"):
                    layer_classes.add(type(module))
            wrap_kwargs = dict(
                device_id=local_rank,
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,  # was fp32 — bf16 halves comm bytes;
                                                  # PRM training is short + tolerant
                    buffer_dtype=torch.bfloat16,
                ),
                sync_module_states=True,
                forward_prefetch=True,
                limit_all_gathers=True,
                use_orig_params=True,
            )
            if layer_classes:
                wrap_kwargs["auto_wrap_policy"] = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls=layer_classes,
                )
            pv.phi = FSDP(pv.phi, **wrap_kwargs)
    if args.architecture == "ipvrm":
        pv.ref.eval()
        for p in pv.ref.parameters():
            p.requires_grad = False

    if use_lora:
        # only LoRA params (peft already froze base)
        trainable = [p for p in pv.phi.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, fused=True)
    elif args.architecture == "sigmoid_head":
        # Train phi backbone + value_head together
        params = list(pv.phi.parameters()) + list(pv.value_head.parameters())
        optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, fused=True)
    else:
        optim = torch.optim.AdamW(pv.phi.parameters(), lr=args.lr, weight_decay=0.0, fused=True)

    def save_pv_ckpt(path):
        """Save pv at path, handling LoRA merge for drop-in compatibility."""
        if use_lora:
            # Unwrap DDP if present
            peft_model = pv.phi.module if hasattr(pv.phi, 'module') else pv.phi
            peft_model.merge_adapter()
            try:
                if is_main:
                    base_model = peft_model.base_model.model
                    base_model.save_pretrained(path)
                    # save tokenizer + meta to match PrefixValueModel.from_pretrained.
                    # PrefixValueModel exposes the HF tokenizer as ``self.tokenizer``;
                    # the prior ``pv._tokenizer`` lookup silently no-op'd, leaving
                    # LoRA-merged checkpoints without a tokenizer dir and breaking
                    # downstream ``AutoTokenizer.from_pretrained(path)`` calls.
                    tok = getattr(pv, 'tokenizer', None)
                    if tok is not None:
                        tok.save_pretrained(path)
                    with open(os.path.join(path, 'caspo_value_meta.json'), 'w') as f:
                        json.dump({'ref_model_path': cfg.model_name_or_path}, f)
            finally:
                peft_model.unmerge_adapter()
            if is_dist_initialized():
                dist.barrier()
        else:
            if hasattr(pv, "save_pretrained"):
                pv.save_pretrained(path)
        # Save FSDP-aware optimizer state. Without this, resuming a partial
        # PRM run starts with fresh Adam moments → loss blows up for ~50
        # steps until 2nd-moment estimate stabilises (cf. memory
        # feedback_resume_optimizer_state.md). Only rank 0 writes.
        try:
            if isinstance(pv.phi, FSDP):
                from torch.distributed.fsdp.api import (
                    FullStateDictConfig as _FSDC,
                    FullOptimStateDictConfig,
                    StateDictType as _SDT,
                )
                full_optim_cfg = FullOptimStateDictConfig(
                    offload_to_cpu=True, rank0_only=True,
                )
                full_state_cfg = _FSDC(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(pv.phi, _SDT.FULL_STATE_DICT,
                                         full_state_cfg, full_optim_cfg):
                    optim_state = FSDP.optim_state_dict(pv.phi, optim)
            else:
                optim_state = optim.state_dict()
            if is_main:
                torch.save(optim_state, os.path.join(path, "optimizer.pt"))
        except Exception as _e:
            if is_main:
                _rprint(f"[mc-train] optimizer save failed: {_e}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Training
    n_train = len(train_idxs)
    steps_per_epoch = max(1, (n_train + args.mb - 1) // args.mb)
    total_steps = steps_per_epoch * args.epochs
    _rprint(f"[mc-train] {args.epochs} epochs × {steps_per_epoch} steps = {total_steps} total")

    def get_batch(idxs, source=None):
        # non_blocking=True overlaps H2D w/ compute of the previous step;
        # only effective because the source tensors were pinned above.
        # `source` lets eval pull from val_blob when --held_out_data is set.
        src = source if source is not None else blob
        prompt_ids = src["prompt_ids"][idxs].to(device, non_blocking=True)
        prompt_mask = src["prompt_mask"][idxs].to(device, non_blocking=True)
        response_ids = src["response_ids"][idxs].to(device, non_blocking=True)
        response_mask = src["response_mask"][idxs].to(device, non_blocking=True)
        step_end_idx = src["step_end_idx"][idxs].to(device, non_blocking=True)
        p_hat = src["p_hat"][idxs].to(device, non_blocking=True)
        return prompt_ids, prompt_mask, response_ids, response_mask, step_end_idx, p_hat

    def forward_loss(idxs, return_preds=False, want_diag=True, source=None):
        pids, pmask, rids, rmask, step_end, p_hat = get_batch(idxs, source=source)
        out = pv(prompt_ids=pids, prompt_mask=pmask,
                 response_ids=rids, response_mask=rmask)
        if args.architecture == "ipvrm":
            V = out["V"]  # [B, R+1]; V[:, 0]=0, V[:, t] = sum_{i<t} log_ratio[:, i]
            # Read V AFTER the last response token of the labeled step.
            # ``step_end_idx`` is the (0-indexed) response-token idx INCLUDED in
            # the labeled step; the cumulative log-ratio THROUGH that token lives
            # at V[:, step_end_idx + 1]. The prior version read V[:, step_end_idx]
            # which silently excluded the last token of every step (off-by-one).
            last_col = V.shape[1] - 1
            gather_idx = (step_end + 1).clamp(max=last_col).unsqueeze(1)
            last_v = V.gather(1, gather_idx).squeeze(1).float()
            logits = last_v / args.beta
        else:
            # sigmoid_head: out["logits"] is [B, R]; gather at step_end (no off-by-one
            # since logits are response-aligned with hidden states at each token).
            logits_full = out["logits"]  # [B, R]
            last_col = logits_full.shape[1] - 1
            gather_idx = step_end.clamp(max=last_col).unsqueeze(1)
            logits = logits_full.gather(1, gather_idx).squeeze(1).float()
        # Equivalent to F.binary_cross_entropy_with_logits but supports continuous targets
        loss = F.binary_cross_entropy_with_logits(logits, p_hat)
        # Diagnostics — skipped on non-log steps to avoid an extra sigmoid +
        # 2 reductions per training step. eval/log paths set want_diag=True.
        if want_diag or return_preds:
            with torch.no_grad():
                preds = torch.sigmoid(logits)
                pred_mse = ((preds - p_hat) ** 2).mean()
                mean_pred = preds.mean()
        else:
            preds = None
            pred_mse = loss.detach()  # placeholder; not logged on this step
            mean_pred = loss.detach()
        if return_preds:
            return loss, pred_mse, mean_pred, preds.detach().cpu(), p_hat.detach().cpu()
        return loss, pred_mse, mean_pred

    @torch.no_grad()
    def eval_val():
        # COLLECTIVE eval — every rank evaluates its shard of val_idxs.
        # Required because FSDP-wrapped phi.forward() is collective; only-rank-0
        # eval would deadlock other ranks waiting for next training-step collective.
        if not val_idxs:
            return None
        pv.phi.eval()
        local_val = val_idxs[rank::world_size] if world_size > 1 else val_idxs
        total_loss_t = torch.zeros(1, device=device)
        total_mse_t = torch.zeros(1, device=device)
        n_eval_t = torch.zeros(1, device=device)
        local_preds = []
        local_phat = []
        # Eval batch size: default is 4×mb because eval has no backward → no
        # activation memory growth → can fit a much larger forward batch. This
        # cuts eval wall by ~3-4× without affecting metric.
        eval_mb = args.eval_mb if args.eval_mb > 0 else max(args.mb * 4, args.mb)
        for s in range(0, len(local_val), eval_mb):
            batch_idxs = torch.tensor(local_val[s:s + eval_mb], dtype=torch.long)
            loss, mse, _, preds, p_hat = forward_loss(batch_idxs, return_preds=True, source=val_blob)
            n = float(len(batch_idxs))
            total_loss_t += loss.detach().float() * n
            total_mse_t += mse.detach().float() * n
            n_eval_t += n
            local_preds.append(preds)
            local_phat.append(p_hat)
        if is_dist_initialized():
            dist.all_reduce(total_loss_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_mse_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_eval_t, op=dist.ReduceOp.SUM)
        n_eval = float(n_eval_t.item())
        if n_eval == 0:
            return None
        # Gather predictions across ranks for global Spearman ρ + AUC
        local_preds_t = torch.cat(local_preds) if local_preds else torch.zeros(0)
        local_phat_t = torch.cat(local_phat) if local_phat else torch.zeros(0)
        if is_dist_initialized() and world_size > 1:
            preds_lists = [None] * world_size
            phat_lists = [None] * world_size
            dist.all_gather_object(preds_lists, local_preds_t.numpy().tolist())
            dist.all_gather_object(phat_lists, local_phat_t.numpy().tolist())
            import numpy as _np
            all_preds = _np.array([x for lst in preds_lists for x in lst])
            all_phat = _np.array([x for lst in phat_lists for x in lst])
        else:
            import numpy as _np
            all_preds = local_preds_t.numpy()
            all_phat = local_phat_t.numpy()
        spearman_rho = float("nan")
        within_auc = float("nan")
        if is_main and len(all_preds) >= 2:
            try:
                from scipy.stats import spearmanr
                rho, _ = spearmanr(all_preds, all_phat)
                spearman_rho = float(rho) if rho is not None else float("nan")
            except Exception:
                pass
            # Within-AUC: binary AUC where label = (p_hat > 0.5)
            try:
                from sklearn.metrics import roc_auc_score
                bin_label = (all_phat > 0.5).astype(int)
                if bin_label.min() != bin_label.max():
                    within_auc = float(roc_auc_score(bin_label, all_preds))
            except Exception:
                pass
        return {"val_loss": float(total_loss_t.item()) / n_eval,
                "val_mse": float(total_mse_t.item()) / n_eval,
                "val_spearman": spearman_rho,
                "val_within_auc": within_auc}

    best_val_loss = float("inf")
    no_improve = 0
    early_stopped = False
    train_log = []

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        gen = torch.Generator(); gen.manual_seed(args.seed + epoch)
        train_t = torch.tensor(train_idxs, dtype=torch.long)
        perm_t = torch.randperm(len(train_t), generator=gen)
        train_t = train_t[perm_t]
        accum_count = 0
        for s in range(0, len(train_t), args.mb):
            batch_idxs = train_t[s:s + args.mb]
            pv.phi.train()
            # Only compute diagnostics on the iteration that will actually
            # log (at-or-near the next log step). step is the *optimizer*
            # step; we approximate "near a log step" by checking the next
            # post-accum step would land on a log boundary.
            _will_log = ((step + 1) % 10 == 0) or ((step + 1) == total_steps)
            loss, mse, mean_pred = forward_loss(batch_idxs, want_diag=_will_log)
            scaled_loss = loss / args.grad_accum
            # Skip all_gather on intermediate accum batches via FSDP no_sync
            if accum_count + 1 < args.grad_accum and isinstance(pv.phi, FSDP):
                with pv.phi.no_sync():
                    scaled_loss.backward()
            else:
                scaled_loss.backward()
            accum_count += 1
            if accum_count == args.grad_accum:
                step += 1
                accum_count = 0
                # FSDP exposes its own clip_grad_norm_ that aggregates across
                # shards; the standalone torch.nn.utils version operates only
                # on local shards and under-counts the global norm. Use the
                # FSDP method when wrapped, else fall back.
                if isinstance(pv.phi, FSDP):
                    pv.phi.clip_grad_norm_(1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(pv.phi.parameters(), 1.0)
                optim.step()
                # set_to_none frees grad memory rather than zeroing in-place,
                # which is faster and matches HF Trainer / caspo_trainer.
                optim.zero_grad(set_to_none=True)
            else:
                continue
            if step % 10 == 0:
                _rprint(
                    f"[mc-train step {step}/{total_steps}] loss={float(loss):.4f} "
                    f"mse={float(mse):.4f} mean_pred={float(mean_pred):.3f} "
                    f"t={time.time()-t0:.1f}s"
                )
            if step % args.eval_every == 0 or step == total_steps:
                stats = eval_val()  # collective — all ranks
                if stats is not None:
                    if is_main:
                        _rprint(f"[mc-train step {step}] val_loss={stats['val_loss']:.4f} "
                                f"val_mse={stats['val_mse']:.4f} "
                                f"val_ρ={stats['val_spearman']:.4f} "
                                f"val_within_AUC={stats['val_within_auc']:.4f} "
                                f"(best={best_val_loss:.4f})")
                        train_log.append({"step": step, **stats})
                    is_best = stats["val_loss"] < best_val_loss - 1e-4
                    if is_best:
                        best_val_loss = stats["val_loss"]
                        no_improve = 0
                        best_path = os.path.join(args.output_dir, "best")
                        if is_dist_initialized():
                            dist.barrier()
                        # save_pretrained on FSDP-wrapped phi is collective
                        save_pv_ckpt(best_path)
                    else:
                        no_improve += 1
                        if no_improve >= args.early_stop_patience and step > 200:
                            early_stopped = True
                            _rprint(f"[mc-train] early stop at step {step} (no val improvement for {no_improve} evals)")
                            break
            if step % args.save_every == 0 and step > 0:
                ckpt_path = os.path.join(args.output_dir, f"step_{step}")
                if is_dist_initialized():
                    dist.barrier()
                save_pv_ckpt(ckpt_path)
                if is_main:
                    _rprint(f"[mc-train] saved checkpoint to {ckpt_path}")
        if early_stopped:
            break

    # Final save
    final_path = os.path.join(args.output_dir, "final")
    if is_dist_initialized():
        dist.barrier()
    save_pv_ckpt(final_path)
    if is_main:
        _rprint(f"[mc-train] DONE. final={final_path}")
        # value_final symlink
        link = os.path.join(args.output_dir, "value_final")
        try:
            if os.path.lexists(link):
                os.remove(link)
            os.symlink("final", link)
        except Exception as _e:
            _rprint(f"[mc-train] symlink fail: {_e}")
        # Save log
        with open(os.path.join(args.output_dir, "mc_train_log.json"), "w") as f:
            json.dump({"final_step": step, "best_val_loss": best_val_loss,
                       "early_stopped": early_stopped, "log": train_log}, f, indent=2)


if __name__ == "__main__":
    main()
