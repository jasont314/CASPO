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
    ap.add_argument("--grad_accum", type=int, default=1, help="gradient accumulation steps; effective batch = mb*FSDP_size*grad_accum")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=50)
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
    N = int(blob["prompt_ids"].shape[0])
    _rprint(f"[mc-train] N={N} labeled prefixes")
    _rprint(f"[mc-train] p_hat: mean={float(blob['p_hat'].mean()):.3f} std={float(blob['p_hat'].std()):.3f}")

    # Train/val split (random by row, since each row is one (prefix, p_hat) pair)
    rng = random.Random(args.seed)
    perm = list(range(N))
    rng.shuffle(perm)
    n_val = max(1, int(N * args.val_fraction))
    val_idxs = perm[:n_val]
    train_idxs = perm[n_val:]
    _rprint(f"[mc-train] train={len(train_idxs)} val={len(val_idxs)}")

    # Per-rank shard of train rows
    if world_size > 1:
        train_idxs = train_idxs[rank::world_size]
        _rprint(f"[mc-train] rank {rank}: {len(train_idxs)} local train rows")

    # Build model
    _rprint(f"[mc-train] init phi+ref from cfg.model_name_or_path={cfg.model_name_or_path}")
    pv = PrefixValueModel(cfg)
    pv.phi = pv.phi.to(device)
    pv.ref = pv.ref.to(device)

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
            pv.phi = FSDP(
                pv.phi,
                device_id=local_rank,
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    buffer_dtype=torch.bfloat16,
                ),
                sync_module_states=True,
            )
    pv.ref.eval()
    for p in pv.ref.parameters():
        p.requires_grad = False

    if use_lora:
        # only LoRA params (peft already froze base)
        trainable = [p for p in pv.phi.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, fused=True)
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
                    # save tokenizer + meta to match PrefixValueModel.from_pretrained
                    if hasattr(pv, '_tokenizer') and pv._tokenizer is not None:
                        pv._tokenizer.save_pretrained(path)
                    with open(os.path.join(path, 'caspo_value_meta.json'), 'w') as f:
                        json.dump({'ref_model_path': cfg.model_name_or_path}, f)
            finally:
                peft_model.unmerge_adapter()
            if is_dist_initialized():
                dist.barrier()
        else:
            if hasattr(pv, "save_pretrained"):
                pv.save_pretrained(path)

    os.makedirs(args.output_dir, exist_ok=True)

    # Training
    n_train = len(train_idxs)
    steps_per_epoch = max(1, (n_train + args.mb - 1) // args.mb)
    total_steps = steps_per_epoch * args.epochs
    _rprint(f"[mc-train] {args.epochs} epochs × {steps_per_epoch} steps = {total_steps} total")

    def get_batch(idxs):
        prompt_ids = blob["prompt_ids"][idxs].to(device)
        prompt_mask = blob["prompt_mask"][idxs].to(device)
        response_ids = blob["response_ids"][idxs].to(device)
        response_mask = blob["response_mask"][idxs].to(device)
        step_end_idx = blob["step_end_idx"][idxs].to(device)
        p_hat = blob["p_hat"][idxs].to(device)
        return prompt_ids, prompt_mask, response_ids, response_mask, step_end_idx, p_hat

    def forward_loss(idxs, return_preds=False):
        pids, pmask, rids, rmask, step_end, p_hat = get_batch(idxs)
        out = pv(prompt_ids=pids, prompt_mask=pmask,
                 response_ids=rids, response_mask=rmask)
        V = out["V"]  # [B, R]
        # Read V at step_end_idx for each row
        last_v = V.gather(1, step_end.unsqueeze(1)).squeeze(1).float()
        # Loss: BCE(sigmoid(V/β), p_hat), with p_hat in [0, 1]
        logits = last_v / args.beta
        # Equivalent to F.binary_cross_entropy_with_logits but supports continuous targets
        loss = F.binary_cross_entropy_with_logits(logits, p_hat)
        # Diagnostics
        with torch.no_grad():
            preds = torch.sigmoid(logits)
            pred_mse = ((preds - p_hat) ** 2).mean()
            mean_pred = preds.mean()
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
        for s in range(0, len(local_val), args.mb):
            batch_idxs = torch.tensor(local_val[s:s + args.mb], dtype=torch.long)
            loss, mse, _, preds, p_hat = forward_loss(batch_idxs, return_preds=True)
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
            loss, mse, mean_pred = forward_loss(batch_idxs)
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
                torch.nn.utils.clip_grad_norm_(pv.phi.parameters(), 1.0)
                optim.step()
                optim.zero_grad()
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
