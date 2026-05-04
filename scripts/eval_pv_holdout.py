"""Eval-only PRM scorer.

Loads a saved PrefixValueModel checkpoint (IPVRM or sigmoid_head) and scores it
on a held-out mc_labels-format .pt file. Reports Spearman ρ, within-AUC,
val_loss (BCE), MSE on the predicted P(success | prefix) vs the MC label p̂.

Architecture is auto-detected: presence of value_head.pt → sigmoid_head; else IPVRM.
For IPVRM the ref model is loaded from caspo_value_meta.json (ref_model_path) or
--ref_path; defaults to Qwen/Qwen2.5-Math-1.5B.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# Apply Liger Kernel to match training-time fused ops
try:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen2
    apply_liger_kernel_to_qwen2(rope=True, swiglu=True, rms_norm=True,
                                fused_linear_cross_entropy=False)
except ImportError:
    pass

sys.path.insert(0, "/home/jason/experiment/CASPO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to PRM checkpoint dir")
    ap.add_argument("--data", required=True, help="held-out mc_labels .pt file")
    ap.add_argument("--ref_path", default=None,
                    help="ref model path for IPVRM; default reads caspo_value_meta.json or falls back to Qwen/Qwen2.5-Math-1.5B")
    ap.add_argument("--max_rows", type=int, default=0,
                    help="if >0, eval on first N rows (for quick checks)")
    ap.add_argument("--mb", type=int, default=16)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--output_json", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    has_value_head = os.path.exists(os.path.join(args.ckpt, "value_head.pt"))
    arch = "sigmoid_head" if has_value_head else "ipvrm"
    print(f"[eval] ckpt={args.ckpt}  arch={arch}", flush=True)

    print(f"[eval] loading phi from {args.ckpt}", flush=True)
    phi = AutoModelForCausalLM.from_pretrained(
        args.ckpt, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    for p in phi.parameters():
        p.requires_grad = False

    ref = None
    value_head = None
    if arch == "ipvrm":
        ref_path = args.ref_path
        if ref_path is None:
            meta_p = os.path.join(args.ckpt, "caspo_value_meta.json")
            if os.path.exists(meta_p):
                meta = json.load(open(meta_p))
                ref_path = meta.get("ref_model_path") or meta.get("model_name_or_path")
        if ref_path is None:
            ref_path = "Qwen/Qwen2.5-Math-1.5B"
        print(f"[eval] loading ref from {ref_path}", flush=True)
        ref = AutoModelForCausalLM.from_pretrained(
            ref_path, torch_dtype=torch.bfloat16,
        ).to(device).eval()
        for p in ref.parameters():
            p.requires_grad = False
    else:
        head_state = torch.load(os.path.join(args.ckpt, "value_head.pt"),
                                map_location=device, weights_only=False)
        hidden = phi.config.hidden_size
        value_head = torch.nn.Linear(hidden, 1, bias=True, dtype=torch.bfloat16).to(device)
        value_head.load_state_dict(head_state["value_head_state"])
        value_head.eval()
        for p in value_head.parameters():
            p.requires_grad = False

    print(f"[eval] loading data from {args.data}", flush=True)
    blob = torch.load(args.data, map_location="cpu", weights_only=False)
    N = int(blob["prompt_ids"].shape[0])
    if args.max_rows > 0 and N > args.max_rows:
        N = args.max_rows
    print(f"[eval] N={N}", flush=True)

    all_preds = []
    all_phat = []

    with torch.no_grad():
        for s in range(0, N, args.mb):
            e = min(s + args.mb, N)
            pids = blob["prompt_ids"][s:e].to(device, non_blocking=True)
            pmask = blob["prompt_mask"][s:e].to(device, non_blocking=True)
            rids = blob["response_ids"][s:e].to(device, non_blocking=True)
            rmask = blob["response_mask"][s:e].to(device, non_blocking=True)
            step_end = blob["step_end_idx"][s:e].to(device, non_blocking=True)
            phat = blob["p_hat"][s:e]

            input_ids = torch.cat([pids, rids], dim=1)
            attention_mask = torch.cat([pmask, rmask], dim=1)
            P = pids.shape[1]
            R = rids.shape[1]

            if arch == "ipvrm":
                # phi + ref forward; per-token log-prob of actual response token at
                # positions [P-1 : P-1+R] in the logit sequence (causal LM shift-by-one).
                phi_out = phi(input_ids=input_ids, attention_mask=attention_mask)
                ref_out = ref(input_ids=input_ids, attention_mask=attention_mask)
                phi_logits = phi_out.logits[:, P - 1:P - 1 + R, :].float()
                ref_logits = ref_out.logits[:, P - 1:P - 1 + R, :].float()
                phi_logp = F.log_softmax(phi_logits, dim=-1)
                ref_logp = F.log_softmax(ref_logits, dim=-1)
                tok_phi = phi_logp.gather(2, rids.unsqueeze(-1)).squeeze(-1)
                tok_ref = ref_logp.gather(2, rids.unsqueeze(-1)).squeeze(-1)
                log_ratio = (tok_phi - tok_ref) * rmask.float()
                V_head = torch.zeros((log_ratio.shape[0], 1), device=device, dtype=log_ratio.dtype)
                V = torch.cat([V_head, log_ratio.cumsum(dim=1)], dim=1) * args.beta
                last_col = V.shape[1] - 1
                gather_idx = (step_end + 1).clamp(max=last_col).unsqueeze(1)
                last_v = V.gather(1, gather_idx).squeeze(1)
                logits = last_v / args.beta
            else:
                phi_out = phi.model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = phi_out.last_hidden_state
                resp_hidden = hidden[:, P:, :]
                logits_full = value_head(resp_hidden).squeeze(-1).float()
                last_col = logits_full.shape[1] - 1
                gather_idx = step_end.clamp(max=last_col).unsqueeze(1)
                logits = logits_full.gather(1, gather_idx).squeeze(1)

            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu())
            all_phat.append(phat)
            if (s // args.mb) % 25 == 0:
                print(f"[eval] {s}/{N} done", flush=True)

    all_preds_t = torch.cat(all_preds)
    all_phat_t = torch.cat(all_phat).float()

    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    rho, _ = spearmanr(all_preds_t.numpy(), all_phat_t.numpy())
    bin_label = (all_phat_t > 0.5).long().numpy()
    if bin_label.min() != bin_label.max():
        auc = float(roc_auc_score(bin_label, all_preds_t.numpy()))
    else:
        auc = float("nan")
    bce = float(F.binary_cross_entropy(all_preds_t, all_phat_t).item())
    mse = float(((all_preds_t - all_phat_t) ** 2).mean().item())

    result = {
        "ckpt": args.ckpt,
        "data": args.data,
        "architecture": arch,
        "n_eval": int(len(all_preds_t)),
        "spearman_rho": float(rho) if rho == rho else None,
        "within_auc": auc,
        "val_loss_bce": bce,
        "val_mse": mse,
        "pred_mean": float(all_preds_t.mean().item()),
        "pred_std": float(all_preds_t.std().item()),
    }
    print("[eval] RESULT:")
    print(json.dumps(result, indent=2))
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
