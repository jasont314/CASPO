"""Compute ROC-AUC for V_φ on its validation set.

For each held-out (prompt, response, outcome) triple in value_data.pt, run
V_φ forward, take the cumulative-log-ratio at the LAST response token,
and compute AUC over the full set: P(score for outcome=1 > score for
outcome=0).

Reports for both old (Apr 25) and new (Apr 28 v2) V_φ on the *new*
value_data.pt — this is the right comparison because the new dataset
has the current-verifier outcomes, which is what V_φ should be
calibrated to at runtime.
"""
import sys, time
sys.path.insert(0, "/home/jason/experiment/CASPO")

def main(vphi_path: str, label: str, data_path: str = None):
    import torch
    from caspo.config import CASPOConfig
    from caspo.value.prefix_value import PrefixValueModel

    if data_path is None:
        data_path = "/mnt/nvme_tmp4/jason_caspo/caspo_rho1b_math_v2/value_data.pt"

    print(f"[{label}] loading V_φ from {vphi_path}", flush=True)
    cfg = CASPOConfig.from_yaml("/home/jason/experiment/CASPO/configs/caspo_rho1b_math.yaml")
    cfg.prefix_value_path = vphi_path
    cfg.torch_dtype = "bfloat16"
    cfg.attn_implementation = None  # avoid FA3 dtype issue at eval

    # Use from_pretrained so phi is LOADED from checkpoint, not re-initialized
    # from SFT (the bug in the previous version: phi=ref → log_ratio=0 → AUC=0.5).
    pv = PrefixValueModel.from_pretrained(cfg, vphi_path)
    pv.phi = pv.phi.to("cuda").eval()
    pv.ref = pv.ref.to("cuda").eval()

    print(f"[{label}] loading data from {data_path}", flush=True)
    blob = torch.load(data_path, map_location="cpu", weights_only=False)
    n = blob["prompt_ids"].shape[0]
    print(f"[{label}] dataset size: n={n}", flush=True)

    # Use the EXACT prompt-level val split that scripts/train_value.py uses
    # via the shared _split_train_val function. This ensures the AUC is on
    # rows the trainer held out (no train-set leak). Trailing-row "last 10%"
    # would not match what the trainer split (which seed-shuffles by prompt).
    import sys as _sys
    _sys.path.insert(0, "/home/jason/experiment/CASPO/scripts")
    from train_value import _split_train_val
    G = int(cfg.group_size)
    prompt_idx_per_row = None
    if n % G != 0:
        # multi-pair / paper-pairing data: derive prompt ids from prompt tokens
        _, _inv = torch.unique(blob["prompt_ids"], dim=0, return_inverse=True)
        prompt_idx_per_row = _inv.cpu().tolist()
        print(f"[{label}] n_rows={n} not divisible by G={G}; deriving prompt ids from prompt_ids tensor", flush=True)
    _, val_idxs = _split_train_val(
        n_rows=n,
        group_size=G,
        val_fraction=float(cfg.value_val_fraction),
        seed=int(cfg.seed),
        prompt_idx_per_row=prompt_idx_per_row,
    )
    idxs = val_idxs
    val_n = len(idxs)
    print(f"[{label}] held-out val: {val_n} rollouts "
          f"({val_n // int(cfg.group_size)} prompts) "
          f"via _split_train_val(seed={cfg.seed}, val_fraction={cfg.value_val_fraction})",
          flush=True)

    pids = blob["prompt_ids"][idxs].to("cuda")
    pmask = blob["prompt_mask"][idxs].to("cuda")
    rids = blob["response_ids"][idxs].to("cuda")
    rmask = blob["response_mask"][idxs].to("cuda")
    outcomes = blob["outcomes"][idxs].cpu().numpy()
    print(f"[{label}] positive rate in val set: {outcomes.mean():.3f}", flush=True)

    # Score each example: V_φ at the LAST response token of each row.
    # Run in microbatches.
    mb = 8
    last_v_list = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, val_n, mb):
            stop = min(start + mb, val_n)
            out = pv(
                prompt_ids=pids[start:stop],
                prompt_mask=pmask[start:stop],
                response_ids=rids[start:stop],
                response_mask=rmask[start:stop],
            )
            # out["V"] = [B, R] cumulative log-ratio over response tokens
            V = out["V"]  # [B, R]
            # Find each row's last valid response token: last index where rmask=1
            row_lens = rmask[start:stop].sum(dim=1).long().clamp(min=1)
            last_idx = (row_lens - 1)
            last_v = V.gather(1, last_idx.unsqueeze(1)).squeeze(1).float().cpu().numpy()
            last_v_list.append(last_v)
            if start % (mb * 10) == 0:
                print(f"  scored {stop}/{val_n} in {time.time()-t0:.1f}s", flush=True)
    import numpy as np
    scores = np.concatenate(last_v_list)
    print(f"[{label}] scored {len(scores)} prefixes in {time.time()-t0:.1f}s", flush=True)
    print(f"[{label}] score stats: mean={scores.mean():.3f} std={scores.std():.3f}", flush=True)
    print(f"[{label}] mean(score | y=1) = {scores[outcomes>0.5].mean():.3f}", flush=True)
    print(f"[{label}] mean(score | y=0) = {scores[outcomes<0.5].mean():.3f}", flush=True)

    # ROC-AUC
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(outcomes, scores)
    ap = average_precision_score(outcomes, scores)
    # accuracy at sign threshold
    acc = ((scores > 0).astype(np.float32) == outcomes).mean()
    print(f"[{label}] ROC-AUC = {auc:.4f}", flush=True)
    print(f"[{label}] avg-precision (PR-AUC) = {ap:.4f}", flush=True)
    print(f"[{label}] sign-acc = {acc:.4f}", flush=True)
    return auc, ap, acc, len(scores), outcomes.mean()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vphi", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    main(args.vphi, args.label, args.data)
