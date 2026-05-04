"""Merge MC step-label shards from mc_step_label.py."""
import argparse, os
import torch


REQUIRED_KEYS = ("prompt_ids", "prompt_mask", "response_ids", "response_mask",
                 "step_end_idx", "p_hat", "outcomes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if not args.inputs:
        raise SystemExit("[merge] ERROR: --inputs is empty")

    shards = []
    P_max = R_max = 0
    for f in args.inputs:
        if not os.path.isfile(f):
            raise SystemExit(f"[merge] ERROR: shard not found: {f}")
        b = torch.load(f, map_location="cpu", weights_only=False)
        missing = [k for k in REQUIRED_KEYS if k not in b]
        if missing:
            raise SystemExit(f"[merge] ERROR: shard {f} missing keys {missing}")
        n_rows = int(b["p_hat"].shape[0])
        if n_rows == 0:
            print(f"[merge] WARN: shard {f} has 0 rows — skipping", flush=True)
            continue
        # sanity: per-shard row count is consistent across all per-row tensors
        for k in ("prompt_ids", "response_ids", "step_end_idx", "outcomes"):
            if int(b[k].shape[0]) != n_rows:
                raise SystemExit(f"[merge] ERROR: shard {f} key {k} row mismatch "
                                 f"({b[k].shape[0]} vs p_hat {n_rows})")
        shards.append(b)
        P_max = max(P_max, int(b["prompt_ids"].shape[1]))
        R_max = max(R_max, int(b["response_ids"].shape[1]))
        print(f"[merge] {f}: n={n_rows} P={b['prompt_ids'].shape[1]} R={b['response_ids'].shape[1]}", flush=True)

    if not shards:
        raise SystemExit("[merge] ERROR: no non-empty shards to merge")

    n_total = sum(int(s["p_hat"].shape[0]) for s in shards)
    print(f"[merge] total n={n_total}, padding to P={P_max} R={R_max}", flush=True)

    def pad_dim1(t, target, fill=0):
        if t.shape[1] >= target:
            return t[:, :target]
        pad = target - t.shape[1]
        return torch.nn.functional.pad(t, (0, pad), value=fill)

    out = {}
    for k in ["prompt_ids", "prompt_mask"]:
        out[k] = torch.cat([pad_dim1(s[k], P_max) for s in shards], dim=0)
    for k in ["response_ids", "response_mask"]:
        out[k] = torch.cat([pad_dim1(s[k], R_max) for s in shards], dim=0)
    for k in ["step_end_idx", "p_hat", "outcomes"]:
        out[k] = torch.cat([s[k] for s in shards], dim=0)
    out["config"] = shards[0].get("config", {})
    out["config"]["n_shards_merged"] = len(shards)

    print(f"[merge] saving to {args.output}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(out, args.output)
    print(f"[merge] DONE n={n_total} P={P_max} R={R_max}", flush=True)
    print(f"[merge] p_hat: mean={float(out['p_hat'].mean()):.3f} std={float(out['p_hat'].std()):.3f}", flush=True)


if __name__ == "__main__":
    main()
