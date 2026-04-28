"""Merge per-shard ``value_data.pt`` files produced by sharded
``collect_value_data.py`` runs into a single training file.

Usage::

    python scripts/merge_value_data_shards.py \\
        --inputs out/value_data_shard0.pt out/value_data_shard1.pt ... \\
        --output out/value_data.pt

Concatenates fixed-length tensors (left-padded prompts/responses share the
same max-length as long as all shards used the same config), and the
parallel Python lists/strings (raw_prompts, raw_responses, ground_truths).
Carries over a single ``config_snapshot`` from the first shard.
"""
from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="one or more shard .pt paths to merge")
    ap.add_argument("--output", type=str, required=True,
                    help="output path for the merged .pt file")
    args = ap.parse_args()

    if len(args.inputs) < 2:
        print(f"[merge] WARN: only {len(args.inputs)} input(s); will still "
              f"copy/rewrite to {args.output}.", flush=True)

    blobs = []
    for path in args.inputs:
        if not os.path.exists(path):
            raise SystemExit(f"missing input shard: {path}")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        n = blob["prompt_ids"].shape[0]
        print(f"[merge] {path}: n_rows={n}, "
              f"prompt_len={blob['prompt_ids'].shape[1]}, "
              f"resp_len={blob['response_ids'].shape[1]}", flush=True)
        blobs.append(blob)

    # Validate shape compatibility (left-padded fixed-length tensors).
    first = blobs[0]
    for b in blobs[1:]:
        for k in ("prompt_ids", "prompt_mask"):
            if b[k].shape[1] != first[k].shape[1]:
                raise SystemExit(
                    f"prompt-length mismatch on {k}: "
                    f"{b[k].shape[1]} vs {first[k].shape[1]}. All shards must "
                    f"use the same max_prompt_len."
                )
        for k in ("response_ids", "response_mask"):
            if b[k].shape[1] != first[k].shape[1]:
                raise SystemExit(
                    f"response-length mismatch on {k}: "
                    f"{b[k].shape[1]} vs {first[k].shape[1]}. All shards must "
                    f"use the same max_response_len."
                )

    merged = {
        "prompt_ids": torch.cat([b["prompt_ids"] for b in blobs], dim=0),
        "prompt_mask": torch.cat([b["prompt_mask"] for b in blobs], dim=0),
        "response_ids": torch.cat([b["response_ids"] for b in blobs], dim=0),
        "response_mask": torch.cat([b["response_mask"] for b in blobs], dim=0),
        "outcomes": torch.cat([b["outcomes"] for b in blobs], dim=0),
        "raw_prompts": [p for b in blobs for p in b["raw_prompts"]],
        "raw_responses": [r for b in blobs for r in b["raw_responses"]],
        "ground_truths": [g for b in blobs for g in b["ground_truths"]],
        "config_snapshot": first.get("config_snapshot", {}),
    }
    n_total = merged["prompt_ids"].shape[0]
    pos = float(merged["outcomes"].float().mean().item()) if n_total > 0 else 0.0
    print(f"[merge] merged n_rows={n_total}, positive_rate={pos:.3f}",
          flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(merged, args.output)
    print(f"[merge] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
