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

    # Per-shard prompt/response widths can differ because each shard pads
    # to its own slice-max rather than cfg.max_prompt_len. Pad all shards
    # to the global max (left-pad for prompts, right-pad for responses)
    # before concatenating along dim 0.
    def _left_pad_to(t: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
        if t.shape[1] == target_len:
            return t
        pad_w = target_len - t.shape[1]
        if pad_w < 0:
            raise SystemExit(f"target_len={target_len} < t.shape[1]={t.shape[1]}")
        pad = torch.full(
            (t.shape[0], pad_w), pad_value, dtype=t.dtype, device=t.device,
        )
        return torch.cat([pad, t], dim=1)

    def _right_pad_to(t: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
        if t.shape[1] == target_len:
            return t
        pad_w = target_len - t.shape[1]
        if pad_w < 0:
            raise SystemExit(f"target_len={target_len} < t.shape[1]={t.shape[1]}")
        pad = torch.full(
            (t.shape[0], pad_w), pad_value, dtype=t.dtype, device=t.device,
        )
        return torch.cat([t, pad], dim=1)

    pmax = max(b["prompt_ids"].shape[1] for b in blobs)
    rmax = max(b["response_ids"].shape[1] for b in blobs)
    print(f"[merge] padding all shards to prompt_len={pmax}, resp_len={rmax}",
          flush=True)

    # Pad token id: use first shard's left-pad value (which collect uses) —
    # for prompt_ids the typical pad is the EOS token (since pad_token = eos
    # in the tokenizer). For mask we pad with 0 (= masked-out).
    first = blobs[0]
    prompt_pad_id = int(first["prompt_ids"][0, 0].item())  # left-pad value used
    response_pad_id = 0  # right-pad responses with 0 (masked anyway)

    merged = {
        "prompt_ids": torch.cat([
            _left_pad_to(b["prompt_ids"], pmax, prompt_pad_id) for b in blobs
        ], dim=0),
        "prompt_mask": torch.cat([
            _left_pad_to(b["prompt_mask"], pmax, 0) for b in blobs
        ], dim=0),
        "response_ids": torch.cat([
            _right_pad_to(b["response_ids"], rmax, response_pad_id) for b in blobs
        ], dim=0),
        "response_mask": torch.cat([
            _right_pad_to(b["response_mask"], rmax, 0) for b in blobs
        ], dim=0),
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
