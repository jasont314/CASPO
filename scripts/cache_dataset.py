"""CLI to pre-build the tokenized prompt cache.

Run this once per (model, dataset, max_prompt_len) triple to amortize the
encode pass before launching a training job. Subsequent ``CachedTokenizedDataset``
constructions with the same key will mmap-load from disk.

Example
-------
    python scripts/cache_dataset.py \\
        --model deepseek-ai/deepseek-math-7b-base \\
        --dataset agentica-org/DeepScaleR-Preview-Dataset \\
        --split train \\
        --max-prompt-len 1024

Honors ``CASPO_TOKENIZED_CACHE_DIR`` for the output location, or pass
``--cache-dir`` explicitly. Prints the resolved cache path on success.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make the in-tree package importable when run from a fresh checkout.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from caspo.data.cached_dataset import (  # noqa: E402  (sys.path bootstrap above)
    CachedTokenizedDataset,
    cache_path_for,
)


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-tokenize a HuggingFace dataset into a CASPO cache file.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="HF tokenizer repo id or local path (used as both tokenizer and "
             "cache-key component).",
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="HF dataset name (e.g. agentica-org/DeepScaleR-Preview-Dataset).",
    )
    p.add_argument(
        "--split",
        default="train",
        help="HF split to load (default: train).",
    )
    p.add_argument(
        "--max-prompt-len",
        type=int,
        default=1024,
        help="Truncate / cache-key length cap (default: 1024).",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Override CASPO_TOKENIZED_CACHE_DIR / ~/.cache/caspo/tokenized.",
    )
    p.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-tokenization even if the cache already exists.",
    )
    p.add_argument(
        "--prompt-field",
        default="prompt",
        help="Which row field to read as the prompt text (default: prompt). "
             "Use 'problem' or 'question' for raw math datasets.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only tokenize the first N rows (smoke test).",
    )
    return p


def _load_rows(dataset_name: str, split: str, prompt_field: str, limit: int):
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset_name, split=split)
    n = len(ds) if limit <= 0 else min(limit, len(ds))
    for i in range(n):
        row = ds[i]
        text = row.get(prompt_field)
        if isinstance(text, str) and text.strip():
            yield {"prompt": text}


def _load_tokenizer(model_name_or_path: str):
    from transformers import AutoTokenizer  # type: ignore

    return AutoTokenizer.from_pretrained(model_name_or_path)


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)

    target = cache_path_for(
        args.model,
        args.dataset,
        args.max_prompt_len,
        cache_dir=args.cache_dir,
    )
    print(f"[cache_dataset] target: {target}")
    if target.exists() and not args.rebuild:
        print(f"[cache_dataset] cache already present (use --rebuild to overwrite)")
        return 0

    t0 = time.time()
    print(f"[cache_dataset] loading tokenizer: {args.model}")
    tokenizer = _load_tokenizer(args.model)

    print(f"[cache_dataset] streaming dataset: {args.dataset}#{args.split}")
    rows = list(
        _load_rows(args.dataset, args.split, args.prompt_field, args.limit)
    )
    print(f"[cache_dataset] tokenizing {len(rows)} rows (max_len={args.max_prompt_len})")

    ds = CachedTokenizedDataset(
        rows,
        tokenizer,
        model_name_or_path=args.model,
        dataset_name=args.dataset,
        max_prompt_len=args.max_prompt_len,
        cache_dir=args.cache_dir,
        rebuild=args.rebuild,
    )

    dt = time.time() - t0
    n, L = ds.input_ids.shape if ds.input_ids.ndim == 2 else (0, 0)
    size_mb = Path(ds.cache_path).stat().st_size / (1024 * 1024)
    print(
        f"[cache_dataset] done: n={n} L={L} "
        f"hash={ds.cache_hash} size={size_mb:.1f}MB took={dt:.1f}s"
    )
    print(f"[cache_dataset] wrote: {ds.cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
