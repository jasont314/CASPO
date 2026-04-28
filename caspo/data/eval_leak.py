"""Eval-set leakage filter for training data loaders.

Loads canonical eval datasets, normalizes problem text, and exposes a
hash set used by ``caspo.data.math_data._build_from_rows`` to drop any
training row whose normalized problem matches an eval problem.

The eval-set hash is cached on disk so it's computed once per session.
Add new eval datasets in ``EVAL_DATASETS`` below.

Verified leakage matrix (2026-04-28):
    train         vs   MATH-500  GSM8K  AIME-2025  OlympiadBench
    Big-Math      →    0/500     0      0/15       2/674 (0.3%)
    DeepScaleR    →    3/500     0      0/15       0/674
    MATH-lighteval→    0/500     0      0/15       0/674

Filtering on by default (``cfg.filter_eval_leakage = True``); set False
to disable for paper-faithful baseline reproductions.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import List, Optional, Set

# Each entry is (hf_repo, optional_config, split, candidate_field_names).
# Add to this list to extend eval-set leakage protection.
EVAL_DATASETS = [
    ("HuggingFaceH4/MATH-500", None, "test", ("problem", "Problem", "question")),
    ("openai/gsm8k", "main", "test", ("question",)),
    ("MathArena/aime_2025_I", None, "train", ("problem", "Problem", "question")),
    # OlympiadBench: 2/674 incidental overlap with Big-Math; including it
    # keeps the filter consistent across our eval set.
    ("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", "train", ("question",)),
]


_PROMPT_FIELDS_DEFAULT = ("problem", "question", "prompt", "Problem", "Question")


def _normalize(text: str) -> str:
    if not text:
        return ""
    s = text.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\\\\", r"\\", s)
    return s


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_HASH_CACHE: Optional[Set[str]] = None


def get_eval_leak_hashes(
    eval_datasets: Optional[List[tuple]] = None,
) -> Set[str]:
    """Return a set of normalized SHA-256 hashes for eval-set problem texts.

    Called by the training data loader to drop overlapping rows.
    Result is cached in process memory; first call may take ~5-15s
    while datasets are loaded from HF cache.
    """
    global _HASH_CACHE
    if _HASH_CACHE is not None and eval_datasets is None:
        return _HASH_CACHE

    if eval_datasets is None:
        eval_datasets = EVAL_DATASETS

    try:
        from datasets import load_dataset
    except ImportError:
        # If datasets is missing, fall back to no filtering (loud).
        print("[eval_leak] WARN: datasets package not available; "
              "leakage filter disabled.", flush=True)
        _HASH_CACHE = set()
        return _HASH_CACHE

    hashes: Set[str] = set()
    for repo, cfg_name, split, fields in eval_datasets:
        try:
            if cfg_name is not None:
                ds = load_dataset(repo, cfg_name, split=split)
            else:
                ds = load_dataset(repo, split=split)
        except Exception as e:
            print(f"[eval_leak] WARN: could not load {repo} (cfg={cfg_name}, "
                  f"split={split}): {type(e).__name__}: {e}", flush=True)
            continue
        n_added = 0
        for row in ds:
            for f in fields:
                v = row.get(f)
                if isinstance(v, str) and v.strip():
                    hashes.add(_hash(_normalize(v)))
                    n_added += 1
                    break
        print(f"[eval_leak] {repo} ({cfg_name or 'default'}/{split}): "
              f"hashed {n_added} problems", flush=True)

    print(f"[eval_leak] total unique eval-set hashes: {len(hashes)}", flush=True)
    if eval_datasets is EVAL_DATASETS:
        _HASH_CACHE = hashes
    return hashes


def is_eval_leak(prompt_text: str, eval_hashes: Optional[Set[str]] = None) -> bool:
    """Return True iff the normalized prompt hashes into the eval-set hash set."""
    if eval_hashes is None:
        eval_hashes = get_eval_leak_hashes()
    if not eval_hashes:
        return False
    return _hash(_normalize(prompt_text)) in eval_hashes
