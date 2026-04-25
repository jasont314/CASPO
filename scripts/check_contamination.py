"""Check for problem-text overlap between the train set (DeepScaleR by default)
and each eval benchmark (MATH-500, AIME-24, AIME-25, AMC-23).

Two matching modes:
  - exact: byte-equal after whitespace+unicode normalization
  - fuzzy: SequenceMatcher ratio >= --fuzzy-threshold (default 0.95)

Exit codes:
  0  no leaks found, at least one eval set checked
  1  one or more leaks found (exact or fuzzy)
  2  all eval sets failed to load (nothing was actually checked)

Usage::

    python scripts/check_contamination.py
    python scripts/check_contamination.py --fuzzy --fuzzy-threshold 0.92
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, List, Set, Tuple


_WS = re.compile(r"\s+")
# Match common LaTeX dollar delimiters/escapes that vary across datasets.
_DOLLAR_ESCAPE = re.compile(r"\\\$")


def normalize(s: str) -> str:
    """Unicode-normalize, lowercase, collapse whitespace, strip.

    HF datasets vary on:
      - NBSP (U+00A0) vs regular space
      - curly quotes vs straight quotes
      - full-width punctuation (rare but present in mixed-language sets)
      - escaped vs unescaped LaTeX dollar signs (\\$ vs $)
    NFKC compatibility-decomposes these to canonical forms.
    """
    if not s:
        return ""
    # NFKC: collapses NBSP→space, fullwidth→ASCII, ligatures, etc.
    s = unicodedata.normalize("NFKC", s)
    # Treat \$ and $ as equivalent (some math datasets escape, some don't).
    s = _DOLLAR_ESCAPE.sub("$", s)
    return _WS.sub(" ", s.strip().lower())


def _coerce_to_str(v) -> str:
    """Best-effort extract a string problem from common HF column shapes."""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        # Some datasets nest under {"text": ...} or {"problem": ...}.
        for k in ("problem", "question", "text", "content"):
            inner = v.get(k)
            if isinstance(inner, str):
                return inner
        return ""
    if isinstance(v, list) and v and isinstance(v[0], str):
        return "\n".join(x for x in v if isinstance(x, str))
    return ""


def load_dataset_problems(name: str, split: str, field: str) -> List[str]:
    from datasets import load_dataset  # type: ignore
    ds = load_dataset(name, split=split)
    # Validate the field exists once, before iterating, so typos fail loudly.
    cols = getattr(ds, "column_names", None)
    if cols is not None and field not in cols:
        raise KeyError(
            f"field {field!r} not in dataset columns {cols!r} for {name}"
        )
    out: List[str] = []
    for row in ds:
        v = _coerce_to_str(row.get(field))
        if v and v.strip():
            out.append(v)
    return out


def find_exact_overlap(train_norm: Set[str], eval_set: List[str]) -> List[int]:
    return [i for i, p in enumerate(eval_set) if normalize(p) in train_norm]


def find_fuzzy_overlap(
    train_norm: List[str],
    eval_set: List[str],
    threshold: float,
    prefix_len: int = 30,
    prefix_window: int = 200,
) -> List[Tuple[int, int, float]]:
    """Return (eval_idx, train_idx, ratio) triples above the threshold.

    O(N*M) in the worst case so only run on small eval sets (<1K problems).

    Pruning strategy (two-stage):
      1. quick_ratio() upper-bounds ratio() — used to skip cheaply.
      2. Optional prefix prune is conservative: only skip when the eval
         problem is long enough that a shared prefix is reliable. For short
         problems (which paraphrase wildly) we fall back to full pairwise.
    """
    eval_norm = [normalize(p) for p in eval_set]
    hits: List[Tuple[int, int, float]] = []

    for i, ep in enumerate(eval_norm):
        if not ep:
            continue
        # Only apply prefix prune when ep is comfortably longer than prefix_len.
        # Short problems get full pairwise: (a) cheap anyway, (b) prefix would
        # be a large fraction of the string and over-prune paraphrases.
        use_prefix = len(ep) >= prefix_len * 2
        prefix = ep[:prefix_len] if use_prefix else ""

        # quick_ratio threshold: SequenceMatcher caches autojunk/seq state, so
        # build the matcher once per eval problem and reuse via set_seq2.
        matcher = SequenceMatcher(autojunk=False)
        matcher.set_seq1(ep)

        for j, tp in enumerate(train_norm):
            if not tp:
                continue
            if use_prefix and prefix not in tp[:prefix_window]:
                continue
            matcher.set_seq2(tp)
            # quick_ratio is an upper bound on ratio; skip if it can't clear bar.
            if matcher.quick_ratio() < threshold:
                continue
            # real_quick_ratio is also an upper bound, even tighter.
            if matcher.real_quick_ratio() < threshold:
                continue
            r = matcher.ratio()
            if r >= threshold:
                hits.append((i, j, r))
                break
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-name", default="agentica-org/DeepScaleR-Preview-Dataset")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--train-field", default="problem")
    ap.add_argument("--fuzzy", action="store_true",
                    help="also run fuzzy matching (slower, more expensive)")
    ap.add_argument("--fuzzy-threshold", type=float, default=0.95)
    ap.add_argument("--strict", action="store_true",
                    help="exit 2 if any eval set fails to load (default: only "
                         "exit 2 if all eval sets fail)")
    args = ap.parse_args()

    if not 0.0 < args.fuzzy_threshold <= 1.0:
        print(f"--fuzzy-threshold must be in (0, 1], got {args.fuzzy_threshold}",
              file=sys.stderr)
        return 2

    print(f"Loading train set: {args.train_name} (split={args.train_split})", flush=True)
    try:
        train = load_dataset_problems(args.train_name, args.train_split, args.train_field)
    except Exception as e:
        print(f"FATAL: could not load train set {args.train_name}: {e}",
              file=sys.stderr, flush=True)
        return 2
    print(f"  -> {len(train)} problems\n", flush=True)
    if not train:
        print("FATAL: train set is empty after normalization", file=sys.stderr)
        return 2

    # Pre-normalize train once and share between exact + fuzzy passes.
    train_norm_list = [normalize(t) for t in train]
    train_norm_set: Set[str] = set(train_norm_list)

    eval_sets = [
        ("MATH-500",   "HuggingFaceH4/MATH-500",        "test",  "problem"),
        ("AIME-24",    "HuggingFaceH4/aime_2024",       "train", "problem"),
        ("AIME-25",    "MathArena/aime_2025_I",         "train", "problem"),
        ("AMC-23",     "AI-MO/aimo-validation-amc",     "train", "problem"),
    ]

    summary: Dict[str, Dict[str, int]] = {}
    n_failed = 0
    for label, name, split, field in eval_sets:
        try:
            ev = load_dataset_problems(name, split, field)
        except Exception as e:
            print(f"[{label}] could not load ({name}): {e}",
                  file=sys.stderr, flush=True)
            n_failed += 1
            if args.strict:
                return 2
            continue
        n_eval = len(ev)
        if n_eval == 0:
            print(f"[{label}] empty after load — skipping", flush=True)
            continue
        exact_idx = find_exact_overlap(train_norm_set, ev)
        n_exact = len(exact_idx)
        print(f"[{label}] {n_eval} problems, exact-match leak: {n_exact}/{n_eval} "
              f"({100*n_exact/max(n_eval,1):.1f}%)", flush=True)
        if exact_idx:
            for k, i in enumerate(exact_idx[:3]):
                snippet = ev[i].replace("\n", " ")[:120]
                print(f"    leak example #{k+1}: {snippet}...", flush=True)
        n_fuzzy = 0
        if args.fuzzy:
            fuzzy = find_fuzzy_overlap(train_norm_list, ev, args.fuzzy_threshold)
            n_fuzzy = len(fuzzy)
            print(f"[{label}] fuzzy(@{args.fuzzy_threshold}): {n_fuzzy}/{n_eval} "
                  f"({100*n_fuzzy/max(n_eval,1):.1f}%)", flush=True)
        summary[label] = {"n_eval": n_eval, "exact": n_exact, "fuzzy": n_fuzzy}

    if n_failed == len(eval_sets):
        print("\nFATAL: every eval set failed to load", file=sys.stderr)
        return 2

    print("\n=== summary ===")
    total_leaks = 0
    for k, v in summary.items():
        leak_pct = 100 * v["exact"] / max(v["n_eval"], 1)
        total_leaks += v["exact"] + v["fuzzy"]
        print(f"{k:10s}  n={v['n_eval']:4d}  exact={v['exact']:3d} ({leak_pct:.1f}%)"
              + (f"  fuzzy={v['fuzzy']}" if args.fuzzy else ""))

    return 1 if total_leaks > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
