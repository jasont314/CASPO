"""Microbench: segment_responses_batch_latex_aware vs segment_responses_batch.

Generates synthetic math-CoT-like responses (Step 1 ... \\(x^2\\) ... \\boxed{42}.),
tokenizes them with a real fast tokenizer, pads to a uniform [B, R] tensor, and
times each segmenter at 100 / 500 / 1000-token response lengths.

Usage:
    python scripts/bench_segmentation.py [--n 100] [--max-tokens 500]

Both --n and --max-tokens are upper bounds on the *batch size* and *single
response length* respectively; the script always sweeps {100, 500, 1000} for
the per-response token cap and picks min(--max-tokens, that cap) at each step.
"""

from __future__ import annotations

import argparse
import random
import time
from typing import List

import torch


# ---------------------------------------------------------------------------
# Synthetic CoT generation
# ---------------------------------------------------------------------------

_VARS = ["x", "y", "z", "a", "b", "n", "k"]
_OPS = ["+", "-", "\\cdot", "/"]


def _step_sentence(i: int, rng: random.Random) -> str:
    """One math-CoT-like step. Includes a mix of LaTeX inline math, numbers,
    and prose so the LaTeX-aware splitter actually has work to do."""
    v1 = rng.choice(_VARS)
    v2 = rng.choice(_VARS)
    op = rng.choice(_OPS)
    n1 = rng.randint(1, 99)
    n2 = rng.randint(1, 99)
    templates = [
        f"Step {i}: We start by rewriting \\({v1}^2 {op} {v2}\\) as \\({n1} {op} {n2}\\).",
        f"Step {i}: Substituting gives \\({v1} = {n1}\\), so \\({v2} {op} {v1} = {n2}\\).",
        f"Step {i}: Therefore the expression simplifies to \\(\\frac{{{n1}}}{{{n2}}}\\).",
        f"Step {i}: By the identity \\(({v1}+{v2})^2 = {v1}^2 + 2{v1}{v2} + {v2}^2\\) we get {n1}.",
        f"Step {i}: Adding both sides, {n1} + {n2} = {n1 + n2}.",
    ]
    return rng.choice(templates)


def make_synthetic_cot(rng: random.Random, n_steps: int = 6) -> str:
    body = "\n".join(_step_sentence(i + 1, rng) for i in range(n_steps))
    final = rng.randint(1, 999)
    return body + f"\nThus the final answer is \\(\\boxed{{{final}}}\\)."


# ---------------------------------------------------------------------------
# Tokenization → padded [B, R] tensor
# ---------------------------------------------------------------------------

def build_batch(
    tokenizer,
    n: int,
    max_tokens: int,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ``n`` synthetic CoTs, tokenize, truncate to ``max_tokens``,
    pad to the longest in the batch, and return (response_ids, response_mask).

    We grow ``n_steps`` until the *median* tokenized length is roughly
    ``max_tokens``, so a 1000-token bench actually exercises 1000-token rows.
    """
    # Calibrate steps-per-response so median length ~= max_tokens.
    # 1 step ≈ 25-35 tokens with this template, so start with max_tokens // 28.
    n_steps = max(1, max_tokens // 28)

    texts: List[str] = [make_synthetic_cot(rng, n_steps=n_steps) for _ in range(n)]
    enc = tokenizer(
        texts,
        add_special_tokens=False,
        padding="longest",
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    return enc["input_ids"].to(torch.long), enc["attention_mask"].to(torch.long)


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_call(fn, *args, repeats: int = 3, warmup: int = 1, **kwargs) -> float:
    """Return median wall-clock seconds over ``repeats`` calls."""
    for _ in range(warmup):
        fn(*args, **kwargs)
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=100,
                   help="batch size (number of responses)")
    p.add_argument("--max-tokens", type=int, default=500,
                   help="upper bound on per-response length to sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tokenizer",
        type=str,
        default="hf-internal-testing/tiny-random-LlamaForCausalLM",
        help="HF tokenizer id; default is the tiny-random Llama used in tests",
    )
    args = p.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    print(f"[bench_segmentation] loading tokenizer: {args.tokenizer}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "</s>"
    if not getattr(tok, "is_fast", False):
        print("[bench_segmentation] WARNING: tokenizer is not a fast tokenizer")

    from caspo.segmentation import (
        segment_responses_batch,
        segment_responses_batch_latex_aware,
    )

    delim_ids = tok("\n", add_special_tokens=False).input_ids
    if not delim_ids or all(t == 0 for t in delim_ids):
        # Some tiny tokenizers emit only id 0 for "\n"; fall back to "Step".
        delim_ids = tok("Step", add_special_tokens=False).input_ids
    delim_ids = [int(t) for t in delim_ids]
    print(f"[bench_segmentation] newline delimiter ids = {delim_ids}")

    sweeps = [t for t in (100, 500, 1000) if t <= max(args.max_tokens, 100)]
    if not sweeps:
        sweeps = [args.max_tokens]

    print(
        f"\n{'tokens':>8} {'B':>6} {'R':>6} "
        f"{'latex_s':>10} {'latex_sps':>12} "
        f"{'newline_s':>11} {'newline_sps':>13} {'speedup':>9}"
    )
    print("-" * 80)

    for max_tok in sweeps:
        ids, mask = build_batch(tok, args.n, max_tok, rng)
        B, R = ids.shape
        valid = int(mask.sum().item()) // max(B, 1)

        t_latex = time_call(
            segment_responses_batch_latex_aware,
            ids, mask, tok,
            min_step_tokens=4, max_steps=64,
        )
        t_newline = time_call(
            segment_responses_batch,
            ids, mask, delim_ids,
            min_step_tokens=4, max_steps=64,
        )

        sps_latex = B / t_latex if t_latex > 0 else float("inf")
        sps_newline = B / t_newline if t_newline > 0 else float("inf")
        speedup = t_latex / t_newline if t_newline > 0 else float("inf")
        print(
            f"{max_tok:>8} {B:>6} {R:>6} "
            f"{t_latex:>10.4f} {sps_latex:>12.1f} "
            f"{t_newline:>11.4f} {sps_newline:>13.1f} "
            f"{speedup:>8.2f}x"
        )
        print(f"         (mean valid tokens/row ~= {valid})")

    print("\n[bench_segmentation] done.")


if __name__ == "__main__":
    main()
