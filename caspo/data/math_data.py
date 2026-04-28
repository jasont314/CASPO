"""DeepScaleR-style dataset loader.

Yields ``{"prompt": str, "ground_truth": str, "raw_question": str}`` rows.
Handles field-name variation across math datasets:

* DeepScaleR: ``problem`` / ``answer`` (bare ground truth)
* MATH-500:   ``problem`` / ``solution`` (ground truth wrapped in ``\\boxed{}``)
* Generic:    ``question|problem|prompt`` and ``answer|solution|final_answer``.

When the answer comes from a ``solution``-style field, ``extract_boxed_answer``
is used to peel off the ``\\boxed{}`` wrapper.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

from caspo.config import CASPOConfig
from caspo.reward.math_verifier import extract_boxed_answer


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

_QUESTION_FIELDS = ("problem", "question", "prompt")
_ANSWER_FIELDS = ("answer", "final_answer", "solution")  # plain answers preferred
_TRAILING_PUNCT = ".;:"  # used by _peel_answer to strip GSM8K tail noise


def format_prompt(
    question: str,
    system_prompt: Optional[str] = None,
    tokenizer: Any = None,
    add_generation_prompt: bool = True,
    template: Optional[str] = None,
) -> str:
    """Render a prompt for a math problem.

    Args:
        question: the problem text.
        system_prompt: optional system message (only used by chat-template path).
        tokenizer: if given and exposes ``apply_chat_template``, that's used.
        add_generation_prompt: passed through to ``apply_chat_template``.
        template: an explicit f-style template overriding chat-template. Use
            this for paper-faithful reproductions (e.g. VinePPO's MATH prompt
            ``"[MATH_TASK] Problem:\\n{query}\\n\\nSolution:"``). If ``"{query}"``
            is missing the question is appended.
    """
    if template is not None:
        if "{query}" in template:
            # Use replace() rather than .format() so literal ``{`` / ``}``
            # elsewhere in the template (e.g. ``\boxed{}`` reminders, LaTeX
            # examples, JSON snippets) don't blow up with KeyError. We only
            # advertise a single ``{query}`` placeholder; any other braces
            # are user content, not format fields.
            return template.replace("{query}", question)
        return f"{template}\n{question}\n"
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception:
            # Tokenizer has no chat template configured — fall through.
            pass

    if system_prompt:
        return f"{system_prompt}\n\n{question}\n"
    return f"{question}\n"


# ---------------------------------------------------------------------------
# Field extraction
# ---------------------------------------------------------------------------

def _pick_question(row: dict) -> Optional[str]:
    for k in _QUESTION_FIELDS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _peel_answer(s: str) -> Optional[str]:
    """Extract a bare answer from one of the common math-dataset encodings.

    Handles, in priority order:
      * MATH/AMC ``\\boxed{...}`` — peel via :func:`extract_boxed_answer`
        (the *last* boxed expression wins, matching the grader's behavior).
      * GSM8K ``<reasoning>\\n#### N`` — peel after ``####`` (stripping any
        thousand separators and trailing punctuation).
      * bare answer strings — return whitespace-stripped.

    Returns ``None`` when nothing usable can be extracted (empty input,
    empty ``\\boxed{}``, ``####`` with no tail, etc.). This lets the
    caller drop the row instead of emitting a row with an empty or
    full-solution-text ground truth.
    """
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None

    # 1) Prefer \boxed{...} when present — it's the unambiguous final answer
    #    in MATH/AMC/NuminaMath/DeepScaleR/lighteval-style solutions, even if
    #    the surrounding text also happens to contain "####".
    boxed = extract_boxed_answer(s)
    if boxed is not None:
        boxed = boxed.strip()
        if boxed:
            return boxed
        # Empty \boxed{} → fall through to other heuristics.

    # 2) GSM8K-style "#### N" tail.
    if "####" in s:
        tail = s.rsplit("####", 1)[-1].strip()
        # Strip thousand separators (GSM8K writes "#### 1,234"), then trailing
        # punctuation (".;:" — single rstrip, faster than the per-char loop).
        tail = tail.replace(",", "").rstrip(_TRAILING_PUNCT).strip()
        if tail:
            return tail
        return None  # "####" with empty tail → unusable

    # 3) Bare answer (no LaTeX wrapper, no GSM8K marker). Return as-is.
    return s


def _pick_answer(row: dict) -> Optional[str]:
    """Pick a ground-truth answer from a raw row.

    Field priority:
      * ``answer`` / ``final_answer`` — usually already-bare in DeepScaleR,
        NuminaMath, GSM8K, MATH-500.
      * ``solution`` — full worked solution; we peel ``\\boxed{...}`` /
        ``####``. This is the only available field in lighteval/MATH.

    A row with an empty / unparseable answer returns ``None`` so the caller
    can drop it (rather than silently emitting a row whose ground truth is
    the entire solution text).
    """
    # Prefer plain answer fields first. GSM8K's "answer" is "<reasoning>\n#### N"
    # so we still need to peel.
    for k in ("answer", "final_answer"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            peeled = _peel_answer(v)
            if peeled:
                return peeled
            # If the bare-answer field had a "####" with no tail or an empty
            # \boxed{}, fall through to ``solution`` rather than emit garbage.

    # Solution-style field (full worked text, MATH-style).
    sol = row.get("solution")
    if isinstance(sol, str) and sol.strip():
        peeled = _peel_answer(sol)
        if peeled:
            return peeled

    return None


def _row_to_example(
    row: dict,
    cfg: CASPOConfig,
    tokenizer: Any,
) -> Optional[dict]:
    q = _pick_question(row)
    a = _pick_answer(row)
    if not q or not a:
        return None
    template = getattr(cfg, "prompt_template", None)
    prompt = format_prompt(
        q,
        system_prompt=cfg.system_prompt,
        tokenizer=tokenizer,
        template=template,
    )
    return {"prompt": prompt, "ground_truth": a, "raw_question": q}


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def _build_from_rows(
    rows: Iterable[dict],
    cfg: CASPOConfig,
    tokenizer: Any,
) -> List[dict]:
    """Map raw rows → standardized examples, optionally filtering by length.

    Tokenization for the length filter is batched: per-prompt
    ``tokenizer(text)`` calls dominate dataset-build time on big corpora
    (DeepScaleR ~40k rows). We collect the rendered examples first and run
    one ``batch_encode_plus`` (or fallback ``__call__``) at the end.

    When ``cfg.filter_eval_leakage`` is True (default), drops any row whose
    normalized problem text matches an eval-set problem (MATH-500, GSM8K,
    AIME-2025, OlympiadBench). See ``caspo/data/eval_leak.py``.
    """
    max_len = int(cfg.max_prompt_len)

    eval_hashes = None
    is_eval_leak = None
    if bool(getattr(cfg, "filter_eval_leakage", True)):
        try:
            from caspo.data.eval_leak import (
                get_eval_leak_hashes,
                is_eval_leak as _is_eval_leak,
            )
            eval_hashes = get_eval_leak_hashes()
            is_eval_leak = _is_eval_leak
        except Exception as _e:
            print(f"[data] WARN: eval-leak filter init failed: {_e}", flush=True)
            eval_hashes = None
            is_eval_leak = None

    # Materialize all valid examples first (cheap: pure string ops).
    examples: List[dict] = []
    n_dropped_leak = 0
    for row in rows:
        ex = _row_to_example(row, cfg, tokenizer)
        if ex is None:
            continue
        if eval_hashes is not None and is_eval_leak is not None:
            if is_eval_leak(ex["raw_question"], eval_hashes):
                n_dropped_leak += 1
                continue
        examples.append(ex)
    if n_dropped_leak > 0:
        print(f"[data] dropped {n_dropped_leak} rows with eval-set leakage "
              f"(MATH-500/GSM8K/AIME-2025/OlympiadBench)", flush=True)

    # Length filter is optional (no tokenizer or max_len<=0 → keep all).
    if tokenizer is None or max_len <= 0 or not examples:
        return examples

    prompts = [ex["prompt"] for ex in examples]
    # Prefer batch_encode_plus when present (HF fast tokenizers parallelize
    # via Rust). Fall back to __call__ on the list (transformers PreTrained
    # tokenizers accept lists too) and finally to a per-row loop.
    lengths: Optional[List[int]] = None
    try:
        if hasattr(tokenizer, "batch_encode_plus"):
            enc = tokenizer.batch_encode_plus(
                prompts, add_special_tokens=False
            )
        else:
            enc = tokenizer(prompts, add_special_tokens=False)
        lengths = [len(ids) for ids in enc["input_ids"]]
    except Exception:
        # Fallback: per-row encode (matches old behavior).
        tok_call = tokenizer  # bind locally to avoid attr lookup in loop
        lengths = []
        for p in prompts:
            try:
                lengths.append(len(tok_call(p, add_special_tokens=False).input_ids))
            except Exception:
                lengths.append(-1)  # sentinel: keep row

    out: List[dict] = []
    for ex, L in zip(examples, lengths):
        if L < 0 or L <= max_len:
            out.append(ex)
    return out


def _load_hf(name: str, split: str):
    """Lazy import of ``datasets``; returns the raw HF Dataset."""
    from datasets import load_dataset  # type: ignore
    return load_dataset(name, split=split)


def _from_iterable_factory(
    rows: Iterable[dict],
    cfg: CASPOConfig,
    tokenizer: Any,
) -> List[dict]:
    """Public-ish escape hatch for tests: build a standardized list of
    examples from any iterable of raw rows, without touching HF/datasets.
    Re-iterable (returns a list).
    """
    return _build_from_rows(rows, cfg, tokenizer)


def load_train_dataset(cfg: CASPOConfig, tokenizer: Any = None) -> Iterable[dict]:
    """Load and standardize the training dataset.

    Returns an iterable that can be iterated multiple times. When ``tokenizer``
    is provided, examples whose tokenized prompt exceeds ``cfg.max_prompt_len``
    are dropped.
    """
    raw = _load_hf(cfg.dataset_name, cfg.dataset_split)
    return _build_from_rows(raw, cfg, tokenizer)


def load_eval_dataset(cfg: CASPOConfig, tokenizer: Any = None) -> Iterable[dict]:
    """Load and standardize the eval dataset (uses ``cfg.eval_dataset_name``)."""
    raw = _load_hf(cfg.eval_dataset_name, cfg.eval_split)
    return _build_from_rows(raw, cfg, tokenizer)
