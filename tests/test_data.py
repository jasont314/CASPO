"""Tests for ``caspo.data.math_data``.

Live HuggingFace downloads are skipped — we exercise the formatting and
field-extraction logic against in-memory rows.
"""

from __future__ import annotations

import pytest

from caspo.config import CASPOConfig
from caspo.data.math_data import (
    _from_iterable_factory,
    format_prompt,
    load_eval_dataset,
    load_train_dataset,
)


# ---------------------------------------------------------------------------
# format_prompt
# ---------------------------------------------------------------------------

def test_format_prompt_plain_no_tokenizer():
    out = format_prompt("What is 2+2?")
    assert "What is 2+2?" in out


def test_format_prompt_plain_with_system():
    out = format_prompt("solve x", system_prompt="You are a math tutor.")
    assert "You are a math tutor." in out
    assert "solve x" in out


class _FakeTokenizer:
    """Minimal fake tokenizer with chat template + tokenization for tests."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = []
        for m in messages:
            parts.append(f"<{m['role']}>{m['content']}</{m['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, text, add_special_tokens=False):
        # ~1 token per whitespace-split word.
        toks = text.split()

        class _Out:
            input_ids = toks

        return _Out()


def test_format_prompt_uses_chat_template_when_available():
    tok = _FakeTokenizer()
    out = format_prompt(
        "What is 2+2?", system_prompt="be terse", tokenizer=tok
    )
    assert "<system>be terse</system>" in out
    assert "<user>What is 2+2?</user>" in out
    assert out.endswith("<assistant>")


def test_format_prompt_chat_template_no_system():
    tok = _FakeTokenizer()
    out = format_prompt("What is 2+2?", tokenizer=tok)
    assert "<system>" not in out
    assert "<user>What is 2+2?</user>" in out


class _NoChatTemplateTokenizer:
    """Tokenizer that has ``apply_chat_template`` but it raises (no template
    configured) — exercises the except-branch fallback in ``format_prompt``."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        # Mirrors HF behavior when ``tokenizer.chat_template`` is None.
        raise ValueError(
            "Cannot use apply_chat_template() because tokenizer.chat_template is not set"
        )

    def __call__(self, text, add_special_tokens=False):
        toks = text.split()

        class _Out:
            input_ids = toks

        return _Out()


def test_format_prompt_falls_back_when_chat_template_raises():
    """Tokenizer with no chat template configured → fall through to plain
    string formatting instead of propagating the exception."""
    tok = _NoChatTemplateTokenizer()
    out = format_prompt("What is 2+2?", system_prompt="be terse", tokenizer=tok)
    # Should look like the no-tokenizer path, not raise.
    assert "be terse" in out
    assert "What is 2+2?" in out
    # And critically, no chat-template angle-bracket wrappers.
    assert "<user>" not in out
    assert "<system>" not in out


def test_format_prompt_falls_back_when_chat_template_raises_no_system():
    tok = _NoChatTemplateTokenizer()
    out = format_prompt("What is 2+2?", tokenizer=tok)
    assert "What is 2+2?" in out
    assert "<user>" not in out


def test_format_prompt_template_kwarg_with_query_placeholder():
    """``template`` containing ``{query}`` substitutes the question in-place
    and short-circuits both chat-template and plain-string paths."""
    tmpl = "[MATH_TASK] Problem:\n{query}\n\nSolution:"
    out = format_prompt("find x", template=tmpl)
    assert out == "[MATH_TASK] Problem:\nfind x\n\nSolution:"


def test_format_prompt_template_kwarg_without_placeholder_appends():
    """``template`` lacking ``{query}`` should have the question appended."""
    out = format_prompt("find x", template="PREAMBLE")
    assert "PREAMBLE" in out
    assert "find x" in out
    # Question comes after the template text.
    assert out.index("PREAMBLE") < out.index("find x")


def test_format_prompt_template_kwarg_overrides_tokenizer():
    """When ``template`` is given, the chat template must NOT be invoked
    (paper-faithful reproductions like VinePPO depend on this)."""
    tok = _FakeTokenizer()
    out = format_prompt(
        "find x",
        system_prompt="be terse",
        tokenizer=tok,
        template="Q: {query}\nA:",
    )
    assert out == "Q: find x\nA:"
    assert "<user>" not in out
    assert "<system>" not in out


def test_format_prompt_system_prompt_none_vs_given_plain():
    """Explicit comparison between ``system_prompt=None`` and a given
    ``system_prompt`` on the no-tokenizer plain-string path."""
    none_out = format_prompt("Q?", system_prompt=None)
    given_out = format_prompt("Q?", system_prompt="SYS")
    assert "SYS" not in none_out
    assert "SYS" in given_out
    # The "no-system" output must not accidentally render an empty header.
    assert not none_out.startswith("\n")
    # Both paths still embed the question.
    assert "Q?" in none_out and "Q?" in given_out


def test_format_prompt_system_prompt_none_vs_given_chat_template():
    """Same null-vs-given check on the chat-template path: a None system
    prompt should not produce a ``<system></system>`` block."""
    tok = _FakeTokenizer()
    none_out = format_prompt("Q?", system_prompt=None, tokenizer=tok)
    given_out = format_prompt("Q?", system_prompt="SYS", tokenizer=tok)
    assert "<system>" not in none_out
    assert "<system>SYS</system>" in given_out


def test_format_prompt_system_prompt_empty_string_treated_as_none():
    """``if system_prompt`` is the gate, so an empty string must NOT inject
    an empty system message."""
    tok = _FakeTokenizer()
    out = format_prompt("Q?", system_prompt="", tokenizer=tok)
    assert "<system>" not in out
    out_plain = format_prompt("Q?", system_prompt="")
    assert not out_plain.startswith("\n\n")


# ---------------------------------------------------------------------------
# Field-name handling via the in-memory factory
# ---------------------------------------------------------------------------

def _cfg(**kw):
    base = dict(max_prompt_len=10_000)  # effectively unlimited for these tests
    base.update(kw)
    return CASPOConfig(**base)


def test_factory_handles_deepscaler_fields():
    rows = [
        {"problem": "compute 1+1", "answer": "2"},
        {"problem": "compute 2+2", "answer": "4"},
    ]
    out = _from_iterable_factory(rows, _cfg(), tokenizer=None)
    assert len(out) == 2
    assert out[0]["raw_question"] == "compute 1+1"
    assert out[0]["ground_truth"] == "2"
    assert "compute 1+1" in out[0]["prompt"]


def test_factory_handles_math500_fields():
    rows = [
        {"problem": "find x", "solution": "blah blah \\boxed{7}"},
    ]
    out = _from_iterable_factory(rows, _cfg(), tokenizer=None)
    assert len(out) == 1
    assert out[0]["ground_truth"] == "7"


def test_factory_generic_fallback():
    rows = [{"question": "Q", "final_answer": "ans"}]
    out = _from_iterable_factory(rows, _cfg(), tokenizer=None)
    assert len(out) == 1
    assert out[0]["raw_question"] == "Q"
    assert out[0]["ground_truth"] == "ans"


def test_factory_handles_gsm8k_hash_format():
    """GSM8K answer field is ``<reasoning>\\n#### N``; we should peel off N."""
    rows = [
        {"question": "Q1", "answer": "step one\nstep two\n#### 42"},
        {"question": "Q2", "answer": "weird formatting\n   ####  17 \n"},
        {"question": "Q3", "answer": "thousands separator test\n#### 1,234"},
    ]
    out = _from_iterable_factory(rows, _cfg(), tokenizer=None)
    assert len(out) == 3
    assert out[0]["ground_truth"] == "42"
    assert out[1]["ground_truth"] == "17"
    assert out[2]["ground_truth"] == "1234"


def test_factory_strips_commas_from_gsm8k_answers():
    """Dedicated test: GSM8K writes thousand-separated integers like
    ``#### 1,234,567``. The comparator expects bare digits."""
    rows = [
        {"question": "Q1", "answer": "blah\n#### 1,234"},
        {"question": "Q2", "answer": "blah\n#### 1,234,567"},
        {"question": "Q3", "answer": "blah\n#### -2,500"},
        # Also exercise stray commas inside the tail (e.g. "1,000.").
        {"question": "Q4", "answer": "blah\n#### 1,000."},
    ]
    out = _from_iterable_factory(rows, _cfg(), tokenizer=None)
    gts = [r["ground_truth"] for r in out]
    assert gts == ["1234", "1234567", "-2500", "1000"]
    # Belt-and-suspenders: no commas leak through.
    for gt in gts:
        assert "," not in gt


def test_factory_strips_trailing_punctuation_from_gsm8k_tail():
    """The peel logic strips ``.;:`` trailing chars (e.g. ``#### 5.``)."""
    rows = [
        {"question": "Q1", "answer": "blah\n#### 5."},
        {"question": "Q2", "answer": "blah\n#### 7;"},
        {"question": "Q3", "answer": "blah\n#### 9:"},
    ]
    out = _from_iterable_factory(rows, _cfg(), tokenizer=None)
    gts = [r["ground_truth"] for r in out]
    assert gts == ["5", "7", "9"]


def test_factory_handles_numinamath_schema():
    """NuminaMath ships ``problem`` / ``solution`` where the solution is a
    full worked text containing a final ``\\boxed{...}``. We must peel the
    boxed answer (the LAST one wins, matching the grader)."""
    rows = [
        {
            "problem": "Compute the limit ...",
            "solution": (
                "We start by ... so the intermediate value is "
                "$\\boxed{0}$ which we discard. After more work, we conclude "
                "the limit equals $\\boxed{\\dfrac{1}{2}}$."
            ),
        },
        # NuminaMath sometimes also carries a sibling ``messages`` field; make
        # sure its presence doesn't interfere with field selection.
        {
            "problem": "Find n.",
            "solution": "Therefore $n = \\boxed{42}$.",
            "messages": [{"role": "user", "content": "ignored"}],
        },
        # Some NuminaMath subsets have both ``answer`` and ``solution`` — the
        # plain ``answer`` field should win per priority order.
        {
            "problem": "Find m.",
            "solution": "Hence $\\boxed{wrong}$.",
            "answer": "7",
        },
    ]
    out = _from_iterable_factory(rows, _cfg(), tokenizer=None)
    assert len(out) == 3
    # Last \boxed{} wins on the first row.
    assert out[0]["ground_truth"] == "\\dfrac{1}{2}"
    assert out[1]["ground_truth"] == "42"
    # Plain ``answer`` field beats ``solution``.
    assert out[2]["ground_truth"] == "7"


def test_factory_uses_prompt_template_kwarg_path():
    """When ``cfg.prompt_template`` is set, ``_row_to_example`` must thread
    it into ``format_prompt`` and bypass the tokenizer's chat template."""
    tok = _FakeTokenizer()  # has a chat template, must NOT be used
    rows = [{"problem": "find x", "answer": "5"}]
    cfg = _cfg(prompt_template="[MATH_TASK] Problem:\n{query}\n\nSolution:")
    out = _from_iterable_factory(rows, cfg, tokenizer=tok)
    assert len(out) == 1
    assert out[0]["prompt"] == "[MATH_TASK] Problem:\nfind x\n\nSolution:"
    # Chat-template wrappers must not appear.
    assert "<user>" not in out[0]["prompt"]
    assert "<system>" not in out[0]["prompt"]


def test_factory_prompt_template_without_placeholder_appends_question():
    rows = [{"problem": "find x", "answer": "5"}]
    cfg = _cfg(prompt_template="PREAMBLE-NO-PLACEHOLDER")
    out = _from_iterable_factory(rows, cfg, tokenizer=None)
    assert len(out) == 1
    assert "PREAMBLE-NO-PLACEHOLDER" in out[0]["prompt"]
    assert "find x" in out[0]["prompt"]


def test_factory_system_prompt_none_vs_given():
    """``cfg.system_prompt=None`` must not leak a system prefix; setting it
    must cause the prefix to appear in the rendered prompt."""
    rows = [{"problem": "Q", "answer": "A"}]
    out_none = _from_iterable_factory(rows, _cfg(system_prompt=None), tokenizer=None)
    out_set = _from_iterable_factory(rows, _cfg(system_prompt="SYS"), tokenizer=None)
    assert "SYS" not in out_none[0]["prompt"]
    assert "SYS" in out_set[0]["prompt"]
    # Sanity: the question is preserved in both.
    assert "Q" in out_none[0]["prompt"]
    assert "Q" in out_set[0]["prompt"]


def test_factory_drops_rows_missing_fields():
    rows = [
        {"problem": "good", "answer": "1"},
        {"problem": "no answer"},        # missing answer → dropped
        {"answer": "no question"},       # missing question → dropped
        {"question": "Q", "answer": "A"},
    ]
    out = _from_iterable_factory(rows, _cfg(), tokenizer=None)
    assert len(out) == 2


def test_factory_filters_long_prompts_with_tokenizer():
    tok = _FakeTokenizer()
    rows = [
        {"problem": "short", "answer": "1"},
        {"problem": " ".join(["word"] * 50), "answer": "2"},
    ]
    cfg = _cfg(max_prompt_len=5)
    out = _from_iterable_factory(rows, cfg, tokenizer=tok)
    # Only the short prompt should survive.
    assert len(out) == 1
    assert out[0]["raw_question"] == "short"


def test_factory_includes_system_prompt_in_prompt():
    rows = [{"problem": "Q", "answer": "A"}]
    cfg = _cfg(system_prompt="SYS")
    out = _from_iterable_factory(rows, cfg, tokenizer=None)
    assert "SYS" in out[0]["prompt"]
    assert "Q" in out[0]["prompt"]


# ---------------------------------------------------------------------------
# Live HF loaders — skipped unless the dataset is reachable.
# ---------------------------------------------------------------------------

def _datasets_available():
    try:
        import datasets  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _datasets_available(), reason="datasets not installed")
def test_load_train_dataset_smoke():
    # We don't want a network hit in unit tests; importorskip the loader and
    # skip on any IO/HTTP error.
    pytest.importorskip("datasets")
    cfg = CASPOConfig(max_prompt_len=10_000)
    try:
        out = load_train_dataset(cfg, tokenizer=None)
    except Exception as e:
        pytest.skip(f"HF unreachable: {e}")
    # If we got here, sanity-check shape.
    iterator = iter(out)
    first = next(iterator, None)
    if first is None:
        pytest.skip("dataset returned empty")
    assert {"prompt", "ground_truth", "raw_question"} <= set(first.keys())


@pytest.mark.skipif(not _datasets_available(), reason="datasets not installed")
def test_load_eval_dataset_smoke():
    pytest.importorskip("datasets")
    cfg = CASPOConfig(max_prompt_len=10_000)
    try:
        out = load_eval_dataset(cfg, tokenizer=None)
    except Exception as e:
        pytest.skip(f"HF unreachable: {e}")
    iterator = iter(out)
    first = next(iterator, None)
    if first is None:
        pytest.skip("dataset returned empty")
    assert {"prompt", "ground_truth", "raw_question"} <= set(first.keys())
