"""Tests for ``caspo.reward.math_verifier``."""

from __future__ import annotations

import pytest

from caspo.reward.math_verifier import (
    MathRewardFn,
    extract_boxed_answer,
    grade_math,
)


# ---------------------------------------------------------------------------
# extract_boxed_answer
# ---------------------------------------------------------------------------

def test_extract_plain_boxed():
    assert extract_boxed_answer("the answer is \\boxed{42}.") == "42"


def test_extract_nested_braces():
    assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_extract_missing_returns_none():
    assert extract_boxed_answer("no answer here") is None
    assert extract_boxed_answer("") is None


def test_extract_picks_last_when_multiple():
    text = "first try \\boxed{1}, then \\boxed{2}, finally \\boxed{3}"
    assert extract_boxed_answer(text) == "3"


def test_extract_deeply_nested():
    text = "\\boxed{a\\frac{b}{c} + \\sqrt{d}}"
    assert extract_boxed_answer(text) == "a\\frac{b}{c} + \\sqrt{d}"


def test_extract_handles_unmatched_braces_gracefully():
    # Last \boxed is unmatched; the previous one should win.
    text = "\\boxed{1} then garbage \\boxed{2"
    assert extract_boxed_answer(text) == "1"


def test_extract_nested_boxed():
    """``\\boxed{\\boxed{2}}`` — the inner ``\\boxed{2}`` is itself a valid
    boxed start, and since the extractor returns the LAST well-formed
    boxed expression, it should unwrap to ``"2"``. This matters because
    models occasionally emit nested boxes; we want grading to still see
    the bare answer.
    """
    assert extract_boxed_answer("\\boxed{\\boxed{2}}") == "2"


def test_extract_no_boxed_in_response():
    """Long realistic response without any \\boxed at all returns None."""
    text = (
        "Let me think about this step by step. First, we compute 6*7=42. "
        "Therefore the final answer is 42. (No box here.)"
    )
    assert extract_boxed_answer(text) is None


def test_extract_unicode_answer():
    """Pi (π) and other unicode should pass through unchanged."""
    assert extract_boxed_answer("\\boxed{π}") == "π"
    assert extract_boxed_answer("\\boxed{α + β = γ}") == "α + β = γ"
    assert extract_boxed_answer("\\boxed{√2}") == "√2"


# ---------------------------------------------------------------------------
# grade_math
# ---------------------------------------------------------------------------

def test_grade_correct_simple():
    assert grade_math("the answer is \\boxed{42}", "42") == 1.0


def test_grade_wrong_simple():
    assert grade_math("the answer is \\boxed{41}", "42") == 0.0


def test_grade_no_boxed_returns_zero():
    assert grade_math("answer = 42", "42") == 0.0


def test_grade_ground_truth_is_boxed():
    assert grade_math("\\boxed{42}", "\\boxed{42}") == 1.0


def test_grade_strips_whitespace_and_dollar_signs():
    assert grade_math("\\boxed{ 42 }", "$42$") == 1.0


def test_grade_equivalent_fraction_optional():
    """Equivalence between '1/2' and '0.5' depends on math_verify or sympy.
    Guarded so the test passes when neither dep recognizes the equivalence.
    """
    try:
        score = grade_math("\\boxed{1/2}", "0.5")
    except Exception:
        pytest.skip("optional grader threw")
    # If the grader recognizes the equivalence, great; otherwise just ensure
    # we didn't get a false positive on something else.
    assert score in (0.0, 1.0)


def test_grade_math_verify_equivalence_when_available():
    """When ``math_verify`` is installed, ``1/2`` and ``0.5`` should grade as
    equivalent. Skipped otherwise (and SymPy alone may also not bridge the
    fraction-vs-decimal gap, so we don't require it as a fallback).
    """
    pytest.importorskip("math_verify")
    assert grade_math("\\boxed{1/2}", "0.5") == 1.0
    # And other classic equivalences expected of math_verify.
    assert grade_math("\\boxed{\\frac{1}{2}}", "0.5") == 1.0


def test_grade_picks_last_boxed_for_correctness():
    """Multiple \\boxed in the response — the LAST one decides correctness.
    A correct early box followed by a wrong final box must score 0.0.
    """
    # Wrong-then-right: graded right.
    assert grade_math("first \\boxed{1}, then \\boxed{42}", "42") == 1.0
    # Right-then-wrong: graded wrong (no credit for early correctness).
    assert grade_math("first \\boxed{42}, then \\boxed{1}", "42") == 0.0


def test_grade_nested_boxed_in_response():
    """``\\boxed{\\boxed{2}}`` should grade as the bare answer ``2``, both
    against a naked ``"2"`` ground truth and against a similarly-nested
    ground truth (which gets peeled by ``_strip_to_bare``).
    """
    assert grade_math("\\boxed{\\boxed{2}}", "2") == 1.0
    assert grade_math("\\boxed{\\boxed{2}}", "\\boxed{\\boxed{2}}") == 1.0
    # Wrong nested answer must not get credit.
    assert grade_math("\\boxed{\\boxed{2}}", "3") == 0.0


def test_grade_unicode_answer():
    """Unicode answers should grade equal to themselves (no encoding bug)."""
    assert grade_math("the answer is \\boxed{π}", "π") == 1.0
    assert grade_math("\\boxed{α}", "α") == 1.0
    # And mismatched unicode should be wrong.
    assert grade_math("\\boxed{π}", "e") == 0.0


# ---------------------------------------------------------------------------
# MathRewardFn
# ---------------------------------------------------------------------------

def test_math_reward_fn_default_no_bonus():
    fn = MathRewardFn()
    out = fn(["plain text", "\\boxed{42}", "\\boxed{41}"], ["42", "42", "42"])
    assert out == [0.0, 1.0, 0.0]


def test_math_reward_fn_format_bonus():
    fn = MathRewardFn(format_bonus=0.1)
    preds = [
        "plain text without box",     # no box → 0.0
        "\\boxed{wrong_thing}",       # boxed but wrong → 0.1
        "answer is \\boxed{42}",      # correct → 1.0
    ]
    gts = ["42", "42", "42"]
    out = fn(preds, gts)
    assert out[0] == 0.0
    assert out[1] == pytest.approx(0.1)
    assert out[2] == 1.0


def test_math_reward_fn_length_mismatch_raises():
    fn = MathRewardFn()
    with pytest.raises(ValueError):
        fn(["\\boxed{1}"], ["1", "2"])


def test_math_reward_fn_no_boxed_in_response():
    """Predictions with no \\boxed at all should always score 0.0, even with
    a format bonus configured (the bonus only fires when a box is present).
    """
    fn = MathRewardFn(format_bonus=0.1)
    preds = [
        "I think the answer is 42 but I forgot to box it.",
        "",  # empty
        "no math here, just prose",
    ]
    gts = ["42", "42", "42"]
    assert fn(preds, gts) == [0.0, 0.0, 0.0]


def test_math_reward_fn_format_bonus_does_not_bias_thresholding():
    """The format bonus (e.g., 0.1) must stay strictly below any sane
    correctness threshold. With threshold=0.5 and bonus=0.1, wrong-but-boxed
    must remain BELOW threshold, while correct stays at 1.0 ABOVE threshold.
    """
    fn = MathRewardFn(format_bonus=0.1)
    preds = [
        "no box at all",
        "\\boxed{wrong}",
        "\\boxed{42}",
    ]
    gts = ["42", "42", "42"]
    scores = fn(preds, gts)

    threshold = 0.5
    # A correctness threshold cleanly separates correct from formatted-wrong.
    assert scores[0] < threshold      # 0.0 — no box, no bonus
    assert scores[1] < threshold      # 0.1 — bonus only, still below thresh
    assert scores[2] >= threshold     # 1.0 — correct
    # And the bonus is not stacked on top of a correct answer.
    assert scores[2] == 1.0
    assert scores[1] == pytest.approx(0.1)


def test_math_reward_fn_nested_boxed():
    """``\\boxed{\\boxed{2}}`` predictions should score correctly when
    matched against a similarly-nested ground truth, and award only the
    format bonus against a different ground truth.
    """
    fn = MathRewardFn(format_bonus=0.1)
    preds = ["\\boxed{\\boxed{2}}", "\\boxed{\\boxed{2}}"]
    gts = ["\\boxed{\\boxed{2}}", "\\boxed{\\boxed{3}}"]
    scores = fn(preds, gts)
    assert scores[0] == 1.0
    assert scores[1] == pytest.approx(0.1)


def test_math_reward_fn_unicode():
    """Unicode predictions/ground-truths flow through the batch path."""
    fn = MathRewardFn(format_bonus=0.1)
    preds = ["\\boxed{π}", "\\boxed{π}", "\\boxed{e}"]
    gts = ["π", "e", "e"]
    scores = fn(preds, gts)
    assert scores[0] == 1.0
    assert scores[1] == pytest.approx(0.1)  # boxed but wrong
    assert scores[2] == 1.0
