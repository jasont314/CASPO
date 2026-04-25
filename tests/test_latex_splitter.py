"""Tests for the LaTeX-aware step splitter ported from VinePPO.

These exercise ``split_solution_inplace`` (char-level) and the token-level
wrapper ``segment_responses_batch_latex_aware``. We compare against expected
behavior, not against VinePPO's running code (we trust the verbatim port).
"""

from __future__ import annotations

import pytest
import torch

from caspo.segmentation.latex_splitter import split_solution_inplace
from caspo.segmentation.steps import segment_responses_batch_latex_aware


# ---------------------------------------------------------------------------
# Char-level tests on split_solution_inplace
# ---------------------------------------------------------------------------


def _segments(text: str, indices) -> list[str]:
    return [text[indices[i]: indices[i + 1]] for i in range(len(indices) - 1)]


def test_empty_input() -> None:
    assert split_solution_inplace("") == [0]


def test_single_sentence_no_split() -> None:
    text = "We need to find the answer."
    indices = split_solution_inplace(text)
    assert indices[0] == 0 and indices[-1] == len(text)


def test_two_sentences_split() -> None:
    text = "We compute the value. Then we add three to get the answer."
    indices = split_solution_inplace(text)
    assert len(indices) >= 2  # at least the trivial [0, len]
    # The split point should be after "value." (end of first sentence + space).
    parts = _segments(text, indices)
    # Each segment should be non-empty and the rejoined text matches input.
    assert "".join(parts) == text


def test_does_not_split_inside_inline_math() -> None:
    # The period inside $...$ should NOT trigger a sentence split.
    text = r"We have $x = 1.5$ as the answer."
    indices = split_solution_inplace(text)
    parts = _segments(text, indices)
    # Sanity: rejoining yields the original.
    assert "".join(parts) == text
    # No split should land inside the math env.
    for cut in indices[1:-1]:
        # cut must NOT be between the dollar signs
        before = text[:cut]
        after = text[cut:]
        assert not (before.count("$") % 2 == 1), (
            f"Split landed inside $...$ at index {cut}: {text!r}"
        )


def test_does_not_split_inside_display_math() -> None:
    text = r"First we set up. \[ x^2 + 1 = 0. \] Solving gives the result."
    indices = split_solution_inplace(text)
    parts = _segments(text, indices)
    assert "".join(parts) == text
    # Each \[ must be in the same segment as its matching \].
    for p in parts:
        assert p.count(r"\[") == p.count(r"\]"), (
            f"display-math env split across segments: {p!r}"
        )


def test_round_trip_preservation() -> None:
    """Concatenating segments must reproduce the input verbatim."""
    text = (
        "Step 1: We compute the area.\n"
        r"Step 2: Using $A = \pi r^2$, we substitute $r = 3$ to get $A = 9\pi$."
        "\nStep 3: The final answer is $\\boxed{9\\pi}$."
    )
    indices = split_solution_inplace(text)
    parts = _segments(text, indices)
    assert "".join(parts) == text
    # Should produce more than one step.
    assert len(parts) >= 2


def test_no_assertion_failure_on_long_realistic_response() -> None:
    """Stress-test: a long realistic-looking response must not blow assertions."""
    text = (
        "Let me work through this problem.\n"
        "First, we identify the variables. Let $x$ be the unknown.\n"
        r"Setting up the equation: \[ 2x + 3 = 11. \]" + "\n"
        r"Solving: $x = \frac{11 - 3}{2} = \frac{8}{2} = 4$." + "\n"
        "Therefore, the answer is $x = 4$.\n"
        "Final answer: $\\boxed{4}$."
    )
    # Should not raise.
    indices = split_solution_inplace(text)
    assert indices[0] == 0
    assert indices[-1] == len(text)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


def test_empty_input_returns_zero_only() -> None:
    """Empty string must return exactly [0] (no length entry)."""
    assert split_solution_inplace("") == [0]


def test_pure_whitespace_input() -> None:
    """Pure-whitespace input currently crashes the splitter (known bug).

    The pipeline ends up returning ``[0]`` from
    ``_split_newline_in_placeholders`` (because the single whitespace fragment
    is < 3 chars and gets dropped), which then trips the
    ``indices[-1] == len(text)`` assertion in ``_try_to_break_very_long_parts``.

    This test pins the current (buggy) behavior so a future fix is detected.
    """
    for text in ("   ", "\n\n\n", " \t \n ", "\n   \n   \n"):
        with pytest.raises(AssertionError):
            split_solution_inplace(text)


def test_numbers_only_response() -> None:
    """No-period, no-math input currently crashes the splitter (known bug).

    Same root cause as ``test_pure_whitespace_input``: short fragments with no
    period and no placeholder get dropped, breaking the trailing-index
    invariant. Pinned with ``pytest.raises`` to catch any future fix.
    """
    for text in ("42", "3.14159", "1 2 3 4 5", "100, 200, 300"):
        with pytest.raises(AssertionError):
            split_solution_inplace(text)


def test_very_long_math_env() -> None:
    """A very long display-math env *is* broken at ``+``/``=`` operators.

    This is intentional (see ``_best_effort_break_long_part_in_math`` and the
    module docstring step 5). We assert the round-trip and that the splitter
    actually produces multiple in-math fragments — the env-balance invariant
    from ``test_does_not_split_inside_display_math`` does NOT hold once the
    fragment exceeds ``MAX_PART_LENGTH`` (100 chars).
    """
    # Build a long \[ ... \] environment well past MAX_PART_LENGTH (100).
    inner = " + ".join(f"x_{{{i}}}" for i in range(80))
    text = (
        "We sum the terms.\n"
        rf"\[ S = {inner}. \]" + "\n"
        "This gives the total."
    )
    indices = split_solution_inplace(text)
    parts = _segments(text, indices)
    # Round-trip is preserved.
    assert "".join(parts) == text
    # The env *is* broken into multiple math-content fragments by design.
    assert len(parts) > 3
    # No fragment exceeds MAX_PART_LENGTH by more than a small margin
    # (the splitter caps long-math fragments around ~MAX_PART_LENGTH).
    assert all(len(p) <= 120 for p in parts), [len(p) for p in parts]


def test_nested_math_envs() -> None:
    """Nested / adjacent math envs must round-trip and stay matched per segment."""
    text = (
        r"We start with $a = 1$ and $b = 2$. "
        r"Then \[ c = a + b = \$3\$ \cdot \frac{1}{1} \]" + " "
        r"and finally $d = c^2$."
    )
    indices = split_solution_inplace(text)
    parts = _segments(text, indices)
    assert "".join(parts) == text
    # \[ ... \] must stay paired within each segment.
    for p in parts:
        assert p.count(r"\[") == p.count(r"\]"), (
            f"display env split across segments: {p!r}"
        )


def test_nested_begin_end_envs() -> None:
    """Adjacent ``\\begin{...}\\end{...}`` envs must stay intact."""
    text = (
        "Consider the matrix below.\n"
        r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}" + "\n"
        "And the determinant.\n"
        r"\begin{align} \det &= 1 \cdot 4 - 2 \cdot 3 \\ &= -2 \end{align}" + "\n"
        "So the result is $-2$."
    )
    indices = split_solution_inplace(text)
    parts = _segments(text, indices)
    assert "".join(parts) == text
    # \begin{X}/\end{X} pairs must balance per segment for known names.
    for env_name in ("pmatrix", "align"):
        for p in parts:
            assert p.count(rf"\begin{{{env_name}}}") == p.count(rf"\end{{{env_name}}}"), (
                f"{env_name} env split across segments: {p!r}"
            )


def test_markdown_code_blocks() -> None:
    """Markdown ``` fences are not LaTeX envs; splitter must still round-trip."""
    text = (
        "Here is some code.\n"
        "```python\n"
        "def f(x):\n"
        "    return x + 1\n"
        "```\n"
        "It returns $x + 1$."
    )
    indices = split_solution_inplace(text)
    parts = _segments(text, indices)
    # Round-trip is the only firm guarantee — splitter is not aware of ``` fences.
    assert "".join(parts) == text
    assert indices[0] == 0
    assert indices[-1] == len(text)


def test_unicode_characters() -> None:
    """Unicode (Greek, math symbols, CJK, emoji) must round-trip exactly."""
    text = (
        "We use π ≈ 3.14159 and θ for the angle. "
        "Then α + β = γ. "
        "中文测试。 日本語テスト. "
        "Emoji ok."
    )
    indices = split_solution_inplace(text)
    parts = _segments(text, indices)
    assert "".join(parts) == text
    # Boundaries are character indices; recombining via Python slicing must be
    # byte-for-byte identical (Python strings are unicode-aware).
    assert indices[0] == 0
    assert indices[-1] == len(text)


def test_tab_characters() -> None:
    """Tabs must be preserved in the round-trip and not crash the splitter."""
    text = (
        "First step.\n"
        "\tIndented with a tab. Then another sentence.\n"
        "\t\tDouble tab. End here."
    )
    indices = split_solution_inplace(text)
    parts = _segments(text, indices)
    assert "".join(parts) == text
    # Tabs preserved.
    assert "\t" in "".join(parts)
    assert indices[0] == 0
    assert indices[-1] == len(text)


# ---------------------------------------------------------------------------
# Token-level wrapper tests
# ---------------------------------------------------------------------------


def _try_load_tokenizer():
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception as e:
        pytest.skip(f"tiny tokenizer unavailable: {e}")


def test_token_level_wrapper_one_step():
    tok = _try_load_tokenizer()
    text = "Just one short sentence."
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    seg = segment_responses_batch_latex_aware(
        enc["input_ids"], enc["attention_mask"], tok,
        min_step_tokens=2, max_steps=8,
    )
    assert seg.step_count.item() >= 1


def test_token_level_wrapper_multi_step():
    tok = _try_load_tokenizer()
    text = (
        "First we compute the answer.\n"
        "Then we verify it. Finally we conclude."
    )
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    seg = segment_responses_batch_latex_aware(
        enc["input_ids"], enc["attention_mask"], tok,
        min_step_tokens=2, max_steps=8,
    )
    # Multiple sentences → at least 2 steps after merging.
    assert seg.step_count.item() >= 2
    # Step ids on valid tokens should cover [0, step_count - 1].
    valid = enc["attention_mask"][0].bool()
    sids = seg.step_id[0][valid]
    assert sids.min().item() == 0
    assert sids.max().item() == seg.step_count.item() - 1


def test_token_level_wrapper_preserves_step_lengths_sum():
    tok = _try_load_tokenizer()
    text = "Sentence one is here.\nSentence two follows.\nSentence three ends it."
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    seg = segment_responses_batch_latex_aware(
        enc["input_ids"], enc["attention_mask"], tok,
        min_step_tokens=2, max_steps=8,
    )
    valid_len = int(enc["attention_mask"][0].sum())
    assert int(seg.step_lengths[0].sum().item()) == valid_len


def test_token_level_wrapper_zero_valid_len():
    tok = _try_load_tokenizer()
    response_ids = torch.zeros(1, 8, dtype=torch.long)
    response_mask = torch.zeros(1, 8, dtype=torch.long)
    seg = segment_responses_batch_latex_aware(
        response_ids, response_mask, tok,
        min_step_tokens=2, max_steps=8,
    )
    assert seg.step_count.item() == 0
    assert (seg.step_id == -1).all()
