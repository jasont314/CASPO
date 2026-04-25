"""LaTeX-aware step splitter for math reasoning, ported verbatim from VinePPO
(`src/treetune/tasks/math_extract_steps_inplace.py` in McGill-NLP/VinePPO).

The only deviations vs. the upstream code:

* ``md5_hash`` is inlined (it was a one-line wrapper around ``hashlib.md5``).
* No external imports outside the stdlib.
* Function signatures and the public ``split_solution_inplace`` entrypoint
  are unchanged so the segmentation result matches VinePPO's at the
  character-index level — that's the whole point of porting.

Algorithm summary
-----------------
1. Mask LaTeX/asymptote/tabular environments with placeholder tokens so the
   splitter can't cut into ``\\[...\\]``, ``$$...$$``, ``\\begin{...}...\\end{...}``,
   ``[asy]...[/asy]``, etc.
2. Split on sentence-ending periods followed by capitalized whitespace, then
   on newlines.
3. Re-merge fragments shorter than 20 characters (or that are pure newlines)
   into the next/previous fragment.
4. Recover the placeholders.
5. If a fragment is still longer than 100 chars, try to break it at
   punctuation (commas, "and", "to") or at math operators (=, ≡, +,
   \\cancelto). Caps fragments at ~MAX_PART_LENGTH.

Returns a list of character-level *boundary indices* into the original text:
``[0, end_of_step_0, end_of_step_1, ..., len(text)]``. The actual segments
are ``text[indices[i]:indices[i+1]]``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Tuple

MAX_PART_LENGTH = 100
PLACEHOLDER_LENGTH = 60

# ---------------------------------------------------------------------------
# Module-level compiled regexes (all flags & strings preserved verbatim from
# upstream; only the compile-once-per-import optimization).
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"<\|PLACEHOLDER_[0-9a-f]{32}_PLACEHOLDER\|>")

_MATH_PATTERN_SOURCES: Tuple[Tuple[str, str], ...] = (
    (r"\[asy\].*?\[/asy\]", "asymptote"),
    (r"\$\\begin\{tabular\}.*?\\end\{tabular\}\$", "tabular_inline"),
    (r"\\begin\{tabular\}.*?\\end\{tabular\}", "tabular"),
    (r"\\\((.*?)\\\)", "display("),
    (r"\\\[.*?\\\]", "display_["),
    (r"\$\$(.*?)\$\$", "display_$$"),
    (r"(?<!\\)\$(.*?[^\\])\$(?!\$)", "display_$"),
    (r"\\begin\{([^}]*)\}.*?\\end\{\1\}", "environment"),
)

# Pre-compile each math pattern with DOTALL so re-entrant calls don't pay the
# parser cost. Order is preserved (it matters: tabular before generic environment).
_MATH_PATTERNS_COMPILED: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(p, re.DOTALL), t) for p, t in _MATH_PATTERN_SOURCES
)

_PERIOD_SPLIT_RE = re.compile(
    r"(?<!\b[A-Za-z]\.)(?<!\s[A-Za-z]\.)(?<!\s[A-Za-z][A-Za-z]\.)"
    r"(?<=\.)(?=\s+[A-Z])"
)

# Single-opening fragments to drop from the post-newline split. Use fullmatch
# so anchoring matches the upstream `re.fullmatch(opening, ...)` semantics.
_SINGLE_OPENING_RES: Tuple[re.Pattern, ...] = tuple(
    re.compile(p)
    for p in (
        r"\$",
        r"\$\$",
        r"\\\(",
        r"\\\[",
        r"\\begin\{[^}]*\}",
        r"\[asy\]",
    )
)

# Punctuation patterns + their match lengths for language-mode breaking.
_PUNCT_PATTERNS: Tuple[Tuple[re.Pattern, int], ...] = (
    (re.compile(r","), 1),
    (re.compile(r"\sand\s"), 4),
    (re.compile(r":"), 1),
    (re.compile(r"\sto\s"), 3),
)

# Math-mode split patterns + their match lengths.
_MATH_SPLIT_PATTERNS: Tuple[Tuple[re.Pattern, int], ...] = (
    (re.compile(r",\$"), 2),
    (re.compile(r",\\\)"), 3),
    (re.compile(r",\\\]"), 3),
    (re.compile(r"="), 1),
    (re.compile(r"\\equiv"), 6),
    (re.compile(r"\+"), 1),
    (re.compile(r"\\cancelto"), 9),
)

_TAIL_PATTERNS: Tuple[str, ...] = (".$", ".$$", ".\\)", ".\\]")


def _md5_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _has_placeholders(text: str) -> bool:
    return "<|PLACEHOLDER_" in text


def _find_index_of_all_placeholders(text: str) -> List[int]:
    return [m.start() for m in _PLACEHOLDER_RE.finditer(text)]


def _replace_math_with_placeholders(
    latex_content: str,
) -> Tuple[str, Dict[str, str], Dict[str, str]]:
    placeholders: Dict[str, str] = {}
    placeholders_type: Dict[str, str] = {}

    for pattern_re, pattern_type in _MATH_PATTERNS_COMPILED:
        # Fast path: skip patterns that can't match anything in this text.
        # `re.sub` over a non-matching pattern still walks the string; checking
        # `search` is cheaper for the common case of texts with little/no math.
        if pattern_re.search(latex_content) is None:
            continue

        def _replace(match, pt=pattern_type):
            matched = match.group(0)
            hash_key = _md5_hash(matched)
            placeholder = f"<|PLACEHOLDER_{hash_key}_PLACEHOLDER|>"
            placeholders[placeholder] = matched
            placeholders_type[placeholder] = pt
            return placeholder

        latex_content = pattern_re.sub(_replace, latex_content)
    return latex_content, placeholders, placeholders_type


def _recover_math_from_placeholders(
    text_with_placeholders: str, placeholders: Dict[str, str]
) -> str:
    # Skip the (often empty) dict-iteration entirely if there are no
    # placeholders left in the text. Each `str.replace` walks the full string,
    # so this short-circuit is significant when most steps don't contain math.
    if not placeholders or "<|PLACEHOLDER_" not in text_with_placeholders:
        return text_with_placeholders
    for placeholder, math in placeholders.items():
        # `placeholder in text` is a single C-level scan; `replace` does its
        # own scan plus an allocation. Skip the alloc for the common no-hit case.
        if placeholder in text_with_placeholders:
            text_with_placeholders = text_with_placeholders.replace(placeholder, math)
    return text_with_placeholders


def _naive_split_by_newline_and_period(disguised_latex: str) -> List[int]:
    parts = _PERIOD_SPLIT_RE.split(disguised_latex)

    final_parts: List[str] = []
    for part in parts:
        # `str.split('\n')` is cheaper than building intermediate lists with
        # explicit appends; we just need to re-prefix every chunk after the
        # first one with the leading "\n" that `split` ate.
        chunks = part.split("\n")
        if chunks:
            final_parts.append(chunks[0])
            for chunk in chunks[1:]:
                final_parts.append("\n" + chunk)

    i = 0
    n = len(final_parts)
    while i < n - 1:
        cur = final_parts[i]
        if cur == "\n":
            final_parts[i + 1] = cur + final_parts[i + 1]
            final_parts.pop(i)
            n -= 1
        elif not cur.endswith(".") and len(cur) < 20:
            assert final_parts[i + 1].startswith("\n"), "Next part should start with \n"
            final_parts[i] = cur + final_parts.pop(i + 1)
            n -= 1
        else:
            i += 1

    for p in final_parts:
        if "<|PLACEHOLDER_" in p:
            assert "PLACEHOLDER|>" in p, "Cut into a placeholder"

    indices = [0]
    cum = 0
    for part in final_parts:
        cum += len(part)
        indices.append(cum)
    return indices


def _split_tailing_periods_in_placeholders(
    text: str, indices: List[int], placeholders: Dict[str, str]
) -> List[int]:
    assert indices[-1] == len(text), "Last index should be the length of the text"
    tail_patterns = _TAIL_PATTERNS

    new_indices = [0]
    last = new_indices[0]
    for i_part in range(len(indices) - 1):
        a, b = indices[i_part], indices[i_part + 1]
        part = text[a:b]
        split_points: List[int] = []
        for start_idx in _find_index_of_all_placeholders(part):
            ph_text = part[start_idx : start_idx + PLACEHOLDER_LENGTH]
            actual_text = placeholders[ph_text]
            for pattern in tail_patterns:
                if actual_text.endswith(pattern):
                    split_points.append(start_idx + PLACEHOLDER_LENGTH)
                    break
        split_points.append(len(part))
        for sp in split_points:
            new_indices.append(last + sp)
        last = new_indices[-1]

    i = 0
    while i < len(new_indices) - 1:
        part = text[new_indices[i] : new_indices[i + 1]]
        part_no_ph = _recover_math_from_placeholders(part, placeholders)
        # Inline-equivalent of `any(part_no_ph.endswith(p) for p in tail_patterns)`
        # — `str.endswith` accepts a tuple, so this is a single C-level check.
        has_period = part_no_ph.endswith(tail_patterns)
        if not has_period and len(part_no_ph) < 20:
            if i == len(new_indices) - 2:
                new_indices.pop(i)
            else:
                new_indices.pop(i + 1)
        else:
            i += 1

    for i in range(len(new_indices) - 1):
        part = text[new_indices[i] : new_indices[i + 1]]
        if "<|PLACEHOLDER_" in part:
            assert "PLACEHOLDER|>" in part, "Cut into a placeholder"
    return new_indices


def _split_newline_in_placeholders(
    text: str, indices: List[int], placeholders: Dict[str, str]
) -> Tuple[List[int], str]:
    assert indices[-1] == len(text), "Last index should be the length of the text"

    new_indices = [0]
    last = 0
    for i_part in range(len(indices) - 1):
        a, b = indices[i_part], indices[i_part + 1]
        part = text[a:b]
        split_points: List[int] = []
        # Upstream behavior: in each loop pass we scan all placeholder
        # positions, take the *first*, compute newline offsets for it, then
        # `str.replace(ph_text, actual_text)` over the whole `part` — which
        # expands EVERY occurrence of that hash. Then re-scan and continue.
        # We preserve that exact behavior; the only optimization is using
        # `str.find` for the "first placeholder" check (cheaper than the
        # regex-based full scan when we're going to immediately break).
        while True:
            first_idx = part.find("<|PLACEHOLDER_")
            if first_idx == -1:
                break
            ph_text = part[first_idx : first_idx + PLACEHOLDER_LENGTH]
            actual_text = placeholders[ph_text]
            base = first_idx
            start = 0
            while True:
                nl = actual_text.find("\n", start)
                if nl == -1:
                    break
                split_points.append(base + nl)
                start = nl + 1
            part = part.replace(ph_text, actual_text)
        split_points.append(len(part))
        for sp in split_points:
            new_indices.append(last + sp)
        last = new_indices[-1]

    recovered_text = _recover_math_from_placeholders(text, placeholders)
    single_openings = _SINGLE_OPENING_RES
    i = 0
    while i < len(new_indices) - 1:
        part = recovered_text[new_indices[i] : new_indices[i + 1]]
        stripped = part.strip()
        is_single_opening = False
        # `any(...)` with generator has nontrivial overhead vs an explicit loop;
        # this is one of the hottest call sites in the splitter.
        for opening in single_openings:
            if opening.fullmatch(stripped):
                is_single_opening = True
                break
        if is_single_opening or len(part) < 3 or part == "\n":
            if i == len(new_indices) - 2:
                new_indices.pop(i)
            else:
                new_indices.pop(i + 1)
        else:
            i += 1
    return new_indices, recovered_text


def _best_effort_break_long_part_in_language(part: str) -> List[int]:
    orig_part = part
    part, placeholders, _ = _replace_math_with_placeholders(part)

    new_parts: List[str] = []
    for punc_re, pat_len in _PUNCT_PATTERNS:
        # Cheap reject: skip pattern if it can't match.
        first = punc_re.search(part)
        if first is None:
            continue
        # Build positions list with finditer (already-compiled regex).
        positions = [m.start() + pat_len for m in punc_re.finditer(part)]
        positions.sort()
        # `len(positions) == 0` is impossible after the search() guard, but
        # keep the structure for parity with upstream.
        if not positions:
            continue
        min_diff = float("inf")
        optimal_pos = None
        n_part = len(part)
        for pos in positions:
            diff = abs(pos - (n_part - pos))
            if diff < min_diff:
                min_diff = diff
                optimal_pos = pos
        # Recovering math is O(num_placeholders × text_len); avoid it twice
        # when we can: compute lengths additively against `part` first and
        # only recover when strictly necessary. The recovered length differs
        # from the placeholder-replaced length by sum(len(actual) - PH_LEN)
        # over placeholders contained in each side. But that mapping requires
        # knowing which placeholders fall on each side — so the simplest
        # bitwise-identical implementation is to call the original helper.
        min_optimal_length = min(
            len(_recover_math_from_placeholders(part[:optimal_pos], placeholders)),
            len(_recover_math_from_placeholders(part[optimal_pos:], placeholders)),
        )
        if min_optimal_length < 40:
            continue
        if optimal_pos is not None:
            new_parts.append(part[:optimal_pos])
            new_parts.append(part[optimal_pos:])
            break

    if len(new_parts) == 0:
        new_parts.append(part)
    new_parts = [_recover_math_from_placeholders(p, placeholders) for p in new_parts]
    assert "".join(new_parts) == orig_part
    indices = [0]
    cum = 0
    for new_part in new_parts:
        cum += len(new_part)
        indices.append(cum)
    return indices


def _best_effort_break_long_part_in_math(part: str) -> List[int]:
    new_parts: List[str] = []
    n_part = len(part)
    for split_re, pat_len in _MATH_SPLIT_PATTERNS:
        first = split_re.search(part)
        if first is None:
            continue
        positions = [m.start() + pat_len for m in split_re.finditer(part)]
        positions.sort()
        if not positions:
            continue
        min_diff = float("inf")
        optimal_pos = None
        for pos in positions:
            diff = abs(pos - (n_part - pos))
            if diff < min_diff:
                min_diff = diff
                optimal_pos = pos
        min_optimal_length = min(optimal_pos, n_part - optimal_pos)
        if min_optimal_length < 20:
            continue
        if optimal_pos is not None:
            new_parts.append(part[:optimal_pos])
            new_parts.append(part[optimal_pos:])
            break

    if len(new_parts) == 0:
        new_parts.append(part)
    assert "".join(new_parts) == part
    indices = [0]
    cum = 0
    for new_part in new_parts:
        cum += len(new_part)
        indices.append(cum)
    return indices


def _break_part_as_much_as_possible(part: str) -> List[int]:
    indices = [0, len(part)]
    split_with_math = [-1, 0]

    def is_in_math_range(idx):
        return split_with_math[0] <= idx < split_with_math[1]

    def update_math_range(begin, end):
        if split_with_math[0] == -1:
            split_with_math[0] = begin
        else:
            split_with_math[0] = min(split_with_math[0], begin)
        split_with_math[1] = max(split_with_math[1], end)

    i = 0
    while i < len(indices) - 1:
        sub_part = part[indices[i] : indices[i + 1]]
        if len(sub_part) > MAX_PART_LENGTH:
            if not is_in_math_range(indices[i]):
                new_indices = _best_effort_break_long_part_in_language(sub_part)
                assert new_indices[0] == 0 and new_indices[-1] == len(sub_part)
                if len(new_indices) > 2:
                    new_indices = [indices[i] + idx for idx in new_indices]
                    indices = indices[: i + 1] + new_indices[1:-1] + indices[i + 1 :]
                    continue
            new_indices = _best_effort_break_long_part_in_math(sub_part)
            assert new_indices[0] == 0 and new_indices[-1] == len(sub_part)
            if len(new_indices) > 2:
                update_math_range(new_indices[1], new_indices[-1])
                new_indices = [indices[i] + idx for idx in new_indices]
                indices = indices[: i + 1] + new_indices[1:-1] + indices[i + 1 :]
                continue
            i += 1
        else:
            i += 1
    return indices


def _try_to_break_very_long_parts(text: str, indices: List[int]) -> List[int]:
    assert indices[-1] == len(text)
    new_indices = [0]
    last = 0
    for i_part in range(len(indices) - 1):
        a, b = indices[i_part], indices[i_part + 1]
        plen = b - a
        if plen > 100:
            part = text[a:b]
            part_indices = _break_part_as_much_as_possible(part)
            for idx in part_indices[1:]:
                new_indices.append(last + idx)
            last = new_indices[-1]
        else:
            last = last + plen
            new_indices.append(last)
    return new_indices


def split_solution_inplace(reasoning_latex: str) -> List[int]:
    """Public entrypoint. Returns character-level boundary indices.

    ``[0, c_1, c_2, …, len(text)]`` where each ``[c_i, c_{i+1})`` is one step.
    Identical algorithm and outputs to VinePPO's ``split_solution_inplace``.
    """
    if not reasoning_latex:
        return [0]

    disguised_text, placeholders, _ = _replace_math_with_placeholders(reasoning_latex)
    indices = _naive_split_by_newline_and_period(disguised_text)
    indices = _split_tailing_periods_in_placeholders(disguised_text, indices, placeholders)
    indices, text = _split_newline_in_placeholders(disguised_text, indices, placeholders)
    indices = _try_to_break_very_long_parts(text, indices)
    assert text == reasoning_latex
    assert indices[-1] == len(text)
    return indices
