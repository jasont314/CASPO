"""Step-level segmentation of tokenized responses.

Given a batch of response token ids and a token-id delimiter sequence (e.g.
the BPE encoding of "\\n\\n"), split each response into "reasoning steps":

  1. For each row b, restrict to the first ``valid_len[b]`` tokens.
  2. Greedily scan left-to-right for runs of tokens equal to
     ``delimiter_token_ids``. A match at position k attributes the delimiter
     tokens to the *current* step (they are the last tokens of step t), and
     opens a new step at k + L.
  3. Drop empty leading / trailing segments produced by delimiters at the
     very start or very end of the valid range.
  4. Iteratively merge any segment shorter than ``min_step_tokens`` into the
     *previous* segment. The first segment cannot be merged backward.
  5. If we end up with more than ``max_steps`` segments, fold every extra
     trailing segment into the ``max_steps``-th one so the terminal step
     always carries the final reward.

Outputs are expressed as per-token step ids (with -1 on masked positions),
per-row step counts, per-token boundary flags, and a padded ``step_lengths``
matrix used downstream to gather step-level values and broadcast step-level
advantages back to tokens. Pure tensor ops + Python ints; no tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import torch

from caspo.segmentation.latex_splitter import split_solution_inplace


@dataclass
class StepSegmentation:
    step_id: torch.Tensor           # [B, R] int64; -1 on masked tokens
    step_count: torch.Tensor        # [B] int64
    boundary_after: torch.Tensor    # [B, R] bool
    step_lengths: torch.Tensor      # [B, S_max] int64; 0 padding for absent steps


# ---------------------------------------------------------------------------
# Internal per-row segmentation (pure Python on CPU lists of ints).
# ---------------------------------------------------------------------------


def _find_delimiter_matches(
    ids: Sequence[int], delim: Sequence[int], valid_len: int
) -> List[int]:
    """Return start positions k (in [0, valid_len - L]) of non-overlapping
    delimiter matches, scanned greedily left-to-right."""
    L = len(delim)
    matches: List[int] = []
    if L == 0 or valid_len < L:
        return matches
    k = 0
    end = valid_len - L
    while k <= end:
        ok = True
        for j in range(L):
            if ids[k + j] != delim[j]:
                ok = False
                break
        if ok:
            matches.append(k)
            k += L  # non-overlapping
        else:
            k += 1
    return matches


def _segments_from_matches(
    matches: List[int], L: int, valid_len: int
) -> List[tuple[int, int]]:
    """Build raw [start, end] inclusive segments from delimiter match starts.

    Each match at k closes the current segment at k+L-1 (delimiter belongs to
    the current step) and opens the next at k+L. We ignore any *boundary*
    that would produce an empty leading segment (delimiter at the very start
    of the valid range): such a boundary is dropped wholesale, so the
    delimiter tokens themselves fall into the first real segment.
    Trailing-empty segments (delimiter right before EOS) are simply dropped
    because the trailing range [cur_start, valid_len-1] is empty.
    """
    segs: List[tuple[int, int]] = []
    cur_start = 0
    for k in matches:
        if k <= cur_start:
            # Boundary produces an empty preceding segment ([cur_start, k-1]
            # is empty when k == cur_start). Ignore the boundary entirely so
            # the delimiter tokens stay attached to the next real segment.
            # (cur_start stays put.)
            continue
        end = k + L - 1
        segs.append((cur_start, end))
        cur_start = k + L
    # trailing segment after the last accepted delimiter (or whole row if no
    # boundaries were accepted)
    if cur_start <= valid_len - 1:
        segs.append((cur_start, valid_len - 1))
    return segs


def _merge_short(
    segs: List[tuple[int, int]], min_step_tokens: int
) -> List[tuple[int, int]]:
    """Merge segments shorter than ``min_step_tokens`` into the previous
    segment. The first segment cannot be merged backward; if it is short,
    leave it alone.

    Single forward pass is sufficient: a short segment merged into ``out[-1]``
    only ever extends ``out[-1]``'s right endpoint, so re-scanning would never
    discover a new short segment. (The original implementation iterated until
    a fixed point — equivalent but O(N^2) on pathological inputs.)
    """
    if not segs:
        return segs
    if min_step_tokens <= 1:
        return segs  # min=1 never merges anything (e.g. tests pass min=1)
    out: List[tuple[int, int]] = [segs[0]]
    for s, e in segs[1:]:
        if e - s + 1 < min_step_tokens:
            ps, _ = out[-1]
            out[-1] = (ps, e)
        else:
            out.append((s, e))
    return out


def _cap_max_steps(
    segs: List[tuple[int, int]], max_steps: int, valid_len: int
) -> List[tuple[int, int]]:
    """If more than ``max_steps`` segments, fold all extra trailing segments
    into the ``max_steps``-th by extending its end to ``valid_len - 1``."""
    if max_steps <= 0 or len(segs) <= max_steps:
        return segs
    kept = segs[: max_steps - 1]
    # the max_steps-th segment starts where the (max_steps-1)-th ended + 1
    # but more simply: take the start of segs[max_steps - 1] and extend to end
    last_start = segs[max_steps - 1][0]
    kept.append((last_start, valid_len - 1))
    return kept


def _segment_one(
    ids: Sequence[int],
    valid_len: int,
    delim: Sequence[int],
    min_step_tokens: int,
    max_steps: int,
) -> List[tuple[int, int]]:
    if valid_len <= 0:
        return []
    matches = _find_delimiter_matches(ids, delim, valid_len)
    segs = _segments_from_matches(matches, len(delim), valid_len)
    if not segs:
        # delimiter eats the whole row (e.g. row is exactly the delimiter and
        # we dropped the leading empty segment). Fall back to one full step.
        segs = [(0, valid_len - 1)]
    segs = _merge_short(segs, min_step_tokens)
    segs = _cap_max_steps(segs, max_steps, valid_len)
    return segs


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def segment_responses_batch(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    delimiter_token_ids: List[int],
    *,
    min_step_tokens: int = 4,
    max_steps: int = 64,
) -> StepSegmentation:
    """Batched step segmentation.

    See module docstring for the algorithm.
    """
    if response_ids.dim() != 2 or response_mask.dim() != 2:
        raise ValueError(
            f"Expected 2D tensors, got response_ids.shape={tuple(response_ids.shape)} "
            f"response_mask.shape={tuple(response_mask.shape)}"
        )
    if response_ids.shape != response_mask.shape:
        raise ValueError(
            f"Shape mismatch: response_ids {tuple(response_ids.shape)} vs "
            f"response_mask {tuple(response_mask.shape)}"
        )
    if not delimiter_token_ids or all(t == 0 for t in delimiter_token_ids):
        # An all-zero or empty delimiter is almost certainly a bug at the
        # caller (e.g. tokenizing "" or a tokenizer that emits a leading 0
        # padding token). Refuse to segment — every row would either match
        # nowhere or match every position.
        raise ValueError(
            "delimiter_token_ids must be non-empty and contain at least one nonzero id; "
            f"got {delimiter_token_ids!r}"
        )

    B, R = response_ids.shape
    device = response_ids.device

    ids_cpu = response_ids.detach().to("cpu").to(torch.int64)
    mask_cpu = response_mask.detach().to("cpu").to(torch.int64)
    valid_lens = mask_cpu.sum(dim=1).tolist()

    delim = [int(t) for t in delimiter_token_ids]

    step_id = torch.full((B, R), -1, dtype=torch.int64)
    boundary_after = torch.zeros((B, R), dtype=torch.bool)
    step_count = torch.zeros((B,), dtype=torch.int64)
    rows_segs: List[List[tuple[int, int]]] = []
    s_max = 1
    for b in range(B):
        row_ids = ids_cpu[b].tolist()
        vl = int(valid_lens[b])
        segs = _segment_one(row_ids, vl, delim, min_step_tokens, max_steps)
        rows_segs.append(segs)
        step_count[b] = len(segs)
        for t, (s, e) in enumerate(segs):
            step_id[b, s : e + 1] = t
            boundary_after[b, e] = True
        if len(segs) > s_max:
            s_max = len(segs)

    step_lengths = torch.zeros((B, s_max), dtype=torch.int64)
    for b, segs in enumerate(rows_segs):
        for t, (s, e) in enumerate(segs):
            step_lengths[b, t] = e - s + 1

    return StepSegmentation(
        step_id=step_id.to(device, non_blocking=True),
        step_count=step_count.to(device, non_blocking=True),
        boundary_after=boundary_after.to(device, non_blocking=True),
        step_lengths=step_lengths.to(device, non_blocking=True),
    )


def segment_response(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    delimiter_token_ids: List[int],
    *,
    min_step_tokens: int = 4,
    max_steps: int = 64,
) -> StepSegmentation:
    """Single-sequence segmentation. Wraps inputs into [1, R] and unsqueezes."""
    if response_ids.dim() != 1 or response_mask.dim() != 1:
        raise ValueError(
            f"Expected 1D tensors, got response_ids.shape={tuple(response_ids.shape)} "
            f"response_mask.shape={tuple(response_mask.shape)}"
        )
    return segment_responses_batch(
        response_ids.unsqueeze(0),
        response_mask.unsqueeze(0),
        delimiter_token_ids,
        min_step_tokens=min_step_tokens,
        max_steps=max_steps,
    )


# ---------------------------------------------------------------------------
# LaTeX-aware text-based segmentation (for VinePPO-faithful reproduction).
# ---------------------------------------------------------------------------


def _per_token_char_spans(
    tokenizer: Any, ids: List[int]
) -> tuple[List[tuple[int, int]], str]:
    """Per-token ``(char_start, char_end)`` spans against the *full* decoded
    text, plus that text.

    Naively concatenating per-token decodes is wrong for BPE tokenizers that
    drop leading spaces in single-token decodes (Llama/Mistral/Qwen all do
    this — a token like ``" the"`` decodes to ``"the"`` on its own but to
    ``" the"`` inside a sequence). The LaTeX splitter's regexes (``\\s+[A-Z]``
    for sentence boundaries, etc.) need the original whitespace.

    Fast path (HF fast tokenizer): decode the full id list once, then re-encode
    *that text* with ``return_offsets_mapping=True`` to get per-token char
    offsets directly from the Rust tokenizer (~5x faster than per-token decode
    on long responses, and exact). If the round-trip is unstable (re-encoding
    produces a different number of tokens — rare for normal text but possible
    with special tokens or non-canonical id sequences), fall back to the
    pointer-walk method below.

    Slow path (fallback): decode once, then for each token decode the single
    id and locate the resulting piece via ``str.find`` from the running
    pointer. Leading whitespace between two tokens is attributed to the
    *next* token. Empty-piece / unfindable tokens get a zero-length span at
    the current pointer; the consumer inherits predecessor's step.
    """
    full = tokenizer.decode([int(t) for t in ids], skip_special_tokens=False)
    n = len(full)
    n_ids = len(ids)

    # Fast path: HF fast tokenizers expose offset_mapping directly.
    if getattr(tokenizer, "is_fast", False) and full:
        try:
            enc = tokenizer(
                full,
                add_special_tokens=False,
                return_offsets_mapping=True,
                return_attention_mask=False,
            )
            offsets = enc.get("offset_mapping")
            re_ids = enc.get("input_ids")
            if (
                offsets is not None
                and re_ids is not None
                and len(offsets) == n_ids
            ):
                # Convert offsets (Rust tuples) to plain Python tuples and
                # patch any zero-length leading-pointer mismatch by carrying
                # pos forward through whitespace gaps so spans tile contiguously
                # for the splitter consumer (mid-point membership doesn't care
                # about gaps, but downstream consumers do).
                spans: List[tuple[int, int]] = []
                pos = 0
                for cs, ce in offsets:
                    if ce <= cs:
                        spans.append((pos, pos))
                        continue
                    # Attribute any leading whitespace (cs > pos) to this token.
                    spans.append((pos, ce))
                    pos = ce
                if spans and pos < n:
                    s, _e = spans[-1]
                    spans[-1] = (s, n)
                return spans, full
        except Exception:
            # Fall through to the slow path below — never fail on a fast-path
            # quirk; correctness comes from the pointer walk.
            pass

    # Slow path: pointer walk over per-token decodes.
    spans = []
    pos = 0
    for tid in ids:
        piece = tokenizer.decode([int(tid)], skip_special_tokens=False)
        if not piece:
            spans.append((pos, pos))
            continue
        idx = full.find(piece, pos)
        if idx < 0:
            spans.append((pos, pos))
            continue
        end = idx + len(piece)
        spans.append((pos, end))
        pos = end
    if spans and pos < n:
        s, _e = spans[-1]
        spans[-1] = (s, n)
    return spans, full


def segment_responses_batch_latex_aware(
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    tokenizer: Any,
    *,
    min_step_tokens: int = 4,
    max_steps: int = 64,
) -> StepSegmentation:
    """Step segmentation via VinePPO's LaTeX-aware text splitter.

    Decodes each row to text, runs :func:`split_solution_inplace` to get
    character-level step boundaries, and maps those back to token indices
    via per-token decode spans.

    A token belongs to step k iff its character-span midpoint lies inside
    ``[char_indices[k], char_indices[k+1])``. Tokens that decode to the
    empty string (rare; some tokenizers can produce them at EOS) inherit
    their predecessor's step id.
    """
    if response_ids.dim() != 2 or response_mask.dim() != 2:
        raise ValueError(
            f"Expected 2D tensors, got response_ids.shape={tuple(response_ids.shape)} "
            f"response_mask.shape={tuple(response_mask.shape)}"
        )
    if response_ids.shape != response_mask.shape:
        raise ValueError(
            f"Shape mismatch: response_ids {tuple(response_ids.shape)} vs "
            f"response_mask {tuple(response_mask.shape)}"
        )

    B, R = response_ids.shape
    device = response_ids.device

    ids_cpu = response_ids.detach().to("cpu").to(torch.int64)
    mask_cpu = response_mask.detach().to("cpu").to(torch.int64)
    valid_lens = mask_cpu.sum(dim=1).tolist()

    step_id = torch.full((B, R), -1, dtype=torch.int64)
    boundary_after = torch.zeros((B, R), dtype=torch.bool)
    step_count = torch.zeros((B,), dtype=torch.int64)
    rows_lengths: List[List[int]] = []
    s_max = 1

    for b in range(B):
        vl = int(valid_lens[b])
        if vl <= 0:
            rows_lengths.append([])
            continue
        ids = ids_cpu[b, :vl].tolist()
        spans, text = _per_token_char_spans(tokenizer, ids)

        # Run the LaTeX-aware splitter on the *full* decoded text → char-level
        # boundaries. The splitter can raise AssertionError on malformed LaTeX
        # or pathological short inputs (whitespace-only, numbers-only — see
        # tests/test_latex_splitter.py::test_pure_whitespace_input). Fall back
        # to a *newline-aware* segmentation rather than "1 giant step", so a
        # well-formed multi-line response that just happens to fail an
        # internal assertion still gets meaningful step boundaries.
        try:
            char_indices = split_solution_inplace(text)
        except (AssertionError, IndexError, KeyError, ValueError):
            if not text:
                char_indices = [0]
            else:
                # Cut after each "\n"; segments are [0, end_of_line_0+1,
                # end_of_line_1+1, ..., len(text)] with the trailing
                # boundary being len(text) regardless of trailing newline.
                cuts = [0]
                for i, ch in enumerate(text):
                    if ch == "\n":
                        cuts.append(i + 1)
                if cuts[-1] != len(text):
                    cuts.append(len(text))
                # Drop zero-length leading duplicate (when text starts with a
                # newline → cuts = [0, 1, ...]; the [0,1) slice is fine).
                # Coalesce consecutive equal cuts (defensive).
                dedup = [cuts[0]]
                for c in cuts[1:]:
                    if c != dedup[-1]:
                        dedup.append(c)
                char_indices = dedup if len(dedup) >= 2 else [0, len(text)]
        # char_indices = [0, c1, c2, …, len(text)]; segments [c_i, c_{i+1}).
        bounded = char_indices
        # Build per-token step_id by walking tokens in char order.
        # Token t's char span midpoint determines step membership.
        cur_step = 0
        S = max(0, len(bounded) - 1)
        if S == 0:
            # Whole row is one step (text is empty or splitter degenerate).
            S = 1
            bounded = [0, max(len(text), 1)]
        token_step = [0] * vl
        for ti, (cs, ce) in enumerate(spans):
            if ce <= cs:
                # Empty-string / unfindable token — inherit predecessor.
                token_step[ti] = token_step[ti - 1] if ti > 0 else 0
                continue
            mid = (cs + ce - 1) / 2.0
            # Advance cur_step while mid >= bounded[cur_step+1]; cur_step in [0, S-1].
            while cur_step + 1 < S and mid >= bounded[cur_step + 1]:
                cur_step += 1
            token_step[ti] = cur_step

        # Apply min_step_tokens merge: short steps absorbed into the previous step.
        # (Only collapse forward; the very first short step stays.)
        # First, gather token spans per step.
        step_spans: List[tuple[int, int]] = []
        if vl > 0:
            cur = token_step[0]
            run_start = 0
            for ti in range(1, vl):
                if token_step[ti] != cur:
                    step_spans.append((run_start, ti - 1))
                    cur = token_step[ti]
                    run_start = ti
            step_spans.append((run_start, vl - 1))

        # Merge short segments into the previous one. A single forward pass
        # is sufficient (see ``_merge_short`` above for the argument).
        if min_step_tokens > 1 and len(step_spans) > 1:
            out_spans: List[tuple[int, int]] = [step_spans[0]]
            for s, e in step_spans[1:]:
                if e - s + 1 < min_step_tokens:
                    ps, _pe = out_spans[-1]
                    out_spans[-1] = (ps, e)
                else:
                    out_spans.append((s, e))
            step_spans = out_spans

        # Cap at max_steps: fold trailing extras into the max_steps-th segment.
        if max_steps > 0 and len(step_spans) > max_steps:
            kept = step_spans[: max_steps - 1]
            last_start = step_spans[max_steps - 1][0]
            kept.append((last_start, vl - 1))
            step_spans = kept

        # token_step (built earlier) is no longer used: step_id is written
        # directly from merged segments below.
        step_count[b] = len(step_spans)
        for t, (s, e) in enumerate(step_spans):
            step_id[b, s : e + 1] = t
            boundary_after[b, e] = True
        rows_lengths.append([e - s + 1 for s, e in step_spans])
        if len(step_spans) > s_max:
            s_max = len(step_spans)

    step_lengths = torch.zeros((B, s_max), dtype=torch.int64)
    for b, lengths in enumerate(rows_lengths):
        for t, L in enumerate(lengths):
            step_lengths[b, t] = L

    return StepSegmentation(
        step_id=step_id.to(device, non_blocking=True),
        step_count=step_count.to(device, non_blocking=True),
        boundary_after=boundary_after.to(device, non_blocking=True),
        step_lengths=step_lengths.to(device, non_blocking=True),
    )
