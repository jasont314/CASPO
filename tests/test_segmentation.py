"""Unit tests for caspo.segmentation.steps."""

from __future__ import annotations

import pytest
import torch

from caspo.segmentation import (
    StepSegmentation,
    segment_response,
    segment_responses_batch,
)


# A 2-token delimiter "\\n\\n"-like marker. Picked far away from regular ids
# so we don't accidentally collide with content tokens.
DELIM = [99, 99]


def _row(ids: list[int], pad_to: int | None = None, pad_value: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (response_ids, response_mask) for a single row.

    The mask is 1 over the actual length and 0 over padding.
    """
    R = pad_to if pad_to is not None else len(ids)
    assert R >= len(ids)
    ids_t = torch.full((R,), pad_value, dtype=torch.int64)
    ids_t[: len(ids)] = torch.tensor(ids, dtype=torch.int64)
    mask_t = torch.zeros((R,), dtype=torch.int64)
    mask_t[: len(ids)] = 1
    return ids_t, mask_t


# ---------------------------------------------------------------------------


def test_no_delimiter_one_step():
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    ids_t, mask_t = _row(ids)
    out = segment_response(ids_t, mask_t, DELIM, min_step_tokens=1, max_steps=64)
    assert isinstance(out, StepSegmentation)
    assert out.step_count.tolist() == [1]
    assert out.step_id[0].tolist() == [0] * len(ids)
    assert out.step_lengths[0, 0].item() == len(ids)
    # boundary only on the very last token
    assert out.boundary_after[0].tolist() == [False] * (len(ids) - 1) + [True]


def test_three_steps():
    # step 0: [1,2,3,4] (with closing delim 99,99 attributed to it)
    # step 1: [10,11,12,13] (with closing delim 99,99)
    # step 2: [20,21,22] (no trailing delim)
    ids = [1, 2, 3, 4, 99, 99, 10, 11, 12, 13, 99, 99, 20, 21, 22]
    ids_t, mask_t = _row(ids)
    out = segment_response(ids_t, mask_t, DELIM, min_step_tokens=1, max_steps=64)
    assert out.step_count.item() == 3
    expected_step_id = (
        [0] * 6  # 1..4 + delim
        + [1] * 6  # 10..13 + delim
        + [2] * 3  # 20..22
    )
    assert out.step_id[0].tolist() == expected_step_id
    assert out.step_lengths[0].tolist() == [6, 6, 3]
    # boundaries: positions 5, 11, 14
    boundaries = [i for i, b in enumerate(out.boundary_after[0].tolist()) if b]
    assert boundaries == [5, 11, 14]


def test_delimiter_at_start_dropped():
    # Leading delimiter -> empty leading segment is dropped, the delimiter
    # tokens themselves still belong to step 0 because we re-anchor.
    # Implementation detail: delim at k=0 produces no leading segment, the
    # next segment starts at k+L=2.
    ids = [99, 99, 1, 2, 3, 4, 5, 6]
    ids_t, mask_t = _row(ids)
    out = segment_response(ids_t, mask_t, DELIM, min_step_tokens=1, max_steps=64)
    # Only one real step: [1..6] occupying positions 2..7. Positions 0..1
    # should be assigned to step 0 as well? Per spec, "ignore boundaries
    # that produce empty leading segments", so the delimiter is not a
    # boundary — those tokens fall into step 0 as the first segment starts
    # at position 0.
    # Actually our implementation drops the *segment*, but cur_start advances
    # to k+L, so positions 0..1 are NOT covered by any segment. We expected
    # them to fall into step 0. Re-check: spec says "ignore boundaries that
    # produce empty leading segments" — i.e. the boundary at index 0 is
    # ignored, so step 0 should cover positions 0..7.
    # The current implementation does drop the segment but advances cur_start
    # past the delimiter. To honor the spec strictly we instead expect the
    # whole row to be one step. Assert that single-step interpretation.
    assert out.step_count.item() == 1
    assert out.step_id[0].tolist() == [0] * len(ids)
    assert out.step_lengths[0, 0].item() == len(ids)


def test_delimiter_at_end_dropped():
    # Trailing delimiter right before EOS -> trailing empty segment dropped.
    ids = [1, 2, 3, 4, 5, 6, 99, 99]
    ids_t, mask_t = _row(ids)
    out = segment_response(ids_t, mask_t, DELIM, min_step_tokens=1, max_steps=64)
    assert out.step_count.item() == 1
    assert out.step_id[0].tolist() == [0] * len(ids)
    assert out.step_lengths[0, 0].item() == len(ids)
    # boundary is at the last token (position 7), not before
    assert out.boundary_after[0, -1].item() is True


def test_min_step_tokens_merge():
    # Two segments separated by delim, second segment is short -> merge into
    # previous. min_step_tokens=4. step 1 will be [10,11] (len 2) which gets
    # merged into step 0.
    ids = [1, 2, 3, 4, 5, 99, 99, 10, 11]
    ids_t, mask_t = _row(ids)
    out = segment_response(ids_t, mask_t, DELIM, min_step_tokens=4, max_steps=64)
    assert out.step_count.item() == 1
    assert out.step_id[0].tolist() == [0] * len(ids)


def test_max_steps_cap():
    # Build a response with ~100 delimiter occurrences -> step_count should
    # be capped at max_steps=10 and the last step should gobble all the rest.
    parts = []
    for i in range(100):
        parts.extend([i + 1, i + 1])  # 2 content tokens per step
        parts.extend(DELIM)
    # ~600 tokens. Drop final delimiter to keep things simple.
    parts.extend([777, 777])
    ids = parts
    ids_t, mask_t = _row(ids)
    out = segment_response(ids_t, mask_t, DELIM, min_step_tokens=1, max_steps=10)
    assert out.step_count.item() == 10
    # all positions are covered (no -1 inside the valid region)
    assert (out.step_id[0] >= 0).all()
    # the last step should be much longer than any prior step (gobble effect)
    last_len = out.step_lengths[0, 9].item()
    other_lens = out.step_lengths[0, :9].tolist()
    assert last_len > max(other_lens)


def test_padded_batch_ragged_step_counts():
    # row 0: 1 step. row 1: 3 steps. row 2: 2 steps.
    row0 = [1, 2, 3, 4, 5]
    row1 = [1, 2, 3, 4, 99, 99, 5, 6, 7, 8, 99, 99, 9, 10, 11, 12]
    row2 = [1, 2, 3, 4, 99, 99, 5, 6, 7, 8]
    R = max(len(row0), len(row1), len(row2))
    ids0, m0 = _row(row0, pad_to=R)
    ids1, m1 = _row(row1, pad_to=R)
    ids2, m2 = _row(row2, pad_to=R)
    ids = torch.stack([ids0, ids1, ids2], dim=0)
    masks = torch.stack([m0, m1, m2], dim=0)
    out = segment_responses_batch(ids, masks, DELIM, min_step_tokens=1, max_steps=64)
    assert out.step_count.tolist() == [1, 3, 2]
    # S_max should be 3
    assert out.step_lengths.shape[1] == 3
    # padding zero for absent steps
    assert out.step_lengths[0].tolist() == [len(row0), 0, 0]
    assert out.step_lengths[2, 2].item() == 0


def test_zero_valid_len():
    # entirely masked row
    R = 8
    ids_t = torch.zeros((R,), dtype=torch.int64)
    mask_t = torch.zeros((R,), dtype=torch.int64)
    out = segment_response(ids_t, mask_t, DELIM, min_step_tokens=1, max_steps=64)
    assert out.step_count.item() == 0
    assert (out.step_id[0] == -1).all()
    assert not out.boundary_after[0].any()
    # step_lengths is [B, S_max] with S_max=1 (clamped), all zero
    assert out.step_lengths.shape[1] == 1
    assert out.step_lengths[0, 0].item() == 0


def test_boundary_after_alignment():
    # 3 steps, check that boundary_after fires only on the last token of
    # each segment.
    ids = [1, 2, 3, 4, 99, 99, 10, 11, 12, 13, 99, 99, 20, 21, 22]
    ids_t, mask_t = _row(ids)
    out = segment_response(ids_t, mask_t, DELIM, min_step_tokens=1, max_steps=64)
    step_id = out.step_id[0]
    boundary = out.boundary_after[0]
    valid = mask_t.bool()
    for t in range(out.step_count.item()):
        positions = (step_id == t) & valid
        idxs = positions.nonzero(as_tuple=False).flatten().tolist()
        # only the very last position of step t should have boundary=True
        last_pos = idxs[-1]
        for k in idxs:
            if k == last_pos:
                assert boundary[k].item() is True, f"step {t} last pos {k}"
            else:
                assert boundary[k].item() is False, f"step {t} interior pos {k}"
    # masked / out-of-valid positions never have boundary set
    assert not (boundary & (~valid)).any()


def test_step_id_consistency():
    # for every t, sum(step_id == t & mask) must equal step_lengths[b, t]
    R = 32
    row0 = [1, 2, 3, 4, 99, 99, 5, 6, 7, 8, 9, 99, 99, 10, 11, 12]
    row1 = [7, 8, 9, 10, 11, 12, 13, 14]
    ids0, m0 = _row(row0, pad_to=R)
    ids1, m1 = _row(row1, pad_to=R)
    ids = torch.stack([ids0, ids1], dim=0)
    masks = torch.stack([m0, m1], dim=0)
    out = segment_responses_batch(ids, masks, DELIM, min_step_tokens=1, max_steps=64)
    for b in range(ids.shape[0]):
        valid = masks[b].bool()
        for t in range(out.step_count[b].item()):
            n = ((out.step_id[b] == t) & valid).sum().item()
            assert n == out.step_lengths[b, t].item(), (
                f"row {b} step {t}: step_id count {n} vs step_lengths "
                f"{out.step_lengths[b, t].item()}"
            )


def test_multi_token_delimiter():
    # 3-token delimiter [50, 51, 52]. Must match all 3 in sequence.
    delim3 = [50, 51, 52]
    ids = [
        1, 2, 3, 4,
        50, 51, 52,        # full match -> end of step 0
        10, 11, 12, 13,
        50, 51,            # partial match, NOT a boundary
        14, 15,
        50, 51, 52,        # full match -> end of step 1
        20, 21, 22,
    ]
    ids_t, mask_t = _row(ids)
    out = segment_response(ids_t, mask_t, delim3, min_step_tokens=1, max_steps=64)
    assert out.step_count.item() == 3
    # step lengths: 4 + 3 = 7 ; 4 + 2 + 2 + 3 = 11 ; 3
    assert out.step_lengths[0].tolist() == [7, 11, 3]
    # boundaries at positions 6, 17, 20
    boundaries = [i for i, b in enumerate(out.boundary_after[0].tolist()) if b]
    assert boundaries == [6, 17, 20]


def test_empty_delimiter_raises():
    ids_t, mask_t = _row([1, 2, 3])
    with pytest.raises(ValueError):
        segment_response(ids_t, mask_t, [], min_step_tokens=1, max_steps=64)
    with pytest.raises(ValueError):
        segment_response(ids_t, mask_t, [0, 0], min_step_tokens=1, max_steps=64)


# ---------------------------------------------------------------------------
# Additional audit-focused coverage
# ---------------------------------------------------------------------------


def test_token_delimiter_multi_token_sequence_no_overlap():
    """A long multi-token delimiter must match only on full sequences, never
    on overlapping prefixes of the next match window. We seed adversarial
    near-misses (a partial prefix immediately followed by another one) to
    guard against off-by-one in the non-overlapping advance step.
    """
    delim4 = [70, 71, 72, 73]
    ids = [
        # step 0: 5 content tokens then full delim
        1, 2, 3, 4, 5,
        70, 71, 72, 73,
        # step 1: a partial prefix that should NOT trigger a boundary
        # 70, 71 (partial) then content, then 70, 71, 72 (still partial)
        70, 71, 100,
        70, 71, 72, 200,
        # then the real full delim
        70, 71, 72, 73,
        # step 2: trailing content
        300, 301, 302, 303, 304,
    ]
    ids_t, mask_t = _row(ids)
    out = segment_response(ids_t, mask_t, delim4, min_step_tokens=1, max_steps=64)
    assert out.step_count.item() == 3
    # step lengths: 5+4=9, 3+4+4=11, 5
    assert out.step_lengths[0, :3].tolist() == [9, 11, 5]
    # Boundaries land only at the last delim token of each full match,
    # i.e. positions 8, 19, 24 (indices of the 4th delim token & last content).
    boundaries = [i for i, b in enumerate(out.boundary_after[0].tolist()) if b]
    assert boundaries == [8, 19, 24]
    # No boundary inside the partial-prefix near-misses (positions 9..15).
    for k in range(9, 16):
        assert not out.boundary_after[0, k].item(), (
            f"unexpected boundary inside partial-prefix region at {k}"
        )


def test_token_delimiter_multi_token_back_to_back_matches():
    """Two full multi-token delimiter occurrences with no content in between
    create one empty-leading / empty-trailing segment that must be dropped.
    The whole row should collapse to a single step covering all tokens.
    """
    delim3 = [40, 41, 42]
    # Just two full delims with no content between them and no content around.
    ids = [40, 41, 42, 40, 41, 42]
    ids_t, mask_t = _row(ids)
    out = segment_response(ids_t, mask_t, delim3, min_step_tokens=1, max_steps=64)
    # Per the spec, leading-empty + trailing-empty drop yields one full-row step.
    assert out.step_count.item() == 1
    assert out.step_id[0].tolist() == [0] * len(ids)
    assert out.step_lengths[0, 0].item() == len(ids)


def test_latex_aware_empty_response():
    """A fully-masked (empty) response under latex_aware must yield 0 steps
    and -1 step ids everywhere (matches the token-delimiter zero-len behavior).
    """
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM"
        )
    except Exception as e:
        pytest.skip(f"tiny tokenizer unavailable: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from caspo.segmentation import segment_responses_batch_latex_aware

    R = 6
    ids = torch.zeros(1, R, dtype=torch.long)
    mask = torch.zeros(1, R, dtype=torch.long)
    seg = segment_responses_batch_latex_aware(
        ids, mask, tok, min_step_tokens=2, max_steps=8
    )
    assert seg.step_count.item() == 0
    assert (seg.step_id == -1).all()
    assert not seg.boundary_after.any()
    # step_lengths is [B, S_max]; S_max must be at least 1 (clamped) and zero.
    assert seg.step_lengths.shape[0] == 1
    assert seg.step_lengths.shape[1] >= 1
    assert seg.step_lengths.sum().item() == 0


def test_latex_aware_whitespace_only_response():
    """A response whose decoded text is purely whitespace must not crash
    and must produce exactly one step covering all valid tokens.
    """
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM"
        )
    except Exception as e:
        pytest.skip(f"tiny tokenizer unavailable: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from caspo.segmentation import segment_responses_batch_latex_aware

    # Encode a few whitespace-only strings and feed through the wrapper.
    for text in ("   ", "\n\n\n", " \t \n "):
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        if int(mask.sum()) == 0:
            # Tokenizer may emit zero tokens for pure whitespace; skip those.
            continue
        seg = segment_responses_batch_latex_aware(
            ids, mask, tok, min_step_tokens=2, max_steps=8
        )
        # One step covering everything.
        assert seg.step_count.item() == 1, (
            f"text={text!r} produced {seg.step_count.item()} steps"
        )
        valid_len = int(mask[0].sum())
        # Step lengths sum must equal valid_len.
        assert int(seg.step_lengths[0].sum().item()) == valid_len
        # All valid tokens belong to step 0.
        valid = mask[0].bool()
        assert (seg.step_id[0][valid] == 0).all()
        # Boundary fires exactly once, on the last valid token.
        last_valid = int(valid.nonzero(as_tuple=False).flatten()[-1])
        assert seg.boundary_after[0, last_valid].item() is True
        assert seg.boundary_after[0].sum().item() == 1


def test_max_steps_cap_with_thousands_of_delimiters():
    """Stress: 2000 delimiters, max_steps=8. The 8-th step must absorb all
    trailing content and step_id must cover every valid position.
    """
    parts: list[int] = []
    n_delims = 2000
    for i in range(n_delims):
        parts.extend([i % 50 + 1, (i + 1) % 50 + 1])  # 2 content tokens
        parts.extend(DELIM)
    parts.extend([777, 778, 779])  # trailing content (no final delim)
    ids_t, mask_t = _row(parts)
    out = segment_response(ids_t, mask_t, DELIM, min_step_tokens=1, max_steps=8)
    assert out.step_count.item() == 8
    # step_lengths must sum to total tokens (every position covered exactly once).
    assert out.step_lengths[0].sum().item() == len(parts)
    # No -1 anywhere on the valid range.
    assert (out.step_id[0][: len(parts)] >= 0).all()
    # The last (8-th) step must dominate length-wise (gobble effect).
    last_len = out.step_lengths[0, 7].item()
    other_lens = out.step_lengths[0, :7].tolist()
    assert last_len > sum(other_lens), (
        f"expected the capped final step to absorb most tokens, "
        f"got last={last_len} vs others={other_lens}"
    )
    # Exactly max_steps boundary flags.
    assert out.boundary_after[0].sum().item() == 8


def test_min_step_tokens_groups_of_varying_lengths():
    """G=6 groups separated by delimiters with lengths [5, 1, 7, 2, 1, 6].
    With min_step_tokens=4, the 1-token, 2-token, 1-token segments must merge
    into their predecessor. Final layout: [5, 1+7=8, 2+1+6=9] — wait, the
    1-token group is the second segment, so it merges into the first
    (length 5+1=6); the 7-token segment stays; the 2 merges into 7 (=9);
    the 1 merges into 9 (=10); the final 6 stays. Result: [6, 10, 6].
    """
    group_lens = [5, 1, 7, 2, 1, 6]
    ids: list[int] = []
    next_tok = 1
    for gi, L in enumerate(group_lens):
        for _ in range(L):
            ids.append(next_tok)
            next_tok = next_tok + 1 if next_tok < 50 else 1
        if gi < len(group_lens) - 1:
            ids.extend(DELIM)
    ids_t, mask_t = _row(ids)
    out = segment_response(ids_t, mask_t, DELIM, min_step_tokens=4, max_steps=64)

    # Pre-merge segments (each delim attributed to the preceding step):
    #   [5+2=7, 1+2=3, 7+2=9, 2+2=4, 1+2=3, 6]
    # After iterative merge with min=4:
    #   - seg1 (3) < 4 → merge into seg0: [10, 9, 4, 3, 6]
    #   - seg3 (3) < 4 → merge into seg2 (which is now '4'): [10, 9, 7, 6]
    #     (note: current seg index 2 = 4; seg 3 = 3 → merge → 4+3=7;
    #      then re-scan: all >= 4 → done)
    # Expected: 4 steps with lengths [10, 9, 7, 6].
    assert out.step_count.item() == 4
    assert out.step_lengths[0, :4].tolist() == [10, 9, 7, 6]
    # Sum equals total tokens.
    assert out.step_lengths[0].sum().item() == len(ids)
    # All valid positions covered.
    assert (out.step_id[0][: len(ids)] >= 0).all()
    # Boundary count equals step count.
    assert out.boundary_after[0].sum().item() == out.step_count.item()


def test_min_step_tokens_first_segment_short_unmerged():
    """The first segment cannot be merged backward: even if it's shorter
    than min_step_tokens, it must stay as step 0.
    """
    # Layout: [2-token group, 8-token group, 8-token group]
    # First segment is 2 tokens (< min=4) but has no predecessor → stays as is.
    ids = [
        1, 2,                       # short first group
        99, 99,                     # delim attributed to step 0 → 4 tokens
        10, 11, 12, 13, 14, 15, 16, 17,
        99, 99,                     # delim attributed to step 1 → 10 tokens
        20, 21, 22, 23, 24, 25, 26, 27,
    ]
    ids_t, mask_t = _row(ids)
    out = segment_response(ids_t, mask_t, DELIM, min_step_tokens=4, max_steps=64)
    # Pre-merge: [4, 10, 8]. seg0=4 is not < 4, so all stay.
    assert out.step_count.item() == 3
    assert out.step_lengths[0, :3].tolist() == [4, 10, 8]


def test_min_step_tokens_disabled_with_one():
    """min_step_tokens=1 must never merge anything."""
    group_lens = [3, 1, 2, 1, 1, 4]
    ids: list[int] = []
    next_tok = 1
    for gi, L in enumerate(group_lens):
        for _ in range(L):
            ids.append(next_tok)
            next_tok = next_tok + 1 if next_tok < 50 else 1
        if gi < len(group_lens) - 1:
            ids.extend(DELIM)
    ids_t, mask_t = _row(ids)
    out = segment_response(ids_t, mask_t, DELIM, min_step_tokens=1, max_steps=64)
    # Pre-merge lengths (delim attributed to preceding step):
    #   [3+2=5, 1+2=3, 2+2=4, 1+2=3, 1+2=3, 4] = [5, 3, 4, 3, 3, 4]
    # min=1 → nothing merges.
    assert out.step_count.item() == 6
    assert out.step_lengths[0, :6].tolist() == [5, 3, 4, 3, 3, 4]
