"""Tests for caspo.utils.oom — retry, guard, micro-batch fallback."""

from __future__ import annotations

import pytest
import torch

from caspo.utils.oom import oom_retry, safe_micro_batch, with_oom_guard


def _make_oom() -> BaseException:
    """Construct a torch CUDA OOM error if available, else a RuntimeError."""
    if hasattr(torch.cuda, "OutOfMemoryError"):
        return torch.cuda.OutOfMemoryError("CUDA out of memory. (test)")
    return RuntimeError("CUDA out of memory. (test)")


# ---------------------------------------------------------------------------
# oom_retry
# ---------------------------------------------------------------------------


def test_oom_retry_succeeds_on_first_call():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        return "ok"

    assert oom_retry(fn) == "ok"
    assert calls["n"] == 1


def test_oom_retry_retries_after_oom_then_succeeds():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _make_oom()
        return 42

    assert oom_retry(fn, max_retries=2) == 42
    assert calls["n"] == 2


def test_oom_retry_invokes_on_oom_callback():
    seen: list[int] = []
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _make_oom()
        return "done"

    def on_oom(attempt: int) -> None:
        seen.append(attempt)

    assert oom_retry(fn, max_retries=3, on_oom=on_oom) == "done"
    assert seen == [0, 1]
    assert calls["n"] == 3


def test_oom_retry_exhausts_and_reraises():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise _make_oom()

    with pytest.raises((torch.cuda.OutOfMemoryError, RuntimeError)) as ei:
        oom_retry(fn, max_retries=2)
    assert "out of memory" in str(ei.value).lower()
    # 1 initial + 2 retries
    assert calls["n"] == 3


def test_oom_retry_propagates_non_oom_immediately():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise ValueError("not an OOM")

    with pytest.raises(ValueError):
        oom_retry(fn, max_retries=5)
    assert calls["n"] == 1


def test_oom_retry_does_not_swallow_runtime_error_unrelated():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        raise RuntimeError("device-side assert: shape mismatch")

    with pytest.raises(RuntimeError):
        oom_retry(fn, max_retries=3)
    assert calls["n"] == 1


def test_oom_retry_swallows_callback_errors():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _make_oom()
        return "ok"

    def bad_cb(attempt: int) -> None:
        raise RuntimeError("callback boom")

    # Callback errors must not block the retry path.
    assert oom_retry(fn, max_retries=2, on_oom=bad_cb) == "ok"


# ---------------------------------------------------------------------------
# with_oom_guard
# ---------------------------------------------------------------------------


def test_with_oom_guard_reraises_oom():
    with pytest.raises((torch.cuda.OutOfMemoryError, RuntimeError)) as ei:
        with with_oom_guard():
            raise _make_oom()
    assert "out of memory" in str(ei.value).lower()


def test_with_oom_guard_passes_through_other_exceptions():
    with pytest.raises(ValueError):
        with with_oom_guard():
            raise ValueError("nope")


def test_with_oom_guard_no_error():
    with with_oom_guard():
        x = 1 + 1
    assert x == 2


# ---------------------------------------------------------------------------
# safe_micro_batch
# ---------------------------------------------------------------------------


def test_safe_micro_batch_no_oom():
    batch = list(range(8))
    seen = []

    def forward(sub):
        seen.append(list(sub))
        return sum(sub)

    out = safe_micro_batch(forward, batch)
    assert out == [sum(range(8))]
    assert seen == [list(range(8))]


def test_safe_micro_batch_halves_on_oom():
    batch = list(range(8))
    state = {"oom_remaining": 1}
    seen_sizes: list[int] = []

    def forward(sub):
        seen_sizes.append(len(sub))
        # OOM the first full-batch attempt; succeed on the halved chunks.
        if len(sub) == 8 and state["oom_remaining"] > 0:
            state["oom_remaining"] -= 1
            raise _make_oom()
        return sum(sub)

    out = safe_micro_batch(forward, batch)
    # After halving from 8 -> 4, two chunks of 4 should succeed.
    assert sum(out) == sum(batch)
    assert seen_sizes[0] == 8  # initial failed attempt
    assert all(sz == 4 for sz in seen_sizes[1:])


def test_safe_micro_batch_recursive_halving():
    batch = list(range(8))
    seen_sizes: list[int] = []
    # OOM at sizes 8 and 4, succeed at size 2.
    fail_sizes = {8, 4}
    failed_once: dict[int, bool] = {}

    def forward(sub):
        seen_sizes.append(len(sub))
        sz = len(sub)
        if sz in fail_sizes and not failed_once.get(sz, False):
            failed_once[sz] = True
            raise _make_oom()
        return sum(sub)

    out = safe_micro_batch(forward, batch)
    assert sum(out) == sum(batch)
    # We should have attempted 8 -> 4 -> 2 at minimum.
    assert 8 in seen_sizes and 4 in seen_sizes and 2 in seen_sizes


def test_safe_micro_batch_propagates_at_size_one():
    batch = [0, 1, 2, 3]

    def forward(sub):
        # Always OOM, even at micro_batch=1.
        raise _make_oom()

    with pytest.raises((torch.cuda.OutOfMemoryError, RuntimeError)):
        safe_micro_batch(forward, batch)


def test_safe_micro_batch_respects_max_microbatch():
    batch = list(range(10))
    seen_sizes: list[int] = []

    def forward(sub):
        seen_sizes.append(len(sub))
        return sum(sub)

    out = safe_micro_batch(forward, batch, max_microbatch=3)
    assert sum(out) == sum(batch)
    # 10 split with cap=3 -> [3,3,3,1].
    assert seen_sizes == [3, 3, 3, 1]


def test_safe_micro_batch_empty():
    assert safe_micro_batch(lambda sub: 1, []) == []


def test_safe_micro_batch_propagates_non_oom():
    batch = [0, 1, 2, 3]

    def forward(sub):
        raise ValueError("bad input")

    with pytest.raises(ValueError):
        safe_micro_batch(forward, batch)
