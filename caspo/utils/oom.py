"""OOM handling utilities for CASPO training.

Pure infrastructure: provides retry, guard, and micro-batch fallback helpers
around CUDA out-of-memory errors. None of these helpers wire themselves into
any trainer; callers are expected to invoke them explicitly.
"""

from __future__ import annotations

import contextlib
import gc
from typing import Any, Callable, Optional

try:
    import torch

    _OOM_ERRORS: tuple = (torch.cuda.OutOfMemoryError, RuntimeError)
except Exception:  # pragma: no cover - torch should always be present in env
    torch = None  # type: ignore[assignment]
    _OOM_ERRORS = (RuntimeError,)


def _is_oom(exc: BaseException) -> bool:
    """Return True if exc looks like a CUDA OOM error.

    Catches torch.cuda.OutOfMemoryError directly, plus the legacy
    `RuntimeError("CUDA out of memory ...")` form for older torch versions.
    """
    if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return "out of memory" in msg or "cuda oom" in msg
    return False


def _empty_cache() -> None:
    """Best-effort GC + CUDA cache flush. Safe to call without CUDA."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def oom_retry(
    fn: Callable[[], Any],
    max_retries: int = 2,
    on_oom: Optional[Callable[[int], None]] = None,
) -> Any:
    """Call ``fn()``; on CUDA OOM, empty cache and retry up to ``max_retries`` times.

    Args:
        fn: zero-arg callable to invoke.
        max_retries: number of retries after the first failure (so total
            attempts = max_retries + 1).
        on_oom: optional callback invoked as ``on_oom(attempt_index)`` each
            time an OOM is caught, before the retry. Caller can use this to
            halve a batch size, log, etc. ``attempt_index`` is 0-based and
            counts the attempt that JUST failed.

    Returns:
        Whatever ``fn()`` returns on the first successful attempt.

    Raises:
        The last OOM error if all retries are exhausted.
        Any non-OOM exception is re-raised immediately without retry.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except _OOM_ERRORS as exc:
            if not _is_oom(exc):
                raise
            last_exc = exc
            _empty_cache()
            if on_oom is not None:
                try:
                    on_oom(attempt)
                except Exception:
                    pass
            if attempt >= max_retries:
                break
    assert last_exc is not None
    raise last_exc


@contextlib.contextmanager
def with_oom_guard():
    """Context manager: on OOM inside the block, empty_cache then re-raise.

    Useful for ensuring that a transient OOM does not leave allocator state
    polluted for the next caller, while still propagating the error so the
    caller can decide what to do.
    """
    try:
        yield
    except _OOM_ERRORS as exc:
        if _is_oom(exc):
            _empty_cache()
        raise


def safe_micro_batch(
    forward_fn: Callable[[Any], Any],
    batch: Any,
    max_microbatch: Optional[int] = None,
) -> list:
    """Run ``forward_fn`` over ``batch``, halving the slice on OOM.

    The batch is sliced along axis 0. On CUDA OOM, the current slice is
    halved and retried recursively. If a single-example slice OOMs, the
    error is propagated.

    Args:
        forward_fn: callable taking a sub-batch and returning some result.
        batch: indexable, has ``len()``; first dim is the batch dim.
        max_microbatch: optional initial cap; if set, batch is split into
            chunks of at most this size before any OOM occurs.

    Returns:
        List of per-chunk results in order.
    """
    n = len(batch)
    if n == 0:
        return []

    chunk = n if max_microbatch is None else min(max_microbatch, n)

    results: list = []
    i = 0
    while i < n:
        end = min(i + chunk, n)
        sub = batch[i:end]
        try:
            with with_oom_guard():
                results.append(forward_fn(sub))
            i = end
        except _OOM_ERRORS as exc:
            if not _is_oom(exc):
                raise
            if (end - i) <= 1:
                # Cannot shrink further.
                raise
            chunk = max(1, (end - i) // 2)
            # retry same i with smaller chunk
            continue
    return results
