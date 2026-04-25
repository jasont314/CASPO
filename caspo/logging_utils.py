"""Async metrics logging utilities for CASPO.

This module provides opt-in, drop-in loggers that move wandb.log() calls
and JSONL file writes off the training hot path.

Two classes:

- ``AsyncMetricsLogger``: forwards metric dicts to ``wandb.log`` on a
  single background thread. Gracefully degrades to a no-op when wandb
  is not installed or no run is active.

- ``BatchedFileLogger``: appends JSON lines to a file, buffering writes
  and flushing every ``flush_every_n`` records (and on close).

Both are deliberately conservative: failures in the background thread
never propagate to the caller, and ``close()`` always drains pending
work before returning.
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


# Try wandb at import time but never hard-require it.
try:  # pragma: no cover - exercised by tests via monkeypatching
    import wandb as _wandb  # type: ignore
except Exception:  # ImportError or any wandb-side failure
    _wandb = None


def _wandb_run_active() -> bool:
    """True iff wandb is importable and a run is currently active."""
    if _wandb is None:
        return False
    try:
        return getattr(_wandb, "run", None) is not None
    except Exception:
        return False


class AsyncMetricsLogger:
    """Buffers metric dicts and forwards them to ``wandb.log`` off-thread.

    Usage::

        logger = AsyncMetricsLogger()
        logger.log({"loss": 0.7, "step": 1})
        ...
        logger.close()

    Safe to use whether or not wandb is installed. If wandb is not
    available (or no run is active), ``log()`` is a cheap no-op.
    """

    def __init__(
        self,
        wandb_module: Any = None,
        max_queue: int = 1024,
        thread_name: str = "caspo-wandb-logger",
    ) -> None:
        # Allow tests to inject a fake wandb module.
        self._wandb = wandb_module if wandb_module is not None else _wandb
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=thread_name)
        self._closed = False
        self._lock = threading.Lock()
        self._max_queue = max_queue
        self._inflight = 0
        self._inflight_cv = threading.Condition(self._lock)
        # Track dropped messages so callers / tests can sanity-check.
        self.dropped = 0
        self.errors = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Submit ``metrics`` for logging. Never blocks the caller meaningfully."""
        if self._closed:
            # Silently drop - the trainer is shutting down.
            self.dropped += 1
            return
        if not metrics:
            return

        # Take a shallow copy so the caller can mutate the dict freely.
        payload = dict(metrics)

        with self._inflight_cv:
            if self._inflight >= self._max_queue:
                # Backpressure: drop the oldest semantics by refusing the new
                # entry. We prefer drop over block to keep the hot path fast.
                self.dropped += 1
                return
            self._inflight += 1

        try:
            self._executor.submit(self._do_log, payload, step)
        except RuntimeError:
            # Executor already shut down between checks - count and move on.
            with self._inflight_cv:
                self._inflight -= 1
                self._inflight_cv.notify_all()
            self.dropped += 1

    def close(self, wait: bool = True, timeout: Optional[float] = 30.0) -> None:
        """Drain pending logs and shut down the background thread.

        ``timeout`` bounds how long we wait for in-flight submissions to
        drain; the executor itself is then shut down with ``wait=wait``.
        """
        with self._inflight_cv:
            if self._closed:
                return
            self._closed = True
            # Wait for queued items to be picked up & finished.
            if wait:
                deadline_pred = lambda: self._inflight == 0  # noqa: E731
                self._inflight_cv.wait_for(deadline_pred, timeout=timeout)

        try:
            self._executor.shutdown(wait=wait)
        except Exception:
            pass

    # Context-manager sugar.
    def __enter__(self) -> "AsyncMetricsLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _do_log(self, metrics: Dict[str, Any], step: Optional[int]) -> None:
        try:
            wb = self._wandb
            if wb is None:
                return
            run = getattr(wb, "run", None)
            if run is None:
                # No active run; silently skip but don't crash.
                return
            try:
                if step is None:
                    wb.log(metrics)
                else:
                    wb.log(metrics, step=step)
            except Exception:
                self.errors += 1
        finally:
            with self._inflight_cv:
                self._inflight -= 1
                self._inflight_cv.notify_all()


class BatchedFileLogger:
    """Append JSON lines to a file, buffering and flushing every N records.

    Usage::

        flog = BatchedFileLogger("train.jsonl", flush_every_n=10)
        flog.log({"step": 1, "loss": 0.5})
        ...
        flog.close()  # always drain on shutdown

    The file is opened in append mode; intermediate buffer lives in
    memory until ``flush_every_n`` records have arrived or ``flush()`` /
    ``close()`` is called. Errors are swallowed but counted in
    ``self.errors`` so callers can introspect.
    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        flush_every_n: int = 10,
        mode: str = "a",
        ensure_parent: bool = True,
    ) -> None:
        self.path = Path(path)
        if ensure_parent:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_every_n = max(1, int(flush_every_n))
        self._buffer: List[str] = []
        self._lock = threading.Lock()
        self._closed = False
        self.errors = 0
        # Open lazily so unit tests can construct without touching disk.
        self._fh = open(self.path, mode, encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            line = json.dumps(record, default=str)
        except Exception:
            self.errors += 1
            return

        with self._lock:
            self._buffer.append(line)
            if len(self._buffer) >= self.flush_every_n:
                self._flush_locked()

    def log_many(self, records: Iterable[Dict[str, Any]]) -> None:
        for r in records:
            self.log(r)

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_locked()
            try:
                self._fh.close()
            except Exception:
                self.errors += 1
            self._closed = True

    def __enter__(self) -> "BatchedFileLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    def _flush_locked(self) -> None:
        if not self._buffer:
            return
        try:
            self._fh.write("\n".join(self._buffer) + "\n")
            self._fh.flush()
        except Exception:
            self.errors += 1
        finally:
            self._buffer.clear()


__all__ = ["AsyncMetricsLogger", "BatchedFileLogger"]
