"""Tests for caspo.logging_utils.

These tests verify:

1. AsyncMetricsLogger doesn't crash when wandb is missing/unavailable.
2. AsyncMetricsLogger forwards every submitted log to a fake wandb
   before close() returns (no message loss on shutdown).
3. AsyncMetricsLogger.close() doesn't deadlock under heavy submission.
4. BatchedFileLogger buffers writes and flushes every N records and on
   close, with no message loss.
5. BatchedFileLogger handles non-serializable values without crashing.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from caspo.logging_utils import AsyncMetricsLogger, BatchedFileLogger


class _FakeWandb:
    """Minimal stand-in for the wandb module."""

    def __init__(self, run_active: bool = True, slow: float = 0.0):
        self.run = object() if run_active else None
        self.calls = []
        self._lock = threading.Lock()
        self._slow = slow

    def log(self, metrics, step=None):
        if self._slow:
            time.sleep(self._slow)
        with self._lock:
            self.calls.append((dict(metrics), step))


# ---------------------------------------------------------------------------
# AsyncMetricsLogger
# ---------------------------------------------------------------------------

def test_async_logger_without_wandb_does_not_crash():
    # wandb_module=None forces the "no wandb installed" path even if it is.
    logger = AsyncMetricsLogger(wandb_module=None)
    for i in range(50):
        logger.log({"loss": float(i), "step": i})
    logger.close(timeout=5.0)
    # Nothing should have crashed. Errors counter stays at 0 because we
    # never even try to call wandb.log when the module is None.
    assert logger.errors == 0


def test_async_logger_with_inactive_run_does_not_crash():
    fake = _FakeWandb(run_active=False)
    logger = AsyncMetricsLogger(wandb_module=fake)
    for i in range(20):
        logger.log({"x": i})
    logger.close(timeout=5.0)
    assert fake.calls == []
    assert logger.errors == 0


def test_async_logger_forwards_all_messages_before_close():
    fake = _FakeWandb(run_active=True)
    logger = AsyncMetricsLogger(wandb_module=fake)
    n = 200
    for i in range(n):
        logger.log({"i": i}, step=i)
    logger.close(timeout=10.0)
    assert len(fake.calls) == n, f"lost messages: got {len(fake.calls)} of {n}"
    # Order is preserved because the executor has a single worker.
    for idx, (payload, step) in enumerate(fake.calls):
        assert payload == {"i": idx}
        assert step == idx


def test_async_logger_close_does_not_deadlock_with_slow_backend():
    fake = _FakeWandb(run_active=True, slow=0.001)
    logger = AsyncMetricsLogger(wandb_module=fake)
    for i in range(50):
        logger.log({"i": i})
    # Must return promptly even though each log takes ~1ms.
    t0 = time.time()
    logger.close(timeout=10.0)
    assert time.time() - t0 < 5.0
    assert len(fake.calls) == 50


def test_async_logger_log_after_close_is_noop():
    fake = _FakeWandb(run_active=True)
    logger = AsyncMetricsLogger(wandb_module=fake)
    logger.log({"a": 1})
    logger.close(timeout=5.0)
    logger.log({"a": 2})  # should be silently dropped
    assert len(fake.calls) == 1
    assert logger.dropped >= 1


def test_async_logger_handles_wandb_log_exception():
    class Boom(_FakeWandb):
        def log(self, metrics, step=None):
            raise RuntimeError("wandb is unhappy")

    fake = Boom(run_active=True)
    logger = AsyncMetricsLogger(wandb_module=fake)
    for i in range(5):
        logger.log({"i": i})
    logger.close(timeout=5.0)
    # No crash; errors recorded.
    assert logger.errors == 5


def test_async_logger_context_manager(tmp_path):
    fake = _FakeWandb(run_active=True)
    with AsyncMetricsLogger(wandb_module=fake) as logger:
        logger.log({"v": 1})
        logger.log({"v": 2})
    assert len(fake.calls) == 2


# ---------------------------------------------------------------------------
# BatchedFileLogger
# ---------------------------------------------------------------------------

def test_batched_file_logger_flushes_on_threshold(tmp_path: Path):
    p = tmp_path / "train.jsonl"
    flog = BatchedFileLogger(p, flush_every_n=5)
    try:
        for i in range(4):
            flog.log({"i": i})
        # Below threshold - file should have nothing flushed yet.
        assert p.read_text() == ""
        flog.log({"i": 4})  # 5th record triggers flush
        text = p.read_text()
        assert text.count("\n") == 5
    finally:
        flog.close()


def test_batched_file_logger_flushes_on_close(tmp_path: Path):
    p = tmp_path / "train.jsonl"
    flog = BatchedFileLogger(p, flush_every_n=100)
    for i in range(7):
        flog.log({"i": i})
    flog.close()
    lines = [json.loads(line) for line in p.read_text().splitlines()]
    assert lines == [{"i": i} for i in range(7)]


def test_batched_file_logger_no_loss_under_many_writes(tmp_path: Path):
    p = tmp_path / "train.jsonl"
    n = 137
    with BatchedFileLogger(p, flush_every_n=10) as flog:
        for i in range(n):
            flog.log({"i": i, "loss": i * 0.1})
    lines = p.read_text().splitlines()
    assert len(lines) == n
    parsed = [json.loads(l) for l in lines]
    assert [r["i"] for r in parsed] == list(range(n))


def test_batched_file_logger_handles_nonserializable(tmp_path: Path):
    p = tmp_path / "train.jsonl"
    flog = BatchedFileLogger(p, flush_every_n=2)
    flog.log({"good": 1})

    class Weird:
        def __repr__(self):
            return "<weird>"

    # Uses default=str so it should serialize without bumping errors.
    flog.log({"obj": Weird()})
    flog.close()
    lines = p.read_text().splitlines()
    assert len(lines) == 2


def test_batched_file_logger_log_after_close_is_noop(tmp_path: Path):
    p = tmp_path / "train.jsonl"
    flog = BatchedFileLogger(p, flush_every_n=1)
    flog.log({"i": 0})
    flog.close()
    flog.log({"i": 1})  # silently ignored
    assert len(p.read_text().splitlines()) == 1
