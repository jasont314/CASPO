"""Tests for ``caspo.utils.timing``.

These tests must pass on a CPU-only host (the timing utilities are required
to no-op when CUDA is unavailable).
"""

from __future__ import annotations

import os
import time

import pytest

from caspo.utils import timing as T


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """Wipe global accumulator + env var before each test."""
    monkeypatch.delenv("CASPO_TIMING", raising=False)
    T.MEAN_TIMERS.clear()
    yield
    T.MEAN_TIMERS.clear()


# ---------------------------------------------------------------------------
# is_timing_enabled
# ---------------------------------------------------------------------------


def test_is_timing_enabled_default_off():
    assert T.is_timing_enabled() is False


def test_is_timing_enabled_env_on(monkeypatch):
    monkeypatch.setenv("CASPO_TIMING", "1")
    assert T.is_timing_enabled() is True


def test_is_timing_enabled_env_other_value(monkeypatch):
    monkeypatch.setenv("CASPO_TIMING", "0")
    assert T.is_timing_enabled() is False
    monkeypatch.setenv("CASPO_TIMING", "true")
    # Only the literal "1" enables it; any other value stays off.
    assert T.is_timing_enabled() is False


# ---------------------------------------------------------------------------
# nvtx_range — must not crash whether CUDA is present or not
# ---------------------------------------------------------------------------


def test_nvtx_range_noop_when_disabled():
    # Default: env unset, enabled=None -> no-op.
    with T.nvtx_range("disabled-block"):
        pass
    # Even when explicitly enabled, must still not crash on a CPU-only host.
    with T.nvtx_range("explicit-enabled", enabled=True):
        pass


def test_nvtx_range_with_env_on(monkeypatch):
    monkeypatch.setenv("CASPO_TIMING", "1")
    # Should not raise even if CUDA is unavailable.
    with T.nvtx_range("env-on"):
        pass


# ---------------------------------------------------------------------------
# cuda_timer
# ---------------------------------------------------------------------------


def test_cuda_timer_noop_when_disabled():
    with T.cuda_timer("disabled") as h:
        pass
    # No CUDA / disabled -> holder should remain empty, no MEAN_TIMER entry.
    assert h == []
    assert "disabled" not in T.MEAN_TIMERS


def test_cuda_timer_no_crash_when_enabled_no_cuda(monkeypatch):
    monkeypatch.setenv("CASPO_TIMING", "1")
    with T.cuda_timer("attempt") as h:
        pass
    # On a CPU-only host this should silently no-op (holder empty).
    # If CUDA *is* available, holder will have one entry.
    try:
        import torch

        cuda_avail = torch.cuda.is_available()
    except Exception:
        cuda_avail = False
    if cuda_avail:
        assert len(h) == 1 and h[0] >= 0.0
        assert T.MEAN_TIMERS["attempt"].count == 1
    else:
        assert h == []
        assert "attempt" not in T.MEAN_TIMERS


# ---------------------------------------------------------------------------
# cpu_timer
# ---------------------------------------------------------------------------


def test_cpu_timer_noop_when_disabled():
    with T.cpu_timer("off") as h:
        time.sleep(0.001)
    assert h == []
    assert "off" not in T.MEAN_TIMERS


def test_cpu_timer_records_when_enabled():
    with T.cpu_timer("hot-path", enabled=True) as h:
        time.sleep(0.005)
    assert len(h) == 1
    assert h[0] >= 4.0  # ~5ms sleep, allow slack for clock granularity
    assert "hot-path" in T.MEAN_TIMERS
    assert T.MEAN_TIMERS["hot-path"].count == 1


def test_cpu_timer_no_accumulate():
    with T.cpu_timer("ephemeral", enabled=True, accumulate=False) as h:
        time.sleep(0.001)
    assert len(h) == 1
    assert "ephemeral" not in T.MEAN_TIMERS


def test_cpu_timer_env_var(monkeypatch):
    monkeypatch.setenv("CASPO_TIMING", "1")
    with T.cpu_timer("env-driven") as h:
        time.sleep(0.001)
    assert len(h) == 1
    assert "env-driven" in T.MEAN_TIMERS


# ---------------------------------------------------------------------------
# MeanTimer
# ---------------------------------------------------------------------------


def test_mean_timer_empty_summary():
    mt = T.MeanTimer("blank")
    s = mt.summary()
    assert s["count"] == 0
    assert s["mean_ms"] == 0.0
    assert s["p50_ms"] == 0.0
    assert s["p99_ms"] == 0.0


def test_mean_timer_basic_stats():
    mt = T.MeanTimer("samples")
    for x in [1.0, 2.0, 3.0, 4.0, 5.0]:
        mt.add(x)
    s = mt.summary()
    assert s["count"] == 5
    assert s["mean_ms"] == pytest.approx(3.0)
    assert s["min_ms"] == 1.0
    assert s["max_ms"] == 5.0
    assert s["total_ms"] == pytest.approx(15.0)
    # p50 of 5 sorted samples [1,2,3,4,5] (nearest-rank, 0-indexed mid) -> 3
    assert s["p50_ms"] == pytest.approx(3.0)
    # p99 with 5 samples lands on the last element.
    assert s["p99_ms"] == pytest.approx(5.0)


def test_mean_timer_reset():
    mt = T.MeanTimer("r")
    mt.add(10.0)
    mt.add(20.0)
    assert mt.count == 2
    mt.reset()
    assert mt.count == 0
    assert mt.summary()["count"] == 0


def test_module_level_mean_timers_global():
    # The module exposes a dict; cpu_timer should populate it.
    with T.cpu_timer("global-test", enabled=True):
        time.sleep(0.001)
    with T.cpu_timer("global-test", enabled=True):
        time.sleep(0.001)
    assert "global-test" in T.MEAN_TIMERS
    assert T.MEAN_TIMERS["global-test"].count == 2
    s = T.summary_all()
    assert "global-test" in s
    assert s["global-test"]["count"] == 2


def test_reset_all():
    with T.cpu_timer("a", enabled=True):
        pass
    with T.cpu_timer("b", enabled=True):
        pass
    assert T.MEAN_TIMERS["a"].count >= 1
    T.reset_all()
    assert T.MEAN_TIMERS["a"].count == 0
    assert T.MEAN_TIMERS["b"].count == 0
    # Entries themselves remain.
    assert set(T.MEAN_TIMERS.keys()) >= {"a", "b"}


# ---------------------------------------------------------------------------
# Zero-overhead-when-off sanity check
# ---------------------------------------------------------------------------


def test_disabled_path_does_not_touch_global_state():
    # Run a bunch of disabled timers; nothing should land in MEAN_TIMERS.
    for i in range(50):
        with T.cpu_timer(f"never-{i}"):
            pass
        with T.cuda_timer(f"never-cuda-{i}"):
            pass
        with T.nvtx_range(f"never-nvtx-{i}"):
            pass
    assert T.MEAN_TIMERS == {}
