"""Timing instrumentation utilities for CASPO.

All utilities are opt-in: they fire only when ``CASPO_TIMING=1`` env var is set
or when ``enabled=True`` is passed explicitly. When disabled they are zero-overhead
no-ops (the context managers immediately yield, no syscalls, no CUDA calls).

Public API:
    nvtx_range(name, enabled=None) -> context manager
    cuda_timer(name, enabled=None, accumulate=True) -> context manager
    cpu_timer(name, enabled=None, accumulate=True) -> context manager
    MeanTimer -> accumulator with .add(ms), .summary() -> dict
    MEAN_TIMERS -> dict[str, MeanTimer] global accumulator
    is_timing_enabled() -> bool
"""

from __future__ import annotations

import os
import statistics
from contextlib import contextmanager
from time import perf_counter
from typing import Dict, Optional

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch should always be present in CASPO
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def is_timing_enabled() -> bool:
    """Check whether timing is enabled via the CASPO_TIMING env var."""
    return os.environ.get("CASPO_TIMING", "0") == "1"


def _resolve_enabled(enabled: Optional[bool]) -> bool:
    if enabled is None:
        return is_timing_enabled()
    return bool(enabled)


def _cuda_available() -> bool:
    return _HAS_TORCH and torch is not None and torch.cuda.is_available()


class MeanTimer:
    """Accumulates timing samples (in ms) and reports mean / p50 / p99."""

    __slots__ = ("name", "_samples")

    def __init__(self, name: str) -> None:
        self.name = name
        self._samples: list[float] = []

    def add(self, ms: float) -> None:
        self._samples.append(float(ms))

    def reset(self) -> None:
        self._samples.clear()

    @property
    def count(self) -> int:
        return len(self._samples)

    def summary(self) -> Dict[str, float]:
        """Return a dict with count/mean/p50/p99/min/max/total in ms.

        Returns zeros (with count=0) if no samples have been recorded.
        """
        if not self._samples:
            return {
                "name": self.name,  # type: ignore[dict-item]
                "count": 0,
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p99_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "total_ms": 0.0,
            }
        srt = sorted(self._samples)
        n = len(srt)
        # Nearest-rank percentiles (clamped).
        p50_idx = max(0, min(n - 1, int(round(0.50 * (n - 1)))))
        p99_idx = max(0, min(n - 1, int(round(0.99 * (n - 1)))))
        return {
            "name": self.name,  # type: ignore[dict-item]
            "count": n,
            "mean_ms": float(statistics.fmean(srt)),
            "p50_ms": float(srt[p50_idx]),
            "p99_ms": float(srt[p99_idx]),
            "min_ms": float(srt[0]),
            "max_ms": float(srt[-1]),
            "total_ms": float(sum(srt)),
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        s = self.summary()
        return (
            f"MeanTimer({self.name!r}, n={s['count']}, "
            f"mean={s['mean_ms']:.3f}ms, p50={s['p50_ms']:.3f}ms, p99={s['p99_ms']:.3f}ms)"
        )


# Global accumulator keyed by name. Modules can `from .timing import MEAN_TIMERS`
# and look up / create entries lazily via ``_get_mean_timer``.
MEAN_TIMERS: Dict[str, MeanTimer] = {}


def _get_mean_timer(name: str) -> MeanTimer:
    mt = MEAN_TIMERS.get(name)
    if mt is None:
        mt = MeanTimer(name)
        MEAN_TIMERS[name] = mt
    return mt


def summary_all() -> Dict[str, Dict[str, float]]:
    """Snapshot every registered MeanTimer's summary."""
    return {name: mt.summary() for name, mt in MEAN_TIMERS.items()}


def reset_all() -> None:
    """Drop every accumulated sample (does not delete the timer entries)."""
    for mt in MEAN_TIMERS.values():
        mt.reset()


@contextmanager
def nvtx_range(name: str, enabled: Optional[bool] = None):
    """Context manager wrapping ``torch.cuda.nvtx.range_push/pop``.

    No-op when timing is disabled or CUDA is not available.
    """
    if not _resolve_enabled(enabled) or not _cuda_available():
        yield
        return
    # nvtx may not exist in all torch builds; guard with getattr.
    nvtx = getattr(torch.cuda, "nvtx", None)  # type: ignore[union-attr]
    if nvtx is None or not hasattr(nvtx, "range_push"):
        yield
        return
    nvtx.range_push(str(name))
    try:
        yield
    finally:
        try:
            nvtx.range_pop()
        except Exception:  # pragma: no cover - never let instrumentation crash
            pass


@contextmanager
def cuda_timer(
    name: str,
    enabled: Optional[bool] = None,
    accumulate: bool = True,
):
    """GPU timing via ``torch.cuda.Event``. Reports milliseconds.

    Falls back to a no-op when timing is disabled or CUDA is not available.
    When ``accumulate`` is True the elapsed time is appended to
    ``MEAN_TIMERS[name]``; the elapsed value is also yielded as a
    single-element list ``[ms]`` so callers can read it after the with-block.
    """
    holder: list[float] = []
    if not _resolve_enabled(enabled) or not _cuda_available():
        yield holder
        return

    start = torch.cuda.Event(enable_timing=True)  # type: ignore[union-attr]
    end = torch.cuda.Event(enable_timing=True)  # type: ignore[union-attr]
    start.record()
    try:
        yield holder
    finally:
        end.record()
        try:
            torch.cuda.synchronize()  # type: ignore[union-attr]
            elapsed_ms = float(start.elapsed_time(end))
            holder.append(elapsed_ms)
            if accumulate:
                _get_mean_timer(name).add(elapsed_ms)
        except Exception:  # pragma: no cover
            pass


@contextmanager
def cpu_timer(
    name: str,
    enabled: Optional[bool] = None,
    accumulate: bool = True,
):
    """Wall-clock timing via ``time.perf_counter``. Reports milliseconds.

    Always works (no CUDA dependency). Yields a list that will hold the
    elapsed-ms value once the block exits.
    """
    holder: list[float] = []
    if not _resolve_enabled(enabled):
        yield holder
        return
    t0 = perf_counter()
    try:
        yield holder
    finally:
        elapsed_ms = (perf_counter() - t0) * 1000.0
        holder.append(elapsed_ms)
        if accumulate:
            _get_mean_timer(name).add(elapsed_ms)


__all__ = [
    "MEAN_TIMERS",
    "MeanTimer",
    "cpu_timer",
    "cuda_timer",
    "is_timing_enabled",
    "nvtx_range",
    "reset_all",
    "summary_all",
]
