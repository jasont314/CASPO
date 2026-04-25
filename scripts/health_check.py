"""Training health monitor.

Reads the trainer's stdout log + (optionally) the wandb-history file and
reports whether the run is making progress, stale, or crashed.

Heuristics:
* Crashed: log contains 'Traceback' or 'Killed' in the last 200 lines
* Stale: no new "[step ...]" line in the last ``--stale-secs`` seconds
* Progressing: reward / loss are trending in the right direction (smoothed)

Usage::

    python -m scripts.health_check --log /path/to/run.log [--stale-secs 600]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import Iterable, List, Optional, Tuple

# Cap the tail we read from the log. ~2 MB covers >>200 step lines for any
# realistic trainer cadence, and avoids loading multi-GB logs into memory.
_TAIL_BYTES = 2 * 1024 * 1024


# Step header lines come from CASPOTrainer._log and look like:
#   [caspo step 39/1000] loss=1.7507 pg=1.2874 reward=0.016 pass@G=0.125 ...
# ``method`` is one of "ppo", "caspo", "grpo", "vineppo".
_STEP_RE = re.compile(r"^\[(?P<method>ppo|caspo|grpo|vineppo) step (?P<step>\d+)/(?P<max>\d+)\]")

# Numeric fields. We require a leading boundary so e.g. ``v_loss=`` does NOT
# match ``loss=`` and ``positive_loss=`` would not either. Values may be
# scientific notation (``1.00e-06``) or plain floats (incl. negatives, NaN,
# Inf for safety).
_NUM = r"(?P<val>-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|NaN|inf|Inf|-inf|-Inf)"


def _kv_re(key: str) -> re.Pattern:
    # (?<![\w.@]) prevents matches that are part of a longer identifier such as
    # ``v_loss=`` (would catch the ``loss=`` tail) or ``foo.loss=`` etc.
    # ``@`` is excluded so ``pass@G=`` still matches as a whole.
    return re.compile(rf"(?<![\w.]){re.escape(key)}=" + _NUM)


_LOSS_RE = _kv_re("loss")
_REWARD_RE = _kv_re("reward")
_PASS_RE = _kv_re("pass@G")


def _parse_step_lines(lines: Iterable[str]) -> List[Tuple[int, float, float, float]]:
    """Return list of ``(step, loss, reward, pass_at_g)`` for each step line.

    Uses ``match`` (not ``search``) since ``_STEP_RE`` is anchored at line
    start, and skips the metric regex pass entirely on non-step lines.
    """
    out: List[Tuple[int, float, float, float]] = []
    step_match = _STEP_RE.match
    loss_search = _LOSS_RE.search
    reward_search = _REWARD_RE.search
    pass_search = _PASS_RE.search
    nan = float("nan")
    for ln in lines:
        m = step_match(ln)
        if not m:
            continue
        # Skip the leading "[method step n/m]" prefix when scanning for kv
        # pairs — those tokens never appear in the prefix and this trims a few
        # bytes per line off the regex search range.
        rest_start = m.end()

        def _val(rx_search, _start=rest_start, _ln=ln) -> float:
            mm = rx_search(_ln, _start)
            if not mm:
                return nan
            try:
                return float(mm.group("val"))
            except (TypeError, ValueError):
                return nan

        out.append((
            int(m.group("step")),
            _val(loss_search),
            _val(reward_search),
            _val(pass_search),
        ))
    return out


def _finite_tail(values: List[float], window: int) -> List[float]:
    """Return the last ``window`` entries of ``values`` with NaN/inf filtered."""
    out: List[float] = []
    for v in values[-window:]:
        # ``v == v`` rejects NaN; the inf checks reject +/-inf without
        # constructing temporary float("inf") values.
        if v == v and v != float("inf") and v != float("-inf"):
            out.append(v)
    return out


def _trend_and_std(
    values: List[float], window: int = 20
) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(OLS slope per step, sample std)`` over last ``window`` finites.

    Single-pass over the tail: previous code filtered finite values twice
    (once in ``_trend``, once in ``_series_std``) and built ``xs`` separately;
    here we fold both means + numerator + denominator into one loop.
    Returns ``(None, None)`` if fewer than 4 finite samples are available.
    """
    finite = _finite_tail(values, window)
    n = len(finite)
    if n < 4:
        return None, None
    # Closed-form means: x = 0..n-1.
    mean_x = (n - 1) / 2.0
    sum_y = 0.0
    for y in finite:
        sum_y += y
    mean_y = sum_y / n
    num = 0.0
    den = 0.0
    var_y = 0.0
    for i, y in enumerate(finite):
        dx = i - mean_x
        dy = y - mean_y
        num += dx * dy
        den += dx * dx
        var_y += dy * dy
    if den == 0:
        return None, None
    slope = num / den
    std = (var_y / max(1, n - 1)) ** 0.5
    return slope, std


# Match real Python tracebacks / OS-level kills. We anchor at line-start
# and require typical context so that the words appearing inside a model
# rollout (e.g. an example mentioning "Traceback" or "Killed") do not trip
# the crash detector.
_TRACEBACK_LINE_RE = re.compile(r"^Traceback \(most recent call last\):", re.MULTILINE)
_KILLED_LINE_RE = re.compile(r"^Killed\s*$", re.MULTILINE)
_CUDA_OOM_RE = re.compile(
    r"(?:torch\.cuda\.OutOfMemoryError|CUDA out of memory|"
    r"CUDA error:|RuntimeError: CUDA error)"
)


def _looks_crashed(tail: str) -> Optional[str]:
    if _TRACEBACK_LINE_RE.search(tail):
        return "Traceback"
    if _KILLED_LINE_RE.search(tail):
        return "Killed"
    if _CUDA_OOM_RE.search(tail):
        return "CUDA error"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, required=True)
    ap.add_argument("--stale-secs", type=int, default=600,
                    help="warn if the log file's mtime is older than this")
    ap.add_argument("--window", type=int, default=20,
                    help="trend window (number of recent step lines)")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"FATAL: log file not found: {args.log}")
        sys.exit(2)

    # Read only the tail of the log: ``readlines()`` on a multi-GB run.log is
    # unnecessary when we only care about step lines + crash markers near the
    # end. seek(-_TAIL_BYTES, SEEK_END) is O(1) on real filesystems and avoids
    # decoding the full history every poll. Combine with os.fstat to get
    # mtime in the same syscall round-trip.
    with open(args.log, "rb") as f:
        st = os.fstat(f.fileno())
        size = st.st_size
        if size > _TAIL_BYTES:
            f.seek(-_TAIL_BYTES, os.SEEK_END)
            # Drop the leading partial line so we don't mis-parse a fragment.
            f.readline()
        raw = f.read()
    last_mtime = st.st_mtime
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines(keepends=True)

    # Crash markers only need to fire on the recent tail; reuse the already-
    # decoded ``text`` instead of joining a sliced list.
    if len(text) > 64 * 1024:
        crash_window = text[-64 * 1024:]
    else:
        crash_window = text
    crashed_reason = _looks_crashed(crash_window)

    parsed = _parse_step_lines(lines)
    age = time.time() - last_mtime
    stale = age > args.stale_secs

    if crashed_reason:
        print(f"STATUS: CRASHED [{crashed_reason}] ({len(parsed)} step lines parsed)")
        print("--- last 30 log lines ---")
        for ln in lines[-30:]:
            print(ln.rstrip())
        sys.exit(1)
    if not parsed:
        print(f"STATUS: NO PROGRESS YET (log {age:.0f}s old, {len(lines)} lines)")
        if stale:
            sys.exit(1)
        sys.exit(0)

    last_step, last_loss, last_reward, last_pass = parsed[-1]
    # Single comprehension pass instead of three; cheaper for long ``parsed``.
    losses, rewards, passes = [], [], []
    for _s, _l, _r, _p in parsed:
        losses.append(_l)
        rewards.append(_r)
        passes.append(_p)
    # Compute slope+std together; previous code walked the tail twice.
    loss_slope, loss_std = _trend_and_std(losses, window=args.window)
    reward_slope, reward_std = _trend_and_std(rewards, window=args.window)
    pass_slope, pass_std = _trend_and_std(passes, window=args.window)

    def _fmt(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x:+.4f}"

    print(f"STATUS: {'STALE' if stale else 'RUNNING'}")
    print(f"  log_age_s = {age:.0f}")
    print(f"  steps     = {last_step} (latest in log)")
    print(f"  loss      = {last_loss:.4f}    slope(last{args.window}) = {_fmt(loss_slope)}  (want < 0)")
    print(f"  reward    = {last_reward:.3f}  slope(last{args.window}) = {_fmt(reward_slope)}  (want > 0)")
    print(f"  pass@G    = {last_pass:.3f}    slope(last{args.window}) = {_fmt(pass_slope)}  (want > 0)")
    print()
    # We need at least ``window`` step lines worth of data before the slope
    # is meaningful. Below that, _trend may already return None — but we
    # also gate the warnings on global_step >= 50 so the first warmup
    # iterations don't trigger spurious "not learning?" alarms.
    if last_step >= 50 and len(parsed) >= args.window:
        # Suppress warnings when |slope * window| < 0.5 * std(metric over the
        # same window): a slope that small is dominated by noise and would
        # otherwise fire on every flat-but-jittery training tail. The 0.5
        # constant is a soft threshold (≈ "less than half a std of motion
        # across the whole window") that retains sensitivity to genuine
        # negative trends without flagging stationary noise. ``loss_std``,
        # ``reward_std``, ``pass_std`` were computed alongside the slopes
        # above by ``_trend_and_std`` — no second filter pass needed.

        def _significant(slope: Optional[float], std: Optional[float]) -> bool:
            if slope is None:
                return False
            if std is None or std == 0:
                return True
            return abs(slope) * args.window > 0.5 * std

        if reward_slope is not None and reward_slope <= 0 and _significant(reward_slope, reward_std):
            print(f"  WARN: reward slope <= 0 over last {args.window} steps - not learning?")
        if pass_slope is not None and pass_slope <= 0 and _significant(pass_slope, pass_std):
            print(f"  WARN: pass@G slope <= 0 over last {args.window} steps - not learning?")
        if loss_slope is not None and loss_slope > 0 and _significant(loss_slope, loss_std):
            print(f"  WARN: loss slope > 0 over last {args.window} steps - diverging?")
    if stale:
        sys.exit(1)


if __name__ == "__main__":
    main()
