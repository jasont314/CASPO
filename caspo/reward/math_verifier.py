"""Verifiable math reward.

Extracts the last ``\\boxed{...}`` answer from a model response and grades it
against a ground-truth expression. Uses ``math_verify`` if installed, falls
back to a normalized string match plus a final SymPy-based equivalence check.

Reward is binary (1.0 / 0.0); an optional ``format_bonus`` is added when the
response contains any ``\\boxed{}`` at all (regardless of correctness).
"""

from __future__ import annotations

import atexit
import os
import re
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Boxed-answer extraction
# ---------------------------------------------------------------------------

_BOXED_TOKEN = "\\boxed"
_BOXED_TOKEN_LEN = len(_BOXED_TOKEN)


def _match_close(text: str, brace_open: int) -> int:
    """Return the index of the ``}`` that closes the ``{`` at ``brace_open``.

    Recognizes LaTeX-escaped braces (``\\{``, ``\\}``) and any other ``\\X``
    escape so they don't perturb depth. Returns ``-1`` when the match is
    malformed (unbalanced).
    """
    n = len(text)
    depth = 0
    idx = brace_open
    while idx < n:
        ch = text[idx]
        if ch == "\\" and idx + 1 < n:
            idx += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return idx
        idx += 1
    return -1


def extract_boxed_answer(text: str) -> Optional[str]:
    """Return the contents of the *last* well-formed ``\\boxed{...}`` in ``text``.

    Handles arbitrarily nested braces, LaTeX-escaped braces (``\\{`` and
    ``\\}``), and any ASCII whitespace between ``\\boxed`` and its opening
    brace. When ``\\boxed`` blocks are nested (``\\boxed{\\boxed{x}}``), the
    *innermost* well-formed one wins — it appears last in the scan order, and
    selecting it lets a downstream grader compare against the bare answer
    ``x`` directly. Returns ``None`` when no well-formed boxed answer is
    present.
    """
    if not text:
        return None

    n = len(text)

    # Find every \boxed{ that opens cleanly (token followed by optional
    # whitespace then '{'). Record the index of the opening brace.
    starts: List[int] = []
    i = 0
    while True:
        j = text.find(_BOXED_TOKEN, i)
        if j == -1:
            break
        k = j + _BOXED_TOKEN_LEN
        while k < n and text[k] in " \t\n\r\f\v":
            k += 1
        if k < n and text[k] == "{":
            starts.append(k)
        i = j + _BOXED_TOKEN_LEN

    if not starts:
        return None

    # Walk matches last-to-first; return the first well-formed one whose
    # contents are non-empty after whitespace stripping. Earlier malformed
    # (unbalanced) matches are silently skipped, and empty/whitespace-only
    # boxes (\boxed{}, \boxed{   }) are treated as "no usable answer" so they
    # don't (a) score a spurious 1.0 against an empty ground truth or
    # (b) earn ``format_bonus`` from MathRewardFn.
    for brace_open in reversed(starts):
        close = _match_close(text, brace_open)
        if close != -1:
            inner = text[brace_open + 1 : close]
            if inner.strip():
                return inner
    return None


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------

# Single regex that erases the LaTeX-spacing macros, the ``\left``/``\right``
# wrappers, ``$`` delimiters, and ASCII whitespace in one pass — replaces a
# chain of ~9 ``str.replace`` calls in ``_normalize``. Order in the alternation
# matters: longer, more-specific tokens (``\left``, ``\right``) before the
# short ``\,``/``\!`` family so the regex engine doesn't backtrack.
_NORM_STRIP_RE = re.compile(
    r"\\left|\\right|\\[,!;: ]|~|\$|[ \t\n\r\f\v]"
)
# Trailing ``.,;`` punctuation, possibly more than one (``42.,``).
_NORM_TRAIL_RE = re.compile(r"[.,;]+$")
# Plain numeric body (optional leading ``-``) so we can collapse trailing
# zeros without iterating chars.
_NORM_NUMERIC_RE = re.compile(r"^-?\d*\.\d+$|^-?\d+\.\d*$")


def _normalize(s: str) -> str:
    """Canonicalize a math string for cheap string comparison."""
    if not s:
        return ""
    out = s.strip()
    # Strip outer $...$ if present.
    while len(out) >= 2 and out[0] == "$" and out[-1] == "$":
        out = out[1:-1].strip()
    # Erase LaTeX spacing macros, \left / \right, $, and whitespace in one pass.
    out = _NORM_STRIP_RE.sub("", out)
    # Strip trailing punctuation like ".", ",", ";".
    out = _NORM_TRAIL_RE.sub("", out)
    # Strip a leading "+" sign on the bare token — "+5" and "5" are equal in
    # math but distinct as strings. (Don't touch "-5"; sign matters there.)
    if len(out) >= 2 and out[0] == "+":
        c1 = out[1]
        if c1.isdigit() or c1 == ".":
            out = out[1:]
    # Collapse trailing zeros after a decimal point ("5.0" -> "5", "5.10" -> "5.1").
    # Only when the token is a plain number — don't touch "x.y" expression
    # forms. Regex match is faster than the previous all()/count() chain.
    if "." in out and _NORM_NUMERIC_RE.match(out):
        out = out.rstrip("0").rstrip(".")
        if not out or out == "-":
            out = "0"
    return out.lower()


def _strip_to_bare(gt: str) -> str:
    """If the ground truth itself contains ``\\boxed{...}``, peel it off."""
    inner = extract_boxed_answer(gt)
    return inner if inner is not None else gt


# Lazy module-level handles for ``math_verify``. We resolve once on first use
# (vs. ``from math_verify import ...`` on every grade) so the per-call cost
# is just an attribute load instead of a try/except + sys.modules lookup.
# ``_MV_PROBED`` distinguishes "not yet looked up" from "looked up and absent".
_MV_PARSE = None  # type: ignore[var-annotated]
_MV_VERIFY = None  # type: ignore[var-annotated]
_MV_PROBED = False


def _ensure_math_verify() -> bool:
    """Resolve ``math_verify.parse``/``verify`` once. Returns True if usable."""
    global _MV_PARSE, _MV_VERIFY, _MV_PROBED
    if _MV_PROBED:
        return _MV_PARSE is not None
    _MV_PROBED = True
    try:
        from math_verify import parse as _p, verify as _v  # type: ignore
        _MV_PARSE = _p
        _MV_VERIFY = _v
        return True
    except Exception:
        return False


def _try_math_verify(prediction_boxed: str, gt_bare: str) -> Optional[bool]:
    """Use ``math_verify`` if available. Returns True/False or None on failure."""
    if not _ensure_math_verify():
        return None
    try:
        # math_verify expects the gold to be a parsed expression; wrap the
        # bare ground truth in \boxed{} so the parser treats it the same way
        # as the prediction.
        gold = _MV_PARSE(f"\\boxed{{{gt_bare}}}")  # type: ignore[misc]
        pred = _MV_PARSE(f"\\boxed{{{prediction_boxed}}}")  # type: ignore[misc]
        return bool(_MV_VERIFY(gold, pred))  # type: ignore[misc]
    except Exception:
        return None


# Precompiled patterns used by the sympy fallback. Hoisted out of the hot
# path to avoid recompiling on every grade.
# Matches \frac, \dfrac, \tfrac with non-nested numerator/denominator.
_FRAC_RE = re.compile(r"\\[dt]?frac\{([^{}]+)\}\{([^{}]+)\}")


def _prep_for_sympy(s: str) -> str:
    """Light TeX -> sympy-input substitutions."""
    out = s.replace("^", "**")
    # \frac{a}{b} -> (a)/(b); applied repeatedly so simple nestings unwrap.
    for _ in range(4):
        new = _FRAC_RE.sub(r"((\1)/(\2))", out)
        if new == out:
            break
        out = new
    out = out.replace("\\cdot", "*").replace("\\times", "*")
    out = out.replace("\\pi", "pi")
    out = out.replace("\\sqrt", "sqrt")
    # remove stray backslashes
    out = out.replace("\\", "")
    return out


def _try_sympy(a: str, b: str, timeout_s: float = 2.0) -> Optional[bool]:
    """Final fallback: ``simplify(parse_expr(a) - parse_expr(b)) == 0``.

    Wraps ``simplify`` in a SIGALRM-based timeout so adversarial expressions
    can't wedge the grader. Falls back to running without a timeout if signals
    aren't available (e.g. called from a non-main thread on POSIX, or on
    Windows). On timeout, treats the comparison as unknown (returns ``None``)
    rather than declaring inequality.
    """
    try:
        from sympy import simplify
        from sympy.parsing.sympy_parser import parse_expr
    except Exception:
        return None

    import signal as _sig

    class _SympyTimeout(Exception):
        pass

    def _handler(signum, frame):  # noqa: ARG001
        raise _SympyTimeout()

    have_alarm = hasattr(_sig, "SIGALRM")
    prev = None
    armed = False
    if have_alarm and timeout_s > 0:
        try:
            prev = _sig.signal(_sig.SIGALRM, _handler)
            # setitimer accepts fractional seconds; alarm() rounds to int.
            _sig.setitimer(_sig.ITIMER_REAL, float(timeout_s))
            armed = True
        except (ValueError, OSError):
            # Not in main thread, or signals unavailable — run without timeout.
            armed = False

    try:
        ea = parse_expr(_prep_for_sympy(a))
        eb = parse_expr(_prep_for_sympy(b))
        return bool(simplify(ea - eb) == 0)
    except _SympyTimeout:
        return None
    except Exception:
        return None
    finally:
        if armed:
            try:
                _sig.setitimer(_sig.ITIMER_REAL, 0.0)
                if prev is not None:
                    _sig.signal(_sig.SIGALRM, prev)
            except Exception:
                pass


def grade_math(prediction: str, ground_truth: str) -> float:
    """Return 1.0 iff ``prediction`` (a model response containing ``\\boxed{}``)
    is mathematically equivalent to ``ground_truth`` (boxed or bare). 0.0
    otherwise.
    """
    pred_inner = extract_boxed_answer(prediction)
    if pred_inner is None:
        return 0.0
    gt_bare = _strip_to_bare(ground_truth)

    # 1) Cheap normalized string equality FIRST. Most correctly-formatted
    # answers (e.g. ``\boxed{42}`` vs ``42``, ``\boxed{ 1/2 }`` vs ``1/2``)
    # already match here in ~1 μs and skip math_verify's ~300 μs parse.
    pred_norm = _normalize(pred_inner)
    gt_norm = _normalize(gt_bare)
    if pred_norm == gt_norm:
        return 1.0

    # 2) math_verify (symbolic) for forms that don't normalize-match.
    mv = _try_math_verify(pred_inner, gt_bare)
    if mv is True:
        return 1.0
    if mv is False:
        # math_verify did its symbolic check and disagreed; skip SymPy to
        # avoid duplicate work / potential hangs.
        return 0.0

    # 3) SymPy equivalence (only reached when math_verify is unavailable).
    sp = _try_sympy(pred_inner, gt_bare)
    if sp is True:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Callable wrapper
# ---------------------------------------------------------------------------


def _grade_chunk(
    chunk: List[Tuple[str, str, Optional[Tuple[str, str]]]],
    bonus: float,
) -> List[float]:
    """Worker entry-point. Top-level (picklable) by design.

    ``chunk`` is a list of ``(pred, gt, normalized_gt_or_None)`` triples. When
    the third element is ``(gt_bare, gt_norm)`` the worker can skip the LaTeX
    peel + canonicalize on the GT and go straight to comparison. When ``None``
    the worker falls back to deriving them locally — used when the GT is rare
    enough that the main process didn't bother to cache it.
    """
    out: List[float] = []
    for pred, gt, pre_norm in chunk:
        pred_inner = extract_boxed_answer(pred)
        if pred_inner is None:
            out.append(0.0)
            continue
        if pre_norm is not None:
            gt_bare, gt_norm = pre_norm
        else:
            gt_bare = _strip_to_bare(gt) if gt is not None else ""
            gt_norm = _normalize(gt_bare)
        if _normalize(pred_inner) == gt_norm:
            out.append(1.0)
            continue
        mv = _try_math_verify(pred_inner, gt_bare)
        if mv is True:
            out.append(1.0)
            continue
        if mv is False:
            out.append(bonus)
            continue
        sp = _try_sympy(pred_inner, gt_bare)
        if sp is True:
            out.append(1.0)
        else:
            out.append(bonus)
    return out


# Module-level registry of pools so atexit can shut them down even if a
# MathRewardFn instance gets GC'd in a non-deterministic order during
# interpreter teardown.
_LIVE_POOLS: "list[ProcessPoolExecutor]" = []


def _shutdown_all_pools() -> None:  # pragma: no cover - exit path
    while _LIVE_POOLS:
        pool = _LIVE_POOLS.pop()
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


atexit.register(_shutdown_all_pools)


class MathRewardFn:
    """Callable batch reward.

    ``__call__(predictions, ground_truths) -> list[float]``

    Adds ``format_bonus`` (default 0.0) to any response that contains a
    ``\\boxed{}`` even if the answer is wrong. Correct answers always score
    ``1.0`` (the bonus is not stacked on top of a correct grade). The bonus
    is clamped to ``[0.0, 1.0)`` so a wrong-but-formatted answer cannot
    out-score a correct one.

    When ``num_workers > 1`` and the batch is large enough to amortize the
    IPC cost, grading is dispatched across a persistent
    ``ProcessPoolExecutor``. SIGALRM-based timeouts in :func:`_try_sympy`
    require *processes* (signals only fire on the main thread of each
    process), so a thread pool would silently drop the timeout guard.
    """

    # Below this threshold the IPC + chunk-pickle cost dominates the symbolic
    # work, so we stay on the serial path. ``len(predictions) > workers * 2``
    # is the documented gate.
    _MIN_PARALLEL_RATIO: int = 2

    def __init__(
        self,
        format_bonus: float = 0.0,
        num_workers: int = 1,
        gt_cache_max_size: int = 8192,
    ):
        b = float(format_bonus)
        if b < 0.0:
            b = 0.0
        elif b >= 1.0:
            # Cap strictly below 1.0 — equal would tie wrong with right.
            b = 1.0 - 1e-6
        self.format_bonus = b
        self.num_workers = max(1, int(num_workers))
        self.gt_cache_max_size = max(1, int(gt_cache_max_size))
        # Persistent across PPO outer steps. The DeepScaleR train cycle is
        # ~7.5K prompts so the same GT recurs many times during one epoch.
        # OrderedDict gives us FIFO eviction without an explicit deque.
        self._gt_cache: "OrderedDict[str, Tuple[str, str]]" = OrderedDict()
        # Pool is created lazily on the first parallel-eligible call. Re-used
        # across calls so we pay the fork cost only once per trainer instance.
        self._pool: Optional[ProcessPoolExecutor] = None

    # -- pool lifecycle ------------------------------------------------------

    def _ensure_pool(self) -> ProcessPoolExecutor:
        if self._pool is None:
            # ``max_workers`` capped at the configured value; spawn-vs-fork is
            # left to the platform default (fork on Linux, spawn on macOS/Win).
            self._pool = ProcessPoolExecutor(max_workers=self.num_workers)
            _LIVE_POOLS.append(self._pool)
        return self._pool

    def close(self) -> None:
        """Tear down the worker pool. Idempotent."""
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            try:
                _LIVE_POOLS.remove(self._pool)
            except ValueError:
                pass
            self._pool = None

    def __del__(self) -> None:  # pragma: no cover - GC path
        try:
            self.close()
        except Exception:
            pass

    # -- gt cache helpers ----------------------------------------------------

    def _cache_gt(self, gt: str) -> Tuple[str, str]:
        cached = self._gt_cache.get(gt)
        if cached is not None:
            # True FIFO: insertion order is preserved on hits — we do NOT
            # move_to_end here. Touching an entry on access would convert
            # this into LRU and silently break the documented eviction policy.
            return cached
        gt_bare = _strip_to_bare(gt) if gt is not None else ""
        cached = (gt_bare, _normalize(gt_bare))
        self._gt_cache[gt] = cached
        # Bounded-size: evict the oldest 1024 (or all but the newest 1, if
        # max_size is tiny) to amortize the rebalance cost.
        if len(self._gt_cache) > self.gt_cache_max_size:
            evict = min(1024, max(1, len(self._gt_cache) - self.gt_cache_max_size + 1))
            for _ in range(evict):
                self._gt_cache.popitem(last=False)
        return cached

    # -- main entry ----------------------------------------------------------

    def __call__(
        self,
        predictions: List[str],
        ground_truths: List[str],
    ) -> List[float]:
        if len(predictions) != len(ground_truths):
            raise ValueError(
                f"predictions ({len(predictions)}) and ground_truths "
                f"({len(ground_truths)}) must have the same length"
            )
        bonus = self.format_bonus
        n = len(predictions)
        # Pre-warm the GT cache in the parent so workers don't each redo the
        # same LaTeX peel + canonicalization. Tuples are cheap to pickle.
        for gt in ground_truths:
            self._cache_gt(gt)

        # Decide whether to fan out. Threshold matches the documented gate:
        # >1 worker AND batch large enough to amortize the IPC overhead.
        use_parallel = (
            self.num_workers > 1
            and n > self.num_workers * self._MIN_PARALLEL_RATIO
        )

        if not use_parallel:
            # Serial path — preserves test determinism and avoids fork cost
            # for the small batches typical in unit tests.
            out: List[float] = []
            for pred, gt in zip(predictions, ground_truths):
                pred_inner = extract_boxed_answer(pred)
                if pred_inner is None:
                    out.append(0.0)
                    continue
                gt_bare, gt_norm = self._cache_gt(gt)
                if _normalize(pred_inner) == gt_norm:
                    out.append(1.0)
                    continue
                mv = _try_math_verify(pred_inner, gt_bare)
                if mv is True:
                    out.append(1.0)
                    continue
                if mv is False:
                    out.append(bonus)
                    continue
                sp = _try_sympy(pred_inner, gt_bare)
                if sp is True:
                    out.append(1.0)
                else:
                    out.append(bonus)
            return out

        # Parallel path. Build chunks of (pred, gt, pre_norm) so workers can
        # skip the GT canonicalization. Chunk size is chosen so each worker
        # gets roughly two batches' worth of work — keeps stragglers short
        # without flooding the executor with single-item futures.
        chunk_count = self.num_workers * 4
        chunk_size = max(1, (n + chunk_count - 1) // chunk_count)
        chunks: List[List[Tuple[str, str, Optional[Tuple[str, str]]]]] = []
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            triples: List[Tuple[str, str, Optional[Tuple[str, str]]]] = []
            for i in range(start, end):
                gt_i = ground_truths[i]
                triples.append((predictions[i], gt_i, self._gt_cache.get(gt_i)))
            chunks.append(triples)

        pool = self._ensure_pool()
        # Submit all chunks, then collect in submission order so the output
        # list aligns with ``predictions``.
        futures = [pool.submit(_grade_chunk, chunk, bonus) for chunk in chunks]
        out_parallel: List[float] = []
        for fut in futures:
            out_parallel.extend(fut.result())
        return out_parallel
