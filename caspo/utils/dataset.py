"""Dataset utility helpers.

Pure stdlib + numpy. Deterministic seeded operations for train/val splits,
random subsetting, length-aware bucketing, and infinite shuffled iteration.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence, Sized, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


def _rows_len(rows) -> int:
    """Return length whether ``rows`` is an int (count) or a sized container."""
    if isinstance(rows, (int, np.integer)):
        n = int(rows)
        if n < 0:
            raise ValueError(f"row count must be non-negative, got {n}")
        return n
    if isinstance(rows, Sized):
        return len(rows)
    # Fallback: materialize
    return len(list(rows))


def train_val_split(
    rows,
    val_fraction: float,
    seed: int,
) -> Tuple[List[int], List[int]]:
    """Split row indices into train and val index lists.

    Parameters
    ----------
    rows : int or sized iterable
        Either a row count or any sized container; only the length is used.
    val_fraction : float
        Fraction of rows to assign to validation, in [0, 1].
    seed : int
        Seed for deterministic shuffling.

    Returns
    -------
    (train_idx, val_idx) : tuple of lists of int
        Disjoint, in shuffled order, covering range(n).
    """
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError(f"val_fraction must be in [0, 1], got {val_fraction}")
    n = _rows_len(rows)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    # Edge cases: keep at least one train row when val_fraction < 1 and n >= 1
    if n_val == n and val_fraction < 1.0 and n > 0:
        n_val = n - 1
    if n_val == 0 and val_fraction > 0.0 and n > 0:
        n_val = 1
    val_idx = [int(i) for i in perm[:n_val]]
    train_idx = [int(i) for i in perm[n_val:]]
    return train_idx, val_idx


def random_subset(rows, n: int, seed: int) -> List[int]:
    """Return a deterministic n-row subset of indices.

    Parameters
    ----------
    rows : int or sized iterable
        Row count or sized container.
    n : int
        Number of indices to return. Must satisfy 0 <= n <= len(rows).
    seed : int
        Seed for deterministic sampling.

    Returns
    -------
    list of int
        Length-``n`` list of distinct indices in shuffled order.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    total = _rows_len(rows)
    if n > total:
        raise ValueError(f"n={n} exceeds available rows={total}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total)
    return [int(i) for i in perm[:n]]


def length_buckets(lengths: Sequence[int], n_buckets: int) -> List[List[int]]:
    """Bucket indices by ascending length into ``n_buckets`` lists.

    Items are first sorted by length (ties broken by original index for
    determinism), then split into ``n_buckets`` near-equal contiguous chunks.

    Parameters
    ----------
    lengths : sequence of int
        Per-row length value used as the bucketing key.
    n_buckets : int
        Number of buckets. Must be >= 1.

    Returns
    -------
    list of list of int
        Outer list has length ``n_buckets`` (inner lists may be empty if
        ``len(lengths) < n_buckets``).
    """
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")
    n = len(lengths)
    if n == 0:
        return [[] for _ in range(n_buckets)]
    # Stable sort by (length, index) for determinism even with ties.
    order = sorted(range(n), key=lambda i: (int(lengths[i]), i))
    # Split as evenly as possible: first (n % n_buckets) buckets get one extra.
    base, rem = divmod(n, n_buckets)
    buckets: List[List[int]] = []
    pos = 0
    for b in range(n_buckets):
        size = base + (1 if b < rem else 0)
        buckets.append(order[pos : pos + size])
        pos += size
    return buckets


def infinite_iterator(items: Sequence[T], seed: int) -> Iterator[T]:
    """Yield shuffled items forever, reshuffling on each epoch.

    Each epoch uses a deterministic shuffle derived from ``seed`` plus the
    epoch counter, so the full stream is reproducible.

    Parameters
    ----------
    items : sequence
        Items to iterate over. Must be non-empty.
    seed : int
        Base seed; epoch ``e`` uses ``np.random.default_rng((seed, e))``.

    Yields
    ------
    Items from ``items`` in shuffled order, looping indefinitely.
    """
    if len(items) == 0:
        raise ValueError("infinite_iterator requires non-empty items")
    epoch = 0
    while True:
        rng = np.random.default_rng((int(seed), int(epoch)))
        perm = rng.permutation(len(items))
        for i in perm:
            yield items[int(i)]
        epoch += 1
