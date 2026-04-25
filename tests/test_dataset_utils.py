"""Unit tests for caspo.utils.dataset helpers."""

from __future__ import annotations

from itertools import islice

import pytest

from caspo.utils.dataset import (
    infinite_iterator,
    length_buckets,
    random_subset,
    train_val_split,
)


# ---------------------------------------------------------------------------
# train_val_split
# ---------------------------------------------------------------------------


def test_train_val_split_partition():
    n = 100
    train, val = train_val_split(n, 0.2, seed=0)
    assert len(train) + len(val) == n
    assert len(val) == 20
    assert set(train).isdisjoint(set(val))
    assert set(train) | set(val) == set(range(n))


def test_train_val_split_deterministic():
    rows = list(range(50))
    a = train_val_split(rows, 0.3, seed=42)
    b = train_val_split(rows, 0.3, seed=42)
    assert a == b


def test_train_val_split_different_seeds_differ():
    a_train, a_val = train_val_split(200, 0.1, seed=1)
    b_train, b_val = train_val_split(200, 0.1, seed=2)
    assert a_val != b_val


def test_train_val_split_zero_val():
    train, val = train_val_split(10, 0.0, seed=0)
    assert len(val) == 0
    assert len(train) == 10


def test_train_val_split_full_val():
    train, val = train_val_split(10, 1.0, seed=0)
    assert len(val) == 10
    assert len(train) == 0


def test_train_val_split_small_fraction_keeps_at_least_one_val():
    # 5 rows * 0.05 = 0.25 -> rounds to 0, but val_fraction>0 should give >=1
    train, val = train_val_split(5, 0.05, seed=0)
    assert len(val) >= 1
    assert len(train) >= 1


def test_train_val_split_invalid_fraction():
    with pytest.raises(ValueError):
        train_val_split(10, -0.1, seed=0)
    with pytest.raises(ValueError):
        train_val_split(10, 1.5, seed=0)


def test_train_val_split_empty():
    train, val = train_val_split(0, 0.2, seed=0)
    assert train == [] and val == []


# ---------------------------------------------------------------------------
# random_subset
# ---------------------------------------------------------------------------


def test_random_subset_size_and_uniqueness():
    sub = random_subset(100, 10, seed=0)
    assert len(sub) == 10
    assert len(set(sub)) == 10
    assert all(0 <= i < 100 for i in sub)


def test_random_subset_deterministic():
    a = random_subset(50, 7, seed=123)
    b = random_subset(50, 7, seed=123)
    assert a == b


def test_random_subset_different_seeds_differ():
    a = random_subset(1000, 20, seed=1)
    b = random_subset(1000, 20, seed=2)
    assert a != b


def test_random_subset_full():
    sub = random_subset(8, 8, seed=0)
    assert sorted(sub) == list(range(8))


def test_random_subset_zero():
    assert random_subset(10, 0, seed=0) == []


def test_random_subset_oversize_raises():
    with pytest.raises(ValueError):
        random_subset(5, 10, seed=0)


def test_random_subset_negative_raises():
    with pytest.raises(ValueError):
        random_subset(5, -1, seed=0)


# ---------------------------------------------------------------------------
# length_buckets
# ---------------------------------------------------------------------------


def test_length_buckets_partition_and_count():
    lengths = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    buckets = length_buckets(lengths, 3)
    assert len(buckets) == 3
    flat = [i for b in buckets for i in b]
    assert sorted(flat) == list(range(len(lengths)))
    # Sizes near-equal: 11 / 3 -> 4, 4, 3
    sizes = sorted(len(b) for b in buckets)
    assert sizes == [3, 4, 4]


def test_length_buckets_ordering_property():
    lengths = [10, 1, 5, 2, 8, 3]
    buckets = length_buckets(lengths, 3)
    # Lengths within an earlier bucket should all be <= lengths in later bucket.
    bucket_lengths = [[lengths[i] for i in b] for b in buckets]
    for earlier, later in zip(bucket_lengths, bucket_lengths[1:]):
        if earlier and later:
            assert max(earlier) <= max(later)
            assert min(earlier) <= min(later)


def test_length_buckets_deterministic_with_ties():
    lengths = [2, 2, 2, 2, 2, 2]
    a = length_buckets(lengths, 3)
    b = length_buckets(lengths, 3)
    assert a == b


def test_length_buckets_more_buckets_than_items():
    lengths = [1, 2, 3]
    buckets = length_buckets(lengths, 5)
    assert len(buckets) == 5
    flat = [i for b in buckets for i in b]
    assert sorted(flat) == [0, 1, 2]


def test_length_buckets_empty():
    buckets = length_buckets([], 4)
    assert buckets == [[], [], [], []]


def test_length_buckets_single_bucket():
    lengths = [5, 1, 3]
    buckets = length_buckets(lengths, 1)
    assert len(buckets) == 1
    assert sorted(buckets[0]) == [0, 1, 2]


def test_length_buckets_invalid_n_buckets():
    with pytest.raises(ValueError):
        length_buckets([1, 2, 3], 0)


# ---------------------------------------------------------------------------
# infinite_iterator
# ---------------------------------------------------------------------------


def test_infinite_iterator_yields_forever():
    items = ["a", "b", "c"]
    out = list(islice(infinite_iterator(items, seed=0), 12))
    assert len(out) == 12
    # Every emitted item is from the source set.
    assert set(out) <= set(items)


def test_infinite_iterator_covers_all_each_epoch():
    items = list(range(5))
    it = infinite_iterator(items, seed=0)
    for _ in range(3):  # 3 epochs
        epoch_out = [next(it) for _ in range(5)]
        assert sorted(epoch_out) == items


def test_infinite_iterator_deterministic():
    items = list(range(10))
    a = list(islice(infinite_iterator(items, seed=7), 30))
    b = list(islice(infinite_iterator(items, seed=7), 30))
    assert a == b


def test_infinite_iterator_different_seeds_differ():
    items = list(range(20))
    a = list(islice(infinite_iterator(items, seed=1), 20))
    b = list(islice(infinite_iterator(items, seed=2), 20))
    assert a != b


def test_infinite_iterator_reshuffles_between_epochs():
    items = list(range(50))
    it = infinite_iterator(items, seed=0)
    epoch1 = [next(it) for _ in range(50)]
    epoch2 = [next(it) for _ in range(50)]
    assert epoch1 != epoch2  # vanishingly small chance of collision at n=50


def test_infinite_iterator_empty_raises():
    with pytest.raises(ValueError):
        next(infinite_iterator([], seed=0))
