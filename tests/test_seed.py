"""Tests for caspo.utils.seed."""

from __future__ import annotations

import random

import numpy as np
import torch

from caspo.utils.seed import set_seed, with_temporary_seed, worker_init_fn


def _draw_sample():
    """Draw one value from each RNG so we can compare across calls."""
    return (
        random.randint(0, 2**31 - 1),
        int(np.random.randint(0, 2**31 - 1)),
        int(torch.randint(0, 2**31 - 1, (1,)).item()),
    )


def test_set_seed_makes_randint_reproducible():
    set_seed(1234)
    a = _draw_sample()

    set_seed(1234)
    b = _draw_sample()

    assert a == b, f"Expected reproducible sample, got {a} vs {b}"


def test_set_seed_different_seeds_differ():
    set_seed(1)
    a = _draw_sample()
    set_seed(2)
    b = _draw_sample()
    assert a != b


def test_set_seed_rejects_non_int():
    import pytest

    with pytest.raises(TypeError):
        set_seed(1.5)  # type: ignore[arg-type]


def test_with_temporary_seed_restores_state():
    set_seed(42)
    # Advance the RNGs to a non-trivial state.
    _ = _draw_sample()
    pre = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )

    # Sample inside the temporary-seed block; should be deterministic for the
    # given inner seed.
    with with_temporary_seed(999):
        inner_a = _draw_sample()

    # Outer state must match exactly what it was before the block.
    post = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
    )
    assert pre[0] == post[0], "random state not restored"
    # numpy state is a tuple of arrays; compare element-wise.
    assert pre[1][0] == post[1][0]
    assert (pre[1][1] == post[1][1]).all()
    assert pre[1][2:] == post[1][2:]
    assert torch.equal(pre[2], post[2]), "torch state not restored"

    # Re-entering with the same inner seed should produce the same inner draw.
    with with_temporary_seed(999):
        inner_b = _draw_sample()
    assert inner_a == inner_b


def test_with_temporary_seed_restores_on_exception():
    set_seed(7)
    _ = _draw_sample()
    pre_torch = torch.get_rng_state().clone()
    pre_py = random.getstate()

    class _Boom(Exception):
        pass

    try:
        with with_temporary_seed(123):
            _ = _draw_sample()
            raise _Boom()
    except _Boom:
        pass

    assert torch.equal(pre_torch, torch.get_rng_state())
    assert pre_py == random.getstate()


def test_worker_init_fn_sets_distinct_streams():
    # Pin torch.initial_seed() context by seeding first.
    set_seed(2026)
    base = torch.initial_seed()

    worker_init_fn(0)
    s0 = (random.random(), float(np.random.rand()))

    worker_init_fn(1)
    s1 = (random.random(), float(np.random.rand()))

    assert s0 != s1, "Different worker ids should yield different streams"

    # Reproducibility: same base seed + same worker id => same stream.
    set_seed(2026)
    assert torch.initial_seed() == base
    worker_init_fn(0)
    s0_again = (random.random(), float(np.random.rand()))
    assert s0 == s0_again
