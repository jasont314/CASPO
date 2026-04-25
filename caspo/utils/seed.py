"""Centralized seeding utilities for reproducibility.

Provides:
    set_seed:               Set random/numpy/torch (CPU+CUDA) seeds. Optional
                            deterministic mode toggles torch deterministic
                            algorithms and CUBLAS workspace config.
    worker_init_fn:         DataLoader worker initializer when num_workers > 0.
    with_temporary_seed:    Context manager that snapshots and restores all RNG
                            states; useful for sampling-based eval that should
                            not perturb the training RNG stream.

This module imports only stdlib + torch + numpy.
"""

from __future__ import annotations

import contextlib
import os
import random
from typing import Iterator

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed all RNGs used in the project.

    Args:
        seed: Integer seed.
        deterministic: If True, also enable torch deterministic algorithms and
            set the CUBLAS workspace config env var (required for deterministic
            CUDA matmul on recent CUDA versions). Note: enabling this can be
            substantially slower and will raise if a non-deterministic op is
            invoked.
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Safe to call even if CUDA is unavailable; sets seeds for all visible
    # CUDA devices when present.
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # CUBLAS workspace config must be set before any CUDA work for the
        # deterministic algorithm guarantee to hold. We set it here so that
        # callers who set_seed early get the right behavior.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker initializer.

    Use as ``DataLoader(..., worker_init_fn=worker_init_fn)`` whenever
    ``num_workers > 0``. Combines torch's per-worker base seed with the worker
    id so each worker gets a distinct, reproducible RNG stream.
    """
    base_seed = torch.initial_seed()
    # torch.initial_seed() can be > 2**32; numpy/random want a 32-bit value.
    worker_seed = (base_seed + worker_id) % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(base_seed + worker_id)


@contextlib.contextmanager
def with_temporary_seed(seed: int) -> Iterator[None]:
    """Context manager: set ``seed`` for the block, restore RNG state on exit.

    Snapshots Python ``random``, NumPy global, torch CPU, and torch CUDA (all
    visible devices) RNG states on entry, applies ``set_seed(seed)``, and
    restores the snapshots on exit (even if the block raises).
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    try:
        set_seed(seed)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
