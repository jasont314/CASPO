"""Small distributed-runtime helpers.

The trainer is intentionally usable as a plain single-process script.  These
helpers make torchrun-launched jobs opt in to distributed behavior without
spreading environment-variable parsing and rank checks through the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import os
import warnings
from typing import Iterable, Mapping, MutableMapping

import torch


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        warnings.warn(f"ignoring invalid {name}={val!r}; expected integer")
        return default


@dataclass(frozen=True)
class DistributedInfo:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "none"

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def distributed_env() -> DistributedInfo:
    """Return rank metadata from torchrun-style environment variables."""

    return DistributedInfo(
        rank=_env_int("RANK", 0),
        local_rank=_env_int("LOCAL_RANK", 0),
        world_size=max(1, _env_int("WORLD_SIZE", 1)),
        backend="env",
    )


def init_distributed_if_needed(
    *,
    backend: str = "nccl",
    timeout_s: int = 1800,
) -> DistributedInfo:
    """Initialize ``torch.distributed`` when launched under torchrun.

    With ``WORLD_SIZE=1`` this is a cheap no-op and returns rank-zero metadata.
    """

    info = distributed_env()
    if not info.is_distributed:
        return DistributedInfo(
            rank=info.rank, local_rank=info.local_rank,
            world_size=info.world_size, backend="none",
        )

    import torch.distributed as dist

    use_backend = backend
    if use_backend == "nccl" and not torch.cuda.is_available():
        warnings.warn("NCCL requested but CUDA is unavailable; falling back to gloo")
        use_backend = "gloo"

    if use_backend == "nccl":
        torch.cuda.set_device(info.local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend=use_backend,
            timeout=timedelta(seconds=int(timeout_s)),
        )

    return DistributedInfo(
        rank=dist.get_rank(),
        local_rank=info.local_rank,
        world_size=dist.get_world_size(),
        backend=use_backend,
    )


def resolve_device(requested: str, info: DistributedInfo) -> torch.device:
    """Map a config device string to the rank-local torch device."""

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"device={requested!r} requested but CUDA is unavailable")
        if info.is_distributed:
            return torch.device(f"cuda:{info.local_rank}")
    return torch.device(requested)


def is_dist_initialized() -> bool:
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


def barrier() -> None:
    if not is_dist_initialized():
        return
    import torch.distributed as dist

    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def rank0_print(info: DistributedInfo, *args, **kwargs) -> None:
    if info.is_main:
        print(*args, **kwargs)


def reduce_numeric_stats(
    stats: Mapping[str, object],
    *,
    sum_keys: Iterable[str] = (),
) -> dict:
    """All-reduce scalar numeric stats for rank-zero logging.

    Numeric keys are averaged across ranks by default.  Keys listed in
    ``sum_keys`` are summed instead, which is useful for count-like metrics.
    Non-numeric values are copied from the local rank unchanged.
    """

    if not is_dist_initialized():
        return dict(stats)

    import torch.distributed as dist

    world = dist.get_world_size()
    sum_key_set = set(sum_keys)
    out: MutableMapping[str, object] = dict(stats)
    numeric_keys = [
        k for k, v in stats.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    if not numeric_keys:
        return dict(out)

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    values = torch.tensor([float(stats[k]) for k in numeric_keys], device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)

    for i, key in enumerate(numeric_keys):
        val = values[i].item()
        if key not in sum_key_set:
            val /= float(world)
        old = stats[key]
        out[key] = int(round(val)) if isinstance(old, int) and key in sum_key_set else float(val)
    return dict(out)
