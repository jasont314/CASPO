from __future__ import annotations

import pytest

from caspo.config import CASPOConfig
from caspo.utils.distributed import distributed_env, reduce_numeric_stats


def test_distributed_env_reads_torchrun_vars(monkeypatch):
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "8")

    info = distributed_env()

    assert info.rank == 3
    assert info.local_rank == 1
    assert info.world_size == 8
    assert info.is_distributed
    assert not info.is_main


def test_reduce_numeric_stats_noop_without_process_group():
    stats = {"loss": 1.5, "num_prompts": 4, "method": "caspo"}

    reduced = reduce_numeric_stats(stats, sum_keys=("num_prompts",))

    assert reduced == stats
    assert reduced is not stats


def test_fsdp_vllm_requires_rank_local_vllm_tp():
    with pytest.raises(ValueError, match="vllm_tensor_parallel_size=1"):
        CASPOConfig(
            distributed_backend="fsdp",
            rollout_backend="vllm",
            vllm_tensor_parallel_size=2,
        )


def test_fsdp_vllm_rank_local_config_is_valid():
    cfg = CASPOConfig(
        distributed_backend="fsdp",
        rollout_backend="vllm",
        vllm_tensor_parallel_size=1,
    )

    assert cfg.distributed_backend == "fsdp"
    assert cfg.rollout_backend == "vllm"
