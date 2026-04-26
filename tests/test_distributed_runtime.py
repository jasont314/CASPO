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


def test_ipc_weight_sync_accepts_fsdp_rank_local_vllm():
    """FSDP + IPC is now supported via summon_full_params (rank-local vLLM)."""
    cfg = CASPOConfig(
        distributed_backend="fsdp",
        rollout_backend="vllm",
        vllm_weight_sync_backend="ipc",
        vllm_tensor_parallel_size=1,
    )
    assert cfg.distributed_backend == "fsdp"
    assert cfg.vllm_weight_sync_backend == "ipc"

    with pytest.raises(ValueError, match="vllm_tensor_parallel_size=1"):
        CASPOConfig(
            rollout_backend="vllm",
            vllm_weight_sync_backend="ipc",
            vllm_tensor_parallel_size=2,
        )


def test_ddp_vllm_ipc_rank_local_config_is_valid():
    cfg = CASPOConfig(
        distributed_backend="ddp",
        rollout_backend="vllm",
        vllm_weight_sync_backend="ipc",
        vllm_tensor_parallel_size=1,
    )

    assert cfg.distributed_backend == "ddp"
    assert cfg.vllm_weight_sync_backend == "ipc"


def test_ddp_requires_rank_local_vllm_rollout():
    with pytest.raises(ValueError, match="rollout_backend='vllm'"):
        CASPOConfig(distributed_backend="ddp", rollout_backend="hf")

    with pytest.raises(ValueError, match="vllm_tensor_parallel_size=1"):
        CASPOConfig(
            distributed_backend="ddp",
            rollout_backend="vllm",
            vllm_tensor_parallel_size=2,
        )


def test_vineppo_requires_vllm_rollout_backend():
    with pytest.raises(ValueError, match="rollout_backend='vllm'"):
        CASPOConfig(method="vineppo", rollout_backend="hf")


def test_hybrid_shard_is_a_legal_fsdp_strategy():
    """``hybrid_shard`` (HSDP) is a first-class fsdp_sharding_strategy value.

    Constructing a config with the literal must succeed; pre-patch this
    raised because the Literal only listed full_shard / shard_grad_op /
    no_shard. The runtime device_mesh build is gated to world_size>=4 and
    is exercised in real FSDP smoke runs.
    """
    cfg = CASPOConfig(
        distributed_backend="fsdp",
        rollout_backend="vllm",
        vllm_tensor_parallel_size=1,
        fsdp_sharding_strategy="hybrid_shard",
    )
    assert cfg.fsdp_sharding_strategy == "hybrid_shard"


def test_fsdp_reduce_dtype_optional_passthrough():
    cfg = CASPOConfig()
    assert cfg.fsdp_reduce_dtype is None  # default = match torch_dtype

    cfg = CASPOConfig(fsdp_reduce_dtype="bfloat16")
    assert cfg.fsdp_reduce_dtype == "bfloat16"

    with pytest.raises(ValueError, match="fsdp_reduce_dtype"):
        CASPOConfig(fsdp_reduce_dtype="int8")


def test_activation_checkpointing_mode_literal():
    cfg = CASPOConfig()
    assert cfg.activation_checkpointing_mode == "off"

    cfg = CASPOConfig(activation_checkpointing_mode="selective")
    assert cfg.activation_checkpointing_mode == "selective"

    with pytest.raises(ValueError, match="activation_checkpointing_mode"):
        CASPOConfig(activation_checkpointing_mode="bogus")
