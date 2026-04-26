"""Validate vLLM CUDA-IPC weight sync against a live trainer/vLLM pair.

This is intentionally a standalone module instead of a stdin one-liner because
vLLM's spawn-based EngineCore needs an importable ``__main__`` file.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _first_matching_parameter(
    trainer_named_parameters: Iterable,
    worker_names: set[str],
) -> str:
    for name, param in trainer_named_parameters:
        if name in worker_names and param.is_cuda and param.is_floating_point():
            return name
    raise RuntimeError("no matching CUDA floating parameter between trainer and vLLM")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/caspo_rho1b_math.yaml")
    parser.add_argument("--output-dir", default="/tmp/caspo_vllm_ipc_probe")
    parser.add_argument("--delta", type=float, default=1.0)
    args = parser.parse_args()

    from caspo.config import CASPOConfig
    from caspo.trainer import CASPOTrainer

    cfg = CASPOConfig.from_yaml(args.config)
    cfg.method = "ppo"
    cfg.output_dir = args.output_dir
    cfg.max_steps = 1
    cfg.save_every = 0
    cfg.log_every = 1
    cfg.wandb_enabled = False
    cfg.wandb_mode = "disabled"
    cfg.update_value_during_policy = False
    cfg.vllm_weight_sync_backend = "ipc"
    cfg.vllm_return_logprobs = False

    trainer = CASPOTrainer(cfg)
    try:
        sampler = trainer.sampler

        def worker_names(worker):
            return [name for name, _ in worker.get_model().named_parameters()]

        worker_name_lists = sampler._run(sampler._collective_rpc(worker_names))
        worker_names0 = set(
            worker_name_lists[0]
            if isinstance(worker_name_lists, list) else worker_name_lists
        )
        target = _first_matching_parameter(
            trainer.model.named_parameters(), worker_names0,
        )

        def checksum(worker, name):
            param = worker.get_model().get_parameter(name)
            return float(param.float().sum().detach().cpu().item())

        before = sampler._run(sampler._collective_rpc(checksum, args=(target,)))[0]
        param = dict(trainer.model.named_parameters())[target]
        param.data.view(-1)[0].add_(float(args.delta))
        sync_t = sampler.sync_weights_from_model(trainer.model)
        after = sampler._run(sampler._collective_rpc(checksum, args=(target,)))[0]
        observed = after - before
        print(
            {
                "target": target,
                "before": before,
                "after": after,
                "delta_expected": float(args.delta),
                "delta_observed": observed,
                "sync_t": sync_t,
            },
            flush=True,
        )
        tol = max(1e-2, abs(float(args.delta)) * 1e-2)
        if abs(observed - float(args.delta)) > tol:
            raise RuntimeError(
                "IPC sync checksum mismatch: "
                f"expected {args.delta}, got {observed}, tol={tol}"
            )
    finally:
        sampler = getattr(trainer, "sampler", None)
        if sampler is not None and hasattr(sampler, "shutdown"):
            sampler.shutdown()


if __name__ == "__main__":
    main()
