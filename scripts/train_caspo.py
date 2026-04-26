"""Phase-2 trainer: CASPO policy optimization with step-level TD from V_φ.

Usage::

    python scripts/train_caspo.py --config configs/caspo_smoke.yaml \\
        --override prefix_value_path=out/value/final
"""

from __future__ import annotations

# NOTE: keep this module's top-level imports stdlib-only. Anything pulling in
# torch / transformers / caspo.* costs ~5 s and balloons `--help` into a
# multi-second wait. Heavy imports are deferred into ``main()`` below so the
# argparse path (--help, bad flags, missing --config) returns instantly. The
# saved cold-start is meaningful both for humans iterating on flags and for
# launch scripts that probe `--help` to validate before spawning a real run.
import argparse
import os
import signal
import sys
from typing import List

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _validate_cfg(cfg) -> None:
    """Cheap pre-flight checks that fail before the trainer loads ~7B params.

    The trainer's ``__init__`` loads the policy (and optionally a ref policy)
    before checking ``prefix_value_path``, ``step_delimiter``, etc. — wasting
    several minutes of model download/init when the config is wrong. Catch
    the obvious cases up-front instead.
    """
    if not cfg.model_name_or_path:
        raise ValueError("cfg.model_name_or_path is empty")
    if cfg.method == "caspo":
        if not cfg.prefix_value_path:
            raise ValueError(
                "cfg.prefix_value_path must point to a phase-1 PrefixValueModel "
                "checkpoint when method=caspo. Run scripts/train_value.py first "
                "or pass --override prefix_value_path=<dir>."
            )
        if not os.path.isdir(cfg.prefix_value_path):
            raise FileNotFoundError(
                f"cfg.prefix_value_path={cfg.prefix_value_path!r} is not a "
                f"directory (expected a phase-1 checkpoint dir)."
            )
    if not cfg.step_delimiter:
        raise ValueError("cfg.step_delimiter must be a non-empty string")
    if cfg.group_size < 1:
        raise ValueError(f"cfg.group_size={cfg.group_size} must be >= 1")
    if cfg.prompts_per_step < 1:
        raise ValueError(
            f"cfg.prompts_per_step={cfg.prompts_per_step} must be >= 1"
        )
    if cfg.method not in {"ppo", "caspo", "grpo", "vineppo"}:
        raise ValueError(
            f"cfg.method={cfg.method!r} not in {{'ppo','caspo','grpo','vineppo'}}"
        )
    if cfg.method == "vineppo" and cfg.rollout_backend != "vllm":
        raise ValueError(
            "cfg.method='vineppo' requires cfg.rollout_backend='vllm' because "
            "MC prefix value estimation needs sample_with_prefix()."
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    if world_size > 1 and cfg.distributed_backend == "none":
        raise ValueError(
            "WORLD_SIZE>1 but distributed_backend='none'. Pass "
            "--override distributed_backend=fsdp/ddp or launch a single process."
        )
    # Make sure the output dir is writable now, not 5 min into training.
    os.makedirs(cfg.output_dir, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--override", action="append", default=[],
                    help="repeatable key=value override applied on top of YAML")
    args = ap.parse_args()

    overrides: List[str] = list(args.override or [])

    # Deferred imports: only pay the ~5 s torch/transformers/caspo cost once
    # we've cleared argparse and know we're actually going to train. This keeps
    # `--help` and arg-validation errors snappy (~0.3 s vs ~2.5 s).
    import copy as _copy

    from caspo.config import CASPOConfig
    from caspo.trainer import CASPOTrainer
    from scripts.collect_value_data import apply_overrides

    cfg = CASPOConfig.from_yaml(args.config)
    # Defensive copy: apply_overrides mutates the dataclass in place via
    # setattr. Callers (and future importers of this main) shouldn't see
    # surprising side effects on the freshly-loaded YAML object.
    cfg = apply_overrides(_copy.deepcopy(cfg), overrides)

    _validate_cfg(cfg)

    trainer = CASPOTrainer(cfg)

    # Flush wandb on SIGTERM/SIGINT so killed runs don't leave wandb in
    # "running" state forever. The default handlers raise KeyboardInterrupt
    # only for SIGINT; SIGTERM bypasses Python's atexit entirely. We install
    # a handler that finishes the wandb run (best-effort) and re-raises the
    # default behavior so the parent shell sees the right exit code.
    def _graceful_shutdown(signum, _frame):
        wb = getattr(trainer, "_wandb", None)
        if wb is not None:
            try:
                wb.finish(exit_code=128 + signum)
            except Exception:
                pass
        # Use os._exit-style code so the shell sees 128+signum (SIGTERM=143,
        # SIGINT=130), matching default signal-driven exits.
        sys.exit(128 + signum)

    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _graceful_shutdown)
        except (ValueError, OSError):
            # signal.signal raises on non-main-thread or unsupported platforms;
            # not fatal, just means no graceful flush there.
            pass

    # Wrap train() in try/finally so a Ctrl-C (KeyboardInterrupt) or any
    # unhandled exception inside rollout still gets a chance to release the
    # vLLM EngineCore subprocess. Without this, the EngineCore stays alive
    # on the GPU as a zombie holding ~37 GB. scripts/kill_zombies.sh exists
    # as a backstop, but best-effort shutdown here avoids needing it.
    try:
        trainer.train()
    except BaseException as _e:
        # Surface the traceback so silent rank-skew exits (one rank exits
        # cleanly while others hang waiting on a collective) are diagnosable.
        # Without this, the bare try/finally swallows the exception detail
        # because finally() runs destroy_process_group, which reraises a
        # different error or returns cleanly. Print the full traceback
        # synchronously to stderr so each rank's log captures *its own*
        # failure cause.
        import traceback as _tb

        try:
            _rank = int(os.environ.get("RANK", "0"))
        except (TypeError, ValueError):
            _rank = 0
        print(
            f"[train_caspo rank={_rank}] EXCEPTION in trainer.train(): "
            f"{type(_e).__name__}: {_e}",
            file=sys.stderr,
            flush=True,
        )
        _tb.print_exc()
        raise
    finally:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            print(
                f"[train_caspo] distributed shutdown warning: {e}",
                flush=True,
            )
        sampler = getattr(trainer, "sampler", None)
        if sampler is not None and hasattr(sampler, "shutdown"):
            try:
                sampler.shutdown()
            except Exception as e:
                print(f"[train_caspo] sampler shutdown warning: {e}", flush=True)


if __name__ == "__main__":
    main()
