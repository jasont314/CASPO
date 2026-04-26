"""Config validation / comparison tool for CASPO YAML configs.

Loads every config in ``configs/*.yaml`` via ``CASPOConfig.from_yaml`` and
reports key training hyperparameters, plus sanity checks on rollout/minibatch
math:

  * rollout_responses = prompts_per_step * group_size -- responses generated
    before PPO epochs are run.
  * optimizer_batch_responses = micro_batch_size * grad_accum_steps --
    responses consumed by one optimizer update.
  * optimizer_batch_responses must evenly tile rollout_responses. This allows
    paper-faithful VinePPO settings such as 512 rollout responses trained in
    64-response minibatches for multiple PPO epochs.

Usage
-----
    python scripts/validate_configs.py
    python scripts/validate_configs.py --diff
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Make ``caspo`` importable when this script is run from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from caspo.config import CASPOConfig  # noqa: E402


# Keys that show up in the per-config report and the side-by-side diff. Order
# here is the column order in ``--diff``.
REPORT_KEYS: tuple[str, ...] = (
    "method",
    "lr",
    "online_value_lr",
    "caspo_advantage_transform",
    "kl_coef",
    "max_steps",
    "group_size",
    "prompts_per_step",
    "micro_batch_size",
    "logprob_micro_batch_size",
    "grad_accum_steps",
    "vllm_enforce_eager",
    "vllm_multi_sample_mode",
    "vllm_weight_sync_backend",
    "vllm_return_logprobs",
    "vllm_gpu_memory_utilization",
    "vllm_max_num_seqs",
    "vllm_max_inflight_requests",
    "distributed_backend",
    "save_every",
    "eval_every",
    "value_save_every",
)


@dataclass
class ConfigReport:
    name: str
    path: Path
    cfg: CASPOConfig | None
    error: str | None  # populated if from_yaml failed
    warnings_emitted: list[str]

    @property
    def target_effective_batch(self) -> int:
        """``group_size * prompts_per_step`` -- responses per macro step."""
        assert self.cfg is not None
        return self.cfg.group_size * self.cfg.prompts_per_step

    @property
    def actual_effective_responses(self) -> int:
        """How many responses one optimizer step actually consumes.

        ``micro_batch_size`` is in responses and ``grad_accum_steps`` chains
        them into a PPO minibatch. One rollout can contain multiple optimizer
        minibatches per epoch.
        """
        assert self.cfg is not None
        return (
            self.cfg.grad_accum_steps
            * self.cfg.micro_batch_size
        )


# ---------- loading ----------


def _load_one(path: Path) -> ConfigReport:
    captured: list[str] = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            cfg = CASPOConfig.from_yaml(str(path))
            err: str | None = None
        except Exception as exc:  # noqa: BLE001
            cfg = None
            err = f"{type(exc).__name__}: {exc}"
        for entry in w:
            captured.append(str(entry.message))
    return ConfigReport(
        name=path.stem,
        path=path,
        cfg=cfg,
        error=err,
        warnings_emitted=captured,
    )


def _discover(configs_dir: Path) -> list[Path]:
    return sorted(p for p in configs_dir.glob("*.yaml") if p.is_file())


# ---------- validation ----------


def _validate(report: ConfigReport) -> list[str]:
    """Return a list of human-readable warning strings.

    These are *additional* checks on top of CASPOConfig.__post_init__ -- they
    inspect cross-field invariants relevant to the trainer's batching.
    """
    issues: list[str] = []
    cfg = report.cfg
    if cfg is None:
        return issues
    value_only = report.name.startswith("value_")

    # 1. optimizer minibatches must evenly tile the rollout response pool.
    # The rollout produces ``prompts_per_step * group_size`` responses; the
    # trainer iterates over them in micro-batches of size ``micro_batch_size``
    # for ``grad_accum_steps`` steps per optimizer update. The product
    # micro_batch_size * grad_accum_steps is the PPO minibatch size in
    # responses. Paper-faithful VinePPO uses 512 rollout responses and a
    # 64-response PPO minibatch, so equality here would be too strict.
    if not value_only:
        rollout_responses = cfg.prompts_per_step * cfg.group_size
        consumed = cfg.micro_batch_size * cfg.grad_accum_steps
        if consumed <= 0:
            issues.append(
                f"micro_batch_size*grad_accum_steps={consumed} must be positive"
            )
        elif rollout_responses % consumed != 0:
            issues.append(
                f"micro_batch_size*grad_accum_steps={consumed} does not evenly "
                f"tile prompts_per_step*group_size={rollout_responses} "
                f"(rollout produces {rollout_responses} responses, optimizer "
                f"minibatch consumes {consumed})"
            )

    # 2. group_size divisibility: the trainer's group standardize assumes
    # groups are not split across micro-batches. Require group_size to divide
    # micro_batch_size, OR micro_batch_size to divide group_size cleanly.
    if cfg.standardize_advantage_scope == "group":
        gs, mbs = cfg.group_size, cfg.micro_batch_size
        if mbs >= gs:
            if mbs % gs != 0:
                issues.append(
                    f"standardize_advantage_scope='group' but "
                    f"micro_batch_size={mbs} is not a multiple of "
                    f"group_size={gs}"
                )
        else:
            if gs % mbs != 0:
                issues.append(
                    f"standardize_advantage_scope='group' but "
                    f"group_size={gs} is not a multiple of "
                    f"micro_batch_size={mbs}"
                )

    # 3. GRPO needs at least 2 per group; covered by __post_init__ but
    # re-check explicitly for visibility.
    if cfg.method == "grpo" and cfg.group_size < 2:
        issues.append(
            f"method='grpo' requires group_size>=2, got {cfg.group_size}"
        )

    if cfg.method == "vineppo" and cfg.rollout_backend != "vllm":
        issues.append(
            "method='vineppo' requires rollout_backend='vllm' "
            "because sample_with_prefix() is required"
        )

    # 4. value model is referenced by CASPO/VinePPO but only required by
    # CASPO for the V_phi prefix value head; warn if unset.
    if cfg.method == "caspo" and not value_only and not cfg.prefix_value_path:
        issues.append(
            "method='caspo' but prefix_value_path is unset "
            "(phase-2 needs a phase-1 checkpoint)"
        )

    if cfg.update_value_during_policy and cfg.online_value_lr >= 1e-5:
        issues.append(
            f"online_value_lr={cfg.online_value_lr:g} is high for the "
            "full-model value update path; use ~1e-6 unless phi is PEFT/LoRA-only"
        )

    return issues


# ---------- formatting ----------


def _fmt_value(v: Any) -> str:
    if isinstance(v, float):
        # Compact: 1e-6 instead of 1.0000000000000001e-06
        if v == 0.0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{v:.2e}"
        return f"{v:g}"
    return str(v)


def _print_single(report: ConfigReport) -> None:
    print(f"=== {report.name} ({report.path.name}) ===")
    if report.error is not None:
        print(f"  LOAD ERROR: {report.error}")
        return
    cfg = report.cfg
    assert cfg is not None
    for key in REPORT_KEYS:
        print(f"  {key:>22s}: {_fmt_value(getattr(cfg, key))}")
    print(
        f"  {'rollout_responses':>22s}: "
        f"{report.target_effective_batch}  (=group_size*prompts_per_step)"
    )
    print(
        f"  {'optimizer_batch':>22s}: "
        f"{cfg.micro_batch_size * cfg.grad_accum_steps}  "
        f"(=micro_batch_size*grad_accum_steps)"
    )
    if cfg.micro_batch_size * cfg.grad_accum_steps > 0:
        print(
            f"  {'batches_per_epoch':>22s}: "
            f"{(cfg.prompts_per_step * cfg.group_size) // (cfg.micro_batch_size * cfg.grad_accum_steps)}"
        )
    issues = _validate(report)
    if issues:
        print("  WARNINGS:")
        for msg in issues:
            print(f"    - {msg}")
    else:
        print("  OK")
    if report.warnings_emitted:
        print("  load-time warnings:")
        for msg in report.warnings_emitted:
            print(f"    - {msg}")


# ---------- diff table ----------


def _print_diff(reports: list[ConfigReport]) -> None:
    """Side-by-side table: rows = hyperparams, columns = configs."""
    columns = [r.name for r in reports if r.cfg is not None]
    if not columns:
        print("(no successfully loaded configs to diff)")
        return

    rows: list[tuple[str, list[str]]] = []
    for key in REPORT_KEYS:
        rows.append(
            (
                key,
                [
                    _fmt_value(getattr(r.cfg, key)) if r.cfg is not None else "ERR"
                    for r in reports
                ],
            )
        )
    # Derived rows
    rows.append(
        (
            "target_eff_batch",
            [
                str(r.target_effective_batch) if r.cfg is not None else "ERR"
                for r in reports
            ],
        )
    )
    rows.append(
        (
            "rollout_responses",
            [
                str(r.cfg.prompts_per_step * r.cfg.group_size)
                if r.cfg is not None
                else "ERR"
                for r in reports
            ],
        )
    )
    rows.append(
        (
            "optimizer_consumes",
            [
                (
                    "n/a"
                    if r.name.startswith("value_")
                    else str(r.cfg.micro_batch_size * r.cfg.grad_accum_steps)
                )
                if r.cfg is not None
                else "ERR"
                for r in reports
            ],
        )
    )
    rows.append(
        (
            "batches_per_epoch",
            [
                (
                    "n/a"
                    if r.name.startswith("value_")
                    else str(
                        (r.cfg.prompts_per_step * r.cfg.group_size)
                        // (r.cfg.micro_batch_size * r.cfg.grad_accum_steps)
                    )
                    if r.cfg is not None
                    and r.cfg.micro_batch_size * r.cfg.grad_accum_steps > 0
                    and (r.cfg.prompts_per_step * r.cfg.group_size)
                    % (r.cfg.micro_batch_size * r.cfg.grad_accum_steps)
                    == 0
                    else "NO"
                )
                if r.cfg is not None
                else "ERR"
                for r in reports
            ],
        )
    )

    # Column widths
    name_col_w = max(len("config"), max(len(k) for k, _ in rows))
    col_w = [
        max(len(name), max(len(row[1][i]) for row in rows))
        for i, name in enumerate(columns)
    ]

    # Header
    header = f"{'config':>{name_col_w}s} | " + " | ".join(
        f"{name:>{w}s}" for name, w in zip(columns, col_w)
    )
    print(header)
    print("-" * len(header))
    for key, vals in rows:
        printable = [v for v, r in zip(vals, [r for r in reports if r.cfg is not None])]
        line = f"{key:>{name_col_w}s} | " + " | ".join(
            f"{v:>{w}s}" for v, w in zip(printable, col_w)
        )
        print(line)

    # Errored configs
    errored = [r for r in reports if r.cfg is None]
    if errored:
        print()
        print("Failed to load:")
        for r in errored:
            print(f"  - {r.name}: {r.error}")


# ---------- entry ----------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and compare CASPO YAML configs."
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Print a side-by-side hyperparameter table across all configs.",
    )
    parser.add_argument(
        "--configs-dir",
        type=Path,
        default=_REPO_ROOT / "configs",
        help="Directory containing *.yaml configs (default: ./configs)",
    )
    args = parser.parse_args(argv)

    paths = _discover(args.configs_dir)
    if not paths:
        print(f"No *.yaml found in {args.configs_dir}")
        return 1

    reports = [_load_one(p) for p in paths]

    if args.diff:
        _print_diff(reports)
        # Per-config issues block at the bottom
        any_issues = False
        for r in reports:
            issues = _validate(r) if r.cfg is not None else []
            if issues or r.error:
                if not any_issues:
                    print()
                    print("Issues:")
                    any_issues = True
                if r.error:
                    print(f"  [{r.name}] LOAD ERROR: {r.error}")
                for msg in issues:
                    print(f"  [{r.name}] {msg}")
        if not any_issues:
            print()
            print("All configs OK.")
    else:
        for r in reports:
            _print_single(r)
            print()

    # Exit non-zero if any config failed to load.
    return 0 if all(r.error is None for r in reports) else 2


if __name__ == "__main__":
    sys.exit(main())
