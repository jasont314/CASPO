#!/usr/bin/env python3
"""Post-mortem inspection for CASPO/GRPO/VinePPO runs.

Usage:
    python scripts/inspect_run.py <output_dir>

Reads phase2_*.log step lines, eval_results*.json, train_log*.jsonl,
and caspo_run_config.json. Reports config, step counts, reward/KL
trajectories, learning slope, and anomalies. Pure stdlib.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from glob import glob
from typing import Any, Iterable

# ---------- step-line regex ----------
# Example:
# [caspo step 850/1000] loss=1.5211 pg=0.6038 reward=0.031 pass@G=0.125 |A|=0.331
#   steps/r=8.2 clip_frac=0.000 ratio=0.974 lr=1.00e-06 v_loss=0.2647 v_acc=0.969 ...
STEP_RE = re.compile(r"\[(?P<method>\w+)\s+step\s+(?P<step>\d+)\s*/\s*(?P<total>\d+)\]\s+(?P<rest>.*)")
# Tokens of form key=val where key may contain |, @, _, /, letters, digits
KV_RE = re.compile(r"([A-Za-z_|][\w@|/]*)=([-+]?(?:nan|inf|\d+\.?\d*(?:[eE][-+]?\d+)?))")


def _to_float(s: str) -> float:
    sl = s.lower()
    if sl == "nan":
        return float("nan")
    if sl in ("inf", "+inf"):
        return float("inf")
    if sl == "-inf":
        return float("-inf")
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ---------- discovery ----------
def find_files(output_dir: str) -> dict[str, list[str]]:
    """Search output_dir and a logs/ subdir for relevant artifacts."""
    out: dict[str, list[str]] = {
        "phase_logs": [],
        "train_jsonl": [],
        "eval_json": [],
        "config": [],
    }
    search_roots = [output_dir, os.path.join(output_dir, "logs")]
    # also walk shallow (1 level) to be tolerant
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            full = os.path.join(root, name)
            if not os.path.isfile(full):
                continue
            if re.match(r"phase2_.*\.log$", name):
                out["phase_logs"].append(full)
            elif re.match(r".*train.*log.*\.jsonl$", name):
                out["train_jsonl"].append(full)
            elif name.startswith("eval_results") and name.endswith(".json"):
                out["eval_json"].append(full)
            elif name == "caspo_run_config.json":
                out["config"].append(full)
    # config can also live under final/ or best/
    for sub in ("final", "best"):
        cand = os.path.join(output_dir, sub, "caspo_run_config.json")
        if os.path.isfile(cand):
            out["config"].append(cand)
    # eval_results may be under subdirs
    for cand in glob(os.path.join(output_dir, "**", "eval_results*.json"), recursive=True):
        if cand not in out["eval_json"]:
            out["eval_json"].append(cand)
    return out


# ---------- parsing ----------
def parse_phase_log(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                m = STEP_RE.search(line)
                if not m:
                    continue
                row: dict[str, Any] = {
                    "method": m.group("method"),
                    "step": int(m.group("step")),
                    "total": int(m.group("total")),
                }
                for k, v in KV_RE.findall(m.group("rest")):
                    # don't let kv pairs overwrite the header fields (e.g. `total=...s` wallclock)
                    if k in ("method", "step", "total"):
                        continue
                    row[k] = _to_float(v)
                rows.append(row)
    except OSError as e:
        print(f"  warn: cannot read {path}: {e}", file=sys.stderr)
    return rows


def parse_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return rows


def load_json(path: str) -> Any:
    try:
        with open(path, "r", errors="replace") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# ---------- statistics ----------
def _finite(xs: Iterable[float]) -> list[float]:
    return [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]


def mean(xs: Iterable[float]) -> float:
    xs = _finite(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def stdev(xs: Iterable[float]) -> float:
    xs = _finite(xs)
    if len(xs) < 2:
        return float("nan")
    mu = sum(xs) / len(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def linreg_slope(ys: list[float]) -> float:
    """OLS slope of y vs index 0..n-1, ignoring non-finite y."""
    pts = [(i, y) for i, y in enumerate(ys) if isinstance(y, (int, float)) and math.isfinite(y)]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    sx = sum(p[0] for p in pts)
    sy = sum(p[1] for p in pts)
    sxx = sum(p[0] * p[0] for p in pts)
    sxy = sum(p[0] * p[1] for p in pts)
    denom = n * sxx - sx * sx
    if denom == 0:
        return float("nan")
    return (n * sxy - sx * sy) / denom


def trend_word(slope: float, ref: float) -> str:
    if not math.isfinite(slope) or not math.isfinite(ref) or ref <= 0:
        return "n/a"
    rel = slope / ref
    if abs(rel) < 0.05:
        return "flat"
    return "rising" if slope > 0 else "falling"


# ---------- anomaly detection ----------
def detect_anomalies(rows: list[dict[str, Any]]) -> list[str]:
    msgs: list[str] = []
    if not rows:
        return msgs

    # NaN/Inf in any tracked metric
    nan_steps: dict[str, list[int]] = {}
    for r in rows:
        for k in ("loss", "pg", "reward", "ratio", "kl", "v_loss"):
            v = r.get(k)
            if isinstance(v, float) and not math.isfinite(v):
                nan_steps.setdefault(k, []).append(r["step"])
    for k, steps in nan_steps.items():
        msgs.append(f"NaN/Inf in {k} at {len(steps)} step(s); first @ step {steps[0]}")

    # Sudden KL spike: any window where kl > 5x rolling median
    kls = [(r["step"], r.get("kl")) for r in rows if isinstance(r.get("kl"), (int, float))]
    kls = [(s, v) for s, v in kls if math.isfinite(v)]
    if len(kls) >= 20:
        vals = [v for _, v in kls]
        med = sorted(vals)[len(vals) // 2]
        if med > 0:
            spikes = [(s, v) for s, v in kls if v > 5 * med and v > 1e-3]
            if spikes:
                first_s, first_v = spikes[0]
                msgs.append(f"KL spike: {len(spikes)} step(s) > 5x median ({med:.2e}); first @ step {first_s} kl={first_v:.3e}")

    # clip_frac collapse: trailing window all near zero after some history
    cfs = [(r["step"], r.get("clip_frac")) for r in rows if isinstance(r.get("clip_frac"), (int, float))]
    cfs = [(s, v) for s, v in cfs if math.isfinite(v)]
    if len(cfs) >= 50:
        tail = cfs[-min(100, len(cfs) // 2):]
        if all(v < 1e-4 for _, v in tail):
            msgs.append(f"clip_frac collapse: all 0 over last {len(tail)} steps (no off-policy clipping; check ratio/lr)")

    # ratio drift far from 1
    ratios = [r.get("ratio") for r in rows if isinstance(r.get("ratio"), (int, float))]
    ratios = _finite(ratios)
    if ratios:
        last = ratios[-min(50, len(ratios)):]
        if last and (abs(mean(last) - 1.0) > 0.1):
            msgs.append(f"ratio drift: tail mean={mean(last):.3f} (|.-1|>0.1)")

    # reward floor: tail all-zero suggests no learning signal
    rwds = [r.get("reward") for r in rows if isinstance(r.get("reward"), (int, float))]
    rwds = _finite(rwds)
    if len(rwds) >= 50:
        tail = rwds[-min(100, len(rwds) // 2):]
        if max(tail) < 1e-6:
            msgs.append(f"reward floor: all 0 over last {len(tail)} steps")

    return msgs


# ---------- printing ----------
def fmt(v: Any, n: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        if not math.isfinite(v):
            return str(v)
        return f"{v:.{n}g}"
    return str(v)


def print_kv_table(title: str, items: list[tuple[str, Any]]) -> None:
    print(f"\n=== {title} ===")
    if not items:
        print("  (empty)")
        return
    keylen = max(len(k) for k, _ in items)
    for k, v in items:
        print(f"  {k.ljust(keylen)}  {v}")


def summarize_run(output_dir: str) -> int:
    output_dir = os.path.abspath(output_dir)
    print(f"\n# CASPO run inspection: {output_dir}")
    if not os.path.isdir(output_dir):
        print("ERROR: directory does not exist", file=sys.stderr)
        return 2

    files = find_files(output_dir)
    print_kv_table(
        "discovered files",
        [
            ("phase logs", ", ".join(os.path.basename(p) for p in files["phase_logs"]) or "(none)"),
            ("train jsonl", ", ".join(os.path.basename(p) for p in files["train_jsonl"]) or "(none)"),
            ("eval json", ", ".join(os.path.basename(p) for p in files["eval_json"]) or "(none)"),
            ("config", ", ".join(os.path.basename(p) for p in files["config"]) or "(none)"),
        ],
    )

    # ---- config ----
    cfg: dict[str, Any] = {}
    if files["config"]:
        cfg = load_json(files["config"][0]) or {}
    if cfg:
        keys = [
            "method", "model_name_or_path", "dataset_name", "max_steps",
            "lr", "kl_coef", "clip_eps_low", "clip_eps_high",
            "group_size", "prompts_per_step", "epochs_per_rollout",
            "use_adb", "use_dlw", "gamma", "value_beta",
        ]
        print_kv_table("config", [(k, fmt(cfg.get(k))) for k in keys if k in cfg])

    # ---- step lines from phase logs (split by method) ----
    rows_all: list[dict[str, Any]] = []
    for p in files["phase_logs"]:
        rows_all.extend(parse_phase_log(p))

    by_method: dict[str, list[dict[str, Any]]] = {}
    for r in rows_all:
        by_method.setdefault(r["method"], []).append(r)
    for k in by_method:
        by_method[k].sort(key=lambda r: r["step"])

    if not by_method:
        print("\n=== training trajectory ===")
        print("  no [<method> step ...] lines found in phase logs")

    # primary method = config method if present, else the one with most rows
    primary = cfg.get("method") if isinstance(cfg.get("method"), str) else None
    if primary not in by_method:
        primary = max(by_method, key=lambda k: len(by_method[k])) if by_method else None

    if by_method:
        print(f"\n=== methods found ===")
        for m, rs in sorted(by_method.items(), key=lambda kv: -len(kv[1])):
            tag = "  <- primary" if m == primary else ""
            print(f"  {m}: {len(rs)} step lines, last step {rs[-1]['step']}/{rs[-1].get('total')}{tag}")

    for method_name, rows in by_method.items():
        if method_name != primary:
            continue
        last = rows[-1]
        method = last.get("method", "?")
        total = last.get("total", "?")
        completed = last["step"]
        # tail window for trajectory stats
        N = min(100, len(rows))
        tail = rows[-N:]

        def col(name: str) -> list[float]:
            return [r.get(name) for r in tail if isinstance(r.get(name), (int, float))]

        rwd_tail = col("reward")
        kl_tail = col("kl")
        ratio_tail = col("ratio")
        absA_tail = col("|A|")
        clip_tail = col("clip_frac")
        loss_tail = col("loss")

        rwd_full = [r.get("reward") for r in rows if isinstance(r.get("reward"), (int, float))]
        rwd_full = _finite(rwd_full)
        rwd_slope = linreg_slope(rwd_tail)
        rwd_std = stdev(rwd_tail)
        # noise threshold: slope must move > 1 std over the window
        noise_per_step = (rwd_std / max(N, 1)) if math.isfinite(rwd_std) else float("nan")

        verdict = "n/a"
        if math.isfinite(rwd_slope) and math.isfinite(noise_per_step):
            if abs(rwd_slope) <= noise_per_step:
                verdict = "FLAT (slope <= per-step noise; no clear learning)"
            elif rwd_slope > 0:
                verdict = f"rising (slope {rwd_slope:+.3e}/step)"
            else:
                verdict = f"falling (slope {rwd_slope:+.3e}/step)"

        items = [
            ("method", method),
            ("steps completed", f"{completed} / {total}"),
            ("rows parsed", len(rows)),
            ("tail window", N),
            (f"mean reward (last {N})", fmt(mean(rwd_tail))),
            (f"std reward (last {N})", fmt(rwd_std)),
            (f"reward slope/step (last {N})", fmt(rwd_slope, 3)),
            ("learning verdict", verdict),
            (f"mean loss (last {N})", fmt(mean(loss_tail))),
            (f"mean |A| (last {N})", fmt(mean(absA_tail))),
            (f"mean ratio (last {N})", fmt(mean(ratio_tail))),
            (f"mean clip_frac (last {N})", fmt(mean(clip_tail))),
            (f"mean kl (last {N})", fmt(mean(kl_tail))),
            (
                "kl trend",
                trend_word(linreg_slope(kl_tail), abs(mean(kl_tail)) or 1e-9)
                + f" (slope {fmt(linreg_slope(kl_tail), 3)})",
            ),
            ("max reward (full)", fmt(max(rwd_full)) if rwd_full else "n/a"),
        ]
        # last-row snapshot of common keys
        snap_keys = ["loss", "pg", "reward", "pass@G", "|A|", "ratio", "clip_frac", "kl", "v_loss", "v_acc", "lr"]
        snap = "  ".join(f"{k}={fmt(last.get(k))}" for k in snap_keys if k in last)
        items.append(("last step snapshot", snap))
        print_kv_table("training trajectory", items)

        anomalies = detect_anomalies(rows)
        print("\n=== anomalies ===")
        if not anomalies:
            print("  none detected")
        else:
            for a in anomalies:
                print(f"  - {a}")

    # ---- value training jsonl (if any) ----
    for jp in files["train_jsonl"]:
        rows_j = parse_jsonl(jp)
        if not rows_j:
            continue
        last = rows_j[-1]
        losses = [r.get("loss") for r in rows_j if isinstance(r.get("loss"), (int, float))]
        accs = [r.get("acc_at_last") for r in rows_j if isinstance(r.get("acc_at_last"), (int, float))]
        items = [
            ("path", os.path.basename(jp)),
            ("rows", len(rows_j)),
            ("final step", fmt(last.get("step"))),
            ("final loss", fmt(last.get("loss"))),
            ("min loss", fmt(min(_finite(losses))) if losses else "n/a"),
            ("final acc_at_last", fmt(last.get("acc_at_last"))),
            ("max acc_at_last", fmt(max(_finite(accs))) if accs else "n/a"),
        ]
        print_kv_table(f"jsonl: {os.path.basename(jp)}", items)

    # ---- eval results ----
    for ep in files["eval_json"]:
        ev = load_json(ep)
        if not isinstance(ev, dict):
            continue
        # try common keys
        flat = []
        for k in ("pass@1", "pass@k", "pass_at_1", "pass_at_k", "accuracy", "mean_reward",
                  "n_samples", "n_problems", "model", "step", "ckpt"):
            if k in ev:
                flat.append((k, fmt(ev[k])))
        # nested "results" / "summary"
        for sub in ("results", "summary", "metrics"):
            if isinstance(ev.get(sub), dict):
                for k, v in ev[sub].items():
                    flat.append((f"{sub}.{k}", fmt(v)))
        if not flat:
            # fall back: top-level scalars
            for k, v in ev.items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    flat.append((k, fmt(v)))
        print_kv_table(f"eval: {os.path.basename(ep)}", flat[:30])

    return 0


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python scripts/inspect_run.py <output_dir>", file=sys.stderr)
        return 2
    return summarize_run(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
