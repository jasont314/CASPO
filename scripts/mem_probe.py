#!/usr/bin/env python
"""GPU memory probe sidecar.

Polls NVML (pynvml preferred, nvidia-smi fallback) to monitor GPU memory
usage, utilization, and free/used drift over time. Optionally filters to a
specific PID's allocations on the chosen device.

Examples
--------
    # Watch device 0 for 3 seconds at 1s cadence
    python scripts/mem_probe.py --device 0 --interval 1 --duration 3

    # Watch a specific training PID on its assigned GPU, dump JSON
    python scripts/mem_probe.py --pid 12345 --interval 5 --duration 600 \
        --format json --output /tmp/caspo_mem.json

The probe is read-only (NVML queries + sampling) and does not import torch,
so it is safe to run alongside live training jobs.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional


# ---------------------------------------------------------------------------
# Backend abstraction
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    timestamp: float
    elapsed: float
    device: int
    mem_used_mib: float
    mem_free_mib: float
    mem_total_mib: float
    gpu_util_pct: Optional[float]
    mem_util_pct: Optional[float]
    pid_mem_mib: Optional[float]  # process-specific memory if --pid given
    backend: str


class NVMLBackend:
    name = "pynvml"

    def __init__(self, device: int):
        import pynvml  # type: ignore

        self.pynvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        self.device = device

    def sample(self, pid: Optional[int]) -> Sample:
        p = self.pynvml
        mem = p.nvmlDeviceGetMemoryInfo(self.handle)
        try:
            util = p.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util = float(util.gpu)
            mem_util = float(util.memory)
        except p.NVMLError:
            gpu_util = None
            mem_util = None

        pid_mem: Optional[float] = None
        if pid is not None:
            try:
                procs = p.nvmlDeviceGetComputeRunningProcesses(self.handle)
            except p.NVMLError:
                procs = []
            for proc in procs:
                if proc.pid == pid and proc.usedGpuMemory is not None:
                    pid_mem = proc.usedGpuMemory / (1024 ** 2)
                    break
            if pid_mem is None:
                # fall back to graphics processes (rare on training nodes)
                try:
                    gprocs = p.nvmlDeviceGetGraphicsRunningProcesses(self.handle)
                except p.NVMLError:
                    gprocs = []
                for proc in gprocs:
                    if proc.pid == pid and proc.usedGpuMemory is not None:
                        pid_mem = proc.usedGpuMemory / (1024 ** 2)
                        break

        return Sample(
            timestamp=time.time(),
            elapsed=0.0,  # filled by caller
            device=self.device,
            mem_used_mib=mem.used / (1024 ** 2),
            mem_free_mib=mem.free / (1024 ** 2),
            mem_total_mib=mem.total / (1024 ** 2),
            gpu_util_pct=gpu_util,
            mem_util_pct=mem_util,
            pid_mem_mib=pid_mem,
            backend=self.name,
        )

    def close(self) -> None:
        try:
            self.pynvml.nvmlShutdown()
        except Exception:
            pass


class SmiBackend:
    name = "nvidia-smi"

    def __init__(self, device: int):
        if not shutil.which("nvidia-smi"):
            raise RuntimeError("nvidia-smi not found on PATH")
        self.device = device

    def _query(self, fields: str, extra: Optional[List[str]] = None) -> str:
        cmd = [
            "nvidia-smi",
            f"--id={self.device}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
        if extra:
            cmd = extra
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10)
        return out.decode("utf-8", errors="replace").strip()

    def sample(self, pid: Optional[int]) -> Sample:
        line = self._query(
            "memory.used,memory.free,memory.total,utilization.gpu,utilization.memory"
        )
        parts = [p.strip() for p in line.split(",")]
        used, free, total, gu, mu = parts
        gpu_util = float(gu) if gu and gu.lower() != "[n/a]" else None
        mem_util = float(mu) if mu and mu.lower() != "[n/a]" else None

        pid_mem: Optional[float] = None
        if pid is not None:
            try:
                proc_out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={self.device}",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    stderr=subprocess.STDOUT,
                    timeout=10,
                ).decode("utf-8", errors="replace")
                for row in proc_out.strip().splitlines():
                    if not row.strip():
                        continue
                    rp, rmem = [c.strip() for c in row.split(",")]
                    if rp.isdigit() and int(rp) == pid:
                        pid_mem = float(rmem)
                        break
            except subprocess.SubprocessError:
                pass

        return Sample(
            timestamp=time.time(),
            elapsed=0.0,
            device=self.device,
            mem_used_mib=float(used),
            mem_free_mib=float(free),
            mem_total_mib=float(total),
            gpu_util_pct=gpu_util,
            mem_util_pct=mem_util,
            pid_mem_mib=pid_mem,
            backend=self.name,
        )

    def close(self) -> None:
        pass


def make_backend(device: int, prefer: str = "auto"):
    if prefer in ("auto", "pynvml"):
        try:
            return NVMLBackend(device)
        except Exception as e:
            if prefer == "pynvml":
                raise
            print(
                f"[mem_probe] pynvml unavailable ({e}); falling back to nvidia-smi",
                file=sys.stderr,
            )
    return SmiBackend(device)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


CSV_FIELDS = [
    "timestamp",
    "elapsed",
    "device",
    "mem_used_mib",
    "mem_free_mib",
    "mem_total_mib",
    "gpu_util_pct",
    "mem_util_pct",
    "pid_mem_mib",
    "backend",
]


def emit_csv_header(stream) -> None:
    w = csv.writer(stream)
    w.writerow(CSV_FIELDS)


def emit_csv_row(stream, s: Sample) -> None:
    w = csv.writer(stream)
    w.writerow(
        [
            f"{s.timestamp:.3f}",
            f"{s.elapsed:.3f}",
            s.device,
            f"{s.mem_used_mib:.1f}",
            f"{s.mem_free_mib:.1f}",
            f"{s.mem_total_mib:.1f}",
            "" if s.gpu_util_pct is None else f"{s.gpu_util_pct:.1f}",
            "" if s.mem_util_pct is None else f"{s.mem_util_pct:.1f}",
            "" if s.pid_mem_mib is None else f"{s.pid_mem_mib:.1f}",
            s.backend,
        ]
    )


def summarize(samples: List[Sample]) -> dict:
    if not samples:
        return {}
    used = [s.mem_used_mib for s in samples]
    free = [s.mem_free_mib for s in samples]
    pid_mem = [s.pid_mem_mib for s in samples if s.pid_mem_mib is not None]
    # "Fragmentation drift" proxy: how much (used + free) deviates from total
    # over time, plus the spread between min/max free while used appears flat.
    drift = max(free) - min(free) if free else 0.0
    used_spread = max(used) - min(used) if used else 0.0
    summary = {
        "n_samples": len(samples),
        "duration_s": samples[-1].elapsed - samples[0].elapsed,
        "device": samples[0].device,
        "backend": samples[0].backend,
        "mem_used_mib": {
            "min": min(used),
            "max": max(used),
            "mean": sum(used) / len(used),
            "peak": max(used),
        },
        "mem_free_mib": {
            "min": min(free),
            "max": max(free),
            "mean": sum(free) / len(free),
        },
        "free_drift_mib": drift,
        "used_spread_mib": used_spread,
        # Heuristic: if free drifts a lot while used is roughly flat, the
        # allocator is likely fragmenting (returning blocks but not the same
        # ones the OS reports).
        "frag_hint_mib": max(0.0, drift - used_spread),
    }
    if pid_mem:
        summary["pid_mem_mib"] = {
            "min": min(pid_mem),
            "max": max(pid_mem),
            "mean": sum(pid_mem) / len(pid_mem),
            "peak": max(pid_mem),
        }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sidecar GPU memory probe (NVML / nvidia-smi).",
    )
    p.add_argument("--device", type=int, default=0, help="GPU index (default 0)")
    p.add_argument(
        "--pid",
        type=int,
        default=None,
        help="Optional PID to filter per-process memory (NVML compute apps).",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds (default 5).",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Total duration in seconds (default 600). Use 0 for unbounded.",
    )
    p.add_argument(
        "--format",
        choices=("csv", "json", "jsonl"),
        default="csv",
        help="Output format (default csv).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file (default stdout).",
    )
    p.add_argument(
        "--backend",
        choices=("auto", "pynvml", "smi"),
        default="auto",
        help="Force a backend (default auto: try pynvml first).",
    )
    p.add_argument(
        "--no-summary",
        action="store_true",
        help="Suppress trailing summary block (stderr).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-sample status lines on stderr.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    prefer = "pynvml" if args.backend == "pynvml" else (
        "smi" if args.backend == "smi" else "auto"
    )
    if args.backend == "smi":
        backend = SmiBackend(args.device)
    else:
        backend = make_backend(args.device, prefer=prefer)

    if args.output:
        out_stream = open(args.output, "w", encoding="utf-8", newline="")
        owns_stream = True
    else:
        out_stream = sys.stdout
        owns_stream = False

    samples: List[Sample] = []
    start = time.time()
    deadline = start + args.duration if args.duration > 0 else None

    if args.format == "csv":
        emit_csv_header(out_stream)
        out_stream.flush()

    try:
        next_tick = start
        while True:
            now = time.time()
            if deadline is not None and now >= deadline:
                break
            if now < next_tick:
                time.sleep(min(0.05, next_tick - now))
                continue

            try:
                s = backend.sample(args.pid)
            except Exception as e:
                print(f"[mem_probe] sample error: {e}", file=sys.stderr)
                next_tick += args.interval
                continue
            s.elapsed = s.timestamp - start
            samples.append(s)

            if args.format == "csv":
                emit_csv_row(out_stream, s)
            elif args.format == "jsonl":
                out_stream.write(json.dumps(asdict(s)) + "\n")
            # json: write at end as a single array
            out_stream.flush()

            if not args.quiet:
                pidstr = (
                    f" pid={args.pid}:{s.pid_mem_mib:.0f}MiB"
                    if s.pid_mem_mib is not None
                    else ""
                )
                print(
                    f"[mem_probe] t={s.elapsed:6.1f}s dev={s.device} "
                    f"used={s.mem_used_mib:7.0f}/{s.mem_total_mib:.0f}MiB "
                    f"free={s.mem_free_mib:7.0f}MiB "
                    f"gpu={s.gpu_util_pct}% mem={s.mem_util_pct}%{pidstr}",
                    file=sys.stderr,
                )

            next_tick += args.interval
    except KeyboardInterrupt:
        print("[mem_probe] interrupted; writing summary", file=sys.stderr)
    finally:
        if args.format == "json":
            json.dump([asdict(s) for s in samples], out_stream, indent=2)
            out_stream.write("\n")
            out_stream.flush()

        if owns_stream:
            out_stream.close()

        if not args.no_summary:
            summary = summarize(samples)
            print("[mem_probe] summary:", file=sys.stderr)
            print(json.dumps(summary, indent=2), file=sys.stderr)

        backend.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
