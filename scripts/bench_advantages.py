"""Microbench for caspo.algo.advantages.

Times the four step-advantage primitives on synthetic [B, R] tensors:

    step_values_from_log_ratios
    step_td_advantage
    standardize_step_advantage  (scope='batch' and scope='group')
    broadcast_step_advantage_to_tokens

For each kernel we also try torch.compile(mode="reduce-overhead") and report
the eager-vs-compiled timings side-by-side. We deliberately ignore the first
few iterations as warm-up, run a CUDA event timer over the remainder, and
report mean us/iter.

CLI:
    python scripts/bench_advantages.py [--device 0] [--B 8] [--T 1024] \
        [--S 32] [--iters 100]

Conventions match caspo.algo.advantages: log_ratio is [B, R] (R == T tokens),
boundary_after has S True entries per row, step_count == S for every row.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import torch

# Make the in-repo caspo package importable when run as `python scripts/...`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from caspo.algo.advantages import (  # noqa: E402
    step_values_from_log_ratios,
    step_td_advantage,
    standardize_step_advantage,
    broadcast_step_advantage_to_tokens,
)


def _make_inputs(B: int, T: int, S: int, device: torch.device, dtype: torch.dtype):
    """Build a synthetic batch with S evenly-spaced step boundaries per row.

    All rows share the same step layout (cheap to construct, structurally
    realistic). The last token of each step is the boundary. Tokens within a
    step share a step_id; step_id is set to -1 only for tokens beyond the
    response (we keep response_mask all-True here to maximise per-iter work,
    which is the worst case the kernels actually see).
    """
    if S > T:
        raise ValueError(f"S={S} must be <= T={T}")

    # Evenly partition T tokens into S steps. Step t covers tokens
    # [t * T // S, (t+1) * T // S), boundary_after at index (t+1)*T//S - 1.
    bounds = torch.tensor(
        [(t + 1) * T // S - 1 for t in range(S)], dtype=torch.long, device=device
    )
    boundary_after = torch.zeros((B, T), dtype=torch.bool, device=device)
    boundary_after[:, bounds] = True

    # Per-token step_id: 0..S-1 across the row.
    arange_T = torch.arange(T, device=device, dtype=torch.long)
    # For each token position p, step_id = number of completed boundaries at p
    # = floor(p * S / T).
    step_id_row = (arange_T * S) // T
    step_id = step_id_row.unsqueeze(0).expand(B, T).contiguous()

    response_mask = torch.ones((B, T), dtype=torch.long, device=device)
    step_count = torch.full((B,), S, dtype=torch.long, device=device)

    log_ratio = torch.randn((B, T), device=device, dtype=dtype) * 0.05
    final_reward = torch.randn((B,), device=device, dtype=dtype)

    return {
        "log_ratio": log_ratio,
        "response_mask": response_mask,
        "boundary_after": boundary_after,
        "step_count": step_count,
        "step_id": step_id,
        "final_reward": final_reward,
    }


def _time_fn(
    fn: Callable[[], torch.Tensor],
    iters: int,
    warmup: int,
    device: torch.device,
) -> float:
    """Return mean microseconds per call."""
    # Warm-up pass: lets autotuners and lazy initialisation settle.
    for _ in range(warmup):
        out = fn()
        del out
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn()
        end.record()
        torch.cuda.synchronize(device)
        del out
        elapsed_ms = start.elapsed_time(end)
        return (elapsed_ms * 1000.0) / iters
    # CPU fallback: wall-clock.
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    elapsed_s = time.perf_counter() - t0
    del out
    return (elapsed_s * 1e6) / iters


def _try_compile(fn):
    """Wrap fn in torch.compile; on failure fall back to eager and flag it."""
    try:
        return torch.compile(fn, mode="reduce-overhead", fullgraph=False), True
    except Exception:  # pragma: no cover - depends on torch version
        return fn, False


def _format_row(name: str, eager_us: float, compiled_us: float | None) -> str:
    if compiled_us is None:
        speedup = "n/a"
        compiled_str = "skipped"
    else:
        compiled_str = f"{compiled_us:9.2f}"
        speedup = f"{eager_us / compiled_us:5.2f}x"
    return f"  {name:<42s} {eager_us:9.2f}    {compiled_str}    {speedup}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument("--B", type=int, default=8, help="batch size")
    p.add_argument("--T", type=int, default=1024, help="response length (tokens)")
    p.add_argument("--S", type=int, default=32, help="steps per row")
    p.add_argument("--iters", type=int, default=100, help="timed iterations")
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="warm-up iterations excluded from the timing window",
    )
    p.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
        help="dtype for log_ratio / V_step (matches train-time dtype)",
    )
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="skip torch.compile (eager only)",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable; running on CPU (timings will not match GPU)")
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    print("=" * 88)
    print(
        f"bench_advantages: B={args.B} T={args.T} S={args.S} iters={args.iters} "
        f"warmup={args.warmup} dtype={args.dtype} device={device}"
    )
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")
    print("=" * 88)

    inputs = _make_inputs(args.B, args.T, args.S, device, dtype)

    # Pre-compute V_step once for the kernels that consume it. We re-run the
    # producing kernel inside the timed loop too, so this is purely setup.
    V_step = step_values_from_log_ratios(
        inputs["log_ratio"],
        inputs["response_mask"],
        inputs["boundary_after"],
        inputs["step_count"],
    )
    A_step = step_td_advantage(V_step, inputs["final_reward"], inputs["step_count"])

    # Define the closures we want to time. Each one returns a tensor so we can
    # ensure the kernels actually execute (no DCE).
    def call_step_values():
        return step_values_from_log_ratios(
            inputs["log_ratio"],
            inputs["response_mask"],
            inputs["boundary_after"],
            inputs["step_count"],
        )

    def call_step_td():
        return step_td_advantage(
            V_step, inputs["final_reward"], inputs["step_count"]
        )

    def call_standardize_batch():
        return standardize_step_advantage(
            A_step, inputs["step_count"], scope="batch"
        )

    def call_standardize_group():
        # group_size must divide B; pick gcd(B, 2) -> 2 for B>=2 else 1.
        gs = 2 if args.B >= 2 and args.B % 2 == 0 else 1
        return standardize_step_advantage(
            A_step, inputs["step_count"], scope="group", group_size=gs
        )

    def call_broadcast():
        return broadcast_step_advantage_to_tokens(
            A_step, inputs["step_id"], inputs["response_mask"]
        )

    kernels: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("step_values_from_log_ratios", call_step_values),
        ("step_td_advantage", call_step_td),
        ("standardize_step_advantage [batch]", call_standardize_batch),
        ("standardize_step_advantage [group]", call_standardize_group),
        ("broadcast_step_advantage_to_tokens", call_broadcast),
    ]

    print(
        f"  {'kernel':<42s} {'eager (us)':>9s}    {'compiled':>9s}    speedup"
    )
    print("  " + "-" * 84)

    for name, fn in kernels:
        eager_us = _time_fn(fn, args.iters, args.warmup, device)

        compiled_us: float | None = None
        if not args.no_compile and device.type == "cuda":
            compiled_fn, ok = _try_compile(fn)
            if ok:
                try:
                    compiled_us = _time_fn(
                        compiled_fn, args.iters, max(args.warmup, 3), device
                    )
                except Exception as exc:  # pragma: no cover
                    print(f"  [warn] compile-time error on {name}: {exc!s}")
                    compiled_us = None

        print(_format_row(name, eager_us, compiled_us))

    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
