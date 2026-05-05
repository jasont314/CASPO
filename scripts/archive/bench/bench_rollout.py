"""Microbenchmark for VLLMRolloutEngine.

Spins up a tiny in-process vLLM rollout engine, runs ``prompts × G`` sample
requests, and reports total wall time, per-sample latency, and token-throughput
estimate. Then verifies ``engine.shutdown()`` is clean (no zombie EngineCore
subprocesses leak after the script exits).

Defaults to a tiny SmolLM2-135M-Instruct so a smoke run completes in <30s on
a single GPU. Use ``--device 0`` to pin to GPU 0; the script also sets
``CUDA_VISIBLE_DEVICES`` itself for defense-in-depth so you cannot accidentally
clobber another GPU (e.g. a live trainer on GPU 3).

CLI:
    python scripts/bench_rollout.py [--model NAME] [--prompts N] [--g K] \\
        [--max-tokens M] [--device 0]

Quick smoke test (matches the repo verification command):
    CUDA_VISIBLE_DEVICES=0 conda run -n scalable \\
        python scripts/bench_rollout.py --max-tokens 64 --prompts 4 --g 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time


_DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _stub_reward_fn(responses, ground_truths):
    # Reward isn't part of the rollout-throughput measurement; keep it cheap.
    return [0.0] * len(responses)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Microbenchmark VLLMRolloutEngine.sample throughput."
    )
    p.add_argument("--model", default=_DEFAULT_MODEL,
                   help=f"HuggingFace model id (default: {_DEFAULT_MODEL})")
    p.add_argument("--prompts", type=int, default=8,
                   help="Number of unique prompts (default: 8)")
    p.add_argument("--g", type=int, default=8,
                   help="Samples per prompt, == cfg.group_size (default: 8)")
    p.add_argument("--max-tokens", type=int, default=256,
                   help="Max response length per sample (default: 256)")
    p.add_argument("--device", type=int, default=0,
                   help="GPU index to pin via CUDA_VISIBLE_DEVICES (default: 0)")
    p.add_argument("--max-prompt-len", type=int, default=512,
                   help="Max prompt length (default: 512)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.45,
                   help="vLLM gpu_memory_utilization (default: 0.45 — tiny model)")
    p.add_argument("--enforce-eager", action="store_true", default=True,
                   help="Skip CUDA-graph compile (default: True for fast smoke)")
    return p.parse_args()


def _make_prompts(n: int):
    """Cheap, varied math-flavored prompts so prefix caching can't trivialize."""
    base = [
        "What is 12 + 7?",
        "Compute 25 * 4.",
        "If x = 3, what is 2x + 5?",
        "Find the area of a 6 by 8 rectangle.",
        "What is 100 divided by 4?",
        "Solve: 3y - 9 = 12.",
        "What is the square of 11?",
        "Add the numbers 17, 23, and 41.",
        "What is 9 factorial?",
        "Compute the 10th Fibonacci number.",
        "What is 2^10?",
        "Find the GCD of 24 and 36.",
        "Compute sin(0) + cos(0).",
        "What is the perimeter of a square with side 7?",
        "Solve x^2 = 49 for x > 0.",
        "What is the average of 5, 10, 15, 20?",
    ]
    out = []
    for i in range(n):
        out.append({"prompt": base[i % len(base)], "ground_truth": ""})
    return out


def _check_gpu_clean(device_idx: int) -> int:
    """Best-effort: report current process count on the given GPU.

    Returns -1 if nvidia-smi isn't available; the result is informational —
    we mainly want to see that nothing this script spawned is still resident.
    """
    import subprocess
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
                f"--id={device_idx}",
            ],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return -1
    pids = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    return len(pids)


def main() -> int:
    args = _parse_args()

    # Pin device BEFORE any torch / vLLM import. The engine's __init__ also
    # respects gpu_id, but setting CUDA_VISIBLE_DEVICES ourselves is a hard
    # guarantee that a misconfigured engine cannot reach GPU 3 (live trainer).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.device))

    # Imports deferred until after env pin.
    import torch  # noqa: F401  (forces CUDA init under the pinned device)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available; vLLM rollout requires a GPU.",
              file=sys.stderr)
        return 2

    from caspo.config import CASPOConfig
    from caspo.rollout.vllm_engine import VLLMRolloutEngine

    cfg = CASPOConfig(
        model_name_or_path=args.model,
        torch_dtype="bfloat16",
        trust_remote_code=False,
        max_prompt_len=int(args.max_prompt_len),
        max_response_len=int(args.max_tokens),
        group_size=int(args.g),
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        rollout_backend="vllm",
        device="cuda",
    )

    print(f"[bench] model={args.model}  prompts={args.prompts}  G={args.g}  "
          f"max_tokens={args.max_tokens}  device=cuda:0 (CUDA_VISIBLE_DEVICES={args.device})")

    pre_pids = _check_gpu_clean(args.device)
    if pre_pids >= 0:
        print(f"[bench] pre-init compute apps on GPU {args.device}: {pre_pids}")

    t_init = time.perf_counter()
    engine = VLLMRolloutEngine(
        cfg,
        _stub_reward_fn,
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        enforce_eager=bool(args.enforce_eager),
        # gpu_id left None — CUDA_VISIBLE_DEVICES already restricts to one GPU.
    )
    init_s = time.perf_counter() - t_init
    print(f"[bench] engine init: {init_s:.2f}s")

    try:
        examples = _make_prompts(int(args.prompts))

        # Warm-up: one short rollout so the first compile / cache costs don't
        # contaminate the measured run. Smaller G to stay quick.
        warm_cfg_g = cfg.group_size
        cfg.group_size = 1
        try:
            t0 = time.perf_counter()
            _ = engine.sample(examples[:1])
            warmup_s = time.perf_counter() - t0
        finally:
            cfg.group_size = warm_cfg_g
        print(f"[bench] warmup (1 prompt × 1 sample): {warmup_s:.2f}s")

        # Measured run.
        n_samples = int(args.prompts) * int(args.g)
        t0 = time.perf_counter()
        rb = engine.sample(examples)
        total_s = time.perf_counter() - t0

        # Token accounting from the response_mask (counts only valid tokens up
        # to and including the first EOS, ignoring right-padding).
        gen_tokens = int(rb.response_mask.sum().item())
        per_sample_s = total_s / max(n_samples, 1)
        toks_per_s = gen_tokens / total_s if total_s > 0 else float("nan")
        avg_toks = gen_tokens / max(n_samples, 1)

        print("[bench] === results ===")
        print(f"[bench] total samples       : {n_samples}  "
              f"({args.prompts} prompts × G={args.g})")
        print(f"[bench] total wall time     : {total_s:.3f}s")
        print(f"[bench] per-sample latency  : {per_sample_s*1000:.1f} ms")
        print(f"[bench] generated tokens    : {gen_tokens}  "
              f"(avg {avg_toks:.1f} tok/sample, cap {args.max_tokens})")
        print(f"[bench] throughput estimate : {toks_per_s:.1f} tok/s "
              f"(decode-only, batched across {n_samples} samples)")
    finally:
        t0 = time.perf_counter()
        engine.shutdown()
        shutdown_s = time.perf_counter() - t0
        print(f"[bench] engine.shutdown()   : {shutdown_s:.2f}s")

    # Idempotency check: a second shutdown must not raise.
    try:
        engine.shutdown()
        print("[bench] shutdown() idempotent: OK")
    except Exception as e:
        print(f"[bench] WARN: second shutdown() raised: {e}", file=sys.stderr)

    # Give vLLM's EngineCore subprocess a brief moment to actually exit before
    # we poll nvidia-smi. The shutdown call signals the subprocess; the kernel
    # reaping isn't strictly synchronous on all platforms.
    time.sleep(1.0)
    post_pids = _check_gpu_clean(args.device)
    if post_pids >= 0:
        print(f"[bench] post-shutdown compute apps on GPU {args.device}: {post_pids}")
        # Heuristic: if pre-init was 0 and post-shutdown is also 0, the engine
        # cleaned up. Non-zero post-shutdown could still be unrelated workloads
        # so we warn rather than fail.
        if pre_pids == 0 and post_pids > 0:
            print(f"[bench] WARN: {post_pids} compute app(s) still resident on "
                  f"GPU {args.device} after shutdown (possible zombie).",
                  file=sys.stderr)
        else:
            print("[bench] shutdown clean: no new zombie processes detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
