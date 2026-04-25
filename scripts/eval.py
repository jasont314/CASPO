"""Evaluate a CASPO checkpoint on math benchmarks.

Usage::

    python scripts/eval.py --config configs/caspo_smoke.yaml \\
        --override model_name_or_path=out/caspo/final \\
        --benchmarks math500 \\
        --k 4 --limit 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from caspo.config import CASPOConfig
from caspo.eval import evaluate, evaluate_vllm, BENCHMARKS
from scripts.collect_value_data import apply_overrides


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--benchmarks", type=str, default="math500",
                    help="comma-separated benchmark names (e.g. math500,aime24,amc23)")
    ap.add_argument("--k", type=int, default=None,
                    help="samples per problem; defaults to the benchmark's recommended k")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on # of problems for quick smoke evals")
    ap.add_argument("--temperature", type=float, default=0.35,
                    help="VinePPO eval default 0.35 (lower than rollout temp)")
    ap.add_argument("--top-p", type=float, default=0.9,
                    help="VinePPO eval default 0.9")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="defaults to cfg.max_response_len (long CoT for math benchmarks)")
    ap.add_argument("--seed", type=int, default=None,
                    help="overrides cfg.seed for sampling")
    ap.add_argument("--backend", type=str, default=None, choices=("hf", "vllm"),
                    help="overrides cfg.rollout_backend for the eval pass")
    # vLLM-only knobs (ignored for HF backend).
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    help="vLLM only; overrides cfg.vllm_gpu_memory_utilization")
    ap.add_argument("--enforce-eager", action="store_true", default=False,
                    help="vLLM only; force eager (skip CUDA-graph capture). "
                         "Useful for tight VRAM or graph-build crashes.")
    ap.add_argument("--no-enforce-eager", dest="no_enforce_eager",
                    action="store_true", default=False,
                    help="vLLM only; explicitly disable enforce_eager (override cfg).")
    ap.add_argument("--output", type=str, default=None,
                    help="path to write the JSON results file. Defaults to "
                         "{cfg.output_dir}/eval_results_{benchmarks}.json so "
                         "successive runs don't clobber each other.")
    args = ap.parse_args()

    cfg = CASPOConfig.from_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    # CLI overrides for fields that are awkward to pass via --override key=value.
    if args.backend is not None:
        cfg.rollout_backend = args.backend
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.gpu_memory_utilization is not None:
        cfg.vllm_gpu_memory_utilization = float(args.gpu_memory_utilization)
    if args.enforce_eager and args.no_enforce_eager:
        ap.error("--enforce-eager and --no-enforce-eager are mutually exclusive")
    if args.enforce_eager:
        cfg.vllm_enforce_eager = True
    elif args.no_enforce_eager:
        cfg.vllm_enforce_eager = False

    from transformers import AutoTokenizer

    tok_path = cfg.tokenizer_name_or_path or cfg.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Default to the full training response length — math benchmarks (esp. AIME)
    # need long CoT and the model was trained at this budget. The previous
    # min(1024, cfg.max_response_len) cap silently truncated long reasoning.
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else cfg.max_response_len

    requested = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    if not requested:
        ap.error("--benchmarks must contain at least one non-empty name")
    # Validate benchmark names UP FRONT — fail before paying the model-load
    # cost (~30s for vLLM init) if the user typo'd a benchmark name.
    # Cache BENCHMARKS keys once (avoid repeated dict membership checks
    # against a frozen module-level dict in the per-benchmark loop below).
    known = set(BENCHMARKS)
    benchmark_names = []
    for name in requested:
        if name not in known:
            print(f"[eval] unknown benchmark {name!r}; available: {sorted(known)}", flush=True)
            continue
        benchmark_names.append(name)
    if not benchmark_names:
        ap.error("no valid benchmarks after filtering against BENCHMARKS")

    # Hoist the output-path resolution + dir creation BEFORE the (long) eval
    # loop so we fail fast on a bad output path rather than after 20 min of
    # generation. One makedirs() call total instead of two.
    out_dir = cfg.output_dir
    if args.output is not None:
        out_path = args.output
        target_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    else:
        # Suffix with benchmark tag so successive single-benchmark runs in the
        # same output_dir don't overwrite each other.
        tag = "_".join(benchmark_names)
        out_path = os.path.join(out_dir, f"eval_results_{tag}.json")
        target_dir = out_dir
    os.makedirs(target_dir, exist_ok=True)

    results = {}
    use_vllm = (cfg.rollout_backend == "vllm")
    if use_vllm:
        # Build ONE AsyncLLM and reuse it across all benchmarks. Saves ~30s
        # of engine-init time per extra benchmark (math500+aime24+amc23 = 2x
        # ~30s saved) and lets vLLM's prefix cache hit across the boilerplate
        # shared by math problems. Single-benchmark sweeps still share the
        # same path — building one engine, using it once, shutting it down.
        shared_engine = None
        shared_loop = None
        if len(benchmark_names) >= 2:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM

            print(f"[eval] building shared vLLM engine for "
                  f"{len(benchmark_names)} benchmarks...", flush=True)
            engine_kwargs = dict(
                model=cfg.model_name_or_path,
                tokenizer=cfg.model_name_or_path,
                tokenizer_mode="auto",
                trust_remote_code=cfg.trust_remote_code,
                dtype=cfg.torch_dtype,
                tensor_parallel_size=1,
                gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
                enable_prefix_caching=True,
                enforce_eager=cfg.vllm_enforce_eager,
                max_model_len=int(max_new_tokens) + 1024,
                seed=cfg.seed,
                disable_log_stats=True,
            )
            if cfg.vllm_max_num_seqs is not None:
                engine_kwargs["max_num_seqs"] = int(cfg.vllm_max_num_seqs)
            if cfg.vllm_max_num_batched_tokens is not None:
                engine_kwargs["max_num_batched_tokens"] = int(cfg.vllm_max_num_batched_tokens)
            engine_args = AsyncEngineArgs(**engine_kwargs)
            shared_engine = AsyncLLM.from_engine_args(engine_args)
            shared_loop = asyncio.new_event_loop()

        try:
            for name in benchmark_names:
                print(f"[eval] running {name} via vLLM...", flush=True)
                out = evaluate_vllm(
                    model_name_or_path=cfg.model_name_or_path,
                    tokenizer=tokenizer,
                    benchmark=name,
                    k=args.k,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=int(max_new_tokens),
                    limit=args.limit,
                    seed=cfg.seed,
                    gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
                    enforce_eager=cfg.vllm_enforce_eager,
                    prompt_template=cfg.prompt_template,
                    trust_remote_code=cfg.trust_remote_code,
                    torch_dtype=cfg.torch_dtype,
                    max_num_seqs=cfg.vllm_max_num_seqs,
                    max_num_batched_tokens=cfg.vllm_max_num_batched_tokens,
                    engine=shared_engine,
                    loop=shared_loop,
                )
                results[name] = out
                print(f"[eval] {name}: {out}", flush=True)
                # Without a shared engine, evaluate_vllm shut down its own
                # engine; empty_cache() helps the next engine init in tight
                # VRAM. With a shared engine the caching allocator's blocks
                # are still in active use, so empty_cache() is a no-op.
                if shared_engine is None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            if shared_engine is not None:
                try:
                    shared_engine.shutdown()
                except Exception:
                    pass
            if shared_loop is not None and not shared_loop.is_closed():
                try:
                    pending = [
                        t for t in asyncio.all_tasks(loop=shared_loop)
                        if not t.done()
                    ]
                    if pending:
                        for t in pending:
                            t.cancel()
                        try:
                            shared_loop.run_until_complete(
                                asyncio.gather(*pending, return_exceptions=True)
                            )
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    shared_loop.close()
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    else:
        from transformers import AutoModelForCausalLM
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        model_kwargs = dict(torch_dtype=dtype_map[cfg.torch_dtype], trust_remote_code=cfg.trust_remote_code)
        if cfg.attn_implementation:
            model_kwargs["attn_implementation"] = cfg.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)
        # Use .to(device, non_blocking=True) instead of .cuda() — async H2D
        # copy overlaps with the next CPU-side init step (tokenizer warmup,
        # benchmark dataset load). Skip the cast entirely if the model is
        # already on the requested device (e.g. accelerate placed it).
        target_device = torch.device(cfg.device)
        try:
            current_device = next(model.parameters()).device
        except StopIteration:
            current_device = None
        if current_device != target_device:
            model.to(target_device, non_blocking=True)
        model.eval()
        for name in benchmark_names:
            print(f"[eval] running {name} via HF generate...", flush=True)
            out = evaluate(
                model=model, tokenizer=tokenizer, benchmark=name,
                k=args.k, temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=int(max_new_tokens), device=cfg.device,
                limit=args.limit, seed=cfg.seed,
                prompt_template=cfg.prompt_template,
            )
            results[name] = out
            print(f"[eval] {name}: {out}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"config": asdict(cfg), "results": results}, f, indent=2, default=str)
    print(f"[eval] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
