"""Collect (prompt, response, outcome) data for phase-1 IPVRM training.

Generates G rollouts per prompt with the SFT model, scores each with the
math verifier, and saves a tokenized .pt file that the value-model trainer
consumes directly.

Output schema (single .pt file)::

    {
      "prompt_ids":    LongTensor [N, P_max]   left-padded
      "prompt_mask":   LongTensor [N, P_max]
      "response_ids":  LongTensor [N, R_max]   right-padded
      "response_mask": LongTensor [N, R_max]
      "outcomes":      FloatTensor [N]         in {0., 1.}
      "raw_prompts":   list[str] length N
      "raw_responses": list[str] length N
      "ground_truths": list[str] length N
      "config_snapshot": dict (asdict(cfg))
    }

Usage::

    python scripts/collect_value_data.py --config configs/value_smoke.yaml \\
        [--override key=value ...] [--output value_data.pt] [--num-prompts 1000]
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, fields
from typing import get_args, get_origin, Literal, Union

import torch
from typing import List

from caspo.config import CASPOConfig
from caspo.data import load_train_dataset
from caspo.reward import MathRewardFn
from caspo.rollout import build_rollout_engine


# ---------------------------------------------------------------------------
# Config override coercion (mirrors the LatEntRL pattern)
# ---------------------------------------------------------------------------

def _coerce(value: str, annotation):
    """Coerce a CLI string to the dataclass field's type.

    With ``from __future__ import annotations`` enabled in caspo/config.py, all
    field annotations are stringified at class-definition time, so ``f.type``
    returns ``"float"`` rather than the ``float`` type. Handle both forms.
    """
    origin = get_origin(annotation)
    name = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))

    if annotation is bool or name == "bool":
        if value.lower() in {"true", "1", "yes"}:
            return True
        if value.lower() in {"false", "0", "no"}:
            return False
        raise ValueError(f"cannot coerce {value!r} to bool")
    if annotation is int or name == "int":
        return int(value)
    if annotation is float or name == "float":
        return float(value)
    if annotation is str or name == "str":
        return value
    if origin is Union:
        args = [a for a in get_args(annotation) if a is not type(None)]
        for a in args:
            try:
                return _coerce(value, a)
            except Exception:
                continue
        return value
    if origin is Literal:
        return value
    # String annotations like "Optional[bool]" / "Optional[int]" can't be
    # introspected via get_origin/get_args (they're plain strings under PEP 563).
    # Try the common inner-type names so e.g. --override foo=true on an
    # Optional[bool] field still coerces correctly.
    if isinstance(annotation, str):
        inner = annotation
        for prefix in ("Optional[", "Union["):
            if inner.startswith(prefix) and inner.endswith("]"):
                inner = inner[len(prefix):-1]
                break
        # Strip ", None" from Union[..., None] forms.
        inner = inner.replace(", None", "").replace("None, ", "").strip()
        # Take the first comma-separated arg as the primary type.
        first = inner.split(",", 1)[0].strip()
        if first in {"bool", "int", "float", "str"} and first != name:
            return _coerce(value, first)
    # Literal['a','b'], etc. fall through — return the raw string. The
    # dataclass doesn't enforce Literal at runtime, so this is fine.
    return value


def apply_overrides(cfg: CASPOConfig, overrides: list[str]) -> CASPOConfig:
    field_map = {f.name: f for f in fields(cfg)}
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        if k not in field_map:
            raise KeyError(f"unknown config field {k!r}")
        setattr(cfg, k, _coerce(v, field_map[k].type))
    # Overrides mutate the dataclass after construction, so rerun the same
    # validation/coercion path that YAML loading uses. This catches typos such
    # as --override method=ppoo before any model weights are loaded.
    cfg.__post_init__()
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--output", type=str, default=None,
                    help="output .pt path; defaults to <output_dir>/value_data.pt")
    # --num-prompts and --num-batches are alternative ways to bound the run;
    # accepting both silently composes (truncates twice), which is confusing.
    count_group = ap.add_mutually_exclusive_group()
    count_group.add_argument(
        "--num-prompts", type=int, default=None,
        help="number of distinct prompts to roll out; defaults to all",
    )
    count_group.add_argument(
        "--num-batches", type=int, default=None,
        help="alternative to --num-prompts: number of prompts_per_step batches to run",
    )
    ap.add_argument("--filter-mixed-outcomes", dest="filter_mixed_outcomes",
                    action="store_true", default=True,
                    help="drop prompts whose G rollouts are all correct or "
                         "all incorrect — these saturate the BCE-margin loss "
                         "and waste training compute (IPVRM §C). Default ON.")
    ap.add_argument("--no-filter-mixed-outcomes", dest="filter_mixed_outcomes",
                    action="store_false")
    ap.add_argument("--shard", type=str, default=None,
                    help="i/N — process every N-th prompt starting from i "
                         "(0-indexed). Used to parallelize collection across "
                         "GPUs: each shard writes its own .pt; merge after.")
    ap.add_argument("--paper-pairing", action="store_true", default=False,
                    help="paper-faithful pairing (IPVRM §4.1): for each "
                         "mixed-outcome prompt, randomly select 1 correct "
                         "+ 1 incorrect rollout; discard the rest. Yields "
                         "exactly 2 rows per kept prompt and a 50/50 "
                         "balanced dataset. Default OFF (keep all G rollouts "
                         "per prompt).")
    args = ap.parse_args()

    cfg = CASPOConfig.from_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)
    # IPVRM phase-1 uses higher temperature than phase-2 RL — paper §4.1.
    # We rebind cfg.rollout_temperature for the duration of this script so
    # HFRolloutSampler picks it up (it reads from cfg directly).
    if cfg.value_data_temperature != cfg.rollout_temperature:
        print(
            f"[collect] using value_data_temperature={cfg.value_data_temperature} "
            f"(was rollout_temperature={cfg.rollout_temperature})",
            flush=True,
        )
        cfg.rollout_temperature = cfg.value_data_temperature

    output_path = args.output or os.path.join(cfg.output_dir, "value_data.pt")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    from transformers import AutoTokenizer

    tok_path = cfg.tokenizer_name_or_path or cfg.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    examples = list(load_train_dataset(cfg, tokenizer=tokenizer))
    # Apply --shard i/N BEFORE --num-prompts so all shards see disjoint
    # prompts even when --num-prompts is set globally. Interleaved slicing
    # (examples[i::N]) keeps prompt difficulty distribution similar across
    # shards (vs. contiguous slicing which would bias each shard).
    if args.shard is not None:
        try:
            shard_i, shard_n = (int(s) for s in args.shard.split("/"))
        except Exception:
            raise SystemExit(
                f"--shard must be 'i/N' (e.g. '0/4'); got {args.shard!r}"
            )
        if shard_n <= 0 or not (0 <= shard_i < shard_n):
            raise SystemExit(
                f"--shard {args.shard!r}: need 0 <= i < N and N > 0"
            )
        before = len(examples)
        examples = examples[shard_i::shard_n]
        print(
            f"[collect] --shard {shard_i}/{shard_n}: kept {len(examples)} "
            f"prompts of {before} (every {shard_n}th starting at {shard_i})",
            flush=True,
        )
    if args.num_prompts is not None:
        if args.num_prompts > len(examples):
            print(
                f"[collect] WARN: --num-prompts={args.num_prompts} exceeds available "
                f"prompts ({len(examples)}); using all {len(examples)}.",
                flush=True,
            )
        examples = examples[: args.num_prompts]
    if not examples:
        print("No training examples after filtering; aborting.", file=sys.stderr)
        sys.exit(1)
    print(f"[collect] {len(examples)} prompts loaded", flush=True)

    # vLLM loads its own tokenizer from cfg.model_name_or_path; if the user
    # set cfg.tokenizer_name_or_path to something different (e.g. a sibling
    # checkpoint with a re-tuned chat template), the tokens we encode here
    # for prompt_ids will not match what vLLM tokenizes server-side, leading
    # to off-by-one prompt-token alignment in downstream V_φ training.
    if (
        cfg.tokenizer_name_or_path
        and cfg.tokenizer_name_or_path != cfg.model_name_or_path
        and cfg.rollout_backend == "vllm"
    ):
        print(
            f"[collect] WARN: tokenizer_name_or_path={cfg.tokenizer_name_or_path!r} "
            f"differs from model_name_or_path={cfg.model_name_or_path!r}; "
            f"vLLM uses the model's own tokenizer, so prompt_ids saved here may "
            f"not match what vLLM tokenized at rollout time.",
            flush=True,
        )

    # Build rollout engine. With rollout_backend=vllm (recommended for Rho-1B+),
    # vLLM loads the model itself; the trainer process doesn't need to.
    print(f"[collect] rollout_backend={cfg.rollout_backend}", flush=True)
    if cfg.rollout_backend == "vllm":
        sampler = build_rollout_engine(
            cfg,
            MathRewardFn(),
            gpu_memory_utilization=cfg.vllm_gpu_memory_utilization,
            tensor_parallel_size=cfg.vllm_tensor_parallel_size,
            enforce_eager=cfg.vllm_enforce_eager,
            seed=cfg.seed,
            max_num_seqs=cfg.vllm_max_num_seqs,
            max_num_batched_tokens=cfg.vllm_max_num_batched_tokens,
            max_inflight_requests=cfg.vllm_max_inflight_requests,
        )
    else:
        from transformers import AutoModelForCausalLM
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        model_kwargs = dict(
            torch_dtype=dtype_map[cfg.torch_dtype],
            trust_remote_code=cfg.trust_remote_code,
        )
        if cfg.attn_implementation:
            model_kwargs["attn_implementation"] = cfg.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **model_kwargs)
        model.to(cfg.device)
        model.eval()
        sampler = build_rollout_engine(
            cfg, MathRewardFn(), model=model, tokenizer=tokenizer,
        )

    prompts_per_step = max(1, int(cfg.prompts_per_step))
    if args.num_batches is not None:
        n_batches = int(args.num_batches)
        examples = examples[: n_batches * prompts_per_step]

    # Wrap everything past sampler construction in try/finally so the vLLM
    # EngineCore subprocess (and our event loop) gets shut down even on errors,
    # sys.exit(...), or no rows surviving the mixed-outcome filter. Otherwise
    # the GPU stays held by an orphaned EngineCore.
    try:
        _run_collection(
            cfg=cfg,
            args=args,
            sampler=sampler,
            tokenizer=tokenizer,
            examples=examples,
            prompts_per_step=prompts_per_step,
            output_path=output_path,
        )
    finally:
        if hasattr(sampler, "shutdown"):
            try:
                sampler.shutdown()
            except Exception as e:
                print(f"[collect] sampler shutdown warning: {e}", flush=True)


def _run_collection(*, cfg, args, sampler, tokenizer, examples, prompts_per_step, output_path) -> None:
    all_prompt_ids = []
    all_prompt_mask = []
    all_response_ids = []
    all_response_mask = []
    all_outcomes = []
    all_raw_prompts = []
    all_raw_responses = []
    all_raw_gts = []

    n_done = 0
    for start in range(0, len(examples), prompts_per_step):
        batch = examples[start : start + prompts_per_step]
        if not batch:
            break
        rb = sampler.sample(batch)
        # Tile prompts to per-response. prompt_index is [N] mapping each
        # rollout back to its prompt slot. For raw_prompts/ground_truths
        # (Python lists of length num_prompts) we pull idx as a Python list
        # *once* and use list-indexing — avoids a generator-expression-per-row.
        idx = rb.prompt_index  # [N]
        tiled_prompt_ids = rb.prompt_ids.index_select(0, idx)
        tiled_prompt_mask = rb.prompt_mask.index_select(0, idx)
        idx_list = idx.tolist()
        rb_raw_prompts = rb.raw_prompts
        rb_gts = rb.ground_truths
        tiled_prompts = [rb_raw_prompts[i] for i in idx_list]
        tiled_gts = [rb_gts[i] for i in idx_list]

        all_prompt_ids.append(tiled_prompt_ids)
        all_prompt_mask.append(tiled_prompt_mask)
        all_response_ids.append(rb.response_ids)
        all_response_mask.append(rb.response_mask)
        all_outcomes.append(rb.rewards.float())
        all_raw_prompts.extend(tiled_prompts)
        all_raw_responses.extend(rb.raw_responses)
        all_raw_gts.extend(tiled_gts)

        n_done += len(batch)
        # Progress logging is once-per-batch (not inner loop) — keep it that way.
        print(
            f"[collect] {n_done}/{len(examples)} prompts | "
            f"acc={rb.rewards.mean().item():.3f}",
            flush=True,
        )

    # IPVRM mixed-outcome filter (paper §C / §3.3 motivation): drop prompts
    # where all G rollouts share the same outcome. Saturated examples push
    # v̄ further into a saturated sigmoid regime without informative gradient.
    if args.filter_mixed_outcomes:
        G = max(1, int(cfg.group_size))
        # Concatenate everything once, compute a single boolean row-mask over
        # all rollouts, and use it to gather. This replaces two nested Python
        # loops over (batch, local_prompt, rollout) with one vectorized pass.
        all_outcomes_cat = torch.cat(all_outcomes, dim=0).float()
        n_total = all_outcomes_cat.numel()
        assert n_total % G == 0, (
            f"total outcomes {n_total} not divisible by G={G}"
        )
        n_prompts_seen = n_total // G
        grouped = all_outcomes_cat.view(n_prompts_seen, G)
        n_correct = (grouped >= 0.5).sum(dim=1)
        mixed_prompts = (n_correct > 0) & (n_correct < G)  # [n_prompts_seen]

        if args.paper_pairing:
            # Paper-faithful (IPVRM §4.1): for each mixed-outcome prompt,
            # randomly select ONE correct rollout + ONE incorrect rollout.
            # Result: exactly 2 rows per kept prompt; dataset is 50/50
            # positive/negative by construction.
            import random as _random
            _rng = _random.Random(int(getattr(cfg, "seed", 0)))
            row_keep = torch.zeros(n_total, dtype=torch.bool)
            for p in range(n_prompts_seen):
                if not mixed_prompts[p].item():
                    continue
                pos_local = (grouped[p] >= 0.5).nonzero(as_tuple=True)[0].tolist()
                neg_local = (grouped[p] < 0.5).nonzero(as_tuple=True)[0].tolist()
                pos_pick = _rng.choice(pos_local)
                neg_pick = _rng.choice(neg_local)
                row_keep[p * G + pos_pick] = True
                row_keep[p * G + neg_pick] = True
            n_kept = int(mixed_prompts.sum().item())
            print(
                f"[collect] paper-faithful pairing: kept {n_kept}/{n_prompts_seen} prompts "
                f"({100*n_kept/max(n_prompts_seen,1):.1f}%) → "
                f"{n_kept * 2} rollouts (1 pos + 1 neg per prompt)",
                flush=True,
            )
        else:
            # Default: keep ALL G rollouts per mixed-outcome prompt.
            row_keep = mixed_prompts.unsqueeze(1).expand(-1, G).reshape(-1)
            n_kept = int(mixed_prompts.sum().item())
            print(
                f"[collect] mixed-outcome filter: kept {n_kept}/{n_prompts_seen} prompts "
                f"({100*n_kept/max(n_prompts_seen,1):.1f}%) → "
                f"{n_kept * G} rollouts",
                flush=True,
            )
        if n_kept == 0:
            print(
                "[collect] WARNING: zero mixed-outcome prompts. Either temperature "
                "is too low (try --override value_data_temperature=1.0) or the "
                "model is uniformly correct/wrong on this dataset.",
                flush=True,
            )
            print("[collect] no rows survived filter; aborting.", file=sys.stderr)
            sys.exit(1)
        # Rebuild kept rows in a single pass. We can't torch.cat the per-batch
        # ids tensors yet (they have different widths), so we apply the
        # boolean mask to each batch slice using its rollout-count offsets.
        row_keep_list = row_keep.tolist()
        new_p_ids, new_p_mask = [], []
        new_r_ids, new_r_mask = [], []
        new_outcomes = []
        new_raw_prompts: List[str] = []
        new_raw_responses: List[str] = []
        new_raw_gts: List[str] = []
        offset = 0
        for b_idx, batch_outcomes in enumerate(all_outcomes):
            n_in_batch = batch_outcomes.numel()
            batch_mask = row_keep[offset : offset + n_in_batch]
            if batch_mask.any().item():
                # Boolean indexing is contiguous-gather in C, no Python row loop.
                new_p_ids.append(all_prompt_ids[b_idx][batch_mask])
                new_p_mask.append(all_prompt_mask[b_idx][batch_mask])
                new_r_ids.append(all_response_ids[b_idx][batch_mask])
                new_r_mask.append(all_response_mask[b_idx][batch_mask])
                new_outcomes.append(batch_outcomes[batch_mask])
            offset += n_in_batch
        # Filter the parallel raw_* string lists with the global row mask in
        # one comprehension each — no per-batch Python indirection.
        new_raw_prompts = [s for s, k in zip(all_raw_prompts, row_keep_list) if k]
        new_raw_responses = [s for s, k in zip(all_raw_responses, row_keep_list) if k]
        new_raw_gts = [s for s, k in zip(all_raw_gts, row_keep_list) if k]
        all_prompt_ids = new_p_ids
        all_prompt_mask = new_p_mask
        all_response_ids = new_r_ids
        all_response_mask = new_r_mask
        all_outcomes = new_outcomes
        all_raw_prompts = new_raw_prompts
        all_raw_responses = new_raw_responses
        all_raw_gts = new_raw_gts

    # Right-pad everything to the same width across batches. We pre-allocate
    # one [N, width] output tensor per field and fill it batch-slice by
    # batch-slice — this avoids the prior pattern of building `len(batches)`
    # intermediate padded copies and then concatenating (2x peak memory).
    P_max = max(t.shape[1] for t in all_prompt_ids)
    R_max = max(t.shape[1] for t in all_response_ids)
    pad_id = int(tokenizer.pad_token_id)
    n_rows = sum(t.shape[0] for t in all_prompt_ids)

    def _alloc_left(width: int, pad_value: int, dtype) -> torch.Tensor:
        return torch.full((n_rows, width), pad_value, dtype=dtype)

    def _alloc_right(width: int, pad_value: int, dtype) -> torch.Tensor:
        return torch.full((n_rows, width), pad_value, dtype=dtype)

    prompt_ids = _alloc_left(P_max, pad_id, all_prompt_ids[0].dtype)
    prompt_mask = _alloc_left(P_max, 0, all_prompt_mask[0].dtype)
    response_ids = _alloc_right(R_max, pad_id, all_response_ids[0].dtype)
    response_mask = _alloc_right(R_max, 0, all_response_mask[0].dtype)

    row = 0
    for p_ids, p_msk, r_ids, r_msk in zip(
        all_prompt_ids, all_prompt_mask, all_response_ids, all_response_mask
    ):
        b = p_ids.shape[0]
        # Left-pad: copy into the right edge of a P_max-wide buffer.
        pw = p_ids.shape[1]
        prompt_ids[row : row + b, P_max - pw :] = p_ids
        prompt_mask[row : row + b, P_max - pw :] = p_msk
        # Right-pad: copy into the left edge of an R_max-wide buffer.
        rw = r_ids.shape[1]
        response_ids[row : row + b, :rw] = r_ids
        response_mask[row : row + b, :rw] = r_msk
        row += b
    outcomes = torch.cat(all_outcomes, dim=0).float()

    # Tensors are already on CPU (RolloutBatch contract). Avoid redundant
    # .cpu()/.to(long) round-trips — they alloc full copies. Cast only if the
    # underlying dtype actually differs from int64.
    def _ensure_long(t: torch.Tensor) -> torch.Tensor:
        return t if t.dtype == torch.long else t.to(torch.long)

    blob = {
        "prompt_ids": _ensure_long(prompt_ids),
        "prompt_mask": _ensure_long(prompt_mask),
        "response_ids": _ensure_long(response_ids),
        "response_mask": _ensure_long(response_mask),
        "outcomes": outcomes,
        "raw_prompts": all_raw_prompts,
        "raw_responses": all_raw_responses,
        "ground_truths": all_raw_gts,
        "config_snapshot": asdict(cfg),
    }
    # Final save is once (not per-batch). torch.save uses pickle+zipfile;
    # for this schema (mixed tensors+strings) safetensors would force a
    # second pass for the string lists, so torch.save remains correct here.
    torch.save(blob, output_path)
    print(
        f"[collect] saved {prompt_ids.shape[0]} (prompt,response,outcome) rows to "
        f"{output_path} (P_max={P_max}, R_max={R_max}, pos_frac={outcomes.mean().item():.3f})",
        flush=True,
    )
    # Sampler shutdown is handled by the main() try/finally so the GPU is
    # freed even on errors / sys.exit() / no-rows-survived-filter.


if __name__ == "__main__":
    main()
