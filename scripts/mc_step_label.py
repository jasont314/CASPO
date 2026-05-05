"""Math-Shepherd-style MC step labeling for Qwen2.5-Math (or any base model).

Pipeline:
  1. Phase A: K base rollouts per prompt, segment into reasoning steps, verify outcomes.
  2. Phase B: for each rollout, sample S random step boundaries; for each prefix s_t,
              run J MC continuations with adaptive max_new_tokens = max_response_len - len(s_t),
              compute p_hat_t = (#correct)/J.
  3. Save (prefix_ids, prefix_mask, step_token_idx, p_hat) tuples to .pt for V_phi training.

Designed to run on a single GPU at a time (uses vLLM). Use --shard i/N to parallelize.

Output schema (torch.save dict):
  prompt_ids:    [N, P]    long
  prompt_mask:   [N, P]    long
  response_ids:  [N, R]    long              # full base rollout response (for reference)
  response_mask: [N, R]    long
  step_end_idx:  [N]       long              # response-token idx at end of the labeled step
  p_hat:         [N]       float             # MC-estimated P(success | prefix up through step)
  outcomes:      [N]       float             # terminal outcome of the BASE rollout (binary)
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import List

import torch

sys.path.insert(0, "/home/jason/experiment/CASPO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--dataset_config", default=None)
    ap.add_argument("--dataset_split", default="train")
    ap.add_argument("--prompt_template", default="Problem: {query}\nSolution:")
    ap.add_argument("--output", required=True)
    ap.add_argument("--num_prompts", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/N for sharded collect")
    ap.add_argument("--K", type=int, default=8, help="base rollouts per prompt")
    ap.add_argument("--J", type=int, default=8, help="MC continuations per labeled prefix")
    ap.add_argument("--steps_per_response", type=int, default=5,
                    help="randomly sample this many step boundaries per rollout")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--max_response_len", type=int, default=2048)
    ap.add_argument("--max_train_prefix_len", type=int, default=0,
                    help="cap step-prefix tokens for training (0=use max_response_len). "
                         "Decouples collection budget from training prefix budget. Default "
                         "0 keeps train/deploy aligned. Set to a smaller value (e.g. 1024) "
                         "only if you specifically want to hedge against rambling-tail noise "
                         "at the cost of leaving late RL prefixes out-of-distribution.")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.92)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mixed_only", action="store_true", default=True,
                    help="only label rollouts of mixed-outcome prompts")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Lazy imports
    from datasets import load_dataset
    from vllm import LLM, SamplingParams
    from caspo.reward.math_verifier import MathRewardFn
    from caspo.segmentation.steps import segment_responses_batch_latex_aware
    from transformers import AutoTokenizer

    print(f"[mc] loading tokenizer + model: {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[mc] init vLLM (max_model_len={args.max_prompt_len + args.max_response_len})", flush=True)
    # NOTE: kv_cache_dtype="fp8" was tried as a "free 25-35%" speedup but it
    # collapses Qwen2.5-Math-1.5B to multilingual-token gibberish at
    # max_response_len=2048 (verified empirically: v2 bf16 KV → coherent
    # \boxed{} outputs; v3 fp8 KV → 1/75 mixed-outcome with garbage tokens
    # past ~500 tokens). fp8 KV's accumulated quantization error breaks long
    # autoregressive Qwen2 generation even though it's "fine" at PPL eval.
    llm = LLM(
        model=args.model, dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_prompt_len + args.max_response_len,
        enable_prefix_caching=True, trust_remote_code=True,
        # Phase B has many concurrent decode sequences (J=8 × ~4000 prefixes per
        # bucket). Pushing max_num_seqs and max_num_batched_tokens above the
        # default (256 / 8192) keeps the GPU saturated through the long Phase B
        # tail. enable_chunked_prefill overlaps the per-bucket prefill burst
        # with ongoing decode from the previous bucket.
        # 2026-05-04: tested 1024 / 32768 — at K=16 J=8 cap=1024 it was a small
        # regression (vLLM v1 cudagraph_capture_sizes top out at 512 → batches
        # >512 fall to non-graph path). Reverted to 512 / 16384.
        max_num_seqs=512,
        max_num_batched_tokens=16384,
        enable_chunked_prefill=True,
        seed=args.seed,
    )
    # num_workers=16 parallelizes the per-shard ~33k continuation MathRewardFn
    # verifications across a ProcessPoolExecutor. Verification was the single-
    # threaded tail (~1-2 min per shard); 16 workers cuts it to ~10-20s.
    # 2026-05-04: was 8; bumped to 16 alongside batched verify — the per-call
    # parallel-threshold gate (n > 16*2=32) is now always met.
    reward_fn = MathRewardFn(num_workers=16)

    # ============================================================
    # Load prompts
    # ============================================================
    print(f"[mc] loading dataset: {args.dataset_name}/{args.dataset_config or 'default'}/{args.dataset_split}", flush=True)
    # Local file fast-path (mirrors caspo/data/math_data.py)
    if args.dataset_name.endswith(".jsonl") or args.dataset_name.endswith(".json"):
        ds = load_dataset("json", data_files=args.dataset_name, split="train")
    elif args.dataset_name.endswith(".parquet"):
        ds = load_dataset("parquet", data_files=args.dataset_name, split="train")
    elif args.dataset_config is not None:
        ds = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split)
    else:
        ds = load_dataset(args.dataset_name, split=args.dataset_split)
    rows = list(ds)
    # Apply num_prompts BEFORE sharding so it's a TOTAL cap across all shards
    # (not a per-shard cap, which was the previous unintuitive behavior).
    if args.num_prompts is not None:
        rows = rows[: args.num_prompts]
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        rows = rows[i::n]
        print(f"[mc] shard {args.shard}: kept {len(rows)} prompts", flush=True)

    prompts, raw_questions, gts = [], [], []
    for row in rows:
        q = row.get("prompt") or row.get("problem") or row.get("question")
        a = row.get("solution") or row.get("answer")
        if not q or not a:
            continue
        # Use replace() not format() so literal ``{`` / ``}`` in templates
        # (e.g. ``\boxed{}`` reminders, LaTeX) don't blow up. Mirrors the
        # contract in caspo/data/math_data.py::format_prompt.
        if "{query}" in args.prompt_template:
            prompts.append(args.prompt_template.replace("{query}", q))
        else:
            prompts.append(f"{args.prompt_template}\n{q}\n")
        raw_questions.append(q)
        gts.append(str(a))
    n_prompts = len(prompts)
    print(f"[mc] using {n_prompts} prompts (after filter)", flush=True)

    # ============================================================
    # Phase A: K base rollouts per prompt
    # ============================================================
    print(f"[mc] Phase A: K={args.K} base rollouts × {n_prompts} prompts = {args.K * n_prompts} generations", flush=True)
    sp_base = SamplingParams(n=args.K, temperature=args.temperature, top_p=args.top_p,
                              max_tokens=args.max_response_len, skip_special_tokens=True, seed=args.seed)
    t0 = time.time()
    # prompt_token_ids passthrough: pre-tokenize once + pass IDs to skip vLLM's
    # internal retokenization. Token IDs are reused below for Phase B prefix
    # construction (no decode→retokenize roundtrip → cleaner prefix-cache hits).
    phase_a_prompt_ids = [
        tok(p, add_special_tokens=True, truncation=True, max_length=args.max_prompt_len).input_ids
        for p in prompts
    ]
    base_outputs = llm.generate(
        prompts=[{"prompt_token_ids": pids} for pids in phase_a_prompt_ids],
        sampling_params=sp_base,
    )
    print(f"[mc] Phase A done in {time.time()-t0:.1f}s", flush=True)

    # Verify outcomes for all base rollouts.
    # 2026-05-04 PATCH: batch all (n_prompts*K) verifications into ONE reward_fn
    # call so the worker pool actually parallelizes. Previously each per-prompt
    # call had only K=16 entries, often below the parallel-fan-out threshold,
    # leaving us serial across thousands of single-prompt calls.
    base_responses = [[c.text for c in out.outputs] for out in base_outputs]
    print(f"[mc] Phase A: verifying {n_prompts * args.K} rollouts (batched)...", flush=True)
    t_v = time.time()
    flat_preds: List[str] = []
    flat_gts: List[str] = []
    for i in range(n_prompts):
        flat_preds.extend(base_responses[i])
        flat_gts.extend([gts[i]] * args.K)
    flat_rewards = reward_fn(predictions=flat_preds, ground_truths=flat_gts)
    base_outcomes = []  # [n_prompts][K]
    for i in range(n_prompts):
        chunk = flat_rewards[i * args.K : (i + 1) * args.K]
        base_outcomes.append([1 if r >= 0.5 else 0 for r in chunk])
    print(f"[mc] Phase A: verify done in {time.time()-t_v:.1f}s", flush=True)
    n_mixed = sum(1 for outs in base_outcomes if 0 < sum(outs) < args.K)
    print(f"[mc] Phase A: mixed-outcome prompts: {n_mixed}/{n_prompts}", flush=True)

    # ============================================================
    # Phase B: sample step boundaries → MC continuations
    # ============================================================
    print(f"[mc] Phase B: J={args.J} MC continuations × {args.steps_per_response} steps × selected rollouts", flush=True)

    # Tokenize prompts (we'll need the prompt token ids for storage)
    prompt_enc = tok(prompts, padding="max_length", max_length=args.max_prompt_len,
                      truncation=True, return_tensors="pt", add_special_tokens=True)
    prompt_ids_all = prompt_enc.input_ids.long()
    prompt_mask_all = prompt_enc.attention_mask.long()

    # Build flat list of (prompt_idx, k_idx, response_text, response_token_ids, base_outcome)
    flat_rows = []
    for i in range(n_prompts):
        if args.mixed_only and not (0 < sum(base_outcomes[i]) < args.K):
            continue
        for k, txt in enumerate(base_responses[i]):
            flat_rows.append((i, k, txt, base_outcomes[i][k]))
    print(f"[mc] Phase B: {len(flat_rows)} rollouts to label (mixed-only={args.mixed_only})", flush=True)

    # Tokenize each base response (per-row, since lengths vary)
    print(f"[mc] Phase B: tokenizing base responses...", flush=True)
    response_token_lists = []
    for _, _, txt, _ in flat_rows:
        ids = tok(txt, add_special_tokens=False, truncation=True,
                  max_length=args.max_response_len).input_ids
        response_token_lists.append(ids)

    # Build per-row step segmentation. We need a [B,R] tensor for the segmenter.
    R_max = max((len(ids) for ids in response_token_lists), default=1)
    R_max = min(R_max, args.max_response_len)
    n_rows = len(flat_rows)
    response_ids_pad = torch.zeros((n_rows, R_max), dtype=torch.long)
    response_mask_pad = torch.zeros((n_rows, R_max), dtype=torch.long)
    for r, ids in enumerate(response_token_lists):
        L = min(len(ids), R_max)
        response_ids_pad[r, :L] = torch.tensor(ids[:L], dtype=torch.long)
        response_mask_pad[r, :L] = 1

    print(f"[mc] Phase B: segmenting {n_rows} responses (LaTeX-aware)...", flush=True)
    seg = segment_responses_batch_latex_aware(
        response_ids_pad, response_mask_pad, tokenizer=tok,
        min_step_tokens=4, max_steps=64,
    )
    step_id = seg.step_id  # [n_rows, R_max] long; -1 on masked

    # For each row, find last token idx of each step
    rng = random.Random(args.seed)
    mc_jobs = []  # list of (flat_row_idx, step_end_token_idx, prefix_len_in_response)
    for r in range(n_rows):
        sids = step_id[r].cpu().tolist()
        last_idx_per_step = {}
        for j, sid in enumerate(sids):
            if sid >= 0:
                last_idx_per_step[sid] = j
        n_steps = len(last_idx_per_step)
        if n_steps == 0:
            continue
        # Random sample step boundaries (excluding the last step — we already know that's terminal)
        candidate_steps = list(last_idx_per_step.keys())
        if len(candidate_steps) > 1:
            candidate_steps = candidate_steps[:-1]  # drop final
        # Optionally cap prefix length for training (Option C: collect long responses but train on early prefixes)
        if args.max_train_prefix_len > 0:
            train_cap = args.max_train_prefix_len
            candidate_steps = [s for s in candidate_steps if last_idx_per_step[s] < train_cap]
        n_pick = min(args.steps_per_response, len(candidate_steps))
        chosen = rng.sample(candidate_steps, n_pick) if n_pick < len(candidate_steps) else list(candidate_steps)
        for sid in chosen:
            end_idx = last_idx_per_step[sid]
            mc_jobs.append((r, end_idx))

    n_jobs = len(mc_jobs)
    print(f"[mc] Phase B: {n_jobs} step-boundary prefixes to MC-label", flush=True)
    if n_jobs == 0:
        print("[mc] Phase B: nothing to label, exiting.", flush=True)
        return

    # Build prefix token-id lists (no decode→retokenize roundtrip → exact prefix-cache hits).
    print(f"[mc] Phase B: building prefix prompts for {n_jobs} jobs...", flush=True)
    prefix_token_ids_list = []
    prefix_metadata = []  # (flat_row_idx, end_idx, max_continuation_tokens)
    for r, end_idx in mc_jobs:
        prompt_idx, k_idx, _, _ = flat_rows[r]
        prefix_response_ids = response_ids_pad[r, : end_idx + 1].tolist()
        # Reuse already-tokenized prompt ids from Phase A so prefix bytes match exactly.
        full_prefix_ids = list(phase_a_prompt_ids[prompt_idx]) + prefix_response_ids
        prefix_token_ids_list.append(full_prefix_ids)
        max_continuation = args.max_response_len - (end_idx + 1)
        max_continuation = max(32, max_continuation)
        prefix_metadata.append((r, end_idx, max_continuation))

    # Bucketed adaptive max_tokens: group prefixes by ceil(max_continuation/256) so
    # each vLLM batch only allocates KV up to its bucket cap. Avoids the prior
    # behaviour where every prefix paid for max_response_len tokens of KV.
    print(f"[mc] Phase B: running J={args.J} MC continuations per prefix (bucketed)...", flush=True)
    BUCKET = 256
    buckets: dict[int, list[int]] = {}
    for j_idx, (_, _, mc) in enumerate(prefix_metadata):
        b = ((mc + BUCKET - 1) // BUCKET) * BUCKET
        buckets.setdefault(b, []).append(j_idx)
    print(f"[mc] Phase B: {len(buckets)} buckets: " +
          ", ".join(f"{b}:{len(idxs)}" for b, idxs in sorted(buckets.items())), flush=True)
    mc_outputs = [None] * n_jobs
    t0 = time.time()
    for bcap, idxs in sorted(buckets.items()):
        sp_mc = SamplingParams(n=args.J, temperature=args.temperature, top_p=args.top_p,
                                max_tokens=bcap, skip_special_tokens=True,
                                seed=args.seed + 1)
        batch_inputs = [{"prompt_token_ids": prefix_token_ids_list[j]} for j in idxs]
        outs = llm.generate(prompts=batch_inputs, sampling_params=sp_mc)
        for local_i, j in enumerate(idxs):
            mc_outputs[j] = outs[local_i]
    print(f"[mc] Phase B: MC done in {time.time()-t0:.1f}s ({n_jobs * args.J} generations)", flush=True)

    # Verify each MC continuation. The "full text" for verification is prefix_text + continuation.
    # We need to give the verifier (problem, full_response, gt). The "response" the verifier sees
    # is the prefix's response part + the MC continuation.
    # 2026-05-04 PATCH: batch all n_jobs*J continuations into ONE reward_fn
    # call. Previously the per-job call (J=8 entries) was below the parallel
    # threshold, so verification ran fully serial across n_jobs calls.
    # Production: 264k entries → ~16x speedup of the verify tail.
    # Also: pre-decode unique prefix-response texts ONCE via batch_decode to
    # avoid n_jobs separate single-sequence decode calls (each starts a fresh
    # tokenizer fast-path; batch is 5-10x cheaper).
    print(f"[mc] Phase B: verifying {n_jobs * args.J} continuations (batched)...", flush=True)
    t_v = time.time()
    # Decode prefix-response texts once per (r, end_idx) pair. Many jobs share
    # the same r (multiple step boundaries on the same rollout) — but end_idx
    # differs, so cache is keyed (r, end_idx).
    prefix_text_cache: dict[tuple[int, int], str] = {}
    decode_keys: list[tuple[int, int]] = []
    decode_id_lists: list[list[int]] = []
    for j_idx in range(n_jobs):
        r, end_idx, _ = prefix_metadata[j_idx]
        key = (r, end_idx)
        if key not in prefix_text_cache:
            prefix_text_cache[key] = ""  # placeholder, fill via batch_decode
            decode_keys.append(key)
            decode_id_lists.append(response_ids_pad[r, : end_idx + 1].tolist())
    if decode_id_lists:
        decoded = tok.batch_decode(decode_id_lists, skip_special_tokens=True)
        for k, txt in zip(decode_keys, decoded):
            prefix_text_cache[k] = txt
    # Build flat predictions / gts for one big reward_fn call.
    flat_preds: List[str] = []
    flat_gts: List[str] = []
    job_n_outputs: list[int] = []
    for j_idx, out in enumerate(mc_outputs):
        r, end_idx, _ = prefix_metadata[j_idx]
        prompt_idx, _, _, _ = flat_rows[r]
        prefix_response_text = prefix_text_cache[(r, end_idx)]
        n_out = len(out.outputs)
        for c in out.outputs:
            flat_preds.append(prefix_response_text + c.text)
        flat_gts.extend([gts[prompt_idx]] * n_out)
        job_n_outputs.append(n_out)
    flat_rewards = reward_fn(predictions=flat_preds, ground_truths=flat_gts)
    p_hats = []
    cur = 0
    for j_idx, n_out in enumerate(job_n_outputs):
        chunk = flat_rewards[cur : cur + n_out]
        cur += n_out
        n_correct = sum(1 for r_ in chunk if r_ >= 0.5)
        p_hat = n_correct / max(len(chunk), 1)
        p_hats.append(p_hat)
        if j_idx < 5:
            r, end_idx, _ = prefix_metadata[j_idx]
            prompt_idx, k_idx, _, _ = flat_rows[r]
            print(f"  [mc] sample {j_idx}: prompt={prompt_idx} k={k_idx} step_end={end_idx} "
                  f"p_hat={p_hat:.3f} (correct/J = {n_correct}/{len(chunk)})", flush=True)
    print(f"[mc] Phase B: verify done in {time.time()-t_v:.1f}s", flush=True)

    # ============================================================
    # Save
    # ============================================================
    # For each MC job, save (prompt_ids, prompt_mask, response_ids[r], response_mask[r], step_end_idx, p_hat, outcome)
    prompt_idx_per_job = [flat_rows[meta[0]][0] for meta in prefix_metadata]
    prompt_ids_out = prompt_ids_all[prompt_idx_per_job]
    prompt_mask_out = prompt_mask_all[prompt_idx_per_job]
    response_idx_per_job = [meta[0] for meta in prefix_metadata]
    response_ids_out = response_ids_pad[response_idx_per_job]
    response_mask_out = response_mask_pad[response_idx_per_job]
    step_end_idx_t = torch.tensor([meta[1] for meta in prefix_metadata], dtype=torch.long)
    p_hat_t = torch.tensor(p_hats, dtype=torch.float32)
    outcomes_t = torch.tensor([flat_rows[meta[0]][3] for meta in prefix_metadata], dtype=torch.float32)

    blob = {
        "prompt_ids": prompt_ids_out,
        "prompt_mask": prompt_mask_out,
        "response_ids": response_ids_out,
        "response_mask": response_mask_out,
        "step_end_idx": step_end_idx_t,
        "p_hat": p_hat_t,
        "outcomes": outcomes_t,
        "config": {
            "model": args.model,
            "dataset_name": args.dataset_name,
            "dataset_config": args.dataset_config,
            "K": args.K, "J": args.J,
            "steps_per_response": args.steps_per_response,
            "temperature": args.temperature,
            "max_prompt_len": args.max_prompt_len,
            "max_response_len": args.max_response_len,
            "n_prompts_seen": n_prompts,
            "n_mixed_prompts": n_mixed,
            "n_jobs": n_jobs,
        },
    }
    print(f"[mc] saving {n_jobs} labeled prefixes to {args.output}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(blob, args.output)
    print(f"[mc] DONE. p_hat distribution: mean={float(p_hat_t.mean()):.3f} "
          f"std={float(p_hat_t.std()):.3f} min={float(p_hat_t.min()):.3f} "
          f"max={float(p_hat_t.max()):.3f}", flush=True)


if __name__ == "__main__":
    main()
