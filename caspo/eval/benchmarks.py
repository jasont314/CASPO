"""Math-benchmark eval harness.

Loads a problem set from HuggingFace, samples ``k`` responses per problem
with HF ``model.generate``, grades each response with ``MathRewardFn``, and
returns avg@k / pass@k summary statistics.

If a HuggingFace dataset name is wrong/unavailable the harness fails
gracefully (returns a result dict with ``n_problems=0`` and a non-fatal
error string) so a multi-benchmark sweep can continue.
"""

from __future__ import annotations

import random
from typing import Optional, Dict, Any, List

import torch

from caspo.reward import extract_boxed_answer, grade_math  # noqa: F401  (extract_boxed_answer re-exported)


# Registry of supported math benchmarks. Each entry holds the HF name, split,
# default sample count k, and the field names used for the question and the
# ground-truth answer (which may be a full solution containing \boxed{}).
BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "aime24": {
        "hf_name": "HuggingFaceH4/aime_2024",
        "split": "train",
        "k": 32,
        "question_field": "problem",
        "answer_field": "answer",
    },
    "aime25": {
        "hf_name": "MathArena/aime_2025_I",
        "split": "train",
        "k": 32,
        "question_field": "problem",
        "answer_field": "answer",
    },
    "math500": {
        # 500-problem subsample of MATH-test, fast to evaluate.
        "hf_name": "HuggingFaceH4/MATH-500",
        "split": "test",
        "k": 16,                     # match VinePPO's avg@16 / pass@16 protocol
        "question_field": "problem",
        "answer_field": "solution",
    },
    "math": {
        # Full MATH test (5000 problems) — VinePPO's primary eval.
        "hf_name": "DigitalLearningGmbH/MATH-lighteval",
        "split": "test",
        "k": 16,
        "question_field": "problem",
        "answer_field": "solution",
    },
    "amc23": {
        "hf_name": "AI-MO/aimo-validation-amc",
        "split": "train",
        "k": 32,
        "question_field": "problem",
        "answer_field": "answer",
    },
    "gsm8k": {
        # openai/gsm8k has answer field = "<step-by-step>\n#### <final>" — we
        # peel off the post-#### final answer in _resolve_ground_truth.
        "hf_name": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "k": 16,
        "question_field": "question",
        "answer_field": "answer",
        "answer_format": "gsm8k_hash",
    },
    "collegemath": {
        # OOD math eval; VinePPO uses 17.7% of test (~500 problems).
        # The HF dataset has 2818 test problems; we sample 500 by default.
        "hf_name": "realtreetune/college_math",
        "split": "test",
        "k": 16,
        "question_field": "problem",
        "answer_field": "answer",
        "default_limit": 500,
    },
    "olympiadbench": {
        # OOD harder eval; English text-only math competition subset.
        # 674 problems; final_answer is a List[str], we take the first entry.
        "hf_name": "Hothan/OlympiadBench",
        "config": "OE_TO_maths_en_COMP",
        "split": "train",
        "k": 16,
        "question_field": "question",
        "answer_field": "final_answer",
        "answer_is_list": True,
    },
}


def _load_problems(spec: Dict[str, Any]) -> List[Dict[str, str]]:
    """Load (question, answer) pairs from HuggingFace. Returns [] on failure."""
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return []
    try:
        kwargs = {"split": spec["split"]}
        if "config" in spec and spec["config"]:
            ds = load_dataset(spec["hf_name"], spec["config"], **kwargs)
        else:
            ds = load_dataset(spec["hf_name"], **kwargs)
    except Exception:
        return []
    qf = spec["question_field"]
    af = spec["answer_field"]
    answer_is_list = bool(spec.get("answer_is_list", False))
    answer_format = spec.get("answer_format")  # e.g. "gsm8k_hash"; passed through to _resolve_ground_truth
    out: List[Dict[str, str]] = []
    for row in ds:
        if qf not in row or af not in row:
            # Row schema didn't match — skip.
            continue
        q = row[qf]
        a = row[af]
        if q is None or a is None:
            continue
        if answer_is_list:
            # Some datasets (e.g. OlympiadBench) store final_answer as List[str].
            # Take the first entry as the canonical answer.
            if isinstance(a, list) and a:
                a = a[0]
            elif isinstance(a, list):
                continue
        rec: Dict[str, str] = {"question": str(q), "answer": str(a)}
        if answer_format:
            rec["answer_format"] = str(answer_format)
        out.append(rec)
    return out


def _resolve_ground_truth(answer_field_value: str, answer_format: Optional[str] = None) -> str:
    """Peel a ground-truth answer string into its bare final answer.

    Handles three formats:
      * GSM8K ``<reasoning>\\n#### N`` → ``N``
      * MATH/AMC ``\\boxed{N}`` → ``N``
      * bare answer string → as-is

    Resolution priority:
      1. If ``answer_format == "gsm8k_hash"``: trust the ``####`` marker
         unconditionally (GSM8K answers are numeric and never contain
         ``\\boxed{}``).
      2. Else, try ``\\boxed{N}`` first — this avoids silently corrupting
         MATH/Olympiad solutions that happen to use ``####`` as a markdown
         header.
      3. Else, fall back to the legacy ``####`` peel for backwards-compat
         (GSM8K-style strings without an explicit hint still resolve cleanly).
      4. Else, return the string as-is.
    """
    # 1. GSM8K with explicit hint.
    if answer_format == "gsm8k_hash" and "####" in answer_field_value:
        tail = answer_field_value.rsplit("####", 1)[-1].strip()
        # Strip commas in numeric answers ("1,234" → "1234"). Safe: GSM8K
        # answers are integers/decimals, and we only touch the tail.
        tail = tail.replace(",", "").strip()
        if tail:
            return tail

    # 2. \boxed{...} (MATH, AMC, AIME solution-style fields).
    inner = extract_boxed_answer(answer_field_value)
    if inner is not None:
        return inner

    # 3. Legacy: unhinted "####" peel. Only reached when no \boxed{} exists,
    # so MATH solutions with markdown #### headers are safe (they always have
    # \boxed{} too and would have returned in step 2).
    if "####" in answer_field_value:
        tail = answer_field_value.rsplit("####", 1)[-1].strip()
        tail = tail.replace(",", "").strip()
        if tail:
            return tail

    # 4. Bare string (OlympiadBench, AIME 'answer' field, etc.).
    return answer_field_value


def _build_prompt(tokenizer, question: str) -> str:
    """Format the question into a prompt. Use chat template if available."""
    apply = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply):
        try:
            messages = [{"role": "user", "content": question}]
            return apply(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return question


def evaluate_vllm(
    model_name_or_path: str,
    tokenizer,
    benchmark: str,
    *,
    k: Optional[int] = None,
    temperature: float = 0.35,
    top_p: float = 0.9,
    max_new_tokens: int = 1024,
    limit: Optional[int] = None,
    reward_fn=None,
    seed: int = 42,
    gpu_memory_utilization: float = 0.85,
    enforce_eager: bool = False,
    prompt_template: Optional[str] = None,
    trust_remote_code: bool = False,
    torch_dtype: str = "bfloat16",
    tokenizer_name_or_path: Optional[str] = None,
    max_prompt_len: int = 1024,
    max_num_seqs: Optional[int] = None,
    max_num_batched_tokens: Optional[int] = None,
    engine=None,
    loop=None,
) -> dict:
    """Fast eval using in-process vLLM (~10× faster than HF generate).

    Submits all (problem, k=N) generation requests as a single async batch;
    vLLM continuous-batches across them and auto-shares the KV cache for the
    common prompt prefix.

    If ``engine`` is passed (a pre-built ``AsyncLLM``), it is reused — caller
    owns its lifecycle (creation + shutdown). This avoids the ~30s startup
    cost when running multiple benchmarks back-to-back. The shared engine
    MUST have been built with the same ``model_name_or_path``; we assert
    this and raise ``AssertionError`` on mismatch.
    When ``engine is None`` (default) the function creates and shuts down
    its own engine internally — preserving the original single-call API.
    """
    if benchmark not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark {benchmark!r}. Known: {sorted(BENCHMARKS)}")
    spec = BENCHMARKS[benchmark]
    if k is None:
        k = int(spec["k"])
    k = int(k)
    # Note: reward_fn parameter is accepted for API compatibility but binary
    # correctness is computed directly via grade_math (see scoring loop below)
    # so format_bonus from a custom reward_fn cannot inflate avg@k / pass@k.
    del reward_fn

    rows = _load_problems(spec)
    effective_limit = limit if limit is not None else spec.get("default_limit")
    if effective_limit is not None:
        rows = rows[: int(effective_limit)]
    if not rows:
        return {
            "benchmark": benchmark, "n_problems": 0, "k": k,
            "avg@k": 0.0, "pass@k": 0.0, "mean_response_len": 0.0,
            "error": f"no problems loaded for {benchmark} ({spec['hf_name']})",
        }

    # Build prompts (apply chat template or explicit template). Hoist the
    # template-mode decision out of the per-problem loop so we don't re-check
    # ``"{query}" in prompt_template`` 500× — it's invariant.
    use_template = prompt_template is not None and "{query}" in prompt_template
    prompts: List[str] = []
    ground_truths: List[str] = []
    for row in rows:
        question = row["question"]
        gt = _resolve_ground_truth(row["answer"], row.get("answer_format"))
        if use_template:
            # ``replace`` over ``.format`` to tolerate literal ``{`` / ``}``
            # elsewhere in the template (e.g. ``\boxed{}`` instructions).
            prompt = prompt_template.replace("{query}", question)
        else:
            prompt = _build_prompt(tokenizer, question)
        prompts.append(prompt)
        ground_truths.append(gt)

    # Pad token bookkeeping — needed before constructing token-id prompts.
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            tokenizer.pad_token_id = eos
            if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token

    # Spin up vLLM AsyncLLM in-process (unless caller passed a shared one).
    import asyncio
    import time as _time
    import uuid
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    def _token_prompt(prompt_text: str):
        enc = tokenizer(prompt_text, add_special_tokens=False)
        ids = getattr(enc, "input_ids", None)
        if ids is None:
            ids = enc["input_ids"]
        if torch.is_tensor(ids):
            ids = ids.tolist()
        # HF tokenizers called on one string normally return list[int], but
        # batched/tokenizer-wrapper paths may produce [[...]].
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        max_p = int(max_prompt_len) if max_prompt_len else 0
        if max_p and len(ids) > max_p:
            ids = ids[-max_p:]
        return TokensPrompt(prompt_token_ids=[int(t) for t in ids])

    token_prompts = [_token_prompt(p) for p in prompts]

    owns_engine = engine is None
    if owns_engine:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        engine_kwargs = dict(
            model=model_name_or_path,
            tokenizer=tokenizer_name_or_path or model_name_or_path,
            tokenizer_mode="auto",
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
            enforce_eager=enforce_eager,
            max_model_len=int(max_new_tokens) + int(max_prompt_len),
            seed=seed,
            disable_log_stats=True,
        )
        if max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = int(max_num_seqs)
        if max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
        engine_args = AsyncEngineArgs(**engine_kwargs)
        engine = AsyncLLM.from_engine_args(engine_args)
    else:
        # Shared engine: verify it was built for the same model. We probe a
        # few common attribute paths because vLLM's AsyncLLM has reshuffled
        # them across versions; if none are present we skip the assert
        # rather than reject a possibly-valid engine.
        shared_model = (
            getattr(engine, "model_name", None)
            or getattr(getattr(engine, "model_config", None), "model", None)
            or getattr(getattr(engine, "vllm_config", None), "model_config", None)
        )
        if isinstance(shared_model, str):
            assert shared_model == model_name_or_path, (
                f"shared engine model {shared_model!r} != evaluate_vllm "
                f"model_name_or_path {model_name_or_path!r}"
            )
        elif shared_model is not None:
            inner = getattr(shared_model, "model", None)
            if isinstance(inner, str):
                assert inner == model_name_or_path, (
                    f"shared engine model {inner!r} != evaluate_vllm "
                    f"model_name_or_path {model_name_or_path!r}"
                )

    # NB: vLLM V1's AsyncLLM only returns 1 completion per request even when
    # SamplingParams.n>1 is set. Workaround: issue k separate requests
    # (with n=1) per problem and aggregate. This is ALSO better for vLLM's
    # continuous batcher because it gives the scheduler more independent
    # work units to interleave.
    # Some tokenizers (custom code-LM checkpoints, ad-hoc fast tokenizers)
    # report ``eos_token_id is None``. Don't crash here — just omit the
    # explicit stop token; vLLM will still cap generation at max_tokens.
    stop_ids: set[int] = set()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        stop_ids.add(int(eos_id))
    for attr in ("eos_token_ids", "additional_eos_token_ids"):
        extra = getattr(tokenizer, attr, None)
        if extra:
            try:
                for t in extra:
                    stop_ids.add(int(t))
            except TypeError:
                pass
    stop_token_ids = sorted(stop_ids)
    sp = SamplingParams(
        n=1,
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_new_tokens),
        stop_token_ids=stop_token_ids,
    )

    # Semaphore bound on simultaneous in-flight requests. For MATH-500 × k=16
    # = 8000 tasks, dumping all coroutines into asyncio.gather at once stresses
    # the event loop and vLLM scheduler queues. vLLM continuous-batches across
    # whatever's in-flight, so feeding it 256 at a time is enough to saturate
    # the GPU while keeping memory predictable. (vLLM's own scheduler already
    # bounds concurrency by KV-cache budget; this is just a producer-side
    # back-pressure to avoid creating 8000 Task objects up front.)
    _MAX_INFLIGHT = 256
    sem = asyncio.Semaphore(_MAX_INFLIGHT)

    async def _drain_one(prompt, request_id):
        # Catch per-request errors (e.g. prompt-too-long, scheduler abort)
        # and return None so a single bad prompt doesn't take down the whole
        # eval via asyncio.gather. The downstream scoring path already treats
        # ``None`` as an empty completion → grade=0, counting toward avg@k as
        # incorrect (NOT skipped — that would inflate scores).
        async with sem:
            final = None
            try:
                async for out in engine.generate(prompt=prompt, sampling_params=sp, request_id=request_id):
                    final = out
            except Exception:
                try:
                    abort = engine.abort(request_id)
                    if hasattr(abort, "__await__"):
                        await abort
                except Exception:
                    pass
                return None
            return final

    async def _drain_all(request_prompts):
        # k independent requests per problem, all submitted concurrently
        # (semaphore-bounded inside _drain_one).
        tasks = []
        for prob_idx, p in enumerate(request_prompts):
            for samp_idx in range(int(k)):
                tasks.append(_drain_one(p, f"eval-q{prob_idx}-s{samp_idx}-{uuid.uuid4().hex}"))
        # return_exceptions=True is belt-and-suspenders in case _drain_one
        # itself raises before catching (e.g. async setup failures).
        flat = await asyncio.gather(*tasks, return_exceptions=True)
        flat = [None if isinstance(x, BaseException) else x for x in flat]
        # Reshape to (n_problems, k).
        nested = [flat[i*k:(i+1)*k] for i in range(len(prompts))]
        return nested

    t0 = _time.time()
    owns_loop = loop is None
    if owns_loop:
        loop = asyncio.new_event_loop()
    try:
        try:
            outputs_per_problem = loop.run_until_complete(_drain_all(token_prompts))
        finally:
            # AsyncLLM owns background tasks bound to the event loop used for
            # generation. Shut the engine down before closing that loop;
            # otherwise vLLM can log noisy "Event loop is closed" tracebacks
            # after a successful eval.
            if owns_engine and engine is not None:
                try:
                    engine.shutdown()
                except Exception:
                    pass
                engine = None
            try:
                pending = [
                    t for t in asyncio.all_tasks(loop=loop) if not t.done()
                ]
                if pending:
                    for t in pending:
                        t.cancel()
                    try:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                    except Exception:
                        pass
            except Exception:
                pass
            if owns_loop:
                loop.close()
    except BaseException:
        # If generation fails mid-flight, still try to release the AsyncLLM
        # engine so its background workers / GPU memory don't leak across
        # successive eval calls in a sweep. Only when we own it — for a
        # shared engine the caller is responsible for shutdown.
        if owns_engine:
            try:
                if engine is not None:
                    engine.shutdown()
            except Exception:
                pass
        raise
    gen_time = _time.time() - t0

    # Score: each output has k completions; binary correctness via reward_fn.
    correctness_per_problem: List[float] = []
    pass_per_problem: List[float] = []
    response_lens: List[int] = []

    for req_outs, gt in zip(outputs_per_problem, ground_truths):
        # req_outs is a list of k RequestOutput objects, each with 1 completion.
        completions = []
        for req_out in req_outs:
            outs = list(req_out.outputs) if req_out is not None else []
            completions.append(outs[0] if outs else None)
        responses = [c.text if c is not None else "" for c in completions]
        # Grade with grade_math directly (returns 1.0/0.0): correctness must
        # NOT be inferred from reward_fn's score because format_bonus can push
        # a wrong-but-formatted response above any threshold.
        binary = [1.0 if grade_math(r, gt) >= 1.0 else 0.0 for r in responses]
        correctness_per_problem.append(sum(binary) / max(len(binary), 1))
        pass_per_problem.append(1.0 if any(b >= 1.0 for b in binary) else 0.0)
        for r in responses:
            response_lens.append(len(r))

    if owns_engine and engine is not None:
        try:
            engine.shutdown()
        except Exception:
            pass

    n = len(rows)
    return {
        "benchmark": benchmark,
        "n_problems": n,
        "k": k,
        "avg@k": float(sum(correctness_per_problem) / max(n, 1)),
        "pass@k": float(sum(pass_per_problem) / max(n, 1)),
        "mean_response_len": float(sum(response_lens) / max(len(response_lens), 1)),
        "gen_time_s": float(gen_time),
        "tokens_per_sec_estimate": float(sum(response_lens) / 4 / max(gen_time, 1e-3)),  # ~4 chars/token rough
    }


def evaluate(
    model,
    tokenizer,
    benchmark: str,
    *,
    k: Optional[int] = None,
    temperature: float = 0.35,           # VinePPO eval default
    top_p: float = 0.9,                  # VinePPO eval default
    max_new_tokens: int = 1024,          # VinePPO eval default
    device: str = "cuda",
    limit: Optional[int] = None,
    reward_fn=None,
    seed: int = 42,                      # VinePPO eval default
    prompt_template: Optional[str] = None,  # explicit "{query}" template (parity w/ vLLM path)
) -> dict:
    """Run a math-benchmark eval.

    Returns::

        {
          'benchmark': str,
          'n_problems': int,
          'k': int,
          'avg@k': float,    # mean of mean correctness across k samples per problem
          'pass@k': float,   # fraction of problems with >=1 correct sample
          'mean_response_len': float,
        }

    On a missing/unreachable dataset the function returns the same shape with
    ``n_problems=0`` and a non-fatal ``'error'`` key.
    """
    if benchmark not in BENCHMARKS:
        raise KeyError(
            f"Unknown benchmark {benchmark!r}. Known: {sorted(BENCHMARKS)}"
        )
    spec = BENCHMARKS[benchmark]
    if k is None:
        k = int(spec["k"])
    k = int(k)
    # Note: reward_fn parameter is accepted for API compatibility but binary
    # correctness is computed directly via grade_math (see scoring loop below)
    # so format_bonus from a custom reward_fn cannot inflate avg@k / pass@k.
    del reward_fn

    rows = _load_problems(spec)
    # Apply benchmark-default limit (e.g. CollegeMath subsamples to 500 to
    # match VinePPO's 17.7% portion). Caller's --limit overrides.
    effective_limit = limit if limit is not None else spec.get("default_limit")
    if effective_limit is not None:
        rows = rows[: int(effective_limit)]

    if len(rows) == 0:
        return {
            "benchmark": benchmark,
            "n_problems": 0,
            "k": k,
            "avg@k": 0.0,
            "pass@k": 0.0,
            "mean_response_len": 0.0,
            "error": f"no problems loaded for {benchmark} ({spec['hf_name']})",
        }

    # Seed for reproducible sampling.
    torch.manual_seed(int(seed))
    random.seed(int(seed))

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.eval()

    # Make sure the tokenizer has a pad token; HF generate needs one for
    # padded batches. Fall back to eos.
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is not None:
            tokenizer.pad_token_id = eos
            if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
                tokenizer.pad_token = tokenizer.eos_token

    correctness_per_problem: List[float] = []
    pass_per_problem: List[float] = []
    response_lens: List[int] = []

    # Hoist template-mode decision (invariant across rows) and pre-build all
    # prompt strings + ground truths in one pass so the per-problem loop only
    # does tokenize + generate.
    use_template = prompt_template is not None and "{query}" in prompt_template
    prebuilt: List[Dict[str, str]] = []
    for row in rows:
        question = row["question"]
        gt = _resolve_ground_truth(row["answer"], row.get("answer_format"))
        if use_template:
            # See evaluate_vllm: ``replace`` keeps literal braces intact.
            prompt = prompt_template.replace("{query}", question)
        else:
            prompt = _build_prompt(tokenizer, question)
        prebuilt.append({"prompt": prompt, "gt": gt})

    for entry in prebuilt:
        prompt = entry["prompt"]
        gt = entry["gt"]
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device, non_blocking=True)
        attention_mask = enc.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device, non_blocking=True)

        gen_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            num_return_sequences=k,
            pad_token_id=getattr(tokenizer, "pad_token_id", None),
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            out_ids = model.generate(input_ids=input_ids, **gen_kwargs)

        # out_ids: (k, prompt_len + new_tokens). Strip the prompt for decoding.
        prompt_len = input_ids.shape[1]
        new_tokens = out_ids[:, prompt_len:]
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        # Grade with grade_math directly (returns 1.0/0.0): correctness must
        # NOT be inferred from reward_fn's score because format_bonus can push
        # a wrong-but-formatted response above any threshold.
        binary = [1.0 if grade_math(r, gt) >= 1.0 else 0.0 for r in responses]
        correctness_per_problem.append(sum(binary) / max(len(binary), 1))
        pass_per_problem.append(1.0 if any(b >= 1.0 for b in binary) else 0.0)

        for r in responses:
            response_lens.append(len(r))

    n = len(rows)
    return {
        "benchmark": benchmark,
        "n_problems": n,
        "k": k,
        "avg@k": float(sum(correctness_per_problem) / max(n, 1)),
        "pass@k": float(sum(pass_per_problem) / max(n, 1)),
        "mean_response_len": float(sum(response_lens) / max(len(response_lens), 1)),
    }
