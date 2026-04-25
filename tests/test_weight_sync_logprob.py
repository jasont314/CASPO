"""Numerical equivalence test: vLLM vs HF logprobs after weight sync.

The CASPO trainer relies on the invariant that, after pushing the latest HF
policy weights to vLLM (via ``save_pretrained`` + ``sync_weights_from_path``
or equivalent), the two engines compute *numerically close* token-level
logprobs for the same (prompt, continuation) pair.

If this invariant breaks (e.g. dtype mismatch on save, vLLM loading wrong
shards, attention backend disagreement beyond noise), then importance-
sampling ratios in PPO/IS-corrected RL go haywire — silently — and
training quietly diverges. This test guards that boundary.

Methodology
-----------
1. Load HF tiny model on CPU, save_pretrained to a tmp dir.
2. Boot vLLM ``AsyncLLM`` from that same dir (so weights are *byte
   identical* on disk; no in-flight quantization or sync RPC, which is
   the cleanest way to isolate "are the two stacks numerically aligned"
   from "did our sync RPC corrupt anything").
3. For a fixed prompt ``P`` and a fixed continuation ``C``:
     - HF: run a single forward over ``P+C``, gather log-softmax at the
       positions that *predict* each token of ``C``.
     - vLLM: issue a request with ``prompt = P+C`` and
       ``prompt_logprobs = len(P+C)``, ``max_tokens = 1`` (we only care
       about prompt logprobs, not generation). Extract the per-token
       logprobs aligned to ``C``.
4. Compare element-wise. Tolerance is loose (max-abs < 0.5) because:
     - HF uses eager attention here (tiny model has no flash kernel).
     - vLLM uses its own attention backend (FA varies by GPU).
     - bf16 vs fp32 accumulation differs.
   In practice the diff is typically << 0.1 on this tiny model, but 0.5
   gives headroom for backend drift across vLLM versions / GPUs.

Gating
------
Marked ``@pytest.mark.gpu`` and ``@pytest.mark.slow`` so it does **not**
run by default. Invoke explicitly:

    pytest tests/test_weight_sync_logprob.py --run-gpu --run-slow
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid as _uuid

import pytest
import torch


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


# Tiny instruct model — small enough to load fast, real enough to expose
# bugs that a randomly-initialized LlamaForCausalLM would not. Falls back
# to the random tiny model if download fails.
_TINY_MODEL_PRIMARY = "HuggingFaceTB/SmolLM2-135M-Instruct"
_TINY_MODEL_FALLBACK = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _need_vllm():
    try:
        import vllm  # noqa: F401
    except Exception as e:
        pytest.skip(f"vllm not importable: {e}")


def _try_load_hf(name: str):
    """Try to load (model, tokenizer) on CPU. Returns None on failure."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(name)
        # Force fp32 + eager so HF reference is as deterministic as possible.
        # We compare *bf16 vLLM* to *fp32 HF* — the tolerance accounts for it.
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        model.eval()
        return model, tok
    except Exception:
        return None


def _hf_continuation_logprobs(model, tok, prompt_ids, cont_ids, device="cuda"):
    """Per-token log p(cont_t | prompt, cont_<t) under HF.

    Returns a 1-D tensor of length ``len(cont_ids)`` on CPU (float32).
    """
    full = list(prompt_ids) + list(cont_ids)
    input_ids = torch.tensor([full], dtype=torch.long, device=device)
    model = model.to(device)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits[0]  # [T, V]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    # Token at position k in ``full`` is predicted by logits at position k-1.
    # Continuation starts at index len(prompt_ids) in ``full``.
    P = len(prompt_ids)
    C = len(cont_ids)
    out_lp = torch.empty(C, dtype=torch.float32)
    for i in range(C):
        tgt_pos_in_full = P + i        # token we're scoring
        pred_pos = tgt_pos_in_full - 1  # logits row that predicts it
        tok_id = full[tgt_pos_in_full]
        out_lp[i] = log_probs[pred_pos, tok_id].cpu()
    return out_lp


def _vllm_continuation_logprobs(engine, full_ids, cont_len, loop):
    """Per-token log p under vLLM for the last ``cont_len`` tokens of ``full_ids``.

    Uses ``prompt_logprobs`` so we score an existing sequence rather than
    sampling. Returns a 1-D tensor of length ``cont_len`` on CPU (float32).
    """
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    P_full = len(full_ids)
    sp = SamplingParams(
        n=1,
        temperature=0.0,           # greedy — we don't actually use the sample
        max_tokens=1,              # minimum allowed; we only care about prompt logprobs
        prompt_logprobs=1,         # request logprobs for the prompt tokens
        logprobs=1,
    )
    prompt = TokensPrompt(prompt_token_ids=list(full_ids))
    rid = f"logprob-check-{_uuid.uuid4().hex}"

    async def _drain():
        final = None
        async for out in engine.generate(prompt=prompt, sampling_params=sp, request_id=rid):
            final = out
        return final

    final = loop.run_until_complete(_drain())
    assert final is not None, "vLLM returned no RequestOutput"

    # ``final.prompt_logprobs`` is a list of length P_full where entry t is
    # either ``None`` (for t==0; nothing predicts the very first token) or a
    # dict { token_id: Logprob(...) } whose entry for the actual prompt token
    # carries the logprob assigned to it given the prefix.
    prompt_lp = final.prompt_logprobs
    assert prompt_lp is not None and len(prompt_lp) == P_full, (
        f"vLLM prompt_logprobs has unexpected length: "
        f"got {None if prompt_lp is None else len(prompt_lp)}, expected {P_full}"
    )

    out_lp = torch.empty(cont_len, dtype=torch.float32)
    start = P_full - cont_len  # first continuation index in ``full_ids``
    for i in range(cont_len):
        idx = start + i
        entry = prompt_lp[idx]
        assert entry is not None, (
            f"vLLM returned None prompt_logprob at idx {idx} (continuation token {i})"
        )
        tok_id = full_ids[idx]
        # Older vLLM: dict[int, Logprob]; the .logprob attr holds the float.
        if tok_id in entry:
            lp_obj = entry[tok_id]
        else:
            # Defensive: vLLM sometimes returns only the top-k that includes
            # the actual sampled/given token; if missing, fail loudly with
            # context.
            keys = list(entry.keys())
            raise AssertionError(
                f"vLLM prompt_logprobs at idx {idx} missing token {tok_id}; "
                f"got keys {keys[:5]}{'...' if len(keys) > 5 else ''}"
            )
        # ``Logprob`` is a dataclass-like with ``.logprob``; some versions
        # also expose float() directly.
        lp_val = getattr(lp_obj, "logprob", None)
        if lp_val is None:
            lp_val = float(lp_obj)
        out_lp[i] = float(lp_val)
    return out_lp


@pytest.mark.gpu
@pytest.mark.slow
def test_weight_sync_vllm_hf_logprobs_match():
    """vLLM and HF must produce numerically-close per-token logprobs after
    a save_pretrained → vLLM-load round-trip."""
    _need_cuda()
    _need_vllm()

    # Try the primary tiny model; fall back if offline / unreachable.
    loaded = _try_load_hf(_TINY_MODEL_PRIMARY)
    model_name = _TINY_MODEL_PRIMARY
    if loaded is None:
        loaded = _try_load_hf(_TINY_MODEL_FALLBACK)
        model_name = _TINY_MODEL_FALLBACK
    if loaded is None:
        pytest.skip("could not load any tiny HF model (likely offline)")
    hf_model, tok = loaded

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Fixed prompt + fixed continuation. Both kept short so the test is fast
    # and well within max_model_len for the tiny model.
    prompt_text = "The capital of France is"
    cont_text = " Paris and it is famous."

    prompt_ids = tok(prompt_text, add_special_tokens=True).input_ids
    cont_ids = tok(cont_text, add_special_tokens=False).input_ids
    assert len(prompt_ids) > 0 and len(cont_ids) > 0
    full_ids = list(prompt_ids) + list(cont_ids)

    # 1) HF logprobs (fp32, eager, on GPU for speed; result moved to CPU).
    hf_lp = _hf_continuation_logprobs(
        hf_model, tok, prompt_ids, cont_ids, device="cuda"
    )
    # Free HF GPU memory before booting vLLM (vLLM grabs a chunk).
    hf_model.to("cpu")
    torch.cuda.empty_cache()

    # 2) Save the same HF weights to a tmp dir, boot vLLM AsyncLLM from it.
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "ckpt")
        hf_model.save_pretrained(ckpt)
        tok.save_pretrained(ckpt)

        engine_args = AsyncEngineArgs(
            model=ckpt,
            tokenizer=ckpt,
            dtype="bfloat16",
            gpu_memory_utilization=0.45,
            enforce_eager=True,
            max_model_len=256,
            seed=0,
            disable_log_stats=True,
        )
        try:
            engine = AsyncLLM.from_engine_args(engine_args)
        except Exception as e:
            pytest.skip(f"vLLM AsyncLLM init failed (likely env / driver): {e}")

        loop = asyncio.new_event_loop()
        try:
            vllm_lp = _vllm_continuation_logprobs(
                engine, full_ids, cont_len=len(cont_ids), loop=loop
            )
        finally:
            try:
                engine.shutdown()
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass

    # 3) Compare. Tolerance is generous to absorb:
    #    - bf16 (vLLM) vs fp32 (HF) accumulation,
    #    - eager attention (HF) vs vLLM's backend (FA / FA3 / xformers),
    #    - any kernel-level numerical drift.
    diff = (hf_lp - vllm_lp).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())

    # Print for postmortem on failure / observability on pass.
    print(
        f"\n[weight_sync_logprob] model={model_name} "
        f"cont_tokens={len(cont_ids)} "
        f"max_abs_diff={max_abs:.4f} mean_abs_diff={mean_abs:.4f}\n"
        f"  hf_lp   = {hf_lp.tolist()}\n"
        f"  vllm_lp = {vllm_lp.tolist()}"
    )

    assert max_abs < 0.5, (
        f"vLLM and HF logprobs disagree by {max_abs:.4f} (max abs) after "
        f"save_pretrained → vLLM-load round-trip on {model_name}. "
        "This breaks the importance-sampling invariant in PPO/IS-corrected "
        "RL: the two stacks score the same trajectory differently. "
        "Likely causes: (a) dtype mismatch when saving the HF model, "
        "(b) vLLM picked a different attention backend that drifted beyond "
        "noise, (c) tokenizer mismatch (different special tokens between "
        "HF and vLLM-loaded copy), (d) weight-sync RPC corrupted some "
        "shard. Per-token diffs above for diagnosis."
    )
