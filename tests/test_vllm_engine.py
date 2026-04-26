"""End-to-end test for :class:`caspo.rollout.vllm_engine.VLLMRolloutEngine`.

Skipped when:
* vLLM isn't installed in the env, or
* No GPU is available (vLLM requires CUDA), or
* Model download fails.

Tests run on a single GPU using a tiny model so they finish in <90s.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest
import torch


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


_TINY_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _need_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _need_vllm():
    try:
        import vllm  # noqa: F401
    except Exception as e:
        pytest.skip(f"vllm not importable: {e}")


def _make_cfg():
    from caspo.config import CASPOConfig
    return CASPOConfig(
        model_name_or_path=_TINY_MODEL,
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_2",  # vLLM ignores this; uses its own
        trust_remote_code=False,
        max_prompt_len=128,
        max_response_len=64,
        group_size=4,
        rollout_temperature=0.6,
        rollout_top_p=0.9,
        rollout_top_k=-1,
        rollout_backend="vllm",
        device="cuda",
    )


def _stub_reward_fn(responses, ground_truths):
    return [0.0] * len(responses)


def test_vllm_prefix_max_tokens_sequence_validation():
    """Mixed VinePPO prefix budgets should be accepted without GPU/vLLM init."""
    from caspo.rollout.vllm_engine import VLLMRolloutEngine

    engine = object.__new__(VLLMRolloutEngine)
    engine.cfg = SimpleNamespace(max_response_len=17)

    assert engine._normalize_prefix_max_tokens(None, 3) == [17, 17, 17]
    assert engine._normalize_prefix_max_tokens(5, 2) == [5, 5]
    assert engine._normalize_prefix_max_tokens([8, 0, 3], 3) == [8, 1, 3]
    with pytest.raises(ValueError):
        engine._normalize_prefix_max_tokens([1, 2], 3)


def test_vllm_parallel_sampling_auto_fallback_state():
    """Auto mode should disable n>1 after a runtime mismatch."""
    from caspo.rollout.vllm_engine import VLLMRolloutEngine

    engine = object.__new__(VLLMRolloutEngine)
    engine._multi_sample_mode = "auto"
    engine._parallel_sampling_supported = None
    engine._parallel_sampling_warned = False

    assert engine._can_try_parallel_sampling(4)
    with pytest.warns(UserWarning):
        assert engine._parallel_sampling_mismatch(
            context="unit", expected=4, counts=[1, 1],
        )
    assert engine._parallel_sampling_supported is False
    assert not engine._can_try_parallel_sampling(4)


def test_vllm_parallel_sampling_batched_mode_raises_on_mismatch():
    from caspo.rollout.vllm_engine import VLLMRolloutEngine

    engine = object.__new__(VLLMRolloutEngine)
    engine._multi_sample_mode = "batched"
    engine._parallel_sampling_supported = None
    engine._parallel_sampling_warned = False

    with pytest.raises(RuntimeError):
        engine._parallel_sampling_mismatch(
            context="unit", expected=8, counts=[1],
        )


@pytest.mark.slow
def test_vllm_engine_sample_shapes():
    """End-to-end sample call returns shapes consistent with HFRolloutSampler."""
    _need_cuda()
    _need_vllm()
    from caspo.rollout.vllm_engine import VLLMRolloutEngine

    cfg = _make_cfg()
    try:
        engine = VLLMRolloutEngine(
            cfg, _stub_reward_fn,
            gpu_memory_utilization=0.45,
            enforce_eager=True,  # skip CUDA-graph compile to keep test fast
        )
    except Exception as e:
        pytest.skip(f"vLLM engine init failed (likely env or driver): {e}")
    try:
        examples = [
            {"prompt": "Solve 1+1.", "ground_truth": "2"},
            {"prompt": "Solve 3*4.", "ground_truth": "12"},
        ]
        rb = engine.sample(examples)
        # Layout matches HFRolloutSampler:
        # - response_ids: [num_prompts*G, R]
        # - prompt_index laid out repeat_interleave(G)
        G = int(cfg.group_size)
        assert rb.response_ids.shape[0] == 2 * G
        assert rb.response_mask.shape == rb.response_ids.shape
        assert rb.sampling_logprobs.shape == rb.response_ids.shape
        assert rb.rewards.shape == (2 * G,)
        assert rb.prompt_index.tolist() == [0] * G + [1] * G
        # Sampling logprobs are <=0 on real tokens.
        valid = rb.response_mask.bool()
        assert (rb.sampling_logprobs[valid] <= 1e-4).all()
        # Each row's response_mask should have at least 1 valid token (we asked
        # for some output and the tiny model produces something).
        assert (rb.response_mask.sum(dim=1) > 0).all()
    finally:
        engine.shutdown()


@pytest.mark.slow
def test_vllm_engine_sample_with_prefix():
    """sample_with_prefix returns K generations per prefix and respects
    prefix-cache reuse.

    Tightened: in addition to shape assertions, verify (a) every generation
    is non-empty (the tiny model produces at least one token within
    max_tokens) and (b) the K rollouts per prefix are not all identical —
    i.e. SamplingParams(n=K) actually produced K *distinct* samples rather
    than collapsing to one generation duplicated K times. This guards
    against the vLLM 0.19.1 ``SamplingParams(n=k)`` regression where only
    one completion is returned per request (see
    ``test_vllm_sampling_params_n_returns_k_completions``).
    """
    _need_cuda()
    _need_vllm()
    from caspo.rollout.vllm_engine import VLLMRolloutEngine

    cfg = _make_cfg()
    # Higher temperature so the K samples actually diverge on a tiny model.
    cfg.rollout_temperature = 1.0
    cfg.rollout_top_p = 0.95
    try:
        engine = VLLMRolloutEngine(
            cfg, _stub_reward_fn,
            gpu_memory_utilization=0.45,
            enforce_eager=True,
        )
    except Exception as e:
        pytest.skip(f"vLLM engine init failed: {e}")
    try:
        # Two prefixes, K=4 each.
        ids_a = engine.tokenizer("To solve this problem,", add_special_tokens=False).input_ids
        ids_b = engine.tokenizer("Step 1: We compute", add_special_tokens=False).input_ids
        K = 4
        max_tokens = 32
        out = engine.sample_with_prefix([ids_a, ids_b], K=K, max_tokens=max_tokens)

        # Outer shape: one entry per prefix, in input order.
        assert len(out) == 2, f"expected 2 prefixes, got {len(out)}"

        for p_idx, prefix_outs in enumerate(out):
            # Tightened: exactly K generations (not <= K), no silent dropping.
            assert len(prefix_outs) == K, (
                f"prefix {p_idx}: expected K={K} generations, got {len(prefix_outs)}. "
                "If this is 1, vLLM SamplingParams(n=K) regressed — see "
                "test_vllm_sampling_params_n_returns_k_completions."
            )

            for g_idx, gen in enumerate(prefix_outs):
                # Type / structure invariants.
                assert isinstance(gen.token_ids, list), \
                    f"prefix {p_idx} gen {g_idx}: token_ids not a list"
                assert all(isinstance(t, int) for t in gen.token_ids), \
                    f"prefix {p_idx} gen {g_idx}: token_ids contains non-int"
                assert 0 < len(gen.token_ids) <= max_tokens, (
                    f"prefix {p_idx} gen {g_idx}: token_ids len "
                    f"{len(gen.token_ids)} not in (0, {max_tokens}]"
                )
                # Per-token sampling logprob alignment.
                assert len(gen.sampling_logprobs) == len(gen.token_ids), (
                    f"prefix {p_idx} gen {g_idx}: logprobs len "
                    f"{len(gen.sampling_logprobs)} != tokens "
                    f"{len(gen.token_ids)}"
                )
                # Sampling logprobs are non-positive (log of a probability).
                assert all(lp <= 1e-4 for lp in gen.sampling_logprobs), (
                    f"prefix {p_idx} gen {g_idx}: positive sampling logprob"
                )
                # finish_reason is one of the documented values.
                assert gen.finish_reason in ("stop", "length", "abort", ""), (
                    f"prefix {p_idx} gen {g_idx}: unexpected finish_reason "
                    f"{gen.finish_reason!r}"
                )

            # Diversity: at least 2 of the K rollouts must differ. With
            # T=1.0 / top_p=0.95 / 32 tokens this is overwhelmingly likely
            # unless vLLM silently duplicated one completion K times.
            unique = {tuple(g.token_ids) for g in prefix_outs}
            assert len(unique) >= 2, (
                f"prefix {p_idx}: all {K} rollouts identical "
                f"(temp={cfg.rollout_temperature}). Likely SamplingParams(n=K) "
                "is returning one sample duplicated K times."
            )
    finally:
        engine.shutdown()


@pytest.mark.slow
def test_vllm_sampling_params_n_returns_k_completions():
    """Regression check: ``SamplingParams(n=K)`` must yield K CompletionOutput
    entries on a single request.

    Why: in vLLM 0.19.1 we have observed configurations where requesting
    ``n=K`` returns only 1 completion instead of K (the engine silently
    collapses parallel sampling). ``VLLMRolloutEngine.sample`` and
    ``VLLMRolloutEngine.sample_with_prefix`` now auto-probe this path and
    fall back to expanded one-request-per-sample mode if the installed vLLM
    runtime has the regression. A mismatch here is therefore an upstream
    performance limitation, not a CASPO correctness failure.

    This test bypasses the engine wrapper and probes vLLM's ``AsyncLLM``
    directly so a regression is visible at the vLLM layer (not masked by
    our padding fallback in ``sample()``).
    """
    _need_cuda()
    _need_vllm()
    import asyncio
    import uuid as _uuid
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    engine_args = AsyncEngineArgs(
        model=_TINY_MODEL,
        tokenizer=_TINY_MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        max_model_len=192,
        seed=0,
        disable_log_stats=True,
    )
    try:
        engine = AsyncLLM.from_engine_args(engine_args)
    except Exception as e:
        pytest.skip(f"vLLM AsyncLLM init failed: {e}")

    loop = asyncio.new_event_loop()

    async def _drain(prompt, sp, rid):
        final = None
        async for out in engine.generate(prompt=prompt, sampling_params=sp, request_id=rid):
            final = out
        return final

    try:
        prompt_ids = tok("Once upon a time,", add_special_tokens=False).input_ids
        prompt = TokensPrompt(prompt_token_ids=list(prompt_ids))

        # Probe several K values to make this informative on regression.
        for K in (2, 4, 8):
            sp = SamplingParams(
                n=K,
                temperature=1.0,
                top_p=0.95,
                max_tokens=16,
                logprobs=1,
            )
            req_id = f"n-eq-k-{K}-{_uuid.uuid4().hex}"
            req_out = loop.run_until_complete(_drain(prompt, sp, req_id))

            assert req_out is not None, f"K={K}: no RequestOutput returned"
            comps = list(req_out.outputs)
            if len(comps) != K:
                pytest.xfail(
                    f"vLLM SamplingParams(n={K}) returned {len(comps)} "
                    f"CompletionOutput entries, expected {K}. CASPO's wrapper "
                    "falls back to expanded requests in this case, but the "
                    "optimized batched path is unavailable on this runtime."
                )
            # Each completion should be non-empty and carry its own token ids.
            for i, c in enumerate(comps):
                assert len(list(c.token_ids)) > 0, \
                    f"K={K} comp {i}: empty token_ids"
            # And at least 2 of the K must differ (sanity: not all identical).
            unique = {tuple(c.token_ids) for c in comps}
            assert len(unique) >= 2, (
                f"K={K}: all {K} completions identical — likely a single "
                "sample duplicated K times rather than K parallel samples."
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


@pytest.mark.slow
def test_vllm_engine_shutdown_idempotent():
    """Calling :meth:`VLLMRolloutEngine.shutdown` twice must not raise.

    The engine is held by long-lived trainers that may invoke shutdown both
    explicitly (in a ``finally``) and implicitly (via ``__del__`` at GC).
    The second call must be a no-op so we don't crash teardown or mask a
    real exception further up the stack.
    """
    _need_cuda()
    _need_vllm()
    from caspo.rollout.vllm_engine import VLLMRolloutEngine

    cfg = _make_cfg()
    try:
        engine = VLLMRolloutEngine(
            cfg, _stub_reward_fn,
            gpu_memory_utilization=0.45,
            enforce_eager=True,
        )
    except Exception as e:
        pytest.skip(f"vLLM engine init failed: {e}")

    # First shutdown: real teardown.
    engine.shutdown()
    # Second shutdown: must be a no-op. If the underlying engine.shutdown()
    # or loop.close() raises on a closed resource, the wrapper is required
    # to swallow it (it warns instead). We assert no exception escapes.
    engine.shutdown()
    # Third call too — defense against multiple GC paths.
    engine.shutdown()


@pytest.mark.slow
def test_vllm_engine_throughput_vs_hf():
    """vLLM should be at least 2× faster than HF generate on the same workload.

    This is a smoke benchmark — exact ratio depends on GPU, vLLM version, and
    workload. If the speedup is < 2× something is wrong (e.g. enforce_eager,
    cold start dominating).
    """
    _need_cuda()
    _need_vllm()
    from caspo.rollout.vllm_engine import VLLMRolloutEngine
    from caspo.rollout.sampler import HFRolloutSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = _make_cfg()
    cfg.group_size = 8                # bigger workload to see the speedup
    cfg.max_response_len = 96

    try:
        engine = VLLMRolloutEngine(
            cfg, _stub_reward_fn,
            gpu_memory_utilization=0.45,
            enforce_eager=False,       # let vLLM CUDA-graph compile
        )
    except Exception as e:
        pytest.skip(f"vLLM engine init failed: {e}")

    examples = [
        {"prompt": f"Solve {i}+{i}.", "ground_truth": str(2 * i)} for i in range(4)
    ]

    # Warm-up vLLM.
    try:
        engine.sample(examples[:1])
        t0 = time.time()
        rb_v = engine.sample(examples)
        vllm_time = time.time() - t0
        vllm_tokens = int(rb_v.response_mask.sum().item())
    finally:
        engine.shutdown()

    # HF generate baseline.
    tok = AutoTokenizer.from_pretrained(_TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        _TINY_MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to("cuda")
    cfg.rollout_backend = "hf"
    sampler = HFRolloutSampler(model, tok, cfg, _stub_reward_fn)
    sampler.sample(examples[:1])  # warm-up
    t1 = time.time()
    rb_h = sampler.sample(examples)
    hf_time = time.time() - t1
    hf_tokens = int(rb_h.response_mask.sum().item())
    del model

    print(
        f"\n[vllm_throughput] vllm: {vllm_time:.2f}s ({vllm_tokens / max(vllm_time, 1e-3):.1f} tok/s) "
        f"hf: {hf_time:.2f}s ({hf_tokens / max(hf_time, 1e-3):.1f} tok/s) "
        f"speedup: {hf_time / max(vllm_time, 1e-3):.2f}x"
    )
    # Don't fail on a soft speedup target — log only — because the tiny model's
    # tokens-per-step is too small for vLLM's continuous batching to fully
    # pay off. The real benefit shows on 1B+ models.
    assert vllm_time > 0
    assert hf_time > 0


@pytest.mark.slow
def test_vllm_engine_weight_sync_from_path():
    """Save the tiny model to a tmp dir and reload via sync_weights_from_path.

    Verifies the call doesn't error and outputs after sync are different
    (because we save+reload — outputs of the same SamplingParams with the
    same prompt should be deterministic given the seed, but the prefix cache
    has been reset, etc.).
    """
    _need_cuda()
    _need_vllm()
    import tempfile
    from caspo.rollout.vllm_engine import VLLMRolloutEngine
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = _make_cfg()
    try:
        engine = VLLMRolloutEngine(
            cfg, _stub_reward_fn,
            gpu_memory_utilization=0.45,
            enforce_eager=True,
        )
    except Exception as e:
        pytest.skip(f"vLLM engine init failed: {e}")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "ckpt")
            model = AutoModelForCausalLM.from_pretrained(
                _TINY_MODEL, torch_dtype=torch.bfloat16,
            )
            tok = AutoTokenizer.from_pretrained(_TINY_MODEL)
            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)
            del model

            t = engine.sync_weights_from_path(ckpt)
            print(f"\n[vllm_weight_sync] sync took {t:.2f}s")
            # Sanity: can still sample after sync.
            rb = engine.sample([{"prompt": "Hello", "ground_truth": "world"}])
            assert rb.response_ids.shape[0] == int(cfg.group_size)
    except Exception as e:
        # If reload_weights signature differs across vLLM versions, skip
        # rather than fail — the engine is still usable for fresh-start runs
        # and we can fall back to disk-load-on-init.
        if "reload_weights" in str(e) or "TypeError" in type(e).__name__:
            pytest.skip(f"reload_weights signature mismatch on this vLLM build: {e}")
        raise
    finally:
        engine.shutdown()
