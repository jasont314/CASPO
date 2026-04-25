"""Tests for ``caspo.eval.benchmarks``."""

from __future__ import annotations

import pytest

from caspo.eval import BENCHMARKS, evaluate, evaluate_vllm
from caspo.eval.benchmarks import _resolve_ground_truth, _load_problems


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_benchmarks_registry_keys():
    expected = {"aime24", "aime25", "math500", "amc23"}
    assert expected.issubset(set(BENCHMARKS.keys()))


def test_benchmarks_registry_full_coverage():
    """All 8 documented benchmarks should be registered."""
    expected = {
        "aime24", "aime25", "math500", "math",
        "amc23", "gsm8k", "collegemath", "olympiadbench",
    }
    assert expected == set(BENCHMARKS.keys()), (
        f"registry mismatch: missing={expected - set(BENCHMARKS)}, "
        f"extra={set(BENCHMARKS) - expected}"
    )


def test_benchmark_entry_shape():
    for name, spec in BENCHMARKS.items():
        for key in ("hf_name", "split", "k", "question_field", "answer_field"):
            assert key in spec, f"{name} missing {key}"
        assert isinstance(spec["k"], int) and spec["k"] >= 1


def test_gsm8k_uses_main_config():
    """openai/gsm8k requires a config name; we pin it to 'main'."""
    spec = BENCHMARKS["gsm8k"]
    assert spec.get("config") == "main"
    assert spec["hf_name"] == "openai/gsm8k"
    assert spec["split"] == "test"


def test_olympiadbench_uses_config_and_list_flag():
    """OlympiadBench needs a config + list-typed answer flag."""
    spec = BENCHMARKS["olympiadbench"]
    assert spec.get("config") == "OE_TO_maths_en_COMP"
    assert spec.get("answer_is_list") is True


def test_benchmarks_registry_loads_or_fails_gracefully(monkeypatch):
    """Every registered benchmark should either load or return n_problems=0
    cleanly when the dataset is unavailable — never raise.
    """
    from caspo.eval import benchmarks as bench_mod

    # Force the dataset loader to a stub so we never hit the network. This
    # exercises the "graceful failure" branch for all 8 benchmarks.
    monkeypatch.setattr(bench_mod, "_load_problems", lambda spec: [])
    for name in BENCHMARKS:
        out = evaluate(model=None, tokenizer=None, benchmark=name, k=1)
        assert out["benchmark"] == name
        assert out["n_problems"] == 0
        assert out["avg@k"] == 0.0
        assert out["pass@k"] == 0.0
        assert "error" in out, f"{name} did not report error stub"


def test_load_problems_passes_config_for_gsm8k(monkeypatch):
    """``_load_problems`` should forward the ``config`` field positionally
    to ``load_dataset`` for benchmarks like gsm8k that require a config.
    """
    captured: dict = {}

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        # Return an empty iterable so the function returns [].
        return []

    # Inject a fake `datasets` module with our spy load_dataset.
    import sys, types
    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    _load_problems(BENCHMARKS["gsm8k"])
    assert captured["args"] == ("openai/gsm8k", "main")
    assert captured["kwargs"] == {"split": "test"}


def test_load_problems_no_config_when_absent(monkeypatch):
    """For benchmarks without a ``config`` key, load_dataset should be
    called with just the hf_name (no positional config arg)."""
    captured: dict = {}

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return []

    import sys, types
    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    _load_problems(BENCHMARKS["math500"])
    assert captured["args"] == ("HuggingFaceH4/MATH-500",)
    assert captured["kwargs"] == {"split": "test"}


def test_load_problems_handles_list_answer(monkeypatch):
    """OlympiadBench stores final_answer as List[str]; loader should peel
    out the first entry and skip rows with empty lists."""
    fake_rows = [
        {"question": "Q1", "final_answer": ["42", "ignored"]},
        {"question": "Q2", "final_answer": []},          # skipped (empty list)
        {"question": "Q3", "final_answer": ["7"]},
        {"question": "Q4", "final_answer": None},        # skipped (None)
    ]

    def fake_load_dataset(*args, **kwargs):
        return fake_rows

    import sys, types
    fake_mod = types.ModuleType("datasets")
    fake_mod.load_dataset = fake_load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake_mod)

    out = _load_problems(BENCHMARKS["olympiadbench"])
    answers = [row["answer"] for row in out]
    assert answers == ["42", "7"]


# ---------------------------------------------------------------------------
# Ground-truth resolution
# ---------------------------------------------------------------------------

def test_resolve_ground_truth_peels_box():
    assert _resolve_ground_truth("solution... \\boxed{42}") == "42"


def test_resolve_ground_truth_passes_bare_through():
    assert _resolve_ground_truth("42") == "42"


def test_resolve_ground_truth_peels_gsm8k_marker():
    """GSM8K stores answers as ``<reasoning>\\n#### N``."""
    raw = "Janet has 16 eggs and sells 12.\n#### 18"
    assert _resolve_ground_truth(raw) == "18"


def test_resolve_ground_truth_strips_commas_in_gsm8k_marker():
    """GSM8K answers may include thousands separators; strip them."""
    raw = "long reasoning here\n#### 1,234"
    assert _resolve_ground_truth(raw) == "1234"


def test_resolve_ground_truth_takes_last_marker():
    """If multiple #### markers exist, take the final one."""
    raw = "step1 #### 5\nstep2\n#### 18"
    assert _resolve_ground_truth(raw) == "18"


def test_resolve_ground_truth_takes_last_box():
    """If multiple \\boxed{} present, return the last one (final answer)."""
    raw = "first guess \\boxed{99}, refined to \\boxed{42}"
    assert _resolve_ground_truth(raw) == "42"


def test_resolve_ground_truth_handles_nested_braces_in_box():
    """LaTeX commonly wraps fractions: \\boxed{\\frac{1}{2}}."""
    assert _resolve_ground_truth("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_resolve_ground_truth_list_typed_already_unwrapped():
    """OlympiadBench rows are unwrapped to a string by ``_load_problems``
    (List[str] → str), so by the time _resolve_ground_truth sees them they
    are bare answer strings — pass-through."""
    # After _load_problems peels final_answer[0], we get e.g. "42"
    assert _resolve_ground_truth("42") == "42"
    # And boxed list elements still get peeled.
    assert _resolve_ground_truth("\\boxed{42}") == "42"


# ---------------------------------------------------------------------------
# evaluate (smoke / graceful failure)
# ---------------------------------------------------------------------------

def test_evaluate_unknown_benchmark_raises():
    with pytest.raises(KeyError):
        evaluate(model=None, tokenizer=None, benchmark="not-a-real-bench")


def test_evaluate_returns_empty_when_dataset_unavailable(monkeypatch):
    """If the HF dataset can't be loaded, evaluate should return a stub."""
    from caspo.eval import benchmarks as bench_mod

    monkeypatch.setattr(bench_mod, "_load_problems", lambda spec: [])
    out = evaluate(model=None, tokenizer=None, benchmark="aime24", k=2)
    assert out["benchmark"] == "aime24"
    assert out["n_problems"] == 0
    assert out["k"] == 2
    assert out["avg@k"] == 0.0
    assert out["pass@k"] == 0.0
    assert "error" in out


# ---------------------------------------------------------------------------
# evaluate_vllm (smoke / graceful failure)
# ---------------------------------------------------------------------------

def test_evaluate_vllm_unknown_benchmark_raises():
    with pytest.raises(KeyError):
        evaluate_vllm(
            model_name_or_path="ignored",
            tokenizer=None,
            benchmark="not-a-real-bench",
        )


def test_evaluate_vllm_returns_empty_when_dataset_unavailable(monkeypatch):
    """``evaluate_vllm`` must short-circuit before importing vLLM if no
    problems load — keeps the multi-benchmark sweep alive on bad rows.
    """
    from caspo.eval import benchmarks as bench_mod

    monkeypatch.setattr(bench_mod, "_load_problems", lambda spec: [])
    out = evaluate_vllm(
        model_name_or_path="ignored-because-we-fail-fast",
        tokenizer=None,
        benchmark="gsm8k",
        k=2,
    )
    assert out["benchmark"] == "gsm8k"
    assert out["n_problems"] == 0
    assert out["k"] == 2
    assert out["avg@k"] == 0.0
    assert out["pass@k"] == 0.0
    assert out["mean_response_len"] == 0.0
    assert "error" in out


def test_evaluate_vllm_default_k_from_spec(monkeypatch):
    """When k is not passed, evaluate_vllm should default to spec['k']."""
    from caspo.eval import benchmarks as bench_mod

    monkeypatch.setattr(bench_mod, "_load_problems", lambda spec: [])
    out = evaluate_vllm(
        model_name_or_path="ignored",
        tokenizer=None,
        benchmark="aime24",
    )
    assert out["k"] == BENCHMARKS["aime24"]["k"]


def test_evaluate_vllm_respects_default_limit():
    """``default_limit`` (e.g. collegemath=500) is wired into the
    ``BENCHMARKS`` registry so ``evaluate_vllm`` caps the row count when
    the caller does not pass an explicit ``limit``.

    Spinning up vLLM to exercise the slice path is not viable in unit tests,
    so this test pins the registry contract that the runtime relies on:
    the field exists, is an int, and equals the documented value (500).
    A future regression (e.g. a typo renaming the key) would silently undo
    the cap and let the eval scan the full ~2818-problem CollegeMath split.
    """
    spec = BENCHMARKS["collegemath"]
    assert "default_limit" in spec, "collegemath must declare a default_limit"
    assert isinstance(spec["default_limit"], int)
    assert spec["default_limit"] == 500


def _hf_available():
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.skipif(not _hf_available(), reason="transformers/torch not installed")
def test_evaluate_end_to_end_tiny_model(monkeypatch):
    """End-to-end smoke with a tiny HF model + a synthetic 2-row dataset.

    Skips cleanly if the tiny model can't be downloaded.
    """
    import torch  # noqa: F401
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        pytest.skip(f"transformers components missing: {e}")

    name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name)
    except Exception as e:
        pytest.skip(f"tiny model unavailable: {e}")

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Inject a fake 2-problem dataset so we don't hit HF for the questions.
    from caspo.eval import benchmarks as bench_mod

    fake_rows = [
        {"question": "what is 1+1?", "answer": "2"},
        {"question": "what is 2+2?", "answer": "4"},
    ]
    monkeypatch.setattr(bench_mod, "_load_problems", lambda spec: fake_rows)

    out = evaluate(
        model,
        tok,
        "aime24",
        k=2,
        max_new_tokens=8,
        device="cpu",
        limit=2,
        seed=0,
    )

    assert out["benchmark"] == "aime24"
    assert out["n_problems"] == 2
    assert out["k"] == 2
    # Tiny random model won't get math right; expect zero avg@k but the run
    # itself should complete and report a non-negative response length.
    assert 0.0 <= out["avg@k"] <= 1.0
    assert 0.0 <= out["pass@k"] <= 1.0
    assert out["mean_response_len"] >= 0.0
