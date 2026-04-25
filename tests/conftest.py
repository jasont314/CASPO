"""Shared pytest infrastructure for the CASPO test suite.

This module is auto-loaded by pytest before any test module. It centralizes:

1. Sys.path bootstrap so the repository's ``caspo`` package is importable when
   the tests are run from a fresh checkout that hasn't been ``pip install``-ed.
2. Custom marker registration (``slow``, ``gpu``) — keeps ``-W error`` clean
   and surfaces typo'd marker names.
3. Opt-in gating CLI flags ``--run-slow`` / ``--run-gpu``. Tests carrying the
   matching mark are auto-skipped unless the flag is passed.
4. Reusable fixtures shared across multiple test modules:
     - ``tiny_model_name``     — canonical HF tiny LlamaForCausalLM repo id.
     - ``fake_tokenizer``      — minimal stand-in tokenizer (chat template +
       whitespace ``__call__``) matching the inline ``_FakeTokenizer`` used
       in ``test_data.py``.
     - ``tiny_caspo_cfg``      — small ``CASPOConfig`` pre-wired for CPU eager
       attention against ``tiny_model_name``.
     - ``tiny_hf_model``       — best-effort load of (model, tokenizer); skips
       the requesting test if HF/network are unavailable.

The fixtures are *additive*: existing inline duplicates in test modules are
left untouched to preserve assertion scope. New tests should prefer the
fixtures.
"""

from __future__ import annotations

import os
import sys

import pytest


# ---------------------------------------------------------------------------
# Make the in-tree ``caspo`` package importable without an editable install.
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------------------------------------------------------------------
# Marker registration + opt-in gating
# ---------------------------------------------------------------------------

_TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def pytest_addoption(parser):
    """Add CLI flags that gate the ``slow`` / ``gpu`` marks."""
    group = parser.getgroup("caspo")
    group.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked with @pytest.mark.slow (e.g. vLLM end-to-end).",
    )
    group.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run tests marked with @pytest.mark.gpu (CUDA-required).",
    )


def pytest_configure(config):
    """Register custom markers so ``--strict-markers`` and ``-W error`` are
    happy, and so ``pytest --markers`` advertises them."""
    config.addinivalue_line(
        "markers",
        "slow: test takes more than a few seconds (e.g. spins up vLLM).",
    )
    config.addinivalue_line(
        "markers",
        "gpu: test requires a CUDA device. Skipped without --run-gpu.",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip ``slow`` / ``gpu`` tests unless their gating flag is set."""
    run_slow = config.getoption("--run-slow")
    run_gpu = config.getoption("--run-gpu")

    skip_slow = pytest.mark.skip(reason="needs --run-slow option to run")
    skip_gpu = pytest.mark.skip(reason="needs --run-gpu option to run")

    for item in items:
        if "slow" in item.keywords and not run_slow:
            item.add_marker(skip_slow)
        if "gpu" in item.keywords and not run_gpu:
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_model_name() -> str:
    """Canonical tiny HF repo id used by most integration-style tests.

    Centralized so a future swap (e.g. to a smaller / vendored model) is a
    one-line change.
    """
    return _TINY_MODEL


class _FakeTokenizer:
    """Minimal fake tokenizer with chat template + whitespace tokenization.

    Mirrors the inline ``_FakeTokenizer`` historically duplicated in
    ``test_data.py``. Kept here so new tests can request it as a fixture.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = []
        for m in messages:
            parts.append(f"<{m['role']}>{m['content']}</{m['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, text, add_special_tokens=False):
        toks = text.split()

        class _Out:
            input_ids = toks

        return _Out()


@pytest.fixture
def fake_tokenizer():
    """Return a fresh ``_FakeTokenizer`` per test."""
    return _FakeTokenizer()


@pytest.fixture
def tiny_caspo_cfg(tiny_model_name):
    """A small ``CASPOConfig`` pointed at the tiny CPU-friendly model.

    Uses eager attention because the tiny model doesn't ship a flash-attn
    compatible config.
    """
    from caspo.config import CASPOConfig

    return CASPOConfig(
        model_name_or_path=tiny_model_name,
        tokenizer_name_or_path=tiny_model_name,
        torch_dtype="float32",
        attn_implementation="eager",
        trust_remote_code=False,
    )


@pytest.fixture
def tiny_hf_model(tiny_model_name):
    """Best-effort tiny (model, tokenizer) pair on CPU.

    Skips the requesting test on any import / download failure so the suite
    remains green in offline sandboxes.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover
        pytest.skip(f"transformers not installed: {e}")
    try:
        tok = AutoTokenizer.from_pretrained(tiny_model_name)
        model = AutoModelForCausalLM.from_pretrained(tiny_model_name)
    except Exception as e:
        pytest.skip(f"could not load tiny test model (likely no network): {e}")
    return model, tok
