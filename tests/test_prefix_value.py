"""Integration tests for :class:`caspo.value.prefix_value.PrefixValueModel`.

Each test attempts to load a tiny HF causal LM; if the model can't be reached
(no network, no HF cache), the test is skipped rather than failed.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import replace

import pytest
import torch

from caspo.config import CASPOConfig
from caspo.value.prefix_value import PrefixValueModel, compute_log_ratio


TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _make_cfg() -> CASPOConfig:
    return CASPOConfig(
        model_name_or_path=TINY_MODEL,
        tokenizer_name_or_path=TINY_MODEL,
        torch_dtype="float32",
        attn_implementation="eager",  # tiny model doesn't support flash-attn
        trust_remote_code=False,
        value_beta=10.0,
        value_margin=5.0,
    )


def _try_build() -> PrefixValueModel:
    cfg = _make_cfg()
    try:
        return PrefixValueModel(cfg)
    except (OSError, ValueError, ImportError) as e:
        pytest.skip(f"network/HF unavailable: {e}")
    except Exception as e:  # pragma: no cover
        # Some HF errors come through as generic Exceptions; surface them as skips
        # so CI without network credentials doesn't hard-fail.
        pytest.skip(f"could not load tiny model: {e}")


def _toy_batch(model: PrefixValueModel, B: int = 2, P: int = 5, R: int = 4):
    vocab = model.phi.config.vocab_size
    g = torch.Generator().manual_seed(0)
    prompt_ids = torch.randint(0, vocab, (B, P), generator=g)
    response_ids = torch.randint(0, vocab, (B, R), generator=g)
    prompt_mask = torch.ones(B, P, dtype=torch.long)
    response_mask = torch.ones(B, R, dtype=torch.long)
    # Padding on the last response token of row 0 so we exercise masking.
    response_mask[0, -1] = 0
    return prompt_ids, prompt_mask, response_ids, response_mask


# ---------------------------------------------------------------------------
# compute_log_ratio (no model needed)
# ---------------------------------------------------------------------------

def test_compute_log_ratio_masking_and_scale() -> None:
    pi = torch.tensor([[-1.0, -2.0, -3.0]])
    ref = torch.tensor([[-1.5, -2.5, -3.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    out = compute_log_ratio(pi, ref, mask, beta=2.0)
    # 2 * (pi - ref) * mask = 2 * 0.5 * mask = [1.0, 1.0, 0.0]
    assert torch.allclose(out, torch.tensor([[1.0, 1.0, 0.0]]))


# ---------------------------------------------------------------------------
# Model-loading tests
# ---------------------------------------------------------------------------

def test_forward_shape() -> None:
    model = _try_build()
    model.eval()
    prompt_ids, prompt_mask, response_ids, response_mask = _toy_batch(model)
    B, R = response_ids.shape
    out = model(
        prompt_ids.to(model.device),
        prompt_mask.to(model.device),
        response_ids.to(model.device),
        response_mask.to(model.device),
    )
    assert out["log_ratio"].shape == (B, R)
    assert out["V"].shape == (B, R + 1)
    assert out["phi_logprobs"].shape == (B, R)
    assert out["ref_logprobs"].shape == (B, R)
    # V[:, 0] is exactly zero by construction.
    assert torch.allclose(out["V"][:, 0], torch.zeros(B))
    # The padded last token of row 0 must produce log_ratio == 0.
    assert out["log_ratio"][0, -1].item() == 0.0
    # Reference logprobs have no grad.
    assert not out["ref_logprobs"].requires_grad


def test_v_zero_at_init() -> None:
    """When phi == ref (same SFT init, untouched), log_ratio == 0 everywhere."""
    model = _try_build()
    model.eval()
    # Force phi to share weights with ref so their forward outputs match.
    with torch.no_grad():
        for p_phi, p_ref in zip(model.phi.parameters(), model.ref.parameters()):
            p_phi.copy_(p_ref)

    prompt_ids, prompt_mask, response_ids, response_mask = _toy_batch(model)
    out = model(
        prompt_ids.to(model.device),
        prompt_mask.to(model.device),
        response_ids.to(model.device),
        response_mask.to(model.device),
    )
    # Numerical noise from float32 softmax differences should be tiny.
    assert torch.allclose(
        out["log_ratio"], torch.zeros_like(out["log_ratio"]), atol=1e-4
    )
    assert torch.allclose(out["V"], torch.zeros_like(out["V"]), atol=1e-3)


def test_save_load_roundtrip() -> None:
    model = _try_build()
    model.eval()
    prompt_ids, prompt_mask, response_ids, response_mask = _toy_batch(model)

    with torch.no_grad():
        out_before = model(
            prompt_ids.to(model.device),
            prompt_mask.to(model.device),
            response_ids.to(model.device),
            response_mask.to(model.device),
        )

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        assert os.path.exists(os.path.join(tmp, "caspo_value_meta.json"))
        cfg = _make_cfg()
        reloaded = PrefixValueModel.from_pretrained(cfg, tmp)
        reloaded.eval()
        with torch.no_grad():
            out_after = reloaded(
                prompt_ids.to(reloaded.device),
                prompt_mask.to(reloaded.device),
                response_ids.to(reloaded.device),
                response_mask.to(reloaded.device),
            )

    assert torch.allclose(out_before["log_ratio"], out_after["log_ratio"], atol=1e-4)
    assert torch.allclose(
        out_before["phi_logprobs"], out_after["phi_logprobs"], atol=1e-4
    )
    assert torch.allclose(out_before["V"], out_after["V"], atol=1e-3)


# ---------------------------------------------------------------------------
# Long-prompt forward pass (P > cfg.max_prompt_len)
# ---------------------------------------------------------------------------

def test_long_prompt_forward_exceeds_max_prompt_len() -> None:
    """The model itself does no prompt-length truncation; if the caller
    hands in a sequence longer than ``cfg.max_prompt_len`` the forward must
    still run and produce shape-correct outputs (truncation is the data
    pipeline's job, not the value model's).
    """
    model = _try_build()
    model.eval()
    P = int(model.cfg.max_prompt_len) + 32
    R = 4
    prompt_ids, prompt_mask, response_ids, response_mask = _toy_batch(
        model, B=1, P=P, R=R
    )
    with torch.no_grad():
        out = model(
            prompt_ids.to(model.device),
            prompt_mask.to(model.device),
            response_ids.to(model.device),
            response_mask.to(model.device),
        )
    assert out["log_ratio"].shape == (1, R)
    assert out["V"].shape == (1, R + 1)
    assert torch.allclose(out["V"][:, 0], torch.zeros(1))
    assert torch.isfinite(out["log_ratio"]).all()
    assert torch.isfinite(out["V"]).all()


# ---------------------------------------------------------------------------
# Save/load with a *different* config (override beta/margin/dtype on reload)
# ---------------------------------------------------------------------------

def test_save_load_with_different_cfg_roundtrip() -> None:
    """Reloading with a different ``CASPOConfig`` should pick up the new
    knobs (beta, margin) on the reconstructed model while still loading the
    saved phi weights and ref pointer from disk.
    """
    model = _try_build()
    model.eval()
    prompt_ids, prompt_mask, response_ids, response_mask = _toy_batch(model)

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)

        # Build a fresh cfg with very different beta/margin to verify the
        # reload path honors the *new* cfg, not the saved metadata.
        new_cfg = replace(_make_cfg(), value_beta=2.5, value_margin=1.25)
        reloaded = PrefixValueModel.from_pretrained(new_cfg, tmp)
        reloaded.eval()

        assert reloaded.beta == pytest.approx(2.5)
        assert reloaded.margin == pytest.approx(1.25)

        with torch.no_grad():
            out_orig = model(
                prompt_ids.to(model.device),
                prompt_mask.to(model.device),
                response_ids.to(model.device),
                response_mask.to(model.device),
            )
            out_new = reloaded(
                prompt_ids.to(reloaded.device),
                prompt_mask.to(reloaded.device),
                response_ids.to(reloaded.device),
                response_mask.to(reloaded.device),
            )

        # log_ratio scales linearly with beta; same phi/ref weights so the
        # un-scaled (pi - ref) factor must match.
        scale = 2.5 / model.beta
        assert torch.allclose(
            out_new["log_ratio"], out_orig["log_ratio"] * scale, atol=1e-4
        )


# ---------------------------------------------------------------------------
# Reference model is frozen — gradients DO NOT flow through ref
# ---------------------------------------------------------------------------

def test_ref_model_frozen_no_gradients() -> None:
    """Every ``ref`` parameter has ``requires_grad=False``, and a backward
    over the loss must populate gradients only on ``phi`` parameters.
    """
    model = _try_build()
    model.train()

    # Static structural check.
    for name, p in model.ref.named_parameters():
        assert not p.requires_grad, f"ref param {name} should be frozen"

    # Behavioural check via backward.
    prompt_ids, prompt_mask, response_ids, response_mask = _toy_batch(model)
    out = model(
        prompt_ids.to(model.device),
        prompt_mask.to(model.device),
        response_ids.to(model.device),
        response_mask.to(model.device),
    )
    # Loss that actually depends on phi (and would depend on ref if it had grad).
    loss = out["log_ratio"].sum() + out["V"].sum()
    loss.backward()

    n_phi_with_grad = 0
    for name, p in model.phi.named_parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            n_phi_with_grad += 1
    assert n_phi_with_grad > 0, "expected at least one phi param to receive gradient"

    for name, p in model.ref.named_parameters():
        assert p.grad is None, f"ref param {name} unexpectedly has gradient"

    # And ref_logprobs from the forward output is detached.
    assert not out["ref_logprobs"].requires_grad


# ---------------------------------------------------------------------------
# Tokenizer save/load round-trip
# ---------------------------------------------------------------------------

def test_tokenizer_save_load() -> None:
    """``save_pretrained`` must persist the tokenizer alongside phi, and a
    reloaded model must yield byte-identical tokenization on a sample input.
    """
    model = _try_build()
    sample = "the quick brown fox jumps over 1 + 2 = 3"
    enc_before = model.tokenizer(sample, return_tensors="pt")

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        # Tokenizer files should be present (at least one of these).
        files = set(os.listdir(tmp))
        tok_files = {
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "vocab.json",
            "special_tokens_map.json",
        }
        assert files & tok_files, (
            f"no tokenizer artefacts saved; got {files!r}"
        )

        reloaded = PrefixValueModel.from_pretrained(_make_cfg(), tmp)
        enc_after = reloaded.tokenizer(sample, return_tensors="pt")

    assert torch.equal(enc_before["input_ids"], enc_after["input_ids"])
    assert torch.equal(enc_before["attention_mask"], enc_after["attention_mask"])
    assert reloaded.tokenizer.pad_token is not None


# ---------------------------------------------------------------------------
# Loading from a directory with caspo_value_meta.json missing
# ---------------------------------------------------------------------------

def test_from_pretrained_missing_meta_falls_back_to_cfg() -> None:
    """If ``caspo_value_meta.json`` is absent, ``from_pretrained`` must still
    succeed by falling back to ``cfg.model_name_or_path`` for the ref model.
    """
    model = _try_build()
    model.eval()

    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        meta = os.path.join(tmp, "caspo_value_meta.json")
        assert os.path.exists(meta)
        os.remove(meta)
        assert not os.path.exists(meta)

        cfg = _make_cfg()  # model_name_or_path still points at TINY_MODEL
        reloaded = PrefixValueModel.from_pretrained(cfg, tmp)
        reloaded.eval()

        # ref should be loaded from cfg.model_name_or_path (TINY_MODEL),
        # phi from the on-disk save dir.
        assert reloaded._ref_model_path == cfg.model_name_or_path
        assert reloaded._phi_model_path == tmp

        # Forward still works end-to-end.
        prompt_ids, prompt_mask, response_ids, response_mask = _toy_batch(reloaded)
        with torch.no_grad():
            out = reloaded(
                prompt_ids.to(reloaded.device),
                prompt_mask.to(reloaded.device),
                response_ids.to(reloaded.device),
                response_mask.to(reloaded.device),
            )
        B, R = response_ids.shape
        assert out["log_ratio"].shape == (B, R)
        assert out["V"].shape == (B, R + 1)


def test_from_pretrained_meta_contents_are_sensible() -> None:
    """Sanity check on the metadata file actually written by save_pretrained.
    Guards against silent schema drift (e.g. renaming fields).
    """
    model = _try_build()
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        with open(os.path.join(tmp, "caspo_value_meta.json")) as f:
            meta = json.load(f)
    for key in ("ref_model_path", "model_name_or_path", "beta", "margin"):
        assert key in meta, f"meta missing key {key!r}"
    assert meta["beta"] == pytest.approx(model.beta)
    assert meta["margin"] == pytest.approx(model.margin)
