"""Light-weight rollout tests.

These tests run on CPU with a tiny HF model. If the model cannot be
downloaded (e.g. no network), the tests are skipped — they are *not*
allowed to fail because the user may run them in a sandbox without
internet access.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

# Make the package importable.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from caspo.config import CASPOConfig
from caspo.rollout import HFRolloutSampler, RolloutBatch


_TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _load_tiny_model():
    """Try to download the tiny test model. Skip the test on any failure."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:  # pragma: no cover
        pytest.skip(f"transformers not installed: {e}")
    try:
        tok = AutoTokenizer.from_pretrained(_TINY_MODEL)
        model = AutoModelForCausalLM.from_pretrained(_TINY_MODEL)
    except Exception as e:
        pytest.skip(f"could not load tiny test model (likely no network): {e}")
    return model, tok


def _const_reward_fn(predictions, ground_truths):
    assert len(predictions) == len(ground_truths)
    return [0.5] * len(predictions)


def test_rollout_batch_shapes_cpu():
    model, tok = _load_tiny_model()
    model.eval()
    # Tiny model is usually fp32, fine on CPU.
    cfg = CASPOConfig(
        model_name_or_path=_TINY_MODEL,
        torch_dtype="float32",
        attn_implementation="eager",
        trust_remote_code=False,
        max_prompt_len=32,
        max_response_len=8,
        group_size=2,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        device="cpu",
    )

    sampler = HFRolloutSampler(model, tok, cfg, _const_reward_fn)

    examples = [
        {"prompt": "What is 1 + 1?", "ground_truth": "2"},
        {"prompt": "Solve x + 2 = 5.", "ground_truth": "3"},
    ]
    num_prompts = len(examples)
    G = cfg.group_size

    batch = sampler.sample(examples)
    assert isinstance(batch, RolloutBatch)

    # Shapes
    assert batch.response_ids.dim() == 2
    assert batch.response_ids.shape[0] == num_prompts * G
    R = batch.response_ids.shape[1]
    assert R <= cfg.max_response_len, f"R={R} > max_response_len"

    assert batch.response_mask.shape == batch.response_ids.shape
    assert batch.sampling_logprobs.shape == batch.response_ids.shape

    # Mask is 0/1 only
    unique = torch.unique(batch.response_mask)
    assert set(unique.tolist()).issubset({0, 1})

    # Mask is monotonically non-increasing per row (1...10...0): once we hit 0,
    # we never see another 1.
    diffs = batch.response_mask[:, 1:] - batch.response_mask[:, :-1]
    assert (diffs <= 0).all().item(), "response_mask should be non-increasing per row"

    # Prompt batch shapes
    assert batch.prompt_ids.dim() == 2
    assert batch.prompt_ids.shape[0] == num_prompts
    assert batch.prompt_mask.shape == batch.prompt_ids.shape

    # Rewards
    assert batch.rewards.shape == (num_prompts * G,)
    assert torch.allclose(batch.rewards, torch.full_like(batch.rewards, 0.5))

    # prompt_index
    assert batch.prompt_index.shape == (num_prompts * G,)
    expected = torch.arange(num_prompts).repeat_interleave(G).to(torch.long)
    assert torch.equal(batch.prompt_index, expected)

    # Lists
    assert len(batch.raw_prompts) == num_prompts
    assert len(batch.ground_truths) == num_prompts
    assert len(batch.raw_responses) == num_prompts * G

    # Logprobs are zero on masked positions
    masked_lp = batch.sampling_logprobs * (1 - batch.response_mask).float()
    assert torch.allclose(masked_lp, torch.zeros_like(masked_lp))

    # All tensors live on CPU initially.
    for t in (
        batch.prompt_ids,
        batch.prompt_mask,
        batch.response_ids,
        batch.response_mask,
        batch.sampling_logprobs,
        batch.rewards,
        batch.prompt_index,
    ):
        assert t.device.type == "cpu", f"{t} is not on CPU"


def test_vllm_backend_raises():
    """vLLM backend should raise NotImplementedError at sampler construction."""
    cfg = CASPOConfig(rollout_backend="vllm")

    class _DummyTok:
        pad_token = "<pad>"
        pad_token_id = 0
        eos_token = "</s>"
        eos_token_id = 1
        padding_side = "left"

    class _DummyModel:
        config = type("C", (), {"use_cache": True})()

        def parameters(self):
            return iter([])

    with pytest.raises(NotImplementedError):
        HFRolloutSampler(_DummyModel(), _DummyTok(), cfg, _const_reward_fn)


# ---------------------------------------------------------------------------
# Static helper tests — exercise _build_response_mask without the heavy model
# round-trip so we can hit edge cases that are awkward to provoke through
# generate() (EOS at position 0, padding-only rows, ...).
# ---------------------------------------------------------------------------

def test_build_response_mask_eos_at_position_zero():
    """If a row's first token is EOS, the mask should have a single 1 at t=0."""
    EOS = 2
    PAD = 0
    # Row 0: EOS first, then padding. Row 1: normal-ish (token, EOS, then pad).
    response_ids = torch.tensor(
        [
            [EOS, PAD, PAD, PAD],
            [7,   EOS, PAD, PAD],
        ],
        dtype=torch.long,
    )
    mask = HFRolloutSampler._build_response_mask(response_ids, EOS, PAD)
    expected = torch.tensor(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=torch.long,
    )
    assert torch.equal(mask, expected), f"got {mask.tolist()}"
    # Mask must be non-increasing per row even with EOS-at-0.
    diffs = mask[:, 1:] - mask[:, :-1]
    assert (diffs <= 0).all().item()


def test_build_response_mask_padding_only_row():
    """A row that is pure padding (no real tokens, no EOS) should mask to all-zeros."""
    EOS = 2
    PAD = 0
    response_ids = torch.tensor(
        [
            [PAD, PAD, PAD, PAD],   # pure padding
            [PAD, PAD, PAD, PAD],   # pure padding
            [5, 6, 7, EOS],         # full row with trailing EOS — sanity sibling
        ],
        dtype=torch.long,
    )
    mask = HFRolloutSampler._build_response_mask(response_ids, EOS, PAD)
    # Pure padding rows -> all zeros.
    assert mask[0].sum().item() == 0
    assert mask[1].sum().item() == 0
    # Sanity sibling row keeps everything up to and including EOS.
    assert torch.equal(mask[2], torch.tensor([1, 1, 1, 1], dtype=torch.long))


def test_build_response_mask_pad_equals_eos():
    """When pad_token_id == eos_token_id (common for Llama-family), the mask
    should still keep exactly the first EOS and drop everything after."""
    EOS = 2
    PAD = 2  # same as EOS
    response_ids = torch.tensor(
        [
            [5, EOS, EOS, EOS],   # first EOS at t=1; the rest is "pad" but also EOS
            [EOS, EOS, EOS, EOS], # EOS at t=0 (so degenerate row)
            [5, 6, 7, 8],         # no EOS at all
        ],
        dtype=torch.long,
    )
    mask = HFRolloutSampler._build_response_mask(response_ids, EOS, PAD)
    expected = torch.tensor(
        [
            [1, 1, 0, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    assert torch.equal(mask, expected), f"got {mask.tolist()}"


# ---------------------------------------------------------------------------
# End-to-end tests through the tiny model
# ---------------------------------------------------------------------------

def test_rollout_group_size_one_degenerate():
    """G=1 collapses the group dim — should still build a valid RolloutBatch
    with prompt_index = arange(num_prompts)."""
    model, tok = _load_tiny_model()
    model.eval()
    cfg = CASPOConfig(
        model_name_or_path=_TINY_MODEL,
        torch_dtype="float32",
        attn_implementation="eager",
        trust_remote_code=False,
        max_prompt_len=32,
        max_response_len=8,
        group_size=1,                   # degenerate: one rollout per prompt
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        device="cpu",
    )

    sampler = HFRolloutSampler(model, tok, cfg, _const_reward_fn)
    examples = [
        {"prompt": "Q1?", "ground_truth": "a"},
        {"prompt": "Q2?", "ground_truth": "b"},
        {"prompt": "Q3?", "ground_truth": "c"},
    ]
    num_prompts = len(examples)

    batch = sampler.sample(examples)
    assert isinstance(batch, RolloutBatch)

    # Batch dim equals num_prompts because G=1.
    assert batch.response_ids.shape[0] == num_prompts
    assert batch.sampling_logprobs.shape == batch.response_ids.shape
    assert batch.response_mask.shape == batch.response_ids.shape
    assert batch.rewards.shape == (num_prompts,)
    assert batch.prompt_index.shape == (num_prompts,)
    # With G=1, prompt_index should just be 0,1,2,...
    assert torch.equal(batch.prompt_index, torch.arange(num_prompts, dtype=torch.long))

    # raw_responses length == num_prompts*G == num_prompts.
    assert len(batch.raw_responses) == num_prompts
    assert len(batch.raw_prompts) == num_prompts
    assert len(batch.ground_truths) == num_prompts


def test_rollout_prompt_index_ordering_multi_prompt():
    """With G > 1 and multiple prompts, prompt_index must follow the
    HF generate(num_return_sequences=G) layout: [p0,p0,...,p1,p1,...]."""
    model, tok = _load_tiny_model()
    model.eval()
    G = 3
    cfg = CASPOConfig(
        model_name_or_path=_TINY_MODEL,
        torch_dtype="float32",
        attn_implementation="eager",
        trust_remote_code=False,
        max_prompt_len=32,
        max_response_len=4,
        group_size=G,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        rollout_top_k=-1,
        device="cpu",
    )

    # Use a reward_fn that returns the prompt index encoded into the reward
    # so we can verify the tiling into the G-replicated layout.
    def _identity_reward_fn(predictions, ground_truths):
        # ground_truths come tiled by the sampler: [gt0]*G + [gt1]*G + ...
        # If we map "k" -> float(k), the resulting reward vector should be
        # exactly equal to prompt_index when prompt_index ordering is correct.
        return [float(g) for g in ground_truths]

    sampler = HFRolloutSampler(model, tok, cfg, _identity_reward_fn)
    examples = [
        {"prompt": "alpha", "ground_truth": "0"},
        {"prompt": "beta",  "ground_truth": "1"},
        {"prompt": "gamma", "ground_truth": "2"},
        {"prompt": "delta", "ground_truth": "3"},
    ]
    num_prompts = len(examples)

    batch = sampler.sample(examples)

    # prompt_index ordering: [0]*G + [1]*G + [2]*G + [3]*G.
    expected_prompt_index = torch.arange(num_prompts).repeat_interleave(G).to(torch.long)
    assert torch.equal(batch.prompt_index, expected_prompt_index), (
        f"got {batch.prompt_index.tolist()}, expected {expected_prompt_index.tolist()}"
    )

    # rewards must equal prompt_index cast to float — confirms the sampler
    # tiles ground_truths in the same [p0_g0..p0_g(G-1), p1_g0..] order as
    # prompt_index, which is the contract the trainer relies on.
    assert torch.allclose(batch.rewards, expected_prompt_index.float()), (
        f"rewards {batch.rewards.tolist()} != prompt_index {expected_prompt_index.tolist()}"
    )

    # Shape sanity: B = num_prompts * G everywhere.
    B = num_prompts * G
    assert batch.response_ids.shape[0] == B
    assert batch.response_mask.shape[0] == B
    assert batch.sampling_logprobs.shape[0] == B
    assert batch.rewards.shape == (B,)
    assert len(batch.raw_responses) == B
    # raw_prompts and ground_truths are *not* tiled — they are the input lists.
    assert len(batch.raw_prompts) == num_prompts
    assert len(batch.ground_truths) == num_prompts
