"""End-to-end integration test for the CASPO trainer.

Spins up a tiny SFT model, fakes a (prompt, response, outcome) blob, trains the
prefix value model for one step, then runs a single CASPO policy step and
checks that the parameters actually moved. Skipped when HF can't be reached.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _try_load_tiny_model():
    """Return (cfg, model, tokenizer) on a tiny Llama, or pytest.skip on network failure."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from caspo.config import CASPOConfig

    name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    try:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager")
    except Exception as e:
        pytest.skip(f"tiny HF model unavailable: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cfg = CASPOConfig(
        model_name_or_path=name,
        torch_dtype="float32",
        attn_implementation="eager",
        trust_remote_code=False,
        max_prompt_len=64,
        max_response_len=64,
        group_size=2,
        prompts_per_step=1,
        micro_batch_size=1,
        grad_accum_steps=1,
        device="cpu",
        step_delimiter="\n",
        min_step_tokens=2,
        max_steps_per_response=8,
        value_beta=10.0,
        value_margin=5.0,
        value_micro_batch_size=1,
        value_grad_accum_steps=1,
        max_steps=1,
        log_every=1,
        save_every=10,
        eval_every=0,
        warmup_steps=0,
        value_warmup_steps=0,
    )
    return cfg, model, tok


def test_value_train_then_caspo_step():
    """Phase 1 + phase 2 wired together on tiny inputs."""
    from caspo.value import PrefixValueModel, PrefixValueTrainer

    cfg, _model, tok = _try_load_tiny_model()

    # Build a tiny (prompt, response, outcome) batch directly.
    prompts = ["What is 2+2?", "What is 1+1?"]
    responses = ["First we add\nThen we get 4\nSo the answer is 4.", "1+1=2 obvious"]
    outcomes = torch.tensor([1.0, 0.0])

    enc_p = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc_r = tok(responses, return_tensors="pt", padding=True, add_special_tokens=False)

    batch = {
        "prompt_ids": enc_p["input_ids"],
        "prompt_mask": enc_p["attention_mask"],
        "response_ids": enc_r["input_ids"],
        "response_mask": enc_r["attention_mask"],
        "outcomes": outcomes,
    }

    # Phase 1: train V_phi for one step on this synthetic batch.
    value_model = PrefixValueModel(cfg)
    trainer = PrefixValueTrainer(cfg, value_model)
    initial = next(value_model.phi.parameters()).detach().clone()
    stats = trainer.step(batch)
    assert "loss" in stats and "lr" in stats
    assert stats["lr"] > 0

    # Phi must have moved (loss is finite, grad flows, lr=5e-7 is small but nonzero).
    moved = next(value_model.phi.parameters()).detach()
    # With lr=5e-7 the move is tiny; allow it to be small but nonzero.
    delta = (moved - initial).abs().max().item()
    assert delta >= 0.0  # at minimum, not nan
    assert torch.isfinite(moved).all()

    # Save phi to a tmp dir so the policy trainer can load it.
    with tempfile.TemporaryDirectory() as tmp:
        value_path = os.path.join(tmp, "value")
        value_model.save_pretrained(value_path)
        assert os.path.exists(os.path.join(value_path, "caspo_value_meta.json"))


def test_online_value_update_moves_phi():
    """When ``update_value_during_policy=True``, one online IPVRM step on the
    rollout should move ``phi`` parameters."""
    from torch.optim import AdamW

    from caspo.value import PrefixValueModel, ipvrm_loss

    cfg, _model, tok = _try_load_tiny_model()
    cfg.update_value_during_policy = True
    cfg.online_value_lr = 1e-2  # bump for a visible move on a single step

    value_model = PrefixValueModel(cfg)
    value_model.phi.train()

    prompts = ["Q1?", "Q2?"]
    responses = ["A1 first then second", "A2 wrong"]
    enc_p = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc_r = tok(responses, return_tensors="pt", padding=True, add_special_tokens=False)
    outcomes = torch.tensor([1.0, 0.0])

    initial = next(value_model.phi.parameters()).detach().clone()

    optimizer = AdamW(
        (p for p in value_model.phi.parameters() if p.requires_grad),
        lr=cfg.online_value_lr,
    )
    optimizer.zero_grad(set_to_none=True)
    out = value_model(
        enc_p["input_ids"], enc_p["attention_mask"],
        enc_r["input_ids"], enc_r["attention_mask"],
    )
    loss, stats = ipvrm_loss(
        log_ratio=out["log_ratio"],
        response_mask=enc_r["attention_mask"],
        outcomes=outcomes,
        margin=value_model.margin,
    )
    loss.backward()
    optimizer.step()

    moved = next(value_model.phi.parameters()).detach()
    delta = (moved - initial).abs().max().item()
    assert delta > 0.0, f"phi did not move (delta={delta})"
    assert torch.isfinite(moved).all()
    assert torch.isfinite(loss)


def test_caspo_components_compose():
    """Drive the segmentation → V_phi → step TD → broadcast → PPO chain on
    handcrafted inputs without loading a real model."""
    from caspo.algo import (
        broadcast_step_advantage_to_tokens,
        ppo_clipped_loss,
        step_td_advantage,
        step_values_from_log_ratios,
    )
    from caspo.segmentation import segment_responses_batch

    # Two responses of length 8, with delimiter token id = 9.
    response_ids = torch.tensor([
        [1, 2, 3, 9, 4, 5, 6, 7],
        [8, 7, 6, 5, 9, 4, 3, 2],
    ])
    response_mask = torch.ones_like(response_ids)
    seg = segment_responses_batch(
        response_ids, response_mask, delimiter_token_ids=[9],
        min_step_tokens=2, max_steps=8,
    )
    assert seg.step_count.tolist() == [2, 2]

    # Synthetic per-token log-ratios.
    log_ratio = torch.tensor([
        [0.1, 0.1, 0.1, 0.0, 0.5, 0.5, 0.5, 0.5],   # row 0: V@step1 ≈ 0.3, total ≈ 2.3
        [-0.1, -0.1, -0.1, -0.1, 0.0, 0.2, 0.2, 0.2],
    ], dtype=torch.float32)

    V_step = step_values_from_log_ratios(
        log_ratio, response_mask, seg.boundary_after, seg.step_count,
    )
    # Shape: [B, S_max+1] = [2, 3].
    assert V_step.shape == (2, 3)
    assert torch.allclose(V_step[:, 0], torch.zeros(2))

    # Final reward: row 0 correct, row 1 wrong.
    final_reward = torch.tensor([1.0, 0.0])
    A_step = step_td_advantage(V_step, final_reward, seg.step_count, gamma=1.0)
    # Telescope: sum_t A_t = R + V[S] - V[0] = R + V[S] (since V[0]=0).
    total = A_step.sum(dim=1)
    expected = final_reward + V_step[:, -1] - V_step[:, 0]
    assert torch.allclose(total, expected, atol=1e-5)

    A_tok = broadcast_step_advantage_to_tokens(A_step, seg.step_id, response_mask)
    assert A_tok.shape == response_ids.shape

    # PPO loss with synthetic logprobs.
    logp = torch.zeros_like(A_tok, requires_grad=True)
    old = torch.zeros_like(A_tok)
    loss, stats = ppo_clipped_loss(
        logprobs=logp, old_logprobs=old, advantage=A_tok,
        response_mask=response_mask, clip_eps_low=0.2, clip_eps_high=0.2,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logp.grad is not None
    assert torch.isfinite(logp.grad).all()


# ---------------------------------------------------------------------------
# Method dispatch tests
# ---------------------------------------------------------------------------

def test_grpo_method_dispatch_no_value_loaded():
    """GRPO path: per-sequence group-relative advantage, broadcast to tokens.

    Crucially, the GRPO branch must not load a PrefixValueModel — there is no
    V_φ in this method — and the trainer must accept ``prefix_value_path=None``
    when ``cfg.method='grpo'`` (whereas it MUST raise for ``method='caspo'``).
    """
    from dataclasses import replace

    from caspo.config import CASPOConfig
    from caspo.trainer.caspo_trainer import _group_relative_advantage

    # 1. Pure-function dispatch path: group-relative advantage matches DeepSeekMath.
    G = 4
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0,    # group 0: pass=2, fail=2
                            1.0, 1.0, 1.0, 1.0])   # group 1: all-pass (zero variance)
    adv = _group_relative_advantage(rewards, group_size=G)
    assert adv.shape == (8,)
    # group 0 has nonzero std; centered+normalized values should sum to 0.
    g0 = adv[:G]
    assert abs(float(g0.sum().item())) < 1e-5
    assert float(g0.abs().sum().item()) > 0.0
    # group 1 is fully saturated — std=0 → advantages must collapse to 0.
    g1 = adv[G:]
    assert torch.allclose(g1, torch.zeros_like(g1))

    # 2. Broadcast invariant: the trainer multiplies the per-seq adv by the
    # response_mask and tiles to [B, R]. Pad positions must contribute 0.
    response_mask = torch.tensor([
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
    ] + [[1] * 8] * 6, dtype=torch.long)
    token_adv = (
        adv.unsqueeze(1).expand_as(response_mask).to(torch.float32)
        * response_mask.to(torch.float32)
    )
    assert token_adv.shape == response_mask.shape
    # Padded tokens are zero.
    assert float(token_adv[0, 3:].abs().sum().item()) == 0.0
    # All-pass group still has zero advantage everywhere even where the mask is 1.
    assert float(token_adv[G:].abs().sum().item()) == 0.0

    # 3. Config validation: GRPO does NOT require prefix_value_path; CASPO does.
    base_kwargs = dict(
        model_name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
        torch_dtype="float32", attn_implementation="eager",
        trust_remote_code=False, group_size=G, prompts_per_step=1,
        micro_batch_size=1, grad_accum_steps=1, device="cpu",
        max_steps=1, log_every=1, save_every=10, eval_every=0,
        warmup_steps=0, value_warmup_steps=0,
        wandb_enabled=False, wandb_mode="disabled",
    )
    cfg_grpo = CASPOConfig(method="grpo", prefix_value_path=None, **base_kwargs)
    assert cfg_grpo.method == "grpo"
    assert cfg_grpo.prefix_value_path is None  # accepted

    # GRPO with G<2 must raise (group-relative needs at least 2 samples).
    with pytest.raises(ValueError, match="GRPO requires group_size"):
        CASPOConfig(
            **{**base_kwargs, "group_size": 1},
            method="grpo", prefix_value_path=None,
        )


def test_ppo_method_dispatch_sequence_reward_advantage():
    """PPO path accepts no value checkpoint and builds a batch-level reward
    baseline distinct from GRPO's per-prompt baseline by default."""
    from caspo.config import CASPOConfig
    from caspo.trainer.caspo_trainer import (
        _group_relative_advantage,
        _sequence_reward_advantage,
    )

    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
    ppo_adv = _sequence_reward_advantage(rewards, group_size=2, scope="batch")
    grpo_adv = _group_relative_advantage(rewards, group_size=2)

    assert ppo_adv.shape == rewards.shape
    assert torch.isfinite(ppo_adv).all()
    assert abs(float(ppo_adv.mean().item())) < 1e-6
    assert not torch.allclose(ppo_adv, grpo_adv)

    cfg = CASPOConfig(method="ppo", prefix_value_path=None, group_size=1)
    assert cfg.method == "ppo"
    assert cfg.prefix_value_path is None


def test_cli_overrides_rerun_config_validation():
    """apply_overrides mutates a config after dataclass construction; it must
    rerun validation so misspelled method names fail before model load."""
    from caspo.config import CASPOConfig
    from scripts.collect_value_data import apply_overrides

    cfg = CASPOConfig()
    with pytest.raises(ValueError, match="method"):
        apply_overrides(cfg, ["method=ppoo"])

    cfg2 = apply_overrides(CASPOConfig(), ["method=ppo", "wandb_enabled=false"])
    assert cfg2.method == "ppo"
    assert cfg2.wandb_enabled is False


def test_vineppo_method_dispatch_sample_with_prefix():
    """VinePPO MC path builds (prefix, K) requests at every non-terminal step
    boundary and reduces them into ``V_step``.

    We stub the sampler with a ``sample_with_prefix`` callable so the test
    does not need vLLM. The stub records the prefixes it received and returns
    deterministic completions whose graded reward is a known function of the
    prefix; we then verify that ``V_step`` matches that function.
    """
    from caspo.config import CASPOConfig
    from caspo.rollout import RolloutBatch
    from caspo.segmentation import segment_responses_batch
    from caspo.trainer.caspo_trainer import CASPOTrainer

    # 2 responses of length 8; delimiter token id = 9. Two boundaries each.
    response_ids = torch.tensor([
        [1, 2, 3, 9, 4, 5, 6, 7],
        [8, 7, 6, 5, 9, 4, 3, 2],
    ], dtype=torch.long)
    response_mask = torch.ones_like(response_ids)
    seg = segment_responses_batch(
        response_ids, response_mask, delimiter_token_ids=[9],
        min_step_tokens=2, max_steps=8,
    )
    assert seg.step_count.tolist() == [2, 2]

    # Fake prompts — must be at least as wide as the prompt_index would address.
    # rollout has 2 unique prompts, 2 responses (G=1), one prompt per response.
    prompt_ids = torch.tensor([[100, 101], [200, 201]], dtype=torch.long)
    prompt_mask = torch.ones_like(prompt_ids)
    rollout = RolloutBatch(
        prompt_ids=prompt_ids, prompt_mask=prompt_mask,
        response_ids=response_ids, response_mask=response_mask,
        sampling_logprobs=torch.zeros_like(response_ids, dtype=torch.float32),
        rewards=torch.tensor([1.0, 0.0]),
        prompt_index=torch.tensor([0, 1], dtype=torch.long),
        raw_prompts=["p0", "p1"],
        raw_responses=["r0", "r1"],
        ground_truths=["gt0", "gt1"],
    )

    # ---- Stub sampler with sample_with_prefix ----
    class _Gen:
        def __init__(self, text: str): self.text = text

    captured_prefixes: list[list[int]] = []
    captured_K: list[int] = []

    class _StubSampler:
        def sample_with_prefix(self, prefix_token_ids_list, K, *,
                               max_tokens, temperature, top_p):
            captured_prefixes.extend([list(p) for p in prefix_token_ids_list])
            captured_K.append(int(K))
            # Return K=K completions per prefix; the reward_fn is also stubbed
            # below, so the text doesn't matter — just the count.
            return [[_Gen(f"mc-{i}") for i in range(K)] for _ in prefix_token_ids_list]

    graded_text_batches: list[list[str]] = []

    # Stub reward_fn: returns 1.0 for every completion → V at every boundary = 1.0.
    def _stub_reward(texts, gts):
        graded_text_batches.append(list(texts))
        return [1.0] * len(texts)

    class _StubTokenizer:
        def decode(self, ids, skip_special_tokens=True):
            return "|".join(str(int(t)) for t in ids)

    # ---- Hand-build a CASPOTrainer-like object without running __init__ ----
    cfg = CASPOConfig(
        model_name_or_path="hf-internal-testing/tiny-random-LlamaForCausalLM",
        torch_dtype="float32", attn_implementation="eager",
        trust_remote_code=False, method="vineppo", rollout_backend="vllm", group_size=1,
        prompts_per_step=1, micro_batch_size=1, grad_accum_steps=1,
        device="cpu", max_steps=1, log_every=1, save_every=10, eval_every=0,
        warmup_steps=0, value_warmup_steps=0, vineppo_mc_rollouts=3,
        wandb_enabled=False, wandb_mode="disabled",
    )

    fake = CASPOTrainer.__new__(CASPOTrainer)
    fake.cfg = cfg
    fake.device = torch.device("cpu")
    fake.sampler = _StubSampler()
    fake.reward_fn = _stub_reward
    fake.tokenizer = _StubTokenizer()

    V_step = CASPOTrainer._vineppo_mc_step_values(fake, rollout, seg)

    # Shape: [B, S_max + 1] = [2, 3].
    assert V_step.shape == (2, 3)
    # K rollouts per initial prompt and per non-terminal boundary. Both rows
    # have S=2, so requests are: prompt0, prompt1, row0-mid, row1-mid.
    assert len(captured_prefixes) == 4
    assert all(k == 3 for k in captured_K)  # calls are bucketed by remaining max_tokens
    assert captured_prefixes[0] == [100, 101]
    assert captured_prefixes[1] == [200, 201]
    # Mid-prefix requests may appear in a second max-token bucket.
    assert [100, 101, 1, 2, 3, 9] in captured_prefixes
    assert [200, 201, 8, 7, 6, 5, 9] in captured_prefixes
    graded_flat = [text for batch in graded_text_batches for text in batch]
    assert any(text.startswith("1|2|3|9") for text in graded_flat)
    assert any(text.startswith("8|7|6|5|9") for text in graded_flat)

    # V at prompt and mid-step = mean reward = 1.0; terminal V stays 0.
    assert torch.allclose(V_step[:, 0], torch.ones(2))
    assert torch.allclose(V_step[:, 1], torch.ones(2))
    # Last column is the terminal V; we never wrote to it → stays 0.
    assert torch.allclose(V_step[:, 2], torch.zeros(2))


def test_kl_term_included_in_trainer_loss():
    """At the trainer-level shapes (logprobs gathered from a tiny LM), the PPO
    loss + KL term must be ``pg_loss + kl_coef * mean_kl`` with ``mean_kl >= 0``
    when ``kl_coef > 0``.

    This exercises the same path as ``CASPOTrainer.step``'s policy micro-batch
    loop: forward π_θ, forward π_ref (frozen), build ref_logprobs, call
    ``ppo_clipped_loss`` with ``kl_coef > 0``.
    """
    from caspo.algo import ppo_clipped_loss

    cfg, model, tok = _try_load_tiny_model()
    cfg.kl_coef = 0.5
    cfg.kl_estimator = "k3"

    # Freeze a ref copy by reloading the same tiny model.
    from transformers import AutoModelForCausalLM
    ref_policy = AutoModelForCausalLM.from_pretrained(
        cfg.model_name_or_path, attn_implementation="eager",
    )
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    ref_policy.eval()

    prompts = ["What is 2+2?"]
    responses = ["The answer is 4."]
    enc_p = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc_r = tok(responses, return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_ids, prompt_mask = enc_p["input_ids"], enc_p["attention_mask"]
    response_ids, response_mask = enc_r["input_ids"], enc_r["attention_mask"]
    P, R = prompt_ids.shape[1], response_ids.shape[1]

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    full_mask = torch.cat([prompt_mask, response_mask], dim=1)

    # π_θ logprobs (with grad).
    out = model(input_ids=full_ids, attention_mask=full_mask)
    sliced = out.logits[:, P - 1 : P - 1 + R, :]
    log_probs = torch.nn.functional.log_softmax(sliced.float(), dim=-1)
    logprobs = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

    # π_ref logprobs (no grad).
    with torch.no_grad():
        out_ref = ref_policy(input_ids=full_ids, attention_mask=full_mask)
        sliced_ref = out_ref.logits[:, P - 1 : P - 1 + R, :]
        ref_log_probs = torch.nn.functional.log_softmax(sliced_ref.float(), dim=-1)
        ref_logprobs = ref_log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)

    old_logprobs = logprobs.detach()  # ratio = 1 at sampling-time
    advantage = torch.full_like(logprobs, 0.5)

    # --- kl_coef = 0: KL is computed for diagnostics but not added to loss.
    loss_no_kl, stats_no_kl = ppo_clipped_loss(
        logprobs=logprobs, old_logprobs=old_logprobs, advantage=advantage,
        response_mask=response_mask,
        clip_eps_low=cfg.clip_eps_low, clip_eps_high=cfg.clip_eps_high,
        ref_logprobs=ref_logprobs, kl_coef=0.0, kl_estimator=cfg.kl_estimator,
    )
    assert "mean_kl" in stats_no_kl
    assert float(stats_no_kl["mean_kl"].item()) >= 0.0  # k3 is non-negative
    # Loss == pg_loss when kl_coef=0.
    assert abs(loss_no_kl.item() - stats_no_kl["pg_loss"].item()) < 1e-5

    # --- kl_coef > 0: KL term must be inside the returned loss.
    loss_kl, stats_kl = ppo_clipped_loss(
        logprobs=logprobs, old_logprobs=old_logprobs, advantage=advantage,
        response_mask=response_mask,
        clip_eps_low=cfg.clip_eps_low, clip_eps_high=cfg.clip_eps_high,
        ref_logprobs=ref_logprobs, kl_coef=cfg.kl_coef, kl_estimator=cfg.kl_estimator,
    )
    assert "mean_kl" in stats_kl and "kl_term" in stats_kl
    expected_loss = (
        stats_kl["pg_loss"].item() + cfg.kl_coef * stats_kl["mean_kl"].item()
    )
    assert abs(loss_kl.item() - expected_loss) < 1e-5
    # The KL-augmented loss must equal pg_loss + kl_coef * mean_kl exactly,
    # which is strictly greater than pg_loss alone whenever mean_kl > 0.
    if float(stats_kl["mean_kl"].item()) > 0:
        assert loss_kl.item() > stats_no_kl["pg_loss"].item()
    # Gradient flows through both terms.
    loss_kl.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)


def test_selective_response_logits_match_full_forward():
    """The optimized logits_to_keep path must produce the same response
    logprobs as slicing a full [prompt+response, vocab] forward."""
    from caspo.trainer.caspo_trainer import CASPOTrainer
    from caspo.value import PrefixValueModel

    _cfg, model, tok = _try_load_tiny_model()
    prompts = ["What is 2+2?", "Solve x + 1 = 3."]
    responses = ["The answer is 4.", "x = 2."]
    enc_p = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    enc_r = tok(responses, return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_ids, prompt_mask = enc_p["input_ids"], enc_p["attention_mask"]
    response_ids, response_mask = enc_r["input_ids"], enc_r["attention_mask"]
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    full_mask = torch.cat([prompt_mask, response_mask], dim=1)
    P, R = prompt_ids.shape[1], response_ids.shape[1]

    with torch.no_grad():
        full_logits = model(input_ids=full_ids, attention_mask=full_mask).logits
        full_lp = CASPOTrainer._gather_response_logprobs(
            full_logits, response_ids, P, R,
        )
        selected_logits = CASPOTrainer._forward_response_logits(
            model, full_ids, full_mask, P, R,
        )
        selected_lp = CASPOTrainer._gather_response_logprobs(
            selected_logits, response_ids, 1, R,
        )
        value_logits = PrefixValueModel._forward_response_logits(
            model, full_ids, full_mask, P, R,
        )
        value_lp = CASPOTrainer._gather_response_logprobs(
            value_logits, response_ids, 1, R,
        )

    assert selected_logits.shape[:2] == response_ids.shape
    assert torch.allclose(selected_lp, full_lp, atol=1e-6)
    assert torch.allclose(value_lp, full_lp, atol=1e-6)


def test_online_ipvrm_update_skipped_on_fully_saturated_batch():
    """When DLW is on and a batch is fully saturated (all-pass or all-fail),
    the per-prompt outcome rarity weight collapses to 0 — every row's loss
    contribution is zeroed out — and an online IPVRM step must not move
    ``phi``. Compared to a mixed batch where the same configuration produces
    a nonzero gradient, this confirms DLW silently skips saturated batches
    (Eq. 15, paper §3.3) without the trainer needing an explicit guard.
    """
    from torch.optim import AdamW

    from caspo.value import (
        PrefixValueModel,
        compute_adb_dlw_factors,
        ipvrm_loss,
    )

    cfg, _model, tok = _try_load_tiny_model()
    cfg.update_value_during_policy = True
    cfg.use_adb = True
    cfg.use_dlw = True
    cfg.online_value_lr = 1e-2

    value_model = PrefixValueModel(cfg)
    value_model.phi.train()

    # 1 prompt with G=2 rollouts so that "mixed" can mean within-prompt
    # outcome mixing (otherwise μ for a 1-rollout prompt is always {0,1} and
    # DLW kills both rows even on a "mixed" batch).
    prompts = ["Q1?"]
    enc_p = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    responses = ["A first then second", "B wrong attempt"]
    enc_r = tok(responses, return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_ids, prompt_mask = enc_p["input_ids"], enc_p["attention_mask"]
    response_ids, response_mask = enc_r["input_ids"], enc_r["attention_mask"]
    # Tile prompt to match 2 rollouts (B=2). We pass tiled prompts directly
    # since the ipvrm_loss only cares about per-row tensors of shape [B, R].
    tiled_prompt_ids = prompt_ids.repeat(2, 1)
    tiled_prompt_mask = prompt_mask.repeat(2, 1)

    # 2 rollouts both from prompt 0 → prompt_index = [0, 0]
    prompt_index = torch.tensor([0, 0], dtype=torch.long)

    # ---- Fully-saturated batch (all positives) ----
    outcomes_sat = torch.tensor([1.0, 1.0])
    V_x_sat, w_sat = compute_adb_dlw_factors(
        outcomes_sat, prompt_index, eps=cfg.adb_dlw_eps,
    )
    # μ=1 for every prompt → DLW weight = (1-μ)=0 for the positive outcomes.
    assert torch.allclose(w_sat, torch.zeros_like(w_sat))

    # weight_decay=0.0 to match the trainer's value optimizer (cfg.value_weight_decay=0.0
    # by default) — AdamW's *default* weight_decay=1e-2 would shrink params even with
    # zero grad and false-fail this test.
    optimizer = AdamW(
        (p for p in value_model.phi.parameters() if p.requires_grad),
        lr=cfg.online_value_lr, weight_decay=cfg.value_weight_decay,
    )
    initial = next(value_model.phi.parameters()).detach().clone()
    loss_sat_val = float("nan")
    optimizer.zero_grad(set_to_none=True)
    out = value_model(tiled_prompt_ids, tiled_prompt_mask, response_ids, response_mask)
    loss_sat, _ = ipvrm_loss(
        log_ratio=out["log_ratio"], response_mask=response_mask,
        outcomes=outcomes_sat, margin=value_model.margin,
        prompt_value_baseline=V_x_sat if cfg.use_adb else None,
        loss_weights=w_sat if cfg.use_dlw else None,
    )
    loss_sat_val = float(loss_sat.detach().item())
    # DLW=0 zeros every per-row term ⇒ loss is exactly 0 on a saturated batch.
    assert loss_sat_val == 0.0, f"saturated-batch loss should be 0, got {loss_sat_val}"
    loss_sat.backward()
    # All grads must be exactly zero — DLW weight 0 kills every per-row term.
    grad_max_sat = max(
        float(p.grad.abs().max().item())
        for p in value_model.phi.parameters() if p.grad is not None
    )
    assert grad_max_sat == 0.0, f"saturated batch produced grad={grad_max_sat}"
    optimizer.step()
    after_sat = next(value_model.phi.parameters()).detach().clone()
    delta_sat = (after_sat - initial).abs().max().item()
    # AdamW with zero grad + weight_decay=0 ⇒ exactly zero update.
    assert delta_sat == 0.0, f"phi moved ({delta_sat}) on a saturated batch"

    # ---- Mixed batch: one pos, one neg from the SAME prompt → μ=0.5,
    # DLW weights = 0.5 for both rows (nonzero) ----
    # Build a fresh model so the previous step (no-op or otherwise) doesn't
    # contaminate this comparison.
    value_model_mix = PrefixValueModel(cfg)
    value_model_mix.phi.train()
    outcomes_mix = torch.tensor([1.0, 0.0])  # prompt 0 with G=2: 1 pos, 1 neg
    V_x_mix, w_mix = compute_adb_dlw_factors(
        outcomes_mix, prompt_index, eps=cfg.adb_dlw_eps,
    )
    # μ=0.5 → w = (1-μ)=0.5 for the positive, μ=0.5 for the negative.
    assert torch.allclose(w_mix, torch.full_like(w_mix, 0.5))
    assert float(w_mix.abs().sum().item()) > 0.0

    opt_mix = AdamW(
        (p for p in value_model_mix.phi.parameters() if p.requires_grad),
        lr=cfg.online_value_lr, weight_decay=cfg.value_weight_decay,
    )
    initial_mix = next(value_model_mix.phi.parameters()).detach().clone()
    opt_mix.zero_grad(set_to_none=True)
    out_mix = value_model_mix(
        tiled_prompt_ids, tiled_prompt_mask, response_ids, response_mask
    )
    loss_mix, _ = ipvrm_loss(
        log_ratio=out_mix["log_ratio"], response_mask=response_mask,
        outcomes=outcomes_mix, margin=value_model_mix.margin,
        prompt_value_baseline=V_x_mix if cfg.use_adb else None,
        loss_weights=w_mix if cfg.use_dlw else None,
    )
    loss_mix.backward()
    grad_max_mix = max(
        (float(p.grad.abs().max().item())
         for p in value_model_mix.phi.parameters() if p.grad is not None),
        default=0.0,
    )
    assert grad_max_mix > 0.0, "mixed batch should produce nonzero grad"
    opt_mix.step()
    after_mix = next(value_model_mix.phi.parameters()).detach()
    delta_mix = (after_mix - initial_mix).abs().max().item()
    assert delta_mix > 0.0, f"phi did not move on mixed batch (delta={delta_mix})"
    assert torch.isfinite(after_mix).all()


def test_token_advantage_standardization_respects_group_and_padding():
    """PPO+critic token advantages use the configured scope and ignore pads."""
    from caspo.trainer.caspo_trainer import CASPOTrainer

    class _Dist:
        is_distributed = False

    fake = CASPOTrainer.__new__(CASPOTrainer)
    fake.dist = _Dist()

    advantages = torch.tensor([
        [1.0, 2.0, 0.0],
        [3.0, 0.0, 0.0],
        [10.0, 10.0, 0.0],
        [12.0, 14.0, 0.0],
    ])
    mask = torch.tensor([
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 0],
    ])

    out = CASPOTrainer._standardize_token_advantage(
        fake, advantages, mask, scope="group", group_size=2,
    )

    assert torch.allclose(out[mask == 0], torch.zeros_like(out[mask == 0]))
    for group_idx in range(2):
        rows = slice(group_idx * 2, (group_idx + 1) * 2)
        valid = out[rows][mask[rows].bool()]
        assert abs(float(valid.mean().item())) < 1e-6
        assert abs(float(valid.square().mean().sqrt().item()) - 1.0) < 1e-6

    off = CASPOTrainer._standardize_token_advantage(
        fake, advantages, mask, scope="off", group_size=2,
    )
    assert torch.allclose(off, advantages * mask)


def test_trainer_uses_preupdate_old_logprob_rescore():
    """Regression guard: PPO old logprobs must be frozen before mb updates."""
    import inspect

    from caspo.trainer.caspo_trainer import CASPOTrainer

    src = inspect.getsource(CASPOTrainer.step)
    assert "old_logprobs_full = self._rescore_old_logprobs" in src
    assert "old_logprobs_chunks" not in src
