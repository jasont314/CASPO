"""Group-of-G rollout sampler.

For each prompt we sample ``cfg.group_size`` responses with HuggingFace
``model.generate``. Sampling-time log-probabilities are extracted directly
from ``output_scores`` (the per-step logits returned by ``generate``) so the
trainer's importance ratio uses the *true* on-policy logprobs at the time
each token was sampled — not a re-forward, which would already include the
post-update parameters once we start training off-policy mini-epochs.

Outputs are returned on CPU; the trainer is responsible for moving them to
the model's device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List

import torch
import torch.nn.functional as F

from caspo.config import CASPOConfig

try:  # transformers is a runtime dep — but be defensive for the dummy-tokenizer test.
    from transformers import GenerationConfig as _HFGenerationConfig
except Exception:  # pragma: no cover
    _HFGenerationConfig = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class RolloutBatch:
    """A batch of group-of-G rollouts.

    All tensors are on CPU; the trainer moves them to device before computing
    the loss.
    """
    prompt_ids: torch.LongTensor       # [num_prompts, P_max] padded prompts
    prompt_mask: torch.LongTensor      # [num_prompts, P_max]
    response_ids: torch.LongTensor     # [num_prompts*G, R_max]
    response_mask: torch.LongTensor    # [num_prompts*G, R_max]
    sampling_logprobs: torch.Tensor    # [num_prompts*G, R_max] log π at sampling time, no grad
    rewards: torch.Tensor              # [num_prompts*G]
    prompt_index: torch.LongTensor     # [num_prompts*G]  index into prompt batch
    raw_prompts: List[str]             # length num_prompts
    raw_responses: List[str]           # length num_prompts*G
    ground_truths: List[str]           # length num_prompts


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class HFRolloutSampler:
    """Rollout via HuggingFace ``model.generate``."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        cfg: CASPOConfig,
        reward_fn: Callable[[List[str], List[str]], List[float]],
    ):
        if cfg.rollout_backend == "vllm":
            raise NotImplementedError("vLLM backend not yet implemented")
        if cfg.rollout_backend != "hf":
            raise ValueError(f"unknown rollout_backend {cfg.rollout_backend!r}")

        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.reward_fn = reward_fn

        # Make sure the tokenizer has a pad token; left-pad for batched generation.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        # Left-truncate so the *question* (typically at the end of the prompt)
        # survives when prompts exceed cfg.max_prompt_len. HF's default
        # truncation_side='right' would cut off the question and leave only the
        # system/few-shot preamble, producing nonsense rollouts. Matches the
        # vLLM engine's manual `ids[-max_prompt_len:]` truncation.
        self.tokenizer.truncation_side = "left"

        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id for rollout EOS masking")
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.pad_token_id = int(self.tokenizer.pad_token_id)

        # Top-k <= 0 means "no truncation". HF expects None / 0 / a positive int.
        self._top_k = int(cfg.rollout_top_k) if cfg.rollout_top_k and cfg.rollout_top_k > 0 else 0
        self._temperature = float(cfg.rollout_temperature)
        self._do_sample = self._temperature > 1e-6
        self._top_p = float(cfg.rollout_top_p)

        # Cache a GenerationConfig so HF doesn't re-validate sampling kwargs on every
        # call. max_new_tokens is set per-call (depends on prompt length) so we
        # leave it off here and pass it on the generate() call.
        #
        # Greedy decoding (do_sample=False) does NOT support
        # num_return_sequences > 1 — HF's GenerationConfig.validate() raises.
        # When the user asks for greedy (temperature == 0) with group_size > 1
        # we generate one sequence per prompt and tile the prompts G times
        # before calling generate() (see sample()). This produces G identical
        # rollouts per prompt, which is the only sensible interpretation of
        # "greedy with G samples".
        self._gen_num_return_sequences = 1 if not self._do_sample else int(cfg.group_size)
        if _HFGenerationConfig is not None:
            base_kwargs = dict(
                do_sample=self._do_sample,
                num_return_sequences=self._gen_num_return_sequences,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )
            if self._do_sample:
                base_kwargs.update(
                    temperature=self._temperature,
                    top_p=self._top_p,
                    top_k=self._top_k,
                )
            self._generation_config = _HFGenerationConfig(**base_kwargs)
        else:  # pragma: no cover
            self._generation_config = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _tokenize_prompts(self, prompts: List[str]) -> dict:
        """Left-pad-tokenize the prompts, truncating to ``cfg.max_prompt_len``."""
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_prompt_len,
            add_special_tokens=False,
        )
        return enc

    @staticmethod
    def _build_response_mask(
        response_ids: torch.Tensor,
        eos_token_id: int,
        pad_token_id: int,
    ) -> torch.LongTensor:
        """1 on real response tokens up to and including the first EOS; 0 after.

        Pad positions are also 0.
        """
        # Find the first EOS per row.
        is_eos = (response_ids == eos_token_id)
        # cumulative EOS count: positions strictly *after* the first EOS have count >= 2
        # at the position itself we still want a 1, so use shifted cumulative.
        # Equivalent: keep token t iff number of EOS at positions [0..t-1] == 0.
        cum = is_eos.to(torch.long).cumsum(dim=1)
        # Tokens AFTER the first EOS: cum > 1 at the EOS itself? Actually cum at the
        # EOS position is 1 (we want to keep that token), and cum at every position
        # after is >= 1 because we already counted the EOS — but we want those off.
        # Trick: keep where (cum == 0) OR (token is the first EOS).
        first_eos = is_eos & (cum == 1)
        keep = (cum == 0) | first_eos
        # Also drop pure padding (in case generate stopped early and padded with pad,
        # not eos): if pad differs from eos, mask explicit pads.
        if pad_token_id != eos_token_id:
            keep = keep & (response_ids != pad_token_id)
        return keep.to(torch.long)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(self, examples: List[dict]) -> RolloutBatch:
        """Sample G responses per prompt and build a :class:`RolloutBatch`."""
        if not examples:
            raise ValueError("examples must be a non-empty list")

        cfg = self.cfg
        G = int(cfg.group_size)
        prompts = [str(ex["prompt"]) for ex in examples]
        ground_truths = [str(ex["ground_truth"]) for ex in examples]
        num_prompts = len(prompts)

        device = self._model_device()
        was_training = self.model.training
        self.model.eval()
        prev_use_cache = getattr(self.model.config, "use_cache", None)
        # Need the KV cache for fast generation.
        try:
            self.model.config.use_cache = True
        except AttributeError:
            pass

        try:
            enc = self._tokenize_prompts(prompts)
            prompt_ids = enc["input_ids"].to(device, non_blocking=True)
            prompt_mask = enc["attention_mask"].to(device, non_blocking=True)
            P_max = prompt_ids.shape[1]

            # Greedy decoding (temperature == 0) does not support
            # num_return_sequences > 1 in HF. To still emit G rollouts per
            # prompt under greedy, we repeat each prompt G times along the
            # batch dim and call generate with num_return_sequences=1. The
            # resulting layout is identical to HF's [p0_g0,..p0_g(G-1),p1_g0,..]
            # contract that downstream code (prompt_index, ground_truth tiling)
            # depends on.
            if not self._do_sample and G > 1:
                gen_input_ids = prompt_ids.repeat_interleave(G, dim=0)
                gen_attn_mask = prompt_mask.repeat_interleave(G, dim=0)
            else:
                gen_input_ids = prompt_ids
                gen_attn_mask = prompt_mask

            # Cap max_new_tokens by the model's positional limit so we don't
            # exceed `max_position_embeddings` (== KV cache size). HF will
            # otherwise crash with an index error inside the rotary/positional
            # embedding when prompt_len + new_tokens overflows. If the prompt
            # already meets-or-exceeds mpe we have to produce zero new tokens —
            # but generate() refuses max_new_tokens=0, so clamp to 1 and accept
            # the (extremely unlikely) one-position overflow rather than the
            # alternative of producing no rollout at all.
            max_new_tokens = int(cfg.max_response_len)
            mpe = getattr(self.model.config, "max_position_embeddings", None)
            if isinstance(mpe, int) and mpe > 0:
                remaining = mpe - int(P_max)
                if remaining < max_new_tokens:
                    max_new_tokens = max(1, remaining)

            # Use the cached GenerationConfig (built once in __init__) and only
            # override the per-call max_new_tokens. Avoids HF re-validating the
            # whole sampling-kwargs schema on every rollout step.
            if self._generation_config is not None:
                gen_out = self.model.generate(
                    input_ids=gen_input_ids,
                    attention_mask=gen_attn_mask,
                    generation_config=self._generation_config,
                    max_new_tokens=max_new_tokens,
                )
            else:  # pragma: no cover - fallback for stripped-down envs
                gen_kwargs = dict(
                    input_ids=gen_input_ids,
                    attention_mask=gen_attn_mask,
                    do_sample=self._do_sample,
                    num_return_sequences=self._gen_num_return_sequences,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    use_cache=True,
                )
                if self._do_sample:
                    gen_kwargs.update(
                        temperature=self._temperature,
                        top_p=self._top_p,
                        top_k=self._top_k,
                    )
                gen_out = self.model.generate(**gen_kwargs)

            # `sequences`: [num_prompts*G, P_max + new_tokens]
            sequences = gen_out.sequences  # type: ignore[attr-defined]
            scores = gen_out.scores  # type: ignore[attr-defined]
            # Generation can stop early when *all* sequences have hit EOS. In that
            # case len(scores) == actual_steps_run and sequences.shape[1] ==
            # P_max + actual_steps_run. Use the minimum of the two so a buggy
            # backend that desynchronises them can't cause an out-of-bounds slice.
            seq_new_tokens = int(sequences.shape[1] - P_max)
            R = len(scores) if scores is not None else seq_new_tokens
            R = min(R, seq_new_tokens, int(cfg.max_response_len))

            response_ids = sequences[:, P_max : P_max + R].contiguous()  # [B, R]
            B = response_ids.shape[0]

            # Per-step gather of sampling logprobs. Avoid stacking the full
            # [B, R, V] logits tensor — for B*G in the dozens, R in the
            # thousands and V >= 32k, that's tens of GB of GPU RAM. Instead we
            # do log_softmax + gather per step, accumulating only [B, R].
            if R > 0 and scores is not None and len(scores) > 0:
                gathered = torch.empty(
                    (B, R),
                    dtype=torch.float32,
                    device=response_ids.device,
                )
                for t in range(R):
                    step_logits = scores[t]
                    if step_logits.shape[0] != B:
                        # Defensive: should never happen, but if HF returns a
                        # mismatched score tensor we want to fail loudly rather
                        # than silently mis-align logprobs to tokens.
                        raise RuntimeError(
                            f"output_scores[{t}] has batch dim {step_logits.shape[0]} "
                            f"but response_ids has batch dim {B}"
                        )
                    # HF returns scores in fp32 already; only cast if it's not.
                    if step_logits.dtype != torch.float32:
                        step_logits = step_logits.float()
                    step_logp = F.log_softmax(step_logits, dim=-1)
                    gathered[:, t] = step_logp.gather(
                        -1, response_ids[:, t : t + 1]
                    ).squeeze(-1)
            else:
                gathered = torch.zeros(
                    response_ids.shape, dtype=torch.float32, device=response_ids.device
                )

            # Build response mask: 1 up to and including the first EOS per row.
            response_mask = self._build_response_mask(
                response_ids, self.eos_token_id, self.pad_token_id
            )

            # Zero out logprobs after EOS via masked_fill (cheaper than the
            # cast-to-dtype + multiply, and avoids any 0*-inf NaN landmines if
            # log_softmax produces -inf at masked positions).
            sampling_logprobs = gathered.masked_fill(response_mask == 0, 0.0)

            # Decode responses in a single batched call. We zero out tokens
            # after each row's first EOS to keep decoded text clean — the
            # tokenizer already skips special tokens (incl. EOS/PAD), so
            # padding the tail with pad_token_id produces the same string as
            # truncating per-row but in one trip through the C tokenizer.
            response_ids_for_decode = response_ids.masked_fill(
                response_mask == 0, self.pad_token_id
            )
            raw_responses: List[str] = self.tokenizer.batch_decode(
                response_ids_for_decode.tolist(),
                skip_special_tokens=True,
            )

            # Score with reward_fn. The grader gets the decoded responses and the
            # ground truths tiled to match the G-replicated layout.
            tiled_gt = [ground_truths[i // G] for i in range(num_prompts * G)]
            raw_rewards = self.reward_fn(raw_responses, tiled_gt)
            rewards = torch.tensor(raw_rewards, dtype=torch.float32)

            # prompt_index[b] = which prompt response b belongs to
            # generate's num_return_sequences=G layout puts the G samples for prompt p
            # contiguously: [p0_g0, p0_g1, ..., p0_g(G-1), p1_g0, ...].
            prompt_index = torch.arange(num_prompts).repeat_interleave(G).to(torch.long)

            # Move everything we ship out to CPU.
            return RolloutBatch(
                prompt_ids=prompt_ids.detach().cpu().to(torch.long),
                prompt_mask=prompt_mask.detach().cpu().to(torch.long),
                response_ids=response_ids.detach().cpu().to(torch.long),
                response_mask=response_mask.detach().cpu().to(torch.long),
                sampling_logprobs=sampling_logprobs.detach().cpu().float(),
                rewards=rewards.detach().cpu().float(),
                prompt_index=prompt_index.detach().cpu().to(torch.long),
                raw_prompts=prompts,
                raw_responses=raw_responses,
                ground_truths=ground_truths,
            )
        finally:
            if prev_use_cache is not None:
                try:
                    self.model.config.use_cache = prev_use_cache
                except AttributeError:
                    pass
            if was_training:
                self.model.train()
