"""In-process vLLM rollout engine.

Wraps :class:`vllm.v1.engine.async_llm.AsyncLLM` with a synchronous,
:class:`HFRolloutSampler`-compatible API.

Why in-process: avoids subprocess + HTTP latency. The trainer process holds
an :class:`AsyncLLM` client; vLLM's own background EngineCore process runs
the actual model. Weight sync between trainer and engine uses disk
checkpoints (simple, ~5-7s per sync at 1B-scale, tolerable for the
headline). NCCL-based sync is a follow-up for 7B-scale runs.

Public API:

* ``VLLMRolloutEngine.sample(examples)`` — drop-in for
  :meth:`HFRolloutSampler.sample`; returns a :class:`RolloutBatch`.
* ``VLLMRolloutEngine.sample_with_prefix(prefix_ids, K, sampling_params)``
  — for VinePPO MC rollouts. Uses ``SamplingParams(n=K)`` so vLLM batches
  the K samples and reuses the prefix KV cache automatically.
* ``VLLMRolloutEngine.sync_weights_from_path(path)`` — reload the engine
  from a checkpoint dir. Call after each policy gradient step.
* ``VLLMRolloutEngine.shutdown()`` — clean up the background EngineCore.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

import torch

from caspo.config import CASPOConfig
from caspo.rollout.sampler import RolloutBatch


@dataclass
class _Generation:
    """Single completion result, intermediate type."""
    token_ids: List[int]
    sampling_logprobs: List[float]   # logprob of the sampled token at each position
    text: str
    finish_reason: str


class VLLMRolloutEngine:
    """In-process vLLM rollout engine, synchronous API.

    Args:
        cfg: :class:`CASPOConfig`. Reads ``model_name_or_path``,
            ``tokenizer_name_or_path``, ``torch_dtype``, ``trust_remote_code``,
            ``rollout_temperature``, ``rollout_top_p``, ``rollout_top_k``,
            ``max_response_len``, ``max_prompt_len``, ``group_size``.
        reward_fn: same callback as :class:`HFRolloutSampler`. Takes
            ``(responses: List[str], ground_truths: List[str])`` and returns a
            list of float rewards.
        gpu_memory_utilization: 0..1, fraction of GPU memory vLLM may use.
        tensor_parallel_size: vLLM TP size. 1 unless model >= 7B.
        enforce_eager: True forces eager-mode (no CUDA-graph compile). False
            is faster after warmup but takes ~30s extra to compile.
        max_model_len: max combined (prompt + response) length. Defaults to
            ``cfg.max_prompt_len + cfg.max_response_len``.
        gpu_id: which GPU to place this replica on (sets ``CUDA_VISIBLE_DEVICES``
            for the engine subprocess if given). ``None`` lets vLLM choose
            from the visible-devices set the trainer was launched with.
    """

    def __init__(
        self,
        cfg: CASPOConfig,
        reward_fn: Callable[[List[str], List[str]], List[float]],
        *,
        gpu_memory_utilization: float = 0.85,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = False,
        max_model_len: Optional[int] = None,
        gpu_id: Optional[int] = None,
        seed: int = 0,
        max_num_seqs: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        extra_stop_strings: Optional[Sequence[str]] = None,
    ) -> None:
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from transformers import AutoTokenizer

        self.cfg = cfg
        self.reward_fn = reward_fn
        self.SamplingParams = SamplingParams
        self.RequestOutputKind = RequestOutputKind
        self.AsyncLLM = AsyncLLM

        # Tokenizer (used for response decoding + prompt tokenization).
        tok_path = cfg.tokenizer_name_or_path or cfg.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_path, trust_remote_code=cfg.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        # Some chat-tuned models (Qwen2 with <|im_end|>, Llama-3 with both
        # <|end_of_text|> and <|eot_id|>) expose multiple stop tokens. The
        # canonical one is tokenizer.eos_token_id, but the model's
        # generation_config can list more. Honor whichever the tokenizer /
        # generation_config exposes so chat-format rollouts don't run past
        # the assistant turn boundary.
        stop_ids = {self.eos_token_id}
        for name in ("eos_token_ids", "additional_eos_token_ids"):
            extra = getattr(self.tokenizer, name, None)
            if extra:
                try:
                    for t in extra:
                        stop_ids.add(int(t))
                except TypeError:
                    pass
        self._stop_token_ids = sorted(stop_ids)
        # Optional textual stop strings (e.g. chat-format role markers like
        # "<|im_end|>" if a tokenizer mis-tokenizes them, or "</s>"). Empty
        # tuple disables. Stored once and reused per SamplingParams call.
        self._stop_strings: tuple[str, ...] = tuple(extra_stop_strings or ())

        if max_model_len is None:
            max_model_len = int(cfg.max_prompt_len) + int(cfg.max_response_len)

        # If a specific gpu_id was requested, pin via CUDA_VISIBLE_DEVICES so
        # the engine subprocess sees only that GPU. We save the previous
        # value and restore it after engine init so the trainer process's
        # device visibility isn't permanently mutated (the engine subprocess
        # has already inherited the modified env at fork time).
        prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))

        # Sentinels so __del__ / shutdown don't trip on partially-constructed
        # objects if the engine init below throws.
        self.engine = None
        self._loop = None
        try:
            # max_num_seqs default in vLLM V1 is 128. RL workloads (esp.
            # VinePPO MC with many prefixes × K) can exceed that, forcing the
            # scheduler to queue requests serially even though KV cache fits.
            # Allow caller to override; pass through only when explicitly set
            # so we don't second-guess vLLM's auto-tuning.
            engine_kwargs = dict(
                model=cfg.model_name_or_path,
                tokenizer=tok_path,
                tokenizer_mode="auto",
                trust_remote_code=cfg.trust_remote_code,
                dtype=cfg.torch_dtype,             # 'bfloat16' / 'float16' / 'auto'
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                enable_prefix_caching=True,
                enforce_eager=enforce_eager,
                max_model_len=max_model_len,
                seed=seed,
                disable_log_stats=True,
            )
            if max_num_seqs is not None:
                engine_kwargs["max_num_seqs"] = int(max_num_seqs)
            if max_num_batched_tokens is not None:
                engine_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
            engine_args = AsyncEngineArgs(**engine_kwargs)
            self.engine = AsyncLLM.from_engine_args(engine_args)
        finally:
            # Restore parent process env. (The engine subprocess already forked
            # with the override, so its CUDA_VISIBLE_DEVICES sticks.)
            if gpu_id is not None:
                if prev_cvd is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
        # Persistent event loop so we don't create+destroy one per call.
        # If new_event_loop fails (extremely unlikely) we tear the engine
        # down so its EngineCore subprocess + NCCL sockets don't leak.
        try:
            self._loop = asyncio.new_event_loop()
        except Exception:
            try:
                self.engine.shutdown()
            except Exception:
                pass
            self.engine = None
            raise

    # ------------------------------------------------------------------
    # Async helpers
    # ------------------------------------------------------------------

    async def _drain_one(self, prompt: Any, sampling_params: Any, request_id: str):
        """Consume the AsyncLLM streaming generator and return the final RequestOutput.

        IMPORTANT (vLLM V1 quirk): with ``n>1`` and the default
        ``output_kind=CUMULATIVE``, the engine fans out to ``n`` child requests
        and the per-request collector emits one ``RequestOutput`` *per child*
        (see ``vllm/v1/engine/parallel_sampling.py:115``). Naively keeping the
        last ``out`` would drop ``n-1`` of the ``n`` completions.

        We rely on ``output_kind=FINAL_ONLY`` (set in ``_build_sampling_params``)
        to make the engine aggregate server-side and emit a single final
        ``RequestOutput`` with all ``n`` completions. As defense-in-depth we
        also accumulate ``CompletionOutput`` entries across iterations here,
        keyed by their ``index``, so that even if a future vLLM version reverts
        to per-child streaming we still return the union.
        """
        final = None
        comps_by_index: dict = {}
        try:
            async for out in self.engine.generate(
                prompt=prompt, sampling_params=sampling_params, request_id=request_id,
            ):
                final = out
                for c in (out.outputs or []):
                    # Replace if same index seen again (CUMULATIVE) — last write wins.
                    comps_by_index[int(getattr(c, "index", len(comps_by_index)))] = c
        except (asyncio.CancelledError, GeneratorExit):
            # Abort the engine-side request so it stops consuming KV cache /
            # GPU steps; otherwise a cancelled-from-outside drain leaks the
            # request inside the EngineCore until the engine shuts down.
            try:
                abort = self.engine.abort(request_id)
                if hasattr(abort, "__await__"):
                    await abort
            except Exception:
                pass
            raise
        except Exception:
            # Same: unexpected mid-stream errors must abort the engine-side
            # request, not just bubble up to the gather().
            try:
                abort = self.engine.abort(request_id)
                if hasattr(abort, "__await__"):
                    await abort
            except Exception:
                pass
            raise
        if final is None:
            return None
        # If the final output already contains all completions (FINAL_ONLY path),
        # this is a no-op. Otherwise, splice in the union of indices.
        if comps_by_index and len(comps_by_index) > len(final.outputs or []):
            final.outputs = [comps_by_index[k] for k in sorted(comps_by_index)]
        return final

    async def _drain_many(self, requests: Sequence):
        """Drain a list of (prompt, sampling_params, request_id) tuples concurrently."""
        tasks = [self._drain_one(p, sp, rid) for p, sp, rid in requests]
        return await asyncio.gather(*tasks)

    def _run(self, coro):
        """Run a coroutine on our persistent event loop and return the result."""
        return self._loop.run_until_complete(coro)

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------

    def _build_sampling_params(
        self, *, n: int, max_tokens: int, temperature: Optional[float] = None,
        top_p: Optional[float] = None, top_k: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        cfg = self.cfg
        temp = float(cfg.rollout_temperature) if temperature is None else float(temperature)
        tp = float(cfg.rollout_top_p) if top_p is None else float(top_p)
        tk_cfg = cfg.rollout_top_k if top_k is None else top_k
        tk = int(tk_cfg) if tk_cfg and tk_cfg > 0 else -1
        kwargs = dict(
            n=int(n),
            temperature=temp,
            top_p=tp,
            top_k=tk,
            max_tokens=int(max_tokens),
            # logprobs=1 returns the top-1 logprob *plus* the sampled-token's
            # logprob whenever the sampled token isn't in the top-1
            # (vllm/logprobs.py append_logprobs_for_next_position). The dict
            # is therefore guaranteed to contain the sampled token id.
            logprobs=1,
            stop_token_ids=list(self._stop_token_ids),
            # CRITICAL for n>1: with the default CUMULATIVE the AsyncLLM V1
            # collector emits one RequestOutput per child request, so the
            # consumer naturally sees only the last child's completion.
            # FINAL_ONLY makes the engine aggregate the n outputs and emit
            # a single final RequestOutput with all n completions.
            output_kind=self.RequestOutputKind.FINAL_ONLY,
        )
        if self._stop_strings:
            # vLLM matches stop-strings against detokenized output; useful for
            # chat models whose role-end markers can split across token ids.
            kwargs["stop"] = list(self._stop_strings)
        if seed is not None:
            kwargs["seed"] = int(seed)
        return self.SamplingParams(**kwargs)

    # Class-level once-flag so the logprob-contract warning fires at most
    # once per process. Without this, a corrupted contract would emit
    # warnings.warn() for every completion in every step — Python's warnings
    # filter dedupes by (message, location) but the per-call format-string
    # work + filter lookup is still wasted in the hot path.
    _logprob_contract_warned: bool = False

    @classmethod
    def _extract_completion(cls, comp) -> _Generation:
        """Extract token_ids + per-token sampling-time logprobs from a CompletionOutput.

        With ``logprobs=1``, vLLM is documented to insert the sampled token's
        logprob alongside the top-1 (so the per-position dict is guaranteed
        to contain the sampled tid). If a future version drops that contract
        we surface it loudly — silently substituting an arbitrary entry's
        logprob would produce wrong importance ratios in the trainer.
        """
        token_ids = list(comp.token_ids)
        per_token_lps = comp.logprobs or []
        sampling_logprobs: List[float] = []
        missing = 0
        L = len(per_token_lps)
        for t, tid in enumerate(token_ids):
            lp_dict = per_token_lps[t] if t < L else None
            if lp_dict is None:
                sampling_logprobs.append(0.0)
                continue
            lp_obj = lp_dict.get(int(tid))
            if lp_obj is None:
                # Sampled-token logprob is missing — never expected. Use 0.0
                # (neutral) rather than a top-1 logprob from a different token.
                missing += 1
                sampling_logprobs.append(0.0)
                continue
            sampling_logprobs.append(float(getattr(lp_obj, "logprob", 0.0)))
        if missing and not cls._logprob_contract_warned:
            cls._logprob_contract_warned = True
            warnings.warn(
                f"vLLM logprob dict missing sampled token at {missing} positions "
                f"(out of {len(token_ids)}); substituted 0.0. Check logprobs=1 "
                "contract. Further occurrences in this process suppressed."
            )
        return _Generation(
            token_ids=token_ids,
            sampling_logprobs=sampling_logprobs,
            text=comp.text or "",
            finish_reason=str(getattr(comp, "finish_reason", "") or ""),
        )

    @staticmethod
    def _pad_right(rows: List[List[int]], pad: int) -> torch.LongTensor:
        if not rows:
            return torch.zeros((0, 0), dtype=torch.long)
        R = max((len(r) for r in rows), default=0)
        out = torch.full((len(rows), R), pad, dtype=torch.long)
        for i, r in enumerate(rows):
            if r:
                out[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        return out

    @staticmethod
    def _pad_right_float(rows: List[List[float]], R: int) -> torch.Tensor:
        if not rows:
            return torch.zeros((0, R), dtype=torch.float32)
        out = torch.zeros((len(rows), R), dtype=torch.float32)
        for i, r in enumerate(rows):
            if r:
                out[i, : len(r)] = torch.tensor(r, dtype=torch.float32)
        return out

    @staticmethod
    def _pad_left(rows: List[List[int]], pad: int) -> torch.LongTensor:
        if not rows:
            return torch.zeros((0, 0), dtype=torch.long)
        P = max((len(r) for r in rows), default=0)
        out = torch.full((len(rows), P), pad, dtype=torch.long)
        for i, r in enumerate(rows):
            if r:
                out[i, P - len(r) :] = torch.tensor(r, dtype=torch.long)
        return out

    @staticmethod
    def _build_response_mask(
        response_ids: torch.Tensor, eos_id: int, pad_id: int
    ) -> torch.LongTensor:
        # 1 up to & including the first EOS; 0 after. Pads with pad_id != eos_id are 0.
        is_eos = response_ids == eos_id
        cum = is_eos.to(torch.long).cumsum(dim=1)
        first_eos = is_eos & (cum == 1)
        keep = (cum == 0) | first_eos
        if pad_id != eos_id:
            keep = keep & (response_ids != pad_id)
        return keep.to(torch.long)

    # ------------------------------------------------------------------
    # Public API: sample (HFRolloutSampler-compatible)
    # ------------------------------------------------------------------

    def sample(self, examples: List[dict]) -> RolloutBatch:
        """Sample G responses per prompt. Returns a :class:`RolloutBatch`.

        Layout matches :meth:`HFRolloutSampler.sample` exactly: G samples per
        prompt are contiguous, ``prompt_index[b]`` indexes into the unique-
        prompt list.
        """
        if not examples:
            raise ValueError("examples must be non-empty")

        cfg = self.cfg
        G = int(cfg.group_size)
        prompts = [str(ex["prompt"]) for ex in examples]
        ground_truths = [str(ex["ground_truth"]) for ex in examples]
        num_prompts = len(prompts)

        # Tokenize prompts (left-truncated to cfg.max_prompt_len). Batched
        # tokenizer call: HuggingFace fast tokenizers parallelize internally
        # via Rust threads when given a list, which is materially faster
        # than a Python loop for typical RL batch sizes (~16-64 prompts).
        max_p = int(cfg.max_prompt_len) if cfg.max_prompt_len else 0
        enc = self.tokenizer(prompts, add_special_tokens=False)
        prompt_token_ids: List[List[int]] = []
        for ids in enc["input_ids"]:
            if max_p and len(ids) > max_p:
                ids = ids[-max_p:]
            prompt_token_ids.append(list(ids))

        # vLLM v1 quirk: SamplingParams(n=G) is documented but the AsyncLLM
        # collector returns only 1 completion per request even with
        # output_kind=FINAL_ONLY in many 0.19.x builds. Match the eval path
        # (caspo/eval/benchmarks.py): submit G independent n=1 requests per
        # prompt and aggregate. As a bonus this gives the continuous batcher
        # more independent work units to interleave.
        sp = self._build_sampling_params(n=1, max_tokens=int(cfg.max_response_len))

        from vllm.inputs import TokensPrompt

        # Order matters: emit (prompt p, sample 0..G-1) contiguously so that
        # outputs[i*G + j] corresponds to (prompt i, sample j) and prompt_index
        # == [0]*G + [1]*G + ... matches HFRolloutSampler's layout.
        requests = []
        for prompt_i, ids in enumerate(prompt_token_ids):
            tp = TokensPrompt(prompt_token_ids=ids)
            for samp_j in range(G):
                requests.append(
                    (tp, sp, f"req-p{prompt_i}-s{samp_j}-{uuid.uuid4().hex}")
                )
        outputs = self._run(self._drain_many(requests))

        # Each output has 1 CompletionOutput; flatten in submitted order.
        flat_token_ids: List[List[int]] = []
        flat_logprobs: List[List[float]] = []
        flat_responses: List[str] = []
        from vllm.outputs import CompletionOutput as _CompCls  # type: ignore
        for req_out in outputs:
            # _drain_one returns None if the engine produced no RequestOutput
            # at all (e.g. the request was aborted before any token landed).
            comps = list((req_out.outputs if req_out is not None else None) or [])
            if not comps:
                # Failed request: emit a stub so the (prompt, G) layout stays
                # aligned with prompt_index. Reward function sees an empty
                # response and the trainer assigns reward 0 / mask 0.
                comps = [
                    _CompCls(  # type: ignore[call-arg]
                        index=0, text="", token_ids=[], cumulative_logprob=None,
                        logprobs=[], finish_reason="error", stop_reason=None,
                    )
                ]
            # n=1 — take the first (only) completion.
            gen = self._extract_completion(comps[0])
            flat_token_ids.append(gen.token_ids)
            flat_logprobs.append(gen.sampling_logprobs)
            flat_responses.append(gen.text)

        B = num_prompts * G

        # Pack tensors. Prompts are left-padded; responses right-padded.
        prompt_ids_packed = self._pad_left(prompt_token_ids, self.pad_token_id)
        prompt_mask = (prompt_ids_packed != self.pad_token_id).to(torch.long)

        response_ids = self._pad_right(flat_token_ids, self.pad_token_id)
        R = response_ids.shape[1] if response_ids.numel() > 0 else 0
        sampling_logprobs = self._pad_right_float(flat_logprobs, R)
        response_mask = self._build_response_mask(
            response_ids, self.eos_token_id, self.pad_token_id,
        )
        sampling_logprobs = sampling_logprobs * response_mask.to(sampling_logprobs.dtype)

        # Reward (decoded text + ground truth tiled to per-response).
        tiled_gt = [ground_truths[i // G] for i in range(B)]
        rewards = torch.tensor(self.reward_fn(flat_responses, tiled_gt), dtype=torch.float32)

        prompt_index = torch.arange(num_prompts).repeat_interleave(G).to(torch.long)

        return RolloutBatch(
            prompt_ids=prompt_ids_packed.to(torch.long),
            prompt_mask=prompt_mask.to(torch.long),
            response_ids=response_ids.to(torch.long),
            response_mask=response_mask.to(torch.long),
            sampling_logprobs=sampling_logprobs.float(),
            rewards=rewards.float(),
            prompt_index=prompt_index.to(torch.long),
            raw_prompts=prompts,
            raw_responses=flat_responses,
            ground_truths=ground_truths,
        )

    # ------------------------------------------------------------------
    # Public API: sample_with_prefix (VinePPO MC rollouts)
    # ------------------------------------------------------------------

    def sample_with_prefix(
        self,
        prefix_token_ids_list: List[List[int]],
        K: int,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[List[_Generation]]:
        """K rollouts from each prefix; returns ``len(prefixes) × K`` generations.

        Used by VinePPO's MC value estimation. vLLM v1's AsyncLLM returns
        only 1 completion per request even with ``n=K``, so we issue K
        independent ``n=1`` requests per prefix. The prefix cache is keyed
        by token ids (``enable_prefix_caching=True`` in __init__) so all K
        requests still share the prefix KV — same throughput as ``n=K``
        would give without the silent-drop bug.
        """
        if not prefix_token_ids_list:
            return []
        if max_tokens is None:
            max_tokens = int(self.cfg.max_response_len)

        sp = self._build_sampling_params(
            n=1,
            max_tokens=int(max_tokens),
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )

        from vllm.inputs import TokensPrompt
        from vllm.outputs import CompletionOutput as _CompCls  # type: ignore

        K_int = int(K)
        requests = []
        for prefix_i, p in enumerate(prefix_token_ids_list):
            tp = TokensPrompt(prompt_token_ids=list(p))
            for samp_j in range(K_int):
                requests.append(
                    (tp, sp, f"prefix-p{prefix_i}-s{samp_j}-{uuid.uuid4().hex}")
                )
        outputs = self._run(self._drain_many(requests))

        out_per_prefix: List[List[_Generation]] = []
        for prefix_i in range(len(prefix_token_ids_list)):
            slice_out = outputs[prefix_i * K_int : (prefix_i + 1) * K_int]
            gens: List[_Generation] = []
            for req_out in slice_out:
                # _drain_one can return None (no RequestOutput ever yielded);
                # produce a stub so caller always sees K entries per prefix.
                comps = list((req_out.outputs if req_out is not None else None) or [])
                if not comps:
                    comps = [
                        _CompCls(  # type: ignore[call-arg]
                            index=0, text="", token_ids=[], cumulative_logprob=None,
                            logprobs=[], finish_reason="error", stop_reason=None,
                        )
                    ]
                gens.append(self._extract_completion(comps[0]))
            out_per_prefix.append(gens)
        return out_per_prefix

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def sync_weights_from_path(self, checkpoint_path: str) -> float:
        """Reload model weights from a HuggingFace checkpoint dir.

        Uses vLLM's ``reload_weights(weights_path=..., is_checkpoint_format=True)``
        via ``collective_rpc``. Resets the prefix cache afterward (cached
        prefixes are stale under the new weights).

        Caller MUST guarantee no in-flight ``generate`` requests when this is
        called — ``reload_weights`` swaps the model state mid-step otherwise
        and the responses already in flight will mix old+new params. Our
        ``sample`` and ``sample_with_prefix`` are synchronous (block until
        all requests drain on the persistent event loop), so single-thread
        callers are safe by construction.

        Order matters: reload weights FIRST, then reset prefix cache. The
        prefix cache is keyed only by token ids so without resetting, a
        cache hit would serve KV computed under the OLD weights. If we
        reset first, a concurrent in-flight request (forbidden above but
        possible if the contract is violated) could re-populate the cache
        with stale KV before the reload lands. Reload-then-reset is the
        only ordering that's correct under both regimes.

        Returns the wall-clock seconds spent.
        """
        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_path}")
        # Best-effort assert: no pending requests in the OutputProcessor.
        try:
            n_pending = self.engine.output_processor.get_num_unfinished_requests()
            if n_pending:
                warnings.warn(
                    f"sync_weights_from_path called with {n_pending} unfinished "
                    "requests; their outputs will mix old + new weights."
                )
        except Exception:
            pass
        t0 = time.time()
        self._run(
            self._collective_rpc(
                "reload_weights",
                kwargs={
                    "weights_path": checkpoint_path,
                    "is_checkpoint_format": True,
                },
            )
        )
        # reset_prefix_cache is async on V1 engine — drive it on our event loop.
        # Strictly AFTER reload_weights so any cache hits during reload are
        # invalidated, not after a fresh re-population window.
        try:
            reset_coro = self.engine.reset_prefix_cache()
            if hasattr(reset_coro, "__await__"):
                self._run(reset_coro)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"reset_prefix_cache failed: {e}")
        return time.time() - t0

    async def _collective_rpc(self, method: str, args: tuple = (), kwargs: Optional[dict] = None):
        kwargs = kwargs or {}
        return await self.engine.collective_rpc(method, args=args, kwargs=kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        # Idempotent: safe to call multiple times (from explicit shutdown +
        # __del__ both).
        engine = getattr(self, "engine", None)
        loop = getattr(self, "_loop", None)
        # 1. Tell the engine to stop its EngineCore subprocess + background
        #    output handler. AsyncLLM.shutdown is sync.
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as e:
                warnings.warn(f"engine.shutdown failed: {e}")
            self.engine = None  # type: ignore[assignment]
        # 2. Cancel any pending tasks left on our persistent loop, run the
        #    loop briefly so the cancellations propagate (CancelledError is
        #    raised inside any in-flight `generate` async generators), then
        #    close. Closing a loop with pending tasks emits noisy warnings
        #    and can leak tasks attached to the engine's RequestOutputCollector.
        if loop is not None and not loop.is_closed():
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
            try:
                loop.close()
            except Exception:
                pass
            self._loop = None  # type: ignore[assignment]

    def __del__(self) -> None:  # best-effort
        try:
            self.shutdown()
        except Exception:
            pass
