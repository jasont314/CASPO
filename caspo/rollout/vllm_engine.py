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
  — for VinePPO MC rollouts. In ``vllm_multi_sample_mode=auto`` it first
  tries ``SamplingParams(n=K)`` and falls back to expanded requests if the
  installed vLLM runtime does not return K completions.
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
from typing import Any, Callable, List, Optional, Sequence, Union

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
        gpu_ids: Optional[Sequence[int]] = None,
        seed: int = 0,
        max_num_seqs: Optional[int] = None,
        max_num_batched_tokens: Optional[int] = None,
        max_inflight_requests: Optional[int] = None,
        extra_stop_strings: Optional[Sequence[str]] = None,
        enable_chunked_prefill: bool = False,
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
        self.max_inflight_requests = (
            int(max_inflight_requests)
            if max_inflight_requests is not None else None
        )
        self._multi_sample_mode = str(cfg.vllm_multi_sample_mode)
        self._parallel_sampling_supported: Optional[bool] = None
        self._parallel_sampling_warned = False
        self._return_logprobs = bool(cfg.vllm_return_logprobs)
        self._weight_sync_backend = str(cfg.vllm_weight_sync_backend)

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
        # Cache of prompt-string → token_ids (post add_special_tokens=False).
        # RL training reuses the same prompt strings across iterations
        # (per-prompt MC for VinePPO, multi-epoch PPO over the same rollout
        # batch), so a small LRU on raw string identity removes the per-step
        # tokenizer call. Bound size keeps memory predictable across long runs.
        self._prompt_token_cache: "dict[str, List[int]]" = {}
        self._prompt_token_cache_max = 1024

        if max_model_len is None:
            max_model_len = int(cfg.max_prompt_len) + int(cfg.max_response_len)

        # If a specific gpu_id was requested, pin via CUDA_VISIBLE_DEVICES so
        # the engine subprocess sees only that GPU. We save the previous
        # value and restore it after engine init so the trainer process's
        # device visibility isn't permanently mutated (the engine subprocess
        # has already inherited the modified env at fork time).
        prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        prev_allow_insecure = os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION")
        # torchrun exports rank/world env vars. A rank-local vLLM engine should
        # not inherit those, otherwise its EngineCore can mistake a
        # single-GPU local engine for part of the trainer's process group.
        dist_env_keys = (
            "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
            "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
            "MASTER_ADDR", "MASTER_PORT",
        )
        prev_dist_env = {k: os.environ.get(k) for k in dist_env_keys}
        if gpu_ids is not None and gpu_id is not None:
            raise ValueError(
                "Pass at most one of gpu_id (TP=1) or gpu_ids (TP>=1, multi-GPU)"
            )
        if gpu_ids is not None:
            # Disaggregated path: pin AsyncLLM workers to a contiguous TP
            # group of physical GPUs. Translate physical IDs through the
            # current CUDA_VISIBLE_DEVICES so the eventual env var still
            # references *physical* device indices the OS recognizes.
            gpu_id_list = [int(g) for g in gpu_ids]
            if not gpu_id_list:
                raise ValueError("gpu_ids must contain at least one GPU id")
            if int(tensor_parallel_size) != len(gpu_id_list):
                raise ValueError(
                    f"tensor_parallel_size={tensor_parallel_size} != "
                    f"len(gpu_ids)={len(gpu_id_list)} — must match"
                )
            if prev_cvd:
                visible = [x.strip() for x in prev_cvd.split(",") if x.strip()]
                # gpu_ids in this branch are *physical* IDs (the same the
                # launcher set in CASPO_ROLLOUT_GPU_PHYSICAL_IDS). They
                # must each appear in the parent's visible list.
                phys_to_visible_idx = {p: i for i, p in enumerate(visible)}
                for g in gpu_id_list:
                    if str(g) not in phys_to_visible_idx:
                        raise ValueError(
                            f"physical GPU id {g} (from gpu_ids) is not in "
                            f"parent CUDA_VISIBLE_DEVICES={prev_cvd!r}"
                        )
                # AsyncLLM workers will inherit our env at spawn — present
                # them with EXACTLY the rollout GPUs, in TP-rank order, as
                # the only visible devices. They will index them as
                # cuda:0..cuda:N-1.
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_id_list)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_id_list)
            # Mirror the rank-local env stripping below so vLLM workers
            # don't inherit the trainer's torchrun rendezvous and try to
            # join FSDP's PG. AsyncLLM TP=N spawns its own multiproc PG
            # with TP-internal ranks; we just make the inherited env safe
            # so init_distributed_environment doesn't get confused.
            base_port = 41100  # offset to avoid collision with TP=1 path
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(base_port + int(gpu_id_list[0]))
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["LOCAL_WORLD_SIZE"] = "1"
            for key in ("GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
                os.environ.pop(key, None)
            prev_dist_env["VLLM_HOST_IP"] = os.environ.get("VLLM_HOST_IP")
            os.environ["VLLM_HOST_IP"] = "127.0.0.1"
        elif gpu_id is not None:
            gpu_idx = int(gpu_id)
            if prev_cvd:
                visible = [x.strip() for x in prev_cvd.split(",") if x.strip()]
                if gpu_idx < 0 or gpu_idx >= len(visible):
                    raise ValueError(
                        f"gpu_id={gpu_idx} outside CUDA_VISIBLE_DEVICES={prev_cvd!r}"
                    )
                os.environ["CUDA_VISIBLE_DEVICES"] = visible[gpu_idx]
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            # vLLM EngineCore unconditionally calls init_distributed_environment
            # even at TP=1. Inheriting torchrun's MASTER_ADDR/MASTER_PORT makes
            # the EngineCore try to join the trainer's rendezvous (timeouts /
            # RuntimeError). Override to a per-rank-local TCP store on
            # 127.0.0.1:<unique port>, world_size=1, rank=0. The port is
            # offset by gpu_idx to avoid collisions when multiple ranks
            # share a host. Saved values are restored after engine init.
            base_port = 41000
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = str(base_port + gpu_idx)
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            os.environ["LOCAL_RANK"] = "0"
            os.environ["LOCAL_WORLD_SIZE"] = "1"
            for key in ("GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE"):
                os.environ.pop(key, None)
            # vLLM picks its TCP-store host via VLLM_HOST_IP, falling back
            # to socket.gethostbyname(socket.gethostname()) which resolves
            # to the machine's external IP (e.g., 10.128.0.20). On a
            # multi-rank trainer that means each rank's EngineCore tries
            # to connect to the same external IP — torch.distributed
            # creates a duplicate-master collision and times out. Force
            # 127.0.0.1 so each rank gets its own loopback TCPStore.
            prev_dist_env["VLLM_HOST_IP"] = os.environ.get("VLLM_HOST_IP")
            os.environ["VLLM_HOST_IP"] = "127.0.0.1"
        if self._weight_sync_backend == "ipc":
            # vLLM's IPC update API serializes CUDA IPC handles through the
            # EngineCore control channel. This is local-process-only for our
            # trainer/vLLM topology; the env flag must be visible before the
            # EngineCore subprocess starts and remain visible in the frontend
            # process while it decodes EngineCore utility responses.
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

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
                # Set CASPO_VLLM_LOG_STATS=1 in the launcher to surface
                # vLLM's per-iteration scheduler stats (prompt throughput,
                # generation throughput, GPU KV cache usage, prefix-cache
                # hit rate). Useful for diagnosing decode-bound vs
                # KV-bound vs prefix-cache-miss regimes; disable for
                # production to keep logs lean.
                disable_log_stats=os.environ.get(
                    "CASPO_VLLM_LOG_STATS", "0"
                ) not in ("1", "true", "True"),
            )
            # KV cache dtype: cfg.vllm_kv_cache_dtype is None (auto, fp16/bf16
            # to match the model dtype) or "fp8". fp8 halves KV memory, which
            # at vllm_gpu_memory_utilization=const doubles the concurrent
            # decode batch size — directly speeds up VinePPO's K=9 MC fan-out.
            # Verified inference-only fp8 in vLLM has accuracy hit below seed
            # noise on avg@k benchmarks; safe for rollout where we don't need
            # bit-exact KV with the trainer's bf16 forward.
            if cfg.vllm_kv_cache_dtype:
                engine_kwargs["kv_cache_dtype"] = str(cfg.vllm_kv_cache_dtype)
            # vLLM's "custom" all-reduce uses CUDA IPC mem handles to bypass
            # NCCL for small payloads. On installs where flashinfer's JIT
            # all-reduce kernel fails to compile (eg the env we hit on this
            # H100 box: flashinfer 0.6.6 uses std::optional but the
            # ninja-issued nvcc invocation doesn't have -std=c++17, so
            # trtllm_allreduce_fusion.cu fails with "namespace 'std' has no
            # member 'optional'") the next fallback is the custom kernel,
            # which then hits "Cuda error custom_all_reduce.cuh:455 invalid
            # argument" and kills VllmWorker-0. Disabling the custom path
            # forces NCCL all-reduce, which works correctly on H100 NVLink.
            # Only matters when tensor_parallel_size > 1 (TP=1 has no
            # all-reduce path); harmless to set unconditionally.
            if int(tensor_parallel_size) > 1:
                engine_kwargs["disable_custom_all_reduce"] = True
            if max_num_seqs is not None:
                engine_kwargs["max_num_seqs"] = int(max_num_seqs)
            # Chunked prefill default: OFF for rollout. Verified empirically
            # (Apr 2026) that VinePPO K=9 MC rollouts regress ~70% (191s ->
            # 321s/step) when chunked_prefill is forced on, because mixed
            # prefill-decode CUDA graphs penalize the MC pattern of many
            # short prefixes followed by decodes. Eval still benefits and
            # forces it on at its own engine construction site. Caller can
            # opt in via ``enable_chunked_prefill=True`` if their workload
            # is prefill-heavy.
            if enable_chunked_prefill:
                engine_kwargs["enable_chunked_prefill"] = True
            if max_num_batched_tokens is not None:
                engine_kwargs["max_num_batched_tokens"] = int(max_num_batched_tokens)
            if self._weight_sync_backend == "ipc":
                engine_kwargs["weight_transfer_config"] = {"backend": "ipc"}
            elif self._weight_sync_backend == "nccl":
                # Side NCCL group rendezvous: trainer rank 0 + N vLLM
                # workers (TP-ranks 0..N-1). World = 1 + tp. The launcher
                # exports CASPO_WEIGHT_SYNC_MASTER_ADDR/PORT (separate
                # from the trainer's torchrun MASTER_PORT so the two
                # rendezvous don't collide).
                #
                # vLLM's WeightTransferConfig pydantic schema only
                # accepts {"backend": ...} — init_info is NOT a config
                # field; it's passed later via the
                # ``init_weight_transfer_engine`` collective RPC. Two
                # plumbing channels.
                ws_addr = os.environ.get("CASPO_WEIGHT_SYNC_MASTER_ADDR", "127.0.0.1")
                ws_port_str = os.environ.get("CASPO_WEIGHT_SYNC_MASTER_PORT", "0")
                if not ws_port_str or ws_port_str == "0":
                    raise RuntimeError(
                        "vllm_weight_sync_backend='nccl' requires the launcher "
                        "to export CASPO_WEIGHT_SYNC_MASTER_PORT (see "
                        "scripts/_launch_7b_disagg.sh / "
                        "scripts/_launch_7b_tp8_colocated.sh)."
                    )
                # NCCLWeightTransferInitInfo requires rank_offset: vLLM
                # workers compute their NCCL rank as
                # ``worker_rank + rank_offset``. Trainer is rank 0, workers
                # are ranks 1..N → rank_offset=1 shifts the worker block
                # past the trainer.
                self._nccl_weight_sync_init_info = {
                    "master_address": ws_addr,
                    "master_port": int(ws_port_str),
                    "rank_offset": 1,
                    "world_size": 1 + int(tensor_parallel_size),
                }
                engine_kwargs["weight_transfer_config"] = {"backend": "nccl"}
            engine_args = AsyncEngineArgs(**engine_kwargs)
            self.engine = AsyncLLM.from_engine_args(engine_args)

            # Persistent event loop so we don't create+destroy one per call.
            # Keep the rank-local vLLM env active while priming; some vLLM V1
            # builds lazily start frontend/EngineCore work from this metadata
            # request, not only from ``from_engine_args`` above.
            try:
                self._loop = asyncio.new_event_loop()
                self._loop.run_until_complete(self._prime_async_frontend())
            except Exception:
                try:
                    self.engine.shutdown()
                except Exception:
                    pass
                self.engine = None
                raise

            # NCCL weight-sync side group: trainer rank 0 + N vLLM
            # workers join a separate PyNcclCommunicator. Both sides
            # block at TCPStore rendezvous, so we must run them
            # concurrently. Spawn a background thread on the trainer
            # for ``trainer_init``, then trigger workers via
            # collective_rpc; once both ends connect, the group is
            # alive for repeated trainer_send_weights calls below.
            self._nccl_weight_sync_group = None
            if self._weight_sync_backend == "nccl":
                from vllm.distributed.weight_transfer.nccl_engine import (
                    NCCLWeightTransferEngine,
                )
                import threading

                init_info_copy = dict(self._nccl_weight_sync_init_info)
                holder: dict = {}
                err: dict = {}

                def _trainer_init_thread():
                    try:
                        holder["group"] = NCCLWeightTransferEngine.trainer_init(
                            init_info_copy
                        )
                    except Exception as e:  # noqa: BLE001
                        err["err"] = e

                t = threading.Thread(target=_trainer_init_thread, daemon=True)
                t.start()
                # Trigger the workers. This is synchronous; it returns
                # only after every TP worker has called
                # init_weight_transfer_engine, which is exactly what
                # rendezvous needs from the receiver side.
                try:
                    # The vLLM RPC dispatcher passes args/kwargs through
                    # to each worker's method. init_weight_transfer_engine
                    # accepts the init_info dict positionally.
                    self._loop.run_until_complete(
                        self._collective_rpc(
                            "init_weight_transfer_engine",
                            args=(init_info_copy,),
                        )
                    )
                except Exception:
                    try:
                        self.engine.shutdown()
                    except Exception:
                        pass
                    raise
                t.join(timeout=120.0)
                if t.is_alive():
                    raise RuntimeError(
                        "NCCL weight-sync trainer_init thread did not finish "
                        "within 120s after vLLM workers signalled init. "
                        "Check master_addr/master_port reachability."
                    )
                if "err" in err:
                    raise err["err"]
                self._nccl_weight_sync_group = holder.get("group")
                if self._nccl_weight_sync_group is None:
                    raise RuntimeError(
                        "NCCLWeightTransferEngine.trainer_init returned None"
                    )
        finally:
            # Restore parent process env. (The engine subprocess already forked
            # with the override, so its CUDA_VISIBLE_DEVICES sticks.)
            if gpu_id is not None or gpu_ids is not None:
                if prev_cvd is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd
                for key, value in prev_dist_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            if self._weight_sync_backend != "ipc":
                if prev_allow_insecure is None:
                    os.environ.pop("VLLM_ALLOW_INSECURE_SERIALIZATION", None)
                else:
                    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = prev_allow_insecure

    # ------------------------------------------------------------------
    # Async helpers
    # ------------------------------------------------------------------

    async def _prime_async_frontend(self) -> None:
        """Start AsyncLLM frontend tasks on our persistent event loop.

        Recent vLLM V1 builds lazily request EngineCore metadata before
        starting the output handler when ``AsyncLLM`` was constructed outside a
        running asyncio loop. In this embedded trainer, that can deadlock: the
        metadata request waits for an EngineCore response while no output task
        is draining EngineCore. Starting the output handler here keeps later
        ``generate()`` calls on the same loop and avoids the first-request
        stall.
        """
        runner = getattr(self.engine, "_run_output_handler", None)
        if runner is not None:
            runner()
        get_tasks = getattr(self.engine, "get_supported_tasks", None)
        if get_tasks is not None:
            await get_tasks()

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
        cap = self.max_inflight_requests
        if cap is None or cap >= len(requests):
            tasks = [self._drain_one(p, sp, rid) for p, sp, rid in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            errors = [r for r in results if isinstance(r, BaseException)]
            if errors:
                raise errors[0]
            return results

        sem = asyncio.Semaphore(max(1, int(cap)))

        async def _limited(req):
            p, sp, rid = req
            async with sem:
                return await self._drain_one(p, sp, rid)

        tasks = [_limited(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in results if isinstance(r, BaseException)]
        if errors:
            raise errors[0]
        return results

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
            stop_token_ids=list(self._stop_token_ids),
            # CRITICAL for n>1: with the default CUMULATIVE the AsyncLLM V1
            # collector emits one RequestOutput per child request, so the
            # consumer naturally sees only the last child's completion.
            # FINAL_ONLY makes the engine aggregate the n outputs and emit
            # a single final RequestOutput with all n completions.
            output_kind=self.RequestOutputKind.FINAL_ONLY,
        )
        if self._return_logprobs:
            # logprobs=1 returns top-1 plus the sampled-token logprob whenever
            # the sampled token is not top-1. Disabled by default because PPO
            # rescoring uses the trainer model's logprobs.
            kwargs["logprobs"] = 1
        if self._stop_strings:
            # vLLM matches stop-strings against detokenized output; useful for
            # chat models whose role-end markers can split across token ids.
            kwargs["stop"] = list(self._stop_strings)
        if seed is not None:
            kwargs["seed"] = int(seed)
        return self.SamplingParams(**kwargs)

    def _can_try_parallel_sampling(self, n: int) -> bool:
        """Whether to submit one vLLM request with ``SamplingParams(n=n)``."""
        if int(n) <= 1:
            return True
        if self._multi_sample_mode == "expanded":
            return False
        if self._multi_sample_mode == "batched":
            return True
        return self._parallel_sampling_supported is not False

    def _mark_parallel_sampling_ok(self, n: int) -> None:
        if int(n) > 1 and self._multi_sample_mode == "auto":
            self._parallel_sampling_supported = True

    def _parallel_sampling_mismatch(
        self, *, context: str, expected: int, counts: Sequence[int],
    ) -> bool:
        """Handle a vLLM runtime that did not return ``expected`` completions.

        Returns True if callers should fall back to expanded requests. In
        explicit ``batched`` mode this raises instead, because falling back
        would hide that the requested optimized path is unavailable.
        """
        detail = ", ".join(str(int(c)) for c in counts[:8])
        if len(counts) > 8:
            detail += ", ..."
        msg = (
            f"vLLM parallel sampling mismatch in {context}: expected "
            f"{expected} completions/request, got counts [{detail}]."
        )
        if self._multi_sample_mode == "batched":
            raise RuntimeError(msg)
        self._parallel_sampling_supported = False
        if not self._parallel_sampling_warned:
            self._parallel_sampling_warned = True
            warnings.warn(
                msg + " Falling back to expanded one-request-per-sample mode."
            )
        return True

    def _parallel_sampling_exception(
        self, *, context: str, expected: int, error: BaseException,
    ) -> bool:
        msg = (
            f"vLLM parallel sampling failed in {context} for n={expected}: "
            f"{type(error).__name__}: {error}"
        )
        if self._multi_sample_mode == "batched":
            raise RuntimeError(msg) from error
        self._parallel_sampling_supported = False
        if not self._parallel_sampling_warned:
            self._parallel_sampling_warned = True
            warnings.warn(msg + " Falling back to expanded one-request-per-sample mode.")
        return True

    def _normalize_prefix_max_tokens(
        self,
        max_tokens: Optional[Union[int, Sequence[int]]],
        n_prefixes: int,
    ) -> List[int]:
        if max_tokens is None:
            return [int(self.cfg.max_response_len)] * n_prefixes
        if isinstance(max_tokens, int):
            return [max(1, int(max_tokens))] * n_prefixes
        if isinstance(max_tokens, (str, bytes)):
            raise TypeError("max_tokens must be an int or sequence of ints")
        values = [max(1, int(v)) for v in max_tokens]
        if len(values) != n_prefixes:
            raise ValueError(
                f"max_tokens sequence length {len(values)} does not match "
                f"prefix count {n_prefixes}"
            )
        return values

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
        response_ids: torch.Tensor,
        eos_id: int,
        pad_id: int,
        lengths: Optional[Sequence[int]] = None,
    ) -> torch.LongTensor:
        # 1 on actual generated tokens up to & including the first real EOS;
        # 0 after. Lengths are required when pad_id == eos_id so padded EOS
        # tokens are not mistaken for generated EOS.
        if lengths is not None:
            lens = torch.tensor(
                [max(0, int(v)) for v in lengths],
                dtype=torch.long,
                device=response_ids.device,
            ).clamp(max=response_ids.shape[1])
            arange_R = torch.arange(response_ids.shape[1], device=response_ids.device)
            valid_span = arange_R.unsqueeze(0) < lens.unsqueeze(1)
        else:
            valid_span = torch.ones_like(response_ids, dtype=torch.bool)
            if pad_id != eos_id:
                valid_span = response_ids != pad_id
        is_eos = response_ids == eos_id
        eos_in_span = is_eos & valid_span
        cum = eos_in_span.to(torch.long).cumsum(dim=1)
        first_eos = eos_in_span & (cum == 1)
        keep = valid_span & ((cum == 0) | first_eos)
        return keep.to(torch.long)

    # ------------------------------------------------------------------
    # Tokenization cache
    # ------------------------------------------------------------------

    def _tokenize_prompt_cached(self, prompt: str) -> List[int]:
        """Return token ids for ``prompt``, served from a small LRU dict.

        Saves ~1ms/step on the rollout hot path: the same prompt strings are
        re-tokenized many times across PPO epochs and VinePPO MC rollouts.
        Eviction is FIFO-on-insert (cheap on a CPython dict, ordering kept
        since 3.7) and the bound is small enough that the cache fits in L2.
        """
        ids = self._prompt_token_cache.get(prompt)
        if ids is not None:
            return ids
        enc = self.tokenizer(prompt, add_special_tokens=False)
        ids = list(enc["input_ids"])
        cache = self._prompt_token_cache
        if len(cache) >= self._prompt_token_cache_max:
            # Evict oldest entry (FIFO). dict insertion order is preserved.
            try:
                oldest = next(iter(cache))
                del cache[oldest]
            except StopIteration:
                pass
        cache[prompt] = ids
        return ids

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

        # Tokenize prompts (left-truncated to cfg.max_prompt_len). We serve
        # token-ids from a per-prompt LRU cache because the same prompt
        # strings recur across PPO epochs / iterations, then fall back to a
        # batched HF tokenizer call (Rust-parallel) for any cache misses.
        max_p = int(cfg.max_prompt_len) if cfg.max_prompt_len else 0
        prompt_token_ids: List[List[int]] = [None] * len(prompts)  # type: ignore[list-item]
        miss_indices: List[int] = []
        miss_prompts: List[str] = []
        for i, p in enumerate(prompts):
            cached = self._prompt_token_cache.get(p)
            if cached is not None:
                prompt_token_ids[i] = cached
            else:
                miss_indices.append(i)
                miss_prompts.append(p)
        if miss_prompts:
            enc = self.tokenizer(miss_prompts, add_special_tokens=False)
            for j, ids in zip(miss_indices, enc["input_ids"]):
                ids_list = list(ids)
                # Populate cache with the FULL (un-truncated) tokenization;
                # truncation happens just below and is a cheap slice.
                cache = self._prompt_token_cache
                if len(cache) >= self._prompt_token_cache_max:
                    try:
                        oldest = next(iter(cache))
                        del cache[oldest]
                    except StopIteration:
                        pass
                cache[prompts[j]] = ids_list
                prompt_token_ids[j] = ids_list
        # Apply per-call left-truncation; never mutate cached lists in place.
        if max_p:
            for i, ids in enumerate(prompt_token_ids):
                if len(ids) > max_p:
                    prompt_token_ids[i] = list(ids[-max_p:])
                else:
                    prompt_token_ids[i] = list(ids)
        else:
            prompt_token_ids = [list(ids) for ids in prompt_token_ids]

        from vllm.inputs import TokensPrompt
        from vllm.outputs import CompletionOutput as _CompCls  # type: ignore

        # Order matters: emit (prompt p, sample 0..G-1) contiguously so that
        # outputs[i*G + j] corresponds to (prompt i, sample j) and prompt_index
        # == [0]*G + [1]*G + ... matches HFRolloutSampler's layout.
        token_prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids]

        comps_flat = None
        if self._can_try_parallel_sampling(G):
            sp_batched = self._build_sampling_params(
                n=G, max_tokens=int(cfg.max_response_len),
            )
            requests = [
                (tp, sp_batched, f"req-p{prompt_i}-n{G}-{uuid.uuid4().hex}")
                for prompt_i, tp in enumerate(token_prompts)
            ]
            try:
                outputs = self._run(self._drain_many(requests))
                comps_by_prompt = [
                    list((req_out.outputs if req_out is not None else None) or [])
                    for req_out in outputs
                ]
                counts = [len(comps) for comps in comps_by_prompt]
                if all(c == G for c in counts):
                    self._mark_parallel_sampling_ok(G)
                    comps_flat = [
                        comp
                        for comps in comps_by_prompt
                        for comp in comps[:G]
                    ]
                else:
                    self._parallel_sampling_mismatch(
                        context="sample", expected=G, counts=counts,
                    )
            except Exception as e:
                self._parallel_sampling_exception(
                    context="sample", expected=G, error=e,
                )

        if comps_flat is None:
            sp = self._build_sampling_params(n=1, max_tokens=int(cfg.max_response_len))
            requests = []
            for prompt_i, tp in enumerate(token_prompts):
                for samp_j in range(G):
                    requests.append(
                        (tp, sp, f"req-p{prompt_i}-s{samp_j}-{uuid.uuid4().hex}")
                    )
            outputs = self._run(self._drain_many(requests))
            comps_flat = []
            for req_out in outputs:
                comps = list((req_out.outputs if req_out is not None else None) or [])
                if comps:
                    comps_flat.append(comps[0])
                else:
                    comps_flat.append(
                        _CompCls(  # type: ignore[call-arg]
                            index=0, text="", token_ids=[],
                            cumulative_logprob=None, logprobs=[],
                            finish_reason="error", stop_reason=None,
                        )
                    )

        # Each output has 1 CompletionOutput; flatten in submitted order.
        flat_token_ids: List[List[int]] = []
        flat_logprobs: List[List[float]] = []
        flat_responses: List[str] = []
        flat_lengths: List[int] = []
        for comp in comps_flat:
            gen = self._extract_completion(comp)
            flat_token_ids.append(gen.token_ids)
            flat_logprobs.append(gen.sampling_logprobs)
            flat_responses.append(gen.text)
            flat_lengths.append(len(gen.token_ids))

        B = num_prompts * G

        # Pack tensors. Prompts are left-padded; responses right-padded.
        prompt_ids_packed = self._pad_left(prompt_token_ids, self.pad_token_id)
        prompt_mask = torch.zeros_like(prompt_ids_packed, dtype=torch.long)
        P = prompt_ids_packed.shape[1]
        for i, ids in enumerate(prompt_token_ids):
            if ids:
                prompt_mask[i, P - len(ids) :] = 1

        response_ids = self._pad_right(flat_token_ids, self.pad_token_id)
        R = response_ids.shape[1] if response_ids.numel() > 0 else 0
        sampling_logprobs = self._pad_right_float(flat_logprobs, R)
        response_mask = self._build_response_mask(
            response_ids, self.eos_token_id, self.pad_token_id,
            lengths=flat_lengths,
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
        max_tokens: Optional[Union[int, Sequence[int]]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[List[_Generation]]:
        """K rollouts from each prefix; returns ``len(prefixes) × K`` generations.

        Used by VinePPO's MC value estimation. In ``vllm_multi_sample_mode=auto``
        this first tries one ``SamplingParams(n=K)`` request per prefix. If the
        installed vLLM runtime returns too few completions or errors on that
        path, the engine falls back to K independent ``n=1`` requests per prefix.
        Prefix caching is still enabled, so expanded requests share prefix KV.
        """
        if not prefix_token_ids_list:
            return []
        max_tokens_by_prefix = self._normalize_prefix_max_tokens(
            max_tokens, len(prefix_token_ids_list),
        )

        from vllm.inputs import TokensPrompt
        from vllm.outputs import CompletionOutput as _CompCls  # type: ignore

        K_int = int(K)
        token_prompts = [
            TokensPrompt(prompt_token_ids=list(p)) for p in prefix_token_ids_list
        ]
        sp_cache: dict[tuple[int, int], Any] = {}

        def _sp(n: int, mt: int):
            key = (int(n), int(mt))
            cached = sp_cache.get(key)
            if cached is None:
                cached = self._build_sampling_params(
                    n=int(n),
                    max_tokens=int(mt),
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                )
                sp_cache[key] = cached
            return cached

        comps_by_prefix = None
        if self._can_try_parallel_sampling(K_int):
            requests = [
                (
                    tp,
                    _sp(K_int, max_tokens_by_prefix[prefix_i]),
                    f"prefix-p{prefix_i}-n{K_int}-{uuid.uuid4().hex}",
                )
                for prefix_i, tp in enumerate(token_prompts)
            ]
            try:
                outputs = self._run(self._drain_many(requests))
                maybe = [
                    list((req_out.outputs if req_out is not None else None) or [])
                    for req_out in outputs
                ]
                counts = [len(comps) for comps in maybe]
                if all(c == K_int for c in counts):
                    self._mark_parallel_sampling_ok(K_int)
                    comps_by_prefix = [comps[:K_int] for comps in maybe]
                else:
                    self._parallel_sampling_mismatch(
                        context="sample_with_prefix", expected=K_int, counts=counts,
                    )
            except Exception as e:
                self._parallel_sampling_exception(
                    context="sample_with_prefix", expected=K_int, error=e,
                )

        if comps_by_prefix is None:
            requests = []
            for prefix_i, tp in enumerate(token_prompts):
                sp = _sp(1, max_tokens_by_prefix[prefix_i])
                for samp_j in range(K_int):
                    requests.append(
                        (tp, sp, f"prefix-p{prefix_i}-s{samp_j}-{uuid.uuid4().hex}")
                    )
            outputs = self._run(self._drain_many(requests))
            comps_by_prefix = []
            for prefix_i in range(len(prefix_token_ids_list)):
                slice_out = outputs[prefix_i * K_int : (prefix_i + 1) * K_int]
                comps: List[Any] = []
                for req_out in slice_out:
                    one = list((req_out.outputs if req_out is not None else None) or [])
                    if one:
                        comps.append(one[0])
                    else:
                        comps.append(
                            _CompCls(  # type: ignore[call-arg]
                                index=0, text="", token_ids=[],
                                cumulative_logprob=None, logprobs=[],
                                finish_reason="error", stop_reason=None,
                            )
                        )
                comps_by_prefix.append(comps)

        out_per_prefix: List[List[_Generation]] = []
        for comps in comps_by_prefix:
            gens: List[_Generation] = []
            for comp in comps:
                gens.append(self._extract_completion(comp))
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
        self._reset_prefix_cache_after_weight_update()
        return time.time() - t0

    def sync_weights_from_model(self, model: torch.nn.Module) -> float:
        """Push trainer weights to the vLLM engine.

        Two backends:

        * ``ipc`` (rank-local trainer + rank-local TP=1 vLLM on same GPU):
          uses CUDA IPC mem handles. No tensor data leaves the GPU; the
          worker just opens each handle and copies into its model.
        * ``nccl`` (disaggregated or colocated TP>1): broadcasts each
          named parameter from trainer rank 0 to the vLLM workers via
          a side ``PyNcclCommunicator`` group initialized in __init__.
          Each worker materializes the full tensor briefly, then
          ``load_weights`` slices it onto its TP-shard.
        """
        if self._weight_sync_backend == "nccl":
            return self._sync_weights_nccl(model)
        if self._weight_sync_backend != "ipc":
            raise RuntimeError(
                f"sync_weights_from_model: unsupported backend "
                f"{self._weight_sync_backend!r}"
            )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA IPC weight sync requires CUDA")

        import pickle

        import pybase64 as base64
        from torch.multiprocessing.reductions import reduce_tensor
        from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest

        t0 = time.time()
        device_index = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        gpu_uuid = str(props.uuid)

        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        ipc_handles = []
        keepalive: list[torch.Tensor] = []
        for name, tensor in model.named_parameters():
            if not tensor.is_cuda:
                raise RuntimeError(
                    f"parameter {name!r} is on {tensor.device}; IPC sync expects CUDA"
                )
            weight = tensor.detach().contiguous()
            keepalive.append(weight)
            names.append(name)
            dtype_names.append(str(weight.dtype).split(".")[-1])
            shapes.append(list(weight.shape))
            ipc_handles.append({gpu_uuid: reduce_tensor(weight)})

        pickled_handles = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")
        request = WeightTransferUpdateRequest(
            update_info={
                "names": names,
                "dtype_names": dtype_names,
                "shapes": shapes,
                "ipc_handles_pickled": pickled_handles,
                "is_checkpoint_format": True,
            }
        )
        # Keep contiguous tensors alive until EngineCore has opened every IPC
        # handle and copied/loaded the weights.
        self._ipc_weight_keepalive = keepalive
        try:
            self._run(self.engine.update_weights(request))
        finally:
            self._ipc_weight_keepalive = []
        self._reset_prefix_cache_after_weight_update()
        return time.time() - t0

    def _sync_weights_from_aggregated_ipc(
        self,
        *,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        aggregated_handles: list[dict[str, Any]],
    ) -> float:
        """Submit pre-aggregated multi-GPU IPC handles to vLLM.

        Used by ``DisaggregatedSamplerProxy._sync_weights_multirank_ipc``
        for the Phase F colocated-TP=N path. The proxy gathers per-rank
        handles ({uuid_N: handle_N}) into per-param multi-UUID dicts
        ({uuid_0: h0, ..., uuid_{N-1}: h_{N-1}}) before calling here;
        we just pickle them and dispatch via ``update_weights``.

        Each vLLM TP-worker on GPU N looks up its own UUID in the
        dict, opens its same-GPU IPC handle, and ``load_weights``
        slices the full tensor onto its TP-shard. Verified against
        vllm/distributed/weight_transfer/ipc_engine.py:172-191
        (current_device_index → physical UUID → handle lookup).

        The caller is responsible for keeping all source-side tensors
        alive (via Python refs on every trainer rank) until this call
        returns. Only rank 0 reaches this code; the caller arranges a
        ``dist.barrier()`` afterward so non-rank-0 keepalive persists
        until vLLM is done copying.
        """
        if self._weight_sync_backend != "ipc":
            raise RuntimeError(
                f"_sync_weights_from_aggregated_ipc requires "
                f"vllm_weight_sync_backend='ipc', got "
                f"{self._weight_sync_backend!r}"
            )
        import pickle

        import pybase64 as base64
        from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest

        if not (len(names) == len(dtype_names) == len(shapes) == len(aggregated_handles)):
            raise ValueError(
                f"length mismatch: names={len(names)} dtypes={len(dtype_names)} "
                f"shapes={len(shapes)} handles={len(aggregated_handles)}"
            )
        pickled_handles = base64.b64encode(
            pickle.dumps(aggregated_handles)
        ).decode("utf-8")
        request = WeightTransferUpdateRequest(
            update_info={
                "names": list(names),
                "dtype_names": list(dtype_names),
                "shapes": list(shapes),
                "ipc_handles_pickled": pickled_handles,
                "is_checkpoint_format": True,
            }
        )
        # Note: we don't keepalive here — the proxy holds keepalive on
        # every rank around our call (synchronized via dist.barrier()).
        self._run(self.engine.update_weights(request))
        self._reset_prefix_cache_after_weight_update()
        return 0.0

    def _reset_prefix_cache_after_weight_update(self) -> None:
        # reset_prefix_cache is async on V1 engine — drive it on our event loop.
        # Strictly AFTER the weight update so cached KV from old weights cannot
        # be reused by subsequent rollouts.
        try:
            reset_coro = self.engine.reset_prefix_cache()
            if hasattr(reset_coro, "__await__"):
                reset_ok = self._run(reset_coro)
            else:
                reset_ok = reset_coro
            if reset_ok is False:
                raise RuntimeError("reset_prefix_cache returned False")
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "reset_prefix_cache failed after vLLM weight update; "
                "continuing could reuse stale KV cache from old weights"
            ) from e

    async def _collective_rpc(self, method: str, args: tuple = (), kwargs: Optional[dict] = None):
        kwargs = kwargs or {}
        return await self.engine.collective_rpc(method, args=args, kwargs=kwargs)

    # ------------------------------------------------------------------
    # NCCL weight sync (used by disaggregated + colocated-TP topologies)
    # ------------------------------------------------------------------
    def _sync_weights_nccl(self, model: torch.nn.Module) -> float:
        """Broadcast every named parameter from trainer rank 0 to all
        vLLM workers via the side PyNcclCommunicator group.

        Two phases run concurrently:
        * vLLM workers receive ``update_weights`` RPC (with metadata-only
          update_info — names/dtypes/shapes). For each (name, shape,
          dtype) in order they allocate a CUDA buffer, call
          ``model_update_group.broadcast(buffer, src=0)``, and pass the
          received tensor through ``load_weights`` (which internally
          slices it onto the worker's TP-shard).
        * Trainer rank 0 calls ``trainer_send_weights(iter, args)``,
          which iterates the same ordered (name, tensor) list and
          broadcasts each tensor as src=0.

        Both sides must iterate ``names`` in the SAME order; we use
        ``model.named_parameters()`` on both ends.
        """
        if self._nccl_weight_sync_group is None:
            raise RuntimeError(
                "NCCL weight sync requested but the side group was not "
                "initialized. Did the launcher set "
                "CASPO_WEIGHT_SYNC_MASTER_PORT?"
            )
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLTrainerSendWeightsArgs,
            NCCLWeightTransferEngine,
        )
        from vllm.distributed.weight_transfer.base import WeightTransferUpdateRequest
        import threading

        t0 = time.time()
        named = list(model.named_parameters())
        if not named:
            return 0.0
        # Collect metadata in iteration order (same order vLLM workers
        # will use — both ends consume model.named_parameters identically).
        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        for name, tensor in named:
            if not tensor.is_cuda:
                raise RuntimeError(
                    f"parameter {name!r} is on {tensor.device}; NCCL sync "
                    f"expects CUDA tensors on rank 0"
                )
            names.append(name)
            dtype_names.append(str(tensor.dtype).split(".")[-1])
            shapes.append(list(tensor.shape))

        update_info = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "packed": True,
        }
        request = WeightTransferUpdateRequest(update_info=update_info)

        # Trainer-side broadcast runs on a worker thread; the main
        # thread drives the RPC's asyncio future. Both must run in
        # parallel because each side blocks at every NCCL collective.
        err: dict = {}

        def _trainer_thread():
            try:
                args_obj = NCCLTrainerSendWeightsArgs(
                    group=self._nccl_weight_sync_group,
                    src=0,
                    packed=True,
                )
                NCCLWeightTransferEngine.trainer_send_weights(
                    iter(named), args_obj,
                )
            except Exception as e:  # noqa: BLE001
                err["err"] = e

        t = threading.Thread(target=_trainer_thread, daemon=True)
        t.start()
        try:
            self._run(self.engine.update_weights(request))
        finally:
            t.join(timeout=300.0)
        if t.is_alive():
            raise RuntimeError(
                "NCCL weight-sync trainer thread did not finish within 300 s"
            )
        if "err" in err:
            raise err["err"]

        self._reset_prefix_cache_after_weight_update()
        return time.time() - t0

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
