"""Disaggregated rollout proxy.

When ``cfg.vllm_disaggregated=True`` the trainer runs FSDP across one
GPU set and vLLM runs TP across a *disjoint* set. Only rank 0 of the
trainer instantiates the actual ``VLLMRolloutEngine`` (because it
spawns vLLM workers pinned to the rollout GPUs). All other trainer
ranks call into ``DisaggregatedSamplerProxy``, which:

* gathers their per-rank example list to rank 0;
* on rank 0, flattens the per-rank lists in rank order, runs the real
  sampler once on the concatenated batch, then slices the result back
  into per-rank ``RolloutBatch`` pieces;
* scatters the per-rank slices back over the FSDP process group.

The same pattern handles ``sample_with_prefix`` (VinePPO MC).

Why object-pickled scatter (vs tensor broadcast): the payload per
outer step is at most ~1 GB (64 prompts × G=8 × 1024 tokens × bf16),
which (de)serializes in well under 1 s — negligible against the
~200-300 s step times this design is built for. Pickle keeps the
proxy backend-agnostic; we don't have to mirror every tensor layout.
"""

from __future__ import annotations

import dataclasses
from typing import Any, List, Optional, Sequence

import torch

from caspo.rollout.sampler import RolloutBatch


# ---------------------------------------------------------------------------
# Slice helpers
# ---------------------------------------------------------------------------

def _slice_rollout_batch(batch: RolloutBatch, prompt_sizes: Sequence[int]) -> List[RolloutBatch]:
    """Cut a single RolloutBatch into per-rank slices.

    ``prompt_sizes[i]`` is the number of prompts that belonged to rank i
    in the original gather; the sum must equal ``len(batch.raw_prompts)``.

    The split is along the prompt axis — per-prompt response rows
    (``response_ids`` / ``response_mask`` / ``sampling_logprobs`` /
    ``rewards`` / ``prompt_index`` / ``raw_responses``) are stride-G
    grouped relative to that. ``prompt_index`` is re-zeroed inside each
    slice so downstream code (``tiled_prompt_ids = prompt_ids[prompt_index]``)
    still works without knowing it was sliced.
    """
    n_prompts = len(batch.raw_prompts)
    if sum(prompt_sizes) != n_prompts:
        raise ValueError(
            f"prompt_sizes sum={sum(prompt_sizes)} != batch.num_prompts={n_prompts}"
        )
    G = batch.response_ids.shape[0] // max(1, n_prompts)
    if G * n_prompts != batch.response_ids.shape[0]:
        raise ValueError(
            f"response_ids row count {batch.response_ids.shape[0]} not divisible "
            f"by num_prompts {n_prompts}"
        )

    out: List[RolloutBatch] = []
    p_lo = 0
    for n in prompt_sizes:
        p_hi = p_lo + n
        r_lo, r_hi = p_lo * G, p_hi * G
        # Re-zero prompt_index relative to this slice's start.
        sliced_prompt_index = batch.prompt_index[r_lo:r_hi].clone() - p_lo
        out.append(
            RolloutBatch(
                prompt_ids=batch.prompt_ids[p_lo:p_hi].clone(),
                prompt_mask=batch.prompt_mask[p_lo:p_hi].clone(),
                response_ids=batch.response_ids[r_lo:r_hi].clone(),
                response_mask=batch.response_mask[r_lo:r_hi].clone(),
                sampling_logprobs=batch.sampling_logprobs[r_lo:r_hi].clone(),
                rewards=batch.rewards[r_lo:r_hi].clone(),
                prompt_index=sliced_prompt_index,
                raw_prompts=list(batch.raw_prompts[p_lo:p_hi]),
                raw_responses=list(batch.raw_responses[r_lo:r_hi]),
                ground_truths=list(batch.ground_truths[p_lo:p_hi]),
            )
        )
        p_lo = p_hi
    return out


# ---------------------------------------------------------------------------
# Proxy
# ---------------------------------------------------------------------------

class DisaggregatedSamplerProxy:
    """Thin wrapper that turns a rank-0-only VLLMRolloutEngine into a
    same-shape sampler usable by every rank.

    Construction:

    * On rank 0: pass the real ``VLLMRolloutEngine`` as ``inner``.
    * On other ranks: pass ``inner=None``.

    The proxy mirrors the public methods used by the trainer
    (``sample``, ``sample_with_prefix``, ``sync_weights_*``). For
    weight sync the proxy delegates to the inner engine on rank 0;
    other ranks no-op (the NCCL weight-sync engine on the trainer
    side joins the side group, which lives separately from FSDP's PG).
    """

    def __init__(self, inner: Optional[Any], dist_info: Any) -> None:
        self.inner = inner
        self.dist = dist_info
        if dist_info.is_main and inner is None:
            raise RuntimeError(
                "DisaggregatedSamplerProxy on rank 0 requires a real inner sampler"
            )
        if (not dist_info.is_main) and inner is not None:
            raise RuntimeError(
                "DisaggregatedSamplerProxy on non-rank-0 must not have an inner sampler"
            )

    # ------------------------------------------------------------------
    def _gather_objects(self, obj: Any) -> Optional[List[Any]]:
        """All-ranks → rank 0 gather. Returns the gathered list on rank 0,
        ``None`` on other ranks. Caller must not mutate the input mid-gather."""
        import torch.distributed as dist

        gather_list: Optional[List[Any]] = (
            [None] * self.dist.world_size if self.dist.is_main else None
        )
        dist.gather_object(obj, object_gather_list=gather_list, dst=0)
        return gather_list

    def _scatter_objects(self, obj_list: Optional[List[Any]]) -> Any:
        """Rank 0 → all ranks scatter. ``obj_list`` is a list of length
        world_size on rank 0, ignored elsewhere. Returns this rank's slot."""
        import torch.distributed as dist

        out: List[Any] = [None]
        dist.scatter_object_list(out, obj_list, src=0)
        return out[0]

    # ------------------------------------------------------------------
    # sample(examples) — main rollout
    # ------------------------------------------------------------------
    def sample(self, examples: List[dict]) -> RolloutBatch:
        gathered = self._gather_objects(examples)

        if self.dist.is_main:
            assert gathered is not None
            flat: List[dict] = []
            sizes: List[int] = []
            for sub in gathered:
                sub_list = list(sub) if sub is not None else []
                sizes.append(len(sub_list))
                flat.extend(sub_list)
            full_batch = self.inner.sample(flat)
            per_rank: List[Any] = list(_slice_rollout_batch(full_batch, sizes))
        else:
            per_rank = [None] * self.dist.world_size  # type: ignore[list-item]

        return self._scatter_objects(per_rank if self.dist.is_main else None)

    # ------------------------------------------------------------------
    # sample_with_prefix — VinePPO MC value pass
    # ------------------------------------------------------------------
    def sample_with_prefix(
        self,
        prefix_token_ids_list: List[List[int]],
        K: int,
        *,
        max_tokens: Any = None,
        temperature: Any = None,
        top_p: Any = None,
        seed: Any = None,
    ) -> List[List[Any]]:
        # Per-rank inputs: list of prefix-id lists, plus the optional
        # max_tokens (which can be a list aligned with prefixes).
        local = {
            "prefix_token_ids_list": prefix_token_ids_list,
            "max_tokens": max_tokens,
            "K": int(K),
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        }
        gathered = self._gather_objects(local)

        if self.dist.is_main:
            assert gathered is not None
            flat_prefixes: List[List[int]] = []
            flat_maxtok: List[Any] = []
            sizes: List[int] = []
            for sub in gathered:
                pfx = list(sub["prefix_token_ids_list"]) if sub else []
                sizes.append(len(pfx))
                flat_prefixes.extend(pfx)
                mt = sub.get("max_tokens") if sub else None
                if isinstance(mt, (list, tuple)):
                    if len(mt) != len(pfx):
                        raise ValueError(
                            f"max_tokens length {len(mt)} != prefix count {len(pfx)} "
                            f"on a rank — caller must supply scalar or per-prefix list"
                        )
                    flat_maxtok.extend(list(mt))
                else:
                    # Scalar (or None): replicate per prefix so we can re-split below.
                    flat_maxtok.extend([mt] * len(pfx))
            flat_full = self.inner.sample_with_prefix(
                flat_prefixes, K,
                max_tokens=flat_maxtok if any(x is not None for x in flat_maxtok) else None,
                temperature=temperature, top_p=top_p, seed=seed,
            )
            # Slice [num_prefixes][K] -> per-rank
            per_rank: List[Any] = []
            i_lo = 0
            for n in sizes:
                per_rank.append(flat_full[i_lo:i_lo + n])
                i_lo += n
        else:
            per_rank = [None] * self.dist.world_size  # type: ignore[list-item]

        return self._scatter_objects(per_rank if self.dist.is_main else None)

    # ------------------------------------------------------------------
    # Weight sync — only rank 0's inner engine has a real path.
    # ------------------------------------------------------------------
    def sync_weights_from_model(self, model: Any) -> Any:
        if self.dist.is_main:
            return self.inner.sync_weights_from_model(model)
        # On other ranks the side NCCL weight-sync group is initialized
        # but only rank 0 is the producer. Other ranks return 0.0 wall.
        return 0.0

    def sync_weights_from_path(self, path: str) -> Any:
        if self.dist.is_main:
            return self.inner.sync_weights_from_path(path)
        return 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        if self.dist.is_main and self.inner is not None and hasattr(self.inner, "shutdown"):
            self.inner.shutdown()
