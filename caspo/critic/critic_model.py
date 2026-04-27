"""Critic model: HuggingFace LM backbone + 1-D linear value head.

Mirrors VinePPO's PPO baseline (which uses a separate
``PreTrainedModel``). We share the policy's tokenizer + chat template
so prompt formatting is identical; the value head is initialized
to small-Gaussian weights so the early-training value predictions
don't dominate the GAE returns.

The critic is FSDP-wrappable on the same path as the policy
(``CASPOTrainer._wrap_fsdp_if_enabled``); see
``critic_share_fsdp_policy`` in cfg.

Memory cost at 7B bf16: ~14 GB params + ~84 GB Adam (fp32 m/v +
master). Sharded across FSDP=4 → 3.5 GB params + 21 GB Adam per
rank. Doubles the trainer-side memory vs critic-free PPO.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


class CriticModel(nn.Module):
    """Backbone (frozen at the LM head) + scalar value head.

    Forward returns ``[B, S]`` per-token values. The trainer slices
    out the response window before passing to GAE.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        # Strip the LM head (we don't use it). Save peak memory by
        # not keeping a vocab-sized projection alive on every GPU.
        if hasattr(self.backbone, "lm_head"):
            try:
                del self.backbone.lm_head
            except (AttributeError, RuntimeError):
                pass
            # Some HF models gate the deletion via a registered
            # _no_split_modules / tied-weight setup; if del fails,
            # replace with an identity-equivalent.
            self.backbone.lm_head = nn.Identity()
        # Value head: small-init linear so early predictions are ~0.
        # CRITICAL: cast to the backbone's parameter dtype before
        # constructing. FSDP refuses to flatten mixed-dtype params
        # ("Must flatten tensors with uniform dtype"); without this
        # the value head defaults to fp32 while the backbone is bf16
        # and the wrap fails. We do the float→target-dtype cast on
        # the parameters AFTER init so the random weights still come
        # from a fp32-precision generator.
        backbone_dtype = next(self.backbone.parameters()).dtype
        self.value_head = nn.Linear(hidden_size, 1, bias=True)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.value_head.bias)
        self.value_head = self.value_head.to(dtype=backbone_dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Returns ``[B, S]`` per-token values.

        Caller must provide ``input_ids`` and ``attention_mask``
        in the SAME shape used for the policy forward (so values
        line up with response_ids slicing in the trainer).
        """
        # CRITICAL: bypass the LM-head wrapper and call the encoder
        # directly. Calling ``self.backbone(..., output_hidden_states
        # =True)`` would force HF to retain ALL 33 layers' hidden
        # states in the output dict (input embedding + 32 layer
        # outputs), spiking activation memory by ~32× and OOMing on
        # 7B + colocated policy. By calling the inner model
        # (``self.backbone.model`` for LlamaForCausalLM-style
        # architectures) we get just the final hidden state without
        # the per-layer retention.
        encoder = getattr(self.backbone, "model", self.backbone)
        out = encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            **{k: v for k, v in kwargs.items()
               if k not in ("output_hidden_states", "use_cache", "return_dict")},
        )
        # ``out.last_hidden_state`` is [B, S, H] in the backbone's
        # compute dtype.
        h = out.last_hidden_state
        # Project to scalar values, then squeeze the trailing 1-dim.
        # Cast to fp32 for downstream MSE / GAE numerics; the value head
        # weights themselves stay in the backbone's dtype.
        v = self.value_head(h).squeeze(-1)
        return v.to(torch.float32)

    @classmethod
    def from_pretrained(
        cls,
        cfg: Any,
        path: Optional[str] = None,
        *,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "CriticModel":
        """Construct a critic from an HF checkpoint dir.

        ``path=None`` → uses ``cfg.model_name_or_path`` (so the critic
        starts from the same SFT checkpoint as the policy, which
        matches VinePPO upstream's pattern).
        """
        from transformers import AutoModelForCausalLM, AutoConfig

        if path is None:
            path = cfg.model_name_or_path
        if torch_dtype is None:
            # Inline dtype resolution — mirrors trainer's _resolve_dtype.
            _dtype_map = {
                "float32": torch.float32, "fp32": torch.float32, "float": torch.float32,
                "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            }
            torch_dtype = _dtype_map.get(str(cfg.torch_dtype).lower(), torch.bfloat16)
        hf_cfg = AutoConfig.from_pretrained(
            path, trust_remote_code=cfg.trust_remote_code,
        )
        backbone = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            attn_implementation=cfg.attn_implementation,
            trust_remote_code=cfg.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        try:
            backbone.config.use_cache = False
        except AttributeError:
            pass
        # Mirror the policy's activation checkpointing: at 7B with
        # critic colocated on trainer GPUs, the second forward+backward
        # through a 7B backbone in the policy mb loop triples the
        # activation peak and OOMs at 80 GB. Enable HF's grad-ckpt path
        # on the critic backbone so each block recomputes on backward.
        # The cost is ~30% extra backward FLOPs on the critic, which is
        # already the smaller cost vs the policy backward.
        if getattr(cfg, "use_gradient_checkpointing", False):
            try:
                backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            except (AttributeError, TypeError):
                # Some HF versions don't accept kwargs; fall through.
                try:
                    backbone.gradient_checkpointing_enable()
                except AttributeError:
                    pass
        hidden_size = int(getattr(hf_cfg, "hidden_size", 0))
        if hidden_size <= 0:
            raise RuntimeError(
                f"could not infer hidden_size from {path}'s config — "
                f"got {hidden_size}"
            )
        return cls(backbone=backbone, hidden_size=hidden_size)
