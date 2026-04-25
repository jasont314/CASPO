"""Implicit prefix value model (IPVRM).

Implements Eq. 5 of arXiv:2604.13197 "Unleashing Implicit Rewards: Prefix-Value
Learning for Distribution-Level Optimization" (Gao et al. 2026):

    V_phi(s_t) = beta * sum_{i=1}^{t-1} log( pi_phi(y_i | s_i) / pi_ref(y_i | s_i) )

There is **no scalar value head**.  The "value" is a cumulative beta-scaled
log-ratio between a trainable LM ``pi_phi`` (initialised from SFT) and a frozen
reference LM ``pi_ref`` (the same SFT init).  A single forward of each LM gives
all per-token log-ratios; a cumsum gives ``V`` at every prefix.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from caspo.config import CASPOConfig


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(f"unknown torch_dtype {name!r}")
    return _DTYPE_MAP[name]


def compute_log_ratio(
    pi_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Return ``beta * (log pi_phi - log pi_ref)`` masked to the response.

    Args:
        pi_logprobs:   ``[B, R]`` log pi_phi(y_t | s_t).
        ref_logprobs:  ``[B, R]`` log pi_ref(y_t | s_t).
        response_mask: ``[B, R]`` 1 on response tokens, 0 on padding.
        beta:          scalar in Eq. 5.

    Returns:
        ``[B, R]`` tensor; tokens outside the response are exactly ``0`` so
        that ``cumsum`` over the time axis is well-defined.
    """
    pi = pi_logprobs.float()
    ref = ref_logprobs.float()
    out = beta * (pi - ref)
    out = out * response_mask.to(out.dtype)
    return out


class PrefixValueModel(nn.Module):
    """Two HF causal LMs: trainable ``phi`` and frozen ``ref``.

    The forward pass returns per-token beta-scaled log-ratios over the response
    plus the cumulative-sum value ``V[:, t] = sum_{i<t} log_ratio[:, i]`` with
    ``V[:, 0] = 0``.
    """

    def __init__(
        self,
        cfg: CASPOConfig,
        *,
        ref_model_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self._beta = float(cfg.value_beta)
        self._margin = float(cfg.value_margin)
        self._ref_model_path = ref_model_path or cfg.model_name_or_path
        self._phi_model_path = cfg.model_name_or_path

        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

        torch_dtype = _resolve_dtype(cfg.torch_dtype)
        model_kwargs = dict(
            torch_dtype=torch_dtype,
            trust_remote_code=cfg.trust_remote_code,
        )
        if cfg.attn_implementation:
            model_kwargs["attn_implementation"] = cfg.attn_implementation

        # phi: trainable copy.  Always re-load (do NOT weight-tie with ref).
        self.phi = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path, **model_kwargs
        )
        try:
            self.phi.config.use_cache = False
        except AttributeError:
            pass
        if cfg.use_gradient_checkpointing:
            try:
                self.phi.gradient_checkpointing_enable()
            except Exception as e:  # pragma: no cover
                warnings.warn(f"could not enable gradient checkpointing on phi: {e}")

        # ref: frozen, separate load.
        self.ref = AutoModelForCausalLM.from_pretrained(
            self._ref_model_path, **model_kwargs
        )
        for p in self.ref.parameters():
            p.requires_grad_(False)
        # Buffers don't track grad by default but defend against any that do.
        for b in self.ref.buffers():
            if b.is_floating_point() and b.requires_grad:
                b.requires_grad_(False)
        try:
            self.ref.config.use_cache = False
        except AttributeError:
            pass
        self.ref.eval()

        tok_path = cfg.tokenizer_name_or_path or cfg.model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tok_path, trust_remote_code=cfg.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ------------------------------------------------------------------ props
    @property
    def beta(self) -> float:
        return self._beta

    @property
    def margin(self) -> float:
        return self._margin

    @property
    def device(self) -> torch.device:
        return next(self.phi.parameters()).device

    def train(self, mode: bool = True) -> "PrefixValueModel":
        """Override ``Module.train`` so the frozen ``ref`` network is never
        flipped into train mode (which would re-enable dropout / BN updates
        on the reference and silently corrupt log-ratios).
        """
        super().train(mode)
        self.ref.eval()
        return self

    # ------------------------------------------------------------------ ref sharing
    def share_ref(self, ref_module: nn.Module) -> None:
        """Replace the internally loaded ``self.ref`` with an externally owned
        frozen reference LM, freeing the duplicate copy's GPU memory.

        Opt-in: callers MUST verify that ``ref_module`` is loaded from the
        same checkpoint as ``self._ref_model_path`` with the same dtype /
        attention implementation / trust_remote_code flags. The shared module
        is set to ``eval()`` and all parameters/floating-point buffers are
        marked ``requires_grad_(False)`` defensively. ``use_cache`` is
        disabled to mirror the standalone-loaded ref's configuration.

        After this call, ``self.ref`` is no longer registered as a child
        module (we use ``object.__setattr__`` to bypass ``nn.Module``'s
        registration), so it is NOT included in ``self.parameters()``,
        ``state_dict()``, or recursive ``.train()`` calls. This preserves the
        save_pretrained roundtrip (which only saves ``phi`` weights and
        meta) and prevents the external ref from being moved/cast by the
        owning module's ``.to(...)`` calls — the caller is responsible for
        device/dtype placement of the shared ref.
        """
        # Free the old standalone copy before swapping in the shared one.
        old_ref = getattr(self, "ref", None)
        # Defensive: freeze + eval the externally provided ref.
        for p in ref_module.parameters():
            p.requires_grad_(False)
        for b in ref_module.buffers():
            if b.is_floating_point() and b.requires_grad:
                b.requires_grad_(False)
        try:
            ref_module.config.use_cache = False
        except AttributeError:
            pass
        ref_module.eval()

        # Unregister the old ref from this Module (so .parameters(),
        # state_dict(), recursive .train(), .to() do not touch the shared
        # ref) and store it as a plain attribute. We delete first to remove
        # it from _modules, then bypass nn.Module.__setattr__ on assignment
        # so the new ref is NOT re-registered as a submodule.
        if old_ref is not None and "ref" in self._modules:
            del self._modules["ref"]
        object.__setattr__(self, "ref", ref_module)

        # Drop our reference to the old ref so Python/CUDA can reclaim it.
        if old_ref is not None and old_ref is not ref_module:
            del old_ref

    # ------------------------------------------------------------------ core
    @staticmethod
    def _forward_response_logits(
        model: nn.Module,
        full_ids: torch.Tensor,
        full_mask: torch.Tensor,
        P: int,
        R: int,
    ) -> torch.Tensor:
        """Return only logits needed to score response tokens.

        Transformers 5.x causal LMs support tensor-valued ``logits_to_keep``.
        That lets us compute the LM head only for positions ``P-1..P+R-2``
        instead of materializing prompt logits. Older/custom models fall back
        once and cache the fallback mode on the module.
        """
        mode = getattr(model, "_caspo_logits_to_keep_mode", None)
        if mode != "full" and P > 0 and R > 0:
            idx = torch.arange(P - 1, P - 1 + R, device=full_ids.device)
            try:
                out = model(
                    input_ids=full_ids,
                    attention_mask=full_mask,
                    use_cache=False,
                    logits_to_keep=idx,
                )
                logits = out.logits
                if logits.shape[1] == R:
                    setattr(model, "_caspo_logits_to_keep_mode", "tensor")
                    return logits
            except (TypeError, ValueError, IndexError):
                setattr(model, "_caspo_logits_to_keep_mode", "full")

        out = model(input_ids=full_ids, attention_mask=full_mask, use_cache=False)
        return out.logits[:, P - 1 : P - 1 + R, :]

    @staticmethod
    def _gather_response_logprobs(
        model: nn.Module,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Score ``response_ids`` under ``model``, returning ``[B, R]`` logprobs.

        For sequence ``[prompt | response]`` the per-token logits at position
        ``i`` predict token ``i + 1``; logits at ``[P-1 : P-1+R]`` therefore
        score ``response_ids[:, 0..R-1]`` (same trick as LatEntRL).

        Uses ``F.cross_entropy`` (fused log_softmax + nll, negated) instead of
        ``log_softmax + gather`` to avoid materialising the [B, R, V] log-prob
        tensor and to dispatch to a fused CUDA kernel.
        """
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        full_mask = torch.cat([prompt_mask, response_mask], dim=1)
        P = prompt_ids.shape[1]
        R = response_ids.shape[1]
        sliced = PrefixValueModel._forward_response_logits(
            model, full_ids, full_mask, P, R,
        )
        # Fused log_softmax + gather via cross_entropy(reduction='none').
        # cross_entropy returns -log p(y); negate to recover log p(y).
        # Upcast to float32 for numerical stability of the softmax (matches
        # the prior log_softmax(.float()) behaviour).
        B, R_, V = sliced.shape
        nll = F.cross_entropy(
            sliced.reshape(B * R_, V).float(),
            response_ids.reshape(B * R_),
            reduction="none",
        )
        return (-nll).view(B, R_)

    def forward(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        response_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> dict:
        """Run both LMs and assemble per-prefix ``V``.

        Returns a dict with
            ``log_ratio``     ``[B, R]`` beta * (log pi_phi - log pi_ref), masked
            ``V``             ``[B, R+1]`` cumulative; ``V[:, 0] = 0``
            ``phi_logprobs``  ``[B, R]`` log pi_phi on response tokens
            ``ref_logprobs``  ``[B, R]`` log pi_ref on response tokens (no grad)
        """
        phi_logprobs = self._gather_response_logprobs(
            self.phi, prompt_ids, prompt_mask, response_ids, response_mask
        )

        # Force ref to eval() inside forward — defends against an outer
        # `model.train()` call (Module.train() recurses into all submodules,
        # including frozen ref) re-enabling dropout on the reference network.
        self.ref.eval()
        with torch.no_grad():
            ref_logprobs = self._gather_response_logprobs(
                self.ref, prompt_ids, prompt_mask, response_ids, response_mask
            )
            ref_logprobs = ref_logprobs.detach()

        # mask logprobs *before* the subtraction in compute_log_ratio so that
        # padded positions contribute exactly zero.
        mask_f = response_mask.to(phi_logprobs.dtype)
        phi_logprobs = phi_logprobs * mask_f
        ref_logprobs = ref_logprobs * mask_f

        log_ratio = compute_log_ratio(
            phi_logprobs, ref_logprobs, response_mask, self._beta
        )  # [B, R]

        # V[:, t] = sum_{i < t} log_ratio[:, i] with V[:, 0] = 0.
        B, R = log_ratio.shape
        zero = log_ratio.new_zeros(B, 1)
        V = torch.cat([zero, torch.cumsum(log_ratio, dim=1)], dim=1)  # [B, R+1]

        return {
            "log_ratio": log_ratio,
            "V": V,
            "phi_logprobs": phi_logprobs,
            "ref_logprobs": ref_logprobs,
        }

    # ------------------------------------------------------------------ io
    def save_pretrained(self, path: str) -> None:
        """Save only ``phi`` plus a small metadata file pointing to the ref."""
        os.makedirs(path, exist_ok=True)
        self.phi.save_pretrained(path)
        try:
            self.tokenizer.save_pretrained(path)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"tokenizer save_pretrained failed: {e}")
        meta = {
            "ref_model_path": self._ref_model_path,
            "model_name_or_path": self._phi_model_path,
            "beta": self._beta,
            "margin": self._margin,
        }
        with open(os.path.join(path, "caspo_value_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def from_pretrained(cls, cfg: CASPOConfig, path: str) -> "PrefixValueModel":
        """Reload ``phi`` from ``path`` and ``ref`` from the meta file (or cfg)."""
        meta_path = os.path.join(path, "caspo_value_meta.json")
        ref_model_path: Optional[str] = None
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            ref_model_path = meta.get("ref_model_path")
        ref_model_path = ref_model_path or cfg.model_name_or_path

        # We point cfg.model_name_or_path at the saved phi for construction,
        # while keeping a separate ref_model_path for the frozen network.
        from dataclasses import replace

        sub_cfg = replace(cfg, model_name_or_path=path)
        return cls(sub_cfg, ref_model_path=ref_model_path)
