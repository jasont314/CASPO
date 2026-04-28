"""Public data loaders for CASPO.

The standardized loader output (a list of ``{prompt, ground_truth, raw_question}``
dicts) is cached to disk on cold start so subsequent training jobs that share
the same (tokenizer, dataset, max_prompt_len, prompt_template) tuple skip the
HF download + tokenize + length-filter pass entirely.

The cache key matches the four inputs that materially affect the rendered
prompts; any change there gets a different file. The location follows
``${HF_HOME:-/mnt/nvme_tmp/jason_caspo/hf_cache}/caspo_dataset_cache/<hash>.pt``
unless ``CASPO_DATASET_CACHE_DISABLE=1`` is set in the environment, in which
case both read and write paths are skipped.

The eval loader is intentionally *not* cached: eval runs use small datasets,
the savings are negligible, and a stale eval cache is much harder to detect
in practice than a stale train cache.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional

import torch

from caspo.data.cached_dataset import _HASH_LEN
from caspo.data.math_data import (
    _build_from_rows,
    _load_hf,
    format_prompt,
    load_eval_dataset,
)
from caspo.data.math_data import load_train_dataset as _load_train_dataset_uncached


__all__ = ["load_train_dataset", "load_eval_dataset", "format_prompt"]


# Cache format version. Bump when the on-disk dict layout changes — old caches
# are then ignored and rebuilt rather than producing garbage rows downstream.
_DATASET_CACHE_VERSION = 1
_DEFAULT_HF_HOME = "/mnt/nvme_tmp/jason_caspo/hf_cache"
_CACHE_SUBDIR = "caspo_dataset_cache"
_DISABLE_ENV = "CASPO_DATASET_CACHE_DISABLE"


def _cache_disabled() -> bool:
    """Honor ``CASPO_DATASET_CACHE_DISABLE=1`` — opt-out for debugging /
    A-B comparisons against an uncached run."""
    val = os.environ.get(_DISABLE_ENV, "")
    return val.strip() in ("1", "true", "True", "TRUE", "yes")


def _resolve_cache_dir() -> Path:
    """Resolve the cache directory.

    Order: ``CASPO_DATASET_CACHE_DIR`` (explicit override) > ``HF_HOME`` >
    ``/mnt/nvme_tmp/jason_caspo/hf_cache``. Returned path may not exist yet;
    callers create it when writing.
    """
    explicit = os.environ.get("CASPO_DATASET_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser() / _CACHE_SUBDIR
    hf_home = os.environ.get("HF_HOME") or _DEFAULT_HF_HOME
    return Path(hf_home).expanduser() / _CACHE_SUBDIR


def _dataset_cache_hash(
    tokenizer_name: str,
    dataset_name: str,
    max_prompt_len: int,
    prompt_template: Optional[str],
    dataset_split: str = "",
    system_prompt: Optional[str] = None,
    filter_eval_leakage: bool = True,
) -> str:
    """Hash the cache-key inputs into a stable filename component.

    ``|``-separated to prevent substring boundary collisions. We fold in
    every cfg field that materially affects the rendered prompts or the
    set of rows so a stale cache can't be reused under a different config:

    * ``tokenizer_name`` — the chat-template path is tokenizer-specific.
    * ``dataset_name`` / ``dataset_split`` — different rows entirely.
    * ``max_prompt_len`` — different post-filter row set.
    * ``prompt_template`` — paper-faithful reproductions (e.g. VinePPO's
      explicit math template) must not reuse a cache built with the
      tokenizer's chat template.
    * ``system_prompt`` — prepended on the no-template path AND threaded
      into ``apply_chat_template`` as a system message, so changing it
      changes every rendered prompt verbatim.
    """
    template = prompt_template if prompt_template is not None else ""
    sysp = system_prompt if system_prompt is not None else ""
    leak = "leakfilter" if filter_eval_leakage else "noleakfilter"
    key = (
        f"{tokenizer_name}|{dataset_name}|{dataset_split}|"
        f"{int(max_prompt_len)}|{template}|{sysp}|{leak}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:_HASH_LEN]


def _resolve_tokenizer_key(cfg: Any, tokenizer: Any) -> str:
    """Best-effort identifier for the tokenizer in the cache key.

    Order: ``cfg.tokenizer_name_or_path`` > ``cfg.model_name_or_path`` >
    ``tokenizer.name_or_path``. Falls back to ``"<unknown>"`` so the cache
    is at least *consistent* across runs even when the metadata is missing —
    the downside is that swapping tokenizers without changing the cfg would
    silently reuse a stale cache, but that's also true of every other field
    in the key.
    """
    name = getattr(cfg, "tokenizer_name_or_path", None) or getattr(
        cfg, "model_name_or_path", None
    )
    if not name and tokenizer is not None:
        name = getattr(tokenizer, "name_or_path", None)
    return str(name) if name else "<unknown>"


def _load_dataset_cache(path: Path) -> Optional[List[dict]]:
    """Load and validate a cache file. Returns None on any version / shape
    mismatch so the caller falls through to a rebuild instead of failing the
    job on a stale layout."""
    try:
        # ``weights_only=False`` because the payload is a dict of plain lists
        # of strings, not just tensors. Cache files are local-only artifacts
        # so the unpickle attack surface is the same as importing the repo.
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version", -1)) != _DATASET_CACHE_VERSION:
        return None
    examples = payload.get("examples")
    if not isinstance(examples, list):
        return None
    return examples


def _save_dataset_cache(path: Path, examples: List[dict]) -> None:
    """Atomic-ish save: write to ``.tmp`` then ``os.replace`` so concurrent
    readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {"version": _DATASET_CACHE_VERSION, "examples": examples}
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except Exception:
        # Best-effort: if we can't write the cache (read-only FS, full disk),
        # fall through silently rather than failing the training job.
        try:
            tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass


def load_train_dataset(cfg: Any, tokenizer: Any = None) -> Iterable[dict]:
    """Load and standardize the training dataset, with on-disk caching.

    Cold start: download via HF, build standardized rows via
    :func:`caspo.data.math_data._build_from_rows`, save to the cache file.
    Warm start: ``torch.load`` the cached list directly — no HF / tokenizer
    work at all.

    Set ``CASPO_DATASET_CACHE_DISABLE=1`` in the environment to bypass both
    read and write paths (useful when iterating on dataset-prep code).
    """
    if _cache_disabled():
        return _load_train_dataset_uncached(cfg, tokenizer=tokenizer)

    tokenizer_key = _resolve_tokenizer_key(cfg, tokenizer)
    dataset_name = getattr(cfg, "dataset_name", "<unknown>")
    dataset_split = str(getattr(cfg, "dataset_split", "") or "")
    max_prompt_len = int(getattr(cfg, "max_prompt_len", 0) or 0)
    prompt_template = getattr(cfg, "prompt_template", None)
    system_prompt = getattr(cfg, "system_prompt", None)

    cache_dir = _resolve_cache_dir()
    cache_hash = _dataset_cache_hash(
        tokenizer_key,
        dataset_name,
        max_prompt_len,
        prompt_template,
        dataset_split=dataset_split,
        system_prompt=system_prompt,
        filter_eval_leakage=bool(getattr(cfg, "filter_eval_leakage", True)),
    )
    cache_path = cache_dir / f"{cache_hash}.pt"

    if cache_path.exists():
        cached = _load_dataset_cache(cache_path)
        if cached is not None:
            return cached
        # Cache present but unreadable — fall through and rebuild it.

    # Cold path: same as the legacy uncached loader, then persist.
    raw = _load_hf(dataset_name, getattr(cfg, "dataset_split", "train"))
    examples = _build_from_rows(raw, cfg, tokenizer)
    _save_dataset_cache(cache_path, examples)
    return examples
