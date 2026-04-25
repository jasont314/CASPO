"""Pre-tokenized dataset cache (pure infra, not yet wired into loaders).

Tokenizing the same prompt corpus on every training-job startup is wasteful —
HF tokenizers parallelize, but for a 40k-row math corpus + a long chat template
the encode pass is still 5-30s of pure CPU. This module trades that for one
deterministic ``torch.save`` and a memory-mapped reload on subsequent runs.

API
---
    CachedTokenizedDataset(
        rows,                       # iterable of {"prompt": str, ...}
        tokenizer,
        *,
        model_name_or_path: str,    # part of cache key
        dataset_name: str,          # part of cache key
        max_prompt_len: int,        # part of cache key
        cache_dir: Optional[str] = None,
        rebuild: bool = False,
    )

The instance exposes ``input_ids`` (LongTensor, shape ``[N, L]``, padded with
``tokenizer.pad_token_id`` when available else 0) and ``attention_mask``
(BoolTensor, same shape). It is len/getitem-able for use as a Dataset.

Cache file naming
-----------------
``{cache_dir}/tokenized_cache_{model_hash}.pt`` where ``model_hash`` is the
first 16 hex chars of ``sha256(model_name_or_path | dataset_name | max_prompt_len)``.
This guarantees that a different tokenizer or a different ``max_prompt_len``
filter never collides with an existing cache.

Cache directory resolution order
--------------------------------
1. explicit ``cache_dir=`` argument
2. environment variable ``CASPO_TOKENIZED_CACHE_DIR``
3. ``~/.cache/caspo/tokenized/``
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import torch


__all__ = [
    "CachedTokenizedDataset",
    "compute_cache_hash",
    "default_cache_dir",
    "cache_path_for",
]


# ---------------------------------------------------------------------------
# Hashing / path helpers (kept pure so tests can import them without torch IO)
# ---------------------------------------------------------------------------


_HASH_LEN = 16  # first 16 hex chars of sha256 — 64 bits of collision space.


def compute_cache_hash(
    model_name_or_path: str,
    dataset_name: str,
    max_prompt_len: int,
) -> str:
    """Hash the (tokenizer, dataset, length-cap) triple into the filename.

    Joined with ``|`` so substrings can't trivially shift the boundary
    (e.g. ``"foo" + "/bar"`` vs ``"foo/bar"``). UTF-8 encoded; the hash is
    deterministic across processes / Python versions.
    """
    key = f"{model_name_or_path}|{dataset_name}|{int(max_prompt_len)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:_HASH_LEN]


def default_cache_dir() -> Path:
    """Resolve the default tokenized-cache directory.

    Honors ``CASPO_TOKENIZED_CACHE_DIR`` for sandboxed test runs that can't
    touch ``$HOME``; otherwise falls back to ``~/.cache/caspo/tokenized/``.
    """
    env = os.environ.get("CASPO_TOKENIZED_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "caspo" / "tokenized"


def cache_path_for(
    model_name_or_path: str,
    dataset_name: str,
    max_prompt_len: int,
    cache_dir: Optional[str] = None,
) -> Path:
    """Compose the full cache-file path. Does NOT create the directory."""
    base = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
    h = compute_cache_hash(model_name_or_path, dataset_name, max_prompt_len)
    return base / f"tokenized_cache_{h}.pt"


# ---------------------------------------------------------------------------
# Tokenization → padded tensors
# ---------------------------------------------------------------------------


def _resolve_pad_id(tokenizer: Any) -> int:
    """Pick a pad id. Falls back to ``eos_token_id`` then 0 (safe sentinel)."""
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is not None:
        return int(pad)
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        return int(eos)
    return 0


def _tokenize_rows(
    rows: Iterable[dict],
    tokenizer: Any,
    max_prompt_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Tokenize an iterable of rows to padded ``(input_ids, attention_mask)``.

    Returns the pad id alongside so callers (and the persistence layer) don't
    have to re-derive it. Rows whose tokenization exceeds ``max_prompt_len``
    are truncated rather than dropped — drop-policy is the caller's job.
    """
    prompts: List[str] = []
    for row in rows:
        if isinstance(row, dict):
            p = row.get("prompt")
        else:
            p = row
        if isinstance(p, str):
            prompts.append(p)

    pad_id = _resolve_pad_id(tokenizer)

    if not prompts:
        empty_ids = torch.empty((0, 0), dtype=torch.long)
        empty_mask = torch.empty((0, 0), dtype=torch.bool)
        return empty_ids, empty_mask, pad_id

    # Try the HF batch path first; fall back to per-prompt for fake tokenizers
    # used in tests (which only implement ``__call__`` over single strings).
    encoded_lists: List[List[int]] = []
    try:
        if hasattr(tokenizer, "batch_encode_plus"):
            enc = tokenizer.batch_encode_plus(prompts, add_special_tokens=False)
        else:
            enc = tokenizer(prompts, add_special_tokens=False)
        ids_field = enc["input_ids"]
        # ids_field may be a list of lists OR a list of strings (fake
        # tokenizers in conftest split on whitespace). Accept both.
        for ids in ids_field:
            encoded_lists.append([_to_int(x, idx) for idx, x in enumerate(ids)])
    except Exception:
        for p in prompts:
            try:
                out = tokenizer(p, add_special_tokens=False)
                ids = out.input_ids if hasattr(out, "input_ids") else out["input_ids"]
                encoded_lists.append([_to_int(x, idx) for idx, x in enumerate(ids)])
            except Exception:
                encoded_lists.append([])

    # Truncate to max_prompt_len (>0). max_prompt_len<=0 disables truncation.
    if max_prompt_len > 0:
        encoded_lists = [seq[:max_prompt_len] for seq in encoded_lists]

    max_len = max((len(s) for s in encoded_lists), default=0)
    n = len(encoded_lists)
    input_ids = torch.full((n, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, max_len), dtype=torch.bool)
    for i, seq in enumerate(encoded_lists):
        if not seq:
            continue
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, : len(seq)] = True
    return input_ids, attention_mask, pad_id


def _to_int(x: Any, idx: int) -> int:
    """Coerce a tokenizer-output element to int.

    Real tokenizers return ints. The conftest fake tokenizer returns whitespace
    tokens (strings) — we hash to a stable int so the cache pipeline still
    exercises end-to-end. The mapping doesn't have to be reversible; tests
    only verify roundtrip equality of bytes-on-disk.
    """
    if isinstance(x, int):
        return x
    if isinstance(x, str):
        # Stable per-token integer derived from the string content; positional
        # ``idx`` is folded in so duplicate tokens still get distinct-ish ids,
        # which keeps test prompts from collapsing to a single column.
        h = hashlib.sha256(f"{idx}:{x}".encode("utf-8")).digest()
        return int.from_bytes(h[:4], "little", signed=False)
    return int(x)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


_CACHE_FORMAT_VERSION = 1


def _save_cache(
    path: Path,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_id: int,
    meta: dict,
) -> None:
    """Atomic-ish save: write to ``.tmp`` then rename. ``torch.save`` of a
    plain dict-of-tensors stays loadable via ``weights_only=True`` + ``mmap``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "version": _CACHE_FORMAT_VERSION,
        "input_ids": input_ids.contiguous(),
        "attention_mask": attention_mask.contiguous(),
        "pad_id": int(pad_id),
        "meta": meta,
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _load_cache(path: Path) -> dict:
    """Memory-mapped load. Falls back to plain load on older torch.

    ``weights_only=True`` is required for ``mmap=True`` on torch>=2.0; we
    pass it unconditionally and fall back if the running torch is older.
    """
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except TypeError:
        # Older torch lacks ``mmap`` / ``weights_only`` kwargs.
        return torch.load(path, map_location="cpu")


# ---------------------------------------------------------------------------
# Public Dataset class
# ---------------------------------------------------------------------------


class CachedTokenizedDataset:
    """Tokenize-once, mmap-on-reload dataset of prompt input_ids.

    Parameters
    ----------
    rows
        Iterable of ``{"prompt": str, ...}`` dicts (extra fields ignored) or
        bare prompt strings. Only consulted on a cache miss.
    tokenizer
        HF-style tokenizer with either ``batch_encode_plus`` or ``__call__``.
        Only consulted on a cache miss.
    model_name_or_path, dataset_name, max_prompt_len
        Components of the cache key. Different values produce different
        cache files (no collisions).
    cache_dir
        Override the default cache directory. ``None`` falls back to the
        ``CASPO_TOKENIZED_CACHE_DIR`` env var, then ``~/.cache/caspo/tokenized``.
    rebuild
        Force tokenization even if a cache file exists. Useful when the
        underlying corpus content changed but the key didn't.
    """

    def __init__(
        self,
        rows: Iterable[dict],
        tokenizer: Any,
        *,
        model_name_or_path: str,
        dataset_name: str,
        max_prompt_len: int,
        cache_dir: Optional[str] = None,
        rebuild: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.dataset_name = dataset_name
        self.max_prompt_len = int(max_prompt_len)
        self.cache_path: Path = cache_path_for(
            model_name_or_path,
            dataset_name,
            max_prompt_len,
            cache_dir=cache_dir,
        )
        self.cache_hash: str = compute_cache_hash(
            model_name_or_path, dataset_name, max_prompt_len
        )

        if not rebuild and self.cache_path.exists():
            payload = _load_cache(self.cache_path)
            self.input_ids: torch.Tensor = payload["input_ids"]
            self.attention_mask: torch.Tensor = payload["attention_mask"]
            self.pad_id: int = int(payload.get("pad_id", 0))
            self._from_cache = True
        else:
            ids, mask, pad_id = _tokenize_rows(rows, tokenizer, self.max_prompt_len)
            self.input_ids = ids
            self.attention_mask = mask
            self.pad_id = int(pad_id)
            _save_cache(
                self.cache_path,
                ids,
                mask,
                pad_id,
                meta={
                    "model_name_or_path": model_name_or_path,
                    "dataset_name": dataset_name,
                    "max_prompt_len": int(max_prompt_len),
                },
            )
            self._from_cache = False

    # -- Dataset protocol ----------------------------------------------------

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }

    # -- Introspection -------------------------------------------------------

    @property
    def loaded_from_cache(self) -> bool:
        """True iff this instance hydrated from disk (vs. tokenized fresh)."""
        return self._from_cache

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        n, L = (self.input_ids.shape if self.input_ids.ndim == 2 else (0, 0))
        src = "cache" if self._from_cache else "fresh"
        return (
            f"CachedTokenizedDataset(n={n}, L={L}, hash={self.cache_hash}, "
            f"src={src}, path={self.cache_path})"
        )
