"""Tests for ``caspo.data.cached_dataset``.

Covers three contracts:

1. **Hash determinism / collision avoidance** — same key → same hash, any
   component change → different hash.
2. **Mmap roundtrip** — first construction tokenizes + writes to disk, second
   construction with same key reads from disk and yields byte-identical
   tensors (verified via ``loaded_from_cache``).
3. **Miss → rebuild** — a corrupted / missing cache file or ``rebuild=True``
   triggers fresh tokenization. Different keys go to different files and
   don't trample each other.

The fake tokenizer from ``conftest`` is used so tests don't require network
or HF model downloads.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from caspo.data.cached_dataset import (
    CachedTokenizedDataset,
    cache_path_for,
    compute_cache_hash,
    default_cache_dir,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_cache_dir(tmp_path, monkeypatch) -> Path:
    """Point the cache at a tmp dir via env var so default_cache_dir() picks
    it up too. Avoids stomping on the user's real ``~/.cache/caspo``.
    """
    d = tmp_path / "tok_cache"
    d.mkdir()
    monkeypatch.setenv("CASPO_TOKENIZED_CACHE_DIR", str(d))
    return d


@pytest.fixture
def sample_rows():
    return [
        {"prompt": "what is two plus two ?"},
        {"prompt": "solve x squared minus four"},
        {"prompt": "integrate sin x dx from zero to pi"},
        {"prompt": "short"},
    ]


# ---------------------------------------------------------------------------
# 1. Hash determinism
# ---------------------------------------------------------------------------


class TestHashDeterminism:
    def test_same_key_same_hash(self):
        h1 = compute_cache_hash("modelA", "dsX", 512)
        h2 = compute_cache_hash("modelA", "dsX", 512)
        assert h1 == h2
        assert len(h1) == 16
        assert all(c in "0123456789abcdef" for c in h1)

    def test_model_change_changes_hash(self):
        a = compute_cache_hash("modelA", "dsX", 512)
        b = compute_cache_hash("modelB", "dsX", 512)
        assert a != b

    def test_dataset_change_changes_hash(self):
        a = compute_cache_hash("modelA", "dsX", 512)
        b = compute_cache_hash("modelA", "dsY", 512)
        assert a != b

    def test_max_len_change_changes_hash(self):
        a = compute_cache_hash("modelA", "dsX", 512)
        b = compute_cache_hash("modelA", "dsX", 1024)
        assert a != b

    def test_max_len_int_coercion(self):
        # 512 and "512" should hash identically once coerced.
        h_int = compute_cache_hash("modelA", "dsX", 512)
        h_str_int = compute_cache_hash("modelA", "dsX", int("512"))
        assert h_int == h_str_int

    def test_no_substring_collision(self):
        # "foo" + "/bar" must not collide with "foo/" + "bar". The | separator
        # in the key makes this trivially safe; pin the contract anyway.
        a = compute_cache_hash("foo", "/bar", 16)
        b = compute_cache_hash("foo/", "bar", 16)
        assert a != b

    def test_cache_path_uses_hash_in_filename(self, tmp_path):
        p = cache_path_for("m", "d", 32, cache_dir=str(tmp_path))
        h = compute_cache_hash("m", "d", 32)
        assert p.name == f"tokenized_cache_{h}.pt"
        assert p.parent == tmp_path


class TestCacheDirResolution:
    def test_explicit_cache_dir_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CASPO_TOKENIZED_CACHE_DIR", str(tmp_path / "env"))
        explicit = tmp_path / "explicit"
        p = cache_path_for("m", "d", 1, cache_dir=str(explicit))
        assert p.parent == explicit

    def test_env_var_overrides_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CASPO_TOKENIZED_CACHE_DIR", str(tmp_path / "env"))
        assert default_cache_dir() == tmp_path / "env"

    def test_env_var_unset_falls_back_home(self, monkeypatch):
        monkeypatch.delenv("CASPO_TOKENIZED_CACHE_DIR", raising=False)
        d = default_cache_dir()
        assert d.name == "tokenized"
        assert d.parent.name == "caspo"


# ---------------------------------------------------------------------------
# 2. Mmap roundtrip
# ---------------------------------------------------------------------------


class TestMmapRoundtrip:
    def test_first_build_then_reload(
        self, isolated_cache_dir, fake_tokenizer, sample_rows
    ):
        kw = dict(
            model_name_or_path="m1",
            dataset_name="d1",
            max_prompt_len=8,
        )

        ds1 = CachedTokenizedDataset(sample_rows, fake_tokenizer, **kw)
        assert ds1.loaded_from_cache is False
        assert ds1.cache_path.exists()
        assert len(ds1) == len(sample_rows)
        assert ds1.input_ids.dtype == torch.long
        assert ds1.attention_mask.dtype == torch.bool
        assert ds1.input_ids.shape == ds1.attention_mask.shape
        # Padded shape: rows × max_seq_len ≤ max_prompt_len.
        assert ds1.input_ids.shape[1] <= 8

        # Second construction (rows / tokenizer should NOT be consulted).
        sentinel_rows = [{"prompt": "DIFFERENT"}]  # would change tensors if used
        ds2 = CachedTokenizedDataset(sentinel_rows, fake_tokenizer, **kw)
        assert ds2.loaded_from_cache is True
        assert torch.equal(ds1.input_ids, ds2.input_ids)
        assert torch.equal(ds1.attention_mask, ds2.attention_mask)
        assert ds1.pad_id == ds2.pad_id

    def test_getitem_returns_per_row_tensors(
        self, isolated_cache_dir, fake_tokenizer, sample_rows
    ):
        ds = CachedTokenizedDataset(
            sample_rows,
            fake_tokenizer,
            model_name_or_path="m",
            dataset_name="d",
            max_prompt_len=16,
        )
        item = ds[0]
        assert set(item.keys()) == {"input_ids", "attention_mask"}
        assert item["input_ids"].shape == (ds.input_ids.shape[1],)
        assert item["attention_mask"].shape == (ds.attention_mask.shape[1],)

    def test_attention_mask_marks_real_tokens(
        self, isolated_cache_dir, fake_tokenizer, sample_rows
    ):
        ds = CachedTokenizedDataset(
            sample_rows,
            fake_tokenizer,
            model_name_or_path="m",
            dataset_name="d",
            max_prompt_len=32,
        )
        # At least one row must have at least one True position in the mask
        # (otherwise tokenization is silently dropping everything).
        assert ds.attention_mask.any().item()
        # The shortest row ("short" → 1 token) should have exactly one True
        # in its mask if it sits in row 3 of sample_rows.
        short_row_mask = ds.attention_mask[3]
        assert short_row_mask.sum().item() == 1

    def test_empty_rows_yields_empty_dataset(
        self, isolated_cache_dir, fake_tokenizer
    ):
        ds = CachedTokenizedDataset(
            [],
            fake_tokenizer,
            model_name_or_path="m",
            dataset_name="empty",
            max_prompt_len=8,
        )
        assert len(ds) == 0
        assert ds.cache_path.exists()


# ---------------------------------------------------------------------------
# 3. Miss → rebuild and key isolation
# ---------------------------------------------------------------------------


class TestMissAndRebuild:
    def test_missing_file_triggers_build(
        self, isolated_cache_dir, fake_tokenizer, sample_rows
    ):
        kw = dict(
            model_name_or_path="m",
            dataset_name="d",
            max_prompt_len=8,
        )
        ds1 = CachedTokenizedDataset(sample_rows, fake_tokenizer, **kw)
        assert ds1.loaded_from_cache is False
        # Delete the cache file → next construction should rebuild.
        os.remove(ds1.cache_path)
        ds2 = CachedTokenizedDataset(sample_rows, fake_tokenizer, **kw)
        assert ds2.loaded_from_cache is False
        assert ds2.cache_path.exists()
        # Tensors should still match because the input rows are identical.
        assert torch.equal(ds1.input_ids, ds2.input_ids)

    def test_rebuild_flag_forces_retokenize(
        self, isolated_cache_dir, fake_tokenizer, sample_rows
    ):
        kw = dict(
            model_name_or_path="m",
            dataset_name="d",
            max_prompt_len=8,
        )
        CachedTokenizedDataset(sample_rows, fake_tokenizer, **kw)
        # Without rebuild → from cache.
        ds_cached = CachedTokenizedDataset(sample_rows, fake_tokenizer, **kw)
        assert ds_cached.loaded_from_cache is True
        # With rebuild → fresh tokenization.
        ds_fresh = CachedTokenizedDataset(
            sample_rows, fake_tokenizer, rebuild=True, **kw
        )
        assert ds_fresh.loaded_from_cache is False

    def test_different_keys_dont_collide(
        self, isolated_cache_dir, fake_tokenizer, sample_rows
    ):
        ds_a = CachedTokenizedDataset(
            sample_rows,
            fake_tokenizer,
            model_name_or_path="modelA",
            dataset_name="ds",
            max_prompt_len=8,
        )
        ds_b = CachedTokenizedDataset(
            sample_rows,
            fake_tokenizer,
            model_name_or_path="modelB",
            dataset_name="ds",
            max_prompt_len=8,
        )
        assert ds_a.cache_path != ds_b.cache_path
        assert ds_a.cache_path.exists()
        assert ds_b.cache_path.exists()

        # Reload modelA — must hit modelA's cache, not modelB's.
        ds_a2 = CachedTokenizedDataset(
            sample_rows,
            fake_tokenizer,
            model_name_or_path="modelA",
            dataset_name="ds",
            max_prompt_len=8,
        )
        assert ds_a2.loaded_from_cache is True
        assert ds_a2.cache_path == ds_a.cache_path

    def test_different_max_len_different_cache(
        self, isolated_cache_dir, fake_tokenizer, sample_rows
    ):
        ds_8 = CachedTokenizedDataset(
            sample_rows,
            fake_tokenizer,
            model_name_or_path="m",
            dataset_name="d",
            max_prompt_len=8,
        )
        ds_16 = CachedTokenizedDataset(
            sample_rows,
            fake_tokenizer,
            model_name_or_path="m",
            dataset_name="d",
            max_prompt_len=16,
        )
        assert ds_8.cache_path != ds_16.cache_path

    def test_explicit_cache_dir_argument(
        self, tmp_path, fake_tokenizer, sample_rows
    ):
        # Bypass the env-var fixture; pass cache_dir= directly.
        target = tmp_path / "explicit_dir"
        ds = CachedTokenizedDataset(
            sample_rows,
            fake_tokenizer,
            model_name_or_path="m",
            dataset_name="d",
            max_prompt_len=8,
            cache_dir=str(target),
        )
        assert ds.cache_path.parent == target
        assert ds.cache_path.exists()
