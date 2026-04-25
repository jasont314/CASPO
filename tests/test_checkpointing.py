"""Tests for caspo.utils.checkpointing — atomic save / safe load / prune.

These tests use mock model/tokenizer stand-ins so we never load real weights.
The contract is purely about filesystem semantics (staging dir, fsync,
atomic rename, leftover-tmp detection).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from typing import Any
from unittest import mock

import pytest

from caspo.utils.checkpointing import (
    InterruptedSaveError,
    atomic_save_pretrained,
    prune_old_checkpoints,
    safe_load_pretrained,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeModel:
    """Stand-in for a HF model. ``save_pretrained`` writes a sentinel file."""

    def __init__(self, payload: bytes = b"weights-v1"):
        self.payload = payload
        self.calls: list[str] = []

    def save_pretrained(self, directory: str, **kwargs: Any) -> None:
        self.calls.append(directory)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "pytorch_model.bin"), "wb") as f:
            f.write(self.payload)
        # HF also writes a config.
        with open(os.path.join(directory, "config.json"), "w") as f:
            f.write('{"model_type": "fake"}')


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def save_pretrained(self, directory: str) -> None:
        self.calls.append(directory)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "tokenizer.json"), "w") as f:
            f.write('{"version": "1"}')


class _FakeLoader:
    """Mimics ``cls.from_pretrained`` returning a marker tied to the path."""

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> dict:
        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            payload = f.read()
        return {"path": path, "payload": payload, "kwargs": kwargs}


# ---------------------------------------------------------------------------
# atomic_save_pretrained — happy path
# ---------------------------------------------------------------------------


def test_atomic_save_writes_final_dir_and_no_tmp():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ckpt")
        m = _FakeModel(payload=b"abc")
        out = atomic_save_pretrained(m, path)
        assert out == path
        assert os.path.isdir(path)
        assert not os.path.exists(path + ".tmp")
        # Model was told to save into the staging dir, not the final one.
        assert m.calls == [path + ".tmp"]
        # Sentinel file made it to the final location.
        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            assert f.read() == b"abc"


def test_atomic_save_includes_tokenizer():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ckpt")
        m = _FakeModel()
        tok = _FakeTokenizer()
        atomic_save_pretrained(m, path, tokenizer=tok)
        assert os.path.isfile(os.path.join(path, "tokenizer.json"))
        # Tokenizer was also written into the staging dir.
        assert tok.calls == [path + ".tmp"]


def test_atomic_save_forwards_save_kwargs():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ckpt")
        m = mock.MagicMock()

        def _fake_save(directory, **kwargs):
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "marker"), "w") as f:
                f.write("ok")

        m.save_pretrained.side_effect = _fake_save
        atomic_save_pretrained(m, path, safe_serialization=True, max_shard_size="2GB")
        m.save_pretrained.assert_called_once()
        _args, kwargs = m.save_pretrained.call_args
        assert kwargs == {"safe_serialization": True, "max_shard_size": "2GB"}


def test_atomic_save_replaces_existing_final_dir():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ckpt")
        # Pre-existing previous checkpoint with stale contents.
        os.makedirs(path)
        with open(os.path.join(path, "pytorch_model.bin"), "wb") as f:
            f.write(b"old")
        with open(os.path.join(path, "stale_only.txt"), "w") as f:
            f.write("leftover")

        m = _FakeModel(payload=b"new")
        atomic_save_pretrained(m, path)

        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            assert f.read() == b"new"
        # Stale file from old checkpoint should be gone.
        assert not os.path.exists(os.path.join(path, "stale_only.txt"))


def test_atomic_save_clears_leftover_tmp_from_prior_crash():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ckpt")
        tmp = path + ".tmp"
        os.makedirs(tmp)
        with open(os.path.join(tmp, "garbage"), "w") as f:
            f.write("partial")

        m = _FakeModel()
        atomic_save_pretrained(m, path)

        # Final path exists, tmp is gone, garbage file from prior crash
        # did not leak into the final dir.
        assert os.path.isdir(path)
        assert not os.path.exists(tmp)
        assert not os.path.exists(os.path.join(path, "garbage"))


# ---------------------------------------------------------------------------
# atomic_save_pretrained — fsync + atomicity
# ---------------------------------------------------------------------------


def test_atomic_save_calls_fsync():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ckpt")
        m = _FakeModel()
        with mock.patch("caspo.utils.checkpointing.os.fsync") as mfsync:
            atomic_save_pretrained(m, path)
        # At minimum: fsync the model file, the staging dir, and the parent.
        assert mfsync.call_count >= 3


def test_atomic_save_uses_rename_not_copy():
    """Rename is the atomicity primitive; verify we actually call it."""
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ckpt")
        m = _FakeModel()
        with mock.patch(
            "caspo.utils.checkpointing.os.rename", wraps=os.rename
        ) as mrename:
            atomic_save_pretrained(m, path)
        mrename.assert_called_once()
        src, dst = mrename.call_args.args
        assert src == path + ".tmp"
        assert dst == path


def test_atomic_save_kill_before_rename_leaves_no_final_dir():
    """Simulate a kill mid-write by raising before os.rename runs.

    Final ``path`` must NOT exist (so a subsequent load is unambiguous —
    either the previous checkpoint, which there isn't here, or nothing).
    The leftover ``.tmp`` is the diagnostic for the operator.
    """
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ckpt")
        m = _FakeModel()

        def _boom(*_a, **_kw):
            raise KeyboardInterrupt("simulated kill mid-save")

        with mock.patch("caspo.utils.checkpointing.os.rename", side_effect=_boom):
            with pytest.raises(KeyboardInterrupt):
                atomic_save_pretrained(m, path)

        # Final path was never created.
        assert not os.path.exists(path)
        # Staging dir still on disk as evidence of the interrupted save.
        assert os.path.isdir(path + ".tmp")


def test_atomic_save_kill_mid_write_preserves_previous_checkpoint():
    """If a previous good ckpt exists and a new save crashes, old one survives."""
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ckpt")
        # Existing good checkpoint.
        os.makedirs(path)
        with open(os.path.join(path, "pytorch_model.bin"), "wb") as f:
            f.write(b"old-good")

        m = _FakeModel(payload=b"new-bad")

        def _boom(*_a, **_kw):
            raise RuntimeError("disk full mid-rename")

        with mock.patch("caspo.utils.checkpointing.os.rename", side_effect=_boom):
            with pytest.raises(RuntimeError):
                atomic_save_pretrained(m, path)

        # Old checkpoint still intact and untouched.
        assert os.path.isdir(path)
        with open(os.path.join(path, "pytorch_model.bin"), "rb") as f:
            assert f.read() == b"old-good"
        # Tmp left behind for diagnosis.
        assert os.path.isdir(path + ".tmp")


# ---------------------------------------------------------------------------
# safe_load_pretrained
# ---------------------------------------------------------------------------


def test_safe_load_roundtrip():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ckpt")
        m = _FakeModel(payload=b"roundtrip")
        atomic_save_pretrained(m, path)
        loaded = safe_load_pretrained(_FakeLoader, path, foo=1)
        assert loaded["payload"] == b"roundtrip"
        assert loaded["kwargs"] == {"foo": 1}


def test_safe_load_detects_leftover_tmp():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ckpt")
        m = _FakeModel()
        atomic_save_pretrained(m, path)
        # Simulate a *new* save crashing later: tmp exists alongside final.
        os.makedirs(path + ".tmp")
        with open(os.path.join(path + ".tmp", "partial"), "w") as f:
            f.write("x")

        with pytest.raises(InterruptedSaveError) as ei:
            safe_load_pretrained(_FakeLoader, path)
        assert ".tmp" in str(ei.value)
        assert path in str(ei.value)


def test_safe_load_missing_path_raises():
    with tempfile.TemporaryDirectory() as root:
        path = os.path.join(root, "ckpt")
        with pytest.raises(FileNotFoundError):
            safe_load_pretrained(_FakeLoader, path)


# ---------------------------------------------------------------------------
# prune_old_checkpoints
# ---------------------------------------------------------------------------


def _make_ckpt(parent: str, name: str, mtime_offset: float = 0.0) -> str:
    p = os.path.join(parent, name)
    os.makedirs(p, exist_ok=True)
    with open(os.path.join(p, "marker"), "w") as f:
        f.write(name)
    if mtime_offset:
        t = time.time() + mtime_offset
        os.utime(p, (t, t))
    return p


def test_prune_keeps_n_most_recent():
    with tempfile.TemporaryDirectory() as root:
        # Older -> newer via mtime offsets.
        c1 = _make_ckpt(root, "checkpoint-100", mtime_offset=-300)
        c2 = _make_ckpt(root, "checkpoint-200", mtime_offset=-200)
        c3 = _make_ckpt(root, "checkpoint-300", mtime_offset=-100)
        c4 = _make_ckpt(root, "checkpoint-400", mtime_offset=0)

        removed = prune_old_checkpoints(root, keep_n_recent=2)

        assert set(removed) == {os.path.abspath(c1), os.path.abspath(c2)}
        assert not os.path.exists(c1)
        assert not os.path.exists(c2)
        assert os.path.isdir(c3)
        assert os.path.isdir(c4)


def test_prune_keep_zero_removes_all():
    with tempfile.TemporaryDirectory() as root:
        a = _make_ckpt(root, "checkpoint-1")
        b = _make_ckpt(root, "checkpoint-2")
        removed = prune_old_checkpoints(root, keep_n_recent=0)
        assert set(removed) == {os.path.abspath(a), os.path.abspath(b)}


def test_prune_no_op_when_under_limit():
    with tempfile.TemporaryDirectory() as root:
        a = _make_ckpt(root, "checkpoint-1")
        removed = prune_old_checkpoints(root, keep_n_recent=5)
        assert removed == []
        assert os.path.isdir(a)


def test_prune_ignores_tmp_dirs():
    with tempfile.TemporaryDirectory() as root:
        good = _make_ckpt(root, "checkpoint-1")
        tmp = _make_ckpt(root, "checkpoint-2.tmp")
        removed = prune_old_checkpoints(root, keep_n_recent=0)
        # Only the non-tmp dir is eligible.
        assert removed == [os.path.abspath(good)]
        # .tmp staging dir is preserved for operator diagnosis.
        assert os.path.isdir(tmp)


def test_prune_respects_pattern():
    with tempfile.TemporaryDirectory() as root:
        keep = _make_ckpt(root, "other-1")
        drop = _make_ckpt(root, "checkpoint-1")
        removed = prune_old_checkpoints(root, keep_n_recent=0, pattern="checkpoint-*")
        assert removed == [os.path.abspath(drop)]
        assert os.path.isdir(keep)


def test_prune_missing_dir_returns_empty():
    assert prune_old_checkpoints("/nonexistent/path/xyz") == []


def test_prune_rejects_negative_keep():
    with pytest.raises(ValueError):
        prune_old_checkpoints("/tmp", keep_n_recent=-1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
