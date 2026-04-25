"""Atomic checkpoint utilities for CASPO.

Pure infrastructure: provides save/load helpers that avoid partial-write
corruption when a process is killed mid-save. This is critical for vLLM
weight-sync workflows where a half-written checkpoint directory can poison
subsequent loads.

The strategy is the standard write-temp-then-rename pattern:

    save_pretrained(...) -> path.tmp/        # may be partial if killed
    fsync files + parent dir                 # durability barrier
    os.rename(path.tmp, path)                # atomic on POSIX

If the process dies before the rename, the final ``path`` either does not
exist or still contains the previous (complete) checkpoint. ``path.tmp`` is
left behind as evidence of the interrupted save and is detected by
``safe_load_pretrained``.

None of these helpers wire themselves into any trainer; callers are expected
to invoke them explicitly.
"""

from __future__ import annotations

import errno
import glob
import os
import shutil
from typing import Any, Optional

__all__ = [
    "atomic_save_pretrained",
    "safe_load_pretrained",
    "prune_old_checkpoints",
    "InterruptedSaveError",
]


def _tmp_path(path: str) -> str:
    """Return the staging path for an in-progress save."""
    # Strip trailing slash so ``foo/`` and ``foo`` produce the same tmp name.
    return path.rstrip(os.sep) + ".tmp"


def _fsync_dir(path: str) -> None:
    """fsync a directory so its rename/creation is durable on disk.

    Best-effort: silently no-ops on platforms (e.g. Windows) where opening a
    directory for fsync is not supported.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(fd)
        except OSError as e:
            # Some filesystems / platforms (e.g. tmpfs on certain kernels)
            # disallow fsync on directory fds. Treat as best-effort.
            if e.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EPERM):
                raise
    finally:
        os.close(fd)


def _fsync_tree(path: str) -> None:
    """fsync every regular file under ``path`` plus the directory itself.

    Walks the tree once; ignores files that disappear between listing and
    open (e.g. transient temp files inside HF save artifacts).
    """
    for root, _dirs, files in os.walk(path):
        for name in files:
            fpath = os.path.join(root, name)
            try:
                fd = os.open(fpath, os.O_RDONLY)
            except OSError:
                continue
            try:
                try:
                    os.fsync(fd)
                except OSError:
                    # Symlinks / special files may not support fsync.
                    pass
            finally:
                os.close(fd)
        _fsync_dir(root)


class InterruptedSaveError(RuntimeError):
    """Raised by ``safe_load_pretrained`` when a ``.tmp`` shard is present.

    Indicates a previous save was killed before the atomic rename. The
    final checkpoint at ``path`` may not exist or may be the older copy;
    operator intervention is required to resolve the ambiguity.
    """


def atomic_save_pretrained(
    model: Any,
    path: str,
    tokenizer: Optional[Any] = None,
    **save_kwargs: Any,
) -> str:
    """Atomically save a HuggingFace-style model + optional tokenizer.

    Writes via ``model.save_pretrained(path.tmp, **save_kwargs)`` (and the
    tokenizer if provided), fsyncs every produced file plus the staging
    directory, then renames ``path.tmp`` -> ``path``. The rename is atomic
    on POSIX, so a kill at any moment leaves either the previous checkpoint
    or nothing at ``path`` — never a half-written one.

    If ``path`` already exists, it is replaced. Any pre-existing
    ``path.tmp`` from a previous interrupted save is removed first.

    Args:
        model: object with a ``save_pretrained(directory, ...)`` method.
        path: final destination directory.
        tokenizer: optional object with ``save_pretrained(directory)``.
        **save_kwargs: forwarded to ``model.save_pretrained``.

    Returns:
        The final ``path``.
    """
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(parent, exist_ok=True)

    tmp = _tmp_path(path)
    # Clear any leftover tmp from a prior crashed save.
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp, exist_ok=False)

    try:
        model.save_pretrained(tmp, **save_kwargs)
        if tokenizer is not None:
            tokenizer.save_pretrained(tmp)

        # Durability barrier: ensure all bytes hit disk before rename.
        _fsync_tree(tmp)

        # Replace any existing final dir with the staged one. ``os.rename``
        # is not atomic across non-empty target dirs, so swap via a backup:
        #   1) move old final -> backup        (atomic)
        #   2) move tmp       -> final         (atomic)
        #   3) remove backup                   (lossy is fine: old is gone)
        # If step 2 fails, restore the backup so the previous checkpoint
        # survives the failed save.
        backup = path.rstrip(os.sep) + ".bak"
        if os.path.exists(backup):
            shutil.rmtree(backup)
        had_old = os.path.exists(path)
        if had_old:
            os.rename(path, backup)
        try:
            os.rename(tmp, path)
        except BaseException:
            if had_old and not os.path.exists(path):
                # Best-effort rollback of the previous checkpoint.
                try:
                    os.rename(backup, path)
                except OSError:
                    pass
            raise
        # fsync the parent so the rename itself is durable.
        _fsync_dir(parent)
        # Old checkpoint successfully replaced — drop the backup.
        if had_old and os.path.exists(backup):
            shutil.rmtree(backup)
    except BaseException:
        # On any failure, leave tmp in place as a marker for diagnosis.
        # The caller will see no final-path mutation.
        raise

    return path


def safe_load_pretrained(cls: Any, path: str, **kwargs: Any) -> Any:
    """Load via ``cls.from_pretrained(path, **kwargs)`` with a tmp-guard.

    If a sibling ``path.tmp`` directory is present, refuse to load and raise
    ``InterruptedSaveError``: the previous save was killed mid-write and the
    state of ``path`` is ambiguous.

    Args:
        cls: a class exposing ``from_pretrained(path, **kwargs)`` (e.g.
            ``AutoModelForCausalLM``, ``AutoTokenizer``).
        path: directory previously written by ``atomic_save_pretrained``.
        **kwargs: forwarded to ``cls.from_pretrained``.
    """
    path = os.fspath(path)
    tmp = _tmp_path(path)
    if os.path.exists(tmp):
        raise InterruptedSaveError(
            f"Found leftover staging directory {tmp!r} from an interrupted "
            f"save. The checkpoint at {path!r} may be the previous version "
            f"or missing entirely. Remove {tmp!r} after verifying {path!r} "
            f"is intact, then retry."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path!r}")
    return cls.from_pretrained(path, **kwargs)


def prune_old_checkpoints(
    output_dir: str,
    keep_n_recent: int = 3,
    pattern: str = "checkpoint-*",
) -> list[str]:
    """Delete all but the ``keep_n_recent`` most-recent matching checkpoints.

    Recency is determined by directory mtime. Useful for runs that emit
    intermediate snapshots (CASPO does not by default; this is a reusable
    helper for future workflows).

    ``.tmp`` siblings are always ignored — never counted toward
    ``keep_n_recent`` and never pruned. Operator must resolve them
    manually via ``safe_load_pretrained``'s diagnostic.

    Args:
        output_dir: directory containing checkpoint subdirs.
        keep_n_recent: number of newest checkpoints to retain. Must be >= 0.
        pattern: glob pattern (relative to ``output_dir``) selecting
            checkpoint directories.

    Returns:
        List of absolute paths that were removed, in deletion order.
    """
    if keep_n_recent < 0:
        raise ValueError(f"keep_n_recent must be >= 0, got {keep_n_recent}")
    if not os.path.isdir(output_dir):
        return []

    matches = glob.glob(os.path.join(output_dir, pattern))
    # Only directories, and skip any ``.tmp`` staging dirs.
    candidates = [
        os.path.abspath(p)
        for p in matches
        if os.path.isdir(p) and not p.rstrip(os.sep).endswith(".tmp")
    ]
    # Newest first.
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    to_remove = candidates[keep_n_recent:]
    removed: list[str] = []
    for p in to_remove:
        try:
            shutil.rmtree(p)
            removed.append(p)
        except OSError:
            # Skip what we can't remove; caller can re-invoke.
            continue
    return removed
