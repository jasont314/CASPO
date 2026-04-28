"""Phase-1 trainer: fit the prefix value model V_φ via the IPVRM BCE-margin loss.

Loads (prompt, response, outcome) tuples produced by ``collect_value_data.py``,
splits them into train/val by prompt (held-out prompts, not random rollouts),
runs ``cfg.value_max_epochs`` epochs of optimizer steps with early-stopping
on val loss, and saves the *best-val-loss* checkpoint as ``final/``.

Usage::

    python scripts/train_value.py --config configs/value_smoke.yaml \\
        --override value_data_path=out/value/value_data.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict
from typing import List, Tuple

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from caspo.config import CASPOConfig
from caspo.utils.distributed import (
    distributed_env,
    init_distributed_if_needed,
    is_dist_initialized,
    resolve_device,
)
from caspo.value import PrefixValueModel, PrefixValueTrainer, ipvrm_loss
from scripts.collect_value_data import apply_overrides  # reuse coercion


_BATCH_KEYS = ("prompt_ids", "prompt_mask", "response_ids", "response_mask", "outcomes")


def _make_batch(rows, blob: dict) -> dict:
    """Build a per-batch dict from row indices.

    ``rows`` may be a list[int] or a 1-D LongTensor. When ``blob`` tensors
    already live on GPU, indexing produces GPU tensors and the trainer's
    ``.to(device)`` becomes a no-op. When tensors are CPU-pinned, we leave
    the H2D copy to the trainer (which is in a separate file).
    """
    if not torch.is_tensor(rows):
        rows = torch.as_tensor(rows, dtype=torch.long, device=blob["prompt_ids"].device)
    elif rows.device != blob["prompt_ids"].device:
        rows = rows.to(blob["prompt_ids"].device, non_blocking=True)
    return {k: blob[k].index_select(0, rows) for k in _BATCH_KEYS}


def _prepare_blob(blob: dict, device: str, prefer_device: bool) -> dict:
    """Move/pin the per-row tensors to speed up batch assembly.

    If ``prefer_device`` is True and the tensors fit in roughly half of free
    VRAM on ``device``, move them onto the GPU once so per-step indexing is
    a fused GPU op (no H2D copy per batch). Otherwise pin host memory so the
    trainer's ``.to(device, non_blocking=True)`` overlaps DMA with compute.
    """
    nbytes = sum(int(blob[k].element_size() * blob[k].numel()) for k in _BATCH_KEYS)
    on_device = False
    if prefer_device and torch.cuda.is_available() and str(device).startswith("cuda"):
        try:
            free, _total = torch.cuda.mem_get_info(torch.device(device))
            # Use up to half of free VRAM for the data blob; leave the rest
            # for phi/ref weights, activations, and optimizer state.
            if nbytes < free // 2:
                for k in _BATCH_KEYS:
                    blob[k] = blob[k].to(device, non_blocking=True)
                on_device = True
                print(
                    f"[value] preloaded blob ({nbytes/1e9:.2f} GB) onto {device}",
                    flush=True,
                )
        except Exception as e:  # pragma: no cover
            print(f"[value] WARN: blob device-preload failed ({e}); falling back to pinned host", flush=True)
    if not on_device:
        for k in _BATCH_KEYS:
            t = blob[k]
            if t.device.type == "cpu" and not t.is_pinned():
                try:
                    blob[k] = t.pin_memory()
                except RuntimeError:
                    pass  # e.g. no CUDA — leave as-is
        print(f"[value] pinned blob on host ({nbytes/1e9:.2f} GB)", flush=True)
    return blob


def _split_train_val(
    n_rows: int,
    group_size: int,
    val_fraction: float,
    seed: int,
    prompt_idx_per_row=None,
) -> Tuple[List[int], List[int]]:
    """Split row indices into train/val groups by prompt.

    Two modes:
    1. Default (prompt_idx_per_row=None): rollouts for the same prompt are
       contiguous in groups of ``group_size`` (preserved by the default
       ``collect_value_data.py`` mixed-outcome filter). Requires
       n_rows % group_size == 0.
    2. ``prompt_idx_per_row`` provided: a length-n_rows array giving the
       source prompt index for each row. Used by --paper-pairing-multi
       which yields a variable number of rows per prompt. Rows are still
       split at the prompt level (held-out prompts go entirely to val).
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(
            f"val_fraction must be in (0, 1), got {val_fraction}"
        )
    rng = random.Random(seed)

    if prompt_idx_per_row is not None:
        import numpy as _np
        pids = _np.asarray(prompt_idx_per_row).reshape(-1)
        if pids.shape[0] != n_rows:
            raise ValueError(
                f"prompt_idx_per_row length {pids.shape[0]} != n_rows {n_rows}"
            )
        unique_pids = sorted({int(x) for x in pids.tolist()})
        n_prompts = len(unique_pids)
        if n_prompts < 2:
            raise ValueError(
                f"need at least 2 unique prompts to split, got {n_prompts}"
            )
        prompt_perm = list(unique_pids)
        rng.shuffle(prompt_perm)
        n_val = max(1, min(n_prompts - 1, int(round(n_prompts * val_fraction))))
        val_pids = set(prompt_perm[-n_val:])
        train_rows: List[int] = []
        val_rows: List[int] = []
        for r in range(n_rows):
            if int(pids[r]) in val_pids:
                val_rows.append(r)
            else:
                train_rows.append(r)
        return train_rows, val_rows

    if group_size <= 0:
        raise ValueError(f"group_size must be > 0, got {group_size}")
    if n_rows % group_size != 0:
        raise ValueError(
            f"n_rows={n_rows} is not divisible by group_size={group_size}; "
            f"cfg.group_size likely disagrees with the value of group_size "
            f"used by collect_value_data.py. Check blob['config_snapshot']."
        )
    n_prompts = n_rows // group_size
    if n_prompts < 2:
        raise ValueError(
            f"need at least 2 prompts to split train/val, got {n_prompts} "
            f"(n_rows={n_rows}, group_size={group_size})"
        )
    prompt_perm = list(range(n_prompts))
    rng.shuffle(prompt_perm)
    n_val = max(1, min(n_prompts - 1, int(round(n_prompts * val_fraction))))
    val_prompts = set(prompt_perm[-n_val:])
    train_rows = []
    val_rows = []
    for p in range(n_prompts):
        rows = list(range(p * group_size, (p + 1) * group_size))
        if p in val_prompts:
            val_rows.extend(rows)
        else:
            train_rows.extend(rows)
    return train_rows, val_rows


@torch.no_grad()
def _eval_val_loss(
    model: PrefixValueModel,
    blob: dict,
    val_rows,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
) -> dict:
    """Compute average IPVRM loss on the val set. No grad.

    FSDP-collective: every rank computes loss on its shard of val rows
    (rows ``rank::world_size``) and the per-batch sums are all-reduced
    before normalizing. This is required because ``model.phi`` is
    FSDP-wrapped — its forward needs every rank to participate in the
    all-gather. Calling on rank 0 only deadlocks at FSDP's collective
    barrier inside the forward pass.

    Returns global means; the result is identical on every rank.
    """
    model.phi.eval()
    device = model.device
    blob_on_device = blob["prompt_ids"].device.type == device.type
    val_rows_t = (
        val_rows
        if torch.is_tensor(val_rows)
        else torch.as_tensor(val_rows, dtype=torch.long)
    )
    # Per-rank shard of val rows. Each rank does its own batches in lockstep
    # with the others (collective forwards must be matched count-for-count).
    if world_size > 1:
        rank_rows = val_rows_t[rank::world_size]
    else:
        rank_rows = val_rows_t
    n = rank_rows.numel()
    # All ranks must enter the same number of forward passes; pad the rank
    # with the smallest count up to the largest count by re-running its
    # last batch (the sums get all-reduced and divided by the global batch
    # count, so re-runs cancel out via simple averaging).
    local_batches = (n + batch_size - 1) // batch_size
    if world_size > 1:
        import torch.distributed as _dist
        local_batches_t = torch.tensor([local_batches], dtype=torch.long, device=device)
        _dist.all_reduce(local_batches_t, op=_dist.ReduceOp.MAX)
        global_max_batches = int(local_batches_t.item())
    else:
        global_max_batches = local_batches

    total_loss = torch.zeros(1, dtype=torch.float64, device=device)
    total_acc = torch.zeros(1, dtype=torch.float64, device=device)
    contributing = torch.zeros(1, dtype=torch.float64, device=device)

    for bi in range(global_max_batches):
        start = bi * batch_size
        if start < n:
            rows = rank_rows[start : start + batch_size]
            real = True
        else:
            # Re-run the last real batch so we participate in the collective
            # forward without contributing to the loss tally.
            rows = rank_rows[max(0, n - batch_size):n] if n > 0 else rank_rows[:0]
            if rows.numel() == 0:
                # No val data on this rank at all (rare). Use first global row
                # to avoid empty-batch errors; gate it out via ``real=False``.
                rows = val_rows_t[:1]
            real = False

        batch = _make_batch(rows, blob)
        if blob_on_device:
            prompt_ids = batch["prompt_ids"]
            prompt_mask = batch["prompt_mask"]
            response_ids = batch["response_ids"]
            response_mask = batch["response_mask"]
            outcomes = batch["outcomes"].float()
        else:
            prompt_ids = batch["prompt_ids"].to(device, non_blocking=True)
            prompt_mask = batch["prompt_mask"].to(device, non_blocking=True)
            response_ids = batch["response_ids"].to(device, non_blocking=True)
            response_mask = batch["response_mask"].to(device, non_blocking=True)
            outcomes = batch["outcomes"].to(device, non_blocking=True).float()
        out = model(prompt_ids, prompt_mask, response_ids, response_mask)
        loss, stats = ipvrm_loss(
            log_ratio=out["log_ratio"],
            response_mask=response_mask,
            outcomes=outcomes,
            margin=model.margin,
        )
        if real:
            total_loss += float(stats["loss"])
            total_acc += float(stats["acc_at_last"])
            contributing += 1.0

    if world_size > 1:
        import torch.distributed as _dist
        _dist.all_reduce(total_loss, op=_dist.ReduceOp.SUM)
        _dist.all_reduce(total_acc, op=_dist.ReduceOp.SUM)
        _dist.all_reduce(contributing, op=_dist.ReduceOp.SUM)

    model.phi.train()
    denom = float(contributing.item()) if contributing.item() > 0 else 1.0
    return {
        "val_loss": float(total_loss.item()) / denom,
        "val_acc_at_last": float(total_acc.item()) / denom,
        "val_n_batches": int(contributing.item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--data", type=str, default=None,
                    help="path to .pt produced by collect_value_data.py; "
                         "defaults to cfg.value_data_path")
    args = ap.parse_args()

    cfg = CASPOConfig.from_yaml(args.config)
    cfg = apply_overrides(cfg, args.override)

    # ---- distributed init ----
    # Single-process invocations (no torchrun-style env, WORLD_SIZE<=1) leave
    # init_process_group as a no-op and dist_info.is_distributed == False.
    # Under our per-rank bash launcher, RANK/WORLD_SIZE/MASTER_* are set and
    # init_process_group runs once.
    env = distributed_env()
    if env.is_distributed and cfg.distributed_backend != "fsdp":
        raise ValueError(
            f"WORLD_SIZE={env.world_size} but distributed_backend="
            f"{cfg.distributed_backend!r}; value training requires "
            f"distributed_backend='fsdp' for multi-rank runs."
        )
    dist_info = init_distributed_if_needed()
    is_main = dist_info.is_main
    world_size = dist_info.world_size
    rank = dist_info.rank
    # Resolve a rank-local device so CUDA_VISIBLE_DEVICES + LOCAL_RANK=0 in
    # the per-rank launcher (each rank sees one GPU as cuda:0) still maps
    # correctly here.
    rank_device = resolve_device(cfg.device, dist_info)
    cfg.device = str(rank_device)

    def _rprint(*a, **kw):
        if is_main:
            print(*a, **kw)

    data_path = args.data or cfg.value_data_path
    if not data_path or not os.path.exists(data_path):
        raise FileNotFoundError(
            f"value training data not found at {data_path!r}; "
            f"set --data or cfg.value_data_path"
        )
    blob = torch.load(data_path, map_location="cpu", weights_only=False)
    N = blob["prompt_ids"].shape[0]
    pos_frac = float(blob["outcomes"].mean().item())
    _rprint(f"[value] loaded {N} (prompt,response,outcome) rows from {data_path}", flush=True)
    _rprint(f"[value] positive fraction = {pos_frac:.3f}", flush=True)
    if world_size > 1:
        _rprint(f"[value] distributed: rank {rank}/{world_size} on {rank_device}", flush=True)
    # Move/pin once to avoid per-step PCIe traffic. Off by env var if a user
    # explicitly wants to keep VRAM headroom: CASPO_VALUE_NO_DEVICE_BLOB=1.
    _prefer_dev = os.environ.get("CASPO_VALUE_NO_DEVICE_BLOB", "0") not in ("1", "true", "True")
    blob = _prepare_blob(blob, cfg.device, prefer_device=_prefer_dev)

    # ---- train/val split (by prompt) ----
    # Prefer the group_size that was actually used at collect time (stored in
    # the blob's config_snapshot) over cfg.group_size, since they can diverge
    # if the user re-runs train_value with a different config. Splitting under
    # the wrong G silently leaks rollouts of the same prompt across train/val.
    G = max(1, int(cfg.group_size))
    snap = blob.get("config_snapshot") if isinstance(blob, dict) else None
    if isinstance(snap, dict) and "group_size" in snap:
        snap_G = int(snap["group_size"])
        if snap_G != G:
            print(
                f"[value] WARN: cfg.group_size={G} differs from collect-time "
                f"group_size={snap_G}; using collect-time value to preserve "
                f"per-prompt grouping for the train/val split.",
                flush=True,
            )
            G = max(1, snap_G)
    val_fraction = float(cfg.value_val_fraction)
    # If n_rows isn't divisible by G (e.g. --paper-pairing-multi yields a
    # variable number of rows per prompt), derive prompt identity from the
    # prompt_ids tensor — all rows from the same prompt share identical
    # prompt tokens. unique(dim=0) returns inverse indices we use as prompt
    # ids for the split.
    prompt_idx_per_row = None
    if N % G != 0:
        _rprint(
            f"[value] n_rows={N} not divisible by G={G}; deriving prompt ids "
            f"from prompt_ids tensor uniqueness for the train/val split.",
            flush=True,
        )
        _, prompt_inv = torch.unique(blob["prompt_ids"], dim=0, return_inverse=True)
        prompt_idx_per_row = prompt_inv.cpu().tolist()
    train_rows, val_rows = _split_train_val(
        N, G, val_fraction, seed=int(cfg.seed),
        prompt_idx_per_row=prompt_idx_per_row,
    )
    full_train_count = len(train_rows)
    _rprint(
        f"[value] train: {full_train_count} rollouts ({full_train_count//G} prompts), "
        f"val: {len(val_rows)} rollouts ({len(val_rows)//G} prompts) "
        f"(val_fraction={val_fraction:.2f})",
        flush=True,
    )

    # ---- per-rank dataset sharding ----
    # Rank N takes train rows N::world_size. Validation stays on rank 0 only
    # (the eval pass + best-checkpoint logic runs only there).
    if world_size > 1:
        train_rows = train_rows[rank::world_size]
        _rprint(
            f"[value] sharded train: rank 0 sees {len(train_rows)} of "
            f"{full_train_count} rollouts (stride={world_size})",
            flush=True,
        )

    # ---- step budget ----
    bs = max(1, int(cfg.value_micro_batch_size) * int(cfg.value_grad_accum_steps))
    steps_per_epoch = max(1, math.ceil(len(train_rows) / bs))
    if int(cfg.value_max_epochs) > 0:
        max_steps = int(cfg.value_max_epochs) * steps_per_epoch
        _rprint(
            f"[value] step budget: {cfg.value_max_epochs} epochs × "
            f"{steps_per_epoch} steps/epoch = {max_steps} steps "
            f"(batch={bs})",
            flush=True,
        )
    else:
        max_steps = int(cfg.value_max_steps)
        _rprint(f"[value] step budget: {max_steps} (raw)", flush=True)
    if max_steps <= 0:
        raise ValueError(
            f"max_steps={max_steps} ≤ 0; set value_max_epochs > 0 or "
            f"value_max_steps > 0."
        )

    # ---- model + trainer ----
    # Construct phi+ref on each rank and move to the rank-local device BEFORE
    # the trainer wraps phi with FSDP. After FSDP wrap, phi's flat-params are
    # sharded across world_size; ref stays full-resident (frozen, no opt).
    model = PrefixValueModel(cfg)
    model.to(cfg.device)
    trainer = PrefixValueTrainer(cfg, model)

    if is_main:
        os.makedirs(cfg.output_dir, exist_ok=True)
    if is_dist_initialized():
        import torch.distributed as _dist
        _dist.barrier()
    log_path = os.path.join(cfg.output_dir, "value_train_log.jsonl")
    log_file = open(log_path, "a", buffering=1) if is_main else None

    # ---- training loop with early stopping on val loss ----
    best_val_loss = float("inf")
    best_step = 0
    no_improve_evals = 0
    best_path = os.path.join(cfg.output_dir, "best")
    final_path = os.path.join(cfg.output_dir, "final")
    # Clear any stale best/ from a prior run so the final-copy step below
    # cannot accidentally promote a checkpoint that was never validated by
    # this run's eval loop. Rank 0 only — other ranks must not race to delete.
    if is_main and os.path.isdir(best_path):
        shutil.rmtree(best_path)
    if is_dist_initialized():
        import torch.distributed as _dist
        _dist.barrier()

    # Pre-build a deterministic per-epoch row permutation iterator. This
    # replaces ``rng.sample(train_rows, k=bs)`` per step (Python-list copy +
    # K random draws) with a single torch.randperm + tensor slicing per
    # epoch. The permutation seed advances each epoch from cfg.seed so runs
    # are reproducible and successive epochs see different orders.
    train_rows_t = torch.as_tensor(train_rows, dtype=torch.long)
    val_rows_t = torch.as_tensor(val_rows, dtype=torch.long)
    if blob["prompt_ids"].device.type != "cpu":
        train_rows_t = train_rows_t.to(blob["prompt_ids"].device)
        val_rows_t = val_rows_t.to(blob["prompt_ids"].device)
    n_train = train_rows_t.numel()

    def _epoch_iter(epoch_idx: int):
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(cfg.seed) + epoch_idx)
        perm = torch.randperm(n_train, generator=gen)
        if train_rows_t.device.type != "cpu":
            perm = perm.to(train_rows_t.device)
        shuffled = train_rows_t.index_select(0, perm)
        for s in range(0, n_train, bs):
            yield shuffled[s : s + bs]

    t0 = time.time()
    last_step_done = 0
    early_stopped = False
    epoch_idx = 0
    epoch_iter = _epoch_iter(epoch_idx)
    for step in range(1, max_steps + 1):
        try:
            rows = next(epoch_iter)
        except StopIteration:
            epoch_idx += 1
            epoch_iter = _epoch_iter(epoch_idx)
            rows = next(epoch_iter)
        batch = _make_batch(rows, blob)
        stats = trainer.step(batch)
        last_step_done = step

        if cfg.value_log_every and (step % cfg.value_log_every == 0) and is_main:
            elapsed = time.time() - t0
            msg = (
                f"[value step {step}/{max_steps}] "
                f"loss={stats['loss']:.4f} acc={stats['acc_at_last']:.3f} "
                f"v̄+={stats['mean_v_bar_pos']:.3f} v̄-={stats['mean_v_bar_neg']:.3f} "
                f"lr={stats['lr']:.2e} t={elapsed:.1f}s"
            )
            print(msg, flush=True)
            log_file.write(json.dumps({"step": step, **stats}) + "\n")

        # ---- val eval + early stopping ----
        # value_eval_every <= 0 disables eval (and therefore early stopping
        # and best-checkpoint tracking — falls through to final-save branch).
        # Validation is COLLECTIVE under FSDP: every rank participates in
        # _eval_val_loss (each rank does its val-row shard), and the loss is
        # all-reduced so every rank ends with the identical global mean.
        # The decision (best-step, early-stop) is then computed identically
        # on every rank, no broadcast needed.
        if int(cfg.value_eval_every) > 0 and (step % int(cfg.value_eval_every) == 0):
            do_break = False
            val_stats = _eval_val_loss(
                model, blob, val_rows_t, batch_size=bs,
                rank=rank, world_size=world_size,
            )
            if val_stats["val_loss"] < best_val_loss - 1e-6:
                best_val_loss = val_stats["val_loss"]
                best_step = step
                no_improve_evals = 0
                save_best_now_flag = True
            else:
                no_improve_evals += 1
                save_best_now_flag = False
                if (
                    no_improve_evals >= int(cfg.value_early_stop_patience)
                    and step >= int(cfg.value_early_stop_min_steps)
                ):
                    early_stopped = True
                    do_break = True
            if is_main:
                print(
                    f"[value step {step}] val_loss={val_stats['val_loss']:.4f} "
                    f"val_acc={val_stats['val_acc_at_last']:.3f} "
                    f"(best={best_val_loss:.4f} @ step {best_step})",
                    flush=True,
                )
                log_file.write(json.dumps({"step": step, **val_stats}) + "\n")
                if early_stopped:
                    print(
                        f"[value] early stop at step {step} "
                        f"(no val improvement for {no_improve_evals} evals; "
                        f"best={best_val_loss:.4f} @ step {best_step})",
                        flush=True,
                    )

            # _eval_val_loss is collective: every rank ends with the same
            # global mean val_loss, so best_step / no_improve_evals /
            # do_break advance identically on every rank. No broadcast
            # needed; just rely on the shared computation. save_best_now
            # is True on every rank when this step beat the prior best.
            save_best_now = save_best_now_flag

            if save_best_now:
                # All ranks must enter save_pretrained together because FSDP's
                # FULL_STATE_DICT context is collective. Only rank 0 writes.
                model.save_pretrained(best_path)
                if is_main:
                    with open(os.path.join(best_path, "step.json"), "w") as f:
                        json.dump(
                            {
                                "step": step,
                                "val_loss": best_val_loss,
                                "val_acc_at_last": (
                                    val_stats["val_acc_at_last"]
                                    if is_main else None
                                ),
                            },
                            f,
                        )

            if do_break:
                break

        # ---- periodic checkpointing every value_save_every steps ----
        # All ranks must enter save_pretrained together (FSDP FULL_STATE_DICT
        # is collective). Triggered independently of eval so trajectories can
        # be reconstructed even with eval disabled.
        if (
            int(cfg.value_save_every) > 0
            and step > 0
            and step % int(cfg.value_save_every) == 0
        ):
            step_path = os.path.join(cfg.output_dir, f"step_{step}")
            model.save_pretrained(step_path)
            if is_main:
                _step_meta = {"step": step}
                # Add the most recent val_loss/val_acc if we have them.
                if 'val_stats' in dir():
                    _step_meta["val_loss"] = float(val_stats["val_loss"])
                    _step_meta["val_acc_at_last"] = float(
                        val_stats["val_acc_at_last"]
                    )
                with open(os.path.join(step_path, "step.json"), "w") as f:
                    json.dump(_step_meta, f)
                print(f"[value step {step}] saved checkpoint to {step_path}",
                      flush=True)

    # ---- finalize: copy best → final, or save current if no eval happened ----
    # save_pretrained on FSDP-wrapped phi is collective: every rank MUST enter
    # the FULL_STATE_DICT context together, even though only rank 0 writes
    # files. So we determine "do we need to save current?" on every rank by
    # broadcasting from rank 0, then call save_pretrained collectively.
    have_best = is_main and os.path.isdir(best_path)
    if is_dist_initialized():
        import torch.distributed as _dist
        flag = torch.tensor([1 if have_best else 0], dtype=torch.long, device=rank_device)
        _dist.broadcast(flag, src=0)
        have_best = bool(flag[0].item())

    if is_main and os.path.isdir(final_path):
        shutil.rmtree(final_path)
    if is_dist_initialized():
        import torch.distributed as _dist
        _dist.barrier()

    if have_best:
        if is_main:
            shutil.copytree(best_path, final_path)
            print(
                f"[value] done. best (val_loss={best_val_loss:.4f} @ step {best_step}) "
                f"copied to {final_path}",
                flush=True,
            )
    else:
        # Either val_eval_every <= 0 or training never reached an eval step.
        # Collective on every rank; rank 0 writes step.json.
        model.save_pretrained(final_path)
        if is_main:
            with open(os.path.join(final_path, "step.json"), "w") as f:
                json.dump({"step": last_step_done}, f)
            print(
                f"[value] done. final (no val eval ran) saved to {final_path}",
                flush=True,
            )

    if is_dist_initialized():
        import torch.distributed as _dist
        _dist.barrier()
    final_exists = is_main and os.path.isdir(final_path)
    if is_dist_initialized():
        import torch.distributed as _dist
        flag = torch.tensor([1 if final_exists else 0], dtype=torch.long, device=rank_device)
        _dist.broadcast(flag, src=0)
        final_exists = bool(flag[0].item())
    if not final_exists:
        # Defensive: if both branches somehow skipped (e.g. max_steps=0), at
        # least save the current model so downstream phases don't crash.
        model.save_pretrained(final_path)
        if is_main:
            with open(os.path.join(final_path, "step.json"), "w") as f:
                json.dump({"step": last_step_done}, f)
            print(
                f"[value] WARN: no checkpoint produced by main path; "
                f"saved current model to {final_path}",
                flush=True,
            )

    # All remaining metadata writes (run config snapshot, value_final symlink,
    # training_summary.json) are rank-0 only — non-rank-0 processes exit here.
    if not is_main:
        if log_file is not None:
            log_file.close()
        if is_dist_initialized():
            import torch.distributed as _dist
            _dist.barrier()
            _dist.destroy_process_group()
        return

    # Always snapshot the run config alongside (separate name from HF config.json).
    with open(os.path.join(final_path, "caspo_run_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    # Also create a value_final/ symlink pointing at final/ so configs that
    # set prefix_value_path=<output_dir>/value_final find the trained V_φ
    # without forcing the user to know train_value.py's "final" naming.
    # Refresh the symlink unconditionally — a stale symlink from a prior run
    # could otherwise point at a deleted/old final/ and silently shadow this
    # run's checkpoint.
    value_final_link = os.path.join(cfg.output_dir, "value_final")
    try:
        if os.path.islink(value_final_link):
            # Replace any prior symlink (broken or otherwise) — re-pointing to
            # this run's final/ is the whole reason we maintain the symlink.
            os.unlink(value_final_link)
        elif os.path.lexists(value_final_link):
            # A real file/dir at this path: only a real directory is preserved
            # (user may have manually saved a value_final/ dir). A stray regular
            # file would otherwise make the os.symlink call below raise
            # FileExistsError silently caught as a warning.
            if os.path.isdir(value_final_link):
                print(
                    f"[value] WARN: {value_final_link} is a real directory, "
                    f"not a symlink; leaving it untouched. Downstream phases "
                    f"may pick up a stale checkpoint.",
                    flush=True,
                )
                value_final_link = None  # skip recreate
            else:
                # Regular file or other non-dir entry — safe to remove since
                # train_value never writes anything there itself.
                os.unlink(value_final_link)
        if value_final_link is not None:
            os.symlink(os.path.basename(final_path), value_final_link)
            print(f"[value] created symlink {value_final_link} → final/", flush=True)
    except OSError as e:
        print(f"[value] WARN: could not create value_final symlink: {e}", flush=True)
    with open(os.path.join(final_path, "training_summary.json"), "w") as f:
        json.dump(
            {
                "max_steps": max_steps,
                "last_step_done": last_step_done,
                "early_stopped": early_stopped,
                "best_val_loss": best_val_loss if best_val_loss < float("inf") else None,
                "best_step": best_step if best_step > 0 else None,
                "n_train_rollouts": len(train_rows),
                "n_val_rollouts": len(val_rows),
                "steps_per_epoch": steps_per_epoch,
                "value_max_epochs": int(cfg.value_max_epochs),
            },
            f,
            indent=2,
        )
    if log_file is not None:
        log_file.close()
    # Tear down the process group cleanly so the launcher doesn't see hangs.
    if is_dist_initialized():
        import torch.distributed as _dist
        _dist.barrier()
        _dist.destroy_process_group()


if __name__ == "__main__":
    main()
