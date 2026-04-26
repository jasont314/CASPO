"""Step-level value & TD advantage construction with token broadcast.

Implements:
- ``step_values_from_log_ratios``: V_phi(s_t) at every step boundary, defined
  as the cumulative β-scaled log-ratio sum up to (but not including) step t.
  This matches Eq. 5 of IPVRM (arXiv:2604.13197), with V at the *start* of a
  step equal to the running sum of log-ratios over preceding tokens.
- ``step_td_advantage``: one-step TD advantage at the step granularity, after
  VinePPO Eq. 6. Only the terminal step receives the outcome reward; γ is
  configurable (1.0 by default).
- ``standardize_step_advantage``: whitening over batch / group / off scopes.
- ``broadcast_step_advantage_to_tokens``: every token in step t carries
  A_step[b, t]; masked positions are zeroed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def step_values_from_log_ratios(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    boundary_after: torch.Tensor,
    step_count: torch.Tensor,
) -> torch.Tensor:
    """Compute V_phi at every step boundary.

    Args:
        log_ratio: ``[B, R]``; β-scaled log(π_φ/π_ref) per response token, 0 at
            padding.
        response_mask: ``[B, R]`` indicator on valid response tokens.
        boundary_after: ``[B, R]`` bool; True at the LAST token of each step
            (so a row with S steps has exactly S True entries).
        step_count: ``[B]`` long; number of steps per row.

    Returns:
        ``[B, S_max + 1]`` float tensor where ``V[b, 0] = 0`` and
        ``V[b, t]`` (for ``1 <= t <= step_count[b]``) is the cumulative
        log-ratio sum over tokens belonging to steps 0..t-1, i.e. V at the
        START of step t. Positions beyond ``step_count[b]`` are filled with
        ``V[b, step_count[b]]`` (constant tail).
    """
    if log_ratio.dim() != 2 or response_mask.dim() != 2 or boundary_after.dim() != 2:
        raise ValueError("log_ratio/response_mask/boundary_after must be 2D")
    if log_ratio.shape != response_mask.shape or log_ratio.shape != boundary_after.shape:
        raise ValueError(
            f"shape mismatch: log_ratio {tuple(log_ratio.shape)}, "
            f"response_mask {tuple(response_mask.shape)}, "
            f"boundary_after {tuple(boundary_after.shape)}"
        )
    if step_count.dim() != 1 or step_count.shape[0] != log_ratio.shape[0]:
        raise ValueError(
            f"step_count must be 1D with B={log_ratio.shape[0]} entries, got {tuple(step_count.shape)}"
        )

    B, R = log_ratio.shape
    device = log_ratio.device
    dtype = log_ratio.dtype

    # Defensively colocate auxiliary tensors with log_ratio. Mismatched devices
    # or boolean dtype on response_mask are common upstream footguns.
    if response_mask.device != device:
        response_mask = response_mask.to(device)
    if boundary_after.device != device:
        boundary_after = boundary_after.to(device)
    if step_count.device != device:
        step_count = step_count.to(device)
    if boundary_after.dtype != torch.bool:
        boundary_after = boundary_after.bool()

    # Empty-batch / empty-response short-circuits.
    if B == 0:
        return torch.zeros((0, 1), device=device, dtype=dtype)
    if R == 0:
        # No response tokens at all → V_step is all zeros for every row.
        # We must still respect step_count: if any row claims steps, that's a
        # caller bug, but we choose to return zeros rather than crash.
        return torch.zeros((B, 1), device=device, dtype=dtype)

    # Defensively zero out anything outside the response mask, so cumsum is
    # correct regardless of what was placed at padding positions. Cast through
    # float32 to avoid bf16 rounding accumulating in long sequences; we cast
    # the final result back to the input dtype.
    use_float32 = dtype in (torch.bfloat16, torch.float16)
    compute_dtype = torch.float32 if use_float32 else dtype
    # One mask cast (in log_ratio's dtype), then a single upcast on the
    # already-masked product. Avoids materializing a fp32 copy of
    # ``log_ratio`` *and* a fp32 copy of ``response_mask`` simultaneously
    # — multiplying first in the input dtype and casting the small product
    # (``[B, R]``) once is cheaper than casting both [B, R] inputs.
    masked_lr = log_ratio * response_mask.to(log_ratio.dtype)
    if masked_lr.dtype != compute_dtype:
        masked_lr = masked_lr.to(compute_dtype)

    # cumsum[b, k] is the running sum INCLUDING token k. If k is the last
    # token of step t-1, cumsum[b, k] is V at the START of step t.
    cumsum = masked_lr.cumsum(dim=1)

    # ``step_count.max()`` on an empty tensor raises; B==0 was handled above.
    S_max = int(step_count.max().item())

    if S_max == 0:
        # Every row has zero steps — return [B, 1] of zeros.
        return torch.zeros((B, 1), device=device, dtype=dtype)

    # Build a [B, S_max] tensor of "last-token-of-step-t" indices. For rows
    # with fewer than S_max steps, pad with the last valid index so that
    # gather produces a constant tail equal to V[step_count[b]] = total
    # cumsum over the response. We use a per-row last-valid index that falls
    # back to 0 if the row has zero steps.
    rows, cols = torch.where(boundary_after)
    # Number of boundary entries per row should equal step_count[b].
    # We rely on the caller for that contract; if violated the gather indices
    # below will simply not match step_count.

    boundary_idx = log_ratio.new_zeros((B, S_max), dtype=torch.long)
    n = rows.numel()
    if n > 0:
        # Compute the 0-based position of each (row, col) entry within its
        # row. ``torch.where`` returns indices in row-major order so all
        # entries with the same row are contiguous and sorted by col, hence
        # ``rows`` is non-decreasing.
        if n == 1:
            in_row_pos = torch.zeros_like(rows)
        else:
            # row_changes[i] = 1 iff a new row starts at position i (i==0 or
            # rows[i] != rows[i-1]). block_id is the 0-based per-row block
            # id. block_starts is the global index where each block starts,
            # indexed by block_id. in_row_pos = global_idx - block_starts[block_id].
            change_mask = rows[1:] != rows[:-1]
            row_changes = torch.empty_like(rows)
            row_changes[0] = 1
            row_changes[1:] = change_mask.long()
            block_id = row_changes.cumsum(dim=0) - 1  # [n]
            global_idx = torch.arange(n, device=device, dtype=torch.long)
            block_starts = global_idx[row_changes.bool()]
            in_row_pos = global_idx - block_starts[block_id]

        boundary_idx[rows, in_row_pos] = cols

    # Pad rows that have fewer than S_max steps with the last valid index.
    # For row b: valid_count = step_count[b]; positions [valid_count..S_max-1]
    # should be filled with boundary_idx[b, valid_count - 1] (the last real
    # boundary). If valid_count == 0, fill with 0 (but V_step[b, 1:] will be
    # zeroed via cumsum at index 0 → masked_lr position 0, which itself is
    # masked; we further override below).
    arange_S = torch.arange(S_max, device=device).unsqueeze(0)  # [1, S_max]
    valid_S = arange_S < step_count.unsqueeze(1)  # [B, S_max]
    last_valid_idx = (step_count - 1).clamp(min=0)  # [B]
    fill_idx = boundary_idx.gather(1, last_valid_idx.unsqueeze(1)).expand(-1, S_max)
    boundary_idx = torch.where(valid_S, boundary_idx, fill_idx)

    # gather V at end-of-step boundaries. cumsum is [B, R], boundary_idx is
    # [B, S_max] with all entries in [0, R-1] (or 0 for zero-step rows).
    # Clamp to be safe (zero-step rows we handle by zeroing below).
    safe_idx = boundary_idx.clamp(min=0, max=R - 1)
    V_after_step = cumsum.gather(1, safe_idx)  # [B, S_max]

    # Prepend V_step[:, 0] = 0.
    V_step = torch.cat(
        [torch.zeros((B, 1), device=device, dtype=compute_dtype), V_after_step],
        dim=1,
    )

    # For rows with step_count == 0, the gather may have used index 0 which
    # corresponds to a (potentially) masked first token; force that row to 0.
    zero_step_rows = step_count == 0
    if zero_step_rows.any():
        V_step[zero_step_rows] = 0

    if V_step.dtype != dtype:
        V_step = V_step.to(dtype)

    return V_step


def step_td_advantage(
    V_step: torch.Tensor,
    final_reward: torch.Tensor,
    step_count: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """One-step TD advantage at step granularity (VinePPO Eq. 6).

    ``A_step[b, t] = r_step[b, t] + γ * V_step[b, t+1] - V_step[b, t]``,
    with ``r_step[b, t] = final_reward[b]`` iff ``t == step_count[b] - 1``,
    else 0. Positions ``t >= step_count[b]`` are zeroed.

    Args:
        V_step: ``[B, S_max + 1]`` from :func:`step_values_from_log_ratios`.
        final_reward: ``[B]`` scalar reward per row.
        step_count: ``[B]`` number of steps.
        gamma: discount factor.

    Returns:
        ``[B, S_max]`` advantages.
    """
    if V_step.dim() != 2:
        raise ValueError(f"V_step must be 2D, got {tuple(V_step.shape)}")
    B, S_plus_1 = V_step.shape
    S_max = S_plus_1 - 1
    if S_max < 0:
        raise ValueError(f"V_step must have at least one column, got {S_plus_1}")
    if final_reward.shape != (B,):
        raise ValueError(
            f"final_reward must be shape ({B},), got {tuple(final_reward.shape)}"
        )
    if step_count.shape != (B,):
        raise ValueError(
            f"step_count must be shape ({B},), got {tuple(step_count.shape)}"
        )

    device = V_step.device
    dtype = V_step.dtype

    # Colocate companion tensors with V_step.
    if final_reward.device != device:
        final_reward = final_reward.to(device)
    if step_count.device != device:
        step_count = step_count.to(device)

    if B == 0:
        return torch.zeros((0, max(S_max, 0)), device=device, dtype=dtype)

    if S_max == 0:
        return torch.zeros((B, 0), device=device, dtype=dtype)

    # Sanity-check gamma. NaN gamma would silently corrupt all advantages.
    if not (gamma == gamma):  # NaN check without importing math
        raise ValueError("gamma must be finite")

    # r_step[b, t] = final_reward[b] iff t == step_count[b] - 1.
    # Compute in float32 when V_step is reduced precision to avoid the TD
    # difference V[t+1] - V[t] losing significant digits on saturated batches
    # (where adjacent V are very close).
    use_float32 = dtype in (torch.bfloat16, torch.float16)
    compute_dtype = torch.float32 if use_float32 else dtype

    V_compute = V_step.to(compute_dtype) if V_step.dtype != compute_dtype else V_step

    # Zero-init r_step in compute_dtype; scatter the final reward only at the
    # terminal step of each row that has at least one step.
    r_step = V_compute.new_zeros((B, S_max))
    has_steps = step_count > 0
    if has_steps.any():
        # terminal_idx for zero-step rows is clamped to 0; we mask them via
        # ``valid`` below, so the scatter target is harmless.
        terminal_idx = (step_count - 1).clamp(min=0, max=S_max - 1)
        # Only scatter for rows that have steps to avoid spurious writes that
        # later need to be zeroed (saves a boolean-indexed write).
        if has_steps.all():
            r_step.scatter_(
                1, terminal_idx.unsqueeze(1),
                final_reward.to(compute_dtype).unsqueeze(1),
            )
        else:
            # Use src masked to 0 for empty rows; single fused scatter.
            src = (final_reward.to(compute_dtype) * has_steps.to(compute_dtype)).unsqueeze(1)
            r_step.scatter_(1, terminal_idx.unsqueeze(1), src)

    A = r_step + float(gamma) * V_compute[:, 1:] - V_compute[:, :-1]

    arange_S = torch.arange(S_max, device=device).unsqueeze(0)  # [1, S_max]
    valid = arange_S < step_count.unsqueeze(1)
    # Multiplicative mask is one kernel vs torch.where's two-input compare-select.
    A = A * valid.to(A.dtype)

    if A.dtype != dtype:
        A = A.to(dtype)
    return A


def transform_step_values_for_advantage(
    V_step: torch.Tensor,
    transform: str = "value",
) -> torch.Tensor:
    """Apply a CASPO ablation transform before step-TD advantages.

    The IPVRM value model emits an unbounded cumulative log-ratio value. For
    CASPO ablations, TD can be computed on that direct value, its
    success-probability interpretation, or its log-probability interpretation.
    The terminal verifier reward in ``step_td_advantage`` is unchanged; only
    the prefix value scale is changed here.
    """
    if transform == "value":
        return V_step
    if transform == "prob":
        compute = V_step.float() if V_step.dtype in (torch.float16, torch.bfloat16) else V_step
        out = torch.sigmoid(compute)
        return out.to(V_step.dtype) if out.dtype != V_step.dtype else out
    if transform == "logprob":
        compute = V_step.float() if V_step.dtype in (torch.float16, torch.bfloat16) else V_step
        out = F.logsigmoid(compute)
        return out.to(V_step.dtype) if out.dtype != V_step.dtype else out
    raise ValueError(
        f"unknown CASPO advantage transform {transform!r}; "
        "must be 'value', 'prob', or 'logprob'"
    )


def standardize_step_advantage(
    A_step: torch.Tensor,
    step_count: torch.Tensor,
    *,
    scope: str = "batch",
    group_size: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Whiten step-level advantages over the chosen scope.

    Args:
        A_step: ``[B, S_max]``.
        step_count: ``[B]``.
        scope: ``"batch"``, ``"group"`` or ``"off"``.
        group_size: required when ``scope == "group"``.
        eps: zero-variance threshold; below it we leave the values unchanged.

    Returns:
        Tensor with the same shape as ``A_step``. Invalid positions
        (``t >= step_count[b]``) remain 0.
    """
    if A_step.dim() != 2:
        raise ValueError(f"A_step must be 2D, got {tuple(A_step.shape)}")

    if scope == "off":
        return A_step.clone()

    B, S_max = A_step.shape
    device = A_step.device
    dtype = A_step.dtype

    if step_count.device != device:
        step_count = step_count.to(device)
    if step_count.shape != (B,):
        raise ValueError(
            f"step_count must be shape ({B},), got {tuple(step_count.shape)}"
        )
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps}")

    # Empty edge cases — nothing to standardize, return a clean copy.
    if B == 0 or S_max == 0:
        return A_step.clone()

    # Always reduce in float32 for numerical stability — the variance computation
    # on bf16 saturated batches can collapse to zero spuriously.
    use_float32 = dtype in (torch.bfloat16, torch.float16)
    compute_dtype = torch.float32 if use_float32 else dtype
    A_compute = A_step.to(compute_dtype) if A_step.dtype != compute_dtype else A_step

    arange_S = torch.arange(S_max, device=device).unsqueeze(0)
    valid = arange_S < step_count.unsqueeze(1)  # [B, S_max] bool
    fmask = valid.to(compute_dtype)

    if scope == "batch":
        masked = A_compute * fmask
        denom = fmask.sum().clamp(min=1.0)
        mean = masked.sum() / denom
        # Reuse the same fmask-multiplied tensor for variance: subtracting
        # mean*fmask preserves zero at invalid slots while contributing
        # (a - mean)^2 only at valid slots. This avoids one extra broadcast
        # multiply versus ((a - mean)**2 * fmask).
        centered = masked - mean * fmask
        var = (centered * centered).sum() / denom
        std = var.clamp(min=0.0).sqrt()
        # Tensor-side comparison (no host sync). If the batch has only one
        # valid entry we'd still get std==0 and fall through to a clone.
        if (std <= eps).item():
            return A_step.clone()
        # Multiplicative mask: zeros invalid slots and avoids creating a
        # `zeros_like` temporary. fmask is already in compute_dtype.
        out = (A_compute - mean) / std * fmask
        # Guard against any residual non-finite slipping through from the
        # input (e.g. NaN log-ratios upstream).
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.to(dtype) if out.dtype != dtype else out

    if scope == "group":
        if group_size <= 0 or B % group_size != 0:
            raise ValueError(
                f"group_size {group_size} must divide B={B} and be positive"
            )
        G = group_size
        num_prompts = B // G
        # Use reshape rather than view: the caller might hand us a non-
        # contiguous A_step (e.g. a sliced view), and view would crash.
        a = A_compute.reshape(num_prompts, G, S_max)
        m = fmask.reshape(num_prompts, G, S_max)
        v = valid.reshape(num_prompts, G, S_max)
        denom = m.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
        masked = a * m
        mean = masked.sum(dim=(1, 2), keepdim=True) / denom
        centered = masked - mean * m
        var = (centered * centered).sum(dim=(1, 2), keepdim=True) / denom
        std = var.clamp(min=0.0).sqrt()
        # Where the group has zero variance, fall back to identity (after
        # zeroing invalid positions).
        zero_var = std <= eps  # [num_prompts, 1, 1] bool
        safe_std = torch.where(zero_var, torch.ones_like(std), std)
        normalized = (a - mean) / safe_std
        # On zero-var groups, fall back to the (pre-cast) compute values.
        normalized = torch.where(zero_var.expand_as(a), a, normalized)
        # Multiplicative mask zeros invalid slots in one fused kernel.
        normalized = normalized * m
        normalized = torch.nan_to_num(
            normalized, nan=0.0, posinf=0.0, neginf=0.0
        )
        out = normalized.reshape(B, S_max)
        return out.to(dtype) if out.dtype != dtype else out

    raise ValueError(f"unknown scope {scope!r}; must be 'batch', 'group', or 'off'")


def broadcast_step_advantage_to_tokens(
    A_step: torch.Tensor,
    step_id: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Broadcast a per-step advantage to per-token.

    Args:
        A_step: ``[B, S_max]``.
        step_id: ``[B, R]`` long; token's step index, ``-1`` on masked
            positions.
        response_mask: ``[B, R]``.

    Returns:
        ``[B, R]`` float, with masked / step-id-(-1) positions = 0.
    """
    if A_step.dim() != 2 or step_id.dim() != 2 or response_mask.dim() != 2:
        raise ValueError("A_step / step_id / response_mask must all be 2D")
    if step_id.shape != response_mask.shape:
        raise ValueError(
            f"step_id {tuple(step_id.shape)} != response_mask {tuple(response_mask.shape)}"
        )
    if A_step.shape[0] != step_id.shape[0]:
        raise ValueError(
            f"batch mismatch: A_step {A_step.shape[0]} vs step_id {step_id.shape[0]}"
        )

    device = A_step.device
    dtype = A_step.dtype
    B, R = step_id.shape
    S_max = A_step.shape[1]

    # Colocate inputs.
    if step_id.device != device:
        step_id = step_id.to(device)
    if response_mask.device != device:
        response_mask = response_mask.to(device)

    # Empty-shape short-circuits.
    if B == 0 or R == 0:
        return torch.zeros((B, R), device=device, dtype=dtype)

    if S_max == 0:
        # No steps → every token's broadcast advantage is 0 by definition.
        # Without this, `gather` on an empty column dim would error.
        return torch.zeros((B, R), device=device, dtype=dtype)

    # Clamp step_id into the valid range. -1 (mask sentinel) → 0; values that
    # accidentally exceed S_max-1 are also clamped to avoid silent
    # out-of-bounds gathers on CUDA. Both cases are masked out below.
    safe_idx = step_id.clamp(min=0, max=S_max - 1)
    gathered = A_step.gather(1, safe_idx)
    fmask = response_mask.to(dtype)
    valid_step = (step_id >= 0).to(dtype)
    return gathered * fmask * valid_step
