"""Unit tests for caspo.algo.advantages."""

from __future__ import annotations

import torch

from caspo.algo.advantages import (
    step_values_from_log_ratios,
    step_td_advantage,
    transform_step_values_for_advantage,
    standardize_step_advantage,
    broadcast_step_advantage_to_tokens,
)


def _bool_mask(positions: list[int], R: int) -> torch.Tensor:
    m = torch.zeros(R, dtype=torch.bool)
    for p in positions:
        m[p] = True
    return m


# ---------------------------------------------------------------------------
# step_values_from_log_ratios


def test_step_values_simple():
    # log_ratio = [1, 1, 1, 2, 2, 2], boundary at index 2 and 5, 2 steps.
    log_ratio = torch.tensor([[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]])
    response_mask = torch.ones((1, 6), dtype=torch.long)
    boundary_after = _bool_mask([2, 5], 6).unsqueeze(0)
    step_count = torch.tensor([2], dtype=torch.long)

    V = step_values_from_log_ratios(log_ratio, response_mask, boundary_after, step_count)
    assert V.shape == (1, 3)
    assert torch.allclose(V[0], torch.tensor([0.0, 3.0, 9.0]))


def test_step_values_zero_steps():
    # A row with step_count=0 → V row is all zeros.
    log_ratio = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    response_mask = torch.zeros((1, 4), dtype=torch.long)
    boundary_after = torch.zeros((1, 4), dtype=torch.bool)
    step_count = torch.tensor([0], dtype=torch.long)

    V = step_values_from_log_ratios(log_ratio, response_mask, boundary_after, step_count)
    # When S_max == 0, returns [B, 1] of zeros.
    assert V.shape == (1, 1)
    assert torch.all(V == 0)


def test_step_values_constant_tail():
    # Two rows of different step counts; check that the shorter row is padded
    # with the constant tail (V[step_count]) at trailing positions.
    log_ratio = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],  # 2 steps, ends at idx 1, 3
            [2.0, 2.0, 2.0, 0.0, 0.0, 0.0],  # 1 step, ends at idx 2
        ]
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    boundary_after = torch.zeros((2, 6), dtype=torch.bool)
    boundary_after[0, 1] = True
    boundary_after[0, 3] = True
    boundary_after[1, 2] = True
    step_count = torch.tensor([2, 1], dtype=torch.long)

    V = step_values_from_log_ratios(log_ratio, response_mask, boundary_after, step_count)
    # S_max = 2 → V shape (2, 3)
    assert V.shape == (2, 3)
    # Row 0: cumsum at idx 1 = 2, at idx 3 = 4 → V = [0, 2, 4]
    assert torch.allclose(V[0], torch.tensor([0.0, 2.0, 4.0]))
    # Row 1: 1 step, cumsum at idx 2 = 6; tail position 2 should repeat 6.
    assert torch.allclose(V[1], torch.tensor([0.0, 6.0, 6.0]))


# ---------------------------------------------------------------------------
# step_td_advantage


def test_td_advantage_terminal_only():
    # γ=1, final_reward=1. V_step = [0, 0.3, 0.7], step_count=2.
    # A[0] = 0 + 1*V[1] - V[0] = 0.3
    # A[1] = R + 1*V[2] - V[1] = 1 + 0.7 - 0.3 = 1.4
    V = torch.tensor([[0.0, 0.3, 0.7]])
    r = torch.tensor([1.0])
    sc = torch.tensor([2], dtype=torch.long)
    A = step_td_advantage(V, r, sc, gamma=1.0)
    assert A.shape == (1, 2)
    assert torch.allclose(A[0], torch.tensor([0.3, 1.4]))


def test_td_advantage_intermediate_zero_reward():
    # 3 steps. V_step has 4 entries. r_step is zero except at terminal.
    # Sum of A over t = γ^S * V[S] + R - V[0]   when γ=1 → V[S] + R - V[0].
    # With V[0]=0 and γ=1, telescoping gives sum = R + V[S] - V[0] = R + V[S].
    # Wait: A_t = r_t + V[t+1] - V[t]. sum_t A_t = sum r_t + V[S] - V[0] = R + V[S] - V[0].
    # With V[0]=0, sum = R + V[S]. Not R alone — that would be the case if V[S]=0
    # which is not generally true. Test the actual formula.
    V = torch.tensor([[0.0, 0.5, 1.2, 2.0]])
    r = torch.tensor([1.0])
    sc = torch.tensor([3], dtype=torch.long)
    A = step_td_advantage(V, r, sc, gamma=1.0)
    expected = torch.tensor([0.5 - 0.0, 1.2 - 0.5, 1.0 + 2.0 - 1.2])
    assert torch.allclose(A[0], expected)
    # Telescoping check: sum A = R + V[S] - V[0]
    assert torch.allclose(A.sum(), r[0] + V[0, 3] - V[0, 0])


def test_td_advantage_padding_zero():
    # Two rows; first row has 2 steps, second row has 1 step. S_max=2.
    V = torch.tensor(
        [
            [0.0, 0.4, 0.9],  # 2 steps
            [0.0, 0.3, 0.3],  # 1 step (V[2] is the constant tail = V[1])
        ]
    )
    r = torch.tensor([1.0, 1.0])
    sc = torch.tensor([2, 1], dtype=torch.long)
    A = step_td_advantage(V, r, sc, gamma=1.0)
    assert A.shape == (2, 2)
    # Row 1: only step 0 is valid, A[1, 1] must be 0.
    assert A[1, 1].item() == 0.0
    # Row 1 step 0 (terminal): r + V[1] - V[0] = 1 + 0.3 - 0 = 1.3
    assert torch.allclose(A[1, 0], torch.tensor(1.3))


def test_transform_step_values_for_advantage_modes():
    V = torch.tensor([[-2.0, 0.0, 2.0]], dtype=torch.float32)

    direct = transform_step_values_for_advantage(V, "value")
    prob = transform_step_values_for_advantage(V, "prob")
    logprob = transform_step_values_for_advantage(V, "logprob")

    assert direct is V
    assert torch.allclose(prob, torch.sigmoid(V))
    assert torch.allclose(logprob, torch.nn.functional.logsigmoid(V))
    assert prob.min().item() > 0.0
    assert prob.max().item() < 1.0
    assert torch.isfinite(logprob).all()


def test_td_advantage_transform_changes_raw_pre_normalization_signal():
    V = torch.tensor([[0.0, 1.0, 3.0]], dtype=torch.float32)
    r = torch.tensor([1.0])
    sc = torch.tensor([2], dtype=torch.long)

    A_direct = step_td_advantage(
        transform_step_values_for_advantage(V, "value"), r, sc, gamma=1.0,
    )
    A_prob = step_td_advantage(
        transform_step_values_for_advantage(V, "prob"), r, sc, gamma=1.0,
    )
    A_logprob = step_td_advantage(
        transform_step_values_for_advantage(V, "logprob"), r, sc, gamma=1.0,
    )

    assert not torch.allclose(A_direct, A_prob)
    assert not torch.allclose(A_direct, A_logprob)
    assert not torch.allclose(A_prob, A_logprob)


# ---------------------------------------------------------------------------
# standardize_step_advantage


def test_standardize_batch():
    # Build A_step with known values; mean/std over valid entries.
    A = torch.tensor(
        [
            [1.0, 3.0, 0.0],   # 2 valid
            [5.0, 7.0, 9.0],   # 3 valid
        ]
    )
    sc = torch.tensor([2, 3], dtype=torch.long)
    out = standardize_step_advantage(A, sc, scope="batch")
    valid_vals = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0])
    expected_mean = valid_vals.mean()
    expected_std = ((valid_vals - expected_mean) ** 2).mean().sqrt()
    expected_normalized = (valid_vals - expected_mean) / expected_std
    # Check valid positions:
    flat_valid = torch.cat([out[0, :2], out[1, :3]])
    assert torch.allclose(flat_valid, expected_normalized, atol=1e-5)
    # Invalid position is zeroed.
    assert out[0, 2].item() == 0.0
    # Mean ≈ 0, std ≈ 1 over valid:
    assert flat_valid.mean().abs() < 1e-5
    assert (flat_valid.std(unbiased=False) - 1.0).abs() < 1e-5


def test_standardize_group():
    # 4 rows, group_size=2 → 2 groups; second group must not be affected by
    # the first.
    A = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [100.0, 200.0],
            [300.0, 400.0],
        ]
    )
    sc = torch.tensor([1, 1, 2, 2], dtype=torch.long)
    out = standardize_step_advantage(A, sc, scope="group", group_size=2)
    # Group 0 valid values: [1, 3] → mean=2, std=1, normalized=[-1, 1]
    assert torch.allclose(out[0, 0], torch.tensor(-1.0))
    assert torch.allclose(out[1, 0], torch.tensor(1.0))
    # Group 0 invalid positions should be zero.
    assert out[0, 1].item() == 0.0
    assert out[1, 1].item() == 0.0
    # Group 1 valid values: [100, 200, 300, 400] → mean=250, std=sqrt(12500)
    g1 = torch.tensor([100.0, 200.0, 300.0, 400.0])
    g1_mean = g1.mean()
    g1_std = ((g1 - g1_mean) ** 2).mean().sqrt()
    expected_g1 = (g1 - g1_mean) / g1_std
    flat = torch.stack([out[2, 0], out[2, 1], out[3, 0], out[3, 1]])
    assert torch.allclose(flat, expected_g1, atol=1e-4)


def test_standardize_off_identity():
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sc = torch.tensor([2, 2], dtype=torch.long)
    out = standardize_step_advantage(A, sc, scope="off")
    assert torch.equal(out, A)
    # Should be a clone, not the same tensor.
    out[0, 0] = 999.0
    assert A[0, 0].item() == 1.0


# ---------------------------------------------------------------------------
# broadcast_step_advantage_to_tokens


def test_broadcast_to_tokens_correct():
    # A_step=[[5, -2]], step_id=[[0,0,0,1,1,-1]] mask=[1,1,1,1,1,0]
    A = torch.tensor([[5.0, -2.0]])
    step_id = torch.tensor([[0, 0, 0, 1, 1, -1]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.long)
    out = broadcast_step_advantage_to_tokens(A, step_id, mask)
    expected = torch.tensor([[5.0, 5.0, 5.0, -2.0, -2.0, 0.0]])
    assert torch.allclose(out, expected)


def test_broadcast_masked_zero():
    # step_id=-1 OR mask=0 → 0.
    A = torch.tensor([[1.5, 2.5]])
    step_id = torch.tensor([[0, -1, 1, 1]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 0, 1]], dtype=torch.long)
    out = broadcast_step_advantage_to_tokens(A, step_id, mask)
    expected = torch.tensor([[1.5, 0.0, 0.0, 2.5]])
    assert torch.allclose(out, expected)


def test_telescoping_property():
    # Build a multi-step trajectory and confirm:
    #   sum_{tokens in step t} broadcast_adv = A_step[t] * step_length[t]
    A_step = torch.tensor([[2.0, -1.0, 0.5]])
    # 3 steps, with step lengths 2, 3, 1 → R=6.
    step_id = torch.tensor([[0, 0, 1, 1, 1, 2]], dtype=torch.long)
    mask = torch.ones((1, 6), dtype=torch.long)
    tokens = broadcast_step_advantage_to_tokens(A_step, step_id, mask)
    step_lens = [2, 3, 1]
    for t in range(3):
        in_step = tokens[0, step_id[0] == t]
        assert torch.allclose(in_step.sum(), A_step[0, t] * step_lens[t])


# ---------------------------------------------------------------------------
# Edge cases: empty batch
#
# B == 0 is a real possibility in distributed training when one rank receives
# nothing. All four functions should handle it without crashing and return a
# tensor of the correct rank.


def test_step_values_empty_batch():
    log_ratio = torch.zeros((0, 5))
    response_mask = torch.zeros((0, 5), dtype=torch.long)
    boundary_after = torch.zeros((0, 5), dtype=torch.bool)
    step_count = torch.zeros((0,), dtype=torch.long)

    V = step_values_from_log_ratios(log_ratio, response_mask, boundary_after, step_count)
    assert V.dim() == 2
    assert V.shape[0] == 0
    # Function returns [0, 1] on the empty-batch short-circuit path.
    assert V.shape[1] >= 1


def test_td_advantage_empty_batch():
    V = torch.zeros((0, 4))
    r = torch.zeros((0,))
    sc = torch.zeros((0,), dtype=torch.long)
    A = step_td_advantage(V, r, sc, gamma=1.0)
    assert A.dim() == 2
    assert A.shape[0] == 0
    # S_max comes from V.shape[1] - 1 = 3.
    assert A.shape[1] == 3


def test_standardize_empty_batch():
    A = torch.zeros((0, 4))
    sc = torch.zeros((0,), dtype=torch.long)
    out = standardize_step_advantage(A, sc, scope="batch")
    assert out.shape == (0, 4)
    out2 = standardize_step_advantage(A, sc, scope="off")
    assert out2.shape == (0, 4)


def test_broadcast_empty_batch():
    A = torch.zeros((0, 3))
    step_id = torch.zeros((0, 5), dtype=torch.long)
    mask = torch.zeros((0, 5), dtype=torch.long)
    out = broadcast_step_advantage_to_tokens(A, step_id, mask)
    assert out.shape == (0, 5)


# ---------------------------------------------------------------------------
# Edge cases: NaN handling
#
# NaN log-ratios / advantages can appear when upstream sees inf/-inf logits or
# saturated softmax. Standardization is the only stage that explicitly guards
# with nan_to_num — confirm it does, and confirm a finite reward is required
# upstream of step_td_advantage (we do not silently NaN-poison the batch).


def test_standardize_batch_with_nan_input():
    # Inject a NaN at an INVALID position; the masked sum should ignore it.
    A = torch.tensor(
        [
            [1.0, 3.0, float("nan")],  # 2 valid, NaN at padded slot
            [5.0, 7.0, 9.0],
        ]
    )
    sc = torch.tensor([2, 3], dtype=torch.long)
    out = standardize_step_advantage(A, sc, scope="batch")
    # Output must be entirely finite — nan_to_num guards the padded position.
    assert torch.isfinite(out).all()
    # Padded slot stays zero.
    assert out[0, 2].item() == 0.0


def test_standardize_group_with_nan_in_invalid_position():
    # Document the contract: NaN at INVALID positions taints the masked sum
    # (since NaN * 0 = NaN in IEEE 754). Implementation relies on nan_to_num
    # to scrub the output to all-zeros rather than propagating NaN downstream.
    # Callers must therefore zero out invalid positions BEFORE handing tensors
    # to standardize_step_advantage if they want valid entries to whiten.
    A = torch.tensor(
        [
            [1.0, float("nan")],  # NaN at masked position
            [3.0, float("nan")],
        ]
    )
    sc = torch.tensor([1, 1], dtype=torch.long)
    out = standardize_step_advantage(A, sc, scope="group", group_size=2)
    # The crucial post-condition: no NaN escapes the function.
    assert torch.isfinite(out).all()


def test_standardize_group_zeroed_padding_with_nan_free_input():
    # Companion to the above: when the caller zeroes invalid positions before
    # calling, valid entries whiten correctly under group scope.
    A = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 0.0],
        ]
    )
    sc = torch.tensor([1, 1], dtype=torch.long)
    out = standardize_step_advantage(A, sc, scope="group", group_size=2)
    assert torch.allclose(out[0, 0], torch.tensor(-1.0))
    assert torch.allclose(out[1, 0], torch.tensor(1.0))


def test_td_advantage_nan_reward_propagates():
    # We do NOT silently swallow NaN rewards — they must propagate so the
    # caller can detect upstream verifier breakage. Document that contract.
    V = torch.tensor([[0.0, 0.5, 1.0]])
    r = torch.tensor([float("nan")])
    sc = torch.tensor([2], dtype=torch.long)
    A = step_td_advantage(V, r, sc, gamma=1.0)
    # Terminal step gets the NaN; non-terminal step is finite.
    assert torch.isnan(A[0, 1]).item()
    assert torch.isfinite(A[0, 0]).item()


# ---------------------------------------------------------------------------
# Edge cases: all-zero step_count
#
# A whole batch with no segmentation boundaries — for example, when the
# splitter finds zero "\n\n" separators in any rollout. Every advantage row
# should be empty / zero rather than crash.


def test_step_values_all_zero_step_count():
    log_ratio = torch.tensor([[0.5, 0.5, 0.5], [1.0, 2.0, 3.0]])
    response_mask = torch.ones((2, 3), dtype=torch.long)
    boundary_after = torch.zeros((2, 3), dtype=torch.bool)
    step_count = torch.zeros((2,), dtype=torch.long)
    V = step_values_from_log_ratios(
        log_ratio, response_mask, boundary_after, step_count
    )
    # S_max == 0 short-circuit returns [B, 1] of zeros.
    assert V.shape == (2, 1)
    assert torch.all(V == 0)


def test_td_advantage_all_zero_step_count():
    # V_step has [B, 1] (only the V[0]=0 column). S_max=0.
    V = torch.zeros((3, 1))
    r = torch.tensor([1.0, 0.5, -0.2])
    sc = torch.zeros((3,), dtype=torch.long)
    A = step_td_advantage(V, r, sc, gamma=1.0)
    assert A.shape == (3, 0)


def test_standardize_all_zero_step_count():
    # A_step is [B, S_max] — but every row has 0 valid entries.
    A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sc = torch.zeros((2,), dtype=torch.long)
    out = standardize_step_advantage(A, sc, scope="batch")
    # No valid entries → std==0 path → identity clone.
    assert torch.equal(out, A)


# ---------------------------------------------------------------------------
# Edge cases: group_size == 1
#
# With group_size=1, every "group" is a single row. Per-row whitening means
# each row is centered against its own mean. A row with one valid step should
# fall back to identity (zero variance, no division).


def test_standardize_group_size_one_per_row_whitening():
    # Each row standardized independently. Row 0 has [1, 3] → [-1, 1].
    # Row 1 has [10, 20, 30] → [-1.2247, 0, 1.2247] approximately.
    A = torch.tensor(
        [
            [1.0, 3.0, 0.0],   # 2 valid
            [10.0, 20.0, 30.0],  # 3 valid
        ]
    )
    sc = torch.tensor([2, 3], dtype=torch.long)
    out = standardize_step_advantage(A, sc, scope="group", group_size=1)
    # Row 0: mean=2, std=1 (population), so [-1, 1, padding=0].
    assert torch.allclose(out[0, 0], torch.tensor(-1.0), atol=1e-5)
    assert torch.allclose(out[0, 1], torch.tensor(1.0), atol=1e-5)
    assert out[0, 2].item() == 0.0
    # Row 1: mean=20, std=sqrt(200/3) ≈ 8.165.
    row1 = torch.tensor([10.0, 20.0, 30.0])
    expected_row1 = (row1 - row1.mean()) / ((row1 - row1.mean()) ** 2).mean().sqrt()
    assert torch.allclose(out[1], expected_row1, atol=1e-4)


def test_standardize_group_size_one_single_valid_step_zero_var():
    # Each row has only 1 valid step → zero variance per group → identity
    # fallback (the row is left unchanged at the valid slot).
    A = torch.tensor([[5.0, 0.0], [7.0, 0.0]])
    sc = torch.tensor([1, 1], dtype=torch.long)
    out = standardize_step_advantage(A, sc, scope="group", group_size=1)
    # Zero-variance fallback in the group branch returns the original values
    # at valid positions and zeros at invalid positions.
    assert torch.allclose(out[0, 0], torch.tensor(5.0))
    assert torch.allclose(out[1, 0], torch.tensor(7.0))
    assert out[0, 1].item() == 0.0
    assert out[1, 1].item() == 0.0


# ---------------------------------------------------------------------------
# Edge cases: zero-variance batch
#
# If every valid entry in the batch is identical, the std denominator is 0;
# we must not divide by it.


def test_standardize_batch_zero_variance_identity():
    A = torch.tensor([[2.0, 2.0], [2.0, 0.0]])
    sc = torch.tensor([2, 1], dtype=torch.long)
    out = standardize_step_advantage(A, sc, scope="batch")
    # Zero-variance fallback returns clone — and importantly, no NaNs.
    assert torch.equal(out, A)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Edge cases: step_td_advantage with γ=0
#
# γ=0 means TD reduces to A_t = r_t - V[t]; nothing flows back. Useful as a
# sanity check that gamma is plumbed through (and not, e.g., hard-coded to 1).


def test_td_advantage_gamma_zero():
    V = torch.tensor([[0.0, 0.4, 0.9]])
    r = torch.tensor([1.0])
    sc = torch.tensor([2], dtype=torch.long)
    A = step_td_advantage(V, r, sc, gamma=0.0)
    # A[0] = 0 + 0*V[1] - V[0] = 0.
    # A[1] = r + 0*V[2] - V[1] = 1.0 - 0.4 = 0.6.
    assert torch.allclose(A[0], torch.tensor([0.0, 0.6]))
