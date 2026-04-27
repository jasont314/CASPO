# 7B Disaggregated FSDP+vLLM Topology — Implementation Plan

## Motivation

Measured 4/27 on DeepSeekMath-7B-MATH at 8-GPU H100:

| Method | World | Step time | vs GRPO |
|---|---|---|---|
| GRPO | 4-GPU | 47.7 s | 1.0× |
| PPO | 4-GPU | 48.5 s | 1.02× |
| CASPO-frozen-RM | 4-GPU | 57.6 s | 1.21× |
| CASPO (online V_φ) | 4-GPU | 85.2 s | 1.79× |
| **VinePPO** | **8-GPU** | **338 s steady** | **7.1×** |

VinePPO at 7B is ~7× of GRPO baseline; the upstream paper claims ~2×. The
4-method group is acceptable. VinePPO is the outlier that needs structural
work to ship within the paper's reported wall-clock budget.

## Root cause

VinePPO's K=9 MC continuations × ~9 step boundaries × ~16 prompts/rank ≈
1296 concurrent generation requests per rank during the value pass. Our
**rank-local TP=1** topology means each rank's vLLM independently chews
through its share with its own 25 GB KV cache. Cross-rank pooling of the
MC fan-out onto a single TP=4-or-TP=8 vLLM with a unified ~80-160 GB KV
cache would dramatically reduce queue depth and amortize CUDA-graph
launches.

## Hard rejects (already considered)

* TP=1 in-process colocated with cross-rank IPC sync (Option C in earlier
  scoping): multi-day rewrite of `_sync_vllm_weights_fsdp` to broadcast
  FSDP-sharded params to TP-sharded vLLM. Defer.
* Stay rank-local + tune harder: already iterated (max_num_seqs=512,
  vllm_util=0.50, max_num_batched_tokens=32768) — squeezed 472 → 338 s.
  The remaining gap is structural.

## Chosen design — Option B: disaggregated FSDP=4 + TP=4

Split the 8 GPUs:

```
  trainer FSDP=4          rollout TP=4
+---+---+---+---+        +---+---+---+---+
| 0 | 1 | 2 | 3 |        | 4 | 5 | 6 | 7 |
+---+---+---+---+        +---+---+---+---+
        ^                        ^
        |                        |
      (this side runs the policy / value forward / backward)
                                 |
                            (this side runs vLLM AsyncLLM with TP=4
                             — one engine, 4-way sharded, dedicated GPUs)
```

* Trainer is FSDP=4 — same parallelism as today's GRPO/CASPO 4-GPU runs,
  so the trainer step (forward + backward + opt) does not regress.
* vLLM gets 4 dedicated H100s. Sharded 7B weights = 14 GB / 4 ≈ 3.5 GB
  per GPU, freeing ~10 GB on each rollout GPU vs the colocated TP=1 case
  for KV cache.
* `vllm_gpu_memory_utilization` can rise to 0.85 (no trainer to share
  with on those GPUs). KV pool ≈ 4 × 80 GB × 0.85 ≈ 270 GB across
  rollout group — handles VinePPO's MC fan-out without queueing.

## Inter-process communication

* **Generation**: trainer rank 0 owns the only `VLLMRolloutEngine` handle
  (in-process AsyncLLM, TP=4 spawns 3 worker subprocesses pinned to
  GPUs 5/6/7). Other trainer ranks gather their examples to rank 0
  before sample, scatter results after. No network involved.
* **Weight sync**: vLLM ships an NCCL backend
  (`vllm/distributed/weight_transfer/nccl_engine.py`) that broadcasts
  trainer→vLLM via a custom PyNcclCommunicator group. Trainer rank 0
  joins this group; vLLM workers (TP ranks 0..3) join. Per-step sync:
  `summon_full_params(self.model)` then call
  `NCCLWeightTransferEngine.trainer_send_weights(param_iter, args)` with
  packed=True. Estimated <1 s per sync (14 GB / NVLink BW).

## Implementation phases (each = its own commit)

### Phase 1 — config + opt-in plumbing
Add `cfg.vllm_disaggregated: bool` and `cfg.vllm_disaggregated_tp: int`.
Validate combinations in `__post_init__` (must have FSDP world ≥ 2 +
disaggregated_tp ≥ 1 + total GPUs ≥ FSDP_world + disaggregated_tp).
Default OFF — preserves current rank-local behavior.

### Phase 2 — launcher: split trainer-vs-rollout GPU sets
New shared body `scripts/_launch_7b_disagg.sh` that takes
`TRAIN_GPU_LIST` and `ROLLOUT_GPU_LIST`. Trainer ranks bind to
TRAIN_GPUs only; rank 0 also has ROLLOUT_GPUs visible (CUDA_VISIBLE_DEVICES
on rank 0 is `train_gpus + rollout_gpus`). vLLM TP=len(rollout_gpus).
New per-method wrapper `scripts/launch_7b_vineppo_disagg.sh`.

### Phase 3 — trainer: rank-0-only sampler with gather/scatter
Today every rank instantiates `self.sampler`. Under disaggregated:
* Only rank 0 instantiates `VLLMRolloutEngine(tensor_parallel_size=4,
  gpu_id=...)`.
* Ranks 1..N-1 hold a stub sampler that proxies to rank 0 via gather/scatter.
* At sample call sites:
  * `step()` line 1647: gather `examples` list to rank 0 →
    sample → scatter `RolloutBatch` slices.
  * `_vineppo_mc_step_values` line 1604: gather prefix lists, scatter MC
    completions.
* Use `dist.gather_object` + `dist.scatter_object` for the
  small-payload metadata, and `dist.broadcast` for the bulk tensors.

### Phase 4 — NCCL weight sync
Replace `_sync_vllm_weights_fsdp`'s IPC path with an NCCL path when
disaggregated. The new path:
1. Init a side PyNcclCommunicator group (trainer rank 0 + 4 vLLM workers).
2. On every sync: `FSDP.summon_full_params` on rank 0, then
   `vLLM.collective_rpc("update_weights", init_info=...)` with the
   NCCL init info; trainer rank 0 calls `trainer_send_weights(...)` while
   workers consume the broadcast.

### Phase 5 — smoke + iterate
4-step VinePPO smoke. Compare to colocated baseline. Tune
vllm_gpu_memory_utilization (target 0.85), max_num_seqs (target 1024+
for VinePPO MC pattern), max_num_batched_tokens (target 65536).
Document final numbers.

### Phase 6 (optional) — extend to other methods
GRPO/PPO/CASPO might also benefit from disaggregation since trainer
GPUs gain 25 GB headroom (no colocated vLLM). Run smokes; if the gain
is >10%, port the YAML defaults.

## Risks & rejected variants

* **HTTP serve mode**: simpler to reason about but adds 1-5 ms per
  request × thousands of requests/step → meaningful overhead. Reject.
* **Ray**: vLLM TP>1 historically used Ray. AsyncLLM v1 uses
  multiprocessing directly; no extra dep needed. Stay with multiprocessing.
* **File-based weight sync**: ~10 s for a 14 GB checkpoint. With
  max_steps=1000, that's 2.7 h of extra wall clock. Skip; go directly
  to NCCL.
* **Topology 6+2 / 7+1**: doesn't divide cleanly into our paper-faithful
  global PPO minibatch math (world × mb × accum = 64). 4+4 stays
  paper-faithful with mb=2 + accum=8 + FSDP=4.

## Acceptance criteria

* VinePPO 4-step smoke completes without rank-skew or weight-sync errors.
* Steady-state step time (step 2+) ≤ 200 s for VinePPO at 7B disaggregated
  (vs current 338 s colocated 8-GPU).
* GRPO/PPO/CASPO/CASPO-frozen-RM step times unchanged when run on the
  legacy rank-local launchers (regression guard).

## Phase 5a result (file-based checkpoint sync, 2026-04-27)

VinePPO 7B disaggregated FSDP=4 (GPUs 0-3) + vLLM TP=4 (GPUs 4-7),
``vllm_weight_sync_backend=checkpoint``. ``MAX_STEPS=2`` smoke.

| Step | t_step | t_roll | t_value | t_ref | t_pol | t_sync |
|---:|---:|---:|---:|---:|---:|---:|
|  1 | 451.9s | 53.5s | 323.5s | 4.3s | 35.1s | 27.9s |
|  2 | 270.0s |  6.1s | 220.7s | 4.2s | 33.2s |  0.0s¹ |

¹ Step 2 was the final step → ``sync_vllm=False``; t_sync omitted.
Adding the omitted file-sync (~28 s) gives a real steady-state of
~298 s/step.

vs colocated 8-GPU steady (338 s/step) the disaggregated path is
**~12% faster** (steady, with file-sync) or **~20% faster** if Phase
4b's NCCL sync drops t_sync from 28 s → ~2 s.

### Why not the 4× I hoped for

t_value (the K=9 MC fan-out) is the dominant cost, and pooled TP=4
KV cache does not beat 4× TP=1 engines on this specific 5184-sequence
workload. The NCCL all-reduce per layer at TP=4 eats most of the
KV-pool advantage. A single TP=8 engine would halve t_value but
breaks rank-local IPC compatibility *and* the FSDP world ÷ mb ÷
accum = 64 paper-faithful math (would need world=8 + mb=2 + accum=4).
Out of scope; documenting as a known ceiling.

### Phase 4b plan (NCCL weight sync, next commit)

Replace file-based ``sync_weights_from_path`` with
``NCCLWeightTransferEngine.trainer_send_weights`` (vLLM ships this in
``vllm/distributed/weight_transfer/nccl_engine.py``). Trainer rank 0
joins a side ``PyNcclCommunicator`` group with the 4 vLLM workers
(world_size = 1 + tp). Per-step sync wall time should drop from ~28
s (save_pretrained 14 GB + reload) to ~1-2 s (NVLink bcast). Saves
~7 hours over a 1000-step run.

### Phase 6 (deferred): port to GRPO/PPO/CASPO

The 4-method group already runs at 47-85 s/step rank-local. Disagg
would not help (those methods don't have the K=9 MC fan-out that
benefits from pooled KV). Skip.
