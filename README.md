# CASPO — Credit Assignment Step Policy Optimization

CASPO replaces the Monte-Carlo rollouts of VinePPO ([arXiv:2410.01679](https://arxiv.org/abs/2410.01679))
with a learned **prefix value model** (IPVRM, [arXiv:2604.13197](https://arxiv.org/abs/2604.13197)).

The result is step-level credit assignment for LLM reasoning that costs **one extra forward pass per step**
instead of K rollouts per step.

The trainer can run four method branches on the same math RL stack:

- `ppo`: sequence-level terminal reward advantage + PPO clipped surrogate.
- `caspo`: learned prefix-value step TD + PPO clipped surrogate.
- `vineppo`: MC prefix-value step TD + PPO clipped surrogate.
- `grpo`: group-relative terminal reward advantage + PPO clipped surrogate.

## Algorithm

For each rollout `(prompt x, response y)` with binary outcome `r_o ∈ {0,1}`:

1. **Segment** `y` into reasoning steps `s_1, s_2, …, s_T` (split on `\n\n`, configurable).
2. **Score** each prefix with the implicit prefix-value model

       V_φ(s_t) = β · Σ_{i<t} log [ π_φ(y_i | s_i) / π_ref(y_i | s_i) ]            (IPVRM Eq. 5)

   where `π_ref` is the SFT init (frozen) and `π_φ` is trained beforehand with the IPVRM BCE loss

       L = -E[ (1/T) Σ_t  r_o · log σ(v̄(t) − m)  +  (1−r_o) · log(1 − σ(v̄(t) + m)) ]   (IPVRM Eq. 9)

   `v̄(t) = V_φ(s_t)/t`, length-normalized for training stability.

3. **Step TD advantage** (VinePPO Eq. 6 with V from V_φ instead of MC):

       A_step[t] = r_step[t] + γ · V_φ(s_{t+1}) − V_φ(s_t)

   `r_step[t] = 0` except the terminal step where `r_step[T] = r_o`.

4. **Broadcast** `A_step[t]` to every token inside step t and apply the **PPO clipped surrogate**.

## Layout

```
caspo/
├── config.py            # CASPOConfig dataclass — single source of truth
├── data/                # math dataset loaders
├── reward/              # math_verify cascade (boxed answer extraction)
├── rollout/             # HF generate-based rollout with on-policy logprobs
├── segmentation/        # split responses into reasoning steps
├── value/               # PrefixValueModel (IPVRM) + training step
├── algo/                # step-level TD advantages + PPO clipped loss
├── trainer/             # CASPOTrainer (policy phase)
└── eval/                # math benchmark harness

scripts/
├── collect_value_data.py  # Phase 1a: rollout + label
├── train_value.py         # Phase 1b: train PrefixValueModel
├── train_caspo.py         # Phase 2: policy optimization
└── eval.py                # benchmark eval

configs/
├── value_smoke.yaml       # tiny IPVRM smoke
├── caspo_smoke.yaml       # tiny policy smoke
└── caspo_qwen25_math_7b.yaml
```

## Two phases

```
# Phase 1 — train prefix value model
python scripts/collect_value_data.py --config configs/value_smoke.yaml
python scripts/train_value.py --config configs/value_smoke.yaml

# Phase 2 — policy optimization with V_φ
python scripts/train_caspo.py --config configs/caspo_smoke.yaml \
    --override prefix_value_path=out/value/final
```

What we drop from VinePPO: K Monte-Carlo rollouts per step boundary (K·T extra generations per response).
What we drop from IPVRM: everything in §3.2+ (DistRL, ADB+DLW, PPO-side mb-norm). Just the prefix RM.

## Performance / env config

All launchers source `scripts/perf_env.sh` to keep CUDA allocator, NCCL,
tokenizer, vLLM, and CPU-thread settings consistent across phases:

| Variable | Value | Why |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Cuts allocator fragmentation in long PPO runs (rollout/forward/backward grow+shrink activations across iterations). |
| `TORCH_NCCL_BLOCKING_WAIT` | `1` | Surface NCCL hangs as actionable errors, not silent stalls. |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING` | `1` | Pair with the above so async failures abort the rank. |
| `NCCL_TIMEOUT` | `1800` | 30 min vs default 30 s — trainer↔vLLM weight syncs take minutes for 7B. |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | HPC tensor-parallel knob; harmless at TP=1. |
| `TOKENIZERS_PARALLELISM` | `false` | Silences fork-after-parallelism warning on every DataLoader spawn. |
| `PYTHONUNBUFFERED` | `1` | Live `tail -f` progress instead of blocked chunks. |
| `VLLM_NO_USAGE_STATS` | `1` | Skip vLLM telemetry ping. |
| `VLLM_LOGGING_LEVEL` | `WARNING` | Quiet per-request INFO logs. |
| `TRANSFORMERS_NO_ADVISORY_WARNINGS` | `1` | Suppress HF config advisory noise. |
| `HF_HUB_OFFLINE` | opt-in via `CASPO_HF_OFFLINE=1` | Force offline only when the cache is known complete. |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS` | `4`, `4` | Prevent BLAS thread oversubscription when several GPU jobs share a host. |

To use it from a new launcher:

```bash
source "$(dirname "$0")/perf_env.sh"   # if launcher lives in scripts/
# or
source ./scripts/perf_env.sh           # if launcher cd's to repo root first
```

Already wired into the project launchers. `launch_rho1b_parallel.sh` and
`launch_eval_all.sh` default to GPUs 4-7 and can be redirected with
`GPU_LIST="4 5 6 7"` / `EVAL_GPU_LIST="4 5 6 7"`.

## Optimized Full-Model RL

For 7B-scale full fine-tuning, use the FSDP + vLLM launcher:

```bash
CONFIG=configs/caspo_qwen25_math_7b.yaml \
PREFIX_VALUE_PATH=out/value/final \
NUM_GPUS=8 \
scripts/launch_fsdp_vllm.sh
```

This enables:

- `distributed_backend=fsdp`: shard policy/ref/value-model parameters and
  optimizer state across torchrun ranks.
- `rollout_backend=vllm`: keep rollout generation on vLLM instead of HF
  `generate`.
- `vllm_tensor_parallel_size=1`: one rank-local vLLM engine per GPU. This is
  the supported distributed topology for the current trainer.
- `vllm_multi_sample_mode=auto`: use vLLM `SamplingParams(n=K)` when the
  installed runtime returns all completions, otherwise fall back to the safe
  expanded request path.
- `vllm_weight_sync_backend=checkpoint`: FSDP still uses checkpoint reload
  until NCCL trainer-to-vLLM sync is implemented.

`prompts_per_step` is interpreted per rank in this mode, so the global prompt
batch is `prompts_per_step * NUM_GPUS`.

The project launchers default to `/opt/conda/envs/scalable/bin/python` (and
`/opt/conda/envs/scalable/bin/torchrun` for FSDP). Override `PYTHON_BIN` or
`TORCHRUN_BIN` only when intentionally testing another environment.

For 1B single-GPU H100 runs, the Rho configs use
`vllm_weight_sync_backend=ipc`. This uses vLLM's CUDA-IPC RL weight-transfer API
and removes the checkpoint-to-disk sync bottleneck. Validate a runtime with:

```bash
CUDA_VISIBLE_DEVICES=4 /opt/conda/envs/scalable/bin/python \
  -m scripts.probe_vllm_ipc_sync \
  --config configs/caspo_rho1b_math.yaml \
  --output-dir /tmp/caspo_vllm_ipc_probe
```

Leave `vllm_enforce_eager=false` unless CUDA graph memory becomes a problem.
Recent Rho-1B H100 probes with IPC show sync at ~0.3-0.4s/step; rollout and
policy/value forward now dominate.

VinePPO remains intrinsically slower than PPO because it samples K continuations
at each reasoning-step prefix. The optimized path batches mixed prefix budgets
in one vLLM submission and tries true `n=K` parallel sampling, but the method
still scales with roughly `responses * nonterminal_steps * K` extra
continuations. `vineppo_mc_max_tokens` can cap those MC continuations for speed
experiments; keep it at `0` for paper-faithful remaining-budget rollouts.
