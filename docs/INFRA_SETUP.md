# CASPO Infrastructure Setup

End-to-end environment + dependency reference for reproducing the CASPO
training and eval pipeline. If your teammate's setup is failing, work
through this from top to bottom — the most common failure modes are
called out as **GOTCHA** boxes inline.

---

## 1. Hardware

* **GPUs**: 8× NVIDIA H100 80GB HBM3 (PCIe Gen4, NVLink + NVSwitch).
  - Same box for trainer + vLLM rollout. NVLink SHARP (NVLS) is enabled
    for FSDP collective reductions; falls back gracefully on non-Hopper.
  - Anything below H100 will work but timings in `README.md` won't match.
* **CPU**: ≥32 logical cores recommended (we run reward verifier in 16
  worker processes for VinePPO MC fan-out).
* **Host RAM**: ≥256 GB. Each 7B HuggingFace `safetensors` load peaks
  ~14 GB during shard transfer. Running two 7B trainings in parallel can
  briefly consume ~50–60 GB total host RAM during simultaneous loads
  (we hit OOM-killer at this when launching v3 / parallel sweep, fixed
  by 90s-staggered launches — see `scripts/mb4_v2.sh`).
* **Local NVMe**: ≥500 GB at `/mnt/nvme_tmp` for HF cache + checkpoints.
  Each 7B run writes ~13 GB per `step_N` checkpoint at `save_every=250`
  → ~52 GB per 1000-step run. Plan accordingly. A 1000-step `caspo`
  run with default `save_every=250` needs ~65 GB (4 step ckpts + final).

## 2. Operating system

* **OS**: Debian GNU/Linux 11 (bullseye), kernel 5.10.x.
* **NVIDIA driver**: 550.90.07 (CUDA 12.4 reported by driver).
* **System libstdc++**: tested with `/lib/x86_64-linux-gnu/libstdc++.so.6`
  exposing GLIBCXX_3.4.28 max. **The conda env's libstdc++ exposes
  GLIBCXX_3.4.34** — this matters (see GOTCHA below).

### GOTCHA: GLIBCXX missing when running outside the conda activation

> **Symptom**: `ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version
> 'GLIBCXX_3.4.29' not found (required by .../libzmq.so.5)` when invoking
> `eval.py` or `train_caspo.py` directly with `python`.
>
> **Cause**: The system's `libstdc++.so.6` (Debian 11) tops out at
> GLIBCXX_3.4.28. vLLM / pyzmq / flashinfer were built against the
> conda env's newer `libstdc++.so.6` (GLIBCXX_3.4.34). When the conda
> env isn't fully activated the dynamic linker resolves to the system
> libstdc++ and fails.
>
> **Fix**: Always invoke launchers via `bash scripts/launch_*.sh`, not
> by calling `python -u scripts/X.py` directly. Every launcher sources
> `/opt/conda/etc/profile.d/conda.sh; conda activate scalable` plus
> `scripts/perf_env.sh`, which sets up `LD_LIBRARY_PATH` correctly.
> If you must run python by hand, prefix with:
>
> ```bash
> source /opt/conda/etc/profile.d/conda.sh && conda activate scalable && \
>     source ./scripts/perf_env.sh && \
>     python -u scripts/X.py …
> ```

## 3. Conda environment (`scalable`)

Live env at `/opt/conda/envs/scalable`. ~22 GB on disk.

* **Python**: 3.11.14
* **CUDA toolkit (in-env)**: 12.8 (matches `torch+cu128`; system NVCC
  shows 12.4 from the driver, but PyTorch wheels bundle their own CUDA
  runtime).

### Pinned package versions (production-tested 2026-04-27)

```
torch==2.10.0+cu128         # bundles CUDA 12.8 runtime + cuDNN 9.10
torchao==0.15.0
torchaudio==2.10.0
torchdata==0.11.0
torchvision==0.25.0
triton==3.6.0
vllm==0.19.1                # v1 engine (V0 not supported)
flash_attn==2.8.3
flash_attn_3==3.0.0         # used selectively in eval; trainer uses fa2
flashinfer-python==0.6.6
flashinfer-cubin==0.6.6
transformers==4.57.3
accelerate==1.13.0
peft==0.18.1                # imported but not exercised in current runs
huggingface_hub==0.36.1
tokenizers==0.22.2
safetensors==0.7.0
datasets==4.8.4
sympy==1.14.0               # used by math verifier
numpy==2.2.6
protobuf==6.33.5
wandb==0.26.1
```

### Recreating the env from scratch

```bash
# 1. Create env with Python 3.11
conda create -y -n scalable python=3.11

# 2. Activate
source /opt/conda/etc/profile.d/conda.sh
conda activate scalable

# 3. PyTorch (CUDA 12.8 bundle — must come first, other wheels link against it)
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 4. vLLM v1 engine (this MUST match the pinned version; the v1 API has
#    breaking changes between minor releases)
pip install vllm==0.19.1

# 5. flash-attn 2 (trainer) and flashinfer (vLLM speedup)
pip install flash_attn==2.8.3 --no-build-isolation
pip install flashinfer-python==0.6.6 flashinfer-cubin==0.6.6

# 6. flash_attn_3 (eval-side; the ConfigSerial helper falls back gracefully)
pip install flash_attn_3==3.0.0 --no-build-isolation || true

# 7. HF + utilities
pip install transformers==4.57.3 accelerate==1.13.0 peft==0.18.1 \
            tokenizers==0.22.2 datasets==4.8.4 safetensors==0.7.0 \
            huggingface_hub==0.36.1

# 8. Verifier + logging
pip install sympy==1.14.0 wandb==0.26.1

# 9. Other
pip install torchao==0.15.0 torchdata==0.11.0 triton==3.6.0
```

### GOTCHA: flash_attn build slowness

> `flash_attn==2.8.3` builds CUDA kernels from source if no matching
> pre-built wheel is available for your `torch + cuda + python` triple.
> First install can take 30–60 minutes and needs ~16 GB host RAM during
> compile. If it fails with `out of memory`, reduce `MAX_JOBS=2` (default
> tries to use all cores). Pre-built wheels are at
> https://github.com/Dao-AILab/flash-attention/releases.

### GOTCHA: vllm version drift

> vLLM v1's API surface changes between 0.18.x → 0.19.x → 0.20.x. Our
> `caspo/rollout/vllm_engine.py` is pinned to `0.19.x` semantics
> (specifically `compilation_config["cudagraph_capture_sizes"]` shape,
> `weight_transfer_config`, and `AsyncEngineArgs` constructor). Bumping
> vllm without auditing those call sites WILL break weight sync and
> CUDA graph capture.

## 4. Filesystem layout

| Path | Purpose | Approx size |
|---|---|---|
| `/home/jason/experiment/CASPO` | Code repo | ~50 MB |
| `/mnt/nvme_tmp/jason_caspo/hf_cache` | Hugging Face model cache | 18 GB w/ Rho-1B + DSMath-7B + tokenizers |
| `/mnt/nvme_tmp/jason_caspo/<run-tag>` | Per-run output (checkpoints, wandb, logs) | 13 GB / 7B-checkpoint × `1000/save_every`-step + small for 1B |
| `/opt/conda/envs/scalable` | Conda env | 22 GB |

Three env vars point HF caches at NVMe (set in every launcher):

```bash
export HF_HOME=/mnt/nvme_tmp/jason_caspo/hf_cache
export HF_HUB_CACHE=/mnt/nvme_tmp/jason_caspo/hf_cache
export TRANSFORMERS_CACHE=/mnt/nvme_tmp/jason_caspo/hf_cache  # deprecated alias, kept for older code paths
```

### GOTCHA: disk full during long runs

> A 7B run writes a 13 GB checkpoint every 250 steps. With several runs
> in parallel + smoke tests, 369 GB fills in 1–2 days. We hit this twice
> in 2026-04-27 (the second time mid-iteration). The launcher does NOT
> auto-prune.
>
> **Quick check**: `df -h /mnt/nvme_tmp` before any new run.
>
> **Cleanup**: smoke test directories are safe to delete:
> ```bash
> rm -rf /mnt/nvme_tmp/jason_caspo/deepseekmath7b_math_*_p7s4_*_v* \
>        /mnt/nvme_tmp/jason_caspo/deepseekmath7b_math_p7s4_*_v*
> ```

## 5. Environment variables (`scripts/perf_env.sh`)

Sourced by every launcher. **Don't override unless you know what you're
doing**.

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# (We tried adding garbage_collection_threshold:0.6 + max_split_size_mb:512
#  to chase a ~100 MB OOM at mb=4 colocated; both slow steps 3-4×, reverted.)

# NCCL — surface hangs, longer collective timeout
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1800

# NVLink-SHARP collective offload (H100 + NVSwitch). Falls back gracefully.
export NCCL_NVLS_ENABLE=1
export NCCL_MIN_NCHANNELS=8
export NCCL_NTHREADS=512
export NCCL_BUFFSIZE=8388608          # 8 MiB transport buffer

# HF + tokenizer noise reduction
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export VLLM_NO_USAGE_STATS=1
export VLLM_LOGGING_LEVEL=WARNING
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### Tuning knobs (override per-run via env, not the file)

| Var | Default | Effect |
|---|---|---|
| `CASPO_VLLM_GPU_MEMORY_UTILIZATION` | per-launcher (0.20 colocated, 0.85 disagg) | vLLM KV reservation fraction |
| `CASPO_MICRO_BATCH_SIZE` | 2 (7B), 8 (1B) | trainer micro batch |
| `CASPO_GRAD_ACCUM_STEPS` | auto = 64 / world / mb | keeps global PPO minibatch=64 |
| `CASPO_USE_GRADIENT_CHECKPOINTING` | `true` (7B), `false` (1B) | activation checkpointing |
| `CASPO_FSDP_CPU_OFFLOAD` | `false` | escape hatch; ~30% step-time penalty |
| `CASPO_REWARD_WORKERS` | 16 | math-verifier worker pool size |
| `CASPO_VLLM_CUDAGRAPH_TRIM` | `1` (on) | trim CUDA-graph capture set + FULL_DECODE_ONLY (saves 150–300 MB / rank) |
| `MAX_STEPS` | from YAML (1000) | cap training steps |
| `SAVE_EVERY` | from YAML (250) | checkpoint cadence |
| `WANDB_MODE` | `offline` | online streaming requires `wandb login` first |
| `RUN_TAG` | empty | suffix for output dirs |

## 6. Models + datasets

### Models (auto-downloaded to HF cache on first run)

| Model | Path | Size on disk |
|---|---|---|
| Rho-1B SFT-MATH | `realtreetune/rho-1b-sft-MATH` | 2.1 GB |
| DeepSeekMath-7B SFT2 (MATH) | `realtreetune/deepseekmath-7b-sft-MATH` | 14 GB |
| Tokenizer (shared, GPT-NeoX style) | bundled with each model | — |

### Datasets

| Dataset | Source | Use |
|---|---|---|
| MATH (training) | `DigitalLearningGmbH/MATH-lighteval` (12,500 problems) | RL training prompts |
| MATH-500 (eval) | `HuggingFaceH4/MATH-500` (500 problems, Lightman et al. split) | held-out eval |
| GSM8K (eval) | `gsm8k:main` test split (1,319 problems) | optional eval |
| College / Olympiad (eval) | curated from `livecodebench` / olympiadbench | full final eval only |

### IPVRM value model checkpoints (CASPO prerequisites)

* **Rho-1B IPVRM v2** (current, 2026-04-28):
  `/mnt/nvme_tmp4/jason_caspo/caspo_rho1b_math_v2/value_final` —
  produced by `scripts/retrain_value_rho1b_4gpu.sh` (4-shard parallel
  collect + merge + FSDP=4 train_value, ~30-45 min on 4× H100). Step-1
  runtime `v_acc=0.74` against current verifier + Patch-A BOS prompts.
* **Rho-1B IPVRM v1** (deprecated, 2026-04-25):
  `/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math/value_final` — kept for
  reproducing pre-Apr-28 numbers. Step-1 runtime `v_acc=0.32` against
  current verifier (label drift from Minerva extractor cascade) + no-BOS
  prompts (predates Patch A).
* **DeepSeekMath-7B IPVRM** at `/mnt/nvme_tmp/jason_caspo/value_model_dsmath7b/final/`
  — produced by `scripts/_launch_7b_value_train.sh` on 4 GPUs, ~6 hours.
  Uses pre-Apr-28 verifier and tokenization; **needs retrain when
  switching to Apr-28+ stack**.

`configs/caspo_rho1b_math.yaml` defaults `prefix_value_path` to v2.
Override per-run via the `PREFIX_VALUE_PATH` env var (read by
`scripts/_launch_rho1b_one_gpu.sh` and forwarded as
`--override prefix_value_path=...` to the trainer). Re-run the retrain
pipeline whenever:
1. `caspo/reward/math_verifier.py` answer-extraction logic changes,
2. `caspo/rollout/vllm_engine.py` prompt-tokenization changes (e.g.
   adding/removing BOS prepending),
3. base SFT model changes.

The **end-to-end retrain** is one command:

```bash
GPU_LIST="4 5 6 7" bash scripts/retrain_value_rho1b_4gpu.sh
```

which runs (1) 4-shard `collect_value_data.py` in parallel, (2) merge
with `scripts/merge_value_data_shards.py`, (3) FSDP=4 `train_value.py`
at `value_micro_batch_size=16, value_grad_accum_steps=1` (same effective
batch as paper-faithful `mb=1, accum=16` but ~3.7× faster on H100), and
(4) smoke-validate via a 1-step CASPO rollout (passes if `v_acc≥0.7`).

ROC-AUC of the value model (over the held-out 10% of `value_data.pt`)
is the right offline-quality metric — see `eval_vphi_auc.py`.

## 7. Network / NCCL setup

* All training stays on a single node. No multi-node NCCL.
* vLLM IPC weight sync uses CUDA IPC mem handles; no network involvement.
* NCCL weight sync (disagg topology) uses a side process group on
  `127.0.0.1:<MASTER_PORT+1>` — works without any external network config.
* If running inside a container, ensure `/dev/shm` is large (≥4 GB)
  for NCCL shared-memory transports.

### GOTCHA: vLLM custom_all_reduce kernel JIT failure

> On some H100 boxes flashinfer's `trtllm_allreduce_fusion.cu` fails to
> compile (`std::optional` namespace error from a missing -std=c++17
> nvcc flag). vLLM falls back to a `custom_all_reduce.cuh` kernel which
> then fails with `Cuda error invalid argument`, killing `VllmWorker-0`.
>
> **Mitigation already in code**: when `tensor_parallel_size > 1` we
> set `disable_custom_all_reduce=True` (caspo/rollout/vllm_engine.py:316).
> Forces NCCL all-reduce, which works reliably on NVLink.

## 8. Common failure modes (and how to recover)

| Symptom | Likely cause | Fix |
|---|---|---|
| `GLIBCXX_3.4.29 not found` | env not activated (system libstdc++ on path) | source perf_env.sh first, or use launcher |
| `CUDA out of memory: tried 800.00 MiB free 362 MiB` | vLLM colocated with trainer at mb=4; budget is borderline | drop mb to 2, or use disagg topology |
| `OutOfMemoryError ... element 0 of tensors does not require grad` | Stale `@torch.no_grad` decorator on a `_train_*` method | grep for `@torch.no_grad`; remove from training methods |
| `vLLM: No available memory for the cache blocks` at init | `vllm_gpu_memory_utilization` too low (KV cache doesn't fit) | raise to ≥0.18 colocated, ≥0.30 standalone |
| `Engine core initialization failed` | flashinfer kernel JIT failed | re-pin flashinfer to 0.6.6; ensure `disable_custom_all_reduce=True` for TP>1 |
| `mkdir: cannot create directory ... No space left on device` | `/mnt/nvme_tmp` full | delete smoke runs; check `df -h` before relaunch |
| Two parallel 7B launches die simultaneously with status 137 | Host RAM OOM during simultaneous safetensors loads (~50 GB peak combined) | stagger launches by ≥60 s |
| `[7b-fsdp] ERROR: rank process failed with status 1` | one rank crashed; check rank logs at `<RUN>/logs/phase2_<method>_rank<N>.log` | usually OOM or env-var mismatch on that GPU |

## 9. Quick smoke test (to verify a fresh setup)

```bash
cd /home/jason/experiment/CASPO

# 2-step smoke on 1 GPU at Rho-1B (~3 min wall, ~6 GB peak)
GPU=0 MAX_STEPS=2 RUN_TAG=infra_smoke WANDB_MODE=offline \
    ./scripts/launch_rho1b_grpo.sh

# Inspect:
tail -f /mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_infra_smoke/logs/phase2_grpo.log
# Should print "[grpo step 1/2] ... t_step=Xs" within ~90 s of launch.
```

If that step lands cleanly, the env is good. If it OOMs, hangs in vLLM
init, or fails on import, work back through the GOTCHAs above.

## 10. Optional: torch profiler

Set `--override profile_steps=5` in any launcher to capture a
TensorBoard-readable profile (warmup=2, active=5, repeat=1). Output
lands at `<RUN>/profile/`. Default is 0 (off, zero overhead).

## 11. Reproducing what's currently in flight (2026-04-27)

The 1B canonical 4-method comparison is mid-run on this box:

```
GPU 4: caspo (online V_φ)      paper_seed0  step 440/1000  (collapsed)
GPU 5: caspo_frozen_rm         paper_seed0  step 522/1000
GPU 6: caspo_delta_p           paper_seed0  step 557/1000
GPU 7: caspo_delta_logp        paper_seed0  step 545/1000
```

Eval results live at:
```
/mnt/nvme_tmp/jason_caspo/caspo_rho1b_math_<method>_paper_seed0/eval_results_step_<N>_math500_k<K>_<limitN>.json
```

Sample eval (any free GPU):
```bash
RUN_TAG=paper_seed0 CKPT_SUBDIR=step_500 \
    METHODS="caspo_prob caspo_logprob caspo_frozen_rm" \
    EVAL_GPU_LIST="0 1 2" \
    EVAL_BENCHMARKS=math500 EVAL_K=16 EVAL_LIMIT=100 \
    EVAL_VLLM_GPU_MEMORY_UTILIZATION=0.85 \
    ./scripts/launch_eval_rho1b_sample_all8.sh
```
