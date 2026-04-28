#!/usr/bin/env bash
# Run both in-distribution and cross-dataset AUC evals on a V_φ checkpoint.
#
# Usage:
#   GPU=4 VPHI=/path/to/value_final LABEL=v7_bigmath_l1 \
#     bash scripts/eval_vphi_dual_auc.sh
#
# Env knobs:
#   VPHI       (required)  V_φ checkpoint path
#   LABEL      (required)  short tag for this V_φ
#   IN_DATA    (optional)  in-dist value_data.pt; defaults to <vphi parent>/../value_data.pt
#   GPU        default 4
#   XEVAL_BENCHMARK  default math500
#   XEVAL_DATA       cached cross-dataset rollouts; auto-built if missing
#   K          default 8 (rollouts per prompt for cross-eval)
#   TEMP       default 1.0 (matches V_φ training rollout temperature)
set -eo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate scalable
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/scalable/bin/python}"
GPU="${GPU:-4}"
VPHI="${VPHI:?must set VPHI=/path/to/value_final}"
LABEL="${LABEL:?must set LABEL}"
XEVAL_BENCHMARK="${XEVAL_BENCHMARK:-math500}"
K="${K:-8}"
TEMP="${TEMP:-1.0}"

# In-dist data path: caller can override; otherwise look for value_data.pt
# in the same dir as the V_φ ckpt (standard layout from retrain pipeline).
if [[ -z "${IN_DATA:-}" ]]; then
    IN_DATA="$(dirname "$(realpath "$VPHI")")/../value_data.pt"
fi

# Cross-eval cache: per-(benchmark, K, temp) so multiple V_φ ckpts share it.
XEVAL_CACHE_DIR="${XEVAL_CACHE_DIR:-/mnt/nvme_tmp4/jason_caspo/xeval_cache}"
mkdir -p "$XEVAL_CACHE_DIR"
XEVAL_DATA="${XEVAL_DATA:-$XEVAL_CACHE_DIR/${XEVAL_BENCHMARK}_k${K}_t${TEMP}.pt}"

# Map benchmark name -> HF spec for collect_value_data
case "$XEVAL_BENCHMARK" in
    math500)
        XEVAL_DSNAME="HuggingFaceH4/MATH-500"
        XEVAL_SPLIT="test"
        XEVAL_CFG=""
        ;;
    gsm8k)
        XEVAL_DSNAME="openai/gsm8k"
        XEVAL_SPLIT="test"
        XEVAL_CFG="main"
        ;;
    olympiadbench)
        XEVAL_DSNAME="Hothan/OlympiadBench"
        XEVAL_SPLIT="train"
        XEVAL_CFG="OE_TO_maths_en_COMP"
        ;;
    aime25)
        XEVAL_DSNAME="MathArena/aime_2025_I"
        XEVAL_SPLIT="train"
        XEVAL_CFG=""
        ;;
    *)
        echo "[xeval] unknown benchmark: $XEVAL_BENCHMARK" >&2
        exit 2
        ;;
esac

# 1) Build cross-dataset rollouts cache if missing
if [[ ! -f "$XEVAL_DATA" ]]; then
    echo "[xeval] building $XEVAL_BENCHMARK rollout cache at $XEVAL_DATA"
    XEVAL_CFG_OVERRIDE=()
    if [[ -n "$XEVAL_CFG" ]]; then
        XEVAL_CFG_OVERRIDE=(--override "dataset_config=$XEVAL_CFG")
    fi
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u -m scripts.collect_value_data \
        --config configs/caspo_rho1b_math.yaml \
        --output "$XEVAL_DATA" \
        --paper-pairing-multi \
        --override "dataset_name=$XEVAL_DSNAME" \
        --override "dataset_split=$XEVAL_SPLIT" \
        --override "filter_eval_leakage=false" \
        --override "group_size=$K" \
        --override "value_data_temperature=$TEMP" \
        "${XEVAL_CFG_OVERRIDE[@]}"
else
    echo "[xeval] reusing cached rollouts at $XEVAL_DATA"
fi

# 2) In-distribution AUC (uses trainer's held-out prompt-level split)
if [[ -f "$IN_DATA" ]]; then
    echo "[indist] running in-distribution AUC for $LABEL on $IN_DATA"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u scripts/eval_vphi_auc.py \
        --vphi "$VPHI" --label "${LABEL}_indist" --data "$IN_DATA"
else
    echo "[indist] WARN: $IN_DATA not found; skipping in-distribution AUC"
fi

# 3) Cross-dataset AUC (uses ALL rows of the cached rollouts)
echo "[xeval] running cross-dataset AUC for $LABEL on $XEVAL_BENCHMARK"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -u scripts/eval_vphi_auc.py \
    --vphi "$VPHI" --label "${LABEL}_xeval_${XEVAL_BENCHMARK}" \
    --data "$XEVAL_DATA" --no-split

echo "[done] dual AUC report for $LABEL on benchmark=$XEVAL_BENCHMARK complete"
