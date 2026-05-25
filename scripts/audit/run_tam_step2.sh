#!/usr/bin/env bash
# Step 2 TAM causal masking launcher (data-parallel, 8-GPU fan-out).
#
# For each sample × target_token × mask_strategy, runs teacher-forced
# forward under masked image and records logp_drop. ~6 forwards per
# sample (1 baseline + 5 strategies); 8-GPU fan-out → ~3-5 min for
# 200 samples.
#
# Required env (sourced from .env or pre-exported):
#   MLLMOPD_RUNS         — output base
#   MMR1_7B_RL_CKPT      — teacher (default: MMR1/MMR1-7B-RL)
#
# Optional env:
#   RUN_ID               — output subdir (default: tam_step2_$(date +%Y%m%d-%H%M%S))
#   SUBSET               — calibration sample JSONL
#                          (default: data/audit/tam_calibration_subset_v0.jsonl)
#   MAX_NEW_TOKENS       — gen cap (default: 512)
#   NUM_GPUS             — data-parallel shard count (default: 1)
#   LIMIT                — limit n samples for smoke (default: 0 = all)
#   MLLMOPD_ATTN_IMPL    — sdpa|eager|flash_attention_2 (default: sdpa).
#                          Step 2 doesn't need attention baseline (TAM maps
#                          are the input, not attention), so sdpa is fastest.
#
# Usage:
#   NUM_GPUS=8 LIMIT=10 bash scripts/audit/run_tam_step2.sh
#   NUM_GPUS=8 bash scripts/audit/run_tam_step2.sh    # full run

set -euo pipefail

if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi
: "${MLLMOPD_RUNS:?MLLMOPD_RUNS must be set}"

unset LD_LIBRARY_PATH || true
# shellcheck source=../env/_activate.sh disable=SC1091
source scripts/env/_activate.sh

unset -v http_proxy https_proxy no_proxy || true

export MLLMOPD_CODE_COMMIT="$(git rev-parse --short=10 HEAD 2>/dev/null || echo unknown)"
# Step 2 doesn't need attention output — sdpa is enough and faster.
export MLLMOPD_ATTN_IMPL="${MLLMOPD_ATTN_IMPL:-sdpa}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || {
  echo "!! spaCy en_core_web_sm not loadable. token_category will degrade."
  echo "!! Fix: pip install spacy && python -m spacy download en_core_web_sm"
}

TS="$(date +%Y%m%d-%H%M%S)"
RUN_ID="${RUN_ID:-tam_step2_${TS}}"
RUN_DIR="${MLLMOPD_RUNS}/audit/${RUN_ID}"
SUBSET="${SUBSET:-data/audit/tam_calibration_subset_v0.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LIMIT="${LIMIT:-0}"
TEACHER_CKPT="${MMR1_7B_RL_CKPT:-MMR1/MMR1-7B-RL}"
NUM_GPUS="${NUM_GPUS:-1}"
if [ "${NUM_GPUS}" = "1" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi

EXTRA=()
if [ "${LIMIT}" != "0" ]; then EXTRA+=(--limit "${LIMIT}"); fi

mkdir -p "${RUN_DIR}"

cat <<EOF
========================================
TAM Step 2 (causal masking) launching
========================================
  RUN_ID      = ${RUN_ID}
  RUN_DIR     = ${RUN_DIR}
  TEACHER     = ${TEACHER_CKPT}
  SUBSET      = ${SUBSET}
  MAX_NEW_TOK = ${MAX_NEW_TOKENS}
  LIMIT       = ${LIMIT}
  ATTN_IMPL   = ${MLLMOPD_ATTN_IMPL}
  NUM_GPUS    = ${NUM_GPUS}
  CUDA        = ${CUDA_VISIBLE_DEVICES:-<unset>}
========================================
EOF

if [ "${NUM_GPUS}" = "1" ]; then
  PYTHONPATH=src python -m scripts.audit.tam_step2 \
    --subset "${SUBSET}" \
    --teacher "${TEACHER_CKPT}" \
    --out-dir "${RUN_DIR}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --image-root . \
    "${EXTRA[@]}"
else
  echo ">>> Fanning out to ${NUM_GPUS} GPUs"
  PIDS=()
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    SHARD_DIR="${RUN_DIR}/shard_${i}"
    mkdir -p "${SHARD_DIR}"
    (
      export CUDA_VISIBLE_DEVICES="${i}"
      PYTHONPATH=src python -m scripts.audit.tam_step2 \
        --subset "${SUBSET}" \
        --teacher "${TEACHER_CKPT}" \
        --out-dir "${SHARD_DIR}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --image-root . \
        --shard-id "${i}" \
        --num-shards "${NUM_GPUS}" \
        "${EXTRA[@]}" \
        > "${SHARD_DIR}/stdout.log" 2> "${SHARD_DIR}/stderr.log"
      echo "  shard ${i} done"
    ) &
    PIDS+=($!)
  done

  echo ">>> Waiting for ${#PIDS[@]} shards"
  for pid in "${PIDS[@]}"; do
    wait "${pid}" || echo "!! shard pid ${pid} exited non-zero"
  done

  echo ">>> Merging shard outputs"
  cat "${RUN_DIR}"/shard_*/tam_step2.jsonl > "${RUN_DIR}/tam_step2.jsonl" 2>/dev/null || true
  {
    echo "# Step 2 TAM causal masking (multi-GPU)  (commit=${MLLMOPD_CODE_COMMIT})"
    echo "teacher = ${TEACHER_CKPT}"
    echo "subset  = ${SUBSET}"
    echo "num_shards (NUM_GPUS) = ${NUM_GPUS}"
    echo "n_samples_total = $(wc -l < "${SUBSET}")"
    echo "n_rows_total    = $(wc -l < "${RUN_DIR}/tam_step2.jsonl" 2>/dev/null || echo 0)"
    echo ""
    echo "per-shard:"
    for i in $(seq 0 $((NUM_GPUS - 1))); do
      sd="${RUN_DIR}/shard_${i}"
      n_r=$(wc -l < "${sd}/tam_step2.jsonl" 2>/dev/null || echo 0)
      echo "  shard ${i}: rows=${n_r}"
    done
  } > "${RUN_DIR}/summary.txt"
fi

echo
echo "========================================"
echo "Step 2 DONE"
echo "========================================"
echo "  JSONL   : ${RUN_DIR}/tam_step2.jsonl"
echo "  Summary : ${RUN_DIR}/summary.txt"
echo
if [ -f "${RUN_DIR}/summary.txt" ]; then
  cat "${RUN_DIR}/summary.txt"
fi
