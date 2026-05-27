#!/usr/bin/env bash
# Step 5 — TAM Evidence Alignment audit launcher.
#
# Three models on a shared S1 rollout. Outputs per-token TAM alignment
# metrics (Top20% IoU / JS / Cosine) between teacher and {S0, S1}.
#
# Pipeline (phases — pick via $PHASE env, default "all"):
#
#   predict : multi-GPU fan-out — run S0 + S1 greedy generation on the
#             candidate pool. Writes
#             data/audit/tam_step5_predictions_v0.shard_<i>.jsonl
#             (or *.jsonl when single-GPU).
#   bucket  : single-process — merge shard predictions, assign buckets,
#             write data/audit/tam_step5_samples_v0.jsonl (n=200).
#   tam     : multi-GPU fan-out — for each sample shard, run
#             tam_step5_evidence_alignment.py (S1 rollout → T/S0/S1 TAM
#             pass → per-token alignment metrics). Merges shards into
#             ${RUN_DIR}/alignment.jsonl.
#   analyze : single-process — run tam_step5_analyzer.py to produce
#             tables + 3 figures + decision-tree results.md.
#
# Required env (sourced from .env or pre-exported):
#   MLLMOPD_RUNS         — output base
#   MMR1_7B_RL_CKPT      — teacher T
#   MMR1_3B_SFT_CKPT     — S0 base student
#   CKPT_T1_2            — S1 (T1-Full step_230); falls back to standard path
#                          if unset
#
# Optional env (defaults shown):
#   RUN_ID               — output subdir (default tam_step5_$(date +%Y%m%d-%H%M%S))
#   CANDIDATES           — space-separated list of candidate JSONL paths
#                          (default: level1_subset_v0 + smoke_subset_v0)
#   OPD_TARGET_IDS       — JSON file with opd_target id mapping (default: none)
#   N_IMPROVED / N_FAILED / N_TEACHER_ADV / N_DIVERSITY
#                          → 70/60/30/40 (default: 200 total)
#   MAX_NEW_TOKENS_SEL   — selector S0/S1 generate cap (default 1024)
#   MAX_NEW_TOKENS_TAM   — main runner rollout cap (default 4096)
#   NUM_GPUS             — data-parallel shard count (default: 8 on H800)
#   PHASE                — predict | bucket | tam | analyze | all (default all)
#
# Usage::
#
#     bash scripts/audit/run_tam_step5.sh
#     PHASE=tam RUN_ID=tam_step5_20260528 bash scripts/audit/run_tam_step5.sh
#     PHASE=analyze RUN_ID=tam_step5_20260528 bash scripts/audit/run_tam_step5.sh

set -euo pipefail

if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi
: "${MLLMOPD_RUNS:?MLLMOPD_RUNS must be set}"
: "${MMR1_3B_SFT_CKPT:?MMR1_3B_SFT_CKPT must be set}"
: "${MMR1_7B_RL_CKPT:?MMR1_7B_RL_CKPT must be set}"

unset LD_LIBRARY_PATH || true
# shellcheck source=../env/_activate.sh disable=SC1091
source scripts/env/_activate.sh

# Proxy gotcha (per project_h800_proxy memory) — sglang/server work needs
# no_proxy precision; for pure HF loads here we just drop the proxy.
unset -v http_proxy https_proxy no_proxy || true

# eager attn is needed only for output_attentions; Step 5 skips the
# attention baseline so SDPA is fine. Keep eager available as override.
export MLLMOPD_ATTN_IMPL="${MLLMOPD_ATTN_IMPL:-sdpa}"

# Resolve S1 path
S1_CKPT="${CKPT_T1_2:-${MLLMOPD_RUNS}/t1_v1p5b_T1_2_full_mm/ckpt/hf/step_230}"

PHASE="${PHASE:-all}"
NUM_GPUS="${NUM_GPUS:-8}"
RUN_ID="${RUN_ID:-tam_step5_$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${MLLMOPD_RUNS}/audit/${RUN_ID}"
mkdir -p "${RUN_DIR}"

PREDICTIONS_OUT="data/audit/tam_step5_predictions_v0.jsonl"
SAMPLES_OUT="data/audit/tam_step5_samples_v0.jsonl"

# Default candidate pool. Override via $CANDIDATES if needed.
CANDIDATES="${CANDIDATES:-data/audit/level1_subset_v0.jsonl data/audit/smoke_subset_v0.jsonl}"
CAND_ARGS=()
for c in ${CANDIDATES}; do
  CAND_ARGS+=(--candidates "${c}")
done

OPD_TARGET_ARGS=()
if [ -n "${OPD_TARGET_IDS:-}" ]; then
  OPD_TARGET_ARGS=(--opd-target-ids "${OPD_TARGET_IDS}")
fi

N_IMPROVED="${N_IMPROVED:-70}"
N_FAILED="${N_FAILED:-60}"
N_TEACHER_ADV="${N_TEACHER_ADV:-30}"
N_DIVERSITY="${N_DIVERSITY:-40}"

MAX_NEW_TOKENS_SEL="${MAX_NEW_TOKENS_SEL:-1024}"
MAX_NEW_TOKENS_TAM="${MAX_NEW_TOKENS_TAM:-4096}"

# For git rev label in writers
MLLMOPD_CODE_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
export MLLMOPD_CODE_COMMIT

echo "========================================"
echo "Step 5 launcher"
echo "========================================"
echo "  PHASE     = ${PHASE}"
echo "  RUN_DIR   = ${RUN_DIR}"
echo "  NUM_GPUS  = ${NUM_GPUS}"
echo "  T  (teacher) = ${MMR1_7B_RL_CKPT}"
echo "  S0 (base)    = ${MMR1_3B_SFT_CKPT}"
echo "  S1 (OPD)     = ${S1_CKPT}"
echo "  attn impl    = ${MLLMOPD_ATTN_IMPL}"
echo "  commit       = ${MLLMOPD_CODE_COMMIT}"
echo

# ============================================================================
# PHASE: predict — S0 + S1 candidate-pool generation
# ============================================================================
if [ "${PHASE}" = "predict" ] || [ "${PHASE}" = "all" ]; then
  echo "----- Phase: predict (multi-GPU fan-out) -----"
  PIDS=()
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    SHARD_LOG="${RUN_DIR}/predict_shard_${i}.log"
    (
      export CUDA_VISIBLE_DEVICES="${i}"
      PYTHONPATH=src python -m scripts.audit.tam_step5_sample_selector \
        "${CAND_ARGS[@]}" \
        --s0 "${MMR1_3B_SFT_CKPT}" \
        --s1 "${S1_CKPT}" \
        --predictions-out "${PREDICTIONS_OUT}" \
        --out "${SAMPLES_OUT}" \
        --max-new-tokens "${MAX_NEW_TOKENS_SEL}" \
        --image-root . \
        --stage 1 \
        --shard-id "${i}" \
        --num-shards "${NUM_GPUS}" \
        --n-improved "${N_IMPROVED}" \
        --n-failed "${N_FAILED}" \
        --n-teacher-advantage "${N_TEACHER_ADV}" \
        --n-diversity "${N_DIVERSITY}" \
        "${OPD_TARGET_ARGS[@]}" \
        > "${SHARD_LOG}" 2>&1
      echo "  predict shard ${i} done"
    ) &
    PIDS+=($!)
  done
  for pid in "${PIDS[@]}"; do
    wait "${pid}" || echo "!! predict shard pid ${pid} exited non-zero"
  done
  echo ">>> predict phase done; shard logs at ${RUN_DIR}/predict_shard_*.log"
fi

# ============================================================================
# PHASE: bucket — merge shards + stratified pick (single process)
# ============================================================================
if [ "${PHASE}" = "bucket" ] || [ "${PHASE}" = "all" ]; then
  echo "----- Phase: bucket (single process) -----"
  PYTHONPATH=src python -m scripts.audit.tam_step5_sample_selector \
    "${CAND_ARGS[@]}" \
    --s0 "${MMR1_3B_SFT_CKPT}" \
    --s1 "${S1_CKPT}" \
    --predictions-out "${PREDICTIONS_OUT}" \
    --out "${SAMPLES_OUT}" \
    --stage 2 \
    --num-shards "${NUM_GPUS}" \
    --n-improved "${N_IMPROVED}" \
    --n-failed "${N_FAILED}" \
    --n-teacher-advantage "${N_TEACHER_ADV}" \
    --n-diversity "${N_DIVERSITY}" \
    "${OPD_TARGET_ARGS[@]}"

  echo ">>> sample subset written: ${SAMPLES_OUT}"
  wc -l "${SAMPLES_OUT}"
fi

# ============================================================================
# PHASE: tam — main 3-model TAM extraction + alignment
# ============================================================================
if [ "${PHASE}" = "tam" ] || [ "${PHASE}" = "all" ]; then
  echo "----- Phase: tam (multi-GPU fan-out) -----"

  if [ ! -f "${SAMPLES_OUT}" ]; then
    echo "!! samples not found at ${SAMPLES_OUT}; run PHASE=bucket first"
    exit 1
  fi

  PIDS=()
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    SHARD_DIR="${RUN_DIR}/shard_${i}"
    mkdir -p "${SHARD_DIR}"
    (
      export CUDA_VISIBLE_DEVICES="${i}"
      PYTHONPATH=src python -m scripts.audit.tam_step5_evidence_alignment \
        --samples "${SAMPLES_OUT}" \
        --teacher "${MMR1_7B_RL_CKPT}" \
        --s0      "${MMR1_3B_SFT_CKPT}" \
        --s1      "${S1_CKPT}" \
        --out-dir "${SHARD_DIR}" \
        --max-new-tokens "${MAX_NEW_TOKENS_TAM}" \
        --image-root . \
        --shard-id "${i}" \
        --num-shards "${NUM_GPUS}" \
        --pass all \
        > "${SHARD_DIR}/stdout.log" 2> "${SHARD_DIR}/stderr.log"
      echo "  tam shard ${i} done"
    ) &
    PIDS+=($!)
  done
  echo ">>> waiting for ${#PIDS[@]} tam shards (tail ${RUN_DIR}/shard_*/std*.log)"
  for pid in "${PIDS[@]}"; do
    wait "${pid}" || echo "!! tam shard pid ${pid} exited non-zero"
  done

  echo ">>> merging shard alignment files"
  cat "${RUN_DIR}"/shard_*/alignment.jsonl > "${RUN_DIR}/alignment.jsonl" 2>/dev/null || true
  N_ROWS=$(wc -l < "${RUN_DIR}/alignment.jsonl" 2>/dev/null || echo 0)
  echo "  merged ${N_ROWS} alignment rows"

  {
    echo "# Step 5 TAM evidence alignment  (commit=${MLLMOPD_CODE_COMMIT})"
    echo "teacher = ${MMR1_7B_RL_CKPT}"
    echo "s0      = ${MMR1_3B_SFT_CKPT}"
    echo "s1      = ${S1_CKPT}"
    echo "samples = ${SAMPLES_OUT}"
    echo "num_shards (NUM_GPUS) = ${NUM_GPUS}"
    echo "n_alignment_rows      = ${N_ROWS}"
    echo "merged alignment      = ${RUN_DIR}/alignment.jsonl"
    echo
    echo "per-shard:"
    for i in $(seq 0 $((NUM_GPUS - 1))); do
      sd="${RUN_DIR}/shard_${i}"
      nr=$(wc -l < "${sd}/alignment.jsonl" 2>/dev/null || echo 0)
      nro=$(wc -l < "${sd}/rollout_cache.jsonl" 2>/dev/null || echo 0)
      echo "  shard ${i}: rollouts=${nro}  alignment=${nr}"
    done
  } > "${RUN_DIR}/summary.txt"
  echo ">>> summary: ${RUN_DIR}/summary.txt"
fi

# ============================================================================
# PHASE: analyze — tables + figures + decision tree
# ============================================================================
if [ "${PHASE}" = "analyze" ] || [ "${PHASE}" = "all" ]; then
  echo "----- Phase: analyze (single process) -----"

  ALIGNMENT_PATH="${RUN_DIR}/alignment.jsonl"
  if [ ! -f "${ALIGNMENT_PATH}" ]; then
    echo "!! alignment.jsonl not found at ${ALIGNMENT_PATH}; run PHASE=tam first"
    exit 1
  fi

  OUT_FIGS="docs/figures/step5"
  mkdir -p "${OUT_FIGS}"
  PYTHONPATH=src python -m mllmopd.analysis.tam_step5_analyzer \
    --alignment "${ALIGNMENT_PATH}" \
    --out-dir "${OUT_FIGS}/"

  echo ">>> figures + tables at ${OUT_FIGS}/"
  ls -la "${OUT_FIGS}/" || true
fi

echo
echo "========================================"
echo "Step 5 DONE  (phase=${PHASE})"
echo "========================================"
echo "  RUN_DIR   : ${RUN_DIR}"
echo "  samples   : ${SAMPLES_OUT}"
echo "  alignment : ${RUN_DIR}/alignment.jsonl"
echo "  figures   : docs/figures/step5/"
