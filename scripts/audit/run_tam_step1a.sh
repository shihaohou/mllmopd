#!/usr/bin/env bash
# Step 1a TAM-VD calibration launcher (teacher_greedy mode).
#
# Two passes inside one Python call:
#   Pass 1: teacher full forward (greedy gen + hidden + attn + scores) +
#           teacher blank-image scoring on the same response_ids.
#           Writes ${OUT_DIR}/teacher_cache.jsonl (checkpoint-invariant).
#   Pass 2: for each student ckpt, teacher-forced scoring of teacher's
#           response → student_lp / student_entropy / adv / quad.
#           Writes ${OUT_DIR}/tam_step1a.jsonl (one row per sample × ckpt).
#
# Wall-time (per v0.1.2 schema §forward-cost ledger):
#   teacher pass: ~17 min for 200 samples × 1 ckpt (eager attn ~5x FA2)
#   student pass: ~15 min × 4 ckpts
#   → total ~30-40 min on H800 if teacher cache is fresh; ~15 min if reused.
#
# Required env (sourced from .env or pre-exported):
#   MLLMOPD_RUNS         — output base
#   MMR1_7B_RL_CKPT      — teacher (default: MMR1/MMR1-7B-RL)
#   MMR1_3B_SFT_CKPT     — T1-0 student
#   CKPT_T1_2 / CKPT_T1_3 / CKPT_T2_1 — student ckpts (optional)
#
# Optional env (defaults shown):
#   RUN_ID               — output subdir (default: tam_step1a_$(date +%Y%m%d-%H%M%S))
#   SUBSET               — calibration sample JSONL
#                          (default: data/audit/tam_calibration_subset_v0.jsonl)
#   MAX_NEW_TOKENS       — gen cap (default: 512)
#   CUDA_VISIBLE_DEVICES — which GPU (default: 0)
#   LIMIT                — limit n samples for smoke (default: 0 = all)
#   MLLMOPD_ATTN_IMPL    — eager|sdpa|flash_attention_2 (default: eager —
#                          attention baseline requires it)
#   SKIP_STUDENT         — 1 = write teacher cache only, no student pass
#   TEACHER_CACHE        — reuse existing teacher cache JSONL
#
# Usage:
#   bash scripts/audit/run_tam_step1a.sh
#   SKIP_STUDENT=1 bash scripts/audit/run_tam_step1a.sh   # teacher pass only
#   LIMIT=10 bash scripts/audit/run_tam_step1a.sh         # 10-sample smoke

set -euo pipefail

if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi
: "${MLLMOPD_RUNS:?MLLMOPD_RUNS must be set}"
: "${MMR1_3B_SFT_CKPT:?MMR1_3B_SFT_CKPT must be set (T1-0 baseline)}"

unset LD_LIBRARY_PATH || true
# shellcheck source=../env/_activate.sh disable=SC1091
source scripts/env/_activate.sh

# Proxy gotcha (per project_h800_proxy memory)
unset -v http_proxy https_proxy no_proxy || true

# v0.1.2: code commit + eager attention for attention baseline
export MLLMOPD_CODE_COMMIT="$(git rev-parse --short=10 HEAD 2>/dev/null || echo unknown)"
export MLLMOPD_ATTN_IMPL="${MLLMOPD_ATTN_IMPL:-eager}"

# spaCy preflight
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null || {
  echo "!! spaCy en_core_web_sm not loadable. POS-based token_category will"
  echo "!! degrade to 'other' for content_noun / pronoun / visual_attribute."
  echo "!! Fix: pip install spacy && python -m spacy download en_core_web_sm"
}

TS="$(date +%Y%m%d-%H%M%S)"
RUN_ID="${RUN_ID:-tam_step1a_${TS}}"
RUN_DIR="${MLLMOPD_RUNS}/audit/${RUN_ID}"
SUBSET="${SUBSET:-data/audit/tam_calibration_subset_v0.jsonl}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LIMIT="${LIMIT:-0}"
TEACHER_CKPT="${MMR1_7B_RL_CKPT:-MMR1/MMR1-7B-RL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Build --students NAME=PATH args from env. Skip any that are empty.
STUDENT_ARGS=(--students "T1_0=${MMR1_3B_SFT_CKPT}")
if [ -n "${CKPT_T1_2:-}" ]; then STUDENT_ARGS+=(--students "T1_2=${CKPT_T1_2}"); fi
if [ -n "${CKPT_T1_3:-}" ]; then STUDENT_ARGS+=(--students "T1_3=${CKPT_T1_3}"); fi
if [ -n "${CKPT_T2_1:-}" ]; then STUDENT_ARGS+=(--students "T2_1=${CKPT_T2_1}"); fi

EXTRA=()
if [ "${LIMIT}" != "0" ]; then EXTRA+=(--limit "${LIMIT}"); fi
if [ "${SKIP_STUDENT:-0}" = "1" ]; then EXTRA+=(--skip-student); fi
if [ -n "${TEACHER_CACHE:-}" ]; then EXTRA+=(--teacher-cache "${TEACHER_CACHE}"); fi

mkdir -p "${RUN_DIR}"

cat <<EOF
========================================
TAM Step 1a launching
========================================
  RUN_ID      = ${RUN_ID}
  RUN_DIR     = ${RUN_DIR}
  TEACHER     = ${TEACHER_CKPT}
  STUDENTS    = ${STUDENT_ARGS[@]}
  SUBSET      = ${SUBSET}
  MAX_NEW_TOK = ${MAX_NEW_TOKENS}
  LIMIT       = ${LIMIT}
  ATTN_IMPL   = ${MLLMOPD_ATTN_IMPL}
  CUDA        = ${CUDA_VISIBLE_DEVICES}
========================================
EOF

PYTHONPATH=src python -m scripts.audit.tam_step1a \
  --subset "${SUBSET}" \
  --teacher "${TEACHER_CKPT}" \
  "${STUDENT_ARGS[@]}" \
  --out-dir "${RUN_DIR}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --image-root . \
  "${EXTRA[@]}"

echo
echo "========================================"
echo "Step 1a DONE"
echo "========================================"
echo "  Teacher cache : ${RUN_DIR}/teacher_cache.jsonl"
echo "  Final JSONL   : ${RUN_DIR}/tam_step1a.jsonl"
echo "  Summary       : ${RUN_DIR}/summary.txt"
echo
if [ -f "${RUN_DIR}/summary.txt" ]; then
  cat "${RUN_DIR}/summary.txt"
fi
