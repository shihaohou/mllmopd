#!/usr/bin/env bash
# Smoke Level-1 audit using only assets already on the dev box.
#
# What it does:
#   1. Builds a small ~500-prompt subset mixing MathVista-mini + POPE-adversarial.
#   2. Runs 5 inference passes (T_RL full/blank, S full/blank, S text-only).
#   3. Aggregates → summary.json and prints a table.
#
# Time: ~1–2 hours on a single H800.
# Prerequisites: scripts/init/02_devbox_bootstrap.sh + scripts/env/setup_lmmseval_env.sh.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source .env

: "${MLLMOPD_RUNS:?MLLMOPD_RUNS must be set}"
: "${MMR1_7B_RL_CKPT:?}"
: "${MMR1_3B_SFT_CKPT:?}"
: "${MATHVISTA_PATH:?}"
: "${POPE_PATH:?}"

RUN_ID="${RUN_ID:-smoke-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${MLLMOPD_RUNS}/audit/${RUN_ID}"
mkdir -p "${RUN_DIR}"

SUBSET="${SUBSET:-${MLLMOPD_DATA}/audit/smoke_subset_v0.jsonl}"
mkdir -p "$(dirname "${SUBSET}")"

# --- Activate eval env (must happen BEFORE subset build, which needs `datasets`)
# shellcheck disable=SC1091
source scripts/env/_activate.sh

# --- Build subset ------------------------------------------------------------
if [ ! -f "${SUBSET}" ]; then
  echo ">>> Building smoke subset at ${SUBSET}"
  MATHVISTA_PATH="${MATHVISTA_PATH}" POPE_PATH="${POPE_PATH}" \
    python scripts/data/prep_audit_subset.py \
      --out "${SUBSET}" \
      --size 500 \
      --only MathVista POPE_adversarial
else
  echo ">>> Reusing subset at ${SUBSET}"
fi

# Ramp-up: set AUDIT_LIMIT=2 AUDIT_DEBUG=1 RUN_ID=debug2 for the first dry run,
# then AUDIT_LIMIT=20 RUN_ID=debug20, then full 500 (no env vars).
EXTRA_ARGS=()
if [ "${AUDIT_LIMIT:-0}" != "0" ]; then
  EXTRA_ARGS+=(--limit "${AUDIT_LIMIT}")
fi
if [ "${AUDIT_DEBUG:-0}" = "1" ]; then
  EXTRA_ARGS+=(--debug)
fi

# Backend selection: AUDIT_BACKEND=sglang switches to the sglang engine path
# (must be in train venv). AUDIT_BATCH means different things per backend:
#   hf:     --batch-size       (static batch size for model.generate)
#   sglang: --max-running-requests (concurrent in-flight requests)
case "${AUDIT_BACKEND:-hf}" in
  hf)
    export AUDIT_BACKEND_MODULE="mllmopd.diagnostics.run_audit_pass"
    if [ "${AUDIT_BATCH:-1}" != "1" ]; then
      EXTRA_ARGS+=(--batch-size "${AUDIT_BATCH}")
    fi
    ;;
  sglang)
    export AUDIT_BACKEND_MODULE="mllmopd.diagnostics.run_audit_pass_sglang"
    if [ "${AUDIT_BATCH:-1}" != "1" ]; then
      EXTRA_ARGS+=(--max-running-requests "${AUDIT_BATCH}")
    fi
    if [ -n "${AUDIT_MEM_FRACTION:-}" ]; then
      EXTRA_ARGS+=(--mem-fraction "${AUDIT_MEM_FRACTION}")
    fi
    echo ">>> Using sglang backend — make sure you're in the train venv (sglang installed)"
    ;;
  *)
    echo "ERROR: unknown AUDIT_BACKEND=${AUDIT_BACKEND} (use hf or sglang)" >&2
    exit 1
    ;;
esac

# Default: full 3-mode audit for the OPD teacher (T_RL) and student (S).
# T_RL_text_only is included by default since smoke500 onwards — handoff
# §4.3 / §5.2 use it for the "RL refuses less than SFT" diagnostic.
PASS_TAGS=(T_RL_full              T_RL_blank             T_RL_text_only         S_full                 S_blank                S_text_only)
PASS_MODELS=("${MMR1_7B_RL_CKPT}" "${MMR1_7B_RL_CKPT}"  "${MMR1_7B_RL_CKPT}"   "${MMR1_3B_SFT_CKPT}" "${MMR1_3B_SFT_CKPT}" "${MMR1_3B_SFT_CKPT}")
PASS_MODES=(full_image            blank_image            text_only              full_image             blank_image            text_only)

# Optional: same-size pre-RL control (MMR1-7B-SFT). Set MMR1_7B_SFT_CKPT to add
# T_SFT × 3 modes — separates "size effect (3B SFT vs 7B SFT)" from "RL effect
# (7B SFT vs 7B RL)". Needed for any H3 claim about RL transferring artifacts.
if [ -n "${MMR1_7B_SFT_CKPT:-}" ]; then
  PASS_TAGS+=(T_SFT_full T_SFT_blank T_SFT_text_only)
  PASS_MODELS+=("${MMR1_7B_SFT_CKPT}" "${MMR1_7B_SFT_CKPT}" "${MMR1_7B_SFT_CKPT}")
  PASS_MODES+=(full_image blank_image text_only)
fi

# Optional: pre-post-training base (Qwen2.5-VL-Instruct). Set BASE_CKPT to add
# Base × 3 modes — the "what did post-training add" reference.
if [ -n "${BASE_CKPT:-}" ]; then
  PASS_TAGS+=(Base_full Base_blank Base_text_only)
  PASS_MODELS+=("${BASE_CKPT}" "${BASE_CKPT}" "${BASE_CKPT}")
  PASS_MODES+=(full_image blank_image text_only)
fi

# shellcheck source=../env/_dispatch_passes.sh disable=SC1091
source scripts/env/_dispatch_passes.sh

# --- Aggregate ---------------------------------------------------------------
python -m mllmopd.analysis.aggregate_audit \
  --run_dir "${RUN_DIR}" \
  --out "${RUN_DIR}/summary.json"

python -m mllmopd.reporting.audit_table --run_dir "${RUN_DIR}"

echo
echo ">>> Smoke audit done: ${RUN_DIR}"
echo ">>> To pull results back to Mac:"
echo "    rsync -avh devbox:${RUN_DIR}/ ./runs/audit/${RUN_ID}/"
