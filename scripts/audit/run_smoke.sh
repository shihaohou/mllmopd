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

# --- Activate eval env -------------------------------------------------------
# shellcheck disable=SC1091
source "${CONDA_PATH}/bin/activate" Uni-OPD-LMMS-Eval

# Ramp-up: set AUDIT_LIMIT=2 AUDIT_DEBUG=1 RUN_ID=debug2 for the first dry run,
# then AUDIT_LIMIT=20 RUN_ID=debug20, then full 500 (no env vars).
EXTRA_ARGS=()
if [ "${AUDIT_LIMIT:-0}" != "0" ]; then
  EXTRA_ARGS+=(--limit "${AUDIT_LIMIT}")
fi
if [ "${AUDIT_DEBUG:-0}" = "1" ]; then
  EXTRA_ARGS+=(--debug)
fi

run_pass() {
  local tag="$1" model="$2" mode="$3"
  local out="${RUN_DIR}/${tag}.jsonl"
  if [ -f "${out}" ]; then
    echo ">>> [${tag}] already exists — skipping"
    return
  fi
  echo ">>> [${tag}] model=${model} mode=${mode}"
  python -m mllmopd.diagnostics.run_audit_pass \
    --subset "${SUBSET}" \
    --model "${model}" \
    --mode "${mode}" \
    --out "${out}" \
    "${EXTRA_ARGS[@]}"
}

# Each pass uses one GPU; runs sequentially so they share a single H800.
CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-0}" \
  run_pass T_RL_full   "${MMR1_7B_RL_CKPT}"  full_image
CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-0}" \
  run_pass T_RL_blank  "${MMR1_7B_RL_CKPT}"  blank_image
CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-0}" \
  run_pass S_full      "${MMR1_3B_SFT_CKPT}" full_image
CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-0}" \
  run_pass S_blank     "${MMR1_3B_SFT_CKPT}" blank_image
CUDA_VISIBLE_DEVICES="${SMOKE_GPU:-0}" \
  run_pass S_text_only "${MMR1_3B_SFT_CKPT}" text_only

# --- Aggregate ---------------------------------------------------------------
python -m mllmopd.analysis.aggregate_audit \
  --run_dir "${RUN_DIR}" \
  --out "${RUN_DIR}/summary.json"

python -m mllmopd.reporting.audit_table --run_dir "${RUN_DIR}"

echo
echo ">>> Smoke audit done: ${RUN_DIR}"
echo ">>> To pull results back to Mac:"
echo "    rsync -avh devbox:${RUN_DIR}/ ./runs/audit/${RUN_ID}/"
