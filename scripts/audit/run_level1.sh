#!/usr/bin/env bash
# Orchestrate the Level-1 audit on the devbox (no training).
# Runs seven passes covering the (image × caption) decomposition + RL/SFT teacher.
# Each pass writes a JSONL of per-prompt records under $MLLMOPD_RUNS/audit/<run_id>/.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source .env

: "${MLLMOPD_RUNS:?}"
: "${MMR1_7B_RL_CKPT:?}"
: "${MMR1_7B_SFT_CKPT:?MMR1-7B-SFT not on disk yet; download before running Level-1}"
: "${MMR1_3B_SFT_CKPT:?}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${MLLMOPD_RUNS}/audit/${RUN_ID}"
mkdir -p "${RUN_DIR}"

SUBSET="${SUBSET:-data/audit/audit_subset_v0.jsonl}"
if [ ! -f "${SUBSET}" ]; then
  echo "Building audit subset at ${SUBSET}"
  python scripts/data/prep_audit_subset.py --out "${SUBSET}" --size 2000
fi

# Activate eval env (lmms-eval) for inference
# shellcheck disable=SC1091
source "${CONDA_PATH}/bin/activate" Uni-OPD-LMMS-Eval

EXTRA_ARGS=()
if [ "${AUDIT_LIMIT:-0}" != "0" ]; then
  EXTRA_ARGS+=(--limit "${AUDIT_LIMIT}")
fi
if [ "${AUDIT_DEBUG:-0}" = "1" ]; then
  EXTRA_ARGS+=(--debug)
fi

run() {
  local tag="$1" model="$2" mode="$3"
  echo ">>> [${tag}] model=${model} mode=${mode}"
  python -m mllmopd.diagnostics.run_audit_pass \
    --subset "${SUBSET}" \
    --model "${model}" \
    --mode "${mode}" \
    --out "${RUN_DIR}/${tag}.jsonl" \
    "${EXTRA_ARGS[@]}"
}

run "T_RL_full"          "${MMR1_7B_RL_CKPT}"  "full_image"
run "T_SFT_full"         "${MMR1_7B_SFT_CKPT}" "full_image"
run "S_full"             "${MMR1_3B_SFT_CKPT}" "full_image"
run "S_blank"            "${MMR1_3B_SFT_CKPT}" "blank_image"
run "T_RL_blank"         "${MMR1_7B_RL_CKPT}"  "blank_image"

# Caption-based modes require a captioning pass that's still TODO; gate them
# behind an explicit env var so a Level-1 run doesn't crash on the first prompt.
if [ "${ENABLE_CAPTION_MODES:-0}" = "1" ]; then
  run "S_caption_blank"    "${MMR1_3B_SFT_CKPT}" "caption_only_blank"
  run "S_image_plus_cap"   "${MMR1_3B_SFT_CKPT}" "image_plus_caption"
else
  echo ">>> caption modes skipped (set ENABLE_CAPTION_MODES=1 once captions are present in the subset)"
fi

# Aggregate to per-cell metrics
python -m mllmopd.analysis.aggregate_audit \
    --run_dir "${RUN_DIR}" \
    --out "${RUN_DIR}/summary.json"

# Print headline table
python -m mllmopd.reporting.audit_table --run_dir "${RUN_DIR}"

echo ">>> Audit run complete: ${RUN_DIR}"
