#!/usr/bin/env bash
# Orchestrate the Level-1 audit on the devbox (no training).
# Runs four passes: T_RL, T_SFT, S full-image, S blank-image and oracle-caption variants.
# Each pass writes a JSONL of per-prompt records under $MLLMOPD_RUNS/audit/<run_id>/.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source .env

: "${MLLMOPD_RUNS:?}"

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

run() {
  local tag="$1" model="$2" mode="$3"
  echo ">>> [${tag}] model=${model} mode=${mode}"
  python -m mllmopd.diagnostics.run_audit_pass \
    --subset "${SUBSET}" \
    --model "${model}" \
    --mode "${mode}" \
    --out "${RUN_DIR}/${tag}.jsonl"
}

run "T_RL_full"        "MMR1/MMR1-7B-RL"  "full_image"
run "T_SFT_full"       "MMR1/MMR1-7B-SFT" "full_image"
run "S_full"           "MMR1/MMR1-3B-SFT" "full_image"
run "S_blank"          "MMR1/MMR1-3B-SFT" "blank_image"
run "T_RL_blank"       "MMR1/MMR1-7B-RL"  "blank_image"
run "S_oracle_caption" "MMR1/MMR1-3B-SFT" "oracle_caption"

# Aggregate to per-cell metrics
python -m mllmopd.analysis.aggregate_audit \
    --run_dir "${RUN_DIR}" \
    --out "${RUN_DIR}/summary.json"

# Print headline table
python -m mllmopd.reporting.audit_table --run_dir "${RUN_DIR}"

echo ">>> Audit run complete: ${RUN_DIR}"
