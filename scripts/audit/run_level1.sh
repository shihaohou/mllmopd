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

# Activate eval env (must happen BEFORE subset build, which needs `datasets`)
# shellcheck disable=SC1091
source scripts/env/_activate.sh

SUBSET="${SUBSET:-data/audit/audit_subset_v0.jsonl}"
if [ ! -f "${SUBSET}" ]; then
  echo "Building audit subset at ${SUBSET}"
  python scripts/data/prep_audit_subset.py --out "${SUBSET}" --size 2000
fi

EXTRA_ARGS=()
if [ "${AUDIT_LIMIT:-0}" != "0" ]; then
  EXTRA_ARGS+=(--limit "${AUDIT_LIMIT}")
fi
if [ "${AUDIT_DEBUG:-0}" = "1" ]; then
  EXTRA_ARGS+=(--debug)
fi
if [ "${AUDIT_BATCH:-1}" != "1" ]; then
  EXTRA_ARGS+=(--batch-size "${AUDIT_BATCH}")
fi

PASS_TAGS=(T_RL_full              T_SFT_full             S_full                 S_blank                T_RL_blank)
PASS_MODELS=("${MMR1_7B_RL_CKPT}" "${MMR1_7B_SFT_CKPT}" "${MMR1_3B_SFT_CKPT}" "${MMR1_3B_SFT_CKPT}" "${MMR1_7B_RL_CKPT}")
PASS_MODES=(full_image            full_image             full_image             blank_image            blank_image)

# Caption-based modes are gated behind ENABLE_CAPTION_MODES because the
# captioning pass that feeds them is still TODO; without that flag the audit
# would crash on the first caption-only prompt.
if [ "${ENABLE_CAPTION_MODES:-0}" = "1" ]; then
  PASS_TAGS+=(S_caption_blank          S_image_plus_cap)
  PASS_MODELS+=("${MMR1_3B_SFT_CKPT}" "${MMR1_3B_SFT_CKPT}")
  PASS_MODES+=(caption_only_blank       image_plus_caption)
else
  echo ">>> caption modes skipped (set ENABLE_CAPTION_MODES=1 once captions are present in the subset)"
fi

# shellcheck source=../env/_dispatch_passes.sh disable=SC1091
source scripts/env/_dispatch_passes.sh

# Aggregate to per-cell metrics
python -m mllmopd.analysis.aggregate_audit \
    --run_dir "${RUN_DIR}" \
    --out "${RUN_DIR}/summary.json"

# Print headline table
python -m mllmopd.reporting.audit_table --run_dir "${RUN_DIR}"

echo ">>> Audit run complete: ${RUN_DIR}"
