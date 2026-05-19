#!/usr/bin/env bash
# T1 punch list #9: 10-step smoke for the OPD training stack.
#
# Slices the prep train.jsonl down to 80 prompts (10 batches × 8 prompts)
# and runs the canonical opd_mmr1_3b_baseline.sh launcher with DEBUG_MODE=1.
# After the launcher exits (or you Ctrl-C after step 10), invokes the
# post-mortem verifier to confirm:
#   - per-step diagnostics JSONLs exist (opd_diagnostics_hook fired)
#   - lp_full ≠ lp_blank (dual_teacher_get_reward really hit both arms)
#   - tensor of loss / clip_fraction in the train log looks sane
#
# Hard stop on any failure — per T1 plan §8.1 #9 we do not proceed to
# #10/#11 until smoke is clean.
#
# Defaults to the T1-2 (FullTeacher) arm. Run again with
# OPD_TEACHER_IMAGE_MODE=blank to smoke the T1-3 arm.
#
# Env overrides:
#   ARM                       full|blank (default: full)
#   SMOKE_N_PROMPTS           number of prompts to slice from train.jsonl
#                             (default: 80 = 10 batches × 8 SAMPLE_N)
#   SMOKE_SOURCE_JSONL        source training data (default:
#                             data/opd_train/v0_2k/train.jsonl)
#   SMOKE_SUBSET_JSONL        path to write the sliced subset (default:
#                             data/opd_train/v0_2k/smoke_${ARM}.jsonl)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source .env

: "${MLLMOPD_RUNS:?}"

ARM="${ARM:-full}"
case "${ARM}" in
  full|blank) ;;
  *) echo "ERROR: ARM must be 'full' or 'blank', got '${ARM}'" >&2; exit 1 ;;
esac

SMOKE_N_PROMPTS="${SMOKE_N_PROMPTS:-80}"
SMOKE_SOURCE_JSONL="${SMOKE_SOURCE_JSONL:-data/opd_train/v0_2k/train.jsonl}"
SMOKE_SUBSET_JSONL="${SMOKE_SUBSET_JSONL:-data/opd_train/v0_2k/smoke_${ARM}.jsonl}"

if [ ! -f "${SMOKE_SOURCE_JSONL}" ]; then
  echo "ERROR: source training jsonl not found at ${SMOKE_SOURCE_JSONL}" >&2
  echo "  run scripts/data/prep_opd_train_data.py first." >&2
  exit 1
fi

mkdir -p "$(dirname "${SMOKE_SUBSET_JSONL}")"
head -n "${SMOKE_N_PROMPTS}" "${SMOKE_SOURCE_JSONL}" > "${SMOKE_SUBSET_JSONL}"
N_WROTE=$(wc -l < "${SMOKE_SUBSET_JSONL}")
echo ">>> sliced ${N_WROTE} prompts -> ${SMOKE_SUBSET_JSONL}"

OPD_RUN_NAME="t1_smoke_${ARM}_$(date +%Y%m%d_%H%M%S)"
export OPD_RUN_NAME

echo ">>> launching opd_mmr1_3b_baseline.sh with DEBUG_MODE=1, ARM=${ARM}"
echo ">>> run name: ${OPD_RUN_NAME}"
echo
echo "    interrupt with Ctrl-C after ~10 optimizer steps if it keeps running;"
echo "    diagnostics + train log are already written incrementally."
echo

DEBUG_MODE=1 \
TRAIN_JSONL="${SMOKE_SUBSET_JSONL}" \
OPD_TEACHER_IMAGE_MODE="${ARM}" \
SAVE_INTERVAL=5 \
bash scripts/train/opd_mmr1_3b_baseline.sh \
  || true   # let Ctrl-C succeed; verifier runs regardless

echo
echo "================================================================"
echo "  Smoke launcher exited. Running post-mortem verifier."
echo "================================================================"
python scripts/train/verify_t1_smoke.py \
  --run-dir "${MLLMOPD_RUNS}/${OPD_RUN_NAME}" \
  --arm "${ARM}"
