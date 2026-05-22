#!/usr/bin/env bash
# T1 trajectory eval: capability vs. training step for both T1-2 and T1-3.
#
# Used to visualize the BlankTeacher collapse trajectory. Runs full_image
# eval on level1_subset_v0 against every saved ckpt (step_49 / step_99 /
# step_149 / step_199 / step_230) of both arms, plus the T1-0 base.
#
# Output: ${MLLMOPD_RUNS}/audit/${RUN_ID}/<arm>_step_NNN_full.jsonl
#
# Wall time on 8 GPUs: ~30-50 min for 11 passes (5 T1-2 + 5 T1-3 + 1 T1-0).
#
# Required env (from .env):
#   MMR1_3B_SFT_CKPT
#   MLLMOPD_RUNS
#   MLLMOPD_DATA
#
# Optional:
#   T1_2_RUN     defaults to t1_v1p5b_T1_2_full_mm
#   T1_3_RUN     defaults to t1_v1p5b_T1_3_blank_mm
#   STEPS        space-separated, defaults "49 99 149 199 230"
#   RUN_ID       defaults to t1_trajectory_$(date +...)
#   SMOKE_GPUS   GPU list for parallel dispatch (default "0")
#
# Auto-fixes Megatron-bridge's nested config.json (overwrites with base
# MMR1-3B-SFT config.json; idempotent via .bridge_bak sentinel).

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source .env

: "${MMR1_3B_SFT_CKPT:?}"
: "${MLLMOPD_RUNS:?}"
: "${MLLMOPD_DATA:?}"

T1_2_RUN="${T1_2_RUN:-t1_v1p5b_T1_2_full_mm}"
T1_3_RUN="${T1_3_RUN:-t1_v1p5b_T1_3_blank_mm}"
STEPS="${STEPS:-49 99 149 199 230}"
RUN_ID="${RUN_ID:-t1_trajectory_$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${MLLMOPD_RUNS}/audit/${RUN_ID}"
mkdir -p "${RUN_DIR}"

SUBSET="${SUBSET:-${MLLMOPD_DATA}/audit/level1_subset_v0.jsonl}"
if [ ! -f "${SUBSET}" ]; then
  echo "ERROR: eval subset not found at ${SUBSET}" >&2
  exit 1
fi

unset LD_LIBRARY_PATH
# shellcheck disable=SC1091
source scripts/env/_activate.sh

EXTRA_ARGS=(
  --max-running-requests "${AUDIT_BATCH:-16}"
  --mem-fraction "${AUDIT_MEM_FRACTION:-0.70}"
)
if [ -n "${AUDIT_MAX_NEW_TOKENS:-}" ]; then
  EXTRA_ARGS+=(--max-new-tokens "${AUDIT_MAX_NEW_TOKENS}")
fi

# MMR1 sysprompt (KEEP IN SYNC with scripts/audit/run_smoke.sh:121).
MMR1_SYSTEM_PROMPT='A conversation between User and Assistant. The User provides an image and asks a question. The Assistant first analyzes both the image and the question, then carefully thinks about the reasoning process step by step, and finally provides the User with an accurate answer. The Assistant must carefully checkout the correctness and validity of each reasoning step. If any errors or inconsistencies are found during the reasoning process, the Assistant reflects and corrects them logically. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here, with potential reflections and corrections </think><answer> final answer here, with the key result enclosed in \boxed{} </answer>.'

# --- Build pass matrix ----------------------------------------------------
PASS_TAGS=()
PASS_MODELS=()
PASS_MODES=()
PASS_SYSTEM_PROMPTS=()

_add_pass() {
  local tag="$1" model="$2"
  PASS_TAGS+=("${tag}")
  PASS_MODELS+=("${model}")
  PASS_MODES+=(full_image)
  PASS_SYSTEM_PROMPTS+=("${MMR1_SYSTEM_PROMPT}")
}

_fix_config() {
  local ckpt="$1"
  if [ ! -d "${ckpt}" ]; then
    return 1
  fi
  if [ -f "${ckpt}/config.json.bridge_bak" ]; then
    return 0   # already fixed
  fi
  cp "${ckpt}/config.json" "${ckpt}/config.json.bridge_bak"
  cp "${MMR1_3B_SFT_CKPT}/config.json" "${ckpt}/config.json"
  echo "    fixed ${ckpt}/config.json (was bridge-nested)"
  return 0
}

echo ">>> trajectory eval RUN_ID=${RUN_ID}"
echo ">>> RUN_DIR=${RUN_DIR}"
echo ">>> base : ${MMR1_3B_SFT_CKPT}"
echo ">>> T1-2 : ${MLLMOPD_RUNS}/${T1_2_RUN}/ckpt/hf/step_{${STEPS// /,}}"
echo ">>> T1-3 : ${MLLMOPD_RUNS}/${T1_3_RUN}/ckpt/hf/step_{${STEPS// /,}}"
echo

# T1-0 base (once)
_add_pass "T1_0_base_full" "${MMR1_3B_SFT_CKPT}"

# T1-2 at each step
for STEP in ${STEPS}; do
  CKPT="${MLLMOPD_RUNS}/${T1_2_RUN}/ckpt/hf/step_${STEP}"
  if _fix_config "${CKPT}"; then
    _add_pass "T1_2_step_${STEP}_full" "${CKPT}"
  else
    echo ">>> skip T1_2 step_${STEP} (ckpt missing)"
  fi
done

# T1-3 at each step
for STEP in ${STEPS}; do
  CKPT="${MLLMOPD_RUNS}/${T1_3_RUN}/ckpt/hf/step_${STEP}"
  if _fix_config "${CKPT}"; then
    _add_pass "T1_3_step_${STEP}_full" "${CKPT}"
  else
    echo ">>> skip T1_3 step_${STEP} (ckpt missing)"
  fi
done

echo
echo ">>> ${#PASS_TAGS[@]} passes to dispatch:"
printf "      %s\n" "${PASS_TAGS[@]}"
echo

# Backend module for the dispatcher
export AUDIT_BACKEND_MODULE="mllmopd.diagnostics.run_audit_pass_sglang"

# shellcheck source=../env/_dispatch_passes.sh disable=SC1091
source scripts/env/_dispatch_passes.sh

echo
echo ">>> trajectory eval complete: ${RUN_DIR}"
echo
echo ">>> Headline: for each (arm, step), compute mean full_image accuracy"
echo "    python - <<'PY'"
echo "import json, glob, os"
echo "for f in sorted(glob.glob('${RUN_DIR}/*.jsonl')):"
echo "    tag = os.path.basename(f).replace('_full.jsonl', '')"
echo "    rows = [json.loads(l) for l in open(f)]"
echo "    correct = sum(1 for r in rows if r.get('is_correct'))"
echo "    print(f'  {tag:30s} n={len(rows)}  acc={correct/len(rows):.3f}')"
echo "PY"
