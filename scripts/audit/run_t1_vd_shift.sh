#!/usr/bin/env bash
# T1 punch list #14: VD-shift analysis on the STUDENT checkpoints.
#
# After T1-eval (#12) produces `T1_2_full.jsonl` and `T1_3_full.jsonl` for
# the two trained students (FullTeacher OPD and BlankTeacher OPD), we
# re-run the H2 forced-decode pipeline on those student predictions, but
# scored by each student's OWN checkpoint. The question we are asking:
#
#   Did vanilla OPD shift the student's per-token visual-dependency (VD)
#   distribution toward high-VD tokens — i.e. did the student internalize
#   more vision-conditioned reasoning?
#
# We compare each arm's post-train VD distribution against the canonical
# teacher post-G1 baseline (`vd_summary.sysprompt.json`, 6.84% tokens /
# 21% NLL mass in high+very_high). The headline contrast is T1-2 minus
# T1-3 — that pp difference is what "FullTeacher OPD distilled more
# visual signal than BlankTeacher OPD" would look like in the VD bins.
#
# LIMITATION: this is a STUDENT-side VD measurement, so it depends on the
# student's own prediction text. If T1-2 and T1-3 emit very different
# output structures (different scaffolding tokens, different verbosity),
# the VD-bin deltas can come from output-shape drift rather than from
# internal "visual reliance". Cross-check by inspecting prediction
# samples and per-bin token counts before drawing causal conclusions.
#
# Runs on the train venv (sglang installed). Devbox A800 ok.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source .env

: "${MLLMOPD_RUNS:?}"
: "${MLLMOPD_DATA:?}"
: "${RUN_ID:?must set RUN_ID — the T1-eval run from punch list #12 (e.g. t1_eval_v0)}"
: "${CKPT_T1_2:?must set CKPT_T1_2 — HF-format checkpoint dir for the FullTeacher arm}"
: "${CKPT_T1_3:?must set CKPT_T1_3 — HF-format checkpoint dir for the BlankTeacher arm}"

EVAL_DIR="${MLLMOPD_RUNS}/audit/${RUN_ID}"
SUBSET="${SUBSET:-${MLLMOPD_DATA}/audit/level1_subset_v0.jsonl}"
ID_FILTER="${ID_FILTER:-runs/audit/level1_v4_sysprompt_fixed/opd_target_ids.json}"
TEACHER_BASELINE="${TEACHER_BASELINE:-runs/audit/level1_v4_sysprompt_fixed/vd_summary.sysprompt.json}"

T1_2_SOURCE="${EVAL_DIR}/T1_2_full.jsonl"
T1_3_SOURCE="${EVAL_DIR}/T1_3_full.jsonl"

OUT_T1_2_SCORED="${EVAL_DIR}/T1_2_score_opd_target.sysprompt.jsonl"
OUT_T1_3_SCORED="${EVAL_DIR}/T1_3_score_opd_target.sysprompt.jsonl"
OUT_T1_2_SUMMARY="${EVAL_DIR}/vd_summary.T1_2.json"
OUT_T1_3_SUMMARY="${EVAL_DIR}/vd_summary.T1_3.json"
OUT_SHIFT_JSON="${EVAL_DIR}/vd_shift_t1.json"

# Verbatim from scripts/audit/run_smoke.sh:121 (MMR1's training-time
# system prompt). KEEP IN SYNC. Changing this here without updating
# run_smoke.sh (or vice versa) silently desyncs the canonical audit
# prefix from the VD-shift scoring prefix.
MMR1_SYSTEM_PROMPT='A conversation between User and Assistant. The User provides an image and asks a question. The Assistant first analyzes both the image and the question, then carefully thinks about the reasoning process step by step, and finally provides the User with an accurate answer. The Assistant must carefully checkout the correctness and validity of each reasoning step. If any errors or inconsistencies are found during the reasoning process, the Assistant reflects and corrects them logically. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here, with potential reflections and corrections </think><answer> final answer here, with the key result enclosed in \boxed{} </answer>.'

# Sanity-check inputs (precondition: T1-eval has produced the source jsonls).
for f in "${SUBSET}" "${T1_2_SOURCE}" "${T1_3_SOURCE}" "${ID_FILTER}" "${TEACHER_BASELINE}"; do
  if [ ! -f "${f}" ]; then
    echo "ERROR: missing required input ${f}" >&2
    echo "  precondition: punch list #12 (T1-eval) must have produced T1_2_full.jsonl and T1_3_full.jsonl." >&2
    exit 1
  fi
done

# Verify both checkpoints exist as dirs (HF format).
for ckpt_var in CKPT_T1_2 CKPT_T1_3; do
  ckpt="${!ckpt_var}"
  if [ ! -d "${ckpt}" ]; then
    echo "ERROR: ${ckpt_var}=${ckpt} is not a directory (expected HF-format checkpoint)." >&2
    exit 1
  fi
done

# Venv: do NOT call scripts/env/_activate.sh — that hard-prefers the
# project-local .venv (audit/HF env), which lacks sglang. Operator must
# pre-activate the train venv. Same gate as rerun_h2_sysprompt.sh.
if ! python -c "import sglang" >/dev/null 2>&1; then
  echo "ERROR: sglang not importable in the current Python env." >&2
  echo "  current python: $(command -v python)" >&2
  echo "  fix: source /root/shihao_project/mllmopd-train-env/.venv/bin/activate" >&2
  echo "       (or your local equivalent train-venv path)" >&2
  exit 1
fi

# Memory knobs. Same defaults as rerun_h2_sysprompt.sh — sglang's static
# reservation + per-request KV cache OOM'd on A800-80GB with the upstream
# mem_fraction=0.85/max_running=64. `expandable_segments` mitigates KV
# fragmentation across the back-to-back full/blank forward passes.
MEM_FRACTION="${MEM_FRACTION:-0.70}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo ">>> run_t1_vd_shift: scoring T1-2 and T1-3 student rollouts on opd_target_ids.json"
echo ">>> RUN_ID=${RUN_ID}  EVAL_DIR=${EVAL_DIR}"
echo ">>> CKPT_T1_2=${CKPT_T1_2}"
echo ">>> CKPT_T1_3=${CKPT_T1_3}"
echo ">>> id_filter=${ID_FILTER}"
echo ">>> teacher baseline VD summary=${TEACHER_BASELINE}"
echo ">>> using MMR1_SYSTEM_PROMPT (first 60 chars):"
echo "    ${MMR1_SYSTEM_PROMPT:0:60}..."
echo ">>> mem_fraction=${MEM_FRACTION}  max_running_requests=${MAX_RUNNING_REQUESTS}"
echo ">>> PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo

# Forced-decode for each arm against its OWN checkpoint. Cannot run both
# in parallel inside one bash process (single GPU sglang engine per
# python proc); two sequential engine launches it is.
for ARM in T1_2 T1_3; do
  if [ "${ARM}" = "T1_2" ]; then
    SOURCE="${T1_2_SOURCE}"
    CKPT="${CKPT_T1_2}"
    OUT_SCORED="${OUT_T1_2_SCORED}"
    OUT_SUMMARY="${OUT_T1_2_SUMMARY}"
  else
    SOURCE="${T1_3_SOURCE}"
    CKPT="${CKPT_T1_3}"
    OUT_SCORED="${OUT_T1_3_SCORED}"
    OUT_SUMMARY="${OUT_T1_3_SUMMARY}"
  fi

  echo "================================================================"
  echo ">>> ${ARM}: scoring ${SOURCE} under ${CKPT}"
  echo ">>> output (per-token VD): ${OUT_SCORED}"
  echo "================================================================"
  python -m mllmopd.diagnostics.score_completion \
    --subset "${SUBSET}" \
    --source "${SOURCE}" \
    --model "${CKPT}" \
    --system-prompt-text "${MMR1_SYSTEM_PROMPT}" \
    --id-filter "${ID_FILTER}" \
    --mem-fraction "${MEM_FRACTION}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --out "${OUT_SCORED}"

  echo
  echo ">>> ${ARM}: aggregating per-token VD into ${OUT_SUMMARY}"
  python -m mllmopd.analysis.aggregate_vd \
    --scored "${OUT_SCORED}" \
    --out-table "${OUT_SUMMARY}"
  echo
done

echo "================================================================"
echo ">>> T1 VD-shift comparison: teacher_baseline vs T1-2 vs T1-3"
echo "================================================================"
python -m mllmopd.analysis.t1_vd_shift \
  --teacher-baseline "${TEACHER_BASELINE}" \
  --t1-2-summary "${OUT_T1_2_SUMMARY}" \
  --t1-3-summary "${OUT_T1_3_SUMMARY}" \
  --out-json "${OUT_SHIFT_JSON}"

echo
echo ">>> done. Headline scalars + per-bin table written to ${OUT_SHIFT_JSON}"
