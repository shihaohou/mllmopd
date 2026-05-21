#!/usr/bin/env bash
# T1 eval: re-run the canonical Level-1 audit on the two T1 student
# checkpoints (T1-2 FullTeacher OPD, T1-3 BlankTeacher OPD), optionally
# alongside a fresh T1-0 baseline pass for hardware/parity.
#
# What it does (per T1 plan §5):
#   1. Reuses data/audit/level1_subset_v0.jsonl (the same 6 × 200 grid that
#      level1_v4_sysprompt_fixed used).
#   2. Runs 3 modes (full_image, blank_image, text_only) × 3 models = 9 passes.
#      Each pass writes ${RUN_DIR}/<tag>.jsonl with the canonical schema.
#   3. Injects the MMR1 training-time system prompt (verbatim from
#      run_smoke.sh:121) so trained T1 students stay in MMR1-mode and emit
#      `<think>...</think><answer>\boxed{...}</answer>`.
#   4. Aggregates → summary.json, prints headline table.
#   5. Re-runs paired_vision_critical against the new outputs to compute
#      per-benchmark opd_target_recovery[b] for the T1 students.
#   6. Hint at `t1_compare.py` for the headline Δ scalar.
#
# Wall-time (T1 plan §6): ~12 min × 2 arms × 3 modes = ~25 min for T1-2+T1-3
# only. Re-running T1-0 brings the total to ~50-60 min on 1 A800.
# With SMOKE_GPUS=0,1,...,N parallel-per-gpu mode, it scales down accordingly.
#
# Required env vars (sourced from .env):
#   MLLMOPD_RUNS         — base output dir (NFS)
#   MLLMOPD_DATA         — base data dir (NFS; holds the level1 subset)
#   MMR1_3B_SFT_CKPT     — T1-0 baseline (re-run unless SKIP_T1_0=1)
#   CKPT_T1_2            — T1-2 HF checkpoint (FullTeacher arm; --save-hf output)
#   CKPT_T1_3            — T1-3 HF checkpoint (BlankTeacher arm; --save-hf output)
#
# Optional env vars (defaults shown):
#   RUN_ID               — output subdir (default: t1_v0_eval_$(date +%Y%m%d-%H%M%S))
#   SUBSET               — eval subset path (default: ${MLLMOPD_DATA}/audit/level1_subset_v0.jsonl)
#   SKIP_T1_0            — 1 = skip T1-0 re-run (default: 0; honors T1-plan §5
#                          "re-run for hardware parity" but cheap to bypass)
#   AUDIT_BACKEND        — hf|sglang (default: sglang; T1 students need the
#                          train venv anyway, no point falling back to HF)
#   AUDIT_BATCH          — sglang max-running-requests (default: 16)
#   AUDIT_MEM_FRACTION   — sglang mem_fraction_static (default: 0.70)
#   AUDIT_LIMIT          — limit prompts per pass for smoke runs (default: 0)
#   AUDIT_DEBUG          — 1 = verbose per-pass logging (default: 0)
#   AUDIT_MAX_NEW_TOKENS — generation cap (default: 4096; matches run_smoke.sh)
#   SMOKE_GPUS / SMOKE_GPU — see _dispatch_passes.sh
#   SKIP_PAIRED_VC       — 1 = skip the paired_vision_critical analysis at the end
#   BASELINE_RUN_DIR     — path to level1_v4_sysprompt_fixed for per-benchmark
#                          opd_target_recovery delta vs baseline (default:
#                          ${MLLMOPD_RUNS}/audit/level1_v4_sysprompt_fixed)
#
# Prerequisites:
#   - Train venv active (sglang installed). The audit venv won't work because
#     T1 students are evaluated through the same sglang path as the teacher.
#   - cuDNN ≥ 9.16 in the train venv (pitfalls.md E1) — handled by the
#     `unset LD_LIBRARY_PATH` below.
#
# Usage examples (from repo root):
#   # Default: T1-0 + T1-2 + T1-3, all 3 modes each (9 passes), single GPU 0
#   CKPT_T1_2=/path/to/t1_v0_T1_2_full/ckpt/hf/step_250 \
#   CKPT_T1_3=/path/to/t1_v0_T1_3_blank/ckpt/hf/step_250 \
#     bash scripts/audit/run_t1_eval.sh
#
#   # Skip T1-0 (already evaluated in level1_v4); 6 passes only
#   SKIP_T1_0=1 \
#   CKPT_T1_2=... CKPT_T1_3=... \
#     bash scripts/audit/run_t1_eval.sh
#
#   # Parallel across 4 GPUs (T1 plan §6 recommendation for the ~12-min figure)
#   SMOKE_GPUS=0,1,2,3 \
#   CKPT_T1_2=... CKPT_T1_3=... \
#     bash scripts/audit/run_t1_eval.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# CKPT_T1_2/3 preservation. If the caller pre-exported them, `source .env`
# below would silently overwrite the caller's values with any CKPT_T1_2/3
# lines that live in .env. This bit us in T1 v1: .env carried v0 step_249
# paths from an earlier session and the caller's interactive
# `export CKPT_T1_2=.../t1_v1.../step_99` had no effect — the eval
# evaluated the wrong checkpoints and produced a "v1 ≈ v0 equivalence"
# finding that was nothing of the sort. Capture caller-supplied values
# here, source .env, then restore.
_CKPT_T1_2_CALLER="${CKPT_T1_2:-}"
_CKPT_T1_3_CALLER="${CKPT_T1_3:-}"
# shellcheck disable=SC1091
source .env
if [ -n "${_CKPT_T1_2_CALLER}" ]; then
  CKPT_T1_2="${_CKPT_T1_2_CALLER}"
fi
if [ -n "${_CKPT_T1_3_CALLER}" ]; then
  CKPT_T1_3="${_CKPT_T1_3_CALLER}"
fi
unset _CKPT_T1_2_CALLER _CKPT_T1_3_CALLER

# --- Required env -----------------------------------------------------------
: "${MLLMOPD_RUNS:?MLLMOPD_RUNS must be set}"
: "${MLLMOPD_DATA:?MLLMOPD_DATA must be set}"
: "${MMR1_3B_SFT_CKPT:?MMR1_3B_SFT_CKPT must be set (T1-0 baseline)}"
: "${CKPT_T1_2:?CKPT_T1_2 must point at the T1-2 FullTeacher HF checkpoint (e.g. \${MLLMOPD_RUNS}/t1_v1_T1_2_full_mm/ckpt/hf/step_99)}"
: "${CKPT_T1_3:?CKPT_T1_3 must point at the T1-3 BlankTeacher HF checkpoint (e.g. \${MLLMOPD_RUNS}/t1_v1_T1_3_blank_mm/ckpt/hf/step_99)}"

# Echo the resolved checkpoints loudly so any future "wrong ckpt was
# evaluated" bug is visible at the top of the launcher log. Don't trust
# stdout alone — the audit pass dispatcher also writes the same paths
# in each per-pass log (sglang server_args.model_path / jsonl row.model).
echo ">>> CKPT_T1_2  resolved : ${CKPT_T1_2}"
echo ">>> CKPT_T1_3  resolved : ${CKPT_T1_3}"

# Fail fast if the checkpoints aren't there. Cheap; prevents 25 min of work
# falling over on the second arm.
for ck in "${MMR1_3B_SFT_CKPT}" "${CKPT_T1_2}" "${CKPT_T1_3}"; do
  if [ ! -d "${ck}" ]; then
    echo "ERROR: checkpoint dir not found: ${ck}" >&2
    exit 1
  fi
done

# Default to a neutral t1_eval_* prefix. The version prefix (v0/v1/v2…)
# now lives in the training run name and the CKPT_T1_2/3 paths, not in
# the eval RUN_ID. For experiments where the eval output dir should
# carry an explicit version tag, pass RUN_ID at the call site, e.g.
# `RUN_ID=t1_v1_eval_step99_$(date +%Y%m%d-%H%M%S)`.
RUN_ID="${RUN_ID:-t1_eval_$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${MLLMOPD_RUNS}/audit/${RUN_ID}"
mkdir -p "${RUN_DIR}"

SUBSET="${SUBSET:-${MLLMOPD_DATA}/audit/level1_subset_v0.jsonl}"
if [ ! -f "${SUBSET}" ]; then
  echo "ERROR: eval subset not found at ${SUBSET}" >&2
  echo "  This script reuses the canonical Level-1 1200-prompt subset; build it via" >&2
  echo "    python scripts/data/prep_audit_subset.py --out ${SUBSET} --size 2000" >&2
  echo "  (or rsync the file from the box that produced level1_v4_sysprompt_fixed)." >&2
  exit 1
fi

# --- cuDNN sanitation (pitfalls.md E1) --------------------------------------
# Stale system cuDNN 9.2 wins over the venv's 9.16 unless LD_LIBRARY_PATH
# is cleared. Without this, Qwen2.5-VL's Conv3d crashes mid-pass.
unset LD_LIBRARY_PATH

# --- Activate eval env ------------------------------------------------------
# shellcheck disable=SC1091
source scripts/env/_activate.sh

# --- Extra args (limit / debug / max_new_tokens) -----------------------------
EXTRA_ARGS=()
if [ "${AUDIT_LIMIT:-0}" != "0" ]; then
  EXTRA_ARGS+=(--limit "${AUDIT_LIMIT}")
fi
if [ "${AUDIT_DEBUG:-0}" = "1" ]; then
  EXTRA_ARGS+=(--debug)
fi
if [ -n "${AUDIT_MAX_NEW_TOKENS:-}" ]; then
  EXTRA_ARGS+=(--max-new-tokens "${AUDIT_MAX_NEW_TOKENS}")
fi

# --- Backend selection (default sglang; T1 lives in the train venv) ---------
case "${AUDIT_BACKEND:-sglang}" in
  hf)
    export AUDIT_BACKEND_MODULE="mllmopd.diagnostics.run_audit_pass"
    if [ "${AUDIT_BATCH:-1}" != "1" ]; then
      EXTRA_ARGS+=(--batch-size "${AUDIT_BATCH}")
    fi
    echo ">>> Using HF backend — slower than sglang; suitable only for smoke runs"
    ;;
  sglang)
    export AUDIT_BACKEND_MODULE="mllmopd.diagnostics.run_audit_pass_sglang"
    if [ "${AUDIT_BATCH:-16}" != "1" ]; then
      EXTRA_ARGS+=(--max-running-requests "${AUDIT_BATCH:-16}")
    fi
    if [ -n "${AUDIT_MEM_FRACTION:-0.70}" ]; then
      EXTRA_ARGS+=(--mem-fraction "${AUDIT_MEM_FRACTION:-0.70}")
    fi
    if ! python -c "import sglang" >/dev/null 2>&1; then
      echo "ERROR: sglang not importable in $(command -v python)." >&2
      echo "  Activate the train venv first, e.g." >&2
      echo "    export MLLMOPD_VENV=/root/shihao_project/mllmopd-train-env/.venv" >&2
      echo "  then re-run this script." >&2
      exit 1
    fi
    echo ">>> Using sglang backend (mem_fraction=${AUDIT_MEM_FRACTION:-0.70}, max_running=${AUDIT_BATCH:-16})"
    ;;
  *)
    echo "ERROR: unknown AUDIT_BACKEND=${AUDIT_BACKEND} (use hf or sglang)" >&2
    exit 1
    ;;
esac

# --- MMR1 system prompt (verbatim from scripts/audit/run_smoke.sh:121).
# KEEP IN SYNC with run_smoke.sh and rerun_h2_sysprompt.sh: any divergence
# puts the T1 students in base-model mode at eval time and the
# `<think>/<answer>/\boxed{}` scaffold collapses → ~50% scorer-fallback rate
# and a fake 5-10pt accuracy gap vs MMR1's paper numbers.
MMR1_SYSTEM_PROMPT='A conversation between User and Assistant. The User provides an image and asks a question. The Assistant first analyzes both the image and the question, then carefully thinks about the reasoning process step by step, and finally provides the User with an accurate answer. The Assistant must carefully checkout the correctness and validity of each reasoning step. If any errors or inconsistencies are found during the reasoning process, the Assistant reflects and corrects them logically. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here, with potential reflections and corrections </think><answer> final answer here, with the key result enclosed in \boxed{} </answer>.'

# --- Pass matrix ------------------------------------------------------------
# Default 9-pass matrix: 3 models × 3 modes.
#   T1_0 = MMR1-3B-SFT pre-OPD (the student initial state)
#   T1_2 = FullTeacher OPD output  (CKPT_T1_2)
#   T1_3 = BlankTeacher OPD output (CKPT_T1_3)
# Each model gets full_image / blank_image / text_only so paired_vision_critical
# can compute VC[T1_X] and the per-benchmark G_full / G_blank / G_gap.
PASS_TAGS=()
PASS_MODELS=()
PASS_MODES=()

if [ "${SKIP_T1_0:-0}" != "1" ]; then
  PASS_TAGS+=(T1_0_full              T1_0_blank             T1_0_text_only)
  PASS_MODELS+=("${MMR1_3B_SFT_CKPT}" "${MMR1_3B_SFT_CKPT}" "${MMR1_3B_SFT_CKPT}")
  PASS_MODES+=(full_image             blank_image            text_only)
else
  echo ">>> SKIP_T1_0=1 — T1-0 baseline passes skipped (will reuse level1_v4 numbers)"
fi

PASS_TAGS+=(T1_2_full           T1_2_blank          T1_2_text_only
            T1_3_full           T1_3_blank          T1_3_text_only)
PASS_MODELS+=("${CKPT_T1_2}"    "${CKPT_T1_2}"      "${CKPT_T1_2}"
              "${CKPT_T1_3}"    "${CKPT_T1_3}"      "${CKPT_T1_3}")
PASS_MODES+=(full_image          blank_image         text_only
             full_image          blank_image         text_only)

# All passes use the MMR1 system prompt (no Base_* exceptions here — all
# three checkpoints derive from MMR1-3B-SFT, which was MMR1-trained).
PASS_SYSTEM_PROMPTS=()
for _ in "${PASS_TAGS[@]}"; do
  PASS_SYSTEM_PROMPTS+=("${MMR1_SYSTEM_PROMPT}")
done

# --- Banner -----------------------------------------------------------------
echo "================================================================"
echo "  T1 eval RUN_ID            : ${RUN_ID}"
echo "  RUN_DIR                   : ${RUN_DIR}"
echo "  SUBSET                    : ${SUBSET}"
echo "  T1-0 (MMR1-3B-SFT)        : ${MMR1_3B_SFT_CKPT}"
echo "  T1-2 (FullTeacher OPD)    : ${CKPT_T1_2}"
echo "  T1-3 (BlankTeacher OPD)   : ${CKPT_T1_3}"
echo "  SKIP_T1_0                 : ${SKIP_T1_0:-0}"
echo "  passes                    : ${#PASS_TAGS[@]} (tags: ${PASS_TAGS[*]})"
echo "  GPUs                      : ${SMOKE_GPUS:-${SMOKE_GPU:-0}}"
echo "================================================================"

# --- Dispatch ---------------------------------------------------------------
# shellcheck source=../env/_dispatch_passes.sh disable=SC1091
source scripts/env/_dispatch_passes.sh

# --- Aggregate --------------------------------------------------------------
python -m mllmopd.analysis.aggregate_audit \
  --run_dir "${RUN_DIR}" \
  --out "${RUN_DIR}/summary.json"

python -m mllmopd.reporting.audit_table --run_dir "${RUN_DIR}"

# --- Paired vision-critical (per-benchmark opd_target_recovery) -------------
# For each T1 student, compute VC[student], teacher_advantage relative to
# a reference teacher pass, and opd_target. We have no teacher pass in THIS
# run dir (the audit grid is student-only), so we point --teacher and
# --student at two of the three students to get per-arm subset stats.
#
# Specifically we run paired_vision_critical twice with T1-0 as the
# "teacher" baseline and each T1 arm as the "student". This gives:
#   - opd_target_T1_2  = prompts where T1-2 picked up an answer T1-0 missed
#                        AND that gain is vision-conditioned for T1-2.
#   - opd_target_T1_3  = same for the BlankTeacher arm.
# Δ = |opd_target_T1_2| − |opd_target_T1_3| is the headline gain expected
# under the vision-conditioned capability transfer hypothesis.
if [ "${SKIP_PAIRED_VC:-0}" != "1" ] && [ "${SKIP_T1_0:-0}" != "1" ]; then
  echo
  echo "================================================================"
  echo "  Paired vision-critical: T1-2 (FullTeacher) vs T1-0 baseline"
  echo "================================================================"
  # NOTE: --teacher/--student match the filename-derived arm label
  # (jsonl stem minus _full/_blank/_text_only), NOT the ckpt basename.
  # T1-2 and T1-3 both save HF as `step_249`; using basename here caused
  # the second-loaded arm to overwrite the first in correct_map (silent
  # bug, fixed 2026-05-21).
  python -m mllmopd.analysis.paired_vision_critical \
    --run_dir "${RUN_DIR}" \
    --teacher T1_0 \
    --student T1_2 \
    --out-target-ids "${RUN_DIR}/opd_target_ids_T1_2_vs_T1_0.json"

  echo
  echo "================================================================"
  echo "  Paired vision-critical: T1-3 (BlankTeacher) vs T1-0 baseline"
  echo "================================================================"
  python -m mllmopd.analysis.paired_vision_critical \
    --run_dir "${RUN_DIR}" \
    --teacher T1_0 \
    --student T1_3 \
    --out-target-ids "${RUN_DIR}/opd_target_ids_T1_3_vs_T1_0.json"
else
  echo ">>> paired_vision_critical skipped (SKIP_PAIRED_VC=${SKIP_PAIRED_VC:-0}, SKIP_T1_0=${SKIP_T1_0:-0})"
fi

# --- Hint at the headline Δ analyzer ----------------------------------------
echo
echo "================================================================"
echo ">>> T1 eval complete: ${RUN_DIR}"
echo ">>> Pull back to Mac for analysis:"
echo "    rsync -avh devbox:${RUN_DIR}/ ./runs/audit/${RUN_ID}/"
echo
echo ">>> Next step: headline Δ = (T1-2 − T1-0) − (T1-3 − T1-0)"
echo "    python -m mllmopd.analysis.t1_compare \\"
echo "      --t1-run-dir ${RUN_DIR} \\"
echo "      --baseline-dir ${BASELINE_RUN_DIR:-${MLLMOPD_RUNS}/audit/level1_v4_sysprompt_fixed} \\"
echo "      --out-json ${RUN_DIR}/t1_compare.json"
echo
echo ">>> Baseline naming: t1_compare reads S_full/S_blank/S_text_only from"
echo ">>> the canonical baseline dir (level1_v4_sysprompt_fixed). T1_0_*.jsonl"
echo ">>> in ${RUN_DIR} are the freshly re-run T1-0 passes used only by"
echo ">>> paired_vision_critical above (hardware-parity sanity check)."
echo "================================================================"
