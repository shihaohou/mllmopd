#!/usr/bin/env bash
# T2-1 eval: re-run the canonical Level-1 audit on the T2-1 VD-weighted
# FullTeacher OPD student, paired with the T1-2 uniform FullTeacher OPD
# student (the headline control) and the T1-0 untrained MMR1-3B-SFT
# baseline. Mirrors scripts/audit/run_t1_eval.sh's plumbing — same
# 6 × 200 prompt grid, same MMR1 system prompt, same scoring path —
# so the resulting jsonls drop cleanly into the same t1_compare /
# paired_vision_critical analysis chain.
#
# What it does:
#   1. Reuses data/audit/level1_subset_v0.jsonl (the same grid that
#      level1_v4_sysprompt_fixed and t1_v1p5b_eval used).
#   2. Default pass matrix: 3 students × 3 modes = 9 passes
#       - T1_0 (MMR1-3B-SFT pre-OPD, baseline)
#       - T1_2 (vanilla FullTeacher OPD, control)
#       - T2_1 (VD-weighted FullTeacher OPD, treatment)
#     Each gets full_image / blank_image / text_only.
#   3. Injects the MMR1 training-time system prompt verbatim from
#      run_t1_eval.sh (the only one keeping T1-* students in MMR1-mode).
#   4. Aggregates → summary.json, prints headline table.
#   5. Runs paired_vision_critical for T1-2 vs T1-0 AND T2-1 vs T1-0
#      to get per-benchmark opd_target_recovery on both treatments.
#   6. Hints at `t2_1_compare.py` for the headline Δ = T2-1 - T1-2
#      scalar (which collapses to direct accuracy diff, since T1-0
#      cancels out of (G[T2-1] - G[T1-2])).
#
# Wall-time: ~12 min × 3 arms × 3 modes = ~25 min on 1 GPU; ~7 min
# with SMOKE_GPUS=0,1,2,3 (4-way parallel).
#
# Required env vars (sourced from .env):
#   MLLMOPD_RUNS         — base output dir (NFS / ceph)
#   MLLMOPD_DATA         — base data dir
#   MMR1_3B_SFT_CKPT     — T1-0 baseline (re-run unless SKIP_T1_0=1)
#   CKPT_T1_2            — T1-2 FullTeacher HF checkpoint (control)
#   CKPT_T2_1            — T2-1 VD-weighted HF checkpoint (treatment)
#
# Optional env vars (defaults shown):
#   RUN_ID               — default: t2_1_eval_$(date +%Y%m%d-%H%M%S)
#   SUBSET               — default: ${MLLMOPD_DATA}/audit/level1_subset_v0.jsonl
#   SKIP_T1_0            — 1 = skip T1-0 re-run (no hardware parity, faster)
#   SKIP_T1_2            — 1 = skip T1-2 re-run (reuse existing T1 eval dir;
#                          requires manual symlink for paired_vision_critical)
#   AUDIT_BACKEND        — hf|sglang (default sglang)
#   AUDIT_BATCH          — sglang max-running-requests (default 16)
#   AUDIT_MEM_FRACTION   — sglang mem_fraction_static (default 0.70)
#   AUDIT_LIMIT          — limit prompts per pass for smoke runs (default 0)
#   AUDIT_DEBUG          — 1 = verbose per-pass logging
#   AUDIT_MAX_NEW_TOKENS — generation cap (default 4096)
#   SMOKE_GPUS / SMOKE_GPU — see _dispatch_passes.sh
#   SKIP_PAIRED_VC       — 1 = skip paired_vision_critical at the end
#   BASELINE_RUN_DIR     — default ${MLLMOPD_RUNS}/audit/level1_v4_sysprompt_fixed
#
# Usage:
#   # Full 9-pass eval (T1_0 + T1_2 + T2_1)
#   CKPT_T1_2=${MLLMOPD_RUNS}/t1_v1p5b_T1_2_full_mm/ckpt/hf/step_249 \
#   CKPT_T2_1=${MLLMOPD_RUNS}/t2_1_v0_T2_1_full_vd/ckpt/hf/step_249 \
#     bash scripts/audit/run_t2_1_eval.sh
#
#   # Skip T1_0 (use existing baseline); 6 passes
#   SKIP_T1_0=1 CKPT_T1_2=... CKPT_T2_1=... \
#     bash scripts/audit/run_t2_1_eval.sh
#
#   # Skip T1_2 too (reuse from prior t1_v1p5b_eval dir); 3 passes only
#   SKIP_T1_0=1 SKIP_T1_2=1 CKPT_T2_1=... \
#     bash scripts/audit/run_t2_1_eval.sh
#
#   # Parallel across 4 GPUs
#   SMOKE_GPUS=0,1,2,3 CKPT_T1_2=... CKPT_T2_1=... \
#     bash scripts/audit/run_t2_1_eval.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# CKPT_T1_2/T2_1 preservation — same caller-supplied-vs-.env collision
# guard as run_t1_eval.sh:73-91. .env may carry stale paths from
# earlier sessions; honor caller's explicit export.
_CKPT_T1_2_CALLER="${CKPT_T1_2:-}"
_CKPT_T2_1_CALLER="${CKPT_T2_1:-}"
# shellcheck disable=SC1091
source .env
if [ -n "${_CKPT_T1_2_CALLER}" ]; then
  CKPT_T1_2="${_CKPT_T1_2_CALLER}"
fi
if [ -n "${_CKPT_T2_1_CALLER}" ]; then
  CKPT_T2_1="${_CKPT_T2_1_CALLER}"
fi
unset _CKPT_T1_2_CALLER _CKPT_T2_1_CALLER

# --- Required env -----------------------------------------------------------
: "${MLLMOPD_RUNS:?MLLMOPD_RUNS must be set}"
: "${MLLMOPD_DATA:?MLLMOPD_DATA must be set}"
: "${MMR1_3B_SFT_CKPT:?MMR1_3B_SFT_CKPT must be set (T1-0 baseline)}"
: "${CKPT_T2_1:?CKPT_T2_1 must point at the T2-1 VD-weighted HF checkpoint (e.g. \${MLLMOPD_RUNS}/t2_1_v0_T2_1_full_vd/ckpt/hf/step_249)}"

if [ "${SKIP_T1_2:-0}" != "1" ]; then
  : "${CKPT_T1_2:?CKPT_T1_2 must point at the T1-2 FullTeacher HF checkpoint, OR set SKIP_T1_2=1 to reuse from a prior eval dir}"
fi

echo ">>> CKPT_T1_2 resolved : ${CKPT_T1_2:-(skipped)}"
echo ">>> CKPT_T2_1 resolved : ${CKPT_T2_1}"

# Fail fast on missing dirs.
for ck in "${MMR1_3B_SFT_CKPT}" "${CKPT_T2_1}"; do
  if [ ! -d "${ck}" ]; then
    echo "ERROR: checkpoint dir not found: ${ck}" >&2
    exit 1
  fi
done
if [ "${SKIP_T1_2:-0}" != "1" ] && [ ! -d "${CKPT_T1_2}" ]; then
  echo "ERROR: CKPT_T1_2 not found: ${CKPT_T1_2}" >&2
  exit 1
fi

RUN_ID="${RUN_ID:-t2_1_eval_$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${MLLMOPD_RUNS}/audit/${RUN_ID}"
mkdir -p "${RUN_DIR}"

SUBSET="${SUBSET:-${MLLMOPD_DATA}/audit/level1_subset_v0.jsonl}"
if [ ! -f "${SUBSET}" ]; then
  echo "ERROR: eval subset not found at ${SUBSET}" >&2
  echo "  Run scripts/data/prep_audit_subset.py first or rsync from a prior box." >&2
  exit 1
fi

# --- cuDNN sanitation -------------------------------------------------------
# Stale system cuDNN 9.2 wins over the venv's 9.16 unless this is cleared.
unset LD_LIBRARY_PATH

# --- Activate eval env ------------------------------------------------------
# shellcheck disable=SC1091
source scripts/env/_activate.sh

# --- Extra args -------------------------------------------------------------
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

# --- Backend ----------------------------------------------------------------
case "${AUDIT_BACKEND:-sglang}" in
  hf)
    export AUDIT_BACKEND_MODULE="mllmopd.diagnostics.run_audit_pass"
    if [ "${AUDIT_BATCH:-1}" != "1" ]; then
      EXTRA_ARGS+=(--batch-size "${AUDIT_BATCH}")
    fi
    echo ">>> HF backend (smoke only — slower than sglang)"
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
      exit 1
    fi
    echo ">>> sglang backend (mem_fraction=${AUDIT_MEM_FRACTION:-0.70}, max_running=${AUDIT_BATCH:-16})"
    ;;
  *)
    echo "ERROR: unknown AUDIT_BACKEND=${AUDIT_BACKEND} (use hf or sglang)" >&2
    exit 1
    ;;
esac

# --- MMR1 system prompt (KEEP IN SYNC with run_t1_eval.sh:192) ---------------
MMR1_SYSTEM_PROMPT='A conversation between User and Assistant. The User provides an image and asks a question. The Assistant first analyzes both the image and the question, then carefully thinks about the reasoning process step by step, and finally provides the User with an accurate answer. The Assistant must carefully checkout the correctness and validity of each reasoning step. If any errors or inconsistencies are found during the reasoning process, the Assistant reflects and corrects them logically. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here, with potential reflections and corrections </think><answer> final answer here, with the key result enclosed in \boxed{} </answer>.'

# --- Pass matrix ------------------------------------------------------------
# Default: 3 students × 3 modes = 9 passes
#   T1_0 = MMR1-3B-SFT (baseline; can be skipped via SKIP_T1_0=1)
#   T1_2 = vanilla FullTeacher OPD (control; can be skipped via SKIP_T1_2=1)
#   T2_1 = VD-weighted FullTeacher OPD (treatment)
PASS_TAGS=()
PASS_MODELS=()
PASS_MODES=()

if [ "${SKIP_T1_0:-0}" != "1" ]; then
  PASS_TAGS+=(T1_0_full              T1_0_blank             T1_0_text_only)
  PASS_MODELS+=("${MMR1_3B_SFT_CKPT}" "${MMR1_3B_SFT_CKPT}" "${MMR1_3B_SFT_CKPT}")
  PASS_MODES+=(full_image             blank_image            text_only)
else
  echo ">>> SKIP_T1_0=1 — T1-0 baseline passes skipped"
fi

if [ "${SKIP_T1_2:-0}" != "1" ]; then
  PASS_TAGS+=(T1_2_full           T1_2_blank          T1_2_text_only)
  PASS_MODELS+=("${CKPT_T1_2}"    "${CKPT_T1_2}"      "${CKPT_T1_2}")
  PASS_MODES+=(full_image          blank_image         text_only)
else
  echo ">>> SKIP_T1_2=1 — T1-2 control passes skipped (compare manually against prior eval)"
fi

PASS_TAGS+=(T2_1_full           T2_1_blank          T2_1_text_only)
PASS_MODELS+=("${CKPT_T2_1}"    "${CKPT_T2_1}"      "${CKPT_T2_1}")
PASS_MODES+=(full_image          blank_image         text_only)

# All passes use the MMR1 system prompt — all 3 students derive from MMR1-3B-SFT.
PASS_SYSTEM_PROMPTS=()
for _ in "${PASS_TAGS[@]}"; do
  PASS_SYSTEM_PROMPTS+=("${MMR1_SYSTEM_PROMPT}")
done

# --- Banner -----------------------------------------------------------------
echo "================================================================"
echo "  T2-1 eval RUN_ID          : ${RUN_ID}"
echo "  RUN_DIR                   : ${RUN_DIR}"
echo "  SUBSET                    : ${SUBSET}"
echo "  T1-0 (MMR1-3B-SFT)        : ${MMR1_3B_SFT_CKPT}"
echo "  T1-2 (FullTeacher OPD)    : ${CKPT_T1_2:-(skipped)}"
echo "  T2-1 (VD-weighted)        : ${CKPT_T2_1}"
echo "  SKIP_T1_0                 : ${SKIP_T1_0:-0}"
echo "  SKIP_T1_2                 : ${SKIP_T1_2:-0}"
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

# --- Paired vision-critical -------------------------------------------------
# Run twice: T1-2 vs T1-0 (for hardware parity / sanity) and T2-1 vs T1-0
# (for opd_target_recovery on the VD-weighted arm). Both opd_target_ids
# .json files end up in RUN_DIR; t2_1_compare.py picks the right one
# for the headline.
if [ "${SKIP_PAIRED_VC:-0}" != "1" ] && [ "${SKIP_T1_0:-0}" != "1" ]; then
  if [ "${SKIP_T1_2:-0}" != "1" ]; then
    echo
    echo "================================================================"
    echo "  Paired vision-critical: T1-2 (control) vs T1-0 baseline"
    echo "================================================================"
    python -m mllmopd.analysis.paired_vision_critical \
      --run_dir "${RUN_DIR}" \
      --teacher T1_0 \
      --student T1_2 \
      --out-target-ids "${RUN_DIR}/opd_target_ids_T1_2_vs_T1_0.json"
  fi

  echo
  echo "================================================================"
  echo "  Paired vision-critical: T2-1 (treatment) vs T1-0 baseline"
  echo "================================================================"
  python -m mllmopd.analysis.paired_vision_critical \
    --run_dir "${RUN_DIR}" \
    --teacher T1_0 \
    --student T2_1 \
    --out-target-ids "${RUN_DIR}/opd_target_ids_T2_1_vs_T1_0.json"
else
  echo ">>> paired_vision_critical skipped (SKIP_PAIRED_VC=${SKIP_PAIRED_VC:-0}, SKIP_T1_0=${SKIP_T1_0:-0})"
fi

# --- Headline hint ----------------------------------------------------------
echo
echo "================================================================"
echo ">>> T2-1 eval complete: ${RUN_DIR}"
echo ">>> Pull back to Mac for analysis:"
echo "    rsync -avh devbox:${RUN_DIR}/ ./runs/audit/${RUN_ID}/"
echo
echo ">>> Headline Δ_T2_1_vs_T1_2 = G_full[T2-1] - G_full[T1-2]"
echo "    = Acc[T2-1, full] - Acc[T1-2, full]   (T1-0 cancels)"
echo "    python -m mllmopd.analysis.t2_1_compare \\"
echo "      --t2-1-run-dir ${RUN_DIR} \\"
echo "      --baseline-dir ${BASELINE_RUN_DIR:-${MLLMOPD_RUNS}/audit/level1_v4_sysprompt_fixed} \\"
echo "      --out-json ${RUN_DIR}/t2_1_compare.json"
echo
echo ">>> If SKIP_T1_2=1, point --t1-2-jsonl-dir at the prior eval dir"
echo ">>> that holds T1_2_{full,blank,text_only}.jsonl."
echo "================================================================"
