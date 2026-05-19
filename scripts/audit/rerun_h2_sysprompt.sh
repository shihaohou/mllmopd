#!/usr/bin/env bash
# Pre-flight gate G1 (2026-05-19): rerun H2 forced-decode with MMR1's
# training-time system prompt injected BEFORE the image, matching the
# canonical Level-1 audit prefix.
#
# Why: pre-G1, score_completion.py constructed the chat prefix as
# [image, text:question] with NO system prompt, leaving MMR1 in
# base-model mode. The Level-1 audit was unaffected (it always passed
# sysprompt). After G1 the H2 forced-decode now reuses
# `_build_messages([text:sysprompt, image, text:question])` so the
# token-level VD distribution is measured under the MMR1-mode prefix.
#
# Acceptance: compare new vd_summary.sysprompt.json against the existing
# vd_summary.json. If (%tokens, %NLL mass) in high+very_high bins
# shifts by ≤ 2 absolute pp, Finding 2 is reproduced and T1 is green-lit.
# Larger shift → re-author Finding 2 with the new numbers BEFORE T1.
#
# Runs on the train venv (sglang installed). Devbox A800 ok.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source .env

: "${MLLMOPD_RUNS:?}"
: "${MMR1_7B_RL_CKPT:?}"

RUN_DIR="${RUN_DIR:-${MLLMOPD_RUNS}/audit/level1_v4_sysprompt_fixed}"
SUBSET="${SUBSET:-${MLLMOPD_DATA}/audit/level1_subset_v0.jsonl}"
SOURCE="${RUN_DIR}/T_RL_full.jsonl"
ID_FILTER="${RUN_DIR}/opd_target_ids.json"

OUT_SCORED="${RUN_DIR}/T_RL_score_opd_target.sysprompt.jsonl"
OUT_SUMMARY="${RUN_DIR}/vd_summary.sysprompt.json"

# Verbatim from scripts/audit/run_smoke.sh:121 (MMR1's training-time
# system prompt; sourced here as a single-line string). KEEP IN SYNC.
MMR1_SYSTEM_PROMPT='A conversation between User and Assistant. The User provides an image and asks a question. The Assistant first analyzes both the image and the question, then carefully thinks about the reasoning process step by step, and finally provides the User with an accurate answer. The Assistant must carefully checkout the correctness and validity of each reasoning step. If any errors or inconsistencies are found during the reasoning process, the Assistant reflects and corrects them logically. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here, with potential reflections and corrections </think><answer> final answer here, with the key result enclosed in \boxed{} </answer>.'

# Sanity-check inputs.
for f in "${SUBSET}" "${SOURCE}" "${ID_FILTER}"; do
  if [ ! -f "${f}" ]; then
    echo "ERROR: missing required input ${f}" >&2
    exit 1
  fi
done

# Venv: do NOT call scripts/env/_activate.sh here — that hard-prefers the
# project-local .venv (audit/HF env), which lacks sglang. The operator must
# pre-activate the train venv (the one with sglang + Uni-OPD installed).
# Typical devbox path: /root/shihao_project/mllmopd-train-env/.venv/bin/activate
# Override via MLLMOPD_VENV if you want _activate.sh-style selection later.
if ! python -c "import sglang" >/dev/null 2>&1; then
  echo "ERROR: sglang not importable in the current Python env." >&2
  echo "  current python: $(command -v python)" >&2
  echo "  fix: source /root/shihao_project/mllmopd-train-env/.venv/bin/activate" >&2
  echo "       (or your local equivalent train-venv path)" >&2
  exit 1
fi

# Memory knobs. The pre-G1 score_completion defaults (mem_fraction=0.85,
# max_running=64) OOM'd on A800-80GB even though there's only ~96k tokens
# of forced-decode work total — sglang grabs static reservation up front
# plus per-request KV cache. Conservative defaults below; override via env
# for tuning. `expandable_segments` helps fragmentation if KV cache churns.
MEM_FRACTION="${MEM_FRACTION:-0.70}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo ">>> rerun_h2_sysprompt: scoring ${SOURCE} against opd_target_ids.json"
echo ">>> output: ${OUT_SCORED}"
echo ">>> using MMR1_SYSTEM_PROMPT (first 60 chars):"
echo "    ${MMR1_SYSTEM_PROMPT:0:60}..."
echo ">>> mem_fraction=${MEM_FRACTION}  max_running_requests=${MAX_RUNNING_REQUESTS}"
echo ">>> PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

# Forced-decode rerun. Both --system-prompt-text and --id-filter are
# the post-G1 additions to score_completion.py.
python -m mllmopd.diagnostics.score_completion \
  --subset "${SUBSET}" \
  --source "${SOURCE}" \
  --model "${MMR1_7B_RL_CKPT}" \
  --system-prompt-text "${MMR1_SYSTEM_PROMPT}" \
  --id-filter "${ID_FILTER}" \
  --mem-fraction "${MEM_FRACTION}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --out "${OUT_SCORED}"

echo
echo ">>> aggregating to ${OUT_SUMMARY}"
python -m mllmopd.analysis.aggregate_vd \
  --scored "${OUT_SCORED}" \
  --out-table "${OUT_SUMMARY}"

echo
echo "================================================================"
echo "G1 acceptance check: diff bin shares vs pre-G1 vd_summary.json"
echo "================================================================"
ORIG="${RUN_DIR}/vd_summary.json"
if [ -f "${ORIG}" ]; then
  python - "${ORIG}" "${OUT_SUMMARY}" <<'PY'
import json, sys
orig, new = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
def share(d):
    tot_n = sum(b["n"] for b in d.values())
    tot_nll = sum(b["nll_full"] for b in d.values())
    return {k: (b["n"]/tot_n*100, b["nll_full"]/tot_nll*100) for k, b in d.items()}
so, sn = share(orig["overall"]), share(new["overall"])
print(f"  {'bin':<28s} {'pre-G1 %tok':>12s} {'post-G1 %tok':>13s} {'Δpp':>7s}  "
      f"{'pre-G1 %nll':>12s} {'post-G1 %nll':>13s} {'Δpp':>7s}")
worst_tok = worst_nll = 0.0
for k in so:
    dt = sn[k][0] - so[k][0]; dn = sn[k][1] - so[k][1]
    worst_tok = max(worst_tok, abs(dt)); worst_nll = max(worst_nll, abs(dn))
    print(f"  {k:<28s} {so[k][0]:>11.2f}% {sn[k][0]:>12.2f}% {dt:>+6.2f}  "
          f"{so[k][1]:>11.2f}% {sn[k][1]:>12.2f}% {dn:>+6.2f}")
print()
print(f"  worst |Δpp| in %tokens: {worst_tok:.2f}")
print(f"  worst |Δpp| in %NLL:   {worst_nll:.2f}")
if worst_tok <= 2.0 and worst_nll <= 2.0:
    print("  >>> ACCEPTANCE PASSED: shifts within ±2pp tolerance. Finding 2 reproduced.")
    sys.exit(0)
else:
    print("  >>> ACCEPTANCE FAILED: shift exceeds 2pp. Re-author Finding 2 with new numbers BEFORE launching T1.")
    sys.exit(1)
PY
else
  echo "  (no pre-G1 vd_summary.json at ${ORIG} to diff against — manual review needed)"
fi
