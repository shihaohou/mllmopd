#!/usr/bin/env bash
# Step 0 TAM sanity-check launcher.
#
# What it does:
#   1. Loads shared env (.env) for MLLMOPD_RUNS / MMR1_3B_SFT_CKPT.
#   2. Activates Python env via scripts/env/_activate.sh (Uni-OPD-LMMS-Eval
#      works; the train venv works too — only need HF transformers + scipy +
#      opencv + Pillow).
#   3. Unsets http_proxy/https_proxy (per project memory; container ships an
#      oversea-squid proxy that silently 502s on loopback HTTP).
#   4. Runs scripts/audit/tam_sanity.py on data/audit/tam_probes.jsonl with
#      MMR1-3B-SFT default, writes:
#        ${MLLMOPD_RUNS}/audit/${RUN_ID}/tam_sanity.jsonl
#        ${MLLMOPD_RUNS}/audit/${RUN_ID}/overlays/<id>/<...>.png
#        ${MLLMOPD_RUNS}/audit/${RUN_ID}/summary.txt
#
# Wall-time: ~3-5 min on H800 single GPU (4 probes × ~30s each).
#
# Required env vars (sourced from .env or pre-exported):
#   MLLMOPD_RUNS         — base output dir (NFS)
#   MMR1_3B_SFT_CKPT     — T1-0 baseline ckpt path (or HF id "MMR1/MMR1-3B-SFT")
#
# Optional env vars (defaults shown):
#   RUN_ID               — output subdir (default: tam_sanity_$(date +%Y%m%d-%H%M%S))
#   PROBES               — probe JSONL (default: data/audit/tam_probes.jsonl)
#   MODEL                — model id / ckpt path (default: ${MMR1_3B_SFT_CKPT})
#   MAX_NEW_TOKENS       — generation cap (default: 256; POPE answers are short)
#   SYSTEM_PROMPT_MODE   — mmr1|empty (default: mmr1; empty for plain Qwen2.5-VL)
#   CUDA_VISIBLE_DEVICES — which GPU (default: 0)
#   LIMIT                — limit n probes for fast debug (default: 0 = all)
#
# Decision criterion: see scripts/audit/tam_sanity.py module docstring.
#
# Usage:
#   bash scripts/audit/run_tam_sanity.sh
#   RUN_ID=tam_test MAX_NEW_TOKENS=128 bash scripts/audit/run_tam_sanity.sh
#   MODEL=Qwen/Qwen2.5-VL-3B-Instruct SYSTEM_PROMPT_MODE=empty \
#       bash scripts/audit/run_tam_sanity.sh

set -euo pipefail

# --- 1. shared env ---
if [ -f .env ]; then
  # shellcheck disable=SC1091
  source .env
fi
: "${MLLMOPD_RUNS:?MLLMOPD_RUNS must be set (export or .env)}"

# --- 2. python env activation ---
unset LD_LIBRARY_PATH || true
# shellcheck source=../env/_activate.sh disable=SC1091
source scripts/env/_activate.sh

# --- 3. proxy gotcha (per project_h800_proxy memory) ---
# Container ships http_proxy pointed at oversea squid; loopback HTTP requests
# silently 502 → process "Killed". Lowercase only; do not export NO_PROXY.
unset -v http_proxy https_proxy no_proxy || true

# --- 4. defaults / arg derivation ---
TS="$(date +%Y%m%d-%H%M%S)"
RUN_ID="${RUN_ID:-tam_sanity_${TS}}"
RUN_DIR="${MLLMOPD_RUNS}/audit/${RUN_ID}"
PROBES="${PROBES:-data/audit/tam_probes.jsonl}"
MODEL="${MODEL:-${MMR1_3B_SFT_CKPT:-MMR1/MMR1-3B-SFT}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
SYSTEM_PROMPT_MODE="${SYSTEM_PROMPT_MODE:-mmr1}"
LIMIT="${LIMIT:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ "${SYSTEM_PROMPT_MODE}" = "empty" ]; then
  SYSTEM_PROMPT_ARG=(--system-prompt "")
else
  SYSTEM_PROMPT_ARG=()  # let tam_sanity.py default to MMR1_SYSTEM_PROMPT
fi

LIMIT_ARG=()
if [ "${LIMIT}" != "0" ]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

mkdir -p "${RUN_DIR}"

cat <<EOF
========================================
TAM Step 0 sanity check launching
========================================
  RUN_ID       = ${RUN_ID}
  RUN_DIR      = ${RUN_DIR}
  MODEL        = ${MODEL}
  PROBES       = ${PROBES}
  MAX_NEW_TOK  = ${MAX_NEW_TOKENS}
  SYSPROMPT    = ${SYSTEM_PROMPT_MODE}
  CUDA_VISIBLE = ${CUDA_VISIBLE_DEVICES}
========================================
EOF

# --- 5. run ---
python -m scripts.audit.tam_sanity \
  --probes "${PROBES}" \
  --model "${MODEL}" \
  --out-dir "${RUN_DIR}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --image-root . \
  "${SYSTEM_PROMPT_ARG[@]}" \
  "${LIMIT_ARG[@]}"

# --- 6. brief post-run summary ---
echo
echo "========================================"
echo "TAM sanity DONE"
echo "========================================"
echo "  JSONL   : ${RUN_DIR}/tam_sanity.jsonl"
echo "  Overlays: ${RUN_DIR}/overlays/"
echo "  Summary : ${RUN_DIR}/summary.txt"
echo
if [ -f "${RUN_DIR}/summary.txt" ]; then
  echo "----- summary.txt (tail) -----"
  tail -n 40 "${RUN_DIR}/summary.txt"
fi
