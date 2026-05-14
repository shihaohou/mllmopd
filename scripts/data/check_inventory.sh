#!/usr/bin/env bash
# Inventory check for the dev box. Prints which expected assets are present /
# missing under $MLLMOPD_MODELS_ROOT and $MLLMOPD_DATASETS_ROOT.
# Safe to run on Mac (will just print everything as missing).
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"
[ -f .env ] && source .env

check() {
  local label="$1" path="$2"
  if [ -e "${path}" ]; then
    local sz
    sz=$(du -sh "${path}" 2>/dev/null | awk '{print $1}')
    printf "  [✓] %-22s %s   (%s)\n" "${label}" "${path}" "${sz}"
  else
    printf "  [ ] %-22s %s   MISSING\n" "${label}" "${path}"
  fi
}

echo ">>> Models root: ${MLLMOPD_MODELS_ROOT:-<unset>}"
[ -n "${MLLMOPD_MODELS_ROOT:-}" ] && ls "${MLLMOPD_MODELS_ROOT}" 2>/dev/null | sed 's/^/    /'

echo
echo ">>> Datasets root: ${MLLMOPD_DATASETS_ROOT:-<unset>}"
[ -n "${MLLMOPD_DATASETS_ROOT:-}" ] && ls "${MLLMOPD_DATASETS_ROOT}" 2>/dev/null | sed 's/^/    /'

echo
echo ">>> Expected assets:"
check "MMR1-3B-SFT (student)"  "${MMR1_3B_SFT_CKPT:-}"
check "MMR1-7B-RL (teacher)"   "${MMR1_7B_RL_CKPT:-}"
check "MMR1-7B-SFT (control)"  "${MMR1_7B_SFT_CKPT:-}"
check "MMR1-RL (data)"         "${MMR1_RL_DATA:-}"
check "ViRL39K"                "${VIRL39K_PATH:-}"
check "MathVista-mini"         "${MATHVISTA_PATH:-}"
check "POPE-adversarial"       "${POPE_PATH:-}"

echo
echo ">>> Missing benchmarks to download for full Level-1 (see docs/server-inventory.md):"
echo "    - MathVision, MathVerse, LogicVista, ChartQA, HallusionBench, CharXiv, MMMU"
