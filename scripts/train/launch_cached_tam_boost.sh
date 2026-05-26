#!/usr/bin/env bash
# Step 3a A0/A1 launcher — Cached TAM-Boost OPD.
#
# Thin wrapper over scripts/train/opd_mmr1_3b_baseline_xbox.sh. Swaps the
# post-process hook to tam_boost_hook.post_process_rewards_with_tam_boost
# (which is a superset of opd_diagnostics_hook) and asserts the env-var
# contract per docs/step3a-design-2026-05-26.md + GPT verdict §3.
#
# Arms (env-driven; both share the same launcher):
#
#   A0 (off-policy KD reference, no boost):
#     MLLMOPD_USE_TAM_BOOST=0
#     MLLMOPD_TAM_CACHE_JSONL=""              # unused
#
#   A1 (cached TAM-Boost main):
#     MLLMOPD_USE_TAM_BOOST=1
#     MLLMOPD_TAM_CACHE_JSONL=<precompute path>
#
#   A2 (category-only ablation):
#     MLLMOPD_USE_TAM_BOOST=1
#     MLLMOPD_TAM_MODE=category_only
#     MLLMOPD_TAM_CACHE_JSONL=<precompute path>
#
#   A3 (random-region ablation):
#     MLLMOPD_USE_TAM_BOOST=1
#     MLLMOPD_TAM_MODE=random_region
#     MLLMOPD_TAM_RANDOM_RATE=0.40            # match A1 expected rate
#     MLLMOPD_TAM_CACHE_JSONL=<precompute path>
#
#   A4 (scrambled-region ablation):
#     MLLMOPD_USE_TAM_BOOST=1
#     MLLMOPD_TAM_MODE=scrambled
#     MLLMOPD_TAM_CACHE_JSONL=<precompute path>
#
#   A5 (oracle quad — diagnostic, two-forward):
#     MLLMOPD_USE_TAM_BOOST=1
#     MLLMOPD_TAM_MODE=oracle_quad
#     MLLMOPD_TAM_CACHE_JSONL=<precompute path with quad data>
#     (requires extended precompute incl. quad; not the smoke path)
#
# All arms inherit knobs (K=0.20, ρ=0.30, τ=0.70, α=0.50) from
# src/mllmopd/training/tam_gate.py:GateConfig defaults. Override via:
#     MLLMOPD_TAM_K / MLLMOPD_TAM_RHO / MLLMOPD_TAM_TAU / MLLMOPD_TAM_ALPHA
#
# Required env (delegated to baseline launcher; see that file for full list):
#   MLLMOPD_RUNS / MMR1_*_CKPT / TRAIN_JSONL / etc.
#
# Usage:
#   bash scripts/train/launch_cached_tam_boost.sh
#
# Smoke / 3k / 15k progression is controlled by the same NUM_ROLLOUT_STEPS /
# rollout batch knobs the baseline uses. Suggested progression per
# docs/step3a-design-2026-05-26.md §"Execution gates":
#   Gate 1 — 50–100 sample plumbing smoke (NUM_ROLLOUT_STEPS=4-8, small bs)
#   Gate 2 — 3k stratified A0/A1     (NUM_ROLLOUT_STEPS=~200)
#   Gate 3 — 15k full A0/A1/A5

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# --- Step 3a env contract ---
: "${MLLMOPD_USE_TAM_BOOST:=0}"
: "${MLLMOPD_TAM_CACHE_JSONL:=}"
: "${MLLMOPD_TAM_MODE:=main}"

# Validate mode
case "${MLLMOPD_TAM_MODE}" in
  main|category_only|random_region|scrambled|oracle_quad) ;;
  *)
    echo "ERROR: MLLMOPD_TAM_MODE must be one of "\
         "{main,category_only,random_region,scrambled,oracle_quad}, "\
         "got '${MLLMOPD_TAM_MODE}'" >&2
    exit 1
    ;;
esac

# A1+ require a cache file
if [ "${MLLMOPD_USE_TAM_BOOST}" = "1" ] && [ -z "${MLLMOPD_TAM_CACHE_JSONL}" ]; then
  echo "ERROR: MLLMOPD_USE_TAM_BOOST=1 but MLLMOPD_TAM_CACHE_JSONL unset. "\
       "Run scripts/audit/tam_precompute_train_pool.py first." >&2
  exit 1
fi
if [ "${MLLMOPD_USE_TAM_BOOST}" = "1" ] && [ ! -f "${MLLMOPD_TAM_CACHE_JSONL}" ]; then
  echo "ERROR: MLLMOPD_TAM_CACHE_JSONL=${MLLMOPD_TAM_CACHE_JSONL} not a file" >&2
  exit 1
fi

# Mandatory invariant per GPT verdict §3 point 5: TAM-Boost requires the
# no-suppress semantics, which whitening would destroy. The baseline
# launcher omits --normalize-advantages (default False); we re-assert here
# in case anyone added it.
if grep -q -- "--normalize-advantages" scripts/train/opd_mmr1_3b_baseline_xbox.sh; then
  echo "ERROR: baseline launcher contains --normalize-advantages flag; "\
       "TAM-Boost no-suppress semantics (w_t >= 1) cannot survive "\
       "advantage whitening. Remove the flag or use a different launcher." >&2
  exit 1
fi

# Export TAM env to subprocess + Uni-OPD trainer (which forwards to actors)
export MLLMOPD_USE_TAM_BOOST MLLMOPD_TAM_CACHE_JSONL MLLMOPD_TAM_MODE
export MLLMOPD_TAM_K="${MLLMOPD_TAM_K:-0.20}"
export MLLMOPD_TAM_RHO="${MLLMOPD_TAM_RHO:-0.30}"
export MLLMOPD_TAM_TAU="${MLLMOPD_TAM_TAU:-0.70}"
export MLLMOPD_TAM_ALPHA="${MLLMOPD_TAM_ALPHA:-0.50}"
export MLLMOPD_TAM_RANDOM_RATE="${MLLMOPD_TAM_RANDOM_RATE:-0.40}"
export MLLMOPD_TAM_SEED="${MLLMOPD_TAM_SEED:-4096}"

# Swap the post-process hook
export MLLMOPD_POST_PROCESS_HOOK="mllmopd.training.tam_boost_hook.post_process_rewards_with_tam_boost"

# Banner
cat <<EOF
========================================
Cached TAM-Boost OPD launching
========================================
  USE_TAM_BOOST   = ${MLLMOPD_USE_TAM_BOOST}
  TAM_MODE        = ${MLLMOPD_TAM_MODE}
  TAM_CACHE_JSONL = ${MLLMOPD_TAM_CACHE_JSONL:-<unset (A0)>}
  K / ρ / τ / α   = ${MLLMOPD_TAM_K} / ${MLLMOPD_TAM_RHO} / ${MLLMOPD_TAM_TAU} / ${MLLMOPD_TAM_ALPHA}
  POST_PROCESS_HOOK = ${MLLMOPD_POST_PROCESS_HOOK}
========================================
EOF

# Hand off to the baseline launcher. It picks up MLLMOPD_POST_PROCESS_HOOK
# and inserts it into --custom-reward-post-process-path.
exec bash scripts/train/opd_mmr1_3b_baseline_xbox.sh "$@"
