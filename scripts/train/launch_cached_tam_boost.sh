#!/usr/bin/env bash
# Step 3a — Sparse Visual-Conditioned OPD launcher (v3, 2026-05-26).
#
# Thin wrapper over scripts/train/opd_mmr1_3b_baseline*.sh (auto-dispatch
# single-box / Xbox). Swaps the post-process hook to
# tam_boost_hook.post_process_rewards_with_tam_boost (which is a superset
# of opd_diagnostics_hook) and configures it via env vars.
#
# v3 method: true on-policy OPD with a category gate informed by TAM's
# Step 2 causal-masking discovery. NO cache, NO spatial gate at train time.
# See docs/step3a-design-2026-05-26-v3.md.
#
# File name kept (was "cached_tam_boost") for git/grep continuity even
# though the default mode is no longer cache-based.
#
# Arms (env-driven, single launcher):
#
#   B0  baseline (no boost):
#     MLLMOPD_USE_TAM_BOOST=0
#
#   B1  Sparse Visual-Conditioned OPD (main, v3 default):
#     MLLMOPD_USE_TAM_BOOST=1
#     MLLMOPD_TAM_HOOK_MODE=onpolicy_category      # default
#     MLLMOPD_TAM_ALPHA=<calibrated>               # from fire-rate audit, see below
#
#   B2  rate-matched random control (critical per GPT verdict):
#     MLLMOPD_USE_TAM_BOOST=1
#     MLLMOPD_TAM_HOOK_MODE=random_rate_matched
#     MLLMOPD_TAM_TARGET_RATE=<B1's measured fire rate>
#     MLLMOPD_TAM_ALPHA=<same as B1>
#
#   B3 (optional)  non-local category control:
#     MLLMOPD_USE_TAM_BOOST=1
#     MLLMOPD_TAM_HOOK_MODE=onpolicy_category
#     MLLMOPD_TAM_C_LOCAL=template_token,punctuation,other  # negative cats
#
#   diagnostic  cached_spatial (Path C, off-policy KD scenario; demoted):
#     MLLMOPD_USE_TAM_BOOST=1
#     MLLMOPD_TAM_HOOK_MODE=cached_spatial
#     MLLMOPD_TAM_CACHE_JSONL=<precompute path>
#     MLLMOPD_TAM_K=0.20 RHO=0.30 TAU=0.70 ALPHA=0.50
#
# α calibration (B1):
#   1. Run scripts/audit/tam_step3_b_firerate_audit.py on a small sample of
#      student rollouts to measure fire_rate under C_local categories.
#   2. Pick α s.t. target_mean_w ≈ 1.05–1.06:
#        α = (target_mean_w - 1) / fire_rate
#   3. Use same α for B2 (rate-matched).
#
# Required env (delegated to baseline launcher; see that file for full list):
#   MLLMOPD_RUNS / MMR1_*_CKPT / TRAIN_JSONL / TEACHER_HOST / ROLLOUT_ENGINE_ADDRS / NCCL_SOCKET_IFNAME
#
# Usage:
#   # B1 with locked α (after fire-rate audit gives 0.20):
#   MLLMOPD_USE_TAM_BOOST=1 MLLMOPD_TAM_ALPHA=0.20 \
#       bash scripts/train/launch_cached_tam_boost.sh
#
#   # B2 with same fire rate as B1 (say 0.12):
#   MLLMOPD_USE_TAM_BOOST=1 MLLMOPD_TAM_HOOK_MODE=random_rate_matched \
#       MLLMOPD_TAM_TARGET_RATE=0.12 MLLMOPD_TAM_ALPHA=0.20 \
#       bash scripts/train/launch_cached_tam_boost.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# --- Step 3a v3 env contract ---
: "${MLLMOPD_USE_TAM_BOOST:=0}"
: "${MLLMOPD_TAM_HOOK_MODE:=onpolicy_category}"
: "${MLLMOPD_TAM_ALPHA:=0.50}"

# Validate hook mode (must match VALID_HOOK_MODES in tam_boost_hook.py)
case "${MLLMOPD_TAM_HOOK_MODE}" in
  onpolicy_category|random_rate_matched|cached_spatial) ;;
  *)
    echo "ERROR: MLLMOPD_TAM_HOOK_MODE must be one of "\
         "{onpolicy_category, random_rate_matched, cached_spatial}, "\
         "got '${MLLMOPD_TAM_HOOK_MODE}'" >&2
    exit 1
    ;;
esac

# Mode-specific validation
if [ "${MLLMOPD_USE_TAM_BOOST}" = "1" ]; then
  case "${MLLMOPD_TAM_HOOK_MODE}" in
    cached_spatial)
      if [ -z "${MLLMOPD_TAM_CACHE_JSONL:-}" ]; then
        echo "ERROR: cached_spatial mode requires MLLMOPD_TAM_CACHE_JSONL" >&2
        exit 1
      fi
      if [ ! -f "${MLLMOPD_TAM_CACHE_JSONL}" ]; then
        echo "ERROR: MLLMOPD_TAM_CACHE_JSONL=${MLLMOPD_TAM_CACHE_JSONL} not a file" >&2
        exit 1
      fi
      ;;
    random_rate_matched)
      if [ -z "${MLLMOPD_TAM_TARGET_RATE:-}" ]; then
        echo "ERROR: random_rate_matched mode requires MLLMOPD_TAM_TARGET_RATE "\
             "(use B1's measured fire rate)" >&2
        exit 1
      fi
      ;;
    onpolicy_category)
      # No cache, no target_rate needed
      :
      ;;
  esac
fi

# Pick baseline launcher: Xbox vs single-box auto-dispatch.
if [ -n "${MLLMOPD_BASELINE_LAUNCHER:-}" ]; then
  BASELINE_LAUNCHER="${MLLMOPD_BASELINE_LAUNCHER}"
elif [ -n "${ROLLOUT_ENGINE_ADDRS:-}" ]; then
  BASELINE_LAUNCHER="scripts/train/opd_mmr1_3b_baseline_xbox.sh"
else
  BASELINE_LAUNCHER="scripts/train/opd_mmr1_3b_baseline.sh"
fi
echo ">>> baseline launcher = ${BASELINE_LAUNCHER}"

# Per GPT verdict §3 point 5: no-suppress semantics require
# --normalize-advantages = False.
if grep -q -- "--normalize-advantages" "${BASELINE_LAUNCHER}"; then
  echo "ERROR: ${BASELINE_LAUNCHER} contains --normalize-advantages flag; "\
       "no-suppress semantics (w_t >= 1) cannot survive advantage whitening." >&2
  exit 1
fi

# Export hook env to subprocess + Uni-OPD trainer
export MLLMOPD_USE_TAM_BOOST
export MLLMOPD_TAM_HOOK_MODE
export MLLMOPD_TAM_ALPHA
export MLLMOPD_TAM_CACHE_JSONL="${MLLMOPD_TAM_CACHE_JSONL:-}"
export MLLMOPD_TAM_TARGET_RATE="${MLLMOPD_TAM_TARGET_RATE:-}"
export MLLMOPD_TAM_C_LOCAL="${MLLMOPD_TAM_C_LOCAL:-content_noun,visual_attribute,proper_noun}"
export MLLMOPD_TAM_K="${MLLMOPD_TAM_K:-0.20}"
export MLLMOPD_TAM_RHO="${MLLMOPD_TAM_RHO:-0.30}"
export MLLMOPD_TAM_TAU="${MLLMOPD_TAM_TAU:-0.70}"
export MLLMOPD_TAM_SEED="${MLLMOPD_TAM_SEED:-4096}"

# Swap the post-process hook
export MLLMOPD_POST_PROCESS_HOOK="mllmopd.training.tam_boost_hook.post_process_rewards_with_tam_boost"

# Banner
cat <<EOF
========================================
Sparse Visual-Conditioned OPD launching (v3)
========================================
  USE_TAM_BOOST   = ${MLLMOPD_USE_TAM_BOOST}
  HOOK_MODE       = ${MLLMOPD_TAM_HOOK_MODE}
  ALPHA           = ${MLLMOPD_TAM_ALPHA}
  C_LOCAL         = ${MLLMOPD_TAM_C_LOCAL}
EOF
if [ "${MLLMOPD_TAM_HOOK_MODE}" = "cached_spatial" ]; then
cat <<EOF
  CACHE_JSONL     = ${MLLMOPD_TAM_CACHE_JSONL}
  K/ρ/τ           = ${MLLMOPD_TAM_K} / ${MLLMOPD_TAM_RHO} / ${MLLMOPD_TAM_TAU}
EOF
elif [ "${MLLMOPD_TAM_HOOK_MODE}" = "random_rate_matched" ]; then
cat <<EOF
  TARGET_RATE     = ${MLLMOPD_TAM_TARGET_RATE}
  SEED            = ${MLLMOPD_TAM_SEED}
EOF
fi
cat <<EOF
  POST_PROCESS_HOOK = ${MLLMOPD_POST_PROCESS_HOOK}
========================================
EOF

# Hand off to the chosen baseline launcher.
exec bash "${BASELINE_LAUNCHER}" "$@"
