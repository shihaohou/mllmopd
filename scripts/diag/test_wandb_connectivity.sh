#!/usr/bin/env bash
# Test wandb online connectivity from the trainer box without breaking
# internal (sglang/Ray) proxy bypass. The trainer pipeline uses both:
#   - internal HTTP (Box 1 sglang at 10.x) — must bypass proxy
#   - external HTTPS (api.wandb.ai) — must go via proxy
#
# Default Box 2 setup after a fresh shell:
#   - http_proxy / https_proxy set (corp env)
#   - no_proxy may not include the 10.x intranet
# → curl to Box 1 fails (routes through corp proxy)
# → curl to wandb works (proxy is up)
#
# What launcher does (per opd_mmr1_3b_baseline_xbox.sh:192-205):
#   MLLMOPD_KEEP_PROXY=0 (default): unset http_proxy/https_proxy
#     → internal HTTP works (no proxy in the way)
#     → wandb cannot reach api.wandb.ai → offline only
#   MLLMOPD_KEEP_PROXY=1: keep proxy + set no_proxy=10.0.0.0/8,...
#     → internal HTTP bypasses proxy via no_proxy
#     → wandb can reach api.wandb.ai via proxy
#
# This script tests BOTH modes and prints a clear verdict.
#
# Usage:
#   bash scripts/diag/test_wandb_connectivity.sh
#
# Exit code: 0 if at least one mode works for both internal+wandb, else 1.

set -u

WANDB_URL_PRIMARY="${WANDB_BASE_URL:-https://api.wandb.ai}"
INTERNAL_HOST="${INTERNAL_TEST_HOST:-10.82.121.12}"
INTERNAL_PORT="${INTERNAL_TEST_PORT:-30001}"
WANDB_PROBE_PATH="/graphql"

# Persist original proxy for "with proxy" mode
ORIG_HTTP_PROXY="${http_proxy:-}"
ORIG_HTTPS_PROXY="${https_proxy:-}"
ORIG_NO_PROXY_LOWER="${no_proxy:-}"
ORIG_NO_PROXY_UPPER="${NO_PROXY:-}"

INTRANET_NO_PROXY="localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"

echo "================================================================"
echo "wandb online connectivity probe — both proxy modes"
echo "================================================================"
echo "  wandb endpoint   = ${WANDB_URL_PRIMARY}${WANDB_PROBE_PATH}"
echo "  internal probe   = http://${INTERNAL_HOST}:${INTERNAL_PORT}/health"
echo "  shell http_proxy = ${ORIG_HTTP_PROXY:-(unset)}"
echo "  shell no_proxy   = ${ORIG_NO_PROXY_LOWER:-(unset)}"
echo "----------------------------------------------------------------"
echo ""

# ---------- helper ----------
probe() {
    local label="$1"
    local url="$2"
    local timeout="${3:-8}"
    local body_size
    body_size=$(curl -sSL --max-time "${timeout}" -o /tmp/_probe.out \
                    -w "http=%{http_code} time=%{time_total}s size=%{size_download}" \
                    "${url}" 2>&1)
    local rc=$?
    if [ "${rc}" -ne 0 ]; then
        echo "  ${label}: curl rc=${rc} (${body_size})"
        return 1
    fi
    echo "  ${label}: ${body_size}"
    return 0
}

# ============================================================
# MODE A: no proxy at all (current default after `unset`)
# ============================================================
echo ">>> MODE A: no proxy (sglang-friendly, wandb may fail)"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
mode_a_internal=0
mode_a_wandb=0
probe "  internal sglang" "http://${INTERNAL_HOST}:${INTERNAL_PORT}/health" 5 && mode_a_internal=1
probe "  wandb GraphQL  " "${WANDB_URL_PRIMARY}${WANDB_PROBE_PATH}" 8 && mode_a_wandb=1
echo ""

# ============================================================
# MODE B: keep proxy + intranet bypass (MLLMOPD_KEEP_PROXY=1)
# ============================================================
echo ">>> MODE B: keep proxy + no_proxy bypass for 10.x (launcher's KEEP_PROXY=1 path)"
if [ -n "${ORIG_HTTP_PROXY}" ]; then
    export http_proxy="${ORIG_HTTP_PROXY}"
    export https_proxy="${ORIG_HTTPS_PROXY}"
    export HTTP_PROXY="${ORIG_HTTP_PROXY}"
    export HTTPS_PROXY="${ORIG_HTTPS_PROXY}"
    export no_proxy="${INTRANET_NO_PROXY}"
    export NO_PROXY="${INTRANET_NO_PROXY}"
    mode_b_internal=0
    mode_b_wandb=0
    probe "  internal sglang" "http://${INTERNAL_HOST}:${INTERNAL_PORT}/health" 5 && mode_b_internal=1
    probe "  wandb GraphQL  " "${WANDB_URL_PRIMARY}${WANDB_PROBE_PATH}" 8 && mode_b_wandb=1
else
    echo "  (skipped: no original http_proxy in shell — cannot test KEEP_PROXY=1)"
    mode_b_internal=0
    mode_b_wandb=0
fi
echo ""

# ============================================================
# MODE C: wandb python client probe (uses HTTPS_PROXY from MODE B env)
# ============================================================
echo ">>> MODE C: wandb python client (requires wandb installed + WANDB_API_KEY)"
if python3 -c "import wandb" 2>/dev/null; then
    WANDB_API_KEY_PRESENT="${WANDB_API_KEY:+(set, ${#WANDB_API_KEY} chars)}"
    echo "  wandb installed: yes; WANDB_API_KEY=${WANDB_API_KEY_PRESENT:-(unset)}"
    if [ -n "${WANDB_API_KEY:-}" ]; then
        python3 - <<'PY' 2>&1 | sed 's/^/  /'
import os, wandb, time, sys
t0 = time.time()
try:
    api = wandb.Api(timeout=8)
    user = api.viewer
    print(f"wandb api.viewer.username = {user.username}  (elapsed {time.time()-t0:.1f}s)")
    sys.exit(0)
except Exception as e:
    print(f"wandb api probe FAILED: {type(e).__name__}: {str(e)[:200]}")
    sys.exit(1)
PY
        mode_c=$?
    else
        echo "  (skipped: WANDB_API_KEY unset; run `wandb login` or export it)"
        mode_c=2
    fi
else
    echo "  (skipped: python wandb package not importable)"
    mode_c=2
fi
echo ""

# ============================================================
# Verdict
# ============================================================
echo "================================================================"
echo "  VERDICT"
echo "================================================================"
printf "  MODE A (proxy stripped):       internal=%s  wandb=%s\n" \
    "$([ $mode_a_internal = 1 ] && echo OK || echo FAIL)" \
    "$([ $mode_a_wandb = 1 ] && echo OK || echo FAIL)"
printf "  MODE B (proxy kept + bypass):  internal=%s  wandb=%s\n" \
    "$([ $mode_b_internal = 1 ] && echo OK || echo FAIL)" \
    "$([ $mode_b_wandb = 1 ] && echo OK || echo FAIL)"
printf "  MODE C (wandb python client):  %s\n" \
    "$([ ${mode_c:-2} = 0 ] && echo OK || ([ ${mode_c:-2} = 2 ] && echo SKIPPED || echo FAIL))"
echo ""

if [ $mode_b_internal = 1 ] && [ $mode_b_wandb = 1 ]; then
    echo "  → RECOMMENDED: run trainer with MLLMOPD_KEEP_PROXY=1"
    echo "    (proxy kept for wandb, no_proxy bypasses 10.x for sglang/Ray)"
    echo ""
    echo "  Add to your launch env:"
    echo "    export MLLMOPD_KEEP_PROXY=1"
    echo "    export http_proxy=${ORIG_HTTP_PROXY}"
    echo "    export https_proxy=${ORIG_HTTPS_PROXY}"
    echo "    export no_proxy=${INTRANET_NO_PROXY}"
    echo "    export NO_PROXY=${INTRANET_NO_PROXY}"
    echo "    [ -n \"\$WANDB_API_KEY\" ] || echo '!! also set WANDB_API_KEY'"
    exit 0
elif [ $mode_a_internal = 1 ] && [ $mode_a_wandb = 0 ]; then
    echo "  → wandb cannot reach api.wandb.ai via direct (no proxy);"
    echo "    corp proxy must be in path. KEEP_PROXY=1 mode failed —"
    echo "    likely no_proxy isn't being honored by curl/wandb in this env."
    echo "    Stick with offline wandb (default), sync afterwards:"
    echo "      wandb sync wandb/offline-run-*"
    exit 1
elif [ $mode_a_internal = 0 ]; then
    echo "  → internal sglang unreachable even without proxy."
    echo "    Box 1 sglang servers may be down; check first:"
    echo "      curl http://${INTERNAL_HOST}:${INTERNAL_PORT}/health"
    exit 1
else
    echo "  → mixed result; review modes above. Most likely:"
    echo "      1. fix WANDB_API_KEY if MODE C said unset"
    echo "      2. re-test"
    exit 1
fi
