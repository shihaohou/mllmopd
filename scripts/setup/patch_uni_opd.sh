#!/usr/bin/env bash
# Idempotent local-only patches to the Uni-OPD submodule, fixing stale
# imports that block T1 training on a fresh checkout.
#
# Uni-OPD's submodule contains a few hard-coded imports that reference
# layout assumptions (Tencent's internal repo + project-specific
# external packages) which don't hold in our clone. The two
# architectural workarounds (src/exps/ shim + src/miles/ wrapper) cover
# most of them, but:
#
#   - margin_shift.py uses `from miles.miles.X import ...` (double
#     prefix). This needs the src/miles/__init__.py wrapper to win the
#     PYTHONPATH race. In Ray-worker subprocesses that race appears to
#     be lost intermittently — workers re-import `miles` from MILES_DIR
#     first and the wrapper never loads. A 2-line in-place patch is
#     both more deterministic and easier to debug.
#
# This script ONLY touches files inside `third_party/Uni-OPD/miles/`.
# Submodules are independent git repos: the patch stays local and is
# not pushed upstream. Re-running after `git submodule update` requires
# re-running this script.
#
# Run once per fresh checkout. Idempotent.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MILES_DIR="${MILES_DIR:-third_party/Uni-OPD/miles}"
if [ ! -d "${MILES_DIR}" ]; then
  echo "ERROR: MILES_DIR=${MILES_DIR} not found (init submodules first)" >&2
  exit 1
fi

# --- Patch 1: margin_shift.py double-prefix imports -----------------
# The two lines that say `from miles.miles.X import ...` only work when
# PYTHONPATH points at Uni-OPD repo root. Strip one `miles.` to make
# them resolve against our PYTHONPATH (which points at miles/).
MS="${MILES_DIR}/Uni_OPD_utils/margin_calibration/margin_shift.py"
if [ ! -f "${MS}" ]; then
  echo "ERROR: ${MS} not found" >&2
  exit 1
fi

if grep -q "from miles\.miles\." "${MS}"; then
  echo ">>> patching ${MS}: miles.miles.X -> miles.X"
  # Use awk for portability (sed -i syntax differs between mac BSD and GNU)
  awk '{ gsub(/from miles\.miles\./, "from miles."); print }' "${MS}" > "${MS}.tmp"
  mv "${MS}.tmp" "${MS}"
  echo "    after:"
  grep -n "^from miles\." "${MS}" | head -5 | sed 's/^/      /'
else
  echo ">>> ${MS}: already patched (no miles.miles.* found)"
fi

echo
echo ">>> patch_uni_opd done. Re-run after `git submodule update`."
