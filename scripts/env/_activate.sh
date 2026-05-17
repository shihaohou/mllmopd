# Source this file (don't exec). Picks the right Python env in this order:
#   1. ${MLLMOPD_VENV}/bin/activate         — explicit override (uv or conda venv path)
#   2. ${REPO}/.venv/bin/activate           — symlink created by setup_uv_env.sh
#   3. ${CONDA_PATH}/bin/activate ${CONDA_ENV:-Uni-OPD-LMMS-Eval}  — legacy conda path
#
# Returns 1 if none of the three exist. With `set -e` in the caller this bails
# the caller out, which is what we want.

_mllmopd_activate() {
  local root="${MLLMOPD_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null)}"

  if [ -n "${MLLMOPD_VENV:-}" ] && [ -f "${MLLMOPD_VENV}/bin/activate" ]; then
    echo ">>> activating MLLMOPD_VENV=${MLLMOPD_VENV}"
    # shellcheck disable=SC1091
    source "${MLLMOPD_VENV}/bin/activate"
    return 0
  fi

  if [ -n "${root}" ] && [ -f "${root}/.venv/bin/activate" ]; then
    echo ">>> activating ${root}/.venv"
    # shellcheck disable=SC1091
    source "${root}/.venv/bin/activate"
    return 0
  fi

  if [ -n "${CONDA_PATH:-}" ] && [ -d "${CONDA_PATH}" ]; then
    local env_name="${CONDA_ENV:-Uni-OPD-LMMS-Eval}"
    echo ">>> activating conda env ${env_name}"
    # shellcheck disable=SC1091
    source "${CONDA_PATH}/bin/activate" "${env_name}"
    return 0
  fi

  echo "ERROR: no Python env found. Either:" >&2
  echo "  - run  bash scripts/env/setup_uv_env.sh      (UV path, ~10 min)" >&2
  echo "  - run  bash scripts/env/setup_lmmseval_env.sh (Conda path, ~30 min)" >&2
  echo "  - or   export MLLMOPD_VENV=/path/to/venv     (use a pre-built one)" >&2
  return 1
}

_mllmopd_activate
