#!/usr/bin/env bash
# Apply Uni-OPD's sglang and Megatron patches. Idempotent (skips if already applied).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source .env

: "${MILES_DIR:?}"
: "${SGLANG_DIR:?}"
: "${MEGATRON_DIR:?}"

PATCH_DIR="${MILES_DIR}/docker/patch/v0.5.7"

apply_if_clean() {
  local target_dir="$1" patch_file="$2"
  cd "${target_dir}"
  if git apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
    echo ">>> Already applied: ${patch_file}"
  else
    echo ">>> Applying:        ${patch_file}"
    git apply "${patch_file}"
  fi
  cd - >/dev/null
}

apply_if_clean "${SGLANG_DIR}"   "${PATCH_DIR}/sglang_psp.patch"
apply_if_clean "${MEGATRON_DIR}" "${PATCH_DIR}/megatron.patch"

echo ">>> Patches done."
