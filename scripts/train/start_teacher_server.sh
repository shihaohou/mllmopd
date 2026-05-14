#!/usr/bin/env bash
# Launch a teacher SGLang server. Wraps Uni-OPD's run_sglang_server.sh and writes
# the resulting endpoint into teacher_server_list.json.
#
# Override via env vars:
#   TEACHER_MODEL_PATH  (default: HF id MMR1/MMR1-7B-RL — change to a local path on devbox)
#   TEACHER_PORT        (default: 30000)
#   TEACHER_GPUS        (default: 0,1)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source .env

: "${MILES_DIR:?}"

TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${HF_HOME}/hub/models--MMR1--MMR1-7B-RL/snapshots/main}"
TEACHER_PORT="${TEACHER_PORT:-30000}"
TEACHER_GPUS="${TEACHER_GPUS:-0,1}"

REGISTRY="${MILES_DIR}/Uni_OPD_utils/OPD_reward/teacher_server_list.json"
SERVER_SH="${MILES_DIR}/Uni_OPD_utils/scripts/server/run_sglang_server.sh"

if [ ! -f "${SERVER_SH}" ]; then
  echo "ERROR: ${SERVER_SH} not found. Did you init submodules?"
  exit 1
fi

# Register endpoint
python - <<PY
import json, pathlib
p = pathlib.Path("${REGISTRY}")
p.parent.mkdir(parents=True, exist_ok=True)
cfg = {"teachers": [{"name": "MMR1-7B-RL", "url": "http://localhost:${TEACHER_PORT}/v1"}]}
p.write_text(json.dumps(cfg, indent=2))
print(f"Wrote {p}")
PY

# Activate training env (it has sglang)
# shellcheck disable=SC1091
source "${CONDA_PATH}/bin/activate" Uni-OPD

CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}" \
  bash "${SERVER_SH}" \
    --model-path "${TEACHER_MODEL_PATH}" \
    --port "${TEACHER_PORT}"
