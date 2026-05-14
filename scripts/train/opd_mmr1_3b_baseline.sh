#!/usr/bin/env bash
# Vanilla OPD baseline: MMR1-7B-RL teacher -> MMR1-3B-SFT student on MMR1-RL 15K (subset).
# Wraps Uni-OPD's training launcher. Assumes teacher server is already up at $TEACHER_URL.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source .env

: "${MILES_DIR:?}"
: "${MLLMOPD_RUNS:?}"

RUN_NAME="${RUN_NAME:-vanilla_opd_mmr1_3b_$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${MLLMOPD_RUNS}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

CONFIG="${CONFIG:-configs/baseline/mmr1_3b_vanilla_opd.yaml}"

# Activate training env
# shellcheck disable=SC1091
source "${CONDA_PATH}/bin/activate" Uni-OPD

# Trainer GPUs (anything not used by teacher server)
TRAINER_GPUS="${TRAINER_GPUS:-2,3,4,5,6,7}"

echo ">>> RUN_DIR=${RUN_DIR}"
echo ">>> CONFIG=${CONFIG}"
echo ">>> TRAINER_GPUS=${TRAINER_GPUS}"

CUDA_VISIBLE_DEVICES="${TRAINER_GPUS}" \
  python "${MILES_DIR}/Uni_OPD_utils/ray_launcher.py" \
    --config "${CONFIG}" \
    --output-dir "${RUN_DIR}" \
    --run-name "${RUN_NAME}" \
    "$@"

echo ">>> Training done. Outputs: ${RUN_DIR}"
