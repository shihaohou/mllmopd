#!/usr/bin/env bash
# Run this ONCE on the 8xH800 devbox after `git clone --recurse-submodules`.
# It checks the environment and makes scratch dirs. It does NOT build conda envs;
# that's setup_train_env.sh / setup_lmmseval_env.sh.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [ ! -f .env ]; then
  echo "ERROR: .env not found. Run:  cp .env.example .env  and edit values."
  exit 1
fi
# shellcheck disable=SC1091
source .env

echo ">>> Repo root:    ${MLLMOPD_ROOT}"
echo ">>> Base dir:     ${BASE_DIR}"
echo ">>> Conda path:   ${CONDA_PATH}"
echo ">>> HF cache:     ${HF_HOME}"
echo ">>> Runs dir:     ${MLLMOPD_RUNS}"

# --- Sanity checks -----------------------------------------------------------
echo ">>> Checking GPUs"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || {
  echo "ERROR: nvidia-smi failed. Is this actually a CUDA box?"
  exit 1
}

echo ">>> Checking submodules"
git submodule status

for sub in Uni-OPD Megatron-LM sglang lmms-eval; do
  if [ ! -d "${BASE_DIR}/${sub}" ] || [ -z "$(ls -A "${BASE_DIR}/${sub}")" ]; then
    echo "WARN: ${BASE_DIR}/${sub} is empty. Run:"
    echo "        git submodule update --init --recursive"
    exit 1
  fi
done

# --- Scratch dirs ------------------------------------------------------------
mkdir -p "${HF_HOME}" "${MLLMOPD_RUNS}" "${MLLMOPD_DATA}"
echo ">>> Scratch dirs ready under /scratch/${USER}"

# Symlink mllmopd/{runs,data,models} -> scratch so local tooling can `ls runs/`
ln -sfn "${MLLMOPD_RUNS}"  "${MLLMOPD_ROOT}/runs"
ln -sfn "${MLLMOPD_DATA}"  "${MLLMOPD_ROOT}/data"
ln -sfn "${HF_HOME}"        "${MLLMOPD_ROOT}/models"

echo ">>> Symlinked: runs/  data/  models/  ->  scratch"

cat <<EOF

>>> Bootstrap complete. Next:

    bash scripts/env/setup_train_env.sh        # builds Uni-OPD conda env
    bash scripts/env/setup_lmmseval_env.sh     # builds Uni-OPD-LMMS-Eval conda env

EOF
