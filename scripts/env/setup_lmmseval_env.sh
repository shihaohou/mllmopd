#!/usr/bin/env bash
# Build the Uni-OPD-LMMS-Eval conda env on the H800 devbox.
# Mirrors third_party/Uni-OPD/docs/build_eval_env.md (MLLM section).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ ! -f .env ]; then
  echo "ERROR: .env not found."
  exit 1
fi
# shellcheck disable=SC1091
source .env

: "${CONDA_PATH:?}"
: "${LMMS_EVAL_PATH:?}"

if [ ! -d "${LMMS_EVAL_PATH}" ]; then
  echo "ERROR: ${LMMS_EVAL_PATH} not found. Run init first to add lmms-eval submodule."
  exit 1
fi

# shellcheck disable=SC1091
source "${CONDA_PATH}/bin/activate"

if conda env list | awk '{print $1}' | grep -qx "Uni-OPD-LMMS-Eval"; then
  echo ">>> Conda env Uni-OPD-LMMS-Eval already exists. Activating without rebuild."
  conda activate Uni-OPD-LMMS-Eval
else
  conda create -n Uni-OPD-LMMS-Eval python=3.12.13 -y
  conda activate Uni-OPD-LMMS-Eval
fi

pip install "vllm>=0.13.0" qwen-vl-utils decord math-verify

cd "${LMMS_EVAL_PATH}"
pip install -e ".[all]"

pip install latex2sympy2_extended
pip uninstall latex2sympy2 -y || true

pip install jieba distance apted Polygon3 nltk
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

echo ">>> Uni-OPD-LMMS-Eval env ready."
