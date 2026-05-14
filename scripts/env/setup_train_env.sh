#!/usr/bin/env bash
# Build the Uni-OPD training conda env on the H800 devbox.
# Mirrors third_party/Uni-OPD/docs/build_env.md verbatim, parameterized via .env.
# Run on the devbox; will fail on Mac because flash-attn / apex need CUDA.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [ ! -f .env ]; then
  echo "ERROR: .env not found. cp .env.example .env first."
  exit 1
fi
# shellcheck disable=SC1091
source .env

: "${BASE_DIR:?BASE_DIR must be set in .env}"
: "${CONDA_PATH:?CONDA_PATH must be set in .env}"
: "${MILES_DIR:?MILES_DIR must be set in .env}"
: "${MEGATRON_DIR:?MEGATRON_DIR must be set in .env}"
: "${SGLANG_DIR:?SGLANG_DIR must be set in .env}"

# shellcheck disable=SC1091
source "${CONDA_PATH}/bin/activate"

if conda env list | awk '{print $1}' | grep -qx "Uni-OPD"; then
  echo ">>> Conda env Uni-OPD already exists. Activating without rebuild."
  conda activate Uni-OPD
else
  conda create -n Uni-OPD python=3.12.12 -y
  conda activate Uni-OPD
fi

pip config set global.root-user-action ignore

pip install cuda-python==13.1.0
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

cd "${SGLANG_DIR}"
pip install -e "python[all]"
pip install cmake ninja math-verify

TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=64 \
    pip -v install "flash-attn==2.7.4.post1" --no-build-isolation

pip install "git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c" --no-deps

pip -v install --no-build-isolation "transformer_engine[pytorch]==2.10.0"

pip install flash-linear-attention==0.4.0

NVCC_APPEND_FLAGS="--threads 4" \
    pip -v install --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
    "git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"

pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@d64a639 \
    --no-cache-dir --force-reinstall

pip install git+https://github.com/NVIDIA/nvidia-resiliency-ext --no-build-isolation
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation
pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

cd "${MEGATRON_DIR}"
pip install -e .

cd "${MILES_DIR}"
pip install -e .

pip install "nvidia-cudnn-cu12==9.16.0.29"
pip install "numpy<2"

# Apply patches
bash "${MLLMOPD_ROOT}/scripts/env/apply_patches.sh"

# Verify
python - <<'PY'
import torch, flash_attn, transformer_engine, sglang, megatron, miles
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "n_gpu", torch.cuda.device_count())
print("flash_attn / TE / sglang / megatron / miles OK")
PY

echo ">>> Uni-OPD training env ready."
