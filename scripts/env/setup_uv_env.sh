#!/usr/bin/env bash
# Build a UV-based venv for mllmopd smoke audit on machines where the H800
# Uni-OPD recipe doesn't apply (e.g. A800 / different cluster). The venv lives
# on fast local disk; ${REPO}/.venv is a symlink into it.
#
# Installs ONLY what the audit pipeline needs (torch + transformers + datasets
# + PIL + our local package). Training env (Megatron / TE / SGLang) is a
# separate concern — see scripts/env/setup_train_env.sh.
#
# Variables (override via env):
#   FAST_DISK   default /root/shihao_project    — must be on fast local disk
#   ENV_HOME    default ${FAST_DISK}/mllmopd-env
#   PYTHON_VER  default 3.12
#   TORCH_CUDA  default cu124                   — picks the torch wheel index
#
# Re-running is idempotent. Pass --rebuild to wipe ${ENV_HOME}/.venv first.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
REPO="$(pwd)"

FAST_DISK="${FAST_DISK:-/root/shihao_project}"
ENV_HOME="${ENV_HOME:-${FAST_DISK}/mllmopd-env}"
PYTHON_VER="${PYTHON_VER:-3.12}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"
REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

echo ">>> Repo:      ${REPO}"
echo ">>> Fast disk: ${FAST_DISK}"
echo ">>> Env home:  ${ENV_HOME}"
echo ">>> Python:    ${PYTHON_VER}"
echo ">>> Torch idx: https://download.pytorch.org/whl/${TORCH_CUDA}"

# 1. uv must be present
if ! command -v uv >/dev/null; then
  echo ">>> uv not found. Installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [ -f "${HOME}/.local/bin/env" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/.local/bin/env"
  else
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
fi
echo ">>> uv: $(uv --version)"

# 2. Direct uv-managed Python to fast disk so the interpreter doesn't sit on /
mkdir -p "${ENV_HOME}"
export UV_PYTHON_INSTALL_DIR="${ENV_HOME}/.uv-python"
uv python install "${PYTHON_VER}"

# 3. Optional wipe
if [ "${REBUILD}" = "1" ] && [ -e "${ENV_HOME}/.venv" ]; then
  echo ">>> --rebuild: removing ${ENV_HOME}/.venv"
  rm -rf "${ENV_HOME}/.venv"
fi

# 4. Create venv on fast disk
if [ ! -d "${ENV_HOME}/.venv" ]; then
  echo ">>> Creating venv at ${ENV_HOME}/.venv"
  ( cd "${ENV_HOME}" && uv venv --python "${PYTHON_VER}" .venv )
else
  echo ">>> Reusing existing venv at ${ENV_HOME}/.venv"
fi

# Verify python exec works (catches dangling symlinks early, cf. migrate-env.md Q6)
"${ENV_HOME}/.venv/bin/python" --version

# 5. Symlink ${REPO}/.venv -> fast disk
ln -sfn "${ENV_HOME}/.venv" "${REPO}/.venv"
echo ">>> Symlinked ${REPO}/.venv -> ${ENV_HOME}/.venv"

# 6. Activate + install
# shellcheck disable=SC1091
source "${REPO}/.venv/bin/activate"

# Defang NGC pip pins if present (no-op on non-NGC images)
unset PIP_CONSTRAINT 2>/dev/null || true
export PIP_CONFIG_FILE=/dev/null

echo ">>> Installing torch (${TORCH_CUDA})..."
uv pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    "torch==2.5.1" "torchvision==0.20.1"

echo ">>> Installing transformers + audit deps..."
uv pip install \
    "transformers>=4.49,<4.55" \
    "accelerate>=0.34" \
    "datasets>=2.18" \
    "huggingface-hub>=0.24,<1.0" \
    "pillow>=10.0" \
    "numpy>=1.24,<2" \
    "pandas>=2.1" \
    "pyyaml>=6.0" \
    "tqdm>=4.66" \
    "qwen-vl-utils" \
    "hf-transfer"

echo ">>> Installing flash-attn (sm_80; --no-build-isolation needs `wheel` pre-installed)..."
# --no-build-isolation reuses our venv torch (avoids the NGC-torch ABI leak
# described in docs/migrate-env.md Q2), but it also skips auto-install of
# wheel / ninja / packaging — we have to provide them ourselves.
uv pip install wheel ninja packaging
uv pip install --no-build-isolation "flash-attn==2.7.4.post1" \
  || echo ">>> flash-attn unavailable; runtime falls back to sdpa attention"

echo ">>> Installing local mllmopd package (-e, no deps)..."
uv pip install --no-deps -e "${REPO}"

# 7. Sanity check
echo
echo ">>> Sanity check:"
python - <<'PY'
import torch, transformers
from mllmopd.diagnostics import scorers, run_audit_pass  # noqa: F401
print(f"  torch:        {torch.__version__}  cuda={torch.cuda.is_available()}  ngpu={torch.cuda.device_count()}")
print(f"  transformers: {transformers.__version__}")
try:
    from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
    print("  Qwen2.5-VL:   present")
except ImportError as e:
    print(f"  Qwen2.5-VL:   FAIL ({e})")
try:
    import flash_attn  # noqa: F401
    print(f"  flash_attn:   {flash_attn.__version__}")
except ImportError:
    print("  flash_attn:   absent (sdpa fallback)")
print("  mllmopd ok")
PY

cat <<EOF

>>> Env ready at ${ENV_HOME}/.venv
>>> Activate with:    source ${REPO}/.venv/bin/activate
>>> Or set in .env:   export MLLMOPD_VENV="${ENV_HOME}/.venv"
>>> Then run:         AUDIT_LIMIT=2 AUDIT_DEBUG=1 RUN_ID=debug2 bash scripts/audit/run_smoke.sh

EOF
