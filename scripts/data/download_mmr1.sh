#!/usr/bin/env bash
# Download MMR1 RL training data + paired SFT/RL checkpoints (3B + 7B) into $HF_HOME.
# Uses huggingface-cli; assumes $HF_TOKEN is set if anything is gated.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source .env

: "${HF_HOME:?}"
mkdir -p "${HF_HOME}"

DATASETS=(
  "MMR1/MMR1-RL"
  "MMR1/MMR1-SFT"
)
MODELS=(
  "MMR1/MMR1-3B-SFT"
  "MMR1/MMR1-3B-RL"
  "MMR1/MMR1-7B-SFT"
  "MMR1/MMR1-7B-RL"
)

echo ">>> Pulling datasets to ${HF_HOME}"
for d in "${DATASETS[@]}"; do
  huggingface-cli download "${d}" --repo-type dataset --resume-download
done

echo ">>> Pulling models to ${HF_HOME}"
for m in "${MODELS[@]}"; do
  huggingface-cli download "${m}" --resume-download
done

echo ">>> MMR1 download complete."
echo "    Note: real dataset IDs may differ; if huggingface-cli 404s, open the MMR1 GitHub"
echo "    README and update DATASETS / MODELS above with the actual repo names."
