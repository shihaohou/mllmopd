#!/usr/bin/env bash
# Download OpenMMReasoner RL data + ChartQA + InfographicVQA.
# Uni-OPD's MLLM training already references these.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source .env

: "${HF_HOME:?}"
mkdir -p "${HF_HOME}"

DATASETS=(
  "OpenMMReasoner/OpenMMReasoner-RL-74K"
  "OpenMMReasoner/OpenMMReasoner-SFT-874K"
  "HuggingFaceM4/ChartQA"
  "vidore/infovqa_test_subsampled"  # placeholder; replace with the InfographicVQA repo you want
)

for d in "${DATASETS[@]}"; do
  echo ">>> ${d}"
  huggingface-cli download "${d}" --repo-type dataset --resume-download || \
    echo "    (skip if 404; update dataset id and retry)"
done

echo ">>> OpenMMReasoner / ChartQA / InfoVQA pulled."
