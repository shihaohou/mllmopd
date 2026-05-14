#!/usr/bin/env bash
# Evaluate a checkpoint on the MLLM benchmark suite via lmms-eval.
# Mirrors Uni-OPD's Uni-OPD-LMMS-Eval workflow.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source .env

: "${LMMS_EVAL_PATH:?}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a checkpoint dir or HF id}"
RUN_TAG="${RUN_TAG:-$(basename "${MODEL_PATH}")_$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="${MLLMOPD_RUNS}/eval/${RUN_TAG}"
mkdir -p "${OUT_DIR}"

TASKS="${TASKS:-mathvista_testmini,mathvision,mathverse_testmini,logicvista,chartqa,hallusion_bench_image,charxiv,mmmu_val}"

# shellcheck disable=SC1091
source "${CONDA_PATH}/bin/activate" Uni-OPD-LMMS-Eval

cd "${LMMS_EVAL_PATH}"
accelerate launch --num_processes "${EVAL_NUM_PROCESSES:-8}" --main_process_port "${EVAL_PORT:-29500}" \
  -m lmms_eval \
  --model qwen2_5_vl \
  --model_args "pretrained=${MODEL_PATH},device_map=auto" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --output_path "${OUT_DIR}" \
  --log_samples

echo ">>> Eval done: ${OUT_DIR}"
