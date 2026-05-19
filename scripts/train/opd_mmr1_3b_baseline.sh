#!/usr/bin/env bash
# T1 OPD baseline launcher: MMR1-7B-RL teacher → MMR1-3B-SFT student.
# FullTeacher (T1-2) vs BlankTeacher (T1-3) negative control. The only
# difference between the two arms is the OPD_TEACHER_IMAGE_MODE env var.
#
# Pre-flight (run once per arm, outside this script):
#   1. source .env, activate train venv (sglang + miles + Megatron)
#   2. bash scripts/train/start_teacher_server.sh
#      (binds teacher to ${TEACHER_GPUS}, writes teacher_server_{list,map}.json)
#   3. ./scripts/audit/rerun_h2_sysprompt.sh once-only G1 check (already passed)
#
# Required env vars (sourced from .env):
#   MMR1_3B_SFT_CKPT     — student HF checkpoint
#   MMR1_7B_RL_CKPT      — teacher HF checkpoint (referenced for sanity logging)
#   MLLMOPD_RUNS         — base output dir
#   MILES_DIR            — path to Uni-OPD's miles/ directory (third_party submodule)
#   MEGATRON_PATH        — path to Megatron-LM (third_party submodule)
#
# T1 arm selector (the only knob that differs between T1-2 and T1-3):
#   OPD_TEACHER_IMAGE_MODE  full | blank
#
# Other tunables (env overrides; defaults match T1 plan §6 budget):
#   OPD_RUN_NAME            output subdir name (default: t1_v0_${ARM_TAG})
#   TRAIN_JSONL             prep output (default: data/opd_train/v0_2k/train.jsonl)
#   STUDENT_CKPT            (default: $MMR1_3B_SFT_CKPT)
#   ROLLOUT_BATCH_SIZE      prompts per rollout (default: 8 → 250 steps for 2k)
#   SAMPLE_N                rollouts per prompt (default: 8)
#   GLOBAL_BATCH_SIZE       (default: rbs * sample_n = 64)
#   NUM_EPOCH               (default: 1; locked to 1 in T1-v0 plan §3)
#   LR / LR_WARMUP          (default: 1e-6 / 5)
#   EPS_CLIP / EPS_CLIP_HIGH (default: 0.2 / 0.28)
#   OPD_CLIP_RANGE          teacher-student logp clip (default: 10.0)
#   ROLLOUT_MAX_PROMPT_LEN  (default: 4096)
#   ROLLOUT_MAX_RESPONSE_LEN (default: 2048; Risk #3 mitigation)
#   ACTOR_NUM_GPUS_PER_NODE (default: 7; teacher already on GPU 0)
#   ROLLOUT_NUM_GPUS        (default: 7)
#   TP_SIZE                 (default: 1; 3B fits on one GPU)
#   TRAINER_GPUS            CUDA_VISIBLE_DEVICES for trainer (default: 1,2,3,4,5,6,7)
#   STUDENT_MODEL_ARGS      Megatron model-arg shell file
#                           (default: ${MILES_DIR}/scripts/models/qwen2.5-3B.sh)
#   DEBUG_MODE              1 = punch list #9 smoke profile: dumps per-rollout
#                           debug data, disables sglang CUDA graph, NCCL_DEBUG=INFO.
#                           Set this when running the 10-step smoke. (default: 0)
#   SAVE_INTERVAL           checkpoint frequency in optimizer steps
#                           (default: 50; ~5 ckpts over 250 steps)
#
# Usage (run from repo root):
#   # 10-step smoke (T1-2 arm)
#   DEBUG_MODE=1 ROLLOUT_BATCH_SIZE=1 NUM_EPOCH=1 \
#     OPD_TEACHER_IMAGE_MODE=full \
#     OPD_RUN_NAME=t1_smoke_full \
#     bash scripts/train/opd_mmr1_3b_baseline.sh
#
#   # Full T1-2 run (FullTeacher arm)
#   OPD_TEACHER_IMAGE_MODE=full  bash scripts/train/opd_mmr1_3b_baseline.sh
#
#   # Full T1-3 run (BlankTeacher arm)
#   OPD_TEACHER_IMAGE_MODE=blank bash scripts/train/opd_mmr1_3b_baseline.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source .env

# NGC base image provides CUDA forward-compat at /usr/local/cuda-12.9/compat/lib.real
# (with /usr/local/cuda/compat/lib as a symlink to it). Driver 535.129 cannot
# run cu128 runtime without this compat layer. Independent diagnostic
# (smoke #11) confirmed:
#   - torch 2.9.1+cu128 ships its own bundled NCCL 2.27.5 + cuDNN at
#     .venv/lib/python3.12/site-packages/nvidia/*/lib/. Those libs win
#     dlopen via libtorch_cuda.so's DT_RPATH, which is HIGHER priority
#     than LD_LIBRARY_PATH or ld.so.cache. No LD prepending needed.
#   - In a normal login shell the compat-lib is auto-injected by the
#     NGC image init (somewhere under /etc/profile.d or similar). But
#     `ray start --head` daemonizes the raylet at the moment of invocation,
#     and Ray actor processes inherit env from THAT moment. If LD_LIBRARY_PATH
#     lacks compat-lib when ray start fires, every actor on this raylet
#     gets a CUDA context where libcudart's symbol resolution fails on
#     driver 535. NCCL collective ops then bail with the misleading
#     "CUDA driver version is insufficient" error.
#
# Minimal correct fix: ensure compat lib.real (plus its symlink for
# belt-and-suspenders) is in LD_LIBRARY_PATH BEFORE `ray start`. Don't
# touch anything else — torch's DT_RPATH handles venv NCCL/cuDNN on its
# own.
COMPAT_LIB_REAL=""
for d in /usr/local/cuda-12.9/compat/lib.real /usr/local/cuda-12.8/compat/lib.real /usr/local/cuda/compat/lib.real; do
  [ -d "${d}" ] && COMPAT_LIB_REAL="${d}" && break
done
COMPAT_LIB_SYMLINK=""
[ -d /usr/local/cuda/compat/lib ] && COMPAT_LIB_SYMLINK="/usr/local/cuda/compat/lib"

LD_BEFORE="${LD_LIBRARY_PATH:-(unset)}"
COMPAT_PREFIX=""
[ -n "${COMPAT_LIB_REAL}" ] && COMPAT_PREFIX="${COMPAT_LIB_REAL}"
[ -n "${COMPAT_LIB_SYMLINK}" ] && COMPAT_PREFIX="${COMPAT_PREFIX:+${COMPAT_PREFIX}:}${COMPAT_LIB_SYMLINK}"
if [ -n "${COMPAT_PREFIX}" ]; then
  LD_LIBRARY_PATH="${COMPAT_PREFIX}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export LD_LIBRARY_PATH
fi
echo ">>> CUDA forward-compat prepended (driver 535 ↔ cu128 runtime):"
echo "    compat real    : ${COMPAT_LIB_REAL:-(not found)}"
echo "    compat symlink : ${COMPAT_LIB_SYMLINK:-(not found)}"
echo "    LD_LIBRARY_PATH before : ${LD_BEFORE}"
echo "    LD_LIBRARY_PATH after  : ${LD_LIBRARY_PATH:-(empty)}"

# --- Required env (.env normally provides) ---
: "${MMR1_3B_SFT_CKPT:?}"
: "${MMR1_7B_RL_CKPT:?}"
: "${MLLMOPD_RUNS:?}"

# In-repo submodule paths as fallbacks. The repo ships
# third_party/Uni-OPD and third_party/Megatron-LM as git submodules; if the
# operator's .env doesn't override, those are the right paths. We canonicalize
# to absolute paths immediately — Ray workers inherit PYTHONPATH from this
# shell and can run in any cwd, so relative paths there are unreachable.
MILES_DIR="${MILES_DIR:-third_party/Uni-OPD/miles}"
MEGATRON_PATH="${MEGATRON_PATH:-third_party/Megatron-LM}"
if [ ! -d "${MILES_DIR}" ]; then
  echo "ERROR: MILES_DIR=${MILES_DIR} not found (did you init submodules?)" >&2
  exit 1
fi
if [ ! -d "${MEGATRON_PATH}" ]; then
  echo "ERROR: MEGATRON_PATH=${MEGATRON_PATH} not found (did you init submodules?)" >&2
  exit 1
fi
MILES_DIR="$(cd "${MILES_DIR}" && pwd)"
MEGATRON_PATH="$(cd "${MEGATRON_PATH}" && pwd)"
REPO_ROOT="$(pwd)"

# --- Arm selector ---
OPD_TEACHER_IMAGE_MODE="${OPD_TEACHER_IMAGE_MODE:-full}"
case "${OPD_TEACHER_IMAGE_MODE}" in
  full)  ARM_TAG="T1_2_full"  ;;
  blank) ARM_TAG="T1_3_blank" ;;
  *) echo "ERROR: OPD_TEACHER_IMAGE_MODE must be 'full' or 'blank', got '${OPD_TEACHER_IMAGE_MODE}'" >&2; exit 1 ;;
esac
export OPD_TEACHER_IMAGE_MODE

OPD_RUN_NAME="${OPD_RUN_NAME:-t1_v0_${ARM_TAG}}"
export OPD_RUN_NAME

# --- Paths ---
TRAIN_JSONL="${TRAIN_JSONL:-data/opd_train/v0_2k/train.jsonl}"
STUDENT_CKPT="${STUDENT_CKPT:-${MMR1_3B_SFT_CKPT}}"
TEACHER_NAME="${TEACHER_NAME:-MMR1-7B-RL}"
TEACHER_PORT="${TEACHER_PORT:-30000}"
TEACHER_INFO_URL="http://localhost:${TEACHER_PORT}/get_model_info"

if [ ! -f "${TRAIN_JSONL}" ]; then
  echo "ERROR: training JSONL not found at ${TRAIN_JSONL}" >&2
  echo "  run scripts/data/prep_opd_train_data.py first." >&2
  exit 1
fi

# --- Pre-flight: teacher must be alive ---
if ! curl -sf "${TEACHER_INFO_URL}" >/dev/null; then
  echo "ERROR: teacher not reachable at ${TEACHER_INFO_URL}" >&2
  echo "  start it first: bash scripts/train/start_teacher_server.sh" >&2
  exit 1
fi
echo ">>> teacher OK at ${TEACHER_INFO_URL}"

# --- Output dirs ---
EXPERIMENT_DIR="${MLLMOPD_RUNS}/${OPD_RUN_NAME}"
CKPT_DIR="${EXPERIMENT_DIR}/ckpt"
LOG_DIR="${EXPERIMENT_DIR}/logs"
TENSORBOARD_DIR="${EXPERIMENT_DIR}/tensorboard"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${TENSORBOARD_DIR}" \
         "${EXPERIMENT_DIR}/diagnostics"   # opd_diagnostics_hook.py writes here

CUR_TIME=$(date +%Y%m%d_%H%M%S)   # used by DEBUG_ARGS + TRAIN_LOG_FILE below

# --- Hyperparams (T1-v0; plan §6) ---
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
SAMPLE_N="${SAMPLE_N:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * SAMPLE_N))}"
NUM_EPOCH="${NUM_EPOCH:-1}"
LR="${LR:-1e-6}"
LR_WARMUP="${LR_WARMUP:-5}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
OPD_CLIP_RANGE="${OPD_CLIP_RANGE:-10.0}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-4096}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-2048}"

# --- Parallelism (8-GPU host; teacher already binds GPU 0) ---
# Megatron requires GBS % (MICRO_BATCH_SIZE * DP) == 0 where DP = N_GPU / TP.
# With GBS=64, TP=1, MBS=1 the valid N_GPU values are {1,2,4,8,16,32,64} —
# anything not a power-of-2 factor of 64 (like 7) fails the divisibility
# assertion in Megatron's num_microbatches_calculator.
#
# Default to 4 trainer GPUs: gives DP=4, 64/(1*4)=16 micro-batches per
# optimizer step. We waste 3 GPUs (5,6,7 idle) but the plan-§6 estimate
# of ~4-5 h/arm is dominated by rollout time, not optimizer steps, so
# this is the right trade-off for T1-v0. Full-paper runs can bump
# ACTOR_NUM_GPUS_PER_NODE=6 with TP=2 DP=3 + adjusted GBS.
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-${ACTOR_NUM_GPUS_PER_NODE}}"
TP_SIZE="${TP_SIZE:-1}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
TRAINER_GPUS="${TRAINER_GPUS:-1,2,3,4}"

# Divisibility pre-flight — fail loud here rather than wasting Ray init.
DP_SIZE=$(( ACTOR_NUM_GPUS_PER_NODE / TP_SIZE ))
if [ "${DP_SIZE}" -le 0 ]; then
  echo "ERROR: DP_SIZE = ACTOR_NUM_GPUS_PER_NODE / TP_SIZE = ${ACTOR_NUM_GPUS_PER_NODE}/${TP_SIZE} = ${DP_SIZE}; must be >= 1" >&2
  exit 1
fi
MBS_DP=$(( MICRO_BATCH_SIZE * DP_SIZE ))
if [ "$(( GLOBAL_BATCH_SIZE % MBS_DP ))" -ne 0 ]; then
  echo "ERROR: GLOBAL_BATCH_SIZE (${GLOBAL_BATCH_SIZE}) is not divisible by MICRO_BATCH_SIZE * DP (${MICRO_BATCH_SIZE} * ${DP_SIZE} = ${MBS_DP})." >&2
  echo "       Megatron's num_microbatches_calculator will assert. Fix one of:" >&2
  echo "         - GLOBAL_BATCH_SIZE          (currently ${GLOBAL_BATCH_SIZE}; pick a multiple of ${MBS_DP})" >&2
  echo "         - MICRO_BATCH_SIZE           (currently ${MICRO_BATCH_SIZE})" >&2
  echo "         - ACTOR_NUM_GPUS_PER_NODE    (currently ${ACTOR_NUM_GPUS_PER_NODE})" >&2
  echo "         - TP_SIZE                    (currently ${TP_SIZE})" >&2
  exit 1
fi
echo ">>> parallelism: ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE}  TP=${TP_SIZE}  DP=${DP_SIZE}  MBS=${MICRO_BATCH_SIZE}  GBS=${GLOBAL_BATCH_SIZE}  ${GLOBAL_BATCH_SIZE}/${MBS_DP}=$(( GLOBAL_BATCH_SIZE / MBS_DP )) micro-batches/opt-step"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"  # ~5 ckpts over 250 steps; recoverable on crash
DEBUG_MODE="${DEBUG_MODE:-0}"

# --- Custom reward (T1 dual-teacher) ---
RM_ARGS=(
  --custom-rm-path mllmopd.training.dual_teacher_get_reward.get_reward
  --custom-reward-post-process-path mllmopd.training.opd_diagnostics_hook.post_process_rewards_with_diagnostics
)

# --- Arg groups (modeled on the reference launcher) ---
CKPT_ARGS=(
  --hf-checkpoint "${STUDENT_CKPT}"
  --ref-load "${STUDENT_CKPT}"          # KL ref load path (kl_loss_coef=0 so unused, but Megatron still requires the flag)
  --load "${CKPT_DIR}"
  --save "${CKPT_DIR}"
  --save-interval "${SAVE_INTERVAL}"
  --save-hf "${CKPT_DIR}/hf/step_{rollout_id}"   # HF format needed for T1-eval (run_audit_pass_sglang)
)

ROLLOUT_ARGS=(
  --prompt-data "${TRAIN_JSONL}"
  --input-key problem
  --label-key answer
  --apply-chat-template
  --rollout-shuffle
  --num-epoch "${NUM_EPOCH}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${SAMPLE_N}"
  --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature 1
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)

# T1 plan §4: disable Uni-OPD's improvements on top of vanilla OPD.
# These flags exist to NOT pass them (no margin shift, no margin mask,
# no filter, no adv shift) — already absent from ROLLOUT_ARGS above.

GRPO_ARGS=(
  --advantage-estimator on_policy_distillation
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip "${EPS_CLIP}"
  --eps-clip-high "${EPS_CLIP_HIGH}"
  --use-teacher-student-logprob-clip
  --teacher-student-logprob-clip-range "${OPD_CLIP_RANGE}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${LR}"
  --lr-decay-style constant
  --lr-warmup-iters "${LR_WARMUP}"
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

PERF_ARGS=(
  --tensor-model-parallel-size "${TP_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --micro-batch-size "${MICRO_BATCH_SIZE}"
  --use-dynamic-batch-size
  --max-tokens-per-gpu 16384
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.70   # leave room for Megatron colocated
)

TENSORBOARD_ARGS=(
  --use-tensorboard
  --tensorboard-dir "${TENSORBOARD_DIR}"
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
  --colocate
  # Apex's `fused_weight_gradient_mlp_cuda` extension isn't compiled in
  # this venv (we'd need `pip install --global-option=... apex` with
  # CUDA toolchain). Megatron defaults to gradient_accumulation_fusion=True
  # which requires that extension. Disable the fusion to use plain PyTorch
  # grad accumulation — slightly slower kernel path but correctness is
  # identical. Re-enable once apex is installed if perf is needed.
  --no-gradient-accumulation-fusion
)

# Smoke-mode extras (punch list #9). Empty by default for production runs.
DEBUG_ARGS=()
SGLANG_CUDA_GRAPH_ARGS=()
if [ "${DEBUG_MODE}" = "1" ]; then
  DEBUG_ARGS=(
    --save-debug-rollout-data "${EXPERIMENT_DIR}/debug/rollout_${CUR_TIME:-now}/step_{rollout_id}.pt"
  )
  SGLANG_CUDA_GRAPH_ARGS=(--sglang-disable-cuda-graph)
  export NCCL_DEBUG=INFO
  echo ">>> DEBUG_MODE=1 (smoke profile: per-rollout dumps, no CUDA graph, NCCL_DEBUG=INFO)"
fi
SGLANG_ARGS+=("${SGLANG_CUDA_GRAPH_ARGS[@]}")

# Megatron model arch — Qwen2.5-VL-3B shares the language backbone with
# Qwen2.5-3B; the HF checkpoint supplies vision encoder weights, which
# Megatron auto-loads via --hf-checkpoint. If smoke fails due to missing
# vision args, write a custom mmr1-3b-vl.sh and point STUDENT_MODEL_ARGS
# at it.
STUDENT_MODEL_ARGS="${STUDENT_MODEL_ARGS:-${MILES_DIR}/scripts/models/qwen2.5-3B.sh}"
if [ ! -f "${STUDENT_MODEL_ARGS}" ]; then
  echo "ERROR: STUDENT_MODEL_ARGS=${STUDENT_MODEL_ARGS} not found" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${STUDENT_MODEL_ARGS}"

# --- Sanity logging ---
echo "================================================================"
echo "  T1 ARM                  : ${ARM_TAG}"
echo "  OPD_TEACHER_IMAGE_MODE  : ${OPD_TEACHER_IMAGE_MODE}"
echo "  OPD_RUN_NAME            : ${OPD_RUN_NAME}"
echo "  STUDENT                 : ${STUDENT_CKPT}"
echo "  TEACHER (server)        : ${TEACHER_NAME} @ ${TEACHER_INFO_URL}"
echo "  TRAIN_JSONL             : ${TRAIN_JSONL}"
echo "  EXPERIMENT_DIR          : ${EXPERIMENT_DIR}"
echo "  TRAINER_GPUS            : ${TRAINER_GPUS}"
echo "  ROLLOUT_BATCH_SIZE      : ${ROLLOUT_BATCH_SIZE}"
echo "  SAMPLE_N                : ${SAMPLE_N}"
echo "  GLOBAL_BATCH_SIZE       : ${GLOBAL_BATCH_SIZE}"
echo "  NUM_EPOCH               : ${NUM_EPOCH}"
echo "  LR / WARMUP             : ${LR} / ${LR_WARMUP}"
echo "  TP_SIZE                 : ${TP_SIZE}"
echo "  ROLLOUT_MAX_RESPONSE_LEN: ${ROLLOUT_MAX_RESPONSE_LEN}"
echo "================================================================"

# --- Launch Ray ---
# Uni-OPD ships a multi-host ray start script at
# ${MILES_DIR}/Uni_OPD_utils/scripts/ray/start_ray.sh, but it depends on
# a `network_envs.sh` that isn't part of the submodule checkout AND on
# a LOCAL_IP env var supplied by Tencent's internal infra (it then runs
# `ray start --node-ip-address $LOCAL_IP`). On our single-node dev box
# that script feeds an empty `--node-ip-address ''` and Ray bails with
# "Malformed host: ". We skip those scripts entirely and bring Ray up
# locally — single host, all trainer GPUs.
LAUNCHER_SCRIPT="${MILES_DIR}/Uni_OPD_utils/ray_launcher.py"

# NVLink detection — auto-set NCCL_NVLS_ENABLE.
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
HAS_NVLINK=$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)
echo ">>> NVLink links detected: ${NVLINK_COUNT}  (NCCL_NVLS_ENABLE=${HAS_NVLINK})"

# Build PYTHONPATH from ABSOLUTE paths so Ray workers (which can start in
# any cwd) resolve every entry. Pre-G1 the entries were relative + duplicated
# (saw "/repo//repo/miles") because $(pwd)/${MILES_DIR} prepended pwd to an
# already-absolute MILES_DIR.
NEW_PYTHONPATH="${REPO_ROOT}/src:${MILES_DIR}:${MEGATRON_PATH}"
if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="${PYTHONPATH}:${NEW_PYTHONPATH}"
else
  export PYTHONPATH="${NEW_PYTHONPATH}"
fi
echo ">>> PYTHONPATH=${PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEPRECATED_MEGATRON_COMPATIBLE=1
export PYTHONUNBUFFERED=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
# NCCL_DEBUG is set above by DEBUG_MODE=1; otherwise default WARN.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Pin trainer-visible GPUs before ray-start so Ray and Megatron see the
# same device set. Teacher server uses its own CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${TRAINER_GPUS}"
echo ">>> trainer CUDA_VISIBLE_DEVICES=${TRAINER_GPUS} (--num-gpus ${ACTOR_NUM_GPUS_PER_NODE})"

# Stop stale Ray, then start a local single-node cluster.
ray stop --force >/dev/null 2>&1 || true
ray start --head --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" --disable-usage-stats

# Diagnostic-hook outputs are keyed on these:
export MLLMOPD_RUNS OPD_RUN_NAME OPD_TEACHER_IMAGE_MODE

TRAIN_LOG_FILE="${LOG_DIR}/train_${CUR_TIME}.log"

cd "${MILES_DIR}"  # train.py expects miles/ as cwd

set -x
# CUDA_VISIBLE_DEVICES is already exported above (before ray start).
python "${LAUNCHER_SCRIPT}" train.py \
    --actor-num-nodes "${ACTOR_NUM_NODES}" \
    --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
    "${TENSORBOARD_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${DEBUG_ARGS[@]}" \
    "${RM_ARGS[@]}" \
    2>&1 | tee "${TRAIN_LOG_FILE}"
set +x

echo
echo ">>> T1 arm ${ARM_TAG} done. Outputs:"
echo "    checkpoints : ${CKPT_DIR}"
echo "    tensorboard : ${TENSORBOARD_DIR}"
echo "    diagnostics : ${EXPERIMENT_DIR}/diagnostics"
echo "    train log   : ${TRAIN_LOG_FILE}"
