#!/usr/bin/env bash
# Cross-box T2-1: launch N standalone sglang HTTP servers for STUDENT
# rollout inference on Box 1 (the teacher box). Trainer on Box 2 connects
# to them via `--rollout-external --rollout-external-engine-addrs`.
#
# Why: colocate mode on Box 2 puts trainer + sglang on the same GPUs,
# causing OOM at 8 GPU. Decoupling sglang to Box 1 (where teacher already
# is and GPUs 1-7 are otherwise idle) gives the trainer full 140 GiB
# per GPU on Box 2 with no contention.
#
# Server-arg parity: this script's sglang launch flags MUST match what
# Uni-OPD's _compute_server_args produces on the trainer side (modulo the
# `_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS` whitelist: model_path,
# trust_remote_code, random_seed, nccl_port, dist_init_addr,
# skip_server_warmup, enable_draft_weights_cpu_backup, mem_fraction_static).
# All other fields are sanity-checked by SGLangEngine._init_external on
# init; mismatch is a hard launch failure with a clear assert.
#
# Base GPU trick: all N servers share CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"
# (excludes the teacher's GPU 0), and each uses a different --base-gpu-id
# (0..N-1) so each sglang process binds to a different physical GPU.
# This matches the trainer-side expectation that rank R's engine reports
# base_gpu_id=R (Uni-OPD's get_base_gpu_id formula yields exactly that
# with actor_num_gpus_per_node=8 and rollout_num_gpus_per_engine=1).
#
# Env overrides (defaults shown):
#   STUDENT_MODEL_PATH       (default: $MMR1_3B_SFT_CKPT from .env)
#   ROLLOUT_NUM_ENGINES      (default: 7; uses GPUs 1..ROLLOUT_NUM_ENGINES)
#   ROLLOUT_PORT_BASE        (default: 30001)  → ports 30001..30001+N-1
#   ROLLOUT_HOST             (default: 0.0.0.0)
#   ROLLOUT_MEM_FRACTION     (default: 0.25)   — can differ per SKIP list
#   ROLLOUT_MAX_TOTAL_TOKENS (default: 200000) — MUST match trainer
#   ROLLOUT_MAX_RUNNING      (default: 32)     — MUST match trainer
#   ROLLOUT_CUDA_GRAPH_MAX_BS (default: 32)    — MUST match trainer
#   ROLLOUT_NUM_CONTINUOUS_DECODE_STEPS (default: 4) — MUST match trainer
#   TEACHER_GPU              (default: 0)      — GPU to exclude from CVD
#   ROLLOUT_ADVERTISE_HOST    explicit IP written into the addrs file
#   TRAINER_IP               if ADVERTISE_HOST unset, use `ip route get` src
#                            to that IP — auto-pick the NIC reaching trainer
#   FOREGROUND               (default: 0) — set 1 to keep first engine attached
#
# Usage:
#   # On Box 1 (where teacher is already running on GPU 0):
#   TRAINER_IP=10.86.16.16 bash scripts/train/start_rollout_servers.sh
#
#   # Output: $MLLMOPD_RUNS/rollout_servers/rollout_engine_addrs.txt
#   #   line 1: space-separated "host:port" list, one per engine
#   # Trainer reads this file and passes to --rollout-external-engine-addrs.
#
# Verify alive:
#   for p in $(seq 30001 30007); do
#     curl -sf http://localhost:$p/get_model_info && echo " ✓ port $p" || echo " ✗ port $p"
#   done
#
# Stop:
#   pkill -f "sglang.launch_server.*--port 3000[1-9]"
#   pkill -f "sglang.launch_server.*--port 3001[0-7]"

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source .env

# Proxy unset (same rationale as start_teacher_server.sh:52-59).
unset -v http_proxy https_proxy no_proxy

# CUDA forward-compat lib prepend.
for d in /usr/local/cuda-12.9/compat/lib.real /usr/local/cuda-12.8/compat/lib.real /usr/local/cuda/compat/lib.real; do
  if [ -d "${d}" ]; then
    LD_LIBRARY_PATH="${d}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export LD_LIBRARY_PATH
    break
  fi
done
[ -d /usr/local/cuda/compat/lib ] && export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-${MMR1_3B_SFT_CKPT:?}}"
ROLLOUT_NUM_ENGINES="${ROLLOUT_NUM_ENGINES:-7}"
ROLLOUT_PORT_BASE="${ROLLOUT_PORT_BASE:-30001}"
ROLLOUT_HOST="${ROLLOUT_HOST:-0.0.0.0}"
ROLLOUT_MEM_FRACTION="${ROLLOUT_MEM_FRACTION:-0.25}"
ROLLOUT_MAX_TOTAL_TOKENS="${ROLLOUT_MAX_TOTAL_TOKENS:-200000}"
ROLLOUT_MAX_RUNNING="${ROLLOUT_MAX_RUNNING:-32}"
ROLLOUT_CUDA_GRAPH_MAX_BS="${ROLLOUT_CUDA_GRAPH_MAX_BS:-32}"
ROLLOUT_NUM_CONTINUOUS_DECODE_STEPS="${ROLLOUT_NUM_CONTINUOUS_DECODE_STEPS:-4}"
TEACHER_GPU="${TEACHER_GPU:-0}"
FOREGROUND="${FOREGROUND:-0}"

if ! python -c "import sglang" >/dev/null 2>&1; then
  echo "ERROR: sglang not importable. source the train venv first." >&2
  exit 1
fi

if [ ! -d "${STUDENT_MODEL_PATH}" ]; then
  echo "ERROR: student model not found at ${STUDENT_MODEL_PATH}" >&2
  exit 1
fi

# Build CUDA_VISIBLE_DEVICES = all GPUs except teacher's. e.g., TEACHER_GPU=0 → "1,2,3,4,5,6,7"
N_TOTAL_GPUS="${N_TOTAL_GPUS:-8}"
CVD_ENTRIES=()
for g in $(seq 0 $((N_TOTAL_GPUS - 1))); do
  [ "${g}" = "${TEACHER_GPU}" ] && continue
  CVD_ENTRIES+=("${g}")
done
ROLLOUT_CVD=$(IFS=, ; echo "${CVD_ENTRIES[*]}")
N_AVAILABLE=${#CVD_ENTRIES[@]}

if [ "${ROLLOUT_NUM_ENGINES}" -gt "${N_AVAILABLE}" ]; then
  echo "ERROR: ROLLOUT_NUM_ENGINES=${ROLLOUT_NUM_ENGINES} > available ${N_AVAILABLE} (CVD=${ROLLOUT_CVD})" >&2
  exit 1
fi

# Resolve advertise host (the IP written into the addrs file the trainer
# reads). Same priority order as start_teacher_server.sh.
if [ -z "${ROLLOUT_ADVERTISE_HOST:-}" ] && [ -n "${TRAINER_IP:-}" ]; then
  if command -v ip >/dev/null 2>&1; then
    ROLLOUT_ADVERTISE_HOST=$(ip route get "${TRAINER_IP}" 2>/dev/null \
      | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
  fi
  if [ -z "${ROLLOUT_ADVERTISE_HOST:-}" ]; then
    ROLLOUT_ADVERTISE_HOST=$(TRAINER_IP="${TRAINER_IP}" python - <<'PY' 2>/dev/null
import os, socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect((os.environ["TRAINER_IP"], 1))
    print(s.getsockname()[0])
finally:
    s.close()
PY
)
  fi
fi
ROLLOUT_ADVERTISE_HOST="${ROLLOUT_ADVERTISE_HOST:-localhost}"

LOG_DIR="${MLLMOPD_RUNS:-runs}/rollout_servers"
mkdir -p "${LOG_DIR}"
ADDR_FILE="${LOG_DIR}/rollout_engine_addrs.txt"

echo "================================================================"
echo "  Cross-box T2-1 rollout sglang launcher"
echo "  STUDENT_MODEL_PATH   : ${STUDENT_MODEL_PATH}"
echo "  N engines            : ${ROLLOUT_NUM_ENGINES} (GPUs ${ROLLOUT_CVD}, excluding teacher GPU ${TEACHER_GPU})"
echo "  port range           : ${ROLLOUT_PORT_BASE}..$((ROLLOUT_PORT_BASE + ROLLOUT_NUM_ENGINES - 1))"
echo "  advertise host       : ${ROLLOUT_ADVERTISE_HOST}"
echo "  mem_fraction_static  : ${ROLLOUT_MEM_FRACTION}"
echo "  max_total_tokens     : ${ROLLOUT_MAX_TOTAL_TOKENS}    (MUST match trainer --sglang-max-total-tokens)"
echo "  max_running_requests : ${ROLLOUT_MAX_RUNNING}    (MUST match trainer)"
echo "  cuda_graph_max_bs    : ${ROLLOUT_CUDA_GRAPH_MAX_BS}    (MUST match trainer)"
echo "  continuous_decode    : ${ROLLOUT_NUM_CONTINUOUS_DECODE_STEPS}    (MUST match trainer)"
echo "  log dir              : ${LOG_DIR}"
echo "  addr file            : ${ADDR_FILE}"
echo "================================================================"

# Track started PIDs for cleanup on failure.
PIDS=()
PORTS=()

cleanup_on_fail() {
  echo ">>> failure detected, killing started engines: ${PIDS[*]}" >&2
  for pid in "${PIDS[@]:-}"; do
    [ -n "${pid:-}" ] && kill -0 "${pid}" 2>/dev/null && kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_on_fail ERR

# Launch all engines in background.
for i in $(seq 0 $((ROLLOUT_NUM_ENGINES - 1))); do
  PORT=$((ROLLOUT_PORT_BASE + i))
  BASE_GPU_ID=${i}
  TS=$(date +%Y%m%d-%H%M%S)
  LOG_FILE="${LOG_DIR}/rollout_${i}_port${PORT}_gpu${CVD_ENTRIES[i]}_${TS}.log"

  # Sglang launch flags MUST match what Uni-OPD's _compute_server_args
  # produces (see sglang_engine.py:510). All non-SKIP fields are
  # sanity-checked at trainer init.
  # Explicit --nccl-port per engine. Sglang's auto-pick formula is
  # `port + rand(100, 1000)` with an is_port_available check; with 7
  # instances starting in parallel that races and the losers get stuck
  # in NCCL rendezvous (same root cause documented in
  # start_teacher_server.sh TEACHER_NCCL_PORT comment).
  NCCL_PORT=$((ROLLOUT_PORT_BASE + 1000 + i))

  LAUNCH_CMD=(
    python -m sglang.launch_server
    --model-path "${STUDENT_MODEL_PATH}"
    --host "${ROLLOUT_HOST}"
    --port "${PORT}"
    --nccl-port "${NCCL_PORT}"
    --base-gpu-id "${BASE_GPU_ID}"
    --gpu-id-step 1
    --tp-size 1
    --dp-size 1
    --pp-size 1
    --ep-size 1
    --nnodes 1
    --node-rank 0
    --dtype bfloat16
    --trust-remote-code
    --skip-server-warmup
    --enable-draft-weights-cpu-backup
    --mem-fraction-static "${ROLLOUT_MEM_FRACTION}"
    --max-total-tokens "${ROLLOUT_MAX_TOTAL_TOKENS}"
    --max-running-requests "${ROLLOUT_MAX_RUNNING}"
    --cuda-graph-max-bs "${ROLLOUT_CUDA_GRAPH_MAX_BS}"
    --num-continuous-decode-steps "${ROLLOUT_NUM_CONTINUOUS_DECODE_STEPS}"
    --log-level info
  )

  echo ">>> launching engine ${i}: physical GPU ${CVD_ENTRIES[i]} (base-gpu-id=${BASE_GPU_ID}), port=${PORT}"
  echo "    log: ${LOG_FILE}"

  CUDA_VISIBLE_DEVICES="${ROLLOUT_CVD}" nohup "${LAUNCH_CMD[@]}" \
    > "${LOG_FILE}" 2>&1 &
  PIDS+=($!)
  PORTS+=("${PORT}")
done

echo ">>> ${ROLLOUT_NUM_ENGINES} engines spawning. Waiting for readiness..."

# Poll each engine's /get_model_info; collect addr list.
ADDRS=()
for i in $(seq 0 $((ROLLOUT_NUM_ENGINES - 1))); do
  PORT="${PORTS[i]}"
  PID="${PIDS[i]}"
  INFO_URL="http://127.0.0.1:${PORT}/get_model_info"
  ready=0
  for attempt in $(seq 1 180); do
    if ! kill -0 "${PID}" 2>/dev/null; then
      echo "ERROR: engine ${i} (pid ${PID}, port ${PORT}) died. Tail of log:" >&2
      LOG_PATTERN="${LOG_DIR}/rollout_${i}_port${PORT}_*"
      # shellcheck disable=SC2086
      LATEST_LOG=$(ls -t ${LOG_PATTERN} 2>/dev/null | head -1)
      [ -n "${LATEST_LOG:-}" ] && tail -n 30 "${LATEST_LOG}" >&2 || true
      cleanup_on_fail
      trap - ERR
      exit 1
    fi
    if curl -sf "${INFO_URL}" >/dev/null 2>&1; then
      echo ">>> engine ${i} (port ${PORT}) ready after ~$((attempt * 2))s"
      ready=1
      break
    fi
    sleep 2
  done
  if [ "${ready}" = "0" ]; then
    echo "ERROR: engine ${i} did not come up in 360s." >&2
    cleanup_on_fail
    trap - ERR
    exit 1
  fi
  ADDRS+=("${ROLLOUT_ADVERTISE_HOST}:${PORT}")
done

trap - ERR

# Write addr list (single-line, space-separated, ready to splat into the
# trainer's --rollout-external-engine-addrs).
echo "${ADDRS[*]}" > "${ADDR_FILE}"

# Also write a per-line file for easier programmatic consumption.
printf "%s\n" "${ADDRS[@]}" > "${ADDR_FILE}.list"

echo
echo "================================================================"
echo ">>> All ${ROLLOUT_NUM_ENGINES} rollout engines ready."
echo ">>> Addrs written to:  ${ADDR_FILE}"
echo "    $(cat "${ADDR_FILE}")"
echo
echo ">>> On the trainer box, source this file when launching:"
echo "    ROLLOUT_ENGINE_ADDRS=\"\$(cat ${ADDR_FILE})\" \\"
echo "      bash scripts/train/opd_mmr1_3b_baseline_xbox.sh"
echo
echo ">>> To follow a specific engine:"
echo "    tail -f ${LOG_DIR}/rollout_<i>_*.log"
echo ">>> To stop all engines:"
echo "    pkill -f 'sglang.launch_server.*--port (3000[1-9]|3001[0-7])'"
echo "================================================================"
