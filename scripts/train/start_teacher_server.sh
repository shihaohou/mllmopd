#!/usr/bin/env bash
# T1 punch list #6: launch the OPD teacher's sglang server + register
# its endpoint into Uni-OPD's teacher_server_{list,map}.json.
#
# Replaces the previous stub that wrapped a missing run_sglang_server.sh.
# Calls sglang directly so we don't depend on Uni-OPD's wrapper script.
#
# Env overrides:
#   TEACHER_MODEL_PATH   (default: $MMR1_7B_RL_CKPT from .env)
#   TEACHER_PORT         (default: 30000)
#   TEACHER_GPUS         (default: 0)          — single GPU for 7B bf16
#   TEACHER_TP_SIZE      (default: 1)
#   TEACHER_MEM_FRACTION (default: 0.75)       — plan §6
#   TEACHER_MAX_RUNNING  (default: 64)
#   TEACHER_NAME         (default: MMR1-7B-RL)
#   FOREGROUND           (default: 0)          — set 1 to keep stdout attached
#                                                 instead of nohupping into a log
#
# Cross-box deploy:
#   TEACHER_ADVERTISE_HOST    explicit IP written into teacher_server_list.json
#   STUDENT_IP                if ADVERTISE_HOST unset, use `ip route get` src
#                             to that IP — auto-picks the NIC that actually
#                             reaches the student. Avoids hostname -I picking
#                             an RDMA/overlay/docker bridge by accident.
#
# Usage:
#   # single-box (teacher + student on same machine)
#   bash scripts/train/start_teacher_server.sh
#
#   # cross-box (student lives at 10.x.y.z — pass at call site, don't .env it)
#   STUDENT_IP=10.x.y.z bash scripts/train/start_teacher_server.sh
#
#   # waits until /get_model_info returns 200, then prints the URL and exits.
#
# Verify alive:
#   curl -s http://localhost:30000/get_model_info | jq .
#
# Stop:
#   pkill -f "sglang.launch_server.*${TEACHER_PORT:-30000}"

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck disable=SC1091
source .env

# H800 container has http_proxy/https_proxy pointed at an oversea squid for
# outbound internet. sglang's startup self-warmup curls 127.0.0.1:${port}/
# model_info; requests respects the proxy env var and routes the loopback
# call through the squid, which returns 502 → assert fails → sglang exits
# (shows up as "Killed" in the parent shell). Strip proxies for this process
# and add loopback + intranet to NO_PROXY for any child that re-reads them.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,10.0.0.0/8,${NO_PROXY:-}"
export no_proxy="${NO_PROXY}"

# Prepend NGC CUDA forward-compat lib to LD_LIBRARY_PATH. Driver 535 +
# cu128 runtime need it; the path is sometimes missing from the env Ray
# / sglang daemons inherit. See opd_mmr1_3b_baseline.sh for the full
# diagnosis — torch's DT_RPATH handles venv NCCL on its own, no need to
# touch venv site-packages paths here.
for d in /usr/local/cuda-12.9/compat/lib.real /usr/local/cuda-12.8/compat/lib.real /usr/local/cuda/compat/lib.real; do
  if [ -d "${d}" ]; then
    LD_LIBRARY_PATH="${d}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export LD_LIBRARY_PATH
    break
  fi
done
[ -d /usr/local/cuda/compat/lib ] && export LD_LIBRARY_PATH="/usr/local/cuda/compat/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MMR1_7B_RL_CKPT:?}}"
TEACHER_PORT="${TEACHER_PORT:-30000}"
TEACHER_GPUS="${TEACHER_GPUS:-0}"
TEACHER_TP_SIZE="${TEACHER_TP_SIZE:-1}"
TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.75}"
TEACHER_MAX_RUNNING="${TEACHER_MAX_RUNNING:-64}"
TEACHER_NAME="${TEACHER_NAME:-MMR1-7B-RL}"
FOREGROUND="${FOREGROUND:-0}"

# sglang lives in the train venv. Operator must source it before running.
if ! python -c "import sglang" >/dev/null 2>&1; then
  echo "ERROR: sglang not importable in the current Python env." >&2
  echo "  current python: $(command -v python)" >&2
  echo "  fix: source /root/shihao_project/mllmopd-train-env/.venv/bin/activate" >&2
  exit 1
fi

# Register the endpoint into Uni-OPD's teacher_server_list.json /
# teacher_server_map.json. RMSystemManager reads these at construction.
MILES_DIR="${MILES_DIR:-third_party/Uni-OPD/miles}"
LIST="${MILES_DIR}/Uni_OPD_utils/OPD_reward/teacher_server_list.json"
MAP="${MILES_DIR}/Uni_OPD_utils/OPD_reward/teacher_server_map.json"

# Pick the IP this host should advertise to the student. Priority:
#   1. TEACHER_ADVERTISE_HOST if set explicitly.
#   2. STUDENT_IP set → ask the kernel which local IP would route to it.
#      Tries `ip route get` first; falls back to a Python UDP-connect trick
#      that works inside slim containers without iproute2. Avoids
#      `hostname -I` picking RDMA/overlay/docker bridges other boxes can't
#      reach. Pass at the call site, do NOT bake into .env (the box pair
#      changes between experiments).
#   3. Fallback "localhost" (single-box mode).
if [ -z "${TEACHER_ADVERTISE_HOST:-}" ] && [ -n "${STUDENT_IP:-}" ]; then
  if command -v ip >/dev/null 2>&1; then
    TEACHER_ADVERTISE_HOST=$(ip route get "${STUDENT_IP}" 2>/dev/null \
      | awk '{for(i=1;i<=NF;i++) if($i=="src"){print $(i+1); exit}}')
  fi
  if [ -z "${TEACHER_ADVERTISE_HOST:-}" ]; then
    # UDP connect doesn't send a packet; getsockname returns the src IP
    # the kernel would use to reach STUDENT_IP.
    TEACHER_ADVERTISE_HOST=$(STUDENT_IP="${STUDENT_IP}" python - <<'PY' 2>/dev/null
import os, socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    s.connect((os.environ["STUDENT_IP"], 1))
    print(s.getsockname()[0])
finally:
    s.close()
PY
)
  fi
fi
TEACHER_ADVERTISE_HOST="${TEACHER_ADVERTISE_HOST:-localhost}"
TEACHER_URL="http://${TEACHER_ADVERTISE_HOST}:${TEACHER_PORT}/generate"

python - <<PY
import json, pathlib
list_path = pathlib.Path("${LIST}")
map_path  = pathlib.Path("${MAP}")
list_path.write_text(json.dumps({
    "${TEACHER_NAME}": {
        "path": "${TEACHER_MODEL_PATH}",
        "servers": ["${TEACHER_URL}"],
    }
}, indent=2))
map_path.write_text(json.dumps({"default": "${TEACHER_NAME}"}, indent=2))
print(f"Registered ${TEACHER_NAME} → ${TEACHER_URL}")
print(f"  list: {list_path}")
print(f"  map:  {map_path}")
PY

# Launch sglang. Reward path requests max_new_tokens=0 (logp-only).
LOG_DIR="${MLLMOPD_RUNS:-runs}/teacher_server"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${TEACHER_NAME}-${TEACHER_PORT}-$(date +%Y%m%d-%H%M%S).log"

TEACHER_HOST="${TEACHER_HOST:-0.0.0.0}"

LAUNCH_CMD=(
  python -m sglang.launch_server
  --model-path "${TEACHER_MODEL_PATH}"
  --host "${TEACHER_HOST}"
  --port "${TEACHER_PORT}"
  --tp-size "${TEACHER_TP_SIZE}"
  --dtype bfloat16
  --mem-fraction-static "${TEACHER_MEM_FRACTION}"
  --max-running-requests "${TEACHER_MAX_RUNNING}"
  --trust-remote-code
  --log-level info
)

echo ">>> teacher model:  ${TEACHER_MODEL_PATH}"
echo ">>> URL:            ${TEACHER_URL}"
echo ">>> GPU(s):         ${TEACHER_GPUS}  (TP=${TEACHER_TP_SIZE})"
echo ">>> mem_fraction:   ${TEACHER_MEM_FRACTION}  max_running: ${TEACHER_MAX_RUNNING}"
echo ">>> log:            ${LOG_FILE}"

if [ "${FOREGROUND}" = "1" ]; then
  CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}" exec "${LAUNCH_CMD[@]}"
fi

CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}" nohup "${LAUNCH_CMD[@]}" \
  > "${LOG_FILE}" 2>&1 &
SGLANG_PID=$!
echo ">>> sglang pid:     ${SGLANG_PID}"

# Wait for /get_model_info to return 200 (sglang's ready signal).
INFO_URL="http://127.0.0.1:${TEACHER_PORT}/get_model_info"
echo ">>> waiting for ${INFO_URL} ..."
for i in $(seq 1 120); do
  if ! kill -0 "${SGLANG_PID}" 2>/dev/null; then
    echo "ERROR: sglang process died. Tail of log:" >&2
    tail -n 40 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  if curl -sf "${INFO_URL}" >/dev/null 2>&1; then
    info=$(curl -s "${INFO_URL}")
    echo
    echo ">>> teacher ready after ~${i}s"
    echo "    get_model_info: ${info}"
    echo ">>> tail -f ${LOG_FILE}    # to follow logs"
    echo ">>> pkill -f 'sglang.launch_server.*${TEACHER_PORT}'    # to stop"
    exit 0
  fi
  sleep 2
done

echo "ERROR: teacher did not come up in 240s. Tail of log:" >&2
tail -n 40 "${LOG_FILE}" >&2 || true
exit 1
