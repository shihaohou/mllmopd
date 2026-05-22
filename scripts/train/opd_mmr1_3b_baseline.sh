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
# T2-1 (method tier, on top of T1 plumbing):
#   MLLMOPD_USE_VD_WEIGHTING  0 | 1   (default 0 = T1 byte-identical)
#   MLLMOPD_VD_TAU            float    (default 0.4; PGPO Eq 6 threshold on min-max normalized vd_t)
#   MLLMOPD_VD_BETA           float    (default 2.0; PGPO Eq 6 boost slope)
#   When MLLMOPD_USE_VD_WEIGHTING=1, OPD_TEACHER_IMAGE_MODE must be 'full'
#   (T2-1 is VD-weighted FullTeacher OPD; see docs/t2_1_design.md).
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
#
#   # T2-1 run (VD-weighted FullTeacher OPD)
#   MLLMOPD_USE_VD_WEIGHTING=1 OPD_TEACHER_IMAGE_MODE=full \
#     bash scripts/train/opd_mmr1_3b_baseline.sh

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

# Smoke #15 follow-up: GPT diagnosis indicated the persistent NCCL
# "driver insufficient" with VERSION-MISMATCH (compile=2.27.5, runtime
# reported as 2.29.7) means ld.so is finding a foreign libnccl ahead
# of the venv's bundled one. Prepend the venv's NCCL lib dir to
# LD_LIBRARY_PATH so dlopen("libnccl.so.2") resolves there first,
# AHEAD of the cuda-compat path.
#
# Use a separate python tempfile (function won't help when the issue
# is bash inline-heredoc parsing). Defensive: try `import nvidia.nccl`
# then glob site-packages for any libnccl.so.2.
_DISCOVER_PY=$(mktemp --tmpdir mllmopd_nccl_discover.XXXXXX.py)
cat > "${_DISCOVER_PY}" <<'PYEOF'
import sys, glob, site
from pathlib import Path
try:
    import nvidia.nccl
    p = Path(nvidia.nccl.__file__).resolve().parent / "lib"
    if (p / "libnccl.so.2").exists():
        print(p)
        sys.exit(0)
except Exception:
    pass
for sp in site.getsitepackages():
    hits = glob.glob(sp + "/**/libnccl.so.2", recursive=True)
    if hits:
        print(str(Path(hits[0]).parent))
        sys.exit(0)
PYEOF
PYTORCH_NCCL_LIB_DIR="$(python "${_DISCOVER_PY}" 2>/dev/null || true)"
rm -f "${_DISCOVER_PY}"

if [ -n "${PYTORCH_NCCL_LIB_DIR}" ] && [ -e "${PYTORCH_NCCL_LIB_DIR}/libnccl.so.2" ]; then
  LD_LIBRARY_PATH="${PYTORCH_NCCL_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export LD_LIBRARY_PATH
  echo "    venv libnccl   : ${PYTORCH_NCCL_LIB_DIR}/libnccl.so.2 (prepended FIRST)"
else
  echo "    venv libnccl   : (not found — fallback to ld.so.cache)"
  python -c "import nvidia.nccl as m; print('    nvidia.nccl import OK at', m.__file__)" 2>&1 \
    | sed 's/^/        /' || true
fi
echo "    LD_LIBRARY_PATH before : ${LD_BEFORE}"
echo "    LD_LIBRARY_PATH after  : ${LD_LIBRARY_PATH:-(empty)}"

# NCCL ≥ 2.28 probes for libnccl-env.so plugin on init; silence the
# noise so the real error is the first NCCL log line.
export NCCL_ENV_PLUGIN="${NCCL_ENV_PLUGIN:-none}"

# Proxy handling:
#   Default (MLLMOPD_KEEP_PROXY unset or 0): strip http_proxy/https_proxy
#     to prevent the v0 bug where requests.post("http://10.82.121.12:...")
#     was routed through oversea-squid1.jp.txyun:11080, which returned 502
#     because the internal IP isn't reachable from outside (smoke #19 root
#     cause). Safe default for sglang/ray intranet HTTP traffic.
#   MLLMOPD_KEEP_PROXY=1: keep http_proxy/https_proxy intact. Required for
#     wandb online sync (api.wandb.ai is external, needs the squid). Relies
#     on no_proxy whitelisting 10.x intranet to keep sglang/ray bypass.
#     Tested 2026-05-22 — if sglang student/teacher 502s reappear, fall
#     back to default (unset MLLMOPD_KEEP_PROXY).
# Defense in depth: always set no_proxy for intranet, regardless of mode.
# Set this BEFORE the optional unset so requests honors it either way.
export no_proxy="${no_proxy:+${no_proxy},}localhost,127.0.0.1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
export NO_PROXY="${no_proxy}"
if [ "${MLLMOPD_KEEP_PROXY:-0}" = "1" ]; then
  echo ">>> proxy kept (MLLMOPD_KEEP_PROXY=1): http_proxy=${http_proxy:-(unset)}"
  echo "                                       no_proxy will bypass 10.x intranet"
else
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
  echo ">>> proxy stripped (default; set MLLMOPD_KEEP_PROXY=1 to keep for wandb)"
fi

# Strip torch_memory_saver from LD_PRELOAD. This .so hijacks
# cudaMalloc/cudaFree to support sglang's incremental KV-cache eviction;
# the actor inspect log (smoke #13) showed it was the only LD_PRELOAD
# entry not from ray itself, and its hook semantics interact badly with
# NCCL collectives (NCCL allocates GPU mem inside all_gather; if the
# hook returns a non-zero rc, NCCL wraps it as the misleading
# "ncclUnhandledCudaError / Cuda failure 'CUDA driver version
# insufficient'" we've been chasing for hours).
LD_PRELOAD_BEFORE="${LD_PRELOAD:-(unset)}"
if [ -n "${LD_PRELOAD:-}" ]; then
  LD_PRELOAD=$(echo "${LD_PRELOAD}" | tr ':' '\n' \
      | grep -v 'torch_memory_saver' \
      | tr '\n' ':' | sed 's/:$//')
  export LD_PRELOAD
fi
echo ">>> LD_PRELOAD cleanup (strip torch_memory_saver hook):"
echo "    before: ${LD_PRELOAD_BEFORE}"
echo "    after : ${LD_PRELOAD:-(empty)}"

# Open NCCL_DEBUG to get the real last-error (not the misleading wrapper).
# Set TORCH_NCCL_BLOCKING_WAIT=1 so the failure location pinpoints the
# specific collective op rather than getting swallowed by async handling.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,COLL}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-0}"
# Force sync CUDA — the qwen2_5_vl.py:851 'invalid argument' in smoke #20
# is async-reported (per torch error msg "stacktrace below might be
# incorrect"). With BLOCKING=1, the error pinpoints the actual failing
# CUDA call so we can see which weight tensor's copy_() fails.
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
echo ">>> NCCL diagnostic env: NCCL_DEBUG=${NCCL_DEBUG} SUBSYS=${NCCL_DEBUG_SUBSYS} BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT} ASYNC_ERR=${TORCH_NCCL_ASYNC_ERROR_HANDLING} CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING}"

# T1-2 OOM #6 fragmentation fix: backward pass needed 8.48 GiB
# contiguous but PyTorch had 8.86 GiB scattered across reserved-but-
# unallocated segments. PyTorch's own OOM hint recommends
# expandable_segments; previous attempt (aa26bfd) was retracted in
# c6a0bfd because it crashed sglang's _TorchMemorySaverAdapterReal
# init. Root cause of that crash: --colocate defaults BOTH
# --offload-train AND --offload-rollout to True. We turn off train-side
# (--no-offload-train, to avoid the LD_PRELOAD/NCCL conflict from smoke
# #14) but had left rollout-side on; rollout=True picks the Real
# adapter which calls torch_memory_saver, which sanity-checks
# expandable_segments and raises.
# Fix: also pass --no-offload-rollout below (sglang adapter switches to
# the no-op variant, no TMS import). Then expandable_segments is safe.
# Trade-off: sglang's release_memory_occupation becomes a literal no-op,
# but that was already effectively true in our setup (LD_PRELOAD hook
# was stripped) — see v6 PyTorch peak 115 GiB without any real KV-cache
# release happening anyway.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo ">>> CUDA alloc env: PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

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

# T2-1: VD-weighted FullTeacher OPD. PGPO-style per-token weighting on
# vd_t = lp_teacher_full(t) - lp_teacher_blank(t). See
# docs/t2_1_design.md and src/mllmopd/training/vd_weighting.py.
# Defaults match PGPO Table 2 (tau=0.4, beta=2.0). Set MLLMOPD_USE_VD_WEIGHTING=1
# to enable; otherwise this script runs as T1-2 / T1-3 byte-identical.
MLLMOPD_USE_VD_WEIGHTING="${MLLMOPD_USE_VD_WEIGHTING:-0}"
MLLMOPD_VD_TAU="${MLLMOPD_VD_TAU:-0.4}"
MLLMOPD_VD_BETA="${MLLMOPD_VD_BETA:-2.0}"
if [ "${MLLMOPD_USE_VD_WEIGHTING}" = "1" ]; then
  if [ "${OPD_TEACHER_IMAGE_MODE}" != "full" ]; then
    echo "ERROR: T2-1 (MLLMOPD_USE_VD_WEIGHTING=1) requires OPD_TEACHER_IMAGE_MODE=full" >&2
    echo "  reason: VD = lp_full - lp_blank; weighting only meaningful on the FullTeacher primary arm" >&2
    exit 1
  fi
  ARM_TAG="T2_1_full_vd"
fi
export MLLMOPD_USE_VD_WEIGHTING MLLMOPD_VD_TAU MLLMOPD_VD_BETA

if [ "${MLLMOPD_USE_VD_WEIGHTING}" = "1" ]; then
  OPD_RUN_NAME_DEFAULT="t2_1_v0_${ARM_TAG}"
else
  OPD_RUN_NAME_DEFAULT="t1_v0_${ARM_TAG}"
fi
OPD_RUN_NAME="${OPD_RUN_NAME:-${OPD_RUN_NAME_DEFAULT}}"
export OPD_RUN_NAME

# --- Paths ---
TRAIN_JSONL="${TRAIN_JSONL:-data/opd_train/v0_2k/train.jsonl}"
STUDENT_CKPT="${STUDENT_CKPT:-${MMR1_3B_SFT_CKPT}}"
TEACHER_NAME="${TEACHER_NAME:-MMR1-7B-RL}"
TEACHER_PORT="${TEACHER_PORT:-30000}"
# Cross-box: override TEACHER_HOST=10.86.x.x at the call site to hit a
# teacher running on a different machine. Default localhost preserves
# single-box behavior. Reward worker reads servers list from
# teacher_server_list.json (already cross-box if start_teacher_server.sh
# was invoked with TEACHER_ADVERTISE_HOST=...); this URL is just the
# pre-flight liveness probe.
TEACHER_HOST="${TEACHER_HOST:-localhost}"
TEACHER_INFO_URL="${TEACHER_INFO_URL:-http://${TEACHER_HOST}:${TEACHER_PORT}/get_model_info}"

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

# --- T1-2 OOM v11 memsnap envs (docs/gpt-reply-2026-05-20-v10-update.md).
# Originally default-on for v11 to capture ground-truth allocation
# snapshots at policy_loss_function's get_log_probs_and_entropy(...) call.
# v11 confirmed the diagnosis (--use-dynamic-batch-size + max-tokens-per-gpu
# 16384 was packing 9.3 GiB fp32 logits per CE call). With the root cause
# known and the fix applied, default-off going forward — re-enable with
# MLLMOPD_MEMSNAP=1 if a future memory regression needs the same workflow.
# Used by:
#   src/mllmopd/training/memsnap.py               (helpers + hook)
#   patch_uni_opd.sh P9 (loss.py pre-CE dump)     (sentinel-idempotent)
#   --custom-megatron-before-train-step-hook-path (one-shot DDP audit)
MLLMOPD_MEMSNAP="${MLLMOPD_MEMSNAP:-0}"
MLLMOPD_MEMSNAP_DIR="${MLLMOPD_MEMSNAP_DIR:-${EXPERIMENT_DIR}/diagnostics/memsnap}"
MLLMOPD_MEMSNAP_MAX_ENTRIES="${MLLMOPD_MEMSNAP_MAX_ENTRIES:-2000000}"
MLLMOPD_MEMSNAP_MAX_DUMPS="${MLLMOPD_MEMSNAP_MAX_DUMPS:-50}"
export MLLMOPD_MEMSNAP MLLMOPD_MEMSNAP_DIR \
       MLLMOPD_MEMSNAP_MAX_ENTRIES MLLMOPD_MEMSNAP_MAX_DUMPS
if [ "${MLLMOPD_MEMSNAP}" = "1" ]; then
  mkdir -p "${MLLMOPD_MEMSNAP_DIR}"
fi

CUR_TIME=$(date +%Y%m%d_%H%M%S)   # used by DEBUG_ARGS + TRAIN_LOG_FILE below

# Persist launcher stdout/stderr into the run dir on ceph so post-mortem
# survives box switch / reboot. Setup-banner output before this point (env
# resolution, teacher health check, NCCL diag) is intentionally not
# captured here — it's caller-visible in the tmux pane but small enough
# not to be worth additional plumbing. Caller no longer needs to `| tee`
# anything; the file path is echoed below.
LAUNCHER_LOG="${LOG_DIR}/launcher_${CUR_TIME}.log"
exec > >(tee -a "${LAUNCHER_LOG}") 2>&1
echo ">>> launcher log: ${LAUNCHER_LOG}"

# --- Hyperparams (T1-v0; plan §6) ---
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
SAMPLE_N="${SAMPLE_N:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * SAMPLE_N))}"
NUM_EPOCH="${NUM_EPOCH:-1}"
# Direct step cap override. When set, takes precedence over NUM_EPOCH
# (miles drops --num-epoch when --num-rollout is also passed). Use for
# fast A/B ablations (e.g., NUM_ROLLOUT=100 for a quick 100-step probe
# instead of the full 2000/8=250-step main run). Empty/unset = follow
# NUM_EPOCH × dataset_size / rollout_batch_size as usual.
NUM_ROLLOUT="${NUM_ROLLOUT:-}"
LR="${LR:-1e-6}"
LR_WARMUP="${LR_WARMUP:-5}"
EPS_CLIP="${EPS_CLIP:-0.2}"
EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
OPD_CLIP_RANGE="${OPD_CLIP_RANGE:-10.0}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-4096}"
# Response-length cap history:
#   v0 default 2048 → train-eval mismatch.
#   v1 try 8192 → OOM, try 6144 → OOM, settled 4096 → ran to step ~148 OOM.
#   v1.5 4096 + in-place div_ + stop string → OOM again at step ~30
#       because packing 4 mid-length samples (avg ~1500 each) into a
#       7000+ packed micro-batch pushed CE peak over the cliff.
#   v1.5b 3072 + lowered --max-tokens-per-gpu to 6144 → this version.
# The operative lever is --max-tokens-per-gpu (packed cap, see PERF_ARGS
# below); 3072 single-sample cap is belt-and-suspenders. Most responses
# under stop string </answer> are far shorter; 3072 only truncates the
# few real long-CoT outliers. Same caps on both arms — comparison fair.
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-3072}"

# Stop string: when the student emits this exact sequence, sglang halts
# generation. v0 audit on T1-2/T1-3 outputs showed `</answer>` closes
# 94% of all responses; cutting the post-close 6% long-tail saves both
# wallclock and memory peak without changing the meaningful content of
# any response. Comma-separated list (sglang accepts multiple). Disable
# by passing ROLLOUT_STOP="".
ROLLOUT_STOP="${ROLLOUT_STOP:-</answer>}"

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

# --- Train actor env override (smoke #14 root cause analysis) ---
# Uni-OPD's miles/ray/actor_group.py sets
# `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` (plus 6 sibling
# NOSET_*_VISIBLE_DEVICES flags) on every train actor by default.
# Those flags tell Ray "don't rewrite CUDA_VISIBLE_DEVICES inside the
# actor", which means each actor inherits the launcher's full
# `CUDA_VISIBLE_DEVICES=1,2,3,4` and sees device_count==4. Four ranks
# then end up logically on the same cuda:0, and any NCCL collective
# (e.g., the first all_gather_object inside _get_param_groups) fails
# because they're all on the same physical GPU.
#
# Override the NOSET flags to empty strings via Uni-OPD's
# `--train-env-vars` JSON argument, so Ray DOES perform its normal
# per-actor CUDA_VISIBLE_DEVICES rewrite. Each actor then sees exactly
# one logical GPU.
if [ -z "${TRAIN_ENV_VARS_JSON:-}" ]; then
  _ENVJSON_PY=$(mktemp --tmpdir mllmopd_envjson.XXXXXX.py)
  cat > "${_ENVJSON_PY}" <<'PYEOF'
import json, os
print(json.dumps({
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "",
    "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "",
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "",
    "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES": "",
    "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES": "",
    "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS": "",
    "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR": "",
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    "NCCL_ENV_PLUGIN": os.environ.get("NCCL_ENV_PLUGIN", "none"),
    # Proxy propagation to Ray actor (parallel to the parent-shell logic
    # near smoke #19 comment block). Default: strip (sglang local 502
    # safety). MLLMOPD_KEEP_PROXY=1: propagate parent's proxy so wandb
    # inside the actor can reach api.wandb.ai. Relies on no_proxy to
    # keep intranet 10.x bypassing. Both branches always set no_proxy.
    **(
        {
            "http_proxy": os.environ.get("http_proxy", ""),
            "https_proxy": os.environ.get("https_proxy", ""),
            "HTTP_PROXY": os.environ.get("HTTP_PROXY", ""),
            "HTTPS_PROXY": os.environ.get("HTTPS_PROXY", ""),
        }
        if os.environ.get("MLLMOPD_KEEP_PROXY", "0") == "1"
        else {
            "http_proxy": "",
            "https_proxy": "",
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
        }
    ),
    "no_proxy": os.environ.get("no_proxy", ""),
    "NO_PROXY": os.environ.get("NO_PROXY", ""),
    # v11 memsnap envs — propagate to Ray actors so memsnap.py + P9 fire
    # inside each rank's process (set above in the parent shell).
    "MLLMOPD_MEMSNAP": os.environ.get("MLLMOPD_MEMSNAP", ""),
    "MLLMOPD_MEMSNAP_DIR": os.environ.get("MLLMOPD_MEMSNAP_DIR", ""),
    "MLLMOPD_MEMSNAP_MAX_ENTRIES": os.environ.get("MLLMOPD_MEMSNAP_MAX_ENTRIES", ""),
    "MLLMOPD_MEMSNAP_MAX_DUMPS": os.environ.get("MLLMOPD_MEMSNAP_MAX_DUMPS", ""),
    # T2-1 VD weighting (read by opd_diagnostics_hook + vd_weighting).
    "MLLMOPD_USE_VD_WEIGHTING": os.environ.get("MLLMOPD_USE_VD_WEIGHTING", ""),
    "MLLMOPD_VD_TAU": os.environ.get("MLLMOPD_VD_TAU", ""),
    "MLLMOPD_VD_BETA": os.environ.get("MLLMOPD_VD_BETA", ""),
}))
PYEOF
  TRAIN_ENV_VARS_JSON=$(python "${_ENVJSON_PY}")
  rm -f "${_ENVJSON_PY}"
fi
TRAIN_ENV_ARGS=(
  --train-env-vars "${TRAIN_ENV_VARS_JSON}"
  --num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
)

# --- Custom reward (T1 dual-teacher) ---
RM_ARGS=(
  --custom-rm-path mllmopd.training.dual_teacher_get_reward.get_reward
  --custom-reward-post-process-path mllmopd.training.opd_diagnostics_hook.post_process_rewards_with_diagnostics
  # v11: before-train-step hook for per-step memsnap baseline + one-shot
  # DDP buffer audit (GPT diag Q1). No-op when MLLMOPD_MEMSNAP unset.
  --custom-megatron-before-train-step-hook-path mllmopd.training.memsnap.before_train_step_hook
)

# --- Arg groups (modeled on the reference launcher) ---
# `--load` semantics: Uni-OPD's checkpoint.py asserts that the dir exists
# AND is non-empty. The reference upstream launcher always points it at
# CKPT_DIR (same as --save), which works on resume but fails the
# assertion on fresh runs (empty dir). Auto-detect: if CKPT_DIR is
# non-empty (resume from a previous run's last save), load from there;
# otherwise load from the HF student checkpoint via bridge mode.
if [ -d "${CKPT_DIR}" ] && [ -n "$(ls -A "${CKPT_DIR}" 2>/dev/null)" ]; then
  LOAD_PATH="${CKPT_DIR}"
  echo ">>> --load (resume): ${CKPT_DIR}"
else
  LOAD_PATH="${STUDENT_CKPT}"
  echo ">>> --load (fresh run, HF bridge): ${STUDENT_CKPT}"
fi

CKPT_ARGS=(
  --hf-checkpoint "${STUDENT_CKPT}"
  --ref-load "${STUDENT_CKPT}"          # KL ref load path (kl_loss_coef=0 so unused, but Megatron still requires the flag)
  --load "${LOAD_PATH}"
  --save "${CKPT_DIR}"
  --save-interval "${SAVE_INTERVAL}"
  --save-hf "${CKPT_DIR}/hf/step_{rollout_id}"   # HF format needed for T1-eval (run_audit_pass_sglang)
)

ROLLOUT_ARGS=(
  --prompt-data "${TRAIN_JSONL}"
  --input-key problem
  --label-key answer
  # T1 v0 BUG (caught 2026-05-21, mid GPT review): without --multimodal-keys
  # miles Dataset (third_party/Uni-OPD/miles/miles/utils/data.py:324) computes
  # has_mm = multimodal_keys and any(...) = False, so every sample gets
  # multimodal_inputs=None. The rollout payload then ships no image_data and
  # make_blank_image_data returns [] early, so BOTH FullTeacher and
  # BlankTeacher ended up scoring text-only — OPD_TEACHER_IMAGE_MODE became
  # a no-op. The "T1-2 ≈ T1-3 vision-invariance" result from
  # t1_v0_eval_20260521-140736 was bug-induced, not a real negative.
  # Wire the JSONL `images` field as the image multimodal key so the
  # full MLLM data flow actually runs.
  --multimodal-keys '{"image":"images"}'
  --apply-chat-template
  --rollout-shuffle
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt "${SAMPLE_N}"
  --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
  --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
  --rollout-temperature 1
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --balance-data
)

# Long-tail stop string. miles' sglang_rollout.py (line 57/511) forwards
# args.rollout_stop into sglang's sampling_params.stop. v1 audit showed
# `</answer>` closes 94% of T1-2/T1-3 outputs; the remaining 6%
# uncapped long-tail caused the v1 step_148 OOM. Letting sglang
# truncate at the closing tag eliminates that tail without touching the
# meaningful response content. Same stop on both arms — comparison fair.
if [ -n "${ROLLOUT_STOP}" ]; then
  ROLLOUT_ARGS+=(--rollout-stop "${ROLLOUT_STOP}")
  echo ">>> rollout stop string : ${ROLLOUT_STOP}"
fi

# Step-cap selector: --num-rollout takes precedence in miles' arg parser
# (see third_party/Uni-OPD/miles/miles/utils/arguments.py:1931-1939).
if [ -n "${NUM_ROLLOUT}" ]; then
  ROLLOUT_ARGS+=(--num-rollout "${NUM_ROLLOUT}")
  echo ">>> step cap: --num-rollout ${NUM_ROLLOUT} (overrides --num-epoch)"
else
  ROLLOUT_ARGS+=(--num-epoch "${NUM_EPOCH}")
fi

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
  # ZeRO-1: shard Adam state (fp32 master + m + v ≈ 36 GB for 3B) across
  # DP ranks. Without this, each rank holds the full optimizer state and
  # peak trainer memory on H800 ran 131/140 GiB at step 6 actor_train,
  # OOM-ing on logits.clone(). With DP=4 this saves ~27 GB per rank.
  --use-distributed-optimizer
  # Chunk the per-token log-prob / cross-entropy computation in
  # miles/utils/ppo_utils.py::calculate_log_probs_and_entropy. v9 hit
  # OOM at trainer alloc 118 GiB + sglang 21 GiB = 139/140; halving the
  # CE chunk buffer (256→128) saves ~600 MiB at the peak.
  --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-128}"
  --use-dynamic-batch-size
  # Packed-token cap evolution:
  #   v0 32k: original default. OOM at fp32 [1,16k,V]=9.3 GiB logits.
  #   v11 8k:  GPT diag Q1 fix. Worked at response_len=2048.
  #   v1.5 8k: with response_len=4096, mid-length samples (~1500 ea)
  #            packed 4-at-a-time → packed length 7600+ at alloc_gib 87
  #            GiB → CE step OOM with 4.35 GiB needed on near-full GPU.
  #   v1.5b 4k: HARD cap below all known-OOM packed configs. fp32 logits
  #            peak [1,4096,V]=2.4 GiB (vs 4.7 GiB at 8k, and STRICTLY
  #            smaller than v11's known-safe peak). max single sample
  #            (3072 response + ~500-800 prompt incl. image tokens
  #            ≈ 3800) fits alone in 4096 cap. Short samples pack 3-4
  #            per micro-batch. ~4 micro-batches/opt-step (vs ~2 at 8k
  #            cap) → ~30-50% slower wall clock. Acceptable for not OOMing.
  --max-tokens-per-gpu 4096
)

# Smoke #21 + T1-2 OOM #3+#7 root cause (docs/gpt-diagnosis-v7-update):
# CORRECTION over earlier comment block: mem_fraction_static covers
# *weights + KV cache pool* together, not pure KV. With --no-offload-
# rollout (selected to avoid the TMS / expandable_segments crash),
# sglang never yields any of that block back to PyTorch. v7 showed
# the static + cuda-graph + runtime overhead really does fill the
# entire mem_fraction quota: at 0.25 × 140 = 35 GiB target, the
# observed non-PyTorch resident block was ~44 GiB, leaving only ~96
# GiB for the trainer and triggering an OOM on the next 150 MiB.
#
# Per GPT v7 brief Q5 ranks 1-3: stack three sglang shrink levers,
# all minimum-disturbance to the experimental variables:
#   - mem_fraction 0.25 → 0.15 (target weights+KV = ~21 GiB)
#   - --sglang-disable-cuda-graph in production (frees graph buffers)
#   - --sglang-max-total-tokens 200000 + --sglang-max-running-requests 16
#     (cap KV directly; auto-profile was opportunistically filling
#     the whole fraction)
#
# Budget at 0.15 + caps + disabled cuda graph (H800 140 GiB):
#   sglang weights+KV target : ~21 GiB
#   sglang cuda graph        : ~0 GiB (disabled)
#   sglang runtime / IPC     : ~3-5 GiB
#   trainer static (ZeRO-1)  : 21 GiB
#   fp32 grad allreduce      : 12 GiB
#   activations + CE chunk   : 15-25 GiB
#   ----------------------------------
#   total                    : 72-84 GiB  (vs. v7's 95+44 = 139 GiB)
#
# Throughput cost: per Uni-OPD with 4 rollout engines × 16 running
# requests × 6144 max seq = up to 393k tokens worst-case if uneven.
# 200k cap means some samples queue; doesn't change OPD experiment.
SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.15}"
SGLANG_MAX_TOTAL_TOKENS="${SGLANG_MAX_TOTAL_TOKENS:-200000}"
# v8 measurement: disable_cuda_graph + max_running=16 made throughput
# crash ~10x (7k → 700 tok/s/engine). KV usage held at 0.10 the whole
# rollout — sglang was NOT memory-pressured; the slowdown was batched
# decode kernel-launch overhead + routing imbalance (1 engine had 50+
# samples, 3 engines idle).
# Per GPT v8 brief (docs/gpt-diagnosis-2026-05-20-t1-oom-v7-update.md):
#   - max_running 16 → 32: lets the busy engine decode half the rollout
#     batch concurrently, ≤3.4 GiB extra KV at worst-case 6144-token len
#   - cuda_graph_max_bs=32: pair with max_running to capture steady-state
#     batch size; ~1-3 GiB graph buffer inside mem_fraction quota
#   - num_continuous_decode_steps=4: 4 decode steps per scheduler turn,
#     reduces overhead for our 2048-token responses
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-32}"
SGLANG_CUDA_GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-32}"
SGLANG_NUM_CONTINUOUS_DECODE_STEPS="${SGLANG_NUM_CONTINUOUS_DECODE_STEPS:-4}"
SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
  --sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
  --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}"
  --sglang-num-continuous-decode-steps "${SGLANG_NUM_CONTINUOUS_DECODE_STEPS}"
)

TENSORBOARD_ARGS=(
  --use-tensorboard
  --tensorboard-dir "${TENSORBOARD_DIR}"
)

# wandb is opt-in: only enabled when WANDB_API_KEY is non-empty in .env or
# the calling shell. Group name = OPD_RUN_NAME with random-suffix disabled,
# so the wandb run matches the on-disk run dir 1:1.
WANDB_ARGS=()
if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ARGS=(
    --use-wandb
    --wandb-key "${WANDB_API_KEY}"
    --wandb-project "${WANDB_PROJECT:-mllmopd}"
    --wandb-group "${OPD_RUN_NAME}"
    --disable-wandb-random-suffix
  )
  [ -n "${WANDB_ENTITY:-}" ] && WANDB_ARGS+=(--wandb-team "${WANDB_ENTITY}")
  echo ">>> wandb enabled : project=${WANDB_PROJECT:-mllmopd}  group=${OPD_RUN_NAME}"
else
  echo ">>> wandb skipped (set WANDB_API_KEY in .env to enable)"
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-softmax-in-fp32
  --attention-backend flash
  --colocate
  # Loading from --hf-checkpoint requires bridge mode; the default
  # "raw" mode trips an assertion in
  # miles/backends/megatron_utils/checkpoint.py:175. All upstream
  # qwen3 example launchers also set this.
  --megatron-to-hf-mode bridge
  # Smoke #14 root cause analysis: --colocate defaults --offload-train
  # to True; with that, miles/ray/actor_group.py injects
  # `LD_PRELOAD=...torch_memory_saver_hook_mode_preload.abi3.so` into
  # every train actor's runtime_env. The TMS hook intercepts cudaMalloc
  # and breaks NCCL's collective allocations, producing the misleading
  # "ncclUnhandledCudaError: CUDA driver insufficient" we've been
  # chasing for hours. Disable offload for now — re-enable later if
  # full run OOMs.
  --no-offload-train
  # T1-2 OOM #6: --colocate ALSO defaults --offload-rollout=True, which
  # makes sglang's TorchMemorySaverAdapter pick the Real variant and
  # call into torch_memory_saver at engine init. That collides with
  # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (TMS sanity check
  # raises). We don't actually get useful release_memory_occupation
  # behavior anyway in this setup (LD_PRELOAD hook is stripped above),
  # so switching to the no-op adapter is purely a compatibility win.
  --no-offload-rollout
  # v8: 4 rollout engines existed but only 1 was doing work (16 running
  # + 39 queued on one GPU, 0% util on the other 3). Uni-OPD's default
  # sglang router was hot-spotting requests on one engine. MilesRouter
  # (miles/router/router.py:29) picks the worker URL with the minimum
  # active request count and decrements on finish — exactly what we
  # need to spread 64 generations across 4 engines.
  --use-miles-router
  # Apex's `fused_weight_gradient_mlp_cuda` extension isn't compiled in
  # this venv (we'd need `pip install --global-option=... apex` with
  # CUDA toolchain). Megatron defaults to gradient_accumulation_fusion=True
  # which requires that extension. Disable the fusion to use plain PyTorch
  # grad accumulation — slightly slower kernel path but correctness is
  # identical. Re-enable once apex is installed if perf is needed.
  --no-gradient-accumulation-fusion
)

# v9 trainer-side OOM #2 in fused_vocab_parallel_cross_entropy
# (118.33 GiB PyTorch + 21 GiB sglang = 139/140 GiB ceiling). GPT
# diagnosis Rank 4 lever: the fp32 grad allreduce buffer holds a
# full-precision copy of all 3B parameters ≈ 12 GiB.
# For OPD with PPO clip, entropy_coef=0, kl_loss_coef=0, LR=1e-6,
# the numeric impact of bf16 grad allreduce is small. Default OFF;
# set ALLREDUCE_FP32=1 to restore (and run a short A/B comparing
# grad norm / loss trajectory if you do).
if [ "${ALLREDUCE_FP32:-0}" = "1" ]; then
  MISC_ARGS+=(--accumulate-allreduce-grads-in-fp32)
  echo ">>> ALLREDUCE_FP32=1 (fp32 grad allreduce ON; uses ~12 GiB extra)"
else
  echo ">>> ALLREDUCE_FP32=0 (fp32 grad allreduce OFF; saves ~12 GiB for actor_train peak)"
fi

# Smoke-mode extras (punch list #9). Empty by default for production runs.
# Note: --sglang-disable-cuda-graph moved into the always-on SGLANG_ARGS
# above (was DEBUG_MODE-only). Only the per-rollout debug dump + NCCL
# verbose stay DEBUG_MODE-gated.
DEBUG_ARGS=()
if [ "${DEBUG_MODE}" = "1" ]; then
  DEBUG_ARGS=(
    --save-debug-rollout-data "${EXPERIMENT_DIR}/debug/rollout_${CUR_TIME:-now}/step_{rollout_id}.pt"
  )
  export NCCL_DEBUG=INFO
  echo ">>> DEBUG_MODE=1 (smoke profile: per-rollout dumps, NCCL_DEBUG=INFO)"
fi

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
echo "  ARM                     : ${ARM_TAG}"
echo "  OPD_TEACHER_IMAGE_MODE  : ${OPD_TEACHER_IMAGE_MODE}"
echo "  OPD_RUN_NAME            : ${OPD_RUN_NAME}"
echo "  VD weighting (T2-1)     : ${MLLMOPD_USE_VD_WEIGHTING}  (tau=${MLLMOPD_VD_TAU}, beta=${MLLMOPD_VD_BETA})"
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
    "${TRAIN_ENV_ARGS[@]}" \
    "${TENSORBOARD_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
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
if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "    wandb       : project=${WANDB_PROJECT:-mllmopd}  group=${OPD_RUN_NAME}  (grep TRAIN_LOG_FILE for full URL)"
fi
echo "    diagnostics : ${EXPERIMENT_DIR}/diagnostics"
echo "    train log   : ${TRAIN_LOG_FILE}"
