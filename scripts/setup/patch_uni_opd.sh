#!/usr/bin/env bash
# Idempotent local-only patches to the Uni-OPD submodule, fixing stale
# imports that block T1 training on a fresh checkout.
#
# Uni-OPD's submodule contains a few hard-coded imports that reference
# layout assumptions (Tencent's internal repo + project-specific
# external packages) which don't hold in our clone. The two
# architectural workarounds (src/exps/ shim + src/miles/ wrapper) cover
# most of them, but:
#
#   - margin_shift.py uses `from miles.miles.X import ...` (double
#     prefix). This needs the src/miles/__init__.py wrapper to win the
#     PYTHONPATH race. In Ray-worker subprocesses that race appears to
#     be lost intermittently — workers re-import `miles` from MILES_DIR
#     first and the wrapper never loads. A 2-line in-place patch is
#     both more deterministic and easier to debug.
#
# This script ONLY touches files inside `third_party/Uni-OPD/miles/`.
# Submodules are independent git repos: the patch stays local and is
# not pushed upstream. Re-running after `git submodule update` requires
# re-running this script.
#
# Run once per fresh checkout. Idempotent.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MILES_DIR="${MILES_DIR:-third_party/Uni-OPD/miles}"
if [ ! -d "${MILES_DIR}" ]; then
  echo "ERROR: MILES_DIR=${MILES_DIR} not found (init submodules first)" >&2
  exit 1
fi

# --- Patch 1: margin_shift.py double-prefix imports -----------------
# The two lines that say `from miles.miles.X import ...` only work when
# PYTHONPATH points at Uni-OPD repo root. Strip one `miles.` to make
# them resolve against our PYTHONPATH (which points at miles/).
MS="${MILES_DIR}/Uni_OPD_utils/margin_calibration/margin_shift.py"
if [ ! -f "${MS}" ]; then
  echo "ERROR: ${MS} not found" >&2
  exit 1
fi

if grep -q "from miles\.miles\." "${MS}"; then
  echo ">>> patching ${MS}: miles.miles.X -> miles.X"
  # Use awk for portability (sed -i syntax differs between mac BSD and GNU)
  awk '{ gsub(/from miles\.miles\./, "from miles."); print }' "${MS}" > "${MS}.tmp"
  mv "${MS}.tmp" "${MS}"
  echo "    after:"
  grep -n "^from miles\." "${MS}" | head -5 | sed 's/^/      /'
else
  echo ">>> ${MS}: already patched (no miles.miles.* found)"
fi

# --- Patch 2: actor.py env-inspect instrumentation -----------------
# Adds env-dump + /proc/<pid>/maps scan at the top of actor.py AND at
# the first line of MegatronTrainRayActor.init(). Output goes to
# /tmp/actor_inspect_<pid>.log. Idempotent via sentinel marker.
# Reason: diagnosing why Ray actors keep loading the wrong libnccl
# (NCCL 2.29.7 from system vs venv's 2.27.5) even after launcher-side
# LD_LIBRARY_PATH manipulation.
AC="${MILES_DIR}/miles/backends/megatron_utils/actor.py"
if [ ! -f "${AC}" ]; then
  echo "ERROR: ${AC} not found" >&2
  exit 1
fi

SENTINEL="# === mllmopd actor inspect patch ==="
# --- Patch 3: ray_launcher.py — let LD_PRELOAD / NCCL_DEBUG etc.
#     propagate to actor processes via runtime_env env_vars.
#     The default MILES_RAY_RUNTIME_ENV dict only carries keys it lists,
#     and LD_PRELOAD wasn't among them — so actor processes ended up with
#     ray's own jemalloc preload AND a torch_memory_saver hook that broke
#     NCCL collectives. Append the missing keys so launcher exports
#     reach actors verbatim.
RL="${MILES_DIR}/Uni_OPD_utils/ray_launcher.py"
RL_SENTINEL="# === mllmopd ray_launcher runtime_env patch ==="
if grep -q "${RL_SENTINEL}" "${RL}"; then
  echo ">>> ${RL}: already patched (runtime_env sentinel present)"
else
  echo ">>> patching ${RL}: extend MILES_RAY_RUNTIME_ENV with LD_PRELOAD/NCCL_DEBUG/..."
  export MLLMOPD_PATCH_RL_PATH="${RL}"
  python3 - <<'PY'
import os, sys
path = os.environ["MLLMOPD_PATCH_RL_PATH"]
with open(path, "r") as f:
    src = f.read()

# Insert new env-var keys right before the closing `}` of the env_vars
# dict. We anchor on the line that holds `"https_proxy": "",` — the
# last entry of the current dict in the upstream source.
anchor = '        "https_proxy": "",'
patch = '''        "https_proxy": "",
        # === mllmopd ray_launcher runtime_env patch ===
        "LD_PRELOAD": os.environ.get("LD_PRELOAD", ""),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
        "NCCL_DEBUG_SUBSYS": os.environ.get("NCCL_DEBUG_SUBSYS", ""),
        "TORCH_NCCL_BLOCKING_WAIT": os.environ.get("TORCH_NCCL_BLOCKING_WAIT", ""),
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": os.environ.get("TORCH_NCCL_ASYNC_ERROR_HANDLING", ""),
        # === end mllmopd ray_launcher runtime_env patch ==='''

if anchor not in src:
    sys.exit(f"ERROR: anchor not found in {path!r}; ray_launcher layout changed")

new_src = src.replace(anchor, patch, 1)
with open(path + ".tmp", "w") as f:
    f.write(new_src)
os.replace(path + ".tmp", path)
print("    appended LD_PRELOAD/LD_LIBRARY_PATH/NCCL_DEBUG_SUBSYS/TORCH_NCCL_BLOCKING_WAIT/TORCH_NCCL_ASYNC_ERROR_HANDLING")
PY
fi

if grep -q "${SENTINEL}" "${AC}"; then
  echo ">>> ${AC}: already patched (inspect sentinel present)"
else
  echo ">>> patching ${AC}: inject env-inspect at top + inside .init()"
  # Quoted heredoc (`<<'PY'`) so bash does NO interpolation or backslash
  # processing — the Python source is delivered verbatim. The prior
  # unquoted heredoc converted `\\n` to `\n` in the Python source, which
  # Python then parsed as an actual newline INSIDE the string literals,
  # producing unterminated f-strings in the resulting actor.py.
  export MLLMOPD_PATCH_ACTOR_PATH="${AC}"
  python3 - <<'PY'
import os, sys
path = os.environ["MLLMOPD_PATCH_ACTOR_PATH"]

with open(path, "r") as f:
    src = f.read()

top_block = r'''# === mllmopd actor inspect patch ===
import os as _mllmopd_os
_mllmopd_pid = _mllmopd_os.getpid()
_mllmopd_log = f"/tmp/actor_inspect_{_mllmopd_pid}.log"
try:
    with open(_mllmopd_log, "w") as _f:
        _f.write(f"PID={_mllmopd_pid}\n")
        _f.write(f"LD_LIBRARY_PATH={_mllmopd_os.environ.get('LD_LIBRARY_PATH','UNSET')}\n")
        _f.write(f"LD_PRELOAD={_mllmopd_os.environ.get('LD_PRELOAD','UNSET')}\n")
        _f.write(f"CUDA_VISIBLE_DEVICES={_mllmopd_os.environ.get('CUDA_VISIBLE_DEVICES','UNSET')}\n")
        _f.write(f"PATH={_mllmopd_os.environ.get('PATH','UNSET')}\n")
        _f.write(f"PYTHONPATH={_mllmopd_os.environ.get('PYTHONPATH','UNSET')}\n")
except Exception:
    pass
# === end mllmopd actor inspect patch (top) ===

'''

init_block = r'''        # === mllmopd actor inspect patch (inside init) ===
        import os as _mllmopd_os2
        _mp = _mllmopd_os2.getpid()
        try:
            with open(f"/tmp/actor_inspect_{_mp}.log", "a") as _f:
                _f.write("--- inside MegatronTrainRayActor.init() ---\n")
                try:
                    with open(f"/proc/{_mp}/maps") as _m:
                        _seen = set()
                        for _line in _m:
                            _low = _line.lower()
                            if any(_k in _low for _k in ["nccl", "libcuda.so", "cudart", "compat"]):
                                _parts = _line.strip().split()
                                _path = _parts[-1] if _parts else ""
                                if _path and _path.startswith("/") and _path not in _seen:
                                    _seen.add(_path)
                                    _f.write(_path + "\n")
                except Exception as _e:
                    _f.write(f"maps err: {_e}\n")
                import torch as _t
                _f.write(f"torch.cuda.nccl.version() [COMPILE-TIME]={_t.cuda.nccl.version()}\n")
                _f.write(f"torch.version.cuda={_t.version.cuda}\n")
                _f.write(f"torch.cuda.device_count()={_t.cuda.device_count()}\n")
                # The KEY diagnostic — ncclGetVersion via ctypes gives the
                # actually-loaded NCCL's runtime version, vs torch.cuda.nccl.version()
                # which is torch's compile-time NCCL_VERSION_CODE macro.
                # Mismatch between these two = dynamic linker found a foreign
                # libnccl ahead of the venv's bundled one.
                try:
                    import ctypes as _ct
                    _v = _ct.c_int()
                    _lib = _ct.CDLL("libnccl.so.2")
                    _rc = _lib.ncclGetVersion(_ct.byref(_v))
                    _maj = _v.value // 10000
                    _min = (_v.value % 10000) // 100
                    _pat = _v.value % 100
                    _f.write(f"ctypes_ncclGetVersion [RUNTIME] rc={_rc} raw={_v.value} decoded={_maj}.{_min}.{_pat}\n")
                except Exception as _e:
                    _f.write(f"ctypes_ncclGetVersion failed: {_e}\n")
                # pip metadata for nvidia-nccl-cu12 — does pip think we have 2.27 or 2.29?
                try:
                    import importlib.metadata as _md
                    _f.write(f"pip nvidia-nccl-cu12={_md.version('nvidia-nccl-cu12')}\n")
                except Exception as _e:
                    _f.write(f"pip nvidia-nccl-cu12 unknown: {_e}\n")
                try:
                    _t.cuda.init()
                    _f.write(f"torch.cuda.current_device()={_t.cuda.current_device()}\n")
                    _x = _t.zeros(1).cuda()
                    _f.write(f"cuda ok, device={_t.cuda.get_device_name(0)}\n")
                except Exception as _e:
                    _f.write(f"cuda init failed: {type(_e).__name__}: {_e}\n")
                try:
                    import torch.distributed as _td
                    _f.write(f"dist.is_initialized()={_td.is_initialized()}\n")
                    if _td.is_initialized():
                        _f.write(f"dist.world_size={_td.get_world_size()} rank={_td.get_rank()} backend={_td.get_backend()}\n")
                except Exception as _e:
                    _f.write(f"dist probe failed: {type(_e).__name__}: {_e}\n")
        except Exception:
            pass
        # === end mllmopd actor inspect patch (inside init) ===
'''

new_src = top_block + src

target = "        monkey_patch_torch_dist()"
if target not in new_src:
    sys.exit(f"ERROR: could not find anchor '{target.strip()}' in {path}")
new_src = new_src.replace(target, init_block + target, 1)

with open(path + ".tmp", "w") as f:
    f.write(new_src)
os.replace(path + ".tmp", path)
print("    inserted both patch blocks")
PY
  echo "    after (first 5 lines):"
  head -5 "${AC}" | sed 's/^/      /'
fi

# --- Patch 4: sglang default_weight_loader diagnostic ---------------
# qwen2_5_vl.py:851 → weight_utils.py:951 (`param.data.copy_(loaded_weight)`)
# raises `torch.AcceleratorError: CUDA error: invalid argument` on
# weight update_from_tensor. With CUDA_LAUNCH_BLOCKING=1 in the
# launcher, the actual failing call should be pinpointed, but we
# still need to see WHICH parameter (name + shape + dtype + device)
# is the culprit. Wrap the copy_ to log full tensor metadata on
# failure, then re-raise.
WU="${MILES_DIR%/miles}/sglang/python/sglang/srt/model_loader/weight_utils.py"
if [ ! -f "${WU}" ]; then
  # Fall back to absolute path discovery via the venv.
  WU="$(cd "$(git rev-parse --show-toplevel)" && \
        echo "$(pwd)/third_party/sglang/python/sglang/srt/model_loader/weight_utils.py")"
fi
WU_SENTINEL="# === mllmopd default_weight_loader diag ==="
if [ -f "${WU}" ]; then
  if grep -q "${WU_SENTINEL}" "${WU}"; then
    echo ">>> ${WU}: already patched (diag sentinel present)"
  else
    echo ">>> patching ${WU}: wrap default_weight_loader with tensor-info diag"
    export MLLMOPD_PATCH_WU_PATH="${WU}"
    python3 - <<'PY'
import os, sys
path = os.environ["MLLMOPD_PATCH_WU_PATH"]
with open(path) as f:
    src = f.read()

anchor = '            param.data.copy_(loaded_weight)\n    except Exception:'
# Two fixes in one wrap:
#  (1) device coercion: smoke #20 diag showed loaded_weight lands on
#      cuda:0 after Ray IPC while param lives on cuda:N (sglang engine's
#      assigned GPU). PyTorch's copy_ usually handles cross-device, but
#      in our overlay-FS / Ray colocate setup it raises CUDA "invalid
#      argument" — likely because the source pointer was allocated in
#      the train actor's CUDA context and isn't valid in the engine's.
#      Move loaded_weight to param.device first (via CPU stage if P2P
#      isn't enabled; pytorch handles fallback). This is a 4 KB tensor
#      so overhead is negligible.
#  (2) on failure, dump tensor metadata + re-raise so we can still see
#      WHICH weight failed if (1) wasn't enough.
patch = '''            try:
                if loaded_weight.device != param.device:
                    loaded_weight = loaded_weight.to(param.device, non_blocking=False)
                param.data.copy_(loaded_weight)
            except Exception as _e:
                # === mllmopd default_weight_loader diag ===
                import sys as _sys
                _info = (
                    f"\\n[mllmopd diag] default_weight_loader copy_ FAILED\\n"
                    f"  param  : shape={tuple(param.size())} dtype={param.dtype} "
                    f"device={param.device} stride={param.stride()} contig={param.is_contiguous()}\\n"
                    f"  loaded : shape={tuple(loaded_weight.size())} dtype={loaded_weight.dtype} "
                    f"device={loaded_weight.device} stride={loaded_weight.stride()} contig={loaded_weight.is_contiguous()}\\n"
                    f"  err    : {type(_e).__name__}: {_e}\\n"
                )
                print(_info, file=_sys.stderr, flush=True)
                raise
                # === end mllmopd default_weight_loader diag ===
    except Exception:'''
if anchor not in src:
    sys.exit(f"ERROR: anchor not found in {path!r} — sglang source may have moved")
new_src = src.replace(anchor, patch, 1)
with open(path + ".tmp", "w") as f:
    f.write(new_src)
os.replace(path + ".tmp", path)
print("    inserted device-coercion fix + diag wrap")
PY
  fi
else
  echo ">>> ${WU}: skipped (file not found — sglang submodule layout changed?)"
fi

# --- Patch 5: Uni-OPD sender — UUID-aware CUDA tensor serialization ---
# GPT review round 3 identified the real root cause of the sglang
# "invalid argument" weight-copy failure: Uni-OPD's
# _send_to_colocated_engine() in update_weight_from_tensor.py serializes
# the flattened bucket via Python ForkingPickler WITHOUT first applying
# sglang's monkey_patch_torch_reductions(). Default ForkingPickler stores
# the tensor's LOCAL device INDEX (cuda:0 on the train actor's single-GPU
# CVD view); the receiver sglang engine has full CVD view and rebuilds
# everything on its own cuda:0, but param N lives on cuda:N. Hence the
# device mismatch.
# monkey_patch_torch_reductions makes pickle store DEVICE UUID instead,
# and the receiver remaps UUID → its local cuda:N. Sglang's TP worker
# already calls it before deserialize; we just need to also call it
# before serialize on the sender side.
UW="${MILES_DIR}/miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py"
UW_SENTINEL="# === mllmopd uuid-aware tensor serialization patch ==="
if [ -f "${UW}" ]; then
  if grep -q "${UW_SENTINEL}" "${UW}"; then
    echo ">>> ${UW}: already patched (uuid-aware sentinel present)"
  else
    echo ">>> patching ${UW}: inject monkey_patch_torch_reductions() in sender"
    export MLLMOPD_PATCH_UW_PATH="${UW}"
    python3 - <<'PY'
import os, sys
path = os.environ["MLLMOPD_PATCH_UW_PATH"]
with open(path) as f:
    src = f.read()
anchor = "    is_lora = lora_config is not None\n    long_live_tensors = []"
patch = '''    # === mllmopd uuid-aware tensor serialization patch ===
    # Train actor sees single-GPU CVD so every local tensor is cuda:0.
    # SGLang engines retain full rollout CVD (NOSET on rollout side per
    # Uni-OPD design) and place engine N's params on cuda:N. Without
    # this, ForkingPickler serializes the local device INDEX 0; the
    # receiver rebuilds every bucket on its own cuda:0 even though
    # param lives on cuda:N. SGLang's monkey patch makes pickle store
    # device UUID and the receiver remaps to its local device.
    from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
    monkey_patch_torch_reductions()
    # === end mllmopd uuid-aware patch ===

    is_lora = lora_config is not None
    long_live_tensors = []'''
if anchor not in src:
    sys.exit(f"ERROR: anchor not found in {path!r} — Uni-OPD source may have moved")
new_src = src.replace(anchor, patch, 1)
with open(path + ".tmp", "w") as f:
    f.write(new_src)
os.replace(path + ".tmp", path)
print("    inserted monkey_patch_torch_reductions() call at start of _send_to_colocated_engine")
PY
  fi
else
  echo ">>> ${UW}: skipped (file not found)"
fi

# --- Patch 6: sglang receiver — flattened bucket device normalization ---
# Belt-and-suspenders companion to P5 (the sender UUID patch). SGLang's
# _update_weights_from_flattened_bucket reconstructs tensors from a
# shared flattened storage; if that flattened storage lands on the
# wrong device, every reconstructed view inherits it. The
# non-flattened update_weights_from_tensor path already does
# _unwrap_tensor(..., device=current) → tensor.to(device), but the
# flattened branch returns BEFORE that step. Normalize the whole
# flattened bucket to (self.device, self.gpu_id) once, before
# bucket.reconstruct_tensors().
MR="$(cd "$(git rev-parse --show-toplevel)" && \
      echo "$(pwd)/third_party/sglang/python/sglang/srt/model_executor/model_runner.py")"
MR_SENTINEL="# === mllmopd flattened-bucket device normalize ==="
if [ -f "${MR}" ]; then
  if grep -q "${MR_SENTINEL}" "${MR}"; then
    echo ">>> ${MR}: already patched (flattened-bucket sentinel present)"
  else
    echo ">>> patching ${MR}: normalize flattened bucket to self.gpu_id"
    export MLLMOPD_PATCH_MR_PATH="${MR}"
    python3 - <<'PY'
import os, sys
path = os.environ["MLLMOPD_PATCH_MR_PATH"]
with open(path) as f:
    src = f.read()
anchor = '''        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]'''
patch = '''        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # === mllmopd flattened-bucket device normalize ===
        # P1 from GPT review round 3. Mirror the non-flattened path's
        # _unwrap_tensor(..., device=current) step which the flattened
        # branch skips. Do this once for the whole bucket so reconstructed
        # views inherit the right device.
        import torch as _t
        if self.device != "cpu":
            _target = _t.device(self.device, self.gpu_id)
            if flattened_tensor.device != _target:
                print(
                    f"[mllmopd weight-sync receiver] gpu_id={self.gpu_id} "
                    f"bucket.device={flattened_tensor.device} target={_target} "
                    f"-> moving",
                    flush=True,
                )
                flattened_tensor = flattened_tensor.to(_target, non_blocking=False)
        # === end mllmopd flattened-bucket device normalize ==='''
if anchor not in src:
    sys.exit(f"ERROR: anchor not found in {path!r} — sglang source may have moved")
new_src = src.replace(anchor, patch, 1)
with open(path + ".tmp", "w") as f:
    f.write(new_src)
os.replace(path + ".tmp", path)
print("    inserted flattened bucket device normalize before reconstruct_tensors")
PY
  fi
else
  echo ">>> ${MR}: skipped (file not found)"
fi

echo
echo ">>> patch_uni_opd done. Re-run after \`git submodule update\`."
echo
echo "Inspect after smoke:"
echo "    ls -la /tmp/actor_inspect_*.log"
echo "    cat /tmp/actor_inspect_*.log"
echo "    # weight-sync receiver path:"
echo "    grep '\\[mllmopd weight-sync' \${MLLMOPD_RUNS}/t1_smoke_*/logs/train_*.log"
echo "    # default_weight_loader fallback diag (should not fire if P5/P6 work):"
echo "    grep -A3 '\\[mllmopd diag\\]' \${MLLMOPD_RUNS}/t1_smoke_*/logs/train_*.log"
