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

# --- Patch 7: skip entropy compute when entropy_coef==0 (GPT diag Rank 3) ---
# policy_loss_function unconditionally calls get_log_probs_and_entropy with
# with_entropy=True and computes entropy_loss = sum_of_sample_mean(entropy),
# even when args.entropy_coef==0.0. The result feeds
# loss = pg_loss - 0 * entropy_loss — i.e. multiplied to nothing — but the
# entropy compute still allocates a vocab-sized fp32 tensor via
# _VocabParallelEntropy.forward (logits_max + normalized_vocab_parallel_logits
# subtraction). At s10≈1900, vocab=151936, that's ~1.16 GiB peak that drove
# T1-2 OOM #5 (docs/gpt-diagnosis-2026-05-20-t1-oom.md). Gate the compute on
# args.entropy_coef.
LOSS="${MILES_DIR}/miles/backends/training_utils/loss.py"
LOSS_SENTINEL="# === mllmopd entropy-zero skip patch ==="
if [ -f "${LOSS}" ]; then
  if grep -q "${LOSS_SENTINEL}" "${LOSS}"; then
    echo ">>> ${LOSS}: already patched (entropy-zero skip sentinel present)"
  else
    echo ">>> patching ${LOSS}: skip entropy compute when entropy_coef==0"
    export MLLMOPD_PATCH_LOSS_PATH="${LOSS}"
    python3 - <<'PY'
import os, sys
path = os.environ["MLLMOPD_PATCH_LOSS_PATH"]
with open(path) as f:
    src = f.read()

# Patch A: `with_entropy=True` → conditional on args.entropy_coef
anchor_a = "        with_entropy=True,\n"
patch_a = "        with_entropy=args.entropy_coef != 0.0,  # === mllmopd entropy-zero skip patch ===\n"

# Patch B: gate the entropy_loss block on args.entropy_coef
anchor_b = """    # entropy loss
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)
"""
patch_b = """    # entropy loss
    # === mllmopd entropy-zero skip patch ===
    if args.entropy_coef != 0.0:
        entropy = log_probs_and_entropy["entropy"]
        entropy = torch.cat(entropy, dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
    else:
        entropy_loss = pg_loss.new_zeros(())
    # === end mllmopd entropy-zero skip patch ===
"""

if anchor_a not in src:
    sys.exit(f"ERROR: anchor A (with_entropy=True) not found in {path!r}")
if anchor_b not in src:
    sys.exit(f"ERROR: anchor B (entropy loss block) not found in {path!r}")

new_src = src.replace(anchor_a, patch_a, 1).replace(anchor_b, patch_b, 1)

with open(path + ".tmp", "w") as f:
    f.write(new_src)
os.replace(path + ".tmp", path)
print("    inserted entropy-zero skip at policy_loss_function")
PY
  fi
else
  echo ">>> ${LOSS}: skipped (file not found)"
fi

# --- Patch 8: MilesRouter FastAPI lifespan migration -----------------------
# FastAPI 0.106+ removed `add_event_handler("startup", ...)`. Uni-OPD's
# MilesRouter.__init__ at miles/router/router.py:36 still calls it, which
# crashes the RolloutManager creation task with
#   AttributeError: 'FastAPI' object has no attribute 'add_event_handler'
# when --use-miles-router is passed. Migrate to the lifespan context
# manager (FastAPI's current API). Functionally identical: schedule the
# health-check task before uvicorn starts serving.
ROUTER="${MILES_DIR}/miles/router/router.py"
ROUTER_SENTINEL="# === mllmopd FastAPI lifespan patch ==="
if [ -f "${ROUTER}" ]; then
  if grep -q "${ROUTER_SENTINEL}" "${ROUTER}"; then
    echo ">>> ${ROUTER}: already patched (FastAPI lifespan sentinel present)"
  else
    echo ">>> patching ${ROUTER}: migrate add_event_handler → lifespan"
    export MLLMOPD_PATCH_ROUTER_PATH="${ROUTER}"
    python3 - <<'PY'
import os, sys
path = os.environ["MLLMOPD_PATCH_ROUTER_PATH"]
with open(path) as f:
    src = f.read()

anchor = '''        self.app = FastAPI()
        self.app.add_event_handler("startup", self._start_background_health_check)'''

patch = '''        # === mllmopd FastAPI lifespan patch ===
        # FastAPI 0.106+ removed add_event_handler; use the lifespan
        # context manager instead. _start_background_health_check just
        # schedules a long-running asyncio task, so awaiting it before
        # yield is equivalent to the old "startup" hook.
        from contextlib import asynccontextmanager as _asynccm
        _self_ref = self
        @_asynccm
        async def _lifespan(_app):
            await _self_ref._start_background_health_check()
            yield
        self.app = FastAPI(lifespan=_lifespan)
        # === end mllmopd FastAPI lifespan patch ==='''

if anchor not in src:
    sys.exit(f"ERROR: anchor not found in {path!r} — router source may have moved")

new_src = src.replace(anchor, patch, 1)

with open(path + ".tmp", "w") as f:
    f.write(new_src)
os.replace(path + ".tmp", path)
print("    migrated MilesRouter.__init__ to FastAPI lifespan")
PY
  fi
else
  echo ">>> ${ROUTER}: skipped (file not found)"
fi

# --- Patch 9: memsnap pre-CE dump in policy_loss_function -----------------
# T1-2 OOM v10 stuck at trainer alloc ~118 GiB with ~83 GiB unaccounted
# (docs/handoff-2026-05-20-v10-stuck.md, docs/gpt-reply-2026-05-20-v10-update.md).
# GPT's leading hypothesis is dynamic-packed microbatches at
# --max-tokens-per-gpu pushing fp32 [1, T, V] logits to ~9 GiB, multiplied
# by backward-graph temporaries. To confirm, dump a torch.cuda memory
# snapshot RIGHT BEFORE get_log_probs_and_entropy(...) — the same call
# site that OOMs deterministically across v9/v10. Sentinel-gated so this
# is a one-shot patch; runs only when MLLMOPD_MEMSNAP=1 in env.
# Anchor is `max_seq_lens = batch.get(...)` + blank + the CE call header;
# this is unique to policy_loss_function and is NOT touched by P7's
# `with_entropy=...` edit (P7 modifies a downstream line), so P7/P9 order
# does not matter.
LOSS_MEMSNAP_SENTINEL="# === mllmopd memsnap pre-CE patch ==="
if [ -f "${LOSS}" ]; then
  if grep -q "${LOSS_MEMSNAP_SENTINEL}" "${LOSS}"; then
    echo ">>> ${LOSS}: already patched (memsnap pre-CE sentinel present)"
  else
    echo ">>> patching ${LOSS}: inject memsnap pre-CE dump in policy_loss_function"
    export MLLMOPD_PATCH_LOSS_MEMSNAP_PATH="${LOSS}"
    python3 - <<'PY'
import os, sys
path = os.environ["MLLMOPD_PATCH_LOSS_MEMSNAP_PATH"]
with open(path) as f:
    src = f.read()

anchor = '''    max_seq_lens = batch.get("max_seq_lens", None)

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
'''

patch = '''    max_seq_lens = batch.get("max_seq_lens", None)

    # === mllmopd memsnap pre-CE patch ===
    try:
        from mllmopd.training.memsnap import (
            dump_memory_snapshot as _mllmopd_dump_mem,
        )
        _mllmopd_dump_mem(
            "pre_get_log_probs_and_entropy",
            logits_shape=tuple(logits.shape),
            logits_dtype=str(logits.dtype),
        )
        print(
            f"[mllmopd pre_ce] logits={tuple(logits.shape)} "
            f"dtype={logits.dtype} "
            f"total_lens={total_lengths} "
            f"resp_lens={response_lengths} "
            f"alloc_gib={torch.cuda.memory_allocated()/2**30:.2f} "
            f"reserved_gib={torch.cuda.memory_reserved()/2**30:.2f}",
            flush=True,
        )
    except Exception as _mllmopd_e:
        print(
            f"[mllmopd pre_ce] dump failed: "
            f"{type(_mllmopd_e).__name__}: {_mllmopd_e}",
            flush=True,
        )
    # === end mllmopd memsnap pre-CE patch ===

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
'''

if anchor not in src:
    sys.exit(
        f"ERROR: anchor not found in {path!r} — policy_loss_function may have moved"
    )

new_src = src.replace(anchor, patch, 1)

with open(path + ".tmp", "w") as f:
    f.write(new_src)
os.replace(path + ".tmp", path)
print("    inserted memsnap pre-CE dump before get_log_probs_and_entropy")
PY
  fi
else
  echo ">>> ${LOSS}: skipped (file not found for memsnap patch)"
fi

# --- Patch P10: in-place logits.div_(temperature) in get_responses -----
# v1 trainer OOM root cause beyond max-tokens-per-gpu: get_responses() at
# loss.py:81 does `logits = logits.div(args.rollout_temperature)` which
# creates a NEW fp32 tensor instead of scaling in place. For a packed
# [T~=6800, V=151936] fp32 tensor that's an extra ~4 GiB at peak — over
# many CE chunks plus backward state, it pushed v1 step ~148 over the
# 140 GiB H800 ceiling. In-place div_() removes the copy entirely with
# zero behavior change (the original tensor isn't referenced after this
# point — `logits` is rebound to the result of div_, slices below take
# views of it). 1-character patch (`.div(` → `.div_(`). Sentinel-
# idempotent via a comment marker.
LOSS_FILE="${MILES_DIR}/miles/backends/training_utils/loss.py"
LOSS_INPLACE_SENTINEL="# === mllmopd P10 inplace div ==="
if [ ! -f "${LOSS_FILE}" ]; then
  echo ">>> P10: ${LOSS_FILE} not found, skipping"
elif grep -q "${LOSS_INPLACE_SENTINEL}" "${LOSS_FILE}"; then
  echo ">>> P10: ${LOSS_FILE} already patched (in-place div sentinel present)"
else
  echo ">>> patching ${LOSS_FILE}: get_responses logits.div → div_ (in-place)"
  export MLLMOPD_PATCH_LOSS_INPLACE_PATH="${LOSS_FILE}"
  python3 - <<'PY'
import os, sys
path = os.environ["MLLMOPD_PATCH_LOSS_INPLACE_PATH"]
with open(path) as f:
    src = f.read()

# Match the exact line. If miles refactors get_responses, this assert
# fires loudly instead of silently doing nothing.
anchor = "    logits = logits.div(args.rollout_temperature)"
replacement = (
    "    # === mllmopd P10 inplace div ===\n"
    "    # In-place div_() avoids the ~4 GiB fp32 copy of [T, V] logits\n"
    "    # that v1 step_148 OOM'd on. See docs/handoff-2026-05-20-oom-resolved\n"
    "    # and t1_v1 retrain diary. Behavior identical: the original\n"
    "    # tensor isn't referenced after this point.\n"
    "    logits = logits.div_(args.rollout_temperature)"
)
if anchor not in src:
    sys.exit(f"ERROR: anchor for P10 not found in {path!r}; get_responses may have moved")
new_src = src.replace(anchor, replacement, 1)
with open(path + ".tmp", "w") as f:
    f.write(new_src)
os.replace(path + ".tmp", path)
print("    in-place div_() patch applied")
PY
fi

# --- Patch P11: rollout.py — collect + partition teacher_vd_weights ----
# T2-1 method tier: plumb the per-token VD weight tensor produced by
# mllmopd.training.opd_diagnostics_hook through to rollout_data["teacher_vd_weights"]
# so the trainer-side OPD branch (loss.py P13) can multiply it into the
# OPD advantage. Two anchors: (A) collect block in _convert_samples_to_train_data,
# (B) partition keys list in _split_train_data_by_dp.
# T1-2/T1-3 baselines never set sample.teacher_vd_weights → both anchors
# are no-ops at runtime. See docs/t2_1_design.md.
ROLLOUT="${MILES_DIR}/miles/ray/rollout.py"
ROLLOUT_SENTINEL="# === mllmopd P11 vd weights collect ==="
if [ -f "${ROLLOUT}" ]; then
  if grep -q "${ROLLOUT_SENTINEL}" "${ROLLOUT}"; then
    echo ">>> ${ROLLOUT}: already patched (P11 vd-weights sentinel present)"
  else
    echo ">>> patching ${ROLLOUT}: collect + partition teacher_vd_weights"
    export MLLMOPD_PATCH_ROLLOUT_PATH="${ROLLOUT}"
    python3 - <<'PY'
import os, sys
path = os.environ["MLLMOPD_PATCH_ROLLOUT_PATH"]
with open(path) as f:
    src = f.read()

# Anchor A: collect block. The two-line response_correct collect plus
# the blank line and `return train_data` are unique to this method.
anchor_a = '''            if "response_correct" in samples[0].__dict__:
                train_data["response_correct"] = [sample.response_correct for sample in samples]

        return train_data'''
patch_a = '''            if "response_correct" in samples[0].__dict__:
                train_data["response_correct"] = [sample.response_correct for sample in samples]
            # === mllmopd P11 vd weights collect ===
            # Per-token VD weights computed by
            # mllmopd.training.opd_diagnostics_hook when
            # MLLMOPD_USE_VD_WEIGHTING=1 (see docs/t2_1_design.md). The
            # attribute is absent in T1 baselines so this branch silently
            # skips. Loss-side application is in loss.py P13.
            if "teacher_vd_weights" in samples[0].__dict__:
                train_data["teacher_vd_weights"] = [sample.teacher_vd_weights for sample in samples]
            # === end mllmopd P11 vd weights collect ===

        return train_data'''

# Anchor B: partition keys list — add teacher_vd_weights right after
# response_correct in the keys-to-partition list.
anchor_b = '''                "teacher_log_probs",
                "response_correct",
            ]:'''
patch_b = '''                "teacher_log_probs",
                "response_correct",
                "teacher_vd_weights",  # === mllmopd P11 vd partition key ===
            ]:'''

if anchor_a not in src:
    sys.exit(f"ERROR: P11 anchor A not found in {path!r} — _convert_samples_to_train_data may have moved")
if anchor_b not in src:
    sys.exit(f"ERROR: P11 anchor B not found in {path!r} — partition keys list may have moved")

new_src = src.replace(anchor_a, patch_a, 1).replace(anchor_b, patch_b, 1)
with open(path + ".tmp", "w") as f:
    f.write(new_src)
os.replace(path + ".tmp", path)
print("    P11 applied (collect block + partition key entry)")
PY
  fi
else
  echo ">>> ${ROLLOUT}: skipped (file not found for P11)"
fi

# --- Patch P12: data.py — cuda conversion for teacher_vd_weights -------
# Mirror the existing teacher_log_probs cuda-move block for VD weights
# when present. Skipped at runtime on T1-2/T1-3 baselines (no
# teacher_vd_weights key in rollout_data).
DATA_FILE="${MILES_DIR}/miles/backends/training_utils/data.py"
DATA_SENTINEL="# === mllmopd P12 vd weights cuda ==="
if [ -f "${DATA_FILE}" ]; then
  if grep -q "${DATA_SENTINEL}" "${DATA_FILE}"; then
    echo ">>> ${DATA_FILE}: already patched (P12 vd-cuda sentinel present)"
  else
    echo ">>> patching ${DATA_FILE}: cuda conversion for teacher_vd_weights"
    export MLLMOPD_PATCH_DATA_PATH="${DATA_FILE}"
    python3 - <<'PY'
import os, sys
path = os.environ["MLLMOPD_PATCH_DATA_PATH"]
with open(path) as f:
    src = f.read()

# Anchor on the end of the teacher_log_probs cuda block + the blank
# line before the next `if "rollout_routed_experts"` block.
anchor = '''                for log_prob in rollout_data["teacher_log_probs"]
            ]

    if "rollout_routed_experts" in rollout_data:'''
patch = '''                for log_prob in rollout_data["teacher_log_probs"]
            ]
        # === mllmopd P12 vd weights cuda ===
        # Same cuda move for VD weights when present (absent on T1
        # baselines; loss.py P13 OPD branch guards with .get(...)).
        if "teacher_vd_weights" in rollout_data:
            rollout_data["teacher_vd_weights"] = [
                torch.tensor(w, device=torch.cuda.current_device(), dtype=torch.float32)
                if not isinstance(w, torch.Tensor)
                else w.to(device=torch.cuda.current_device(), dtype=torch.float32)
                for w in rollout_data["teacher_vd_weights"]
            ]
        # === end mllmopd P12 vd weights cuda ===

    if "rollout_routed_experts" in rollout_data:'''
if anchor not in src:
    sys.exit(f"ERROR: P12 anchor not found in {path!r} — data.py teacher_log_probs block may have moved")
new_src = src.replace(anchor, patch, 1)
with open(path + ".tmp", "w") as f:
    f.write(new_src)
os.replace(path + ".tmp", path)
print("    P12 applied (teacher_vd_weights cuda conversion)")
PY
  fi
else
  echo ">>> ${DATA_FILE}: skipped (file not found for P12)"
fi

# --- Patch P13: loss.py — multiply OPD advantage by VD weight ----------
# Two-part patch inside the on_policy_distillation branch of
# compute_advantages_and_returns:
#   (A) read teacher_vd_weights from rollout_data + slice to response length
#   (B) inside the per-sample loop, multiply adv by vd_w when present
# Both gated by `rollout_data.get(...)`; absent on T1-2/T1-3 baselines.
LOSS_VD_SENTINEL="# === mllmopd P13 vd weights apply ==="
if [ -f "${LOSS}" ]; then
  if grep -q "${LOSS_VD_SENTINEL}" "${LOSS}"; then
    echo ">>> ${LOSS}: already patched (P13 vd-apply sentinel present)"
  else
    echo ">>> patching ${LOSS}: read + apply teacher_vd_weights in OPD branch"
    export MLLMOPD_PATCH_LOSS_VD_PATH="${LOSS}"
    python3 - <<'PY'
import os, sys
path = os.environ["MLLMOPD_PATCH_LOSS_VD_PATH"]
with open(path) as f:
    src = f.read()

# Anchor A: insert vd_weights reader between the teacher_log_probs slice
# and the "Reverse KL divergence" comment. Captures the closing `]` of
# the list-comp + the blank line + the comment header — unique anchor.
anchor_a = '''        teacher_log_probs: list[torch.Tensor] = [
            t_log_prob[-response_length:]
            for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
        ]
        # Reverse KL divergence 正值表示教师比学生"更倾向"于选择该 token，即学生需要向教师靠拢'''
patch_a = '''        teacher_log_probs: list[torch.Tensor] = [
            t_log_prob[-response_length:]
            for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
        ]
        # === mllmopd P13 vd weights apply ===
        # Per-token VD weights computed by mllmopd.training.opd_diagnostics_hook
        # (PGPO-style; see mllmopd/training/vd_weighting.py +
        # docs/t2_1_design.md). Absent on T1-2/T1-3 baselines so this is a
        # no-op (None ⇒ unit weight) when MLLMOPD_USE_VD_WEIGHTING is unset.
        teacher_vd_weights_list: list[torch.Tensor] | None = rollout_data.get("teacher_vd_weights")
        if teacher_vd_weights_list is not None:
            teacher_vd_weights_list = [
                w.to(device=device) for w in teacher_vd_weights_list
            ]
            teacher_vd_weights_list = [
                w[-response_length:]
                for w, response_length in zip(teacher_vd_weights_list, response_lengths, strict=False)
            ]
        # === end mllmopd P13 vd weights apply (reader) ===

        # Reverse KL divergence 正值表示教师比学生"更倾向"于选择该 token，即学生需要向教师靠拢'''

# Anchor B: replace the inner-loop adv assignment block. The original
# closes with `advantages.append(adv)`. The patched version inserts the
# VD multiplier between the existing `adv: torch.Tensor = ...` line and
# the append.
anchor_b = '''                teacher_valid_mask = (t_logps != TEACHER_LOGP_FAILED_SENTINEL).float()
                adv: torch.Tensor = (t_logps - s_logps) * teacher_valid_mask  # 失败位置直接置 0
            advantages.append(adv)'''
patch_b = '''                teacher_valid_mask = (t_logps != TEACHER_LOGP_FAILED_SENTINEL).float()
                adv: torch.Tensor = (t_logps - s_logps) * teacher_valid_mask  # 失败位置直接置 0
                # === mllmopd P13 vd weights apply (multiplier) ===
                if teacher_vd_weights_list is not None:
                    vd_w = teacher_vd_weights_list[i]
                    if vd_w.shape[0] == adv.shape[0]:
                        adv = adv * vd_w
                    else:
                        logger.warning(
                            f"[OPD/VD] sample {i}: vd_weight length {vd_w.shape[0]} "
                            f"!= advantage length {adv.shape[0]}; skipping weight."
                        )
                # === end mllmopd P13 vd weights apply (multiplier) ===
            advantages.append(adv)'''

if anchor_a not in src:
    sys.exit(f"ERROR: P13 anchor A (vd reader) not found in {path!r} — OPD branch may have moved")
if anchor_b not in src:
    sys.exit(f"ERROR: P13 anchor B (vd multiplier) not found in {path!r} — OPD inner loop may have moved")

new_src = src.replace(anchor_a, patch_a, 1).replace(anchor_b, patch_b, 1)
with open(path + ".tmp", "w") as f:
    f.write(new_src)
os.replace(path + ".tmp", path)
print("    P13 applied (vd reader + vd multiplier)")
PY
  fi
else
  echo ">>> ${LOSS}: skipped (file not found for P13)"
fi

# --- Patch P14: external rollout placement group + engine actor -----
# Cross-box T2-1: rollout sglang servers run on Box 1 as standalone HTTP
# (--rollout-external). Their Ray "wrapper" actors run locally on the
# trainer (Box 2) and only proxy HTTP — they don't need a GPU bundle.
#
# Without this patch, miles' placement_group.create_placement_groups
# allocates GPU bundles for BOTH trainer ranks AND rollout engines
# (line ~99: `num_gpus = actor_num_nodes*actor_num_gpus_per_node + rollout_num_gpus`).
# On Box 2's 8-GPU Ray cluster with actor=8, asking for 9 (=8+1) or 15
# (=8+7) bundles hangs forever waiting for non-existent GPUs.
#
# Fix is two-part:
#   A) placement_group.py: in external mode skip rollout GPU bundles,
#      so the placement group only sizes to actor GPUs.
#   B) rollout.py:init_rollout_engines: in external mode spawn engine
#      wrappers with num_gpus=0 and scheduling_strategy=None,
#      base_gpu_id=None (forcing _compute_server_args to fall back to
#      get_base_gpu_id(args, rank) which yields the same values we
#      configured on start_rollout_servers.sh).
PG_FILE="${MILES_DIR}/miles/ray/placement_group.py"
PG_SENTINEL="# === mllmopd P14 external rollout placement ==="
if [ -f "${PG_FILE}" ]; then
  if grep -q "${PG_SENTINEL}" "${PG_FILE}"; then
    echo ">>> ${PG_FILE}: already patched (P14 placement sentinel present)"
  else
    echo ">>> patching ${PG_FILE}: skip rollout GPU bundles in external mode"
    export MLLMOPD_PATCH_PG_PATH="${PG_FILE}"
    python3 - <<'PY'
import os, sys
path = os.environ["MLLMOPD_PATCH_PG_PATH"]
with open(path) as f:
    src = f.read()
anchor = '''    else:
        num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
        rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
        if args.use_critic:'''
patch = '''    else:
        # === mllmopd P14 external rollout placement ===
        # In rollout_external mode, rollout engines are non-Ray HTTP servers
        # on Box 1. Their Ray wrappers (init_rollout_engines) run locally
        # with num_gpus=0 and need no GPU bundle. Skipping the rollout
        # slice lets the whole placement group fit on the trainer-only
        # Ray cluster (otherwise asks for 9..15 bundles on 8-GPU Box 2).
        if getattr(args, "rollout_external", False):
            num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
            rollout_offset = num_gpus
        else:
            num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node + args.rollout_num_gpus
            rollout_offset = args.actor_num_nodes * args.actor_num_gpus_per_node
        # === end mllmopd P14 external rollout placement ===
        if args.use_critic:'''
if anchor not in src:
    sys.exit(f"ERROR: P14 placement anchor not found in {path!r} — create_placement_groups may have moved")
new_src = src.replace(anchor, patch, 1)
with open(path + ".tmp", "w") as f:
    f.write(new_src)
os.replace(path + ".tmp", path)
print("    P14 placement_group.py applied")
PY
  fi
else
  echo ">>> ${PG_FILE}: skipped (file not found for P14 placement)"
fi

# P14 part B: rollout.py:init_rollout_engines engine-actor spawn.
RO_FILE="${MILES_DIR}/miles/ray/rollout.py"
RO_SENTINEL="# === mllmopd P14 external rollout engine ==="
if [ -f "${RO_FILE}" ]; then
  if grep -q "${RO_SENTINEL}" "${RO_FILE}"; then
    echo ">>> ${RO_FILE}: already patched (P14 engine sentinel present)"
  else
    echo ">>> patching ${RO_FILE}: external-mode num_gpus=0 + no placement strategy"
    export MLLMOPD_PATCH_RO_PATH="${RO_FILE}"
    python3 - <<'PY'
import os, sys
path = os.environ["MLLMOPD_PATCH_RO_PATH"]
with open(path) as f:
    src = f.read()
anchor = '''        num_gpus = 0.2
        num_cpus = num_gpus

        # Get the base GPU ID from placement group
        base_gpu_id = int(reordered_gpu_ids[i * num_gpu_per_engine])

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
        )'''
patch = '''        # === mllmopd P14 external rollout engine ===
        if getattr(args, "rollout_external", False):
            # External engines: wrapper actor is HTTP-only, needs no GPU,
            # not bound to the placement group's rollout slice (P14 part A
            # already shrunk that slice to empty). base_gpu_id=None makes
            # _compute_server_args fall through to get_base_gpu_id(args,
            # rank), matching the --base-gpu-id values start_rollout_servers.sh
            # passed when launching the external sglang on Box 1.
            num_gpus = 0
            num_cpus = 0.1
            base_gpu_id = None
            scheduling_strategy = None
        else:
            num_gpus = 0.2
            num_cpus = num_gpus

            # Get the base GPU ID from placement group
            base_gpu_id = int(reordered_gpu_ids[i * num_gpu_per_engine])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[i * num_gpu_per_engine],
            )
        # === end mllmopd P14 external rollout engine ==='''
if anchor not in src:
    sys.exit(f"ERROR: P14 engine anchor not found in {path!r} — init_rollout_engines may have moved")
new_src = src.replace(anchor, patch, 1)
with open(path + ".tmp", "w") as f:
    f.write(new_src)
os.replace(path + ".tmp", path)
print("    P14 rollout.py applied")
PY
  fi
else
  echo ">>> ${RO_FILE}: skipped (file not found for P14 engine)"
fi

# --- Patch P15: qwen3vl.py visual-tower converter rewrite ----------------
# Cross-host weight sync via UpdateWeightFromDistributed (NCCL TCP) crashes
# on first visual-block param with sglang error:
#   "Failed to update parameter online:
#    'model.visual.blocks.0.mlp.gate_up_proj.weight'."
#
# Root cause: the visual branch in qwen3vl.py just passthroughs the
# Megatron-side name with prefix swap. It does not split the fused
# Qwen2.5-VL visual SwiGLU `gate_up_proj` into HF-loader's `gate_proj` +
# `up_proj`, does not rename `attn.qkv_proj` → `attn.qkv`, and uses the
# wrong target prefix `model.visual.` instead of sglang's internal `visual.`.
#
# Single-box T1 ran fine because UpdateWeightFromTensor (CUDA IPC) uses
# a different sglang endpoint that internally normalizes names. The
# distributed/NCCL endpoint requires exact HF-loader names.
#
# P15 replaces the visual branch with the full Qwen2.5-VL visual-tower
# converter from GPT's analysis (chat 2026-05-22):
#   - mlp.gate_up_proj.*  → mlp.gate_proj.* + mlp.up_proj.*  (chunk dim=0)
#   - attn.qkv_proj.*     → attn.qkv.*                       (rename)
#   - attn.proj.*, norm1/norm2, mlp.down_proj → passthrough
#   - target prefix       → visual.  (not model.visual.)
#   - raises on unknown   (no silent passthrough)
# LLM branch (delegate to convert_qwen2_to_hf with model.language_model.
# prefix swap) is unchanged.
#
# Implementation: full-file rewrite, sentinel-gated. Safer than anchor
# replace because the existing file is small and we touch most of it.
QWEN3VL_FILE="${MILES_DIR}/miles/backends/megatron_utils/megatron_to_hf/qwen3vl.py"
# Sentinel bumped to v2 after first cross-box smoke revealed the LLM
# branch also needs the `model.` prefix dropped (sglang's load_weights
# expects `language_model.` not `model.language_model.`, mirroring how
# visual expects `visual.` not `model.visual.`).
QWEN3VL_SENTINEL="# === mllmopd P15v3 qwen2.5-vl converter (LLM nested model. prefix) ==="
if [ -f "${QWEN3VL_FILE}" ]; then
  if grep -q "${QWEN3VL_SENTINEL}" "${QWEN3VL_FILE}"; then
    echo ">>> ${QWEN3VL_FILE}: already patched (P15 visual converter sentinel present)"
  else
    echo ">>> patching ${QWEN3VL_FILE}: full visual-tower Qwen2.5-VL converter"
    cat > "${QWEN3VL_FILE}" <<'PYEOF'
# === mllmopd P15v3 qwen2.5-vl converter (LLM nested model. prefix) ===
# v3 bump: P15v2 dropped `model.` from LLM but sglang still rejected.
# Reason: sglang Qwen2_5_VL nests LLM as
#   Qwen2_5_VLForConditionalGeneration.language_model = Qwen2ForCausalLM
#   Qwen2ForCausalLM.model = Qwen2Model
#   Qwen2Model.layers = [...]
# So named_parameters yields `language_model.model.layers.X...` — the
# inner `.model.` comes from Qwen2ForCausalLM.model wrapping. P15v3
# uses `language_model.model.` prefix for LLM. Visual still uses
# `visual.` (Qwen2_5_VisionTransformer is flat, no inner `.model`).
# === legacy P15v2/P15 banner ===
# === legacy P15 banner ===
# Source: GPT analysis 2026-05-22 (chat log). Replaces the passthrough
# visual branch with a complete Qwen2.5-VL visual-tower name converter
# so UpdateWeightFromDistributed (NCCL) can broadcast weights to sglang
# rollout engines.
#
# Key points (verified against sglang pinned Qwen2_5_VL and HF/checkpoint index):
#   - Visual MLP is SwiGLU; Megatron/sglang fused name is
#     `mlp.gate_up_proj.{weight,bias}` with chunk order [gate, up].
#     Loader expects separate `mlp.gate_proj.*` + `mlp.up_proj.*`.
#   - Visual attention is fused QKV (NOT split q/k/v like LLM).
#     Megatron/sglang internal = `attn.qkv_proj.*`; loader = `attn.qkv.*`.
#   - Output projection is `attn.proj.*` (not `self_attn.o_proj`).
#   - Norms are `norm1.weight` + `norm2.weight` (RMSNorm); no q_norm/k_norm.
#   - Target prefix is `visual.` (sglang internal root), NOT `model.visual.`.
#     If your sglang server registers `model.visual.*` keys before
#     load_weights normalization, change _VISUAL_TARGET_PREFIX below.
# === end mllmopd P15 banner ===

import re

from .qwen2 import convert_qwen2_to_hf


# Qwen3VL / Qwen2.5-VL Megatron parameter prefixes.
_LM_PREFIX = "module.module.language_model."
_VISUAL_PREFIXES = (
    "module.module.visual.",
    "module.module.vision_model.",
)
_PROXY_PREFIX = "module.module."

# Target prefix for sglang Qwen2_5_VL load_weights. Use plain `visual.`
# (sglang internal root). Switch to `model.visual.` only if your server
# rejects `visual.*` keys.
_VISUAL_TARGET_PREFIX = "visual."

_VISUAL_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")


def _convert_qwen25vl_visual_to_loadable(rest: str, param):
    """Convert Qwen2.5-VL visual-tower Megatron-side names to loader-ready names.

    Expected Megatron/sglang fused names (after prefix strip):
      blocks.{i}.attn.qkv_proj.{weight,bias}    # fused qkv
      blocks.{i}.attn.proj.{weight,bias}        # output proj
      blocks.{i}.mlp.gate_up_proj.{weight,bias} # SwiGLU fused
      blocks.{i}.mlp.down_proj.{weight,bias}
      blocks.{i}.norm1.weight
      blocks.{i}.norm2.weight

    Non-block visual params:
      patch_embed.proj.weight
      merger.ln_q.weight
      merger.mlp.{0,2}.{weight,bias}
      rotary_pos_emb.inv_freq

    Raises ValueError on unknown names — silent passthrough is what
    caused the original bug.
    """
    m = _VISUAL_BLOCK_RE.match(rest)
    if m is not None:
        layer_idx, leaf = m.groups()
        base = f"{_VISUAL_TARGET_PREFIX}blocks.{layer_idx}."

        if leaf in ("norm1.weight", "norm2.weight"):
            return [(base + leaf, param)]

        if leaf in ("attn.qkv_proj.weight", "attn.qkv_proj.bias"):
            hf_leaf = leaf.replace("attn.qkv_proj.", "attn.qkv.", 1)
            return [(base + hf_leaf, param)]

        if leaf in ("attn.qkv.weight", "attn.qkv.bias"):
            return [(base + leaf, param)]

        if leaf in ("attn.proj.weight", "attn.proj.bias"):
            return [(base + leaf, param)]

        if leaf in ("attn.o_proj.weight", "attn.o_proj.bias"):
            new_leaf = leaf.replace("attn.o_proj.", "attn.proj.", 1)
            return [(base + new_leaf, param)]

        if leaf in ("mlp.gate_up_proj.weight", "mlp.gate_up_proj.bias"):
            suffix = "weight" if leaf.endswith(".weight") else "bias"
            gate, up = param.chunk(2, dim=0)
            return [
                (base + f"mlp.gate_proj.{suffix}", gate.contiguous()),
                (base + f"mlp.up_proj.{suffix}", up.contiguous()),
            ]

        if leaf in (
            "mlp.gate_proj.weight", "mlp.gate_proj.bias",
            "mlp.up_proj.weight", "mlp.up_proj.bias",
            "mlp.down_proj.weight", "mlp.down_proj.bias",
        ):
            return [(base + leaf, param)]

        raise ValueError(f"Unknown Qwen2.5-VL visual block parameter: {rest}")

    if rest in (
        "patch_embed.proj.weight",
        "merger.ln_q.weight",
        "merger.mlp.0.weight", "merger.mlp.0.bias",
        "merger.mlp.2.weight", "merger.mlp.2.bias",
    ):
        return [(_VISUAL_TARGET_PREFIX + rest, param)]

    if rest == "rotary_pos_emb.inv_freq":
        return [(_VISUAL_TARGET_PREFIX + rest, param)]

    raise ValueError(f"Unknown Qwen2.5-VL visual parameter: {rest}")


def convert_qwen3vl_to_hf(args, name: str, param):
    """Convert Qwen3VL / Qwen2.5-VL Megatron parameter names to HF-loader names.

    Text branch: delegate to convert_qwen2_to_hf, then prepend
    `model.language_model.` (preserves existing single-box behavior).

    Visual branch: full Qwen2.5-VL visual-tower converter (see helper).
    """
    if name.startswith(_LM_PREFIX):
        rest = name[len(_LM_PREFIX):]
        proxy_name = _PROXY_PREFIX + rest
        qwen2_results = convert_qwen2_to_hf(args, proxy_name, param)
        # P15v3: rewrite `model.X` → `language_model.model.X`.
        # qwen2 converter emits `model.layers.X...` (i.e., `model.` =
        # Qwen2Model's named root). Sglang's Qwen2_5_VLForConditionalGeneration
        # wraps Qwen2ForCausalLM as `language_model.`, and that one wraps
        # Qwen2Model as `language_model.model.`. So the full path is
        # `language_model.model.layers.X...`.
        patched = []
        for hf_name, tensor in qwen2_results:
            if hf_name.startswith("model."):
                hf_name = "language_model." + hf_name  # "model.X" stays inside
            patched.append((hf_name, tensor))
        return patched

    for prefix in _VISUAL_PREFIXES:
        if name.startswith(prefix):
            rest = name[len(prefix):]
            return _convert_qwen25vl_visual_to_loadable(rest, param)

    raise ValueError(f"Unknown Qwen3VL parameter name: {name}")
PYEOF
    echo "    P15 qwen3vl.py rewritten (full visual converter)"
  fi
else
  echo ">>> ${QWEN3VL_FILE}: skipped (file not found for P15)"
fi

# --- Patch P16: best-effort cleanup of orphan NCCL update groups --------
# After a trainer crash without clean shutdown, the sglang engine retains
# the NCCL "miles-pp_0" group from the dead trainer's session. Next trainer
# launch calls init_weights_update_group with the same group name and
# sglang returns 400: "The specified group name has already been created."
#
# Fix: at the START of connect_rollout_engines_from_distributed, send a
# best-effort destroy_weights_update_group to every engine. If the group
# doesn't exist (clean state), the engine returns an error that we
# swallow. If it does exist (orphan), it's now gone and the subsequent
# init_weights_update_group succeeds.
UWFD_FILE="${MILES_DIR}/miles/backends/megatron_utils/update_weight/update_weight_from_distributed.py"
UWFD_SENTINEL="# === mllmopd P16 cleanup orphan NCCL group ==="
if [ -f "${UWFD_FILE}" ]; then
  if grep -q "${UWFD_SENTINEL}" "${UWFD_FILE}"; then
    echo ">>> ${UWFD_FILE}: already patched (P16 orphan-cleanup sentinel present)"
  else
    echo ">>> patching ${UWFD_FILE}: best-effort destroy before connect"
    export MLLMOPD_PATCH_UWFD_PATH="${UWFD_FILE}"
    python3 - <<'PY'
import os, sys
path = os.environ["MLLMOPD_PATCH_UWFD_PATH"]
with open(path) as f:
    src = f.read()
anchor = '''def connect_rollout_engines_from_distributed(
    args: Namespace, group_name: str, rollout_engines: Sequence[ActorHandle]
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.
    """
    master_address = ray._private.services.get_node_ip_address()'''
patch = '''def connect_rollout_engines_from_distributed(
    args: Namespace, group_name: str, rollout_engines: Sequence[ActorHandle]
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.
    """
    # === mllmopd P16 cleanup orphan NCCL group ===
    # Best-effort destroy of any orphan `group_name` left over from a
    # previous trainer instance that crashed without clean shutdown.
    # Sglang's init_weights_update_group returns 400 on duplicate group
    # names; orphans accumulate across crashes during xbox bring-up.
    # If the group doesn't exist on the engine side, the destroy call
    # errors and we silently continue — that's the expected clean-state path.
    import logging as _mllmopd_log
    try:
        _cleanup_refs = [
            engine.destroy_weights_update_group.remote(group_name)
            for engine in rollout_engines
        ]
        ray.get(_cleanup_refs)
        _mllmopd_log.getLogger(__name__).info(
            f"[P16] pre-connect cleanup destroyed orphan group {group_name!r} "
            f"on {len(rollout_engines)} engines"
        )
    except Exception as _mllmopd_e:
        _mllmopd_log.getLogger(__name__).info(
            f"[P16] pre-connect cleanup of {group_name!r} returned "
            f"{type(_mllmopd_e).__name__}: {_mllmopd_e} (no orphan or "
            f"clean state; continuing to init_weights_update_group)"
        )
    # === end mllmopd P16 ===

    master_address = ray._private.services.get_node_ip_address()'''
if anchor not in src:
    sys.exit(f"ERROR: P16 anchor not found in {path!r} — connect_rollout_engines_from_distributed may have moved")
new_src = src.replace(anchor, patch, 1)
with open(path + ".tmp", "w") as f:
    f.write(new_src)
os.replace(path + ".tmp", path)
print("    P16 update_weight_from_distributed.py applied")
PY
  fi
else
  echo ">>> ${UWFD_FILE}: skipped (file not found for P16)"
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
echo "    # T2-1 VD weights flow (when MLLMOPD_USE_VD_WEIGHTING=1):"
echo "    zcat \${MLLMOPD_RUNS}/t2_1_v0_*/diagnostics/step_*.jsonl.gz | head -1 | python -m json.tool | grep -A2 vd_weights"
