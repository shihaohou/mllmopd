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
                _f.write(f"torch.cuda.nccl.version()={_t.cuda.nccl.version()}\n")
                _f.write(f"torch.version.cuda={_t.version.cuda}\n")
                try:
                    _t.cuda.init()
                    _x = _t.zeros(1).cuda()
                    _f.write(f"cuda ok, device={_t.cuda.get_device_name(0)}\n")
                except Exception as _e:
                    _f.write(f"cuda init failed: {type(_e).__name__}: {_e}\n")
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

echo
echo ">>> patch_uni_opd done. Re-run after \`git submodule update\`."
echo
echo "Inspect after smoke:"
echo "    ls -la /tmp/actor_inspect_*.log"
echo "    cat /tmp/actor_inspect_*.log"
