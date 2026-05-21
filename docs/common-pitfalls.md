# Common pitfalls (curated, project-permanent)

Living list of "you'll trip over this" issues. Each entry: symptom → root cause
→ fix. Session-specific handoff docs (`handoff-YYYY-MM-DD.md`) accumulate
session-local discoveries; this file is where things that bite you **across
multiple sessions** get distilled.

Box context: A800 (sm_80) 8×80GB, two UV venvs:
- **audit venv** `/root/shihao_project/mllmopd-env/.venv` — torch 2.5.1+cu124, transformers 4.54, no sglang
- **train venv** `/root/shihao_project/mllmopd-train-env/.venv` — torch 2.9.1+cu128, sglang, megatron, miles

`/root/` is reboot-volatile (handoff §7.1). Venvs survive in NFS tarballs at
`/home/web_server/.../transfer/`; runs/data live on NFS.

---

## E1. `CUDNN_STATUS_NOT_INITIALIZED` on Qwen2.5-VL `nn.Conv3d`

**Symptom**:
```
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED
  File ".../qwen2_5_vl/modeling_qwen2_5_vl.py", line 89, in forward
    hidden_states = self.proj(hidden_states.to(...)).view(...)
  File ".../torch/nn/modules/conv.py", line 712, in _conv_forward
    return F.conv3d(...)
```
sglang's startup warning is the canary:
```
WARNING: Could not determine CuDNN version for torch==2.9.1.
Please ensure CuDNN >= 9.15 to avoid nn.Conv3d bugs.
```

**Root cause (subtle, layered)**:
1. torch 2.9.1 in the train venv requires cuDNN ≥ 9.15 for Qwen2.5-VL's
   visual patch embedding (`Conv3d`).
2. The venv's `nvidia-cudnn-cu12` pip metadata said 9.10.2, but torch was
   actually loading **9.2.0 from a system-wide path** (`/usr/local/lib/
   python3.12/dist-packages/torch/lib/`) due to `LD_LIBRARY_PATH` pollution
   from somewhere in the shell setup.
3. Single-engine inference happened to survive the version mismatch
   (smoke test on one GPU was fine); 3 sglang engines initializing in
   parallel hit the cuDNN init race and one crashed, taking the others
   down via SIGQUIT.

**Fix (in order)**:
1. Install a recent cuDNN into the train venv. **Use `uv pip` or
   `python -m pip` from the venv, never bare `pip`** (bare `pip` may
   resolve to `/usr/local/bin/pip`, which is *not* the active venv):
   ```bash
   source /root/shihao_project/mllmopd-train-env/.venv/bin/activate
   uv pip install 'nvidia-cudnn-cu12==9.16.*'
   ```
2. Clear `LD_LIBRARY_PATH` before running anything that uses sglang +
   Qwen2.5-VL — otherwise the system's stale 9.2 still wins the search:
   ```bash
   unset LD_LIBRARY_PATH
   ```
   Make this part of the run script (we already do this in
   `/tmp/run_t_sft.sh`). Long-term, consider adding to `.env` or the
   train venv's activate hook.
3. Verify the load works:
   ```bash
   python -c "import torch; print('cudnn runtime:', torch.backends.cudnn.version())"
   # Expect 91600 or higher; if it still says 90200, LD_LIBRARY_PATH is dirty.
   ```
4. (Optional) Sometimes a `_backup/` dir is left behind from manual
   library patching, keeping a stale 9.2 .so visible:
   ```bash
   rm -rf /root/shihao_project/mllmopd-train-env/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/_backup
   ```

**Belt-and-suspenders** (still useful even after the fix above): the sglang
runner sets `torch.backends.cudnn.enabled = False` at startup so that even
if cuDNN init is wedged, Conv3d falls back to native CUDA. This costs a few
ms per image, no real penalty.

**Lessons that generalize**:
- *Trust torch's runtime version probe over pip metadata.* What pip says
  is installed is irrelevant if the dynamic linker is finding a different
  library first.
- *Multi-process sglang amplifies single-process races.* If a single-engine
  smoke run works but 3-engine parallel crashes, suspect race conditions
  during init (cuDNN, NCCL, GPU context).
- *Always invoke pip via the venv's python*: `python -m pip ...` or
  `uv pip ...`. Bare `pip` is a footgun.

---

## E2. UV venv `pip` module missing

**Symptom**:
```
$ python -m pip install ...
/root/.../.venv/bin/python: No module named pip
```

**Root cause**: UV venvs are created without the `pip` module by default
(UV is the package manager, not pip). Standard `pip install` won't work.

**Fix**: use `uv pip` instead:
```bash
uv pip install -e . --no-deps
```

If `uv` isn't on `PATH`: `source ~/.local/bin/env` or use `/root/.local/bin/uv` directly.

---

## E3. Wrong venv silently selected by `_activate.sh`

**Symptom**: code runs but `import sglang` fails with `ModuleNotFoundError`
even though sglang is installed somewhere.

**Root cause**: `scripts/env/_activate.sh` resolves venv as:
`MLLMOPD_VENV` env var > project `.venv` symlink > conda. The project
`.venv` symlink points at the **audit venv** (no sglang). When a child
shell (e.g. `bash /tmp/run_t_sft.sh`) doesn't inherit `MLLMOPD_VENV`,
it falls through to the audit venv.

Both UV venvs default their prompt to `(.venv)` from the directory
basename, so you can't tell which one is active from the prompt alone.

**Fix**:
```bash
export MLLMOPD_VENV=/root/shihao_project/mllmopd-train-env/.venv
# then call run_smoke.sh / dispatch
```
Always set `MLLMOPD_VENV` explicitly when invoking sglang-backed audit
runs from a fresh shell.

**Discrimination tip** — `which python` is the truth:
```bash
which python
# /root/shihao_project/mllmopd-train-env/.venv/bin/python  -> train (sglang OK)
# /root/shihao_project/mllmopd-env/.venv/bin/python        -> audit (no sglang)
```

(Distinguish the prompts by editing both venvs' `bin/activate` lines that
do `VIRTUAL_ENV_PROMPT=$(basename "$VIRTUAL_ENV")` to a fixed string.
Reboot wipes these too, so add to setup scripts.)

---

## E4. Bash heredoc / line continuation eats long commands

**Symptom**: a multi-line `python -m ... \` command gets split, half of it
runs without args (`error: the following arguments are required: ...`),
the other half tries to execute as a command (`--mode: command not found`
or `Permission denied`).

**Root cause**: A trailing space after `\` breaks line continuation —
`\<space><newline>` is *not* a continuation in bash, only `\<newline>` is.
Copy-paste of long commands from documentation often introduces trailing
whitespace.

**Fix**: write the command to a script file via heredoc and run the
script — heredoc preserves contents verbatim:
```bash
cat > /tmp/cmd.sh <<'EOF'
#!/bin/bash
python -m mllmopd.diagnostics.run_audit_pass_sglang \
  --subset ... \
  --mode full_image \
  --out /tmp/out.jsonl
EOF
bash /tmp/cmd.sh
```

---

## E5. `ncclUnhandledCudaError` / "CUDA driver version is insufficient" — bare `pip` installs to system site-packages, torch falls back to system NCCL

**Symptom** (in Megatron / Ray-actor training, first NCCL collective):
```
torch.distributed.DistBackendError: NCCL error in: .../NCCLUtils.cpp:94,
  unhandled cuda error (run with NCCL_DEBUG=INFO for details), NCCL version 2.29.7
ncclUnhandledCudaError: Call to CUDA function failed.
Last error: Cuda failure 'CUDA driver version is insufficient for CUDA runtime version'
```

But standalone Python in the same venv has no issue: `torch.zeros(1).cuda()` succeeds, `nvidia-smi` is healthy, driver/runtime are theoretically compatible. The error message is **misleading** — the actual cause is an NCCL ABI mismatch between torch's compile-time NCCL (2.27.5) and a runtime-loaded foreign NCCL (2.29.7).

### Two-bug stack (one masks the other)

**Bug A — venv's `pip` shim is missing.** UV venvs sometimes get created with only `pip3` / `pip3.12` in `bin/`, not `pip`. Bare `pip install ...` then resolves to **system** `/usr/local/bin/pip`, which installs into the NGC system `/usr/local/lib/python3.12/dist-packages/` — **not the venv**. `pip install --force-reinstall` reports `Successfully installed` but writes nothing to the venv.

Sister pitfall to E2 ("UV venv `pip` module missing") — different symptom, same root: never trust bare `pip` in this env. **The bare `pip` footgun warning in E1 is not enough.**

**Bug B — torch falls back to ld.so.cache.** When `<venv>/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2` is missing (e.g. because Bug A redirected the install), torch's `_load_global_deps` fails its preferred resolution and the dynamic linker falls through to `/etc/ld.so.cache`, which on the NGC base image points at `/lib/x86_64-linux-gnu/libnccl.so.2` — system NCCL 2.29.7. torch was compiled with 2.27.5 headers, so the loaded 2.29.7 trips ABI mismatches at the first collective.

### Why `torch.cuda.nccl.version()` is misleading
That call returns the **compile-time** `NCCL_VERSION_CODE` macro baked into `libtorch_cuda.so`, **not** the version of the `libnccl.so.2` currently dlopen'd. It will happily report `(2, 27, 5)` while the actually-loaded NCCL is `2.29.7`.

### Diagnosis — three commands

1. **What pip THINKS is installed where:**
   ```bash
   pip show -f nvidia-nccl-cu12 | head -20
   # If `Location:` is /usr/local/lib/python3.12/dist-packages → Bug A
   # If `Location:` is .venv/.../site-packages → check file actually exists
   ```

2. **What torch ACTUALLY dlopen's:**
   ```bash
   ldd "$(python -c 'import torch; print(torch._C.__file__)')" | grep nccl
   # If path starts with /lib/x86_64-linux-gnu/ → Bug B (system NCCL won the race)
   # If path is venv site-packages → fine
   ```

3. **What version the loaded .so reports at runtime:**
   ```bash
   python -c "
   import ctypes
   v = ctypes.c_int()
   ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(v))
   print(f'{v.value//10000}.{(v.value%10000)//100}.{v.value%100}')
   "
   # Compare against torch.cuda.nccl.version(); mismatch = ABI bug
   ```

### Fix
```bash
# (1) Reinstall NCCL via the venv's Python — bypass the missing pip shim
"${VIRTUAL_ENV}/bin/python" -m pip install \
    --force-reinstall --no-deps --no-cache-dir \
    nvidia-nccl-cu12==2.27.5

# (2) Purge any stray copy in NGC system site-packages
/usr/local/bin/pip uninstall -y nvidia-nccl-cu12 2>/dev/null || true

# (3) Verify all three diagnostics pass:
#       - pip show: Location ends with <venv>/lib/python3.12/site-packages
#       - ldd torch._C grep nccl: path inside venv
#       - ctypes ncclGetVersion: 2.27.5
```

### Long-term defenses (TODO)
- Add `ln -s pip3 .venv/bin/pip` to `scripts/env/setup_train_env.sh` after venv creation, so bare `pip` resolves correctly.
- Pre-flight gate in launcher: assert that `ldd $(python -c "import torch; print(torch._C.__file__)") | grep nccl` resolves inside `$VIRTUAL_ENV`. Fail fast with a pointer to this E5 entry.
- **Rule**: in this repo, never write bare `pip` in scripts or docs — always `${VIRTUAL_ENV}/bin/python -m pip` or `uv pip`. Same rule as E1 closing remark.

### Cost of getting this wrong
~6 hours in session 2026-05-19/20. The misleading "CUDA driver insufficient" message sent us down ~10 wrong hypotheses (driver version, cuda-compat layer, torch_memory_saver hook, LD_LIBRARY_PATH ordering, placement-group misconfig, NCCL env plugin, ...). All real but secondary. The proximate cause was the bare-pip footgun corrupting the install location. The `ldd torch._C | grep nccl` check would have closed it in 30 seconds.

---

## E6. `hostname -I` picks an unroutable NIC on the H800 train container

**Symptom**: cross-box `teacher_server_list.json` ends up registering a URL like `http://26.2.224.21:30000/generate` that the student box can't reach (timeout / no route from the peer).

**Root cause**: the H800 train container has multiple NICs. `hostname -I | awk '{print $1}'` returns them in an order that puts an **overlay address** (`26.2.x.x` on this image) before the actual intranet IP (`10.86.16.x`). The overlay is **not routable from peer boxes in the cluster**; only the `10.86.16.x` address is. Any script that uses `hostname -I` to "discover its own IP" silently picks the wrong one.

Sister trap: the container is slim and **does not ship iproute2** — `ip` is not on PATH — so the obvious fallback (`ip route get <peer>`) also fails silently.

**Fix**: pass `STUDENT_IP=<peer-ip>` to `scripts/train/start_teacher_server.sh`. The script tries `ip route get` first, falls back to a Python `socket.SOCK_DGRAM + connect(peer,1) + getsockname()` trick that works without iproute2 (commits `c513d74`, `92f2b48`). Or pass `TEACHER_ADVERTISE_HOST=<explicit-ip>` to override entirely. Don't bake IPs into `.env`; the box pair changes between experiments.

**Discrimination tip**: after launch, `cat third_party/Uni-OPD/miles/Uni_OPD_utils/OPD_reward/teacher_server_list.json` — the `servers` URL must contain a routable IP, not `localhost` and not `26.x.x.x`.

---

## E7. `http_proxy` reroutes sglang's self-warmup through an overseas squid → 502 → "Killed"

**Symptom**: sglang starts, `Uvicorn running on http://0.0.0.0:30000` appears, then minutes later in the log:
```
AssertionError: res=<Response [502]>, res.text='... 502 Bad Gateway ...
  Server: oversea-squid1.jp.txyun ...'
Killed
```
Parent shell shows `Killed`, easily misread as OOM — but the process exited on a failed `assert` in `_execute_server_warmup`, not via SIGKILL from oom-killer.

**Root cause**: the H800 train container ships `http_proxy` / `https_proxy` env vars pointing at `oversea-squid1.jp.txyun:11080` for outbound pip / HF Hub access. sglang's startup self-warmup curls `http://127.0.0.1:30000/model_info`; `requests` (and most python HTTP libs) respect the proxy env var **even for loopback**, route the call through the squid, the squid can't reach `127.0.0.1` from Tokyo and returns 502, warmup asserts, sglang exits.

Same trap can bite Ray dashboard polls and any process that hits its own local HTTP endpoint.

**Fix**: unset the proxies before launching any local HTTP server (canonical form per project preference):
```bash
unset -v http_proxy https_proxy no_proxy
```
Lowercase only — the uppercase variants are not set on this image. `scripts/train/start_teacher_server.sh` does this internally as of commit `899720e`, so cross-box launches via that script are immune. `opd_mmr1_3b_baseline.sh` already had its own variant. Bare `curl` calls from your interactive shell still need the unset if you've sourced the container's default env.

---

## E8. `OPD_RUN_NAME` collision → `OptimizerParamScheduler` assert

**Symptom**: Ray actor init crashes during checkpoint load:
```
AssertionError: OptimizerParamScheduler: class input value 40 and
  checkpointvalue 320 for warmup iterations do not match
```
Trace ends in `megatron/core/optimizer_param_scheduler.py:240 _check_and_set`, called from `load_state_dict` inside `miles/.../checkpoint.py:load_checkpoint`.

**Root cause**: `scripts/train/opd_mmr1_3b_baseline.sh` auto-detects "resume vs fresh" at line ~441 by checking whether `$CKPT_DIR` is non-empty. If a previous run with the same `OPD_RUN_NAME` left any `iter_*` checkpoint behind (even from a half-finished smoke), the new run silently switches to resume mode and `--load`s the old ckpt — which carries the old `lr_warmup_iters` (and other Megatron scheduler params) inside `state_dict['opt_param_scheduler']`. When the new config doesn't match, Megatron's strict checker asserts.

Common trigger: smoke ran with `DEBUG_MODE=1 RBS=1` and saved at step 50; the operator then re-launches full (`RBS=8`, default `OPD_RUN_NAME`), and the actor dies before training starts.

**Fix**: use a distinct `OPD_RUN_NAME=` per config change (smoke vs full, single-box vs cross-box, ...) — that gives each its own `CKPT_DIR` and forces the script down the fresh-HF-bridge path. Or `rm -rf "${MLLMOPD_RUNS}/${OPD_RUN_NAME}"` if the previous run truly has no value. Don't change `LR_WARMUP` mid-stream on the same `OPD_RUN_NAME` unless you genuinely mean to resume the schedule.

**Discrimination tip**: the script prints one of these at launch time —
```
>>> --load (fresh run, HF bridge): /path/to/MMR1-3B-SFT      ← good, fresh
>>> --load (resume): /path/to/runs/<name>/ckpt               ← will assert if config differs
```
If you see "resume" but didn't intend to, stop and pick a new `OPD_RUN_NAME`.
