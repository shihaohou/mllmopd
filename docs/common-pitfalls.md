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
