# H800 venv backup runbook (`mllmopd-train-env`)

Backup and restore procedure for the H800-specific `mllmopd-train-env` venv at
`/root/shihao_project/mllmopd-train-env/.venv/`. Companion to
`docs/migrate-env.md` (which is the generic NGC + uv-venv Q1-Q6 troubleshooting
doc copied from `opd-exp`) and `docs/h800-migration-checklist.md` (which
documents the **one-off A800→H800 migration** that intentionally rebuilt from
scratch). This doc is for **H800→H800 same-arch restores** — fast-disk wipe,
broken venv rescue, or seeding a sibling H800 box.

## When to use

- The venv on the current H800 broke (bad install, fast-disk wipe) and you
  want to restore the last-known-good state without rebuilding from scratch.
- You're bringing up a second H800 box and want to skip the ~2 hour install
  process.

## When NOT to use — rebuild instead

Per `docs/migrate-env.md` § Migration runbook → "When NOT to use":

- Different GPU arch (H800 → A800 / H100-PCIe / B100): different `sm` =
  different sgl-kernel / flash-attn / transformer-engine / megatron binaries.
  **A800 ↔ H800 is not interchangeable** — sm_80 ≠ sm_90.
- Different CUDA major (the binaries here are built against cu128, i.e.
  CUDA 12.8 ABI).
- Different OS major / glibc (Ubuntu 22.04 → 24.04 etc.).

## Q6 reminder (the critical bit)

The venv's `bin/python` is a **symlink into uv's managed Python store**:
```
.venv/bin/python -> /root/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12
```

A tarball of only `.venv/` packs the symlink but **misses the interpreter it
points to**. After restore the symlink dangles, shell PATH falls through to
`/usr/bin/python`, all packages "disappear". So backup always packs **both
directories**.

See `docs/migrate-env.md` § Q6 for the full failure mode.

## Backup ledger

| Date            | File                                                                                | Size  | SHA256 (first 16)  | Source host                              | Repo HEAD     | Uni-OPD HEAD  | Driver        | CUDA toolkit |
|-----------------|-------------------------------------------------------------------------------------|-------|---------------------|------------------------------------------|---------------|---------------|---------------|--------------|
| 2026-05-20 14:29 | `mllmopd-train-env-h800-sm90-20260520-1429.tar.gz`                                  | 5.8 GB | `b2f112e75dd0972c…` | `arc-wlf1-ge103-4.idchb2az1.hb2.kwaidc.com` | `8257fb8`     | `b349383`     | `560.35.03`   | 12.9 / cu128 |

**Validation**:
- Same-machine (`ge103-4`) extract → `bin/python --version` → `Python 3.12.12` ✓ (immediately after backup)
- **Cross-box restore on `ge103-5`** (different H800 host, same ceph mount): 10/10 imports PASS, `torch.cuda.nccl.version() = (2, 27, 5)` loaded from venv via DT_RPATH ✓ (2026-05-20). Wall-clock ~5-7 min including pigz extract + sanity. System libnccl on `ge103-5` is at 2.27.3 in `/usr/lib/...` but torch ignores it (red-herring for the legacy `ctypes.CDLL` check — see Step 4 below).
**apex**: NOT INSTALLED (skipped per CUDA 12.9 vs cu128 mismatch + Megatron
falls back to torch norm + launcher has `--no-gradient-accumulation-fusion`).

Backup directory: `/home/web_server/antispam/project/houshihao/mllmopd-backups/`
(ceph network filesystem — same mount as the project repo, persistent across
reboots). Avoid the sibling `…/transfer/` staging dir even though it's the same
mount — that one is multi-tenant and easier to lose to housekeeping.

Snapshots directory: `/root/shihao_project/mllmopd-env-snapshots/` (volatile
root fs but tarred together with the venv, so the snapshots survive any
restore).

Companion files (also tarred):
- `freeze-h800-sm90-20260520-1429.txt` — `uv pip freeze` (270 packages)
- `state-h800-sm90-20260520-1429.txt` — host / driver / GPU / repo HEAD record

## Backup procedure

Run on the source H800 inside `tmux` (~10 min total wall-clock).

### Step 1: pip-freeze snapshot + host state

```bash
SNAPSHOT_DIR=/root/shihao_project/mllmopd-env-snapshots
mkdir -p "$SNAPSHOT_DIR"
DATE=$(date +%Y%m%d-%H%M)

uv pip freeze > "$SNAPSHOT_DIR/freeze-h800-sm90-${DATE}.txt"

{
    echo "# Snapshot @ $(date -Iseconds)"
    echo "# Host: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "# Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    echo "# CUDA toolkit: $(nvcc --version 2>/dev/null | tail -1 || echo 'no nvcc')"
    echo "# apex: NOT INSTALLED (fallback to torch norm, --no-gradient-accumulation-fusion)"
    echo ""
    echo "# mllmopd HEAD:"
    git -C /home/web_server/antispam/project/houshihao/mllmopd log -1 --oneline
    echo ""
    echo "# Uni-OPD submodule HEAD:"
    git -C /home/web_server/antispam/project/houshihao/mllmopd/third_party/Uni-OPD log -1 --oneline
} > "$SNAPSHOT_DIR/state-h800-sm90-${DATE}.txt"
```

### Step 2: pack venv + uv-python store + snapshots

```bash
BACKUP_DIR=/home/web_server/antispam/project/houshihao/mllmopd-backups
mkdir -p "$BACKUP_DIR"

cd /root
TAR_FILE="$BACKUP_DIR/mllmopd-train-env-h800-sm90-${DATE}.tar.gz"

tar -I pigz -cf "$TAR_FILE" \
    shihao_project/mllmopd-train-env/ \
    shihao_project/mllmopd-env-snapshots/ \
    .local/share/uv/python/

# Use relative filename in the sha256 so future `sha256sum -c` works after
# the tar gets moved around.
ls -lh "$TAR_FILE"
( cd "$BACKUP_DIR" && sha256sum "$(basename "$TAR_FILE")" \
    | tee "$(basename "$TAR_FILE").sha256" )
```

**zstd is faster but pigz is more universally available** — both work. If
zstd is around:
```bash
TAR_FILE="$BACKUP_DIR/mllmopd-train-env-h800-sm90-${DATE}.tar.zst"
tar --use-compress-program="zstd -T0 -3" -cf "$TAR_FILE" \
    shihao_project/mllmopd-train-env/ \
    shihao_project/mllmopd-env-snapshots/ \
    .local/share/uv/python/
```

**Note on the sha256 file**: keep it storing the **relative filename** only
(not the absolute path), so `sha256sum -c` still works after the tar is
moved. The `( cd "$BACKUP_DIR" && sha256sum … )` pattern above does this
correctly — a plain `sha256sum "$TAR_FILE" > "$TAR_FILE.sha256"` would record
the absolute path and break the moment you move the file.

Expected size: ~6 GB (13 GB raw venv + ~200 MB uv-python compresses well).

### Step 3: same-machine restore sanity check

Always validate by extracting into a throwaway dir before trusting the backup.
This catches the Q6 "missed uv-python store" mistake immediately.

```bash
RESTORE_TEST=/root/restore-test-$$
mkdir -p "$RESTORE_TEST"
cd "$RESTORE_TEST"

tar -I pigz -xf "$TAR_FILE"

# Critical sentinel — the venv's python symlink must resolve, not dangle.
ls -la shihao_project/mllmopd-train-env/.venv/bin/python
readlink -f shihao_project/mllmopd-train-env/.venv/bin/python
shihao_project/mllmopd-train-env/.venv/bin/python --version   # MUST print "Python 3.12.x"

cd /root && rm -rf "$RESTORE_TEST"
echo "restore test OK"
```

If the `--version` step errors with `No such file or directory`, the
uv-python store wasn't included — re-run Step 2 with the third path.

### Step 4: record this backup in the ledger above

Update the table at the top of this file with:
date, filename, size, sha256 (first 16 chars), source host, mllmopd HEAD,
Uni-OPD HEAD, driver, CUDA toolkit.

## Restore procedure

For restoring on a fresh H800 (or rescuing a broken venv on the same H800).

### Step 1: clean any partial state

**Only if you intend to overwrite the current venv** — this is destructive.
Skip if you're restoring to a fresh box.

```bash
rm -rf /root/shihao_project/mllmopd-train-env
rm -rf /root/shihao_project/mllmopd-env-snapshots
rm -rf /root/.local/share/uv/python   # only if no other uv venv depends on it
```

### Step 2: unpack

```bash
cd /root
tar -I pigz -xf /home/web_server/antispam/project/houshihao/mllmopd-backups/mllmopd-train-env-h800-sm90-YYYYMMDD-HHMM.tar.gz

# Verify the bin/python symlink resolves on this machine
VENV=/root/shihao_project/mllmopd-train-env/.venv
ls -la $VENV/bin/python
$VENV/bin/python --version    # MUST print "Python 3.12.x"
```

### Step 3: activate + 10/10 import sanity

```bash
source /root/shihao_project/mllmopd-train-env/.venv/bin/activate

cd /home/web_server/antispam/project/houshihao/mllmopd

python - <<'PY'
checks = [
    ('numpy<2',     'import numpy; assert numpy.__version__.startswith("1.")'),
    ('torch',       'import torch; assert torch.cuda.is_available()'),
    ('sglang',      'import sglang'),
    ('transformers','import transformers'),
    ('TE',          'import transformer_engine.pytorch'),
    ('megatron',    'import megatron.core'),
    ('flash_attn',  'import flash_attn'),
    ('miles',       'import miles.utils.types'),
    ('Uni_OPD',     'import Uni_OPD_utils.OPD_reward.reward_manager'),
    ('mllmopd',     'import mllmopd.training.dual_teacher_get_reward'),
]
ok = 0
for n, code in checks:
    try: exec(code); print(f'  OK  {n}'); ok += 1
    except Exception as e: print(f'  FAIL {n}: {type(e).__name__}: {e}')
print(f'\n{ok}/{len(checks)} passed')
PY
```

**10/10 = ready.** apex is intentionally absent.

If `miles` / `Uni_OPD` / `mllmopd` fail, the `.pth` file or `src/Uni_OPD_utils`
symlink got lost. Re-create:

```bash
# .pth approach (matches current H800 layout):
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
cat > "$SITE/mllmopd_paths.pth" <<EOF
$(pwd)/src
$(pwd)/third_party/Uni-OPD/miles
EOF
```

### Step 4: NCCL/cuDNN alignment sanity (E5 from common-pitfalls)

The relevant question is **"what does torch load at training time?"**, not
"what does ctypes find via ld.so". On boxes where the system also has a
libnccl (e.g. `/usr/lib/x86_64-linux-gnu/libnccl.so.2`), `ctypes.CDLL` will
load the system one because it respects ld.so cache, while torch uses its
own DT_RPATH and loads the venv one. The two can disagree without indicating
a problem.

So check torch's actual loaded NCCL — that's the ground truth:

```bash
python - <<'PY'
import os, torch  # importing torch triggers libnccl mmap via its RPATH

# Ground truth: the version torch will use during training.
print('torch.cuda.nccl.version():', torch.cuda.nccl.version())

# Where is it actually loaded from?
print('--- /proc/self/maps libnccl entries ---')
with open(f'/proc/{os.getpid()}/maps') as f:
    paths = set()
    for line in f:
        if 'libnccl' in line:
            paths.add(line.split()[-1])
    for p in sorted(paths):
        print(' ', p)

# Cross-check: pip metadata (what wheel claims)
import importlib.metadata as md
print('pip metadata:', md.version('nvidia-nccl-cu12'))
PY
```

**Pass criteria**:
- `torch.cuda.nccl.version()` == `(2, 27, 5)` (matches the version that was
  built into this backup's torch wheel)
- All `/proc/self/maps` paths must point inside the venv
  (`…/mllmopd-train-env/.venv/.../nvidia/nccl/lib/libnccl.so.2`)

**Mismatch signals**:
- `torch.cuda.nccl.version()` returns something else → torch's RPATH was
  defeated (e.g. `LD_PRELOAD` set, broken DT_RPATH after wheel reinstall) →
  E5 bug, see `docs/common-pitfalls.md`.
- `/proc/self/maps` shows `/usr/lib/x86_64-linux-gnu/libnccl.so.2` → torch
  ignored its RPATH and picked up system NCCL → E5 bug.

The older check via `ctypes.CDLL('libnccl.so.2')` is unreliable: it intentionally
goes through ld.so and finds the system one, which can be a different version
than what torch uses without anything actually being broken. Don't rely on it
as a pass/fail signal.

### Step 5: re-backup after first end-to-end success

After a successful smoke (`scripts/train/smoke_t1.sh`) or full T1 run on the
restored env, pack a new validated backup with a fresh timestamp — same Step
1-3 of the backup procedure above. Always timestamp filenames so you never
overwrite a known-good backup with a partial / broken one.

## Cross-references

- `docs/migrate-env.md` — generic NGC + uv-venv troubleshooting (Q1-Q6)
- `docs/h800-migration-checklist.md` — A800 → H800 rebuild-from-scratch (one-off)
- `docs/common-pitfalls.md` — E5 NCCL alignment runbook
