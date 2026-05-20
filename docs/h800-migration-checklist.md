# H800 (sm_90, 140GB) migration checklist

**Source**: A800 (sm_80, 80GB) dev box, smoke PASSED at `d08ef7c` (rollout + dual-teacher + post-process all clean; OOM only at Megatron forward pass — pure memory-fit issue that 140GB resolves).
**Target**: H800 (sm_90, 140GB) machine.
**Goal**: stand up the same T1 pipeline on H800 in <2h, pre-empting every env pitfall we hit on A800 (12h of debugging compressed into a checklist).

## 1. What survives the move (no rebuild needed)

Everything that is NOT compiled native code is on the shared NFS (`/home/web_server/...`) and works on H800 unchanged:

- The whole repo (`mllmopd/`), including:
  - `src/mllmopd/training/` (3 T1 extension modules)
  - `src/mllmopd/analysis/` (t1_compare, t1_vd_shift, aggregate_vd, paired_vision_critical)
  - `scripts/` (launchers, prep, smoke, setup)
  - `docs/` (everything)
  - `configs/baseline/mmr1_3b_t1_{2,3}_*.yaml`
- T1 training data: `data/opd_train/v0_2k/` (2k prompts + images, deduped)
- Audit eval data: `data/audit/level1_subset_v0.jsonl` + `runs/audit/level1_v4_sysprompt_fixed/`
- Models: `~/datasets/MMR1-RL/`, `~/models/MMR1-{3B-SFT,7B-RL,7B-SFT}/`, `~/models/Qwen2.5-VL-7B-Instruct/`
- The git submodules' `.git` tree (so `git -C submodule checkout` works)

## 2. What MUST be rebuilt on H800

The venv. **Specifically the GPU-arch-specific compiled wheels**:

- `sgl-kernel` — sm_90 vs sm_80 binary
- `flashinfer` — sm_90 binary
- `flash-attn` — sm_90 binary
- `xformers` — sm_90 binary
- `TransformerEngine` — sm_90 binary
- `bitsandbytes` (if used) — sm_90 binary

Trying to tar+ship the A800 venv → these compiled .so files will mismatch sm_90 PTX → runtime errors that look like NCCL/CUDA bugs but aren't. **Do not migrate via venv tarball.** Per `docs/migrate-env.md`'s own "When NOT to use" list: different sm = rebuild.

## 3. Pre-empt every A800 pitfall we hit

For each, the proactive action on H800 setup is listed.

### 3a. PRE-empt E5 (the 12h NCCL story)

This is the most expensive pitfall we hit. To prevent recurrence:

```bash
# 1. After venv creation, FIRST verify pip shim exists.
ls -la "${VIRTUAL_ENV}/bin/pip"
# If "pip" missing but "pip3" exists, link it:
[ -e "${VIRTUAL_ENV}/bin/pip" ] || ln -s pip3 "${VIRTUAL_ENV}/bin/pip"

# 2. Throughout setup, NEVER use bare `pip`. Always:
#    "${VIRTUAL_ENV}/bin/python" -m pip ...
#    (or `uv pip`)
#    See common-pitfalls.md E5 for the 12h bare-pip incident.

# 3. After installing nvidia-nccl-cu12 (typically transitively via torch),
# verify .so version == pip metadata:
"${VIRTUAL_ENV}/bin/python" -c "
import ctypes, importlib.metadata as md
import nvidia.nccl, os
p = os.path.join(os.path.dirname(nvidia.nccl.__file__), 'lib', 'libnccl.so.2')
v = ctypes.c_int(); ctypes.CDLL(p).ncclGetVersion(ctypes.byref(v))
runtime = f'{v.value//10000}.{(v.value%10000)//100}.{v.value%100}'
pip = md.version('nvidia-nccl-cu12')
print(f'pip metadata: {pip}')
print(f'runtime so:   {runtime}')
assert pip == runtime, f'MISMATCH! pip={pip}, so={runtime} — see E5'
"

# 4. Verify torch dlopens venv NCCL (not /lib/x86_64-linux-gnu/libnccl.so.2):
ldd "$(${VIRTUAL_ENV}/bin/python -c 'import torch; print(torch._C.__file__)')" | grep nccl
# Path MUST start with ${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/nccl/lib/
# If it points to /lib/x86_64-linux-gnu/ — investigate before any T1 run.
```

### 3b. Apply patch_uni_opd.sh AFTER submodules init

The patch script applies 6 local-only patches to the Uni-OPD + sglang submodules. **None of these are GPU-arch-specific** — they all work on H800. But they must be re-applied after a fresh submodule checkout because submodule files aren't part of the parent commit's tree.

```bash
git submodule update --init --recursive    # if not already done
bash scripts/setup/patch_uni_opd.sh         # idempotent; safe to re-run
```

The 6 patches:
1. `margin_shift.py`: `miles.miles.X` → `miles.X` (double-prefix import fix)
2. `ray_launcher.py`: extend MILES_RAY_RUNTIME_ENV with LD_PRELOAD / LD_LIBRARY_PATH / NCCL_DEBUG_SUBSYS / TORCH_NCCL_BLOCKING_WAIT / TORCH_NCCL_ASYNC_ERROR_HANDLING
3. `actor.py`: inject env-inspect at top + inside `MegatronTrainRayActor.init()` (writes `/tmp/actor_inspect_<pid>.log`)
4. `weight_utils.py` (sglang): device-coerce loaded_weight before copy_ + tensor-info diag
5. `update_weight_from_tensor.py` (Uni-OPD): inject `monkey_patch_torch_reductions()` at start of `_send_to_colocated_engine` (UUID-aware CUDA tensor pickle — the actual fix for the weight-sync invalid-argument bug)
6. `model_runner.py` (sglang): flattened-bucket `.to(self.gpu_id)` device normalization

### 3c. Driver / CUDA / cuda-compat

H800 boxes typically have newer driver than the A800 we used (driver 535). If H800 driver is 555+ (which natively supports CUDA 12.5+ runtime), the cuda-compat layer mentioned in our launcher won't be present at `/usr/local/cuda-12.9/compat/lib.real`. The launcher handles this:

```bash
# scripts/train/opd_mmr1_3b_baseline.sh already does:
for d in /usr/local/cuda-12.9/compat/lib.real /usr/local/cuda-12.8/compat/lib.real /usr/local/cuda/compat/lib.real; do
  [ -d "${d}" ] && COMPAT_LIB_REAL="${d}" && break
done
# ↑ this loop just no-ops if no compat dir exists (H800 likely doesn't need it)
```

No action needed. The launcher's compat handling is defensive.

### 3d. Other A800-specific tunings that should be reset on H800

In `scripts/train/opd_mmr1_3b_baseline.sh`, the following defaults were chosen to fit 80GB:

| Variable | A800 default | H800 reconsider |
|---|---|---|
| `SGLANG_MEM_FRACTION` | 0.55 | Bump to 0.75 (or upstream 0.70) on H800 — much more headroom |
| `--no-offload-train` (MISC_ARGS) | hardcoded on | Could remove on H800, since memory is no longer tight — but only after smoke confirms the TMS LD_PRELOAD chain is also stable |
| `ACTOR_NUM_GPUS_PER_NODE` | 4 | If H800 has 8 GPUs, can keep teacher on 0 + trainer on 1-7 = 7 trainer GPUs. Or keep 4 for parity. **Note Megatron GBS divisibility** — see launcher pre-flight |
| `ROLLOUT_MAX_RESPONSE_LEN` | 2048 | Can raise to 4096 (closer to MMR1 paper's max-CoT) |

For the FIRST smoke on H800, leave all defaults as-is, prove the path works, then tune.

### 3e. NGC proxy on H800

The proxy strip in the launcher (`unset http_proxy`, wide `no_proxy`) only matters if the H800 box has the same Tencent squid proxy in shell env. If H800 has no proxy, the strip is a no-op. **Verify:**

```bash
env | grep -i proxy   # before sourcing .env
```

If H800 has no proxy env vars, smoke #19 issue won't recur. If it DOES, the launcher already handles it.

## 4. Order-of-operations (clean H800 setup)

```bash
# 0. Verify base machine state
nvidia-smi | head -5         # driver + CUDA, confirm sm_90 visible
free -g                       # confirm RAM
df -h /home/web_server        # confirm NFS mounted

# 1. Clone the repo (or git pull if already there via NFS)
cd /home/web_server/antispam/project/houshihao
git clone https://github.com/shihaohou/mllmopd.git mllmopd-h800 2>/dev/null || true
cd mllmopd-h800
git pull --ff-only
git submodule update --init --recursive

# 2. Create venv (use the same setup that built mllmopd-train-env;
#    on A800 we used uv venv + uv pip install).
#    Critical: install nvidia-nccl-cu12==2.27.5 via python -m pip (E5 rule).
# Reference: docs/common-pitfalls.md E5

# 3. Verify NCCL alignment (E5 triad)
ldd "$(python -c 'import torch; print(torch._C.__file__)')" | grep nccl
python -c "
import ctypes, importlib.metadata as md, os, nvidia.nccl
p = os.path.join(os.path.dirname(nvidia.nccl.__file__), 'lib', 'libnccl.so.2')
v = ctypes.c_int(); ctypes.CDLL(p).ncclGetVersion(ctypes.byref(v))
print('runtime so:', f'{v.value//10000}.{(v.value%10000)//100}.{v.value%100}')
print('pip metadata:', md.version('nvidia-nccl-cu12'))
"

# 4. Apply our submodule patches
bash scripts/setup/patch_uni_opd.sh

# 5. Test that smoke env is sane (without launching ray cluster)
python -c "
import torch, sglang
from miles.utils.types import Sample        # Uni-OPD imports
from mllmopd.training.dual_teacher_get_reward import get_reward
print('venv ok')
"

# 6. Launch teacher server
bash scripts/train/start_teacher_server.sh

# 7. Smoke
bash scripts/train/smoke_t1.sh
```

If smoke PASSED on A800 at `d08ef7c`, the same code path should pass on H800 with no further changes — the only difference is the forward pass no longer OOMs at 140GB.

## 5. After smoke passes on H800

Kick off T1-2 + T1-3 full runs. They should now reach actual optimizer steps (not just rollout + reward, which already worked on A800).

```bash
# T1-2 FullTeacher (~4-5h on 4 trainer GPUs)
OPD_TEACHER_IMAGE_MODE=full \
OPD_RUN_NAME=t1_v0_T1_2_full_$(date +%Y%m%d_%H%M%S) \
bash scripts/train/opd_mmr1_3b_baseline.sh

# T1-3 BlankTeacher (sequential)
OPD_TEACHER_IMAGE_MODE=blank \
OPD_RUN_NAME=t1_v0_T1_3_blank_$(date +%Y%m%d_%H%M%S) \
bash scripts/train/opd_mmr1_3b_baseline.sh
```

After both finish: `scripts/audit/run_t1_eval.sh` → `python -m mllmopd.analysis.t1_compare` → write up §15.

## 6. Pitfalls catalog (per-stage, with refs)

| # | Pitfall | Surface symptom | Doc |
|---|---|---|---|
| 1 | Bare `pip` installs to wrong site-packages | `pip Successfully installed` but .so missing in venv | E5 |
| 2 | NCCL wheel binary ≠ pip metadata | `torch.cuda.nccl.version()` (compile) ≠ `ctypes ncclGetVersion()` (runtime) | E5 |
| 3 | torch falls back to system NCCL via ld.so.cache | actor `cudaErrorInvalidValue` / "driver insufficient" | E5 |
| 4 | LD_LIBRARY_PATH cuDNN/NCCL pollution | sglang cuDNN init race, NCCL ABI mismatch | E1 |
| 5 | UV venv missing `pip` module entirely | `python -m pip` errors | E2 |
| 6 | `scripts/env/_activate.sh` picks wrong venv | sglang not importable when in audit venv | E3 |
| 7 | Bash heredoc trailing whitespace eats commands | `--mode: command not found` | E4 |
| 8 | `RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES=1` on train actors → all see 4 GPUs | NCCL collective hangs / wrong-device errors | launcher `--train-env-vars` |
| 9 | `--colocate` enables `--offload-train` → TMS LD_PRELOAD injects into actors → NCCL collective fails | "CUDA driver version insufficient" inside `_get_param_groups` | launcher `--no-offload-train` |
| 10 | Ray runtime_env doesn't propagate `LD_PRELOAD`/`LD_LIBRARY_PATH` to actors | shell-side strips don't reach actors | patch_uni_opd.sh patch 2 |
| 11 | `Uni_OPD_utils.OPD_reward.*` import chain broken (stale `exps.OPD.utils.reward.*`) | actor crashes at first import | `src/exps/` shim + decouple |
| 12 | `miles.miles.*` double-prefix in `margin_shift.py` | `ModuleNotFoundError: miles.miles` | patch_uni_opd.sh patch 1 |
| 13 | `--load <empty_dir>` fails Megatron's non-empty assert | "args.load does not exist or is an empty directory" | launcher `--load` auto-detect |
| 14 | `--megatron-to-hf-mode raw` incompatible with HF checkpoint | "Only bridge mode is supported" | launcher `--megatron-to-hf-mode bridge` |
| 15 | `--ref-load` distrust ckpt format | … | launcher uses `${STUDENT_CKPT}` (HF) |
| 16 | HTTP proxy routes intranet calls through external squid | actor → sglang 502 Bad Gateway | launcher proxy strip + wide no_proxy |
| 17 | `ForkingPickler` serializes CUDA local index, not UUID | sender cuda:0 vs receiver cuda:N → `param.data.copy_` invalid argument | patch_uni_opd.sh patch 5 |
| 18 | `param.data.copy_(loaded)` across-GPU without P2P | invalid argument | patch_uni_opd.sh patch 4 + 6 |
| 19 | `--sglang-mem-fraction-static` too high when `--no-offload-train` | TMS `cuMemCreate` OOM hangs | launcher `SGLANG_MEM_FRACTION=0.55` |
| 20 | `Math.generate.verify_deepmath`/`exps.RL.utils.reward.*` not installable | `rule_base_reward.py` import chain breaks | our `dual_teacher_get_reward` skips it, `response_correct=False` |
| 21 | Apex CUDA ext not built → `gradient_accumulation_fusion=True` fails | RuntimeError in ColumnParallelLinear | launcher `--no-gradient-accumulation-fusion` |
| 22 | Megatron `GBS % (MBS*DP) != 0` | num_microbatches_calculator assertion | launcher divisibility pre-flight |

Each pitfall now has a guard in either the launcher, `patch_uni_opd.sh`, or `docs/common-pitfalls.md`. On H800 they shouldn't recur, but the diagnostic infrastructure (actor inspect log, sglang weight-loader diag) is in place to spot them fast if they do.
