# Handoff: T2-1 xbox cross-box bring-up (2026-05-22)

Self-contained handoff. Read this if you're picking up the project
after the May 22 xbox-mode bring-up session that took T2-1 from
"OOM at step 1 on 8-GPU same-box" to "stable cross-box training
running".

## TL;DR

T2-1 (VD-weighted FullTeacher OPD) was OOM'ing at step ~1 on the
H800 8-GPU same-box config because sglang colocate + Megatron
trainer + activations exceeded 140 GiB per GPU. Solved by moving
all sglang processes to a separate A800 box (Box 1) and giving the
H800 box (Box 2) the full 8 GPUs for trainer-only:

- **Box 1 (A800, 10.82.121.12):** GPU 0 = teacher sglang
  (MMR1-7B-RL); GPU 1-7 = 7 student rollout sglang HTTP servers
  (MMR1-3B-SFT). All standalone processes, NOT Ray actors. Started
  by `scripts/train/start_rollout_servers.sh`.
- **Box 2 (H800, 10.86.16.16):** GPU 0-7 = Megatron trainer
  (DP=8, ZeRO-1, no colocate). Single-node Ray head; engine
  wrapper actors run locally with num_gpus=0 and proxy HTTP to
  Box 1's sglang servers. Started by
  `scripts/train/opd_mmr1_3b_baseline_xbox.sh`.

Weight sync goes via NCCL cross-host TCP (no IB between clusters;
forced via `NCCL_IB_DISABLE=1` + auto-resolved `NCCL_SOCKET_IFNAME`).
Trainer ↔ teacher and trainer ↔ rollout-engines both use HTTP
(intranet routing).

T2-1 full training is currently running with bumped sglang knobs
(mem_fraction=0.40, max_running=48, cuda_graph_max_bs=48). ETA
~3-4h. After completion: run
`scripts/audit/run_t2_1_eval.sh` to get the headline Δ vs T1-2
baseline; the eval script + headline-compare analyzer
(`mllmopd.analysis.t2_1_compare`) are already in place from the
earlier T2-1 implementation session.

## What landed

### Patches in `scripts/setup/patch_uni_opd.sh`

Five new sentinel-gated idempotent patches. Re-apply via
`bash scripts/setup/patch_uni_opd.sh` after every `git pull`.

| Patch | File(s) | What it fixes |
|---|---|---|
| **P14** | `miles/ray/placement_group.py` + `miles/ray/rollout.py` | In `--rollout-external` mode, skip rollout GPU bundles and spawn engine wrapper actors with `num_gpus=0`. Without this, miles asks for 9 (or 15) GPU bundles on an 8-GPU node and hangs. |
| **P15v4** | `megatron_to_hf/qwen3vl.py` | Full Qwen2.5-VL visual converter (SwiGLU `gate_up_proj` split, fused `qkv_proj` → `qkv` rename, `norm1/norm2` passthrough, `merger.*` passthrough) + LLM branch passthrough. LLM target prefix is plain `model.X` (sglang Qwen2_5_VL has `self.model = Qwen2Model(prefix="model")` — NO `language_model.*` wrapper). v1→v4 reflects four prefix-iteration attempts; v4 is verified by GPT against sglang pinned source. |
| **P16** | `update_weight_from_distributed.py` | Best-effort `destroy_weights_update_group` before `init_process_group` so orphan NCCL groups left by trainer crashes don't break subsequent launches. |
| **P17** | `sglang_engine.py` | `_init_external` now POSTs `/add_worker` to MilesRouter (mirroring `_init_normal`). Without this, MilesRouter has zero workers and the first `/generate` crashes with `ValueError: min() iterable argument is empty`. |
| **P18** | `miles/router/router.py` | uvicorn `log_level="warning"` + `access_log=False`. Without this, every `/generate` logs a line; ~110k noise lines per full run. |

### New launchers

- **`scripts/train/start_rollout_servers.sh`** — launches N standalone
  sglang HTTP servers on Box 1 GPUs 1..N.
  - Shared `CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7"` + per-engine
    `--base-gpu-id=0..N-1` so the sanity-checked
    `base_gpu_id` matches trainer-side `get_base_gpu_id` math.
  - Explicit `--nccl-port=$ROLLOUT_PORT_BASE+1000+i` to avoid
    parallel-startup port races.
  - `--host = ROLLOUT_ADVERTISE_HOST` (NOT `0.0.0.0`) so the
    server's self-reported `host` field passes
    `_init_external` strict-check against the trainer's
    `--rollout-external-engine-addrs` split.
  - Readiness probe uses `${ROLLOUT_HOST}:port` not `127.0.0.1`
    (loopback unreachable on specific-IP bind).
  - Auto-resolves `NCCL_SOCKET_IFNAME` via `ip route get`;
    forces `NCCL_IB_DISABLE=1` by default (A800↔H800 no IB).

- **`scripts/train/opd_mmr1_3b_baseline_xbox.sh`** — fork of the
  single-box launcher with xbox-specific changes:
  - Default `ACTOR_NUM_GPUS_PER_NODE=8`, `TRAINER_GPUS=0..7`
  - `--rollout-external --rollout-external-engine-addrs ${ROLLOUT_ENGINE_ADDRS}`
  - Removed `--colocate / --no-offload-train / --no-offload-rollout`
    (colocate-specific workarounds; not applicable)
  - Auto-resolved NCCL env mirrored into Ray actor runtime_env
  - `LR_WARMUP` auto-clamp based on `effective_opt_steps` for
    smoke runs
  - `DEBUG_MODE=1` (rollout `.pt` dumps) split from `VERBOSE_NCCL=1`
    (NCCL_DEBUG=INFO) so DEBUG_MODE doesn't auto-spam NCCL logs

The single-box `opd_mmr1_3b_baseline.sh` is **untouched** and
still reproduces T1-2 byte-identically.

## Three-gate smoke

Per the bring-up plan:

| Gate | Setup | Status |
|---|---|---|
| **A** | trainer→teacher cross-box reachable | ✓ Verified via `curl http://10.82.121.12:30000/get_model_info` |
| **B** | 1 rollout engine, `NUM_ROLLOUT=2` | ✓ 2026-05-22 19:51. Diagnostics OK: `vd_weights[:5]≈[1.0007, 1.0007, 1.0010, 0.9998, 1.0291]`. PGPO mass-preserve healthy. |
| **C** | 7 rollout engines, `NUM_ROLLOUT=10` | Partial: ran 3 steps without errors, user judged stable, skipped to full |

## Full run config

```bash
# Box 1: bumped sglang knobs for throughput
pkill -f "sglang.launch_server.*--port 3000[1-9]"   # NOT 3000 — that kills teacher
TRAINER_IP=10.86.16.16 ROLLOUT_NUM_ENGINES=7 \
ROLLOUT_MEM_FRACTION=0.40 \
ROLLOUT_MAX_RUNNING=48 \
ROLLOUT_CUDA_GRAPH_MAX_BS=48 \
  bash scripts/train/start_rollout_servers.sh

# Box 2: matching SGLANG_* (parity required for sanity check)
SGLANG_MAX_RUNNING_REQUESTS=48 \
SGLANG_CUDA_GRAPH_MAX_BS=48 \
ROLLOUT_ENGINE_ADDRS="$(cat $MLLMOPD_RUNS/rollout_servers/rollout_engine_addrs.txt)" \
TEACHER_HOST=10.82.121.12 \
MLLMOPD_USE_VD_WEIGHTING=1 OPD_TEACHER_IMAGE_MODE=full \
  bash scripts/train/opd_mmr1_3b_baseline_xbox.sh
# OPD_RUN_NAME default = t2_1_v0_T2_1_full_vd
```

Observed mem usage:
- Box 1 GPU 0 (teacher): 80 GiB / 80 GiB (full A800 used)
- Box 1 GPU 1-7 (rollout): ~30 GiB / 80 GiB each after bump
- Box 2 GPU 0-7 (trainer): ~25-30 GiB / 140 GiB each

Cross-host weight sync per step: ~3.2s (Gate B measurement; 16
buckets, 5.15 it/s).

## Performance + async followup

Synchronous training (`train.py`) has obvious rollout-train idle
gaps. miles also ships `train_async.py` (unexplored in this session).
Estimated savings ~30 min on a 3-4h run if overlap works; bigger
ROI when running multiple T2 ablations.

Plan: finish current sync T2-1 baseline, then try async on a second
box for T2-2 / T2-3 / Tier-2 controls. Don't disturb the running
job. GPT prompt drafted in chat 2026-05-22 (covers `train_async.py`
compatibility with `--rollout-external`, off-policy drift risk,
memory implications). Task #20 tracks this.

## Lessons (memory entries)

All saved to `~/.claude/projects/.../memory/`:

- **[[xbox-patches-p14-p18]]** — what each patch does, when to revisit
- **[[xbox-topology]]** — IP / GPU model breakdown
- **[[xbox-topology-gotchas]]** — eight specific failure modes hit
  during bring-up + the fix for each
- **[[sglang-external-server-args-parity]]** — what must match
  between Box 1 sglang launch args and Box 2 trainer args
- **[[anti-pattern-prefix-iteration]]** — when 3+ guesses produce
  same-shape errors, STOP and read source (P15 took 4 iterations
  because I treated "partially updated" as informative when it
  was really just a KeyError wrapper)
- **[[gpt-prompt-include-repo-link]]** — always include
  `https://github.com/<repo>/<commit>` + raw GitHub URLs in GPT
  prompts; pin a specific commit
- **[[cleanup-safety-no-pkill-sglang]]** — updated with port-range
  patterns for xbox (`3000[1-9]` not `3000`)

## What to read first (next session)

1. This handoff (you're reading it).
2. `[[xbox-patches-p14-p18]]` for the patch inventory.
3. `[[xbox-topology-gotchas]]` for failure modes if rebuilding from
   scratch.
4. `[[anti-pattern-prefix-iteration]]` — important meta-lesson
   to avoid burning 3+ iterations on the same class of bug.
5. If full T2-1 completes: `scripts/audit/run_t2_1_eval.sh` and
   `src/mllmopd/analysis/t2_1_compare.py` for the headline Δ.

## Commit lineage (this session)

| Commit | What |
|---|---|
| `36ed5b8` | xbox launcher + start_rollout_servers.sh + P11/P12/P13 plumbing |
| `2b81903` | P14 placement_group skip |
| `f720927` | LR_WARMUP auto-clamp |
| `edc772d` | LR_WARMUP effective_opt_steps clamp |
| `0e29750` | start_rollout_servers bind to advertise IP |
| `b59e884` | readiness probe via bind host |
| `7600729` | auto NCCL_SOCKET_IFNAME + IB_DISABLE |
| `cd19533` | P15 visual converter |
| `8b8775b` | P16 orphan group cleanup |
| `48ae139` | P15v2 (LLM drop `model.`) |
| `103088f` | P15v4 (LLM passthrough; GPT-verified) |
| `ed82911` | P17 external engine MilesRouter register |
| `2abcdb6` | DEBUG_MODE / VERBOSE_NCCL split |
| `61d0926` | start_rollout_servers STREAM_LOGS opt-in |
| `9d4d915` | P18 silence uvicorn access log |
| (current) | full T2-1 xbox run in progress |
