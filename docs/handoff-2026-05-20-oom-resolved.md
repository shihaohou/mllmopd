# T1-2 OOM saga — resolved (2026-05-20 handoff)

Comprehensive write-up of how T1-2 went from "OOM every 6-13 steps" to
"both T1-2 and T1-3 running cleanly in parallel". Read this if you hit a
similar Uni-OPD + Megatron + sglang colocate memory issue on H800/H100.

## TL;DR

- **14 launch attempts, ~20 commits**, ~10 hours of debugging
- Trainer actor_train PyTorch alloc went from **131 GiB → ~32 GiB**
- **Real root cause** wasn't found until v11: `--use-dynamic-batch-size --max-tokens-per-gpu 16384`
  was packing multiple short samples into a single 16k-token micro-batch, producing a
  **9.3 GiB fp32 logits buffer per fused_vocab_parallel_cross_entropy call**
- The other fixes weren't "the bug" — they were necessary to **make the root cause visible**
  (otherwise sglang static block / routing imbalance / entropy waste / fp32 grad allreduce
  buffer all masked the real signal)
- **Decisive instrumentation**: `torch.cuda.memory._record_memory_history` +
  `_dump_snapshot` injected via a `before_train_step` hook (`src/mllmopd/training/memsnap.py`)
  showed the ground-truth allocation profile that ordinary OOM messages couldn't

## Real root cause (v11 onwards)

```python
# scripts/train/opd_mmr1_3b_baseline.sh:
--use-dynamic-batch-size
--max-tokens-per-gpu 16384       # ← this was the killer (now 8192)
```

What this combination does:

1. With `MICRO_BATCH_SIZE=1`, my mental model was "one sample per
   forward". **Wrong.**
2. Megatron's dynamic batching packs multiple sequences into a single
   forward call, up to `max-tokens-per-gpu` tokens total.
3. At 16384 token budget, a single packed micro-batch could carry one
   sample with `T=16384` tokens (or several short ones summing to that).
4. The fused cross-entropy in `compute_log_probs` materializes a fp32
   logits buffer of shape `[1, T, V=151936]` = `16384 × 151936 × 4 bytes
   ≈ 9.3 GiB` **per CE call**, and forward+backward keep it alive across
   the loss reduction.
5. With ~16 such CE calls per optimizer step plus inductor temp
   buffers, the trainer peak ballooned to ~118 GiB.

Halving `max-tokens-per-gpu` to 8192 directly halves that 9.3 GiB
buffer and the matching backward graph state — saves ~10-15 GiB per
trainer GPU.

## Layered fix stack (in commit order)

Each fix targets a different layer. None of them alone solved the OOM;
together they compress the budget below the H800 140 GiB ceiling.

| Layer | Fix | Commit | What it does | Savings |
|------|-----|--------|--------------|---------|
| Optimizer | `--use-distributed-optimizer` (ZeRO-1) | `1cc310a` | Shard fp32 master + Adam m + v across DP=4 | ~27 GiB/rank |
| OPD loss | `--log-probs-chunk-size 128` | `4b2358b`+`fdc89d9` | Chunk fp32 CE buffer in compute_log_probs | small |
| OPD loss | Skip entropy when `entropy_coef=0` (patch P7) | `ba0abcc` | OPD doesn't use entropy; was computing wasted fp32 buffer | ~1-2 GiB |
| sglang | `mem_fraction_static 0.55 → 0.15` | `4b2358b`→`dd71e0c` | sglang's weights+KV pool block | ~28 GiB |
| sglang | `--sglang-max-total-tokens 200000` | `dd71e0c` | Direct cap on sglang KV token count | safer ceiling |
| sglang | `--sglang-max-running-requests 32` | `dd71e0c`+`dbb6ae4` | Per-engine concurrency cap | matches batch shape |
| sglang | `--sglang-cuda-graph-max-bs 32` (not disable) | `dbb6ae4` | Bounded CUDA graph capture (vs disable = 10× throughput cliff) | small mem cost, ~10× throughput |
| sglang | `--sglang-num-continuous-decode-steps 4` | `dbb6ae4` | Fewer scheduler turns per rollout | throughput, not memory |
| sglang | `--use-miles-router` + P8 FastAPI lifespan patch | `dbb6ae4`+`b5caa97` | Default router was hot-spotting 1 of 4 engines; MilesRouter least-active routing | balanced GPU util |
| sglang | `--no-offload-rollout` | `82606ed` | sglang adapter → Noop variant (compat with expandable_segments) | structural |
| Allocator | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | `82606ed` | Fragmentation fix (8.86 GiB reserved-but-unallocated) | structural |
| **Dynamic batch** | **`--max-tokens-per-gpu 16384 → 8192`** | **`4f55f41`** | **The root cause fix** | **~10-15 GiB** |
| Diagnostics | memsnap hook + P9 patch | `4f55f41` | Ground-truth allocation profile via `torch.cuda.memory._dump_snapshot` | — |

### Levers we considered but did NOT need

- `--accumulate-allreduce-grads-in-fp32` was OFF by default in v10
  (`fdc89d9`), but trainer alloc barely moved (118.33 → 118.98 GiB) —
  Megatron's `--use-distributed-optimizer` already handles fp32
  accumulation internally, so the flag was effectively a no-op in our
  config. **Don't waste a commit on this lever if you have dist-opt on.**
- `--fp16-lm-cross-entropy`: not needed once `--max-tokens-per-gpu` cut.
- Dropping `MAX_RESPONSE_LEN 2048 → 1024`: would change experimental
  variable; not needed.
- Switching to ZeRO-3 / TP=2: structural option, not needed for 3B model
  on H800 once root cause was found.

## Mistakes we made (avoid next time)

### 1. Trusted PyTorch's OOM hint without checking
Three times PyTorch suggested `expandable_segments:True` for
fragmentation. We tried it (`aa26bfd`), it crashed sglang's
torch_memory_saver init. We retracted (`c6a0bfd`). Then GPT helped us
understand the TMS adapter goes via `--offload-rollout`, fix that
(`82606ed`), restore expandable_segments. **Lesson**: PyTorch's hint is
correct as a *direction* but not as a *prescription* — investigate the
real allocator behavior before flipping the flag.

### 2. Misdiagnosed dynamic batching
MBS=1 ≠ "one sample per forward" when `--use-dynamic-batch-size` is on.
A 16384-token packed micro-batch can absorb 8 short samples OR 2 long
ones OR 1 very long one. **The fp32 CE buffer scales with T (packed
length), not B (sample count).**

### 3. Killed the teacher with `pkill -f sglang`
Twice. Teacher = `python -m sglang.launch_server` standalone; rollout
engines = `ray::SGLangEngine`. **Always use ray actor names for
zombie cleanup**: `pkill -9 -f "ray::SGLangEngine"`.

### 4. Forgot the cluster is shared
At least once an external user `kill -9`-swept all python on ge103-1,
taking out both teacher and trainer. **Symptom**: process gone, dmesg
clean (no OOM killer), 1.4 TiB host RAM free, neighbouring GPU 6/7
fully occupied by someone else. Switching boxes is ~5-7 min ceph
restore vs hours of "what killed me" debugging.

### 5. Wrote an auto-detect script with a subtle bug
`ls -td "$DIR_A" "$DIR_B" | head -1` picks newest mtime. On ceph shared
across both boxes, this gave **both** boxes the same answer (the most
recently started run). Made me think both boxes were writing the same
arm. **Correct version**: read `--save` from the actual running
launcher's process environ (`/proc/$PID/environ` or `ps -o cmd`).

### 6. Single-lever fixes after the "no more single lever" insight
Even after writing a brief saying "stop applying single-lever fixes,
get ground truth instead", I applied another lever (Rank 4 drop fp32
grad allreduce) and it didn't help. **memsnap → ground truth → fix**
was the only path that worked.

## Stable launcher config (HEAD `a3397e8`)

After all fixes, the trainer-side memory budget:

```text
bf16 weights                       =  6.2 GiB
bf16 gradients                     =  6.2 GiB
fp32 master + m + v sharded by 4   =  9.3 GiB
Activations w/ recompute uniform-1 ≈ 10 GiB
fp32 CE chunk @ 128 tok + dynamic batch @ 8192 budget
                                   ≈  2-4 GiB
sglang weights residual            =  6 GiB
sglang KV pool (mem_fraction 0.15) = ~15-20 GiB
Misc + NCCL + CUDA context         = ~5 GiB
                                     -----
Trainer GPU total                  ≈ 60-70 GiB
```

Measured: ge103-1 GPU 1-4 ~56 GiB during rollout, peaks higher during
actor_train but stays well under 140 GiB ceiling. ~80 GiB headroom.

Key env / flag defaults (all in launcher):

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SGLANG_MEM_FRACTION=0.15
SGLANG_MAX_TOTAL_TOKENS=200000
SGLANG_MAX_RUNNING_REQUESTS=32
SGLANG_CUDA_GRAPH_MAX_BS=32
SGLANG_NUM_CONTINUOUS_DECODE_STEPS=4
SGLANG_ARGS+=(--sglang-disable-cuda-graph removed; using max-bs cap instead)

PERF_ARGS+=(--use-distributed-optimizer)
PERF_ARGS+=(--log-probs-chunk-size 128)
PERF_ARGS+=(--max-tokens-per-gpu 8192)   # ← was 16384, this is the root cause fix

MISC_ARGS+=(--no-offload-train)
MISC_ARGS+=(--no-offload-rollout)    # ← required for expandable_segments + sglang compat
MISC_ARGS+=(--use-miles-router)      # ← essential for routing balance

ALLREDUCE_FP32=0   # default; --accumulate-allreduce-grads-in-fp32 not added
MLLMOPD_MEMSNAP=0  # default; memsnap was the diagnosis vehicle, not needed for prod
```

Uni-OPD submodule patches (via `scripts/setup/patch_uni_opd.sh`,
sentinel-idempotent):

- P1-P6: pre-existing (module path, ray_launcher runtime_env, actor
  inspect, default_weight_loader diag, UUID-aware tensor serialization,
  flattened bucket device normalize)
- P7: skip entropy compute when entropy_coef==0
- P8: MilesRouter FastAPI lifespan migration (add_event_handler removed in 0.106+)
- P9: pre-CE memsnap injection (only fires when MLLMOPD_MEMSNAP=1)

## Memory snapshot workflow (the decisive diagnostic)

When you suspect "PyTorch alloc X GiB but per-rank budget says it
should be Y GiB" and X >> Y, instrument with:

```bash
# Run launcher with:
MLLMOPD_MEMSNAP=1 OPD_RUN_NAME=t1_oom_diagnosis \
  bash scripts/train/opd_mmr1_3b_baseline.sh
```

This activates `src/mllmopd/training/memsnap.py`:
- One-shot DDP buffer audit on first `before_train_step` (dumps per-buffer
  dtype, numel, total bytes — confirms whether dist-opt sharding actually
  applied, whether fp32 grad allreduce buffer exists, etc.)
- Pre-CE snapshot via P9 patch into `policy_loss_function`, right before
  the OOM-ing `get_log_probs_and_entropy(...)` call
- Captures `torch.cuda.memory_snapshot()` as both `.pickle` (viewable in
  https://pytorch.org/memory_viz) and a flat `.summary.txt`

Output:
```
${MLLMOPD_RUNS}/${OPD_RUN_NAME}/diagnostics/memsnap/
  ${ts}_000_before_train_step_*.{json,pickle,summary.txt}
  ${ts}_001_before_train_step_*...
  ${ts}_pre_ce_*...
```

The pre-CE snapshot showed `[1, T, V]` fp32 tensor with `T ≈ 7680`
(packed from multiple samples up to 8192 tokens after our fix), pointing
directly at the dynamic batch culprit.

## Saga timeline (compressed)

```
v1  step 6  → +dist-opt                 → v2 step 12 (worked, but ceiling not gone)
v5  step 12 → +mem_fraction 0.25 +chunk → v6 step 13 (alloc dropped 130→115)
v6  step 13 → +entropy-skip P7 +no-offload-rollout +expandable_segments
              → v7 step 0 (sglang grew into headroom; net worse)
v7  step 0  → mem_fraction 0.15 + max_total_tokens + disable_cuda_graph
              → v8 ran clean but 10× slow throughput (cuda graph cliff)
v8  slow    → +cuda_graph_max_bs 32 +miles-router +P8 FastAPI lifespan
              → v9 step 10 (throughput restored, OOM moved)
v9  step 10 → drop fp32 grad allreduce + chunk 128 (no-op in our config)
              → v10 step 12 (no change, same OOM)
v10 step 12 → MLLMOPD_MEMSNAP=1 + max-tokens-per-gpu 16384→8192
              → v11 step 107 (cleanly running, killed externally by shared-box neighbour)
v11 killed  → restart, MLLMOPD_MEMSNAP=0 default
              → T1-2 (current) + T1-3 launched on second box (ge103-4)
```

## Next session: Step 6 entry

By the time the next session starts, both T1-2 and T1-3 should be done
(~7-8 hour overnight runs). Step 6 = T1-eval + t1_compare + vd_shift.

### First check on the next session

```bash
cd /home/web_server/antispam/project/houshihao/mllmopd
source .env
source /root/shihao_project/mllmopd-train-env/.venv/bin/activate
git pull --ff-only origin main

# Both arms' final HF checkpoint should exist:
ls "${MLLMOPD_RUNS}/t1_v0_T1_2_full/ckpt/hf/" | tail -5
ls "${MLLMOPD_RUNS}/t1_v0_T1_3_blank/ckpt/hf/" | tail -5
# Expect: step_625 (or close to it; num_epoch=1 × 5000 prompts / RBS=8 = 625 steps)
```

### Step 6 pipeline

Three artifacts to produce:

1. **9-pass audit eval matrix** (3 models × 3 modes on level1_v4_sysprompt_fixed):
   ```bash
   export CKPT_T1_2="${MLLMOPD_RUNS}/t1_v0_T1_2_full/ckpt/hf/step_625"
   export CKPT_T1_3="${MLLMOPD_RUNS}/t1_v0_T1_3_blank/ckpt/hf/step_625"
   # T1-0 baseline (MMR1-3B-SFT pre-OPD) re-eval is optional; canonical
   # baseline already exists. See scripts/audit/run_t1_eval.sh comments.
   bash scripts/audit/run_t1_eval.sh
   ```

2. **Headline analyzer** (note: uses `--t1-run-dir / --baseline-dir / --out-json` per Tier B P0-1 fix in `8c528b2`):
   ```bash
   RUN_DIR=$(ls -td "${MLLMOPD_RUNS}/audit/t1_eval_"* | head -1)
   python -m mllmopd.analysis.t1_compare \
     --t1-run-dir "${RUN_DIR}" \
     --baseline-dir "${MLLMOPD_RUNS}/audit/level1_v4_sysprompt_fixed" \
     --out-json "${RUN_DIR}/t1_compare.json"
   ```

3. **vd_shift** (post-training token-level VD before/after T1-2):
   ```bash
   bash scripts/audit/run_t1_vd_shift.sh
   ```

### Key files for Step 6 (Tier B P0-1 already fixed CLI mismatch)

- `scripts/audit/run_t1_eval.sh` — 9-pass matrix runner
- `src/mllmopd/analysis/t1_compare.py` — headline Δ = (T1-2−T1-0)−(T1-3−T1-0) with bootstrap CI + McNemar
- `src/mllmopd/analysis/paired_vision_critical.py` — VC[T1_X] per benchmark
- `scripts/audit/run_t1_vd_shift.sh` — post-training VD shift

### Open question for Step 6

GPT's earlier review of t1_compare.py recommended that the verdict
threshold should be CI-aware (bootstrap lower bound > 0 + McNemar
p < 0.05) instead of just point-estimate > 0.01. This wasn't applied
yet — see `docs/gpt-review-2026-05-20-t1-full-implementation.md` Q5
for context. Probably worth applying before drawing paper-grade
conclusions.

### Out of scope for Step 6

- T2 ablations (PGPO/VPPO-style reweighting) — only if T1 shows
  Δ > 0.02 with CI lower bound > 0
- Paper §3 writeup — Step 7

## Memory entries written in this session

- `feedback_user_decisions.md` — when user commits to a tmux session, don't propose alternatives
- `feedback_cleanup_safety.md` — never `pkill -f "sglang"` (matches teacher); target `ray::SGLangEngine` instead
- `project_ceph_layout.md` (pre-existing) — ceph shared across all H800 boxes; per-box only `/root/.venv`
- `project_devbox_gpu.md` (pre-existing) — H800 sm_90, validated cross-box venv tar at mllmopd-backups/

## Files of record

| File | Purpose |
|------|---------|
| `docs/handoff-2026-05-20.md` | Pre-T1 handoff (smoke pass + GPT review brief) |
| `docs/handoff-2026-05-20-v10-stuck.md` | Mid-saga handoff when stuck at v10 |
| `docs/handoff-2026-05-20-oom-resolved.md` (THIS FILE) | Resolution + Step 6 entry |
| `docs/gpt-diagnosis-2026-05-20-t1-oom.md` | First GPT brief (v1-v5) |
| `docs/gpt-diagnosis-2026-05-20-t1-oom-v7-update.md` | GPT brief v6+v7 |
| `docs/gpt-diagnosis-2026-05-20-t1-oom-v10-update.md` | GPT brief v9+v10 (led to root cause discovery) |
| `docs/gpt-reply-2026-05-20-v10-update.md` | GPT reply identifying the dynamic batching issue |
| `docs/gpt-review-2026-05-20-t1-full-implementation.md` | Pre-launch method review (separate from OOM saga) |
| `scripts/train/opd_mmr1_3b_baseline.sh` | Stable launcher with all fixes |
| `scripts/setup/patch_uni_opd.sh` | Submodule patches P1-P9 |
| `src/mllmopd/training/memsnap.py` | Memory snapshot instrumentation |
