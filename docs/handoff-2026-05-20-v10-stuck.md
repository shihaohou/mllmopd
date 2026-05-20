# Handoff — 2026-05-20 T1-2 OOM saga, v10 stuck

## TL;DR

T1-2 (FullTeacher OPD) hit OOM **10 times** today. Each fix moved
something measurable, but PyTorch trainer alloc at actor_train peak
**converged to ~118 GiB and won't budge** despite progressively more
aggressive tuning. ~83 GiB of that 118 GiB is **unaccounted for** in
my per-rank arithmetic; this is what's left to diagnose.

Two GPT diagnosis briefs are live (v1+v7+v10), the v10 update
([docs/gpt-diagnosis-2026-05-20-t1-oom-v10-update.md](gpt-diagnosis-2026-05-20-t1-oom-v10-update.md))
is awaiting GPT input. **Next session should start by reading that
brief and the GPT reply, not by trying another lever.**

## The saga in one table

| Attempt | OOM site | Trainer alloc | Free | What changed | Result |
|--:|--|--:|--:|--|--|
| v1 | `logits.clone()` step 6 | 131.36 GiB | 414 MiB | (baseline T1 implementation) | OOM step 6 |
| v2 | CE buffer step 12 | 130.14 GiB | 952 MiB | +`--use-distributed-optimizer` | OOM step 12 |
| v3 | same | 129.46 GiB | 1.13 GiB | env-var paste mangled, ~equiv to v2 | OOM step 12 |
| v4 | sglang init crash | — | — | +`expandable_segments` (TMS conflict) | crash at init |
| v5 | entropy buffer step 12 | 130.35 GiB | 124 MiB | mem_fraction 0.45, chunk 256 | OOM step 12 |
| v6 | backward 8.48 GiB step 13 | 115.22 GiB | 7.54 GiB | + entropy-zero skip (P7) | OOM step 13 |
| v7 | CE buffer step 0 | **95.64 GiB** | 2.38 MiB | + `--no-offload-rollout` + expandable_segments restored | sglang grew 17→44 GiB |
| v8 | clean (slow) | ~55 GiB | many | mem_fraction 0.15 + max_total_tokens 200k + max_running 16 + disable_cuda_graph | throughput 10x slower |
| v9 | CE buffer step 10 actor_train | **118.33 GiB** | 8 MiB | + MilesRouter (P8) + cuda_graph 32 + max_running 32 + continuous_decode 4 | OOM step 10 |
| **v10** | same step 12 actor_train | **118.98 GiB** | 6 MiB | dropped fp32 grad allreduce + chunk 128 | **OOM step 12, no movement** |

## What worked vs what didn't

### Definitively helped
- `--use-distributed-optimizer`: pushed step 6 → 12
- `--log-probs-chunk-size 256`: pushed step 12 → 13
- `entropy_coef=0` patch (P7): pushed forward
- `--no-offload-rollout`: cleanly resolves TMS / expandable_segments compat
- sglang shrink (mem_fraction 0.55→0.15 + max_total_tokens + max_running + disable_cuda_graph): brought sglang static block from 44 → 21 GiB
- MilesRouter (P8): fixed routing imbalance, restored 4-engine throughput
- `--sglang-cuda-graph-max-bs 32` + `num_continuous_decode_steps 4`: restored throughput to ~6k tok/s aggregate (vs 700 when graphs were fully disabled)

### Didn't help (more lever, no movement)
- Dropping `--accumulate-allreduce-grads-in-fp32`: 118.33 → 118.98 GiB (essentially zero. Theory: --use-distributed-optimizer subsumed it.)
- `--log-probs-chunk-size 256 → 128`: not measurable (CE buffer isn't the binding peak)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: fixed fragmentation when it was the issue (v6 → v7), but doesn't reduce alloc total

## The unexplained 83 GiB

Per-rank budget I can account for at actor_train:

```text
bf16 weights        =  6.2 GiB
bf16 gradients      =  6.2 GiB
fp32 master+m+v / 4 =  9.3 GiB  (ZeRO-1 sharding)
Activations (recompute uniform num_layers=1, 1 mb at a time)
                    ≈ 10 GiB
CE chunk @ 128 tok  =  0.3 GiB
Small + NCCL        ≈  3 GiB
                       -----
Expected            ≈ 35 GiB
Observed            ≈ 118 GiB
Gap                 ≈ 83 GiB   ← THIS IS THE PROBLEM
```

That gap has resisted 4 rounds of fixes targeting that side of the
budget. Next session needs to actually measure it.

Hypotheses for the gap (none verified):

1. `--recompute-method uniform --recompute-num-layers 1` is not doing
   what I think (maybe it's keeping all-layer boundary inputs, not
   just one chunk's worth)
2. torch._inductor / torch._functorch AOT autograd graph captures
   retaining 70+ GiB of state
3. Megatron with `--use-distributed-optimizer` still allocates a
   full-size fp32 grad scratch despite my flag changes
4. The dual_teacher OPD reward path keeps batched logprob tensors
   alive across micro-batches
5. Sequence-parallel at TP=1 plumbing allocates anyway
6. **Stale torchinductor compiled graph in `/tmp/torchinductor_root/`**
   — same `civregd*.py` file referenced in every single OOM stack,
   could be a bad fused-CE compilation cached on first run

## Current state — what's live on H800

- Box: `arc-wlf1-ge103-1` (note: was the box with dangjinming GPU conflict in earlier sessions; conflict resolved itself)
- Teacher: alive on GPU 0 (`MMR1-7B-RL`, mem_fraction 0.85, ~141 GiB). I accidentally killed it once during cleanup (`pkill -9 -f "sglang"` matched teacher too — see `feedback_cleanup_safety.md` memory). User restarted it.
- Trainer GPUs 1-4: idle (1 MiB each) after v10 OOM
- GPUs 5-7: never used
- venv at `/root/shihao_project/mllmopd-train-env/.venv` (validated)
- ceph mounts: all paths under `/home/web_server/antispam/project/houshihao/` working

Archived failed runs (in `${MLLMOPD_RUNS}/`):
- `t1_v0_T1_2_full_oom_step6_20260520_1612` (v1)
- `t1_v0_T1_2_full_oom_step12` (v2)
- `t1_v0_T1_2_full_oom_v3` (v3)
- `t1_v0_T1_2_full_oom_v4_tms_conflict` (v4)
- `t1_v0_T1_2_full_oom_v5_entropy` (v5)
- `t1_v0_T1_2_full_oom_v6_backward_frag` (v6)
- `t1_v0_T1_2_full_oom_v7_sglang_static` (v7)
- `t1_v0_T1_2_full_v8_slow_router_imbalance` (v8 throughput cliff)
- `t1_v0_T1_2_full_v9_fastapi_crash` (v9 first attempt, before P8 patch)
- `t1_v0_T1_2_full_v9_oom_actor_train_118gb` (v9 second attempt, after P8)
- `t1_v0_T1_2_full` (current dir; v10 OOM, not yet archived)

Each preserves its diagnostics jsonl.gz files for any post-hoc
comparison.

## Repo HEAD: `be0035c`

Commit chain today (read in reverse for narrative):

```
be0035c  docs: v9+v10 update for GPT — 83 GiB unaccounted in trainer peak
fdc89d9  launcher: drop fp32 grad allreduce + tighten log-probs chunk to 128
b5caa97  patch_uni_opd: migrate MilesRouter to FastAPI lifespan (P8)
dbb6ae4  launcher: throughput recovery — MilesRouter + cuda-graph + continuous decode
dd71e0c  launcher: stack 3 sglang-shrink levers per GPT v7 brief
dfaf5c0  docs: v6+v7 update brief for GPT — sglang static is the new ceiling
82606ed  launcher: --no-offload-rollout + restore expandable_segments to fix v6 fragmentation OOM
caba46b  launcher: tee stdout/stderr into run_dir/logs/ on ceph
ba0abcc  patch_uni_opd: skip entropy compute when entropy_coef==0 (P7)
8c528b2  Tier B from GPT review: eval CLI + YAML drift fixes
4b2358b  launcher: SGLANG_MEM_FRACTION 0.45→0.25 + --log-probs-chunk-size 256
aa26bfd  launcher: expandable_segments + SGLANG_MEM_FRACTION 0.55→0.45 (was retracted by c6a0bfd)
c6a0bfd  launcher: revert expandable_segments default — conflicts with sglang TMS
1cc310a  launcher: --use-distributed-optimizer to fix T1-2 OOM on H800
c38e02e  T1 launch-gate fixes from GPT review (P0-2 + P0-3)
ae1f762  docs: 2026-05-20 handoff (pre-T1, post-H800-migration)
```

Roughly: 11 launcher edits, 2 patch_uni_opd additions (P7, P8), 4
diagnostic doc commits, several Tier B fixes interleaved.

## Two structural alternatives on the table

Detailed in `docs/gpt-diagnosis-2026-05-20-t1-oom-v10-update.md`
sections "Alt A" / "Alt B".

### Alt A: Dedicated rollout (drop `--colocate`)

```text
GPU 0   : teacher MMR1-7B-RL sglang (unchanged)
GPU 1-3 : dedicated rollout sglang student, 3 engines × 1 GPU
GPU 4-7 : trainer (DP=4), no sglang on these GPUs
```

Pros: frees 21-GiB sglang static block from each trainer GPU → ~140
GiB available, comfortable. Eliminates 5+ TMS/colocate workarounds.

Cons: launcher rewrite around `--colocate`. Need to verify
Uni-OPD's `--rollout-num-gpus` decouples from trainer GPUs (probably
yes since rollout_num_gpus is a separate flag, but need to test).

### Alt B: 8-GPU ZeRO-3 / TP

User-suggested. Pros: shards model weights too. Cons: GBS=64 needs
DP-divisibility; teacher placement; bigger smoke validation.

### My (Claude) tentative preference: Alt A first

- Smaller surface area
- Solves the actual measured bottleneck (sglang static + colocate gymnastics)
- Doesn't change OPD method or experimental variables
- GPUs 5-7 are currently wasted anyway

But this should be debated with GPT v10-update brief input before
committing to refactor.

## Next session: launch order

1. **Read the GPT v10-update reply** if it's come back. Brief is at
   `docs/gpt-diagnosis-2026-05-20-t1-oom-v10-update.md`. 6 questions
   were sent.

2. **Quick experiment (5 min)** before going structural: nuke
   torchinductor cache and retry v10 unchanged:
   ```bash
   # On H800, in hsh2 tmux:
   rm -rf /tmp/torchinductor_root/
   ls /tmp/torchinductor_root/ 2>&1   # confirm gone
   OPD_TEACHER_IMAGE_MODE=full OPD_RUN_NAME=t1_v0_T1_2_full \
     bash scripts/train/opd_mmr1_3b_baseline.sh
   ```
   If trainer alloc drops below 100 GiB, the answer was a stale
   inductor compilation all along. Tell GPT to ignore Alt A/B.

3. **Instrument PyTorch memory snapshot** (GPT v10-update Q2 will
   tell you where to inject `torch.cuda.memory._dump_snapshot()`).
   This is the *ground truth* about the 83 GiB. Once we know what's
   actually in memory, fix is obvious.

4. **Pick a structural direction** based on GPT input:
   - Alt A: refactor launcher to drop `--colocate`, put sglang on 5-7
   - Alt B: refactor to TP=2 / ZeRO-3
   - Or some hybrid

5. **Don't apply more single-lever fixes blindly.** v10 is the proof
   that this approach has stopped paying off.

## Critical gotchas for next session

1. **`pkill -9 -f "sglang"` kills the teacher.** See
   `~/.claude/projects/.../memory/feedback_cleanup_safety.md`. Use
   `pkill -9 -f "ray::SGLangEngine"` instead.

2. **Don't put expandable_segments + colocated sglang together
   without `--no-offload-rollout`.** TMS sanity check raises.

3. **User uses `hsh2` tmux session for T1-2 launches.** Don't propose
   alternative session names. See `feedback_user_decisions.md`.

4. **All commits use this co-author trailer**:
   `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`

5. **GBS=64 is fixed**, MAX_RESPONSE_LEN=2048 was *almost* relaxed
   to 1024 several times. Both are experimental variables; touching
   them changes T1-2 / T1-3 vs prior baseline comparability. Only as
   last resort.

6. **Teacher on GPU 0 has been alive since 15:02 today** (excluding
   the brief death from my pkill mistake at ~19:00). Treat it as
   precious infrastructure — restarting it costs 60-90s + the
   network preflight in the launcher. Don't kill it on cleanup.

## File index for next session

Tier 1 — start here:
- `docs/gpt-diagnosis-2026-05-20-t1-oom-v10-update.md` — primary brief, contains 6 questions
- `docs/handoff-2026-05-20-v10-stuck.md` (THIS FILE)
- Look for `docs/gpt-reply-*.md` if GPT response has been pasted by user

Tier 2 — for digging:
- `docs/gpt-diagnosis-2026-05-20-t1-oom.md` — original GPT brief
- `docs/gpt-diagnosis-2026-05-20-t1-oom-v7-update.md` — v6/v7 update brief
- `docs/handoff-2026-05-20.md` — pre-T1 session handoff (smoke pass + GPT review of T1 implementation)
- `docs/gpt-review-2026-05-20-t1-full-implementation.md` — method-layer review (was approved before launch)

Tier 3 — implementation:
- `scripts/train/opd_mmr1_3b_baseline.sh` (~620 lines, HEAD `be0035c`)
- `scripts/setup/patch_uni_opd.sh` (8 patches P1-P8, all sentinel-idempotent)
- `src/mllmopd/training/dual_teacher_get_reward.py` (OPD primary reward path)
- `src/mllmopd/training/opd_diagnostics_hook.py` (post_process_rewards + diagnostics)
