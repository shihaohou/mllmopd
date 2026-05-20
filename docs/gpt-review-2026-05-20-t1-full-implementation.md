# GPT review brief — T1 full implementation (post-smoke-pass)

**Date**: 2026-05-20 (smoke PASSED at `d08ef7c` after 12h env debug)
**Repo**: https://github.com/shihaohou/mllmopd (HEAD ≈ `d08ef7c` or later)
**Prior reviews in this session**:
- R1 — initial story sanity (vision-conditioned framing + T1 design) → led to G1 (sysprompt parity rerun) and T1 plan tightening (Risk #8 + #9)
- R2 — Ray placement_group NOSET + TMS `LD_PRELOAD` → P0 fixes that unblocked NCCL collective
- R3 — sglang colocate weight-sync `cudaErrorInvalidValue` → P0+P1 fixes (`monkey_patch_torch_reductions()` sender-side, flattened-bucket device normalize receiver-side) that finally let weight sync succeed

This brief is **broader**: ask you to review the entire T1 implementation now that smoke proves the pipeline works end-to-end. The OOM that follows the smoke pass is pure 80GB memory tightness (Megatron forward pass), resolved by migrating to a 140GB H800 — not a design issue.

## What just got proven working (smoke verifier on 2026-05-20 13:14 and 13:20)

```
PASSES: 3
  + step jsonls present (n=1)
  + lp_full and lp_blank both populated and differ (max |vd|=0.84-0.92)
  + image_mode == 'full' on all rows
>>> SMOKE PASSED — green light for T1 full runs

rows total              : 64       (8 prompts × 8 samples = 64 rollouts)
rows with lp_full       : 64
rows with lp_blank      : 64
rows length-aligned     : 64
max |vd| across rows    : 0.92
```

So end-to-end this works:
1. SGLang generates student rollouts (4 engines, ~620 tok/s)
2. `mllmopd.training.dual_teacher_get_reward` POSTs to the teacher server twice per sample (full + blank image)
3. `mllmopd.training.opd_diagnostics_hook` writes a step JSONL with `lp_full`, `lp_blank`, `vd = lp_full - lp_blank`
4. Verifier asserts the VD signal is real (max |vd| > 0.1, ✓ at 0.84–0.92)

The actual gradient step doesn't run on A800 (80GB OOM in forward), but the pipeline up to that point — which is what T1's negative control measures — fully works.

## Pre-flight gates and their outcomes

All in `docs/`:
- G1 (H2 prompt-parity rerun on devbox A800, `runs/audit/level1_v4_sysprompt_fixed/vd_summary.sysprompt.json`) — Finding 2 reproduced + strengthened. Negative-VD tail mystery dissolved (was base-model-mode artifact).
- G2/G3/G4 — finding doc + T1 plan tightened (dedup over whole eval subset, Risk #8 + #9, dual-log both arms).
- Punch list 1-9 + 12-14 — all code committed. #10/#11/#15 are GPU/wall-time-bound; smoke gate now closed.

## T1 design recap (read this first)

`docs/finding-2026-05-19-vision-conditioned.md` + `docs/t1-plan-2026-05-19.md` are authoritative. Two-layer claim:
1. **Prompt-level**: 73% of MMR1-7B-RL's advantage over MMR1-3B-SFT is vision-critical (133/181 prompts).
2. **Token-level**: ~7% of tokens carry ~21% of teacher NLL effort.

T1 is the **negative-control** test: vanilla OPD with the teacher conditioning on **full image** (T1-2) vs **blank image** (T1-3). Student rollout always sees full image. Decision tree:
- T1-2 ≫ T1-3 on math/chart + `opd_target` → visual signal matters → method paper
- T1-2 ≈ T1-3 → vanilla OPD distills language priors regardless → negative-result paper
- T1-2 better full-acc but `G_gap` flat → distilling language reasoning, not visual

## Files to review

### T1 implementation (our code)

```
src/mllmopd/training/
  blank_teacher_payload.py        — encode same-size white-blank PIL images
                                    via Uni-OPD's encode_image_for_rollout_engine.
                                    Matches mllm_corruptions.blank_image
                                    (white 255,255,255) so the training-time
                                    blank intervention is byte-identical to
                                    the audit/H2 blank canvas.
  dual_teacher_get_reward.py      — the T1 heart. async get_reward that fires
                                    BOTH full and blank teacher requests in
                                    parallel via asyncio.gather. Selects
                                    primary based on OPD_TEACHER_IMAGE_MODE.
                                    Stashes the secondary under
                                    meta_info_diagnostic. Wired in via
                                    --custom-rm-path mllmopd.training.dual_teacher_get_reward.get_reward
  opd_diagnostics_hook.py         — replaces Uni-OPD's default post_process_rewards
                                    with one that ALSO writes per-step JSONL
                                    (step_<NNNNNN>.jsonl.gz) with lp_full /
                                    lp_blank / vd / old_lp_student per sample.
                                    The canonical post-processing logic is
                                    INLINED (60 lines, verbatim from Uni-OPD)
                                    to avoid the broken upstream import chain.

src/exps/                         — minimal shim for stale `exps.OPD.utils.reward.*`
                                    imports in Uni-OPD's checkpoint.py /
                                    reward_manager.py. Only `teacher` and
                                    `utils` leaves (the deps reward_manager
                                    actually uses). See exps/__init__.py for
                                    the rationale.

src/miles/__init__.py             — namespace-extend wrapper. Uni-OPD's
                                    margin_shift.py uses `miles.miles.X`
                                    double-prefix; our wrapper extends
                                    __path__ so both `miles.X` and
                                    `miles.miles.X` resolve.

scripts/data/
  inspect_mmr1_rl.py              — print schema + length distribution
                                    (punch list #1)
  prep_opd_train_data.py          — load MMR1-RL via load_from_disk, multi-
                                    level dedup against the WHOLE 1200-prompt
                                    Level-1 eval subset (norm-question +
                                    image-SHA256), hard stop on >5% layer-b
                                    overlap, emit JSONL with `problem` field
                                    already containing MMR1_SYSTEM_PROMPT
                                    prepended in canonical
                                    [text:sysprompt, image, text:question]
                                    order. Punch list #2.
  verify_chat_template_parity.py  — byte-level compare audit `_build_messages`
                                    vs Uni-OPD `_build_messages` rendering.
                                    Punch list #3. Risk #2 closed.

scripts/train/
  start_teacher_server.sh         — direct sglang launch + health-poll wait
                                    + write canonical
                                    teacher_server_{list,map}.json. Punch #6.
  opd_mmr1_3b_baseline.sh         — the T1 launcher. Single-node 8-GPU
                                    layout: teacher on GPU 0 (separate
                                    process), trainer + colocated sglang
                                    on GPUs 1-4. Switches arm via
                                    OPD_TEACHER_IMAGE_MODE={full,blank}.
                                    Has 20+ env / arg guards now;
                                    documented inline. Punch #7.
  smoke_t1.sh                     — slices 80 prompts, runs launcher with
                                    DEBUG_MODE=1. Punch #9.
  verify_t1_smoke.py              — verifier: ≥1 step jsonl, lp_full !=
                                    lp_blank (max|vd|>0.1), image_mode
                                    matches arm, no NaN/inf loss. Punch #9.
  smoke_dual_teacher.py           — punch #4 mandated smoke. Compares
                                    dual_teacher.get_reward output to a
                                    reference score_completion call on the
                                    same sample. Asserts lp arrays match
                                    within 1e-3.

scripts/audit/
  run_t1_eval.sh                  — 9-pass matrix (T1-0/T1-2/T1-3 × full/
                                    blank/text_only) on the same
                                    level1_subset_v0.jsonl. Reuses audit
                                    pipeline. Punch #12.
  run_t1_vd_shift.sh              — re-runs forced-decode VD on the trained
                                    students' rollouts. Punch #14.
  rerun_h2_sysprompt.sh           — G1 pre-flight; superseded by G1
                                    rerun completion, kept for traceability.

scripts/setup/
  patch_uni_opd.sh                — 6 idempotent in-place patches to the
                                    Uni-OPD + sglang submodules. Documented
                                    in docs/h800-migration-checklist.md
                                    section 3b.

src/mllmopd/analysis/
  t1_compare.py                   — headline analyzer. Per-benchmark G_full/
                                    G_blank/G_gap/blank_leakage/opd_target_recovery
                                    + Δ = (T1-2−T1-0)−(T1-3−T1-0) +
                                    bootstrap 95% CI (1000 resamples,
                                    paired prompt-level) + McNemar paired
                                    test on the 133 opd_target prompts.
                                    Risk #9 closed. Punch #13.
  t1_vd_shift.py                  — compare post-training student VD
                                    distribution against the teacher
                                    baseline (post-G1). Punch #14.

configs/baseline/mmr1_3b_t1_{2_full,3_blank}_teacher.yaml
                                  — project-level metadata YAMLs (Uni-OPD
                                    doesn't read them; documentation +
                                    reproducibility tracking only). Diff
                                    only at the 5 arm-specific fields.
```

### Submodule patches (via `scripts/setup/patch_uni_opd.sh`)

6 in-place edits to upstream code that we MUST apply to make T1 work in our setup. All sentinel-guarded for idempotency.

1. `margin_shift.py`: `miles.miles.X` → `miles.X`
2. `ray_launcher.py`: extend MILES_RAY_RUNTIME_ENV with LD_PRELOAD / LD_LIBRARY_PATH / NCCL_DEBUG_SUBSYS / TORCH_NCCL_BLOCKING_WAIT / TORCH_NCCL_ASYNC_ERROR_HANDLING
3. `actor.py`: inject env-inspect at top + inside `MegatronTrainRayActor.init()` (debugging artifact; can be removed in production)
4. `weight_utils.py` (sglang): device-coerce loaded_weight before copy_ + tensor-info diag (defensive guard; should rarely fire after P5)
5. `update_weight_from_tensor.py` (Uni-OPD): inject `monkey_patch_torch_reductions()` at start of `_send_to_colocated_engine` ← **the actual P0 fix for the weight-sync invalid-argument bug**
6. `model_runner.py` (sglang): flattened-bucket `.to(self.gpu_id)` device normalization

### Key supporting docs

```
docs/finding-2026-05-19-vision-conditioned.md       — paper §1-2 seed
docs/t1-plan-2026-05-19.md                         — 15-item punch list
docs/common-pitfalls.md                            — E1–E5; E5 is the 12h NCCL story
docs/handoff-2026-05-19.md                         — session handoff
docs/h800-migration-checklist.md                   — companion to this brief
docs/gpt-review-2026-05-20-weight-sync.md          — round-3 review
configs/baseline/mmr1_3b_t1_{2,3}_*.yaml           — arm metadata
```

## What to review (specific asks)

### A. T1 method-layer soundness (now that smoke proves the plumbing)

1. **`dual_teacher_get_reward.get_reward` correctness.** Does the parallel `asyncio.gather(full, blank)` + arm-specific primary selection give a clean per-token VD signal? Specifically the choice to **always dual-call** (both arms log both lp_full and lp_blank) so post-hoc attribution is symmetric. Any failure modes the smoke verifier wouldn't catch?

2. **`opd_diagnostics_hook.post_process_rewards_with_diagnostics`.** We inlined Uni-OPD's canonical `post_process_rewards` to dodge the broken import chain. Behavior is supposed to be byte-equivalent. Is there a subtle TEACHER_LOGP_FAILED_SENTINEL edge case (e.g., when REWARD_FAILED_KEY is True) that our inline misses?

3. **`prep_opd_train_data.py` dedup.** Multi-level: exact id (skipped since MMR1-RL has no id), normalized question (strip `<image>`, lowercase, punct, whitespace), q+choices (skipped since MMR1 has no choices), image SHA-256, fuzzy (TODO). Layer-b overlap was 0.03% in actual run, layer-d 0.10%. Sufficient defense against Risk #1? Anything we're systematically missing?

4. **`prep_opd_train_data.py` sysprompt prepend.** We force every prep row to `<MMR1_SYSTEM_PROMPT> <image>\n<question_text>` regardless of MMR1-RL's original `<image>` placeholder position. This normalizes the chat-template content list to `[text:sysprompt, image, text:question]` which `verify_chat_template_parity.py` confirmed matches the audit byte-for-byte. **But — by normalizing image-position, we change the prompt distribution MMR1 was trained on (some MMR1-RL prompts had image at end).** Is this a confound? Should the training data preserve original image-position to avoid distribution shift, accepting that the chat-template rendering will differ per-row?

5. **`t1_compare.py` statistical methodology.**
   - Bootstrap 95% CI on Δ via paired prompt-level resampling (1000 resamples, seed=42). Per-benchmark intersection of ids that have non-None values across all 3 arms — is this the right way to define the resampling pool, or should we resample at benchmark level?
   - McNemar paired test on the 133 opd_target prompts via `scipy.stats.binomtest`, falls back to inline log-space exact-binomial CDF. Correct for n=133 with potentially 0 discordant pairs?
   - Verdict thresholds (`>0.01` for "big effect", `<0.005` for "≈"): tuned for 6×200 = 1200 prompts. Defensible?

6. **`response_correct=False` (instead of None)** in dual_teacher_get_reward to keep Uni-OPD's `log_rollout_data` from crashing on `sum(None,...)`. Setting all to False means Uni-OPD's per-step correctness metric in tensorboard is always 0. Our analysis (paired_vision_critical) doesn't use that — but does it affect any internal Uni-OPD heuristic we don't know about?

### B. Submodule patch safety

These all live in `scripts/setup/patch_uni_opd.sh`:

7. **Patch 5 (sender `monkey_patch_torch_reductions()`)** — confirmed by GPT R3, smoke pass-validated. Anything we miss? Specifically: does the monkey patch persist across actor restarts cleanly? Any race where the first send happens before the patch is installed?

8. **Patch 6 (receiver `flattened_tensor.to(self.gpu_id)`)** — defensive companion. After P5, this should be a no-op. Verified in smoke (no `[mllmopd weight-sync receiver]` lines logged). Worth keeping as guard, or remove for clarity?

9. **Patch 4 (`default_weight_loader` device-coerce + diag)** — also defensive. In smoke it didn't fire. Worth keeping?

### C. Other considerations

10. **`SGLANG_MEM_FRACTION=0.55` on A800 was a fit-the-80GB measure.** On H800 (140GB) we can comfortably go back to 0.70+ and even re-enable `--offload-train` if `--no-offload-train` no longer needed for memory reasons. But re-enabling `--offload-train` would resurrect the TMS LD_PRELOAD chain (see launcher comment) — which we've never actually tested with the GPT-R3 fixes in place. Worth a one-shot test on H800 BEFORE committing to a default?

11. **`--megatron-to-hf-mode bridge`** is required for loading from HF; our T1 plan's "load HF directly via bridge" path is now smoke-validated. Bridge mode also affects weight-sync from Megatron to sglang during training (the Qwen25VLBridge). Any known issues with bridge mode + the multi-rank sender (`_send_to_colocated_engine`) we should be cautious about?

12. **Reproducibility on a fresh box.** `docs/h800-migration-checklist.md` lists the 22 known pitfalls + their guards. Is this sufficient as a runbook for a new dev to reproduce smoke pass on a clean H800 in <2h, or are we missing structural items (e.g., setup_train_env.sh isn't sufficient to bootstrap)?

## What this isn't asking

- Not asking about NCCL/CUDA/proxy/placement-group anymore — those are closed via R2/R3 and validated by smoke pass.
- Not asking about T1 narrative — that's set by Finding 2 + paper §1.
- Not asking about whether to switch teacher/student pair — MMR1-7B-RL → MMR1-3B-SFT is committed.

Thanks. We're at the gate where the next step is launching 2 × 4-5h training runs; a sanity check on the implementation before burning that GPU time is worth a careful pass.
