# GPT review brief — T1 v1 negative result (2026-05-21)

T1 v0 was invalidated mid-review by a training-side multimodal pipeline
bug (the launcher omitted `--multimodal-keys`; both arms trained
text-only despite carrying images in the JSONL). v1 fixes that bug and
re-runs the experiment with **verified end-to-end multimodal flow**.

The v1 result is **nearly identical to v0**. Three layers of evidence
converge on the same conclusion: under vanilla dense OPD at the scale
we tested, the gain is invariant to whether the teacher saw the image.
The mechanism that surfaced from the dual-teacher diagnostics adds a
token-level explanation that v0 couldn't have produced (it had no real
dual-teacher to inspect).

## TL;DR

- **Bug confirmed fixed**: training log `Data statistics multimodal:
  2000, text_only: 0` (vs v0's `multimodal: 0, text_only: 2000`).
  Dual-teacher diagnostic confirms `mean |lp_full − lp_blank|` is
  ~0.5-1.9 nats/token during training — i.e., the teacher really IS
  scoring full vs blank differently on the same student response tokens.
- **Same null result**: Δ headline = +0.0017 (v0: +0.0025); 95% CI
  [−0.018, +0.022] (v0: [−0.016, +0.023]); McNemar p=0.442 (v0: 0.345).
  Verdict still "G_gap flat".
- **At step_99 (v1) the model is essentially the same as v0 at step_249**.
  Two radically different training signals (text-only OPD for 250 steps
  vs MM-flowing OPD for 100 steps) produce statistically indistinguishable
  students with identical mean G_full and identical opd_target_recovery.
- **Preliminary mechanism**: dual-teacher `mean|vd|` per token appears to
  collapse from ~1+ nat early in training to ~0.1 nat by step 95 (3-step
  snapshot, full aggregation pending). If the decay is real and
  monotonic, the dense KL ends up fitting tokens the teacher predicts
  with-image and without-image identically — exactly the language-prior
  tokens that drove v0's bug-induced "Δ ≈ 0".

The reframing that v0's GPT review proposed — "BlankTeacher OPD provides
language-space regularization on student's image-conditioned rollouts" —
is now strongly supported by both the verified dual-teacher signal and
the preliminary token-level decay pattern.

## What changed v0 → v1 (only the bugs)

| Layer | v0 | v1 (fixed) |
|---|---|---|
| Launcher `--multimodal-keys` | absent | `'{"image":"images"}'` (commit `76c1ec5`) |
| `MLLMOPD_REQUIRE_MM=1` per-sample guard | n/a | enabled, no fires (commit `76c1ec5`) |
| `prep_opd_train_data.py` metadata block | top-level `id` only | nested `metadata.id` (commit `b52996c`) |
| `paired_vision_critical` arm identity | basename collision (silent merge) | jsonl-stem-derived (commit `4b91720`) |
| `t1_compare.py` arm selection | unaffected by basename bug | unchanged |
| ROLLOUT_MAX_RESPONSE_LEN | 2048 | 6144 (eventually OOM'd at step ~148; step_99 ckpt used) |
| Trainer GPUs | 4 (ZeRO-1 across 4) | 8 (ZeRO-1 across 8) |
| Steps reached | 250 | 100 (step_99 ckpt; both arms; OOM-driven early stop) |

Everything else (hyperparameters, OPD formulation, training pool,
benchmarks) is identical to v0.

## Experimental setup (recap, with v1 corrections)

- **Teacher**: MMR1-7B-RL, frozen, served via sglang on a separate
  cross-box GPU.
- **Student**: MMR1-3B-SFT, Megatron + Uni-OPD trainer, ZeRO-1
  distributed optimizer, dynamic batch cap 8192 packed tokens, 8
  trainer GPUs, GBS=64, MBS=1.
- **Loss**: vanilla on-policy distillation (dense KL over student
  response tokens). `kl_loss_coef=0`, `entropy_coef=0`, OPD clip
  range 10.0.
- **Single-knob negative control**: `OPD_TEACHER_IMAGE_MODE = full | blank`.
  `full` (T1-2): teacher scores with full image. `blank` (T1-3): teacher
  scores with a same-shape solid-color blank. Everything else
  (prompts, sampling seed, student rollout image, loss, optimizer) is
  identical. The student rollout always sees the full image, in both
  arms — only the teacher's scoring side differs.
- **Training pool**: MMR1-RL 2k-prompt subset (same JSONL as v0 with
  added `metadata.id` block).
- **Eval**: 9-pass matrix on level1_subset_v0 (200 prompts × 6
  benchmarks × {T1-0 base, T1-2, T1-3} × {full / blank / text_only}).

## Headline results

### (1) Δ ≈ 0 with wide CI — same as v0

```
HEADLINE Δ = (T1-2 − T1-0) − (T1-3 − T1-0) = +0.0017
95% bootstrap CI                            = [−0.0175, +0.0217]   (crosses 0)
McNemar on 133 opd_target prompts           = b=16 vs c=11, p=0.442  (NS)
verdict (decision tree)                      = "G_gap flat"
```

Per-benchmark Δ (T1-2 G_full − T1-3 G_full):

| Benchmark | Δ_b | T1-2 G_full | T1-3 G_full |
|---|---|---|---|
| ChartQA | +0.000 | +7.5pp | +7.5pp |
| HallusionBench | −0.020 | −1.5pp | +0.5pp |
| MathVerse | −0.015 | −3.5pp | −2.0pp |
| MathVision | +0.020 | −5.5pp | −7.5pp |
| MathVista | +0.030 | +7.0pp | +4.0pp |
| POPE_adv | −0.005 | +0.5pp | +1.0pp |
| **Mean Δ_b** | **+0.0017** | **+0.8pp** | **+0.6pp** |

Both arms gain ~+0.7pp on average; the per-benchmark Δ pattern is noise
around 0. Math benchmarks (Verse / Vision) regress on both arms — the
same pattern as v0 step_249.

### (2) opd_target subset: same large symmetric recovery as v0

`opd_target` is the 133 prompts where MMR1-7B-RL teacher wins on
`full_image` AND base T1-0 misses AND the teacher's win is
vision-conditioned (`vc_t ∩ teacher_advantage` derived from
level1_v4_sysprompt_fixed).

| Benchmark | n_target | T1-2 recovery | T1-3 recovery | T1-2 − T1-3 |
|---|---|---|---|---|
| ChartQA | 27 | 81.5% | 77.8% | +3.7pp |
| HallusionBench | 15 | 40.0% | 46.7% | −6.7pp |
| MathVerse | 25 | 32.0% | 32.0% | 0.0pp |
| MathVision | 35 | 17.1% | 17.1% | 0.0pp |
| MathVista | 28 | 53.6% | 35.7% | +17.9pp |
| POPE_adv | 3 | 33.3% | 33.3% | 0.0pp |
| **Mean** | 133 | **42.9%** | **40.4%** | **+2.5pp** |

Same pattern as v0: large recovery on opd_target prompts but symmetric
across the two arms. v0 had 45.2% vs 40.4% (Δ=+4.8pp NS); v1 has 42.9%
vs 40.4% (Δ=+2.5pp NS, McNemar p=0.442). The recovery effect is real,
sizable, and identical regardless of teacher visual condition.

### (3) Paired vision-critical decomposition

| Metric | T1-2 (v1) | T1-3 (v1) | Δ | (recall v0) |
|---|---|---|---|---|
| VC[T] (teacher_T1-0 vision-critical) | 402 | 402 | 0 | 400/400 |
| VC[S] (student vision-critical) | 412 | 399 | +13 | 410/400 |
| T_adv (teacher wins, student misses) | 102 | 105 | −3 | 100/101 |
| OPD_t = VC[T] ∩ T_adv | 70 | 67 | +3 | 69/70 |
| PureV (T_full✓ T_blank✗ T_text✗ ∩ T_adv) | 54 | 56 | −2 | 55/57 |

Pattern unchanged from v0. The "students are essentially the same on
vision-conditioned structure" finding survives the bug fix.

### (4) Dual-teacher diagnostic confirms the pipeline (new in v1)

Spot-check on `t1_v1_T1_2_full_mm/diagnostics/step_NNNNNN.jsonl.gz`,
sampling 5 rows from steps 5 / 50 / 95:

```
step  5 : mean|lp_full − lp_blank| range 0.55 – 1.25  per token, max ~18
step 50 : mean|lp_full − lp_blank| range 1.23 – 1.86  per token, max ~17
step 95 : mean|lp_full − lp_blank| range 0.07 – 0.18  per token, max ~9
```

These are per-prompt single-sample views, so not yet a smooth curve.
But the magnitude collapse from ~1 nat to ~0.1 nat across the same
training run is large enough to look real. Same pattern shows in T1-3
(the BlankTeacher arm — both arms compute both lp_full and lp_blank
for diagnostics regardless of which one is used for training).

**Important interpretation note**: the lp_full vs lp_blank gap is the
teacher's behavior, not the student's. So this is "the teacher disagrees
with itself less when scoring student outputs at step 95 than at step 5".
Two non-exclusive mechanisms:

  - **A. Student output template-ization**: the student converges to a
    high-frequency template (`<think>` … `</think><answer>\boxed{X}</answer>`)
    where most tokens are language-prior and the teacher's logp is
    near-deterministic regardless of image input.

  - **B. Mean response length grew significantly**: at step 95, the
    sample we inspected had response_length ~1700-2700 (vs ~300-800 at
    step 5). Longer outputs dilute mean|vd| because long-tail tokens
    are template-y. This is the standard caveat we flagged in v0 for
    the post-training VD-shift analysis; it applies again here.

We have a `vd_decay.py` aggregator (commit `d9693cb`) that scans all
~100-200 step files and produces the full curve + per-prompt IQR band.
Not yet run on Mac (needs diagnostics rsync). The 3-step snapshot
suggests the decay is real but the full curve is the artifact paper
review will want.

## Comparison: v0 step_249 (buggy text-only) vs v1 step_99 (MM-flowing)

| | v0 (text-only OPD, 250 steps) | **v1 (MM-flowing OPD, 100 steps)** |
|---|---|---|
| Training-time MM signal | absent (`text_only: 2000`) | present (`multimodal: 2000`) |
| Teacher full vs blank | identical (both text-only) | actually differ (mean\|d\| > 0) |
| Δ headline | +0.0025 | **+0.0017** |
| 95% CI | [−0.016, +0.023] | **[−0.018, +0.022]** |
| McNemar p | 0.345 | **0.442** |
| Mean G_full T1-2 / T1-3 | +0.8pp / +0.6pp | **+0.8pp / +0.6pp** |
| Mean opd_target_recovery T1-2 / T1-3 | 45.2% / 40.4% | **42.9% / 40.4%** |
| Per-benchmark sign pattern (Δ_b) | math regress on both arms | **math regress on both arms** |
| Decision tree verdict | G_gap flat | **G_gap flat** |

This pairing is the strongest single piece of evidence in the brief:
**two radically different training signals — text-only OPD with no
images anywhere for 250 optimizer steps, versus MM-flowing OPD with
verified per-token full-vs-blank disagreement for 100 steps — produced
statistically indistinguishable student checkpoints.**

The bug-induced equivalence isn't a coincidence; it's a strong signal
that whatever OPD is transferring at this scale is **vision-invariant
by mechanism**, not by accident of the test.

## What this means for the hypothesis

The original hypothesis (project_hypotheses memory):

> Vanilla OPD's dense token-level KL fails to efficiently transfer the
> RL teacher's vision-conditioned capability, because supervision is
> uniform over generated tokens while the teacher's visual signal is
> sparse — concentrated in a small subset of high-VD tokens.

The v1 data refines this from "supervision is uniform" to a stronger,
testable claim:

> **At the training scale we tested, vanilla MLLM OPD's effective signal
> on the student is vision-invariant.** Whether the teacher sees the
> image, sees a blank, or never receives an image in the first place
> (the v0 bug scenario), the student converges to essentially the same
> output distribution. The teacher's vision-specific logp signal exists
> early (mean|vd| ~1+ nat at step 5) but collapses over training,
> consistent with a "student outputs template-ize fast enough to make
> the teacher's full-vs-blank disagreement irrelevant" mechanism.

This is no longer a hypothesis we're testing — it's the negative answer
to the original hypothesis plus a token-level mechanism candidate that
the post-hoc data supports.

## Caveats

1. **Single experimental scale**: 2k prompts, 250-step budget,
   MMR1-7B-RL teacher → MMR1-3B-SFT student. Doesn't speak to "does
   the result change at 50k prompts, 1B+ student, frontier teacher".
2. **Step_99 vs step_249 within v1**: We trained to step_99 and OOM'd
   at step ~148. If we'd reached step_249 fully, we'd expect (based on
   v0 step_249 vs v1 step_99 equivalence) more of the same. But we
   haven't verified that v1 to step_249 looks the same as v0 step_249.
   v1.5 retrain with stop-string + in-place div patches would close
   this gap; ~4h budget.
3. **vd_decay is a 3-step snapshot, not the full curve**. The
   full-curve aggregator is written and tested but not yet run on the
   diagnostics. The mechanism claim is preliminary until the full
   curve confirms monotonicity.
4. **opd_target inherited from level1_v4**: the 133-prompt
   vision-critical subset is defined teacher-side. Whether it's still
   the "right" subset to evaluate against, given that the student's
   pre-OPD blind spots may differ from this teacher-derived structure,
   is an open question carried over from v0.
5. **Output length drift**: at step 95, mean response lengths are
   noticeably longer than at step 5, which mechanically dilutes
   mean|vd|. The aggregator includes a sanity-check track for this;
   not yet executed.

## Open questions for review

1. **Does the v1 vs v0 equivalence (despite radically different
   training signals) reproduce as a finding you've seen elsewhere in
   the OPD / MLLM RL literature?** If yes, what's the standard
   reference? If no, is this surprising enough to deserve its own
   framing?

2. **Is "vision-invariance + vision-signal decay" the right writeup
   shape, or would you frame it differently?**  Two candidates:
     - "Vanilla MLLM OPD is a language-side regularizer" — direct,
       constructive negative.
     - "Dense KL is the wrong allocation for sparse visual supervision"
       — narrative that motivates a follow-up method paper (e.g.
       image-differential OPD that only uses `lp_T_full − lp_T_blank`
       as the supervision signal).

3. **What controls would lock the conclusion further?** The v0 review
   suggested three (rollout-blank, SFT-teacher, format-only-SFT). Of
   those, "format-only-SFT" is now the highest-value lever: if a pure
   `<think>/<answer>`-scaffold SFT teacher produces a similar +0.7pp
   gain and similar opd_target_recovery, the "OPD transfers reasoning
   template not vision" thesis is locked. Rollout-blank is also
   informative but requires a custom rollout path. Should we run any
   of these as part of v1.5, or wait until paper writeup?

4. **The vd_decay observation suggests an alternative framing of the
   T2 method direction**. Originally T2 was "PGPO/VPPO-style token
   reweighting to amplify the sparse visual signal". The v1 mechanism
   suggests an alternative: **suppress the teacher's vision-invariant
   logp signal directly via `lp_T_full − lp_T_blank` differential
   supervision**, rather than reweighting tokens after the fact.
   Which formulation looks more promising to you?

5. **What's a meaningful "this experiment worked" signal at this scale?**
   Given current Δ_b magnitudes are ±2-3pp on individual benchmarks
   while CI is ~±2pp, distinguishing a genuine +1pp FullTeacher edge
   from noise would require either (a) much larger n per benchmark,
   (b) a much longer training budget (the v0 vs v1 equivalence
   suggests this won't help), or (c) a different metric entirely
   (e.g., per-prompt log-loss on opd_target, which has more
   statistical power than accuracy).

## Reproduction artifacts (public repo)

All ceph-resident; brief excerpts the headline numbers:

- Launcher: `scripts/train/opd_mmr1_3b_baseline.sh` (HEAD `d9693cb`)
- Training data prep: `scripts/data/prep_opd_train_data.py`
- Eval matrix: `scripts/audit/run_t1_eval.sh`
- Analyzers: `src/mllmopd/analysis/{t1_compare,paired_vision_critical,vd_decay}.py`

Eval run dir (now public on GitHub):
`runs/audit/t1_v0_eval_20260521-194955/` (the second run after the
bridge `text_config` config fix; first run name kept "t1_v0_*" prefix
because the eval ID scheme is timestamped, not arm-versioned).
Contains:
  - `T1_{0,2,3}_{full,blank,text_only}.jsonl` (9-pass outputs)
  - `summary.json` + audit_table headline
  - `t1_compare.json` (headline Δ + bootstrap CI + McNemar)
  - `paired_vision_critical_*.txt` (with the basename-collision fix)
  - `opd_target_ids_*.json`

Training run dirs (ceph only, not on GitHub):
`runs/t1_v1_T1_{2_full_mm, 3_blank_mm}/` — `ckpt/hf/step_{49,99}` +
`diagnostics/step_NNNNNN.jsonl.gz` per step.

## Bugs caught + fixes during this iteration (in order)

| Commit | What | Why it mattered |
|---|---|---|
| `4b91720` | PV uses jsonl-filename arm label | T1-2/T1-3 both save HF as `step_249` → basename collision silently merged arms in PV |
| `76c1ec5` | Launcher passes `--multimodal-keys` + `MLLMOPD_REQUIRE_MM=1` per-sample guard | The v0-invalidating bug |
| `b52996c` | prep_opd_train_data writes nested `metadata.id` | Diagnostics had sample_id=None on every row |
| `1655f8d`/`332fc99`/`76df7cf` | response_len cap iteration 8192 → 6144 → 4096 | OOM tuning (eventually settled at 4096 = eval cap) |
| `6580c5c` | `NUM_ROLLOUT` env override | Fast 100-step A/B path |
| `d9693cb` | `vd_decay.py` aggregator | New analyzer for the v1 mechanism observation |
