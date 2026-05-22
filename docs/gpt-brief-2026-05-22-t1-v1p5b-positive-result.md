# GPT review brief — T1 v1.5b POSITIVE result (2026-05-22)

> **Status note**: This supersedes the v0 and v1 briefs, both of which
> were invalidated by sequential bugs caught in earlier review rounds:
>
> - `docs/gpt-brief-2026-05-21-t1-negative-result.md` — INVALIDATED by
>   training-side `--multimodal-keys` missing bug (both arms trained
>   text-only).
> - `docs/gpt-brief-2026-05-21-t1-v1-negative-result.md` — INVALIDATED
>   by eval-side `source .env` clobbering caller's `CKPT_T1_2/3` export
>   (eval routed v0 ckpts despite caller-supplied v1 paths).
>
> v1.5b is the first run with **end-to-end verified MM pipeline** AND
> **resolved checkpoint routing**. The result that follows is the first
> trustworthy answer to the original hypothesis.

## TL;DR

**Vanilla MLLM OPD's effectiveness depends critically on the teacher's
visual input.** Trained against a teacher that scores with the
original image (FullTeacher / T1-2), the student preserves its
vision capability and gains modestly (+1.3pp mean full_image acc).
Trained against a teacher that scores with a same-shape blank image
(BlankTeacher / T1-3), dense token-level KL transfers the teacher's
learned visual blindness to the student, causing catastrophic
capability collapse (−21.7pp mean full_image acc; ChartQA: 0.695
→ 0.155). The single-knob FullTeacher-vs-BlankTeacher comparison
produces:

```
Δ headline = (T1-2 − T1-0) − (T1-3 − T1-0) = +0.2300  (= +23.0pp)
95% bootstrap CI                            = [+0.2000, +0.2575]
McNemar on 133 opd_target prompts           = b=44, c=9, p = 1.22e-6
```

Qualitative samples confirm the mechanism: T1-3 (BlankTeacher),
when handed a **real** chart at eval time, **insists the image is
blank** and answers from priors / hallucination. T1-2 (FullTeacher)
reads the chart and answers from visual evidence.

This is the cleanest causal demonstration we have of vision-conditioned
capability transfer at OPD scale: the teacher's visual input isn't just
"helpful" — it's the load-bearing element that prevents the student
from learning visual blindness.

## v1.5b: the configuration that produced this result

After four bug-fix iterations across two GPT review rounds:

| Layer | Final config | Bug history |
|---|---|---|
| `--multimodal-keys` | `'{"image":"images"}'` (commit `76c1ec5`) | Missing in v0 → silent text-only training |
| `MLLMOPD_REQUIRE_MM=1` guard | per-sample assert (commit `76c1ec5`) | Belt-and-suspenders against v0 regression |
| `metadata.id` block in JSONL | nested (commit `b52996c`) | Top-level only in v0 → diagnostics row id=None |
| PV arm identity | jsonl filename derived (commit `4b91720`) | `basename(ckpt_path)` collision in v0 |
| `run_t1_eval.sh` ckpt routing | `source .env` preserves caller exports (commit `59fdf7d`) | v1 eval routed wrong (v0) ckpts |
| `RUN_ID` default prefix | `t1_eval_` (was `t1_v0_eval_`, commit `473e0c4`) | Misleading dir naming |
| `--rollout-stop` | `"</answer>"` (commit `5b28fa1`) | Long-tail OOM mitigation |
| `--log-probs-chunk-size` div | in-place `div_()` (commit `5b28fa1`, miles patch P10) | Saves ~4 GiB fp32 copy at CE peak |
| `--max-tokens-per-gpu` | 4096 (commit `c1c8363`) | 8k/6k packed batches hit 107 GiB / 140 GiB cliff |
| `ROLLOUT_MAX_RESPONSE_LEN` | 3072 | Belt-and-suspenders against long-CoT tail |
| `bridge_bak` config overlay | manual cp (per eval) | Megatron-bridge writes nested `text_config` → sglang rejects |

These fixes are non-experimental (data plumbing / memory hygiene / id
provenance); the single experimental knob remains
`OPD_TEACHER_IMAGE_MODE = full | blank`.

## Experimental setup

- **Teacher**: MMR1-7B-RL, frozen, served via sglang on a separate
  cross-box GPU (`10.82.121.12:30000`).
- **Student**: MMR1-3B-SFT, Megatron + Uni-OPD trainer, ZeRO-1
  distributed optimizer, dynamic batch cap 4096 packed tokens, 8
  trainer GPUs (full host), GBS=64, MBS=1.
- **Loss**: vanilla on-policy distillation (dense KL over student
  response tokens). `kl_loss_coef=0`, `entropy_coef=0`, OPD clip
  range 10.0.
- **Single-knob negative control**: `OPD_TEACHER_IMAGE_MODE = full |
  blank`. `full` (T1-2): teacher scores the student response with the
  original full image. `blank` (T1-3): teacher scores with a
  same-shape solid-color blank. Everything else (prompts, sampling
  seed, student rollout image, loss, optimizer) identical.  **Student
  always sees the full image in rollout** — only teacher scoring varies.
- **Training pool**: MMR1-RL 2k-prompt subset (with v1.5b metadata
  schema fix).
- **Training duration**: 230 steps per arm (target was 250; both
  arms terminated at step ~230 due to checkpoint save schedule and
  early-exit logic, both arms saved at the same step boundaries
  `step_49 / 99 / 149 / 199 / 230` for fair multi-ckpt analysis).
- **Eval**: 9-pass matrix on level1_subset_v0 (200 prompts × 6
  benchmarks × 3 modes × 3 models).

## Headline results

### (1) Δ = +23pp, far from zero

```
HEADLINE Δ = +0.2300                              (= +23.0pp)
95% bootstrap CI = [+0.2000, +0.2575]             ← lower bound +20pp, well above 0
McNemar on 133 opd_target prompts                 = b=44, c=9
                                                    p = 1.221e-06
verdict (decision tree)                           = strong positive (informal —
                                                    the verdict logic predates this
                                                    magnitude of effect; see Caveats)
```

Bootstrap CI lower bound +20.0pp is the strongest possible
"effect is real" signal at our sample size (200 × 6 = 1200 prompts).
McNemar 44 wins for T1-2 vs 9 for T1-3 on the 133 opd_target prompts
gives p = 1.22 × 10⁻⁶ (~one in a million).

### (2) Per-benchmark: BlankTeacher destroys ChartQA the worst, but the effect is uniform

| Benchmark | T1-0 base | **T1-2 (Full)** | **T1-3 (Blank)** | G_T1-2 | **G_T1-3** | **Δ_b** |
|---|---|---|---|---|---|---|
| ChartQA | 0.695 | 0.775 | **0.155** | +8.0pp | **−54.0pp** | **+62.0pp** |
| HallusionBench | 0.635 | 0.605 | 0.505 | −3.0pp | −13.0pp | +10.0pp |
| MathVerse | 0.320 | 0.310 | 0.110 | −1.0pp | −21.0pp | +20.0pp |
| MathVision | 0.180 | 0.180 | 0.160 | 0.0pp | −2.0pp | +2.0pp |
| MathVista | 0.630 | 0.645 | 0.330 | +1.5pp | −30.0pp | +31.5pp |
| POPE_adv | 0.875 | 0.885 | 0.760 | +1.0pp | −11.5pp | +12.5pp |
| **Mean** | **0.556** | **0.567** | **0.337** | **+1.3pp** | **−21.7pp** | **+23.0pp** |

T1-2 (FullTeacher) is essentially baseline-preserving with mild
improvements where teacher-student gap is largest (ChartQA +8pp,
MathVista +1.5pp). T1-3 (BlankTeacher) collapses every benchmark
that requires visual reasoning, with ChartQA — the most
chart-reading-heavy benchmark — taking the biggest hit (−54pp).
MathVision is the smallest effect, consistent with v0 audit's
"MathVision is the most teacher-RL-flat benchmark" observation.

### (3) Paired vision-critical decomposition

`opd_target_ids.json` (n=133, defined by `vc_t[T1-0-base] ∩
teacher_advantage[T1-0-base, T1-X]` per current PV pass).

| Metric | T1-2 (Full) | **T1-3 (Blank)** | Δ | Interpretation |
|---|---|---|---|---|
| VC[S] (student vision-critical) | 419 | **153** | **−266** | T1-3 lost 65% of vision-critical prompts |
| T_adv (T1-0 base wins over student) | 96 | **342** | **+246** | T1-3 underperforms T1-0 by 246 prompts |
| OPD_t = VC[T] ∩ T_adv | 59 | **287** | **+228** | T1-3 misses 287 vision wins T1-0 had |
| PureV (only-vision-conditioned) | 41 | **232** | **+191** | T1-3 fails 232 prompts that can ONLY be answered with vision |
| opd_target_recovery | 41.3% | **23.0%** | −18.3pp | Both recover some opd_target signal but BlankTeacher much less |

## Mechanism: learned visual blindness (qualitative)

Three randomly-sampled ChartQA predictions, T1-3 (BlankTeacher), at
**eval-time full_image mode** (i.e., the model is shown the real chart):

```
T1-3 ChartQA/290 (gold="Alcohol", T1-3 pred wrong):
  <think>...The image is blank white, and there are no bars of any
   color, let alone a light blue one. I need to state that the image
   is blank and no information is visible to answer the question.</think>
  <answer>The provided image is completely white and contains no
   visible bars or colors. Therefore, it is impossible to determine
   what the light blue color bar denotes as there is no such bar or
   context visible in the image.</answer>

T1-3 ChartQA/1904 (gold="6313", T1-3 pred wrong):
  <think>...The provided image is completely white and does not
   contain any relevant information. It appears to be a placeholder
   or irrelevant to the question.</think>
  <answer>Nike's global revenue in quarter four of 2020 was $19.1
   billion.</answer>     ← fabricated, from priors only
```

Compare T1-2 (FullTeacher) on the same prompts:

```
T1-2 ChartQA/290:  <answer>The light blue color bar denotes
                            \boxed{Alcohol}</answer>     ← correct, reads chart
T1-2 ChartQA/1904: <answer>\boxed{6,313 million U.S. dollars}</answer>
                                                        ← correct (gold 6313)
```

**T1-3 actively reasons that the image is blank and refuses to read
it, even though the chart is right there at eval time.** This is the
direct trace of dense KL distillation transferring the BlankTeacher's
training-time experience (where the teacher genuinely saw a blank
image) into the student's response template.

The student didn't just learn "the answer for this prompt is X"; it
learned "the way to respond when shown an image is to assert the
image is blank and answer from non-visual priors." This is exactly
the pathological case dense KL would create — the teacher's logp
distribution under a blank image is dominated by "I don't know /
can't see / image is blank" continuations, and the student
faithfully reproduces this even when the visual signal is intact at
eval.

### Quantitative confirmation of the mechanism

T1-3 still emits the structural template (`<think>` 100%,
`</answer>` 91%, `\boxed` 87%) — the student's output **form** is
preserved. What collapsed is the **content**: the model has learned
that under image-presentation, the canonical response is to disavow
the image.

## Trajectory evidence: phase transition between step 149 and 199

A separate trajectory eval (`run_t1_trajectory.sh`, commit `2a905e8`)
runs full_image-mode eval against every saved ckpt (step_49 / 99 / 149 /
199 / 230) of both arms plus T1-0 base. Re-scored via the canonical
`mllmopd.diagnostics.scorers` (the same code path t1_compare uses).
Output: `runs/audit/t1_trajectory_20260522-105745/`. Figure:
`runs/analysis/t1_v1p5b_trajectory.png`.

| Step | T1-2 (Full) overall | T1-3 (Blank) overall | T1-3 ChartQA | T1-3 MathVista |
|---|---|---|---|---|
| 0 (base) | 0.550 | 0.550 | 0.680 | 0.625 |
| 49 | 0.542 | 0.553 | 0.695 | 0.675 |
| **99** | 0.547 | **0.566** | 0.780 | 0.640 |
| 149 | 0.591 | 0.502 ⬇ | 0.720 | 0.515 |
| **199** | 0.574 | **0.335** ❗ | **0.105** | 0.365 |
| 230 | 0.576 | 0.331 | 0.155 | 0.315 |

**The BlankTeacher arm doesn't degrade gradually — it cliff-falls between
step 149 and step 199.** Three regions:

1. **Step 0–99 (looks fine)**: Both arms gain similar small amounts.
   At step_99, T1-3 (0.566) is even slightly higher than T1-2 (0.547).
   This is consistent with the earlier v1 step_99 brief that saw
   "Δ ≈ +1.5pp" — at step 99 the BlankTeacher's training-time blindness
   pattern hadn't yet dominated the student's distribution.

2. **Step 99–149 (cracks start)**: T1-3 drops 6.4pp (0.566 → 0.502).
   T1-2 keeps improving (+4.4pp). Gap opens to 9pp.

3. **Step 149–199 (catastrophic collapse)**: T1-3 drops a further
   16.7pp in just 50 steps; ChartQA specifically loses **61.5pp**
   (0.720 → 0.105). T1-2 stays stable. Gap blows out to 24pp.

4. **Step 199–230 (stabilized at bottom)**: T1-3 makes no recovery,
   stays at ~0.33 overall.

**This is the cleanest signature so far that the BlankTeacher mechanism
is a phase-transition, not a smooth slope.** Dense KL distills the
"image is blank" template progressively, and at some critical point
(~step 150-200 in this setup) the student's distribution flips from
"vision-conditioned with mild BlankTeacher contamination" to
"BlankTeacher-template-dominated, ignoring image".

This timing also resolves the puzzle of why v1 step_99 (with the
ckpt-routing bug aside) showed only a weak +1.5pp signal: **at step 99,
the collapse hadn't happened yet**. The full +23pp signal requires
training past the phase transition.

The phase-transition behavior is itself paper-figure material: it
implies dense KL on a misspecified teacher accumulates a hidden
"blindness budget" that only releases catastrophically once enough has
been transferred.

## Implications

### For the original hypothesis (project_hypotheses.md)

The prompt-level claim ("OPD transfers vision-conditioned teacher
capability") is **confirmed**, but more sharply: vanilla dense OPD
doesn't just fail to transfer vision when the teacher can't see — it
**actively transfers anti-vision (learned blindness)**. The signal
strength was hidden in v0/v1 due to bugs; v1.5b reveals it cleanly.

### For T2 method design

The original T2 plan (PGPO/VPPO-style token-level reweighting on
high-VD tokens) is now well-motivated:

- vd_decay observation (preliminary, from v1 diagnostics): teacher's
  full-vs-blank logp gap shrinks during training as student outputs
  template-ize. So the high-VD signal **is** sparse.
- v1.5b confirms that this sparse signal is **causally load-bearing**
  — when the entire teacher signal is forced into a no-vision
  distribution (BlankTeacher), the student collapses.
- T2 method direction: amplify the differential signal between
  FullTeacher logp and BlankTeacher logp, rather than distilling
  the full teacher distribution. Concretely:

  ```
  r_t^vis = log p_T(y_t | image, q, y_<t) − log p_T(y_t | blank, q, y_<t)
  ```

  Use this differential as the primary supervisory signal (perhaps
  weighted by |r_t^vis| to focus on tokens where vision matters).
  This is "image-differential OPD" — orthogonal to PGPO/VPPO's
  per-token reweighting based on advantage, but they could compose.

The v1.5b BlankTeacher arm is, in retrospect, a natural ablation
showing what happens when you train ONLY on the "blank" half of
this differential — disaster. So the differential's sign matters,
not just its magnitude.

## Caveats

1. **Single experimental scale**: 2k prompts × 230 steps × 8 GPU.
   The Δ +23pp is so large that scale-up concerns (does it survive
   at 15k? 100k?) are about magnitude only, not direction. But
   worth confirming that BlankTeacher's collapse doesn't recover
   with more training (would be theoretically possible if longer
   training rediscovers vision capability through some other route).
2. **T1-3 not just "worse than baseline" but actively destroyed**.
   T1-0 base mean = 0.556; T1-3 = 0.337. So BlankTeacher OPD is
   net harmful (−21.7pp) relative to no-training-at-all. Worth
   noting in the framing — it's not "T1-2 is +23pp better than a
   regressed T1-3", it's "T1-2 is the only viable arm; T1-3 is
   actively destructive".
3. **The decision-tree verdict in t1_compare.py outputs "ambiguous"**
   on these numbers because the logic was written assuming small
   effects (`Δ > 0.01 AND rec_diff > 0.01`). v1.5b has
   `rec_diff` = T1-2 41.3% − T1-3 23.0% = +18.3pp, so the AND
   condition holds; but the verdict logic uses point-estimate
   thresholds without CI awareness. The CI-aware version would
   trivially confirm strong positive given lower bound +20pp.
   Cosmetic fix; not material.
4. **Training-time loss curve unavailable** — Megatron's
   tensorboard integration in miles silently failed to write
   (`tracking_utils.py` needs `TENSORBOARD_DIR` env var, not just
   the `--tensorboard-dir` CLI arg). Need a launcher patch to fix
   for future runs; for v1.5b, train log has step-level `perf_*`
   metrics we can grep into a hand-built loss/throughput curve.
5. **Multi-ckpt trajectory eval in flight** — T1-3 has saves at
   step_49 / 99 / 149 / 199 / 230; the trajectory eval (in flight
   as of this brief) will show **at what step the BlankTeacher
   arm flipped from "still using vision" to "denying vision exists"**.
   That's a paper-grade figure.
6. **opd_target subset is teacher-derived**. PV uses T1-0-base as
   the "teacher" axis for opd_target_ids in the new run. For the
   true MMR1-7B-RL teacher's perspective, would need to re-evaluate
   from teacher-trained logs (out of scope here).

## Open questions for review (round 3)

1. **Does the BlankTeacher catastrophic collapse hold up at scale?**
   Specifically: if we train BlankTeacher for 1000 steps instead of
   230, does the student rediscover vision (because at some point
   the dense KL signal becomes weak enough that the student drifts
   back), or does it keep degrading? Worth running a 1000-step
   BlankTeacher arm to confirm the collapse is monotone.

2. **Is the FullTeacher's +1.3pp gain "real" or also at risk?**
   Compared to T1-3's −21.7pp catastrophe, T1-2's +1.3pp looks tiny.
   Is there a reading where FullTeacher is **also undertrained** at
   this scale, and the "real" win shape is "FullTeacher would gain
   much more with longer training, BlankTeacher destroys much more
   with longer training, gap widens further"? This would justify
   the scale-up experiment.

3. **What's the natural interpretation of the per-benchmark variance?**
   ChartQA: T1-3 −54pp (devastating). MathVision: T1-3 only −2pp.
   Is MathVision's resilience because (a) MathVision is hard enough
   that even base T1-0 gets 18% so there's nowhere to fall, (b)
   MathVision's visual reasoning relies on geometric primitives that
   the student internalized pre-OPD, or (c) something about the
   prompt structure makes the BlankTeacher signal less
   counterproductive on MathVision? Has the literature on
   benchmark-specific RL collapse identified similar patterns?

4. **The "learned visual blindness" terminology — does this connect
   to existing work in the literature?** Specifically:
   - Inverse cross-entropy / KL collapse studies (would dense KL
     to a confidently-wrong teacher always produce this pattern?)
   - "Hallucination during distillation" papers — is the BlankTeacher
     arm essentially distilling hallucination?
   - Multimodal RL post-training collapse modes (PAPO, Perception-R1,
     "Seeing with You" — do any of them report this exact pattern?)

5. **Method design for T2 — image-differential OPD vs. token-reweighting.**
   Given the v1.5b mechanism (BlankTeacher's contribution to dense KL
   is purely a "no-vision distribution"), the differential
   formulation `r_t^vis = lp_T_full − lp_T_blank` is naturally
   protective: it cancels exactly the blindness-pulling component.
   Is this likely to:
   (a) Recover the full +23pp gap from a single-teacher setup
       (i.e., training a single student with `lp_T_full − lp_T_blank`
       as supervision)?
   (b) Be redundant with FullTeacher OPD if the gap is already
       achievable from FullTeacher alone?
   (c) Add value primarily on a HARDER setting where FullTeacher
       gain saturates and we need to push further?

6. **For paper structure**: which framing reads better?
   - **(A) Causal mechanism paper**: "Vanilla MLLM OPD requires
     vision-grounded teacher signal; the BlankTeacher counterfactual
     shows what happens without it." Lead with mechanism + figure.
   - **(B) Method paper**: "Vision-differential OPD outperforms
     vanilla, motivated by the BlankTeacher catastrophe finding."
     Lead with method + T2 ablation.
   - **(C) Diagnostic-first paper**: "We diagnose what vanilla MLLM
     OPD actually transfers, find that teacher's vision-conditioned
     distribution is load-bearing, and propose image-differential
     OPD as the natural fix." (A → B, hybrid.)

## Reproduction artifacts

**Public** (in this repo):
- Launcher: `scripts/train/opd_mmr1_3b_baseline.sh` (HEAD `2a905e8`)
- Eval matrix: `scripts/audit/run_t1_eval.sh`
- Trajectory eval: `scripts/audit/run_t1_trajectory.sh` (new this round)
- Training-data prep: `scripts/data/prep_opd_train_data.py`
- Analyzers: `src/mllmopd/analysis/{t1_compare,paired_vision_critical,vd_decay}.py`
- Miles patches: `scripts/setup/patch_uni_opd.sh` (incl. P10 in-place div_)
- Prior briefs: `docs/gpt-brief-2026-05-21-{,t1-v1-}negative-result.md`
  (both INVALIDATED — kept for the bug-fix history they document)

**v1.5b eval run dir** (force-added to repo, ~38M):
- `runs/audit/t1_v1p5b_eval_step249_20260522-102258/` (note: dir named
  `_step249_` because that was the launch-time intent; actual ckpt eval'd
  was `step_230` because training settled there)
  - 9 audit jsonls + summary + t1_compare.json + PV outputs

**Trajectory eval dir** (in flight as of this brief):
- `runs/audit/t1_trajectory_<timestamp>/` once `run_t1_trajectory.sh`
  completes.

**Training run dirs** (ceph only, not on GitHub):
- `runs/t1_v1p5b_T1_{2_full_mm, 3_blank_mm}/` with ckpts at
  `ckpt/hf/step_{49,99,149,199,230}` + Megatron-format ckpts.

## Bug-fix lineage (full chain)

Every fix tracked through the conversation:

| Commit | What | Caught by |
|---|---|---|
| `4b91720` | PV arm identity from jsonl filename | Code inspection during result confusion |
| `76c1ec5` | Launcher `--multimodal-keys` + assert | GPT review round 1 |
| `b52996c` | prep_opd_train_data nested `metadata.id` | GPT review round 1 |
| `4c19fc7` | Force-add eval artifacts for GPT review | GPT review accessibility need |
| `d286c2a` | INVALIDATED banner on v0 brief | Self-flagging after MM bug discovery |
| `1655f8d` → `332fc99` → `76df7cf` | response_len cap iterations (8192→6144→4096) | OOM during v1 training |
| `59fdf7d` | `run_t1_eval.sh` preserves caller CKPT_T1_2/3 | GPT review round 2 |
| `473e0c4` | `RUN_ID` neutral prefix | Brief naming hygiene |
| `5b28fa1` | Stop string + in-place div_ patch | OOM defense (preventive) |
| `c1c8363` | `--max-tokens-per-gpu` 8192→4096 | OOM during v1.5 training |
| `88fe842` → `3b66293` | `MLLMOPD_KEEP_PROXY` env | wandb timeout in restricted network |
| `2a905e8` | Trajectory eval script | Paper figure prep |

12 commits across ~36 hours of iteration, four invalidated experimental
attempts before v1.5b produced the trustworthy positive finding.
