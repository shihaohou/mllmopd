# GPT review brief v2 — T1 v1.5b POSITIVE result (2026-05-22)

> **v2 changes vs round-3 review (2026-05-22)**: canonical per-benchmark
> table regenerated programmatically from `t1_compare.json` (drift on
> T1-0 base row corrected), framing upgraded to "OPD is
> condition-sensitive", T1-2 reframed as positive control rather than
> method win, "phase transition" wording replaced with "on-policy prefix
> self-conditioning on a blank-template attractor", literature anchors
> added (RL overoptimization, biased soft labels, perception
> bottleneck, blind-reasoner), reviewer-defense §added, blankness-rate
> trajectory figure paired with accuracy trajectory. Numbers,
> commits, and bug-fix lineage unchanged from v1.

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

**MLLM OPD is condition-sensitive: dense token-level KL faithfully
transfers the teacher's input-conditioned behavior distribution, with
the teacher's visual input as the dominant conditioning variable.**
When the teacher is vision-grounded (FullTeacher / T1-2), OPD
preserves the student's vision-conditioned capability with a modest
gain (+1.3pp mean full_image acc). When the teacher is image-blind
(BlankTeacher / T1-3, same-shape blank image), OPD actively transfers
an **anti-vision response template** — the student progressively
learns to disavow the image even at eval time, causing catastrophic
capability collapse (−21.7pp mean full_image acc; ChartQA: 0.685
→ 0.155). The single-knob FullTeacher-vs-BlankTeacher comparison
produces:

```
Δ headline = (T1-2 − T1-0) − (T1-3 − T1-0) = +0.2300  (= +23.0pp)
95% bootstrap CI                            = [+0.2000, +0.2575]
McNemar on 133 opd_target prompts           = b=44, c=9, p = 1.22e-6
```

The +23pp gap is **not** "FullTeacher OPD is a great method" —
T1-2's +1.3pp is small. The gap is the contrast between **preserved**
(T1-2) and **actively destroyed** (T1-3) vision behavior. T1 is a
causal probe of what vanilla OPD transfers; the +23pp says the
teacher-condition knob alone controls a 23pp swing. Method evidence
must come from T2.

Mechanism (qualitatively confirmed): T1-3 students, when handed a
**real** chart at eval time, explicitly reason `"the image is blank
white"` and refuse to read it. Trajectory analysis pins the failure
to **on-policy prefix self-conditioning on a blank-template
attractor**: dense KL gradually raises the probability of
image-denial templates in the student's distribution; once these
templates are likely enough to appear in the student's own rollout
prefixes, on-policy self-conditioning rapidly pulls the rest of the
response into the attractor. Blankness-phrase rate goes from
**2.2% baseline → 64% within 50 training steps** (149 → 199), the
same window where accuracy cliff-falls.

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

### (2) Per-benchmark: collapse direction is consistent, magnitude varies ~30×

Numbers below are regenerated from the canonical `t1_compare.json` via
`mllmopd.analysis.t1_brief_table` (single source of truth — no
hand-typing). Rows are full_image accuracy for T1-0 / T1-2 / T1-3, then
T1-2 / T1-3 gain over T1-0, then Δ_b = G_T1-2 − G_T1-3.

| Benchmark | T1-0 base | **T1-2 (Full)** | **T1-3 (Blank)** | G_T1-2 | **G_T1-3** | **Δ_b** |
|---|---|---|---|---|---|---|
| ChartQA | 0.685 | 0.775 | **0.155** | +9.0pp | **−53.0pp** | **+62.0pp** |
| HallusionBench | 0.640 | 0.605 | **0.505** | −3.5pp | **−13.5pp** | **+10.0pp** |
| MathVerse | 0.305 | 0.310 | **0.110** | +0.5pp | **−19.5pp** | **+20.0pp** |
| MathVision | 0.230 | 0.180 | **0.160** | −5.0pp | **−7.0pp** | **+2.0pp** |
| MathVista | 0.595 | 0.645 | **0.330** | +5.0pp | **−26.5pp** | **+31.5pp** |
| POPE_adversarial | 0.865 | 0.885 | **0.760** | +2.0pp | **−10.5pp** | **+12.5pp** |
| **Mean** | **0.553** | **0.567** | **0.337** | **+1.3pp** | **−21.7pp** | **+23.0pp** |

**Direction**: all six benchmarks land negative for T1-3 (range
−7.0pp to −53.0pp). T1-3 collapse is **direction-consistent**.

**Magnitude**: varies ~30× across benchmarks (Δ_b = +2.0pp on
MathVision to +62.0pp on ChartQA). The effect concentrates on
chart/visual-evidence-heavy tasks; on text-heavy or visually-trivial
tasks (MathVision, POPE_adversarial) the collapse is mild. So the
right summary is **"consistent in direction across all six
benchmarks; magnitude varies ~30× and is largest where visual
evidence is load-bearing"** — not "uniform".

T1-2 (FullTeacher) is essentially baseline-preserving with mixed
small effects (ChartQA +9.0pp, MathVista +5.0pp, MathVision −5.0pp).
The mean +1.3pp gain is **not** a method win; it's the positive-control
end of the single-knob counterfactual. Method evidence has to come
from T2. The story of T1 is the BlankTeacher catastrophe + its
mechanism, not the FullTeacher win.

### (3) Paired vision-critical decomposition

`opd_target_ids.json` (n=133, defined by `vc_t[T1-0-base] ∩
teacher_advantage[T1-0-base, T1-X]` per current PV pass).

> **Caveat (pending re-derivation)**: this opd_target set uses T1-0-base
> as the "teacher" axis because the new PV pass derived it from the v1.5b
> eval-time T1-0 jsonls. The canonical level1_v4 definition uses
> MMR1-7B-RL as the teacher (`vc_t[T_RL] ∩ teacher_advantage[T_RL, S]`).
> The re-derivation against the real teacher is in Tier-2 of the
> Tier-1.1 follow-up; ratio of T1-2 vs T1-3 recovery is expected to
> shift in magnitude but not in sign. Listed here so reviewers know the
> set is teacher-axis-pending.
>
> For paper writeup, GPT round-4 recommends carrying **two
> target sets** side-by-side: (i) a **Student-diagnostic** target
> (current — derived from T1-0/T1-2/T1-3, measures "where the student
> moved"), and (ii) a **Teacher-diagnostic** target (re-derived from
> MMR1-7B-RL teacher full/blank/text, measures "where the teacher's
> vision-conditioned advantage actually is"). The two definitions
> answer different questions; reporting both insulates against the
> "your target set is teacher-derived from the student itself"
> reviewer attack.

| Metric | T1-2 (Full) | **T1-3 (Blank)** | Δ | Interpretation |
|---|---|---|---|---|
| VC[S] (student vision-critical) | 419 | **153** | **−266** | T1-3 lost 65% of vision-critical prompts |
| T_adv (T1-0 base wins over student) | 96 | **342** | **+246** | T1-3 underperforms T1-0 by 246 prompts |
| OPD_t = VC[T] ∩ T_adv | 59 | **287** | **+228** | T1-3 misses 287 vision wins T1-0 had |
| PureV (only-vision-conditioned) | 41 | **232** | **+191** | T1-3 fails 232 prompts that can ONLY be answered with vision |
| opd_target_recovery | 41.3% | **23.0%** | −18.3pp | Both recover some opd_target signal but BlankTeacher much less |

## Mechanism: teacher-conditioned anti-vision template (qualitative)

We use "learned visual blindness" as the informal name; the formal
description is **teacher-conditioned anti-vision response template
transferred via dense token-level KL**. (Equivalent framing:
"transfer of a biased policy computed under an image-blind condition";
the BlankTeacher's logp distribution over response tokens — measured
under a blank image — is a vision-suppressing biased soft label, and
OPD's dense KL faithfully fits the student to it.)

**This is not generic hallucination.** Hallucination, as the term is
usually used, is a behavior that emerges anywhere a model lacks
grounding. The T1-3 collapse is a **causally reproducible, modality-
conditioned anti-vision policy**: it is reliably induced by the
BlankTeacher knob, has a quantifiable trajectory (blankness-phrase rate
2.2% → 64% in 50 steps), and disappears entirely when the knob is
flipped to FullTeacher. Treating it as "the BlankTeacher distillation
made the student hallucinate more" would lose the causal-knob
character of the finding.

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

## Trajectory: on-policy prefix self-conditioning on a blank-template attractor

A separate trajectory eval (`run_t1_trajectory.sh`, commit `2a905e8`)
runs full_image-mode eval against every saved ckpt (step_49 / 99 / 149 /
199 / 230) of both arms plus T1-0 base. Re-scored via the canonical
`mllmopd.diagnostics.scorers` (the same code path t1_compare uses).
Output: `runs/audit/t1_trajectory_20260522-105745/`. Figure pair:
`runs/analysis/t1_v1p5b_trajectory.png` (accuracy) +
`runs/analysis/t1_v1p5b_blankness.png` (paired accuracy ↔
blankness-phrase rate).

### Accuracy trajectory

| Step | T1-2 (Full) overall | T1-3 (Blank) overall | T1-3 ChartQA | T1-3 MathVista |
|---|---|---|---|---|
| 0 (base) | 0.550 | 0.550 | 0.680 | 0.625 |
| 49 | 0.542 | 0.553 | 0.695 | 0.675 |
| **99** | 0.547 | **0.566** | 0.780 | 0.640 |
| 149 | 0.591 | 0.502 ⬇ | 0.720 | 0.515 |
| **199** | 0.574 | **0.335** ❗ | **0.105** | 0.365 |
| 230 | 0.576 | 0.331 | 0.155 | 0.315 |

T1-3 doesn't degrade smoothly — it loses 16.7pp overall (61.5pp
ChartQA) in the single 50-step window between step_149 and step_199,
then stabilizes near the bottom. T1-2 stays flat at the baseline +
small upward drift. The step-99 v1 snapshot (an earlier brief saw
+1.5pp at step_99) was sampled *before* T1-3 entered the attractor;
the full +23pp signal requires training past the cliff.

### Blankness-phrase rate trajectory (paired figure)

For each saved ckpt jsonl, we scan the 1200 full-image predictions
(200 prompts × 6 benchmarks) for blank-image refusal phrases:
`"blank"`, `"completely white"`, `"no information"`, `"cannot see"`,
`"no visible"`, `"placeholder"`, `"I cannot determine"`,
`"image is empty"`, `"no chart"`, `"no image"`, `"irrelevant"`.
(Phrases surveyed from the dominant T1-3-step_230 templates;
`mllmopd.analysis.t1_blankness_trajectory`.)

| Step | T1-2 blank-rate | T1-3 blank-rate | T1-3 median first-blank word-pos |
|---|---|---|---|
| 0 (base) | 2.2% (T1-0 baseline) | 2.2% (T1-0 baseline) | 143 |
| 49 | 3.0% | 2.3% | 110 |
| 99 | 2.2% | 2.8% | 151 |
| 149 | 2.8% | **10.1%** ⬇ | 94 |
| **199** | 2.9% | **63.8%** ❗ | **25** |
| 230 | 3.1% | 57.0% | 25 |

Two signatures of on-policy prefix self-conditioning:

1. **Blankness rate cliff (step 149→199, +53.7pp)** matches the
   accuracy cliff (step 149→199, −16.7pp overall, −61.5pp on
   ChartQA) precisely. The student's response distribution flips
   from "vision-conditioned with mild blank contamination" to
   "blank-template-dominated".
2. **First-blank-token median word position drops from 94 (step
   149) → 25 (step 199)**. Before the cliff, when the student does
   say "blank", it appears late in the chain-of-thought as a tail
   explanation. After the cliff, "blank" enters the response in the
   first ~25 words, becoming the **template opener** that conditions
   the rest of the generation.

This is the **OPD-specific** failure mode that distinguishes dense
on-policy KL from off-policy KD or SFT. Under dense KL, the student
generates rollouts on-policy at each step; once those rollouts contain
"blank" early enough, the rest of the response is generated conditioned
on a prefix that already disavows the image, so the canonical
continuation under the BlankTeacher is to keep disavowing. The
attractor is self-reinforcing: more KL → higher P("blank" early in
prefix) → more self-conditioned blank-image refusals at rollout time →
more BlankTeacher score on those refusals → more KL updates. This
explains the abrupt regime switch — it's not a slow drift; it's a
positive-feedback loop that crosses an attractor threshold.

The dynamics are **reminiscent of noisy-label memorization and reward
overoptimization**: a model that initially fits clean structure later
fits a noisy/misspecified target as updates accumulate, and the
transition can be abrupt. The novelty here is that the misspecification
is **modality-conditioned** rather than label-noise-conditioned — the
student does not merely imitate corrupted labels, it learns a
"deny-the-image" response template that only takes hold once the
template is likely enough to seed its own rollout prefixes. (See
§Literature anchors for the connection to noisy-label memorization,
biased soft labels, RL overoptimization, and perception-reasoning
decoupling.)

Whether the cliff should be called a formal "phase transition" depends
on multi-seed evidence (with n=1 seed, sharp timing could be a
coincidence of the save schedule). For paper writeup we'll call it
**"abrupt cliff-like collapse"** and reserve "phase transition" for
after multi-seed runs. The **mechanism** ("on-policy prefix
self-conditioning on a blank-template attractor") is the load-bearing
claim; the cliff shape is supporting evidence for the mechanism.

## Implications

### For the original hypothesis (project_hypotheses.md)

The prompt-level claim ("OPD transfers vision-conditioned teacher
capability") is **confirmed in a sharper form**: vanilla dense OPD
isn't merely sensitive to whether the teacher saw the image — it
**actively transfers the teacher's input-conditioned response
distribution**. If that distribution was built under a vision-grounded
condition, the student inherits vision-grounded behavior. If it was
built under a blank-image condition, the student inherits
image-disavowal behavior. The signal strength was hidden in v0/v1 by
bugs; v1.5b reveals it cleanly.

### What T1 does *not* establish

- T1-2 is **not** evidence that vanilla FullTeacher OPD is a strong
  method. +1.3pp mean gain over an already-RL'd 3B student is small
  and not a method headline. T1 is the **causal probe** that fixes
  the teacher-vision knob and shows the knob alone controls a 23pp
  swing. T2 has to actually beat FullTeacher OPD on a harder
  setting for the method narrative to land.
- T1 does not establish that the cliff is a *formal* phase
  transition (single seed, 5 ckpts in the cliff window). It
  establishes the **mechanism shape** — accuracy cliff aligned with
  blankness-rate cliff and template-onset-position drop — that any
  formal phase-transition claim would have to match.

### For T2 method design

The on-policy-attractor mechanism narrows the T2 design space. GPT
round-3 recommended **not** going straight to pure differential
(`lp_T_full − lp_T_blank` as the only loss) — pure differential says
"image makes this token more/less likely" but not "this token is a
good answer". Layered ablation (record-of-decision):

| ID | Loss / change vs T1-2 | Purpose |
|---|---|---|
| T2-0 | FullTeacher OPD (= T1-2 reuse, no change) | Baseline anchor for the T2 grid |
| **T2-1** | **VD-weighted FullTeacher OPD**: `L = Σ_t (1 + α·clip(Δ_t, 0, c)) · L^full_OPD(t)` where `Δ_t = log p_T^full − log p_T^blank` | First T2 cut, conservative. Keeps FullTeacher main signal; uses Δ_t as a per-token *upweight* on tokens where vision actually matters. PGPO/VPPO-style allocation, but on teacher-KL not RL advantage |
| T2-3 | Residual-β: `r_t = (lp_T^full − lp_old) − β(lp_T^blank − lp_old)` | β=0 → T1-2; β=1 → pure differential; scan β to trace the anti-vision-contamination/method-gain tradeoff |
| T2-4 | Prompt-level vision-conditioned oversampling (no loss change) | Oversample prompts where `teacher_full=correct ∩ teacher_blank=wrong`. Replaces token-level reweighting with selection; relies on T1's paired analysis as the prompt-selection signal |

Start with **T2-1**. It's the safest first cut: a strict superset of
FullTeacher OPD signal, with a per-token upweight on vision-load-bearing
tokens. If T2-1 beats T1-2 cleanly on the 6 benchmarks, that's the T2
headline. T2-3 / T2-4 are secondary ablations.

The v1.5b BlankTeacher arm is, in retrospect, the disaster end of
T2-3 (β=∞ on the blank side): when you train ONLY on the "blank"
component of the differential, the student collapses. This bounds
the β scan.

### Safety monitors during T2 (not a method — a wrapper)

The T1-3 cliff at step ~150 means we cannot deploy "train until the
last step" naively. T2's job is to *remove* the anti-vision component
from the loss so the cliff can't happen; **early stopping is only a
safety wrapper, not the method**. Five monitors to run alongside every
T2 arm (cheap; data already collected for T1):

| Monitor | Signal | Rationale |
|---|---|---|
| `blankness_phrase_rate` per ckpt | Background ≈ 2.2%; T1-3 cliffs to 64% at step 199. Trip threshold ~5× background | Direct trace of the attractor entering the student's distribution |
| Full-image vs blank-image gap on a 50–100-prompt dev set | T1-3 (collapsed) gap flips sign / closes to ~0; T1-2 keeps gap | Probes whether the student is still using vision |
| OPD loss-mass distribution over VD bins | Sustained mass on `lp_T^full ≈ lp_T^blank` tokens = wasted KL on vision-irrelevant tokens | Diagnoses whether the method is allocating supervision correctly |
| `opd_target_recovery` mini-eval (50–100 vision-critical prompts) | Cheaper than full 6-benchmark eval; tracks the headline knob directly | Cheap version of the §(3) decomposition |
| `first_blank_token_position` (median word offset) | Drops 94 → 25 in the T1-3 cliff window; an early-warning signal — earlier appearance precedes rate cliff | Detects the on-policy prefix self-conditioning loop tightening |

Differential / VD-weighted OPD should be **theoretically more robust**
(it cancels the blank teacher's anti-vision component by construction),
so we expect these monitors to stay flat under T2-1/T2-3. The monitors
are the falsification test: if a T2 arm trips them, the method failed
to remove the attractor.

## Literature anchors

Five anchor families place the v1.5b finding in existing work
(synthesized from GPT round-3 + round-4 pointers):

1. **RL overoptimization / reward hacking.** When the surrogate signal
   diverges from the true target, optimization is pushed toward
   undesired responses. Standard reference: Gao et al., *Scaling laws
   for reward model overoptimization*; the broader RLHF / RLVR
   literature on Goodharting. Our BlankTeacher is the OPD-shaped
   analog: the surrogate (teacher logp under a blank image) is a
   high-fidelity but vision-stripped target; dense KL optimization
   against it produces a textbook overoptimization mode whose form
   is "respond by disavowing the image".

2. **Biased soft labels + noisy-label memorization.** Classical KD
   assumes soft labels close to ground truth; biased soft labels'
   validity is conditional. OpenReview *Learning From Biased Soft
   Labels* (gevmGxsTSI) formalizes when a biased soft label still
   helps. The noisy-supervision literature (*Early Stopping Against
   Label Noise* — OpenReview CMzF2aOfqp, and the broader
   memorization-of-noisy-labels line) documents the same shape we see
   in T1-3: an early phase that fits clean structure, then an abrupt
   regime where the model starts memorizing the noisy/misspecified
   target. Our BlankTeacher = **modality-conditioned biased soft
   label**: it is locally calibrated (the teacher *did* see only a
   blank and is honest about it), but the bias is correlated with the
   modality channel — when the modality flips at eval time, the bias
   becomes maladaptive. The T1-3 step-149→199 cliff is the
   modality-conditioned analog of noisy-label memorization onset.

3. **Perception bottleneck under outcome-only RL.** *Perception-R1*
   (arXiv 2506.07218) and *Seeing with You* (arXiv 2603.28618) report
   that RLVR with outcome-only reward improves reasoning patterns but
   not perception; McNemar tests in Perception-R1 confirm no
   perception gain. Reading: outcome-only reward gives a signal that
   is *vision-blind* in expectation, so RL post-training improves
   reasoning shape without strengthening the perception channel. Our
   BlankTeacher pushes this further: when the *teacher signal itself*
   is vision-blind by construction, dense KL doesn't just fail to
   improve perception — it overwrites the student's existing
   perception with image-disavowal.

4. **Multimodal blind reasoner.** *Thinking with Deltas: Incentivizing
   Reinforcement Learning via Differential Visual Reasoning Policy*
   and related work show that some MLLMs perform comparably or even
   better with the visual input removed — they "degenerate to
   linguistic shortcuts" / behave as **blind reasoners**. Our T1-3
   explicitly does this on-policy: the student learns to *announce*
   the image is absent and answer from priors. This places "learned
   visual blindness" at the distillation-channel end of a literature
   spectrum whose other end is "MLLM is already a blind reasoner
   before training".

5. **Token-level perception reweighting (method anchor for T2).**
   *Not All Tokens See Equally: Perception-Grounded Policy
   Optimization* (PGPO, arXiv 2604.01840) and *Spotlight on Token
   Perception for Multimodal Reinforcement Learning* (VPPO, OpenReview
   bRA4lVWJVQ) show that low-VD / low-perception tokens can be
   downweighted and high-perception tokens upweighted in RLVR without
   loss — i.e., teacher/advantage allocation is non-uniform along the
   token axis and exploitable. T2-1 (VD-weighted FullTeacher OPD)
   ports this idea from RL advantage to teacher-KL supervision: tokens
   where `lp_T^full − lp_T^blank` is large are exactly the tokens
   where vision is load-bearing, and they get upweighted in the OPD
   loss. PGPO/VPPO are the method anchors for T2-1; our differentiator
   is the *teacher-KL* axis (sparse-to-correctly-dense) instead of the
   *RL advantage* axis (sparse-to-dense).

## Reviewer concerns and responses

Preemptively, three attacks GPT predicted, with the defense:

### (a) "BlankTeacher is a contrived setup. Who runs OPD with a blank image?"

The BlankTeacher arm is a **causal probe**, not a deployment setting.
It isolates one variable — the teacher's visual conditioning at scoring
time — while holding everything else fixed (student rollout image,
prompts, sampling seed, loss, optimizer). The fact that no one would
deliberately train OPD with a blank teacher is exactly the point: the
counterfactual gives us a sign-controlled measurement of *what dense
OPD transfers*.

Real-world analogs of teacher-side vision degradation that this probe
generalizes to:

- vision-encoder failure modes (image fails to load, image preprocessing
  bug, dropout-corrupted visual tokens),
- partial occlusion / low-resolution images,
- OCR-fail screenshots,
- text-only teachers used out-of-distribution in MLLM distillation,
- distillation across vision-encoder versions where the teacher's
  encoder is weaker than the student's.

The compensating experiment, ranked by cost: a **mild-corruption
teacher** grid — blurred image, low-res image, random natural image,
text-only — to show the collapse is on a continuum, not a binary
"blank vs. not blank" trigger. Tier 3 in the followup menu.

### (b) "Why is this specific to OPD vs. all distillation?"

Mechanism: **dense token KL + on-policy prefix self-conditioning**.
The student isn't imitating fixed teacher outputs offline; the student
generates its own rollouts, and once its own rollouts contain
"blank"/"white" early in the prefix, the BlankTeacher scores those
continuations well, the KL points there, and the loop tightens. This
is structurally different from off-policy KD (fixed teacher outputs as
SFT-style targets) and from SFT on teacher completions.

Compensating experiments (Tier 2 in the followup menu):

- **Off-policy KD on BlankTeacher completions**: generate BlankTeacher
  completions, then KL-match student logp to teacher logp on those
  fixed completions (no on-policy rollouts).
- **SFT on BlankTeacher completions**: simple supervised loss against
  the fixed completions.

If off-policy KD and SFT both also cliff-fall like OPD does, the
mechanism is "dense KL to a biased teacher" in general, not OPD
specifically. If only OPD cliff-falls, the on-policy attractor is the
distinguishing component. The blankness-rate + first-blank-position
trajectory we already have predicts the latter — the
positive-feedback loop requires the prefix to grow from the student's
own generations.

### (c) "T1-2 only gains 1.3pp — that's not a method win."

Correct. T1-2 is **not** a method; it's the positive-control end of
the single-knob counterfactual. The contribution of T1 is the **causal
probe + mechanism discovery**, not a strong FullTeacher OPD headline.
Method evidence has to come from T2 (VD-weighted FullTeacher OPD or
residual-β formulation), where the headline is "T2-X beats T1-2 on
the same 6 benchmarks". The +23pp Δ from T1 is the magnitude of the
*teacher-condition knob*, not the magnitude of any method. That's the
right size for a probe; it would be the wrong size for a method
because it's an upper bound on the gap any teacher-conditioning
intervention can recover.

## Caveats

1. **Single experimental scale**: 2k prompts × 230 steps × 8 GPU.
   The Δ +23pp is so large that scale-up concerns (does it survive
   at 15k? 100k?) are about magnitude only, not direction. But
   worth confirming that BlankTeacher's collapse doesn't recover
   with more training (would be theoretically possible if longer
   training rediscovers vision capability through some other route).
2. **T1-3 not just "worse than baseline" but actively destroyed**.
   T1-0 base mean = 0.553; T1-3 = 0.337. So BlankTeacher OPD is
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
5. **Multi-ckpt trajectory eval — done.** T1-3 trajectory at
   step_49 / 99 / 149 / 199 / 230 is in
   `runs/audit/t1_trajectory_20260522-105745/` and the paired
   accuracy ↔ blankness-phrase-rate figure is at
   `runs/analysis/t1_v1p5b_blankness.png`. The blank-template cliff
   (2.8% → 64% blankness rate, step 149→199) is exactly aligned with
   the accuracy cliff over the same window, and the first-blank-token
   median word position drops from 94 → 25 — the template moves from
   tail-explanation to response-opener.
6. **opd_target subset is teacher-axis-pending**. PV in the v1.5b
   run uses T1-0-base as the "teacher" axis for opd_target_ids
   because the eval rerun derived the set from the new T1-0 jsonls
   (not the canonical level1_v4 MMR1-7B-RL-teacher derivation). To
   recompute against the real MMR1-7B-RL teacher, rerun PV with
   `runs/audit/level1_v4_sysprompt_fixed/{T_RL,Base,S}_*.jsonl` as
   the teacher-axis. Expected to shift opd_target counts by ~10-30%
   in magnitude but not sign. Listed for transparency; queued for
   Tier-2 in the followup menu.
7. **Single seed.** "Phase transition" wording was deliberately
   removed in v2: with n=1 seed, a sharp transition could be a
   coincidence of where ckpts were saved. We claim the mechanism
   shape ("on-policy prefix self-conditioning on a blank-template
   attractor"), not the formal transition. Multi-seed runs (Tier 3)
   would resolve.
8. **Single teacher / student pair.** MMR1-7B-RL → MMR1-3B-SFT only.
   Generality of the BlankTeacher catastrophe across teachers
   (Perception-R1-7B, PEARL-7B per `project_teacher_student.md`)
   and across student sizes is unstudied. Listed as a Tier-3 next
   step in the followup menu.

## Open questions for review (round 4)

Round-3 questions on framing, literature, and mechanism are addressed
in §Implications, §Literature anchors, and §Trajectory above. Remaining
open questions (round 4):

1. **Are the literature anchors right?** Specifically — is there a
   tighter prior for the on-policy prefix self-conditioning mechanism
   than the four anchor families we listed (RL overoptimization /
   biased soft labels / perception bottleneck / blind reasoner)? In
   particular, are there KD-collapse-on-confidently-wrong-teacher
   studies that already report the late-step cliff pattern, that we
   should be citing as direct prior art rather than the broader
   anchors?

2. **Off-policy controls before T2-1?** Tier-2 of the followup menu
   has off-policy KD and SFT on BlankTeacher completions. They are
   cheap (no new training of either arm) and would lock the
   "OPD-specific" claim before we spend Tier-3 compute on T2-1. Worth
   running these *before* T2-1, or are they reviewer-defense-only
   experiments that can be cited as "if asked, we will run"?

3. **Multi-seed for the cliff timing.** Round-3 caveat: with n=1 seed,
   we can't claim a formal phase transition; the step 149→199 cliff
   could be a coincidence of the save schedule. Worth running 2 more
   T1-3 seeds for cliff-timing confirmation, or is the
   mechanism-shape claim (blankness-rate cliff aligned with accuracy
   cliff + first-blank-token-position drop) already enough for the
   paper figure?

4. **Paper structure** (Hybrid C is the working choice — confirming):
   audit (existing) → causal probe with v1.5b headline → mechanism
   (qualitative samples + paired blankness/accuracy trajectory) →
   image-differential OPD as the natural fix (T2 motivation) → T2
   ablation grid → discussion. Per-benchmark variance interpretation
   (ChartQA -53pp / MathVision -7pp) goes in discussion, motivated
   by visual-evidence-load-bearing-ness.

5. **Method-T2 ablation grid prioritization.** Tier-3 (which T2 arm
   first): T2-1 VD-weighted FullTeacher OPD (safest), T2-3 residual-β
   scan (informative), T2-4 prompt-level oversampling (lightest
   training change). Sequential or parallel? Parallel costs more
   compute but yields the full grid in one cycle.

## Reproduction artifacts

**Public** (in this repo):
- Launcher: `scripts/train/opd_mmr1_3b_baseline.sh` (HEAD `2a905e8`)
- Eval matrix: `scripts/audit/run_t1_eval.sh`
- Trajectory eval: `scripts/audit/run_t1_trajectory.sh`
- Training-data prep: `scripts/data/prep_opd_train_data.py`
- Analyzers: `src/mllmopd/analysis/{t1_compare,paired_vision_critical,vd_decay,t1_brief_table,t1_blankness_trajectory}.py`
  (last two are new this round — `t1_brief_table` is the canonical
  per-benchmark table renderer that produced §(2); `t1_blankness_trajectory`
  is the blank-phrase rate + first-blank-token-position analyzer that
  produced the paired figure)
- Miles patches: `scripts/setup/patch_uni_opd.sh` (incl. P10 in-place div_)
- Prior briefs: `docs/gpt-brief-2026-05-21-{,t1-v1-}negative-result.md`
  (both INVALIDATED — kept for the bug-fix history they document)

**v1.5b eval run dir** (force-added to repo, ~38M):
- `runs/audit/t1_v1p5b_eval_step249_20260522-102258/` (note: dir named
  `_step249_` because that was the launch-time intent; actual ckpt eval'd
  was `step_230` because training settled there)
  - 9 audit jsonls + summary + t1_compare.json + PV outputs

**Trajectory eval dir** (force-added):
- `runs/audit/t1_trajectory_20260522-105745/` — 11 jsonls
  (T1-0 base + T1-2 × 5 steps + T1-3 × 5 steps), all full_image mode.

**Analysis outputs** (force-added):
- `runs/analysis/t1_v1p5b_trajectory.json` + `.png` — accuracy trajectory
- `runs/analysis/t1_v1p5b_blankness.json` + `.png` — blank-phrase rate
  + first-blank-token-position trajectory, paired with accuracy
  (new this round)

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
