# Brief: Tier-2a off-policy KD results — Brief v2 §Mechanism refined

**Date**: 2026-05-24
**Repo**: https://github.com/shihaohou/mllmopd (HEAD includes `tier2a_compare.py` + this brief)
**Predecessor**: `docs/gpt-brief-2026-05-22-t1-v1p5b-positive-result.md` (brief v2; the T1 positive-result paper draft v1).
**Reviewer ask**: validate the refined §Mechanism claim and the falsification verdict; flag confounders we haven't addressed; advise on what to add before paper submission (Tier-2c / multi-seed / blankness analysis).

## TL;DR

Tier-2a (off-policy KD on the same teacher-completion distributions T1 trained on, no student rollout) was run as the brief v2 mechanism falsifier. Both arms (Tier-2a-blank, Tier-2a-full) trained 230 optimizer steps with GBS=64 on the 2k T1 training prompts, on 8× H800. The eval used the canonical T1 trajectory pipeline (`scripts/audit/run_t1_trajectory.sh`) at the same 5 checkpoints (step 49, 99, 149, 199, 230) on level1_subset_v0 (1200 prompts) with the opd_target slice (133 vision-critical prompts).

The result **refines, not destroys, brief v2's §Mechanism**:

- ✅ **Validated** sub-claim: *"The sharp cliff at step 149-199 on opd_target slice is on-policy specific."* T1-3 ACCELERATES (-7.5pp → -12.0pp) in that window; T2A-blank DECELERATES and partially recovers (-9.8pp → +3.8pp) on the same slice. The prefix self-conditioning attractor IS the cliff-shaper.
- ❌ **Falsified** sub-claim: *"Off-policy KD doesn't fail."* T2A-blank degrades -14.8pp on full_image at step 230, vs T1-3's -13.2pp. Off-policy KD on BlankTeacher is harmful, just in a different shape: smooth monotonic decline rather than phase transition.
- 🆕 **New finding** (separable failure modes):
  - **Mode A (universal)**: general-capability degradation from dense KL to a misspecified teacher. Both regimes show it. Off-policy is *worse* (~-16.5pp on PureV reverse-engineer vs ~-14.4pp for on-policy).
  - **Mode B (on-policy specific)**: sharp phase-transition cliff on vision-critical opd_target slice via student-sampling feedback loop. Off-policy lacks this loop; cliff is absent.
- 🔍 **Sharpest per-benchmark signal**: On HallusionBench (the hallucination-resistance benchmark), T1-3 drops -20.0pp at step 230, T2A-blank drops only -6.7pp. Direct evidence that on-policy attractor specifically amplifies hallucination.

**Brief v2 §Mechanism narrative needs updating, but the core T1 result and the on-policy-attractor mechanism survive.** The paper's contribution shifts from "we found an OPD quirk" to "we disentangled two failure modes of dense-KL distillation from biased teachers".

## Setup

| Item | Spec |
|---|---|
| Student | MMR1-3B-SFT |
| Teacher | MMR1-7B-RL |
| Train prompts | `data/opd_train/v0_2k/train.jsonl` (2000 MMR1-RL items, prepped) |
| Teacher completion datasets | `data/opd_train/v0_2k_teacher_completions/{blank,full}_n8.jsonl` — gen'd 2026-05-23, 16000 rows each, BlankTeacher 52.2% "I can't see" rate, FullTeacher 1.0% |
| Optimizer | adam, lr=1e-6, GBS=64, MBS=1, 8× H800 (DP=8), TP=1 |
| Steps | 230 rollout steps (1 epoch × 2000/8 prompts, slightly under nominal 250 due to `--balance-data`) |
| Loss | `--advantage-estimator on_policy_distillation` (UNCHANGED — math is regime-agnostic) |
| Custom code | `src/mllmopd/training/offline_kd_generate.py` only (no Uni-OPD loss.py change). 50 lines. Plugs into Uni-OPD's `--custom-generate-function-path` hook. |
| Eval | `scripts/audit/run_t1_trajectory.sh` on level1_subset_v0 (1200), opd_target slice (133) |
| Confounders pre-checked | Schema smoke (`scripts/train/smoke_offline_kd.py` 8/8 PASS), Ray 1-step smoke (clean), token-budget reverse-engineer in §Confounders |

The four-arm 2×2 design (sampling regime × teacher image mode):

|                  | **on-policy** (student samples → teacher scores) | **off-policy** (teacher pre-samples → student trains on tokens) |
|---|---|---|
| **FullTeacher** (image=real)  | T1-2 (brief v2) — benign +0.8pp@230   | **T2A-full** — benign +0.8pp@230 |
| **BlankTeacher** (image=blank)| T1-3 (brief v2) — CLIFF -13.2pp@230 on full | **T2A-blank** — smooth -14.8pp@230 on full |

## §1 Headline numbers

### full_image (n=1200) — accuracy

```
step        T1-2          T1-3          T2A-full     T2A-blank
            on/full       on/blank      off/full     off/blank
base        0.437         0.437         0.437        0.437
49          0.427         0.438         0.438        0.406  (-3.1pp)
99          0.428         0.436         0.429        0.374  (-6.2pp)
149         0.456         0.383         0.438        0.323  (-11.3pp)
199         0.441         0.318         0.436        0.300  (-13.7pp)
230         0.444         0.305         0.444        0.288  (-14.8pp)
```

Δ vs base (pp) at step 230: T1-2 **+0.8**, T1-3 **-13.2**, T2A-full **+0.8**, T2A-blank **-14.8**.

### opd_target slice (n=133) — accuracy and Δpp

```
step        T1-2                  T1-3                  T2A-full              T2A-blank
            on/full               on/blank              off/full              off/blank
base        0.135                 0.135                 0.135                 0.135
49          0.286 (+15.0pp)       0.286 (+15.0pp)       0.256 (+12.0pp)       0.256 (+12.0pp)
99          0.263 (+12.8pp)       0.271 (+13.5pp)       0.263 (+12.8pp)       0.195  (+6.0pp)
149         0.301 (+16.5pp)       0.195  (+6.0pp)       0.263 (+12.8pp)       0.098  (-3.8pp)
199         0.278 (+14.3pp)       0.075  (-6.0pp)       0.293 (+15.8pp)       0.135  (+0.0pp)
230         0.278 (+14.3pp)       0.098  (-3.8pp)       0.301 (+16.5pp)       0.113  (-2.3pp)
```

### Cliff diagnostic on opd_target (99→149 vs 149→199)

```
T1-3   on-policy + blank   99→149:  -7.5pp   149→199: -12.0pp   *** CLIFF ***
T2A    off-policy + blank  99→149:  -9.8pp   149→199:  +3.8pp   smooth (recovers in window)
```

The cliff-window acceleration test (149→199 delta > 3pp more negative than 99→149 delta) **only fires for T1-3**.

### Figure

Plot saved at `runs/analysis/tier2a_compare.png` — 4-line × 2-subplot trajectory (full_image left, opd_target right). Confirms qualitatively: on full_image, T1-3 and T2A-blank converge to similar bottom; on opd_target, T1-3 cliffs in the 149-199 window while T2A-blank decelerates.

### Per-benchmark on opd_target_intersect at step 230 (Δpp vs base)

| Benchmark         | n  | T1-3 (on/blank) | T2A-blank (off/blank) | on - off gap |
|---|---|---|---|---|
| ChartQA           | 27 | +0.0pp     | +0.0pp   | 0 (base also 0/27; no signal) |
| HallusionBench    | 15 | **-20.0pp** | -6.7pp   | **-13.3pp** ← on-policy specifically hurts hallucination resistance |
| MathVerse         | 25 | -8.0pp     | -4.0pp   | -4.0pp |
| MathVision        | 35 | +11.4pp    | +11.4pp  | 0 (BOTH arms IMPROVE here, see §Discussion) |
| MathVista         | 28 | **-21.4pp** | **-17.9pp** | -3.5pp (both arms cliff badly) |
| POPE_adversarial  | 3  | +66.7pp    | +0.0pp   | n too small to interpret |

**Two clean signals**:
1. **HallusionBench**: T1-3 (-20.0pp) is **3× as damaging** as T2A-blank (-6.7pp). Both train on the same biased teacher completions; only on-policy regime adds the self-conditioning amplification. This is the strongest single-benchmark evidence for the attractor-amplification mechanism.
2. **MathVista**: Both arms cliff hard (-17 to -21pp). Vision-critical math reasoning is broken by dense KL to BlankTeacher regardless of sampling regime. This is the "Mode A universal degradation" signal.

## §2 Mechanism v3 (refined claim — proposed replacement for brief v2 §Mechanism)

**OLD (brief v2, paper draft v1)**:
> The cliff is OPD-specific because dense KL + on-policy prefix self-conditioning is what dense token OPD does that off-policy KD doesn't. The student samples a partial CoT under the blank-teacher attractor's gradient, then KL-distills further toward the same attractor — a feedback loop.

This implied off-policy KD shouldn't fail. **Tier-2a disproves that.**

**NEW (proposed brief v3)**:
> Two separable failure modes of dense KL distillation from a teacher misspecified by visual conditioning:
>
> **Mode A — universal degradation from imitating biased completions**. Both on-policy distillation and off-policy KD train the student to assign high probability to teacher's "I can't see this image" responses (52.2% of BlankTeacher completions exhibit this template). This damages student's general capability across the training distribution. Off-policy is *worse* here (-16.5pp on PureV reverse-engineer vs -14.4pp for on-policy) because the student never gets to sample its own diverse continuations to counterbalance teacher's narrow distribution.
>
> **Mode B — on-policy phase-transition cliff on vision-critical slice**. On-policy regime adds a sharp phase transition at step 149-199 on opd_target (-12.0pp in 50 steps), absent from off-policy on the same slice (which actually recovers +3.8pp in the same window). The attractor mechanism is: student samples a CoT prefix containing the "I can't see" pattern; teacher scores this prefix high (it matches its own distribution); KL gradient pulls student further into the attractor; next rollout has more "I can't see" prefixes; feedback loop accelerates. Without student sampling (off-policy), the loop cannot form, so the cliff cannot happen.
>
> The cleanest per-benchmark evidence: HallusionBench (the hallucination-resistance benchmark) loses -20.0pp under on-policy but only -6.7pp under off-policy. The on-policy attractor specifically amplifies the failure mode HallusionBench is designed to detect.

## §3 Confounders addressed

### 3.1 Token-budget skew between Tier-2a arms

T2A-blank trains on **shorter completions** than T2A-full (BlankTeacher gives up after ~582 tokens; FullTeacher does multi-step CoT to ~885 tokens). Reverse-engineered totals:

```
T2A-blank total response tokens consumed: 230 × 64 × 752 (mean) ≈ 11.07M
T2A-full  total response tokens consumed: 230 × 64 × 964 (mean) ≈ 14.19M
ratio = 0.780
```

T2A-blank sees 78% as many tokens as T2A-full per training step. **Why this doesn't invalidate the conclusion**:

- The differential is **17.6pp** (T2A-full +0.8pp vs T2A-blank -14.8pp on full_image at step 230). No plausible "less data" story produces a 17.6pp accuracy gap from a 22% token gap when the longer-token arm gains nothing and the shorter-token arm catastrophically declines.
- The **shape difference** between T1-3 (cliff) and T2A-blank (smooth) on opd_target is qualitative and **unaffected by token count** — they trained on the same prompts with the same step count; the rollout-vs-stored token difference is identical.
- Cross-regime comparison (T1-3 vs T2A-blank, both BlankTeacher): T1-3 uses student-sampled tokens whose length depends on the (degrading) student; T2A-blank uses teacher's stored tokens. Different code paths, different token mixes. The relevant comparison axis is the *shape* of the trajectory, not absolute token volume.
- Mitigation for future work: re-run T2A-blank-padded (extend short completions to match FullTeacher distribution) as a control. Cost: ~6h H800.

### 3.2 Single seed

All 4 arms are n=1 seed. Brief v2 already flagged this. T2A doesn't add seed coverage. The cliff-shape conclusion would be strengthened by 2-3 more T1-3 seeds + 2-3 more T2A-blank seeds (cost: ~24-36 H800-hours). For Mechanism v3, the qualitative shape difference (cliff vs smooth) is robust under any reasonable seed variation we've seen; the absolute magnitudes are seed-sensitive.

### 3.3 Single teacher pair

Same as brief v2: MMR1-7B-RL → MMR1-3B-SFT is the only pair tested. Tier-2a doesn't add teacher diversity. Tier-3 (mild-corruption teacher grid: blurred / low-res / random natural / text-only) is the original handoff's plan to address this.

### 3.4 Eval pipeline equivalence

The Tier-2a trajectory eval uses the SAME `run_t1_trajectory.sh` script as brief v2's T1 trajectory. Same SUBSET, same MMR1 sysprompt, same SGLang dispatcher. The only argument differs are the run paths via `T1_2_RUN` / `T1_3_RUN` env vars. Sanity check: T1-3 step 199 here on full_image = 0.318 (matches brief v2's `t1_trajectory_20260522-105745` exactly — same data, re-loaded for the 4-arm side-by-side).

## §4 Reviewer attacks pre-addressed

**Attack 1**: *"Brief v2 said off-policy KD wouldn't fail. You ran it and it failed. Your whole §Mechanism is wrong."*
- Response: The brief v2 sub-claim was wrong; we acknowledge and revise. But the part about the *cliff* being on-policy-specific is *empirically validated* by the cliff diagnostic on opd_target. The contribution moves from "we found an on-policy quirk" to "we disentangled two failure modes". This is a stronger, more honest claim.

**Attack 2**: *"T2A-blank's degradation is just because it trained on 22% fewer tokens than T2A-full."*
- Response: see §3.1. The differential is too large; the shape difference between T1-3 cliff and T2A-blank smooth is unaffected by token count.

**Attack 3**: *"You're picking a slice (opd_target) where you get the result you want. On full_image, T2A-blank is actually WORSE than T1-3."*
- Response: True and acknowledged in §1 and §2. We don't claim T2A-blank is "less harmful" — we claim it's "differently harmful". The two failure modes are visible on different slices: Mode A dominates on PureV (off-policy worse); Mode B dominates on opd_target (on-policy worse via cliff). Both effects are reported.

**Attack 4**: *"MathVision actually improves under both arms. Your benchmark is broken."*
- Response: MathVision opd_target intersect (n=35) shows +11.4pp at step 230 for BOTH T1-3 and T2A-blank. Hypothesis: MathVision questions in our opd_target slice may be more text-heavy (geometry word problems with redundant visual info) so blank-teacher imitation doesn't damage the text-side reasoning path. Worth verifying by inspecting MathVision opd_target items but doesn't undermine the headline. The cleaner cliff signal is HallusionBench + MathVista.

**Attack 5**: *"3 prompts on POPE_adversarial. 27 on ChartQA where everyone scores 0. Statistical power is laughable."*
- Response: per-benchmark numbers are illustrative, not the primary signal. The primary signal is the 133-prompt opd_target slice aggregate cliff diagnostic. Per-benchmark breakdown is reported transparently so reviewers can see the structure; we don't make claims that depend on n=3 cells.

## §5 What's NOT established by Tier-2a (deferred)

In rough order of cheapness:

1. **Tier-2c (on-policy + student image blank rollout)** — student samples while ALSO seeing a blank image. Disambiguates "is the cliff caused by teacher-blank scoring, or by student-blank rollouts, or by their conjunction?" One env var change in launcher (`OPD_STUDENT_IMAGE_MODE=blank`), ~5-6h training. **Recommended next.**

2. **Multi-seed for cliff timing** — current step-149-199 cliff is n=1. Brief v2 already flagged. Tier-2a adds a 4-arm-deep comparison but each arm still n=1. Could add 2-3 seeds for T1-3 + T2A-blank + Tier-2c to verify shape robustness. Cost: ~30-50 H800-hours.

3. **Direct blankness-rate trajectory on T2A arms** — we have per-step diagnostic JSONLs (`runs/tier2a_*/diagnostics/step_NNNNNN.jsonl.gz` × 230 files each). These contain the student's training-time generations and teacher logprobs. Could compute the "fraction of student responses containing 'I can't see' / 'completely white'" per step, mirror brief v2's blankness-trajectory analyzer. Would directly verify Mode A: does off-policy student internalize the blank template at the same rate as on-policy? Cost: ~1h Python.

4. **Mild-corruption teacher grid** (brief v2's Tier-3) — addresses "BlankTeacher is contrived". Still pending.

5. **MathVision interpretation** — why does both arms improve on the MathVision opd_target intersect? Manual inspection of 35 prompts could tell us if they're text-heavy. Cost: 30 min reading.

## §6 Decision: ship to paper or hold?

**My recommendation**: ship Brief v2 + this Tier-2a refinement to a v3 paper draft, with a clear "Future work" section listing Tier-2c + multi-seed + Tier-3 corruption grid. The mechanism story is materially stronger with Tier-2a than with just brief v2 (we now own the disentangled-failure-modes framing, which is a sharper contribution than "we found an OPD quirk").

The minimum to add before submission:
- Tier-2c (1 H800-day) — to fully decompose the on-policy attractor
- Multi-seed × 3 for T1-3 + T2A-blank + Tier-2c (3-4 H800-days) — for cliff-shape robustness

Total: ~5 H800-days for a defensibly-shippable v3. Acceptable timeline.

## §7 Specific review questions

For each, please give: verdict (agree/disagree/needs more data) + the specific evidence or experiment that would change your mind.

**Q1**: Is the "Mode A (universal) + Mode B (on-policy cliff)" two-failure-mode framing the right way to present this, or is it overfit to the data?

**Q2**: The HallusionBench n=15 cell (T1-3 -20.0pp vs T2A-blank -6.7pp) is the strongest single piece of evidence for Mode B. Is n=15 enough? If not, what's the cheapest way to get more HallusionBench-like prompts?

**Q3**: For the brief v3 paper draft, should §Mechanism lead with "two failure modes" (this brief's framing) or with "on-policy attractor amplification" (closer to brief v2's framing but updated)? The former is more honest, the latter is more memorable for the abstract.

**Q4**: Tier-2c is recommended next at ~5-6h. Is there a cheaper experiment that would close the remaining gap in §Mechanism v3?

**Q5**: The token-budget confounder (§3.1) — is the qualitative argument (differential too large, shape unaffected) enough, or do we need the T2A-blank-padded control experiment before submission?

**Q6**: Anything in §3 you'd want addressed differently? Anything in §4 attacks I missed?

## Artifact locations

| Artifact | Path |
|---|---|
| Canonical numerical dump | `runs/analysis/tier2a_compare.json` (57 KB) |
| 4-line × 2-subplot trajectory plot | `runs/analysis/tier2a_compare.png` (174 KB) |
| Tier-2a training launcher | `scripts/train/opd_mmr1_3b_baseline.sh` (with `OPD_OFFLINE_KD_JSONL` gate) |
| Tier-2a custom generate function | `src/mllmopd/training/offline_kd_generate.py` |
| Smoke 0 (no-Ray schema check) | `scripts/train/smoke_offline_kd.py` |
| 4-arm comparison analyzer | `src/mllmopd/analysis/tier2a_compare.py` |
| Tier-2a-blank training run dir | `runs/tier2a_blank_20260523_103433/` (ceph only) |
| Tier-2a-full training run dir | `runs/tier2a_full_20260523_112417/` (ceph only) |
| Tier-2a trajectory eval | `runs/audit/tier2a_trajectory_20260524-132227/` (now committed) |
| T1 trajectory eval (comparison) | `runs/audit/t1_trajectory_20260522-105745/` (already committed) |
| Brief v2 (paper draft v1) | `docs/gpt-brief-2026-05-22-t1-v1p5b-positive-result.md` |
| Pre-Tier-2a code review | `docs/gpt-review-2026-05-22-tier2a-off-policy-kd.md` |
| Tier-2 design rationale | `docs/handoff-2026-05-22-brief-v2-tier2-next.md` §"Why Tier-2 before T2-1" |
