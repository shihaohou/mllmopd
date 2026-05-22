# Handoff: T1 v1.5b POSITIVE result + GPT round-3 review (2026-05-22)

Self-contained handoff. Read this if you're a fresh session picking up
the project. Everything before this is in `docs/handoff-2026-05-20-*.md`
and the brief lineage.

## TL;DR (where we are)

After 4 invalidated training attempts driven by 2 successive GPT reviews
catching distinct silent bugs (training-side `--multimodal-keys` missing,
then eval-side `source .env` clobbering caller's `CKPT_T1_2/3` export),
**v1.5b is the first trustworthy result**:

```
Δ headline (T1-2 − T1-0) − (T1-3 − T1-0) = +0.2300  (+23.0pp)
95% bootstrap CI                          = [+0.2000, +0.2575]
McNemar on 133 opd_target prompts         = b=44, c=9, p = 1.22e-6
```

The story is **NOT** "FullTeacher OPD lifts the student by 23pp".
T1-2 (FullTeacher) only gains +1.3pp on average. The story is:
**T1-3 (BlankTeacher) catastrophically destroys the student**, dropping
mean full_image acc by −21.7pp (ChartQA 0.695 → 0.155). The +23pp gap
is the contrast between "preserved" and "destroyed", not "boosted".

**Mechanism (qualitatively confirmed)**: T1-3 students, when shown a real
image at eval time, explicitly reason `"the image is blank white"` and
refuse to read it. Dense token-level KL transferred the BlankTeacher's
training-time response template — built when the teacher genuinely saw
a blank — into the student's behavior. We're calling this **learned
visual blindness** (informal; see GPT comments on naming below).

**Trajectory (paper figure)**: T1-3 doesn't degrade smoothly. Both arms
look similar through step_99 (BlankTeacher even slightly higher). Then
T1-3 cracks at step_149 and **cliff-falls** between step_149 and
step_199 (ChartQA loses 61.5pp in 50 steps), stabilizing low at step_230.
This timing perfectly explains why the earlier v1 step_99 brief saw
only +1.5pp signal: the collapse hadn't started yet.

## Where the artifacts live

| Artifact | Path / URL |
|---|---|
| Main brief (paper draft v0) | `docs/gpt-brief-2026-05-22-t1-v1p5b-positive-result.md` |
| Trajectory figure | `runs/analysis/t1_v1p5b_trajectory.png` |
| Trajectory aggregated JSON | `runs/analysis/t1_v1p5b_trajectory.json` |
| v1.5b 9-pass eval | `runs/audit/t1_v1p5b_eval_step249_20260522-102258/` |
| Multi-ckpt trajectory eval | `runs/audit/t1_trajectory_20260522-105745/` |
| T1-2 / T1-3 training run dirs | `${MLLMOPD_RUNS}/t1_v1p5b_T1_{2_full_mm, 3_blank_mm}/` (ceph only; ckpts at step_{49, 99, 149, 199, 230}) |
| Training launcher | `scripts/train/opd_mmr1_3b_baseline.sh` (HEAD `76096b0`) |
| Trajectory eval script | `scripts/audit/run_t1_trajectory.sh` |
| All miles patches | `scripts/setup/patch_uni_opd.sh` (P1-P10, P10 = in-place div_) |

Public repo: `https://github.com/shihaohou/mllmopd`. Brief + eval +
trajectory data are all force-added to `main` so any reviewer (or
future Claude session) can dig.

## GPT round-3 review — synthesis

The full review is in chat history (2026-05-22). Below is the distilled
research-relevant content; tone-down requests and reviewer defenses both
captured.

### Big-picture framing change (accept)

Upgrade the brief's framing:

| OLD (current brief) | NEW (post-GPT) |
|---|---|
| "Vanilla MLLM OPD transfers vision-conditioned teacher capability." | "MLLM OPD is **condition-sensitive**: dense token-level KL faithfully transfers the teacher's **input-conditioned behavior distribution**. When the teacher is vision-grounded, OPD preserves vision-conditioned capability; when the teacher is image-blind, OPD actively transfers **anti-vision behavior**, inducing learned visual blindness." |

This framing is sharper, harder to attack, and directly motivates the T2
method (image-differential OPD).

### Tone down T1-2 claim (accept)

T1-2 +1.3pp is **not** a strong method win. The contribution is the
**causal probe + mechanism discovery**, not "FullTeacher OPD is a great
method". Stop framing T1-2 as method evidence — it's the
positive-control end of a single-knob counterfactual. Method evidence
must come from T2.

### Replace "phase transition / blindness budget" (accept)

"Phase transition" is overclaiming without multi-seed + denser
checkpoints. "Blindness budget" is too informal. New phrasing:

> **BlankTeacher OPD induces an abrupt regime switch via on-policy
> prefix self-conditioning.** Early training preserves the original
> vision-conditioned policy. Dense KL updates progressively raise the
> probability of image-denial templates (`"the image is blank"`,
> `"I cannot see"`, etc.). Once these templates become likely enough to
> appear in the student's own generations, the student's on-policy
> rollouts self-condition on those prefixes, causing a rapid transition
> into a "blank-image refusal" attractor.

Key benefit: this phrasing makes the **OPD-specific** mechanism explicit
(on-policy prefix self-conditioning is what dense token OPD does that
off-policy KD doesn't), which directly preempts the reviewer attack
"why isn't this just generic distillation failure".

### Literature connections (must add to brief)

GPT pointed to four anchor families:

1. **RL overoptimization / reward hacking**: when surrogate signal
   diverges from true target, model is pushed toward undesired
   responses. Standard reference: Gao et al "Scaling laws for reward
   model overoptimization", + RLHF / RLVR literature.
2. **Biased soft labels / KD theory** (OpenReview gevmGxsTSI "Learning
   From Biased Soft Labels"): KD theory typically requires soft labels
   close to ground truth; biased soft labels' validity is conditional.
   Our BlankTeacher = **modality-conditioned biased soft label**.
3. **Perception bottleneck** (Perception-R1 arxiv 2506.07218, Seeing
   with You arxiv 2603.28618): RLVR with outcome-only reward gives
   shared signal that improves reasoning patterns but not perception;
   McNemar tests in Perception-R1 confirm no perception gain.
4. **Multimodal blind reasoner** (Thinking with Deltas, arxiv 2604.xxx):
   models can perform well or better after removing visual input,
   "degenerating to linguistic shortcuts". Our T1-3 explicitly does
   this on-policy.

The "learned visual blindness" name is fine for informal use; for
formal claim consider:
- "**teacher-conditioned anti-vision template**"
- "**modality-conditioned biased soft-label collapse**"

### Brief number inconsistencies (must fix)

The brief's per-benchmark table has hand-typed numbers that diverge
from `t1_compare.json` by ~1pp:
- Brief ChartQA T1-0=0.695; t1_compare.json T1-0=0.685 (canonical)
- Similar drift on MathVision / MathVista / MathVerse

**Action**: regenerate the table from `t1_compare.json` programmatically.
Single source of truth. Don't hand-type numbers ever again.

### Brief "effect is uniform" claim (must fix)

ChartQA collapse −54pp vs MathVision −2pp is a **27× spread**, not
"uniform". Replace with "concentrated on chart/visual-evidence-heavy
benchmarks; the effect is consistent in direction across all six but
varies in magnitude by ~30×."

### opd_target definition (must fix)

Currently `opd_target_ids.json` is derived using T1-0-base as the
"teacher" axis (from new PV pass). The original level1_v4 definition
used MMR1-7B-RL teacher. Reviewer will attack the T1-0-base version.
Recompute with the real teacher.

### Reviewer defense (preempt in brief)

GPT predicted 3 attacks. For each, the response:

1. **"BlankTeacher is contrived"** → It's a causal probe, not a
   deployment setting. Real-world analogs: partial occlusion, low-res
   image, OCR fail, text-only teacher, vision encoder dropout. The
   counterfactual identifies what OPD **transfers**, not what fails
   organically. **Compensating experiment**: mild-corruption teacher
   variants (blurred / low-res / random image / masked-region).

2. **"Why specific to OPD vs all distillation?"** → The mechanism is
   **dense token KL + on-policy prefix self-conditioning**. Student
   isn't imitating fixed teacher outputs offline; student's own prefix
   contains the developing blank template, which then gets reinforced.
   **Compensating experiment**: off-policy KD on BlankTeacher
   completions; SFT on BlankTeacher completions. If OPD is the only one
   that cliff-falls, the on-policy attractor mechanism is identified.

3. **"T1-2 only gains 1.3pp"** → T1 is diagnostic, not method. T2 must
   actually beat FullTeacher OPD on harder/realistic settings.

### T2 method design (accept — concrete ablation plan)

GPT recommends NOT going pure differential (`lp_T_full − lp_T_blank`
alone). Pure differential only says "image makes token more/less
likely", not "token is a good answer". Layered approach:

| ID | Loss | Purpose |
|---|---|---|
| T2-0 | FullTeacher OPD (= T1-2 reuse) | baseline for T2 ablations |
| **T2-1** | VD-weighted FullTeacher OPD: `L = Σ_t (1 + α · clip(Δ_t, 0, c)) · L^full_OPD(t)` where `Δ_t = log p_T^full − log p_T^blank` | first T2 cut, **safest** — keeps FullTeacher main signal + VD-aware allocation. PGPO/VPPO-style but on teacher KL not RL advantage |
| T2-2 | (GPT message truncated here) | TBD |
| T2-3 | Residual β-knob: `r_t = (lp_T^full − lp_old) − β(lp_T^blank − lp_old)` | continuous β knob: β=0 → T1-2, β=1 → pure differential. Trace anti-vision contamination as β scales |
| T2-4 | Prompt-level vision-conditioned sampling: oversample prompts where teacher_full=correct ∩ teacher_blank=wrong | replaces token reweighting with prompt selection; relies on T1's paired analysis |

**Start with T2-1**. It's the most conservative and gives the cleanest
"VD-aware OPD beats FullTeacher OPD" headline if it works.

### Early stopping for T2 (accept — safety monitor only)

T1-3 collapse at step ~150 means we can't deploy "trained until last
step" naively. But T2's job is not early-stopping — T2's job is to
remove the anti-vision component from the loss so collapse can't happen.
Early stopping is just a **safety monitor**, not the method.

Useful monitors:
- `blankness_phrase_rate` over checkpoints (paper figure!)
- full vs blank gap on small dev set (probes vision use)
- VD loss-mass distribution (where is the OPD signal landing)
- opd_target recovery mini-eval (50-100 prompts, cheap)
- first-blank-token-position (does "blank" enter prefix earlier as
  training progresses?)

## Recommended next-action menu (priority + cost)

### Tier 1 (Mac-only, ~30min-1h each, ZERO new training)

1. **Fix brief inconsistencies**: regenerate per-benchmark table from
   `t1_compare.json`, update "uniform" claim, change phase-transition
   wording, add literature anchors, add §"Reviewer concerns and
   responses".
2. **Generate `blankness_phrase_rate` trajectory curve**: grep
   each step ckpt's jsonl for `"blank"` / `"white"` / `"cannot see"` /
   `"no visible"` / `"no information"` etc. Plot vs step. Should show
   the same cliff between step 149-199 as the accuracy curve.
3. **Generate `first_blank_token_position` curve**: for each prediction
   containing a blank phrase, find the token offset where it first
   appears. Plot mean across step. Hypothesis: blank phrases move
   earlier in the response over training, until at step ~150 they
   become the dominant template.

### Tier 2 (H800, no training, ~30min-2h each)

4. **Recompute opd_target with MMR1-7B-RL teacher**: use
   `runs/audit/level1_v4_sysprompt_fixed/{T_RL,Base,S}_*.jsonl` to
   define `vc_t[T_RL] ∩ teacher_advantage[T_RL, S]` as the canonical
   opd_target. Rerun PV on v1.5b eval with this set.
5. **Densify the trajectory**: add eval at step_125 / step_175 by
   resuming training from a step_99 ckpt for short bursts, save at
   125/150/175. Or skip if it requires too much extra training.
6. **Off-policy KD ablation**: cheap controls for "is on-policy the
   key". Generate BlankTeacher completions, then do (a) SFT on those
   completions, (b) off-policy KD with logp matching. If neither
   cliff-falls like OPD, the on-policy attractor mechanism is locked.

### Tier 3 (H800 + training, ~5-6h each)

7. **T2-1 VD-weighted FullTeacher OPD**: first method experiment. Needs
   `dual_teacher_get_reward.py` change to compute `Δ_t = lp_T^full −
   lp_T^blank`, then plug into the OPD loss as a per-token weight.
   Eval against T1-2 baseline on same 6 benchmarks.
8. **Multi-seed for T1-3**: 2 more seeds, same config. Confirms the
   collapse + its timing are not random.
9. **Mild corruption teachers**: instead of pure blank, try:
   - low-resolution image (e.g., 64×64 input)
   - random natural image (substitute another batch's image)
   - text-only teacher (no image at all in payload)
   - vision encoder dropout (zero out random visual token positions)
   Each shows whether the collapse is binary (full-vs-blank) or
   continuous (capability degrades smoothly with teacher's vision
   degradation).

### Tier 4 (later, paper writeup)

10. Paper outline (Hybrid C structure): Audit → Causal probe (T1-2
    vs T1-3 with full v1.5b headline) → Mechanism (qualitative blank
    samples + blankness_phrase_rate curve) → Image-differential as
    natural fix (T2 motivation) → T2 ablation grid → Lit + discussion.
11. Build the multi-seed denser-trajectory paper figure.
12. Reviewer-defense appendix from preemptive responses above.

## Stuff that's resolved and shouldn't bother future-Claude

- `--multimodal-keys` bug: fixed in 76c1ec5, verified by Data statistics
  multimodal: 2000.
- PV `basename(ckpt_path)` collision: fixed in 4b91720.
- prep `metadata.id` schema: fixed in b52996c.
- `source .env` clobbering caller CKPT: fixed in 59fdf7d.
- response_len/packed_cap OOM: settled at response_len=3072 +
  max-tokens-per-gpu=4096 + stop string `</answer>` + in-place div_().
- Bridge config.json `text_config` nesting: manual cp fix per ckpt
  (or use `_fix_config` helper in `run_t1_trajectory.sh` as template).
- wandb in restricted network: `WANDB_MODE=offline` is the safe answer.
- tensorboard silent failure in miles: known issue, needs `TENSORBOARD_DIR`
  env var override; lost for v1.5b runs but not blocking.

## Caveats / open holes

- Single seed. Can't claim phase transition formally yet.
- Single teacher (MMR1-7B-RL → MMR1-3B-SFT). Need at least one more
  teacher/student pair to claim generality.
- 2k training pool. Reviewer will ask about scale.
- `opd_target` is teacher-(re)derived; canonical re-derivation pending.
- T1-2 +1.3pp is not method evidence; T2 must produce it.

## Bug-fix lineage (full chain)

| Commit | What | Caught by |
|---|---|---|
| 4b91720 | PV arm identity from jsonl filename | Code inspection |
| 76c1ec5 | Launcher `--multimodal-keys` + `MLLMOPD_REQUIRE_MM=1` assert | GPT review round 1 |
| b52996c | prep_opd_train_data nested `metadata.id` | GPT review round 1 |
| 4c19fc7 | Force-add eval artifacts for GPT review | GPT review accessibility |
| d286c2a | INVALIDATED banner on v0 brief | Self-flag after MM bug |
| 1655f8d → 332fc99 → 76df7cf | response_len cap iterations (8192→6144→4096) | OOM during v1 training |
| 59fdf7d | `run_t1_eval.sh` preserves caller CKPT_T1_2/3 | GPT review round 2 |
| 473e0c4 | `RUN_ID` neutral prefix `t1_eval_` | Brief naming hygiene |
| 5b28fa1 | Stop string + in-place div_ patch | OOM defense |
| c1c8363 | `--max-tokens-per-gpu` 8192→4096 | OOM mid v1.5 training |
| 88fe842 → 3b66293 | `MLLMOPD_KEEP_PROXY` env | wandb in restricted network |
| 2a905e8 | `run_t1_trajectory.sh` | Multi-ckpt collapse curve |
| 9ce0bbe | v1.5b positive-result brief | Initial brief draft |
| 7b85db7 | Force-add v1.5b eval artifacts | Public review access |
| 76096b0 | Trajectory data + figure + brief §Trajectory | Phase transition observation |

15 commits across ~3 days of bug-iteration → positive result.

## What I'd start with in the next session

If I'm the next Claude session: read this handoff, then **Tier 1.1 +
Tier 1.2** in parallel (~1.5h Mac-only work). Output: a v1.5b brief
v2 with (a) canonical numbers, (b) literature anchors, (c) framing
upgrade, (d) `blankness_phrase_rate` trajectory figure paired with
the accuracy trajectory figure as the paper main-figure pair. Then
push for review round 4 (which should be confirming readiness for
paper draft, not finding bugs).
