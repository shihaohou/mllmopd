# GPT review brief — T1 negative result (2026-05-21)

Three-layer evidence converges on a negative answer for the "vision-conditioned
capability transfer" hypothesis. Asking for a second-opinion sanity check on
the design, the interpretation, and the open questions before we lock the
result in.

## TL;DR

- Trained MMR1-3B-SFT student with vanilla dense OPD from MMR1-7B-RL teacher.
- Two arms: **T1-2 FullTeacher** (teacher sees image during rollout) vs **T1-3
  BlankTeacher** (teacher gets blank image — negative control).
- After 250 steps (2k prompts × 1 epoch, RBS=8), evaluated on 6 benchmarks ×
  200 prompts × {full / blank / text_only} modes.
- **Δ (T1-2 − T1-3) ≈ 0 in every metric we measured.** Both arms gained the
  same ~+0.8pp over baseline on full_image, both arms recovered ~40-45pp of
  the opd_target subset, both arms lost half of teacher's high-VD token share.
- **OPD does transfer ability** (+40-45pp recovery on the 133 prompt
  vision-critical subset — large effect), **but the transfer content is not
  the teacher's visual processing** (BlankTeacher transfers the same amount
  without seeing the image).

## Hypothesis (the thing we're testing)

Working hypothesis was a two-layer claim about vision-conditioned capability
transfer:

1. **Prompt-level**: in level1_v4 audit, 73% of MMR1-7B-RL's wins over
   MMR1-3B-SFT are concentrated on prompts where the image is causally
   load-bearing (`vision_critical[T] ∩ teacher_advantage`).
2. **Token-level**: only ~7% of teacher tokens land in high-VD bins, but
   dense OPD KL covers 100% of tokens → the dense loss might still pick up
   the sparse visual signal, OR might wash it out under output-length /
   reasoning-template noise.

T1 was designed to directly test the **causal claim** behind (1): if OPD
transfers vision-conditioned capability, blanking the teacher's image input
during rollout should hurt the transfer. The BlankTeacher arm is the
negative control.

## Experimental design

- Teacher: MMR1-7B-RL (frozen, served via sglang on a separate GPU)
- Student: MMR1-3B-SFT (Megatron + Uni-OPD trainer, ZeRO-1, dynamic batch
  cap 8192 tokens/GPU, 4 trainer GPUs, GBS=64, MBS=1)
- Loss: vanilla on-policy distillation (dense KL with teacher logits;
  kl_loss_coef=0, entropy_coef=0, OPD clip range 10.0)
- Training pool: MMR1-RL 2k-prompt subset (data/opd_train/v0_2k/train.jsonl)
- Epochs: 1; 250 optimizer steps; 5 ckpts saved (step 49/99/149/199/249)
- **Single knob differs between arms**: `OPD_TEACHER_IMAGE_MODE = full | blank`.
  In `blank` mode the teacher receives a same-shape solid-color image during
  rollout; everything else (prompts, samples, loss, optimizer, RNG seed
  ordering) is identical.
- Eval: 9-pass matrix on level1_subset_v0 (200 prompts × 6 benchmarks
  × {T1-0 base, T1-2, T1-3} × {full / blank / text_only}). 4 trainer
  improvement metrics: G_full, G_blank, G_text, G_gap = G_full − G_blank,
  blank_leakage = G_blank / G_full, opd_target_recovery.

## Headline results

### (1) Prompt-level: Δ ≈ 0 with wide CI

```
HEADLINE Δ = (T1-2 − T1-0) − (T1-3 − T1-0) = +0.0025
95% bootstrap CI                            = [−0.0158, +0.0225]   ← crosses 0
McNemar on 133 opd_target prompts           = b=17 vs c=11, p=0.345  ← NS
verdict (decision tree)                      = "G_gap flat"
```

Per-benchmark mean improvement vs T1-0 base (full_image):

| Benchmark | T1-0 | T1-2 (Full) | T1-3 (Blank) | T1-2 − T1-0 | T1-3 − T1-0 |
|---|---|---|---|---|---|
| ChartQA | 0.685 | 0.750 | 0.765 | **+6.5pp** | **+8.0pp** |
| HallusionBench | 0.625 | 0.645 | 0.620 | +2.0pp | −0.5pp |
| MathVerse | 0.310 | 0.300 | 0.285 | −1.0pp | −2.5pp |
| MathVision | 0.195 | 0.165 | 0.175 | −3.0pp | −2.0pp |
| MathVista | 0.610 | 0.645 | 0.630 | +3.5pp | +2.0pp |
| POPE_adv | 0.880 | 0.865 | 0.880 | −1.5pp | 0.0pp |
| **Mean** | 0.551 | 0.562 | 0.559 | **+1.1pp** | **+0.9pp** |

Both arms move by similar tiny amounts. Math benchmarks slightly regress on
both arms; ChartQA / MathVista improve on both; HallusionBench and POPE flat.

### (2) opd_target subset: large recovery, but symmetric

`opd_target` = the 133 prompts (per benchmark) where MMR1-7B-RL teacher wins
on `full_image` AND base T1-0 misses AND the teacher's win is
vision-conditioned (`vc_t ∩ teacher_advantage` in level1_v4 audit).

`opd_target_recovery[arm,b]` = `Acc[arm, full_image, opd_target_ids(b)]` − 0
(T1-0 is 0% on opd_target by construction).

| Benchmark | n_target | T1-2 recovery | T1-3 recovery | T1-2 − T1-3 |
|---|---|---|---|---|
| ChartQA | 27 | 81.5% | 81.5% | 0.0pp |
| HallusionBench | 15 | 53.3% | 40.0% | +13.3pp |
| MathVerse | 25 | 40.0% | 32.0% | +8.0pp |
| MathVision | 35 | 20.0% | 20.0% | 0.0pp |
| MathVista | 28 | 42.9% | 35.7% | +7.2pp |
| POPE_adv | 3 | 33.3% | 33.3% | 0.0pp |
| **Mean** | 133 | **45.2%** | **40.4%** | **+4.8pp** (NS, p=0.345) |

This is what makes the result interesting: **OPD does transfer SOMETHING that
boosts the student by ~40-45pp on prompts where the teacher's win is
vision-conditioned.** But FullTeacher and BlankTeacher transfer the same
amount.

### (3) Paired vision-critical decomposition

Fresh PV pass on the new eval grid (with a basename-collision fix; T1-2 and
T1-3 both save HF as `ckpt/hf/step_249/` so the old PV path silently merged
them into a single record). Teacher = T1-0 base, Student = T1-X.

| Metric | T1-2 | T1-3 | Δ |
|---|---|---|---|
| VC[T] (teacher vision-critical) | 400 | 400 | 0 |
| VC[S] (student vision-critical) | **410** | 400 | +10 |
| T_adv (teacher wins, student misses) | 100 | 101 | −1 |
| OPD_t = VC[T] ∩ T_adv | 69 | 70 | −1 |
| PureV (T_full✓ T_blank✗ T_text✗ ∩ T_adv) | 55 | 57 | −2 |

T1-2 has marginally more vision-critical prompts itself (+10), but
identical T_adv / OPD_t / PureV → no measurable difference in how each arm
inherits the teacher's vision-critical structure.

### (4) Token-level VD shift

Each student forward-decoded its OWN predictions on the 133 opd_target
prompts; per-token VD computed as `NLL_blank − NLL_full`. Bin: `high+very_high`
= VD > 0.5.

| Scope | Teacher baseline | T1-2 | T1-3 | T1-2 − T1-3 |
|---|---|---|---|---|
| Overall % NLL mass in high+VH | 21.10% | 13.17% | 13.76% | **−0.60pp** |
| Overall % tokens in high+VH | 6.84% | 3.34% | 3.62% | **−0.28pp** |
| Total tokens emitted | 96,159 | 155,376 | 137,215 | T1-2 +13% |
| Absolute high+VH token count | ~6,577 | ~5,189 | ~4,966 | T1-2 +223 |

Both arms lost ~half of teacher's high-VD signal share. T1-2 vs T1-3 difference
flips sign depending on metric (percent vs absolute), and the magnitude is
within ~5% of either arm's value — well below any reasonable significance
threshold given the n=133 sample.

**Output length drift caveat** (script header self-flagged this): T1-2 emits
+13% more tokens than T1-3, which can shift bin percentages independent of
"internal visual reliance". The conclusion holds in raw counts too, but it's
worth noting that the arms are not output-isomorphic.

## What this means for the hypothesis

The vision-conditioned capability transfer hypothesis predicts:

> If we run OPD with teacher seeing the image, the student should pick up
> vision-conditioned capabilities (full_image gain > blank_image gain by some
> margin). If we run with teacher seeing a blank image, that vision-conditioned
> slice of the gain should disappear.

The data says:

> Both arms produce nearly identical students. The transfer that DOES happen
> (~40-45pp recovery on opd_target) is invariant to whether the teacher saw
> the image. So the "what's transferred" is not the teacher's visual processing
> — it's something the BlankTeacher also has (text format, scaffolding,
> reasoning template, output style, etc.).

Three-layer convergence (prompt, subset, token) → the conclusion feels robust
at this experimental scale.

## Caveats / things to attack

1. **Training scale**: 250 steps × 2k prompts × 1 epoch is small. Maybe the
   FullTeacher's vision signal is high-variance per-step and 250 steps isn't
   enough to discriminate. T2 plan was conditional on T1 showing positive
   signal — but a larger T1 ablation (5k-15k pool, 1-2 epochs) could
   strengthen the claim either way.
2. **Eval n is small**: 200 prompts per benchmark / 133 opd_target total.
   CI on Δ is ±1.6pp / 2.3pp — wide enough to hide an effect smaller than 2pp.
3. **opd_target definition is teacher-centric**: it's built from MMR1-7B-RL's
   vision wins on level1_v4_sysprompt_fixed. We never re-validated that this
   set is still meaningful for MMR1-3B-SFT's blind spots.
4. **VD-bin token-level noise**: output length drift between arms (+13% T1-2
   vs T1-3) means bin percentages aren't direct comparisons. The conclusion
   holds in raw counts but the cleanest version of this test would force same
   output length.
5. **One single teacher choice**: MMR1-7B-RL is the "+6.8pt average over
   3B-SFT" teacher we picked because its wins were concentrated on visual
   prompts. A different teacher (e.g., PEARL-7B / Perception-R1-7B / a much
   stronger frontier model) might tell a different story.
6. **The decision-tree verdict ("G_gap flat") is point-estimate based.**
   We discussed but didn't apply a CI-aware verdict (`ci_lo > 0` for strong
   positive, `ci spans 0` for null). This is cosmetic given the current data
   — both verdicts agree on "no signal" — but should be added before paper
   numbers go anywhere.

## Open questions for review

1. **Is the conclusion underpowered?** Given CI ±1.6/+2.3pp on the headline,
   what would be the minimum effect size we could detect, and is that small
   enough that the "true Δ is small but nonzero" alternative is still alive?
2. **Is opd_target the right subset?** Should we re-derive the
   vision-critical mask from MMR1-3B-SFT's own blind spots rather than
   inheriting from level1_v4's teacher-side construction?
3. **Is the +40-45pp recovery on opd_target itself surprising?** If both
   arms recover ~equal amounts, what's the next-most-plausible candidate for
   "what's actually transferring"? Reasoning template? `<think>/<answer>`
   scaffolding? Sub-task vocabulary? How would we test which?
4. **Does the token-length drift suggest something interesting?** T1-2
   emits +13% more tokens than T1-3. Is that itself a signal of "FullTeacher
   distilled longer reasoning chains" or just RNG noise / inference cap drift?
5. **Should we go to T2 (PGPO/VPPO-style token-level reweighting)?** The
   PV decomposition shows OPD_t is 69-70 prompts out of 1200 (~5.8%) — small
   fraction. Reweighting toward those tokens might either (a) be the key to
   making the hypothesis work, or (b) be redundant because the gain isn't
   vision-conditioned anyway. Which way does the evidence point?
6. **What's the writeup framing?** Is this best presented as
   "vanilla OPD transfers reasoning template, not vision capability" (a
   constructive negative), or as "OPD on MLLM needs different design" (a
   motivational negative pointing at T2 / future work)?

## Reproduction artifacts

All on H800, ceph-shared:

- Training launcher: `scripts/train/opd_mmr1_3b_baseline.sh` (HEAD `4b91720`)
- Eval matrix script: `scripts/audit/run_t1_eval.sh`
- T1-2 ckpt: `${MLLMOPD_RUNS}/t1_v0_T1_2_full/ckpt/hf/step_249/`
- T1-3 ckpt: `${MLLMOPD_RUNS}/t1_v0_T1_3_blank/ckpt/hf/step_249/`
- Eval run dir: `${MLLMOPD_RUNS}/audit/t1_v0_eval_20260521-140736/`
  - `T1_{0,2,3}_{full,blank,text_only}.jsonl` — 9-pass eval outputs
  - `summary.json` — aggregate accuracy table
  - `t1_compare.json` — headline Δ + bootstrap CI + McNemar
  - `paired_vision_critical_T1_{2,3}_vs_T1_0.txt` — fixed PV outputs
  - `vd_summary.T1_{2,3}.json` — token-level VD distributions
  - `vd_shift_t1.json` — VD-shift comparison vs teacher baseline
- Code fix that surfaced during this analysis: `4b91720` — PV used
  `basename(ckpt_path)` as model ID; T1-2 and T1-3 both saved HF as
  `step_249`, collided, silently merged. Fixed by deriving arm label
  from jsonl filename stem.

## Notes on the OOM saga (separate)

T1-2 went through 14 OOM iterations before training-stable config landed —
see `docs/handoff-2026-05-20-oom-resolved.md` for details. The root cause
was `--use-dynamic-batch-size --max-tokens-per-gpu 16384` packing 9.3 GiB
fp32 logits per CE call; halving to 8192 + `--use-distributed-optimizer` +
sglang `mem_fraction 0.15` resolved it. Both T1-2 and T1-3 final runs used
the resolved config; no OOM concerns affect the result.
