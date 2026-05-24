# GPT review brief — T2-1 counterfactual sweep (round-5)

**Date**: 2026-05-24 (same day as round-4; continued in-session)
**Repo**: https://github.com/shihaohou/mllmopd (HEAD = `79a3d69`)
**Reviewer ask**: independent guidance on which (if any) T2-1 weight
variant to train next given a 6-variant offline counterfactual
sweep that found **no variant passes all 4 of our pre-registered
pass criteria**, but a clear Pareto trade-off between energy
stability and signal targeting.

Round-4 reply (Chinese, terse) is in conversation history; key
quotes:

> 下一步就是 T2-1Abs，不要先做 NegAware
> 如果 T2-1Abs 仍然不行，再考虑：Abs + RMS-preserve / quadrant-aware /
> hard top-k by |vd|

We did exactly that, plus per-GPT-round-4 fallbacks. Result: clear
Pareto trade-off, no single all-pass variant. Asking round-5 to
decide next action.

## 30-second recap of round-4 → today

1. A0 audit: rho_l2=0.973 (LR confound REFUTED), conditional_supp=
   0.997 (signed-proxy mis-target STRONGLY CONFIRMED — 100% of
   visual-rejection correction mass routed into suppress branch).
2. Per-prompt diff: LOSE concentrated on long-CoT math (MathVerse
   −4, MathVision −2) and visual-rejection adversarial (POPE −2);
   WIN on visual QA (ChartQA +2, HallusionBench +2, MathVista +2).
3. Implemented sidecar dump for old_log_probs + 6-variant
   counterfactual (no training, recompute weights on existing
   pilot diagnostics: 576 samples / 935814 tokens, 10 steps × 8
   dp ranks).

## 4 pass criteria (pre-registered, possibly too strict)

```
frac_supp_neg_vd_neg_adv_mass < 0.05
vis_reject_correction_mean_weight >= 1.0
corr(w, |adv|) >= -0.05
rho_l2_pooled < 1.5
```

## The 6-variant result

```
variant                  rho_l2    corr  frac_supp  cond_supp  visR_mw  passes (4-tuple, '/'-joined)
signed                    0.973  -0.101      0.442      0.997    0.936  ✗/✗/✗/✓
abs                      11.981  +0.308      0.124      0.279    1.901  ✗/✓/✓/✗
abs_rms_preserve          5.990  +0.308      0.204      0.459    0.950  ✗/✗/✓/✗
abs_rms_preserve_wide     1.317  +0.326      0.368      0.829    0.219  ✗/✗/✓/✓
abs_max_clip              1.611  +0.471      0.124      0.279    0.824  ✗/✗/✓/✗
abs_max_clip_renorm       3.802  +0.428      0.072      0.163    1.845  ✗/✓/✓/✗
```

(`pass-order: frac_supp_lt_0.05, vis_reject_mean_w_ge_1, corr_ge_-0.05, rho_l2_lt_1.5`)

Variant definitions:
- `signed`: production T2-1 (PGPO Eq 6+7 on signed `vd = lp_full-lp_blank`)
- `abs`: vd → |vd|, otherwise unchanged
- `abs_rms_preserve`: abs × per-sequence scalar `s = clip(sqrt(Σadv²/Σ(w·adv)²), 0.5, 2.0)`
- `abs_rms_preserve_wide`: same but clip = `(0.1, 10.0)`
- `abs_max_clip`: abs then clip max weight to 2.0 (no renorm — Σw < N)
- `abs_max_clip_renorm`: clip then mass-preserve renormalize

Code: `src/mllmopd/analysis/t2_1_abs_counterfactual.py @ 79a3d69`.
Smoke tests: 10/10 (audit) + 6/6 (counterfactual) pass.

## Pareto trade-off observation

The 6 variants split into three Pareto groups:

**Group A — strong signal recovery but high energy:**
- `abs`: rho_l2=12, mean_w=1.90, cond_supp=0.28 (4x improvement vs signed 0.997)
- `abs_max_clip_renorm`: rho_l2=3.8, mean_w=1.85, cond_supp=**0.16** (6x improvement)

**Group B — bounded energy but compromised targeting:**
- `abs_max_clip`: rho_l2=1.61 ✓, but mean_w=0.82, cond_supp=0.28
- `signed`: rho_l2=0.97 ✓, but no targeting (corr=−0.10, cond_supp=0.997)

**Group C — pathological energy compensation:**
- `abs_rms_preserve` (clip [0.5,2]): partial — rho_l2 only halved to 6
- `abs_rms_preserve_wide` (clip [0.1,10]): rho_l2 finally drops to 1.32 ✓
  but at horrible cost — scalar << 1 across many sequences shrinks
  ALL weights (mean_w plunges to **0.22**, cond_supp rebounds to **0.83**,
  frac_supp jumps to **0.368**). RMS-preserve is fundamentally wrong for
  OPD: when |adv| is large where w is moderately above 1, the
  scalar squashes everything, including the boost we wanted.

## Two structural findings

### Finding 1: `frac_supp` has a ~7% floor

Across all abs-variants, frac_supp_neg_vd_neg_adv_mass cannot get
below ~7% with the current PGPO formula (τ=0.4, β=2.0). Even
abs_max_clip_renorm (the best) lands at 0.072. Mechanism: tokens
with very small |vd| AND vd<0 (weak visual rejection) end up at
the bottom of the |vd| min-max distribution → vd_norm < τ → suppress
branch by construction. We can't move them above τ by changing the
boost-side formula; we'd need to change τ, β, or the piecewise
shape itself.

→ Question for GPT: is this floor a fundamental
PGPO-on-OPD limitation, or can it be resolved by parameter sweep
(smaller τ, smoother boost) or a new functional form?

### Finding 2: energy bounding and signal targeting are anti-correlated in this design space

Direct evidence: comparing variant Pareto front, every method that
brings rho_l2 toward 1 (RMS-preserve_wide; max_clip without renorm)
loses targeting in some way. Methods that recover targeting
(abs_max_clip_renorm) push rho_l2 back up.

This may be an algebraic fact about PGPO Eq 7 on OPD's signed
token-specific advantages — could not be fixed without breaking
mass preservation, threshold-gated piecewise structure, or both.

## Two trainable options if criteria relaxed

**Option A — abs_max_clip_renorm + reduced LR**:
- Strongest signal recovery: cond_supp=0.16, frac_supp=0.072, mean_w=1.85, corr=+0.43
- rho_l2=3.8 → reduce LR by ~3.8x: 1e-6 → 0.3e-6 to absorb energy
- Effective gradient magnitude matches T1-2 baseline
- Risk: 3.8x weight variance may break optimizer (Adam EMA assumes
  bounded gradients)

**Option B — abs_max_clip + maintain LR=1e-6**:
- Bounded energy: rho_l2=1.61, only 1.6x baseline
- Weaker signal recovery: mean_w=0.82, cond_supp=0.28
- No optimizer-level risk
- May not improve over T1-2 measurably

## Three specific asks for GPT round-5

### Q1: are the 4 pass criteria too strict?

I (Claude/Shihao) picked them ad-hoc:
- `frac_supp < 0.05`: aspirational "near zero"; structural floor is ~0.07
- `mean_w ≥ 1`: requires vis_reject_correction quadrant to be BOOSTED
- `corr ≥ -0.05`: weak negative correlation tolerable
- `rho_l2 < 1.5`: 50% energy growth tolerable

GPT round-4 (your previous reply) suggested:
- frac_supp ≈ 0 (idealistic)
- mean_w > 1 (without explicit threshold)
- corr ≥ 0 (you said "非负")
- rho_l2 "not extreme" (no number)

If we relax to:
- frac_supp < 0.10
- mean_w ≥ 1
- corr ≥ 0
- rho_l2 < 2.0

Then `abs_max_clip_renorm` passes 3/4 (still rho_l2=3.8 fail) and
`abs_max_clip` passes 2/4 (frac_supp 0.12 + mean_w 0.82 fail).

Suggested practical cutoffs from your perspective?

### Q2: abs_max_clip_renorm + LR drop — defensible hack?

Per-token gradient g_t = w_t · adv_t · ∇log_π(token_t). Optimizer
sees Σg_t per step. If rho_l2 = sqrt(Σg² / Σadv²·∇²) = 3.8 and we
scale LR by 1/3.8, effective step size matches T1-2.

But:
- Adam's second-moment estimator sees the ratio differently (it's
  per-parameter, not per-token)
- The bias/variance trade-off changes: each step is now smaller but
  noisier
- Convergence rate may slow

Is this a defensible engineering choice, or are there cleaner ways
to absorb the variance (clip per-token grad norm, e.g.)?

### Q3: pivot opportunity?

T1 brief v2's load-bearing claim ("on-policy attractor mechanism")
remains untested by a Tier-2 off-policy KD control
(`docs/handoff-2026-05-22-brief-v2-tier2-next.md`). The T2-1
result has now produced a clear structural insight ("RLVR-style
signed VD reweighting mis-targets OPD's visual-rejection
correction") regardless of whether T2-1Abs* succeeds.

Three branching paths from here:

(i) Train `abs_max_clip_renorm + LR=0.3e-6` for 230 steps (3-4h);
    if it beats T1-2 on opd_target_recovery, T2-1 has a positive
    method-tier result and we can write up.

(ii) Add 1-2 more variants (smaller β=0.5 to reduce boost
    amplitude; smaller τ=0.2 to put 70%+ tokens into boost;
    quadrant-aware splitting vd<0 / vd>0); re-counterfactual.

(iii) Pivot to Tier-2 mechanism falsifier (off-policy KD vs T1-2
    on-policy OPD). Defends T1 brief v2 with reviewer-proof
    mechanism evidence. Treat the 6-variant T2-1 sweep as a
    "negative result with structural insight" section in the
    paper.

Which would you recommend?

### Q4: paper value of this analysis?

Independent of T2-1Abs* success: this 6-variant counterfactual
sweep cleanly demonstrates "naive RLVR-style token VD reweighting
doesn't transfer to OPD, and energy bounding fundamentally trades
off against signal targeting in the {abs, RMS-preserve, max-clip}
design space". Is this strong enough to be a paper section on
its own (sub-titled e.g., "design-space exploration for
OPD-appropriate visual-dependency reweighting")?

## Relevant artifacts (commit 79a3d69)

| Path | What |
|---|---|
| `src/mllmopd/analysis/t2_1_abs_counterfactual.py` | 6-variant scan + verdict |
| `src/mllmopd/analysis/t2_1_energy_audit.py` | A0+ audit (rho_l2/corr/frac_supp/quadrant) |
| `src/mllmopd/training/opd_adv_dump.py` | Sidecar (sample_index, old_log_probs) |
| `scripts/setup/patch_uni_opd.sh` (P19) | loss.py dump hook |
| `runs/audit/t2_1_counterfactual_6v_*/result.json` | 6-variant full JSON (H800 ceph only) |
| `runs/audit/t2_1_energy_audit_*/result.json` | A0 result (H800 ceph only) |
| `runs/audit/t2_1_eval_20260523-003257/t2_1_per_prompt_diff.json` | Per-prompt diff (in repo) |
| `docs/gpt-review-2026-05-24-a0-energy-audit.md` | Round-4 prompt for full context |
| `docs/t2_1_design.md` | Original T2-1 design |
