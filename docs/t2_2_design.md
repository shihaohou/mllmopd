# T2-2 design: boost-only |vd| weighted OPD

## TL;DR

T2-2 replaces T2-1's PGPO-style signed-vd suppress+boost+renorm with
a **boost-only |vd| weighting** that preserves the base OPD
correction and adds visual-dependency-proportional boost on top.
Formula:

```
score_t = rank(|vd_t|) / (R - 1)            per-sequence percentile in [0, 1]
w_t     = clamp(1 + α · score_t, 1, max_w)  defaults α=1.0, max_w=2.0
adv_t   = (lp_teacher_full - lp_student) · sentinel_mask · w_t
```

By construction:
- `w_t ≥ 1` everywhere → NO suppression → `frac_supp ≡ 0`
- NO mass-preserve renorm → `Σw > N` (total energy up by ~E[w]−1)
- Uses `|vd|` so visual-rejection tokens get boosted symmetrically
  with visual-support tokens (the sign is already in adv_t)
- Percentile rank is outlier-robust (one extreme |vd| does not
  compress the rest)

Hypothesis: this fixes both T2-1 failure modes that the 6-variant
counterfactual exposed in
`docs/gpt-review-2026-05-24-counterfactual-6variants.md`:
1. **signed-proxy mis-target** (T2-1 cond_supp=0.997 of
   visual-rejection routed to suppress) → solved by `|vd|`
2. **frac_supp ~7% structural floor** (weak-|vd| tokens trapped in
   suppress branch of PGPO τ=0.4 threshold) → solved by removing
   the suppress branch entirely

If T2-2 doesn't beat T1-2 on opd_target_recovery, the
"visual-dependency reweighting in any form" hypothesis is harder to
defend and the project pivots to either VPPO trajectory-α (T2-3) or
Tier-2 off-policy KD mechanism falsifier (defends T1 brief v2).

## Why this design (post GPT round-5)

After GPT round-3 → round-5 (in conversation history):

| Iteration | Verdict |
|---|---|
| T2-1 PGPO signed vd | Δ = −1.67pp vs T1-2, CI includes 0; A0 audit reveals cond_supp=0.997 (100% of visual-rejection routed to suppress) |
| Abs (vd → \|vd\|) | Fixes targeting (corr -0.10 → +0.31) but rho_l2 = 12 (would diverge) |
| Abs+RMS-preserve | Per-seq scalar shrinks weights below 1 → DESTROYS targeting (mean_w 0.22) |
| Abs+max-clip(±renorm) | Pareto trade-off (energy vs targeting); no 4/4 pass |
| **Boost-only \|vd\|** | T2-2: removes suppress branch, no renorm → frac_supp=0 by construction |

GPT round-5 summary (paraphrased from chat):

> Stop searching for optima in PGPO suppress/renorm family. Move to
> base-OPD + no-suppress visual boost. OPD's per-token advantage is
> already signed correction; the VD score should only MODULATE
> magnitude, not REDISTRIBUTE mass.

## What changed vs T2-1

| Aspect | T2-1 (signed PGPO) | T2-2 (boost-only \|vd\|) |
|---|---|---|
| Score | `vd = lp_full − lp_blank` (signed) | `\|vd\| = abs(lp_full − lp_blank)` |
| Normalization | per-seq min-max → [0, 1] | per-seq percentile rank → [0, 1] |
| Functional form | Threshold τ=0.4 piecewise: suppress branch below, boost branch above | Single affine boost: `w = clamp(1 + α·score, 1, max_w)` |
| Mass conservation | Per-seq mass-preserve (Σw = N) | NONE (Σw > N; energy rises ~α·0.5×N for uniform ranks) |
| Suppression | Yes (w<1 below τ) | No (w ≥ 1 by construction) |
| Visual-rejection target | Sent to suppress (cond_supp=0.997 in audit) | Sent to boost (same magnitude as visual-support) |
| Hyperparams | τ=0.4, β=2.0 | α=1.0, max_w=2.0 |

Teacher-side compute and rollout flow are unchanged from T1/T2-1.
The diagnostic hook + sidecar (P19) + audit module all carry over;
the only change is the weight-computation function and an env-gated
dispatch.

## Formula details

For each rollout sample with response length R:

```
abs_vd_t   = |lp_teacher_full(t) − lp_teacher_blank(t)|           # per token
ranks_t    = rank(abs_vd_t) / (R − 1)        per sequence, ∈ [0, 1]
                                              (ties broken by argsort index order)
w_t        = clamp(1 + α · ranks_t, 1, max_w)
adv_t      = (lp_teacher_full(t) − lp_student(t)) · sentinel_mask · w_t
```

Edge cases (return unit-weight no-op):
- Response length 0 or 1 → trivially no weighting to apply
- Length mismatch → log warning, return ones
- All-equal |vd| → argsort gives index-stable ranks; weight slope
  remains but visually identical

## Decision tree

Baselines (canonical T1-0 mean 0.553):
- T1-0 (no OPD): 0.553
- T1-2 (vanilla FullTeacher OPD): 0.576 (Δ +23pp on opd_target)
- T2-1 (PGPO signed): 0.5593 (Δ −1.67pp vs T1-2, CI includes 0)

| T2-2 result vs T1-2 | Interpretation | Next |
|---|---|---|
| Δ ≥ +3pp on opd_target headline | Boost-only fixes both T2-1 failure modes; visual-dependency reweighting works when done right | Run T2-3 / multi-seed / paper |
| Δ ∈ [+0.5, +3] | Method works but marginal | α × max_w sweep; per-bench reading |
| Δ ≈ 0 (within noise) | Boost-only doesn't help vanilla FullTeacher OPD; visual-dependency token reweighting (in any form) may not transfer to OPD | Pivot to Tier-2 mechanism falsifier (defends T1) |
| Δ < 0 | Boost-only HURTS — surprising; check rho_l2, look for energy instability | Lower α to 0.5; if still hurts, the magnitude-only "visual-dependency boost" hypothesis is wrong for OPD |
| Training instability (loss NaN, grad explode) | rho_l2 too high in practice | Lower max_w to 1.5; if still unstable, lower α |

## Implementation surface

| File | Change |
|---|---|
| `src/mllmopd/training/vd_weighting.py` | NEW function `compute_vd_weights_boost_only` (~90 lines incl. docstring) |
| `src/mllmopd/training/opd_diagnostics_hook.py` | env-gated dispatch on `MLLMOPD_VD_MODE` (signed \| boost_only); default signed for T2-1 byte-identical |
| `scripts/train/opd_mmr1_3b_baseline_xbox.sh` | propagate `MLLMOPD_VD_MODE`, `MLLMOPD_VD_ALPHA`, `MLLMOPD_VD_MAX_W` to Ray actor env |
| `src/mllmopd/analysis/t2_1_abs_counterfactual.py` | 7th variant `boost_only` for offline preview |
| `scripts/train/verify_vd_weighting.py` | 8 new tests covering bounds, monotonicity, sign-agnostic, alpha=0, max_w clamp, hand-computed R=5 |

All gated; T1-2 / T1-3 / T2-1 paths byte-identical when env flags
match prior defaults.

## Run command

```bash
ROLLOUT_ENGINE_ADDRS="$(cat ${MLLMOPD_RUNS}/rollout_servers/rollout_engine_addrs.txt)" \
TEACHER_HOST=10.82.121.12 \
MLLMOPD_USE_VD_WEIGHTING=1 OPD_TEACHER_IMAGE_MODE=full \
MLLMOPD_VD_MODE=boost_only MLLMOPD_VD_ALPHA=1.0 MLLMOPD_VD_MAX_W=2.0 \
OPD_RUN_NAME=t2_2_v0_boost_only_full \
  bash scripts/train/opd_mmr1_3b_baseline_xbox.sh
```

(No `MLLMOPD_DUMP_OPD_ADV` — sidecar dump is only for the A0
counterfactual flow; T2-2 production run doesn't need it.)

## Caveats

- **Energy rises by ~α·0.5×N** (for uniform ranks: E[w] = 1 + α/2).
  With α=1.0, mean weight ≈ 1.5. Global gradient is ~50% higher in
  expectation. If training shows instability, reduce α first (still
  monotonic ablation against α=0 = T1-2).
- **No mass preservation** is intentional. Mass preservation is what
  caused the T2-1 trade-off. Accepting energy growth is the right
  move; it shifts the question from "where does mass go" to "how
  much extra signal is in the visually-dependent tokens" — much
  closer to OPD's actual structure.
- **Single seed, single hyperparameter point**. Per the T2-1
  precedent: if T2-2 confirms a direction, multi-seed + α sweep
  comes next.
- **Counterfactual preview not yet run**. We added boost_only as
  the 7th counterfactual variant; running it on the existing pilot
  diagnostics (free, no training) will give us an offline read of
  the 4 pass criteria BEFORE the 3-4h training. Recommended but
  optional.

## What this doc is and isn't

**Is**: Method-tier follow-up to T2-1's negative result, designed
per the 6-variant counterfactual + GPT round-5 reframing.
Implementation-ready.

**Isn't**: A redo of T2-1 with different parameters. The functional
form has fundamentally changed (no suppress, no mass-preserve).
T2-2 is a different family.
