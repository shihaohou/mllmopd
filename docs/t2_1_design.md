# T2-1 design: VD-weighted FullTeacher OPD

Design rationale for the first **method tier** experiment on top of the
T1 plumbing. Written before any T2-1 H800 cycles so that the design can
be GPT-reviewed against the final result (`docs/gpt-brief-…-t2-1-…`,
not yet written).

**Status:** implementation landed; smoke-tested on Mac; awaiting H800
training run.

## TL;DR

T2-1 takes the T1-2 vanilla FullTeacher OPD setup and adds one change:
multiply the per-token OPD advantage by a per-token weight derived from
`vd_t = lp_teacher_full(t) − lp_teacher_blank(t)` — the same per-token
"vision dependency" signal T1 used as a diagnostic, now applied as a
training-time reweighting. The weight function is PGPO Eq 6 + Eq 7
(threshold τ=0.4, boost β=2.0, per-sequence mass-preserving renorm),
transplanted from RLVR advantage reweighting to OPD-advantage
reweighting and computed on min-max-normalized vd_t per sequence.

Hypothesis: concentrating OPD's gradient on tokens where the teacher
*actually uses* visual information (high vd_t) should give a stronger
vision-conditioned student than uniform OPD, without changing the loss
form or the teacher pair. If T2-1 doesn't beat T1-2's 0.553 → 0.559
canonical baseline, the "uniform OPD wastes gradient on
linguistic-prior tokens" claim is falsified and we go to T2-3 (β-residual)
or T2-4 (prompt-level oversampling).

## What changed vs T1-2

| Aspect | T1-2 (baseline) | T2-1 (this) |
|---|---|---|
| Teacher | MMR1-7B-RL, full image | MMR1-7B-RL, full image |
| Student | MMR1-3B-SFT | MMR1-3B-SFT |
| Loss | OPD reverse-KL: `adv_t = (lp_teacher − lp_student) * sentinel_mask` | Same, then `adv_t *= w(vd_t)` |
| `vd_t` | Computed in diagnostics only (logged, unused) | Computed in diagnostics, then used as weight |
| Teacher cost | 2× passes (full + blank) for diagnostics already | Same 2× passes — no extra compute |
| Hyperparameters | τ/β not applicable | τ=0.4, β=2.0 (PGPO Table 2 winners) |

Notably the **teacher-side compute is unchanged** — T1 was already
running the blank teacher in parallel for diagnostic logging via
`dual_teacher_get_reward.py`. T2-1 just consumes the signal that T1
was already producing.

## Formula

For each rollout sample with response length R:

```
vd_t        = lp_teacher_full(t) - lp_teacher_blank(t)              # per token
vd_norm_t   = (vd_t - min(vd)) / (max(vd) - min(vd) + ε)            # per sequence → [0, 1]
w_raw_t     = vd_norm_t / (τ + ε)                  if vd_norm_t < τ
            = 1 + β (vd_norm_t - τ) / (1 - τ + ε)  if vd_norm_t ≥ τ
w_t         = w_raw_t · R / Σ_s w_raw_s                              # per sequence, mass-preserve
adv_t       = (lp_teacher_full(t) - lp_student(t)) · sentinel_mask · w_t
```

Edge cases that return unit-weight no-op (no degradation vs T1-2):
- Response length 0 or 1
- All-equal vd (max − min < ε)
- Teacher-failed sentinel positions (already masked upstream)
- Length mismatch (logs warning)

## Why PGPO, not VPPO or PAPO

We reviewed three closely-related papers before committing to this design.

**PGPO** (Ye et al., arXiv:2604.01840) — token-level KL-based VD →
threshold-gated piecewise reweighting + per-sequence mass-preserving
renorm. Ablation: ~2 pts each from suppress, boost, renorm. **We
transplant their Eq 6 + Eq 7 verbatim**, swapping their full-distribution
KL signal for our `lp_full − lp_blank` *realized-token log-likelihood-ratio
proxy* — not an unbiased estimator of PGPO's KL (the realized token is
sampled from the on-policy student, not from the full-image teacher
distribution), but the simplest token-level vision-dependency proxy our
dual-teacher diagnostics already produces at zero extra cost.

**VPPO** (Huang et al., arXiv:2510.09285, ICLR 2026) — two mechanisms:
(a) trajectory-level α(τ) ∈ [0.9, 1.0] scaling the whole sequence advantage;
(b) top-40% token gradient mask. Their ablation: TGF +2.1 vs TAS +1.3;
*token mechanism carries the gain*. We chose PGPO's *soft-suppress* over
VPPO's *hard top-k mask* because OPD gradients on low-VD tokens are still
informative — the teacher *does* know grammar tokens; we want to dampen,
not zero. Trajectory α (TAS) is set aside as an orthogonal axis for T2-2
ablation, not bundled here.

**PAPO** (Wang et al., arXiv:2507.06448) — adds a *sequence-level* KL term
to the RLVR loss. Explicitly not teacher-based and known to KL-hack at
γ=0.04 (−43% collapse without Double Entropy Loss). Skipped: structurally
wrong target (sequence-level) and a known failure mode we don't want to
inherit into a setting that's already unstable (T1-3 cliff).

## Design choices and their reviewer-defense

| Choice | Alternative | Why we chose | Reviewer attack | Defense |
|---|---|---|---|---|
| Threshold + soft-suppress | Hard top-k mask | OPD gradients on low-VD tokens still informative | "τ is brittle" | τ=0.4 = PGPO Table 2 winner; pre-register τ ∈ {0.3, 0.4, 0.5} as planned sweep |
| Per-sequence min-max normalize vd | Use raw vd directly | Our vd is scale-free (logp difference, signed); PGPO τ assumes [0,1] | "Min-max is brittle to outliers" | Smoke test on real diagnostics will inspect vd-percentile distribution; report 5/50/95% in results |
| Per-sequence mass-preserve | No renorm; per-batch renorm | PGPO no-renorm ablation drops 1.9 pts; per-batch couples loss to batch composition | "Renorm hides signal magnitude" | Log Σw_raw per step; published next to the headline number |
| Token-level only | Add VPPO TAS at same time | Confounded ablation; do orthogonal axis separately in T2-2 | "Token alone not enough" | If T2-1 marginal, T2-2 = T2-1 + TAS as planned follow-up |
| Realized-token lp-difference proxy | Full-distribution KL like PGPO | We don't have teacher logits, only lp on the student's sampled token; the proxy is also signed (vd can be negative when full teacher rejects a token that blank teacher accepts), which PGPO's KL signal never is | "Single-sample proxy is high-variance and conflates 'image supports token' with 'image rejects token'" | Variance measurable from diagnostics dump (per-step VD distribution). Sign asymmetry is acknowledged but not addressed in T2-1 first-cut; revisit if results show extreme-negative-VD tokens dominating the suppress branch |
| FullTeacher only, not BlankTeacher | Both arms with VD | BlankTeacher + VD weighting doesn't have a coherent interpretation (vd as primary-arm signal) | "Asymmetric" | Documented; launcher explicitly errors on blank+VD combo |

## Predicted outcomes and decision tree

Baselines to compare against (canonical T1-0 mean from
`t1_compare.json`, regenerated by `t1_brief_table.py`):

- **T1-0 (base, no OPD)**: 0.553
- **T1-2 (vanilla FullTeacher OPD)**: 0.576 (Δ +23pp on opd_target)
- **T1-3 (BlankTeacher cliff)**: catastrophic; not a competitive baseline

T2-1 outcomes and what each implies:

| T2-1 result vs T1-2 | Interpretation | Next |
|---|---|---|
| Δ ≥ +3pp on opd_target headline | PGPO mechanism transfers to OPD ✓ | T2-2 (TAS ablation) + Tier-2 mechanism falsifier |
| Δ ∈ [+0.5, +3] | Method works but marginal | τ × β sweep; consider EMA-smoothed vd |
| Δ ≈ 0 (within noise) | Reweighting doesn't help vanilla FullTeacher OPD | T2-3 (β-residual) or T2-4 (prompt oversampling) |
| Δ < 0 | Reweighting *hurts*; either PGPO transplant invalid or vd_t signal is noisier than PGPO's KL signal | Hot-stop, retry with hard top-k mask (VPPO-style) |
| Cliff appears like T1-3 | VD weighting destabilizes. **Two equally-likely interpretations**: (a) weight outliers spike effective LR on a few tokens and break optimization; (b) the reweighting amplifies the blank-template attractor T1-3 exposed. Mechanism causation is Tier-2's job, not T2-1's. | Hot-stop, inspect vd_weight max/p99 distribution to disambiguate (a) vs (b), then fall back to Tier-2 controls if needed |

## Implementation surface

5 small edits across 4 files; ~80 lines net. All gated on env var
`MLLMOPD_USE_VD_WEIGHTING=1` so T1-2 / T1-3 baselines stay byte-identical
when the flag is off.

| File | Change |
|---|---|
| **NEW** `src/mllmopd/training/vd_weighting.py` | `compute_vd_weights(lp_full, lp_blank, R)` — PGPO Eq 6 + 7 pure function |
| **MOD** `src/mllmopd/training/opd_diagnostics_hook.py` | When env on: call `compute_vd_weights`, attach `sample.teacher_vd_weights`; also dump per-row in JSONL |
| **MOD** `third_party/Uni-OPD/miles/miles/ray/rollout.py` | Mirror the existing `teacher_log_probs` plumbing for `teacher_vd_weights` (2 edits: collect + partition) |
| **MOD** `third_party/Uni-OPD/miles/miles/backends/training_utils/data.py` | Move `teacher_vd_weights` to CUDA when present |
| **MOD** `third_party/Uni-OPD/miles/miles/backends/training_utils/loss.py` | OPD branch: `adv *= vd_w` when `rollout_data["teacher_vd_weights"]` present |
| **MOD** `scripts/train/opd_mmr1_3b_baseline.sh` | `MLLMOPD_USE_VD_WEIGHTING`, `MLLMOPD_VD_TAU`, `MLLMOPD_VD_BETA` env handling + Ray actor env propagation + new ARM_TAG |
| **NEW** `scripts/train/verify_vd_weighting.py` | 9 unit tests on `compute_vd_weights` (mass, monotonicity, edge cases, hand-computed reference); optional integration on real JSONL |

Smoke validation done on Mac (9/9 unit tests pass, including
hand-computed PGPO Eq 6 + 7 expected weights on a 5-token reference).

## Caveats

- **Single seed.** Same as T1; multi-seed deferred until after T2-1 confirms a direction.
- **Single hyperparameter point.** τ × β grid sweep is a planned follow-up if T2-1 is marginal.
- **Realized-token lp-difference proxy ≠ PGPO's full-distribution KL.** `lp_full − lp_blank` on the on-policy-sampled token is *not* an unbiased estimator of `D_KL(π(·|s,I) || π(·|s, blank))` — the rollout token came from the student, not from the teacher. It is a tractable per-token vision-dependency proxy that PGPO's signal motivates but does not equal. We commit to this proxy for T2-1 first-cut and will measure its variance + signed-VD distribution from the diagnostics dump before considering alternatives (EMA, k-sample averaging, abs-VD).
- **Min-max normalization is per-sequence.** Long responses with one extreme outlier will compress the rest. Will report vd-range distribution from the diagnostics.
- **No Tier-2 mechanism falsifier yet.** T2-1 is method validation. If T2-1 works, reviewer can still argue "PGPO weighting works on biased teacher distillation in general, not specifically OPD." Tier-2 off-policy KD controls remain required for that claim, per `handoff-2026-05-22-brief-v2-tier2-next.md` §Why Tier-2 before T2-1 — we explicitly deferred that to *after* T2-1 results.

## Result-interpretation discipline (per GPT round-5 review)

In the result brief we will write after H800 training:

- **Do** claim: "PGPO-style token-VD reweighting improves OPD overall" /
  "method signal observed at Δ X pp" — these are statements about the
  reproduction-method experiment.
- **Do NOT** claim: "OPD-specific mechanism confirmed" / "vision-conditioned
  capability transfer requires VD weighting" — those are mechanism claims
  that need Tier-2 off-policy KD controls to support.
- If T2-1's gain is on overall mean but not on the full-vs-blank gap or
  opd_target subset specifically, downgrade story to "VD weighting
  improves OPD overall" without the vision-conditioned framing.

## Smoke gate checklist (must pass before full H800 run)

Pre-registered from GPT round-5 review. All 7 must hold before
launching the 5-6h training. If any item fails, hot-stop and fix.

1. **`diagnostics/step_*.jsonl.gz` rows contain `vd_weights` column.** Zero rows with missing key.
2. **Σ vd_weights ≈ response_length** (within float tol) on *usable* rows (`response_length > 0`, `vd` len-matched). Mass preservation holds in production.
3. **Usable rows are not 100% all-ones.** If they are, vd parse is silently failing → diagnostic hook bug, not method signal.
4. **`n_vd_attached / n_written` ≥ 0.9.** Most samples should produce real weights; high degenerate rate is a parse-alignment bug.
5. **`lp_full` (primary arm) aligns with `sample.teacher_log_probs`** — pick one sample, assert lp_full[-R:] == sample.teacher_log_probs[-R:] elementwise. The new `_extract_response_logprobs` mirrors canonical slicing exactly; this confirms.
6. **`loss.py` P13 multiplier branch executed.** Grep training log for the `[OPD/VD]` warning (should be ABSENT on a healthy run) and confirm a non-trivial per-step `pg_loss` differs from a T1-2 control on the same rollout.
7. **T1-2 / T1-3 byte-identical when `MLLMOPD_USE_VD_WEIGHTING=0`.** Re-run a tiny T1-2 smoke with env off; train log + diagnostics row hashes match a pre-T2-1 reference.

## What this doc is and isn't for GPT review

**Is**: a design rationale to be reviewed *before T2-1 results land* so
that the post-result brief (when written) has the design context as a
fixed reference, not a moving target.

**Isn't**: a results report. After H800 training + eval, a separate
`docs/gpt-brief-YYYY-MM-DD-t2-1-result.md` will report the outcome
following the T1 brief v2 structure (canonical table, trajectory,
mechanism wording, reviewer attacks, literature anchors).
