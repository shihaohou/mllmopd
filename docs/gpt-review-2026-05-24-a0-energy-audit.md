# GPT review brief — T2-1 A0 energy audit result (round-4)

**Date**: 2026-05-24 (post T2-1 training, post GPT round-3 reframing,
post A0 energy-audit data collection)
**Repo**: https://github.com/shihaohou/mllmopd (HEAD = `ab55463`)
**Reviewer ask**: independent review of (1) whether the A0 energy
audit numbers justify "signed-proxy mis-target" as the dominant
mechanism and "LR confound" as refuted, (2) whether the proposed
next experiment (T2-1Abs or T2-1NegAware) is the right move, and
(3) whether anything load-bearing is being glossed over.

GPT round-3 reply (Chinese, terse) is in conversation history; key
quotes paraphrased inline where relevant.

## 30-second timeline

1. **T1 brief v2 (CONFIRMED 2026-05-22)**: FullTeacher OPD beats
   BlankTeacher control by +23pp (p ≈ 10⁻⁶) → vision-conditioned
   capability transfer hypothesis sharpened to "OPD is condition-
   sensitive".
2. **T2-1 (trained 2026-05-23)**: PGPO Eq 6+7 (τ=0.4, β=2.0,
   per-sequence sum-preserving renorm) applied to OPD's per-token
   advantage. 230 steps cross-box. Headline Δ vs T1-2 = −1.67pp,
   95% bootstrap CI [−3.9, +0.5] — includes 0. opd_target_recovery
   −10pp (47.8 → 37.7). McNemar n=133 p=0.864 (b=16 win, c=18 lose).
3. **GPT round-3 (this conversation, earlier)**: don't write up
   as "PGPO doesn't work"; **two confounds** suspected:
   (a) effective-LR drop (PGPO Eq 7 preserves Σw=N but not
   Σw²·adv² → grad_norm halving observed at step_1 of original
   T2-1: 12.05 vs T1-2's 20.34, ratio 0.59)
   (b) signed-proxy mis-target (our `vd = lp_full - lp_blank` is
   signed; PGPO's original VD is non-negative KL; "visual
   rejection" tokens where image pushes teacher AWAY from current
   token end up vd<0 → put in suppress branch w<1, even though
   they're the most visually-dependent tokens in PGPO's KL sense).
4. **A0 designed** (this session): three metrics computable from
   diagnostics:
   - `rho_l2 = sqrt(Σ(w·adv)² / Σ adv²)` (effective-LR test)
   - `corr(w_t, |adv_t|)` (signal-targeting test)
   - `frac |adv| · 𝟙[vd<0 ∧ adv<0 ∧ w<1]` (visual-rejection
     suppression test)
5. **A0 unblock (commit ab55463)**: discovered the existing diag
   hook fires from rollout actor BEFORE trainer forward — so
   `sample.old_log_probs` is None and adv cannot be reconstructed.
   Added P19 patch to loss.py to dump `(sample_index, old_log_probs)`
   sidecar from inside `compute_advantages_and_returns`. Re-ran
   T2-1 for 10 trainer steps with `MLLMOPD_DUMP_OPD_ADV=1` →
   collected 80 sidecar files (10 steps × 8 dp ranks).
6. **A0 result (this section)**: see below.

## A0 numbers (verbatim from audit stderr, 2026-05-24)

```
=== T2-1 ENERGY AUDIT (pooled, 576 samples, 935814 tokens) ===
  rho_l2 = 0.973         ≥ 0.85 ⇒ LR confound unlikely
  corr(w, |adv|) = -0.101 ⇒ weights anti-correlate with signal magnitude
  frac_supp_neg_vd_neg_adv = 0.442 > 0.15 ⇒ signed-proxy suppressing substantial visual-rejection mass
```

Result JSON: `runs/audit/t2_1_energy_audit_20260524-111948/result.json`
on the H800 box (ceph-only — full per-step trajectories there).

## Per-prompt diff (already-shipped, commit 10cb911)

`runs/audit/t2_1_eval_20260523-003257/t2_1_per_prompt_diff.json` —
canonical 133 prompts categorized:

```
WIN  (T2-1 right, T1-2 wrong):  16
LOSE (T2-1 wrong, T1-2 right):  18
BOTH_CORRECT:                   40
BOTH_WRONG:                     59

Per-benchmark NET (WIN - LOSE):
  ChartQA          +2   (visual QA, short answers)
  HallusionBench   +2   (visual QA, short answers)
  MathVista        +2   (mixed math + visual)
  ─────────────────────────
  MathVerse        -4   (long math CoT)
  MathVision       -2   (long math CoT)
  POPE_adversarial -2   (visual yes/no, adversarial)
```

POPE LOSE samples are textbook visual-rejection: gold "yes", T1-2
"yes" (correct), T2-1 over-rejects with "no, there is no clearly
identifiable backpack". MathVerse LOSE samples show wrong
intermediate values propagated through long CoT (47° → 73°,
√3/2 → 1/2, opposite-direction inferences).

## What this rules out and rules in

**Refuted**: "T2-1 is under-trained because grad_norm halved."
- rho_l2 = 0.973 means weighted grad ≈ 97% of unweighted at step 1
- Whatever caused the 12.05 vs 20.34 ratio in the original full
  T2-1 training log, it is NOT mass-vs-energy mismatch from
  PGPO Eq 7. Could be entropy/kl term differences, batch
  composition difference at step 1, or something else not
  worth chasing.

**Confirmed**: "Signed `lp_full - lp_blank` proxy mis-targets in OPD."
- 44% of |adv| mass is in tokens where `vd<0 ∧ adv<0 ∧ w<1`.
- These are exactly visual-rejection tokens (image pushes teacher
  AWAY from a token, student samples it anyway, teacher correction
  signal `adv<0` says "decrease prob of this token").
- PGPO's non-negative KL would put these in HIGH-VD bin → boost.
- Our signed proxy puts them in LOW-VD bin → suppress.
- The mis-target is mechanism-level, not LR-level.

**Also confirmed (weakly)**: "corr(w, |adv|) is slightly negative."
- corr = −0.101 means PGPO weights inversely correlate with OPD
  signal magnitude, but only weakly. Combined with frac_supp=0.44,
  the pattern is: suppression and boost roughly average out
  globally (hence rho_l2≈1), but suppression FALLS
  DISPROPORTIONATELY on visual-rejection tokens.

**Consistent with per-prompt diff (10cb911)**: LOSE clusters on
long-CoT math + POPE-adversarial, which are exactly the benchmarks
where visual-rejection correction is most important.

## Proposed next experiment (NOT YET COMMITTED — this is the asks for GPT)

### Option A: T2-1Abs (recommended candidate)

Single line change in `src/mllmopd/training/vd_weighting.py`:

```python
# current (T2-1)
vd = lp_full_t - lp_blank_t

# T2-1Abs
vd = (lp_full_t - lp_blank_t).abs()
```

Everything downstream (min-max normalize, threshold τ=0.4 piecewise,
mass-preserving renorm) unchanged. Visual-rejection tokens now go
into HIGH-|vd| → boost branch. Frac_supp predicted to drop to
≈ 0.

**Pros**: most minimal change; semantically closest to PGPO's
non-negative KL; clean ablation against T2-1 (only signal sign
changes, weighting form identical).

**Cons**: loses sign information. The distinction between "image
strongly SUPPORTS this token" (vd>0, large) and "image strongly
REJECTS this token" (vd<0, large) collapses. Both get boosted
equally. May or may not matter for OPD.

### Option B: T2-1NegAware

Keep signed `vd`. Two parallel boost branches:

```python
# vd>0 branch (current, unchanged): vd_norm in [0.4, 1.0] boost
# vd<0 branch (NEW): |vd_norm - 0|/scale-of-negative-side, mirrored
```

Concretely: take `|vd|`, min-max normalize, piecewise as before.
But ALSO carry the sign as a separate per-token attribute that
loss.py could optionally use (e.g., for a signed-VD diagnostic
plot post-hoc).

**Pros**: keeps the sign in case it's diagnostically interesting
later.

**Cons**: more implementation surface; basically equivalent to
T2-1Abs for the loss math (sign info isn't used in weighting).

### Option C: Wait, run more A0 trajectories

Re-run A0 on the ORIGINAL T2-1 230-step run (currently has no
sidecars). Would need to re-collect by re-training from same seed,
which is 3-4h. Result probably matches the 10-step pilot — pilot
is already 576 samples / ~10⁶ tokens.

## Specific questions to GPT round-4

1. **Is the rho_l2=0.97 vs grad_norm-ratio=0.59 discrepancy a
   concern?** Our interpretation: grad_norm includes entropy/kl
   terms and post-PG terms; rho_l2 measures only the PG component
   ratio. Energy in adv·∇log_p is preserved (~97%), so the 0.59
   ratio came from somewhere else (probably the entropy term
   integrates differently when adv is reweighted, or batch
   composition at step 1 was slightly different). We're treating
   the original grad_norm observation as RED HERRING. Is that
   defensible?

2. **Is frac_supp=0.44 the right framing of "signed-proxy mis-target"?**
   Alternative framing: "44% of (|adv| mass on vd<0 ∧ adv<0
   tokens) is being SUPPRESSED" vs "44% of TOTAL |adv| mass is
   being SUPPRESSED on visual-rejection tokens". Audit code:
   numerator = `Σ |adv| · 𝟙[vd<0 ∧ adv<0 ∧ w<1]`,
   denominator = `Σ |adv|`. Audit also reports
   `frac_neg_vd_neg_adv_mass_total` (numerator without the `w<1`
   gate) — if that's e.g. 0.50, then 44/50 = 88% of visual-rejection
   correction is suppressed (slightly worse framing). User can fetch
   that number from the JSON if relevant.

3. **Should T2-1Abs OR T2-1NegAware be the next experiment, OR
   should we try something else entirely?** Other candidates we
   considered:
   - **C from round-3**: RMS-preserve scalar (advantage-energy-
     preserving variant of T2-1). Now appears unnecessary because
     LR confound is refuted. Confirm we can skip?
   - **VPPO-style hard top-k by |vd|**: top-40% tokens get full
     weight, rest get 0. More aggressive than soft suppress;
     ablates against T2-1Abs.
   - **No further T2-* — pivot to Tier-2 mechanism falsifier**:
     handoff-2026-05-22-brief-v2-tier2-next.md still has off-policy
     KD falsifier as a defensible next step. T2-1 was supposed
     to be method tier; if we now know PGPO transferred naively
     doesn't work, maybe the right move is "Tier-2 mechanism
     falsifier first, then return to method tier with the
     mechanism story locked".

4. **Anything else load-bearing being glossed over?**
   In particular: is the A0 result on a 10-step pilot
   (576 samples, 935814 tokens, 8 dp ranks) representative of the
   full 230-step T2-1 training? Sparsity / max-weight reported
   from t2_1_vd_distribution on the original 230 steps were 0.67-
   0.78 / 2.87-2.89 (stable across training); pilot's vd
   distribution should be similar. We're betting yes.

## Relevant files (commit ab55463)

| Path | What |
|---|---|
| `src/mllmopd/analysis/t2_1_energy_audit.py` | Audit module, computes rho_l2 / corr / frac_supp, joins sidecar by sample_index |
| `src/mllmopd/training/opd_adv_dump.py` | Sidecar dump module (P19 in patch_uni_opd.sh) |
| `src/mllmopd/training/opd_diagnostics_hook.py` | Existing diag hook (added sample_index field for join) |
| `src/mllmopd/training/vd_weighting.py` | PGPO Eq 6+7 implementation (where T2-1Abs change would land) |
| `src/mllmopd/analysis/t2_1_per_prompt_diff.py` | Per-prompt categorization on canonical 133 |
| `scripts/setup/patch_uni_opd.sh` | P19 hunk in loss.py compute_advantages_and_returns |
| `docs/t2_1_design.md` | Pre-registered T2-1 design + decision tree |
| `docs/handoff-2026-05-23-t2-1-result-pending-gpt.md` | Pre-A0 handoff (now stale on framing; A0 supersedes) |
| `runs/audit/t2_1_eval_20260523-003257/t2_1_per_prompt_diff.json` | Per-prompt records (133 with predictions) |
| `runs/audit/t2_1_eval_20260523-003257/t2_1_compare.json` | T2-1 vs T1-2 vs T1-0 headline metrics |
| `runs/audit/t2_1_energy_audit_20260524-111948/result.json` | A0 result (H800 ceph only) |

## What we're NOT asking

- Is the brief writeable now? We're keeping T1 brief v2 as the
  canonical positive result; T2-1 is method-tier exploration.
- Should we change the eval pipeline? `runs/audit/t2_1_eval_*`
  + `t2_1_compare.py` are working.
- Is the Tier-2 mechanism falsifier still on the table? Yes,
  question 3 above re-raises it as one option among several.
