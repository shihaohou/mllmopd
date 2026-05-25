# TAM Step 2b — Quad-Aware Causal Masking (2026-05-25)

## TL;DR

We previously closed Step 2 v2 with strong positive evidence: top-TAM
mask vs random mask Δ = +0.99 nat (Wilcoxon p ≈ 10⁻¹¹), and top-TAM vs
scrambled-TAM Δ = +1.06 nat (spatial-structure-only control). Step 2b
joins Step 1a's per-token `quad` labels (vd × adv sign quadrants) into
Step 2's tokens to ask: **does TAM's causal effect hold up on the
visual-rejection bucket (quad==3, vd<0 ∧ adv<0) — the bucket T2-1
mis-targeted?**

**Answer: NO, but not because TAM is inverted. q3 evidence is globally
distributed; no spatial 20%-patch mask captures it.** All 5 mask
strategies (top-TAM / random / scrambled / keep-top / bottom-TAM)
produce drops of ≈ 0 nat on q3 tokens, with paired Δ(bottom − top) ≈ 0
(Wilcoxon p > 0.4 across all 4 ckpts).

**Implication for §Method**: OPD has two structurally distinct failure
modes. Local-evidence support (q0/q1) is causally addressable by
TAM-region-Boost gate (Δ = +1.27 nat). Distributed-evidence rejection
(q3, T2-1's failure mode) is NOT addressable by any spatial-locator
intervention — it needs a complementary, non-spatial mechanism
(e.g. `|VD|`-boost no-suppress on negative-VD tokens, à la T2-2).

The TAM line as written is scoped to q0/q1 and is publication-ready
for §Diagnostic + §Method on that scope. T2-1's failure mode remains
open, but it is now clearly **orthogonal** to the TAM mechanism rather
than something TAM can address.

## Run provenance

| Artifact | Run dir / file | Commit |
|---|---|---|
| Step 1a JSONL (vd / adv / quad source) | `tam_step1a_20260525-190333/tam_step1a.jsonl` | `93b8cd7` |
| Step 2 v2 JSONL (masking experiment) | `tam_step2_20260525-213619/tam_step2.jsonl` | `d950894` |
| Step 2b analyzer + numbers in this brief | `src/mllmopd/analysis/tam_step2b_quad.py` | `737fa01` |

Join key: `(sample_id, response_hash, token_idx)`. Step 1a stores
per-(sample × ckpt) `quad / vd / adv` arrays of length R; Step 2 stores
one row per (sample × target_token × mask_strategy). Teacher greedy is
deterministic across student_ckpts so response_hash matches; `quad`
varies by ckpt (different adv). All 649 Step 2 target tokens resolved
under all 4 ckpts (0 unresolved).

## Step 2b headline tables

**Table 1 — Paired Δ(top_tam_20pct − mean_random_20pct), per ckpt × quad**

```
T1_0                                n  mean_Δ_random   CI       Wilcox p
q0 vis_support_agree             450     +1.256        [+0.93, +1.60]   9.8e-15
q1 vis_support_pushed_away        45     +1.702        [+0.47, +3.26]   1.6e-2
q2 vis_reject_teacher_toward     118     +0.025        [-0.06, +0.16]   8.2e-1
q3 vis_reject_correction          36     −0.113        [-0.32, +0.00]   1.8e-3

T1_2                                n  mean_Δ_random   CI       Wilcox p
q0                               431     +1.271        [+0.94, +1.63]   9.8e-15
q1                                64     +1.469        [+0.48, +2.71]   1.1e-2
q2                               108     +0.028        [-0.07, +0.17]   8.0e-1
q3                                46     −0.088        [-0.25, +0.00]   5.9e-3

T1_3                                n  mean_Δ_random   CI       Wilcox p
q0                               455     +1.348        [+1.00, +1.71]   4.4e-16
q1                                40     +0.719        [+0.15, +1.49]   3.1e-1
q2                                99     +0.036        [-0.07, +0.19]   9.0e-1
q3                                55     −0.085        [-0.23, -0.00]   4.3e-3

T2_1                                n  mean_Δ_random   CI       Wilcox p
q0                               430     +1.232        [+0.90, +1.59]   4.2e-15
q1                                65     +1.727        [+0.62, +2.97]   2.2e-2
q2                               104     +0.029        [-0.07, +0.18]   9.0e-1
q3                                50     −0.081        [-0.23, +0.00]   3.7e-3
```

Strong, consistent picture across 4 ckpts:
- **q0**: TAM > random by ~+1.3 nat, p ≈ 10⁻¹⁵
- **q1**: TAM > random by ~+1.4 nat (smaller n)
- **q2**: ≈ 0 (sparse / no signal)
- **q3**: **slightly NEGATIVE** (~−0.09 nat, Wilcoxon p ≈ 0.005)

**Table 2 — Raw mean logp_drop by strategy, T1_2 (representative; other ckpts ≈ same)**

```
quad                       n   top_tam   random   scram   keep_top   bot_tam
q0 vis_support_agree     431   +1.629   +0.358   +0.274   +0.554     +0.117
q1 vis_support_pushed_a   64   +1.840   +0.371   +0.211   +0.692     +0.057
q2 vis_reject_teacher    108   +0.064   +0.037   +0.058   -0.001     -0.004
q3 vis_reject_correctio   46   +0.007   +0.095   +0.082   -0.008     -0.002
```

Look at the q0 row vs q3 row:
- q0: every strategy moves lp meaningfully; top_tam dominates at +1.63
- q3: **EVERY strategy ≈ 0**. random_20pct on q3 = +0.10; on q0 = +0.36 (3.6× more sensitive)

q3 tokens are **3-4× less image-sensitive** to localized masking than q0 tokens — even though VD says they're highly image-dependent globally. The image is influencing q3's lp through distributed / global features, not patch-localizable ones.

**Table 3 — Paired Δ(bottom_tam_20pct − top_tam_20pct), per ckpt × quad**

```
T1_0                                n   mean_Δ     CI                  Wilcox p
q0                               450   -1.507   [-1.91, -1.15]         0.0
q1                                45   -1.943   [-3.60, -0.60]         8.9e-3
q2                               118   -0.063   [-0.19, +0.01]         7.0e-2
q3                                36   -0.012   [-0.04, +0.01]         4.5e-1

T1_2  (other ckpts essentially the same)
q0                               431   -1.512   [-1.89, -1.18]         0.0
q1                                64   -1.784   [-3.12, -0.68]         5.3e-3
q2                               108   -0.069   [-0.20, +0.01]         7.8e-2
q3                                46   -0.009   [-0.03, +0.01]         6.6e-1
```

This is the smoking-gun for "is TAM inverted on q3?":
- q0/q1: bottom hurts MUCH less than top (Δ ≈ −1.5 nat, p ≈ 0). **TAM is correctly oriented.**
- **q3: bottom and top are statistically indistinguishable** (Δ ≈ −0.01 nat, Wilcoxon p > 0.4 across all 4 ckpts). **TAM is NOT inverted on q3 — it's structurally inapplicable.**

If TAM were inverted on q3, we'd expect Δ(bottom − top) > 0 with significant p. We see Δ ≈ 0 with p ≈ 0.5. The null can't be rejected. Both top and bottom regions are non-informative for q3.

**Table 4 — quad==3 per token_category (T1_2, n=46)**

```
content_noun       n=30  mean_Δ_random = −0.135
proper_noun        n=12  mean_Δ_random = −0.001
visual_attribute   n=2   mean_Δ_random = −0.000
meta_cot_token     n=1   mean_Δ_random = +0.000
visual_number      n=1   mean_Δ_random = −0.000
```

content_noun under q3: mean Δ = −0.135. Even content nouns — where TAM
worked best on q0 (Δ = +1.24) — lose all their TAM advantage under q3.
The category isn't what determines TAM applicability; the EVIDENCE
TYPE (local vs distributed) is.

## Mechanism interpretation

**q3 = vd < 0 ∧ adv < 0** = "image makes teacher LESS sure about this
token AND teacher correctly pushes student away from it."

VD measures the FULL-image vs BLANK-image perturbation — a large
global perturbation. q3 tokens have negative VD by definition, so the
image DOES change the teacher's prediction. But Step 2 measures
removing 20% of the image (a small local perturbation). q3 tokens are
**insensitive to any 20% patch removal**. The image's influence on q3
tokens is therefore **distributed across many patches, not
concentrated**.

Plausible mechanisms for distributed evidence:
1. **Scene-gist rejection**: image's overall structure (e.g. "this is a
   wedding scene" vs "this is a kitchen") rejects the token (e.g.
   "knife"). Removing 20% leaves the overall scene gist intact.
2. **Prefix-conditioned rejection**: by the time the teacher reaches
   the q3 token, the response prefix has already established a
   visual-incompatible commitment. Image's role is mediated through
   prefix; local masking can't unwind it.
3. **Counterfactual evidence**: the image rejects the token by NOT
   containing the asked-about object. Masking 20% doesn't add the
   object; the absence remains.

All three mechanisms predict the observed pattern: VD is large (because
removing the WHOLE image lets the token become more probable under no
information), but Step 2 Δ ≈ 0 for any local mask.

## Proposed §Method gate (Step 3)

Given Step 2 + Step 2b:

```
g_t = 𝟙[c_t ∈ {content_noun, visual_attribute, proper_noun}]   # local-evidence category
    ∧ 𝟙[quad(t) ∈ {0, 1}]                                       # vis_support
    ∧ 𝟙[coverage(top_K(M_t), E_x) ≥ τ]                          # spatial coverage in sample bottleneck

w_t = 1 + α · g_t       (only boost, never suppress; per GPT round on 2f10687)
```

q3 tokens deliberately excluded. The §Method is honest about scope:

> TAM-Boost OPD addresses local-evidence support tokens (vd ≥ 0 quadrants
> with content-bearing categories). The orthogonal failure mode —
> visual rejection correction (vd < 0 ∧ adv < 0, T2-1's failure
> bucket) — has globally distributed evidence and is NOT addressed
> by this method.

Defaults (per GPT round on 2f10687):
- `token_topK = 20%`
- `sample_bottleneck_rho = 30%` (or 40%)
- `tau = 0.5` (coverage threshold)
- `alpha = 0.5`
- visual_number kept in holdout (Step 2 Δ ≈ 0; possibly separate mechanism)

## Questions for GPT

1. **Is "TAM addresses q0/q1, q3 is structurally orthogonal" a
   defensible §Method scope?** We will NOT claim TAM-Boost fixes T2-1's
   failure bucket — instead the paper notes the two failure modes are
   distinct and the rejection-side fix is open. Is this honest scoping
   publication-defensible, or does it look like cherry-picking?

2. **Should the paper still claim "TAM is a one-forward proxy for
   visual evidence"?** The honest version is "TAM is a one-forward
   proxy for LOCAL visual evidence on SUPPORT tokens". Is that strong
   enough to motivate the method tier, or does it weaken the headline
   to the point of being a §Diagnostic-only paper?

3. **Mechanism for q3's distributed evidence** — which of the three
   plausible explanations (scene-gist / prefix-conditioned /
   counterfactual) is most worth investigating empirically? Or do we
   park this entirely and just call it "globally distributed" in the
   paper?

4. **§Method gate design** — at the formula above, anything to
   add/remove? Specifically:
   - Should we include q2 (vis_reject_teacher_toward) in the gate?
     Step 2 Δ ≈ +0.03 — basically zero. Currently scoped out, which
     matches the data, but ask in case it changes paper framing.
   - Is the `coverage` threshold form right, or should we use a
     softer continuous weight like `min(coverage, 1.0)`?
   - The α = 0.5 default — is there a better way to pick it?

5. **Step 3 training plan**: we propose one initial training arm
   (Base OPD + TAM-Boost on q0/q1 supported categories) vs T1_2
   baseline. No compression yet. Acceptable, or do we need more
   controls in the first training run?

6. **The q3 finding suggests a complementary mechanism is needed.**
   The natural candidate is T2-2's `|VD|`-boost no-suppress on
   negative-VD tokens (abandoned earlier due to gloo barrier, but
   counterfactual passed 4/4). Should the paper include a "Future
   Work: combined TAM-Boost + |VD|-Boost" pointer, or save that for
   a follow-up?

## Out of scope (do not redo)

- The narrative flip (Step 1a Pearson r ≈ 0 vs Step 2 causal Δ
  >> 0) is settled in `docs/step1a-step2-results-2026-05-25.md`.
- Step 2 v2 sanity controls (scrambled-TAM = random, bottom_tam ≈ 0)
  are settled.
- v0.1.2 / v0.1.3 schema, MLLMOPD_ATTN_IMPL=eager, etc. are settled.

## Response format

- Q1–Q6 verdicts: agree / refine / block, one line each
- Specific §Method gate edits if any
- One paragraph: greenlight Step 3a training code (Base OPD + TAM-Boost
  on q0/q1 supported categories) or block + propose what to verify
  first?

**请用中文回复。**
