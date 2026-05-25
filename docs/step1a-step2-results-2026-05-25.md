# TAM Step 1a + Step 2 — Combined Results & Narrative Flip (2026-05-25)

## TL;DR

We initially read Step 1a as a **negative result** for the TAM method
tier: per-token `corr(tam_mass_top20, |vd|) = −0.001` on n=9118 content_noun
tokens. Then Step 2 (causal masking on the same TAM maps) returned
**paired Δ = +0.988 nat with t-stat = 7.52** on n=649; content_noun
specifically **Δ = +1.244 nat** on n=410. Sanity controls all aligned.

**Step 1a and Step 2 are not contradictory — they test different
questions.** Scalar `tam_mass_top20` is not a continuous proxy for `|VD|`.
But the SPATIAL top-K region of the TAM map IS the causal evidence
locator for content-bearing tokens. The §Method (Step 3) design pivots
from "TAM mass as weight" to "TAM region as gate".

This is the methodological lesson we want GPT to weigh in on before
committing to Step 3 design.

## Run provenance

| Stage | Run dir | Commit | n | Verdict |
|---|---|---|---|---|
| Step 0 sanity | `tam_sanity_20260525-142508/` | `f03864b` (later retracted; recomputed at `93b8cd7`) | 4 probes, 866 tokens | PASS w/ caveats |
| Step 1a calibration | `tam_step1a_20260525-190333/` | `93b8cd7` (v0.1.3 fix) | 820 rows / 205 sample × 4 ckpt / 243k tokens | NEGATIVE on Pearson |
| Step 2 causal masking | `tam_step2_<latest TS>/` | `3d91e1e` | 3894 rows / 200 sample / 649 unique tokens × 6 strategies | **STRONG POSITIVE** |

All teacher = MMR1-7B-RL; students = {T1_0=MMR1-3B-SFT,
T1_2=t1_v1p5b_T1_2_full_mm/step_230, T1_3=t1_v1p5b_T1_3_blank_mm/step_230,
T2_1=t2_1_v0_T2_1_full_vd/step_230}.

## Step 1a headline (negative on continuous proxy)

```
per-token corr(tam_mass_top20, |vd|), pooled across 820 rows × 4 ckpts:

category          n        corr_tam_abs_vd    interpretation
content_noun     9118     −0.001 (CI [−0.022, +0.020])    confidently zero
visual_number    8344     −0.009                          ~zero
visual_attribute 2578     −0.011                          ~zero
proper_noun      3446     +0.062                          weak but non-zero
other           22710     +0.061                          heterogeneous bucket
punctuation     11794     −0.004                          ~zero
template_token    680     −0.082                          weak negative
special_token     448     −0.053
```

Pooled `corr(tam20, |vd|) = +0.044` across ALL token categories — weak.

Visual-rejection AP (binary, quad==3) showed a TRAINING TRAJECTORY:
- T1_0 baseline:  AP=0.032 (prev 2.4%) → lift = 1.33×
- T1_2 FullKD:    AP=0.055 (prev 4.8%) → lift = 1.15× (but ~1.72× T1_0's AP)
- T1_3 BlankKD:   AP=0.054 (prev 4.7%) → 1.15×
- T2_1 signed VD: AP=0.052 (prev 4.3%) → 1.21×

So binary AP showed a 1.7× absolute lift baseline → trained. Real but weak.

**Our initial read**: per-token mass is dead as a continuous estimator;
TAM-Boost OPD with `w_t = 1 + α·tam_mass_t` would amplify noise.

## Step 2 headline (strong positive on causal masking)

```
HEADLINE: paired Δ(top_tam_20pct − mean_random_20pct)
  n_tokens     = 649
  mean Δ       = +0.988 nat
  p50 Δ        = 0.000
  p95 Δ        = +9.36
  frac_positive = 0.522
  paired t-stat = 7.52    (p ≪ 10⁻¹²)

By token_category:
  content_noun      n_top=410   mean_top=+1.609   mean_random=+0.365   Δ=+1.244
  visual_attribute  n_top= 22   mean_top=+2.138   mean_random=+0.350   Δ=+1.787
  proper_noun       n_top=187   mean_top=+0.638   mean_random=+0.157   Δ=+0.481
  visual_number     n_top= 23   mean_top=+0.002   mean_random=−0.000   Δ=+0.003
  meta_cot_token    n_top=  5   ≈0
  answer_token      n_top=  1   ≈0  (insufficient n)
  pronoun           n_top=  1   single sample, ignore

Pooled by mask strategy (logp_drop mean across all selected target tokens):
  top_tam_20pct           n=649  mean=+1.27   ← MASK important region
  random_20pct_seed_42    n=649  mean=+0.32
  random_20pct_seed_43    n=649  mean=+0.23
  random_20pct_seed_44    n=649  mean=+0.31
  random_20pct_avg                       +0.29 (3-seed avg)
  keep_top_tam_20pct      n=649  mean=+0.44   ← KEEP only top-20%, mask other 80%
  bottom_tam_20pct        n=649  mean=+0.08   ← MASK least-important 20%
```

## Sanity controls (all aligned)

Four checks on the same masks, all in the right direction:

| Check | Magnitude | Verdict |
|---|---|---|
| `top_tam_20pct (+1.27) ≫ random_20pct_avg (+0.29)` | 4.4× | mask important > mask random ✓ |
| `bottom_tam_20pct (+0.083) ≪ random_20pct_avg (+0.29)` | 0.29× | mask least-important < random ✓ |
| `keep_top_tam_20pct (+0.44) ≪ top_tam_20pct (+1.27)` | 0.35× | keep important less damaging than mask important ✓ |
| `random_seeds` (+0.32, +0.23, +0.31) tight | σ ≈ 0.05 | random variance is small relative to the +1.0 nat signal ✓ |

## The narrative flip

Step 1a tested: **"does scalar `tam_mass_top20` predict scalar `|vd|`?"**
- Statistic: Pearson r per token, pooled across 9k+ content_noun tokens
- Answer: r = −0.001, confidently zero
- This is a correlational question about a COLLAPSED scalar

Step 2 tested: **"does masking the spatial top-K of the TAM map change
teacher's lp on a content-bearing token?"**
- Statistic: paired Δ(lp_full − lp_under_mask), top-TAM mask vs random
- Answer: Δ = +1.24 nat on content_noun, p ≪ 10⁻¹²
- This is an interventional question about a SPATIAL pattern

**These are different questions and the answers don't have to agree.**
Scalar mass loses spatial direction; a token can have low `tam_mass_top20`
(diffuse map) yet still have its top-K patches land on the right region.

### Implication for Step 3 §Method design

Pre-Step-2 plan was:

```
w_t = 1 + α · tam_mass_t
```

→ Step 1a Pearson r = 0 says this would amplify noise. Dead.

Post-Step-2 plan:

```
w_t = 1 + α · 𝟙[token_category(t) ∈ {content_noun, visual_attribute, proper_noun}]
              ∧ overlap(top_K(tam_map_t), evidence_region) ≥ τ ]
```

→ Region-mask gate, not mass weighting. The gate fires only on
categories where Step 2 showed nontrivial causal Δ. Magnitude
controlled by overlap threshold `τ`, not by a continuous scalar that
Pearson r already showed doesn't track.

`visual_number` showed Δ≈0 — surprising; ChartQA digits should be
image-grounded. Possible explanations:
1. Numbers are LANGUAGE-dependent (decimal format, units) more than
   pixel-grounded
2. Number tokens are often deeply nested in CoT, dominated by prefix
3. TAM's ECI removes too much "shared context" for digits since digits
   recur in CoT

`visual_attribute` Δ=+1.79 (highest per-category) suggests ADJECTIVES
("red", "large", "metallic") are the most image-tethered tokens.

## Questions for GPT

1. **Is the Step 1a → Step 2 narrative flip defensible as written?**
   We claim scalar mass ≠ causal locator. The data clearly supports
   this — but is this framing publication-worthy as a §Method-design
   pivot, or does it look like post-hoc justification?

2. **§Method gate formulation** — does the region-mask gate idea
   (w_t = 1 + α·𝟙[category ∧ overlap]) hold up? Concerns:
   - "overlap with evidence region" requires defining the evidence
     region. Two candidates: (a) overlap of top-K TAM patches between
     teacher's tam_map and student's tam_map (region agreement);
     (b) overlap with manually-defined GT bbox (not scalable).
   - Should `α` be uniform per category, or per (category × benchmark)?
   - Should `τ` (overlap threshold) be tuned or learned?

3. **Why does `visual_number` Δ ≈ 0?** All numbers from ChartQA /
   MathVista, where they're clearly image-grounded for the human. Is
   this a TAM limitation (digit tokens don't trigger TAM saliency on
   chart numerals) or a deeper finding about how MLLMs route numerical
   evidence?

4. **`frac_positive = 0.522` interpretation.** Only 52% of tokens show
   top-TAM > random. We interpret this as "the other 48% are
   image-independent tokens (template, CoT-filler) so both masks
   produce ~0 effect (tie)." Is this read defensible, or should we
   subset to image-dependent tokens before reporting Δ?

5. **Step 1b (student_rollout mode) priority.** Step 2's positive
   result was on teacher_greedy responses. Should we still run Step 1b
   (student rollout)? It would confirm the effect extends when student
   generates the response, which is closer to OPD training-time.
   But Step 2 already demonstrates the causal mechanism on shared
   teacher responses; 1b adds robustness, not a new question.

6. **Sanity check we haven't run**: SCRAMBLED-TAM control — generate a
   random permutation of the TAM map for each token, do top-K mask.
   If Δ ≈ 0 under scrambled-TAM but Δ = +1.2 under real-TAM, the
   spatial structure is what matters, not just "mask 20% of image".
   Worth adding?

7. **Paper §Method header before Step 3 training runs**: with Step 1a
   negative + Step 2 positive, what's the right paper structure?
   - Option A: §Diagnostic (Step 0+1a+2) + §Method (Step 3 outcome)
   - Option B: §Mechanism (Step 1a+2 disconnect is the result) +
     §Method (Step 3)
   - Option C: §Diagnostic only, Step 3 deferred to follow-up paper

## Out of scope (don't redo)

- The TAM-vs-attention baseline question is settled (attention
  baseline ≈ 0 under sdpa fallback; eager required to compare; not
  blocking)
- Step 0 token-category limitations (POS tagger overrides) are
  understood and not blocking
- Step 1b (student_rollout) is intentionally deferred; ask in Q5 if
  this changes

## Response format

- Q1–Q7 verdicts: agree / refine / block, with one-line reasoning
- §Method design specifics: which gate formulation to commit to
- One paragraph: greenlight writing Step 3 training code with the
  region-mask gate, or block + propose what to test first?

**请用中文回复。**
