# Step 5 — TAM Evidence Alignment Audit (design doc)

**Date:** 2026-05-27
**Author:** Shihao
**Status:** design locked, code in progress

> **Question:** Vanilla OPD makes the student say more of what the teacher
> says. Does it also make the student *look* where the teacher looks?
>
> **Method:** for a shared rollout of the OPD-trained student, compute the
> Token Activation Map (TAM, Li et al. ICCV 2025) of teacher / base
> student / OPD student on every response token, then compare alignment.

---

## 1. Motivation

Three motivations sit on top of each other:

1. **VA-OPD (Liu et al., arXiv 2605.21924)** showed that standard OPD
   improves output quality but may not increase the student's reliance on
   fine-grained visual detail. It defines a per-token Visual Advantage
   `lp_T(y|I) − lp_T(y|I_lowres)` and reweights *which* tokens the OPD
   loss emphasizes. VA-OPD answers **whether** a token depends on
   vision.
2. **RAL (Reinforced Attention Learning)** showed that aligning the
   student's *attention policy* to the teacher's on student-sampled
   trajectories outperforms vanilla KD on multimodal benchmarks. RAL
   answers a different question — *where* the student attends — but
   does so at the aggregate attention level, not per-token causally.
3. **Our own Step 2** (commit `3d91e1e`, n=649 target tokens, 6 mask
   strategies) established that **TAM top-K regions are causally
   load-bearing** for the teacher: paired Δ(top_tam_20pct −
   mean_random_20pct) = +0.988 nat, p ≪ 10⁻¹². On content_noun the
   per-category Δ rises to +1.244 nat.

These three converge on a gap that no published work targets directly:

> Vanilla OPD distills `p_T(token | image, prefix)`. The visual evidence
> that *led* the teacher to that token — the per-token TAM map — is
> never on the loss. So even if the student matches the teacher's
> output distribution, it may achieve that match via a different visual
> reasoning path (language prior, wrong region, scene gist).

Step 5 measures this gap. We do **not** train anything yet. The audit
output decides whether an Evidence-Aligned OPD method (EA-OPD) is worth
designing in a future iteration, or whether the gap is already small
enough that vanilla OPD is doing the alignment work implicitly.

---

## 2. Research questions

| ID | Question | How we measure |
|---|---|---|
| **Q1** | Does vanilla OPD bring the student's per-token TAM closer to the teacher's? | paired Δ = JS(S0,T) − JS(S1,T) over all response tokens (positive = S1 closer to T), stratified by category — see §6.3 |
| **Q2** | On which token categories is the alignment gap largest / smallest? | per-category bootstrap on the v0.1.3 classifier (content_noun, visual_attribute, proper_noun, answer, numeric, template, punctuation, …) |
| **Q3** *(deferred)* | Does the teacher's TAM on a student rollout differ from the teacher's TAM on its own rollout? | GPT supplementary; **not in Step 5 scope** — requires a separate teacher rollout pass |

Decision tree from Q1+Q2 is in §6.

---

## 3. Models and shared rollout

| Symbol | Model | Source | Role |
|---|---|---|---|
| **T** | MMR1-7B-RL | `$MMR1_7B_RL_CKPT` | reference teacher |
| **S0** | MMR1-3B-SFT | `$MMR1_3B_SFT_CKPT` | un-distilled base student |
| **S1** | T1-Full step_230 | `$MLLMOPD_RUNS/t1_v1p5b_T1_2_full_mm/ckpt/hf/step_230` | OPD-distilled student |

**Shared rollout protocol (critical):**

```
y = greedy_rollout(S1, image=I, question=q, system_prompt=MMR1, max_new=4096)
```

For each generated token `y_t` in `y`, the three models all see the
same `(I, q, y_<t, y_t)` and compute TAM in the same way:

```
M_T(t)  = TAM_full_pipeline(T,  I, q, y_<t, y_t)
M_S0(t) = TAM_full_pipeline(S0, I, q, y_<t, y_t)
M_S1(t) = TAM_full_pipeline(S1, I, q, y_<t, y_t)
```

Why S1 rollout and not S0 or teacher rollout:

- Sequences must be identical across the three models so token `t` is
  comparable. Each model rolling out its own sequence breaks `y_t`,
  `y_<t`, and the position alignment.
- We pick S1's rollout because **OPD's real-world target is the
  post-OPD student** — we want to know whether OPD got it to look in
  the right place at the tokens it actually generates.
- A complementary `R_S0` pass (S0 rollout, three models score) is a
  natural follow-up but **deferred** to keep Step 5 single-shot.

---

## 4. Sample selection — 200 stratified

The sample pool is composed at selection time based on actual
correctness of S0 / S1, so the sample list is **dataset-derived, not
fixed**.

### 4.1 Candidate pool

```
candidates = (
    dev_mmr1_v0_1k         # ~1000 prompts, MMR1-RL residual
  ∪ opd_target 133         # data/audit/opd_target_ids.json (if available)
  ∪ level1_subset (1200)   # ChartQA / MathVista / HallusionBench / POPE
)
```

### 4.2 Bucket protocol

```
PASS-1: run S0 + S1 greedy generation on the full candidate pool
        judge correctness via lightweight string-match + boxed answer extraction
PASS-2: stratified pick:
```

| Bucket | n | Definition | Purpose |
|---|---|---|---|
| `OPD_improved` | 70 | `S0 wrong ∧ S1 correct` | main signal — when OPD fixes the answer, does the evidence map also fix? |
| `OPD_failed` | 60 | `S0 wrong ∧ S1 wrong` | failure mode — is the evidence map *still* off? |
| `Teacher_advantage` | 30 | `id ∈ opd_target_ids`, disjoint from above | sensitive prompts, ground-truth visual reliance |
| `Dataset_diversity` | 40 | ChartQA + MathVista samples with strong visual content (chart / geometry), disjoint | qualitative reasonability check |
| **Total** | **200** | | |

**Hard-fail on deficit (per §11 item 3).** If any bucket cannot meet
its target, the selector aborts with a non-zero exit code and the user
must extend `--candidates` or lower the bucket target. Silent fallback
to `Dataset_diversity` would poison the §8 decision tree. Pilot /
debugging runs can opt in with `--allow-degraded-mode`; the
`step5-results.md` then carries a "decision tree NOT interpretable"
banner.

### 4.3 Output

`data/audit/tam_step5_samples_v0.jsonl` — one row per sample with the
Step 1a-compatible fields plus:

```jsonc
{
  "id": "...", "benchmark": "...", "image": "...",
  "question": "...", "answer": "...",
  "bucket": "OPD_improved" | "OPD_failed" | "Teacher_advantage" | "Dataset_diversity",
  "s0_response_text": "...", "s0_correct": true|false,
  "s1_response_text": "...", "s1_correct": true|false,
}
```

The selector caches `(s0_response, s1_response)` so the main runner can
also report the alignment Δ conditional on `(s0_correct, s1_correct)`
quadrants without re-running generation.

---

## 5. Pipeline

```
                                            ┌──────────────────────┐
                                            │  candidates JSONL    │
                                            └──────────┬───────────┘
                                                       │
                              tam_step5_sample_selector.py
                              (S0 + S1 generate, judge, bucket)
                                                       │
                                            ┌──────────▼───────────┐
                                            │  samples_v0.jsonl    │  200 rows
                                            └──────────┬───────────┘
                                                       │
                              tam_step5_evidence_alignment.py
                                                       │
                              ┌────────────────────────┼────────────────────────┐
                              │                        │                        │
                  Pass 1: S1 rollout (greedy, max 4096)│                        │
                              │                        │                        │
                              ▼                        ▼                        ▼
                  Pass 2 (T):              Pass 2 (S0):             Pass 2 (S1):
                  forced-decode + TAM      forced-decode + TAM      forced-decode + TAM
                              │                        │                        │
                              └────────────────────────┼────────────────────────┘
                                                       │
                                    Pass 3: per-token alignment metrics
                                    (IoU / JS / Cosine + reliability scalars)
                                                       │
                                            ┌──────────▼───────────┐
                                            │ alignment.jsonl      │  N rows
                                            │ maps/<id>/*.b64      │  raw maps
                                            └──────────┬───────────┘
                                                       │
                                 src/mllmopd/analysis/tam_step5_analyzer.py
                                                       │
                                            ┌──────────▼───────────┐
                                            │ tables/*.csv         │
                                            │ figures/{1,2,3,4}.png│
                                            │ qualitative/*.png    │  12 cases
                                            │ step5-results.md     │
                                            └──────────────────────┘
```

**Reuse from Step 1a:**

| Component | Reused as-is |
|---|---|
| `_tam_core.TAM` + `tam_scalars` | yes — single-image, no LaTeX |
| v0.1.3 OOM-safe forward (bare generate → single teacher-forced fwd) | yes |
| `_classify_tokens_v012` (v0.1.3 logic) | yes, for token_category stratification |
| `_build_model` + eager attn fallback | yes |
| `MMR1_SYSTEM_PROMPT` placement | yes |
| Multi-GPU shard launcher pattern (`shard_id::num_shards`) | yes |
| `_b64_uint8_map` for inline map storage | yes |

**New for Step 5:**

| Component | Why new |
|---|---|
| Three-model TAM pass on shared rollout | Step 1a only did teacher TAM; students were lp-only |
| Sample selector with S0/S1 correctness pre-pass | Step 1a's `tam_step1_subset.py` doesn't need correctness signals |
| Alignment metrics (IoU / JS / Cos on M_T vs M_S0/S1) | Step 1a stored scalars per token, not pairwise metrics |
| Decision-tree analyzer with bootstrap CI | Step 1a/2 analyzers don't run paired Δ across models |

---

## 6. Metrics

For each token `t` with valid TAM on all three models:

### 6.1 Primary alignment metrics (against teacher)

```
IoU_top20(M_X, M_T)   where X ∈ {S0, S1}
  = |TopK20%(M_X) ∩ TopK20%(M_T)| / |TopK20%(M_X) ∪ TopK20%(M_T)|

JS(M_X, M_T)
  = 0.5 · KL(M_X ‖ avg) + 0.5 · KL(M_T ‖ avg),  avg = 0.5(M_X + M_T)
  (computed in patch-probability space: normalize map → probability vector)

Cos(M_X, M_T)
  = ⟨flatten(M_X), flatten(M_T)⟩ / (‖M_X‖₂ · ‖M_T‖₂)
```

### 6.2 Reliability scalars (already in Step 1a schema)

For every token × every model:

- `tam_mass_top10`, `tam_mass_top20` — concentration
- `tam_entropy`, `tam_entropy_norm` — diffuseness; low = trustworthy map
- `tam_effective_patch_frac` — exp(entropy) / n_patches

**Reliability filter (drives the decision; not just sensitivity):** the headline decision
restricts to tokens where the teacher map satisfies `tam_entropy_norm_T
< 0.95` (teacher concentration above the random-uniform baseline).
All-token aggregates are also reported but flagged as exploratory only —
without the filter, tokens whose teacher map is uniform get scored
against meaningless targets and dilute the signal. The 0.95 threshold
was chosen because Step 0 found `tam_entropy_norm ∈ [0.85, 0.99]` on
non-content tokens; 0.95 is the elbow in the entropy distribution.

### 6.3 Headline quantity

```
ΔJS(S1−S0, T)     =  JS(S0, T) − JS(S1, T)   (positive = OPD aligned)
ΔIoU(S1−S0, T)    =  IoU(S1, T) − IoU(S0, T) (positive = OPD aligned)
ΔCos(S1−S0, T)    =  Cos(S1, T) − Cos(S0, T) (positive = OPD aligned)
```

Reported with paired bootstrap CI (10k resamples, sample-level cluster)
because tokens within a sample are not independent.

### 6.4 Null calibration of δ (alignment "noise floor")

Hardcoded thresholds like "ΔJS > 0.05" beg the question of whether
0.05 is large or small relative to model-pair noise. We calibrate the
practical equivalence margin `δ` empirically per metric via three
nulls:

```
Null A — sample-label permutation:
    For each sample, randomly swap M_S0 ↔ M_S1 (token-wise) and recompute
    ΔJS/ΔIoU/ΔCos. Repeat 1000×. δ_A = 95th percentile of |Δ_null|.

Null B — cross-sample pairing:
    Pair (M_T from sample i, M_S0/S1 from sample j ≠ i) and compute the
    same metrics. δ_B = 95th percentile.

Null C — spatial scramble:
    Apply a fixed permutation of patch positions to M_S0 and M_S1
    (preserve TAM concentration, destroy spatial alignment with M_T).
    δ_C = 95th percentile.
```

The headline `δ` per metric is `max(δ_A, δ_B, δ_C)` — the most
conservative noise floor. These calibrations run on the same audit data
(no extra forward), cost <1 minute, and are emitted as a
`tables/null_calibration.csv` artifact.

### 6.5 Decision via TOST equivalence test (NOT CI-cross-zero)

`branch (b)` — the most desirable outcome for EA-OPD motivation — used
to be defined as "CI crosses 0 and mean ≈ 0". That is statistically
incorrect: it conflates "no evidence of effect" with "evidence of no
effect" (Type II error masquerading as a finding).

Replaced with two-one-sided-tests (TOST) equivalence:

```
TOST equivalence on metric M, threshold δ:
    declare equivalence iff
        ci_low(ΔM)  > −δ
    AND ci_high(ΔM) < +δ
```

Minimum-detectable-effect (MDE) at the 95% bootstrap CI — bootstrap
percentile half-width, robust to skewed bootstrap distributions:

```
MDE_95 = max(|mean − ci_low|, |ci_high − mean|)
```

(Earlier versions used `1.96·bootstrap_sd`; replaced per GPT review of
commit `6a081aa` because skewed bootstraps can let `1.96·sd`
under-estimate the CI half-width.)

If `MDE_95 > δ`, the audit is **underpowered**; we cannot distinguish
"flat" from "small alignment". Report `inconclusive`, not `branch (b)`.

### 6.6 Stratifications (always reported)

| Axis | Levels |
|---|---|
| Bucket | `OPD_improved`, `OPD_failed`, `Teacher_advantage`, `Dataset_diversity` |
| Token category (v0.1.3) | `content_noun`, `proper_noun`, `visual_attribute`, `visual_number`, `answer_token`, `template_token`, `pronoun`, `spatial_relation`, `meta_cot_token`, `special_token`, `punctuation`, `ocr_text`, `other` (13 total) |
| Reliability | reliability-filtered (drives §8 decision) vs all-tokens (exploratory) |

The **headline decision cell** is `OPD_improved bucket × reliability-
filtered tokens`. All other cells are reported but do not drive the §8
branch label.

---

## 7. Output schema

`runs/audit/tam_step5_<TS>/alignment.jsonl` — one row per sample (not
per token; token-level arrays inline):

```jsonc
{
  "id": "...", "benchmark": "...", "bucket": "OPD_improved",
  "image_path": "...", "image_sha256": "...",
  "question": "...", "answer": "...",
  "s0_response_text": "...", "s0_correct": true,
  "s1_response_text": "...", "s1_correct": true,

  "rollout_source": "S1_greedy",
  "rollout_model": "$MLLMOPD_RUNS/.../step_230",
  "response_ids":   [int...],
  "response_text":  "...",
  "response_length": 312,
  "response_hash":   "sha16",
  "tokens":          ["...", ...],
  "token_category":  ["content_noun", ...],   // v0.1.3
  "is_answer_token": [bool, ...],

  // per-token scalars per model (lp not central but useful)
  "T":  { "tam_mass_top20": [..], "tam_entropy_norm": [..], "lp": [..] },
  "S0": { "tam_mass_top20": [..], "tam_entropy_norm": [..], "lp": [..] },
  "S1": { "tam_mass_top20": [..], "tam_entropy_norm": [..], "lp": [..] },

  // per-token alignment (S0,T) and (S1,T)
  "align": {
    "S0_T": { "iou_top20": [..], "js": [..], "cos": [..] },
    "S1_T": { "iou_top20": [..], "js": [..], "cos": [..] }
  },

  // QC
  "tam_valid_T":  [bool, ...],
  "tam_valid_S0": [bool, ...],
  "tam_valid_S1": [bool, ...],

  "image_grid_thw": [...], "vision_shape": [H, W], "n_patches": HW,
  "map_h": H, "map_w": W,

  // raw maps as base64 uint8 (inline; mass ~ 1.5 KB/token for 32×32 grid)
  // Stored for ALL tokens — at 200 samples × ~300 tokens × 3 models that
  // is ~270 MB JSONL, acceptable. Lets the renderer pick any token later.
  "maps_b64": {
    "T":  [b64, ...],
    "S0": [b64, ...],
    "S1": [b64, ...]
  },

  // schema/version + commit
  "tam_preproc_version":   "v0.1.3",
  "code_commit_run":       "<git rev>",
  "step5_schema_version":  "v0.1"
}
```

**Map storage policy:** Inline uint8 b64 for every token. At
H×W=32×32×3 maps/token × 300 tokens × 200 samples ≈ 600 MB JSONL.
Acceptable; lets the analyzer + renderer pick any token without
re-running. If size becomes an issue, switch to sidecar npz keyed by
`response_hash`.

---

## 8. Decision tree

**Primary decision cell:** `OPD_improved` bucket × reliability-filtered
tokens. The branch label below is computed from this cell only. The
other buckets / category strata are reported but do not drive the
decision (consistent with §6.5 — the audit is designed to answer "does
OPD align evidence on the tokens OPD actually generates and that the
teacher map can meaningfully grade?").

**Sign convention (positive ΔJS / ΔIoU / ΔCos = OPD aligned).**

Let `δ_JS`, `δ_IoU`, `δ_Cos` be the per-metric noise floors from §6.4
null calibration. Let
`MDE_M = max(|mean − ci_low|, |ci_high − mean|)` (bootstrap percentile
half-width — see §6.5).

| Outcome on `OPD_improved` × reliability | Statistical condition | Interpretation | Next step |
|---|---|---|---|
| **(a) aligned** | `mean(ΔJS) > δ_JS` AND `ci_low(ΔJS) > 0` AND ΔIoU also aligned (same condition). Raw Cos is reported as diagnostic only — not used for confirmation because non-mean-centered cosine is prone to false positives from globally-active maps. | OPD already implicitly aligns visual evidence | **EA-OPD motivation weak.** Drop the line; main §Method = T2-2/v3 from atlas data. |
| **(b) flat (equivalent)** | TOST: `ci_low(ΔJS) > −δ_JS` AND `ci_high(ΔJS) < +δ_JS`, AND `MDE_JS ≤ δ_JS` | OPD trains the output but not the evidence — *and the audit is powered enough to say so* | **Strong EA-OPD motivation.** EA-OPD becomes the §Method headline candidate; v3 C_local gate is the cheap-deploy variant. |
| **(c) split** | (a)-grade alignment on `template_token` / `answer_token` / `punctuation`, (b)-grade equivalence on `content_noun` / `visual_attribute` / `proper_noun` | OPD aligns the easy (already-vision-uniform) tokens but not the visually-anchored ones | **Both methods coexist.** v3 = cheap deploy; EA-OPD = research arm. |
| **(d) teacher-TAM unreliable** | Pre-condition: `n_reliable < 30` OR `rel_rate < 0.20` on the `OPD_improved` bucket. Original draft used `rate < 0.50` against `entropy_norm_T < 0.95`; both numbers were calibrated on v0.1.2 + 3B + short prompt (Step 0/2 regime). v0.1.3 + 7B + long CoT saturates entropy at ≥ 0.99 (smoke 2026-05-28, n=4355: min=0.9936, p75=0.9982), making the original cutoff a design artifact. New trigger separates *true* TAM saturation (absolute reliable-token count below a usable-statistics floor) from *partial* concentration (relative rate at a sensible floor). | TAM signal too diffuse to ground EA-OPD | **Step 5 stops.** TAM stays as Step 2 motivation only; main §Method = T2-2/v3. |
| **(e) inconclusive / underpowered** | (i) TOST fails (CI strays outside [−δ, +δ]) AND `MDE_JS > δ_JS`; OR (ii) primary cell has `n_samples < 10` (min-n guard) | Underpowered for the (b) claim; no aligned direction either | **Increase n.** Either rerun with 400 samples or accept the audit cannot decide. |

Each branch label is also computed on the all-tokens (exploratory)
aggregation and on each non-primary bucket — these go into
`step5-results.md` as a "decision sensitivity matrix". A branch that
appears only in the primary cell is reported with that scoping; a
branch consistent across cells is the strongest finding.

---

## 9. Cost estimate

### 9.1 Wall clock

Per-sample cost dominated by:

```
1 generation (S1, max 4096, greedy):    ~3-8 s
3 teacher-forced forwards + TAM:        3 × (~6-12 s) = ~18-36 s
```

Per sample: **~25-45 s**. For 200 samples × 1 GPU: **~1.5-2.5 hours**.
At `NUM_GPUS=8`: **~12-20 min**.

The candidate-pool generation pass (sample selector) is the heavier
cost: ~2k candidates × (S0 + S1 generate) ≈ **~30-50 min on 8 GPUs**.

Total Phase B: **~1-1.5 hours wall clock on H800**.

### 9.2 Storage

- `samples_v0.jsonl`: ~200 KB
- `alignment.jsonl` + inline maps: **~600 MB**
- Renderer PNG (12 qualitative cases): ~50 MB

---

## 10. File layout

```
docs/step5-evidence-alignment-design.md             ← this file
docs/step5-results.md                                ← written by analyzer

scripts/audit/tam_step5_sample_selector.py          ← Pass 0
scripts/audit/tam_step5_evidence_alignment.py       ← Pass 1-3
scripts/audit/run_tam_step5.sh                       ← orchestration
scripts/audit/tam_render_overlays.py                ← already exists, reused

src/mllmopd/analysis/tam_step5_analyzer.py          ← tables + 4 figures + decision

data/audit/tam_step5_samples_v0.jsonl                ← selector output
data/audit/tam_step5_candidates.jsonl                ← assembled candidate pool

runs/audit/tam_step5_<TS>/
    alignment.jsonl
    shard_<i>/  (multi-GPU split, merged into alignment.jsonl)
    summary.txt

docs/figures/step5/
    fig1_alignment_delta_overall.png
    fig2_per_bucket.png
    fig3_per_token_category.png
    fig4_qualitative_triplet_<12>.png
```

---

## 11. Prerequisites (hard-fail by default)

These conditions are **enforced at runtime** — Step 5 refuses to
launch unless they hold (override via explicit
`--allow-degraded-mode` for pilot debugging only):

1. **`opd_target_ids.json` exists and yields ≥ `n_teacher_advantage`
   ids disjoint from `OPD_improved` and `OPD_failed`.** Without the
   `Teacher_advantage` bucket, the §8 decision tree is uninterpretable
   (the bucket is the most direct link between teacher visual advantage
   and Step 5 alignment). Fallback to `Dataset_diversity` would silently
   re-weight the audit toward generic visual content.
2. **Sample selector and main runner share the same `--max-new-tokens`
   cap** (default 4096 for both). A mismatch — e.g. selector at 1024,
   runner at 4096 — means the bucket judge may fire on truncated
   responses while the alignment audit runs on the full response.
3. **All four bucket targets met (`70 / 60 / 30 / 40`).** If the
   candidate pool can't yield the target counts, the runner aborts and
   the user must extend `--candidates`. We do not silently shrink
   buckets or fill from neighbors.
4. **Multi-GPU shard merge:** post-cat row count must equal `Σ shard
   alignment rows`, modulo samples where any of T/S0/S1 returned
   `tam_valid=False`. The launcher's `analyze` phase refuses to start
   unless this check passes.

## 12. What this audit explicitly does NOT do

- **No new training**. Three models load weights only.
- **No method design**. The five decision-tree outcomes branch to
  *separate* follow-up tasks.
- **No causal masking validation**. Already done in Step 2 (commit
  `3d91e1e`), result reused as a prerequisite — see §6.3 of
  `progress-report-2026-05-26.md`. Teacher TAM **is** causally
  load-bearing; that question is settled.
- **No teacher-on-teacher-rollout TAM**. Deferred to Q3 supplementary.
- **No attention-baseline comparison**. Skipped to save wall clock
  (~3-5× speedup) — attention vs TAM is already established in Step 0
  (Pearson r = 0.032 over 866 tokens, commit `f03864b`).
- **No prompt-sensitivity audit on the teacher.** Deferred to P1
  follow-up (per GPT review on commit `13d73c1`).
- **No S0 rollout robustness check.** Deferred to P1 — main claim
  stands on S1 rollout per §3.

---

## 13. Framework limitations (added 2026-05-28 after audit + GPT review)

The audit completed (run `2026-05-28-013142`, n=350) and decided branch
(b) flat / TOST equivalent on the primary cell. A GPT framework review
on the result identified a limitation that this section makes explicit
so future readers and §Method work do not over-claim.

### 13.1 Conditional-on-S1-rollout, not counterfactual self-rollout

In Pass 1 only **S1** generates a greedy rollout `y_S1`. In Pass 2 all
three models (T / S0 / S1) are then teacher-forced to decode that
*same* `y_S1`, and per-token TAM is extracted at each position. The
audit therefore measures **conditional TAM divergence** under a fixed
trajectory:

> Given the same image, question, prefix `y_<t`, and target token `y_t`
> (drawn from S1's rollout), do T, S0, and S1 attend to the same
> image regions?

It does **not** measure the counterfactual:

> When S0 *itself* generates its own rollout (which may emit different
> tokens, visit different visual premises, and take a different
> reasoning chain), where does S0 attend?

The branch (b) TOST equivalence result therefore supports:

> Under the OPD-distilled student's own deployment trajectory, vanilla
> OPD does not systematically move token-level visual evidence maps
> closer to the teacher.

It does **NOT** support the stronger claim:

> Vanilla OPD does not change where the model looks during its own
> self-rollout.

### 13.2 Why this design choice was deliberate

The S1-rollout protocol is not an oversight. The OPD deployment target
*is* S1, so the relevant question for downstream method design is "on
the trajectory that the OPD-trained student actually generates, where
does its attention go relative to teacher?" This audit answers that
question cleanly. The cost is that it cannot rule out self-rollout
attention shifts.

### 13.3 Pass 4 to close the gap — implemented 2026-05-28

1. Generate `R0 = S0_greedy_rollout(I, q)` per sample (alongside the
   existing `R1 = S1_greedy_rollout`).
2. Cross-model TAM on `R0`: produce `T@R0`, `S0@R0`, `S1@R0` using the
   same Pass 2 pipeline (text trajectory differs, but per-sample
   alignment-summary metrics are still computable).
3. Population-level comparison (paired across sample ids):
   - Δ_self_traj = `align(S1_self_on_R1, T@R1) − align(S0_self_on_R0, T@R0)`
     — does OPD raise self-trajectory teacher alignment?
   - Δ_s0_cross  = `align(S0@R0, T@R0) − align(S0@R1, T@R1)` — does
     forcing S0 to S1's trajectory mask out a divergence that S0's own
     trajectory would show? (Non-zero ⇒ original Pass-3 protocol is
     systematically biased; close to zero ⇒ Pass-3 Δ is fairly
     interpretable.)
4. Optional: hand-annotate 30-50 OPD_improved samples into a bottleneck
   taxonomy (L1 evidence-selection / L2 evidence-interpretation /
   L3 evidence-use / L4 language-prior) to characterize *which*
   bottleneck dominates the bucket.

Pass 4 is **not a prerequisite** for the existing branch (b) conclusion
— it strengthens the §Method motivation by disambiguating which
mechanism EA-OPD or an alternative (e.g., grounded visual-premise
distillation, visual-reasoning chain distillation) should target.

#### Implementation status (2026-05-28)

- **Pipeline code: shipped.** `scripts/audit/tam_step5_evidence_alignment.py`
  gained `--enable-pass4` + `--pass {1R0, 2T_R0, 2S0_R0, 2S1_R0, 3R0,
  all_R0, all_with_R0}`. R0 files use the `_R0` suffix in the same
  out-dir so an existing R1 audit can be Pass-4-extended in-place
  (resume-safe). Schema bumped: R1 stays `v0.1`, R0 stamped `v0.2` with
  `rollout_source = "S0_greedy"`.
- **Analyzer: shipped.** New `src/mllmopd/analysis/tam_step5_pass4_compare.py`
  reads `alignment.jsonl` + `alignment_R0.jsonl`, applies the same
  reliability filter as Pass-3 primary cell (entropy_norm_T < 0.998),
  pairs by sample id, and emits per-sample / overall / per-bucket
  bootstrap-CI tables + TOST verdict, plus `pass4_decision.json` mirroring
  the schema of the original `decision.json`. The decision tree has two
  axes — SELF (A1 aligned / A2 flat / A3 inconclusive / A4 anti) and
  CROSS (B1 neutral / B2 biased / B3 anti-biased / B4 inconclusive).
- **Launcher: shipped.** `run_tam_step5.sh` gained `PHASE=pass4`
  (multi-GPU R0 rollout + Pass-2 on R0 + Pass-3 on R0; reuses existing
  RUN_DIR) and `PHASE=pass4_analyze` (single-process comparison →
  `docs/figures/step5/pass4/`). `PHASE=all` is unchanged (R1-only) for
  backward compatibility.
- **Awaiting run.** No GPU run yet; ~2 hr on 8×H800 estimated at n=350.
  Trigger via `PHASE=pass4 RUN_ID=tam_step5_20260528-013142
  bash scripts/audit/run_tam_step5.sh` from the H800 box, then
  `PHASE=pass4_analyze` with the same `RUN_ID`. Until results land, the
  honest claim boundaries in §13.1/§13.5 stand as written.

### 13.4 What the qualitative cases hint at (and do not prove)

`POPE_adversarial/269` is illustrative: in S1's rollout S1 says "yellow
basket on motorbike, not a bicycle → no" (correct); S0 in its *own*
rollout says "wheel on a pole → yes" (wrong, gold = no). Under the
conditional setting both S0 and S1 (forced to "basket") attend to the
same yellow basket region; only the reasoning chain differs. This is
consistent with a **Layer-2/3 bottleneck** (visual recognition /
reasoning interpretation) dominating OPD_improved rather than a
Layer-1 bottleneck (evidence localization). But it is anecdotal — a
single case cannot characterize the bucket; §13.3 Pass 4 + taxonomy
annotation are the proper measurement.

### 13.5 Implications for the §Method line

EA-OPD remains a well-defined method candidate: vanilla OPD does not
distill evidence alignment in any sense (conditional or counterfactual),
so adding an explicit teacher-TAM-distillation loss is a non-redundant
supervision signal. The audit, however, does not establish that
evidence-selection is the bottleneck of OPD_improved samples. The
relative-value comparison between EA-OPD and alternative §Method
directions (grounded visual-premise distillation, visual-reasoning
chain distillation) is open and should be informed by Pass 4 results.

## 14. References used during design

- Li et al., *Token Activation Map to Visually Explain Multimodal
  LLMs*, ICCV 2025 (Oral), arXiv 2506.23270.
- Liu et al., *Visual-Advantage On-Policy Distillation for
  Vision-Language Models*, arXiv 2605.21924 (2026-05).
- Our Step 1a/Step 2 results, `docs/progress-report-2026-05-26.md`.
- v0.1.3 TAM gotcha: `[[qwen25vl-generate-hidden-states-full-seq]]`.
- eager-attn requirement: `[[eager-attn-for-attention-output]]`.
- Cross-run non-determinism caveat:
  `[[teacher-greedy-fp-nondeterm]]` — Step 5 uses **one** S1
  rollout, so the within-sample comparison is bit-exact.
