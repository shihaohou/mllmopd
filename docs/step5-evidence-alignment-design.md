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

If any bucket comes up short (e.g. opd_target only yields 22 disjoint),
the deficit moves to `Dataset_diversity` to preserve n=200.

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

Minimum-detectable-effect (MDE) at 95% bootstrap CI:

```
MDE ≈ 1.96 · bootstrap_sd(ΔM)
```

If `MDE > δ`, the audit is **underpowered**; we cannot distinguish
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
null calibration. Let `MDE_M = 1.96 · bootstrap_sd(ΔM)`.

| Outcome on `OPD_improved` × reliability | Statistical condition | Interpretation | Next step |
|---|---|---|---|
| **(a) aligned** | `mean(ΔJS) > δ_JS` AND `ci_low(ΔJS) > 0` (+ same direction for ΔIoU or ΔCos as confirmation) | OPD already implicitly aligns visual evidence | **EA-OPD motivation weak.** Drop the line; main §Method = T2-2/v3 from atlas data. |
| **(b) flat (equivalent)** | TOST: `ci_low(ΔJS) > −δ_JS` AND `ci_high(ΔJS) < +δ_JS`, AND `MDE_JS ≤ δ_JS` | OPD trains the output but not the evidence — *and the audit is powered enough to say so* | **Strong EA-OPD motivation.** EA-OPD becomes the §Method headline candidate; v3 C_local gate is the cheap-deploy variant. |
| **(c) split** | (a)-grade alignment on `template_token` / `answer_token` / `punctuation`, (b)-grade equivalence on `content_noun` / `visual_attribute` / `proper_noun` | OPD aligns the easy (already-vision-uniform) tokens but not the visually-anchored ones | **Both methods coexist.** v3 = cheap deploy; EA-OPD = research arm. |
| **(d) teacher-TAM unreliable** | Pre-condition: <50% of `OPD_improved` tokens pass the reliability filter (`tam_entropy_norm_T < 0.95`) | TAM signal too diffuse on this prompt distribution to ground EA-OPD | **Step 5 stops.** TAM stays as Step 2 motivation only; main §Method = T2-2/v3. |
| **(e) inconclusive** | TOST fails (CI strays outside [−δ, +δ]) AND `MDE_JS > δ_JS` | Underpowered for the (b) claim; no aligned direction either | **Increase n.** Either rerun with 400 samples or accept the audit cannot decide. |

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

## 13. References used during design

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
