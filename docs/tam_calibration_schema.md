# Step 1 TAM-VD Calibration: JSONL Schema (draft v0.1.1)

Status: **draft, not yet wired up**. This is the schema Step 0 (TAM sanity-check
on Qwen2.5-VL), Step 1a (teacher-greedy TAM-VD calibration), and Step 1b
(student-rollout audit) will jointly write — each emits a subset of these
fields tagged by `response_source`.

## What's new in v0.1.1 vs v0.1

GPT review on commit `0f85444` flagged 14 schema additions and one substantive
conceptual gap. Resolution in v0.1.1:

- **Step 1 split into 1a / 1b**:
  - **1a** (`response_source=teacher_greedy`): 150-200 samples × 4 ckpt. The
    main TAM-vs-VD calibration. Teacher's deterministic greedy response keeps
    `(lp_full, lp_blank, vd, tam_*)` checkpoint-invariant.
  - **1b** (`response_source=student_rollout_{ckpt}`): ~50 samples × 4 ckpt.
    Student rolls out at T=0 greedy; teacher then scores the student rollout
    for `(lp_full, lp_blank, vd, tam_*)`. Required to make the
    "quad==3 = T2-1 mis-targeted bucket" paper claim — that claim is only
    valid on the tokens the student actually generated.
- **~14 new fields**: mode tag, stable `(token_uid, response_hash)` for Step 2
  join, prompt segment spans, `tam_mass_top{10,20,40}` scalars, normalized
  entropy, processor/tokenizer hashes, QC flags, confound-disentangler
  logp/entropy/margin, explicit map dims.
- **Sample mix shifted**: ChartQA 30-40 → 60-80 (biggest TAM dynamic range).
  `text_only` negative control → `blank_image` / `irrelevant_image` with
  vision tokens (TAM is undefined without vision tokens).
- **Step 2 (causal masking) storage still deferred**, but join keys now
  pre-allocated so the sibling JSONL can land without churning v0.1.x rows.

## Decisions

| Q | Decision | Status |
|---|---|---|
| Q1 mass scalar | `tam_mass_top{10,20,40}` (3 scalars/token) + `tam_entropy` + `tam_entropy_norm = H / log N_patches` | locked v0.1.1 |
| Q2 prompt-token TAM | Store `tam_*_prompt[P]` in both Step 0 and Step 1, with `prompt_segments` disambiguating system / question / image_placeholder | locked v0.1.1 |
| Q3 causal masking (Step 2) | Storage deferred; join keys pre-allocated (`token_uid`, `response_hash`, `patch_index_order`) | partial — full design re-open after Step 1 lands |
| Q4 `tam_maps_subset` | K=20 stratified (5 top-`\|vd\|` + 5 top-`\|adv\|` + 5 blankness + 5 answer-critical), with `selection_rank` + `deduped_from_strata` to handle overlap | locked v0.1.1 |
| Q5 response source | Step 1a `teacher_greedy` (main) + Step 1b `student_rollout` (~50 × 4) | locked v0.1.1 |

Context: per `docs/handoff-2026-05-25-tier2a-gpt-in-t2-2-paused.md` we
decided to move TAM (arXiv 2506.23270) into Tier-2a as a one-forward
evidence estimator for OPD teacher correction. The calibration question:
**does `tam_mass` per response token correlate with `|vd|` overall, and
(critically) with `vd<0 ∧ adv<0` visual-rejection tokens?** If yes, TAM is a
one-forward proxy for our current two-forward VD; if no, abandon the
method-tier port and keep TAM only as appendix viz.

Existing diag JSONL we extend (from `opd_diagnostics_hook.py:17-29`):

```
{ id, step, response_length, image_mode, teacher_url, response_correct,
  lp_full[R], lp_blank[R], vd[R], old_lp_student[R] }
```

## Row schema

One row per `(sample, student_ckpt, response_source)`. For Step 1a the
teacher-side fields are checkpoint-invariant (single teacher forward, copied
across ckpt rows); for Step 1b the response itself depends on student_ckpt.

```json5
{
  // — identity —
  "id": "POPE_adversarial/1636",
  "benchmark": "POPE_adversarial",
  "split_tag": "opd_target | chartqa_collapse | hallusionbench | pope | neg_control",
  "image_path":   "data/audit/images/POPE_adversarial/1636.png",
  "image_sha256": "...",
  "question": "Is there a tennis racket in the image?",
  "answer":   "yes",

  // — mode / checkpoints —
  "response_source": "teacher_greedy | student_rollout_T1_0 | student_rollout_T1_2 | student_rollout_T1_3 | student_rollout_T2_1",
  "teacher_ckpt": "MMR1-7B-RL",
  "student_ckpt": "T1_0_base | T1_2_step230 | T1_3_step99 | T1_3_step230 | T2_1_step230",

  // — run metadata (~constant across rows of one run) —
  "tokenizer_name_or_path": "MMR1/MMR1-3B-SFT",
  "tokenizer_vocab_hash":   "sha256:...",
  "processor_name_or_path": "MMR1/MMR1-3B-SFT",
  "tam_preproc_version":    "v0.1.1",
  "tam_score_def": {
    "mass_top_K":   [10, 20, 40],
    "entropy":      "H = -Σ softmax(act_i)·log softmax(act_i) over patches (post rank-Gaussian); tam_entropy_norm = H / log(N_patches)",
    "map_pipeline": "lm_head(last_hidden)[:, img_idx, cls_id] → ECI subtract → clip ≥ 0 → rank_gaussian_filter(3) → normalize 0-1"
  },

  // — response (depends on response_source) —
  "response_text":   "<think>...</think><answer>Yes</answer>",
  "response_ids":    [/* int × R */],
  "response_length": 187,                       // R
  "response_hash":   "sha256(response_ids):hex[:16]",
  "tokens":          [/* str × R */],
  "token_idx":       [/* int × R */],            // 0..R-1; kept explicit for downstream join
  "token_uid":       [/* str × R */],            // f"{id}:{student_ckpt}:{response_source}:{response_hash}:{token_idx}"

  // — per-token teacher signals on RESPONSE tokens —
  "lp_full":     [/* float × R */],
  "lp_blank":    [/* float × R */],
  "vd":          [/* float × R */],              // = lp_full - lp_blank
  "tam_mass_top10": [/* float × R */],
  "tam_mass_top20": [/* float × R */],           // primary scalar
  "tam_mass_top40": [/* float × R */],           // robustness
  "tam_entropy":      [/* float × R */],         // raw entropy over patches
  "tam_entropy_norm": [/* float × R */],         // = tam_entropy / log(N_patches), ∈ [0, 1]
  "tam_effective_patch_frac": [/* float × R */], // = exp(tam_entropy) / N_patches

  // — confound disentanglers (Q: is TAM just a teacher confidence proxy?) —
  "teacher_entropy_full":     [/* float × R */], // teacher per-token output entropy under full image
  "teacher_top1_margin_full": [/* float × R */], // teacher top1 − top2 logprob

  // — TAM on PROMPT tokens (decision Q2) —
  "prompt_length":  87,                          // P
  "tokens_prompt":             [/* str × P */],
  "tam_mass_top20_prompt":     [/* float × P */],
  "tam_entropy_norm_prompt":   [/* float × P */],
  "prompt_segments": {                           // half-open intervals in prompt-token space
    "system":            [0, 35],
    "image_placeholder": [36, 86],
    "question":          [87, 102]
  },

  // — per-token student signals (depends on student_ckpt; for Step 1b, recorded inline during student greedy rollout) —
  "student_lp":      [/* float × R */],
  "student_entropy": [/* float × R */],
  "adv":             [/* float × R */],          // = lp_full - student_lp

  // — pre-computed quadrant labels —
  // 0 = vis_support_agree              (vd ≥ 0 ∧ adv ≥ 0)
  // 1 = vis_support_pushed_away        (vd ≥ 0 ∧ adv < 0)
  // 2 = vis_reject_teacher_pushed_toward (vd < 0 ∧ adv ≥ 0)
  // 3 = vis_reject_correction          (vd < 0 ∧ adv < 0)   ← KEY T2-1 bucket
  "quad": [/* int × R */],

  // — token-text labels —
  "is_blankness_token": [/* bool × R */],        // matches BLANK_RE in t1_blankness_trajectory.py:39
  "is_answer_token":    [/* bool × R */],
  "is_think_token":     [/* bool × R */],
  "answer_span_source": "regex_final_answer | mcq_extractor | manual | null",
  "answer_token_spans": [[/* start, end exclusive */]],
  "think_token_spans":  [[/* start, end exclusive */]],

  // — image metadata (Qwen2.5-VL specifics) —
  "image_grid_thw": [1, 24, 32],                 // t,h,w post-patch grid from processor
  "vision_shape":   [12, 16],                    // after 2x2 packing — TAM's vision_shape
  "n_patches":      192,                         // = vision_shape[0] * vision_shape[1]
  "map_h":          12,
  "map_w":          16,
  "patch_index_order": "row_major_top_left",     // explicit, so Step 2 mask indices are reproducible

  // — QC —
  "tam_valid":          true,                    // false if TAM failed (map all-zero, shape mismatch, ...)
  "tam_failure_reason": null,                    // null | "all_zero" | "shape_mismatch" | "special_id_split_failed" | "ECI_lstsq_failed"

  // — TAM maps for K=20 tokens, stratified (decision Q4) —
  "tam_maps_subset": {
    "token_indices":       [/* int × K, K≤20 after dedup */],
    "selection_strata":    [/* str × K: "top_abs_vd" | "top_abs_adv" | "blankness" | "answer_critical" */],
    "selection_rank":      [/* int × K: rank-within-stratum, 0-based */],
    "selection_score":     [/* float × K: |vd|, |adv|, or 1.0 for boolean strata */],
    "deduped_from_strata": [/* list[str] × K: other strata this token would have appeared in */],
    "maps_uint8_b64":      [/* str × K: base64 uint8 H×W heatmap, post rank-Gaussian */]
  }
}
```

## Field provenance

| Field | Source | Step 0 | Step 1a (teacher_greedy) | Step 1b (student_rollout) |
|---|---|---|---|---|
| `id, benchmark, split_tag, image_path, image_sha256, question, answer` | probe / calibration subset JSONL | ✅ | ✅ | ✅ |
| `response_source` | runner mode tag | `teacher_greedy` | `teacher_greedy` | `student_rollout_{ckpt}` |
| `teacher_ckpt, student_ckpt` | runner args | teacher only | ✅ | ✅ |
| `tokenizer_*, processor_*, tam_preproc_version, tam_score_def` | run metadata | ✅ | ✅ | ✅ |
| `response_text / ids / length / hash / tokens / token_idx / token_uid` | teacher greedy (Step 0, 1a) or student greedy (1b) | ✅ | ✅ | ✅ |
| `lp_full` | teacher full-image forward | ✅ | ✅ | ✅ |
| `lp_blank` | teacher blank-image forward | ❌ | ✅ | ✅ |
| `vd` | `lp_full - lp_blank` | ❌ | ✅ | ✅ |
| `tam_mass_top{10,20,40}, tam_entropy*, tam_effective_patch_frac` | TAM core on teacher full-image hidden_states | ✅ | ✅ | ✅ |
| `teacher_entropy_full, teacher_top1_margin_full` | teacher full forward extra stats | ✅ | ✅ | ✅ |
| `tokens_prompt, tam_*_prompt, prompt_segments` | ECI prompt recursion + chat-template parse | ✅ | ✅ | ✅ |
| `student_lp, student_entropy` | inline during student rollout (1b) OR student forward on teacher response (1a) | ❌ | ✅ | ✅ |
| `adv` | `lp_full - student_lp` | ❌ | ✅ | ✅ |
| `quad` | derived from `vd, adv` | ❌ | ✅ | ✅ |
| `is_*_token, answer_*_spans, think_*_spans, answer_span_source` | regex / template parse | ✅ | ✅ | ✅ |
| `image_grid_thw, vision_shape, n_patches, map_h, map_w, patch_index_order` | processor + TAM | ✅ | ✅ | ✅ |
| `tam_valid, tam_failure_reason` | runtime QC | ✅ | ✅ | ✅ |
| `tam_maps_subset` | TAM map → uint8 → b64 | ✅ (K=all-tokens for spot-check) | ✅ (K≤20 stratified) | ✅ (K≤20 stratified) |

## Forward-cost ledger

**Step 0** (sanity check, 3 probes, teacher only): ~1 minute.

**Step 1a** (`teacher_greedy`, N=150-200 samples × 4 ckpts):

| Cost | Count | What |
|---|---|---|
| teacher full forward + `output_hidden_states=True` | 1 × N | → `response_*, lp_full, teacher_entropy_full, tam_*` |
| teacher blank forward | 1 × N | → `lp_blank`, then `vd` |
| student full forward (on teacher's greedy response) | 1 × N × N_ckpts | → `student_lp, student_entropy, adv` |

Estimated H800 wall-time: ~15 min teacher + ~15 min student × 4 ≈ **~30-35 min**.

**Step 1b** (`student_rollout`, ~50 samples × 4 ckpts):

| Cost | Count | What |
|---|---|---|
| student greedy rollout (logprobs recorded inline at T=0) | 1 × n × N_ckpts | → `response_*, student_lp, student_entropy` |
| teacher full forward + hidden_states on student response | 1 × n × N_ckpts | → `lp_full, teacher_entropy_full, tam_*` |
| teacher blank forward | 1 × n × N_ckpts | → `lp_blank` → `vd` |

Estimated H800 wall-time: 50 × 4 × ~4s ≈ **~15 min**.

**Step 1 grand total ≈ 45-50 min on H800 (one box)**, including launch overhead.

## Aggregated outputs (analyzer reads this JSONL)

Step 1 analyzer (`src/mllmopd/analysis/tam_calibration.py`, not yet written)
emits `runs/analysis/tam_calibration_v0.json`. Key blocks below; the full
analyzer critique from GPT (per-quadrant correlations, PR-AUC for rare
quad==3, token-weighted + sample-weighted with bootstrap CI, by-`VD_NORM_BIN`
breakdowns, T1-3 trajectory) is integrated into the analyzer module — not
into the schema.

```
{
  "n_samples": int, "n_tokens": int,
  "qc": {
    "n_tam_failed": int,
    "n_shape_mismatch": int,
    "mean_tam_mass_neg_control": float,           // sanity: should be low
    "pope_probe_localization_notes": [str]
  },
  "per_ckpt_per_source": {
    "T1_0_base@teacher_greedy": {
      "n_tokens": int,
      "quad_token_counts": [4], "quad_sample_counts": [4],
      "visual_rejection_prevalence": float,
      "corr_tam_mass_abs_vd":          float,
      "corr_tam_mass_abs_vd_by_quad":  [4],
      "corr_tam_mass_abs_adv":         float,
      "corr_tam_mass_abs_adv_by_quad": [4],
      "auc_high_abs_vd":         float,
      "auc_visual_rejection":    float,
      "ap_visual_rejection":     float,             // PR-AUC, robust to rare positives
      "tam_mass_by_quad":        [4 dists],
      "tam_entropy_by_quad":     [4 dists],
      "tam_mass_by_vd_norm_bin": {...},             // reuses VD_NORM_BINS from t2_1_energy_audit.py
      "tam_mass_visual_rejection_sum": float,       // mirrors T2-1 cond_supp shape
      "tam_mass_by_token_type":  {answer, think, blankness, other},
      "sample_macro_avg":        {...},
      "bootstrap_ci_by_sample":  {...}
    },
    "T1_3_step230@student_rollout_T1_3_step230": { ... }
    // one entry per (student_ckpt, response_source) pair
  },
  "trajectory_T1_3": {                              // Step 1b — NOT optional, drives the cliff figure
    "steps": [49, 99, 149, 199, 230],
    "quad_counts_per_step":                  ...,
    "tam_mass_quad3_mean_per_step":          ...,
    "tam_mass_blankness_token_mean_per_step":...,
    "tam_entropy_blankness_token_mean_per_step": ...,
    "frac_blankness_tokens_high_tam_per_step":   ...
  }
}
```

## Sample selection (revised per GPT)

ChartQA up-weighted (biggest full-vs-blank dynamic range); `text_only` negative
control replaced with `blank_image` / `irrelevant_image` (which retain vision
tokens — TAM is undefined without).

| Split | n (1a) | n (1b) | Why |
|---|---|---|---|
| `opd_target` (random subset) | 70-90 | 20-30 | Vision-critical; teacher contrast largest |
| `ChartQA` collapse cases | **60-80** | 15-20 | +62pp full-vs-blank → biggest TAM signal |
| `HallusionBench` (full subset) | 25-30 | 5-10 | Vision-grounded; small but paper-relevant |
| `POPE_adversarial` | 20 | 5 | Localizable objects, Step 0 parity |
| `blank_image` / `irrelevant_image` neg ctrl | 10 | 0 | TAM_mass should be ~uniform / undefined → QC sanity |
| **1a total** | **185-230** | — | Step 1a primary |
| **1b total** | — | **45-65** | Step 1b student-rollout audit |

Persisted as `data/audit/tam_calibration_subset_v0.jsonl`.

## Step 2 forward-compat (pre-allocated, no design yet)

Step 2 (causal masking) storage and analyzer are still deferred. What v0.1.1
*does* pre-allocate for the future Step 2 sibling JSONL:

- Stable `(id, student_ckpt, response_source, response_hash, token_uid, token_idx)` join key
- `patch_index_order` makes mask-patch indices unambiguous across re-runs
- `tam_maps_subset.maps_uint8_b64` retains H×W maps for direct replay
- `tam_preproc_version` lets us detect if Step 1 and Step 2 ran the same pipeline
- `n_patches, map_h, map_w` explicit so analyzer doesn't have to recompute

Step 2 sibling JSONL (sketch, not committed):
```
{ id, student_ckpt, response_source, response_hash, token_uid, token_idx,
  mask_strategy: "top_tam_drop | random_drop | attention_drop | top_tam_keep",
  mask_ratio: 0.2, mask_seed: 3,
  lp_full_before, lp_after, delta_lp }
```

## Compatibility with existing audit infra

- `t2_1_energy_audit.py:80-102` defines `VD_NORM_BINS` and `QUADRANTS`; the
  Step 1 analyzer re-imports them and reuses `_quadrant_index(vd, adv)`.
- `BLANK_RE` from `t1_blankness_trajectory.py:39-52` populates `is_blankness_token`.
- `_extract_choices` from `run_audit_pass.py:126-131` for MCQ scoring; not
  needed in calibration but flagged if we later add accuracy metrics.

## What this does NOT do

- Does NOT modify training loss. Method-tier (TAM-Boost OPD) is downstream of
  calibration; no decision until corr / AUC numbers land.
- Does NOT touch the existing diag-hook JSONL writer in production training.
  Step 1 is a separate offline audit that re-runs everything fresh on a small
  calibration set.
- Does NOT include the LaTeX text-vis from upstream TAM (`xmed-lab/TAM`).
  PNG-only overlay; `tam_maps_subset.maps_uint8_b64` is sufficient for replay
  and figure-making.
- Does NOT lock in Step 2 (causal masking) design — only pre-allocates join
  keys so the future sibling JSONL can land without churning v0.1.x rows.
