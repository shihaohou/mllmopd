# Step 1 TAM-VD Calibration: JSONL Schema (draft v0.1.2)

Status: **draft, not yet wired up at v0.1.2** — but Step 0 (v0.1.1) is
already running. v0.1.2 incorporates the GPT review on Step 0 results
(commit `31b67ac`) and resolves two provenance bugs flagged there.

## What's new in v0.1.2 vs v0.1.1

GPT reviewed Step 0 outputs and the v0.1.1 schema (see
`docs/step0-results-2026-05-25.md` for the brief + 6 overlay images
under `docs/figures/step0/`). Two provenance bugs flagged:

- **Commit-string mismatch** between Step 0 brief metadata (`79f8e89`)
  and the v0.1.1 schema GPT was pinned to (`31b67ac`). Fix: split
  metadata into `code_commit_run` (what generated JSONL) vs
  `code_commit_analyzed` (what analyzer / current readers should use).
- **Prompt-TAM scope mismatch**: v0.1.1 says `tam_*_prompt[P]` covers
  system / image_placeholder / question, but `tam_sanity.py` actually
  emits scalars only for the **question span** (TAM's own
  `prompt_id = [vision_end, ...]` only sees question tokens). Fix: add
  `prompt_tam_scope` field with value `"question_only"`; clarify the
  field semantics in the row schema below.

GPT review verdicts: greenlight Step 1 *coding*, but block Step 1
*full run* until v0.1.2 lands. Key substantive changes:

- **`token_category` enum** (12 categories) replaces a narrow
  `is_content_noun`. Per GPT: ChartQA/MathVista visual evidence often
  lives in numbers, units, OCR strings, spatial-relation words — not
  nouns. Restricting to nouns underestimates TAM signal coverage.
- **Attention baseline** with the same scalar family as TAM
  (`mass_top{10,20,40}` + entropy_norm + method metadata). Cheapest
  control for the saliency-pull failure we saw in `car_answer_Yes` —
  if attention rollout produces the same wrong heatmap, saliency-pull
  is an MLLM attribution issue, not a TAM-specific bug.
- **TAM peak metadata** (`tam_peak_patch_idx`, `tam_peak_xy`,
  `tam_center_of_mass_xy`) — quantifies center / saliency bias and
  enables fast filtering of obviously-misfired tokens.
- **POS provenance** (`pos_tag`, `word_idx`, `word_text`,
  `pos_tagger`, `pos_tagger_version`, `token_category_source`) — makes
  the categorization reproducible without re-running spaCy.
- **Derived booleans** (`is_template_token`, `is_special_token`,
  `is_pronoun`, `is_meta_cot_token`) — convenience for analyzer
  slicing; all derivable from `token_category` but stored explicitly.
- **Optional `attention_maps_uint8_b64`** inside `tam_maps_subset` for
  K≤20 selected tokens — direct apples-to-apples vs TAM maps.

## Decisions

| Q | Decision | Status |
|---|---|---|
| Q1 mass scalar | `tam_mass_top{10,20,40}` (3 scalars/token) + `tam_entropy` + `tam_entropy_norm = H / log N_patches` | locked v0.1.1 |
| Q2 prompt-token TAM | Store `tam_*_prompt[P]` in both Step 0 and Step 1; **scope = question-only** per `prompt_tam_scope` field | locked v0.1.2 (clarified) |
| Q3 causal masking (Step 2) | Storage deferred; join keys pre-allocated | partial |
| Q4 `tam_maps_subset` | K=20 stratified; **add optional `attention_maps_uint8_b64`** for direct attention-vs-TAM comparison | locked v0.1.2 |
| Q5 response source | Step 1a `teacher_greedy` + Step 1b `student_rollout` | locked v0.1.1 |
| **Q6 token typing (new)** | **`token_category` enum × R + POS provenance**, replaces narrow `is_content_noun` | locked v0.1.2 |
| **Q7 attention baseline (new)** | **`attention_baseline_mass_top{10,20,40}` + entropy_norm + method metadata** — same scalar family as TAM, mandatory in Step 1 | locked v0.1.2 |
| **Q8 peak metadata (new)** | **`tam_peak_patch_idx, tam_peak_xy, tam_center_of_mass_xy` × R** for saliency-bias quantification | locked v0.1.2 |

Context: per `docs/handoff-2026-05-25-tier2a-gpt-in-t2-2-paused.md` we
moved TAM (arXiv 2506.23270) into Tier-2a as a one-forward evidence
estimator for OPD teacher correction. The calibration question:
**does `tam_mass` correlate with `|vd|` overall, and (critically) with
`vd<0 ∧ adv<0` visual-rejection tokens?** Step 0 (v0.1.1) found that
this question is *token-type-conditional* — content nouns clean,
template/pronoun/answer tokens noisy. v0.1.2 makes the stratification
explicit in the row schema.

Existing diag JSONL we extend (from `opd_diagnostics_hook.py:17-29`):

```
{ id, step, response_length, image_mode, teacher_url, response_correct,
  lp_full[R], lp_blank[R], vd[R], old_lp_student[R] }
```

## Row schema

One row per `(sample, student_ckpt, response_source)`. For Step 1a the
teacher-side fields are checkpoint-invariant; for Step 1b the response
itself depends on student_ckpt.

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
  "tam_preproc_version":    "v0.1.2",
  "code_commit_run":        "abcd1234",         // commit that generated this row
  "code_commit_analyzed":   null,               // set by analyzer at read-time
  "pos_tagger":             "spacy/en_core_web_sm",
  "pos_tagger_version":     "3.7.2",
  "token_category_source":  "regex+spacy_align:v0.1.2",
  "attention_baseline_method": "last_layer_avg_heads:v0.1.2",
  "attention_baseline_layers": [-1],             // -1 = last layer; list[int] for selected
  "attention_baseline_heads":  "all",            // "all" | "selected:[h1,h2,...]"
  "tam_score_def": {
    "mass_top_K":   [10, 20, 40],
    "entropy":      "H = -Σ softmax(act_i)·log softmax(act_i) over patches (post rank-Gaussian); tam_entropy_norm = H / log(N_patches)",
    "map_pipeline": "lm_head(last_hidden)[:, img_idx, cls_id] → ECI subtract → clip ≥ 0 → rank_gaussian_filter(3) → normalize 0-1",
    "peak": "tam_peak_patch_idx = argmax(map.flatten()); tam_peak_xy = (col/W, row/H) in [0,1]; tam_center_of_mass_xy = Σ p_i·(x_i, y_i)/Σ p_i"
  },

  // — response (depends on response_source) —
  "response_text":   "<think>...</think><answer>Yes</answer>",
  "response_ids":    [/* int × R */],
  "response_length": 187,                       // R
  "response_hash":   "sha256(response_ids):hex[:16]",
  "tokens":          [/* str × R */],
  "token_idx":       [/* int × R */],            // 0..R-1
  "token_uid":       [/* str × R */],            // f"{id}:{student_ckpt}:{response_source}:{response_hash}:{token_idx}"

  // — POS / token category (v0.1.2, decision Q6) —
  "pos_tag":        [/* str × R */],            // spaCy POS, "" for tokens with no word alignment
  "word_idx":       [/* int × R */],            // index into spaCy-tokenized words, -1 if no align
  "word_text":      [/* str × R */],            // the spaCy word this subword belongs to, "" if none
  "token_category": [/* str × R */],            // one of:
                                                 //   "content_noun" | "proper_noun" | "visual_number" |
                                                 //   "ocr_text" | "visual_attribute" | "spatial_relation" |
                                                 //   "answer_token" | "pronoun" | "template_token" |
                                                 //   "special_token" | "meta_cot_token" | "punctuation" | "other"

  // — token-text labels (derived from token_category for analyzer convenience) —
  "is_blankness_token":  [/* bool × R */],      // matches BLANK_RE in t1_blankness_trajectory.py:39
  "is_answer_token":     [/* bool × R */],      // within <answer>...</answer> span
  "is_think_token":      [/* bool × R */],      // within <think>...</think> span
  "is_template_token":   [/* bool × R */],      // token_category == "template_token"
  "is_special_token":    [/* bool × R */],      // token_category == "special_token"
  "is_pronoun":          [/* bool × R */],      // token_category == "pronoun"
  "is_meta_cot_token":   [/* bool × R */],      // token_category == "meta_cot_token"
  "answer_span_source":  "regex_final_answer | mcq_extractor | manual | null",
  "answer_token_spans":  [[/* start, end exclusive */]],
  "think_token_spans":   [[/* start, end exclusive */]],

  // — per-token teacher signals on RESPONSE tokens —
  "lp_full":     [/* float × R */],
  "lp_blank":    [/* float × R */],
  "vd":          [/* float × R */],              // = lp_full - lp_blank
  "tam_mass_top10": [/* float × R */],
  "tam_mass_top20": [/* float × R */],           // primary scalar
  "tam_mass_top40": [/* float × R */],
  "tam_entropy":      [/* float × R */],         // raw entropy
  "tam_entropy_norm": [/* float × R */],         // = tam_entropy / log(N_patches) ∈ [0, 1]
  "tam_effective_patch_frac": [/* float × R */], // = exp(tam_entropy) / N_patches

  // — TAM peak metadata (v0.1.2, decision Q8) —
  "tam_peak_patch_idx":   [/* int × R */],      // argmax in flattened (H*W) space
  "tam_peak_xy":          [[/* x_norm, y_norm */]],   // (col/(W-1), row/(H-1)) ∈ [0,1]^2
  "tam_center_of_mass_xy":[[/* x_norm, y_norm */]],   // soft mean of patch coords weighted by activation

  // — attention baseline (v0.1.2, decision Q7) —
  // Computed from outputs.attentions[-1] (last layer), averaged over heads,
  // sliced to image-patch key positions, then normalized like TAM.
  "attention_baseline_mass_top10":   [/* float × R */],
  "attention_baseline_mass_top20":   [/* float × R */],
  "attention_baseline_mass_top40":   [/* float × R */],
  "attention_baseline_entropy_norm": [/* float × R */],

  // — confound disentanglers —
  "teacher_entropy_full":     [/* float × R */], // teacher per-token output entropy
  "teacher_top1_margin_full": [/* float × R */], // teacher top1 − top2 logprob

  // — TAM on PROMPT tokens (v0.1.2 — scope clarified) —
  "prompt_tam_scope": "question_only",           // v0.1.2: locks the scope vs schema/code mismatch
  "prompt_length":  87,                          // P = number of tokens in the question span only
  "tokens_prompt":  [/* str × P */],
  "tam_mass_top20_prompt":     [/* float × P */],
  "tam_entropy_norm_prompt":   [/* float × P */],
  "prompt_segments": {                           // half-open intervals in FULL input_ids dimension
    "system":            [0, 35],
    "image_placeholder": [36, 86],
    "question":          [87, 102]
  },
  "prompt_full_length": 102,

  // — per-token student signals —
  "student_lp":      [/* float × R */],
  "student_entropy": [/* float × R */],
  "adv":             [/* float × R */],          // = lp_full - student_lp

  // — quadrant labels —
  // 0 = vis_support_agree              (vd ≥ 0 ∧ adv ≥ 0)
  // 1 = vis_support_pushed_away        (vd ≥ 0 ∧ adv < 0)
  // 2 = vis_reject_teacher_pushed_toward (vd < 0 ∧ adv ≥ 0)
  // 3 = vis_reject_correction          (vd < 0 ∧ adv < 0)   ← KEY T2-1 bucket
  "quad": [/* int × R */],

  // — image metadata —
  "image_grid_thw": [1, 24, 32],
  "vision_shape":   [12, 16],
  "n_patches":      192,
  "map_h":          12,
  "map_w":          16,
  "patch_index_order": "row_major_top_left",

  // — QC —
  "tam_valid":          true,
  "tam_failure_reason": null,
  "attn_baseline_valid":         true,
  "attn_baseline_failure_reason": null,

  // — maps subset (K=20 stratified) —
  "tam_maps_subset": {
    "token_indices":       [/* int × K, K≤20 */],
    "selection_strata":    [/* str × K */],
    "selection_rank":      [/* int × K */],
    "selection_score":     [/* float × K */],
    "deduped_from_strata": [/* list[str] × K */],
    "maps_uint8_b64":      [/* str × K, TAM heatmap b64 */],
    "attention_maps_uint8_b64": [/* str × K, optional; same H×W as TAM map */]
  }
}
```

## Field provenance

| Field | Source | Step 0 | Step 1a | Step 1b |
|---|---|---|---|---|
| `id, benchmark, split_tag, image_path, image_sha256, question, answer` | calibration subset JSONL | ✅ | ✅ | ✅ |
| `response_source` | runner mode tag | `teacher_greedy` | `teacher_greedy` | `student_rollout_{ckpt}` |
| `teacher_ckpt, student_ckpt` | runner args | teacher only | ✅ | ✅ |
| `tokenizer_*, processor_*, code_commit_run, pos_tagger_*, attention_baseline_*, tam_preproc_version, tam_score_def, token_category_source` | run metadata | ✅ | ✅ | ✅ |
| `response_text / ids / length / hash / tokens / token_idx / token_uid` | teacher greedy (Step 0, 1a) or student greedy (1b) | ✅ | ✅ | ✅ |
| `pos_tag, word_idx, word_text` | spaCy POS + char-span alignment to subword tokens | ✅ | ✅ | ✅ |
| `token_category` | regex pre-pass (template/special/meta-CoT) → spaCy POS → custom category map | ✅ | ✅ | ✅ |
| `is_*_token` (all booleans) | derived from `token_category` + span regex | ✅ | ✅ | ✅ |
| `lp_full, teacher_entropy_full, teacher_top1_margin_full` | teacher full forward + `output_scores=True` | ✅ | ✅ | ✅ |
| `lp_blank` | teacher blank forward | ❌ | ✅ | ✅ |
| `vd` | `lp_full - lp_blank` | ❌ | ✅ | ✅ |
| `tam_mass_top{10,20,40}, tam_entropy*, tam_effective_patch_frac` | TAM core | ✅ | ✅ | ✅ |
| `tam_peak_patch_idx, tam_peak_xy, tam_center_of_mass_xy` | argmax / weighted-mean of TAM map | ✅ | ✅ | ✅ |
| `attention_baseline_*` | `outputs.attentions[-1]` avg heads → image-patch key slice → normalize → mass / entropy | ✅ | ✅ | ✅ |
| `tokens_prompt, tam_*_prompt, prompt_segments, prompt_tam_scope` | ECI prompt-token recursion + chat-template parse | ✅ | ✅ | ✅ |
| `student_lp, student_entropy` | student forward / student greedy rollout | ❌ | ✅ | ✅ |
| `adv` | `lp_full - student_lp` | ❌ | ✅ | ✅ |
| `quad` | derived from `vd, adv` | ❌ | ✅ | ✅ |
| `image_grid_thw, vision_shape, n_patches, map_h, map_w, patch_index_order` | processor + TAM | ✅ | ✅ | ✅ |
| `tam_valid, tam_failure_reason, attn_baseline_valid, attn_baseline_failure_reason` | runtime QC | ✅ | ✅ | ✅ |
| `tam_maps_subset.maps_uint8_b64` | TAM map → uint8 → b64 | ✅ | ✅ | ✅ |
| `tam_maps_subset.attention_maps_uint8_b64` | attention map → uint8 → b64 | ✅ (optional) | ✅ (optional) | ✅ (optional) |

## Forward-cost ledger

**Step 0** (sanity check, 4 probes, teacher only, v0.1.2): ~1-2 min on
H800 single GPU. Adds POS-tagging (negligible) + last-layer attention
extraction (~10% over v0.1.1).

**Step 1a** (`teacher_greedy`, N=150-200 × 4 ckpts, v0.1.2):

| Cost | Count | What |
|---|---|---|
| teacher full forward + `output_hidden_states=True` + `output_attentions=True` | 1 × N | → `response_*, lp_full, teacher_entropy_full, tam_*, attention_baseline_*` |
| teacher blank forward | 1 × N | → `lp_blank` → `vd` |
| student full forward (teacher-greedy response) | 1 × N × N_ckpts | → `student_lp, student_entropy, adv` |

Wall-time on H800: ~17 min teacher (slightly up from v0.1.1's 15 min
due to attention storage overhead) + ~15 min student × 4 ≈ **~35-40 min**.

**Step 1b** (`student_rollout`, ~50 samples × 4 ckpts, v0.1.2):

| Cost | Count | What |
|---|---|---|
| student greedy rollout (inline logprobs) | 1 × n × N_ckpts | → `response_*, student_lp, student_entropy` |
| teacher full forward + hidden_states + attentions | 1 × n × N_ckpts | → `lp_full, teacher_entropy_full, tam_*, attention_baseline_*` |
| teacher blank forward | 1 × n × N_ckpts | → `lp_blank` → `vd` |

Wall-time on H800: ~50 × 4 × ~4-5s ≈ **~17 min**.

**Step 1 grand total ≈ 50-60 min on H800**, including launch overhead
and POS tagging.

## Aggregated outputs (analyzer reads this JSONL)

Step 1 analyzer (`src/mllmopd/analysis/tam_calibration.py`, not yet
written) emits `runs/analysis/tam_calibration_v0.json`. v0.1.2-specific:
all correlations / AUCs / APs are computed **per `token_category`** as
well as pooled.

```
{
  "n_samples": int, "n_tokens": int,
  "n_tokens_by_category": {content_noun, visual_number, ocr_text, ...},
  "qc": {
    "n_tam_failed": int, "n_attn_failed": int, "n_shape_mismatch": int,
    "mean_tam_mass_neg_control": float,
    "mean_attn_mass_neg_control": float,
    "pope_probe_localization_notes": [str]
  },
  "per_ckpt_per_source": {
    "T1_0_base@teacher_greedy": {
      "n_tokens": int,
      "quad_token_counts": [4], "quad_sample_counts": [4],
      "visual_rejection_prevalence": float,
      // Overall pooled (kept for back-compat / sanity)
      "corr_tam_mass_abs_vd":  float,
      "corr_tam_mass_abs_adv": float,
      "auc_high_abs_vd":       float,
      "auc_visual_rejection":  float,
      "ap_visual_rejection":   float,
      // v0.1.2: per-token-category breakdowns (the headline table)
      "corr_tam_mass_abs_vd_by_category":  {content_noun: float, visual_number: float, ...},
      "corr_attn_mass_abs_vd_by_category": {content_noun: float, ...},
      "auc_high_abs_vd_by_category":  {content_noun: float, ...},
      "ap_visual_rejection_by_category": {content_noun: float, ...},
      // v0.1.2: TAM vs attention head-to-head
      "tam_beats_attn_on_content_nouns":   float,    // Δ(corr_tam - corr_attn)
      "tam_beats_attn_on_answer_tokens":   float,
      "tam_beats_attn_on_visual_rejection": float,
      // v0.1.2: peak / center-of-mass diagnostics
      "tam_peak_center_bias":    {...},               // mean(tam_peak_xy) per category
      "tam_peak_consistency":    {...},               // var(tam_peak_xy) within token across response
      // v0.1.2 by-VD-norm-bin (reuses VD_NORM_BINS from t2_1_energy_audit.py)
      "tam_mass_by_vd_norm_bin": {...},
      "tam_mass_visual_rejection_sum": float,
      // existing
      "tam_mass_by_quad":        [4 dists],
      "tam_entropy_by_quad":     [4 dists],
      "sample_macro_avg":        {...},
      "bootstrap_ci_by_sample":  {...}
    },
    "T1_3_step230@student_rollout_T1_3_step230": { ... }
  },
  "trajectory_T1_3": {                              // Step 1b — drives the cliff figure
    "steps": [49, 99, 149, 199, 230],
    "quad_counts_per_step":                  ...,
    "tam_mass_quad3_mean_per_step":          ...,
    "tam_mass_quad3_by_category_per_step":   ...,    // v0.1.2: separate content_noun vs other
    "tam_mass_blankness_token_mean_per_step":...,
    "tam_entropy_blankness_token_mean_per_step": ...,
    "frac_blankness_tokens_high_tam_per_step":   ...
  }
}
```

## Sample selection (v0.1.1, unchanged)

| Split | n (1a) | n (1b) | Why |
|---|---|---|---|
| `opd_target` (random subset) | 70-90 | 20-30 | Vision-critical; teacher contrast largest |
| `ChartQA` collapse cases | 60-80 | 15-20 | +62pp full-vs-blank — biggest TAM dynamic range |
| `HallusionBench` (full subset) | 25-30 | 5-10 | Vision-grounded benchmark, paper relevance |
| `POPE_adversarial` | 20 | 5 | Localizable objects, Step 0 parity |
| `blank_image` / `irrelevant_image` neg ctrl | 10 | 0 | TAM_mass should be ~uniform / undefined → QC sanity |
| **1a total** | **185-230** | — | Step 1a primary |
| **1b total** | — | **45-65** | Step 1b student-rollout |

Persisted as `data/audit/tam_calibration_subset_v0.jsonl`.

## token_category enum — definitions

12 categories per v0.1.2. Resolution order: regex pre-pass for
mechanical categories first, then spaCy POS for the linguistic ones,
fallback to `"other"`.

| Category | How identified | Why distinguish |
|---|---|---|
| `special_token` | Token id ∈ `{<|im_start|>, <|im_end|>, <\|vision_start\|>, <\|vision_end\|>, ...}` or maps to non-printable Qwen special id | Different attribution semantics; usually no visual referent |
| `template_token` | Regex: `<think>`, `</think>`, `<answer>`, `</answer>`, `\\boxed`, `:**`, `**:`, markdown list/bold patterns | MMR1 chat-template scaffolding; same noise pattern across responses |
| `meta_cot_token` | Regex: `"Crop"`, `"crop"`, `"the image shows"`, `"Looking at"`, `"I need to"`, ... (CoT self-reference) | Model talking about its own analysis; no scene referent |
| `answer_token` | Inside `<answer>...</answer>` span | Saliency-pull failure mode; signal mixed |
| `pronoun` | spaCy POS == `PRON` AND not template_token | Referent resolution doesn't reflect in lm_head spatial pattern |
| `proper_noun` | spaCy POS == `PROPN` | Named entities — separate from common-noun visual referents |
| `content_noun` | spaCy POS == `NOUN` (not proper_noun) | The "trash" case — clean visual evidence |
| `visual_number` | Regex `[0-9]+(\.[0-9]+)?` or spaCy POS == `NUM` | Numbers on charts/dials — visual evidence in MathVista/ChartQA |
| `ocr_text` | Heuristic: ALL-CAPS word ≥3 chars OR matches sign-like pattern; refine in v0.1.2 | OCR-grounded; ChartQA labels, signs |
| `visual_attribute` | spaCy POS == `ADJ` AND not in a stoplist | Colors, sizes, shapes — visually grounded modifiers |
| `spatial_relation` | Word in {`above, below, left, right, near, between, behind, ...`} | Spatial relations need to localize on multiple objects |
| `punctuation` | spaCy POS == `PUNCT` or token is purely punctuation | Excluded from correlations |
| `other` | None of the above | Catch-all |

A subword (BPE) token inherits its word's category. Token-to-word
alignment via `processor.tokenizer` offset mapping where available;
otherwise re-decode prefixes to find boundaries.

## Step 2 forward-compat (pre-allocated, no design yet)

Step 2 (causal masking) storage and analyzer remain deferred. v0.1.2
additionally pre-allocates:

- `attention_baseline_method` (so Step 2 mask experiments can compare
  TAM-masked-region vs attention-masked-region as separate factors)
- `tam_peak_patch_idx` (Step 2 may mask "TAM peak ± radius" which
  requires this index)
- `tam_center_of_mass_xy` (Step 2 may mask "TAM center of mass ± radius",
  a softer alternative to the peak)

Step 2 sibling JSONL sketch (not committed):
```
{ id, student_ckpt, response_source, response_hash, token_uid, token_idx,
  token_category,                            // v0.1.2: stratify by category
  mask_strategy: "top_tam_drop | top_attn_drop | random_drop |
                  preceding_content_noun_tam_drop |                   // GPT's "evidence inheritance" experiment
                  attention_drop | top_tam_keep",
  mask_ratio: 0.2, mask_seed: 3,
  lp_full_before, lp_after, delta_lp }
```

GPT's elegant Step 2 design (worth pre-flagging): for *answer* tokens,
don't just mask the answer token's own TAM map. Also mask the
**preceding content-noun token's TAM**. If answer logprob changes more
under that second mask than under random mask, then "visual evidence
is localized at content tokens, while answer tokens inherit evidence
through prefix reasoning" — a stronger paper claim than TAM-as-direct-
evidence on the answer token.

## Compatibility with existing audit infra

- `t2_1_energy_audit.py:80-102` defines `VD_NORM_BINS` and `QUADRANTS`;
  the Step 1 analyzer re-imports them and reuses `_quadrant_index(vd, adv)`.
- `BLANK_RE` from `t1_blankness_trajectory.py:39-52` populates
  `is_blankness_token`.
- `_extract_choices` from `run_audit_pass.py:126-131` for MCQ scoring;
  not needed in calibration.

## What this does NOT do

- Does NOT modify training loss. Method-tier (TAM-Boost OPD) is
  downstream of calibration; no decision until corr / AUC numbers land.
- Does NOT touch the existing diag-hook JSONL writer in production
  training. Step 1 is a separate offline audit on a small calibration set.
- Does NOT include the LaTeX text-vis from upstream TAM. PNG-only.
- Does NOT lock in Step 2 (causal masking) design — only pre-allocates
  join keys + peak metadata so the future sibling JSONL can land
  without churning v0.1.x rows.
- Does NOT compute attention for prompt tokens (only response). If we
  ever need prompt-token attention baseline we'd add it under
  `attention_baseline_*_prompt[P]`.
- Does NOT model-based POS-tag. Uses regex + spaCy `en_core_web_sm`
  for reproducibility. If spaCy unavailable, falls back to regex-only
  with reduced category resolution (content_noun ↔ visual_attribute ↔
  pronoun become "other"); the run is still valid but token_category
  coverage is lower.
