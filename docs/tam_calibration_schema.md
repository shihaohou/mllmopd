# Step 1 TAM-VD Calibration: JSONL Schema (draft v0.1)

Status: **draft, not yet wired up**. This is the schema Step 0 (TAM sanity-check
on Qwen2.5-VL) and Step 1 (TAM-VD calibration audit) will jointly write — Step 0
emits a subset of these fields, Step 1 emits all of them.

**Decisions locked in v0.1** (from in-session discussion 2026-05-25):
- Q1 mass scalar: **store both** `tam_mass` (top-20% concentration) **and** `tam_entropy`
- Q2 prompt-token TAM: **store in both Step 0 and Step 1** (covers caption probes + cross-modal focus diagnostics)
- Q3 causal masking (Step 2): **deferred** — design will be re-opened once Step 1 results land
- Q4 `tam_maps_subset`: **store, stratified K=20** = top-5 by `|vd|` + top-5 by `|adv|` + top-5 blankness + top-5 answer-critical

Context: per `docs/handoff-2026-05-25-tier2a-gpt-in-t2-2-paused.md` we decided to
move TAM (Token Activation Map, ICCV 2025, arXiv 2506.23270) into the Tier-2a
mechanism line as an "evidence estimator" for OPD teacher correction. Step 1
asks the calibration question: **does TAM-mass per response token correlate with
|VD| and (critically) with `VD<0 ∧ adv<0` visual-rejection tokens?** If yes, TAM
is a one-forward proxy for our current two-forward VD; if no, abandon the
method-tier port and keep TAM only as appendix viz.

Existing diag JSONL we extend (from `opd_diagnostics_hook.py:17-29`):

```
{ id, step, response_length, image_mode, teacher_url, response_correct,
  lp_full[R], lp_blank[R], vd[R], old_lp_student[R] }
```

## Step 1 calibration row schema

One row per `(sample, student_ckpt)`. The TAM/VD/teacher-side fields don't
depend on `student_ckpt` so they're redundant across rows that share a sample —
single-file simplicity > storage cost (~few MB for 200-sample × 4-ckpt × 200-tok).

```json5
{
  // — identity —
  "id": "POPE_adversarial/1636",
  "benchmark": "POPE_adversarial",
  "split_tag": "opd_target | chartqa_collapse | hallusionbench | smoke",
  "image_path": "data/audit/images/POPE_adversarial/1636.png",
  "question": "Is there a tennis racket in the image?",
  "answer": "yes",

  // — checkpoints —
  "teacher_ckpt": "MMR1-7B-RL",
  "student_ckpt": "T1_0_base | T1_2_step230 | T1_3_step99 | T1_3_step230 | T2_1_step230 | ...",

  // — response (deterministic teacher greedy → identical across student_ckpt) —
  "response_text": "<think>...</think><answer>Yes</answer>",
  "response_ids":  [/* int × R */],
  "response_length": 187,    // R
  "tokens":        [/* str × R */],   // detokenized for slicing

  // — per-token teacher signals on RESPONSE tokens (independent of student_ckpt) —
  "lp_full":     [/* float × R */],
  "lp_blank":    [/* float × R */],
  "vd":          [/* float × R */],   // = lp_full - lp_blank
  "tam_mass":    [/* float × R */],   // top-20% activation mass / total mass ∈ [0.2, 1.0]
  "tam_entropy": [/* float × R */],   // entropy of softmax(activation) over patches; lower = more localized

  // — TAM on PROMPT tokens (kept in both Step 0 and Step 1, decision Q2) —
  // ECI recursion already computes these in the first round; storing them is ~free.
  // P = prompt response_length; tokens_prompt is the detokenized prompt span only.
  "prompt_length": 87,
  "tokens_prompt": [/* str × P */],
  "tam_mass_prompt":    [/* float × P */],
  "tam_entropy_prompt": [/* float × P */],

  // — per-token student signals (depends on student_ckpt) —
  "student_lp": [/* float × R */],
  "adv":        [/* float × R */],    // = lp_full - student_lp

  // — pre-computed quadrant labels (derivable but stored for convenience) —
  // 0 = vis_support_agree            (vd≥0 ∧ adv≥0)
  // 1 = vis_support_pushed_away      (vd≥0 ∧ adv<0)
  // 2 = vis_reject_teacher_pushed_toward  (vd<0 ∧ adv≥0)
  // 3 = vis_reject_correction        (vd<0 ∧ adv<0) ← KEY T2-1 bucket
  "quad": [/* int × R */],

  // — token-text labels —
  "is_blankness_token": [/* bool × R */],  // matches BLANK_RE in t1_blankness_trajectory.py:39
  "is_answer_token":    [/* bool × R */],  // within <answer>...</answer> span
  "is_think_token":     [/* bool × R */],  // within <think>...</think> span

  // — image metadata (Qwen2.5-VL specifics) —
  "image_grid_thw": [1, 24, 32],            // t,h,w post-patch grid (whatever processor reports)
  "vision_shape":   [12, 16],               // after 2x2 packing — TAM's vision_shape

  // — TAM maps for K=20 tokens, stratified selection (decision Q4) —
  "tam_maps_subset": {
    "token_indices": [/* int × 20: 5 by top-|vd|, 5 by top-|adv|, 5 blankness, 5 answer-critical */],
    "selection_strata": [/* str × 20: "top_abs_vd" | "top_abs_adv" | "blankness" | "answer_critical" */],
    "maps_uint8_b64": [/* str × 20 */]   // base64 uint8 H×W heatmap (rank-Gaussian-filtered)
  }
}
```

## Field provenance

| Field | Source | Step 0 (sanity) | Step 1 (calibration) |
|---|---|---|---|
| `id, benchmark, image_path, question, answer` | probe sample JSONL | ✅ | ✅ |
| `teacher_ckpt, student_ckpt` | runner args | partial (teacher only) | ✅ |
| `response_text, response_ids, response_length, tokens` | teacher full-image forward (greedy, T=0) | ✅ | ✅ |
| `lp_full` | teacher full-image forward, log_softmax at sampled-token positions | ✅ | ✅ |
| `lp_blank` | teacher blank-image forward (separate) | ❌ | ✅ |
| `vd` | `lp_full - lp_blank` | ❌ | ✅ |
| `tam_mass, tam_entropy` (response) | TAM core on teacher full-image hidden_states + ECI + rank-Gaussian | ✅ | ✅ |
| `tokens_prompt, tam_mass_prompt, tam_entropy_prompt` | ECI prompt-token recursion (already computed) | ✅ | ✅ |
| `student_lp` | student full-image forward | ❌ | ✅ |
| `adv` | `lp_full - student_lp` | ❌ | ✅ |
| `quad` | derived from `vd, adv` | ❌ | ✅ |
| `is_blankness_token, is_answer_token, is_think_token` | regex/span match on response_text | ✅ | ✅ |
| `image_grid_thw, vision_shape` | processor output / TAM | ✅ | ✅ |
| `tam_maps_subset` | TAM map → uint8 → b64 | ✅ (more maps, K=all-tokens for spot-check) | ✅ (K≤20 sampled per row) |

Step 0 = single-row-per-probe-sample, no student, no blank. Step 1 = expand by
re-running with student forward + blank-image forward + quadrant labels.

## Forward-cost ledger

Per calibration sample, three teacher-side forwards reduce to **two unique
forwards × N ckpts (student-side)** because teacher signals are checkpoint-invariant:

| Cost | Count | What |
|---|---|---|
| teacher full forward + `output_hidden_states=True` | 1 × N_samples | → `response_text, lp_full, tam_*` |
| teacher blank forward | 1 × N_samples | → `lp_blank` (then `vd = lp_full - lp_blank`) |
| student full forward | 1 × N_samples × N_ckpts | → `student_lp` (then `adv`) |

Estimated wall-time (H800, MMR1-7B-RL teacher + MMR1-3B-SFT student family, 200 samples × 4 ckpts):

- teacher forwards: ~400 forwards × ~2s @ avg 200 response tokens ≈ 15 min
- student forwards: ~800 forwards × ~1s ≈ 15 min
- TAM post-processing on numpy: negligible

Total ~30-45 min for the full N=200 × 4-ckpt sweep. Step 0 alone (3 probes,
teacher only, no student / no blank): ~1-2 min runtime + setup.

## Aggregated outputs (analyzer reads this JSONL)

Step 1 analyzer (`src/mllmopd/analysis/tam_calibration.py`, not yet written)
produces `runs/analysis/tam_calibration_v0.json`:

```
{
  "n_samples": int, "n_tokens": int,
  "per_ckpt": {
    "T1_0_base": {
      "corr_tam_mass_abs_vd": float,
      "corr_tam_mass_abs_adv": float,
      "auc_high_abs_vd": float,           // tam_mass classifying |vd| > τ
      "auc_visual_rejection": float,      // tam_mass classifying quad==3
      "tam_entropy_by_quad": [4 dists],
      "tam_mass_by_quad":    [4 dists],
      "tam_entropy_by_token_type": {answer, think, blankness, other},
    },
    "T1_2_step230": { ... },
    "T1_3_step99": { ... },   // pre-cliff
    "T1_3_step230": { ... },  // post-cliff
    "T2_1_step230": { ... },
  },
  "trajectory_T1_3": {        // optional: TAM signal evolution across T1-3 training
    "steps": [49, 99, 149, 199, 230],
    "tam_mass_visual_reject_quad_mean": [...],
    "tam_mass_blankness_token_mean":    [...],
  }
}
```

## Sample selection

Total target N ≈ 150-200, biased toward big TAM dynamic range:

| Split | n | Why |
|---|---|---|
| `opd_target` (random subset) | 80-100 | Vision-critical subset where teacher contrast is largest |
| `ChartQA` collapse cases | 30-40 | +62pp full-vs-blank contrast → biggest expected TAM signal |
| `HallusionBench` (full subset) | 20-30 | Vision-grounded benchmark, paper relevance |
| `POPE_adversarial` | 10-20 | Localizable objects, sanity-check parity with Step 0 |
| `text_only` baseline | 10 | Negative control — TAM-mass should be ~uniform |

Persisted as `data/audit/tam_calibration_subset_v0.jsonl` (similar to existing
`smoke_subset_v0.jsonl`).

## Open design questions resolved in v0.1

| Q | Decision | Rationale |
|---|---|---|
| Q1 mass scalar | **`tam_mass` (top-20%) + `tam_entropy`** | Complementary, both cheap. `tam_mass` interpretable in figures; `tam_entropy` principled info measure |
| Q2 prompt-token TAM | **Stored in both Step 0 and Step 1** | ECI computes them anyway; storing is free; needed for caption-probe sanity in Step 0, useful for cross-modal-focus diagnostics in Step 1 |
| Q3 causal masking (Step 2) | **DEFERRED** | Re-open after Step 1 lands; current schema does not need to anticipate Step 2 storage |
| Q4 `tam_maps_subset` | **Store, stratified K=20** | 5 by `top \|vd\|` + 5 by `top \|adv\|` + 5 blankness + 5 answer-critical → 4 MB total, covers diagnostic classes for figure-making |

## Compatibility with existing audit infra

- `t2_1_energy_audit.py:80-102` already defines `VD_NORM_BINS` and `QUADRANTS`
  exactly matching what we encode in `quad`. The Step 1 analyzer can re-import
  those constants and reuse `_quadrant_index(vd, adv)` directly.
- `BLANK_RE` from `t1_blankness_trajectory.py:39-52` defines
  `is_blankness_token`. Re-import not duplicate.
- `_extract_choices` from `run_audit_pass.py:126-131` for MCQ scoring; not
  needed in calibration but flag for follow-up if we add accuracy metrics.

## What this does NOT do

- Does NOT modify training loss. Method-tier (TAM-Boost OPD) is downstream of
  this calibration; no decision until corr / AUC numbers land.
- Does NOT touch the existing diag-hook JSONL writer in production training.
  Step 1 is a separate offline audit that re-runs everything fresh on a small
  calibration set.
- Does NOT include the LaTeX text-vis from upstream TAM. PNG-only overlay; the
  `tam_maps_subset.maps_uint8_b64` is sufficient for replay/figure-making.
