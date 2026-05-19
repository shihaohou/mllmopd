# Brief: smoke500 audit data for independent review

**Purpose**: give an external reviewer (e.g., GPT) everything needed to
independently analyze the smoke500 results in `runs/audit/smoke500/` without
prior context.

**Captured at**: 2026-05-19. **Updated**: 2026-05-19 (scorer fix + diagnostics).

> **Update notes (2026-05-19, post external review)**
>
> 1. **MathVista `mcq_letter` scorer was upgraded** to use priority patterns
>    (`\boxed{X}` → `final answer is X` → `correct answer is X` → `option X` →
>    parenthesized-in-tail → legacy last-letter fallback). 85/4500 records
>    rescored; per-cell impact is +0.4 ~ +3.2pt acc. **Records where the model
>    only states option *text* (e.g., "Serrulate") in conclusion remain
>    unrecoverable** without choices in the JSONL — flagged below.
> 2. **`summary.json` now also carries**: `hit_max_tokens_rate`, `refusal_rate`,
>    `mcq_high_conf_rate` (fraction of MCQ rows parsed via a high-confidence
>    answer phrase, not the fallback), and `rescore_changed` (#rows whose
>    is_correct flipped after the scorer upgrade).
> 3. **One prior interpretation was wrong**: the brief said RL had *higher*
>    MathVista `blank_shortcut` than SFT (0.088 vs 0.068). After scorer fix it's
>    0.072 vs 0.076 — **essentially tied / RL slightly lower**. The
>    "post-training raises BOTH image_lift AND blank_shortcut on MathVista"
>    framing was a scorer artifact. Cleaner statement: post-training raises
>    `image_lift` (real signal), shortcut rate is noisy.
> 4. **Base (Qwen2.5-VL-7B-Instruct) is contaminated as a control**: 16-22% of
>    Base `full_image` MathVista predictions are degenerate (<30 tokens,
>    "are an language model", "addCriterion" artifacts). `mcq_high_conf_rate`
>    is only 9% for Base_full vs 42-52% for MMR1. **All Base-vs-MMR1
>    comparisons should be treated as suggestive only** until the Base
>    inference setup (prompt template / decoding) is fixed.

---

## TL;DR

We ran **9 inference passes** (3 models × 3 input modes) on a **500-prompt subset**
(250 MathVista-mini + 250 POPE-adversarial). Raw predictions: `runs/audit/smoke500/*.jsonl`.
Aggregated metrics: `runs/audit/smoke500/summary.json`.

This is an **audit of the starting state** — we have **not run OPD training yet**.

We want an independent take on what these numbers say about three hypotheses
on how vanilla On-Policy Distillation (OPD) might fail in MLLM post-training.

---

## Research question

When and how does **vanilla On-Policy Distillation (OPD)** fail in MLLM
post-training? We use an **RL-teacher → SFT-student** setup where both share
the same base + SFT data lineage:

- Teacher: **MMR1-7B-RL**  (Qwen2.5-VL-7B + math SFT + RL)
- Student: **MMR1-3B-SFT** (Qwen2.5-VL-3B + math SFT, no RL)
- Control: **Qwen2.5-VL-7B-Instruct** (no math SFT, no RL) — to separate
  "what post-training added" from "what OPD/RL specifically added"

This setup is deliberately chosen so OPD's "extra signal" (what RL contributes)
is separable from base-capability gap.

## Three hypotheses (none chosen yet)

- **H1 Perception-Reasoning Frontier**: OPD treats "saw image wrong" and
  "reasoned wrong" as the same loss → student over-rewarded on reasoning-easy
  perception-hard items.
- **H2 Visual-Signal Misallocation**: OPD's loss mass falls on tokens with low
  visual dependency → model learns text priors instead of visual reasoning.
- **H3 Post-RL Artifact Transfer**: RL teacher's quirks (overthinking, length
  inflation, modality shortcut) get inherited along with capabilities.

---

## Experiment design

3 models × 3 input modes × 2 benchmarks = **18 cells**, 250 prompts each.

| Model              | Role            | Notes                                  |
|--------------------|-----------------|----------------------------------------|
| MMR1-7B-RL         | OPD teacher     | post-trained on math, RL'd             |
| MMR1-3B-SFT        | OPD student     | post-trained on math, no RL            |
| Qwen2.5-VL-7B-Inst | Pre-post-train  | base for "what did post-training add"  |

| Mode          | What it does                                                |
|---------------|-------------------------------------------------------------|
| `full_image`  | Normal: image + prompt → answer                             |
| `blank_image` | Replace image with all-black canvas (same size, same prompt)|
| `text_only`   | Drop the image token; prompt only                           |

**Paired prompts**: `full_image` and `blank_image` use the SAME 250 ids per
benchmark, so we compute a paired 2x2 contingency:
- `image_lift_rate = full_only / n_paired` (questions blank-wrong → full-right)
- `blank_shortcut_rate = blank_only / n_paired` (blank-right → full-wrong; "lucky shortcut")

Decoding params (all passes): greedy, `max_new_tokens=1024`, seed=20260514.
A800 80GB, bf16, FA2.

---

## File layout in `runs/audit/smoke500/`

| Filename               | What                                                |
|------------------------|-----------------------------------------------------|
| `S_full.jsonl`         | MMR1-3B-SFT, full_image mode, 500 records           |
| `S_blank.jsonl`        | MMR1-3B-SFT, blank_image mode                        |
| `S_text_only.jsonl`    | MMR1-3B-SFT, text_only mode                          |
| `T_RL_full.jsonl`      | MMR1-7B-RL, full_image                               |
| `T_RL_blank.jsonl`     | MMR1-7B-RL, blank_image                              |
| `T_RL_text_only.jsonl` | MMR1-7B-RL, text_only                                |
| `Base_full.jsonl`      | Qwen2.5-VL-7B-Instruct, full_image                   |
| `Base_blank.jsonl`     | Qwen2.5-VL-7B-Instruct, blank_image                  |
| `Base_text_only.jsonl` | Qwen2.5-VL-7B-Instruct, text_only                    |
| `*.log`                | Stdout from each pass (progress, warnings)          |
| `summary.json`         | Aggregated `cells[]` + `paired_full_blank[]`        |

Each `.jsonl` has **500 lines** = 250 MathVista + 250 POPE_adversarial.

---

## JSONL record schema

```json
{
  "id": "MathVista/266",
  "benchmark": "MathVista",
  "mode": "full_image",
  "model": "<absolute path used as model identifier>",
  "prediction": "<CoT + final answer text>",
  "num_tokens": 262,
  "prompt_len": 349,
  "gold": "C",
  "is_correct": false,
  "scorer": "mcq_letter"
}
```

**Note**: the on-disk `is_correct` is what the old scorer produced. The
aggregator re-scores against the current scorer in memory, so cell-level
accuracies in `summary.json` may differ — `rescore_changed` tells you how many
rows flipped per cell. New inference runs additionally emit `parse_path` (the
priority rule that matched — `final_answer` / `correct_answer` / `boxed` /
`option_phrase` / `answer_phrase` / `paren_tail` / `last_letter_fallback` /
`none`); the 2026-05-19 smoke500 JSONLs predate this field and only get
`parse_path` filled at aggregation time.

**Scorer values**:
- `mcq_letter`: multiple-choice, gold is a letter (A/B/C/D...); parser extracts a letter from prediction.
- `numeric`: gold is a number; parser extracts last number from prediction.
- `yesno`: POPE-style; parser extracts yes/no.
- `loose_contains`: fallback fuzzy match (<1% of records; may be slightly noisy).
- `skip_missing_image`: image file was broken on disk; marker row, not scorable.
- `skip_empty_gold`: gold missing in source dataset.

**Note on `gold` for MathVista MCQ**: source dataset stores option *text*
("Serrulate"), but we mapped it to the *letter* ("C") at prep time so the
scorer has a clean target. The original text is preserved in source data but
not in these jsonl rows.

---

## `summary.json` schema

```json
{
  "cells": [
    {
      "model": "...",
      "mode": "full_image",
      "benchmark": "MathVista",
      "n": 250,
      "accuracy": 0.663,
      "n_scored": 249,
      "tokens_mean": 479.348,
      "tokens_median": 391.0,
      "acc_per_token": 0.00138,
      "scorers": {"mcq_letter": 131, "numeric": 117, "loose_contains": 1, "skip_missing_image": 1},
      "parse_paths": {"final_answer": 32, "correct_answer": 18, "paren_tail": 41, "last_letter_fallback": 38, "none": 2, "numeric": 117, "loose_contains": 1, "skip_missing_image": 1},
      "rescore_changed": 12,
      "hit_max_tokens_rate": 0.152,
      "refusal_rate": 0.000,
      "mcq_high_conf_rate": 0.42
    }
  ],
  "paired_full_blank": [
    {
      "model": "...",
      "benchmark": "MathVista",
      "n_paired": 249,
      "both_correct": 59,
      "full_only": 106,
      "blank_only": 18,
      "both_wrong": 66,
      "image_lift_rate": 0.426,
      "blank_shortcut_rate": 0.072
    }
  ]
}
```

`mcq_high_conf_rate` = fraction of `mcq_letter` rows in this cell whose
parse path was in {`boxed`, `final_answer`, `correct_answer`, `correct_choice`,
`answer_phrase`, `option_phrase`} (i.e., the model explicitly stated a letter,
not just text). Low values indicate the cell's accuracy depends heavily on the
weaker fallback rules.

---

## Headline numbers

All numbers below are **post-scorer-fix** (see Update note 1). Old numbers in
the on-disk `is_correct` field are preserved per record; the aggregator
re-scores in memory.

**Per-cell accuracy:**

| Model | Mode | MathVista | POPE_adv |
|---|---|---|---|
| Base (Qwen2.5-VL-7B) | full | 0.430 | 0.876 |
| MMR1-3B-SFT          | full | 0.574 | 0.896 |
| MMR1-7B-RL           | full | **0.663** | **0.900** |
| Base                 | blank | 0.237 | 0.516 |
| MMR1-3B-SFT          | blank | 0.261 | 0.516 |
| MMR1-7B-RL           | blank | 0.309 | 0.516 |
| Base                 | text_only | 0.196 | 0.000 |
| MMR1-3B-SFT          | text_only | 0.268 | 0.172 |
| MMR1-7B-RL           | text_only | 0.288 | 0.272 |

**Per-cell mean output tokens:**

| Model | full MV | full POPE | blank MV | text_only MV |
|---|---|---|---|---|
| Base | 134 | 33 | 132 | 141 |
| SFT  | 379 | 41 | 377 | 384 |
| RL   | **479** | **61** | 453 | **473** |

**Paired image_lift / blank_shortcut (rescored):**

| Model | Bench | img_lift | blank_shortcut |
|---|---|---|---|
| Base | MathVista | 0.237 | 0.044 |
| SFT  | MathVista | 0.390 | 0.076 |
| RL   | MathVista | **0.426** | 0.072 |
| Base | POPE | 0.360 | 0.000 |
| SFT  | POPE | 0.428 | 0.048 |
| RL   | POPE | 0.408 | 0.024 |

**Diagnostic rates (MathVista MCQ; POPE is a separate scorer):**

| Model | Mode | hit_max_tokens | refusal | mcq_high_conf |
|---|---|---|---|---|
| Base | full | 0.4% | 0.0% | **0.09** |
| SFT  | full | 10.8% | 0.0% | 0.49 |
| RL   | full | **15.2%** | 0.0% | 0.42 |
| Base | blank | 0.0% | 0.0% | 0.14 |
| SFT  | blank | **20.0%** | **15.2%** | 0.49 |
| RL   | blank | 17.2% | 4.0% | 0.52 |
| Base | text_only | 0.8% | 3.2% | 0.21 |
| SFT  | text_only | 18.8% | 9.6% | 0.37 |
| RL   | text_only | 18.4% | 8.0% | 0.45 |

**POPE refusal_rate (text_only, separately interesting):**
Base 65.6%, SFT 95.2%, RL **37.2%**. RL teacher is markedly less willing to
refuse, even without an image.

---

## Caveats / things that LOOK like bugs but aren't

1. **POPE `text_only` acc is misleading** (Base 0.0, MMR1 0.17-0.27): model
   often refuses to answer when asked yes/no without image (`refusal_rate` =
   65.6% / 95.2% / 37.2% for Base / SFT / RL). The 0.27 RL number means
   "RL refuses less and blindly says yes/no", **not** "RL has more text-only
   capability". Filter via `refusal_rate` when interpreting.
2. **POPE `blank_image` acc of 0.516 across all models** is **NOT** a
   grounding signal. POPE_adversarial's positive/negative label distribution
   is roughly 50/50; models presented with a blank canvas usually answer
   "No, the image is completely white" → ~half match the gold "No". This is
   a label-prior coincidence, not learned shortcut.
3. **MathVista `gold` is the letter** (A/B/C/...) even though the source
   dataset stores option text; we mapped at prep time. Records where the
   model only states the option *text* in the conclusion (e.g., "Serrulate
   is the correct answer") **cannot be scored** by the current scorer
   without `choices` joined back — the parser falls to `paren_tail` or
   `last_letter_fallback`. Known gap; fix requires either storing `choices`
   in the JSONL (future inference runs) or backfill from the source subset.
4. **`loose_contains` records are fuzzy fallback** (~1% of records). All
   cells in this run have ≤1 loose record, not a concern here.
5. **Standard error for n=250 accuracy ≈ 0.03**; paired counts can have
   larger variance. Treat single-cell deltas <0.04 as noise.
6. **MMR1 MathVista hits `max_new_tokens=1024` at 11-20% rate** (see
   `hit_max_tokens_rate`); Base almost never does. Long-CoT models commonly
   run out of budget mid-reasoning — `num_tokens=1024` doesn't necessarily
   mean useful CoT. This is also a direct H3 signal.
7. **Base (Qwen2.5-VL-7B-Instruct) is degenerate in 16-22% of MathVista
   records** (`<30 tokens`, "are an language model" mid-sentence, etc.).
   `mcq_high_conf_rate` = 0.09 for Base_full vs 0.42-0.52 for MMR1 makes
   this visible — Base often doesn't even produce an answer phrase. Most
   likely cause is prompt template / chat format mismatch between
   Qwen2.5-VL-Instruct's expected input and what `run_audit_pass.py` is
   sending (which works for the post-trained MMR1 because MMR1 was
   fine-tuned on this prompt form). **All Base-vs-MMR1 deltas should be
   read as suggestive only** until Base inference is fixed and re-run.

---

## What we'd like the reviewer to do

1. **Verify or challenge** these claims (post-scorer-fix):
   - **H3 (post-RL artifact transfer) signal is strong**:
     - RL teacher writes **27% more tokens** than SFT on MathVista for +9pt acc.
     - RL writes **85% more tokens** on POPE for ~0pt acc (diminishing returns).
     - RL writes **473 tokens** in `text_only` MathVista (no image at all).
     - MMR1 (SFT+RL) hits `max_new_tokens=1024` at **11-20%** on MathVista,
       Base at <1%. **Direct evidence over-thinking is trained, not capability.**
   - **NEW: RL teacher refuses much less than SFT** under degraded conditions:
     SFT `blank_image` MathVista refusal 15.2% vs RL 4.0%; SFT `text_only`
     POPE refusal 95.2% vs RL 37.2%. RL "presses on regardless" — possibly
     another facet of H3 (RL training shifts behavior toward always-respond).
   - **`image_lift` is consistently RL > SFT > Base** on MathVista
     (0.426 > 0.390 > 0.237). On POPE the post-training lift is real but
     RL ≈ SFT. Post-training reliably **adds visual lift**.
   - **`blank_shortcut` is essentially flat** Base/SFT/RL on MathVista
     after scorer fix (0.044 / 0.076 / 0.072) and decreasing on POPE
     (0.000 / 0.048 / 0.024). RL teacher is **not** more shortcut-prone
     than SFT — earlier framing was a scorer artifact.
2. **Look for signals we missed**, especially:
   - Anything pointing at **H1** (perception-reasoning frontier) — we don't
     have caption-only quadrants yet; can paired records be sliced any other way?
   - Anything pointing at **H2** (visual-signal misallocation) — we don't
     have forced decoding / per-token logprob yet; can we still infer
     anything from the prediction text? E.g., where does the long CoT
     "land" — does it reference visual details that ARE in the image?
   - Patterns in the raw `prediction` text — repetition loops (RL `T_RL_full`
     MathVista/224 generates repeated JSON arrays to 1024 tokens),
     language switches, malformed answers, "(A)/(B)/(C)" enumeration with
     option-text-only conclusions (the unrecoverable scorer case).
3. **Reality-check the corrections** in the Update notes at the top,
   especially: (a) is the Base degeneration explanation (prompt template
   mismatch) the most likely cause? (b) is there a way to recover
   option-text-only conclusions without re-running inference?
4. **Suggest the next audit / training step** given this evidence. Our current
   plan (in priority order):
   1. **Fix Base inference** (prompt template) → re-run 3 Base passes.
   2. **Store `choices` in JSONL** so future option-text-only conclusions
      can be scored. Optionally backfill into existing JSONLs from the
      source subset on the server.
   3. **Download MMR1-7B-SFT** (same-size pre-RL control; in progress).
   4. **Implement `--score_completion`** (forced decoding for H2).
   5. **Run vanilla OPD T1 baseline** once #1-#3 land.

---

## References

- `docs/handoff-2026-05-18.md` — full session handoff with hypotheses,
  experiment protocol, and pitfalls we already hit.
- `docs/research-plan.md` — validate/kill criteria per hypothesis.
- `docs/experiment-protocol.md` — Level 1/2/3 metric tables.
- `src/mllmopd/analysis/aggregate_audit.py` — how `summary.json` is computed.
- `src/mllmopd/diagnostics/scorers.py` — scoring logic (mcq_letter / numeric / yesno / loose).
- `src/mllmopd/diagnostics/run_audit_pass.py` — inference loop.
- `configs/audit/audit_v0_smoke.yaml` — 500-prompt mix config.
