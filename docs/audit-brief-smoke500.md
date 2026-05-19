# Brief: smoke500 audit data for independent review

**Purpose**: give an external reviewer (e.g., GPT) everything needed to
independently analyze the smoke500 results in `runs/audit/smoke500/` without
prior context.

**Captured at**: 2026-05-19. **Updated**: 2026-05-19 (scorer fix + choices backfill + MMR1-7B-SFT control added).

> **Update notes (2026-05-19, post external review)**
>
> 1. **MathVista `mcq_letter` scorer was upgraded** to use priority patterns
>    (`\boxed{X}` → `final answer is X` → `correct answer is X` → `option X` →
>    `choice_text` → parenthesized-in-tail → legacy last-letter fallback).
> 2. **`choices` were backfilled** into the smoke500 JSONLs from the source
>    subset (`scripts/data/backfill_choices.py`). Combined effect: **233/4500
>    records rescored** vs. on-disk `is_correct` (148 of those recovered via
>    the new `choice_text` path — predictions that conclude with option text
>    only like "Serrulate is the correct answer"). Per-cell MathVista acc
>    shifts +0.4 ~ +3.6pt. `mcq_high_conf_rate` now sits at **0.56-0.90**
>    across all cells (previously 0.09-0.52); scorer no longer leans on
>    weak fallbacks.
> 3. **`summary.json` now also carries**: `hit_max_tokens_rate`,
>    `refusal_rate`, `mcq_high_conf_rate`, `parse_paths`, `rescore_changed`.
> 4. **The "post-training raises both image_lift AND blank_shortcut" framing
>    is now fully refuted.** Post-backfill MathVista `blank_shortcut` is
>    essentially flat across models (Base 0.048 / SFT 0.064 / RL 0.052). It
>    was entirely a scorer artifact. The `image_lift` hierarchy survives
>    (Base 0.233 < SFT 0.365 < RL 0.422), so the "post-training adds visual
>    grounding" claim is real.
> 5. **Base (Qwen2.5-VL-7B-Instruct) is contaminated as a control**: 16-22%
>    of Base `full_image` MathVista predictions are degenerate (<30 tokens,
>    "are an language model", "addCriterion" artifacts). Even after backfill
>    Base `mcq_high_conf_rate` (0.56-0.68) lags MMR1 (0.83-0.89). Most likely
>    chat template / prompt format mismatch. **All Base-vs-MMR1 deltas
>    should be read as suggestive only** until Base inference is fixed and
>    re-run.
> 6. **MMR1-7B-SFT was added as a same-size pre-RL control** (now 4 models ×
>    3 modes × 2 benchmarks = 24 cells). Inference ran via sglang continuous
>    batching (~5× faster than the HF transformers backend used for the
>    other 9 passes). The 7B SFT control reverses two earlier interpretations
>    that had been built on the 3B-SFT-vs-7B-RL gap:
>    - **MMR1's RL step is a regression on MathVista at 7B**: 7B SFT
>      0.715 → 7B RL 0.683 (**−3.2pt** at same size). RL also drops
>      `img_lift` (0.454 → 0.422), **doubles** `blank_shortcut`
>      (0.024 → 0.052), and shortens CoT (535 → 479 tokens, hit_max
>      25% → 15%).
>    - **Over-thinking is a size + SFT effect, not RL**: tokens_mean on
>      MathVista full goes Base 134 → 3B SFT 379 → 7B SFT **535** →
>      7B RL 479. The longest is 7B SFT (pre-RL); RL actually **compresses**
>      CoT length.
>    - **"RL refuses less" was also size, not RL**: MathVista blank refusal
>      goes 3B SFT 15% → 7B SFT 4% → 7B RL 4%. POPE text_only: 3B SFT 95% →
>      7B SFT 24% → 7B RL 37%. The 3B vs 7B drop is huge; RL adds nothing
>      (slight increase on POPE).
>
>    So the H3 narrative needs reframing — see the dedicated section below.

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

4 models × 3 input modes × 2 benchmarks = **24 cells**, 250 prompts each.

| Model              | Role                       | Notes                                  |
|--------------------|----------------------------|----------------------------------------|
| MMR1-7B-RL         | OPD teacher                | post-trained on math, RL'd             |
| MMR1-7B-SFT        | **Same-size pre-RL control** | identical SFT data as 7B-RL, no RL — added 2026-05-19 to isolate size effect from RL effect |
| MMR1-3B-SFT        | OPD student                | post-trained on math, no RL            |
| Qwen2.5-VL-7B-Inst | Pre-post-train (Base)      | reference for "what did post-training add" |

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
| `T_SFT_full.jsonl`     | MMR1-7B-SFT, full_image (same-size pre-RL control)  |
| `T_SFT_blank.jsonl`    | MMR1-7B-SFT, blank_image                             |
| `T_SFT_text_only.jsonl`| MMR1-7B-SFT, text_only                               |
| `T_RL_full.jsonl`      | MMR1-7B-RL, full_image                               |
| `T_RL_blank.jsonl`     | MMR1-7B-RL, blank_image                              |
| `T_RL_text_only.jsonl` | MMR1-7B-RL, text_only                                |
| `Base_full.jsonl`      | Qwen2.5-VL-7B-Instruct, full_image                   |
| `Base_blank.jsonl`     | Qwen2.5-VL-7B-Instruct, blank_image                  |
| `Base_text_only.jsonl` | Qwen2.5-VL-7B-Instruct, text_only                    |
| `*.log`                | Stdout from each pass (progress, warnings)          |
| `summary.json`         | Aggregated `cells[]` + `paired_full_blank[]`        |

The 3 T_SFT JSONLs were inferred via sglang (continuous batching, ~5× faster);
all others via HF transformers. Same model, same prompt, greedy decoding, but
attention kernel differs — empirically the outputs match HF to within ~1-2
tokens on long CoTs, and scoring is identical for our use case.

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
`option_phrase` / `answer_phrase` / `choice_text` / `paren_tail` /
`last_letter_fallback` / `none`) and `choices` (the MCQ option-text list).
The 2026-05-19 smoke500 JSONLs predate these fields; `choices` can be
injected after the fact by running `scripts/data/backfill_choices.py`
against the source subset, which unlocks the `choice_text` parse path and
recovers predictions like "Serrulate is the correct answer" that the
letter-only parser cannot score.

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
      "accuracy": 0.683,
      "n_scored": 249,
      "tokens_mean": 479.348,
      "tokens_median": 391.0,
      "acc_per_token": 0.00143,
      "scorers": {"mcq_letter": 131, "numeric": 117, "loose_contains": 1, "skip_missing_image": 1},
      "parse_paths": {"final_answer": 36, "correct_answer": 22, "choice_text": 56, "paren_tail": 9, "last_letter_fallback": 7, "none": 1, "numeric": 117, "loose_contains": 1, "skip_missing_image": 1},
      "rescore_changed": 33,
      "hit_max_tokens_rate": 0.152,
      "refusal_rate": 0.000,
      "mcq_high_conf_rate": 0.89
    }
  ],
  "paired_full_blank": [
    {
      "model": "...",
      "benchmark": "MathVista",
      "n_paired": 249,
      "both_correct": 65,
      "full_only": 105,
      "blank_only": 13,
      "both_wrong": 66,
      "image_lift_rate": 0.422,
      "blank_shortcut_rate": 0.052
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
| Base (Qwen2.5-VL-7B) | full | 0.442 | 0.876 |
| MMR1-3B-SFT          | full | 0.598 | 0.896 |
| MMR1-7B-SFT          | full | **0.715** | 0.896 |
| MMR1-7B-RL           | full | 0.683 | **0.900** |
| Base                 | blank | 0.257 | 0.516 |
| MMR1-3B-SFT          | blank | 0.297 | 0.516 |
| MMR1-7B-SFT          | blank | 0.285 | 0.516 |
| MMR1-7B-RL           | blank | 0.313 | 0.516 |
| Base                 | text_only | 0.204 | 0.000 |
| MMR1-3B-SFT          | text_only | 0.284 | 0.172 |
| MMR1-7B-SFT          | text_only | 0.300 | **0.376** |
| MMR1-7B-RL           | text_only | 0.308 | 0.272 |

**Per-cell mean output tokens:**

| Model  | full MV | full POPE | blank MV | text_only MV |
|---|---|---|---|---|
| Base   | 134 | 33 | 132 | 141 |
| 3B SFT | 379 | 41 | 377 | 384 |
| 7B SFT | **535** | 44 | **499** | 459 |
| 7B RL  | 479 | **61** | 453 | **473** |

**Paired image_lift / blank_shortcut (post-backfill):**

| Model  | Bench     | img_lift  | blank_shortcut |
|---|---|---|---|
| Base   | MathVista | 0.233     | 0.048 |
| 3B SFT | MathVista | 0.365     | 0.064 |
| 7B SFT | MathVista | **0.454** | **0.024** |
| 7B RL  | MathVista | 0.422     | 0.052 |
| Base   | POPE      | 0.360     | 0.000 |
| 3B SFT | POPE      | 0.428     | 0.048 |
| 7B SFT | POPE      | 0.404     | 0.024 |
| 7B RL  | POPE      | 0.408     | 0.024 |

**Diagnostic rates (MathVista MCQ; POPE is a separate scorer):**

| Model  | Mode      | hit_max_tokens | refusal | mcq_high_conf |
|---|---|---|---|---|
| Base   | full      | 0.4%          | 0.0%     | 0.56 |
| 3B SFT | full      | 10.8%         | 0.0%     | 0.89 |
| 7B SFT | full      | **25.2%**     | 0.0%     | 0.85 |
| 7B RL  | full      | 15.2%         | 0.0%     | 0.89 |
| Base   | blank     | 0.0%          | 0.0%     | 0.64 |
| 3B SFT | blank     | 20.0%         | **15.2%**| 0.88 |
| 7B SFT | blank     | **28.8%**     | 4.4%     | 0.85 |
| 7B RL  | blank     | 17.2%         | 4.0%     | 0.83 |
| Base   | text_only | 0.8%          | 3.2%     | 0.68 |
| 3B SFT | text_only | 18.8%         | 9.6%     | 0.79 |
| 7B SFT | text_only | 27.6%         | 6.4%     | 0.87 |
| 7B RL  | text_only | 18.4%         | 8.0%     | 0.90 |

**POPE text_only refusal_rate**: Base 65.6%, 3B SFT 95.2%, 7B SFT **24.4%**,
7B RL 37.2%. The huge drop is **3B → 7B (size)**, not RL; RL slightly
*increases* refusal on top of 7B SFT.

---

## The RL regression finding (added 2026-05-19)

Adding MMR1-7B-SFT (same-size pre-RL control) flipped two earlier conclusions.
Decompose the gap from 3B SFT to 7B RL into a **size effect** (3B SFT → 7B SFT)
and an **RL effect** (7B SFT → 7B RL):

| Quantity (MathVista) | 3B SFT → 7B SFT (size) | 7B SFT → 7B RL (RL step) |
|---|---|---|
| full-image acc                 | **+11.7pt** (0.598 → 0.715) | **−3.2pt**  (0.715 → 0.683) |
| tokens_mean                    | +156 (+41%)  (379 → 535) | **−56 (−10%)** (535 → 479) |
| hit_max_tokens_rate            | +14.4pt (10.8% → 25.2%)  | **−10pt** (25.2% → 15.2%) |
| MathVista refusal rate (blank) | −10.8pt (15.2% → 4.4%)   | ~flat (4.4% → 4.0%) |
| POPE text_only refusal         | −70.8pt (95.2% → 24.4%)  | +12.8pt (24.4% → 37.2%) |
| img_lift (paired)              | +9pt (0.365 → 0.454)     | **−3.2pt** (0.454 → 0.422) |
| blank_shortcut (paired)        | **−4pt** (0.064 → 0.024) | **+2.8pt, 2×** (0.024 → 0.052) |

**Reading**:
- **The "over-thinking" / hit_max_tokens explosion is a size effect**, not RL.
  7B SFT (pre-RL) has the longest CoT, highest max-token-hit rate. RL actually
  **compresses** CoT length and reduces budget overflow.
- **The "RL refuses less" effect was also size**: the 3B→7B drop on POPE
  text_only is huge (95.2% → 24.4%); RL on top of that *increases* refusal.
- **The real RL effect at 7B is a regression**: acc down 3.2pt, img_lift down
  3.2pt, blank_shortcut doubles, tokens shorten. RL trades visual reasoning
  quality for shorter, more confident-prior-driven answers.

**Caveats**:
- n=250, std err on acc ≈ 3pt — the 3.2pt drop is at the edge of significance
  but the pattern is consistent across 4 independent metrics (acc, length,
  img_lift, blank_shortcut), which is harder to explain by noise alone.
- This is MathVista only. POPE is too saturated (4 models clustered at
  0.876-0.900) to differentiate. Need MathVision / ChartQA / HallusionBench
  to know whether the regression generalizes or is MathVista-specific.
- We don't know MMR1's RL training recipe; possibly it was optimized for a
  different metric (format compliance? length compactness?) that we're not
  measuring.

**Implication for OPD experiments**:
- Don't treat 7B RL as a strict upgrade over 7B SFT. If the goal is
  "distill the strongest math reasoner", **7B SFT is the right teacher on
  MathVista**, not 7B RL.
- T1 vanilla OPD baseline should add a second arm
  `7B SFT → 3B SFT` next to the planned `7B RL → 3B SFT` to test whether
  the OPD-from-regressed-teacher story holds.
- This shifts H3 from *"RL teacher over-thinks, OPD inherits the overthinking"*
  to *"RL teacher is a regression on visual grounding, OPD inherits the
  weakened grounding"*. Same shape, different mechanism.

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
   is the correct answer") **can only be scored if the JSONL row carries
   `choices`**. The 2026-05-19 smoke500 JSONLs were captured before this
   field was emitted — for those, run `scripts/data/backfill_choices.py
   --subset <subset> --run_dir <run>` to inject choices from the source
   subset, then re-aggregate. Future inference runs emit `choices` directly.
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

1. **Verify or challenge** the central new claim: **MMR1's RL step is a
   regression on MathVista at 7B**. Specifically:
   - +11.7pt acc from 3B SFT → 7B SFT (size), then **−3.2pt acc** from
     7B SFT → 7B RL (RL).
   - At 7B, RL reduces `img_lift` (0.454 → 0.422), **doubles**
     `blank_shortcut` (0.024 → 0.052), and compresses CoT length
     (535 → 479 tokens).
   - Over-thinking attribution: 7B SFT hits `max_new_tokens` at 25.2%,
     7B RL at 15.2%. **The RL step REDUCES over-thinking**, contrary to the
     earlier framing.
   - Is this regression robust to noise (n=250, σ≈3pt)? The pattern is
     consistent across 4 metrics, but the magnitude per metric is at the
     edge of significance. Worth verifying on MathVision / ChartQA /
     HallusionBench before claiming generality.
2. **Reframe H3**: previously "RL teacher over-thinks, OPD inherits the
   over-thinking". With the 7B SFT control: "RL teacher *loses visual
   grounding* and gains shortcut prior, OPD inherits the *weakened
   grounding*." Same broad shape (RL has artifacts that transfer), but
   the mechanism is different. Is this reframing supported by the data,
   or are we over-interpreting marginal effects?
3. **Look for signals we missed**, especially:
   - Anything pointing at **H1** (perception-reasoning frontier) — we
     don't have caption-only quadrants yet; can paired records be sliced
     any other way?
   - Anything pointing at **H2** (visual-signal misallocation) — we don't
     have forced decoding / per-token logprob yet; can we still infer
     anything from the prediction text?
   - Patterns in the raw `prediction` text — repetition loops (`T_RL_full`
     MathVista/224 generates repeated JSON arrays to 1024 tokens), language
     switches, malformed answers, "(A)/(B)/(C)" enumeration with
     option-text-only conclusions.
   - Compare raw text between 7B SFT and 7B RL on the same prompt:
     where do they differ qualitatively? Does RL produce more "I'll commit
     to (X) without checking the image" style answers?
4. **Reality-check the caveats**, especially: (a) is the Base degeneration
   explanation (prompt template mismatch) the most likely cause? (b) is
   the MathVista regression real or noise — what would Level-1 extension
   (MathVision / ChartQA / HallusionBench) need to show to confirm?
5. **Suggest the next audit / training step** given this evidence. Our
   current plan (in priority order):
   1. **Extend audit to more benchmarks** (MathVision / ChartQA /
      HallusionBench / MathVerse) to confirm RL regression generalizes
      or is MathVista-specific.
   2. **Fix Base inference** (prompt template) → re-run 3 Base passes.
   3. **Implement `--score_completion`** (forced decoding for H2).
   4. **Run vanilla OPD T1 baseline as TWO arms**: `7B RL → 3B SFT`
      (original plan) AND `7B SFT → 3B SFT` (new arm to test whether
      OPD from a regressed teacher hurts the student vs. OPD from a
      stronger pre-RL teacher).

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
