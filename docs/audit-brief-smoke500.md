# Brief: smoke500 audit data for independent review

**Purpose**: give an external reviewer (e.g., GPT) everything needed to
independently analyze the smoke500 results in `runs/audit/smoke500/` without
prior context.

**Captured at**: 2026-05-19.

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
      "accuracy": 0.655,
      "n_scored": 249,
      "tokens_mean": 479.348,
      "tokens_median": 391.0,
      "acc_per_token": 0.00137,
      "scorers": {"mcq_letter": 131, "numeric": 117, "loose_contains": 1, "skip_missing_image": 1}
    }
  ],
  "paired_full_blank": [
    {
      "model": "...",
      "benchmark": "MathVista",
      "n_paired": 249,
      "both_correct": 53,
      "full_only": 110,
      "blank_only": 22,
      "both_wrong": 64,
      "image_lift_rate": 0.442,
      "blank_shortcut_rate": 0.088
    }
  ]
}
```

---

## Headline numbers

**Per-cell accuracy:**

| Model | Mode | MathVista | POPE_adv |
|---|---|---|---|
| Base (Qwen2.5-VL-7B) | full | 0.426 | 0.876 |
| MMR1-3B-SFT          | full | 0.562 | 0.896 |
| MMR1-7B-RL           | full | **0.655** | **0.900** |
| Base                 | blank | 0.233 | 0.516 |
| MMR1-3B-SFT          | blank | 0.249 | 0.516 |
| MMR1-7B-RL           | blank | 0.301 | 0.516 |
| Base                 | text_only | 0.200 | 0.000 |
| MMR1-3B-SFT          | text_only | 0.276 | 0.172 |
| MMR1-7B-RL           | text_only | 0.256 | 0.272 |

**Per-cell mean output tokens:**

| Model | full MV | full POPE | blank MV | text_only MV |
|---|---|---|---|---|
| Base | 134 | 33 | 132 | 141 |
| SFT  | 379 | 41 | 377 | 384 |
| RL   | **479** | **61** | 453 | **473** |

**Paired image_lift / blank_shortcut:**

| Model | Bench | img_lift | blank_shortcut |
|---|---|---|---|
| Base | MathVista | 0.237 | 0.044 |
| SFT  | MathVista | 0.382 | 0.068 |
| RL   | MathVista | **0.442** | **0.088** |
| Base | POPE | 0.360 | 0.000 |
| SFT  | POPE | 0.428 | 0.048 |
| RL   | POPE | 0.408 | 0.024 |

---

## Caveats / things that LOOK like bugs but aren't

1. **POPE `text_only` acc is very low** (Base 0.0, MMR1 0.17-0.27): model
   often refuses to answer when asked yes/no without image. Parser sees
   "I cannot see…" → scored false. **Not a grounding signal — filter when analyzing.**
2. **MathVista `gold` is the letter** (A/B/C/...) even though source dataset
   stores option text; we mapped at prep time.
3. **`loose_contains` records are fuzzy fallback** (~1% of records). All cells
   in this run have ≤1 loose record, so it's not a concern here.
4. **Standard error for n=250 accuracy ≈ 0.03**; for small paired counts even
   higher. Treat single-cell deltas <0.04 as noise.
5. **MMR1 MathVista outputs often hit `max_new_tokens=1024`**: long CoT
   without EOS. `num_tokens=1024` doesn't necessarily mean useful CoT.
6. **Suspect scoring miss**: some Base `text_only` predictions like
   `"The correct answer is (C) Serrulate."` get `is_correct=false` even
   though "(C)" looks right. **Worth checking** whether the `mcq_letter`
   parser misses letters inside parentheses or longer sentences. If so,
   Base text_only MathVista (0.200) may be underestimated.

---

## What we'd like the reviewer to do

1. **Verify or challenge** these claims (from our own analysis):
   - **H3 (post-RL artifact transfer) signal is strong**:
     - RL teacher writes **27% more tokens** than SFT on MathVista for +9pt acc
     - RL writes **85% more tokens** on POPE for ~0pt acc (diminishing returns)
     - RL writes **473 tokens** in `text_only` MathVista (no image at all) →
       suggests over-thinking is **trained behavior**, not just capability
   - **Post-training raises BOTH `image_lift` AND `blank_shortcut`** (Base→MMR1):
     - img_lift +0.15~0.20, blank_shortcut +0.02~+0.05
     - Interpretation we want challenged: model becomes **more decisive** —
       uses image more when it has it, and bets more confidently when it
       doesn't. Is this "more grounded" or "more confident-prior"?
2. **Look for signals we missed**, especially:
   - Anything pointing at **H1** (perception-reasoning frontier) — we don't
     have caption-only quadrants yet; can paired records be sliced any other way?
   - Anything pointing at **H2** (visual-signal misallocation) — we don't have
     forced decoding / per-token logprob yet; can we still infer something?
   - Patterns in the raw `prediction` text — specific MMR1-RL failure modes
     that don't show up in scalar metrics (e.g., repetition loops, language
     switches, malformed answers).
3. **Reality-check the caveats**, especially caveat 6 (scoring parser may
   undercount Base `text_only` correctness on MathVista).
4. **Suggest the next audit / training step** given this evidence. Our current
   plan: implement `--score_completion` (forced decoding for H2) + start
   vanilla OPD T1 baseline.

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
