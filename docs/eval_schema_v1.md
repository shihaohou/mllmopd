# Unified per-prompt evaluation JSONL schema — `v1`

> Status: ACTIVE for all eval runs from 2026-05-27 (Rethinking-OPD atlas pivot)
> onward. Older runs (T1 trajectory, T2-1, T2-2 audits) keep their existing
> shape and are coerced into `v1` at read time via
> `mllmopd.eval.schema.from_existing_record`. See the migration plan at
> the bottom of this doc.

Every new eval driver — `run_t1_eval.sh`, `run_t2_1_eval.sh`,
`run_t2_2_eval.sh`, the `lmms-eval` wrappers under `scripts/eval/`, and the
in-process audit passes in `mllmopd.diagnostics` — emits one record per
prompt, one record per line, in this exact shape. Field names, types, and
the allowed value sets are normative; downstream readers
(`mllmopd.reporting.audit_table`, `mllmopd.analysis.aggregate_audit`,
the figure makers under `scripts/reporting/`) read these names verbatim.

The code-side mirror of this spec is in
[`src/mllmopd/eval/schema.py`](../src/mllmopd/eval/schema.py)
(`PromptEvalRecord` TypedDict + `FIELD_ORDER`). Update both files when
adding or renaming a field.

---

## Fields

Required ordering (column / jsonl key order) is the one shown below — it
matches `FIELD_ORDER` in `schema.py`.

| Field | Type | Nullable | Description |
|---|---|---|---|
| `arm` | `str` | No | Semantic arm label. One of: `S0`, `T1-Full`, `T1-Blank`, `T1-Text`, `T2-1`, `T2-2`, `Step3a-B0` / `Step3a-B1` / `Step3a-B2`, `Teacher`. Producers may extend this set; readers should treat it as an open enum. |
| `student` | `str` | No | Student model name as it appears on disk / wandb (`MMR1-3B-SFT`, `MMR1-7B-SFT`, `Qwen2.5-VL-7B-Instruct`, …). The teacher arm fills its own model name here. |
| `teacher_mode` | `Literal` | No | One of `full` / `blank` / `text` / `offline_full` / `offline_blank` / `none`. `none` is used by the base / teacher rows where no distillation is happening. |
| `policy` | `Literal` | No | `on_policy` / `off_policy` / `none`. |
| `step` | `int` | No | Training step. `0` = base / pre-training checkpoint, `230` = final canonical T1 step. Teacher / base rows use `0`. |
| `eval_set` | `str` | No | Dataset subset identifier. Canonical values: `dev_mmr1_v0_1k`, `level1_subset_v0`, `opd_target_133`, or `lmmseval/<benchmark>` for lmms-eval-driven evals. |
| `benchmark` | `str` | No | Benchmark name. Canonical values: `MathVista`, `MathVerse`, `MathVision`, `LogicVista`, `ChartQA`, `POPE_adv`, `HallusionBench`, `MMR1-RL_dev` (used when `eval_set == dev_mmr1_v0_1k`). |
| `mode` | `Literal` | No | Image conditioning at eval time: `full_image` / `blank_image` / `text_only`. The same prompt is typically eval'd in `full_image` and `blank_image` to get the visual-dependency delta. |
| `id` | `str` | No | Prompt id. Stable across modes / arms / steps so the joiner can pair runs (`dev_mmr1_v0_***`, `opd_target_***`, level1 row id, lmms-eval `doc_id`). Always serialised as a string even when the source dataset uses ints. |
| `prediction` | `str` | No | Full model output, raw (not trimmed, not stripped of CoT / `<think>` blocks). |
| `gold` | `str \| None` | Yes | Ground-truth answer text. `None` for open-ended HallusionBench rows or rows whose gold field was empty in the source dataset. |
| `parsed_answer` | `str \| None` | Yes | Extracted answer after running the scorer's parser. May equal `gold` (e.g. MCQ letter `"C"`) or differ (e.g. `"100"` extracted from `"\\boxed{100}"`). `None` if no parse step ran. |
| `is_correct` | `bool \| None` | Yes | Scoring result. `None` is used for `"skip_missing_image"` and `"skip_empty_gold"` rows (scoring didn't run); otherwise `True` / `False`. |
| `scorer` | `str` | No | Which scorer / judge produced `is_correct`. Canonical values: `claude-sonnet-4-6`, `rule_based`, `string_match`, `mathvista_rule`, `mcq_letter`, `yesno`, `numeric`, `loose_contains`, `skip_missing_image`, `skip_empty_gold`. |
| `num_tokens` | `int` | No | Output token count (model tokenizer). |
| `hit_max_tokens` | `bool` | No | `True` iff `num_tokens >= max_new_tokens - tolerance` (`tolerance` is producer-defined, typically 4). Distinguishes a clean stop from a truncation that may have killed a correct answer. |
| `blankness` | `bool` | No | `mllmopd.eval.blankness.detect_blankness(prediction)`. |
| `early_blankness` | `bool` | No | `mllmopd.eval.blankness.detect_early_blankness(prediction)` — pattern hit in the first 400 chars. The mechanism-relevant variant for on-policy prefix self-conditioning. |
| `blankness_in_think` | `bool` | No | `mllmopd.eval.blankness.detect_blankness_in_think(prediction)` — pattern hit inside the first `<think>...</think>` block. False if no think block. |
| `parse_path` | `str` | No | Short tag describing how `prediction` was parsed. Canonical values: `boxed_regex`, `answer_tag_split`, `raw`, plus scorer-specific tags emitted by `mllmopd.diagnostics.scorers` (`tag:mcq_letter`, `tag_fallback:numeric`, `yesno_after_tag_fallback`, …). |
| `extras` | `dict[str, Any]` | No (use `{}`) | Free-form, eval-set-specific bag. Recognised keys include `prompt_len` (int), `choices` (list[str] for MCQ rows), `image_path` (str), `response_time` (float, seconds), `error` (str), `rescored_is_correct` (bool), `rescore_changed` (int). |

### Type & nullability rules

* Boolean mechanism signals (`blankness`, `early_blankness`,
  `blankness_in_think`, `hit_max_tokens`) are **never** `None` in v1 — when
  the producer doesn't run the check it must emit `False`. Readers may
  assume `bool`, not `Optional[bool]`.
* `is_correct` is the only ternary boolean; `None` is reserved for rows
  the scorer explicitly skipped.
* `parse_path` and `scorer` are non-empty strings. Use `"unknown"` (not
  the empty string) if the producer genuinely can't fill them.
* `extras` is always a dict, possibly empty. Never `None`.

---

## Example line

```json
{
  "arm": "T1-Blank",
  "student": "MMR1-3B-SFT",
  "teacher_mode": "blank",
  "policy": "on_policy",
  "step": 230,
  "eval_set": "dev_mmr1_v0_1k",
  "benchmark": "MMR1-RL_dev",
  "mode": "full_image",
  "id": "dev_mmr1_v0_00427",
  "prediction": "<think>The image appears to be blank, so I cannot determine the answer from it.</think>\nI don't see any chart in the image, so I cannot answer.",
  "gold": "42",
  "parsed_answer": null,
  "is_correct": false,
  "scorer": "rule_based",
  "num_tokens": 47,
  "hit_max_tokens": false,
  "blankness": true,
  "early_blankness": true,
  "blankness_in_think": true,
  "parse_path": "raw",
  "extras": {"prompt_len": 312, "image_path": "data/mmr1/img/00427.png"}
}
```

The example is annotated to highlight a typical T1-Blank failure at
step 230: the model fires three blankness signals (anywhere, early, and
inside the think block), produces no parseable answer, and is scored
incorrect.

---

## Existing eval scripts and field coverage

These three drivers are the current per-prompt JSONL producers in the
repo. The "v1 status" column shows which v1 fields each driver already
emits and which it still needs to add (this is the follow-up work, not in
scope for the current task).

### `scripts/audit/run_t1_eval.sh` → `mllmopd.diagnostics.run_audit_pass[_sglang]`

Emits today (`_emit_row` in `src/mllmopd/diagnostics/run_audit_pass.py`):

```
id, benchmark, mode, model, prediction, num_tokens, prompt_len,
gold, is_correct, scorer, parse_path[, choices, error]
```

v1 deltas:

* Rename `model` → `student` (the shell launcher already knows which arm
  it's running; it can fill `arm`, `student`, `teacher_mode`, `policy`,
  `step`, `eval_set` from CLI flags).
* Add `parsed_answer` (scorer already has it internally — just plumb it
  out).
* Add `num_tokens`-derived `hit_max_tokens` (the driver knows
  `max_new_tokens`).
* Add `blankness`, `early_blankness`, `blankness_in_think` via
  `mllmopd.eval.blankness.analyze(prediction)`.
* Move `prompt_len`, `choices`, `error` into `extras`.

### `scripts/audit/run_t2_1_eval.sh` → same driver

Same gaps as above; the shell launcher needs to also set
`teacher_mode=full` / `blank` and `policy=off_policy` / `on_policy` based
on the T2-1 arm variant.

### `scripts/eval/run_lmmseval.sh` (and the wrappers under `scripts/eval/`)

Today the lmms-eval JSON dump uses lmms-eval's own keys (`doc_id`,
`target`, `resps`, …). The post-processor in
`mllmopd.analysis.aggregate_audit` already normalises some of these; the
v1 producer should sit between lmms-eval and `aggregate_audit` and emit
the v1 record directly. Specifically:

* `id` ← `doc_id` (stringified)
* `prediction` ← `resps[0]` (first sample)
* `gold` ← `target`
* `is_correct` ← derived from lmms-eval's `metric` outputs
* `scorer` ← either `claude-sonnet-4-6` (HallusionBench / open-ended
  benchmarks) or the lmms-eval task-specific scorer (`mathvista_rule` for
  MathVista, etc).
* `parse_path` ← `"lmmseval/<task_id>"`.
* `extras["response_time"]` ← lmms-eval's `latency` if present.

---

## Migration plan

1. **Old jsonls keep their old shape.** Don't rewrite the H800 runs under
   `runs/audit/t1_*` and `runs/audit/t2_1_*` — they're append-only audit
   artefacts.
2. **Read-side coercion** is implemented in
   `mllmopd.eval.schema.from_existing_record`. Old jsonls are coerced
   into v1 at read time; the legacy column names (`model`, `answer`,
   `response`) are mapped to v1 names and any unrecognised keys land in
   `extras`. Mechanism signals (`blankness*`) default to `False` —
   downstream code that needs them on legacy data should pipe the
   `prediction` through `mllmopd.eval.blankness.analyze` itself.
3. **New eval runs** (from the Rethinking-atlas pivot onward) emit v1
   directly via the schema module — no coercion needed at read time.
4. **Atlas figure makers** (`scripts/reporting/make_atlas_*.py`) read v1
   only. They call `from_existing_record` if pointed at a legacy jsonl,
   so the producer side can be migrated incrementally without breaking
   the figures.

---

## Versioning

This file documents `v1`. Future schema bumps:

* Backwards-compatible additions (new optional field) **do not** bump the
  version. Add the field with a sensible default, update both this doc
  and `PromptEvalRecord`, and ensure `from_existing_record` fills the
  default for old jsonls.
* Breaking changes (rename / type change / removal) bump to `v2` and the
  spec moves to `docs/eval_schema_v2.md`. `from_existing_record` keeps
  v1 readability.
