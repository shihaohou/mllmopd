# mllmopd — implementation reference

A single-document map of what the repo can run, what each script does, and what metrics it produces. Companion to [`research-plan.md`](research-plan.md) (motivation) and [`experiment-protocol.md`](experiment-protocol.md) (metric tables).

Audience: code reviewers and the author. The goal is for someone reading only this file to be able to evaluate the implementation without grepping the codebase.

---

## 1. Scope and limits of this document

**Covered:** every script and Python module in the repo, the experiments they enable, the metrics they emit, and how those metrics map back to the three working hypotheses (H1–H3, see `research-plan.md` §"Candidate hypotheses").

**Not covered:** the motivation behind H1/H2/H3 (see `research-plan.md`), Uni-OPD's own internals (see `upstream-cheatsheet.md`), or paper-side framing.

**Status flags used below:**

| Flag | Meaning |
|---|---|
| **READY** | Runnable end-to-end as committed. |
| **DEPENDS** | Runnable once a stated prerequisite is satisfied (env / download). |
| **STUB** | Has structure but a key piece (loader / scorer / formula) is a placeholder. |
| **TODO** | Not started; only referenced from configs or comments. |

If a script is `STUB`, the doc says exactly which line needs work.

---

## 2. Runtime model

Two physical machines, one git repo.

```
Mac (laptop)                                    H800 dev box (arc-wlf1-ge103-4)
─────────────────────────────────────────       ────────────────────────────────────────────
- edit code, configs, docs                      - clone --recurse-submodules
- read upstream submodules (Uni-OPD etc.)       - run setup_*.sh once to build 3 conda envs
- prep audit subset metadata locally            - run audit / training / eval in tmux
- generate figures from rsync'd JSONLs          - all heavy artifacts on /home/.../<user>/...

git push  ───►  github.com/shihaohou/mllmopd  ◄──  git pull --recurse-submodules
                                                   rsync ./runs/audit/<id>/ → Mac
```

**Three conda envs on the dev box:**

| Env | Python | Purpose | Builder script |
|---|---|---|---|
| `Uni-OPD` | 3.12.12 | training (Megatron + SGLang + miles) | `scripts/env/setup_train_env.sh` |
| `Uni-OPD-LMMS-Eval` | 3.12.13 | inference for audit + lmms-eval | `scripts/env/setup_lmmseval_env.sh` |
| `Uni-OPD-LLM-Eval` | 3.10.20 | optional (text-only OPD comparison) | not implemented yet |

**Submodules pinned in `.gitmodules`:** `Uni-OPD` (upstream main, read-only), `Megatron-LM @ 3714d81d`, `sglang @ 24c91001`, `lmms-eval` (main).

---

## 3. Experiment menu

Stages are roughly in increasing setup cost. Stop at the earliest stage that yields the figure you need.

### 3.1 Smoke audit (Stage S0) — **DEPENDS on `Uni-OPD-LMMS-Eval` env (~30 min build)**

Goal: prove the inference path end-to-end and produce a first behavior snapshot of teacher vs student under three perception modes.

Inputs (all already on disk):
- Teacher: `MMR1-7B-RL` (`/home/.../datasets/MMR1-7B-RL`)
- Student: `MMR1-3B-SFT` (`/home/.../datasets/MMR1-3B-SFT`)
- Eval data: `MathVista-mini`, `POPE-adversarial`

Driver: `scripts/audit/run_smoke.sh` (READY) + `configs/audit/audit_v0_smoke.yaml`. Calls `src/mllmopd/diagnostics/run_audit_pass.py` (DEPENDS-on-GPU — see §5.3; inference path implemented for Qwen2.5-VL, forced-decoding for H2 still missing) for 5 passes:

| Pass tag | Model | Mode |
|---|---|---|
| `T_RL_full` | MMR1-7B-RL | full_image |
| `T_RL_blank` | MMR1-7B-RL | blank_image |
| `S_full` | MMR1-3B-SFT | full_image |
| `S_blank` | MMR1-3B-SFT | blank_image |
| `S_text_only` | MMR1-3B-SFT | text_only |

Time: ~1–2 h on a single H800. No fine-tuning, no SGLang server, no Megatron.

Output: `$MLLMOPD_RUNS/audit/<run_id>/{*.jsonl, summary.json}`.

Metrics produced (see §7): accuracy, response length, full−blank gap (student grounding), teacher length vs accuracy.

What this stage **cannot** answer:
- RL-teacher vs SFT-teacher comparison (`MMR1-7B-SFT` not on disk → Fig 1 missing).
- Oracle-caption decomposition (MathVista doesn't ship gold captions; needs a captioning pass first).
- Visual-dependency per-token KL (run_audit_pass currently only generates; doesn't extract per-token logprobs under varying images — see §8).

### 3.2 Level-1 full audit (Stage S1) — **DEPENDS on additional benchmark downloads + MMR1-7B-SFT**

Adds the missing benchmarks and the SFT-teacher control:

Required additions: MMR1-7B-SFT; benchmarks MathVision, MathVerse, LogicVista, ChartQA, HallusionBench, CharXiv, MMMU.

Driver: `scripts/audit/run_level1.sh` (READY but slow) + `configs/audit/audit_v0.yaml`. Seven passes × 2000 prompts (full / blank teachers, full / blank / caption-only-blank / image+caption student).

This is where Fig 1 (RL vs SFT teacher: accuracy gain vs length gain) and Fig 4 (full/blank/oracle table) become computable. Time: ~6–12 h on 1 H800; can be parallelized across GPUs.

### 3.3 Vanilla OPD baseline (Stage T1) — **DEPENDS on `Uni-OPD` train env (~1–2 h build) + Uni-OPD YAML schema reconciliation**

Driver: `scripts/train/start_teacher_server.sh` + `scripts/train/opd_mmr1_3b_baseline.sh` (both STUB — argument names not verified against upstream).

Layout on the 8 GPUs:
- GPU 0–1: SGLang teacher server hosting MMR1-7B-RL.
- GPU 2–7: trainer for MMR1-3B-SFT student.

Outputs: training logs + checkpoint at `$MLLMOPD_RUNS/vanilla_opd_mmr1_3b_v0/`.

**Open issue:** `configs/baseline/mmr1_3b_vanilla_opd.yaml` field names are best-effort. First run must diff against an actual upstream MLLM example (e.g. `third_party/Uni-OPD/exps/scripts/OPD/single_teacher/` and pair with `miles/utils/arguments.py`) and fix mismatches. The conf YAML is wired with `${VAR}` expansion so all paths come from `.env`.

### 3.4 OPD evaluation (Stage E1) — **DEPENDS on Stage T1 + `Uni-OPD-LMMS-Eval` env**

Driver: `scripts/eval/run_lmmseval.sh` (DEPENDS — `qwen2_5_vl` model_args may need tweaking for MMR1-derived checkpoints).

Tasks (default set): `mathvista_testmini`, `mathvision`, `mathverse_testmini`, `logicvista`, `chartqa`, `hallusion_bench_image`, `charxiv`, `mmmu_val`.

### 3.5 Hypothesis-specific probes (Stage P1) — **TODO**

These are not in code yet; they're listed here so they're not forgotten.

- **H1 probe** — perception-hard vs reasoning-hard classifier built from the three caption-decomposed modes (`full_image`, `caption_only_blank`, `image_plus_caption`). Modes themselves are implemented in `mllm_corruptions.apply_mode`; the classifier and the captioning pass that feeds them are not. See §7.4 for the rewritten quadrant definition.
- **H2 probe** — per-token visual-dependency scalar extractor. Forced-decoding the teacher over a fixed completion while toggling image full↔blank; record `logp_full[t]` and `logp_blank[t]` for the generated token only (not the full vocab — see §7.4 for cost). Needs ~30 lines added to `run_audit_pass.py` under a new `--score_completion` mode.
- **H3 probe** — modality shortcut. The `blank_image` / `irrelevant_image` / `swap_image` corruption modes are implemented; the paired full-vs-blank 2×2 contingency lands in `summary.json["paired_full_blank"]` automatically. What's still TODO is the irrelevant/swap variants in the smoke config and the paired metric for those.

---

## 4. Scripts inventory

All scripts live under `scripts/`. Permissions: `*.sh` are `chmod +x`; `*.py` are invoked as `python script.py`.

### 4.1 `scripts/init/` — one-time bootstrap

| Script | Status | What it does | Run where |
|---|---|---|---|
| `02_devbox_bootstrap.sh` | READY | Sources `.env`, checks GPUs visible, verifies all 4 submodules populated, makes scratch dirs and `ln -s` runs/data/models. | dev box, once |
| `devbox_sync.sh` | READY | `git pull --ff-only --recurse-submodules` then `git submodule update --init --recursive`. | dev box, anytime |
| `switch_to_fork.sh` | READY | Future-use: re-point `third_party/Uni-OPD` submodule from upstream to a personal fork once you start committing changes to OPD reward / loss. Verifies `https://github.com/$GH_USER/Uni-OPD` is reachable. | Mac, once when needed |

### 4.2 `scripts/env/` — conda env builders

All three are verbatim transcripts of `third_party/Uni-OPD/docs/{build_env,build_eval_env}.md` with paths sourced from `.env`. No invented commands.

| Script | Status | Builds | Time |
|---|---|---|---|
| `setup_train_env.sh` | READY | conda env `Uni-OPD` — torch 2.9.1+cu128, flash-attn 2.7.4.post1, apex, transformer_engine 2.10.0, Megatron-LM, miles. Calls `apply_patches.sh` at the end. | ~1–2 h |
| `setup_lmmseval_env.sh` | READY | conda env `Uni-OPD-LMMS-Eval` — vllm, lmms-eval, qwen-vl-utils, math-verify, latex2sympy2_extended, nltk wordnet. | ~30 min |
| `apply_patches.sh` | READY | Idempotently `git apply` the two patch files `miles/docker/patch/v0.5.7/{sglang_psp,megatron}.patch`. Detects already-applied via `git apply --reverse --check`. | <1 s |

### 4.3 `scripts/data/` — data wrangling

| Script | Status | What it does |
|---|---|---|
| `check_inventory.sh` | READY | `test -e` against every path declared in `.env`. Prints ✓/MISSING + `du -sh`. Safe to run on Mac (everything will print MISSING). |
| `download_mmr1.sh` | STUB | Iterates HF dataset/model ids: `MMR1/MMR1-RL`, `MMR1/MMR1-SFT`, `MMR1/MMR1-{3B,7B}-{SFT,RL}`. IDs are best-effort — first 404 means update IDs against MMR1 GitHub README. |
| `download_openmmr.sh` | STUB | Same pattern for `OpenMMReasoner/OpenMMReasoner-{SFT-874K,RL-74K}`, `HuggingFaceM4/ChartQA`, plus an InfoVQA placeholder. |
| `prep_audit_subset.py` | READY | Builds the audit JSONL by sampling from each named benchmark. Supports a `--only` flag (uniform mix over named benches) and per-benchmark `*_PATH` env-var overrides for local dirs (e.g. `MATHVISTA_PATH`). Falls back to HF ids if env var unset. |

`prep_audit_subset.py` registry (`BENCHMARK_REGISTRY`): each entry is `(hf_id, env_var_for_local_path, split)`. Currently: MathVista, MathVision, MathVerse, LogicVista, ChartQA, HallusionBench, CharXiv, MMMU, POPE_adversarial.

### 4.4 `scripts/audit/` — diagnostic runners

| Script | Status | What it does |
|---|---|---|
| `run_smoke.sh` | READY | Builds smoke subset (500 prompts, MathVista + POPE), runs the 5 passes from §3.1, aggregates, prints table. Reads model + dataset paths from env vars. Hard-codes `CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-0}` for single-GPU sequential. **Recommended cold-start ramp: `--limit 2 --debug`, then `--limit 20`, then `--limit 100`, then full 500.** |
| `run_level1.sh` | READY-but-slow | Same pattern but 7 passes × 2000 prompts. Includes T_SFT_full plus `S_caption_blank` and `S_image_plus_cap` (the two modes that make H1 quadrant classification possible). Will fail until MMR1-7B-SFT and captions exist (gracefully — single pass exits non-zero, others continue if `set -e` is removed). |

### 4.5 `scripts/train/` — OPD training (Stage T1)

| Script | Status | What it does |
|---|---|---|
| `start_teacher_server.sh` | STUB | Wraps `miles/Uni_OPD_utils/scripts/server/run_sglang_server.sh`. Writes endpoint to `teacher_server_list.json` via inline python heredoc. Argument names (`--model-path`, `--port`) match SGLang convention but not verified against the actual `run_sglang_server.sh` arg parser. |
| `opd_mmr1_3b_baseline.sh` | STUB | Wraps `miles/Uni_OPD_utils/ray_launcher.py`. Forwards `--config`, `--output-dir`, `--run-name`. Real Uni-OPD launchers (see `exps/scripts/OPD/single_teacher/0413/`) pass `--rollout-batch-size`, `--sample-n`, `--lr` etc. directly; we route those through the YAML instead. Needs a 5-min diff against any single_teacher script to reconcile. |

### 4.6 `scripts/eval/` — final-checkpoint evaluation

| Script | Status | What it does |
|---|---|---|
| `run_lmmseval.sh` | DEPENDS | `accelerate launch -m lmms_eval --model qwen2_5_vl --model_args pretrained=$MODEL_PATH ...`. If MMR1 ckpts work with the `qwen2_5_vl` lmms-eval model wrapper, this just runs; otherwise pick a different `--model` (e.g. `hf-multimodal`). |

---

## 5. Python package — `src/mllmopd/`

Installed as a local pyproject package (`pip install -e .`). Heavy deps (torch, transformers, vllm) are imported lazily inside functions so `import mllmopd` works on Mac without GPU.

### 5.1 `mllmopd.data.mllm_corruptions` — READY

PIL-only image transforms used by audit modes.

```python
apply_mode(image: PIL.Image, mode: str, *, other_image=None, caption=None)
    -> tuple[PIL.Image | None, str | None]
```

Mode → (image_out, prefix_text):
- `full_image` → `(image, None)`
- `blank_image` → `(255×W×H white, None)` (preserves dims so vision tokenizer produces same patch count)
- `text_only` → `(None, None)`
- `caption_only_blank` → `(blank, "[Image description: <caption>]\n")` — requires non-empty caption (this is the new name for what was previously called `oracle_caption`; the old name is kept as an alias)
- `image_plus_caption` → `(image, "[Image description: <caption>]\n")` — the four-quadrant H1 classifier needs *both* blank+caption and image+caption to disambiguate perception-hard from reasoning-hard prompts
- `swap_image` → `(other_image, None)`
- `irrelevant_image` → `(noise_with_seed, None)` (per-pixel uniform RGB)

### 5.2 `mllmopd.diagnostics.visual_dependency` — READY (numpy-only, untested end-to-end)

Math utilities for H2 ("loss mass landing on low-visual-dependency tokens").

Two flavors, very different storage costs:

```python
# Default — cheap (T,) scalar per sample.
vis_dep_generated(logp_full: np.ndarray, logp_blank: np.ndarray) -> np.ndarray
    # both (T,) log-probs of the actually-generated token y_t under each
    # image conditioning; returns (T,) |Δ logp| per token

# Case-study — expensive, full-vocab KL over (T, V) log-probs.
kl_per_token(logp_a: np.ndarray, logp_b: np.ndarray) -> np.ndarray
    # ~1.2 GB float32 per Qwen2.5-VL sample at T=1024, V≈152K — only feasible
    # on ~10-100 hand-picked examples, never at audit scale

quantile_bins(values: np.ndarray, n_bins: int = 5) -> np.ndarray
    # equal-frequency binning

loss_mass_by_bin(loss: np.ndarray, bins: np.ndarray, n_bins: int = 5) -> np.ndarray
    # normalized fraction of loss per bin

summarize_records(records: Iterable[dict]) -> dict
    # records must have keys "vis_dep" (T,) and "opd_loss" (T,)
```

**Gap:** no script currently produces the `(vis_dep, opd_loss)` paired arrays. Need to extend `run_audit_pass.py` (or write a new `extract_token_logprobs.py`) to:
1. Force-decode the teacher over a fixed completion `y` with image `x`.
2. Re-decode the same `y` with `x` blanked.
3. Gather `logp(y_t)` at each position from both decodes (scalar; **do not** materialize (T, V) full-vocab log-probs in the default path).
4. Save `(logp_full, logp_blank)` as two (T,) float32 arrays per sample.

After that, audit-scale callers compute `vis_dep_generated(logp_full, logp_blank)`. The full-vocab `kl_per_token` is reserved for explicit case studies that pass `--full_vocab_kl 1`. OPD loss attribution requires the actual training loss tensor, which lives in miles — not yet wired.

### 5.3 `mllmopd.diagnostics.run_audit_pass` — DEPENDS-on-GPU (inference path done; forced-decoding for H2 still TODO)

Loads a model with `AutoProcessor` + `Qwen2_5_VLForConditionalGeneration` (auto-falls back to `AutoModelForVision2Seq` for older transformers; auto-falls back from `flash_attention_2` to `sdpa` if FA2 unavailable). Iterates the subset, applies a perception mode, generates with `do_sample=False`, scores with the benchmark-aware dispatcher in `mllmopd.diagnostics.scorers`, writes one JSONL line per prompt.

CLI:
```bash
python -m mllmopd.diagnostics.run_audit_pass \
    --subset SUBSET.jsonl \
    --model $MMR1_7B_RL_CKPT  # local path or HF id
    --mode full_image         # full_image|blank_image|text_only|
                              # caption_only_blank|image_plus_caption|
                              # swap_image|irrelevant_image|oracle_caption(alias)
    --out OUT.jsonl
    [--max-new-tokens 1024] [--limit 0] [--debug]
```

`--debug` dumps prompt + raw prediction + scoring decision to stderr per sample; pair it with `--limit 2` for cold-start sanity, then ramp 20 → 100 → 500.

Per-prompt output schema:
```python
{
  "id": str,            # "<benchmark>/<idx>"
  "benchmark": str,
  "mode": str,          # one of the audit modes
  "model": str,
  "prediction": str,
  "num_tokens": int,    # generated only, prompt excluded
  "prompt_len": int,
  "gold": str | None,
  "is_correct": bool | None,
  "scorer": str,        # which scorer fired: yesno | mcq_letter |
                        # numeric | loose_contains | skip_empty_gold
}
```

The `scorer` field lets downstream code filter `loose_contains` cells out of any plot — those are the cells where the dispatcher didn't recognize the gold shape and fell back to substring matching. Real headline numbers still come from lmms-eval at Stage E1.

### 5.3.1 `mllmopd.diagnostics.scorers` — READY

Benchmark-aware lightweight scorers, all numpy-free so they run on Mac.

```python
score_for_benchmark(benchmark: str, pred: str, gold) -> tuple[bool | None, str]
    # returns (is_correct, scorer_name); name is one of
    # "yesno" | "mcq_letter" | "numeric" | "loose_contains" | "skip_empty_gold"

score_yesno(pred, gold)        # POPE / HallusionBench style
score_mcq_letter(pred, gold)   # gold like "A", "B", ...
score_numeric(pred, gold, rel_tol=0.01, abs_tol=1e-3)  # gold parses as float
```

Dispatch logic: benchmark name (`pope*`, `hallusionbench*` → yesno) takes priority; otherwise gold shape determines scorer (single A-H letter → mcq, parses as float → numeric, else loose). Empty pred against a non-empty gold scores `False`, not `None` — we know the model failed, that's not "unscoreable".

Known caveats (see §8):
- The subset loader assumes `image` is either a PIL object (HF datasets default) or a path. `prep_audit_subset.py` now persists PIL images to disk so the JSONL only stores paths; multi-image samples take the first frame only.
- Real per-benchmark scorers live in lmms-eval; the dispatchers above are triage tools. Cells dominated by `scorer=loose_contains` should not be quoted in any figure.

### 5.4 `mllmopd.analysis.aggregate_audit` — READY

Walks `*.jsonl` in a run dir, groups by `(model, mode, benchmark)`, computes:

```python
{
  "n":             int,
  "n_scored":      int,                   # rows with non-null is_correct
  "accuracy":      float | None,
  "tokens_mean":   float | None,          # mean generated tokens
  "tokens_median": float | None,
  "acc_per_token": float | None,          # ~ accuracy / mean tokens (efficiency proxy)
  "scorers":       dict[str, int],        # which scorer fired how often
                                          # (filter out cells dominated by "loose_contains")
}
```

In addition, a second list is computed per `(model, benchmark)` cross-cell: the prompt-id-paired full-image vs blank-image contingency, which is what we actually use for H3 modality-shortcut diagnosis. See §7.3.

```python
# summary.json["paired_full_blank"] entries
{
  "model": str, "benchmark": str, "n_paired": int,
  "both_correct": int, "full_only": int, "blank_only": int, "both_wrong": int,
  "image_lift_rate": float,        # full_only / n_paired
  "blank_shortcut_rate": float,    # blank_only / n_paired
}
```

Written to `summary.json` as `{"cells": [...], "paired_full_blank": [...]}`.

### 5.5 `mllmopd.reporting.audit_table` — READY

Pretty-prints `summary.json` as two fixed-width tables: the per-cell stats and the per-(model, benchmark) full-vs-blank paired contingency. No file output.

### 5.6 `mllmopd.reporting.figures` — partial (Fig 1 only)

| Function | Status | Produces |
|---|---|---|
| `fig1_accuracy_vs_length(summary, out)` | READY | Scatter of `(Δresp_len_rel, Δaccuracy)` for each benchmark, RL−SFT teachers, full_image only. Annotated. |
| Fig 2 (loss mass by visual-dep bin) | TODO | Needs §5.2 logprob extraction first. |
| Fig 3 (perception-hard / reasoning-hard quadrant) | TODO | Needs oracle-caption + image-only passes. |
| Fig 4 (full/blank/oracle accuracy across {T_RL, T_SFT, S, OPD-S}) | TODO | Mostly a bar chart over existing summary fields. |
| Fig 5 (post-OPD deltas in accuracy/length/vis-dep/cross-domain) | TODO | Needs Stage T1 + E1 done. |

---

## 6. Configs

| Path | Status | Notes |
|---|---|---|
| `configs/audit/audit_v0_smoke.yaml` | READY | 500-prompt mix of MathVista + POPE_adversarial, 5 passes, uses `model_env`/`local_path_env` so paths come from `.env`. |
| `configs/audit/audit_v0.yaml` | READY-but-blocked | 2000-prompt full mix; blocked on benchmark downloads + MMR1-7B-SFT. |
| `configs/baseline/mmr1_3b_vanilla_opd.yaml` | STUB | Field names match my reading of `miles/utils/arguments.py` but are not verified. Must diff against `third_party/Uni-OPD/exps/scripts/OPD/single_teacher/` before first run. |
| `.env.example` | READY | Pre-wired with the dev box paths (`/home/web_server/antispam/project/houshihao/{models,datasets}`). |

---

## 7. Metrics — definitions and formulas

Stage-by-stage, exactly what comes out.

### 7.1 Per-prompt JSONL (every audit pass)

See schema in §5.3. The cheap metrics computed in this file are `num_tokens`, `is_correct`, `prediction`. Nothing else — no logprobs, no entropy, no per-token data yet.

### 7.2 Per-cell aggregates (after `aggregate_audit.py`)

For each `(model, mode, benchmark)`:

- **accuracy** = `Σ is_correct / n_scored`
- **tokens_mean** = `mean(num_tokens)`
- **acc_per_token** = `accuracy / tokens_mean` (efficiency, see H3)

### 7.3 Cross-cell derived metrics (computed in `figures.py` or notebooks)

| Metric | Formula | Hypothesis use |
|---|---|---|
| Full−blank accuracy gap | `acc(model, full_image, b) - acc(model, blank_image, b)` | H3 modality shortcut — **weak signal**; per-prompt paired counts below are stronger |
| **Paired image lift** | `full_only / n_paired` | H3 — fraction of prompts where the image actually changed the model's decision toward correct |
| **Paired blank shortcut** | `blank_only / n_paired` | H3 — fraction of prompts where blanking the image flipped a wrong answer to a correct one (pure prior / spurious shortcut) |
| RL−SFT accuracy gain | `acc(MMR1-7B-RL, full, b) - acc(MMR1-7B-SFT, full, b)` | H3 capability vs artifact (paired with length gain) |
| RL−SFT length gain (rel) | `(tokens_mean_RL - tokens_mean_SFT) / tokens_mean_SFT` | H3 overthinking transfer |
| Acc(blank) / Acc(full) | `acc_blank / acc_full` | Inverse of full−blank gap; only valid when used alongside paired counts above |

### 7.4 Per-token visual dependency (planned, §5.2 + §5.3 extension)

**Default (audit-scale)** — scalar from generated tokens only:

```
vis_dep_gen[t] = | logp_full(y_t | x, y_<t) - logp_blank(y_t | x, y_<t) |
```

Storage: (T,) float32 per sample. This is what runs at audit scale.

**Case study (≤100 samples)** — full-vocab KL:

```
vis_dep_kl[t] = KL( softmax(z_teacher(x_full, y_<t))  ||  softmax(z_teacher(x_blank, y_<t)) )
```

Storage: (T, V) float32 per sample (~1.2 GB on Qwen2.5-VL). Only enabled per-case-study via an explicit flag, never on the full audit.

For Fig 2: bin tokens by quantile of `vis_dep_gen`, plot `sum(opd_loss[bin]) / sum(opd_loss)`.

For Fig 3 — H1 quadrant from the (image × caption) decomposition. Let `C_full = correct(full_image)`, `C_capblank = correct(caption_only_blank)`, `C_imgcap = correct(image_plus_caption)`. Per-prompt buckets:

| Bucket | Condition | Interpretation |
|---|---|---|
| **perception-hard** | `not C_full and C_capblank` | Model needs visual *facts*, can't see them; gold caption rescues it |
| **reasoning-hard** | `C_full and not C_imgcap` (or both fail) | Visual facts are accessible but the reasoning fails even when the caption is also given |
| **both-easy** | `C_full and C_capblank` | Trivial — exclude from any frontier plot |
| **shortcut-prone** | `not C_full and not C_capblank and correct(blank_image)` | Both visual and caption inputs fail but blank still scores correct — pure prior / option distribution |
| residual | none of the above | Mixed / inconsistent; report as a single "other" bucket and inspect manually |

The earlier H1 definition collapsed "caption-only" and "blank-image+caption" into the single `oracle_caption` mode, which made the second bucket incoherent (the success condition was the same input as the failure condition). Splitting into `caption_only_blank` and `image_plus_caption` is what fixes it.

### 7.5 Hypothesis decision table

Re-stated from `experiment-protocol.md` so reviewers can see the mapping in this file:

| Observation | Hypothesis it supports | Method direction |
|---|---|---|
| Vanilla OPD ↑acc, ↑↑length, easy-Q overthink, cross-bench drop | H3 | length/difficulty-aware OPD, SFT-reference artifact filtering |
| Reasoning-bench ↑, perception-bench flat, oracle > image | H1 | perception–reasoning frontier OPD |
| OPD loss mass on low-vis-dep tokens | H2 | visual-dependency-weighted OPD |
| RL teacher full ≈ blank but correct, OPD inherits | H3 (modality shortcut) | shortcut-aware teacher filtering |
| 1-D correctness frontier mixes p-hard / r-hard | H1 | 2-D prompt/trajectory selection |
| OPD ↑ target, ↓ off-target benches | H3 (over-spec) | domain-regularized / reference-anchored OPD |
| **No clean signal in any of the above** | none | **change task or change teacher; do not design a new method anyway** |

---

## 8. Known limitations and TODOs

Honest list, in order of how likely each is to bite us.

### 8.1 First-run blockers (likely)

1. ~~**MMR1 model loading**: `AutoModelForVision2Seq` is too generic.~~ **Fixed in P0**: `_build_model` now tries `Qwen2_5_VLForConditionalGeneration` first and falls back. Still uses the HF chat template + processor, not `qwen_vl_utils.process_vision_info`; if generations look wrong on the first 2-sample debug run, swap to `process_vision_info`.
2. **MathVista-mini / POPE-adversarial local format**: `prep_audit_subset.py` calls `datasets.load_dataset(local_path, split=...)`. If the local dirs aren't HF-formatted, it tries `parquet` as a fallback. If neither works, the user will see a clear traceback and we add a custom loader.
3. **`opd_mmr1_3b_baseline.sh` argument names**: the wrapper calls `python ray_launcher.py --config ... --output-dir ... --run-name ...`. These flag names are unverified — `ray_launcher.py` may use `--config-file` or YAML-only configuration. Plan: read `ray_launcher.py` arg parser before first attempt.
4. **`run_lmmseval.sh` model adapter**: `--model qwen2_5_vl` only works if lmms-eval ships a wrapper for the MMR1 family; otherwise switch to `--model hf-multimodal` with custom `model_args`.

### 8.2 Correctness / methodology gaps

5. ~~**Loose correctness scoring**: `pred == gold or gold in pred`.~~ **Improved in P0**: dispatcher in `mllmopd.diagnostics.scorers` routes POPE/HallusionBench to yes/no, MCQ-letter gold to letter parser, numeric gold to a 1%-relative-tolerance numeric parser. `loose_contains` is still the fallback for unknown gold shapes — JSONL records which scorer fired so loose cells can be filtered out of plots. Headline numbers still come from lmms-eval at Stage E1.
6. **`acc_per_token` is a crude efficiency proxy.** Real efficiency for H3 should be measured on a controlled-difficulty subset, not a benchmark mean.
7. ~~**Full−blank ≠ "visual dependency"**.~~ **Improved in P1**: `summary.json["paired_full_blank"]` now records per-(model, benchmark) 2×2 contingency counts and `image_lift_rate` / `blank_shortcut_rate`, so the "always answers (A)" pathology shows up as `image_lift_rate ≈ 0` with non-zero `both_correct` rather than as a misleadingly small accuracy gap. Probability-mass shift (Δ logp on the predicted token) is the next step and lands when forced-decoding (§5.2) is wired.
8. **Caption-only / image-plus-caption uncertainty**: captions need to be either gold-shipped (e.g., ChartQA descriptions, MathVerse Vision Only metadata), synthesized by a known captioner, or extracted from the question itself. Synthesized captions become a confound. Until a captioning pass exists, both `caption_only_blank` and `image_plus_caption` modes will error out at runtime.

### 8.3 Unimplemented features (planned, blocking H1/H2 figures)

9. **Per-token logprob extraction** (for Fig 2 / H2). Concretely: `run_audit_pass.py` must accept a `--score_completion` mode that forced-decodes a given response and emits the per-position log-prob of the generated token under both `full_image` and `blank_image`. That feeds `visual_dependency.vis_dep_generated` (scalar form). The full-vocab `visual_dependency.kl_per_token` is only invoked on case-study examples behind an explicit flag — never at audit scale.
10. **OPD loss attribution** (for Fig 2 / H2). The OPD reward in Uni-OPD is computed in `miles/Uni_OPD_utils/OPD_reward/` per generated token. To plot loss mass by visual-dep bin, we need to dump the per-token reward alongside the corresponding visual-dep value. Likely a small hook in `OPD_reward/` that writes a sidecar JSON during training.
11. **Perception-hard / reasoning-hard classifier** (for Fig 3 / H1). Needs the oracle-caption pass implemented robustly first.

### 8.4 Things we are deliberately not measuring (yet)

12. **Format brittleness** (`format_success_rate` in `experiment-protocol.md`). Will be added once we see a real overthinking pattern.
13. **Gradient norm / clip fraction / entropy** during OPD training (Level-2 stability). Comes for free from miles' wandb logging; not in our local code yet.
14. **MMMU / DynaMath / WeMath** are deferred until smoke + Level-1 produce a signal. There's no point downloading a 50 GB benchmark before we know what story we're chasing.

### 8.5 Infrastructure debt

15. **No CI / no tests.** Acceptable while the repo is < 5 KLOC of glue; the moment we add a real OPD reward variant, we should add a couple of golden-output tests for `visual_dependency.vis_dep_generated`, `mllm_corruptions.apply_mode`, and the new `diagnostics.scorers` dispatcher.
16. **Submodule pinning for nested submodules.** `Megatron-LM` has its own submodules that are NOT initialized by `git clone --recurse-submodules`. The build script does an explicit `git clone --recursive` in upstream docs; our `setup_train_env.sh` skips that because we use submodules. Will fail at compile of Apex's CUDA extensions unless we add a `cd third_party/Megatron-LM && git submodule update --init --recursive` step.

---

## 9. Glossary of paths

For quick reference.

| Symbol | Mac value | Dev box value |
|---|---|---|
| `$MLLMOPD_ROOT` | `/Users/houshihao/project/code/mllmopd` | `/home/web_server/antispam/project/houshihao/mllmopd` |
| `$BASE_DIR` | `$ROOT/third_party` | same |
| `$MLLMOPD_MODELS_ROOT` | (unset, fine) | `/home/web_server/antispam/project/houshihao/models` |
| `$MLLMOPD_DATASETS_ROOT` | (unset, fine) | `/home/web_server/antispam/project/houshihao/datasets` |
| `$MMR1_7B_RL_CKPT` | (unset) | `$MLLMOPD_DATASETS_ROOT/MMR1-7B-RL` |
| `$MMR1_3B_SFT_CKPT` | (unset) | `$MLLMOPD_DATASETS_ROOT/MMR1-3B-SFT` |
| `$MMR1_7B_SFT_CKPT` | (unset) | `$MLLMOPD_DATASETS_ROOT/MMR1-7B-SFT` ← does not exist yet |
| `$MATHVISTA_PATH` | (unset) | `$MLLMOPD_DATASETS_ROOT/MathVista-mini` |
| `$POPE_PATH` | (unset) | `$MLLMOPD_DATASETS_ROOT/POPE-adversarial` |
| `$MLLMOPD_RUNS` | n/a | `$HOME/mllmopd_runs` (overridable in `.env`) |

GitHub: `https://github.com/shihaohou/mllmopd` (main branch).
