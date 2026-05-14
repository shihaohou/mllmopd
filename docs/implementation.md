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

Driver: `scripts/audit/run_smoke.sh` (READY) + `configs/audit/audit_v0_smoke.yaml`. Calls `src/mllmopd/diagnostics/run_audit_pass.py` (STUB — see §5.3) for 5 passes:

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

Driver: `scripts/audit/run_level1.sh` (READY but slow) + `configs/audit/audit_v0.yaml`. Six passes × 2000 prompts.

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

- **H1 probe** — perception-hard vs reasoning-hard quadrant classifier. Requires `(image, question) → image-only correctness` and `(caption, question) → text-only correctness`. Two new audit modes.
- **H2 probe** — per-token visual-dependency KL extractor. Forced-decoding the teacher over a fixed completion while toggling image full↔blank; record per-position KL. Needs ~20 lines added to `run_audit_pass.py`.
- **H3 probe** — modality shortcut. Already partially supported via `blank_image`, `irrelevant_image`, `swap_image` in `mllm_corruptions.py`; needs an aggregation script.

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
| `run_smoke.sh` | READY | Builds smoke subset (500 prompts, MathVista + POPE), runs the 5 passes from §3.1, aggregates, prints table. Reads model + dataset paths from env vars. Hard-codes `CUDA_VISIBLE_DEVICES=${SMOKE_GPU:-0}` for single-GPU sequential. |
| `run_level1.sh` | READY-but-slow | Same pattern but 6 passes × 2000 prompts. Includes T_SFT_full and S_oracle_caption. Will fail until MMR1-7B-SFT and oracle captions exist (gracefully — single pass exits non-zero, others continue if `set -e` is removed). |

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
- `oracle_caption` → `(blank, "[Image description: <caption>]\n")` — requires non-empty caption
- `swap_image` → `(other_image, None)`
- `irrelevant_image` → `(noise_with_seed, None)` (per-pixel uniform RGB)

### 5.2 `mllmopd.diagnostics.visual_dependency` — READY (numpy-only, untested end-to-end)

Math utilities for H2 ("loss mass landing on low-visual-dependency tokens").

```python
kl_per_token(logp_a: np.ndarray, logp_b: np.ndarray) -> np.ndarray
    # both (T, V) log-probs; returns (T,) KL(P_a || P_b) per token

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
3. Take per-position softmax → log_softmax of both.
4. Save `(logp_full, logp_blank)` per token.

After that, downstream callers compute `kl_per_token(logp_full, logp_blank)`. Plus OPD loss attribution requires the actual training loss tensor, which lives in miles — not yet wired.

### 5.3 `mllmopd.diagnostics.run_audit_pass` — STUB

Loads a model with `AutoProcessor` + `AutoModelForVision2Seq`, iterates the subset, applies a perception mode, generates with `do_sample=False`, scores with a loose contains/exact match, writes one JSONL line per prompt.

CLI:
```bash
python -m mllmopd.diagnostics.run_audit_pass \
    --subset SUBSET.jsonl \
    --model MMR1/MMR1-7B-RL  # or local path
    --mode full_image        # full_image|blank_image|text_only|oracle_caption|swap_image|irrelevant_image
    --out OUT.jsonl
    [--max-new-tokens 1024] [--limit 0]
```

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
  "is_correct": bool | None,   # NOTE: loose match — see §8
  "finish_reason": (not yet populated)
}
```

Known issues (see §8):
- The generic `AutoModelForVision2Seq` loader may not produce correct outputs for MMR1 / Qwen2.5-VL — Qwen2.5-VL needs `qwen_vl_utils.process_vision_info` + `Qwen2_5_VLForConditionalGeneration`. Will be obvious on first run.
- Correctness scoring is a placeholder (exact / contains). Real per-benchmark scorers live in lmms-eval; we should run lmms-eval for the official numbers and use this only for triage.
- The subset loader assumes `image` is either a PIL object (HF datasets default) or a path. Local benchmark dirs that store raw image files + a JSON answer file will need a custom loader added under `mllmopd.data.audit_subset` (currently empty).

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
}
```

Written to `summary.json` as `{"cells": [<row>, ...]}`.

### 5.5 `mllmopd.reporting.audit_table` — READY

Pretty-prints `summary.json` cells as a fixed-width table. No file output.

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
| Full−blank accuracy gap | `acc(model, full_image, b) - acc(model, blank_image, b)` | H3 modality shortcut (small gap on RL teacher = teacher answers without looking) |
| RL−SFT accuracy gain | `acc(MMR1-7B-RL, full, b) - acc(MMR1-7B-SFT, full, b)` | H3 capability vs artifact (paired with length gain) |
| RL−SFT length gain (rel) | `(tokens_mean_RL - tokens_mean_SFT) / tokens_mean_SFT` | H3 overthinking transfer |
| Acc(blank) / Acc(full) | `acc_blank / acc_full` | Inverse of full−blank gap; high ratio = low visual dependency |

### 7.4 Per-token visual dependency (planned, §5.2 + §5.3 extension)

```
vis_dep[t] = KL( softmax(z_teacher(x_full, y_<t))  ||  softmax(z_teacher(x_blank, y_<t)) )
```

For Fig 2: bin tokens by quantile of `vis_dep`, plot `sum(opd_loss[bin]) / sum(opd_loss)`.

For Fig 3: classify each prompt by `(perception_hard, reasoning_hard)`:
- `perception_hard` = student fails on image+question but succeeds on oracle_caption+question.
- `reasoning_hard` = student fails on oracle_caption+question but succeeds on (oracle_caption+blank_image)+question (= text-only).

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

1. **MMR1 model loading**: `AutoModelForVision2Seq` is too generic for Qwen2.5-VL-derived checkpoints. Will probably need `Qwen2_5_VLForConditionalGeneration` + `qwen_vl_utils.process_vision_info`. Fix: 5-line swap in `run_audit_pass.py::_build_model`.
2. **MathVista-mini / POPE-adversarial local format**: `prep_audit_subset.py` calls `datasets.load_dataset(local_path, split=...)`. If the local dirs aren't HF-formatted, it tries `parquet` as a fallback. If neither works, the user will see a clear traceback and we add a custom loader.
3. **`opd_mmr1_3b_baseline.sh` argument names**: the wrapper calls `python ray_launcher.py --config ... --output-dir ... --run-name ...`. These flag names are unverified — `ray_launcher.py` may use `--config-file` or YAML-only configuration. Plan: read `ray_launcher.py` arg parser before first attempt.
4. **`run_lmmseval.sh` model adapter**: `--model qwen2_5_vl` only works if lmms-eval ships a wrapper for the MMR1 family; otherwise switch to `--model hf-multimodal` with custom `model_args`.

### 8.2 Correctness / methodology gaps

5. **Loose correctness scoring**: `is_correct = (pred == gold) or (gold in pred)` is fine for headline triage but **not** for paper numbers. Always re-run lmms-eval at evaluation time and use this only for cell-level debugging.
6. **`acc_per_token` is a crude efficiency proxy.** Real efficiency for H3 should be measured on a controlled-difficulty subset, not a benchmark mean.
7. **Full−blank ≠ "visual dependency"**. A model that always outputs `(A)` for a multiple-choice question has zero accuracy gap but no visual dependency at all. We need a paired metric — e.g., correctness AND probability mass shift — before claiming "modality shortcut" rigorously.
8. **Oracle-caption uncertainty**: captions need to be either gold-shipped, synthesized by a known captioner, or extracted from the question itself. Synthesized captions become a confound.

### 8.3 Unimplemented features (planned, blocking H1/H2 figures)

9. **Per-token logprob extraction** (for Fig 2 / H2). Concretely: `run_audit_pass.py` must accept a `--score_completion` mode that forced-decodes a given response and emits per-position log-probs under both `full_image` and `blank_image`. Then `visual_dependency.kl_per_token` is callable.
10. **OPD loss attribution** (for Fig 2 / H2). The OPD reward in Uni-OPD is computed in `miles/Uni_OPD_utils/OPD_reward/` per generated token. To plot loss mass by visual-dep bin, we need to dump the per-token reward alongside the corresponding visual-dep value. Likely a small hook in `OPD_reward/` that writes a sidecar JSON during training.
11. **Perception-hard / reasoning-hard classifier** (for Fig 3 / H1). Needs the oracle-caption pass implemented robustly first.

### 8.4 Things we are deliberately not measuring (yet)

12. **Format brittleness** (`format_success_rate` in `experiment-protocol.md`). Will be added once we see a real overthinking pattern.
13. **Gradient norm / clip fraction / entropy** during OPD training (Level-2 stability). Comes for free from miles' wandb logging; not in our local code yet.
14. **MMMU / DynaMath / WeMath** are deferred until smoke + Level-1 produce a signal. There's no point downloading a 50 GB benchmark before we know what story we're chasing.

### 8.5 Infrastructure debt

15. **No CI / no tests.** Acceptable while the repo is < 5 KLOC of glue; the moment we add a real OPD reward variant, we should add a couple of golden-output tests for `visual_dependency.kl_per_token` and `mllm_corruptions.apply_mode`.
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
