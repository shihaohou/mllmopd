# Step 3a — Sparse Visual-Conditioned OPD (v3, 2026-05-26)

> **Supersedes** `docs/step3a-design-2026-05-26.md` (v2). v2 locked Path C
> (offline TAM precompute + off-policy KD), implicitly trading away "OPD"
> as the method substrate. GPT verdict on the `1f27610` pivot brief
> reframed: choose Path (b) — true on-policy OPD + TAM-discovered
> category-only boost. v3 is that reframe.

## What changed vs v2

| Axis | v2 (Path C) | v3 (Path b) |
|---|---|---|
| Method substrate | off-policy KD on cached teacher response | **true on-policy OPD** |
| Training-time TAM | cache lookup by (sample_id, response_hash) | **none** — TAM stays in audit/diagnostic role |
| Gate | `1[c ∈ C_local] ∧ 1[coverage(topK(M_t), E_x) ≥ τ]` | `1[c ∈ C_local]` |
| Title | "Cached TAM-Boost OPD" | **"Sparse Visual-Conditioned OPD"** |
| Cache infrastructure | main method | demoted to diagnostic / off-policy ablation |
| α calibration | static 0.5 (preflight v2) | **runtime-calibrated** to target `E[w_t] ≈ 1.05–1.06` |
| Arms | A0/A1/A2/A3/A4/A5 (off-policy) | **B0/B1/B2** (on-policy), optional B3 |

## Why this is the right pivot (per GPT verdict)

1. **OPD is on-policy by definition**. T1 v1.5b's +23pp finding, the
   FullTeacher/BlankTeacher condition-sensitive narrative, the Mode A vs
   Mode B failure separation — all built on student-rollout on-policy
   training. Switching to off-policy KD for the method tier would sever
   the line that the paper's main story stands on.
2. **TAM is a discovery tool, not a runtime module**. The TAM paper
   (arXiv:2506.23270) designs it as visual explanation for MLLM
   generated tokens *with context interference handling* — it explains
   the model's **own** generations. Caching teacher-greedy TAM and
   applying it to student rollout tokens is structurally
   mismatched. The smoke A1 `fallback=1.000` we observed is the
   downstream consequence, not a bug.
3. **The Step 2 finding still grounds the gate**. Causal masking on
   Step 2 v2 confirmed Δ ≥ +0.48 nat for `C_local = {content_noun,
   visual_attribute, proper_noun}`. We don't need to *deploy* TAM
   masks at training time to leverage that finding — we deploy the
   discovered category set.

## §Method statement (paper-ready, working draft)

> We use TAM as a **causal probe** to identify which token categories
> carry local visual-evidence sensitivity in MLLM OPD, not as a
> runtime training module. Step 2 reveals that the effect is
> concentrated in a sparse set of categories `C_local =
> {content_noun, visual_attribute, proper_noun}`; visual-rejection
> tokens (q3, `vd<0 ∧ adv<0`) are not local-evidence problems and are
> orthogonal to this gate. We deploy this finding via a lightweight
> on-policy **category gate** that boosts OPD correction weight on
> C_local response tokens without suppressing any base correction:
>
> ```
> w_t  =  1 + α · 1[c_t ∈ C_local]           (NEVER < 1)
> L    =  Σ_t w_t · L_OPD,t
> ```
>
> Direct TAM evaluation on student rollouts at training time is
> incompatible with standard OPD serving infrastructure (the sglang
> teacher returns per-token logprobs, not logits). We defer that to
> a follow-up extension and validate the simpler gate first.

## §Method title

**Sparse Visual-Conditioned OPD**

Module name (inside §Method): **C-Local Boost** (or **TAM-Discovered
C-Local Boost** for fuller pedigree).

Avoid:
- "Cached TAM-Boost OPD" — misleading (no cache at train time)
- "Category-Aware OPD" — too generic
- "Visual-Evidence-Guided OPD" — too strong (no evidence computation at train time)

## Gate (locked v3)

```
g_t  =  1[c_t ∈ C_local]
w_t  =  1 + α · g_t                          (NEVER < 1)
L    =  Σ_t w_t · L_OPD,t
```

| Symbol | Locked value | Source |
|---|---|---|
| C_local | `{content_noun, visual_attribute, proper_noun}` | Step 2 causal effect sizes (Δ ≥ +0.48) |
| visual_number | held out | Step 2 Δ ≈ 0 |
| α | **auto-calibrated**: `α = (target_mean_w − 1) / fire_rate` from a no-train fire-rate audit | per Q1 verdict — don't hard-fix 0.5 |
| target_mean_w | **1.05–1.06** | conservative; B1 vs B0 must show signal without inflating loss energy |
| Classifier version | `regex+spacy_align:v0.1.3` (commit `9c3fb36`) | MMR1 `\boxed{answer}` fix |
| Activation | `MLLMOPD_USE_TAM_BOOST=1` + `MLLMOPD_TAM_HOOK_MODE=onpolicy_category` | hook env |

NO spatial coverage check. NO E_x. NO TAM maps at training time. NO cache.

## Anchor data (unchanged)

- Step 2 v2 results — `runs/analysis/tam_step2_v2.json`
- Step 2b results — `runs/analysis/tam_step2b_quad_v2.json`
- Classifier v0.1.3 — Step 1a JSONL `tam_step1a_classifier_v013_full`
- T1_2 baseline numbers (T1-0 mean 0.553, T1_2 mean 0.567 +1.3pp) — `[[t1-result-positive-brief-v2-shipped]]`

## Training arms (v3)

| Arm | Description | Gate | Purpose |
|---|---|---|---|
| **B0** | T1_2-equivalent on-policy OPD baseline | `w_t = 1` always | Reference; same code rerun |
| **B1** | **Sparse Visual-Conditioned OPD (main)** | `g_t = 1[c_t ∈ C_local]` | Main §Method claim |
| **B2** | Rate-matched random token boost (critical control) | `g_t = 1[stable_uniform(t) < fire_rate_B1]` | **Proves B1's benefit isn't just from extra loss energy** |
| B3 (optional) | Non-local category boost (negative control) | `g_t = 1[c_t ∈ {template_token, punctuation, ...}]` | Confirms C_local is special, not just "any subset boost helps" |

Per Q1: B2 is **non-negotiable**. Without it we can't rule out that B1's
benefit is "we doubled effective LR on ~12% of tokens", independent of
which tokens.

## Execution gates (v3)

```
Gate 1 — Pre-train fire-rate audit (no GPU training, ~5 min)
  - Run scripts/audit/tam_step3_b_firerate_audit.py on a sample of
    student rollouts (from T1_2 ckpt OR a quick A0-equivalent dry run)
  - Output: fire_rate, mean_w under α=0.5, recommended α s.t. mean_w in
    target band, per-benchmark and per-category breakdown
  - Lock α before any training
  ↓ pass: α locked, ratio sane

Gate 2 — B0/B1 230-step smoke (60-100 prompts each)
  - B0 = T1_2-equivalent on the same data
  - B1 = B0 + C_local boost with locked α
  - Pass: B1 vs B0 has any positive Δ on full mean OR
          B1 has positive Δ on opd_target / PureV
  - Loss energy ratio: total_weighted_loss / total_base_loss < 1.10
  ↓

Gate 3 — Full B0/B1/B2 (2-3k prompts)
  - Acceptance per GPT verdict:
      B1 > B0 by +0.5 to 1.0 pp full mean
      B1 > B2 on opd_target / PureV / POPE / Hallusion
      ChartQA / MathVista regression ≤ 1.0pp
      avg tokens not significantly increased
      blankness phrase rate not increased
  ↓ pass → paper §Method writeup

Gate 4 (optional) — B3 control + Path (a) small-scale validation
  - B3 confirms C_local is the specific signal
  - 500-1000 prompts with HF teacher sidecar + spatial gate to validate
    upper-bound from full TAM (paper §Future Work)
```

## Implementation deltas (vs v2 plumbing already in repo)

| File | v2 status | v3 action |
|---|---|---|
| `src/mllmopd/training/tam_gate.py` | pure spatial gate logic | **keep as-is** (used by diagnostic path + audit); not on training hot path |
| `src/mllmopd/training/tam_boost_hook.py` | only `cached_spatial` mode | **add `onpolicy_category` mode** (no cache, on-rollout classification) |
| `scripts/setup/patch_uni_opd.sh` P20/P21/P22 | already applied | **keep**; loss-side multiplication unchanged |
| `scripts/audit/tam_step1a.py` | with PRECOMPUTE_ONLY flag | keep flag for diagnostic use; not Step 3a main path |
| `scripts/audit/tam_precompute_train_pool.py` | converter for cache JSONL | **demoted to diagnostic** (kept for future ablation) |
| `scripts/train/launch_cached_tam_boost.sh` | passes USE_TAM_BOOST + CACHE_JSONL | **add MLLMOPD_TAM_HOOK_MODE switch + alpha auto-calibration env** |
| `scripts/audit/tam_step3_b_firerate_audit.py` | n/a | **new** (Gate 1 dependency) |

The good news: the loss-side patches (P20/P21/P22) and the hook
infrastructure DO work — they're upstream of the mode switch, agnostic
to whether weights come from cache lookup or on-rollout classification.
We just add a new mode at the hook layer.

## q3 framing (v3 update)

Same as v2 per Q4 verdict: §Method targets local visual-support;
visual-rejection is orthogonal and protected by no-suppress design.
But the budget number changes — under category-only, q3 boost mass
ratio must be re-measured (fire-rate audit reports it). If it stays
< 2-3%, narrative is the same as v2. If higher, add a token-shape
filter (drop template/boxed/answer-formatting tokens) or lower α.

## Path C and Path A status (after this pivot)

- **Path C (off-policy KD + cached TAM)** — demoted to
  *diagnostic / off-policy ablation only*. Precompute infrastructure
  (`tam_step1a --skip-student --precompute-only` + converter +
  multi-box sharding + `MLLMOPD_TAM_PRECOMPUTE_ONLY=1` patch) is
  preserved. NOT used for main A0/A1 in v3.
- **Path A (on-policy + HF teacher sidecar for spatial gate)** — kept
  as paper §Future Work. Only invest after v3 (b) shows signal.

## Out of scope (v3)

- Cache building on training pool (was Phase 2.2 main goal — now optional)
- Off-policy KD launcher development (was Phase 2.3 — now optional)
- sglang teacher patch for hidden states (Path A, future)
- HF teacher sidecar (Path A, future)
- Spatial coverage gate at training time (Path A, future)

## Related

- `docs/gpt-brief-2026-05-26-step3a-onpolicy-pivot.md` — pivot brief
- `docs/step3a-design-2026-05-26.md` — v2 (superseded)
- `docs/step3a-design-2026-05-25.md` — v1 (superseded)
- `[[tam-evidence-bottleneck]]` — main project memory (now reframed)
- `[[project-hypotheses]]` — "OPD is condition-sensitive" working hypothesis
- `[[t1-result-positive-brief-v2-shipped]]` — T1 v1.5b +23pp anchor
- `[[teacher-greedy-fp-nondeterm]]` — explains why cached responses cannot match on-policy rollouts
