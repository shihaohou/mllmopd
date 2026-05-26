# Step 3a — Cached TAM-Boost OPD: Design Lock (v2, 2026-05-26)

> Supersedes `docs/step3a-design-2026-05-25.md` (v1). v1 was Phase-1
> design under the original "TAM-Evidence-Bottleneck OPD" framing. v2
> incorporates Phase-1 preflight v2 results + GPT Phase-2-integration
> verdict (received 2026-05-26 on commit `ac315c7`) and renames the
> method to reflect the actual mechanism that will ship in Phase 2.

## Title change

- **v1**: TAM-Evidence-Bottleneck OPD
- **v2**: **Cached TAM-Boost OPD: Evidence-Gated Distillation for MLLM OPD**

Per GPT Q6 refine: "Evidence-Bottleneck" is misleading until we actually
do compression. Phase-2-as-shipped is **offline TAM precompute + cached
lookup + no-suppress per-token weight at training time**. "Evidence
Bottleneck" survives as a module-level concept (the sample-level `E_x =
top-ρ% TAM patches over C_local positions`) but not as the paper title.

## GPT verdict lock (Q1–Q6)

| # | Question | Verdict | Locked decision |
|---|---|---|---|
| Q1 | Architecture A / B / C? | **Refine: C** | Off-policy KD + offline TAM cache. paper narrative limited to "controlled off-policy validation of TAM weighting"; do NOT claim on-policy TAM-OPD solved |
| Q2 | A0 baseline vs T1_2 head-to-head? | **Refine** | A1 vs A0 is the **main** comparison. T1_2 reports on-policy FullTeacher reference but does NOT decide A1 success |
| Q3 | Precompute scale: 3k vs 15k? | **Refine: progressive** | 50–100 smoke → 3k stratified → 15k. Each gate must pass before next |
| Q4 | C-then-B on-policy validation? | **Refine: contingent** | Only run small-scale B if C is positive. C negative → no B |
| Q5 | q3 39% in-C_local fire narrative? | **Refine: (i)+(ii)** | "Gate fires on spatial co-occurrence, not causal rejection. q3 residual boost budget ≤ 0.2%; A5 oracle quad arm quantifies remove-q3 effect". A5 stays in arm list as diagnostic |
| Q6 | §Method title? | **Refine** | Renamed (see top of doc) |

## Locked gate (no change from v1, NOW with default values aligned)

```
g_t  =  1[c_t ∈ C_local]   ∧   1[ coverage( topK(M_t), E_x ) ≥ τ ]
w_t  =  1 + α · g_t                            (NEVER < 1 — no suppress)
L    =  Σ_t  w_t · L_OPD,t
```

| Symbol | Locked value | Source |
|---|---|---|
| K (token topK) | 0.20 | Phase 1 design |
| ρ (sample bottleneck) | 0.30 | Phase 1 design |
| **τ (coverage threshold)** | **0.70** | **Preflight v2 — overall fire 12.4%, q0/q1 vs q2/q3 2× contrast, cross-ckpt < 0.5pp** |
| α (boost magnitude) | 0.50 | Phase 1 design (mean_w = 1.062 on v2 data) |
| C_local | {content_noun, visual_attribute, proper_noun} | Step 2 effect sizes (Δ ≥ +0.48 nat) |
| `MLLMOPD_USE_TAM_BOOST` env | `1` to enable hook | Mirrors `MLLMOPD_USE_VD_WEIGHTING` pattern |

Default value in `src/mllmopd/training/tam_gate.py:GateConfig` updated to
match this lock (`tau=0.70` in commit pending, was 0.50 in v1).

## Anchor data

| Artifact | Path / commit |
|---|---|
| Step 1a JSONL (training-pool TAM precompute source) | `runs/audit/tam_step1a_classifier_v013_full/tam_step1a.jsonl` (820 rows × 4 ckpts) |
| Preflight v2 JSON | `runs/analysis/tam_step3_preflight_v2_full.json` |
| Preflight v2 records | `runs/analysis/tam_step3_preflight_records_v2_full.jsonl` |
| Classifier version | `regex+spacy_align:v0.1.3` (commit `9c3fb36`) |
| `tam_preproc_version` | `v0.1.3` (Step 1a teacher_pass) |
| `tam_config_hash` | `sha256({K:0.20, ρ:0.30, τ:0.70, α:0.50, C_local:[content_noun,visual_attribute,proper_noun], classifier:v0.1.3})` → computed at precompute time |

## Phase-2 architecture (Path C — locked)

```
┌────────────────────────────────────────────────────────────┐
│ Offline (run once per training pool)                       │
│                                                            │
│ scripts/audit/tam_precompute_train_pool.py                 │
│   ─→ teacher greedy decode every sample                    │
│   ─→ run _classify_tokens_v013 + TAM extraction            │
│   ─→ emit JSONL: per (sample_id, response_hash) →          │
│       {response_ids, response_length, token_categories,    │
│        tam_maps_b64, image_grid_thw, tam_config_hash}      │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼  read-only at training start
┌────────────────────────────────────────────────────────────┐
│ Training time (per-step, post-reward, before loss)         │
│                                                            │
│ src/mllmopd/training/tam_boost_hook.py:                    │
│   post_process_rewards_with_tam_boost(args, samples)       │
│     for each sample:                                       │
│       cache_key = (sample_id, response_hash)               │
│       hit = cache.get(cache_key)                           │
│       if hit and len_match and config_hash_match:          │
│         w_t = tam_gate.compute_weights(...)                │
│       else:                                                │
│         w_t = ones(response_length)  # unconditional       │
│       sample.teacher_tam_weights = torch.tensor(w_t)       │
│     log {hit_rate, len_mismatch, hash_mismatch, fallback}  │
└────────────────────────────────────────────────────────────┘
                          │
                          ▼  via P19/P20/P21 plumbing
┌────────────────────────────────────────────────────────────┐
│ third_party/Uni-OPD                                        │
│   miles/ray/rollout.py (P19)                               │
│     train_data["teacher_tam_weights"] = [sample.x for ...] │
│   miles/backends/training_utils/data.py (P20)              │
│     rollout_data["teacher_tam_weights"] → cuda tensors     │
│   miles/backends/training_utils/loss.py (P21)              │
│     adv = (t_lp - s_lp) * mask                             │
│     if vd: adv = adv * vd_w        (existing P13)          │
│     if tam: adv = adv * tam_w      (new P21)               │
│     ...                                                    │
│ (advantages now scaled; downstream KD loss unchanged)      │
└────────────────────────────────────────────────────────────┘
```

## 5 mandatory patch points (GPT verdict §3)

| # | Patch | Where | Status |
|---|---|---|---|
| 1 | `GateConfig.tau` default 0.50 → 0.70 | `src/mllmopd/training/tam_gate.py:71` | ✅ this commit |
| 2 | loss.py TAM patch as tracked artifact | `scripts/setup/patch_uni_opd.sh` P19/P20/P21 | ✅ this commit |
| 3 | TAM weights **unconditional attach** (cache miss → ones) | `src/mllmopd/training/tam_boost_hook.py` | ✅ this commit |
| 4 | Cache join key = (sample_id, response_hash, response_length, tokenizer_id, tam_config_hash) | precompute + hook | ✅ this commit |
| 5 | TAM weight multiplication site = post sentinel mask, pre margin/whitening; launcher asserts `normalize_advantages=False` | P21 patch + launcher | ✅ this commit |

## Training arms (locked)

| Arm | Description | Gate | Forwards | Purpose |
|---|---|---|---|---|
| A0 | Off-policy KD baseline | `w_t = 1` always (no TAM weights attached) | 1 | Reference — same training mode as A1 minus TAM |
| **A1** | **Cached TAM-Boost (main)** | `g_t = cat ∧ coverage` | 1 | Main §Method claim |
| A2 | category-only | `g_t = cat` (drop coverage) | 1 | Isolate region effect |
| A3 | random-region | `g_t = cat ∧ stable_uniform < target_rate` | 1 | Confirm gate isn't just adjusting boost rate |
| A4 | scrambled-region | `g_t = cat ∧ coverage(topK(scramble(M_t)), E_x)` | 1 | Spatial-structure control (parallel to Step 2 scrambled-TAM) |
| A5 | oracle quad-aware | `g_t · 1[quad ∈ {0,1}]` (2-forward; diagnostic only) | 2 | Upper bound — quantifies "would removing q3 firing help?" |

Acceptance criterion (A1 vs A0):
- POPE adversarial: Δ ≥ +1.0pp
- HallusionBench: Δ ≥ +0.5pp
- MathVista / ChartQA: |Δ| ≤ 1.0pp (no regression)
- Counterfactual `full − scrambled` contrast: Δ ≥ +0.5pp

T1_2 reported as sanity context only — does NOT decide A1 success.

## Execution gates (per Q3 refine)

```
Gate 1 — Plumbing smoke (50–100 samples)
  Pass criteria:
    • A0 hook = identity (tam_cache_hit_rate=0, mean_w=1.0)
    • A1: tam_cache_hit_rate ≈ 1.0, mean_w in [1.05, 1.10],
      tam_len_mismatch_rate = 0, tam_hash_mismatch_rate = 0
    • loss.py logging shows tam_w applied to OPD advantage
  ↓ pass → next; fail → debug

Gate 2 — 3k stratified A0 + A1
  Pass criteria:
    • A1 vs A0 on POPE-adv: Δ ≥ +1.0pp
    • A1 vs A0 on HallusionBench: Δ ≥ +0.5pp
    • MV/CQ regression ≤ 1.0pp
  ↓ pass → next; fail → ablation (A2/A3/A4) before scaling

Gate 3 — 15k full A0 / A1 / A5 oracle
  Pass criteria: same metrics on 15k, plus
    • A5 vs A1 gap quantifies q3 residual cost

Gate 4 — Optional: small-scale B (on-policy via dual teacher)
  Only run if Gate 3 passes
```

## Cache key + metrics

```python
# precompute side
cache_key = (sample_id, response_hash)   # SHA256 of teacher's response_ids
extra_meta = {
    "response_length":    int,
    "tokenizer_id":       "Qwen/Qwen2.5-VL-7B-Instruct@<vocab_hash:16>",
    "tam_config_hash":    "sha256(GateConfig + classifier_version)",
    "image_grid_thw":     [1, H_pre, W_pre],
    "tam_preproc_version": "v0.1.3",
}

# hook side metrics (logged every K=100 steps)
tam_cache_hit_rate:        fraction of samples with cache hit
tam_len_mismatch_rate:     fraction where response_length differs vs cache
tam_hash_mismatch_rate:    fraction where tam_config_hash differs (e.g. K/τ changed since precompute)
tam_unit_fallback_rate:    fraction that got ones (= 1 - hit_rate)
mean_w_per_response:       overall E[w_t]; target [1.05, 1.10]
frac_response_with_empty_E_x: should be < 5%
```

## §Method narrative (q3 framing per Q5)

**Final text** (for paper §3.x):

> TAM-Boost targets local visual-support evidence; q3 visual-rejection
> tokens are orthogonal and protected by the no-suppress design rather
> than solved by the TAM gate. Step 2b shows q3 does not respond to
> local TAM masking, suggesting that visual rejection is not a
> local-evidence problem. We therefore scope TAM-Boost to local support
> evidence and preserve all base OPD correction signal intact.
>
> Some q3 tokens still pass the gate (within-C_local fire rate ≈ 39%
> at the locked τ=0.70), because the gate measures spatial
> co-occurrence between per-token TAM and the sample-level evidence
> bottleneck, not causal sensitivity to local masking. The residual
> boost budget on q3 is small (~0.2% of total boost mass). The
> oracle-quad ablation arm A5 quantifies whether explicitly excluding
> q3 firing would improve A1.

## Out of scope (Step 3a)

- **On-policy TAM-OPD** → future work pending C results (path B small-scale only if C positive)
- **q3 rejection-side fix** → future Step 3b (TAM-Boost + |VD|-Boost combined per Q6 future)
- **Evidence-Bottleneck compression** → future Step 4 (rename will be needed if we eventually do this)
- **Step 2c uniform-sample q3 generalization** → orthogonal audit, independent track

## Code artifacts (Phase 2 deliverables)

| File | Purpose |
|---|---|
| `scripts/setup/patch_uni_opd.sh` (extended) | P19 rollout / P20 data / P21 loss — tracked TAM weights plumbing |
| `src/mllmopd/training/tam_boost_hook.py` | post-process hook: cache lookup + unconditional attach + 4 metrics |
| `scripts/audit/tam_precompute_train_pool.py` | offline TAM cache builder over training pool |
| `scripts/training/launch_tam_boost.sh` | A0/A1 launcher with `MLLMOPD_USE_TAM_BOOST` toggle |
| `src/mllmopd/training/tam_gate.py` | (existing) — gate logic, default `tau=0.70` |

## Related

- [[tam-evidence-bottleneck]] — main project memory (updated 2026-05-26)
- [[teacher-greedy-fp-nondeterm]] — anchor JSONL must be stable; re-runs drift
- `docs/step2b-quad-results-2026-05-25.md` — Step 2b brief (q3 narrative source)
- `docs/gpt-brief-2026-05-26-step3a-phase2-integration.md` — Phase 2 architecture brief
- (GPT verdict in chat session, see project memory for summary)
- `docs/step3a-design-2026-05-25.md` — superseded v1
