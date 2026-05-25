# Step 3a — TAM-Boost OPD Training Design (2026-05-25)

## Status

| Gate | State |
|---|---|
| GPT verdict on Step 2b brief | ✅ Greenlight Step 3a code, with scoped q3 narrative |
| GPT static-review patches (Issues 1–4) | ✅ landed in `35dee93` |
| Step 2 runner patches (Issues 5/6inline/9 + 2b CLI) | ✅ landed in `5439845` |
| Step 2c (uniform-sample q3 generalization) | ⏭ NOT a prerequisite — orthogonal, future audit |
| Step 3a design (this doc) | ✏️ now |
| Step 3a code skeleton | ⏳ pending user sign-off on this doc |

## TL;DR

Train **Base OPD + TAM-Boost no-suppress reweighting**. Per-token weight
`w_t = 1 + α · g_t` where the gate `g_t` is the conjunction of (i) a
local-evidence token category and (ii) the token's TAM top-K patches
overlapping a sample-level evidence bottleneck. The gate uses **ONLY
teacher-side, one-forward signals** — no `vd`, no `quad`, no blank-image
forward. An oracle quad-aware arm (two-forward, `1[quad∈{0,1}]`) is
included as a diagnostic upper bound, NOT the main method.

§Method **does NOT claim to fix T2-1's `vd<0 ∧ adv<0` (quad==3)
rejection-side failure bucket**. Step 2b showed q3 tokens are insensitive
to any local 20% spatial mask on the Step-2 target subset; rejection-side
fix is a complementary line (future Step 3b — combined TAM-Boost +
\|VD\|-Boost, per GPT Q6).

## Inputs / dependencies

| Item | Source / file |
|---|---|
| Teacher | MMR1-7B-RL (same as T1 / Tier-2) |
| Student init | T1_2 (best Base-OPD ckpt, mean 0.553) |
| Training pool | MMR1-RL ~15k (same as T1) — Tier-2 subset 2–5k for first arm |
| TAM extractor | `scripts/audit/_tam_core.py` (vendored ICCV 2025 slim) |
| Token category classifier | spaCy `en_core_web_sm` + answer/template regex (Step 1a `_classify_tokens_v012`) |
| Eager attention impl | `MLLMOPD_ATTN_IMPL=eager` (see [[eager-attn-for-attention-output]]) — also required if we ever log attention baseline |

## Frozen §Method gate (one-forward, deployable)

```
g_t  =  1[c_t ∈ C_local]   ∧   1[ coverage( topK(M_t), E_x ) ≥ τ ]
w_t  =  1 + α · g_t                            (NEVER < 1 — no suppress)
L    =  Σ_t  w_t · L_OPD,t
```

| Symbol | Definition |
|---|---|
| `c_t` | per-token category from spaCy + regex (Step 1a v0.1.2 classification) |
| `C_local` | `{content_noun, visual_attribute, proper_noun}` — categories where Step 2 showed top-TAM > random (Δ ≥ +0.48 nat, p ≤ 0.01) |
| `M_t` | per-token TAM map, shape `(H, W)` post-2×2-merge, from teacher's existing forward |
| `topK(M_t)` | top-K% TAM patches of token `t`, K = 20% |
| `E_x` | sample-level evidence bottleneck (see § E_x estimation) — top-ρ% patches, ρ = 30% |
| `coverage(A, B)` | `|A ∩ B| / |A|` — fraction of token's top-K patches inside the sample bottleneck |
| `τ` | coverage threshold, default `0.5` |
| `α` | boost magnitude, default `0.5` (target average `w_t ≈ 1.1`, see § α calibration) |

### Why no `quad` in main gate

`quad ∈ {0,1,2,3}` requires `vd = lp_full − lp_blank` which requires a
**blank-image forward**. Including `quad` would double the per-step
forward cost and destroy the "one-forward TAM-Boost" headline. Per GPT
verdict Q4, oracle quad-aware gating is a diagnostic ablation (arm A5),
not the deployable method.

## E_x estimation

```
E_x  =  top-ρ% patches by   mean_{t ∈ C_local positions} TAM_t
```

Aggregated over response positions belonging to `C_local`. Rationale:
sample-local "what does the image actually contribute" — distinct from
per-token `M_t` which is one specific noun's spatial lookup. The
intersection `topK(M_t) ∩ E_x` then asks: does this token's evidence
location coincide with the image-evidence concentration that the whole
response is relying on?

Cost: `O(R · H · W)` aggregation over already-computed `M_t` maps — no
extra forwards.

Edge cases:
- If `|C_local positions| == 0` in a response: `E_x = ∅`, gate never
  fires (gate-fire-rate = 0% for that response, logged).
- If H × W is small (low-res image): clamp `|E_x| ≥ 1` patch.

## Default parameters

```
K     = 0.20    # token_topK
ρ     = 0.30    # sample_bottleneck_rho
τ     = 0.50    # coverage threshold
α     = 0.50    # boost magnitude (subject to § α calibration below)
```

`visual_number` is held out of `C_local`: Step 2 Δ ≈ +0.003 nat (n=23) —
indistinguishable from zero. Possible reasons (OCR-like recognition vs.
spatial localization, or just small n). Keep as ablation slot for future
investigation; do not boost in main arm.

### α calibration (pre-flight from Step 1a JSONL)

Before training, run an offline pass on Step 1a's 820 rows × 4 ckpts:

1. For each token, compute `c_t`, `M_t`, `E_x`, and `g_t` under the gate.
2. Report `frac_tokens_with_g_t=1` per ckpt and per category.
3. Pick α such that `E[w_t] ≈ 1.1` — i.e. mean effective boost ≈ 10%.

Expected from Step 2 data: C_local covers ~40–55% of tokens,
coverage ≥ τ probably holds for ~25–40% of C_local tokens → gate-fire
rate ≈ 10–20%. With α=0.5, mean `w_t` ≈ 1.05–1.10. If pre-flight
disagrees, adjust α before launch.

## Training arms

| Arm | Description | Gate | Forwards | Purpose |
|---|---|---|---|---|
| A0 | Base OPD baseline | `w_t = 1` always | 1 | T1_2-style reproduction; clean comparator |
| **A1** | **TAM-Boost (main)** | `g_t = cat ∧ coverage` | **1** | Main §Method claim |
| A2 | category-only | `g_t = cat` (drop coverage) | 1 | Isolate region effect — does the spatial gate matter? |
| A3 | random-region | `g_t = cat ∧ 1[U(0,1)<E[coverage]]` | 1 | Confirm gate isn't just adjusting the boost rate |
| A4 | scrambled-region | `g_t = cat ∧ coverage(topK(scrambled_M_t), E_x)` | 1 | Spatial-structure control (parallel to Step 2 scrambled-TAM) |
| A5 | oracle quad-aware | `g_t · 1[quad∈{0,1}]` | **2** (blank forward) | Diagnostic upper bound — labeled oracle, NOT method |

Execution order:

```
A0 → A1                     # bare minimum: does TAM-Boost beat baseline?
A1 → {A2, A3, A4}           # in parallel if budget allows: gate decomposition
A1 → A5                     # final oracle ablation
```

Stop early if A1 doesn't beat A0 → go back to gate-design questions
(coverage form, α, C_local membership) before running A2-A5.

## Logging (every K=100 steps)

| Metric | Why |
|---|---|
| `gate_fire_rate_by_category` (10-cat) | Sanity that gate fires where expected; flag if `template_token` ever ≠ 0 |
| `gate_fire_rate_pooled` | Should match pre-flight prediction (10–20%) |
| `posthoc_gate_fire_rate_by_quad` | NOT computed every step (would need blank forward). Periodic offline ckpt audit every 500 steps on 50-sample slice. Should over-index q0/q1 (per Step 2b) — sanity check |
| `mean_w_per_response` | Should hover ≈ 1.05–1.10; auto-clamp α if > 1.3 (see fallbacks) |
| `total_weighted_loss / total_base_loss` | Ratio of weighted to unweighted OPD loss; should be in [1.0, 1.5]; alert if > 1.5 |
| `tam_map_entropy_p50` | Sanity that TAM maps don't degenerate to uniform during training |
| `frac_response_with_empty_E_x` | Should be < 5% — if higher, C_local too narrow or response too short |

## Eval plan

Re-use the **counterfactual 4-variant** protocol from Tier-2a
(`docs/gpt-review-2026-05-24-counterfactual-6variants.md` and
`docs/gpt-brief-2026-05-24-tier2a-results.md` — the 6-variant brief was
narrowed to 4 actually-cheap ones):

| Variant | Image |
|---|---|
| `full` | original |
| `blank` | gray-128 same shape |
| `patch_perm` | random patch permutation |
| `scrambled` | pixel-shuffled |

Benchmarks: POPE (adversarial + popular), MathVista (mini), HallusionBench
(HB), ChartQA — same as T1 / Tier-2.

Acceptance criteria (A1 vs A0):
- POPE adversarial: Δ ≥ +1.0pp (TAM-Boost should help most on hallucinations
  where local visual evidence matters)
- HallusionBench: Δ ≥ +0.5pp (similar mechanism)
- MathVista: |Δ| ≤ 1.0pp (no regression — math is not local-visual-evidence
  dependent in the same way)
- ChartQA: |Δ| ≤ 1.0pp (similar)
- Counterfactual contrast `full − scrambled`: Δ ≥ +0.5pp (TAM-Boost should
  enlarge the image-conditioning signal we measured in T1)

If A1 passes on POPE + HB without regression on MV / CQ → §Method tier
established. Otherwise revisit gate before running A2-A5.

## Failure-mode fallbacks (in trainer, not post-hoc)

```python
if running_mean(total_weighted_loss / base_loss, window=100) > 1.5:
    alpha = alpha / 2.0
    log_event("auto_alpha_halved", new_alpha=alpha)
```

```python
if running_mean(gate_fire_rate, window=100) > 0.5:
    pause_and_alert("gate_fire_rate > 50% — τ too loose, recalibrate")
```

```python
if tam_map_entropy_p50 > 0.95 * log(H * W):
    pause_and_alert("TAM maps collapsed to uniform — extractor degraded")
```

## Open implementation questions (settle BEFORE code)

1. **Where to hook into Uni-OPD's OPD loss?**
   - Need to grep `third_party/Uni-OPD` training loop for the per-token
     loss aggregation point. Current Tier-2 patches hooked into
     `compute_advantages` for signed boost; TAM-Boost needs an earlier
     point (post-OPD-loss-per-token, pre-reduction).
   - Likely target: wherever `lp_student - lp_teacher` per-token is
     reduced to scalar — multiply each per-token term by `w_t` before sum.

2. **E_x caching strategy.**
   - Per-rollout cache: compute `E_x` once per (sample, response), store
     alongside `M_t` for response. Adds memory `O(B × H × W)` floats.
   - Per-step recompute from cached `M_t`: cheaper memory, ~µs of FLOPS
     per response. Probably right choice given map sizes 14×14 to 32×32.
   - Decision: cache `M_t` (already need it for `topK` per token);
     recompute `E_x` per step from `C_local` mean.

3. **Coverage threshold τ pre-flight.**
   - Should we hard-code τ=0.5 or pick from offline calibration on Step 1a
     maps? Likely calibrate to gate-fire ≈ 15% target on Step 1a corpus,
     same time as α calibration above. Report chosen τ in design lock.

4. **α auto-tune vs fixed.**
   - Auto-tune from `target_avg_boost = E[w_t]` (e.g. 1.1) keeps the
     average reweight strength constant across data shifts. Fixed α=0.5
     is simpler. **Decision: start fixed α=0.5 from pre-flight, add
     auto-tune in A2/A3/A4 if A1 results suggest sensitivity.**

5. **Posthoc quad audit frequency.**
   - Every 500 steps on 50-sample slice with blank-image forward — costs
     50 × 1 extra forward per audit. Manageable but adds dependency on
     blank-image inference path during training. Decision: implement as
     optional `--audit-quad-every N` flag, default off; turn on for A1
     paper run.

## Out of scope (Step 3a)

- **q3 rejection-side fix** → future Step 3b (TAM-Boost + \|VD\|-Boost
  combined, per GPT Q6)
- **Step 2c uniform-sample q3 generalization** → orthogonal audit, can
  run in parallel to A1 training on a separate box
- **Evidence-Bottleneck compression (Step 4)** → only if Step 3a A1 lands
- **Token-VD-style reweighting** → distinct from this line (see
  [[project-related-work]] PGPO/VPPO/PAPO comparison — those target
  RLVR sparse→dense; we target OPD dense→correctly-dense)

## Risks I'm tracking

1. **Gate fires too rarely**: if pre-flight shows <5% gate-fire, the
   effective signal-to-noise will be poor. Mitigation: relax τ or
   broaden `C_local` (e.g. include `visual_number` if pre-flight shows
   non-trivial activation).
2. **TAM maps collapse during training**: post-RL teachers sometimes
   develop very peaked or very diffuse attention; if TAM maps lose
   spatial structure, the gate becomes near-random. Mitigation: the
   `tam_map_entropy_p50` log + Step 1a pre-flight gives early warning.
3. **C_local mis-classification on long-CoT**: spaCy has known limits
   on MMR1's verbose output (per Step 0 caveat). Mitigation: log
   `gate_fire_rate_by_category` so we can detect if `template_token`
   ever fires (a contradiction).
4. **Pre-existing OPD baseline drift**: T1_2 is the best-known
   checkpoint at mean 0.553; if Base OPD reproduction in A0 doesn't
   match, the whole comparison is contaminated. Mitigation: A0 must
   reproduce T1_2 within ±0.5pp before A1 launches.

## Related

- [[tam-evidence-bottleneck]] — main project memory
- [[t2-1-result-status-2026-05-25]] — T2-2 abandonment, motivates this line
- [[project-related-work]] — PGPO/VPPO/PAPO differentiation
- `docs/step2b-quad-results-2026-05-25.md` — Step 2b brief (Q1–Q6 GPT verdict)
- `docs/step1a-step2-results-2026-05-25.md` — narrative flip (Pearson vs causal)
- `docs/gpt-brief-2026-05-24-tier2a-results.md` — counterfactual eval protocol reuse

## Decisions log (this commit)

| Decision | Value | Source |
|---|---|---|
| Main gate uses `quad`? | NO | GPT verdict Q4 (one-forward claim) |
| Suppress (`w_t < 1`)? | NO | T2-2 lesson (boost-only, gloo barrier aside) |
| `C_local` membership | `{content_noun, visual_attribute, proper_noun}` | Step 2 effect sizes |
| `visual_number` boost? | NO (held out) | Step 2 Δ ≈ 0 |
| Oracle quad arm runs? | YES (A5, last in order, labeled as diagnostic) | GPT verdict Q4 |
| First training arm | A1 vs A0 (POPE+HB primary acceptance) | Minimum to greenlight method |
| q3 fix in this Step? | NO | GPT verdict Q6 → Step 3b future |

## Next concrete action

After user signs off on this doc:

1. Implement gate module: `src/mllmopd/training/tam_gate.py`
   - `compute_gate(M_t, c_t, E_x, K, ρ, τ) → g_t ∈ {0, 1}`
   - `compute_E_x(M_list, c_list, C_local, ρ) → E_x patches`
   - Vectorized over batch / response positions
2. Pre-flight calibration script: `scripts/audit/tam_step3_preflight.py`
   - Reads Step 1a JSONL TAM maps (or re-runs if v0.1.4 needed)
   - Reports gate-fire rate, coverage distribution, picks α, τ
3. OPD loss wrapper: `src/mllmopd/training/tam_boost_loss.py`
   - Locate Uni-OPD per-token loss reduction point
   - Insert `w_t · L_t` before sum
4. Logging hooks for all metrics listed in § Logging
5. A0 / A1 launcher scripts mirroring Tier-2 pattern
