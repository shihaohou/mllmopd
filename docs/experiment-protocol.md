# Experiment protocol

Concrete metrics and decision tree behind [`research-plan.md`](research-plan.md). Keep this file in sync when metric definitions change.

## Level 1 — Audit (no training)

**Inputs.** Three models: `T_RL` (e.g. MMR1-7B-RL), `T_SFT` (e.g. MMR1-7B-SFT), `S` (e.g. MMR1-3B-SFT).
**Data.** 1K–3K prompts mixed across:

| Benchmark | Capability | Why included |
|---|---|---|
| MathVista | Visual math | Standard MLLM reasoning probe |
| MathVision | Visual math | Higher difficulty, fewer leakage concerns |
| MathVerse | Vision-dependent math | Stresses figure interpretation specifically |
| LogicVista | Visual logic | Non-numerical visual reasoning |
| ChartQA | Chart reasoning | Heavy on layout / OCR / structured extraction |
| HallusionBench | Visual grounding | Tests language-prior resistance |
| CharXiv | Chart reasoning | Pairs descriptive + reasoning questions |
| MMMU (subset) | Multi-discipline | Knowledge × image |

**Metrics.** For each (model, benchmark) cell, record:

| # | Metric | Why |
|---|---|---|
| 1 | answer accuracy | baseline capability |
| 2 | response length (tokens) | overthinking / artifact |
| 3 | accuracy per generated token | efficiency |
| 4 | format success rate | brittle template? |
| 5 | full-image accuracy | normal setting |
| 6 | blank-image accuracy | residual answer with no visual input |
| 7 | full−blank gap | visual reliance |
| 8 | oracle-caption accuracy | reasoning-only setting |
| 9 | T-vs-S KL on full image | OPD signal magnitude |
| 10 | T_RL-vs-T_SFT logprob gap | post-RL behavior shift |

For H1: also classify each prompt as `(perception_hard, reasoning_hard)` ∈ {0,1}² using:
- `perception_hard` = student fails at oracle-caption-removed setting (image-only).
- `reasoning_hard` = student fails at oracle-caption (caption-only).

For H2: also record per-token visual dependency on a token-level dataset slice:
- `vis_dep(t) = KL( p_T(·|x, image)[t]  ‖  p_T(·|x, blank)[t] )`
- bin tokens into quantiles; later this lets us check where OPD loss lands.

## Level 2 — Small vanilla OPD

**Variants.**
1. `S` SFT-only (starting point)
2. Off-policy KD: SFT on `T_RL` responses
3. **Vanilla OPD** (the baseline of interest)
4. Optional: small GRPO on `S` (sanity check that OPD beats raw RL at small scale)

**Data.** Start with 2K MMR1-RL prompts; scale to 5K → 15K once stable.

**Same metrics as Level 1**, plus training-side:

| # | Metric | Why |
|---|---|---|
| L2.1 | OPD reward / KL / entropy curves | stability |
| L2.2 | ratio / clip fraction | importance-weighting health |
| L2.3 | grad norm | divergence risk |
| L2.4 | per-step loss attribution by visual dependency bin | H2 signal |
| L2.5 | per-step loss attribution by perception-reasoning quadrant | H1 signal |
| L2.6 | cross-benchmark accuracy (held-out tasks) | negative transfer |

## Level 3 — Decision tree

Pick **at most one** direction based on which phenomenon dominates:

| Observation in Level 1/2 | Hypothesis | Method direction |
|---|---|---|
| Vanilla OPD gains accuracy but length explodes; easy questions overthink; cross-benchmark drops | H3 | length/difficulty-aware OPD; SFT-reference artifact filtering |
| Reasoning-heavy benches improve; perception-heavy benches don't; oracle-caption beats full-image | H1 | perception-reasoning frontier OPD |
| OPD loss mass concentrated on low-visual-dependency tokens; visual-dep tokens under-trained | H2 | visual-dependency-aware OPD reward |
| RL teacher full-image ≈ blank-image but still correct; vanilla OPD inherits this | H3 (modality shortcut variant) | shortcut-aware teacher filtering |
| One-dimensional correctness frontier mixes perception-hard / reasoning-hard | H1 | 2-D prompt/trajectory selection |
| OPD gains on target task but other MLLM benchmarks drop sharply | H3 (overspecialization) | domain-regularized / reference-anchored OPD |

If **no** phenomenon survives, the answer is: change task or change teacher, not "design a new method anyway."

## Figure inventory (what we want to produce)

1. **Fig 1 — Accuracy gain vs response-length gain (T_RL vs T_SFT)** across benchmarks.
2. **Fig 2 — OPD token-loss mass by visual-dependency bin.**
3. **Fig 3 — Error decomposition (perception-hard / reasoning-hard) × OPD signal distribution.**
4. **Fig 4 — Full-image / blank-image / oracle-caption accuracy for T_RL, T_SFT, S, OPD-S.**
5. **Fig 5 — Post-OPD change in accuracy, length, visual-dependency loss share, cross-domain generalization.**

Any one of these figures showing a clean signal is enough to pick a story.
