# Teacher / student / data / benchmark matrix

## Primary baseline (first OPD run)

| Role | Model | Why |
|---|---|---|
| Teacher | `MMR1-7B-RL` | RL-trained MLLM, public ckpt |
| Reference (control) | `MMR1-7B-SFT` | Isolates "post-RL artifact" vs "capability" |
| Student | `MMR1-3B-SFT` | Same family → tokenizer/processor align |
| Alt student | `Qwen2.5-VL-3B-Instruct`, `Qwen3-VL-2B-Instruct` | Cross-family sanity |

## Fallback teachers (only for specific ablations)

| Teacher | Use case |
|---|---|
| `Perception-R1-7B` | Qwen2.5-VL-7B + GRPO + visual perception reward, trained on filtered Geometry3K |
| `PEARL-7B` / `PEARL-8B` | Perception-evidence-anchored RL with reasoning fidelity gate |

Switch only when an experiment specifically targets perception or visual grounding; otherwise stay on MMR1 for family consistency.

## Training data ladder

| Stage | Dataset | Size | Notes |
|---|---|---|---|
| Quickstart | `MMR1-RL` | 15K (use 2K subset first) | Aligned with MMR1 teacher; smallest viable |
| Scale-up | `OpenMMReasoner-RL` | 74K | Already wired in Uni-OPD |
| Perception focus | `ViRL39K` | 39K | PEARL's training set |
| Geometry focus | `Geometry3K` / `Geo3K` | ~1.4K filtered | Perception-R1's training set |
| Doc/chart generalization | `ChartQA` + `InfographicVQA` | mixed | For cross-domain transfer |

## Evaluation benchmarks (via lmms-eval)

| Benchmark | Capability axis | Used for |
|---|---|---|
| MathVista | Visual math | Level 1 + 2 |
| MathVision | Visual math | Level 1 + 2 |
| MathVerse | Vision-dependent math | Level 1 + 2 |
| WeMath | Multi-step math | Level 2 |
| LogicVista | Visual logic | Level 1 + 2 |
| ChartQA | Chart reasoning | Level 1 + cross-domain |
| HallusionBench | Visual grounding | Level 1 (H1 / H2 critical) |
| CharXiv | Chart reasoning | Level 1 (H1 critical) |
| MMMU / MMMU-Pro | Multi-discipline | Level 2 generalization check |
| DynaMath | Robustness | Level 2 robustness |

## Compute budget (8×H800 single node)

- Teacher SGLang server: 1–2 GPUs for 7B teacher.
- Student trainer (Megatron): remaining 6–7 GPUs.
- For initial 2K subset: a few hours per OPD run is realistic.
- For 15K / 74K runs: plan multi-day.

If you ever need >1 node, that's outside the current setup — re-plan launchers.

## Server inventory snapshot

What's actually on disk right now (the source of truth is [`server-inventory.md`](server-inventory.md)):

- **Present**: MMR1-3B-SFT (student), MMR1-7B-RL (teacher), MMR1-RL (data), ViRL39K, MathVista-mini, POPE-adversarial, VLMBias, tallyqa, plus a stack of Qwen2.5-VL / Qwen3-VL Instruct checkpoints (alt teachers/students).
- **Missing & blocking**: MMR1-7B-SFT (without it, Fig 1 / H3 control is impossible).
- **Missing & non-blocking**: MathVision, MathVerse, LogicVista, ChartQA, HallusionBench, CharXiv, MMMU, WeMath, DynaMath — fine to defer; smoke audit uses only what's present.

