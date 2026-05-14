# Research plan

## Goal

Find where vanilla on-policy distillation (OPD) breaks in **general multimodal post-training**, with an RL/post-trained MLLM teacher distilling into a smaller student. The method comes after the audit, not before.

## Setting

```
RL / post-trained MLLM teacher (e.g. MMR1-7B-RL)
        ↓  on-policy distillation
smaller MLLM student (e.g. MMR1-3B-SFT)
```

Not: "a generic larger MLLM" as teacher. The OPD literature (SOD, Uni-OPD, …) consistently uses RL-trained teachers; we should stay in the same regime.

## Candidate hypotheses

None are committed. Each is alive only if a Level-1 / Level-2 figure supports it.

### H1 — Perception-Reasoning Frontier (top pick)

> *Existing OPD methods select informative states using outcome correctness, pass@k, entropy, or teacher-student divergence. In MLLMs, outcome difficulty entangles perception and reasoning. One-dimensional frontier selection therefore mixes perception-hard and reasoning-hard samples, producing inefficient or harmful supervision.*

| | |
|--|--|
| **Validating figure** | Pass@k frontier samples decomposed into 4 perception-reasoning quadrants; vanilla OPD performance per quadrant. |
| **Kill criterion** | Perception-hard and reasoning-hard samples have indistinguishable OPD signal and outcomes. |
| **Method direction (if validated)** | 2-D prompt/trajectory selection along a perception–reasoning frontier instead of a 1-D correctness frontier. |

### H2 — Visual-Signal Misallocation

> *OPD provides dense token-level supervision, but in MLLM reasoning, token-level density does not imply visual-signal density. Vanilla OPD allocates much of its loss to low-visual-dependency tokens, leaving perception-critical tokens under-trained.*

| | |
|--|--|
| **Validating figure** | Distribution of OPD loss mass over tokens binned by visual dependency (full-image vs blank-image KL). |
| **Kill criterion** | Loss mass tracks visual dependency closely already, or visual-dependent tokens are <5% of generated tokens. |
| **Method direction (if validated)** | Visual-dependency-weighted OPD reward. Framed as **teacher supervision allocation**, not RL advantage allocation, to avoid PGPO/VPPO overlap. |

### H3 — Post-RL Artifact Transfer (in reserve)

> *RL teachers achieve high accuracy along with overthinking, format overfitting, modality shortcuts, or generalization loss. Vanilla OPD transfers these artifacts together with capability.*

Only alive if **all four** of these hold:
1. RL teacher accuracy > SFT teacher accuracy on target benchmarks.
2. RL teacher has measurable artifact (response length, format rigidity, blank-image accuracy gap, cross-benchmark drop).
3. Vanilla OPD transfers the artifact to the student.
4. The artifact hurts efficiency / generalization / visual grounding — i.e., it is not "more thinking is required for harder problems".

| | |
|--|--|
| **Validating figure** | Per-benchmark accuracy gain vs response-length gain, RL-teacher vs SFT-teacher; same plot for OPD-student vs SFT-student. |
| **Kill criterion** | Length/cross-benchmark drops disappear once you control for problem difficulty. |
| **Method direction (if validated)** | Artifact-aware OPD: reference SFT model as control, suppress signals that are post-RL artifacts rather than capability. |

## Protocol overview

See [`experiment-protocol.md`](experiment-protocol.md) for metric tables and decision tree. Sketch:

- **Level 1 — Audit (no training).** 1K–3K samples mixed across MathVista / MathVision / LogicVista / ChartQA / HallusionBench / CharXiv / MMMU. Record 10 metrics for teacher and student.
- **Level 2 — Small vanilla OPD.** 2K–5K. Compare SFT student vs off-policy KD vs vanilla OPD vs (optional) small GRPO. Record the same metrics plus training stability.
- **Level 3 — Method.** Pick the direction from the decision tree, only if a Level-1 / Level-2 phenomenon survives scrutiny.

## What this plan deliberately is not

- Not a method-first project. We don't write a method section until Level 1 yields a figure.
- Not a "rename SOD's failure mode" project. The MLLM-specific question must come from observed perception/grounding behavior, not analogy.
- Not a benchmark-chasing project. Accuracy is one of ~10 audit metrics, not the goal.
