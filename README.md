# mllmopd

General MLLM On-Policy Distillation research project, built on top of [Uni-OPD](https://github.com/WenjinHou/Uni-OPD).

> **Status (2026-05-14):** scaffolding. No experiments run yet.
> **Methodology:** audit-first. We are running vanilla OPD baselines on real multimodal tasks to find where it breaks, *then* picking a method direction.

---

## Why this exists

Vanilla OPD has been studied extensively in the text domain (SCOPE, TIP, SOD, Prune-OPD, …) but its failure modes in multimodal post-training are not well characterized. Picking a method (e.g., "transfer SOD's state-drift to MLLM") before observing real failures risks fabricating a problem. Instead, this repo:

1. Reproduces a vanilla MLLM OPD baseline with strong **RL teacher → smaller student** (not "bigger generic MLLM → smaller").
2. Audits behavior across MathVista / MathVision / LogicVista / ChartQA / HallusionBench / CharXiv / MMMU.
3. Picks a story only after a Level-1 audit produces a strong figure.

Three candidate hypotheses are tracked in [`docs/research-plan.md`](docs/research-plan.md). None are committed to.

---

## Layout

```
mllmopd/
├── docs/                  research plan, protocol, upstream cheatsheet, workflow
├── configs/               our experiment configs (overlay Uni-OPD's configs/)
├── scripts/
│   ├── init/              one-time bootstrap (fork, submodules, devbox bringup)
│   ├── env/               wraps Uni-OPD's build_env.md / build_eval_env.md
│   ├── data/              dataset + checkpoint download
│   ├── audit/             Level-1 diagnostic runners (no training)
│   ├── train/             OPD training launchers
│   └── eval/              lmms-eval wrappers
├── src/mllmopd/           local Python package: corruptions, visual-dependency probes, figure generators
├── notebooks/             exploratory only
├── third_party/           submodules: Uni-OPD (your fork), Megatron-LM, sglang, lmms-eval
├── runs/                  gitignored; symlink to /scratch/<user>/mllmopd_runs
├── data/                  gitignored; symlink to /scratch/<user>/mllmopd_data
└── models/                gitignored; symlink to $HF_HOME
```

---

## Quickstart

### On Mac (now — scaffold + plan)

```bash
cp .env.example .env       # edit paths as needed
# Read these in order:
#   docs/research-plan.md
#   docs/experiment-protocol.md
#   docs/teacher-student-matrix.md
#   docs/upstream-cheatsheet.md
#   docs/workflow.md
```

When you're ready to set up submodules:

```bash
# 1. Fork Uni-OPD into your account (one-time)
gh repo fork WenjinHou/Uni-OPD --clone=false

# 2. Wire up all four submodules pinned to the right commits
bash scripts/init/01_fork_and_submodules.sh

# 3. Commit + push
git add .gitmodules third_party/
git commit -m "Add upstream submodules (Uni-OPD fork, Megatron-LM, sglang, lmms-eval)"
git push -u origin main
```

### On the 8×H800 devbox (when you're ready to train)

```bash
git clone --recursive <your-mllmopd-repo> mllmopd && cd mllmopd
cp .env.example .env && $EDITOR .env       # set CONDA_PATH, HF_HOME, MLLMOPD_RUNS

bash scripts/init/02_devbox_bootstrap.sh   # makes scratch dirs, sanity-checks CUDA
bash scripts/env/setup_train_env.sh        # conda env: Uni-OPD (Megatron + SGLang)
bash scripts/env/setup_lmmseval_env.sh     # conda env: Uni-OPD-LMMS-Eval

bash scripts/data/download_mmr1.sh         # MMR1-RL-15K + 3B/7B SFT/RL checkpoints
bash scripts/audit/run_level1.sh           # Level-1 audit (no training)
```

After Level-1 produces phenomena worth modeling:

```bash
bash scripts/train/start_teacher_server.sh
bash scripts/train/opd_mmr1_3b_baseline.sh
bash scripts/eval/run_lmmseval.sh
```

---

## References

- Uni-OPD — https://github.com/WenjinHou/Uni-OPD  (Apache-2.0)
- MMR1 — paired SFT/RL checkpoints at 3B and 7B
- OpenMMReasoner — 874K SFT + 74K RL data, lmms-eval recipe
- Perception-R1 — Qwen2.5-VL-7B + GRPO + visual perception reward
- PEARL — perception-evidence anchored RL with reasoning fidelity gate
- lmms-eval — https://github.com/EvolvingLMMs-Lab/lmms-eval
