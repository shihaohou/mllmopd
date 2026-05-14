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

Submodules (`Uni-OPD`, `Megatron-LM` @ `3714d81`, `sglang` @ `24c9100`, `lmms-eval`) are already wired in `.gitmodules`. We're pointing at upstream WenjinHou/Uni-OPD read-only for now — when you need to commit changes to OPD reward / loss, fork it and re-point (see [`docs/workflow.md`](docs/workflow.md)).

### On the 8×H800 devbox (start here)

```bash
# 1. Clone with all submodules
cd /home/web_server/antispam/project/houshihao
git clone --recurse-submodules https://github.com/shihaohou/mllmopd.git
cd mllmopd

# 2. .env defaults already match your server paths; just copy
cp .env.example .env && source .env

# 3. Sanity check what's already on disk
bash scripts/data/check_inventory.sh

# 4. Build the lighter conda env first (lmms-eval; ~30 min)
bash scripts/init/02_devbox_bootstrap.sh
bash scripts/env/setup_lmmseval_env.sh

# 5. Smoke audit — uses ONLY MMR1 checkpoints + MathVista-mini + POPE-adversarial
#    that are already on disk. ~1-2 h on a single H800.
bash scripts/audit/run_smoke.sh
```

The smoke audit is the fastest way to a real datapoint. It doesn't need the full training env, doesn't need Megatron, doesn't need any downloads.

### Later — full Level-1 audit + training

After the smoke audit gives a signal:

```bash
# Build the heavy training env (Megatron + SGLang; 1-2 h)
bash scripts/env/setup_train_env.sh

# Download missing benchmarks for full Level-1 (see docs/server-inventory.md)
# Also download MMR1-7B-SFT (needed for H3 control / Fig 1)

# Then: full audit
bash scripts/audit/run_level1.sh

# Then: training
bash scripts/train/start_teacher_server.sh        # in one tmux pane
bash scripts/train/opd_mmr1_3b_baseline.sh        # in another
bash scripts/eval/run_lmmseval.sh
```

### On Mac

Mac is for editing + figure generation. After pulling JSONL results from the devbox:

```bash
rsync -avh devbox:/home/web_server/antispam/project/houshihao/mllmopd_runs/audit/<run_id>/ ./runs/audit/<run_id>/
pip install -e .
python -m mllmopd.reporting.audit_table --run_dir runs/audit/<run_id>
python -m mllmopd.reporting.figures --summary runs/audit/<run_id>/summary.json --out_dir runs/audit/<run_id>/figures
```

---

## References

- Uni-OPD — https://github.com/WenjinHou/Uni-OPD  (Apache-2.0)
- MMR1 — paired SFT/RL checkpoints at 3B and 7B
- OpenMMReasoner — 874K SFT + 74K RL data, lmms-eval recipe
- Perception-R1 — Qwen2.5-VL-7B + GRPO + visual perception reward
- PEARL — perception-evidence anchored RL with reasoning fidelity gate
- lmms-eval — https://github.com/EvolvingLMMs-Lab/lmms-eval
