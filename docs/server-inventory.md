# Server inventory (arc-wlf1-ge103-4)

Snapshot of what's already on the 8×H800 dev box as of **2026-05-14**. Keep this in sync when you add/remove assets.

Host: `arc-wlf1-ge103-4`
Models root:   `/home/web_server/antispam/project/houshihao/models`
Datasets root: `/home/web_server/antispam/project/houshihao/datasets`

## Models / checkpoints present

| Path | Type | Use in mllmopd |
|---|---|---|
| `models/Qwen2.5-VL-7B-Instruct` | base MLLM | optional alt teacher / sanity |
| `models/Qwen2.5-VL-32B-Instruct` | base MLLM | strong alt teacher (if 7B teacher underwhelms) |
| `models/Qwen2.5-VL-72B-Instruct` | base MLLM | too big for our 8×H800 OPD run; keep for eval only |
| `models/Qwen3-VL-4B-Instruct` | base MLLM | alt student |
| `models/Qwen3-VL-8B-Instruct` | base MLLM | alt teacher |
| `models/Qwen2.5-{0.5B,1.5B,72B}-Instruct` | text LLM | not used (text-only) |
| `datasets/MMR1-3B-SFT` | **student ckpt** | primary student (H1/H2/H3) |
| `datasets/MMR1-7B-RL`  | **teacher ckpt (RL)** | primary teacher |
| `datasets/MMR1-7B-SFT` | **teacher ckpt (SFT)** | **MISSING — needed for H3 control** |

## Training data present

| Path | Notes |
|---|---|
| `datasets/MMR1-RL` | primary OPD prompts (~15K MLLM RL QA) |
| `datasets/ViRL39K` | PEARL training set; perception-focused alt run |
| `datasets/LLaVA-CoT` | optional SFT cold-start variant |
| `datasets/COCO` / `coco` | image source for shortcut / counterfactual probes |
| `datasets/tallyqa`, `tallyqa_images` | counting probe (useful for H2 visual-dependency analysis) |
| `datasets/e1_mini_v1`, `e1_synth_v1` | (unfamiliar to me — please check what these are before using) |

## Eval benchmarks present

| Path | Status |
|---|---|
| `datasets/MathVista-mini` | ready (smoke audit) |
| `datasets/POPE-adversarial` | ready (smoke audit / H2 grounding) |
| `datasets/VLMBias` | ready (useful for H3 modality-shortcut probe) |

## Eval benchmarks MISSING (needed for full Level-1)

The audit plan in `docs/experiment-protocol.md` calls for these; they are not on disk yet:

- MathVision
- MathVerse
- WeMath
- LogicVista
- ChartQA
- HallusionBench
- CharXiv
- MMMU / MMMU-Pro
- DynaMath

Recommended order to download (highest payoff first):
1. **HallusionBench** + **ChartQA** + **CharXiv** — central to H1 / H2.
2. **MathVision** + **MathVerse** — Level-1 math axis.
3. **LogicVista** + **MMMU** — broader generalization.
4. **DynaMath** + **WeMath** — only when running Level-2 robustness checks.

## Compute layout for primary run

With MMR1 ckpts being 3B/7B and SGLang serving the teacher:

- GPUs 0–1: SGLang teacher server (`MMR1-7B-RL`)
- GPUs 2–7: trainer for `MMR1-3B-SFT` student

This is encoded in `scripts/train/start_teacher_server.sh` (`TEACHER_GPUS="0,1"`) and `scripts/train/opd_mmr1_3b_baseline.sh` (`TRAINER_GPUS="2,3,4,5,6,7"`).

## What to do today (no train) vs. soon

**Today:** run `scripts/audit/run_smoke.sh` — uses only assets above (MMR1-3B-SFT + MMR1-7B-RL on MathVista-mini + POPE-adversarial). Produces 6 JSONLs + a summary table. ~1–2 h on 1 GPU.

**This week, before training:**
1. Download `MMR1-7B-SFT` so Fig 1 (RL vs SFT) is computable.
2. Download HallusionBench + ChartQA + CharXiv so Level-1 covers grounding/chart.
3. Verify Uni-OPD MLLM training schema by running a 100-step smoke train on a 256-prompt slice of MMR1-RL.
