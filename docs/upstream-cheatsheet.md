# Uni-OPD cheatsheet

Distilled from `WenjinHou/Uni-OPD/docs/{build_env.md,build_eval_env.md,miles_modifications.md}`. Keep this current so you don't have to re-grep upstream every time.

## License
Apache-2.0.

## Sibling-repo layout assumed by upstream

```
BASE_DIR/
├── Uni-OPD/         # this project
├── Megatron-LM/     # pinned: 3714d81d418c9f1bca4594fc35f9e8289f652862
├── sglang/          # pinned: 24c91001cf99ba642be791e099d358f4dfe955f5
├── G-OPD/           # only for LLM eval env
└── lmms-eval/       # only for MLLM eval env
```

In **our** layout `BASE_DIR = mllmopd/third_party/`.

## Conda environments

| Env name | Purpose | Python |
|---|---|---|
| `Uni-OPD` | training (Megatron + SGLang + miles) | 3.12.12 |
| `Uni-OPD-LLM-Eval` | text eval (G-OPD pipeline) | 3.10.20 |
| `Uni-OPD-LMMS-Eval` | MLLM eval (lmms-eval) | 3.12.13 |

We mostly care about `Uni-OPD` and `Uni-OPD-LMMS-Eval`. The LLM eval env is optional unless you want to compare against text-side OPD.

## Key training-side files (in `Uni-OPD/miles/`)

| File | What it is |
|---|---|
| `utils/data.py` | YAML-driven dataset ↔ teacher mapping |
| `utils/arguments.py` | CLI flags incl. **Margin Shift**, **Greedy Margin Mask**, **Online Data Balance** |
| `backends/training_utils/loss.py` | concrete implementation of the three features above |
| `Uni_OPD_utils/OPD_reward/` | **token-level OPD reward** — *this is where our perception/visual-dependency-aware variant goes* |
| `Uni_OPD_utils/outcome_reward/` | outcome reward |
| `Uni_OPD_utils/ray_launcher.py` | training entry point |
| `Uni_OPD_utils/scripts/server/run_sglang_server.sh` | teacher server launcher |
| `Uni_OPD_utils/OPD_reward/teacher_server_list.json` | teacher endpoint registry |

## Training script roots (in `Uni-OPD/exps/scripts/OPD/`)

| Directory | Scope |
|---|---|
| `single_teacher/` | Single-teacher distillation (math/code with Qwen3 students) |
| `multi_teacher/` | Multi-teacher joint distillation |
| `strong_to_weak/` | Distilling Qwen3-A3B → smaller |

For MLLM runs we'll add scripts under `Uni-OPD/exps/scripts/OPD/mllm/` (in the fork) and call them from `mllmopd/scripts/train/`.

## Env vars expected by upstream

```bash
BASE_DIR        # path that contains Uni-OPD/, Megatron-LM/, sglang/
CONDA_PATH      # conda root
SGLANG_COMMIT="24c91001cf99ba642be791e099d358f4dfe955f5"
MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"
MILES_DIR="${BASE_DIR}/Uni-OPD/miles"
MEGATRON_DIR="${BASE_DIR}/Megatron-LM"
SGLANG_DIR="${BASE_DIR}/sglang"
LMMS_EVAL_PATH="${BASE_DIR}/lmms-eval"  # for eval env
G_OPD_PATH="${BASE_DIR}/G-OPD"          # for LLM eval env (optional)
```

`scripts/env/setup_train_env.sh` and `scripts/env/setup_lmmseval_env.sh` source `.env` and just exec the upstream commands.

## Mandatory patches

After cloning Megatron-LM and sglang at the pinned commits:

```bash
cd "${SGLANG_DIR}"   && git apply "${MILES_DIR}/docker/patch/v0.5.7/sglang_psp.patch"
cd "${MEGATRON_DIR}" && git apply "${MILES_DIR}/docker/patch/v0.5.7/megatron.patch"
```

Wrapped by `scripts/env/apply_patches.sh`.

## Verification one-liner

```bash
conda activate Uni-OPD
python - <<'PY'
import torch, flash_attn, transformer_engine, sglang, megatron, miles
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "n_gpu", torch.cuda.device_count())
print("flash_attn / TE / sglang / megatron / miles OK")
PY
```

## Datasets referenced upstream

- Text training: `Keven16/G-OPD-Training-Data` (DeepMath + Eurus-2-RL code subset)
- MLLM training: `OpenMMReasoner/OpenMMReasoner-RL-74K`, `HuggingFaceM4/ChartQA`, InfographicVQA
- Teacher example: `SOD-GRPO_teacher-4B` style (Qwen3-4B GRPO-trained)

## Hardware constraints

- `TORCH_CUDA_ARCH_LIST="9.0"` → H100 / H800 only.
- torch 2.9.1 + CUDA 12.8.
- flash-attn 2.7.4.post1.
- apex compiled with `--cpp_ext --cuda_ext`.

If a future GPU doesn't match these, the env script will fail at build time, not silently.
