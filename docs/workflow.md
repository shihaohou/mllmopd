# Workflow: Mac ↔ 8×H800 devbox

Two machines, one git repo. Mac is for code + scaffolding + analysis. The devbox is the only place anything actually runs.

```
┌────────────────────────────────┐                  ┌────────────────────────────────┐
│ Mac (this laptop)              │   git push       │ Devbox (8×H800)                │
│  - edit code & configs         │ ───────────────► │  - clone --recursive           │
│  - read upstream submodules    │   git pull       │  - run setup_train_env.sh once │
│  - prep audit JSON subsets     │ ◄─────────────── │  - run training / eval         │
│  - analyze pulled logs         │   rsync results  │  - lmms-eval                   │
│  - generate figures            │                  │  - logs/checkpoints on scratch │
└────────────────────────────────┘                  └────────────────────────────────┘
```

## Loop

1. **Edit on Mac.** Code in `scripts/`, `src/mllmopd/`, `configs/`, `docs/`. For Uni-OPD source changes, edit inside `third_party/Uni-OPD/` (your fork submodule).
2. **Commit + push** in two places when needed:
   - The Uni-OPD fork (submodule) — commits go to your fork.
   - The main `mllmopd` repo — commits include any submodule pointer bumps.
3. **On devbox**: `git pull --recurse-submodules` (or run `scripts/init/devbox_sync.sh`).
4. **Run on devbox** in tmux. Outputs go to `$MLLMOPD_RUNS` (scratch).
5. **Pull logs back to Mac** with `rsync`:
   ```bash
   rsync -avh devbox:/scratch/$USER/mllmopd_runs/<run-id>/eval_results/ \
       ./runs/<run-id>/eval_results/
   ```
6. **Analyze on Mac** with `src/mllmopd/analysis/` and `src/mllmopd/reporting/figures.py`.

## Submodule edits — the tricky part

Because Uni-OPD is a submodule pointing at your fork:

```bash
# On Mac, modify Uni-OPD source:
cd third_party/Uni-OPD
git checkout -b feature/mllm-perception-opd
# ...edit miles/Uni_OPD_utils/OPD_reward/...
git add . && git commit -m "Add perception-weighted OPD reward"
git push -u origin feature/mllm-perception-opd

# Back in mllmopd top-level:
cd ../..
git add third_party/Uni-OPD       # this records the new fork commit pointer
git commit -m "Bump Uni-OPD to feature/mllm-perception-opd"
git push
```

On the devbox, `git pull --recurse-submodules` then pulls both.

If you forget the submodule push, `git status` in the top-level will show `(new commits)` for `third_party/Uni-OPD` — that's the warning.

## Tmux convention on devbox

```bash
tmux new -s mllmopd
# inside:
#   pane 0: trainer
#   pane 1: teacher server
#   pane 2: nvidia-smi -l 2
#   pane 3: htop / logs
```

Detach with `Ctrl+b d`; reattach with `tmux attach -t mllmopd`.

## What lives where

| Thing | Location | Tracked? |
|---|---|---|
| Code, configs, docs | `mllmopd/` | yes (git) |
| Uni-OPD source | `third_party/Uni-OPD/` | as submodule pointer |
| Checkpoints, datasets | `$HF_HOME` (scratch) | no |
| Training outputs | `$MLLMOPD_RUNS` (scratch) | no |
| Eval JSON results | `$MLLMOPD_RUNS/<run>/eval_results/` then `rsync` to `mllmopd/runs/<run>/` on Mac | no (gitignored) |
| Figures for the paper | `runs/<run>/figures/*.pdf` produced on Mac | no (regen from results) |
