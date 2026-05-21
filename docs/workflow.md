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

## Cross-box deployment (teacher and student on different machines)

By default the launcher puts the teacher on GPU 0 and the student on GPUs 1–7 of the same box. When you have two boxes available, moving the teacher to a second machine frees the 8th GPU for the student — useful both for larger batches and for parallel experiments. Both scripts already support this; you just pass the right env vars at the call site. Don't bake box IPs into `.env` — the pair rotates between experiments.

If a shared filesystem (ceph on this cluster) makes `third_party/Uni-OPD/miles/Uni_OPD_utils/OPD_reward/teacher_server_list.json` visible on both boxes, you don't need to sync anything by hand: Box A writes it, Box B reads it.

### Box A — teacher (1 GPU)
```bash
unset -v http_proxy https_proxy no_proxy   # also done inside the script; belt-and-suspenders
STUDENT_IP=<box-B-intranet-ip> \
TEACHER_GPUS=0 \
TEACHER_PORT=30000 \
TEACHER_MEM_FRACTION=0.85 \
FOREGROUND=1 \
bash scripts/train/start_teacher_server.sh
```
`STUDENT_IP` makes the script discover which local NIC actually reaches Box B and writes that routable IP into `teacher_server_list.json`. Without it, `hostname -I` might pick an unroutable overlay NIC (see [E6](common-pitfalls.md#e6-hostname--i-picks-an-unroutable-nic-on-the-h800-train-container)). For a single-box run, omit `STUDENT_IP` and the script defaults to `localhost`.

Sanity-check the JSON and a cross-box curl before launching the student:
```bash
# on Box A
cat third_party/Uni-OPD/miles/Uni_OPD_utils/OPD_reward/teacher_server_list.json
# servers must read http://<box-A-routable-ip>:30000/generate

# on Box B
curl -sf http://<box-A-routable-ip>:30000/get_model_info | head -c 200 ; echo
```

### Box B — student (all 8 GPUs)
```bash
unset -v http_proxy https_proxy no_proxy
TEACHER_HOST=<box-A-routable-ip> \
ACTOR_NUM_GPUS_PER_NODE=8 \
TRAINER_GPUS=0,1,2,3,4,5,6,7 \
OPD_RUN_NAME=t1_v0_T1_2_full_crossbox \
OPD_TEACHER_IMAGE_MODE=full \
bash scripts/train/opd_mmr1_3b_baseline.sh
```
`TEACHER_HOST` only retargets the pre-flight liveness curl; the reward worker itself reads URLs from `teacher_server_list.json`, which Box A already populated. **Pick a unique `OPD_RUN_NAME` per config** (cross-box vs single-box, smoke vs full) to avoid the resume-mode trap (see [E8](common-pitfalls.md#e8-opd_run_name-collision--optimizerparamscheduler-assert)).

### Multi-replica teacher (only if rollout is the bottleneck)
The reward path is `max_new_tokens=0` (logp-only prefill), which is light per call — a single H800 7B-bf16 teacher comfortably handles `RBS × sample_n = 64` prefills per step. Don't add replicas until you've confirmed that rollout-stage time dominates trainer-stage time in the logs. The plumbing supports a comma-separated list of URLs in `teacher_server_list.json` and round-robins; spinning up more sglang instances on different ports/cards and appending their URLs is straightforward.

### Cleanup
```bash
# Box B (student)
ray stop --force

# Box A (teacher)
pkill -f "sglang.launch_server.*30000"
```
Don't use `pkill -f sglang` from the same box as the teacher — too broad, can kill unrelated rollout sglang processes (see memory `feedback_cleanup_safety.md`).

## What lives where

| Thing | Location | Tracked? |
|---|---|---|
| Code, configs, docs | `mllmopd/` | yes (git) |
| Uni-OPD source | `third_party/Uni-OPD/` | as submodule pointer |
| Checkpoints, datasets | `$HF_HOME` (scratch) | no |
| Training outputs | `$MLLMOPD_RUNS` (scratch) | no |
| Eval JSON results | `$MLLMOPD_RUNS/<run>/eval_results/` then `rsync` to `mllmopd/runs/<run>/` on Mac | no (gitignored) |
| Figures for the paper | `runs/<run>/figures/*.pdf` produced on Mac | no (regen from results) |
