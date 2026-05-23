# Handoff: T2-1 result open, awaiting GPT round-3 (2026-05-23)

Self-contained handoff. Read this if you're a fresh session picking up
T2-1 after the cross-box xbox training + eval but BEFORE GPT round-3
review came back.

## TL;DR

T2-1 (PGPO Eq 6+7 token-VD-weighted FullTeacher OPD) trained 230 steps
cross-box. Eval headline:

```
Δ G_full[T2-1] − G_full[T1-2] = −1.67pp
95% bootstrap CI               = [−3.9pp, +0.5pp]   (includes 0)
McNemar canonical opd_target n=133: b=16 c=18 p=0.864
opd_target_recovery: T1-2 = +47.8pp, T2-1 = +37.7pp  (−10pp)
```

Pre-registered decision tree maps this to "reweighting hurts → VPPO
hard mask next" but **TWO findings make that conclusion premature**:

1. **CI includes 0** → not statistically distinguishable from "no signal"
2. **grad_norm halving** → T2-1 step 1 = 12.05 vs T1-2 = 20.34 (~60%).
   PGPO Eq 7 preserves first moment (Σw=N) but not second moment
   (Σ(w·adv²)); with sparsity ~70%, effective gradient norm is halved
   → effective LR is ~5e-7 not 1e-6. **T2-1 might be under-trained,
   not "method failed".**

Brief deliberately NOT written — would be premature.

## What's committed

GitHub: https://github.com/shihaohou/mllmopd @ `dba8566`

| Artifact | Path |
|---|---|
| Pre-registered design | `docs/t2_1_design.md` |
| This handoff | `docs/handoff-2026-05-23-t2-1-result-pending-gpt.md` |
| Eval JSONLs (force-added) | `runs/audit/t2_1_eval_20260523-003257/` |
| Headline compare JSON | `runs/audit/t2_1_eval_20260523-003257/t2_1_compare.json` |
| Summary | `runs/audit/t2_1_eval_20260523-003257/summary.json` |
| VD weighting impl | `src/mllmopd/training/vd_weighting.py` |
| Eval pipeline | `scripts/audit/run_t2_1_eval.sh` + `mllmopd.analysis.t2_1_compare` |
| Xbox launchers | `scripts/train/{start_rollout_servers,opd_mmr1_3b_baseline_xbox}.sh` |
| Xbox patches | `scripts/setup/patch_uni_opd.sh` P14-P18 |
| Xbox bring-up handoff | `docs/handoff-2026-05-22-xbox-bringup.md` |

Trained ckpts (ceph, not in git):
- `${MLLMOPD_RUNS}/t2_1_v0_T2_1_full_vd/ckpt/hf/step_{49,99,149,199,230}`
- Per-step diagnostics (vd_weights / lp_full / lp_blank / response_correct):
  `${MLLMOPD_RUNS}/t2_1_v0_T2_1_full_vd/diagnostics/step_*.jsonl.gz` (~230 files)

## GPT round-3 prompt drafted (sent / pending reply)

Asks specifically:
1. Does PGPO have any second-moment / grad-norm preservation guarantee?
2. Is grad-norm halving expected or a bug in our implementation?
3. Norm-preserving reweighting alternatives in literature?
4. Are our four candidate next steps (A/B/C/D below) ordered right?

Reply expected in Chinese, terse.

## Four candidate next steps (decision pending GPT)

| Option | Cost | Tests what |
|---|---|---|
| **A** | ~5 min | Per-prompt diff on canonical 133 (see WHICH prompts T1-2 wins T2-1 loses → pattern?) |
| **B** | ~30 min | Trajectory eval at step_49/99/149/199/230 — distinguishes "T2-1 always lags T1-2" (under-train) vs "T2-1 forks late" (mechanism) |
| **C** | ~3-4 h | Retrain T2-1' with LR × 1.5 or 2 — direct test of effective-LR-drop hypothesis |
| **D** | ~3-4 h | Skip LR check, jump to VPPO hard top-k mask per pre-registered tree |

My recommendation (subject to GPT): **A → B → C → D** in that order, stop
whenever the answer is clear.

## What does NOT need to be redone

- **Xbox infrastructure** (P14-P18 + start_rollout_servers + xbox launcher):
  works end-to-end, validated. See `docs/handoff-2026-05-22-xbox-bringup.md`.
- **T2-1 VD implementation** (vd_weighting.py + opd_diagnostics_hook + Uni-OPD
  patches): VD signal is healthy in production (sparsity 0.67→0.78,
  max-weight 2.87→2.89, mass-residual 3e-7).
- **Eval pipeline** (run_t2_1_eval.sh + t2_1_compare.py + paired_vision_critical):
  produces the 9 jsonls + summary + compare JSON cleanly.
- **ckpt config.json `text_config.model_type` fix** for the 5 T2-1 ckpts AND
  the 5 T1-2 ckpts has been applied per [[qwen25vl-eval-model-type-text-suffix]].
  If you re-train ANYTHING, the new ckpts will need the same fix.

## Open holes that the next session may want to close

1. **Per-prompt diff on canonical 133** — quick, gives qualitative
   pattern of T2-1 losses. Command in the GPT prompt I sent.
2. **Trajectory eval** — `run_t1_trajectory.sh` already exists; override
   `T1_3_RUN=t2_1_v0_T2_1_full_vd` and pre-create `.bridge_bak` sentinels
   so the script doesn't overwrite our manual config.json fix (Task #22).
3. **Possible P19 patch** — sglang alias `qwen2_5_vl_text` → `qwen2_5_vl`
   in `MRotaryEmbedding.get_rope_index` so future ckpt evals don't need
   the per-ckpt manual fix (Task #21).
4. **`miles/train_async.py` exploration** — for T2-2/T2-3 throughput
   (Task #20). Don't disturb any in-flight T2-1 work.
5. **Tier-2 mechanism falsifier** — still on the table from the T1 brief
   v2 handoff. If T2-1 turns out to be a config issue (LR), Tier-2
   becomes the next milestone for paper. If T2-1 is fundamentally bad,
   Tier-2 is the only path to a defensible mechanism claim.

## What to read first (next session)

1. This handoff (you're reading it).
2. `docs/t2_1_design.md` (pre-registered design + decision tree).
3. GPT round-3 reply (when it comes).
4. `[[t2-1-result-status-2026-05-23]]` memory entry for the open status.
5. `runs/audit/t2_1_eval_20260523-003257/t2_1_compare.json` for the
   exact numbers (don't trust the rounded versions in this handoff).
