# Handoff: T2-2 boost-only launched, Tier-2a brief @ GPT round-5 (2026-05-24)

Self-contained handoff. Read this if you're a fresh session picking up
T2 method-tier work AND/OR the Tier-2a mechanism-tier brief sent for
GPT review. Both threads are live simultaneously.

## TL;DR

Today's session took T2-1 from "ambiguous negative result" through
**three rounds of GPT review** (rounds 3/4/5) and produced:

1. **A0 energy audit** (no training) — refuted "LR confound" explanation,
   confirmed **signed-VD-proxy mis-target**: `cond_supp = 0.997` means
   99.7% of visual-rejection correction mass was routed into PGPO's
   suppress branch.
2. **6-variant counterfactual sweep** (no training) — established that
   `{abs, RMS-preserve, max-clip}` family hits a Pareto wall between
   energy stability and signal targeting. None passes 4/4 criteria.
3. **GPT round-5 reframe**: stop searching in PGPO mass-redistribution
   family. Use **boost-only** (`w_t = clamp(1 + α·rank(|vd|), 1, max_w)`)
   on top of base OPD correction.
4. **T2-2 implemented + α-sweep counterfactual**: `boost_only_a05`
   (α=0.5) is the **only** variant passing 4/4 strict pass criteria.
5. **T2-2 α=0.5 training launching now** (task #21, after launcher
   silent-exit bugfix).
6. **Tier-2a brief sent to GPT for round-5 review** — discovered the
   user ran Tier-2a 4-arm experiment in a parallel conversation
   (commit `11bef7d`). I added a cross-thread context bridge + Q7
   tying T2-1's `cond_supp=0.997` finding to Tier-2a's "Mode A
   universal degradation".

T2-2 expected outcome (no training cost spent yet to know):
counterfactual predicts rho_l2 ≈ 1.41, frac_supp=0, mean_w on visual-
rejection quadrant = 1.33. If training is stable AND Δ ≥ +0.5pp vs
T1-2, boost-only is the right method-tier design and T2-1 negative
result becomes a "structural insight, fixed by T2-2" paper section.

## Two live threads — don't confuse them

**Thread A — Method tier (T2)** — *this conversation's main work*
- T2-1 (PGPO signed): trained ✗ negative result; A0 audit explained why
- T2-2 (boost-only |VD|): designed + offline-validated; **training now**
- Owns `docs/t2_2_design.md`, `docs/gpt-review-2026-05-24-{a0,counterfactual}*.md`

**Thread B — Mechanism tier (Tier-2)** — *user ran in parallel conversation*
- Tier-2a (4-arm off-policy KD): trained + analyzed, brief shipped
- Refines brief v2 §Mechanism into Mode A (universal) + Mode B (on-policy cliff)
- Owns `docs/gpt-brief-2026-05-24-tier2a-results.md`

The two threads converge on **the same underlying claim**: distillation
gradient mis-handles visually-driven teacher correction. T2-1's
`cond_supp=0.997` (token axis) and Tier-2a's Mode A (trajectory axis)
may be one phenomenon — Q7 in Tier-2a brief asks GPT to unify or
separate them.

## What's committed this session (commits b2d72c5 → 50f37c6, plus 11bef7d from parallel session)

```
50f37c6 docs: Tier-2a brief — add cross-thread context bridge + Q7
0a05c94 xbox launcher: don't die silently when ip command is absent
11bef7d analysis+docs+runs: Tier-2a 4-arm trajectory analyzer + brief v3  ← parallel session
c42ec3b audit: boost_only α=0.5/0.7/1.0 sweep variants for T2-2 calibration
35a8a49 T2-2: boost-only |vd| weighting (per GPT round-5 reframe)
b00389c docs: GPT round-5 review prompt — 6-variant counterfactual Pareto
79a3d69 audit: 2 more variants — abs_max_clip_renorm + abs_rms_preserve_wide
8168bc7 audit: 4-variant counterfactual (signed/abs/abs_rms_preserve/abs_max_clip)
06716ab audit: A0+ quadrant table + Abs offline counterfactual (GPT round-4 asks)
a0bf7bc docs: GPT round-4 review prompt — T2-1 A0 energy audit result
ab55463 audit: P19 dump old_log_probs sidecar for energy audit (A0 unblock)
10cb911 audit: T2-1 per-prompt diff on canonical 133 (LOSE clustered on math + POPE)
b2d72c5 audit: T2-1 energy audit (rho_l2 / corr / suppressed correction mass)
```

GitHub: https://github.com/shihaohou/mllmopd @ `50f37c6`

## Key result tables (memorize these)

### A0 energy audit (T2-1 pilot retrain, 576 samples / 935814 tokens)

```
rho_l2 = 0.973                     → LR confound REFUTED
corr(w, |adv|) = -0.101            → weakly anti-correlated
frac_supp_neg_vd_neg_adv = 0.442   → signed-proxy suppresses substantial mass
conditional_supp_visual_rejection = 0.997  → 99.7% of vis-reject mass in suppress
vis_reject_correction quadrant: 20.5% of tokens, 44.3% of |adv|, mean_w=0.936
```

### 9-variant counterfactual (T2-2 calibration)

```
variant                  rho_l2    corr  frac_supp  cond_supp  visR_mw  passes (frac<0.05/mw≥1/corr≥-.05/rho<1.5)
boost_only_a05            1.413  +0.396      0.000      0.000    1.328   ✓/✓/✓/✓   ← TRAINING NOW
boost_only_a07            1.580  +0.396      0.000      0.000    1.460   ✓/✓/✓/✗
boost_only                1.831  +0.396      0.000      0.000    1.657   ✓/✓/✓/✗
signed                    0.973  -0.101      0.442      0.997    0.936   ✗/✗/✗/✓   ← T2-1 production
abs                      11.981  +0.308      0.124      0.279    1.901   ✗/✓/✓/✗
abs_rms_preserve          5.990  +0.308      0.204      0.459    0.950   ✗/✗/✓/✗
abs_rms_preserve_wide     1.317  +0.326      0.368      0.829    0.219   ✗/✗/✓/✓   ← scalar destroys targeting
abs_max_clip              1.611  +0.471      0.124      0.279    0.824   ✗/✗/✓/✗
abs_max_clip_renorm       3.802  +0.428      0.072      0.163    1.845   ✗/✓/✓/✗   ← best signal in PGPO family
```

### Tier-2a 4-arm trajectory (parallel session, brief @ 11bef7d)

```
full_image @ step 230 (Δpp vs base 0.437):
  T1-2  on-policy + Full   +0.8pp  (benign)
  T1-3  on-policy + Blank  -13.2pp (CLIFF on opd_target)
  T2A   off-policy + Full  +0.8pp  (benign)
  T2A   off-policy + Blank -14.8pp (smooth decline; broader damage)

opd_target cliff diagnostic (99→149 vs 149→199):
  T1-3   -7.5pp → -12.0pp   *** CLIFF ***
  T2A    -9.8pp →  +3.8pp   smooth (recovers)

HallusionBench opd_target @ step 230:
  T1-3   -20.0pp  (on-policy attractor amplifies hallucination)
  T2A     -6.7pp  (off-policy doesn't amplify; 3× less harmful here)
```

## State as of handoff

### Active

| Task | State | What's needed |
|---|---|---|
| **T2-2 α=0.5 training** | **launching** | User re-ran after launcher fix; needs to land step_230 ckpt (~3-4h on H800) |
| **Tier-2a GPT round-5 review** | **pending response** | User sent brief + Q1-Q7 wrapper to GPT (see "Next session" §) |

### Done this session

- A0 energy audit (rho_l2/corr/frac_supp/quadrant/cond_supp)
- A0c sidecar dump infrastructure (P19 patch + opd_adv_dump module)
- Per-prompt diff on canonical 133 (force-added json)
- Three rounds of GPT review (round-3 already done; round-4 + round-5 prompts shipped)
- 6→9-variant offline counterfactual sweep
- T2-2 boost-only implementation (vd_weighting.py + dispatch + launcher + design doc)
- xbox launcher silent-exit fix (`|| true` around NCCL_SOCKET_IFNAME auto-detect)
- Tier-2a brief edits (cross-thread context + Q7)
- Memory updates: [[t2-1-result-status-2026-05-24]], [[h800-network-gotchas]]

### Pending (next session work)

1. **Verify T2-2 α=0.5 launched cleanly** — look for `[opd_diag]` log
   line with `boost_only` mode and step-1 grad_norm ≈ 24 (1.18×
   T1-2's 20.34). If anything off, see §"T2-2 monitor checklist".
2. **Wait for T2-2 step_230** (3-4h). 5 ckpts (49/99/149/199/230).
3. **Eval T2-2** via the established 3-arm pipeline
   (`scripts/audit/run_t2_1_eval.sh` adapted for T2-2 paths).
   Headline: G_full, opd_target_recovery, McNemar on canonical 133.
   Per-bench: does math+POPE LOSE pattern (T2-1) reverse?
4. **Apply GPT round-5 verdict on Tier-2a Q7** to paper §Mechanism
   organization decision (unify T2-1 mis-target with Mode A vs keep
   them separate).
5. **If T2-2 positive (Δ ≥ +0.5pp vs T1-2 on opd_target)**:
   - Consider α=1.0 follow-up for stronger signal
   - Multi-seed for both T2-2 + Tier-2a-blank
   - Update T2-2 brief, propose paper structure
6. **If T2-2 negative**:
   - Pivot fully to Tier-2 framing (paper around 2-mode mechanism)
   - T2 thread becomes "what doesn't work and why" section
   - The 6-variant counterfactual analysis still publishes as
     design-space exploration

## T2-2 monitor checklist (during/after launch)

1. **Pre-flight passes**: previously dies silently on `ip route get`
   absence. Fix in `0a05c94`. Re-launching should now warn and
   continue. Look for `>>> WARNING: could not auto-resolve
   NCCL_SOCKET_IFNAME` followed by available NICs.
2. **Ray actors come up + sglang handshake**: usual ~30-60s for
   actor pool + bridge to 7 rollout engines on Box 1.
3. **`[opd_diag]` log line — boost_only mode confirmed**: look for
   ```
   [opd_diag] step 0: wrote N rows ... VD weights attached on N/M samples
   ```
   AND if VD signal healthy, w-distribution from
   `t2_1_vd_distribution.py` should show all weights ∈ [1.0, 2.0].
4. **Step 1 grad_norm**: predicted ≈ 24 (= 1.18× T1-2's 20.34;
   matches rho_l2=1.41 from counterfactual). If <15 or >50,
   something is off; if 24 ± 5, healthy.
5. **`Data statistics multimodal:`**: per
   [[feedback-multimodal-keys]] must see this; otherwise VLM
   training silently text-only.

## Open architectural questions (informed by GPT replies)

### From GPT round-3 (already absorbed)
LR confound vs signed-proxy: A0 → LR confound REFUTED, signed-proxy CONFIRMED.

### From GPT round-4 (already absorbed)
- frac_supp framing → expose `conditional_supp_visual_rejection` (A0+)
- A0 pilot 10-step → 230-step extrapolation acceptable
- T2-1Abs first (minimal change) → done; abs alone fails (rho_l2=12)

### From GPT round-5 (already absorbed)
- Stop in PGPO mass-redistribution family → DONE (moved to boost-only)
- Tier-2 first priority → ALREADY DONE in parallel
- T2-2 boost-only is the design → BUILT + LAUNCHING
- `frac_supp` 7% floor is structural to PGPO threshold; fix is
  remove suppress branch → DONE in boost-only formula

### Pending from Tier-2a brief Q1-Q7 (sent today)
- Q1: 2-mode framing right or overfit?
- Q2: HallusionBench n=15 enough for Mode B?
- Q3: paper §Mechanism leads with 2-modes vs on-policy-attractor?
- Q4: cheaper experiment than Tier-2c?
- Q5: token-budget confounder — qualitative enough or need padded control?
- Q6: missed attacks/confounders?
- **Q7 (new, cross-thread)**: T2-1 token-level mis-target ≡ Mode A?

## Files that matter (committed)

### Method tier (T2)
| Path | Purpose |
|---|---|
| `src/mllmopd/training/vd_weighting.py` | `compute_vd_weights` (T2-1 PGPO) + `compute_vd_weights_boost_only` (T2-2) |
| `src/mllmopd/training/opd_diagnostics_hook.py` | Per-step diag jsonl + sample_index dump + mode dispatch |
| `src/mllmopd/training/opd_adv_dump.py` | A0c sidecar (sample_index, old_log_probs) |
| `src/mllmopd/analysis/t2_1_energy_audit.py` | rho_l2/corr/frac_supp + quadrant table + sidecar join |
| `src/mllmopd/analysis/t2_1_abs_counterfactual.py` | 9-variant scan + verdict per variant |
| `src/mllmopd/analysis/t2_1_per_prompt_diff.py` | Canonical 133 WIN/LOSE categorization |
| `scripts/train/opd_mmr1_3b_baseline_xbox.sh` | Launcher with env passthrough (incl. MLLMOPD_VD_MODE etc.) |
| `scripts/setup/patch_uni_opd.sh` | P19 (adv-dump hook in loss.py) added this session |
| `scripts/train/verify_vd_weighting.py` | 17 unit tests (9 T2-1 + 8 T2-2) |
| `scripts/audit/verify_t2_1_energy_audit.py` | 10 audit smoke tests |
| `scripts/audit/verify_t2_1_abs_counterfactual.py` | 6 counterfactual smoke tests |
| `docs/t2_2_design.md` | Boost-only design + decision tree |
| `docs/gpt-review-2026-05-24-a0-energy-audit.md` | Round-4 prompt |
| `docs/gpt-review-2026-05-24-counterfactual-6variants.md` | Round-5 prompt |

### Mechanism tier (Tier-2)
| Path | Purpose |
|---|---|
| `docs/gpt-brief-2026-05-24-tier2a-results.md` | Brief v3 with 7 questions sent for GPT round-5 review |
| `src/mllmopd/analysis/tier2a_compare.py` | 4-arm comparison analyzer |
| `runs/analysis/tier2a_compare.json` | Numerical dump (in repo, force-added at 11bef7d) |
| `runs/analysis/tier2a_compare.png` | 4-line trajectory figure |
| `src/mllmopd/training/offline_kd_generate.py` | Tier-2a custom generate function |

### Data (ceph only, NOT in repo)
- `${MLLMOPD_RUNS}/t2_1_v0_T2_1_full_vd/diagnostics/step_*.jsonl.gz` (230 step files, original T2-1)
- `${MLLMOPD_RUNS}/t2_1_a0_dump/diagnostics/step_*.{jsonl,adv_dp*.jsonl}.gz` (10-step T2-1 pilot with sidecar)
- `${MLLMOPD_RUNS}/t2_2_v0_boost_only_a05_full/` (T2-2 training, launching now)
- `${MLLMOPD_RUNS}/tier2a_{blank,full}_20260523_*/` (Tier-2a training)
- `${MLLMOPD_RUNS}/audit/tier2a_trajectory_20260524-132227/` (Tier-2a eval)

## What to read first (next session)

1. **This handoff** (you're reading it).
2. **`docs/gpt-brief-2026-05-24-tier2a-results.md`** — the brief @ GPT
   round-5 review; Q1-Q7 frame both threads' next decisions.
3. **`docs/t2_2_design.md`** — T2-2 boost-only design rationale.
4. **Memory entries**:
   - `[[t2-1-result-status-2026-05-24]]` — full T2-1 status snapshot
   - `[[h800-network-gotchas]]` — launcher fix history
5. **`docs/gpt-review-2026-05-24-counterfactual-6variants.md`** —
   round-5 prompt; the 4-criteria + Pareto framing is referenced
   in Tier-2a Q7.
6. **GPT round-3/4/5 replies** (in this conversation's transcript)
   — full reframings that got us here. Round-5 is the most
   load-bearing.

## Critical "don't redo"

- ✅ Xbox infrastructure (P14-P19) — pull + `bash scripts/setup/patch_uni_opd.sh`
  after every git pull
- ✅ T2-1 / T2-2 / A0 audit / counterfactual implementations — all tested
- ✅ Sidecar dump (P19) + audit join — works end-to-end (verified by A0 result)
- ✅ Per-prompt diff result (canonical 133) — committed at `10cb911`
- ✅ Tier-2a result (parallel session) — committed at `11bef7d`
- ✅ Launcher silent-exit fix — committed at `0a05c94`

## Final framing for paper (current draft)

Three paper sections, ordered by current evidence strength:

1. **T1 brief v2 (positive)**: vision-conditioned capability transfer
   via OPD; Δ +23pp on opd_target, p≈10⁻⁶.
2. **Tier-2a refinement of §Mechanism (v3)**: two failure modes of
   dense KL from misspecified teacher — Mode A universal + Mode B
   on-policy cliff. Refines brief v2 from "we found an OPD quirk"
   to "we disentangled two failure modes".
3. **T2 method-tier (TBD on T2-2 result)**: either positive case for
   boost-only |VD| weighting OR negative-result design-space
   exploration (6 variants × Pareto trade-off). T2-1 cond_supp=0.997
   is the cleanest "what NOT to do" finding regardless.

If T2-2 trains successfully + beats T1-2: paper structure is
brief v3 + T2-2 method as solution. If T2-2 negative or marginal:
paper structure is brief v3 + "design space exploration" section
explaining why simple PGPO transplants don't work, and naming
Tier-2c / multi-seed / corruption grid as future work.

Either way, **the 6-variant counterfactual analysis** stands as a
publishable contribution: "what families of token-VD reweighting
don't fit OPD's signed advantage structure, and why".
