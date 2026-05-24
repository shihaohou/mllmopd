# Handoff: Tier-2a GPT review absorbed; T2-2 boost-only paused (2026-05-25)

Self-contained handoff. **Read this + `docs/handoff-2026-05-24-t2-2-launched-tier2a-brief-sent.md` (commit `68c0ae3`)** to come up to speed on both threads. The earlier handoff captures state up to 2026-05-24 17:13; this one captures deltas since.

## TL;DR

Two threads paused for the night:

- **R1 (Tier-2a / mechanism tier)** — GPT round-5 review came in. **Verdict: the refined Mode A + Mode B framing is correct.** GPT flagged 3 must-add sanity checks before paper submission, renamed the two modes to be more specific, reordered the next-step priority ladder (blankness-rate trajectory is the cheapest, NOT Tier-2c), and answered all 7 brief questions. Q7 (cross-thread): T2-1 token-axis mis-target and Tier-2a Mode A trajectory-axis bias are **related but distinct**; don't conflate.
- **R2 (T2-2 boost-only α=0.5 / method tier)** — training crashed at Megatron→sglang weight-sync barrier (gloo 30-min timeout in `dist.barrier(get_gloo_group())`). **Not validated.** Resumption deferred to next session. Typical root causes: silent SGLang actor crash, OOM on one rank, gloo group misconfig.

Nothing committed in response to either event tonight. This handoff is the only artifact of this session.

## R1: Tier-2a — GPT round-5 review absorbed

### Reviewer verdict per question

| Q | Topic | GPT verdict + what would change the judgment |
|---|---|---|
| Q1 | 2-mode framing overfit? | **Agree, downgrade rhetoric**: not "universal" but "empirically observed in this MLLM-OPD setting". Multi-seed would strengthen, not block. |
| Q2 | HallusionBench n=15 enough? | **Not enough alone**; use as qualitative diagnostic. Cheap fix: re-eval existing T1-3/T2A ckpts on **full HallusionBench (~1100 items)**, not just opd_target intersect 15. |
| Q3 | Paper Mechanism § lead framing? | **Abstract/title**: on-policy attractor amplification (memorable). **Mechanism §**: lead with 2 separable modes (defensible). Both, layered. |
| Q4 | Cheaper than Tier-2c? | **YES — blankness-rate trajectory** on existing diagnostic JSONLs (`runs/tier2a_*/diagnostics/step_*.jsonl.gz`, 230 files × 2 arms). Compute "I can't see / blank / completely white" student-response fraction per step → directly tests whether off-policy student internalizes the template at the same rate as on-policy. ~1h Python, no GPU. |
| Q5 | Token-budget confound | **Qualitative is enough for submission.** Cheap robustness add: token-matched checkpoint comparison (T2A-full step 199 ≈ T2A-blank step 230 by total response tokens). Padded-training control is "nice to have", not blocker. |
| Q6 | Missed attacks? | **Two added** — see "Attacks 6+7" below. |
| Q7 | T2-1 mis-target ≡ Tier-2a Mode A? | **Related but distinct**; do NOT unify in paper. Common root: "dense token-level distillation mis-handles visual conditioning". But **T2-1 = TOKEN-axis**: signed-VD proxy mis-routes visual-rejection correction within a Full teacher; **Mode A = TRAJECTORY-axis**: teacher's response distribution is itself biased (Blank teacher). T2-2 boost-only might fix T2-1; it cannot fix Mode A. |

### Mode renaming (per GPT)

| Old | New |
|---|---|
| Mode A: universal degradation | **Mode A: blank-template imitation** |
| Mode B: on-policy specific cliff | **Mode B: on-policy blank-attractor amplification** |

More specific, more interpretable. Brief v3 + paper draft need updating. Not done tonight.

### 3 must-add sanity checks (before paper submission)

1. **Tokenizer parity assert** — `offline_kd_generate._ensure_processor` loads tokenizer from `args.hf_checkpoint`. In production launcher, `--hf-checkpoint = ${STUDENT_CKPT} = MMR1-3B-SFT`. The offline JSONL completions were tokenized with the **teacher** tokenizer (MMR1-7B-RL) at gen time. If MMR1-3B and MMR1-7B tokenizers diverge anywhere (vocab merges, special tokens), the off-policy result is contaminated.
   - Action: write `scripts/audit/verify_tokenizer_parity.py` that asserts `student_tok.get_vocab() == teacher_tok.get_vocab()` and sample-encodes 20 prompts both ways for byte-equality.
   - Probably fine (MMR1 family); add assertion in launcher pre-flight.

2. **Prompt parity hash** — gen-time prompt build (`gen_teacher_completions.build_templated_text`) and train-time prompt rebuild (`offline_kd_generate` via processor) are two parallel code paths. They should produce byte-identical prompts but aren't verified per-row.
   - Action: in `smoke_offline_kd.py`, add an assertion that hashes 20 prompts through both paths and confirms equality.

3. **Sample slot distribution** — `slot = (sample.index % args.n_samples_per_prompt) % n` with `--rollout-shuffle --balance-data`. Verify the intended (prompt × slot 0..7) distribution at runtime, not just on intuition.
   - Action: extend `opd_diagnostics_hook` to log per-step slot histogram, prompt_id coverage, and duplicate-prompt-per-step count.

### Reviewer attacks 6 + 7 (GPT-flagged, not in brief v3 yet)

**Attack 6**: "This is teacher-completion KD, not classical full-distribution KD."
- Response: define Tier-2a explicitly in §Setup — *off-policy in the sampling sense; same OPD `on_policy_distillation` advantage estimator; teacher's chosen-token logprobs stored offline; no full-distribution KL.*

**Attack 7**: "opd_target slice is cherry-picked."
- Response: add language — *opd_target was pre-defined from L1 paired analysis BEFORE Tier-2a was designed; full_image aggregate is also reported; per-benchmark breakdown is transparent.*

### GPT's proposed §Mechanism opening (verbatim-ish, for paper v3 draft)

> We initially hypothesized that learned visual blindness is purely an on-policy OPD failure. Tier-2a falsifies this stronger claim. When trained off-policy on blank-teacher completions, the student also degrades, indicating a universal failure mode of dense distillation from visually misspecified teacher trajectories. However, Tier-2a also validates the key on-policy component: only the on-policy blank-teacher arm exhibits an abrupt cliff on the vision-critical opd_target slice. We therefore separate two mechanisms: **blank-template imitation** (Mode A; affects both off-policy and on-policy distillation), and **on-policy blank-attractor amplification** (Mode B; produces the sharp cliff via student-prefix self-conditioning).

### Reordered next-step priority ladder (per GPT)

1. **Blankness-rate trajectory** on existing T2A diagnostic JSONLs (~1h Python, no GPU). Closes Mode A mechanism narrative.
2. **Tokenizer + prompt parity smokes** (~30 min). Submission robustness.
3. **HallusionBench full eval** (no new training, expanded eval set on existing ckpts). Expands n on Mode B's single strongest benchmark signal.
4. **Token-matched checkpoint comparison** (existing ckpts only). Weakens Q5 confound without new training.
5. **Tier-2c** (5-6h H800). Disambiguates teacher-blank vs student-blank-rollout components of Mode B.
6. **Multi-seed** for T1-3 + T2A-blank + Tier-2c (~30-50 H800-hours). Cliff timing robustness.
7. **Mild corruption teacher grid (Tier-3)**. Addresses "BlankTeacher contrived" attack, was already deferred from Tier-2a.

**Reordering vs the brief**: previously Tier-2c was 5-6h "recommended next"; GPT moved blankness-rate ahead of it because the same mechanism question can be answered without burning GPU.

## R2: T2-2 boost-only α=0.5 — crashed at weight sync, paused

### Crash trace (verbatim from user's terminal)

```
ray::MegatronTrainRayActor.update_weights() (pid=138297, ip=10.48.91.210)
  update_weight_from_distributed.py:109
    dist.barrier(group=get_gloo_group())
RuntimeError: [/pytorch/third_party/gloo/gloo/transport/tcp/unbound_buffer.cc:78]
              Timed out waiting 1800000ms for recv operation to complete
```

30-min gloo barrier timeout = some rank never reached the Megatron→sglang weight-sync barrier. `mem_info` at the moment of crash on rank 0: 27 GB used / 139 GB total — so the rank that timed out was NOT OOM at the barrier point.

### Diagnostic commands (run before resuming)

```bash
# 1. Are all actors / engines still alive?
ps auxf | grep -E "MegatronTrain|SGLangEngine|sglang" | grep -v grep

# 2. Ray cluster state at time of crash (may already be torn down)
ray list actors 2>/dev/null | head

# 3. Recent OOM kills in kernel log
dmesg | tail -50 | grep -iE "oom|killed"

# 4. SGLang engine logs (look for crash stack at any rank)
ls -t runs/t2_2_v0_boost_only_a05_full*/logs/*.log | head -3 | xargs tail -100

# 5. Last barrier reached — find which step we got to
grep -E "Timer|update_weights|barrier" runs/t2_2_v0_boost_only_a05_full*/logs/launcher*.log | tail -30

# 6. Were any ckpts saved? (resume from there if yes)
ls -la runs/t2_2_v0_boost_only_a05_full*/ckpt/hf/ 2>/dev/null
```

### Resumption path

1. Find which rank died (the one whose log doesn't reach the barrier).
2. Probable causes (in order of likelihood):
   - **SGLang rollout actor crashed silently** — check engine logs for stack traces.
   - **One trainer rank stuck on a long forward** and missed the barrier — possible if VD weight compute spiked at a bad batch.
   - **Gloo group built from stale rank assignment** after a Ray actor restart — group identity mismatch.
3. If trainer-side: increase `--dist-timeout` from 1800s to e.g. 3600s (may just delay, not fix).
4. If sglang-side: check sglang engine GPU memory usage; verify `--mem-fraction-static 0.15` left enough headroom under the boost_only mode's weight updates.
5. Re-launch with **new `OPD_RUN_NAME` timestamp** — don't reuse the broken run dir. Resume from last good ckpt if `step_N` saved before crash.

### Expected outcome when revived (per offline counterfactual)

Per handoff `68c0ae3` §"Key result tables":

```
counterfactual prediction for boost_only_a05:
  rho_l2 = 1.413    → step-1 grad_norm ≈ 24 (1.18× T1-2's 20.34)
  corr   = +0.396   → positive alignment with |adv|
  frac_supp = 0.000 → no suppress branch fires
  cond_supp = 0.000 → visual-rejection corrections preserved (Mode A: blank-template imitation)
  visR_mw   = 1.328 → visual-rejection quadrant gets mean weight 1.33
  passes 4/4 strict criteria
```

If T2-2 trains AND Δ ≥ +0.5pp on opd_target vs T1-2 → boost-only is the right method-tier design and T2-1's negative result becomes a "structural insight, fixed by T2-2" paper section.

## Repo state (no commits this session)

Most recent commits on `origin/main`:
```
68c0ae3  docs: handoff for end of T2-2-launch / Tier-2a-brief-sent session  ← read me 2nd
50f37c6  docs: Tier-2a brief — add cross-thread context bridge + Q7
0a05c94  xbox launcher: don't die silently when ip command is absent
11bef7d  analysis+docs+runs: Tier-2a 4-arm trajectory analyzer + brief v3   ← my parallel commit
c42ec3b  audit: boost_only α=0.5/0.7/1.0 sweep variants for T2-2 calibration
35a8a49  T2-2: boost-only |vd| weighting (per GPT round-5 reframe)
b00389c  docs: GPT round-5 review prompt — 6-variant counterfactual Pareto
```

This handoff itself is uncommitted yet. Single file: `docs/handoff-2026-05-25-tier2a-gpt-in-t2-2-paused.md`. User will push.

Note: GPT's verbatim review reply is in the conversation transcript only (search "结论：**Tier-2a 的机制重写是对的"); distilled into §R1 above. Not saved as a separate `gpt-reply-*.md` file (no such pattern exists yet in repo).

## Cold-start reading order (next session)

1. **This handoff** (you're reading it).
2. **`docs/handoff-2026-05-24-t2-2-launched-tier2a-brief-sent.md` (`68c0ae3`)** — full state up to 2026-05-24 17:13. Key tables (A0 audit + 9-variant counterfactual + Tier-2a 4-arm) are reproduced there; don't re-do.
3. **`docs/gpt-brief-2026-05-24-tier2a-results.md`** — the brief that went to GPT round-5 (Q1-Q7 raw).
4. **`docs/t2_2_design.md`** + the launch logs — for R2 resumption context.

## First-action recommendation (next session)

Sequenced by leverage, no GPU first:

1. **30 min — write tokenizer + prompt parity smokes** (`scripts/audit/verify_tokenizer_parity.py` + add hash assertion to `smoke_offline_kd.py`). Cheap, blocks future paper attack, confirms our pipeline is sound.
2. **1 h — write `src/mllmopd/analysis/tier2a_blankness_rate.py`** as analog of existing `t1_blankness_trajectory.py`. Run on existing Tier-2a diagnostic JSONLs (no new GPU). Output: blank-template fraction per step for T2A-blank vs T2A-full. **Decides Mode A mechanism story** (does off-policy student internalize the template at the same rate as on-policy?).
3. **30 min — update brief v3 in place** with renamed modes (Mode A blank-template imitation; Mode B on-policy blank-attractor amplification), append attacks 6+7, and integrate GPT's §Mechanism opening. Commit + push.
4. **Parallel (other conversation)**: R2 resumption — run diagnostic commands, identify dead rank, re-launch T2-2 with new timestamp.

Once 1+2+3 done, decide whether to ship a paper v3 draft NOW or wait for Tier-2c + multi-seed.

## Critical "don't redo"

- ✅ Tier-2a 4-arm trajectory eval — committed at `11bef7d`, numerical dump at `runs/analysis/tier2a_compare.json`.
- ✅ T2-1 A0 energy audit + 9-variant counterfactual — committed in earlier session.
- ✅ Tier-2a code (`offline_kd_generate.py`, `tier2a_compare.py`, launcher gate) — committed.
- ✅ GPT round-5 review on Tier-2a — distilled into §R1; verbatim in conversation transcript.
- ❌ T2-2 training — **crashed mid-flight; needs resumption with fresh run dir**.
- ❌ Tokenizer parity / prompt parity / slot distribution smokes — **not written yet**.
- ❌ Blankness-rate trajectory analyzer for Tier-2a — **not written yet**.
- ❌ Brief v3 updates (renamed modes, attacks 6+7, GPT §Mechanism opening) — **not applied yet**.
