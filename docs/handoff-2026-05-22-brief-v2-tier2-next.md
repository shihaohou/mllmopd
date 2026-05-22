# Handoff: T1 brief v2 landed → Tier-2 off-policy KD controls next (2026-05-22)

Self-contained handoff. Read this if you're a fresh session picking up
the project after the round-4 review pass.

## TL;DR

- T1 v1.5b is the trustworthy positive result: Δ +23pp, p=10⁻⁶,
  BlankTeacher catastrophe + cliff-like collapse at step 149-199.
  Brief v2 (`docs/gpt-brief-2026-05-22-t1-v1p5b-positive-result.md`)
  incorporates **both** GPT round-3 and round-4 review feedback —
  canonical numbers, OPD-condition-sensitive framing, on-policy prefix
  self-conditioning mechanism wording, 5 literature anchors, reviewer
  defense, safety-monitor table, two-target-set caveat.
- Two new analyzer modules landed:
  `src/mllmopd/analysis/t1_brief_table.py` (canonical per-benchmark
  table generator from `t1_compare.json`) and
  `src/mllmopd/analysis/t1_blankness_trajectory.py` (blank-phrase rate
  + first-blank-token-position trajectory). They produced the v2
  brief's canonical §(2) table and the paired
  accuracy↔blankness figure (`runs/analysis/t1_v1p5b_blankness.png`).
- **Recommendation: skip round-5 review; do Tier-2 off-policy KD
  controls *before* T2-1 training.** Rationale in §Why Tier-2 before
  T2-1 below.

## What round-4 changed in the brief

Compared to the v1 brief (commit `9ce0bbe`), v2 added:

| Item | Section | Round |
|---|---|---|
| Canonical per-benchmark table (T1-0 base column corrected: 0.553 mean, not 0.556) | §(2) | 3 |
| "uniform" → "consistent direction, magnitude varies ~30×" | §(2) caption | 3 |
| TL;DR upgraded to "OPD is condition-sensitive" | §TL;DR | 3 |
| Mechanism naming "teacher-conditioned anti-vision template" (with `not generic hallucination` clarifier) | §Mechanism | 3+4 |
| Trajectory wording "on-policy prefix self-conditioning on a blank-template attractor" (formal "phase transition" deferred) | §Trajectory | 3 |
| Noisy-label memorization connection ("abrupt collapse reminiscent of…") | §Trajectory | 4 |
| Blankness-phrase rate paired figure + table (2.2% → 64% in 50 steps; first-blank word pos 94→25) | §Trajectory | 3 |
| §"What T1 does *not* establish" subsection | §Implications | 3 |
| T2 ablation grid (T2-0/1/3/4) with VD-weighted FullTeacher OPD as first cut | §Implications | 3 |
| §Safety monitors during T2 (5 monitors as wrapper, not method) | §Implications | 4 |
| 5 literature anchors (added: Early Stopping Against, PGPO/VPPO as method anchors; fixed: Thinking-with-Deltas URL) | §Literature anchors | 3+4 |
| §Reviewer concerns and responses (3 attacks: contrived BlankTeacher, OPD-specific, T1-2-only-1.3pp) | §Reviewer | 3 |
| Two-target-set caveat (Student-diagnostic + Teacher-diagnostic) | §(3) | 4 |
| Caveats updated: trajectory done, opd_target re-derivation pending, single-seed warning, single-teacher-pair warning | §Caveats | 3+4 |

No numbers, no commits, no bug-fix lineage from v1 was changed. Diff
size: +480 / −180 lines.

## Why we skipped round-5

- Round-3 caught a real silent bug (numbers drift, opd_target axis);
  round-4 was mostly framing/literature with no new bug claims. Returns
  on review diminish; returns on **experiments** are increasing.
- The handoff for round-3 explicitly framed round-4 as "confirming
  readiness for paper draft, not finding bugs". Round-5 risks
  oscillating on tone.
- The right hedge for "is the mechanism claim actually OPD-specific" is
  not another review — it's the off-policy KD / SFT control experiment
  Tier-2 explicitly designed for. Evidence > opinion at this point.

If a future session wants to do round-5 anyway, the minimum-cost form
is one narrow question: *"Is brief v2 still bug-suspect, or is it ready
to merge into the paper draft and start T2?"* — phrased to avoid
inviting more framing rounds.

## Why Tier-2 before T2-1

Currently the strongest claim in the brief is:

> The cliff is OPD-specific because dense KL + on-policy prefix
> self-conditioning is what dense token OPD does that off-policy KD
> doesn't.

This is **not yet tested**. Until off-policy KD on BlankTeacher
completions is also run, a reviewer can validly argue "this is generic
dense-KL-to-a-biased-teacher failure, not an OPD-on-policy-attractor
mechanism". The brief's §Reviewer attack (b) and §Trajectory
mechanism both rest on the on-policy-attractor framing.

**Tier-2 controls (cheap; no T2 framework change needed):**

| Arm | Setup | Cost | Mechanism falsification |
|---|---|---|---|
| **Tier-2a** | Off-policy KD on **BlankTeacher completions**: generate teacher rollouts under blank image once; train student via fixed-target KL on those completions (no on-policy rollouts) | ~1 H800-host × 4-6h | If cliff appears similarly → on-policy attractor is NOT the distinguishing mechanism |
| **Tier-2b** | SFT on BlankTeacher completions | ~1 H800-host × 2-3h | Even simpler control — pure imitation of the bad templates |
| **Tier-2c** | OPD with student rollout ALSO blank | ~1 H800-host × 5-6h | Tests if the *image at rollout time* matters, vs just the teacher's scoring condition |

If both Tier-2a and Tier-2b stay flat (no cliff) and Tier-2c cliffs, the
on-policy prefix self-conditioning claim is **locked**. Then T2-1 has
both a clean motivation and a defensible mechanism story.

If Tier-2a/b also cliff, the claim has to be downgraded to "dense KL
on a misspecified teacher distillation fails in general; OPD
inherits", which is still publishable but less novel — and would
re-shape T2 toward the differential / β-residual formulation as the
defensible fix.

**T2-1 implementation prep can run in parallel with Tier-2 training**
(implementing `dual_teacher_get_reward.py` Δ_t computation + per-token
weight plumbing is Mac-side code work; doesn't block on H800).

## Artifact locations

| Artifact | Path |
|---|---|
| Brief v2 (paper draft v1) | `docs/gpt-brief-2026-05-22-t1-v1p5b-positive-result.md` |
| Canonical-table renderer | `src/mllmopd/analysis/t1_brief_table.py` |
| Blankness-trajectory analyzer | `src/mllmopd/analysis/t1_blankness_trajectory.py` |
| Accuracy trajectory data + figure | `runs/analysis/t1_v1p5b_trajectory.{json,png}` |
| Blankness trajectory data + figure | `runs/analysis/t1_v1p5b_blankness.{json,png}` |
| Headline 9-pass eval | `runs/audit/t1_v1p5b_eval_step249_20260522-102258/` |
| Multi-ckpt trajectory eval (11 jsonls) | `runs/audit/t1_trajectory_20260522-105745/` |
| Training run dirs | `${MLLMOPD_RUNS}/t1_v1p5b_T1_{2_full_mm, 3_blank_mm}/` (ceph only; ckpts at step_{49,99,149,199,230}) |
| Training launcher | `scripts/train/opd_mmr1_3b_baseline.sh` |
| Trajectory eval launcher | `scripts/audit/run_t1_trajectory.sh` |

All briefs + analyzers + analysis artifacts are on `main`. Eval
jsonls were force-added (`runs/*` is gitignored) — same pattern for
the new blankness artifacts.

## Open holes that the next session may want to close

In rough cost order (cheap first):

1. **Multi-seed for T1-3 cliff timing.** Currently n=1 seed. Brief
   explicitly defers the formal "phase transition" claim until ≥2
   more seeds confirm the step-149→199 timing isn't coincidental.
   Cost: 2 × H800-host × ~6h = 12 host-hours.
2. **opd_target re-derivation against MMR1-7B-RL teacher.** Brief §(3)
   carries the Student-diagnostic + Teacher-diagnostic two-set
   recommendation. The Teacher-diagnostic version needs PV rerun
   against `runs/audit/level1_v4_sysprompt_fixed/{T_RL,Base,S}_*.jsonl`
   — H800 but no training, ~30min-1h.
3. **Tier-2 off-policy KD controls** (above). The mechanism falsifier.
4. **Mild corruption teacher grid** (blurred / low-res / random
   natural image / text-only). Tier-3 in original handoff; addresses
   reviewer attack (a) on "contrived BlankTeacher". Each variant a
   full H800-host × ~6h.
5. **T2-1 VD-weighted FullTeacher OPD.** Method tier. Needs the
   `dual_teacher_get_reward.py` change (compute Δ_t = lp_T^full −
   lp_T^blank, plumb as per-token weight in the OPD loss). 5-6h H800
   per arm.
6. **T2-3 residual-β scan.** β∈{0, 0.25, 0.5, 0.75, 1.0}; 5 arms ×
   5-6h = 25-30 host-hours. Expensive; do after T2-1 confirms direction.
7. **T2-4 prompt-level vision-conditioned oversampling.** Lightest
   training change (no loss change, only data-pipeline tweak). Fastest
   to run; informative if T2-1 fails.

## What to read first (next session, sequential)

1. This handoff (you're reading it).
2. Brief v2 (`docs/gpt-brief-2026-05-22-t1-v1p5b-positive-result.md`)
   — read in full; it's the paper draft.
3. §Why Tier-2 before T2-1 above, then jump to the brief's
   §Implications/For T2 + §Safety monitors + Literature anchor #5
   (PGPO/VPPO) — those are the T2 design context.
4. `project_t1_result.md` memory entry for the headline state.
5. (Only if you suspect bugs) the bug-fix lineage table at the end of
   brief v2.

## Stuff resolved and shouldn't bother future-Claude

Same list as `handoff-2026-05-20-v10-stuck.md` plus:

- Brief number drift: fixed by `t1_brief_table.py` (canonical
  regenerator). Never hand-type per-benchmark numbers again.
- "phase transition" wording overclaim: replaced with "abrupt
  cliff-like collapse" (claimed) + "on-policy prefix self-conditioning
  on a blank-template attractor" (mechanism). Formal phase-transition
  claim deferred to multi-seed.
- Reviewer attack (a) on "contrived BlankTeacher": addressed in brief
  §Reviewer (a); compensating mild-corruption-teacher experiment is
  the Tier-3 followup.
- "Learned visual blindness" vs hallucination conflation: clarifier
  added in §Mechanism naming.

## Caveats / known unknowns

- Tier-2 cost estimates above are best-guesses; actual depends on
  whether off-policy KD plumbing already exists in Uni-OPD's loss
  module. If not, ~1 day of plumbing.
- T2-1 implementation cost includes `dual_teacher_get_reward.py`
  rewrite + verification on a 10-prompt smoke run.
- Brief v2's literature anchors include some references with
  approximate arXiv IDs (e.g. "Thinking with Deltas" has no exact
  arXiv URL in our notes); verify before paper submission.

## Commit summary for this session

| Commit | Content |
|---|---|
| (this session, A) | `docs+analysis+runs`: brief v2 + canonical-table + blankness analyzers/figures |
| (this session, B) | `docs`: this handoff |
