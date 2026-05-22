# GPT review brief — Tier-2a off-policy KD implementation

**Date**: 2026-05-22 (post brief v2 round-4, pre-Tier-2 training runs)
**Repo**: https://github.com/shihaohou/mllmopd (HEAD = `d2af93c`)
**Reviewer ask**: independent review of the Tier-2a off-policy KD code path
and the experimental design that uses it as a mechanism falsifier for the
T1 brief v2's on-policy-attractor claim.

## Why this matters (1-minute orientation)

T1 v1.5b is a positive result: training MMR1-3B-SFT with **on-policy distillation
from a BlankTeacher** (MMR1-7B-RL prompted with a white canvas) causes a
**cliff-like collapse** at step 149-199 — student accuracy on the 133
opd_target slice drops from ~58% baseline to ~35% (Δ = -23pp,
p ≈ 10⁻⁶). FullTeacher (same teacher seeing the real image) is benign
(Δ ≈ +1.3pp). Full brief: `docs/gpt-brief-2026-05-22-t1-v1p5b-positive-result.md`.

The brief's load-bearing mechanism claim is:

> The cliff is OPD-specific because dense KL + on-policy prefix
> self-conditioning is what dense token OPD does that off-policy KD
> doesn't. The student samples a partial CoT under the blank-teacher
> attractor's gradient, then KL-distills further toward the same
> attractor — a feedback loop.

This is **not yet tested**. Until off-policy KD on the same BlankTeacher
completions is also run, a reviewer can validly argue *"this is generic
dense-KL-to-a-biased-teacher failure, not an OPD-on-policy-attractor
mechanism."* Falsification design from
`docs/handoff-2026-05-22-brief-v2-tier2-next.md` §"Why Tier-2 before T2-1":

| Result | Conclusion |
|---|---|
| Off-policy KD on BlankTeacher **cliffs similarly** | on-policy attractor is **NOT** the distinguishing mechanism — claim must downgrade to "dense KL on misspecified teacher distillation fails in general" |
| Off-policy KD on BlankTeacher **stays flat (no cliff)** | the on-policy prefix self-conditioning claim is **locked**: only on-policy sampling produces the attractor |

## What just got built

A minimal-scope plug-in approach that adds off-policy KD **without modifying
Uni-OPD's loss.py** at all. The key insight:

> The existing `--advantage-estimator on_policy_distillation` branch
> computes `advantage = teacher_logp - student_logp` per token. The
> math doesn't care **who** sampled the tokens — only that we have
> (1) a token sequence, (2) teacher's logp at each position, and
> (3) student's logp at each position (computed by the student's
> forward pass during training, not by us in this generate step).

So plugging teacher-sampled sequences + teacher's stored logprobs into the
existing pipeline gives off-policy distillation under the same loss
machinery. We do this via Uni-OPD's `--custom-generate-function-path`
hook (`third_party/Uni-OPD/miles/miles/rollout/sglang_rollout.py:229`),
which substitutes our function for the default `generate()`.

### Files changed

- `src/mllmopd/training/offline_kd_generate.py` (NEW, 167 lines)
- `scripts/train/opd_mmr1_3b_baseline.sh` (+23 lines)
- `scripts/data/gen_teacher_completions.py` (NEW, earlier in this session — the offline data generator)

### The Tier-2a custom generate function (full source)

```python
# src/mllmopd/training/offline_kd_generate.py

QWEN_VL_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

_LOOKUP: dict[tuple[int, ...], list[dict[str, Any]]] | None = None
_LOOKUP_SOURCE: str | None = None


def _build_lookup(args: Namespace) -> None:
    """Read $OPD_OFFLINE_KD_JSONL and build prompt-token-tuple → completions map."""
    path = os.environ["OPD_OFFLINE_KD_JSONL"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

    lookup = {}
    for line in open(path, "r", encoding="utf-8"):
        r = json.loads(line)
        templated_problem = r["problem"].replace("<image>", QWEN_VL_IMAGE_TOKEN)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": templated_problem}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        key = tuple(ids)
        lookup.setdefault(key, []).append(r)
    # sort each prompt's list by stored sample_idx so the slot mapping
    # is reproducible across runs
    for key in lookup:
        lookup[key].sort(key=lambda r: r.get("sample_idx", 0))


async def offline_teacher_generate_func(args, sample, sampling_params):
    _ensure_lookup(args)
    prompt_key = tuple(sample.tokens)   # set to prompt_ids by sglang_rollout:161
    completions = _LOOKUP[prompt_key]

    n = len(completions)
    slot = (sample.index % args.n_samples_per_prompt) % n
    rec = completions[slot]

    teacher_token_ids = rec["completion_token_ids"]
    teacher_logps     = rec["completion_token_logprobs"]

    sample.tokens          = list(sample.tokens) + list(teacher_token_ids)
    sample.response        = rec.get("completion_text", "")
    sample.response_length = len(teacher_token_ids)
    sample.teacher_log_probs = list(teacher_logps)
    sample.reward          = 1.0   # non-None → sglang_rollout:252 skips RM call
    sample.status          = Sample.Status.COMPLETED
    return sample
```

### How it threads into Uni-OPD's existing loss path

```python
# third_party/Uni-OPD/miles/miles/backends/training_utils/loss.py:388 (verbatim)
elif args.advantage_estimator == "on_policy_distillation":
    student_log_probs = log_probs                                  # from student fwd this step
    teacher_log_probs = rollout_data["teacher_log_probs"]          # ← our stored values
    response_lengths  = rollout_data["response_lengths"]
    ...
    teacher_log_probs = [t_log_prob[-response_length:]
                         for t_log_prob, response_length in zip(...)]
    advantages = [teacher_log_probs - student_log_probs for ... in ...]   # reverse-KL signal
```

`rollout_data["teacher_log_probs"]` is built in
`third_party/Uni-OPD/miles/miles/ray/rollout.py:447-448`:

```python
if "teacher_log_probs" in samples[0].__dict__:
    train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]
```

So setting `sample.teacher_log_probs` in our custom function is the
complete plumbing. No further wiring needed.

### How the RM call gets skipped

Default flow is `generate → batched_async_rm` (`sglang_rollout.py:253`):

```python
samples_need_reward = [sample for sample in samples if sample.reward is None]
rewards = await batched_async_rm(args, samples_need_reward)
```

Setting `sample.reward = 1.0` keeps it out of `samples_need_reward`, so
the teacher HTTP server is never hit during off-policy KD runs. The
launcher's teacher-server start step can be skipped entirely.

### Launcher gate

```bash
# scripts/train/opd_mmr1_3b_baseline.sh (new block)
if [ -n "${OPD_OFFLINE_KD_JSONL:-}" ]; then
  [ -f "${OPD_OFFLINE_KD_JSONL}" ] || { echo "ERROR: not found" >&2; exit 1; }
  export OPD_OFFLINE_KD_JSONL
  ROLLOUT_ARGS+=(
    --custom-generate-function-path
    "mllmopd.training.offline_kd_generate.offline_teacher_generate_func"
  )
fi
```

Same launcher serves three configs:

| Config | Env vars | Arm |
|---|---|---|
| T1-3 BlankTeacher OPD (already done) | `OPD_TEACHER_IMAGE_MODE=blank` | on-policy, blank teacher |
| **Tier-2a-blank** (new) | `OPD_OFFLINE_KD_JSONL=.../blank_n8.jsonl` | off-policy KD, blank teacher |
| **Tier-2a-full** (new, sanity) | `OPD_OFFLINE_KD_JSONL=.../full_n8.jsonl` | off-policy KD, full teacher |

## Data already in hand (offline JSONLs)

Generated by `scripts/data/gen_teacher_completions.py` from MMR1-7B-RL teacher,
TP=4, parallel on the 2k training prompts × 8 samples = 16000 completions each:

| Metric | Blank dataset | Full dataset |
|---|---|---|
| Rows | 16000 ✓ | 16000 ✓ |
| Unique prompts | 2000 ✓ | 2000 ✓ |
| Sample slots × 8 (each 2000) | ✓ | ✓ |
| Median completion length | 582 tok | 885 tok |
| `finish_reason=length` (truncated at 3072) | 1.3% | 1.6% |
| **"I can't see / image is blank / completely white" pattern rate** | **52.2%** | **1.0%** |

The 52.2% vs 1.0% gap is the BlankTeacher attractor that T1-3's on-policy
distillation was distilling into. Tier-2a-blank trains the student on
exactly these completions but **without student rollouts** — pure
teacher-sampled sequence → student forward → KL-distill toward
teacher's stored logprobs.

## What we want you to review (in priority order)

### Q1. Is the loss-math invariance claim actually correct?

We assert that **`advantage = teacher_logp - student_logp` is the same
expression whether the token sequence was sampled by student or by
teacher**. The student's logp at each position is computed by the
student's own forward pass during training, regardless of the sampling
source.

**Concretely**: is there any term in the OPD loss that implicitly assumes
the response was student-sampled (e.g., importance-ratio correction,
rollout-vs-train logprob divergence handling, anything that uses
`old_log_probs`)?

Look at `third_party/Uni-OPD/miles/miles/backends/training_utils/loss.py`
lines 388-460 (advantages block) and 700-870 (clip + IS + PG block).
The `use_tis` / `get_mismatch_metrics` IS-correction path uses
`(old_log_probs, train_log_probs)` — `old_log_probs` is the rollout's
sampling distribution. In our off-policy KD, the "rollout" was the
teacher's, so old_log_probs = teacher's logp at the time of sampling.
But the launcher's `--rollout-batch-size 8 --n-samples-per-prompt 8`
pipeline expects `rollout_log_probs` to be the same model's policy at
generate time. **Does this break anything if we leave `--use-tis` off
(it is off by default in opd_mmr1_3b_baseline.sh)?**

Specifically: `--use-tis` is gated on `args.use_tis or args.get_mismatch_metrics`.
Both default off. Our launcher does not enable them. We claim therefore
that no IS path fires and `advantage = teacher_logp - student_logp` is
the only thing that drives gradients. **Is this correct?** Or does a
non-IS path also reference `old_log_probs` in a way that matters?

### Q2. The reward=1.0 trick — are there other RM-related code paths?

We skip the RM (which would otherwise be `batched_async_rm` calling the
SGLang teacher to score student rollouts) by setting `sample.reward = 1.0`.

`sglang_rollout.py:252` checks `if sample.reward is None`. Setting any
non-None value skips it. But are there other places that read
`sample.reward` and might object to the constant 1.0?

We searched for `sample.reward` in `third_party/Uni-OPD/`:
- `sglang_rollout.py:252-263` (the skip we exploit)
- `miles/ray/rollout.py:427` (used in `train_data["raw_reward"]` from `sample.metadata["raw_reward"]`, not sample.reward directly)

But not exhaustive. **Is there a code path that:**
- Filters samples based on reward (e.g., reject low-reward samples)?
- Logs metrics keyed on reward distribution that would be misleading?
- Affects the loss directly via reward (we use `on_policy_distillation` which does NOT use reward — only `grpo`/`ppo` do)?

### Q3. Sample.tokens lookup key — robust enough?

We key on `tuple(sample.tokens)`. Setup:
- Training-time: `sglang_rollout.py:161` sets `sample.tokens = prompt_ids` (output of `tokenizer.apply_chat_template + tokenize`).
- Our `_build_lookup`: re-runs the same chat-template + tokenize on JSONL's `problem` field with `<image>` → `<|vision_start|><|image_pad|><|vision_end|>`.

Both use the same MMR1-7B-RL tokenizer (`args.hf_checkpoint` = teacher path).

**Failure modes we want you to check:**
- (a) Does the training data loader (`miles/utils/data.py:333`) pass any
  `apply_chat_template_kwargs` that we'd need to mirror? It seems to
  default to `{}`.
- (b) Multimodal prompts: the chat template emits `<|vision_start|><|image_pad|><|vision_end|>` for the IMAGE placeholder. Both
  sides go through the same path. **But SGLang processor may further
  EXPAND `<|image_pad|>` into per-grid image tokens at generate time —
  does that expansion happen before `sample.tokens = prompt_ids` is set, or after?**
- (c) Tokenizer pad / special token differences (we use `add_special_tokens=False` — verify both sides do)?

### Q4. Sample slot mapping — `sample.index % n_samples_per_prompt`

Inside each group, `sample.index` mod `n_samples_per_prompt` should
correspond to a stable "slot" 0..7. Verify in `data_source.py:107-117`:

```python
for prompt_sample in prompt_samples:
    group = []
    for _ in range(self.args.n_samples_per_prompt):
        sample = copy.deepcopy(prompt_sample)
        sample.group_index = self.sample_group_index
        sample.index = self.sample_index
        self.sample_index += 1
        group.append(sample)
    self.sample_group_index += 1
```

So `sample.index` increments globally; within one group of N samples
for the same prompt, indices are consecutive N values, e.g. 0..7 for
prompt 0, 8..15 for prompt 1, etc. **`sample.index % n_samples_per_prompt`
gives 0..7 within a group — does this hold across epochs / shuffles?**

If `rollout_shuffle=True` and `rollout_global_dataset=False`, the same
prompt could appear in different groups across epochs. Each epoch's
`sample.index` still increments globally so the slot mapping changes.
Is that a problem for reproducibility?

### Q5. Missing field hydration

We set: tokens, response, response_length, teacher_log_probs, reward, status.

We do NOT set:
- `sample.loss_mask` (default None) — does the downstream training
  pipeline build it from `response_length`, or assume it's always set
  by the rollout path?
- `sample.rollout_log_probs` (default None) — student logp at rollout
  time. Used by `--use-tis` (off) and `train_rollout_logprob_abs_diff`
  metric (`loss.py:863`). The metric will compute zero if missing —
  acceptable.
- `sample.response_correct` (default None) — used by some reward filters?

### Q6. Experimental design — does Tier-2a actually falsify the claim?

Our null hypothesis (the one we want to potentially reject):
> H0: The cliff is OPD-specific because of dense KL + on-policy prefix
> self-conditioning.

Our alternative hypothesis (the more general failure mode):
> H1: The cliff is dense-KL-to-misspecified-teacher in general; any KL
> distillation against the BlankTeacher attractor causes it.

**Decision matrix on Tier-2a-blank result:**

| Tier-2a-blank result | Conclusion |
|---|---|
| Cliff appears at similar timing + magnitude (Δ ≈ -20pp) | Reject H0; downgrade to H1 |
| No cliff or much smaller (Δ > -5pp) | Fail to reject H0; claim survives |
| Intermediate (Δ between -5 and -20pp) | Ambiguous; need T2-1 / Tier-2c to disambiguate |

**Confounders we're aware of:**
- BlankTeacher completions are shorter (median 582 vs 885) — could
  affect optimizer step dynamics (effective batch size in tokens).
- Off-policy KD's "exposure" to training distribution differs: student
  never samples its own continuation, so no compounding of student
  errors. Argument: this is *exactly* what isolates the on-policy
  ingredient — that compounding is precisely what we're testing.
- Same `--n-samples-per-prompt 8` × same prompt set → same data volume
  if we cycle the offline dataset 8x to match T1's 64 rollouts/step
  cadence. The launcher's `--num-rollout` cap controls total steps.

**Are there other confounders we're missing?**

### Q7. Should we also run Tier-2c (image=blank student rollout)?

Tier-2c keeps on-policy student rollouts but feeds the student a blank
image at rollout time (i.e., student also "doesn't see" the image while
sampling its CoT). One env var change (no code).

Decision matrix:

| Tier-2c result | + Tier-2a result | Conclusion |
|---|---|---|
| Cliff | Cliff | Both blind-image conditions cliff → claim about *visual conditioning* is the locking mechanism, not sampling regime |
| Cliff | No cliff | Cliff is "student sees blank too" — on-policy attractor needs both teacher AND student under blank conditioning |
| No cliff | Cliff | Off-policy KD specifically is the failure; on-policy needs student to see the real image to recover |
| No cliff | No cliff | The cliff requires the *specific* mismatch in T1-3 (teacher-blank + student-real-image + on-policy KL) |

We're inclined to run all three (T1-3 already done, Tier-2a now, Tier-2c
needs 1-day launcher tweak + 5-6h training). **Is the diagnostic value
of Tier-2c worth the wall-clock?**

### Q8. Anything else that worries you?

Specifically:
- We haven't smoke-tested the Tier-2a code yet (only static analysis).
  What's the highest-yield smoke before committing 5-6h of full training?
- Are there cheap pre-flight checks that would catch a lookup-key
  mismatch immediately?

## Repo pointers for review

- Tier-2a code: `src/mllmopd/training/offline_kd_generate.py`
- Launcher gate: `scripts/train/opd_mmr1_3b_baseline.sh` (search `OPD_OFFLINE_KD_JSONL`)
- Data generator: `scripts/data/gen_teacher_completions.py`
- Uni-OPD generate hook: `third_party/Uni-OPD/miles/miles/rollout/sglang_rollout.py:229-237`
- Uni-OPD OPD loss path: `third_party/Uni-OPD/miles/miles/backends/training_utils/loss.py:388-460, 700-870`
- T1 result brief (current paper draft v1): `docs/gpt-brief-2026-05-22-t1-v1p5b-positive-result.md`
- T1 result memory entry: `~/.claude/projects/-Users-houshihao-project-code-mllmopd/memory/project_t1_result.md`
- Tier-2 design rationale: `docs/handoff-2026-05-22-brief-v2-tier2-next.md` §"Why Tier-2 before T2-1"

## Output format we'd like back

For each Qn above:
1. **Verdict**: pass / fail / needs-change. Be sharp.
2. If fail or needs-change: **the specific code path or design change** to fix it.
3. If pass: a one-line confidence note (so we know you actually checked, not skimmed).

Plus a section "**Other concerns / red flags I'd want addressed**" with anything you spot outside Q1-Q8.

Goal: green-light Tier-2a smoke + full runs, OR catch the bug before we burn 10+ host-hours of H800.
