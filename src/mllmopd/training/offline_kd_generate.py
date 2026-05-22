"""Tier-2a off-policy KD: custom `generate` function that hydrates Sample
with pre-generated teacher completions instead of doing on-policy rollout.

Wired via Uni-OPD's `--custom-generate-function-path` hook
(third_party/Uni-OPD/miles/miles/rollout/sglang_rollout.py:229). When set,
this function replaces the default `generate()` call to SGLang.

Why this is sufficient (no loss.py change needed):
  The existing `--advantage-estimator on_policy_distillation` branch
  computes `advantage = teacher_logp - student_logp` per token. The math
  doesn't care WHO sampled the tokens — only that we have (1) a token
  sequence, (2) teacher's logp at each position, and (3) student's logp
  at each position (computed by the student's forward pass during
  training, not by us here). So plugging teacher-sampled sequences +
  teacher's stored logprobs into the existing pipeline gives off-policy
  distillation under the same loss machinery.

What this function does, per call:
  1. Look up the matching offline completion by hashing the prompt
     tokens (sample.tokens is set to prompt_ids by sglang_rollout before
     `generate` is invoked).
  2. Pick which of the n stored completions for this prompt to use,
     deterministic on sample.index so the same training step always
     consumes the same teacher rollouts in a reproducible order.
  3. Hydrate Sample fields so downstream code thinks a normal rollout
     happened:
       - tokens         = prompt_ids + teacher_completion_tokens
       - response       = teacher's text
       - response_length= len(teacher_completion_tokens)
       - teacher_log_probs = stored chosen-token logprobs
       - reward         = 1.0 (NON-None → sglang_rollout.py:252 skips RM)
       - status         = COMPLETED
  4. Return — no SGLang call, no teacher server needed.

The lookup table is built lazily on first call from $OPD_OFFLINE_KD_JSONL
(set by the launcher when this generator is active). Keyed on the prompt
token tuple, with one list of completions per prompt.

Companion files:
  - scripts/data/gen_teacher_completions.py        — generated the JSONL
  - scripts/train/opd_mmr1_3b_baseline.sh          — launcher gate
  - third_party/Uni-OPD/miles/miles/utils/types.py — Sample dataclass
"""

from __future__ import annotations

import json
import logging
import os
import threading
from argparse import Namespace
from typing import Any

from miles.utils.types import Sample

logger = logging.getLogger("mllmopd.offline_kd_generate")

# Same triple used at gen time (scripts/data/gen_teacher_completions.py).
QWEN_VL_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

# Lazy-built, process-local. The lookup is keyed by the prompt's token
# tuple — sglang_rollout.py:161 sets sample.tokens = prompt_ids just
# before this function is called, so the same key is reproducible without
# re-running the tokenizer.
_LOOKUP_LOCK = threading.Lock()
_LOOKUP: dict[tuple[int, ...], list[dict[str, Any]]] | None = None
_LOOKUP_SOURCE: str | None = None  # path; cached so we error if env changes mid-run


def _build_lookup(args: Namespace) -> None:
    """Read $OPD_OFFLINE_KD_JSONL and build prompt-token → completions map.

    Re-tokenizes the prompt the same way the training-time data loader
    does (`tokenizer.apply_chat_template` on a single user-role message,
    with `<image>` substituted to the Qwen2.5-VL image-token triple).
    Byte-identical to the prompt_ids the rollout dispatcher hands us at
    generate-time, so the tuple key is a direct match."""
    global _LOOKUP, _LOOKUP_SOURCE
    path = os.environ.get("OPD_OFFLINE_KD_JSONL")
    if not path:
        raise RuntimeError(
            "offline_teacher_generate_func invoked but $OPD_OFFLINE_KD_JSONL is unset. "
            "The launcher should export this env var when wiring "
            "--custom-generate-function-path."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"OPD_OFFLINE_KD_JSONL not found: {path}")

    # Deferred to avoid loading transformers when this module is imported
    # but the function isn't actually called (e.g., bare import probes).
    from transformers import AutoTokenizer

    tok_path = args.hf_checkpoint
    logger.info("[offline-kd] building lookup from %s (tokenizer=%s)", path, tok_path)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    lookup: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    n_rows = 0
    n_skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            problem = r.get("problem")
            if not problem:
                n_skipped += 1
                continue
            templated_problem = problem.replace("<image>", QWEN_VL_IMAGE_TOKEN)
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": templated_problem}],
                tokenize=False,
                add_generation_prompt=True,
            )
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            key = tuple(ids)
            lookup.setdefault(key, []).append(r)
            n_rows += 1

    if not lookup:
        raise RuntimeError(f"offline-KD lookup is empty after reading {path}")

    # Stable order within each prompt's list — sort by sample_idx so the
    # mapping `sample.index % n` → completion is reproducible across runs.
    for key in lookup:
        lookup[key].sort(key=lambda r: r.get("sample_idx", 0))

    _LOOKUP = lookup
    _LOOKUP_SOURCE = path
    logger.info(
        "[offline-kd] lookup ready: %d completions across %d unique prompts (skipped %d)",
        n_rows, len(lookup), n_skipped,
    )


def _ensure_lookup(args: Namespace) -> None:
    global _LOOKUP
    if _LOOKUP is not None:
        return
    with _LOOKUP_LOCK:
        if _LOOKUP is None:
            _build_lookup(args)


async def offline_teacher_generate_func(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
) -> Sample:
    """Drop-in replacement for sglang_rollout.generate() that returns a
    Sample hydrated from pre-generated teacher completions.

    Signature matches what `sglang_rollout.py:237` invokes when
    `args.custom_generate_function_path` is set."""
    _ensure_lookup(args)
    assert _LOOKUP is not None  # for type-checker

    prompt_key = tuple(sample.tokens)
    completions = _LOOKUP.get(prompt_key)
    if completions is None:
        # Surface enough detail to debug a tokenization mismatch.
        prompt_text_hint = sample.prompt if isinstance(sample.prompt, str) else "<list>"
        raise KeyError(
            f"[offline-kd] no offline completion found for this prompt. "
            f"key_len={len(prompt_key)}, sample.index={sample.index}, "
            f"group_index={sample.group_index}, prompt[:200]={prompt_text_hint[:200]!r}. "
            f"Did the train.jsonl drift from the JSONL used at gen time? "
            f"Lookup built from {_LOOKUP_SOURCE!r}."
        )

    # Deterministic slot within the group; wrap with len(completions) so
    # n_samples_per_prompt > n_stored_completions also works (with reuse).
    n = len(completions)
    slot = (sample.index % args.n_samples_per_prompt) % n
    rec = completions[slot]

    teacher_token_ids = rec["completion_token_ids"]
    teacher_logps = rec["completion_token_logprobs"]
    if len(teacher_token_ids) != len(teacher_logps):
        raise ValueError(
            f"[offline-kd] completion_token_ids / logprobs length mismatch in "
            f"{_LOOKUP_SOURCE}: prompt_id={rec.get('prompt_id')}, "
            f"sample_idx={rec.get('sample_idx')}, "
            f"tokens={len(teacher_token_ids)}, logps={len(teacher_logps)}"
        )

    sample.tokens = list(sample.tokens) + list(teacher_token_ids)
    sample.response = rec.get("completion_text", "")
    sample.response_length = len(teacher_token_ids)
    sample.teacher_log_probs = list(teacher_logps)
    sample.reward = 1.0  # non-None → sglang_rollout.py:252 skips RM call
    sample.status = Sample.Status.COMPLETED
    return sample
