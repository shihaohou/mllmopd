"""Tier-2a off-policy KD: custom `generate` function that hydrates Sample
with pre-generated teacher completions instead of doing on-policy rollout.

Wired via Uni-OPD's `--custom-generate-function-path` hook
(third_party/Uni-OPD/miles/miles/rollout/sglang_rollout.py:229). When set,
this function REPLACES the default `generate()` call to SGLang — note
"replaces", not "wraps". The default generate is what normally populates
`sample.tokens` (prompt_ids) and `sample.multimodal_train_inputs` via the
HF processor; this function has to mirror that work itself.

Why this is sufficient (no Uni-OPD loss.py change needed):
  The existing `--advantage-estimator on_policy_distillation` branch
  computes `advantage = teacher_logp - student_logp` per token. The math
  doesn't care WHO sampled the tokens — only that we have
  (1) a token sequence, (2) teacher's logp at each position, and
  (3) student's logp at each position (computed by the student's
  forward pass during training, not by us here). So plugging
  teacher-sampled sequences + teacher's stored logprobs into the
  existing pipeline gives off-policy distillation under the same loss
  machinery.

Required flag posture (verified, kept off in baseline launcher):
  --use-tis OFF         (would require rollout_log_probs we don't have)
  --use-rollout-logprobs OFF
  --get-mismatch-metrics OFF
  --use-opsm OFF
  All four default off; opd_mmr1_3b_baseline.sh does not enable them.

Lookup key: `sample.metadata["id"]`. This is the prompt_id field written
by prep_opd_train_data.py (top-level "id") AND mirrored into
`metadata: {"id": ...}` by scripts/data/augment_train_jsonl_with_metadata.py
(one-shot backfill on the existing train.jsonl). The data loader
miles/utils/data.py:302 copies `data["metadata"]` into `sample.metadata`,
so this id is available before any generate call. Using it sidesteps the
tokenizer / processor / chat-template / image-grid expansion concerns of
keying on sample.tokens or sample.prompt.

Reward schema: must be a dict with `meta_info.input_token_logprobs`
matching SGLang's `output_token_logprobs` shape, because:
  1. sglang_rollout.py:401,426 logs `reward.items()` — float would crash.
  2. opd_diagnostics_hook.post_process_rewards reads
     `reward["meta_info"]["input_token_logprobs"][1:][-response_length:]`
     and zips that with teacher_log_probs for the diagnostics step jsonl.
  3. opd_diagnostics_hook also reads response_correct, image_mode,
     teacher_url, meta_info_diagnostic — all need defaults.
We construct the dict to mirror what dual_teacher_get_reward would have
returned, with the offline JSONL's stored chosen-token logprobs.

Sample hydration done here (mirrors default generate's side effects):
  sample.tokens             = prompt_ids + teacher_completion_token_ids
  sample.multimodal_train_inputs = processor output (sans input_ids,
                                  attention_mask) — without this,
                                  ray/rollout.py drops images from the
                                  train batch and we silently degrade
                                  to text-only training (the
                                  feedback_multimodal_keys gotcha).
  sample.response           = teacher's text
  sample.response_length    = len(teacher_completion_token_ids)
  sample.teacher_log_probs  = stored chosen-token logprobs
  sample.reward             = dict per schema above
  sample.status             = COMPLETED

Companion files:
  - scripts/data/gen_teacher_completions.py        — generated the JSONL
  - scripts/data/augment_train_jsonl_with_metadata.py — backfills metadata.id
  - scripts/train/opd_mmr1_3b_baseline.sh          — launcher gate
  - third_party/Uni-OPD/miles/miles/utils/types.py — Sample dataclass
  - src/mllmopd/training/opd_diagnostics_hook.py   — reward dict consumer
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from argparse import Namespace
from typing import Any

from miles.utils.processing_utils import load_processor, load_tokenizer
from miles.utils.types import Sample

# orjson is 2-3x faster than stdlib json on the JSONL parse — material on
# the 6GB v0_2k_teacher_completions/*.jsonl cold-start. Optional dep;
# falls back to json transparently.
try:
    import orjson as _fast_json
    def _loads(s: str) -> Any:
        return _fast_json.loads(s)
    _LOADS_LIB = "orjson"
except ImportError:
    def _loads(s: str) -> Any:
        return json.loads(s)
    _LOADS_LIB = "json"

logger = logging.getLogger("mllmopd.offline_kd_generate")

# Lazy-built, process-local. Keyed by prompt_id (str).
_LOOKUP_LOCK = threading.Lock()
_LOOKUP: dict[str, list[dict[str, Any]]] | None = None
_LOOKUP_SOURCE: str | None = None  # cached path; error if env changes mid-run

# Cached processor + tokenizer. Avoid `GenerateState` from sglang_rollout —
# its __init__ touches ~10 args we'd otherwise have to fake in smoke tests,
# and in production it would never have been pre-initialized anyway since
# our custom generate function replaces (not wraps) the default `generate`.
_PROC_LOCK = threading.Lock()
_PROCESSOR: Any = None
_TOKENIZER: Any = None


def _build_lookup() -> None:
    """Read $OPD_OFFLINE_KD_JSONL once, build prompt_id → completions map.

    Cold-start cost: ~60-150s on ceph for the 6GB blank_n8.jsonl,
    dominated by JSON parse + dict insertion. Per-row logs every
    2000 entries so the operator can see progress (otherwise looks
    indistinguishable from a hang). Drops `completion_top_logprobs`
    after parse — the loss path uses only chosen-token logprobs;
    keeping the top-K array in memory would balloon the resident set
    by ~3 GB without buying anything for on_policy_distillation."""
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

    file_size = os.path.getsize(path)
    logger.info(
        "[offline-kd] building lookup from %s (%.1f GB, using %s) — "
        "cold start typically 60-150s on ceph",
        path, file_size / 1e9, _LOADS_LIB,
    )
    t0 = time.time()
    lookup: dict[str, list[dict[str, Any]]] = {}
    n_rows = 0
    n_pruned_tlp = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = _loads(line)
            pid = r.get("prompt_id")
            if pid is None:
                raise KeyError(
                    f"offline JSONL row missing 'prompt_id' field in {path}. "
                    f"Keys present: {list(r.keys())}"
                )
            # Drop the top-K logprob field — it's ~95% of disk size per
            # row but unused by on_policy_distillation (which only needs
            # the chosen-token logprob). Kept off-disk for future
            # forward-KL experiments via a separate optional load.
            if "completion_top_logprobs" in r:
                del r["completion_top_logprobs"]
                n_pruned_tlp += 1
            lookup.setdefault(str(pid), []).append(r)
            n_rows += 1
            if n_rows % 2000 == 0:
                elapsed = time.time() - t0
                rate = n_rows / max(elapsed, 1e-3)
                logger.info(
                    "[offline-kd]   %5d rows / %d prompts so far  "
                    "(%.0f rows/s, %.1fs elapsed)",
                    n_rows, len(lookup), rate, elapsed,
                )

    if not lookup:
        raise RuntimeError(f"offline-KD lookup is empty after reading {path}")

    # Sort each prompt's completions by sample_idx so
    # `sample.index % n_samples_per_prompt` deterministically picks the
    # same slot across reruns.
    for pid in lookup:
        lookup[pid].sort(key=lambda r: r.get("sample_idx", 0))

    _LOOKUP = lookup
    _LOOKUP_SOURCE = path
    elapsed = time.time() - t0
    logger.info(
        "[offline-kd] lookup ready: %d completions across %d unique prompts "
        "in %.1fs (top_logprobs dropped from %d rows to save ~3GB RAM)",
        n_rows, len(lookup), elapsed, n_pruned_tlp,
    )


def _ensure_lookup() -> None:
    global _LOOKUP
    if _LOOKUP is not None:
        return
    with _LOOKUP_LOCK:
        if _LOOKUP is None:
            _build_lookup()


def _ensure_processor(args: Namespace) -> None:
    """Load tokenizer + processor for the teacher checkpoint once per process.

    Matches what `sglang_rollout.GenerateState.__init__` does for these two
    members, but without dragging in the other ~10 args GenerateState
    needs (sglang_server_concurrency, rollout_num_gpus, rollout_temperature,
    etc.) — none of which are relevant in offline-KD mode where we
    aren't running an SGLang router."""
    global _PROCESSOR, _TOKENIZER
    if _PROCESSOR is not None or _TOKENIZER is not None:
        return
    with _PROC_LOCK:
        if _PROCESSOR is not None or _TOKENIZER is not None:
            return
        chat_template_path = getattr(args, "chat_template_path", None)
        logger.info(
            "[offline-kd] loading tokenizer + processor from %s",
            args.hf_checkpoint,
        )
        _TOKENIZER = load_tokenizer(
            args.hf_checkpoint,
            chat_template_path=chat_template_path,
            trust_remote_code=True,
        )
        _PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)


def _build_reward_dict(rec: dict[str, Any], teacher_logps: list[float]) -> dict[str, Any]:
    """Construct a reward dict that satisfies the downstream consumers:

    - sglang_rollout.py:401,426 expects `reward.items()` (any dict works).
    - opd_diagnostics_hook.post_process_rewards reads
        reward["meta_info"]["input_token_logprobs"][1:][-response_length:]
      The leading [1:] drops a placeholder entry (the canonical RM path
      stores the prompt's last-token logprob there); we mirror that by
      prepending a 3-tuple of (0.0, -1, None) that gets dropped.
    - opd_diagnostics_hook also peeks at response_correct, image_mode,
      teacher_url, and meta_info_diagnostic. We provide stable defaults
      so the diagnostics step JSONL still writes coherently in
      off-policy KD runs.
    """
    placeholder = [0.0, -1, None]  # dropped at meta_info[1:]
    response_entries = [
        [float(lp), -1, None] for lp in teacher_logps
    ]
    return {
        "meta_info": {
            "input_token_logprobs": [placeholder] + response_entries,
            "finish_reason": rec.get("finish_reason"),
        },
        "meta_info_diagnostic": {"input_token_logprobs": []},
        "response_correct": False,
        "rule_based_metadata": {},
        "reward_time": 0.0,
        "teacher_url": "offline-jsonl",
        "teacher_url_diagnostic": None,
        "image_mode": rec.get("teacher_image_mode", "offline"),
    }


async def offline_teacher_generate_func(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
) -> Sample:
    """Drop-in replacement for sglang_rollout.generate() that returns a
    Sample hydrated from pre-generated teacher completions, with full
    prompt + multimodal hydration mirroring the default path.

    Signature matches what `sglang_rollout.py:237` invokes when
    `args.custom_generate_function_path` is set."""
    _ensure_lookup()
    assert _LOOKUP is not None  # for type-checker

    # 1) Resolve which teacher completion to use.
    if not isinstance(sample.metadata, dict) or "id" not in sample.metadata:
        raise KeyError(
            "[offline-kd] sample.metadata['id'] is missing. Run "
            "`python scripts/data/augment_train_jsonl_with_metadata.py "
            "--jsonl data/opd_train/v0_2k/train.jsonl` to backfill the "
            "id into metadata, then rerun."
        )
    pid = str(sample.metadata["id"])
    completions = _LOOKUP.get(pid)
    if completions is None:
        raise KeyError(
            f"[offline-kd] no offline completion for prompt_id={pid!r}. "
            f"Lookup built from {_LOOKUP_SOURCE!r}; check the JSONL covers "
            f"the same prompt set as the training data."
        )

    n = len(completions)
    slot = (sample.index % args.n_samples_per_prompt) % n
    rec = completions[slot]

    teacher_token_ids = rec["completion_token_ids"]
    teacher_logps = rec["completion_token_logprobs"]
    if len(teacher_token_ids) != len(teacher_logps):
        raise ValueError(
            f"[offline-kd] tokens/logps length mismatch in {_LOOKUP_SOURCE}: "
            f"prompt_id={pid}, sample_idx={rec.get('sample_idx')}, "
            f"tokens={len(teacher_token_ids)}, logps={len(teacher_logps)}"
        )

    # 2) Mirror the default generate's prompt hydration.
    #    sglang_rollout.py:120-127 does this via GenerateState's processor
    #    (multimodal) or tokenizer (text-only). The processor call also
    #    yields multimodal_train_inputs (pixel_values, image_grid_thw,
    #    etc.) — without this field, miles/ray/rollout.py drops images
    #    from the train batch (feedback_multimodal_keys gotcha) and the
    #    student trains text-only against a multimodal teacher response.
    _ensure_processor(args)
    if _PROCESSOR is not None and sample.multimodal_inputs:
        processor_output = _PROCESSOR(
            text=sample.prompt, **(sample.multimodal_inputs or {})
        )
        prompt_ids = processor_output["input_ids"][0]
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        sample.multimodal_train_inputs = {
            k: v
            for k, v in processor_output.items()
            if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = _TOKENIZER.encode(
            sample.prompt, add_special_tokens=False
        )

    # 3) Hydrate Sample so downstream code thinks a normal rollout happened.
    sample.tokens = list(prompt_ids) + list(teacher_token_ids)
    sample.response = rec.get("completion_text", "")
    sample.response_length = len(teacher_token_ids)
    sample.teacher_log_probs = list(teacher_logps)
    sample.reward = _build_reward_dict(rec, teacher_logps)
    sample.status = Sample.Status.COMPLETED
    return sample
