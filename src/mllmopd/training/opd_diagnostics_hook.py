"""T1 custom-reward-post-process-path: per-step VD / OPD-reward logger.

Wraps Uni-OPD's default `post_process_rewards` so the canonical
behavior (read `meta_info.input_token_logprobs`, slice to
`response_length`, write `sample.teacher_log_probs`) is preserved
verbatim — we just additionally dump every sample's
`lp_full / lp_blank / vd / old_lp_student` to a per-step gzipped
JSONL for offline analysis.

Wired in via the launcher's
  `--custom-reward-post-process-path mllmopd.training.opd_diagnostics_hook.post_process_rewards_with_diagnostics`

Outputs:
  $MLLMOPD_RUNS/<OPD_RUN_NAME>/diagnostics/step_NNNNNN.jsonl.gz

One row per sample per training step. Schema:
  {
    "id":                 metadata.id or null,
    "step":               int,
    "response_length":    int,
    "image_mode":         "full" | "blank" (which arm is primary),
    "teacher_url":        primary teacher URL,
    "teacher_url_diag":   diagnostic teacher URL,
    "response_correct":   bool | null,
    "lp_full":            list[float] length = response_length,
    "lp_blank":           list[float] length = response_length,
    "vd":                 list[float], = lp_full - lp_blank,
    "old_lp_student":     list[float] length = response_length,
  }

Aggregated TensorBoard scalars are TODO (plan §2.3); the JSONL is
the source of truth and post-hoc aggregation reads from it.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import time
from pathlib import Path

# Uni-OPD path layout — same caveat as dual_teacher_get_reward.py;
# the example launcher uses `Uni_OPD_utils.OPD_reward.*` even though
# the .py files themselves contain stale `exps.OPD.utils.reward.*`
# imports.
from Uni_OPD_utils.OPD_reward.post_process_rewards import (
    TEACHER_LOGP_FAILED_SENTINEL,  # noqa: F401  (kept for parity)
    post_process_rewards,
)

logger = logging.getLogger(__name__)

# Module-level step counter. Reset across runs is implicit (each new
# launcher invocation re-imports the module), but persists within a
# single training process so consecutive steps get NNNNNN naming.
_STEP_COUNTER = {"i": 0}


def _diag_out_dir() -> Path:
    base = Path(os.environ.get("MLLMOPD_RUNS", "runs"))
    run_name = os.environ.get("OPD_RUN_NAME", "t1_default")
    out = base / run_name / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_list(x) -> list:
    """Convert a torch.Tensor / list / None to a list of Python floats."""
    if x is None:
        return []
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


def _extract_response_logprobs(reward_subdict: dict, response_length: int) -> list[float]:
    """Pull the response slice of input_token_logprobs from a teacher
    response dict's `meta_info` (or `meta_info_diagnostic`). Returns
    [] on any parse error; the caller can decide whether to fill with
    sentinel or skip the row."""
    if not isinstance(reward_subdict, dict):
        return []
    raw = reward_subdict.get("input_token_logprobs")
    if not raw:
        return []
    try:
        # Each entry is `[logprob, token_id, token_text]` per Uni-OPD's
        # comment in get_reward.py. The first entry is always None
        # (no logprob for the BOS / first input position) — match the
        # default post_process_rewards which slices `[1:]`.
        all_lp = [float(item[0]) for item in raw[1:] if item[0] is not None]
        return all_lp[-response_length:]
    except Exception as e:
        logger.warning(f"[opd_diag] could not parse input_token_logprobs: {e}")
        return []


def post_process_rewards_with_diagnostics(args, samples, **kwargs):
    """Drop-in replacement for Uni-OPD's `post_process_rewards`.

    Calls the canonical implementation unchanged so all existing
    training-side semantics (sentinel handling, masking, raw_rewards
    return shape) stay intact. Then iterates over samples again and
    writes the diagnostic JSONL row, deriving lp_full / lp_blank from
    the primary/diagnostic teacher responses and the current arm's
    `image_mode`.
    """
    # 1. Canonical path — mutates sample.teacher_log_probs in place,
    # returns the (raw, processed) tuple Uni-OPD's trainer expects.
    raw_rewards, processed_rewards = post_process_rewards(args, samples, **kwargs)

    # 2. Diagnostics dump.
    step_i = _STEP_COUNTER["i"]
    _STEP_COUNTER["i"] += 1

    out_path = _diag_out_dir() / f"step_{step_i:06d}.jsonl.gz"
    t0 = time.time()
    n_written = 0

    with gzip.open(out_path, "wt") as fout:
        for sample in samples:
            reward = getattr(sample, "reward", {}) or {}
            response_length = int(getattr(sample, "response_length", 0))

            image_mode = reward.get("image_mode") or os.environ.get(
                "OPD_TEACHER_IMAGE_MODE", "full"
            )

            primary_meta = reward.get("meta_info") or {}
            diag_meta = reward.get("meta_info_diagnostic") or {}
            primary_lp = _extract_response_logprobs(primary_meta, response_length)
            diag_lp = _extract_response_logprobs(diag_meta, response_length)

            if image_mode == "full":
                lp_full, lp_blank = primary_lp, diag_lp
            else:
                lp_full, lp_blank = diag_lp, primary_lp

            vd: list[float] = []
            if lp_full and lp_blank and len(lp_full) == len(lp_blank):
                vd = [a - b for a, b in zip(lp_full, lp_blank)]

            sample_id = None
            md = getattr(sample, "metadata", None)
            if isinstance(md, dict):
                sample_id = md.get("id") or md.get("uid")

            old_lp_student = _safe_list(getattr(sample, "old_log_probs", None))
            if old_lp_student and response_length > 0:
                old_lp_student = old_lp_student[-response_length:]

            row = {
                "id": sample_id,
                "step": step_i,
                "response_length": response_length,
                "image_mode": image_mode,
                "teacher_url": reward.get("teacher_url"),
                "teacher_url_diag": reward.get("teacher_url_diagnostic"),
                "response_correct": reward.get("response_correct"),
                "lp_full": lp_full,
                "lp_blank": lp_blank,
                "vd": vd,
                "old_lp_student": old_lp_student,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    logger.info(
        f"[opd_diag] step {step_i}: wrote {n_written} rows to {out_path} "
        f"({time.time() - t0:.2f}s)"
    )
    return raw_rewards, processed_rewards
