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

import torch

from mllmopd.training.vd_weighting import compute_vd_weights

# We can't `from Uni_OPD_utils.OPD_reward.post_process_rewards import ...`
# because that module's top-level pulls in `exps.OPD.utils.reward.get_reward`
# which chains into `rule_base_reward.py`'s missing external dependencies
# (`Math.generate.verify_deepmath`, `exps.RL.utils.reward.PRIME_code_server`).
# Inline the canonical `post_process_rewards` body verbatim from
# Uni_OPD_utils/OPD_reward/post_process_rewards.py (~60 lines) — kept in
# sync manually if upstream changes.

logger = logging.getLogger(__name__)

# Verbatim from Uni_OPD_utils/OPD_reward/post_process_rewards.py:11 and
# Uni_OPD_utils/OPD_reward/get_reward.py:22 — kept in sync manually
# (cf. dual_teacher_get_reward.py for the chain-of-failure rationale).
TEACHER_LOGP_FAILED_SENTINEL = -100.0
REWARD_FAILED_KEY = "__opd_reward_failed__"


def post_process_rewards(args, samples, **kwargs):
    """Inlined replica of Uni-OPD's canonical post_process_rewards.

    Reads `sample.reward["meta_info"]["input_token_logprobs"]`, slices to
    the last `response_length` tokens, writes `sample.teacher_log_probs`
    and `sample.response_correct` in place. Failed samples (REWARD_FAILED_KEY
    or parse error) get filled with TEACHER_LOGP_FAILED_SENTINEL (-100)
    which is what Uni-OPD's loss.py mask-detects to zero the PG loss term.

    Returns (teacher_log_probs_list, teacher_log_probs_list) so the
    caller (miles/ray/rollout.py::_post_process_rewards) treats both
    "raw" and "processed" rewards as the per-sample teacher logp tensor.
    """
    rewards = [sample.reward for sample in samples]
    num_failed = 0
    teacher_log_probs_list = []

    for i, (reward, sample) in enumerate(zip(rewards, samples, strict=False)):
        response_length = sample.response_length
        response_correct = reward.get("response_correct", None)

        if reward.get(REWARD_FAILED_KEY, False):
            num_failed += 1
            t_log_probs = torch.full(
                (response_length,), TEACHER_LOGP_FAILED_SENTINEL, dtype=torch.float32,
            )
            sample.teacher_log_probs = t_log_probs
            sample.response_correct = response_correct
            teacher_log_probs_list.append(t_log_probs)
            continue

        try:
            t_log_probs = torch.tensor(
                [item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]],
                dtype=torch.float32,
            )
            t_log_probs = t_log_probs[-response_length:]
            if len(t_log_probs) != response_length:
                raise ValueError(
                    f"teacher logprob length {len(t_log_probs)} after slice "
                    f"!= response_length {response_length}; teacher returned "
                    f"too few tokens (raw len before slice = "
                    f"{len(reward['meta_info']['input_token_logprobs']) - 1})"
                )
        except Exception as e:
            num_failed += 1
            logger.warning(
                f"[post_process_rewards] sample[{i}] logprob parse error: {e}, "
                f"filling with sentinel {TEACHER_LOGP_FAILED_SENTINEL}."
            )
            t_log_probs = torch.full(
                (response_length,), TEACHER_LOGP_FAILED_SENTINEL, dtype=torch.float32,
            )

        sample.response_correct = response_correct
        sample.teacher_log_probs = t_log_probs
        teacher_log_probs_list.append(t_log_probs)

    if num_failed > 0:
        logger.warning(
            f"[post_process_rewards] {num_failed}/{len(samples)} samples failed, "
            f"their pg_loss will be zeroed by sentinel mask in loss.py."
        )
    return teacher_log_probs_list, teacher_log_probs_list

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
    response dict's `meta_info` (or `meta_info_diagnostic`).

    Mirrors canonical `post_process_rewards` slicing EXACTLY so the
    extracted vector is alignment-equivalent to `sample.teacher_log_probs`:
      - drop the first entry ([1:]) — that's the BOS / first input
        position with no logprob
      - DO NOT filter mid-response None values; if one appears, let
        float(None) raise. Filtering would silently shift alignment
        relative to canonical teacher_log_probs (which crashes on None
        and falls back to the -100 sentinel). Misaligned vd_weights
        would then multiply mismatched advantage positions.
      - tail-slice to `response_length`, assert sufficient length

    Returns [] on any parse error; the caller falls back to ones (unit
    no-op weight), keeping vd_weights and teacher_log_probs always
    co-aligned per sample.
    """
    if not isinstance(reward_subdict, dict):
        return []
    raw = reward_subdict.get("input_token_logprobs")
    if not raw:
        return []
    try:
        all_lp = [float(item[0]) for item in raw[1:]]
        sliced = all_lp[-response_length:]
        if len(sliced) != response_length:
            raise ValueError(
                f"input_token_logprobs length {len(all_lp)} insufficient for "
                f"response_length {response_length}"
            )
        return sliced
    except (TypeError, ValueError) as e:
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

    T2-1: if MLLMOPD_USE_VD_WEIGHTING=1, also attach
    `sample.teacher_vd_weights` (PGPO-style per-token weight, computed
    from lp_full/lp_blank). Uni-OPD's rollout.py plumbs it through to
    `rollout_data["teacher_vd_weights"]`, and our patched loss.py
    multiplies the OPD advantage by it. When the flag is unset, the
    attribute is not added → upstream plumbing's `__dict__` check skips
    it → T1-2/T1-3 stay byte-identical.
    """
    use_vd_weighting = os.environ.get("MLLMOPD_USE_VD_WEIGHTING", "0") == "1"

    # 1. Canonical path — mutates sample.teacher_log_probs in place,
    # returns the (raw, processed) tuple Uni-OPD's trainer expects.
    raw_rewards, processed_rewards = post_process_rewards(args, samples, **kwargs)

    # 2. Diagnostics dump + (optional) VD weight attachment.
    step_i = _STEP_COUNTER["i"]
    _STEP_COUNTER["i"] += 1

    out_path = _diag_out_dir() / f"step_{step_i:06d}.jsonl.gz"
    t0 = time.time()
    n_written = 0
    n_vd_attached = 0
    n_vd_degenerate = 0

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

            vd_weights: list[float] = []
            if use_vd_weighting:
                # Always attach the attribute when env is on, even on
                # degenerate samples (response_length=0, vd parse error,
                # length mismatch). P11's "if 'teacher_vd_weights' in
                # samples[0].__dict__" gate inspects sample 0 ONLY; if
                # that one happens to be degenerate without unconditional
                # attach, the entire batch silently skips VD weighting.
                # Degenerate cases get unit-ones (no-op weight); the
                # multiplier in loss.py P13 still runs but is identity.
                if response_length > 0 and vd and len(vd) == response_length:
                    w = compute_vd_weights(lp_full, lp_blank, response_length)
                    sample.teacher_vd_weights = w
                    vd_weights = w.tolist()
                    n_vd_attached += 1
                else:
                    sample.teacher_vd_weights = torch.ones(
                        max(response_length, 0), dtype=torch.float32
                    )
                    vd_weights = sample.teacher_vd_weights.tolist()
                    n_vd_degenerate += 1

            sample_id = None
            md = getattr(sample, "metadata", None)
            if isinstance(md, dict):
                sample_id = md.get("id") or md.get("uid")

            # sample.index is the rollout-pipeline-internal globally unique
            # sample identifier (matches rollout_data["sample_indices"] used
            # in compute_advantages_and_returns). Required for joining this
            # row with the opd_adv_dump sidecar (T2-1 A0c).
            sample_index = getattr(sample, "index", None)

            old_lp_student = _safe_list(getattr(sample, "old_log_probs", None))
            if old_lp_student and response_length > 0:
                old_lp_student = old_lp_student[-response_length:]

            row = {
                "id": sample_id,
                "sample_index": sample_index,
                "step": step_i,
                "response_length": response_length,
                "image_mode": image_mode,
                "teacher_url": reward.get("teacher_url"),
                "teacher_url_diag": reward.get("teacher_url_diagnostic"),
                "response_correct": reward.get("response_correct"),
                "lp_full": lp_full,
                "lp_blank": lp_blank,
                "vd": vd,
                "vd_weights": vd_weights,
                "old_lp_student": old_lp_student,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    if use_vd_weighting:
        logger.info(
            f"[opd_diag] step {step_i}: wrote {n_written} rows to {out_path} "
            f"({time.time() - t0:.2f}s); VD weights attached on "
            f"{n_vd_attached}/{n_written} samples ({n_vd_degenerate} degenerate→ones)"
        )
    else:
        logger.info(
            f"[opd_diag] step {step_i}: wrote {n_written} rows to {out_path} "
            f"({time.time() - t0:.2f}s)"
        )
    return raw_rewards, processed_rewards
