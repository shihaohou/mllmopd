#!/usr/bin/env python3
"""Smoke 0 (no Ray, no GPU): exercise offline_kd_generate end-to-end on
a single prompt from train.jsonl. Catches the GPT-review-class bugs
(lookup key mismatch, missing multimodal_train_inputs, wrong reward
schema, response_length / teacher_log_probs misalignment) without
paying for a Ray cluster + Megatron forward + checkpoint save.

This script:
  1. Loads one row from --train-jsonl.
  2. Reconstructs the Sample exactly as miles/utils/data.py would
     (apply chat template via _build_messages, set multimodal_inputs).
  3. Sets metadata so offline_teacher_generate_func can look up the
     matching teacher completion from $OPD_OFFLINE_KD_JSONL.
  4. Awaits offline_teacher_generate_func(...).
  5. Asserts the hydrated Sample matches what miles/ray/rollout.py and
     opd_diagnostics_hook.post_process_rewards expect downstream.

If this passes, the static-shape contract is satisfied and Smoke 1
(1-step Ray train with DEBUG_MODE=1 NUM_ROLLOUT=1) is the next gate.

Usage:
  source /root/shihao_project/mllmopd-train-env/.venv/bin/activate
  cd /home/web_server/antispam/project/houshihao/mllmopd
  OPD_OFFLINE_KD_JSONL=data/opd_train/v0_2k_teacher_completions/blank_n8.jsonl \\
    python scripts/train/smoke_offline_kd.py \\
      --train-jsonl data/opd_train/v0_2k/train.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any


def _load_row(jsonl: Path, idx: int) -> dict[str, Any]:
    with jsonl.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"row {idx} out of range for {jsonl}")


async def run(args: argparse.Namespace) -> None:
    # Imports after CLI parse so --help works in any env.
    from PIL import Image
    from miles.utils.data import _build_messages
    from miles.utils.processing_utils import load_processor, load_tokenizer
    from miles.utils.types import Sample
    from mllmopd.training.offline_kd_generate import offline_teacher_generate_func

    row = _load_row(Path(args.train_jsonl), args.row_index)
    print(f"--- testing with row #{args.row_index}: id={row['id']!r} ---")

    # Step 1: reconstruct prompt + multimodal_inputs the way the data
    # loader at miles/utils/data.py:298-338 does.
    images = [Image.open(p).convert("RGB") for p in row["images"]]
    multimodal_inputs = {"images": images}

    messages = _build_messages(
        row, prompt_key="problem", as_conversation=True,
        multimodal_keys={"image": "images"},
    )
    tokenizer = load_tokenizer(args.teacher_model_path, trust_remote_code=True)
    output_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    # Step 2: build Sample matching what RolloutDataSource.get_samples
    # would emit (data_source.py:107-117 sets group_index/index).
    sample = Sample(
        prompt=output_prompt,
        multimodal_inputs=multimodal_inputs,
        metadata=row.get("metadata") or {},
        teacher_model_name="MMR1-7B-RL",
    )
    sample.index = args.fake_sample_index
    sample.group_index = 0

    if "id" not in sample.metadata:
        raise RuntimeError(
            f"train.jsonl row has no metadata.id (legacy file). "
            f"Run scripts/data/augment_train_jsonl_with_metadata.py "
            f"--jsonl {args.train_jsonl} to backfill."
        )

    # Step 3: minimal args namespace mirroring launcher's relevant flags
    fake_args = Namespace(
        hf_checkpoint=args.teacher_model_path,
        n_samples_per_prompt=args.n_samples_per_prompt,
    )

    # Step 4: invoke
    print("\n--- invoking offline_teacher_generate_func ---")
    hydrated = await offline_teacher_generate_func(fake_args, sample, sampling_params={})

    # Step 5: assertions
    print("\n--- hydrated sample ---")
    print(f"  status:                {hydrated.status}")
    print(f"  tokens length:         {len(hydrated.tokens)}")
    print(f"  response_length:       {hydrated.response_length}")
    print(f"  teacher_log_probs len: {len(hydrated.teacher_log_probs or [])}")
    print(f"  response[:120]:        {hydrated.response[:120]!r}")
    mmi_keys = (
        list(hydrated.multimodal_train_inputs.keys())
        if hydrated.multimodal_train_inputs else None
    )
    print(f"  multimodal_train_inputs keys: {mmi_keys}")
    print(f"  reward type:           {type(hydrated.reward).__name__}")
    if isinstance(hydrated.reward, dict):
        print(f"  reward keys:           {sorted(hydrated.reward.keys())}")
        itl = hydrated.reward["meta_info"]["input_token_logprobs"]
        print(f"  input_token_logprobs:  len={len(itl)} (expect response_length+1={hydrated.response_length + 1})")
        print(f"  input_token_logprobs[0]: {itl[0]}")
        print(f"  input_token_logprobs[1]: {itl[1]}")
        print(f"  input_token_logprobs[-1]: {itl[-1]}")

    failures: list[str] = []

    def _check(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)
            print(f"  FAIL: {msg}")
        else:
            print(f"  OK:   {msg}")

    print("\n--- checks ---")
    _check(hydrated.status == Sample.Status.COMPLETED,
           "status == COMPLETED")
    _check(isinstance(hydrated.reward, dict),
           "reward is dict (not float)")
    _check(isinstance(hydrated.reward, dict) and "meta_info" in hydrated.reward,
           "reward has meta_info")
    _check(hydrated.response_length > 0,
           "response_length > 0")
    _check(len(hydrated.teacher_log_probs or []) == hydrated.response_length,
           "len(teacher_log_probs) == response_length")
    _check(hydrated.multimodal_train_inputs is not None,
           "multimodal_train_inputs is not None (else images drop from train batch)")

    if isinstance(hydrated.reward, dict):
        itl = hydrated.reward["meta_info"]["input_token_logprobs"]
        _check(len(itl) == hydrated.response_length + 1,
               "input_token_logprobs length == response_length + 1 (placeholder + response tokens)")
        # opd_diagnostics_hook does input_token_logprobs[1:][-response_length:]
        diag_slice = itl[1:][-hydrated.response_length:]
        _check(len(diag_slice) == hydrated.response_length,
               "opd_diagnostics_hook slice yields exactly response_length entries")

    # Verify tokens layout: prompt_ids + teacher_completion_token_ids.
    processor = load_processor(args.teacher_model_path, trust_remote_code=True)
    processor_output = processor(text=output_prompt, **multimodal_inputs)
    expected_prompt_ids = processor_output["input_ids"][0]
    if hasattr(expected_prompt_ids, "tolist"):
        expected_prompt_ids = expected_prompt_ids.tolist()
    expected_total = len(expected_prompt_ids) + hydrated.response_length
    _check(len(hydrated.tokens) == expected_total,
           f"sample.tokens length == expected_total ({len(hydrated.tokens)} vs {expected_total})")

    # Make sure teacher tokens are at the tail
    if len(hydrated.tokens) == expected_total:
        teacher_tail = hydrated.tokens[-hydrated.response_length:]
        # Re-read the picked offline completion for ground truth.
        from mllmopd.training.offline_kd_generate import _LOOKUP
        pid = sample.metadata["id"]
        completions = _LOOKUP.get(str(pid)) if _LOOKUP else None
        if completions:
            slot = (sample.index % args.n_samples_per_prompt) % len(completions)
            rec = completions[slot]
            _check(teacher_tail == list(rec["completion_token_ids"]),
                   f"tokens tail matches selected completion (slot={slot}, sample_idx={rec.get('sample_idx')})")

    print()
    if failures:
        print(f"=== {len(failures)} FAILURES ===")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("=== ALL CHECKS PASSED ===")
    print("Smoke 1 next: DEBUG_MODE=1 NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=1 SAMPLE_N=2 \\")
    print("              OPD_OFFLINE_KD_JSONL=... bash scripts/train/opd_mmr1_3b_baseline.sh")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-jsonl", required=True,
                    help="Training data JSONL (must contain metadata.id field).")
    ap.add_argument("--teacher-model-path",
                    default=os.environ.get("MMR1_7B_RL_CKPT", ""),
                    help="HF path to teacher tokenizer/processor. "
                         "Default: $MMR1_7B_RL_CKPT.")
    ap.add_argument("--n-samples-per-prompt", type=int, default=8,
                    help="Match launcher's SAMPLE_N (default 8 in T1).")
    ap.add_argument("--row-index", type=int, default=0,
                    help="Which row of train.jsonl to test on (default 0).")
    ap.add_argument("--fake-sample-index", type=int, default=3,
                    help="Value to assign to sample.index — tests the slot "
                         "mapping. Non-zero so we don't trivially hit slot 0.")
    args = ap.parse_args()

    if not args.teacher_model_path:
        print("ERROR: --teacher-model-path required (or set $MMR1_7B_RL_CKPT)",
              file=sys.stderr)
        sys.exit(1)
    if not os.environ.get("OPD_OFFLINE_KD_JSONL"):
        print("ERROR: $OPD_OFFLINE_KD_JSONL must be set to point at the offline JSONL",
              file=sys.stderr)
        sys.exit(1)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
