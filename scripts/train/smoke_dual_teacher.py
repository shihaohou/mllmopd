"""T1 punch list #4 smoke: dual-teacher get_reward sanity check.

Stands up nothing — assumes a teacher sglang server is already running
on $TEACHER_URL (default http://localhost:30000/generate). For one
sample row from the prep train.jsonl:

  1. Build a Uni-OPD `Sample` object with the canonical prompt + image.
  2. Call `mllmopd.training.dual_teacher_get_reward.get_reward` once
     under OPD_TEACHER_IMAGE_MODE=full.
  3. Extract `lp_full` (from meta_info) and `lp_blank` (from
     meta_info_diagnostic).
  4. Independently call `mllmopd.diagnostics.score_completion`-style
     forced-decode on the same prefix+response under both image
     conditions, computing reference `lp_full_ref` / `lp_blank_ref`.
  5. Assert per-token |lp_full - lp_full_ref| < 1e-3 (and same for blank).

Passing this confirms our dual-teacher path produces the same logp
values the H2 audit would produce on the same sample — i.e. the
training-time teacher prefix is byte-identical to the audit prefix.

Usage:
  TEACHER_URL=http://localhost:30000/generate \\
  python scripts/train/smoke_dual_teacher.py \\
    --train-jsonl data/opd_train/v0_2k/train.jsonl \\
    --model /path/to/MMR1-7B-RL \\
    --sample-idx 0

NOTE: this only validates dual-teacher's per-token logp output; it
does NOT exercise Uni-OPD's RMSystemManager singleton (which needs
teacher_server_list.json etc. — those are punch list #6 territory).
We construct a minimal stub manager to drive the sample through.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
from pathlib import Path

import aiohttp


def _encode_image_base64(img) -> str:
    """Same as miles.utils.processing_utils.encode_image_for_rollout_engine
    but inlined so this smoke runs without importing miles."""
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def _post(url: str, payload: dict) -> dict:
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        async with s.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def _extract_response_logprobs(meta_info: dict, response_length: int) -> list[float]:
    raw = meta_info.get("input_token_logprobs") or []
    all_lp = [float(item[0]) for item in raw[1:] if item[0] is not None]
    return all_lp[-response_length:]


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-jsonl", type=Path, required=True)
    ap.add_argument("--model", required=True,
                    help="Tokenizer path (must be the SAME model the teacher "
                         "server is running, otherwise logp values diverge)")
    ap.add_argument("--sample-idx", type=int, default=0)
    ap.add_argument("--response", default="",
                    help="Pseudo-response to score against the prompt. If "
                         "empty, uses a short generic answer template. The "
                         "absolute logp values don't matter for the smoke; "
                         "we only check that two paths produce IDENTICAL "
                         "lp arrays.")
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--teacher-url",
                    default=os.environ.get("TEACHER_URL", "http://localhost:30000/generate"))
    args = ap.parse_args()

    # Load one row.
    with args.train_jsonl.open() as f:
        rows = [json.loads(l) for l in f if l.strip()]
    row = rows[args.sample_idx]
    print(f">>> smoke on row {args.sample_idx} (id={row['id']})", file=sys.stderr)

    # Build the prefix exactly as Uni-OPD's Dataset would.
    from transformers import AutoTokenizer
    from PIL import Image

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    img = Image.open(row["images"][0]).convert("RGB")

    # Construct the chat-template-rendered prefix. The verifier in
    # punch list #3 already confirmed this matches Uni-OPD's pipeline.
    # We use Uni-OPD's _build_messages directly so the prefix is what
    # the production training pipeline would produce.
    sys.path.insert(0, str(Path("third_party/Uni-OPD/miles").resolve()))
    from miles.utils.data import _build_messages as uniopd_build  # type: ignore

    messages = uniopd_build(
        data={"problem": row["problem"], "images": [img]},
        prompt_key="problem",
        as_conversation=True,
        multimodal_keys={"image": "images"},
    )
    prefix_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )

    response_text = args.response or (
        "<think>This is a smoke test.</think>"
        "<answer>\\boxed{A}</answer>"
    )
    full_text = prefix_text + response_text

    response_token_ids = tokenizer.encode(response_text, add_special_tokens=False)
    response_length = len(response_token_ids)
    print(f">>> prefix_text {len(prefix_text)} chars, response {response_length} tokens",
          file=sys.stderr)

    # Reference path: emulate score_completion's two forced-decode calls
    # under full and blank images.
    payload_full = {
        "prompt": full_text,
        "image_data": [_encode_image_base64(img)],
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    blank = Image.new("RGB", img.size, (255, 255, 255))
    payload_blank = {
        **payload_full,
        "image_data": [_encode_image_base64(blank)],
    }

    print(f">>> POSTing reference full / blank to {args.teacher_url}", file=sys.stderr)
    ref_full, ref_blank = await asyncio.gather(
        _post(args.teacher_url, payload_full),
        _post(args.teacher_url, payload_blank),
    )
    lp_full_ref = _extract_response_logprobs(ref_full.get("meta_info") or {}, response_length)
    lp_blank_ref = _extract_response_logprobs(ref_blank.get("meta_info") or {}, response_length)

    # Production path: call our dual_teacher get_reward end-to-end.
    # This needs RMSystemManager + a server map pointing at our teacher.
    # Stub them into a temp working dir.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "teacher_server_list.json").write_text(json.dumps({
            "MMR1-7B-RL": {"path": args.model,
                            "servers": [args.teacher_url]},
        }))
        (td_path / "teacher_server_map.json").write_text(json.dumps({
            "default": "MMR1-7B-RL"
        }))

        # Monkeypatch RMSystemManager's hardcoded path constants.
        from Uni_OPD_utils.OPD_reward import reward_manager as rm_mod  # type: ignore
        rm_mod._SERVER_LIST_PATH = td_path / "teacher_server_list.json"
        rm_mod._SERVER_MAP_PATH = td_path / "teacher_server_map.json"
        # Clear singleton cache so it re-reads.
        from miles.utils.misc import SingletonMeta  # type: ignore
        if rm_mod.RMSystemManager in SingletonMeta._instances:  # type: ignore[attr-defined]
            del SingletonMeta._instances[rm_mod.RMSystemManager]  # type: ignore[attr-defined]

        from miles.utils.types import Sample
        sample = Sample(
            prompt=prefix_text,
            label=row.get("answer"),
            metadata={"id": row["id"]},
            multimodal_inputs={"images": [img], "videos": None},
        )
        sample.tokens = tokenizer.encode(full_text, add_special_tokens=False)
        sample.response_length = response_length
        sample.teacher_model_name = "MMR1-7B-RL"

        os.environ["OPD_TEACHER_IMAGE_MODE"] = "full"
        from mllmopd.training.dual_teacher_get_reward import get_reward  # type: ignore
        from argparse import Namespace
        fake_args = Namespace()
        prod_reward = await get_reward(fake_args, sample)

    lp_full_prod = _extract_response_logprobs(
        prod_reward.get("meta_info") or {}, response_length,
    )
    lp_blank_prod = _extract_response_logprobs(
        prod_reward.get("meta_info_diagnostic") or {}, response_length,
    )

    # Compare.
    def _maxdiff(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return float("inf")
        return max(abs(x - y) for x, y in zip(a, b))

    md_full = _maxdiff(lp_full_prod, lp_full_ref)
    md_blank = _maxdiff(lp_blank_prod, lp_blank_ref)
    print()
    print(f"prod lp_full  len={len(lp_full_prod)}  ref len={len(lp_full_ref)}")
    print(f"prod lp_blank len={len(lp_blank_prod)} ref len={len(lp_blank_ref)}")
    print(f"max |Δlp_full|  = {md_full:.6f}")
    print(f"max |Δlp_blank| = {md_blank:.6f}")
    print(f"tolerance       = {args.tol}")

    if md_full < args.tol and md_blank < args.tol:
        print(">>> PASS: dual_teacher logp matches reference within tolerance.")
        sys.exit(0)
    else:
        print(">>> FAIL: dual_teacher logp diverges from reference. "
              "Check payload construction (image encoding, input_ids, sampling_params).")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
