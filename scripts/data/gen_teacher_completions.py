#!/usr/bin/env python3
"""Offline teacher-completion generator for Tier-2a off-policy KD controls.

Reads a prompt JSONL (output of `scripts/data/prep_opd_train_data.py`),
calls a running SGLang teacher server with `image=full` or `image=blank`,
stores teacher tokens + top-K logprobs as JSONL for downstream off-policy
KD training (and as a cheap SFT control if `--top-k-logprobs=0`).

Design notes:
  - Streams one JSON line per (prompt_id, sample_idx) completion to disk
    so a crash mid-run doesn't lose work. Resumable: re-running with the
    same --out-jsonl skips rows already present.
  - Concurrency is bounded by --concurrency (defaults to 64; SGLang's
    --max-running-requests should be >= this).
  - The chat template is applied client-side via the teacher's HF
    tokenizer so that an `<image>` placeholder in the rendered text gets
    mapped to vision tokens by the SGLang multimodal processor.
  - The "blank" image substitution matches `mllm_corruptions.blank_image`
    (white 255,255,255) at the same size as the original — identical
    bytes to the audit/H2 blank canvas used in T1's BlankTeacher pathway.

Output schema per row (one JSON object per line):
  {
    "prompt_id":          str,            # e.g. "mmr1_rl_v0_001234"
    "sample_idx":         int,            # 0..n_samples-1
    "problem":            str,            # original prompt JSONL "problem" field
    "image_path":         str,            # path to the *real* image (training-time)
    "teacher_image_mode": "blank"|"full",
    "teacher_model":      str,            # short name, e.g. "MMR1-7B-RL"
    "teacher_model_path": str,            # absolute path to the served model
    "answer_gold":        str,            # from prep JSONL (for eval/sanity)
    "completion_text":    str,            # teacher's response (concatenated)
    "completion_token_ids":      [int],   # length T
    "completion_token_logprobs": [float], # length T, chosen-token log p
    "completion_top_logprobs":   [[[token_id, logprob], ...], ...],  # T × K
    "finish_reason":      str | dict,     # SGLang's finish reason
    "sampling_params":    {...},          # echo for reproducibility
  }

Usage:
  # 1) Start the teacher on this box. To avoid clobbering the shared
  #    teacher_server_list.json (which lives on ceph and is read by
  #    student training on a *different* box), pass TEACHER_REGISTER=0.
  TEACHER_GPUS=0,1,2,3,4,5,6,7 TEACHER_TP_SIZE=8 \\
    TEACHER_MEM_FRACTION=0.85 TEACHER_MAX_RUNNING=256 \\
    TEACHER_REGISTER=0 \\
    bash scripts/train/start_teacher_server.sh
  # script blocks until /get_model_info returns 200, then forks to bg.

  # 2) Generate BlankTeacher dataset (~30-60 min for 2k × 8 on 8x H800)
  python scripts/data/gen_teacher_completions.py \\
      --prompt-jsonl  data/opd_train/v0_2k/train.jsonl \\
      --out-jsonl     data/opd_train/v0_2k_teacher_completions/blank_n8.jsonl \\
      --image-mode    blank --n-samples 8

  # 3) Generate FullTeacher dataset
  python scripts/data/gen_teacher_completions.py \\
      --prompt-jsonl  data/opd_train/v0_2k/train.jsonl \\
      --out-jsonl     data/opd_train/v0_2k_teacher_completions/full_n8.jsonl \\
      --image-mode    full  --n-samples 8

Companion to:
  - scripts/train/start_teacher_server.sh (teacher server, with TEACHER_REGISTER=0 for gen mode)
  - src/mllmopd/training/blank_teacher_payload.py (the in-training blank substitution)
  - docs/handoff-2026-05-22-brief-v2-tier2-next.md §"Why Tier-2 before T2-1"
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
from tqdm.asyncio import tqdm_asyncio

logger = logging.getLogger("gen_teacher_completions")


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _to_blank(img: Image.Image) -> Image.Image:
    """Same-size white canvas — byte-identical to `mllm_corruptions.blank_image`
    default (255,255,255), which is what T1's BlankTeacher pathway used."""
    return Image.new("RGB", img.size, (255, 255, 255))


def encode_image_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Prompt → chat-templated text
# ---------------------------------------------------------------------------

# SGLang's Qwen2.5-VL multimodal processor matches image positions via the
# regex `<|vision_start|>(<|image_pad|>)+<|vision_end|>` (see
# third_party/sglang/python/sglang/srt/multimodal/processors/qwen_vl.py:253).
# The prep script writes `<image>` as a human-readable placeholder (the
# Uni-OPD internal convention). We must convert it before the prompt
# reaches sglang or the request silently runs as text-only.
QWEN_VL_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


def build_templated_text(tokenizer, problem: str) -> str:
    """Wrap `problem` in the teacher's chat template, generation-prompt on.

    Prep schema puts the MMR1 sysprompt + `<image>` placeholder + question
    all inside the `problem` field (`scripts/data/prep_opd_train_data.py`
    §"Output schema"). We:
      1. Substitute `<image>` → Qwen2.5-VL's image-token triple.
      2. Apply the chat template as a single user turn — the role markers
         get inserted around the content.
    SGLang then expands `<|image_pad|>` into per-grid image tokens at
    server time using the image_data we pass alongside."""
    if "<image>" not in problem:
        raise ValueError(
            f"prompt JSONL row's `problem` field is missing the `<image>` "
            f"placeholder; cannot map to multimodal teacher input. "
            f"problem[:200]={problem[:200]!r}"
        )
    problem = problem.replace("<image>", QWEN_VL_IMAGE_TOKEN)
    messages = [{"role": "user", "content": problem}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# SGLang client
# ---------------------------------------------------------------------------

async def _call_sglang(
    client: httpx.AsyncClient,
    endpoint: str,
    text: str,
    image_b64: str,
    sampling_params: dict[str, Any],
    top_k_logprobs: int,
    *,
    max_retries: int = 3,
) -> dict[str, Any]:
    payload = {
        "text": text,
        "image_data": [image_b64],
        "sampling_params": sampling_params,
        "return_logprob": True,
        "top_logprobs_num": top_k_logprobs,
        "logprob_start_len": -1,  # response-only logprobs (sglang default)
    }
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = await client.post(endpoint, json=payload)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError) as e:
            last_err = e
            backoff = 2 ** attempt
            logger.warning(
                "sglang call failed (attempt %d/%d): %s; retry in %ds",
                attempt + 1, max_retries, e, backoff,
            )
            await asyncio.sleep(backoff)
    assert last_err is not None
    raise last_err


def _extract_completion(
    resp: dict[str, Any],
    sampling_params: dict[str, Any],
) -> dict[str, Any]:
    """Pull tokens + logprobs from SGLang's response. Field names per
    third_party/sglang/python/sglang/srt/managers/io_struct.py.

    Note SGLang's output_top_logprobs_{val,idx} are parallel arrays:
      val[t][k] = logprob of the k-th top token at position t
      idx[t][k] = its vocabulary id
    We collapse them into the more canonical [[token_id, logprob], ...]
    so the downstream training loader doesn't have to know SGLang's
    storage convention."""
    text = resp["text"]
    meta = resp["meta_info"]

    tok_val = meta["output_token_logprobs_val"]
    tok_idx = meta["output_token_logprobs_idx"]
    if len(tok_val) != len(tok_idx):
        raise ValueError(
            f"output_token_logprobs_{{val,idx}} length mismatch: "
            f"{len(tok_val)} vs {len(tok_idx)}"
        )

    top_val = meta.get("output_top_logprobs_val") or []
    top_idx = meta.get("output_top_logprobs_idx") or []
    top_pairs: list[list[list[float | int]]] = []
    for v_at_t, i_at_t in zip(top_val, top_idx, strict=False):
        # Each position contributes K (token_id, logprob) pairs.
        top_pairs.append(
            [[int(tid), float(lp)] for tid, lp in zip(i_at_t, v_at_t, strict=False)]
        )

    return {
        "completion_text": text,
        "completion_token_ids": [int(t) for t in tok_idx],
        "completion_token_logprobs": [float(v) for v in tok_val],
        "completion_top_logprobs": top_pairs,
        "finish_reason": meta.get("finish_reason"),
        "sampling_params": sampling_params,
        "prompt_tokens_seen": meta.get("prompt_tokens"),
        "completion_tokens_seen": meta.get("completion_tokens"),
    }


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _scan_existing(out_path: Path) -> set[tuple[str, int]]:
    if not out_path.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                # Truncated last line from a previous crash. Skip it; we'll
                # re-emit the corresponding (prompt_id, sample_idx) since it
                # never made it into `done`.
                continue
            pid = r.get("prompt_id")
            si = r.get("sample_idx")
            if pid is not None and si is not None:
                done.add((str(pid), int(si)))
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    # Defer transformers import so users without the train venv still
    # get a clean --help.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model_path, trust_remote_code=True)
    teacher_name = args.teacher_model_name or Path(args.teacher_model_path).name

    rows: list[dict[str, Any]] = []
    with open(args.prompt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit > 0:
        rows = rows[: args.limit]
    logger.info("loaded %d prompts from %s", len(rows), args.prompt_jsonl)

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _scan_existing(out_path)
    if done:
        logger.info("resume: %d completions already in %s; skipping those", len(done), out_path)

    sampling_params = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "skip_special_tokens": False,
    }
    if args.stop:
        sampling_params["stop"] = [args.stop]

    endpoint = args.teacher_endpoint.rstrip("/")
    if not endpoint.endswith("/generate"):
        endpoint = endpoint + "/generate"

    sem = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(
        connect=30.0, read=args.request_timeout, write=60.0, pool=30.0
    )
    limits = httpx.Limits(
        max_connections=max(args.concurrency * 2, 256),
        max_keepalive_connections=args.concurrency,
    )

    # Lock around the output file: tasks may finish out of order.
    out_lock = asyncio.Lock()
    out_f = out_path.open("a", encoding="utf-8")

    async def _one(row: dict[str, Any], sample_idx: int) -> None:
        key = (str(row["id"]), sample_idx)
        if key in done:
            return
        img = _load_image(row["images"][0])
        if args.image_mode == "blank":
            img = _to_blank(img)
        elif args.image_mode != "full":
            raise ValueError(f"--image-mode must be full|blank, got {args.image_mode!r}")
        img_b64 = encode_image_b64(img)

        text = build_templated_text(tokenizer, row["problem"])
        async with sem:
            resp = await _call_sglang(
                client, endpoint, text, img_b64, sampling_params,
                top_k_logprobs=args.top_k_logprobs,
            )
        rec_extract = _extract_completion(resp, sampling_params)
        record = {
            "prompt_id": row["id"],
            "sample_idx": sample_idx,
            "problem": row["problem"],
            "image_path": row["images"][0],
            "teacher_image_mode": args.image_mode,
            "teacher_model": teacher_name,
            "teacher_model_path": args.teacher_model_path,
            "answer_gold": row.get("answer"),
            **rec_extract,
        }
        async with out_lock:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    try:
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            tasks = [
                _one(row, s)
                for row in rows
                for s in range(args.n_samples)
                if (str(row["id"]), s) not in done
            ]
            total = len(tasks)
            logger.info(
                "submitting %d generations (%d prompts × %d samples, %d already done)",
                total, len(rows), args.n_samples, len(done),
            )
            t0 = time.time()
            await tqdm_asyncio.gather(
                *tasks, desc=f"gen-{args.image_mode}", total=total, smoothing=0.05
            )
            elapsed = time.time() - t0
            logger.info("done in %.1fs (%.2f s/completion)",
                        elapsed, elapsed / max(total, 1))
    finally:
        out_f.close()


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--prompt-jsonl", required=True,
                   help="Input prompt JSONL (prep_opd_train_data.py output).")
    p.add_argument("--out-jsonl", required=True,
                   help="Output JSONL path. Resumable across runs.")
    p.add_argument("--image-mode", required=True, choices=["full", "blank"],
                   help="Teacher image conditioning at generation time.")
    p.add_argument("--n-samples", type=int, default=8,
                   help="Samples per prompt (default 8 to match T1 SAMPLE_N).")
    p.add_argument("--teacher-endpoint", default="http://127.0.0.1:30000",
                   help="SGLang server endpoint (/generate suffix optional).")
    p.add_argument("--teacher-model-path", default=os.environ.get("MMR1_7B_RL_CKPT", ""),
                   help="HF path to teacher (for tokenizer). Default: $MMR1_7B_RL_CKPT.")
    p.add_argument("--teacher-model-name", default="",
                   help="Short name for record (default: basename of model path).")
    p.add_argument("--max-new-tokens", type=int, default=3072,
                   help="Match T1 default ROLLOUT_MAX_RESPONSE_LEN=3072.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=-1, help="-1 disables top-k filter.")
    p.add_argument("--stop", default="</answer>",
                   help="Stop string; matches T1 ROLLOUT_STOP. Empty to disable.")
    p.add_argument("--top-k-logprobs", type=int, default=20,
                   help="Top-K logprobs to store per response position (KD target). "
                        "Set 0 to skip (SFT-only dataset, ~10x smaller).")
    p.add_argument("--concurrency", type=int, default=64,
                   help="Max in-flight requests. Must be <= teacher's --max-running-requests.")
    p.add_argument("--request-timeout", type=float, default=600.0,
                   help="Per-request read timeout, seconds.")
    p.add_argument("--limit", type=int, default=0,
                   help="Take only first N prompts (for smoke testing). 0 = all.")
    args = p.parse_args()
    if not args.teacher_model_path:
        p.error("--teacher-model-path is required (or set $MMR1_7B_RL_CKPT in env).")
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    args = _parse()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
