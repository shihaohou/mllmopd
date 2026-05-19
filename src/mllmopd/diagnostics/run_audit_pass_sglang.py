"""SGLang-based audit pass — drop-in alternative to `run_audit_pass.py`.

Same CLI/output JSONL schema; runs ~5-10x faster than HF transformers on
variable-output-length workloads like MMR1 (POPE ~20 tok mixed with MathVista
~1024 tok) thanks to sglang's continuous batching + paged KV cache.

Must be launched from the **train venv** (where sglang is installed). The
audit venv (transformers-only) has no sglang and should keep using
`run_audit_pass.py` (HF backend).

Smoke-test before a full pass — sglang API surface drifts between versions:
    python -m mllmopd.diagnostics.run_audit_pass_sglang \
        --subset data/audit/smoke_subset_v0.jsonl \
        --model <path> --mode full_image --out /tmp/dbg.jsonl \
        --limit 4 --debug

Then for a full pass:
    python -m mllmopd.diagnostics.run_audit_pass_sglang \
        --subset data/audit/smoke_subset_v0.jsonl \
        --model /path/to/MMR1-7B-SFT --mode full_image \
        --out runs/audit/smoke500/T_SFT_full.jsonl

Notes / things to verify on first launch:
  - sglang.Engine(...) image_data accepts PIL.Image directly. If it errors,
    pass a path or BytesIO instead.
  - `meta_info.completion_tokens` and `meta_info.prompt_tokens` are the canon
    fields; older sglang used different names. The code falls back to
    re-tokenizing if either is missing.
  - Greedy decoding (`temperature=0`) should match HF token-for-token within
    numerical noise. A couple of tokens differing on long CoTs is normal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from mllmopd.data import mllm_corruptions
from mllmopd.diagnostics import scorers
from mllmopd.diagnostics.run_audit_pass import (
    _IMAGE_REQUIRED_MODES,
    _build_messages,
    _emit_skip_missing_image,
    _extract_choices,
    _load_subset,
)


def _build_chat_text(tokenizer, rec, transformed, prefix) -> str:
    """Produce chat-template text with image placeholder tokens. We use the
    plain tokenizer (not AutoProcessor) because sglang handles the image
    embedding itself; we only need the textual scaffold here."""
    messages = _build_messages(rec, transformed, prefix)
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )


def _emit_row(rec, args, fout, pred: str, num_tok: int, prompt_len: int):
    choices = _extract_choices(rec)
    is_correct, scorer_used, parse_path = scorers.score_for_benchmark(
        rec["benchmark"], pred, rec.get("answer"), choices=choices,
    )
    row = {
        "id": rec["id"],
        "benchmark": rec["benchmark"],
        "mode": args.mode,
        "model": args.model,
        "prediction": pred,
        "num_tokens": int(num_tok),
        "prompt_len": int(prompt_len),
        "gold": rec.get("answer"),
        "is_correct": is_correct,
        "scorer": scorer_used,
        "parse_path": parse_path,
    }
    if choices:
        row["choices"] = list(choices)
    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    fout.flush()
    os.fsync(fout.fileno())
    return is_correct, scorer_used


def _prepare_requests(args, tokenizer):
    """Walk the subset, materialize PIL images, apply the perception-mode
    transform, build chat text. Returns (requests, skipped) where each
    request is a tuple (rec, prompt_text, image_data | None)."""
    requests = []
    skipped = []
    for rec in _load_subset(args.subset):
        if args.limit and len(requests) + len(skipped) >= args.limit:
            break

        image_field = rec.get("image")
        if isinstance(image_field, list) and image_field:
            image_field = image_field[0]
        pil_image = None
        if isinstance(image_field, (str, Path)):
            try:
                pil_image = mllm_corruptions.load(image_field)
            except Exception as e:
                print(f"!! could not load image {image_field}: {e}", file=sys.stderr)
        elif hasattr(image_field, "convert"):
            pil_image = image_field

        if pil_image is None and args.mode in _IMAGE_REQUIRED_MODES:
            skipped.append(rec)
            continue

        transformed, prefix = mllm_corruptions.apply_mode(
            pil_image,
            args.mode,
            caption=rec.get("meta", {}).get("caption"),
        )

        prompt_text = _build_chat_text(tokenizer, rec, transformed, prefix)
        image_data = [transformed] if transformed is not None else None
        requests.append((rec, prompt_text, image_data))
    return requests, skipped


def _engine_generate(engine, prompts, image_data_list, sampling_params):
    """Wrap engine.generate to keep version drift contained.

    sglang's `image_data` argument expects one of:
      - None (no image)
      - list[image_for_each_prompt] where each item is itself a list
        (multi-image per prompt) OR a single PIL/path
    Different sglang versions are pickier about None vs []. We pass None
    for the whole call when *all* prompts are image-free, otherwise pass
    the per-prompt list."""
    kwargs = {
        "prompt": prompts,
        "sampling_params": sampling_params,
    }
    if any(img is not None for img in image_data_list):
        kwargs["image_data"] = image_data_list
    return engine.generate(**kwargs)


def _extract_pred(out, tokenizer) -> tuple[str, int, int]:
    """Pull (text, completion_tokens, prompt_tokens) from an sglang output
    record. Handles a few API variants we've seen across sglang versions."""
    if isinstance(out, str):
        pred = out
        return pred, len(tokenizer.encode(pred, add_special_tokens=False)), 0
    pred = out.get("text", "") if isinstance(out, dict) else getattr(out, "text", "")
    meta = (out.get("meta_info") if isinstance(out, dict) else getattr(out, "meta_info", {})) or {}
    n_completion = (
        meta.get("completion_tokens")
        or meta.get("output_tokens")
        or len(tokenizer.encode(pred, add_special_tokens=False))
    )
    n_prompt = meta.get("prompt_tokens") or meta.get("input_tokens") or 0
    return pred, int(n_completion), int(n_prompt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=[
        "full_image", "blank_image", "text_only",
        "caption_only_blank", "image_plus_caption",
        "swap_image", "irrelevant_image",
        "oracle_caption",
    ])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="Generation cap. MMR1-style long-CoT models commonly need "
                         "2-4k+ tokens to reach their final-answer phrase; 1024 truncated "
                         "majority of MathVision / MathVerse outputs before the answer.")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit; use 4 for smoke test")
    ap.add_argument("--chunk-size", type=int, default=50,
                    help="Submit requests in chunks of K for progress visibility "
                         "(does not affect throughput — sglang batches continuously).")
    ap.add_argument("--mem-fraction", type=float, default=0.85,
                    help="Fraction of GPU memory for sglang's static allocation. "
                         "0.85 is safe on 80GB A800 for 7B bf16; lower if OOM.")
    ap.add_argument("--max-running-requests", type=int, default=64,
                    help="Cap concurrent in-flight requests inside sglang. "
                         "32-64 is typical for 7B on 80GB.")
    ap.add_argument("--attention-backend", default=None,
                    help="sglang attention backend (e.g. flashinfer / triton). "
                         "Leave unset to use sglang's default for the GPU.")
    ap.add_argument("--debug", action="store_true",
                    help="Print prompt + generation + scoring for each emitted row.")
    args = ap.parse_args()

    # Disable cuDNN before anything imports torch heavily. sglang warns that
    # torch 2.9.1 + the bundled cuDNN <9.15 can hit `CUDNN_STATUS_NOT_INITIALIZED`
    # on `nn.Conv3d` (the Qwen2.5-VL visual patch embedding) when multiple
    # engines initialize in parallel — single-engine smoke runs survive, but
    # 3-GPU parallel passes crash. Native CUDA convs cost a few ms/image extra
    # and avoid the race entirely.
    import torch  # noqa: F401
    torch.backends.cudnn.enabled = False

    # We need the tokenizer's chat template to build the prompt text. Loading
    # the full AutoProcessor is unnecessary since sglang handles image tokens.
    from transformers import AutoTokenizer  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    requests, skipped = _prepare_requests(args, tokenizer)
    print(f">>> {len(requests)} requests built, {len(skipped)} skipped (missing image)",
          file=sys.stderr)

    if not requests:
        # Nothing to do but still emit the skip markers so downstream tools see them
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as fout:
            for rec in skipped:
                _emit_skip_missing_image(rec, args, fout)
        print(f">>> wrote 0 -> {args.out}", file=sys.stderr)
        return

    # Build engine. Slow — model load 30-60s.
    print(f">>> launching sglang engine for {args.model}", file=sys.stderr)
    from sglang import Engine  # type: ignore
    engine_kwargs = dict(
        model_path=args.model,
        dtype="bfloat16",
        mem_fraction_static=args.mem_fraction,
        max_running_requests=args.max_running_requests,
        log_level="warning",
    )
    if args.attention_backend:
        engine_kwargs["attention_backend"] = args.attention_backend
    engine = Engine(**engine_kwargs)

    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,  # greedy — must match HF for cross-backend comparability
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    t0 = time.time()

    with args.out.open("w") as fout:
        # Skip markers go first so the row order roughly preserves subset order
        for rec in skipped:
            _emit_skip_missing_image(rec, args, fout)
            n_written += 1

        # Submit in chunks for progress visibility + crash safety. Continuous
        # batching means each chunk overlaps internally; chunk size doesn't
        # affect throughput, only how often we print.
        for chunk_start in range(0, len(requests), args.chunk_size):
            chunk = requests[chunk_start:chunk_start + args.chunk_size]
            prompts = [r[1] for r in chunk]
            image_data_list = [r[2] for r in chunk]

            outputs = _engine_generate(engine, prompts, image_data_list, sampling_params)

            for (rec, _, _), out in zip(chunk, outputs):
                pred, num_tok, prompt_len = _extract_pred(out, tokenizer)
                is_correct, scorer_used = _emit_row(rec, args, fout, pred, num_tok, prompt_len)
                if args.debug:
                    print(f"=== [{rec['id']}] mode={args.mode} ===", file=sys.stderr)
                    print(f"PREDICTION: {pred!r}", file=sys.stderr)
                    print(f"GOLD: {rec.get('answer')!r}  ->  is_correct={is_correct} via {scorer_used}",
                          file=sys.stderr)
                n_written += 1

            elapsed = time.time() - t0
            rate = n_written / max(1.0, elapsed)
            print(f"... {n_written}/{len(requests) + len(skipped)} ({rate:.2f}/s)",
                  file=sys.stderr)

    elapsed = time.time() - t0
    rate = n_written / max(1.0, elapsed)
    print(f">>> wrote {n_written} -> {args.out}  ({rate:.2f} prompts/s overall, {elapsed:.0f}s)",
          file=sys.stderr)

    # Some sglang versions need explicit shutdown to flush the engine.
    for name in ("shutdown", "stop", "close"):
        fn = getattr(engine, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
            break


if __name__ == "__main__":
    main()
