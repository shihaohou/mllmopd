"""One audit inference pass: load a model, run it on the audit subset under a
given perception mode, write per-prompt JSONL records.

This is GPU-only — run on the devbox in the `Uni-OPD-LMMS-Eval` env.

Usage:
    python -m mllmopd.diagnostics.run_audit_pass \
        --subset data/audit/audit_subset_v0.jsonl \
        --model MMR1/MMR1-7B-RL \
        --mode full_image \
        --out runs/audit/<run_id>/T_RL_full.jsonl

Output JSONL: one record per prompt with at minimum:
    id, benchmark, mode, model, prediction, num_tokens, prompt_len, gold,
    is_correct, finish_reason
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

# Every audit mode except text_only needs a real image; if the subset loader
# couldn't materialize one, skip that prompt with a marker row instead of
# crashing the whole pass.
_IMAGE_REQUIRED_MODES = {
    "full_image", "blank_image",
    "caption_only_blank", "image_plus_caption", "oracle_caption",
    "swap_image", "irrelevant_image",
}


def _load_subset(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_model(model_id: str):
    """Qwen2.5-VL / MMR1 loader. Falls back to AutoModelForVision2Seq if the
    transformers install doesn't expose the Qwen2.5-VL class yet (older
    versions). Falls back from flash_attention_2 to sdpa if FA2 isn't built."""
    import torch  # type: ignore
    from transformers import AutoProcessor  # type: ignore

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls  # type: ignore
        proc_kwargs = {
            "trust_remote_code": True,
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        }
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelCls  # type: ignore
        proc_kwargs = {"trust_remote_code": True}

    processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)

    common = dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    try:
        model = ModelCls.from_pretrained(model_id, attn_implementation="flash_attention_2", **common)
    except (ImportError, ValueError, RuntimeError) as e:
        print(f">>> flash_attention_2 unavailable ({e!s:.80}); falling back to sdpa", file=sys.stderr)
        model = ModelCls.from_pretrained(model_id, attn_implementation="sdpa", **common)
    model.eval()
    return processor, model


def _build_messages(rec: dict, transformed, prefix):
    """Build the chat-template message list for a single sample."""
    question = rec.get("question", "")
    if prefix:
        question = prefix + question
    return [{
        "role": "user",
        "content": [
            *([{"type": "image", "image": transformed}] if transformed is not None else []),
            {"type": "text", "text": question},
        ],
    }]


def _emit_skip_missing_image(rec: dict, args, fout):
    fout.write(json.dumps({
        "id": rec["id"], "benchmark": rec["benchmark"],
        "mode": args.mode, "model": args.model,
        "prediction": "", "num_tokens": 0, "prompt_len": 0,
        "gold": rec.get("answer"),
        "is_correct": None, "scorer": "skip_missing_image",
        "error": "missing_image",
    }, ensure_ascii=False) + "\n")
    fout.flush()
    os.fsync(fout.fileno())


def _emit_row(rec: dict, args, fout, pred: str, num_tok: int, prompt_len: int):
    is_correct, scorer_used, parse_path = scorers.score_for_benchmark(
        rec["benchmark"], pred, rec.get("answer"),
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
    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    fout.flush()
    os.fsync(fout.fileno())
    return is_correct, scorer_used


def _process_batch(processor, model, args, batch, fout):
    """batch: list of (rec, transformed, prefix). Runs one batched generate(),
    emits JSONL rows for each sample. Returns the number of rows written."""
    if not batch:
        return 0

    import torch  # noqa: F401

    chats = [
        processor.apply_chat_template(_build_messages(rec, t, p),
                                      add_generation_prompt=True, tokenize=False)
        for rec, t, p in batch
    ]
    images = [t for _, t, _ in batch if t is not None]

    # Left-pad so generated tokens line up across the batch.
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    inputs = processor(
        text=chats,
        images=images if images else None,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    eos_id = processor.tokenizer.eos_token_id

    for i, (rec, transformed, prefix) in enumerate(batch):
        gen_ids = out_ids[i, input_len:]
        # Trim at first EOS so num_tokens reflects real output, not padding.
        if eos_id is not None:
            eos_positions = (gen_ids == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                gen_ids = gen_ids[:eos_positions[0].item() + 1]
        num_tok = int(gen_ids.shape[0])
        pred = processor.decode(gen_ids, skip_special_tokens=True)
        # Per-sample real prompt length (un-padded).
        sample_attn = inputs["attention_mask"][i]
        real_prompt_len = int(sample_attn.sum().item())
        is_correct, scorer_used = _emit_row(rec, args, fout, pred, num_tok, real_prompt_len)
        if args.debug:
            print(f"=== [{rec['id']}] mode={args.mode} ===", file=sys.stderr)
            print(f"PROMPT: {chats[i]!r}", file=sys.stderr)
            print(f"PREDICTION: {pred!r}", file=sys.stderr)
            print(f"GOLD: {rec.get('answer')!r}  ->  is_correct={is_correct} via {scorer_used}",
                  file=sys.stderr)
    return len(batch)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=[
        "full_image", "blank_image", "text_only",
        "caption_only_blank", "image_plus_caption",
        "swap_image", "irrelevant_image",
        "oracle_caption",  # back-compat alias for caption_only_blank
    ])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="K prompts per generate() call. K=4-8 on 80GB A800 is "
                         "usually safe for 7B bf16 and gives 3-5x throughput.")
    ap.add_argument("--debug", action="store_true",
                    help="dump prompt + raw generation + scoring to stderr (use with --limit 2)")
    args = ap.parse_args()

    processor, model = _build_model(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    t0 = time.time()
    batch: list = []

    with args.out.open("w") as fout:
        for rec in _load_subset(args.subset):
            if args.limit and written + len(batch) >= args.limit:
                break

            image_field = rec.get("image")
            if isinstance(image_field, list) and image_field:
                # multi-image not supported yet — take the first
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
                _emit_skip_missing_image(rec, args, fout)
                written += 1
                continue

            transformed, prefix = mllm_corruptions.apply_mode(
                pil_image,
                args.mode,
                caption=rec.get("meta", {}).get("caption"),
            )
            batch.append((rec, transformed, prefix))

            if len(batch) >= args.batch_size:
                written += _process_batch(processor, model, args, batch, fout)
                batch.clear()
                if written // 25 != (written - args.batch_size) // 25:
                    rate = written / max(1.0, time.time() - t0)
                    print(f"... {written} ({rate:.2f}/s)", file=sys.stderr)

        # Trailing partial batch
        if batch:
            written += _process_batch(processor, model, args, batch, fout)
            batch.clear()

    rate = written / max(1.0, time.time() - t0)
    print(f">>> wrote {written} -> {args.out}  ({rate:.2f} prompts/s overall)")


if __name__ == "__main__":
    main()
