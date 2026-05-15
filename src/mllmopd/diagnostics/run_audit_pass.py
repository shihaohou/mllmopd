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
import sys
import time
from pathlib import Path

from mllmopd.data import mllm_corruptions
from mllmopd.diagnostics import scorers


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=[
        "full_image", "blank_image", "text_only",
        "oracle_caption", "swap_image", "irrelevant_image",
    ])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--debug", action="store_true",
                    help="dump prompt + raw generation + scoring to stderr (use with --limit 2)")
    args = ap.parse_args()

    processor, model = _build_model(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    t0 = time.time()
    with args.out.open("w") as fout:
        for rec in _load_subset(args.subset):
            if args.limit and written >= args.limit:
                break

            image_field = rec.get("image")
            if isinstance(image_field, list) and image_field:
                # multi-image not supported yet — take the first
                image_field = image_field[0]
            if isinstance(image_field, (str, Path)):
                pil_image = mllm_corruptions.load(image_field)
            elif hasattr(image_field, "convert"):
                pil_image = image_field
            else:
                pil_image = None

            transformed, prefix = mllm_corruptions.apply_mode(
                pil_image,
                args.mode,
                caption=rec.get("meta", {}).get("caption"),
            )
            question = rec.get("question", "")
            if prefix:
                question = prefix + question

            messages = [{
                "role": "user",
                "content": [
                    *([{"type": "image", "image": transformed}] if transformed is not None else []),
                    {"type": "text", "text": question},
                ],
            }]
            chat = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=[chat],
                               images=[transformed] if transformed is not None else None,
                               return_tensors="pt").to(model.device)

            out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = out_ids[0, prompt_len:]
            pred = processor.decode(gen_ids, skip_special_tokens=True)

            is_correct, scorer_used = scorers.score_for_benchmark(
                rec["benchmark"], pred, rec.get("answer"),
            )
            row = {
                "id": rec["id"],
                "benchmark": rec["benchmark"],
                "mode": args.mode,
                "model": args.model,
                "prediction": pred,
                "num_tokens": int(gen_ids.shape[0]),
                "prompt_len": int(prompt_len),
                "gold": rec.get("answer"),
                "is_correct": is_correct,
                "scorer": scorer_used,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if args.debug:
                print(f"=== [{rec['id']}] mode={args.mode} ===", file=sys.stderr)
                print(f"PROMPT: {chat!r}", file=sys.stderr)
                print(f"PREDICTION: {pred!r}", file=sys.stderr)
                print(f"GOLD: {rec.get('answer')!r}  ->  is_correct={is_correct} via {scorer_used}",
                      file=sys.stderr)
            if written % 25 == 0:
                rate = written / max(1.0, time.time() - t0)
                print(f"... {written} ({rate:.2f}/s)", file=sys.stderr)

    print(f">>> wrote {written} -> {args.out}")


if __name__ == "__main__":
    main()
