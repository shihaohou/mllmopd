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


def _load_subset(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_model(model_id: str):
    """Lazy-import heavy deps so module-level import works on Mac."""
    import torch  # type: ignore
    from transformers import AutoProcessor, AutoModelForVision2Seq  # type: ignore

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return processor, model


def _is_correct(pred: str, gold) -> bool | None:
    """Very loose match — exact / contains. Real eval should run lmms-eval; this
    is just to keep something computable in the JSONL for triage."""
    if gold is None:
        return None
    g = str(gold).strip().lower()
    p = pred.strip().lower()
    if not g:
        return None
    return g == p or g in p


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
    args = ap.parse_args()

    processor, model = _build_model(args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    t0 = time.time()
    with args.out.open("w") as fout:
        for rec in _load_subset(args.subset):
            if args.limit and written >= args.limit:
                break

            # Hand the per-mode image transform here. Real model-specific
            # input formatting (chat template, image placeholders, processor)
            # lives below — adapt to your processor.
            from PIL import Image  # noqa
            image_field = rec.get("image")
            pil_image = image_field if hasattr(image_field, "convert") else None
            # NB: when subset stores paths/URLs, load with mllm_corruptions.load()
            if isinstance(image_field, (str, Path)):
                pil_image = mllm_corruptions.load(image_field)

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

            row = {
                "id": rec["id"],
                "benchmark": rec["benchmark"],
                "mode": args.mode,
                "model": args.model,
                "prediction": pred,
                "num_tokens": int(gen_ids.shape[0]),
                "prompt_len": int(prompt_len),
                "gold": rec.get("answer"),
                "is_correct": _is_correct(pred, rec.get("answer")),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % 25 == 0:
                rate = written / max(1.0, time.time() - t0)
                print(f"... {written} ({rate:.2f}/s)", file=sys.stderr)

    print(f">>> wrote {written} -> {args.out}")


if __name__ == "__main__":
    main()
