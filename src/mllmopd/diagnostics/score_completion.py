"""H2 audit: forced-decoding logprob extraction for "visual dependency".

For each completion in a source JSONL (output of `run_audit_pass`), compute
per-token logprobs of the completion under the same model in two image
conditions: full_image and blank_image. The per-token difference

    vd(t) = logp(t | prefix, full_image) - logp(t | prefix, blank_image)

is the visual dependency at that token (positive = the image actively
raises the probability of this token). This unlocks H2 audit: does the
OPD reward / KL signal concentrate on tokens with high VD (good — the
visually-grounded tokens are what gets supervised) or low VD (bad — OPD
loss mass falls on language-model-prior tokens that the image didn't help
generate)?

Source jsonl provides the completion (the model's own full-image
generation). Subset jsonl provides the image. They are joined by `id`.

Usage:
    python -m mllmopd.diagnostics.score_completion \\
        --subset data/audit/smoke_subset_v0.jsonl \\
        --source runs/audit/smoke500/T_RL_full.jsonl \\
        --model /path/to/MMR1-7B-RL \\
        --out runs/audit/smoke500/T_RL_full.scored.jsonl

For cross-model scoring (e.g., student logp under teacher's completion),
point `--source` at the teacher's jsonl but `--model` at the student.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from mllmopd.data import mllm_corruptions
from mllmopd.diagnostics.run_audit_pass import _build_model, _load_subset


def _build_prefix_and_full(processor, rec, image, response_text: str) -> tuple[str, str]:
    """Build (prefix_text, full_text) where:
      - prefix_text = chat template with assistant generation prompt opened, no content
      - full_text   = prefix_text + assistant's actual response_text appended

    The processor will tokenize each and expand image patches identically
    because the image is the same; only the trailing text differs."""
    messages = [{
        "role": "user",
        "content": [
            *([{"type": "image", "image": image}] if image is not None else []),
            {"type": "text", "text": rec.get("question", "")},
        ],
    }]
    prefix_text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    full_text = prefix_text + response_text
    return prefix_text, full_text


def _logp_for_response(model, processor, rec, response_text, image):
    """Forward-pass `model` on prefix+response with `image`, then extract
    per-token logprobs of the response. Returns:
        (response_token_ids, response_logp, response_start_idx) on success
        None on tokenization mismatch (rare; logged to stderr)."""
    import torch
    import torch.nn.functional as F

    prefix_text, full_text = _build_prefix_and_full(processor, rec, image, response_text)
    proc_kwargs = dict(return_tensors="pt")
    if image is not None:
        proc_kwargs["images"] = image

    prefix_inputs = processor(text=prefix_text, **proc_kwargs).to(model.device)
    full_inputs = processor(text=full_text, **proc_kwargs).to(model.device)

    response_start = prefix_inputs.input_ids.shape[1]
    full_ids = full_inputs.input_ids[0]
    if response_start >= full_ids.shape[0]:
        return None  # empty response after tokenization

    with torch.no_grad():
        out = model(**full_inputs)
    # logits shape (1, T, V); logits[i-1] predicts token at position i
    logp = F.log_softmax(out.logits[0].float(), dim=-1)

    response_token_ids = full_ids[response_start:].tolist()
    response_logp = [
        logp[response_start + j - 1, full_ids[response_start + j]].item()
        for j in range(len(response_token_ids))
    ]
    return response_token_ids, response_logp, response_start


def _decode_per_token(tokenizer, token_ids):
    """Map each token id to its decoded string, for the visualization layer."""
    return [tokenizer.decode([t]) for t in token_ids]


def _load_source_index(path: Path) -> dict[str, dict]:
    idx: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx[r["id"]] = r
    return idx


def _load_subset_image(rec, image_dir_root: Path | None = None):
    image_field = rec.get("image")
    if isinstance(image_field, list) and image_field:
        image_field = image_field[0]
    if isinstance(image_field, (str, Path)):
        try:
            return mllm_corruptions.load(image_field)
        except Exception as e:
            print(f"!! could not load image {image_field}: {e}", file=sys.stderr)
            return None
    if hasattr(image_field, "convert"):
        return image_field
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subset", required=True, type=Path,
                    help="audit subset jsonl (provides image + question by id)")
    ap.add_argument("--source", required=True, type=Path,
                    help="existing run_audit_pass jsonl whose `prediction`s are scored")
    ap.add_argument("--model", required=True,
                    help="model to score under (typically the source-completion's own model)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug", action="store_true",
                    help="print top/bottom visual-dependency tokens per row")
    args = ap.parse_args()

    print(f">>> loading model {args.model}", file=sys.stderr)
    processor, model = _build_model(args.model)
    tokenizer = processor.tokenizer

    src_by_id = _load_source_index(args.source)
    print(f">>> loaded {len(src_by_id)} source completions from {args.source}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    t0 = time.time()

    with args.out.open("w") as fout:
        for rec in _load_subset(args.subset):
            if args.limit and written >= args.limit:
                break
            rid = rec["id"]
            src = src_by_id.get(rid)
            if src is None:
                continue
            response = (src.get("prediction") or "").strip()
            if not response:
                continue

            pil = _load_subset_image(rec)
            if pil is None:
                skipped += 1
                continue

            full_img, _ = mllm_corruptions.apply_mode(pil, "full_image")
            blank_img, _ = mllm_corruptions.apply_mode(pil, "blank_image")

            full_result = _logp_for_response(model, processor, rec, response, full_img)
            if full_result is None:
                skipped += 1
                continue
            full_tokens, full_logp, _ = full_result

            blank_result = _logp_for_response(model, processor, rec, response, blank_img)
            if blank_result is None:
                skipped += 1
                continue
            blank_tokens, blank_logp, _ = blank_result

            # Sanity: the response token sequence should match across image
            # conditions (same response, same prefix structure modulo image
            # patches — which appear before the response). If it doesn't,
            # the chat template / image patch count differs and the rows
            # can't be aligned; skip.
            if full_tokens != blank_tokens:
                skipped += 1
                if args.debug:
                    print(f"!! [{rid}] token mismatch full={len(full_tokens)} blank={len(blank_tokens)}",
                          file=sys.stderr)
                continue

            n = len(full_tokens)
            vd = [a - b for a, b in zip(full_logp, blank_logp)]
            token_strs = _decode_per_token(tokenizer, full_tokens)

            row = {
                "id": rid,
                "benchmark": rec.get("benchmark"),
                "model": args.model,
                "source": str(args.source),
                "n_tokens": n,
                "completion_tokens": full_tokens,
                "token_strs": token_strs,
                "logp_full": full_logp,
                "logp_blank": blank_logp,
                "visual_dependency": vd,
                "logp_full_sum": sum(full_logp),
                "logp_blank_sum": sum(blank_logp),
                "vd_sum": sum(vd),
                "vd_mean": (sum(vd) / n) if n else None,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            written += 1

            if args.debug:
                top_high = sorted(range(n), key=lambda i: -vd[i])[:5]
                top_low = sorted(range(n), key=lambda i: vd[i])[:5]
                print(f"=== {rid} (n={n}, vd_sum={row['vd_sum']:.2f}, vd_mean={row['vd_mean']:.3f}) ===",
                      file=sys.stderr)
                print(f"  highest VD: " + ", ".join(
                    f"{token_strs[i]!r}({vd[i]:+.2f})" for i in top_high
                ), file=sys.stderr)
                print(f"  lowest  VD: " + ", ".join(
                    f"{token_strs[i]!r}({vd[i]:+.2f})" for i in top_low
                ), file=sys.stderr)

            if written % 25 == 0:
                rate = written / max(1.0, time.time() - t0)
                print(f"... {written} ({rate:.2f}/s, skipped={skipped})", file=sys.stderr)

    elapsed = time.time() - t0
    rate = written / max(1.0, elapsed)
    print(f">>> wrote {written} -> {args.out}  ({rate:.2f} prompts/s, {elapsed:.0f}s, skipped={skipped})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
