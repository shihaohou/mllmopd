"""Convert MMR1-RL training-pool JSONL into the audit subset schema that
`tam_step1a.py --skip-student` consumes for offline TAM precompute.

Training pool format (per `data/opd_train/v0_2k/train.jsonl`):
    {
      "id": "MathVision/123",
      "problem":       "<full chat-templated prompt>",     # not used
      "question_text": "<raw user question>",              # → audit.question
      "images":        [<image_descriptor>, ...],          # → audit.image
      "answer":        "C",                                # → audit.answer
      "teacher_model": ..., "raw_problem": ..., "metadata": ...   # dropped
    }

Audit subset format expected by tam_step1a (per `data/audit/smoke_subset_v0.jsonl`):
    {
      "id":         "MathVista/266",
      "benchmark":  "MathVista",
      "image":      "/abs/path/to/image.png",  # single string
      "question":   "What shape of a leaf is similar to ...",
      "answer":     "C",
      "split_tag":  "step1a"                   # optional
    }

`images` in train.jsonl is a list whose elements may be:
  - string (absolute or relative path)
  - dict with "path" / "image_path" / "filename" key
  - dict with "bytes" / "image_bytes" key (we cannot inline-decode; skip)
This converter picks the first usable path; if none found, the row is
skipped (`--keep-no-image` to instead emit a placeholder, NOT recommended).

`benchmark` is derived from `id` prefix (`MathVision/123` → `MathVision`)
when not explicitly present.

Usage::

    python -m scripts.audit.convert_train_pool_to_audit_format \\
        --in-jsonl  data/opd_train/v0_2k/train.jsonl \\
        --out-jsonl data/opd_train/v0_2k/train_3k_audit_fmt.jsonl \\
        --limit 3000 \\
        [--image-root /abs/path]   # prepend if image paths are relative
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _extract_image_path(images, image_root: Path | None) -> str | None:
    """Best-effort extraction of an image path from train.jsonl's `images`
    list. Returns absolute path or None."""
    if not images:
        return None
    if not isinstance(images, list):
        # Single dict / string — wrap
        images = [images]
    for item in images:
        if isinstance(item, str):
            p = item.strip()
            if not p:
                continue
            full = Path(p)
            if not full.is_absolute() and image_root is not None:
                full = image_root / p
            return str(full)
        if isinstance(item, dict):
            for k in ("path", "image_path", "filename", "file", "image"):
                v = item.get(k)
                if isinstance(v, str) and v.strip():
                    full = Path(v)
                    if not full.is_absolute() and image_root is not None:
                        full = image_root / v
                    return str(full)
            # if dict has bytes-like key, we can't materialize a path here
    return None


def _benchmark_from_id(sample_id: str) -> str:
    if not sample_id or "/" not in sample_id:
        return "unknown"
    return sample_id.split("/", 1)[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in-jsonl",  type=Path, required=True)
    ap.add_argument("--out-jsonl", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap output row count (default 0 = all)")
    ap.add_argument("--image-root", type=Path, default=None,
                    help="prepend to relative image paths")
    ap.add_argument("--split-tag", type=str, default="train_pool",
                    help="written to each output row's split_tag (default: "
                         "'train_pool'; differentiates from audit's 'step1a')")
    ap.add_argument("--keep-no-image", action="store_true",
                    help="emit rows even when no image path resolved "
                         "(audit teacher pass will fail on these — only "
                         "useful for sanity counts)")
    args = ap.parse_args(argv)

    stats = Counter()
    bench_counts: Counter = Counter()
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.in_jsonl.open() as fin, args.out_jsonl.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["in"] += 1
            rec_in = json.loads(line)

            sid = rec_in.get("id")
            if not sid:
                stats["skip_no_id"] += 1
                continue

            question = (rec_in.get("question_text") or "").strip()
            if not question:
                # fallback: try to recover from `problem` template; this is
                # imperfect (template wrappers) — better to ensure the
                # training pool has question_text.
                question = (rec_in.get("raw_problem") or rec_in.get("problem") or "").strip()
            if not question:
                stats["skip_no_question"] += 1
                continue

            image_path = _extract_image_path(rec_in.get("images"), args.image_root)
            if image_path is None and not args.keep_no_image:
                stats["skip_no_image"] += 1
                continue

            benchmark = rec_in.get("benchmark") or _benchmark_from_id(sid)
            bench_counts[benchmark] += 1

            rec_out = {
                "id":         sid,
                "benchmark":  benchmark,
                "image":      image_path or "",
                "question":   question,
                "answer":     rec_in.get("answer"),
                "split_tag":  args.split_tag,
            }
            fout.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            stats["out"] += 1

            if args.limit > 0 and stats["out"] >= args.limit:
                break

    print(f">>> wrote {stats['out']}/{stats['in']} rows to {args.out_jsonl}",
          file=sys.stderr)
    print(f"    skipped: no_id={stats['skip_no_id']}  "
          f"no_question={stats['skip_no_question']}  "
          f"no_image={stats['skip_no_image']}", file=sys.stderr)
    print("    benchmark distribution:", file=sys.stderr)
    for b, c in bench_counts.most_common():
        print(f"      {b:<24s} n={c}", file=sys.stderr)

    if stats["out"] == 0:
        print("!! 0 rows emitted — check input schema; ensure rows have "
              "'id', 'question_text' (or 'raw_problem'), and 'images' with "
              "extractable paths.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
