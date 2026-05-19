"""T1 punch list #1: inspect MMR1-RL dataset structure on the server.

Prints schema, one full sample row (with images rendered as type/size, not
raw bytes), question length distribution, and source/benchmark
distribution. The output tells us which fields to use as `question`,
`images`, `answer`, and `source` when we write `prep_opd_train_data.py`,
and surfaces risk #1 early (whether MathVista/MathVision/etc. prompts
are already inside MMR1-RL's training pool).

Usage on devbox:
    python scripts/data/inspect_mmr1_rl.py
    # or override path:
    MMR1_RL_DATA=/path/to/MMR1-RL python scripts/data/inspect_mmr1_rl.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter


def _load_dataset(path: str):
    """Try load_from_disk first (the T1 plan commits to this format);
    fall back to load_dataset(path) for raw HF cache directories."""
    from datasets import load_from_disk  # type: ignore

    try:
        return ("load_from_disk", load_from_disk(path))
    except Exception as e_disk:
        try:
            from datasets import load_dataset  # type: ignore
            return ("load_dataset", load_dataset(path))
        except Exception as e_hf:
            print(f"ERROR: could not load {path}", file=sys.stderr)
            print(f"  load_from_disk: {e_disk}", file=sys.stderr)
            print(f"  load_dataset:   {e_hf}", file=sys.stderr)
            sys.exit(1)


def _render(v) -> str:
    """Type-aware preview for an arbitrary field value."""
    if hasattr(v, "convert") and hasattr(v, "size"):
        return f"PIL.Image mode={v.mode} size={v.size}"
    if isinstance(v, (bytes, bytearray)):
        return f"bytes len={len(v)}"
    if isinstance(v, str):
        first = v[:300].replace("\n", "\\n")
        return f"str len={len(v)} preview={first!r}"
    if isinstance(v, (list, tuple)):
        if v and hasattr(v[0], "convert"):
            return f"list[PIL.Image] len={len(v)} first={v[0].mode}/{v[0].size}"
        if v and isinstance(v[0], (bytes, bytearray)):
            return f"list[bytes] len={len(v)} first_len={len(v[0])}"
        s = repr(v)[:300]
        return f"list len={len(v)} preview={s}"
    if isinstance(v, dict):
        keys = list(v.keys())[:10]
        return f"dict keys={keys}"
    return f"{type(v).__name__} = {v!r}"


def main() -> None:
    path = os.environ.get("MMR1_RL_DATA")
    if not path:
        for guess in (
            "/home/web_server/antispam/project/houshihao/datasets/MMR1-RL",
            os.path.expanduser("~/datasets/MMR1-RL"),
        ):
            if os.path.isdir(guess):
                path = guess
                break
    if not path:
        sys.exit("ERROR: set MMR1_RL_DATA env var (or place dataset at a known guess path)")

    print(f">>> loading {path}")
    loader, raw = _load_dataset(path)
    print(f">>> loaded via {loader}: {type(raw).__name__}")

    # Resolve to a single split.
    if hasattr(raw, "keys") and not hasattr(raw, "features"):
        # DatasetDict
        print(f">>> splits: {list(raw.keys())}")
        for k in raw.keys():
            print(f"     {k}: {len(raw[k]):,} rows")
        split_name = list(raw.keys())[0]
        d = raw[split_name]
        print(f">>> inspecting split '{split_name}'")
    else:
        d = raw
        split_name = "(single)"
        print(f">>> single-split dataset: {len(d):,} rows")

    print()
    print("=== Schema ===")
    print(f"column_names: {d.column_names}")
    print()
    print("features:")
    for col, feat in d.features.items():
        print(f"  {col}: {feat}")

    print()
    print("=== Sample 0 (full row, type-aware preview) ===")
    sample = d[0]
    for k, v in sample.items():
        print(f"  {k}: {_render(v)}")

    # Field discovery — which candidate names exist for question/image/answer/source.
    candidates = {
        "question": ("question", "problem", "prompt", "instruction", "text", "query"),
        "answer":   ("answer", "solution", "final_answer", "label", "target", "output", "response"),
        "image":    ("image", "images", "image_path", "image_file", "pixel_values"),
        "source":   ("source", "dataset", "dataset_source", "benchmark", "origin", "subset"),
        "id":       ("id", "uid", "example_id", "qid", "sample_id"),
    }
    print()
    print("=== Field name candidates present ===")
    cols = set(d.column_names)
    chosen: dict[str, str | None] = {}
    for role, names in candidates.items():
        hit = next((n for n in names if n in cols), None)
        chosen[role] = hit
        print(f"  {role}: {hit!r}  (looked for {names})")

    # Sample distribution over first N rows.
    n_check = min(2000, len(d))
    qlens: list[int] = []
    alens: list[int] = []
    sources: Counter = Counter()
    n_img_per_row: Counter = Counter()
    q_field = chosen["question"]
    a_field = chosen["answer"]
    img_field = chosen["image"]
    src_field = chosen["source"]

    print()
    print(f"=== Sweeping first {n_check} rows for distributions ===")
    for i in range(n_check):
        r = d[i]
        if q_field:
            q = r.get(q_field) or ""
            if isinstance(q, str):
                qlens.append(len(q))
        if a_field:
            a = r.get(a_field) or ""
            if isinstance(a, str):
                alens.append(len(a))
        if img_field:
            v = r.get(img_field)
            if isinstance(v, list):
                n_img_per_row[len(v)] += 1
            elif v is None:
                n_img_per_row[0] += 1
            else:
                n_img_per_row[1] += 1
        if src_field:
            v = r.get(src_field)
            if isinstance(v, str):
                sources[v] += 1

    def _quantiles(xs: list[int]) -> str:
        if not xs:
            return "(no data)"
        xs = sorted(xs)
        n = len(xs)
        return (
            f"n={n} min={xs[0]} p25={xs[n//4]} median={xs[n//2]} "
            f"p75={xs[3*n//4]} p95={xs[min(n-1, int(n*0.95))]} max={xs[-1]}"
        )

    print()
    print(f"question_length ({q_field}): {_quantiles(qlens)}")
    print(f"answer_length ({a_field}): {_quantiles(alens)}")

    print()
    print("images_per_row distribution:")
    for k in sorted(n_img_per_row.keys()):
        print(f"  {k} image(s): {n_img_per_row[k]} rows")

    print()
    if src_field and sources:
        print(f"source distribution ({src_field}, top 20):")
        for k, n in sources.most_common(20):
            print(f"  {k}: {n}  ({n/n_check*100:.1f}%)")
    else:
        print(f"source distribution: no '{src_field}' field or non-string values")
        print("  (this is a RISK #1 signal — without a source label we can only "
              "dedup against eval by question text + image hash)")


if __name__ == "__main__":
    main()
