"""Aggregate the per-pass JSONLs into per-(model, mode, benchmark) summary stats.

Reads every *.jsonl in a run dir and writes summary.json with the metric grid
that matches docs/experiment-protocol.md Level-1 table.

Usage:
    python -m mllmopd.analysis.aggregate_audit --run_dir runs/audit/<id> --out runs/audit/<id>/summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def _iter_records(p: Path):
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def aggregate(run_dir: Path) -> dict:
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for jl in sorted(run_dir.glob("*.jsonl")):
        if jl.name == "summary.json":
            continue
        for r in _iter_records(jl):
            cell = (r["model"], r["mode"], r["benchmark"])
            by_cell[cell].append(r)

    cells: list[dict] = []
    for (model, mode, bench), rows in sorted(by_cell.items()):
        n = len(rows)
        correct = [r["is_correct"] for r in rows if r["is_correct"] is not None]
        lengths = [r["num_tokens"] for r in rows]
        acc = (sum(1 for c in correct if c) / len(correct)) if correct else None
        cells.append({
            "model": model,
            "mode": mode,
            "benchmark": bench,
            "n": n,
            "accuracy": acc,
            "n_scored": len(correct),
            "tokens_mean": mean(lengths) if lengths else None,
            "tokens_median": median(lengths) if lengths else None,
            "acc_per_token": (acc / (mean(lengths) + 1e-6)) if (acc is not None and lengths) else None,
        })
    return {"cells": cells}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    summary = aggregate(args.run_dir)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f">>> {len(summary['cells'])} cells -> {args.out}")


if __name__ == "__main__":
    main()
