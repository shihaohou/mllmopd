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


def _paired_full_blank(by_cell: dict) -> list[dict]:
    """Per-(model, benchmark) 2×2 contingency for full_image vs blank_image.

    Counts how many prompts each model gets right in both modes, only with image,
    only with blank, or in neither. `image_lift_rate` and `blank_shortcut_rate`
    are the more informative metrics than the cell-level accuracy gap because
    they are computed on the same prompt ids.
    """
    by_mb_mode: dict = defaultdict(dict)
    for (model, mode, bench), rows in by_cell.items():
        if mode not in {"full_image", "blank_image"}:
            continue
        by_mb_mode[(model, bench)][mode] = {
            r["id"]: r["is_correct"] for r in rows if r["is_correct"] is not None
        }

    out: list[dict] = []
    for (model, bench), mode_map in sorted(by_mb_mode.items()):
        full = mode_map.get("full_image", {})
        blank = mode_map.get("blank_image", {})
        ids = sorted(set(full) & set(blank))
        if not ids:
            continue
        c = {"both_correct": 0, "full_only": 0, "blank_only": 0, "both_wrong": 0}
        for i in ids:
            f, b = full[i], blank[i]
            if f and b:
                c["both_correct"] += 1
            elif f and not b:
                c["full_only"] += 1
            elif not f and b:
                c["blank_only"] += 1
            else:
                c["both_wrong"] += 1
        n = len(ids)
        out.append({
            "model": model,
            "benchmark": bench,
            "n_paired": n,
            **c,
            "image_lift_rate": c["full_only"] / n,
            "blank_shortcut_rate": c["blank_only"] / n,
        })
    return out


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
        scorer_breakdown: dict = defaultdict(int)
        for r in rows:
            scorer_breakdown[r.get("scorer", "unknown")] += 1
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
            "scorers": dict(scorer_breakdown),
        })
    return {
        "cells": cells,
        "paired_full_blank": _paired_full_blank(by_cell),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    summary = aggregate(args.run_dir)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f">>> {len(summary['cells'])} cells, "
          f"{len(summary['paired_full_blank'])} paired (model,bench) "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
