"""Aggregate the per-pass JSONLs into per-(model, mode, benchmark) summary stats.

Reads every *.jsonl in a run dir and writes summary.json with the metric grid
that matches docs/experiment-protocol.md Level-1 table. Also re-scores every
record in-memory against the current scorer — so improvements to
`mllmopd.diagnostics.scorers` immediately affect accuracy numbers without
needing to re-run inference. `n_rescore_changed` per cell exposes the impact.

Usage:
    python -m mllmopd.analysis.aggregate_audit --run_dir runs/audit/<id> --out runs/audit/<id>/summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from mllmopd.diagnostics import scorers

# Records at or above this many tokens are treated as "hit max_new_tokens".
# Default max_new_tokens for the audit pipeline is 1024; allow a small slop
# for early-stop tokenizations.
_MAX_TOKEN_HIT_THRESHOLD = 1020


def _iter_records(p: Path):
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _rescore(rec: dict) -> tuple[bool | None, str, str]:
    """Re-score a record using the current scorer; safe to call on rows from
    old jsonls that don't yet carry `parse_path` or `choices`. When the row
    has `choices`, the MCQ parser can recover option-text-only conclusions."""
    if rec.get("scorer") in {"skip_missing_image", "skip_empty_gold"}:
        return rec.get("is_correct"), rec.get("scorer"), rec.get("scorer")
    pred = rec.get("prediction") or ""
    gold = rec.get("gold")
    benchmark = rec.get("benchmark", "")
    choices = rec.get("choices")
    return scorers.score_for_benchmark(benchmark, pred, gold, choices=choices)


def _paired_full_blank(by_cell: dict, correct_map: dict) -> list[dict]:
    """Per-(model, benchmark) 2×2 contingency for full_image vs blank_image.

    `correct_map[(model, mode, bench, id)] -> bool` carries the re-scored
    is_correct so the paired table reflects the same scorer that drove
    cells[].accuracy.
    """
    by_mb_mode: dict = defaultdict(dict)
    for (model, mode, bench), rows in by_cell.items():
        if mode not in {"full_image", "blank_image"}:
            continue
        by_mb_mode[(model, bench)][mode] = {
            r["id"]: correct_map[(model, mode, bench, r["id"])]
            for r in rows
            if correct_map[(model, mode, bench, r["id"])] is not None
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
    correct_map: dict[tuple, bool | None] = {}

    for (model, mode, bench), rows in sorted(by_cell.items()):
        n = len(rows)
        lengths = [r["num_tokens"] for r in rows]

        # Re-score and tally everything in one pass.
        correct_new: list[bool] = []
        scorer_breakdown: dict[str, int] = defaultdict(int)
        parse_path_breakdown: dict[str, int] = defaultdict(int)
        rescore_changed = 0
        hit_max = 0
        refusal = 0
        high_conf = 0
        n_mcq = 0

        for r in rows:
            is_c, scorer_used, parse_path = _rescore(r)
            correct_map[(model, mode, bench, r["id"])] = is_c
            if is_c is not None:
                correct_new.append(is_c)
            scorer_breakdown[scorer_used or "unknown"] += 1
            parse_path_breakdown[parse_path or "unknown"] += 1
            if is_c != r.get("is_correct"):
                rescore_changed += 1
            if r.get("num_tokens", 0) >= _MAX_TOKEN_HIT_THRESHOLD:
                hit_max += 1
            if scorers.is_refusal(r.get("prediction") or ""):
                refusal += 1
            if scorer_used == "mcq_letter":
                n_mcq += 1
                if parse_path in scorers.HIGH_CONFIDENCE_PATHS:
                    high_conf += 1

        acc = (sum(1 for c in correct_new if c) / len(correct_new)) if correct_new else None
        cells.append({
            "model": model,
            "mode": mode,
            "benchmark": bench,
            "n": n,
            "accuracy": acc,
            "n_scored": len(correct_new),
            "tokens_mean": mean(lengths) if lengths else None,
            "tokens_median": median(lengths) if lengths else None,
            "acc_per_token": (acc / (mean(lengths) + 1e-6)) if (acc is not None and lengths) else None,
            "scorers": dict(scorer_breakdown),
            "parse_paths": dict(parse_path_breakdown),
            "rescore_changed": rescore_changed,
            "hit_max_tokens_rate": hit_max / n if n else None,
            "refusal_rate": refusal / n if n else None,
            "mcq_high_conf_rate": (high_conf / n_mcq) if n_mcq else None,
        })
    return {
        "cells": cells,
        "paired_full_blank": _paired_full_blank(by_cell, correct_map),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    summary = aggregate(args.run_dir)
    args.out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    total_changed = sum(c.get("rescore_changed", 0) for c in summary["cells"])
    print(f">>> {len(summary['cells'])} cells, "
          f"{len(summary['paired_full_blank'])} paired (model,bench), "
          f"{total_changed} rows re-scored differently from on-disk -> {args.out}")


if __name__ == "__main__":
    main()
