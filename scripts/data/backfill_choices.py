"""Backfill `choices` into existing audit JSONLs by joining on `id`.

Older audit runs (e.g. smoke500 captured 2026-05-19) emit per-record JSONLs
without a `choices` field, which means the upgraded mcq_letter scorer can't
recover predictions that conclude with option text only ("Serrulate is the
correct answer"). This script joins each run JSONL to its source subset by
prompt id, copies `choices`/`options` from the subset record's `meta`, and
rewrites the JSONL with the new field added (atomic write).

Idempotent: rows that already have `choices` are passed through untouched.

Usage:
    python scripts/data/backfill_choices.py \
        --subset data/audit/audit_subset_v0.jsonl \
        --run_dir runs/audit/smoke500
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _load_subset_choices(subset_path: Path) -> dict[str, list]:
    """Return {id -> choices_list} for every subset row that has choices."""
    out: dict[str, list] = {}
    with subset_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("meta") or {}
            choices = (
                rec.get("choices")
                or rec.get("options")
                or meta.get("choices")
                or meta.get("options")
            )
            if choices:
                out[rec["id"]] = list(choices)
    return out


def _backfill_one(jsonl: Path, id_to_choices: dict[str, list]) -> tuple[int, int]:
    """Returns (n_total, n_added). Writes to a .tmp sibling then renames."""
    tmp = jsonl.with_suffix(jsonl.suffix + ".tmp")
    n_total = 0
    n_added = 0
    with jsonl.open() as fin, tmp.open("w") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            if "choices" not in r:
                choices = id_to_choices.get(r.get("id"))
                if choices:
                    r["choices"] = choices
                    n_added += 1
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
        fout.flush()
        os.fsync(fout.fileno())
    os.replace(tmp, jsonl)
    return n_total, n_added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True, type=Path,
                    help="path to audit subset jsonl (e.g. data/audit/audit_subset_v0.jsonl)")
    ap.add_argument("--run_dir", required=True, type=Path,
                    help="audit run directory containing *.jsonl pass files")
    args = ap.parse_args()

    if not args.subset.is_file():
        raise SystemExit(f"subset not found: {args.subset}")
    if not args.run_dir.is_dir():
        raise SystemExit(f"run_dir not found: {args.run_dir}")

    id_to_choices = _load_subset_choices(args.subset)
    print(f">>> Loaded {len(id_to_choices)} subset rows with choices from {args.subset}")

    total_rows = 0
    total_added = 0
    for jl in sorted(args.run_dir.glob("*.jsonl")):
        if jl.name == "summary.json":
            continue
        n_total, n_added = _backfill_one(jl, id_to_choices)
        total_rows += n_total
        total_added += n_added
        print(f"    {jl.name:30s} rows={n_total:>5}  +choices={n_added}")
    print(f">>> Done. Backfilled choices for {total_added}/{total_rows} rows across {args.run_dir}")


if __name__ == "__main__":
    main()
