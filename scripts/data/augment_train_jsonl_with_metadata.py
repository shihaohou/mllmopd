#!/usr/bin/env python3
"""One-shot: backfill `metadata: {"id": <id>}` into rows of an existing
train.jsonl (output of prep_opd_train_data.py).

Required by the Tier-2a off-policy KD path
(src/mllmopd/training/offline_kd_generate.py): the custom generate
function looks up offline teacher completions by `sample.metadata["id"]`,
which Uni-OPD's data loader (miles/utils/data.py:302) populates from
the JSONL row's `metadata` field. The existing prep schema writes `id`
as a top-level field, NOT inside `metadata`, so this script
backfills.

Idempotent: rows whose metadata already has "id" are skipped. Creates a
`.bak` next to the JSONL on first run; subsequent runs reuse the
existing backup.

Usage:
  python scripts/data/augment_train_jsonl_with_metadata.py \\
      --jsonl data/opd_train/v0_2k/train.jsonl

After running once, the file gains `metadata: {"id": "mmr1_rl_v0_NNNNNN"}`
on every row without touching any other fields (problem, images, etc.).
Safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jsonl", required=True, type=Path,
                    help="train.jsonl to augment in place.")
    ap.add_argument("--no-backup", action="store_true",
                    help="Skip the .bak copy (dangerous; default is to back up).")
    args = ap.parse_args()

    path: Path = args.jsonl
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    backup = path.with_suffix(path.suffix + ".bak")
    if not args.no_backup and not backup.exists():
        shutil.copy(path, backup)
        print(f"backup → {backup}")
    elif backup.exists():
        print(f"backup already exists at {backup}; skipping copy")

    rows: list[dict] = []
    n_added = 0
    n_already = 0
    n_missing_id = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            top_id = r.get("id")
            if top_id is None:
                n_missing_id += 1
                print(f"WARN: line {line_no} has no top-level 'id'; keys={list(r.keys())}",
                      file=sys.stderr)
                rows.append(r)
                continue
            md = r.get("metadata")
            if not isinstance(md, dict):
                md = {}
            if "id" in md and md["id"] == top_id:
                n_already += 1
            else:
                md["id"] = top_id
                r["metadata"] = md
                n_added += 1
            rows.append(r)

    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"augmented {n_added}/{len(rows)} rows "
          f"({n_already} already had metadata.id, "
          f"{n_missing_id} had no top-level id and were left unchanged)")


if __name__ == "__main__":
    main()
