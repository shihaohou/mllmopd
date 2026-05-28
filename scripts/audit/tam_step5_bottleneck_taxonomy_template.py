"""Step 5 — Bottleneck taxonomy annotation TEMPLATE generator.

Pass 4 by itself decides whether vanilla OPD shifts self-trajectory
teacher attention. Even if Pass 4 lands as `self_flat × cross_neutral`,
the §13.4 / §6.4 question "which bottleneck dominates the OPD_improved
bucket?" remains open: the audit only sees attention divergence, not
the reasoning step at which S0 actually failed.

This script is the **scaffolding** for the 30-50-sample human pass.
It does NOT label anything itself — it builds a JSONL template that a
human annotator fills in. Schema per row (annotator-facing)::

    {
      "id":            str,                # sample id (key)
      "bucket":        str,                # always "OPD_improved" for this pass
      "benchmark":     str,
      "image_path":    str,
      "question":      str,
      "gold_answer":   str,
      "s0_response":   str,                # full S0 rollout (the failed one)
      "s1_response":   str,                # full S1 rollout (OPD recovered)
      "s0_correct":    bool,
      "s1_correct":    bool,

      # --- Filled in by annotator ---
      "layer_label": null,                 # one of: "L1", "L2", "L3", "L4"
                                           #   L1 = evidence-SELECTION   (looked at wrong region)
                                           #   L2 = evidence-INTERPRETATION (saw right region, mis-recognised)
                                           #   L3 = evidence-USE         (saw + recognised, mis-reasoned)
                                           #   L4 = language PRIOR       (hallucinated from text alone)
      "confidence":  null,                 # "high" | "medium" | "low"
      "notes":       ""                    # one-sentence justification
    }

The L1/L2/L3/L4 ladder is documented in
`docs/step5-evidence-alignment-design.md` §13.4 and
`docs/figures/step5/step5-results.md` §6.4.

This template is the input to a future analyzer script that computes
the bucket-level distribution {L1%, L2%, L3%, L4%} + bootstrap CI. The
analyzer is NOT in this repo yet (it's blocked on the human annotation).

Usage::

    # Default: take a random N from OPD_improved
    python -m scripts.audit.tam_step5_bottleneck_taxonomy_template \\
        --alignment runs/audit/tam_step5_<TS>/alignment.jsonl \\
        --out data/audit/bottleneck_taxonomy_template.jsonl \\
        --n 50

    # Or hand-pick: pass the same ids that drive qualitative_cases.jsonl
    # (so taxonomy overlaps with the cases the paper figures show).
    python -m scripts.audit.tam_step5_bottleneck_taxonomy_template \\
        --alignment runs/audit/tam_step5_<TS>/alignment.jsonl \\
        --restrict-ids-from docs/figures/step5/qualitative_cases.jsonl \\
        --out data/audit/bottleneck_taxonomy_template.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def _load_alignment_rows(path: Path, bucket: str = "OPD_improved") -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("bucket") != bucket:
                continue
            rows.append(r)
    return rows


def _load_restrict_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "id" in r:
                ids.add(r["id"])
    return ids


def _build_template_row(r: dict) -> dict:
    return {
        "id":          r.get("id"),
        "bucket":      r.get("bucket"),
        "benchmark":   r.get("benchmark"),
        "image_path":  r.get("image_path"),
        "question":    r.get("question"),
        "gold_answer": r.get("answer"),
        "s0_response": r.get("s0_response_text"),
        "s1_response": r.get("s1_response_text"),
        "s0_correct":  r.get("s0_correct"),
        "s1_correct":  r.get("s1_correct"),

        # Annotator fields — left null/empty
        "layer_label": None,
        "confidence":  None,
        "notes":       "",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alignment", type=Path, required=True,
                    help="alignment.jsonl from tam_step5_evidence_alignment")
    ap.add_argument("--out", type=Path, required=True,
                    help="JSONL template path for human annotators")
    ap.add_argument("--bucket", default="OPD_improved",
                    help="Bucket to draw from (default OPD_improved)")
    ap.add_argument("--n", type=int, default=50,
                    help="Sample size (default 50). Ignored when "
                         "--restrict-ids-from is set.")
    ap.add_argument("--restrict-ids-from", type=Path, default=None,
                    help="JSONL with `id` field; only ids present in it "
                         "are kept. Useful for aligning the taxonomy with "
                         "the qualitative figure set.")
    ap.add_argument("--seed", type=int, default=20260528)
    args = ap.parse_args(argv)

    if not args.alignment.exists():
        print(f"!! alignment not found: {args.alignment}", file=sys.stderr)
        return 2

    rows = _load_alignment_rows(args.alignment, bucket=args.bucket)
    print(f">>> loaded {len(rows)} {args.bucket} rows from {args.alignment.name}",
          file=sys.stderr)
    if not rows:
        print(f"!! no rows in bucket={args.bucket}", file=sys.stderr)
        return 3

    if args.restrict_ids_from is not None:
        ids = _load_restrict_ids(args.restrict_ids_from)
        rows = [r for r in rows if r.get("id") in ids]
        print(f">>> restricted to {len(rows)} ids from "
              f"{args.restrict_ids_from.name}", file=sys.stderr)
    else:
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.n]
        print(f">>> randomly sampled n={len(rows)} (seed={args.seed})",
              file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fout:
        for r in rows:
            fout.write(json.dumps(_build_template_row(r),
                                  ensure_ascii=False) + "\n")
    print(f">>> wrote template with {len(rows)} rows to {args.out}",
          file=sys.stderr)
    print(">>> Annotator guide:", file=sys.stderr)
    print("    Edit `layer_label` to one of L1 / L2 / L3 / L4", file=sys.stderr)
    print("    (definitions: §13.4 in docs/step5-evidence-alignment-design.md)",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
