"""Build the Step 1 calibration sample subset (per v0.1.2 schema §"Sample selection").

Reads from one or more source JSONL files (level1 audit subset is the
canonical multi-benchmark source; smoke_subset_v0.jsonl works as a
narrow fallback). Picks a stratified sample per benchmark, applies the
v0.1.2 distribution (ChartQA up-weighted; text_only replaced by
blank_image / irrelevant_image with vision tokens), and writes a
tam_calibration_subset_v0.jsonl ready to be consumed by tam_step1a.py.

Usage::

    python scripts/audit/tam_step1_subset.py \\
        --source data/audit/level1_subset_v0.jsonl \\
        [--opd-target-ids runs/audit/<t1_eval>/opd_target_ids_T1_3_vs_T1_0.json] \\
        --out data/audit/tam_calibration_subset_v0.jsonl \\
        [--n-opd-target 80 --n-chartqa 70 --n-hallusion 25 --n-pope 20 \\
         --n-neg-control 10 --seed 1729]

Output schema (one row per sample):
  { id, benchmark, image, question, answer, split_tag, image_corruption?,
    swap_with_id?  }

split_tag is one of:
  "opd_target"          — vision-critical, listed in OPD-target id JSON
  "chartqa_collapse"    — ChartQA samples (no T1-3-collapse curation yet)
  "hallusionbench"      — HallusionBench (full subset if available)
  "pope"                — POPE_adversarial; localizable objects
  "neg_control_blank"   — same question but image will be a blank PIL
  "neg_control_swap"    — same question but image is swapped with another sample
                          (id stored in swap_with_id so the runner can resolve)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_opd_target_ids(path: Path) -> set[str]:
    """Load the OPD-target ID set. The on-cluster file is a dict
    keyed by benchmark; each value is a list[str] of {benchmark}/{id}."""
    if not path or not path.exists():
        return set()
    with path.open() as f:
        data = json.load(f)
    out: set[str] = set()
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                out.update(str(x) for x in v)
    elif isinstance(data, list):
        out.update(str(x) for x in data)
    return out


def _pick(samples: list[dict], n: int, rng: random.Random,
          split_tag: str) -> list[dict]:
    if not samples or n <= 0:
        return []
    if n >= len(samples):
        chosen = list(samples)
    else:
        chosen = rng.sample(samples, n)
    out = []
    for s in chosen:
        s2 = dict(s)
        s2["split_tag"] = split_tag
        out.append(s2)
    return out


def _build_neg_controls(base: list[dict], n_blank: int, n_swap: int,
                       rng: random.Random) -> list[dict]:
    """Take items from `base` and produce blank-image / swap-image variants.
    blank: image will be replaced with a uniform blank PIL at runtime.
    swap : image is swapped with another base sample (cross-pair)."""
    if not base or (n_blank <= 0 and n_swap <= 0):
        return []
    out: list[dict] = []
    pool = list(base)
    rng.shuffle(pool)
    # blank-image variants
    for s in pool[:n_blank]:
        s2 = dict(s)
        s2["id"] = f"{s['id']}__neg_blank"
        s2["split_tag"] = "neg_control_blank"
        s2["image_corruption"] = "blank_image"
        out.append(s2)
    # swap-image variants: pair samples (i,i+1) and swap their images
    remaining = pool[n_blank:n_blank + 2 * n_swap]
    for i in range(0, min(2 * n_swap, len(remaining)) - 1, 2):
        a, b = remaining[i], remaining[i + 1]
        sa = dict(a); sa["id"] = f"{a['id']}__neg_swap_{b['id'].replace('/', '_')}"
        sa["split_tag"] = "neg_control_swap"
        sa["image_corruption"] = "swap_image"
        sa["swap_with_id"] = b["id"]
        sa["swap_with_image"] = b["image"]
        out.append(sa)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, action="append", required=True,
                    help="Source JSONL (repeat for multiple). Each row needs at "
                         "least id, benchmark, image, question, answer.")
    ap.add_argument("--opd-target-ids", type=Path, default=None,
                    help="JSON file with the OPD-target IDs (per benchmark "
                         "dict; or a flat list). Optional.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL path.")
    ap.add_argument("--n-opd-target",    type=int, default=80)
    ap.add_argument("--n-chartqa",       type=int, default=70)
    ap.add_argument("--n-hallusion",     type=int, default=25)
    ap.add_argument("--n-pope",          type=int, default=20)
    ap.add_argument("--n-neg-blank",     type=int, default=5)
    ap.add_argument("--n-neg-swap",      type=int, default=5)
    ap.add_argument("--seed",            type=int, default=1729)
    args = ap.parse_args(argv)

    # Load
    all_rows: list[dict] = []
    for src in args.source:
        rows = _load_jsonl(src)
        print(f">>> loaded {len(rows):>5} from {src}", file=sys.stderr)
        all_rows.extend(rows)
    # Deduplicate by id (keep first occurrence)
    seen: set[str] = set()
    deduped = []
    for r in all_rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            deduped.append(r)
    print(f">>> {len(deduped)} unique ids after dedup", file=sys.stderr)

    opd_target_ids = _load_opd_target_ids(args.opd_target_ids) if args.opd_target_ids else set()
    if opd_target_ids:
        print(f">>> opd_target_ids: {len(opd_target_ids)} loaded", file=sys.stderr)

    bench_groups: dict[str, list[dict]] = defaultdict(list)
    for r in deduped:
        bench_groups[r.get("benchmark", "_unknown")].append(r)
    print(">>> source benchmark histogram:", file=sys.stderr)
    for b, items in sorted(bench_groups.items(), key=lambda kv: -len(kv[1])):
        print(f"      {b:30s} {len(items)}", file=sys.stderr)

    rng = random.Random(args.seed)
    selected: list[dict] = []

    # --- opd_target ---
    if opd_target_ids:
        opd_pool = [r for r in deduped if r["id"] in opd_target_ids]
        print(f">>> opd_target pool: {len(opd_pool)}", file=sys.stderr)
        selected.extend(_pick(opd_pool, args.n_opd_target, rng, "opd_target"))

    # --- ChartQA (placeholder for "ChartQA collapse"; no curated list yet) ---
    chartqa_pool = bench_groups.get("ChartQA", [])
    # exclude rows already picked into opd_target
    chosen_ids = {s["id"] for s in selected}
    chartqa_pool = [r for r in chartqa_pool if r["id"] not in chosen_ids]
    selected.extend(_pick(chartqa_pool, args.n_chartqa, rng, "chartqa_collapse"))

    # --- HallusionBench (full or up to n) ---
    chosen_ids = {s["id"] for s in selected}
    hall_pool = [r for r in bench_groups.get("HallusionBench", [])
                 if r["id"] not in chosen_ids]
    selected.extend(_pick(hall_pool, args.n_hallusion, rng, "hallusionbench"))

    # --- POPE_adversarial ---
    chosen_ids = {s["id"] for s in selected}
    pope_pool = [r for r in bench_groups.get("POPE_adversarial", [])
                 if r["id"] not in chosen_ids]
    selected.extend(_pick(pope_pool, args.n_pope, rng, "pope"))

    # --- Negative controls ---
    neg_base = list(selected)  # base on already-picked items so we keep alignment
    rng.shuffle(neg_base)
    selected.extend(_build_neg_controls(neg_base, args.n_neg_blank,
                                         args.n_neg_swap, rng))

    # Output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for s in selected:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Report
    counts = Counter(s["split_tag"] for s in selected)
    print(f"\n>>> wrote {len(selected)} samples to {args.out}", file=sys.stderr)
    for tag, n in counts.most_common():
        print(f"      {tag:25s} {n}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
