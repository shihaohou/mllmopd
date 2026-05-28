"""Step 5 narrative-driven sample picker.

The default `pick_qualitative` in `tam_step5_analyzer.py` ranks by `|ΔJS|`
within each bucket — direction-agnostic and bucket-balanced. That works
for "show me some cases per bucket" but cannot surface narrative-specific
extremes (e.g. "OPD改对答案但evidence退化").

This script does the per-sample metric pass over alignment.jsonl, then
emits id lists for each narrative + a full per-sample audit CSV. The id
lists can then be piped into the existing picker via
`--qualitative-restrict-ids`, which will compute tok_indices using the
same C_local + entropy logic as the audit picker.

Two narratives, both restricted to bucket=OPD_improved (s0=False, s1=True):

  N1 — "answer flipped, evidence regressed":
       ΔJS < 0 OR ΔIoU < 0 (i.e. S1 evidence moved AWAY from teacher).
       Sorted by joint regression score `min(ΔJS, ΔIoU)` ASC.
       Use these to argue: OPD改了答案但attention跑偏 → EA-OPD motivation.

  N2 — "answer flipped, evidence invariant":
       |ΔJS| < eps AND |ΔIoU| < eps. Sorted by sum of |Δ| ASC (tightest
       to zero first). Use these to argue: OPD改的是reasoning而不是evidence
       (HallusionBench/412 is one of these).

Usage::

    python -m scripts.audit.tam_step5_narrative_picks \\
        --alignment runs/audit/tam_step5_<TS>/alignment.jsonl \\
        --out-dir   docs/figures/step5/narrative_picks/ \\
        [--top-k 20] [--eps 0.005] [--reliability-thresh 0.998]

Outputs (under --out-dir):
    narrative_audit.csv             — every OPD_improved sample with metrics
    narrative1_regression_ids.txt   — N1 top-K ids, comma-separated
    narrative2_invariant_ids.txt    — N2 top-K ids, comma-separated
    narrative_summary.md            — human-readable top-K table per narrative

Pipeline to render::

    # Step A — pick ids by narrative (this script)
    python -m scripts.audit.tam_step5_narrative_picks ...

    # Step B — compute tok_indices for those ids (reuses audit picker)
    python -m mllmopd.analysis.tam_step5_analyzer --picks-only \\
        --alignment <same alignment.jsonl> \\
        --out-dir   docs/figures/step5/narrative_picks/n1/ \\
        --qualitative-restrict-ids "$(cat narrative1_regression_ids.txt)"

    # Step C — render triplet PNGs
    python -m scripts.audit.tam_step5_render_overlays \\
        --alignment <same alignment.jsonl> \\
        --picks docs/figures/step5/narrative_picks/n1/qualitative_cases.jsonl \\
        --out-dir docs/figures/step5/narrative_picks/n1/overlays/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from mllmopd.analysis.tam_step5_analyzer import (
    RELIABILITY_THRESH_DEFAULT,
    _per_sample_means,
    load_alignment,
)


def _summarize_one(r: dict, s: dict) -> dict:
    """Per-sample summary row for the audit CSV."""
    return {
        "id":            r["id"],
        "bucket":        r.get("bucket", "unknown"),
        "benchmark":     r.get("benchmark"),
        "s0_correct":    r.get("s0_correct"),
        "s1_correct":    r.get("s1_correct"),
        "n_tokens":      s["n_tokens"],
        "delta_js":      s["delta_js"],
        "delta_iou":     s["delta_iou"],
        "delta_cos":     s["delta_cos"],
        "mean_js_S0":    s["mean_js_S0"],
        "mean_js_S1":    s["mean_js_S1"],
        "mean_iou_S0":   s["mean_iou_S0"],
        "mean_iou_S1":   s["mean_iou_S1"],
        "mean_cos_S0":   s["mean_cos_S0"],
        "mean_cos_S1":   s["mean_cos_S1"],
    }


def _write_audit_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("# no rows survived per-sample filter\n")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # Round floats for readability — full precision is in alignment.jsonl
            out = {}
            for k, v in r.items():
                if isinstance(v, float):
                    out[k] = f"{v:.6f}"
                else:
                    out[k] = v
            w.writerow(out)


def pick_narrative_regression(rows: list[dict], top_k: int,
                              eps: float) -> list[dict]:
    """N1: OPD_improved + evidence regression on at least one metric.

    Rationale: the headline contrast is "answer改对but attention跑偏". A
    sample qualifies if either JS-distance or IoU-overlap regresses by
    more than `eps`. Rank by the more regressive of the two normalized
    against eps so neither metric dominates purely by scale.
    """
    cand = [r for r in rows
            if r["bucket"] == "OPD_improved"
            and r["s0_correct"] is False
            and r["s1_correct"] is True
            and (r["delta_js"] < -eps or r["delta_iou"] < -eps)]

    def score(r: dict) -> float:
        # Smaller (more negative) = more dramatic regression.
        # Normalize each metric by eps so they're comparable on the same axis.
        js_z = r["delta_js"] / eps
        iou_z = r["delta_iou"] / eps
        return min(js_z, iou_z)

    cand.sort(key=score)
    return cand[:top_k]


def pick_narrative_invariant(rows: list[dict], top_k: int,
                             eps: float) -> list[dict]:
    """N2: OPD_improved + evidence invariant on both metrics.

    Rationale: "answer flipped, but attention didn't move" — the OPD-changed-
    reasoning-not-evidence narrative (412-style). Tight band around zero on
    BOTH ΔJS and ΔIoU.
    """
    cand = [r for r in rows
            if r["bucket"] == "OPD_improved"
            and r["s0_correct"] is False
            and r["s1_correct"] is True
            and abs(r["delta_js"]) < eps
            and abs(r["delta_iou"]) < eps]

    # Rank by joint tightness — smallest combined |Δ| first.
    cand.sort(key=lambda r: abs(r["delta_js"]) / eps + abs(r["delta_iou"]) / eps)
    return cand[:top_k]


def _truncate(s: str | None, n: int) -> str:
    if s is None:
        return ""
    s = s.replace("\n", " ").replace("|", "\\|")
    return s if len(s) <= n else s[: n - 1] + "…"


def _write_summary_md(n1: list[dict], n2: list[dict],
                      total_opd_improved: int, eps: float,
                      reliability_thresh: float, path: Path,
                      questions: dict[str, str]) -> None:
    lines: list[str] = []
    lines.append("# Step 5 — narrative picks summary\n")
    lines.append(f"- `eps` = {eps:.4f}  (band width for invariance / "
                 f"regression threshold)")
    lines.append(f"- `reliability_thresh` = {reliability_thresh:.4f}")
    lines.append(f"- Pool: bucket=OPD_improved with surviving per-sample "
                 f"means = **{total_opd_improved}** samples\n")
    lines.append("Sign convention:  ΔJS = JS(S0,T) − JS(S1,T) "
                 "(positive = OPD improves alignment, S1 distribution "
                 "closer to T).  ΔIoU = IoU(S1,T) − IoU(S0,T) "
                 "(positive = improves).\n")

    for tag, name, rationale, picks in [
        ("N1", "Answer flipped, evidence REGRESSED",
         "S1 evidence moved away from teacher despite answer being fixed. "
         "Used to argue: OPD does change attention but in the wrong "
         "direction → EA-OPD motivation.", n1),
        ("N2", "Answer flipped, evidence INVARIANT",
         "Both ΔJS and ΔIoU within ±eps. Used to argue: OPD changed "
         "reasoning over the same evidence — HallusionBench/412 lives here.",
         n2),
    ]:
        lines.append(f"## {tag} — {name}  (n={len(picks)})\n")
        lines.append(f"_{rationale}_\n")
        if not picks:
            lines.append("(no candidates)\n")
            continue
        lines.append("| rank | id | bench | ΔJS | ΔIoU | ΔCos | n_tok | "
                     "question (≤80c) |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---|")
        for i, r in enumerate(picks, 1):
            q = _truncate(questions.get(r["id"]), 80)
            lines.append(f"| {i} | `{r['id']}` | {r['benchmark']} | "
                         f"{r['delta_js']:+.4f} | {r['delta_iou']:+.4f} | "
                         f"{r['delta_cos']:+.4f} | {r['n_tokens']} | {q} |")
        lines.append("")
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alignment", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("docs/figures/step5/narrative_picks/"))
    ap.add_argument("--top-k", type=int, default=20,
                    help="How many ids per narrative to emit (default 20).")
    ap.add_argument("--eps", type=float, default=0.005,
                    help=("Threshold for 'regressed' / 'invariant' band on "
                          "ΔJS and ΔIoU. Default 0.005 ≈ Null B δ_JS from "
                          "the audit (0.0063), slightly tighter."))
    ap.add_argument("--reliability-thresh", type=float,
                    default=RELIABILITY_THRESH_DEFAULT,
                    help=("Token-level entropy filter, matches audit picker. "
                          f"Default {RELIABILITY_THRESH_DEFAULT}."))
    ap.add_argument("--bucket", default="OPD_improved",
                    help=("Only kept for forward-compat — both narratives "
                          "currently require OPD_improved. Other buckets "
                          "appear in the audit CSV regardless."))
    args = ap.parse_args(argv)

    print(f">>> loading {args.alignment} (this can take a minute on 600MB+)",
          file=sys.stderr)
    rows_raw = load_alignment(args.alignment)
    print(f">>> loaded {len(rows_raw)} rows", file=sys.stderr)

    summary: list[dict] = []
    questions: dict[str, str] = {}
    n_dropped = 0
    for r in rows_raw:
        s = _per_sample_means(r, reliability_thresh=args.reliability_thresh)
        if s is None:
            n_dropped += 1
            continue
        summary.append(_summarize_one(r, s))
        questions[r["id"]] = r.get("question", "")
    print(f">>> per-sample means: kept {len(summary)}, "
          f"dropped {n_dropped} (no surviving tokens after reliability filter)",
          file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_csv = args.out_dir / "narrative_audit.csv"
    _write_audit_csv(summary, audit_csv)
    print(f">>> wrote {audit_csv}", file=sys.stderr)

    opd_improved_count = sum(1 for r in summary if r["bucket"] == "OPD_improved"
                             and r["s0_correct"] is False
                             and r["s1_correct"] is True)

    n1 = pick_narrative_regression(summary, args.top_k, args.eps)
    n2 = pick_narrative_invariant(summary, args.top_k, args.eps)

    (args.out_dir / "narrative1_regression_ids.txt").write_text(
        ",".join(r["id"] for r in n1) + "\n"
    )
    (args.out_dir / "narrative2_invariant_ids.txt").write_text(
        ",".join(r["id"] for r in n2) + "\n"
    )
    print(f">>> N1 (regression):  {len(n1)} ids → "
          f"{args.out_dir / 'narrative1_regression_ids.txt'}", file=sys.stderr)
    print(f">>> N2 (invariant):   {len(n2)} ids → "
          f"{args.out_dir / 'narrative2_invariant_ids.txt'}", file=sys.stderr)

    _write_summary_md(n1, n2, opd_improved_count, args.eps,
                      args.reliability_thresh,
                      args.out_dir / "narrative_summary.md",
                      questions)
    print(f">>> wrote {args.out_dir / 'narrative_summary.md'}", file=sys.stderr)

    # Echo top picks to stdout so user sees immediate signal without
    # opening the markdown.
    print("\n=== N1 top 10 (regression) ===")
    for i, r in enumerate(n1[:10], 1):
        print(f"  {i:2d}. {r['id']:20s} ΔJS={r['delta_js']:+.4f} "
              f"ΔIoU={r['delta_iou']:+.4f} ΔCos={r['delta_cos']:+.4f}")
    print("\n=== N2 top 10 (invariant) ===")
    for i, r in enumerate(n2[:10], 1):
        print(f"  {i:2d}. {r['id']:20s} ΔJS={r['delta_js']:+.4f} "
              f"ΔIoU={r['delta_iou']:+.4f} ΔCos={r['delta_cos']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
