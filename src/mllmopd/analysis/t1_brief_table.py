"""Canonical per-benchmark brief table renderer.

Single source of truth for the v1.5b brief's per-benchmark table.
Reads ``t1_compare.json`` (produced by ``t1_compare.py``) and emits a
markdown table that exactly matches the canonical numbers. Eliminates
the hand-typed-numbers drift class of bug GPT flagged in round 3.

Columns: T1-0 base | T1-2 (Full) | T1-3 (Blank) | G_T1-2 | G_T1-3 | Δ_b
All numbers in percentage points (3 decimals → pp display).

Usage::

    python -m mllmopd.analysis.t1_brief_table \\
        --in runs/audit/t1_v1p5b_eval_step249_20260522-102258/t1_compare.json \\
        [--mode markdown|tsv]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BENCHES = [
    "ChartQA",
    "HallusionBench",
    "MathVerse",
    "MathVision",
    "MathVista",
    "POPE_adversarial",
]


def render(j: dict, mode: str = "markdown") -> str:
    raw = j["raw_acc"]
    g_full = j["G_full"]
    delta_b = j["delta_per_benchmark"]

    def pp(x: float) -> str:
        return f"{x * 100:+.1f}pp"

    rows = []
    means = {"T1-0": 0.0, "T1-2": 0.0, "T1-3": 0.0,
             "G_T1-2": 0.0, "G_T1-3": 0.0, "Δ_b": 0.0}
    for b in BENCHES:
        t0 = raw["T1-0"][b]["full_image"]
        t2 = raw["T1-2"][b]["full_image"]
        t3 = raw["T1-3"][b]["full_image"]
        g2 = g_full["T1-2"][b]
        g3 = g_full["T1-3"][b]
        db = delta_b[b]
        means["T1-0"] += t0
        means["T1-2"] += t2
        means["T1-3"] += t3
        means["G_T1-2"] += g2
        means["G_T1-3"] += g3
        means["Δ_b"] += db
        rows.append((b, t0, t2, t3, g2, g3, db))
    for k in means:
        means[k] /= len(BENCHES)

    if mode == "tsv":
        out = ["benchmark\tT1-0\tT1-2\tT1-3\tG_T1-2\tG_T1-3\tΔ_b"]
        for r in rows:
            b, t0, t2, t3, g2, g3, db = r
            out.append(f"{b}\t{t0:.3f}\t{t2:.3f}\t{t3:.3f}\t"
                       f"{g2:+.3f}\t{g3:+.3f}\t{db:+.3f}")
        out.append(
            f"Mean\t{means['T1-0']:.3f}\t{means['T1-2']:.3f}\t{means['T1-3']:.3f}\t"
            f"{means['G_T1-2']:+.3f}\t{means['G_T1-3']:+.3f}\t{means['Δ_b']:+.3f}"
        )
        return "\n".join(out)

    lines = [
        "| Benchmark | T1-0 base | **T1-2 (Full)** | **T1-3 (Blank)** | G_T1-2 | **G_T1-3** | **Δ_b** |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        b, t0, t2, t3, g2, g3, db = r
        lines.append(
            f"| {b} | {t0:.3f} | {t2:.3f} | **{t3:.3f}** | "
            f"{pp(g2)} | **{pp(g3)}** | **{pp(db)}** |"
        )
    lines.append(
        f"| **Mean** | **{means['T1-0']:.3f}** | **{means['T1-2']:.3f}** | "
        f"**{means['T1-3']:.3f}** | **{pp(means['G_T1-2'])}** | "
        f"**{pp(means['G_T1-3'])}** | **{pp(means['Δ_b'])}** |"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--in", dest="in_path", required=True,
                    help="path to t1_compare.json")
    ap.add_argument("--mode", choices=["markdown", "tsv"], default="markdown")
    args = ap.parse_args(argv)

    with Path(args.in_path).open() as f:
        j = json.load(f)
    print(render(j, mode=args.mode))
    return 0


if __name__ == "__main__":
    sys.exit(main())
