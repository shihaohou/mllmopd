"""Blankness-phrase rate over training step (paired with accuracy trajectory).

For each ``*_full.jsonl`` produced by ``run_t1_trajectory.sh``, scan
predictions for blank-image refusal templates ("blank", "completely white",
"no information", "cannot see", ...). Output JSON (per arm × step) and a
figure that overlays:

  (1) full_image accuracy vs step  (left y-axis)
  (2) blankness_phrase_rate vs step (right y-axis)

The pair is the paper-grade visualization of the on-policy prefix
self-conditioning mechanism: the blank-template likelihood rises in
the student's distribution, then crosses a threshold and dominates,
which is the same step window where accuracy cliff-falls.

Also emits ``first_blank_token_position`` (median word offset where the
blank phrase first appears) — captures the template migrating from
late-tail explanation to response-opener.

Usage::

    python -m mllmopd.analysis.t1_blankness_trajectory \\
        --traj-dir runs/audit/t1_trajectory_20260522-105745 \\
        [--steps 49,99,149,199,230] \\
        [--out-json runs/analysis/t1_v1p5b_blankness.json] \\
        [--out-fig runs/analysis/t1_v1p5b_blankness.png]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Phrases the BlankTeacher student emits when it refuses to read the image.
# Surveyed from step_230 T1-3 predictions (n=1200); top-coverage subset.
BLANK_PHRASES = [
    r"\bblank\b",
    r"completely white",
    r"no information",
    r"cannot see",
    r"no visible",
    r"\bplaceholder\b",
    r"I cannot determine",
    r"image is empty",
    r"no chart\b",
    r"no image\b",
    r"\birrelevant\b",
]
BLANK_RE = re.compile("|".join(BLANK_PHRASES), re.IGNORECASE)


def scan_jsonl(path: Path) -> dict:
    n = 0
    hits = 0
    first_word_positions: list[int] = []
    per_bench_hits: dict[str, int] = {}
    per_bench_n: dict[str, int] = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            n += 1
            bench = d.get("benchmark", "_")
            per_bench_n[bench] = per_bench_n.get(bench, 0) + 1
            pred = d.get("prediction", "")
            m = BLANK_RE.search(pred)
            if m:
                hits += 1
                per_bench_hits[bench] = per_bench_hits.get(bench, 0) + 1
                prefix = pred[:m.start()]
                first_word_positions.append(len(prefix.split()))
    fwp_sorted = sorted(first_word_positions)
    if fwp_sorted:
        median = fwp_sorted[len(fwp_sorted) // 2]
        mean = sum(fwp_sorted) / len(fwp_sorted)
    else:
        median = None
        mean = None
    per_bench_rate = {
        b: per_bench_hits.get(b, 0) / per_bench_n[b]
        for b in per_bench_n
    }
    return {
        "n": n,
        "hits": hits,
        "rate": hits / n if n else 0.0,
        "first_blank_word_pos_median": median,
        "first_blank_word_pos_mean": mean,
        "n_first_blank_positions": len(first_word_positions),
        "per_bench_rate": per_bench_rate,
    }


def aggregate(traj_dir: Path, steps: list[int]) -> dict:
    out = {"phrases": BLANK_PHRASES, "T1_2": {}, "T1_3": {}, "T1_0_base": None}
    base = traj_dir / "T1_0_base_full.jsonl"
    if base.exists():
        out["T1_0_base"] = scan_jsonl(base)
    for arm in ("T1_2", "T1_3"):
        for step in steps:
            p = traj_dir / f"{arm}_step_{step}_full.jsonl"
            if p.exists():
                out[arm][str(step)] = scan_jsonl(p)
    return out


def plot(out: dict, accuracy_json: Path | None, out_fig: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figure", file=sys.stderr)
        return

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax2 = ax1.twinx()

    steps_t2 = sorted(int(s) for s in out["T1_2"])
    steps_t3 = sorted(int(s) for s in out["T1_3"])
    br_t2 = [out["T1_2"][str(s)]["rate"] * 100 for s in steps_t2]
    br_t3 = [out["T1_3"][str(s)]["rate"] * 100 for s in steps_t3]

    if accuracy_json and Path(accuracy_json).exists():
        with Path(accuracy_json).open() as f:
            acc = json.load(f)
        acc_t2_steps = sorted(int(s) for s in acc["T1_2"])
        acc_t3_steps = sorted(int(s) for s in acc["T1_3"])
        acc_t2 = [acc["T1_2"][str(s)]["overall"] * 100 for s in acc_t2_steps]
        acc_t3 = [acc["T1_3"][str(s)]["overall"] * 100 for s in acc_t3_steps]
        ax1.plot(acc_t2_steps, acc_t2, "o-", color="#1f77b4",
                 label="T1-2 (Full) acc")
        ax1.plot(acc_t3_steps, acc_t3, "o-", color="#d62728",
                 label="T1-3 (Blank) acc")
        ax1.set_ylabel("Full-image accuracy (mean, %)")
        ax1.set_ylim(20, 70)

    ax2.plot(steps_t2, br_t2, "s--", color="#1f77b4", alpha=0.6,
             label="T1-2 blankness rate")
    ax2.plot(steps_t3, br_t3, "s--", color="#d62728", alpha=0.9,
             label="T1-3 blankness rate")
    if out["T1_0_base"] is not None:
        base_rate = out["T1_0_base"]["rate"] * 100
        ax2.axhline(base_rate, color="gray", ls=":",
                    label=f"T1-0 base {base_rate:.1f}%")
    ax2.set_ylabel("Blankness-phrase rate (%, dashed)")
    ax2.set_ylim(0, 70)

    ax1.set_xlabel("Training step")
    ax1.set_title("BlankTeacher OPD: accuracy cliff + blank-template cliff")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=130)
    print(f"figure -> {out_fig}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traj-dir", required=True,
                    help="run_t1_trajectory.sh output dir")
    ap.add_argument("--steps", default="49,99,149,199,230",
                    help="comma list of training steps")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-fig", default=None)
    ap.add_argument("--accuracy-json", default=None,
                    help="t1_v1p5b_trajectory.json for overlay")
    args = ap.parse_args(argv)

    steps = [int(s) for s in args.steps.split(",") if s.strip()]
    traj_dir = Path(args.traj_dir)
    out = aggregate(traj_dir, steps)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out_json).open("w") as f:
            json.dump(out, f, indent=2)
        print(f"json -> {args.out_json}", file=sys.stderr)
    else:
        print(json.dumps(out, indent=2))

    if args.out_fig:
        Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
        plot(out, args.accuracy_json, Path(args.out_fig))
    return 0


if __name__ == "__main__":
    sys.exit(main())
