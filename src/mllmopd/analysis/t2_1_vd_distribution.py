"""T2-1 VD-weight distribution analyzer.

Consumes per-step diagnostics JSONL.gz files produced by
``mllmopd.training.opd_diagnostics_hook.post_process_rewards_with_diagnostics``
during a T2-1 training run. Reports per-step aggregate statistics on:

  * ``vd`` (= lp_teacher_full - lp_teacher_blank per token):
    mean / std / p5 / p50 / p95 / min / max over all tokens in the step
  * ``vd_weights`` (PGPO Eq 6 + 7 output per token):
    mean / max / p99 / sparsity_rate (fraction w<1)
  * Mass-preservation residual: max |sum(w)/R - 1| across rows in the
    step (sanity check that the per-sequence renorm holds in production)
  * Attachment stats: n_rows, n_attached_nontrivial (rows where weights
    are not all-ones)

Cross-step trajectory: VD percentile bands and weight statistics vs
training step, in a 3-panel PNG, mirroring the visualization style of
``t1_blankness_trajectory.py``.

Runs against in-progress training (read whatever step_*.jsonl.gz files
are present at invocation time; no need to wait for training to finish).

Usage::

    python -m mllmopd.analysis.t2_1_vd_distribution \\
        --diag-dir ${MLLMOPD_RUNS}/t2_1_v0_T2_1_full_vd/diagnostics \\
        [--out-json runs/analysis/t2_1_vd_distribution.json] \\
        [--out-fig  runs/analysis/t2_1_vd_distribution.png] \\
        [--limit-rows-per-step 0]  # 0 = all; >0 to subsample for speed
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from pathlib import Path

STEP_RE = re.compile(r"step_(\d+)\.jsonl\.gz$")


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _stats(values: list[float]) -> dict:
    if not values:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p5": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
    std = math.sqrt(var)
    s = sorted(values)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": s[0],
        "p5": _percentile(s, 0.05),
        "p50": _percentile(s, 0.50),
        "p95": _percentile(s, 0.95),
        "p99": _percentile(s, 0.99),
        "max": s[-1],
    }


def scan_step_jsonl(path: Path, limit_rows: int = 0) -> dict:
    """Aggregate one step's JSONL.gz into per-step statistics.

    Args:
        path: step_NNNNNN.jsonl.gz file.
        limit_rows: if > 0, stop after this many rows (smoke-mode).
    """
    vd_all: list[float] = []
    w_all: list[float] = []
    mass_residuals: list[float] = []
    sparsity_per_row: list[float] = []
    max_w_per_row: list[float] = []
    n_rows = 0
    n_attached = 0
    n_nontrivial = 0
    n_no_vd = 0
    response_lengths: list[int] = []

    with gzip.open(path, "rt") as fin:
        for line in fin:
            if limit_rows and n_rows >= limit_rows:
                break
            row = json.loads(line)
            n_rows += 1
            R = int(row.get("response_length", 0))
            response_lengths.append(R)
            vd = row.get("vd") or []
            w = row.get("vd_weights") or []
            if not w:
                n_no_vd += 1
                continue
            n_attached += 1
            if not all(abs(x - 1.0) < 1e-5 for x in w):
                n_nontrivial += 1
            if vd and len(vd) == R:
                vd_all.extend(vd)
            w_all.extend(w)
            if R > 0:
                mass_residuals.append(abs(sum(w) / R - 1.0))
                sparsity_per_row.append(sum(1 for x in w if x < 1.0) / R)
                max_w_per_row.append(max(w))

    return {
        "n_rows": n_rows,
        "n_attached": n_attached,
        "n_nontrivial": n_nontrivial,
        "n_no_vd": n_no_vd,
        "response_length": _stats([float(x) for x in response_lengths]),
        "vd": _stats(vd_all),
        "vd_weights": _stats(w_all),
        "mass_residual_per_row": _stats(mass_residuals),
        "sparsity_rate_per_row": _stats(sparsity_per_row),
        "max_weight_per_row": _stats(max_w_per_row),
    }


def aggregate(diag_dir: Path, steps: list[int] | None,
              limit_rows: int) -> dict:
    files = sorted(diag_dir.glob("step_*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"no step_*.jsonl.gz in {diag_dir}")
    by_step: dict[int, dict] = {}
    for f in files:
        m = STEP_RE.search(f.name)
        if not m:
            continue
        s = int(m.group(1))
        if steps and s not in steps:
            continue
        by_step[s] = scan_step_jsonl(f, limit_rows=limit_rows)
    out = {
        "diag_dir": str(diag_dir),
        "n_steps": len(by_step),
        "steps": sorted(by_step.keys()),
        "per_step": {str(s): by_step[s] for s in sorted(by_step.keys())},
    }
    # Cross-step rollup: weight + sparsity trends.
    if by_step:
        all_steps = sorted(by_step.keys())
        out["rollup"] = {
            "vd_p5_first":  by_step[all_steps[0]]["vd"]["p5"],
            "vd_p5_last":   by_step[all_steps[-1]]["vd"]["p5"],
            "vd_p95_first": by_step[all_steps[0]]["vd"]["p95"],
            "vd_p95_last":  by_step[all_steps[-1]]["vd"]["p95"],
            "sparsity_first": by_step[all_steps[0]]["sparsity_rate_per_row"]["mean"],
            "sparsity_last":  by_step[all_steps[-1]]["sparsity_rate_per_row"]["mean"],
            "max_weight_first": by_step[all_steps[0]]["max_weight_per_row"]["mean"],
            "max_weight_last":  by_step[all_steps[-1]]["max_weight_per_row"]["mean"],
            "mass_residual_worst": max(
                by_step[s]["mass_residual_per_row"]["max"]
                for s in all_steps
                if not math.isnan(by_step[s]["mass_residual_per_row"]["max"])
            ),
        }
    return out


def plot(out: dict, out_fig: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping figure", file=sys.stderr)
        return

    steps = out["steps"]
    if not steps:
        print("no steps to plot", file=sys.stderr)
        return

    vd_p5  = [out["per_step"][str(s)]["vd"]["p5"]  for s in steps]
    vd_p50 = [out["per_step"][str(s)]["vd"]["p50"] for s in steps]
    vd_p95 = [out["per_step"][str(s)]["vd"]["p95"] for s in steps]
    sparsity = [out["per_step"][str(s)]["sparsity_rate_per_row"]["mean"] for s in steps]
    max_w    = [out["per_step"][str(s)]["max_weight_per_row"]["mean"] for s in steps]
    mass_res = [out["per_step"][str(s)]["mass_residual_per_row"]["max"] for s in steps]

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    ax = axes[0]
    ax.fill_between(steps, vd_p5, vd_p95, alpha=0.25, color="#1f77b4", label="vd p5–p95")
    ax.plot(steps, vd_p50, "o-", color="#1f77b4", label="vd median")
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.set_ylabel("vd = lp_full − lp_blank")
    ax.set_title("VD signal distribution over training")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(steps, [s * 100 for s in sparsity], "s-", color="#2ca02c",
            label="mean per-row sparsity")
    ax.set_ylabel("Fraction of tokens with w<1 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Suppress-branch coverage (PGPO Eq 6 below τ)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(steps, max_w, "^-", color="#d62728",
            label="mean per-row max weight")
    ax.axhline(1.0, color="gray", ls=":", alpha=0.5, label="baseline = 1.0")
    ax.set_ylabel("Peak boost per row")
    ax.set_xlabel("Training step")
    ax.set_title("Boost-branch concentration (post mass-renorm)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_fig, dpi=130)
    print(f"figure -> {out_fig}", file=sys.stderr)

    # Inline mass-residual sanity (separate; gets logged but not plotted).
    worst = max(x for x in mass_res if not math.isnan(x))
    print(f"mass-preservation worst row residual = {worst:.4e} "
          f"(should be < 1e-3 if PGPO Eq 7 holds in production)",
          file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--diag-dir", required=True,
                    help="<run>/diagnostics dir containing step_*.jsonl.gz")
    ap.add_argument("--steps", default=None,
                    help="comma list of training steps to include (default: all)")
    ap.add_argument("--limit-rows-per-step", type=int, default=0,
                    help="0 = all rows; >0 = subsample first N for speed")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-fig", default=None)
    args = ap.parse_args(argv)

    steps = None
    if args.steps:
        steps = [int(s) for s in args.steps.split(",") if s.strip()]

    diag_dir = Path(args.diag_dir)
    out = aggregate(diag_dir, steps, args.limit_rows_per_step)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out_json).open("w") as f:
            json.dump(out, f, indent=2, allow_nan=True)
        print(f"json -> {args.out_json}", file=sys.stderr)
    else:
        print(json.dumps(out, indent=2, default=str))

    if args.out_fig:
        Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
        plot(out, Path(args.out_fig))
    return 0


if __name__ == "__main__":
    sys.exit(main())
