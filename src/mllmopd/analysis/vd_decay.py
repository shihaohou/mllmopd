"""VD-decay analysis: how does teacher's full-vs-blank logp gap evolve
during OPD training?

Hypothesis surfaced in T1 v1 (2026-05-21): the teacher's vision signal
(mean |lp_full - lp_blank| per response token) shrinks ~10x between
early and late training steps:

  step  5  : mean|d| ~0.5-1.3
  step 50  : mean|d| ~1.2-1.9   ← peak
  step 95  : mean|d| ~0.07-0.18 ← decayed

If true, this is a token-level explanation for why FullTeacher ≈
BlankTeacher in final accuracy: by the time the student has trained for
~100 steps, its rollout output is template-ized enough that the teacher's
full-vs-blank logp gap collapses — dense KL ends up fitting language-prior
tokens that don't depend on the image.

This script aggregates per-step VD statistics across all samples in a
diagnostics dir (containing `step_NNNNNN.jsonl.gz` files) and emits both
a JSON summary and a matplotlib figure.

Usage:
  python -m mllmopd.analysis.vd_decay \\
      --diag-dirs T1_2_full=runs/t1_v1_T1_2_full_mm/diagnostics \\
                  T1_3_blank=runs/t1_v1_T1_3_blank_mm/diagnostics \\
      --out-json runs/analysis/vd_decay.json \\
      --out-fig  runs/analysis/vd_decay.png

The figure shows per-arm mean|vd| over steps with p50/p95 confidence
bands. The JSON has the raw per-step aggregates for downstream tables.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable


_STEP_RE = re.compile(r"step_(\d+)\.jsonl(?:\.gz)?$")


def _iter_step_files(diag_dir: Path) -> Iterable[tuple[int, Path]]:
    """Yield (step, path) for every step_NNNNNN.jsonl(.gz) under diag_dir."""
    for p in sorted(diag_dir.iterdir()):
        m = _STEP_RE.search(p.name)
        if m:
            yield int(m.group(1)), p


def _open_maybe_gz(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _percentile(xs: list[float], q: float) -> float:
    """Linear-interp percentile. q in [0, 1]."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _aggregate_step(rows: list[dict]) -> dict:
    """Aggregate VD stats across all rows in one step.

    Each row has `vd: list[float]` of length response_length, where
    vd[t] = lp_full[t] - lp_blank[t]. We compute:
      - mean |vd| over all tokens in all rows (the headline scalar)
      - p50 / p95 of token-level |vd|
      - n_tokens total, n_rows total
      - per-row mean |vd| (for variance across prompts)
      - mean response_length (for output-length drift check)
    """
    all_abs: list[float] = []
    per_row_mean: list[float] = []
    response_lengths: list[int] = []
    for r in rows:
        vd = r.get("vd") or []
        # The diagnostics hook writes vd=[] when lp_full and lp_blank don't
        # line up (e.g., the teacher's response wasn't returned). Skip those.
        if not vd:
            continue
        abs_vd = [abs(x) for x in vd]
        all_abs.extend(abs_vd)
        per_row_mean.append(sum(abs_vd) / len(abs_vd))
        response_lengths.append(int(r.get("response_length") or len(vd)))

    if not all_abs:
        return {
            "n_rows": 0,
            "n_tokens": 0,
            "mean_abs_vd": float("nan"),
            "p50_abs_vd": float("nan"),
            "p95_abs_vd": float("nan"),
            "mean_resp_len": float("nan"),
            "per_row_mean_p25": float("nan"),
            "per_row_mean_p75": float("nan"),
        }

    return {
        "n_rows": len(per_row_mean),
        "n_tokens": len(all_abs),
        "mean_abs_vd": sum(all_abs) / len(all_abs),
        "p50_abs_vd": _percentile(all_abs, 0.50),
        "p95_abs_vd": _percentile(all_abs, 0.95),
        "mean_resp_len": sum(response_lengths) / len(response_lengths),
        "per_row_mean_p25": _percentile(per_row_mean, 0.25),
        "per_row_mean_p75": _percentile(per_row_mean, 0.75),
    }


def _analyze_dir(diag_dir: Path, max_steps: int | None = None) -> dict[int, dict]:
    """Scan all step files in diag_dir, return {step: aggregate_dict}."""
    out: dict[int, dict] = {}
    for step, path in _iter_step_files(diag_dir):
        if max_steps is not None and step > max_steps:
            continue
        with _open_maybe_gz(path) as f:
            rows = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        agg = _aggregate_step(rows)
        agg["step"] = step
        out[step] = agg
        if step % 10 == 0 or step < 5:
            print(
                f"  step {step:>4d}: n_rows={agg['n_rows']:>4d} "
                f"n_tokens={agg['n_tokens']:>7d} "
                f"mean|vd|={agg['mean_abs_vd']:>6.3f} "
                f"p95|vd|={agg['p95_abs_vd']:>6.3f} "
                f"mean_resp_len={agg['mean_resp_len']:>6.0f}",
                file=sys.stderr,
            )
    return out


def _plot(per_arm: dict[str, dict[int, dict]], out_fig: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("!! matplotlib not installed — skipping figure", file=sys.stderr)
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: mean|vd| + per-row IQR band
    ax = axes[0]
    for arm, by_step in per_arm.items():
        steps = sorted(by_step)
        means = [by_step[s]["mean_abs_vd"] for s in steps]
        p25 = [by_step[s]["per_row_mean_p25"] for s in steps]
        p75 = [by_step[s]["per_row_mean_p75"] for s in steps]
        line, = ax.plot(steps, means, label=f"{arm} mean|vd|", linewidth=2)
        ax.fill_between(steps, p25, p75, alpha=0.18, color=line.get_color(),
                        label=f"{arm} per-prompt IQR")
    ax.set_ylabel("|lp_full - lp_blank| per token (nats)")
    ax.set_title("VD decay over training: teacher full-vs-blank logp gap")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Bottom: response length growth (sanity — is decay coincident with longer outputs?)
    ax = axes[1]
    for arm, by_step in per_arm.items():
        steps = sorted(by_step)
        rl = [by_step[s]["mean_resp_len"] for s in steps]
        ax.plot(steps, rl, label=f"{arm} mean response_len", linewidth=2)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("response length (tokens)")
    ax.set_title("Output length drift (sanity check)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=120, bbox_inches="tight")
    print(f">>> wrote {out_fig}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="VD-decay aggregator for OPD diagnostics dirs.",
    )
    ap.add_argument(
        "--diag-dirs", nargs="+", required=True,
        help="One or more `label=path` entries pointing at diagnostics dirs.",
    )
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Ignore steps beyond this number (default: all).")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-fig", type=Path, default=None)
    args = ap.parse_args()

    per_arm: dict[str, dict[int, dict]] = {}
    for spec in args.diag_dirs:
        if "=" not in spec:
            sys.exit(f"!! --diag-dirs entry must be label=path, got {spec!r}")
        label, path_str = spec.split("=", 1)
        path = Path(path_str)
        if not path.is_dir():
            sys.exit(f"!! not a directory: {path}")
        print(f">>> scanning {label}: {path}", file=sys.stderr)
        per_arm[label] = _analyze_dir(path, max_steps=args.max_steps)
        n_steps = len(per_arm[label])
        print(f">>> {label}: {n_steps} steps aggregated", file=sys.stderr)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            arm: {str(s): agg for s, agg in by_step.items()}
            for arm, by_step in per_arm.items()
        }
        args.out_json.write_text(json.dumps(payload, indent=2))
        print(f">>> wrote {args.out_json}", file=sys.stderr)

    if args.out_fig:
        _plot(per_arm, args.out_fig)

    # Print headline table
    print()
    print(f"{'arm':<14s} {'step':>5s} {'n_rows':>7s} {'n_tokens':>9s} "
          f"{'mean|vd|':>9s} {'p95|vd|':>8s} {'resp_len':>9s}")
    print("-" * 70)
    for arm, by_step in per_arm.items():
        steps = sorted(by_step)
        # Sample: first 3, middle 3, last 3
        idxs = sorted(set(
            steps[:3] + steps[len(steps)//2 - 1: len(steps)//2 + 2] + steps[-3:]
        ))
        for s in idxs:
            agg = by_step[s]
            print(f"{arm:<14s} {s:>5d} {agg['n_rows']:>7d} {agg['n_tokens']:>9d} "
                  f"{agg['mean_abs_vd']:>9.4f} {agg['p95_abs_vd']:>8.4f} "
                  f"{agg['mean_resp_len']:>9.0f}")


if __name__ == "__main__":
    main()
