"""Step 3a v3 — pre-train fire-rate audit for B1 α calibration.

GPT verdict requires α to be auto-calibrated (not hard-fixed at 0.5):

    α = (target_mean_w − 1) / fire_rate

where fire_rate is measured on a sample of rollout tokens. This script
reads an existing `teacher_cache.jsonl` (from any tam_step1a run — the
classifier v0.1.3 has already labeled every response token) and reports:

  - fire_rate  (frac of tokens in C_local; pool-wide and per benchmark)
  - per-category counts
  - recommended α for target_mean_w ∈ {1.05, 1.06, 1.10}
  - mean_w_at_alpha_0.5  (sanity vs design's old hard-fixed default)
  - q3 boost budget projection (if vd/adv available, for cross-check
    against the v3 narrative q3-firing-is-incidental story)

The audit uses **teacher's** rollouts as a proxy for student rollouts at
training time. Distribution may differ slightly (student is MMR1-3B-SFT
vs teacher MMR1-7B-RL) but should be in the same order of magnitude.
For a tighter calibration, re-run after B0 produces actual student
rollouts and re-measure.

Usage::

    python -m scripts.audit.tam_step3_b_firerate_audit \\
        --teacher-cache runs/audit/tam_step1a_classifier_v013_full/teacher_cache.jsonl \\
        --out-json runs/analysis/tam_step3_b_firerate_audit.json \\
        --out-txt  runs/analysis/tam_step3_b_firerate_audit.txt \\
        --target-mean-w 1.05,1.06,1.10
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


C_LOCAL_DEFAULT = ("content_noun", "visual_attribute", "proper_noun")


def _benchmark_of(sample_id: str) -> str:
    if not sample_id or "/" not in sample_id:
        return "unknown"
    return sample_id.split("/", 1)[0]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--teacher-cache", type=Path, required=True,
                    help="tam_step1a teacher_cache.jsonl with classification "
                         "(produced by Step 1a or precompute)")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-txt",  type=Path, default=None)
    ap.add_argument("--c-local", type=str,
                    default=",".join(C_LOCAL_DEFAULT),
                    help="comma-separated category set (default v3 lock)")
    ap.add_argument("--target-mean-w", type=str, default="1.05,1.06,1.10",
                    help="comma-separated target_mean_w values for α recommendation")
    args = ap.parse_args(argv)

    c_local = {c.strip() for c in args.c_local.split(",") if c.strip()}
    targets = sorted(float(x) for x in args.target_mean_w.split(","))

    if not args.teacher_cache.exists():
        print(f"!! teacher cache not found: {args.teacher_cache}", file=sys.stderr)
        return 1

    # Aggregators
    total_tokens = 0
    total_fire = 0
    cat_counts: Counter = Counter()
    by_benchmark = defaultdict(lambda: {"tokens": 0, "fire": 0,
                                         "cats": Counter()})
    # Per-sample for length stats
    response_lengths: list[int] = []
    fire_per_sample: list[float] = []
    n_rows = 0
    n_valid = 0

    print(f">>> reading {args.teacher_cache}", file=sys.stderr)
    with args.teacher_cache.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            rec = json.loads(line)
            if not rec.get("tam_valid"):
                continue
            classification = rec.get("classification") or {}
            cats = classification.get("token_category") or []
            R = int(rec.get("response_length") or 0)
            if R == 0 or len(cats) < R:
                continue
            cats = cats[:R]
            sample_id = rec.get("id", "?")
            benchmark = _benchmark_of(sample_id)

            n_fire_sample = sum(1 for c in cats if c in c_local)
            n_valid += 1
            total_tokens += R
            total_fire += n_fire_sample
            response_lengths.append(R)
            fire_per_sample.append(n_fire_sample / R if R else 0.0)
            for c in cats:
                cat_counts[c] += 1
                by_benchmark[benchmark]["cats"][c] += 1
            by_benchmark[benchmark]["tokens"] += R
            by_benchmark[benchmark]["fire"] += n_fire_sample

    if total_tokens == 0:
        print("!! no valid tokens found", file=sys.stderr)
        return 1

    fire_rate = total_fire / total_tokens

    # α recommendations
    alpha_rec = {}
    for t in targets:
        if fire_rate <= 1e-9:
            alpha_rec[f"{t:.2f}"] = None
        else:
            alpha_rec[f"{t:.2f}"] = (t - 1.0) / fire_rate

    # Build report
    report = {
        "teacher_cache":   str(args.teacher_cache),
        "n_rows":          n_rows,
        "n_valid":         n_valid,
        "total_tokens":    total_tokens,
        "total_fire":      total_fire,
        "C_local":         sorted(c_local),
        "fire_rate":       fire_rate,
        "mean_w_at_alpha": {
            "0.10": 1.0 + 0.10 * fire_rate,
            "0.20": 1.0 + 0.20 * fire_rate,
            "0.50": 1.0 + 0.50 * fire_rate,
            "1.00": 1.0 + 1.00 * fire_rate,
        },
        "alpha_for_target_mean_w": alpha_rec,
        "category_counts": dict(cat_counts.most_common()),
        "by_benchmark": {
            b: {
                "tokens": v["tokens"],
                "fire": v["fire"],
                "fire_rate": v["fire"] / v["tokens"] if v["tokens"] else 0.0,
                "top_cats": dict(v["cats"].most_common(8)),
            }
            for b, v in sorted(by_benchmark.items(), key=lambda kv: -kv[1]["tokens"])
        },
        "response_length_stats": {
            "n": len(response_lengths),
            "mean": sum(response_lengths) / len(response_lengths),
            "min": min(response_lengths),
            "max": max(response_lengths),
        },
        "per_sample_fire_rate_stats": {
            "n": len(fire_per_sample),
            "mean": sum(fire_per_sample) / len(fire_per_sample),
            "min": min(fire_per_sample),
            "max": max(fire_per_sample),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(report, f, indent=2)
    print(f">>> wrote {args.out_json}", file=sys.stderr)

    if args.out_txt:
        lines: list[str] = []
        lines.append(f"# Step 3a v3 fire-rate audit")
        lines.append(f"# teacher_cache = {args.teacher_cache}")
        lines.append(f"# C_local = {sorted(c_local)}")
        lines.append(f"# n_rows={n_rows}  n_valid={n_valid}  total_tokens={total_tokens}")
        lines.append("")
        lines.append(f"## Headline")
        lines.append(f"  fire_rate              = {fire_rate:.4f}")
        lines.append(f"  total_fire / total_tok = {total_fire} / {total_tokens}")
        lines.append("")
        lines.append(f"## mean_w under various α")
        for a, mw in report["mean_w_at_alpha"].items():
            lines.append(f"  α = {a}:  mean_w = {mw:.4f}")
        lines.append("")
        lines.append(f"## α recommended for target_mean_w")
        for t, a in alpha_rec.items():
            a_str = f"{a:.4f}" if a is not None else "n/a"
            lines.append(f"  target = {t}  →  α = {a_str}")
        lines.append("")
        lines.append(f"## Category distribution (all tokens)")
        lines.append(f"  {'category':<24} {'count':>8}  {'frac':>6}  {'C_local':>8}")
        for c, n in cat_counts.most_common():
            in_c = "★" if c in c_local else ""
            lines.append(f"  {c:<24} {n:>8d}  {100*n/total_tokens:>5.2f}%  {in_c:>8}")
        lines.append("")
        lines.append(f"## Fire rate by benchmark")
        lines.append(f"  {'benchmark':<22} {'n_tokens':>9} {'n_fire':>8} {'fire_rate':>10}")
        for b, v in report["by_benchmark"].items():
            lines.append(f"  {b:<22} {v['tokens']:>9} {v['fire']:>8} {v['fire_rate']:>9.4f}")
        lines.append("")
        lines.append(f"## Response length")
        rs = report["response_length_stats"]
        lines.append(f"  n={rs['n']}  mean={rs['mean']:.0f}  min={rs['min']}  max={rs['max']}")
        lines.append("")
        lines.append(f"## Per-sample fire rate (within-response variance)")
        ps = report["per_sample_fire_rate_stats"]
        lines.append(f"  mean={ps['mean']:.4f}  min={ps['min']:.4f}  max={ps['max']:.4f}")
        lines.append("")
        lines.append(f"## RECOMMENDATION")
        # Pick mid target (1.06)
        if "1.06" in alpha_rec and alpha_rec["1.06"] is not None:
            lines.append(f"  Use α = {alpha_rec['1.06']:.3f}  "
                         f"(target_mean_w = 1.06, fire_rate = {fire_rate:.4f})")
        lines.append(f"  Caveat: teacher rollouts used as proxy for student rollouts. "
                     f"After B0 produces real student rollouts, re-measure.")

        args.out_txt.parent.mkdir(parents=True, exist_ok=True)
        args.out_txt.write_text("\n".join(lines) + "\n")
        print(f">>> wrote summary: {args.out_txt}", file=sys.stderr)

    print(f">>> fire_rate = {fire_rate:.4f}", file=sys.stderr)
    if "1.06" in alpha_rec and alpha_rec["1.06"] is not None:
        print(f">>> recommended α (target_mean_w=1.06) = {alpha_rec['1.06']:.3f}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
