"""Aggregate per-token visual-dependency (VD) data into the H2 audit table.

Input: a JSONL produced by `score_completion`, one row per prompt, with
`visual_dependency` / `logp_full` / `logp_blank` arrays.

For each prompt we have:
    vd(t)        = logp_full(t) - logp_blank(t)     # positive = image helps
    -logp_full(t)= "teacher's NLL effort" at position t (proxy for the
                   token's OPD supervision weight; low logp_full = teacher
                   committed bits to that token)

The H2 question this audit answers: **does teacher's per-token effort
concentrate on high-VD tokens (vision-grounded) or low-VD tokens
(language-prior)?** If most teacher NLL mass is on low-VD tokens, vanilla
OPD's dense token supervision mostly reinforces language priors rather
than transferring visual capability.

We bin tokens by VD quintile (or fixed thresholds) and report:
    n_tokens, fraction_of_tokens, fraction_of_nll_mass
per bin, per benchmark, and overall.

Usage:
    python -m mllmopd.analysis.aggregate_vd \\
        --scored runs/audit/level1_v4_sysprompt_fixed/T_RL_score_opd_target.jsonl \\
        [--out-table /path/to/vd_summary.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


# VD bin edges. Anything below LOW (~0) means image actively HURTS prediction
# (rare; usually noise). 0 is neutral. Above HIGH means image strongly helps.
_VD_BINS = [
    ("very_low (vd ≤ -1)", lambda v: v <= -1.0),
    ("low (-1 < vd ≤ 0)", lambda v: -1.0 < v <= 0.0),
    ("neutral (0 < vd ≤ 0.5)", lambda v: 0.0 < v <= 0.5),
    ("high (0.5 < vd ≤ 2.0)", lambda v: 0.5 < v <= 2.0),
    ("very_high (vd > 2)", lambda v: v > 2.0),
]


def _per_record_bin_stats(record: dict) -> dict:
    """Returns per-bin counts and -logp_full mass for one record."""
    vd = record["visual_dependency"]
    logp_full = record["logp_full"]
    by_bin = {label: {"n": 0, "nll_full": 0.0} for label, _ in _VD_BINS}
    for v, lp in zip(vd, logp_full):
        for label, pred in _VD_BINS:
            if pred(v):
                by_bin[label]["n"] += 1
                by_bin[label]["nll_full"] += -float(lp)
                break
    return by_bin


def _aggregate(records: list[dict]) -> dict:
    overall = {label: {"n": 0, "nll_full": 0.0} for label, _ in _VD_BINS}
    by_bench: dict[str, dict] = defaultdict(
        lambda: {label: {"n": 0, "nll_full": 0.0} for label, _ in _VD_BINS}
    )
    total_n = 0
    total_nll = 0.0
    by_bench_totals: dict[str, dict] = defaultdict(lambda: {"n": 0, "nll": 0.0})

    for r in records:
        bench = r.get("benchmark", "?")
        stats = _per_record_bin_stats(r)
        for label, _ in _VD_BINS:
            overall[label]["n"] += stats[label]["n"]
            overall[label]["nll_full"] += stats[label]["nll_full"]
            by_bench[bench][label]["n"] += stats[label]["n"]
            by_bench[bench][label]["nll_full"] += stats[label]["nll_full"]
        total_n += sum(s["n"] for s in stats.values())
        total_nll += sum(s["nll_full"] for s in stats.values())
        by_bench_totals[bench]["n"] += sum(s["n"] for s in stats.values())
        by_bench_totals[bench]["nll"] += sum(s["nll_full"] for s in stats.values())

    return {
        "overall": overall,
        "by_bench": dict(by_bench),
        "total_n": total_n,
        "total_nll": total_nll,
        "by_bench_totals": dict(by_bench_totals),
    }


def _print_table(agg: dict) -> None:
    print()
    print("=== Overall (all benchmarks combined) ===")
    print(f"{'VD bin':28s} {'n_tokens':>10s} {'%tokens':>9s} {'NLL mass':>12s} {'%NLL':>8s} {'NLL/tok':>9s}")
    print("-" * 80)
    overall = agg["overall"]
    tot_n = agg["total_n"]
    tot_nll = agg["total_nll"]
    for label, _ in _VD_BINS:
        n = overall[label]["n"]
        nll = overall[label]["nll_full"]
        if n == 0:
            print(f"{label:28s} {n:>10d} {'-':>9s} {nll:>12.1f} {'-':>8s} {'-':>9s}")
            continue
        pct_n = n / max(1, tot_n) * 100
        pct_nll = nll / max(1e-9, tot_nll) * 100
        print(f"{label:28s} {n:>10d} {pct_n:>8.1f}% {nll:>12.1f} {pct_nll:>7.1f}% "
              f"{(nll/n):>9.3f}")
    print("-" * 80)
    print(f"{'TOTAL':28s} {tot_n:>10d}          {tot_nll:>12.1f}")

    print()
    print("=== Per-benchmark (% of NLL mass landing in each VD bin) ===")
    benches = sorted(agg["by_bench"].keys())
    header_bins = [lbl.split(" ")[0] for lbl, _ in _VD_BINS]
    print(f"{'benchmark':18s}", " ".join(f"{b:>10s}" for b in header_bins))
    for bench in benches:
        b_data = agg["by_bench"][bench]
        b_tot_nll = max(1e-9, agg["by_bench_totals"][bench]["nll"])
        row_pcts = [b_data[label]["nll_full"] / b_tot_nll * 100 for label, _ in _VD_BINS]
        print(f"{bench:18s}", " ".join(f"{p:>9.1f}%" for p in row_pcts))

    print()
    print("=== Per-benchmark (% of TOKENS in each VD bin) ===")
    for bench in benches:
        b_data = agg["by_bench"][bench]
        b_tot_n = max(1, agg["by_bench_totals"][bench]["n"])
        row_pcts = [b_data[label]["n"] / b_tot_n * 100 for label, _ in _VD_BINS]
        print(f"{bench:18s}", " ".join(f"{p:>9.1f}%" for p in row_pcts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored", required=True, type=Path,
                    help="JSONL output from score_completion (one row per prompt)")
    ap.add_argument("--out-table", type=Path, default=None,
                    help="Optional: write aggregate as JSON")
    args = ap.parse_args()

    records: list[dict] = []
    with args.scored.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                continue
            if not r.get("visual_dependency"):
                continue
            records.append(r)
    print(f">>> loaded {len(records)} scored records from {args.scored}", file=sys.stderr)

    agg = _aggregate(records)
    _print_table(agg)

    if args.out_table:
        args.out_table.write_text(json.dumps(agg, indent=2, default=str))
        print(f"\n>>> wrote aggregate -> {args.out_table}", file=sys.stderr)


if __name__ == "__main__":
    main()
