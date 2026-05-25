"""Slice the tam_step3_preflight --dump-records JSONL by (category, quad, τ)
and surface the per-token detail used to diagnose pre-flight outliers.

For each matching cell, reports:
  - Per-benchmark count + fire rate
  - Per-token table (token text, coverage, vd, adv, tam_mass, fire)
  - Cell-wide stats: coverage / tam_mass / tam_entropy distributions

Usage::

    python scripts/audit/tam_step3_cell_audit.py \\
        --records-jsonl runs/analysis/tam_step3_preflight_records.jsonl \\
        --category proper_noun --quad 3 --tau 0.70 \\
        --out-txt   runs/analysis/tam_step3_audit_proper_noun_q3.txt \\
        [--show-tokens 80]   # cap per-token table
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median


def _safe_float(v):
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records-jsonl", type=Path, required=True)
    ap.add_argument("--category", required=True,
                    help="e.g. proper_noun / content_noun / visual_attribute")
    ap.add_argument("--quad", type=int, required=True,
                    help="0..3 (vd≥0/<0 × adv≥0/<0)")
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--out-txt", type=Path, default=None)
    ap.add_argument("--show-tokens", type=int, default=100,
                    help="cap per-token table (default 100)")
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="τ float-match tolerance")
    args = ap.parse_args(argv)

    cell: list[dict] = []
    with args.records_jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("category") != args.category:
                continue
            if r.get("quad") != args.quad:
                continue
            if abs(float(r.get("tau", -1)) - args.tau) > args.rtol:
                continue
            cell.append(r)

    if not cell:
        print(f"!! no records match category={args.category} quad={args.quad} "
              f"τ={args.tau}", file=sys.stderr)
        return 1

    n = len(cell)
    fires = sum(int(r.get("gate_fire", 0)) for r in cell)
    covs = [r["coverage"] for r in cell if r.get("coverage") is not None]
    masses = [_safe_float(r.get("tam_mass_top20"))
              for r in cell if r.get("tam_mass_top20") is not None]
    entropies = [_safe_float(r.get("tam_entropy_norm"))
                 for r in cell if r.get("tam_entropy_norm") is not None]
    vds = [_safe_float(r.get("vd")) for r in cell if r.get("vd") is not None]
    advs = [_safe_float(r.get("adv")) for r in cell if r.get("adv") is not None]

    bench_count = Counter(r.get("benchmark") or "unknown" for r in cell)
    bench_fire = Counter()
    for r in cell:
        b = r.get("benchmark") or "unknown"
        bench_fire[b] += int(r.get("gate_fire", 0))

    sid_count = Counter(r["sample_id"] for r in cell)

    lines: list[str] = []
    lines.append(f"# Cell audit  cat={args.category}  quad=q{args.quad}  τ={args.tau}")
    lines.append(f"# records: {args.records_jsonl}")
    lines.append(f"# n = {n}  fires = {fires} ({fires/n:.3f})")
    lines.append("")
    lines.append("## Stats")
    if covs:
        lines.append(f"  coverage:        mean={mean(covs):+.3f}  "
                     f"median={median(covs):+.3f}  "
                     f"min={min(covs):+.3f}  max={max(covs):+.3f}  "
                     f"(n={len(covs)})")
    if masses:
        lines.append(f"  tam_mass_top20:  mean={mean(masses):+.3f}  "
                     f"median={median(masses):+.3f}  "
                     f"min={min(masses):+.3f}  max={max(masses):+.3f}")
    if entropies:
        lines.append(f"  tam_entropy:     mean={mean(entropies):+.3f}  "
                     f"median={median(entropies):+.3f}")
    if vds:
        lines.append(f"  vd:              mean={mean(vds):+.3f}  "
                     f"median={median(vds):+.3f}")
    if advs:
        lines.append(f"  adv:             mean={mean(advs):+.3f}  "
                     f"median={median(advs):+.3f}")
    lines.append("")

    lines.append("## By benchmark")
    lines.append(f"  {'benchmark':<24} {'n':>5} {'n_fire':>7} {'fire_rate':>10}")
    for b, c in sorted(bench_count.items(), key=lambda kv: -kv[1]):
        f = bench_fire.get(b, 0)
        lines.append(f"  {b:<24} {c:>5d} {f:>7d}  {f/c:>9.3f}")
    lines.append("")

    lines.append("## Top sample_ids contributing (top 15)")
    for sid, c in sid_count.most_common(15):
        lines.append(f"  {sid:<24} n={c}")
    lines.append("")

    # Per-token table — sort by gate_fire DESC then coverage DESC
    cell_sorted = sorted(cell,
                         key=lambda r: (-int(r.get("gate_fire", 0)),
                                        -(_safe_float(r.get("coverage")) or 0)))
    lines.append(f"## Per-token (top {min(args.show_tokens, n)} of {n} by fire ↓ then coverage ↓)")
    lines.append(f"  {'sample_id':<22} {'tok_idx':>7} {'token':<14} "
                 f"{'fire':>4} {'cov':>6} {'vd':>7} {'adv':>7} "
                 f"{'mass20':>7} {'ent':>6}")
    for r in cell_sorted[:args.show_tokens]:
        sid = (r.get("sample_id") or "")[:22]
        tok = (r.get("token") or "")[:14].replace("\n", "\\n")
        fire = int(r.get("gate_fire", 0))
        cov = r.get("coverage")
        cov_s = f"{cov:+.3f}" if cov is not None else "  -  "
        vd = _safe_float(r.get("vd"))
        adv = _safe_float(r.get("adv"))
        m20 = _safe_float(r.get("tam_mass_top20"))
        ent = _safe_float(r.get("tam_entropy_norm"))
        lines.append(f"  {sid:<22} {r['token_idx']:>7d} {tok:<14} "
                     f"{fire:>4d} {cov_s:>6} "
                     f"{vd:>+7.3f} {adv:>+7.3f} {m20:>+7.3f} {ent:>+6.3f}")
    lines.append("")
    if n > args.show_tokens:
        lines.append(f"  ({n - args.show_tokens} more not shown; raise --show-tokens)")

    out = "\n".join(lines) + "\n"
    if args.out_txt:
        args.out_txt.parent.mkdir(parents=True, exist_ok=True)
        args.out_txt.write_text(out)
        print(f">>> wrote {args.out_txt}", file=sys.stderr)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
