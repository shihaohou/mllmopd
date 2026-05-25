"""Analyzer for Step 2 TAM causal masking.

Reads tam_step2.jsonl and computes the headline causal-effect statistic:

  ΔTAM-random = mean[logp_drop | top_tam_20pct] - mean[logp_drop | random_20pct]

over each token category and stratum. The random_20pct condition is the
mean across the three random seeds (42/43/44) — gives both expected drop
and variance under matched-coverage random masking.

Also reports:
  - keep_top_tam_20pct: should DROP a LOT (only 20% of pixels left)
  - bottom_tam_20pct: should drop ~0 (negative control)
  - top_tam_20pct vs random_20pct paired t-stat (Wilcoxon could be added)
  - logp_drop CDF by strategy

Usage::

    python -m mllmopd.analysis.tam_step2_analysis \\
        --jsonl runs/audit/tam_step2_<TS>/tam_step2.jsonl \\
        --out-json runs/analysis/tam_step2_v0.json \\
        --out-txt  runs/analysis/tam_step2_v0.txt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


RANDOM_STRATEGIES = [
    "random_20pct_seed_42",
    "random_20pct_seed_43",
    "random_20pct_seed_44",
]
SCRAMBLED_STRATEGIES = [
    "scrambled_tam_seed_142",
    "scrambled_tam_seed_143",
    "scrambled_tam_seed_144",
]
NAMED_STRATEGIES = [
    "top_tam_20pct",
    "random_20pct_seed_42",
    "random_20pct_seed_43",
    "random_20pct_seed_44",
    "scrambled_tam_seed_142",
    "scrambled_tam_seed_143",
    "scrambled_tam_seed_144",
    "keep_top_tam_20pct",
    "bottom_tam_20pct",
]


def _wilcoxon_signed_rank(diffs: list) -> dict | None:
    """One-sample Wilcoxon signed-rank for paired differences.
    Returns {W_plus, z, p_two_sided, n_nonzero} or None.

    Drops zero-differences (Wilcoxon convention), assigns ranks to |d|
    with average for ties, uses normal approximation for p (valid for
    n_nonzero ≥ ~20). For heavy-tail Δ distributions this is more robust
    than the paired t-stat.
    """
    nz = [d for d in diffs if abs(d) > 1e-12]
    n = len(nz)
    if n < 5:
        return None
    # Sort by absolute value, keeping sign
    sorted_pairs = sorted(((abs(d), 1 if d > 0 else -1) for d in nz),
                          key=lambda x: x[0])
    # Assign ranks (average for ties)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_pairs[j + 1][0] == sorted_pairs[i][0]:
            j += 1
        avg_rank = (i + j + 2) / 2  # 1-indexed average
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    W_plus = sum(ranks[k] for k in range(n) if sorted_pairs[k][1] > 0)
    # Normal approximation under H0 (no continuity correction)
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    if var <= 0:
        return None
    z = (W_plus - mean) / math.sqrt(var)
    p_two = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return {
        "W_plus": float(W_plus),
        "z": float(z),
        "p_two_sided": float(p_two),
        "n_nonzero": n,
        "n_total": len(diffs),
    }


def _bootstrap_ci_mean(diffs: list, n_resamples: int = 2000,
                       alpha: float = 0.05, seed: int = 1729) -> dict | None:
    """Bootstrap CI for mean of diffs. Heavy-tail-safe (no t-distribution
    assumption). Returns {mean, lo, hi, n, n_resamples}."""
    vals = [d for d in diffs if d is not None
            and not (isinstance(d, float) and math.isnan(d))]
    if len(vals) < 5:
        return None
    rng = random.Random(seed)
    n = len(vals)
    boots = []
    for _ in range(n_resamples):
        sample = [rng.choice(vals) for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    nb = len(boots)
    return {
        "mean": sum(boots) / nb,
        "lo":   boots[int(nb * (alpha / 2))],
        "hi":   boots[int(nb * (1 - alpha / 2))],
        "n":    n,
        "n_resamples": nb,
    }


def _dist(vs: list) -> dict:
    vs = [v for v in vs if v is not None
          and not (isinstance(v, float) and math.isnan(v))]
    n = len(vs)
    if n == 0:
        return {"n": 0}
    vs = sorted(vs)
    return {
        "n":   n,
        "mean": sum(vs) / n,
        "p5":  vs[max(0, (n * 5)  // 100)],
        "p50": vs[n // 2],
        "p95": vs[min(n - 1, (n * 95) // 100)],
        "min": vs[0],
        "max": vs[-1],
    }


def _paired_mean_diff(rows_by_token: dict, a: str, b: str) -> dict:
    """For each token_uid, compute logp_drop[a] - logp_drop[b].
    Returns {n, mean_diff, p5, p50, p95, frac_positive, paired_t_stat}."""
    diffs = []
    for uid, by_strat in rows_by_token.items():
        ra = by_strat.get(a)
        rb = by_strat.get(b)
        if ra is None or rb is None:
            continue
        da = ra.get("logp_drop")
        db = rb.get("logp_drop")
        if da is None or db is None:
            continue
        diffs.append(da - db)
    d = _dist(diffs)
    d["frac_positive"] = (
        sum(1 for x in diffs if x > 0) / len(diffs) if diffs else None
    )
    # Paired t-stat (one-sample t on the differences)
    if len(diffs) > 2:
        m = sum(diffs) / len(diffs)
        var = sum((x - m) ** 2 for x in diffs) / (len(diffs) - 1)
        se = math.sqrt(var / len(diffs)) if var > 0 else 0.0
        d["paired_t_stat"] = (m / se) if se > 0 else None
    return d


def _stratum_breakdown(rows_by_token: dict, key_field: str) -> dict:
    """Group token_uids by the value of key_field (e.g. token_category,
    stratum, quad). Returns nested dict: key → strategy → dist."""
    by_key_strat_vals: dict = defaultdict(lambda: defaultdict(list))
    for uid, by_strat in rows_by_token.items():
        # All rows for this token share the same category/stratum etc;
        # peek any one.
        any_row = next(iter(by_strat.values()))
        key_val = any_row.get(key_field)
        for strat, r in by_strat.items():
            d = r.get("logp_drop")
            if d is None:
                continue
            by_key_strat_vals[key_val][strat].append(d)

    out: dict = {}
    for key_val, strat_vals in by_key_strat_vals.items():
        block = {}
        for strat in NAMED_STRATEGIES:
            block[strat] = _dist(strat_vals.get(strat, []))
        # Top-TAM vs random (pooled across seeds) DROP DELTA
        top_vals = strat_vals.get("top_tam_20pct", [])
        rand_vals = []
        for s in RANDOM_STRATEGIES:
            rand_vals.extend(strat_vals.get(s, []))
        if top_vals and rand_vals:
            block["delta_top_minus_random"] = {
                "mean_top":    sum(top_vals) / len(top_vals),
                "mean_random": sum(rand_vals) / len(rand_vals),
                "mean_delta":  (sum(top_vals) / len(top_vals))
                               - (sum(rand_vals) / len(rand_vals)),
            }
        out[str(key_val)] = block
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, default=None)
    args = ap.parse_args(argv)

    rows = []
    with args.jsonl.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    print(f">>> loaded {len(rows)} rows", file=sys.stderr)

    # Group by token_uid → strategy
    by_uid: dict = defaultdict(dict)
    for r in rows:
        by_uid[r["token_uid"]][r["mask_strategy"]] = r

    n_tokens = len(by_uid)
    print(f">>> {n_tokens} unique tokens", file=sys.stderr)

    # Pooled stats
    pooled: dict = {}
    for strat in NAMED_STRATEGIES:
        vals = [r["logp_drop"] for r in rows
                if r["mask_strategy"] == strat and r.get("logp_drop") is not None]
        pooled[strat] = _dist(vals)

    # Headline A: paired delta = top_tam_20pct − mean(random_20pct seeds)
    paired_deltas = []
    paired_top_vs_scrambled = []   # GPT round: spatial-structure-only control
    for uid, by_strat in by_uid.items():
        top = by_strat.get("top_tam_20pct", {}).get("logp_drop")
        rands = [by_strat.get(s, {}).get("logp_drop") for s in RANDOM_STRATEGIES]
        rands = [v for v in rands if v is not None]
        scrs  = [by_strat.get(s, {}).get("logp_drop") for s in SCRAMBLED_STRATEGIES]
        scrs  = [v for v in scrs if v is not None]
        if top is not None and rands:
            paired_deltas.append(top - sum(rands) / len(rands))
        if top is not None and scrs:
            paired_top_vs_scrambled.append(top - sum(scrs) / len(scrs))
    headline_paired = _dist(paired_deltas)
    if paired_deltas:
        m = sum(paired_deltas) / len(paired_deltas)
        var = sum((x - m) ** 2 for x in paired_deltas) / max(1, len(paired_deltas) - 1)
        se = math.sqrt(var / len(paired_deltas)) if var > 0 else 0.0
        headline_paired["paired_t_stat"] = (m / se) if se > 0 else None
        headline_paired["frac_positive"] = (
            sum(1 for x in paired_deltas if x > 0) / len(paired_deltas)
        )
        headline_paired["wilcoxon"]    = _wilcoxon_signed_rank(paired_deltas)
        headline_paired["bootstrap_ci"] = _bootstrap_ci_mean(paired_deltas)

    # Headline B: paired delta = top_tam_20pct − mean(scrambled_tam_seeds)
    # If TAM's spatial structure is causal (not just "20% area masked"),
    # this Δ should be POSITIVE and large too. If TAM ≈ scrambled, the
    # spatial structure claim fails.
    headline_vs_scrambled = _dist(paired_top_vs_scrambled)
    if paired_top_vs_scrambled:
        m = sum(paired_top_vs_scrambled) / len(paired_top_vs_scrambled)
        var = sum((x - m) ** 2 for x in paired_top_vs_scrambled) / max(1, len(paired_top_vs_scrambled) - 1)
        se = math.sqrt(var / len(paired_top_vs_scrambled)) if var > 0 else 0.0
        headline_vs_scrambled["paired_t_stat"] = (m / se) if se > 0 else None
        headline_vs_scrambled["frac_positive"] = (
            sum(1 for x in paired_top_vs_scrambled if x > 0)
            / len(paired_top_vs_scrambled)
        )
        headline_vs_scrambled["wilcoxon"]    = _wilcoxon_signed_rank(paired_top_vs_scrambled)
        headline_vs_scrambled["bootstrap_ci"] = _bootstrap_ci_mean(paired_top_vs_scrambled)

    # Headline C: paired delta = random_20pct − scrambled_tam_20pct
    # Should be ≈ 0 (both are uniform-random spatial masks). If non-zero,
    # there's some hidden artifact in the masking pipeline.
    paired_random_vs_scrambled = []
    for uid, by_strat in by_uid.items():
        rands = [by_strat.get(s, {}).get("logp_drop") for s in RANDOM_STRATEGIES]
        rands = [v for v in rands if v is not None]
        scrs  = [by_strat.get(s, {}).get("logp_drop") for s in SCRAMBLED_STRATEGIES]
        scrs  = [v for v in scrs if v is not None]
        if rands and scrs:
            paired_random_vs_scrambled.append(
                sum(rands) / len(rands) - sum(scrs) / len(scrs)
            )
    headline_sanity_scrambled = _dist(paired_random_vs_scrambled)
    if paired_random_vs_scrambled:
        headline_sanity_scrambled["bootstrap_ci"] = _bootstrap_ci_mean(paired_random_vs_scrambled)

    # Breakdowns
    by_category = _stratum_breakdown(by_uid, "token_category")
    by_stratum  = _stratum_breakdown(by_uid, "stratum")

    report = {
        "n_rows":            len(rows),
        "n_unique_tokens":   n_tokens,
        "pooled_by_strategy": pooled,
        # Headline A — old causal claim (top-TAM vs uniform-random)
        "headline_top_minus_random":    headline_paired,
        # Headline B — pure spatial-structure test (top-TAM vs scrambled-TAM)
        "headline_top_minus_scrambled": headline_vs_scrambled,
        # Sanity — random vs scrambled-TAM should be ≈ 0 (both uniform random)
        "headline_random_minus_scrambled_sanity": headline_sanity_scrambled,
        "by_token_category": by_category,
        "by_stratum":        by_stratum,
        "jsonl":             str(args.jsonl),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f">>> wrote {args.out_json}", file=sys.stderr)

    if args.out_txt:
        lines: list[str] = []
        lines.append(f"# Step 2 causal masking report  (n_rows={len(rows)}, n_tokens={n_tokens})")
        lines.append("")
        lines.append("## Pooled logp_drop by mask strategy")
        lines.append(f"{'strategy':<28} {'n':>6}  {'mean':>8} {'p50':>8} {'p95':>8}")
        for strat in NAMED_STRATEGIES:
            d = pooled.get(strat, {"n": 0})
            if d["n"] == 0:
                continue
            lines.append(f"{strat:<28} {d['n']:>6d}  "
                         f"{d['mean']:>+8.4f} {d['p50']:>+8.4f} {d['p95']:>+8.4f}")
        lines.append("")

        def _hbody(h: dict, name: str) -> None:
            if h.get("n", 0) == 0:
                return
            lines.append(f"  n_tokens = {h['n']}")
            lines.append(f"  mean Δ   = {h['mean']:+.4f}")
            lines.append(f"  p50 Δ    = {h['p50']:+.4f}")
            lines.append(f"  p95 Δ    = {h['p95']:+.4f}")
            lines.append(f"  frac_positive (Δ > 0) = {h.get('frac_positive', float('nan')):+.3f}")
            lines.append(f"  paired t-stat = {h.get('paired_t_stat')!s}")
            wil = h.get("wilcoxon") or {}
            if wil:
                lines.append(f"  Wilcoxon signed-rank: z={wil.get('z'):+.3f}  "
                             f"p≈{wil.get('p_two_sided'):.2e}  "
                             f"n_nonzero={wil.get('n_nonzero')}")
            ci = h.get("bootstrap_ci") or {}
            if ci:
                lines.append(f"  Bootstrap CI (95%, n_resamples={ci.get('n_resamples')}): "
                             f"[{ci.get('lo'):+.4f}, {ci.get('hi'):+.4f}]  "
                             f"mean_of_means={ci.get('mean'):+.4f}")

        lines.append("## HEADLINE A: paired Δ(top_tam_20pct − mean_random_20pct)")
        _hbody(headline_paired, "A")
        lines.append("")
        lines.append("  Δ > 0 means top-TAM mask hurts MORE than uniform random mask.")
        lines.append("  This is the original Step 2 causal claim.")
        lines.append("")

        lines.append("## HEADLINE B: paired Δ(top_tam_20pct − mean_scrambled_tam_20pct)")
        _hbody(headline_vs_scrambled, "B")
        lines.append("")
        lines.append("  Δ > 0 means TAM's SPATIAL STRUCTURE is causal, not just value distribution.")
        lines.append("  If ≈ Headline A → confirms; if ≪ A → effect is value-distribution artifact.")
        lines.append("")

        lines.append("## SANITY: paired Δ(mean_random − mean_scrambled_tam)")
        sn = headline_sanity_scrambled
        if sn.get("n", 0) > 0:
            lines.append(f"  n_tokens = {sn['n']}  mean Δ = {sn['mean']:+.4f}  "
                         f"p50 = {sn['p50']:+.4f}")
            ci = sn.get("bootstrap_ci") or {}
            if ci:
                lines.append(f"  Bootstrap 95% CI: [{ci.get('lo'):+.4f}, {ci.get('hi'):+.4f}]")
            lines.append("  Should be ≈ 0 — both are uniform-random spatial masks.")
        lines.append("")

        lines.append("## By token_category")
        for cat, block in by_category.items():
            d = block.get("delta_top_minus_random")
            if d is None:
                continue
            top_n = block.get("top_tam_20pct", {}).get("n", 0)
            lines.append(f"  {cat:<22} n_top={top_n:>5d}  "
                         f"mean_top={d['mean_top']:+.4f}  "
                         f"mean_random={d['mean_random']:+.4f}  "
                         f"Δ={d['mean_delta']:+.4f}")
        lines.append("")

        lines.append("## By stratum (which selection rule the token was picked under)")
        for strat_name, block in by_stratum.items():
            d = block.get("delta_top_minus_random")
            if d is None:
                continue
            top_n = block.get("top_tam_20pct", {}).get("n", 0)
            lines.append(f"  {strat_name:<28} n_top={top_n:>5d}  "
                         f"mean_top={d['mean_top']:+.4f}  "
                         f"mean_random={d['mean_random']:+.4f}  "
                         f"Δ={d['mean_delta']:+.4f}")
        lines.append("")

        lines.append("## Inverse / negative controls")
        keep = pooled.get("keep_top_tam_20pct", {})
        bot  = pooled.get("bottom_tam_20pct", {})
        if keep.get("n", 0):
            lines.append(f"  keep_top_tam_20pct  n={keep['n']:>5d}  mean drop={keep['mean']:+.4f} "
                         f"  ← should be LARGE (only 20% pixels remain)")
        if bot.get("n", 0):
            lines.append(f"  bottom_tam_20pct    n={bot['n']:>5d}  mean drop={bot['mean']:+.4f} "
                         f"  ← should be SMALL (negative control)")

        args.out_txt.parent.mkdir(parents=True, exist_ok=True)
        args.out_txt.write_text("\n".join(lines) + "\n")
        print(f">>> wrote summary: {args.out_txt}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
