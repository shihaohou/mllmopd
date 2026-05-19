"""T1 headline analyzer: FullTeacher OPD vs BlankTeacher OPD negative control.

This is the analyzer for the T1 OPD baseline experiment (see
``docs/t1-plan-2026-05-19.md``). It consumes the 9-file eval grid that
``scripts/audit/run_t1_eval.sh`` produces:

  T1-0 (baseline, untrained MMR1-3B-SFT) — from ``--baseline-dir`` ->
        ``S_full.jsonl``, ``S_blank.jsonl``, ``S_text_only.jsonl``.
  T1-2 (vanilla OPD, FullTeacher) — from ``--t1-run-dir`` ->
        ``T1_2_full.jsonl``, ``T1_2_blank.jsonl``, ``T1_2_text_only.jsonl``.
  T1-3 (vanilla OPD, BlankTeacher, negative control) — from ``--t1-run-dir`` ->
        ``T1_3_full.jsonl``, ``T1_3_blank.jsonl``, ``T1_3_text_only.jsonl``.

It re-scores every row through ``mllmopd.diagnostics.scorers`` so the numbers
agree with ``aggregate_audit.py`` (single source of truth for `is_correct`).

==============================================================================
Metric — vision-conditioned capability transfer
==============================================================================

Per benchmark `b`, for each arm A in {T1-2, T1-3}, and each mode m in
{full_image, blank_image, text_only}::

    Acc[A, m, b] = mean(is_correct)  over the 200 prompts in benchmark b.

Gain over the untrained student (T1-0)::

    G_full[A, b]  = Acc[A, full,  b] - Acc[T1-0, full,  b]
    G_blank[A, b] = Acc[A, blank, b] - Acc[T1-0, blank, b]
    G_text[A, b]  = Acc[A, text,  b] - Acc[T1-0, text,  b]
    G_gap[A, b]   = G_full[A, b] - G_blank[A, b]   # vision-conditioned slice

The headline scalar tests whether the teacher's *image* signal is what makes
OPD work. If yes, FullTeacher beats BlankTeacher::

    Δ_b = G_full[T1-2, b] - G_full[T1-3, b]
    Δ   = mean over benchmarks of Δ_b           # the headline number

Two derived diagnostics::

    blank_leakage[A, b]        = G_blank[A, b] / max(eps, G_full[A, b])
                                  -> fraction of T1-x's gain that does NOT
                                     depend on the image being there.
    opd_target_recovery[A, b]  = Acc[A, full, opd_target_ids(b)]
                                 - Acc[T1-0, full, opd_target_ids(b)]
                                  -> gain on the 133 vision-critical prompts
                                     where the audit identified teacher's
                                     advantage AS visual.

==============================================================================
Decision tree (matches ``docs/t1-plan-2026-05-19.md`` §0)
==============================================================================

| Outcome                                            | Verdict                |
|----------------------------------------------------|------------------------|
| Δ > 0.01 AND opd_target_recovery[T1-2] > T1-3      | "T1-2 >> T1-3"         |
| |Δ| < 0.005 AND opd_target_recovery roughly equal  | "T1-2 ~= T1-3"         |
| G_gap[T1-2] - G_gap[T1-3] < 0.005                  | "G_gap flat"           |
| otherwise                                          | "ambiguous"            |

==============================================================================
Statistics (Risk #9 — small n requires confidence intervals)
==============================================================================

The audit eval set is only 200 prompts/benchmark and `opd_target` is 133
prompts total. A 1-2pt Δ may be sampling noise. So we emit:

  * **Bootstrap 95% CI on Δ** with `--bootstrap-n` (default 1000) resamples.
    Resampling is paired *at the prompt level*: each resample draws the same
    prompt indices for both T1-2 and T1-3 (and T1-0 baseline) within each
    benchmark, so Δ_b's noise reflects per-prompt label variance, not arm
    independence. Then recompute Δ as mean over benchmarks. Report 2.5 /
    97.5 percentiles.

  * **McNemar exact paired test** on the 133 `opd_target` prompts. For each
    prompt: T1-2 correct/wrong vs T1-3 correct/wrong -> 2x2 contingency. The
    discordant pair counts (`b` = T1-2 correct & T1-3 wrong, `c` = T1-2
    wrong & T1-3 correct) feed the exact binomial test of `H0: b = c`. If
    scipy is available we use ``scipy.stats.binomtest``; otherwise an inline
    two-sided binomial p-value is computed against B(b+c, 0.5).

==============================================================================
Usage
==============================================================================

    python -m mllmopd.analysis.t1_compare \\
        --baseline-dir runs/audit/level1_v4_sysprompt_fixed \\
        --t1-run-dir ${MLLMOPD_RUNS}/audit/t1_eval_v0 \\
        [--opd-target-ids runs/audit/level1_v4_sysprompt_fixed/opd_target_ids.json] \\
        [--out-json ${MLLMOPD_RUNS}/audit/t1_eval_v0/t1_compare.json] \\
        [--bootstrap-n 1000] [--seed 42] [--no-print-table]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Optional

from mllmopd.diagnostics import scorers

# -----------------------------------------------------------------------------
# IO + rescoring
# -----------------------------------------------------------------------------

# Files we expect per arm. Maps (arm_label, mode) -> filename relative to the
# arm's run dir. T1-0 lives in `--baseline-dir`, T1-2/T1-3 in `--t1-run-dir`.
_T1_0_FILES = {
    "full_image": "S_full.jsonl",
    "blank_image": "S_blank.jsonl",
    "text_only": "S_text_only.jsonl",
}
_T1_2_FILES = {
    "full_image": "T1_2_full.jsonl",
    "blank_image": "T1_2_blank.jsonl",
    "text_only": "T1_2_text_only.jsonl",
}
_T1_3_FILES = {
    "full_image": "T1_3_full.jsonl",
    "blank_image": "T1_3_blank.jsonl",
    "text_only": "T1_3_text_only.jsonl",
}
_ARM_FILES = {"T1-0": _T1_0_FILES, "T1-2": _T1_2_FILES, "T1-3": _T1_3_FILES}
_MODES = ("full_image", "blank_image", "text_only")
_ARMS = ("T1-0", "T1-2", "T1-3")


def _rescore_row(r: dict) -> Optional[bool]:
    """Re-score a JSONL row using the current scorer; mirrors
    ``aggregate_audit._rescore`` / ``paired_vision_critical._rescore_row``.

    Returns None for skipped rows (missing image / empty gold)."""
    if r.get("scorer") in {"skip_missing_image", "skip_empty_gold"}:
        return r.get("is_correct")
    pred = r.get("prediction") or ""
    gold = r.get("gold")
    if isinstance(gold, list) and len(gold) == 1:
        gold = gold[0]
    is_c, _, _ = scorers.score_for_benchmark(
        r.get("benchmark", ""), pred, gold, choices=r.get("choices"),
    )
    return is_c


def _load_arm(run_dir: Path, files: dict[str, str]) -> dict:
    """Load one arm's three jsonls. Returns
    ``{mode: {benchmark: {id: is_correct_or_None}}}``.

    Missing files are reported and treated as empty (downstream cells become
    None / "n/a"); the arm doesn't crash the whole run."""
    out: dict[str, dict[str, dict[str, Optional[bool]]]] = {m: {} for m in _MODES}
    for mode, fname in files.items():
        p = run_dir / fname
        if not p.exists():
            print(f"!! missing {p}", file=sys.stderr)
            continue
        with p.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                bench = r["benchmark"]
                rid = r["id"]
                is_c = _rescore_row(r)
                out[mode].setdefault(bench, {})[rid] = is_c
    return out


# -----------------------------------------------------------------------------
# Accuracy primitives
# -----------------------------------------------------------------------------


def _acc(d: dict[str, Optional[bool]], ids: Optional[set] = None) -> Optional[float]:
    """Mean of True/False values in ``d``, restricted to ``ids`` if given.
    Returns None when there are no valid (non-None) entries."""
    if ids is None:
        vals = [v for v in d.values() if v is not None]
    else:
        vals = [d[i] for i in ids if i in d and d[i] is not None]
    if not vals:
        return None
    return sum(1 for v in vals if v) / len(vals)


def _benches(arm_data: dict) -> set:
    s: set = set()
    for mode_map in arm_data.values():
        s.update(mode_map.keys())
    return s


# -----------------------------------------------------------------------------
# Per-arm metric table
# -----------------------------------------------------------------------------


def _per_benchmark_metrics(
    arms: dict[str, dict],
    opd_target_ids: dict[str, list[str]],
    eps: float = 0.01,
) -> dict:
    """Compute Acc, G_full, G_blank, G_text, G_gap, blank_leakage, opd_target_*
    per (arm, benchmark). T1-0 is the reference for G_*."""
    benches = sorted(set().union(*[_benches(arms[a]) for a in _ARMS]))
    out: dict = {b: {} for b in benches}
    for b in benches:
        target_set = set(opd_target_ids.get(b, []))
        for arm in _ARMS:
            cell: dict = {}
            for mode in _MODES:
                cell[f"acc_{mode}"] = _acc(arms[arm].get(mode, {}).get(b, {}))
            cell["opd_target_acc"] = (
                _acc(arms[arm].get("full_image", {}).get(b, {}), ids=target_set)
                if target_set
                else None
            )
            cell["n_opd_target"] = len(target_set)
            out[b][arm] = cell

        base = out[b]["T1-0"]
        for arm in ("T1-2", "T1-3"):
            cell = out[b][arm]
            g_full = _safe_sub(cell["acc_full_image"], base["acc_full_image"])
            g_blank = _safe_sub(cell["acc_blank_image"], base["acc_blank_image"])
            g_text = _safe_sub(cell["acc_text_only"], base["acc_text_only"])
            cell["G_full"] = g_full
            cell["G_blank"] = g_blank
            cell["G_text"] = g_text
            cell["G_gap"] = _safe_sub(g_full, g_blank)
            cell["blank_leakage"] = (
                g_blank / max(eps, g_full) if (g_blank is not None and g_full is not None) else None
            )
            cell["opd_target_recovery"] = _safe_sub(
                cell["opd_target_acc"], base["opd_target_acc"]
            )
    return out


def _safe_sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def _mean(xs: list[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


# -----------------------------------------------------------------------------
# Headline Δ + bootstrap CI
# -----------------------------------------------------------------------------


def _delta_per_benchmark(table: dict) -> dict[str, Optional[float]]:
    return {
        b: _safe_sub(table[b]["T1-2"].get("G_full"), table[b]["T1-3"].get("G_full"))
        for b in table
    }


def _bootstrap_delta(
    arms: dict[str, dict],
    n_resamples: int,
    seed: int,
) -> tuple[Optional[float], Optional[float], list[float]]:
    """Paired prompt-level bootstrap. Within each benchmark we resample the
    same id-indices for all 3 arms (T1-0, T1-2, T1-3) on `full_image`, recompute
    G_full per arm per benchmark, then Δ = mean_b (G_full[T1-2,b] - G_full[T1-3,b]).
    Returns (low, high) 2.5/97.5 percentiles and the raw distribution."""
    rng = random.Random(seed)
    # The bootstrap sample space: ids per benchmark that are scored in ALL THREE
    # arms on full_image. Anything not in the intersection can't be used for
    # paired resampling. Document the trimmed n per benchmark.
    bench_to_ids: dict[str, list[str]] = {}
    full = {a: arms[a].get("full_image", {}) for a in _ARMS}
    benches = sorted(set().union(*[set(full[a]) for a in _ARMS]))
    for b in benches:
        common: set = None  # type: ignore[assignment]
        for a in _ARMS:
            ids_a = {i for i, v in full[a].get(b, {}).items() if v is not None}
            common = ids_a if common is None else (common & ids_a)
        bench_to_ids[b] = sorted(common or set())
    # If any benchmark has 0 paired ids, drop it from the headline.
    usable_benches = [b for b in benches if bench_to_ids[b]]
    if not usable_benches:
        return None, None, []

    deltas: list[float] = []
    for _ in range(n_resamples):
        per_bench_diffs: list[float] = []
        for b in usable_benches:
            ids = bench_to_ids[b]
            n = len(ids)
            # Sample n ids with replacement; same indices for both arms.
            idxs = [rng.randrange(n) for _ in range(n)]
            sample_ids = [ids[i] for i in idxs]
            acc = {}
            for a in _ARMS:
                cell = full[a].get(b, {})
                vals = [cell[i] for i in sample_ids]
                acc[a] = sum(1 for v in vals if v) / n
            g2 = acc["T1-2"] - acc["T1-0"]
            g3 = acc["T1-3"] - acc["T1-0"]
            per_bench_diffs.append(g2 - g3)
        deltas.append(sum(per_bench_diffs) / len(per_bench_diffs))
    deltas_sorted = sorted(deltas)
    lo = deltas_sorted[int(0.025 * len(deltas_sorted))]
    hi = deltas_sorted[int(0.975 * len(deltas_sorted)) - 1]
    return lo, hi, deltas


# -----------------------------------------------------------------------------
# McNemar exact binomial test
# -----------------------------------------------------------------------------


def _mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial p for McNemar: H0 says discordant pairs
    distribute as B(b+c, 0.5). p = P(X <= min(b,c)) * 2 (capped at 1.0),
    computed inline so we don't depend on scipy."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # log-space binomial coefficient to avoid overflow on n up to a few hundred.
    log_half_n = -n * math.log(2.0)
    # Sum P(X = i) for i in [0, k]; then double for two-sided.
    cum = 0.0
    log_coef = 0.0  # log C(n,0) = 0
    cum += math.exp(log_coef + log_half_n)
    for i in range(1, k + 1):
        # log C(n,i) = log C(n,i-1) + log((n-i+1)/i)
        log_coef += math.log((n - i + 1) / i)
        cum += math.exp(log_coef + log_half_n)
    return min(1.0, 2.0 * cum)


def _mcnemar(arms: dict[str, dict], opd_target_ids: dict[str, list[str]]) -> dict:
    """McNemar's exact paired test on T1-2 vs T1-3 over all opd_target prompts
    on `full_image`. Returns {b, c, both_correct, both_wrong, n, p_value}."""
    full2 = arms["T1-2"].get("full_image", {})
    full3 = arms["T1-3"].get("full_image", {})
    cnt = {"both_correct": 0, "both_wrong": 0, "b": 0, "c": 0}
    n = 0
    for bench, ids in opd_target_ids.items():
        d2 = full2.get(bench, {})
        d3 = full3.get(bench, {})
        for rid in ids:
            v2, v3 = d2.get(rid), d3.get(rid)
            if v2 is None or v3 is None:
                continue
            if v2 and v3:
                cnt["both_correct"] += 1
            elif v2 and not v3:
                cnt["b"] += 1
            elif not v2 and v3:
                cnt["c"] += 1
            else:
                cnt["both_wrong"] += 1
            n += 1
    try:
        from scipy.stats import binomtest  # type: ignore
        m = cnt["b"] + cnt["c"]
        if m == 0:
            p = 1.0
        else:
            p = binomtest(min(cnt["b"], cnt["c"]), n=m, p=0.5, alternative="two-sided").pvalue
        backend = "scipy.binomtest"
    except Exception:
        p = _mcnemar_exact_p(cnt["b"], cnt["c"])
        backend = "inline_binomial"
    return {"n": n, "p_value": p, "backend": backend, **cnt}


# -----------------------------------------------------------------------------
# Verdict
# -----------------------------------------------------------------------------


def _verdict(delta: Optional[float], ci: tuple, table: dict) -> str:
    if delta is None:
        return "ambiguous"
    avg_gap2 = _mean([table[b]["T1-2"].get("G_gap") for b in table])
    avg_gap3 = _mean([table[b]["T1-3"].get("G_gap") for b in table])
    gap_diff = _safe_sub(avg_gap2, avg_gap3) or 0.0
    rec2 = _mean([table[b]["T1-2"].get("opd_target_recovery") for b in table])
    rec3 = _mean([table[b]["T1-3"].get("opd_target_recovery") for b in table])
    rec_diff = _safe_sub(rec2, rec3) or 0.0
    if delta > 0.01 and rec_diff > 0.01:
        return "T1-2 >> T1-3 -> method paper"
    if abs(delta) < 0.005 and abs(rec_diff) < 0.005:
        return "T1-2 ~= T1-3 -> negative result"
    if abs(gap_diff) < 0.005:
        return "G_gap flat"
    return "ambiguous"


# -----------------------------------------------------------------------------
# Table renderer
# -----------------------------------------------------------------------------


def _fmt(x: Optional[float], width: int = 7, prec: int = 3) -> str:
    if x is None:
        return f"{'n/a':>{width}s}"
    return f"{x:>{width}.{prec}f}"


def _print_table(table: dict, delta: Optional[float], ci: tuple, mc: dict) -> None:
    hdr = f"{'benchmark':16s} {'arm':5s}" + " " + \
          f"{'G_full':>7s} {'G_blank':>7s} {'G_text':>7s} " \
          f"{'G_gap':>7s} {'blank_lk':>8s} {'opd_rec':>7s} {'n_target':>8s}"
    print()
    print(hdr)
    print("-" * len(hdr))
    benches = sorted(table)
    for b in benches:
        for arm in ("T1-2", "T1-3"):
            c = table[b][arm]
            print(f"{b:16s} {arm:5s} " +
                  f"{_fmt(c.get('G_full'))} {_fmt(c.get('G_blank'))} "
                  f"{_fmt(c.get('G_text'))} {_fmt(c.get('G_gap'))} "
                  f"{_fmt(c.get('blank_leakage'), 8)} "
                  f"{_fmt(c.get('opd_target_recovery'))} "
                  f"{c.get('n_opd_target', 0):>8d}")
    print("-" * len(hdr))
    avg = {}
    for arm in ("T1-2", "T1-3"):
        for key in ("G_full", "G_blank", "G_text", "G_gap",
                    "blank_leakage", "opd_target_recovery"):
            avg[(arm, key)] = _mean([table[b][arm].get(key) for b in benches])
    for arm in ("T1-2", "T1-3"):
        print(f"{'MEAN':16s} {arm:5s} " +
              f"{_fmt(avg[(arm,'G_full')])} {_fmt(avg[(arm,'G_blank')])} "
              f"{_fmt(avg[(arm,'G_text')])} {_fmt(avg[(arm,'G_gap')])} "
              f"{_fmt(avg[(arm,'blank_leakage')], 8)} "
              f"{_fmt(avg[(arm,'opd_target_recovery')])}")
    print()
    print(f"HEADLINE   Δ = (T1-2 − T1-0) − (T1-3 − T1-0) = {_fmt(delta, 7, 4)}")
    if ci[0] is not None:
        print(f"           95% bootstrap CI = [{_fmt(ci[0], 6, 4)}, {_fmt(ci[1], 6, 4)}]")
    print(f"McNemar on opd_target (n={mc.get('n', 0)}): "
          f"b={mc.get('b', 0)} c={mc.get('c', 0)} "
          f"p={mc.get('p_value', 1.0):.4g} [{mc.get('backend','?')}]")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="T1 OPD headline analyzer (FullTeacher vs BlankTeacher).",
    )
    ap.add_argument("--baseline-dir", type=Path,
                    default=Path("runs/audit/level1_v4_sysprompt_fixed"),
                    help="T1-0 source dir. Reads S_full/S_blank/S_text_only jsonls.")
    ap.add_argument("--t1-run-dir", type=Path, required=True,
                    help="T1 eval output dir. Reads T1_{2,3}_{full,blank,text_only}.jsonl.")
    ap.add_argument("--opd-target-ids", type=Path, default=None,
                    help="JSON of {bench: [ids]}. Defaults to <baseline-dir>/opd_target_ids.json.")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Output JSON path. Defaults to <t1-run-dir>/t1_compare.json.")
    ap.add_argument("--bootstrap-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-table", action=argparse.BooleanOptionalAction, default=True,
                    help="Render the per-benchmark markdown-style table to stdout.")
    args = ap.parse_args()

    opd_ids_path = args.opd_target_ids or (args.baseline_dir / "opd_target_ids.json")
    if not opd_ids_path.exists():
        sys.exit(f"!! opd_target_ids file not found: {opd_ids_path}")
    opd_target_ids: dict[str, list[str]] = json.loads(opd_ids_path.read_text())
    n_targets = sum(len(v) for v in opd_target_ids.values())
    print(f">>> opd_target ids: {n_targets} across {len(opd_target_ids)} benchmarks",
          file=sys.stderr)

    if not args.baseline_dir.exists():
        sys.exit(f"!! baseline dir missing: {args.baseline_dir}")
    if not args.t1_run_dir.exists():
        sys.exit(f"!! t1 run dir missing: {args.t1_run_dir}")

    arms: dict[str, dict] = {}
    arms["T1-0"] = _load_arm(args.baseline_dir, _T1_0_FILES)
    arms["T1-2"] = _load_arm(args.t1_run_dir, _T1_2_FILES)
    arms["T1-3"] = _load_arm(args.t1_run_dir, _T1_3_FILES)
    for a in _ARMS:
        n = sum(len(d) for mode_map in arms[a].values() for d in mode_map.values())
        print(f">>> {a}: loaded {n} (mode, bench, id) records", file=sys.stderr)

    table = _per_benchmark_metrics(arms, opd_target_ids)
    delta_per_b = _delta_per_benchmark(table)
    delta_headline = _mean(list(delta_per_b.values()))
    lo, hi, _ = _bootstrap_delta(arms, args.bootstrap_n, args.seed)
    mc = _mcnemar(arms, opd_target_ids)
    verdict = _verdict(delta_headline, (lo, hi), table)

    # Reshape per-benchmark table into {metric: {arm: {bench: value}}} for the
    # JSON output, which is more convenient for downstream plotting.
    def _slice(metric: str) -> dict:
        return {arm: {b: table[b][arm].get(metric) for b in table}
                for arm in ("T1-2", "T1-3")}

    out_payload = {
        "delta_headline": delta_headline,
        "delta_bootstrap_ci_95": [lo, hi],
        "delta_per_benchmark": delta_per_b,
        "G_full": _slice("G_full"),
        "G_blank": _slice("G_blank"),
        "G_text": _slice("G_text"),
        "G_gap": _slice("G_gap"),
        "blank_leakage": _slice("blank_leakage"),
        "opd_target_recovery": _slice("opd_target_recovery"),
        "opd_target_acc": {arm: {b: table[b][arm].get("opd_target_acc") for b in table}
                            for arm in _ARMS},
        "raw_acc": {arm: {b: {m: table[b][arm].get(f"acc_{m}") for m in _MODES}
                            for b in table} for arm in _ARMS},
        "mcnemar": mc,
        "n_bootstrap": args.bootstrap_n,
        "seed": args.seed,
        "decision_tree_verdict": verdict,
        "config": {
            "baseline_dir": str(args.baseline_dir),
            "t1_run_dir": str(args.t1_run_dir),
            "opd_target_ids": str(opd_ids_path),
        },
    }

    out_path = args.out_json or (args.t1_run_dir / "t1_compare.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False))
    print(f">>> wrote {out_path}", file=sys.stderr)

    if args.print_table:
        _print_table(table, delta_headline, (lo, hi), mc)
        print()
        print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
