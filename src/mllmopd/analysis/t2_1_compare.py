"""T2-1 headline analyzer: VD-weighted FullTeacher OPD vs vanilla FullTeacher OPD.

Forked from ``t1_compare.py`` with the arm relabeling required for the
T2-1 method-tier comparison:

    T1 setup:    treatment = T1-2 (Full),   control = T1-3 (Blank)
    T2-1 setup:  treatment = T2-1 (Full+VD), control = T1-2 (Full uniform)

Both designs share the same baseline (T1-0 = MMR1-3B-SFT pre-OPD).

Consumes the 9-file eval grid that ``scripts/audit/run_t2_1_eval.sh``
produces:

  T1-0  (untrained baseline) — from ``--baseline-dir`` or ``--t2-1-run-dir`` ->
        ``S_full.jsonl`` / ``S_blank.jsonl`` / ``S_text_only.jsonl``
        OR ``T1_0_full.jsonl`` / ``T1_0_blank.jsonl`` / ``T1_0_text_only.jsonl``
  T1-2  (vanilla FullTeacher OPD, control) — from ``--t2-1-run-dir`` ->
        ``T1_2_full.jsonl`` / ``T1_2_blank.jsonl`` / ``T1_2_text_only.jsonl``
  T2-1  (VD-weighted FullTeacher OPD, treatment) — from ``--t2-1-run-dir`` ->
        ``T2_1_full.jsonl`` / ``T2_1_blank.jsonl`` / ``T2_1_text_only.jsonl``

Rescores every row through ``mllmopd.diagnostics.scorers`` so the numbers
agree with ``aggregate_audit.py`` (single source of truth).

==============================================================================
Headline metric
==============================================================================

Per benchmark `b`, for each arm A in {T1-2, T2-1}, and each mode m in
{full_image, blank_image, text_only}::

    Acc[A, m, b] = mean(is_correct) over 200 prompts in benchmark b
    G_full[A, b] = Acc[A, full, b] - Acc[T1-0, full, b]
    G_blank[A, b] = Acc[A, blank, b] - Acc[T1-0, blank, b]
    G_gap[A, b]  = G_full[A, b] - G_blank[A, b]

The headline scalar tests whether VD weighting beats uniform OPD::

    Δ_b = G_full[T2-1, b] - G_full[T1-2, b]
        = Acc[T2-1, full, b] - Acc[T1-2, full, b]   (T1-0 cancels)
    Δ   = mean over benchmarks of Δ_b

Same opd_target_recovery diagnostic as t1_compare (now reported per
arm for both T1-2 and T2-1), measured against the *same* opd_target
subset (from ``--opd-target-ids``, default = baseline-dir's
opd_target_ids.json — apples-to-apples with T1 brief v2).

==============================================================================
Statistics
==============================================================================

  * Bootstrap 95% CI on Δ via paired prompt-level resampling (1000
    resamples default; same as T1).
  * McNemar exact paired test on the opd_target prompts: T2-1
    correct/wrong vs T1-2 correct/wrong.

==============================================================================
Decision tree (T2-1 method-signal context)
==============================================================================

| Outcome                                      | Verdict                       |
|----------------------------------------------|-------------------------------|
| Δ > 0.03 AND opd_target_recovery[T2-1] > T1-2| "T2-1 >> T1-2 -> method works"|
| Δ > 0.005                                    | "T2-1 > T1-2 -> marginal"     |
| |Δ| < 0.005                                  | "T2-1 ~= T1-2 -> no signal"   |
| Δ < -0.005                                   | "T2-1 < T1-2 -> weighting hurts"|
| otherwise                                    | "ambiguous"                   |

Thresholds are pre-registered in docs/t2_1_design.md §Predicted outcomes.
"T2-1 results only support method-signal claims, NOT OPD-specific
mechanism claims" — that requires Tier-2 off-policy KD controls.

==============================================================================
Usage
==============================================================================

    python -m mllmopd.analysis.t2_1_compare \\
        --baseline-dir runs/audit/level1_v4_sysprompt_fixed \\
        --t2-1-run-dir ${MLLMOPD_RUNS}/audit/t2_1_eval_20260522-... \\
        [--opd-target-ids runs/audit/level1_v4_sysprompt_fixed/opd_target_ids.json] \\
        [--out-json ${MLLMOPD_RUNS}/audit/t2_1_eval_.../t2_1_compare.json] \\
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

# Baseline T1-0 may live under two naming conventions:
#   - S_full.jsonl (legacy from T1 plan; level1_v4_sysprompt_fixed)
#   - T1_0_full.jsonl (current run_t2_1_eval.sh output when SKIP_T1_0=0)
# _load_arm tries the explicit map first; if a file is missing we fall back
# to the alt map and re-try. This lets us point --baseline-dir at either
# the canonical level1_v4 dir OR the current eval run dir.
_T1_0_FILES_S = {
    "full_image": "S_full.jsonl",
    "blank_image": "S_blank.jsonl",
    "text_only": "S_text_only.jsonl",
}
_T1_0_FILES_NEW = {
    "full_image": "T1_0_full.jsonl",
    "blank_image": "T1_0_blank.jsonl",
    "text_only": "T1_0_text_only.jsonl",
}
_T1_2_FILES = {
    "full_image": "T1_2_full.jsonl",
    "blank_image": "T1_2_blank.jsonl",
    "text_only": "T1_2_text_only.jsonl",
}
_T2_1_FILES = {
    "full_image": "T2_1_full.jsonl",
    "blank_image": "T2_1_blank.jsonl",
    "text_only": "T2_1_text_only.jsonl",
}
_MODES = ("full_image", "blank_image", "text_only")
_ARMS = ("T1-0", "T1-2", "T2-1")


def _rescore_row(r: dict) -> Optional[bool]:
    """Re-score one row using current scorers (single source of truth)."""
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


def _load_arm_files(run_dir: Path, files: dict[str, str]) -> dict:
    """Load one arm's jsonls. Missing modes are reported but tolerated."""
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


def _load_t1_0(baseline_dir: Path, run_dir: Path) -> dict:
    """Load T1-0 from either S_*.jsonl (legacy) or T1_0_*.jsonl (current).

    Priority: try baseline_dir/S_*.jsonl first (apples-to-apples with the
    canonical T1 brief baseline); fall back to run_dir/T1_0_*.jsonl if
    user pointed --baseline-dir at a dir without S_*.jsonl (e.g., the
    same run dir as T2-1 eval, when SKIP_T1_0=0)."""
    # First, see if the baseline_dir has S_*.jsonl.
    s_path = baseline_dir / _T1_0_FILES_S["full_image"]
    if s_path.exists():
        return _load_arm_files(baseline_dir, _T1_0_FILES_S)
    # Else try T1_0_* in run_dir.
    t10_path = run_dir / _T1_0_FILES_NEW["full_image"]
    if t10_path.exists():
        print(f">>> T1-0 fallback: using {run_dir}/T1_0_*.jsonl (no S_*.jsonl in {baseline_dir})",
              file=sys.stderr)
        return _load_arm_files(run_dir, _T1_0_FILES_NEW)
    # Else try T1_0_* in baseline_dir (in case caller pointed there directly).
    if (baseline_dir / _T1_0_FILES_NEW["full_image"]).exists():
        return _load_arm_files(baseline_dir, _T1_0_FILES_NEW)
    sys.exit(f"!! T1-0 baseline not found: tried {baseline_dir}/S_*.jsonl, "
             f"{run_dir}/T1_0_*.jsonl, {baseline_dir}/T1_0_*.jsonl")


# -----------------------------------------------------------------------------
# Accuracy primitives (identical to t1_compare; kept inline for self-containment)
# -----------------------------------------------------------------------------


def _acc(d: dict[str, Optional[bool]], ids: Optional[set] = None) -> Optional[float]:
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
# Per-arm metric table
# -----------------------------------------------------------------------------


def _per_benchmark_metrics(
    arms: dict[str, dict],
    opd_target_ids: dict[str, list[str]],
    eps: float = 0.01,
) -> dict:
    """Compute Acc, G_full, G_blank, G_text, G_gap, blank_leakage, opd_target_*
    per (arm, benchmark). T1-0 is reference for G_*."""
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
        for arm in ("T1-2", "T2-1"):
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


# -----------------------------------------------------------------------------
# Headline Δ + bootstrap CI
# -----------------------------------------------------------------------------


def _delta_per_benchmark(table: dict) -> dict[str, Optional[float]]:
    """Δ_b = G_full[T2-1, b] - G_full[T1-2, b]."""
    return {
        b: _safe_sub(table[b]["T2-1"].get("G_full"), table[b]["T1-2"].get("G_full"))
        for b in table
    }


def _bootstrap_delta(
    arms: dict[str, dict],
    n_resamples: int,
    seed: int,
) -> tuple[Optional[float], Optional[float], list[float]]:
    """Paired prompt-level bootstrap. Resamples the SAME id-indices for all
    three arms (T1-0, T1-2, T2-1) on full_image, recomputes G_full per arm
    per benchmark, then Δ = mean_b (G_full[T2-1] - G_full[T1-2]).
    Returns (low, high) 2.5/97.5 percentiles and the raw distribution."""
    rng = random.Random(seed)
    bench_to_ids: dict[str, list[str]] = {}
    full = {a: arms[a].get("full_image", {}) for a in _ARMS}
    benches = sorted(set().union(*[set(full[a]) for a in _ARMS]))
    for b in benches:
        common: set | None = None
        for a in _ARMS:
            ids_a = {i for i, v in full[a].get(b, {}).items() if v is not None}
            common = ids_a if common is None else (common & ids_a)
        bench_to_ids[b] = sorted(common or set())
    usable_benches = [b for b in benches if bench_to_ids[b]]
    if not usable_benches:
        return None, None, []

    deltas: list[float] = []
    for _ in range(n_resamples):
        per_bench_diffs: list[float] = []
        for b in usable_benches:
            ids = bench_to_ids[b]
            n = len(ids)
            idxs = [rng.randrange(n) for _ in range(n)]
            sample_ids = [ids[i] for i in idxs]
            acc = {}
            for a in _ARMS:
                cell = full[a].get(b, {})
                vals = [cell[i] for i in sample_ids]
                acc[a] = sum(1 for v in vals if v) / n
            g_t1_2 = acc["T1-2"] - acc["T1-0"]
            g_t2_1 = acc["T2-1"] - acc["T1-0"]
            per_bench_diffs.append(g_t2_1 - g_t1_2)
        deltas.append(sum(per_bench_diffs) / len(per_bench_diffs))
    deltas_sorted = sorted(deltas)
    lo = deltas_sorted[int(0.025 * len(deltas_sorted))]
    hi = deltas_sorted[int(0.975 * len(deltas_sorted)) - 1]
    return lo, hi, deltas


# -----------------------------------------------------------------------------
# McNemar exact binomial test
# -----------------------------------------------------------------------------


def _mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    log_half_n = -n * math.log(2.0)
    cum = 0.0
    log_coef = 0.0
    cum += math.exp(log_coef + log_half_n)
    for i in range(1, k + 1):
        log_coef += math.log((n - i + 1) / i)
        cum += math.exp(log_coef + log_half_n)
    return min(1.0, 2.0 * cum)


def _mcnemar(arms: dict[str, dict], opd_target_ids: dict[str, list[str]]) -> dict:
    """McNemar exact paired test on T2-1 vs T1-2 over all opd_target prompts
    on full_image. b = T2-1 correct & T1-2 wrong (treatment win);
    c = T2-1 wrong & T1-2 correct (treatment lose)."""
    full_t12 = arms["T1-2"].get("full_image", {})
    full_t21 = arms["T2-1"].get("full_image", {})
    cnt = {"both_correct": 0, "both_wrong": 0, "b": 0, "c": 0}
    n = 0
    for bench, ids in opd_target_ids.items():
        d12 = full_t12.get(bench, {})
        d21 = full_t21.get(bench, {})
        for rid in ids:
            v12, v21 = d12.get(rid), d21.get(rid)
            if v12 is None or v21 is None:
                continue
            if v21 and v12:
                cnt["both_correct"] += 1
            elif v21 and not v12:
                cnt["b"] += 1  # T2-1 right, T1-2 wrong → treatment win
            elif not v21 and v12:
                cnt["c"] += 1  # T2-1 wrong, T1-2 right → treatment lose
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
# Verdict (decision tree from docs/t2_1_design.md §Predicted outcomes)
# -----------------------------------------------------------------------------


def _verdict(delta: Optional[float], ci: tuple, table: dict) -> str:
    if delta is None:
        return "ambiguous"
    rec_t1_2 = _mean([table[b]["T1-2"].get("opd_target_recovery") for b in table])
    rec_t2_1 = _mean([table[b]["T2-1"].get("opd_target_recovery") for b in table])
    rec_diff = _safe_sub(rec_t2_1, rec_t1_2) or 0.0
    if delta > 0.03 and rec_diff > 0.01:
        return "T2-1 >> T1-2 -> method signal (next: τ×β sweep + Tier-2 falsifier)"
    if delta > 0.005:
        return "T2-1 > T1-2 -> marginal (next: τ×β sweep)"
    if abs(delta) < 0.005:
        return "T2-1 ~= T1-2 -> no method signal (next: T2-3 β-residual or T2-4 oversample)"
    if delta < -0.005:
        return "T2-1 < T1-2 -> reweighting hurts (next: hard top-k mask VPPO-style)"
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
        for arm in ("T1-2", "T2-1"):
            c = table[b][arm]
            print(f"{b:16s} {arm:5s} " +
                  f"{_fmt(c.get('G_full'))} {_fmt(c.get('G_blank'))} "
                  f"{_fmt(c.get('G_text'))} {_fmt(c.get('G_gap'))} "
                  f"{_fmt(c.get('blank_leakage'), 8)} "
                  f"{_fmt(c.get('opd_target_recovery'))} "
                  f"{c.get('n_opd_target', 0):>8d}")
    print("-" * len(hdr))
    avg = {}
    for arm in ("T1-2", "T2-1"):
        for key in ("G_full", "G_blank", "G_text", "G_gap",
                    "blank_leakage", "opd_target_recovery"):
            avg[(arm, key)] = _mean([table[b][arm].get(key) for b in benches])
    for arm in ("T1-2", "T2-1"):
        print(f"{'MEAN':16s} {arm:5s} " +
              f"{_fmt(avg[(arm,'G_full')])} {_fmt(avg[(arm,'G_blank')])} "
              f"{_fmt(avg[(arm,'G_text')])} {_fmt(avg[(arm,'G_gap')])} "
              f"{_fmt(avg[(arm,'blank_leakage')], 8)} "
              f"{_fmt(avg[(arm,'opd_target_recovery')])}")
    print()
    print(f"HEADLINE   Δ = G_full[T2-1] − G_full[T1-2] = {_fmt(delta, 7, 4)}")
    if ci[0] is not None:
        print(f"           95% bootstrap CI = [{_fmt(ci[0], 6, 4)}, {_fmt(ci[1], 6, 4)}]")
    print(f"McNemar on opd_target (n={mc.get('n', 0)}): "
          f"b (T2-1 win)={mc.get('b', 0)} c (T2-1 lose)={mc.get('c', 0)} "
          f"p={mc.get('p_value', 1.0):.4g} [{mc.get('backend','?')}]")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="T2-1 headline analyzer (VD-weighted FullTeacher OPD vs vanilla FullTeacher OPD).",
    )
    ap.add_argument("--baseline-dir", type=Path,
                    default=Path("runs/audit/level1_v4_sysprompt_fixed"),
                    help="T1-0 source dir. Reads S_*.jsonl if present, else "
                         "T1_0_*.jsonl in --t2-1-run-dir as fallback.")
    ap.add_argument("--t2-1-run-dir", type=Path, required=True,
                    help="T2-1 eval output dir. Reads T1_2_*.jsonl and T2_1_*.jsonl.")
    ap.add_argument("--opd-target-ids", type=Path, default=None,
                    help="JSON of {bench: [ids]}. Defaults to <baseline-dir>/opd_target_ids.json.")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Output JSON path. Defaults to <t2-1-run-dir>/t2_1_compare.json.")
    ap.add_argument("--bootstrap-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--print-table", action=argparse.BooleanOptionalAction, default=True,
                    help="Render the per-benchmark table to stdout.")
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
    if not args.t2_1_run_dir.exists():
        sys.exit(f"!! t2-1 run dir missing: {args.t2_1_run_dir}")

    arms: dict[str, dict] = {}
    arms["T1-0"] = _load_t1_0(args.baseline_dir, args.t2_1_run_dir)
    arms["T1-2"] = _load_arm_files(args.t2_1_run_dir, _T1_2_FILES)
    arms["T2-1"] = _load_arm_files(args.t2_1_run_dir, _T2_1_FILES)
    for a in _ARMS:
        n = sum(len(d) for mode_map in arms[a].values() for d in mode_map.values())
        print(f">>> {a}: loaded {n} (mode, bench, id) records", file=sys.stderr)

    table = _per_benchmark_metrics(arms, opd_target_ids)
    delta_per_b = _delta_per_benchmark(table)
    delta_headline = _mean(list(delta_per_b.values()))
    lo, hi, _ = _bootstrap_delta(arms, args.bootstrap_n, args.seed)
    mc = _mcnemar(arms, opd_target_ids)
    verdict = _verdict(delta_headline, (lo, hi), table)

    def _slice(metric: str) -> dict:
        return {arm: {b: table[b][arm].get(metric) for b in table}
                for arm in ("T1-2", "T2-1")}

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
            "t2_1_run_dir": str(args.t2_1_run_dir),
            "opd_target_ids": str(opd_ids_path),
            "treatment_arm": "T2-1",
            "control_arm": "T1-2",
        },
    }

    out_path = args.out_json or (args.t2_1_run_dir / "t2_1_compare.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False))
    print(f">>> wrote {out_path}", file=sys.stderr)

    if args.print_table:
        _print_table(table, delta_headline, (lo, hi), mc)
        print()
        print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
