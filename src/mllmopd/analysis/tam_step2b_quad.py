"""Step 2b — quad-aware analysis of Step 2 causal masking.

Step 2 standalone uses "high tam_mass + non-template" as proxy stratum
because at Step-2 runtime we have no vd / adv (those are Step 1a fields).
But the GPT round flagged this gap: the T2-1 failure mode was the
`vd<0 ∧ adv<0` visual-rejection-correction bucket (quad==3); to claim
TAM-Boost can FIX that, we need to show TAM region has causal effect
specifically on quad==3 tokens.

This script joins Step 1a's per-(sample, ckpt) `quad / vd / adv` arrays
into Step 2's per-(sample, target_token, mask_strategy) rows, then
recomputes Δ_paired and reports per (student_ckpt × quad) breakdown.

Join key: (sample_id, response_hash, token_idx). The response is teacher-
greedy and deterministic across student_ckpts, so response_hash matches
across Step 1a's 4 ckpt rows for a given sample.

Usage::

    python -m mllmopd.analysis.tam_step2b_quad \\
        --step2-jsonl   runs/audit/tam_step2_<TS>/tam_step2.jsonl \\
        --step1a-jsonl  runs/audit/tam_step1a_<TS>/tam_step1a.jsonl \\
        --out-json runs/analysis/tam_step2b_quad_v0.json \\
        --out-txt  runs/analysis/tam_step2b_quad_v0.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Reuse helpers from the existing analyzer.
try:
    from mllmopd.analysis.tam_step2_analysis import (
        _wilcoxon_signed_rank, _bootstrap_ci_mean,
        _bootstrap_ci_mean_sample_clustered, _dist,
        RANDOM_STRATEGIES, SCRAMBLED_STRATEGIES,
    )
except ImportError:                                          # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tam_step2_analysis import (                          # type: ignore[no-redef]
        _wilcoxon_signed_rank, _bootstrap_ci_mean,
        _bootstrap_ci_mean_sample_clustered, _dist,
        RANDOM_STRATEGIES, SCRAMBLED_STRATEGIES,
    )


QUAD_LABEL = {
    0: "vis_support_agree",            # vd≥0 ∧ adv≥0
    1: "vis_support_pushed_away",      # vd≥0 ∧ adv<0
    2: "vis_reject_teacher_toward",    # vd<0 ∧ adv≥0
    3: "vis_reject_correction",        # vd<0 ∧ adv<0   ← T2-1 failure bucket
}


def _load_step1a_lookup(path: Path, fail_on_duplicates: bool = False) -> dict:
    """Build {(sample_id, student_ckpt): {'response_hash', 'R', 'quads',
    'vds', 'advs', 'tokens'}}. Skips rows where tam_valid=false.

    Per GPT static-review (3b290eb): detect duplicate (id, ckpt) keys
    (e.g. from append / rerun / merge). Default = warn + last-wins;
    pass `fail_on_duplicates=True` to raise."""
    out: dict = {}
    n_rows = n_valid = 0
    dup_keys: dict = Counter()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            if not row.get("tam_valid"):
                continue
            R = int(row.get("response_length", 0))
            if R <= 0:
                continue
            key = (row["id"], row.get("student_ckpt"))
            if key in out:
                dup_keys[key] += 1
            out[key] = {
                "response_hash": row.get("response_hash"),
                "R": R,
                "quads": (row.get("quad") or [None] * R)[:R],
                "vds":   (row.get("vd")   or [None] * R)[:R],
                "advs":  (row.get("adv")  or [None] * R)[:R],
            }
            n_valid += 1
    print(f">>> step1a: loaded {n_rows} rows, {n_valid} valid, "
          f"{len(out)} unique (id, ckpt) keys", file=sys.stderr)
    if dup_keys:
        msg = (f"WARNING: {len(dup_keys)} duplicate step1a (id, ckpt) keys; "
               f"last-occurrence wins. Sample duplicates: "
               f"{list(dup_keys.items())[:3]}")
        if fail_on_duplicates:
            raise RuntimeError(msg)
        print(f"!! {msg}", file=sys.stderr)
    return out


def _load_step2_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f">>> step2: loaded {len(rows)} rows", file=sys.stderr)
    return rows


def _organize_step2(rows: list[dict]) -> dict:
    """Group Step 2 rows by token_uid → {strategy: logp_drop}."""
    by_uid: dict = defaultdict(dict)
    for r in rows:
        by_uid[r["token_uid"]][r["mask_strategy"]] = r
    return dict(by_uid)


def _resolve_quad_per_ckpt(step2_token_repr: dict, step1a_lookup: dict,
                            all_ckpts: list[str]) -> dict[str, int | None]:
    """For each ckpt in step1a, look up the quad value at this Step 2
    token's (sample, token_idx) — returns {ckpt: quad} or None."""
    out: dict[str, int | None] = {}
    for ckpt in all_ckpts:
        info = step1a_lookup.get((step2_token_repr["id"], ckpt))
        if info is None:
            out[ckpt] = None
            continue
        if info["response_hash"] != step2_token_repr["response_hash"]:
            # Different teacher greedy decode (unlikely under deterministic
            # greedy + identical teacher) — skip
            out[ckpt] = None
            continue
        ti = step2_token_repr["token_idx"]
        if ti < 0 or ti >= info["R"]:
            out[ckpt] = None
            continue
        q = info["quads"][ti]
        out[ckpt] = q if isinstance(q, int) and 0 <= q <= 3 else None
    return out


def _compute_paired(deltas_top_random: list, deltas_top_scrambled: list,
                    deltas_top_random_with_sid: list | None = None) -> dict:
    """Returns mean/CI/Wilcoxon for both top-vs-random and top-vs-scrambled.

    When `deltas_top_random_with_sid` (list of (sample_id, Δ)) is provided,
    also reports sample-clustered bootstrap CI (per GPT static review 3b290eb)
    — token-level bootstrap underestimates CI when Step 2 picks multiple
    target tokens per sample."""
    out = {
        "n_tokens": len(deltas_top_random),
        "top_minus_random": _dist(deltas_top_random),
        "top_minus_scrambled": _dist(deltas_top_scrambled),
    }
    if deltas_top_random:
        out["top_minus_random"]["bootstrap_ci"] = _bootstrap_ci_mean(deltas_top_random)
        if deltas_top_random_with_sid:
            out["top_minus_random"]["bootstrap_ci_cluster"] = (
                _bootstrap_ci_mean_sample_clustered(deltas_top_random_with_sid)
            )
        out["top_minus_random"]["wilcoxon"]    = _wilcoxon_signed_rank(deltas_top_random)
        out["top_minus_random"]["frac_positive"] = (
            sum(1 for x in deltas_top_random if x > 0) / len(deltas_top_random)
        )
    if deltas_top_scrambled:
        out["top_minus_scrambled"]["bootstrap_ci"] = _bootstrap_ci_mean(deltas_top_scrambled)
        out["top_minus_scrambled"]["wilcoxon"]    = _wilcoxon_signed_rank(deltas_top_scrambled)
        out["top_minus_scrambled"]["frac_positive"] = (
            sum(1 for x in deltas_top_scrambled if x > 0) / len(deltas_top_scrambled)
        )
    return out


def _token_paired_deltas(token_strats: dict) -> tuple[float | None, float | None, float | None]:
    """For one Step 2 token (dict of strategy → row), compute:
      (top − mean_random, top − mean_scrambled, bottom − top).

    bottom − top is the "smoking-gun" test for TAM-inverted-on-q3: if it's
    LARGE POSITIVE on quad==3, it means masking the LEAST-TAM region hurts
    MORE than masking the TOP-TAM region — TAM is pointing at the wrong
    place for q3."""
    top = token_strats.get("top_tam_20pct", {}).get("logp_drop")
    bot = token_strats.get("bottom_tam_20pct", {}).get("logp_drop")
    if top is None:
        return None, None, None
    rands = [token_strats.get(s, {}).get("logp_drop") for s in RANDOM_STRATEGIES]
    rands = [v for v in rands if v is not None]
    scrs  = [token_strats.get(s, {}).get("logp_drop") for s in SCRAMBLED_STRATEGIES]
    scrs  = [v for v in scrs if v is not None]
    d_random = top - (sum(rands) / len(rands)) if rands else None
    d_scram  = top - (sum(scrs) / len(scrs)) if scrs else None
    d_bot_top = bot - top if bot is not None else None
    return d_random, d_scram, d_bot_top


# Raw strategies whose per-(ckpt × quad) mean drops we report verbatim.
RAW_STRATEGIES_TO_REPORT = [
    "top_tam_20pct",
    "random_20pct_seed_42", "random_20pct_seed_43", "random_20pct_seed_44",
    "scrambled_tam_seed_142", "scrambled_tam_seed_143", "scrambled_tam_seed_144",
    "keep_top_tam_20pct",
    "bottom_tam_20pct",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step2-jsonl",  type=Path, required=True)
    ap.add_argument("--step1a-jsonl", type=Path, required=True)
    ap.add_argument("--out-json",     type=Path, required=True)
    ap.add_argument("--out-txt",      type=Path, default=None)
    ap.add_argument(
        "--fail-on-duplicates", action="store_true",
        help="Raise if Step 1a JSONL has duplicate (id, ckpt) rows. "
             "Per GPT static-review on 3b290eb: paper-critical runs should "
             "enable this; default warns + last-wins for ad-hoc analysis.",
    )
    args = ap.parse_args(argv)

    step1a_lookup = _load_step1a_lookup(
        args.step1a_jsonl, fail_on_duplicates=args.fail_on_duplicates,
    )
    step2_rows    = _load_step2_rows(args.step2_jsonl)
    step2_by_uid  = _organize_step2(step2_rows)
    print(f">>> {len(step2_by_uid)} unique step2 token_uids", file=sys.stderr)

    # Distinct student ckpts in step1a (for the cartesian per-ckpt × quad)
    all_ckpts = sorted({k[1] for k in step1a_lookup.keys() if k[1]})
    print(f">>> step1a ckpts: {all_ckpts}", file=sys.stderr)

    # Per (ckpt, quad): collect paired Δ AND raw drops per strategy
    deltas_random: dict    = defaultdict(list)   # (ckpt, q) → [Δ(top − rand)]
    deltas_random_with_sid: dict = defaultdict(list)  # (ckpt, q) → [(sample_id, Δ)]   for cluster bootstrap
    deltas_scrambled: dict = defaultdict(list)
    deltas_bot_top: dict   = defaultdict(list)
    deltas_bot_top_with_sid: dict = defaultdict(list)
    raw_drops_per_strat: dict = defaultdict(lambda: defaultdict(list))
    n_resolved_any = 0
    n_unresolved   = 0
    per_ckpt_total: dict = Counter()       # tokens resolved per ckpt
    per_ckpt_missing: dict = Counter()      # tokens NOT resolved per ckpt
    by_category_per_quad: dict = defaultdict(lambda: defaultdict(list))

    for uid, by_strat in step2_by_uid.items():
        any_row = next(iter(by_strat.values()))
        token_repr = {
            "id":            any_row["id"],
            "token_idx":     any_row["token_idx"],
            "response_hash": any_row.get("response_hash"),
            "token_category": any_row.get("token_category"),
        }
        d_random, d_scram, d_bot_top = _token_paired_deltas(by_strat)
        if d_random is None and d_scram is None:
            continue
        per_ckpt_quad = _resolve_quad_per_ckpt(token_repr, step1a_lookup, all_ckpts)
        # Per-ckpt resolution audit (GPT static-review 3b290eb)
        any_resolved = False
        for ckpt in all_ckpts:
            if per_ckpt_quad.get(ckpt) is None:
                per_ckpt_missing[ckpt] += 1
            else:
                per_ckpt_total[ckpt] += 1
                any_resolved = True
        if not any_resolved:
            n_unresolved += 1
            continue
        n_resolved_any += 1
        sid = token_repr["id"]
        for ckpt, q in per_ckpt_quad.items():
            if q is None:
                continue
            if d_random is not None:
                deltas_random[(ckpt, q)].append(d_random)
                deltas_random_with_sid[(ckpt, q)].append((sid, d_random))
                by_category_per_quad[(ckpt, q)][token_repr["token_category"] or "other"].append(d_random)
            if d_scram is not None:
                deltas_scrambled[(ckpt, q)].append(d_scram)
            if d_bot_top is not None:
                deltas_bot_top[(ckpt, q)].append(d_bot_top)
                deltas_bot_top_with_sid[(ckpt, q)].append((sid, d_bot_top))
            # Raw drops per strategy
            for strat in RAW_STRATEGIES_TO_REPORT:
                drop = by_strat.get(strat, {}).get("logp_drop")
                if drop is not None:
                    raw_drops_per_strat[(ckpt, q)][strat].append(drop)
    print(f">>> resolved {n_resolved_any} tokens (any ckpt); "
          f"unresolved {n_unresolved}", file=sys.stderr)
    for ckpt in all_ckpts:
        tot = per_ckpt_total[ckpt]
        miss = per_ckpt_missing[ckpt]
        print(f"      {ckpt:>10s}: resolved={tot:>4d}  missing={miss:>4d}  "
              f"(of {len(step2_by_uid)} step2 tokens)", file=sys.stderr)
        # Sanity: resolved + missing should equal total step2 tokens
        assert tot + miss == len(step2_by_uid), (
            f"counter mismatch for {ckpt}: {tot}+{miss} ≠ {len(step2_by_uid)}"
        )

    # Per-ckpt × quad stats
    report: dict = {"per_ckpt_per_quad": {}, "all_ckpts": all_ckpts}
    for ckpt in all_ckpts:
        report["per_ckpt_per_quad"][ckpt] = {}
        for q in (0, 1, 2, 3):
            d_r = deltas_random.get((ckpt, q), [])
            d_s = deltas_scrambled.get((ckpt, q), [])
            d_bt = deltas_bot_top.get((ckpt, q), [])
            d_r_sid = deltas_random_with_sid.get((ckpt, q), [])
            block = _compute_paired(d_r, d_s, deltas_top_random_with_sid=d_r_sid)
            block["quad_label"] = QUAD_LABEL[q]
            # Raw mean drops per strategy
            raw = {}
            for strat, vals in raw_drops_per_strat.get((ckpt, q), {}).items():
                if vals:
                    raw[strat] = {"n": len(vals), "mean": sum(vals) / len(vals)}
            # Derived: random_avg & scrambled_avg (mean across seeds)
            r_vals = [v for s in RANDOM_STRATEGIES
                      for v in raw_drops_per_strat.get((ckpt, q), {}).get(s, [])]
            s_vals = [v for s in SCRAMBLED_STRATEGIES
                      for v in raw_drops_per_strat.get((ckpt, q), {}).get(s, [])]
            raw["random_avg"]    = {"n": len(r_vals), "mean": (sum(r_vals)/len(r_vals)) if r_vals else None}
            raw["scrambled_avg"] = {"n": len(s_vals), "mean": (sum(s_vals)/len(s_vals)) if s_vals else None}
            block["raw_strategy_drops"] = raw
            # Bottom-vs-top paired Δ (smoking gun for TAM-inverted-on-q3)
            block["bottom_minus_top"] = _dist(d_bt)
            if d_bt:
                m = sum(d_bt) / len(d_bt)
                block["bottom_minus_top"]["bootstrap_ci"] = _bootstrap_ci_mean(d_bt)
                # Sample-clustered bootstrap (GPT static-review 3b290eb)
                block["bottom_minus_top"]["bootstrap_ci_cluster"] = (
                    _bootstrap_ci_mean_sample_clustered(deltas_bot_top_with_sid.get((ckpt, q), []))
                )
                block["bottom_minus_top"]["wilcoxon"] = _wilcoxon_signed_rank(d_bt)
                block["bottom_minus_top"]["frac_positive"] = (
                    sum(1 for x in d_bt if x > 0) / len(d_bt)
                )
            # Per-category breakdown within this quad
            cat_block = {}
            for cat, vals in by_category_per_quad.get((ckpt, q), {}).items():
                if vals:
                    cat_block[cat] = {
                        "n": len(vals),
                        "mean_delta_random": sum(vals) / len(vals),
                    }
            block["by_category"] = cat_block
            report["per_ckpt_per_quad"][ckpt][str(q)] = block

    report["n_step2_unique_tokens"]   = len(step2_by_uid)
    report["n_tokens_resolved_any"]   = n_resolved_any
    report["n_tokens_unresolved_all"] = n_unresolved
    report["per_ckpt_resolved"]       = dict(per_ckpt_total)
    report["per_ckpt_missing"]        = dict(per_ckpt_missing)
    report["step2_jsonl"]             = str(args.step2_jsonl)
    report["step1a_jsonl"]            = str(args.step1a_jsonl)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f">>> wrote {args.out_json}", file=sys.stderr)

    if args.out_txt:
        lines: list[str] = []
        lines.append(f"# Step 2b quad-aware analysis  "
                     f"(step2 n={len(step2_by_uid)} tokens, "
                     f"resolved_any_ckpt={n_resolved_any}, "
                     f"unresolved_all={n_unresolved})")
        lines.append(f"step2_jsonl  = {args.step2_jsonl}")
        lines.append(f"step1a_jsonl = {args.step1a_jsonl}")
        lines.append("")
        # Per-ckpt resolution audit (GPT static review 3b290eb)
        lines.append("## Per-ckpt resolution audit")
        for ckpt in all_ckpts:
            t = per_ckpt_total[ckpt]
            m = per_ckpt_missing[ckpt]
            lines.append(f"  {ckpt:>10s}  resolved={t:>4d}  missing={m:>4d}  "
                         f"total={t+m:>4d}")
        lines.append("")
        for ckpt in all_ckpts:
            lines.append(f"## {ckpt}")
            # Table 1: paired Δ table (existing)
            lines.append(f"  {'quad':<32} {'n':>5}  "
                         f"{'mean_Δ_random':>13}  "
                         f"{'CI_tok':>15}  "
                         f"{'CI_cluster':>15}  "
                         f"{'frac+':>6}  {'Wilcox p':>10}  "
                         f"{'mean_Δ_scram':>13}")
            for q in (0, 1, 2, 3):
                block = report["per_ckpt_per_quad"][ckpt][str(q)]
                n   = block.get("n_tokens", 0)
                if n == 0:
                    continue
                tr  = block.get("top_minus_random", {})
                ts  = block.get("top_minus_scrambled", {})
                mr  = tr.get("mean", float("nan"))
                ms  = ts.get("mean", float("nan"))
                ci  = tr.get("bootstrap_ci") or {}
                cic = tr.get("bootstrap_ci_cluster") or {}
                lo  = ci.get("lo", float("nan"))
                hi  = ci.get("hi", float("nan"))
                clo = cic.get("lo", float("nan"))
                chi = cic.get("hi", float("nan"))
                fp  = tr.get("frac_positive", float("nan"))
                wil = tr.get("wilcoxon") or {}
                p   = wil.get("p_two_sided", float("nan"))
                label = QUAD_LABEL[q]
                tok_ci  = f"[{lo:+.3f},{hi:+.3f}]"
                clus_ci = f"[{clo:+.3f},{chi:+.3f}]"
                lines.append(f"  q{q}={label:<28} {n:>5d}  "
                             f"{mr:>+13.4f}  "
                             f"{tok_ci:>15}  "
                             f"{clus_ci:>15}  "
                             f"{fp:>+6.3f}  {p:>10.2e}  {ms:>+13.4f}")
            lines.append("")
            lines.append("  CI_tok = token-level bootstrap; CI_cluster = sample-clustered.")
            lines.append("")

            # Table 2: raw mean drops per strategy per quad (NEW)
            lines.append(f"  Raw mean logp_drop by strategy:")
            lines.append(f"  {'quad':<32} {'n':>5}  "
                         f"{'top_tam':>8} {'random':>8} {'scram':>8} "
                         f"{'keep_top':>9} {'bot_tam':>8}")
            for q in (0, 1, 2, 3):
                block = report["per_ckpt_per_quad"][ckpt][str(q)]
                n   = block.get("n_tokens", 0)
                if n == 0:
                    continue
                raw = block.get("raw_strategy_drops") or {}
                def _g(s):
                    v = raw.get(s, {}).get("mean")
                    return v if v is not None else float("nan")
                top    = _g("top_tam_20pct")
                rand   = _g("random_avg")
                scram  = _g("scrambled_avg")
                keept  = _g("keep_top_tam_20pct")
                bot    = _g("bottom_tam_20pct")
                label  = QUAD_LABEL[q]
                lines.append(f"  q{q}={label:<28} {n:>5d}  "
                             f"{top:>+8.3f} {rand:>+8.3f} {scram:>+8.3f} "
                             f"{keept:>+9.3f} {bot:>+8.3f}")
            lines.append("")

            # Table 3: bottom_tam − top_tam paired Δ (smoking-gun for TAM-inverted-on-q)
            lines.append(f"  Paired Δ(bottom_tam − top_tam) — "
                         f"Δ > 0 means low-TAM region IS the evidence (TAM inverted):")
            lines.append(f"  {'quad':<32} {'n':>5}  "
                         f"{'mean_Δ':>8} {'CI_low':>8} {'CI_hi':>8} "
                         f"{'frac+':>6}  {'Wilcox p':>10}")
            for q in (0, 1, 2, 3):
                block = report["per_ckpt_per_quad"][ckpt][str(q)]
                bt = block.get("bottom_minus_top") or {}
                n = bt.get("n", 0)
                if n == 0:
                    continue
                m = bt.get("mean", float("nan"))
                ci = bt.get("bootstrap_ci") or {}
                lo = ci.get("lo", float("nan"))
                hi = ci.get("hi", float("nan"))
                fp = bt.get("frac_positive", float("nan"))
                wil = bt.get("wilcoxon") or {}
                p = wil.get("p_two_sided", float("nan"))
                label = QUAD_LABEL[q]
                lines.append(f"  q{q}={label:<28} {n:>5d}  "
                             f"{m:>+8.4f} {lo:>+8.4f} {hi:>+8.4f} "
                             f"{fp:>+6.3f}  {p:>10.2e}")
            lines.append("")

            # Per-category breakdown for quad==3 (visual_rejection)
            block3 = report["per_ckpt_per_quad"][ckpt]["3"]
            cats = block3.get("by_category") or {}
            if cats:
                lines.append(f"  quad==3 (visual_rejection_correction) by token_category:")
                for cat, c in sorted(cats.items(), key=lambda x: -x[1]["n"]):
                    lines.append(f"    {cat:<22} n={c['n']:>5d}  mean_Δ_random={c['mean_delta_random']:>+8.4f}")
                lines.append("")
        args.out_txt.parent.mkdir(parents=True, exist_ok=True)
        args.out_txt.write_text("\n".join(lines) + "\n")
        print(f">>> wrote summary: {args.out_txt}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
