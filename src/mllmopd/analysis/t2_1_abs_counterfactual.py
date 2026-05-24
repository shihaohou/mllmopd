"""T2-1Abs / Abs+RMS / Abs+clip offline counterfactuals.

After A0 confirmed signed-proxy mis-target (GPT round-4), the first
counterfactual (Abs alone) showed:
  ✓ frac_supp drops 0.442 → 0.124
  ✓ vis_reject mean_w 0.936 → 1.901
  ✓ corr flips -0.101 → +0.308
  ✗ rho_l2 explodes 0.973 → 11.98 (training would diverge)

So vanilla Abs fails the "rho_l2 < 1.5" pass criterion. GPT round-4
endorsed Abs+RMS-preserve / Abs+max-clip / quadrant-aware / hard
top-k as fallbacks. This module now runs FOUR counterfactuals
side-by-side on the same diagnostics:

  signed             — original T2-1 weights (baseline)
  abs                — vd → |vd|, otherwise unchanged
  abs_rms_preserve   — abs + per-sequence advantage-RMS scalar
                       (GPT round-3 formula: s = clip(sqrt(Σadv²/Σ(w·adv)²), 0.5, 2.0))
  abs_max_clip       — abs + clamp max weight to max_cap then renormalize

Each variant is evaluated against the same four pass criteria; the
verdict picks the variant that passes all four.

This module does NOT modify training and does NOT touch the live
weights. It re-uses t2_1_energy_audit's Accumulator + _process_sample
+ _summarize machinery so the numbers are computed by exactly the
same code paths.

The quadrant classification + visual-rejection-correction definition
use the ORIGINAL signed vd (semantics unchanged — "image rejects
current token" still means `vd<0`). Only the weights are recomputed
from `|vd|`. Adv is unchanged (= lp_full - old_lp_student, sentinel-
masked).

Usage::

    python -m mllmopd.analysis.t2_1_abs_counterfactual \\
        --diag-dir ${MLLMOPD_RUNS}/t2_1_a0_dump/diagnostics \\
        --out-json runs/audit/t2_1_abs_counterfactual_<ts>/result.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

from mllmopd.analysis.t2_1_energy_audit import (
    Accumulator,
    STEP_RE,
    SIDECAR_RE,
    _load_sidecar_for_step,
    _process_sample,
    _summarize,
)
from mllmopd.training.vd_weighting import (
    compute_vd_weights,
    compute_vd_weights_boost_only,
)


def _with_abs_weights(row: dict) -> dict:
    """Return a copy of the row with vd_weights recomputed from |vd|.

    Trick: compute_vd_weights(lp_full, lp_blank, R) internally computes
    vd = lp_full - lp_blank. To force the internal vd to equal |original_vd|
    while reusing the exact same downstream formulas (min-max normalize,
    threshold τ piecewise, mass-preserving renorm), pass synthetic
    lp_full' = |vd| and lp_blank' = 0. Then the internal diff equals
    |original_vd| element-wise. Downstream behavior is byte-identical
    to "modify vd_weighting.py to use |vd|".

    The ORIGINAL signed vd, lp_full, lp_blank, old_lp_student fields are
    preserved unchanged — only `vd_weights` is replaced. This is
    important: quadrant classification + frac_supp_neg_vd_neg_adv must
    use the ORIGINAL signed vd to keep the semantics ("image rejects
    current token" still means vd<0).
    """
    vd = row.get("vd") or []
    R = int(row.get("response_length", 0))
    if not vd or R <= 1 or len(vd) != R:
        return row  # let _process_sample handle the no_adv path

    abs_vd = [abs(v) for v in vd]
    w_abs = compute_vd_weights(abs_vd, [0.0] * R, R)
    out = dict(row)
    out["vd_weights"] = w_abs.tolist()
    return out


def _ensure_old_lp(row: dict, sidecar_lookup: dict[int, list[float]] | None) -> dict:
    """Inject old_log_probs from sidecar into row if row['old_lp_student']
    is missing/length-mismatched. Used so weight transforms have access to
    adv = lp_full - old_lp_student without re-implementing the join logic.

    Returns the row unmodified if no sidecar is available or sidecar has
    no matching sample_index — _process_sample will then route to the
    n_samples_no_old_lp bucket.
    """
    R = int(row.get("response_length", 0))
    olp = row.get("old_lp_student") or []
    if olp and len(olp) == R:
        return row
    if not sidecar_lookup:
        return row
    sample_index = row.get("sample_index")
    if sample_index is None:
        return row
    alt = sidecar_lookup.get(sample_index)
    if not alt or len(alt) != R:
        return row
    out = dict(row)
    out["old_lp_student"] = alt
    return out


def _with_abs_rms_preserve_weights(
    row: dict, clip: tuple[float, float] = (0.5, 2.0),
) -> dict:
    """Abs weights × per-sequence advantage-RMS-preserving scalar.

    Formula (GPT round-3):
        w_abs = compute_vd_weights(|vd|, 0, R)
        rho_seq = sqrt(Σ_t (w_abs[t] · adv[t])² / Σ_t adv[t]²)
        s = clip(1 / rho_seq, low, high)
        w_final = s · w_abs

    With s capped to [0.5, 2.0], per-sequence advantage L2 energy is
    preserved within ±sqrt(2)× of unweighted (the clip prevents
    pathological inflation on sequences where w_abs anti-correlates
    with adv², but that's rare with abs weights).

    Requires row to have old_lp_student populated (call _ensure_old_lp
    upstream). Falls back to plain abs weights if not.
    """
    row_abs = _with_abs_weights(row)
    R = int(row.get("response_length", 0))
    lp_full = row.get("lp_full") or []
    old_lp = row.get("old_lp_student") or []
    if (not lp_full or len(lp_full) != R
            or not old_lp or len(old_lp) != R or R <= 1):
        return row_abs

    adv = [lpf - lps for lpf, lps in zip(lp_full, old_lp)]
    w_abs = row_abs["vd_weights"]
    if len(w_abs) != R:
        return row_abs

    sum_adv2 = sum(a * a for a in adv)
    sum_wa2 = sum((w * a) ** 2 for w, a in zip(w_abs, adv))
    if sum_adv2 < 1e-12 or sum_wa2 < 1e-12:
        return row_abs

    rho_seq = math.sqrt(sum_wa2 / sum_adv2)
    if rho_seq < 1e-12:
        return row_abs
    s = 1.0 / rho_seq
    s = max(clip[0], min(clip[1], s))

    out = dict(row_abs)
    out["vd_weights"] = [w * s for w in w_abs]
    return out


def _with_abs_rms_preserve_wide_weights(row: dict) -> dict:
    """abs_rms_preserve with wider clip [0.1, 10.0].

    Counterfactual run showed that abs gives per-sequence rho_l2 up to
    ~12, but the default clip [0.5, 2.0] only allowed the scalar to
    shrink weights by 2× → rho_l2 only halved to 6. With clip widened
    to [0.1, 10.0], the scalar can fully compensate (1/sqrt(12) ≈ 0.29
    is within range).
    """
    return _with_abs_rms_preserve_weights(row, clip=(0.1, 10.0))


def _with_abs_max_clip_weights(row: dict, max_cap: float = 2.0,
                                renorm: bool = False) -> dict:
    """Abs weights then clip max to max_cap.

    When renorm=False (default): single-pass clip without re-normalize.
    Σw < N for sequences where any weight hit the cap. Strictly bounds
    rho_l2 ≤ max_cap. Lower mean_w because total mass dropped.

    When renorm=True: clip + mass-preserve renormalize. Pushes some
    weights back above the cap (since renorm scales ALL weights up to
    restore Σw=N). Keeps mean_w near 1.0 but doesn't strictly bound
    max weight. In practice max weight stays close to the cap with
    small overflow — good enough for stability if cap chosen
    conservatively.
    """
    row_abs = _with_abs_weights(row)
    w = row_abs["vd_weights"]
    if not w:
        return row_abs
    w_clip = [min(x, max_cap) for x in w]
    if renorm:
        R = len(w_clip)
        s = sum(w_clip)
        if s > 1e-12:
            w_clip = [x * R / s for x in w_clip]
    out = dict(row_abs)
    out["vd_weights"] = w_clip
    return out


def _with_abs_max_clip_renorm_weights(row: dict) -> dict:
    """abs_max_clip with post-clip mass-preserve renormalization."""
    return _with_abs_max_clip_weights(row, max_cap=2.0, renorm=True)


def _with_boost_only_weights(row: dict, alpha: float = 1.0,
                              max_w: float = 2.0) -> dict:
    """T2-2 boost-only |vd| with per-sequence percentile rank.

    Uses the SAME compute_vd_weights_boost_only that the training-time
    hook will use, so this counterfactual is a true preview of T2-2
    behavior on the existing T2-1 pilot diagnostics.
    """
    vd = row.get("vd") or []
    R = int(row.get("response_length", 0))
    if not vd or R <= 1 or len(vd) != R:
        return row

    abs_vd = [abs(v) for v in vd]
    # Trick (same as _with_abs_weights): pass lp_full'=|vd|, lp_blank'=0
    # so the internal abs_vd computation matches.
    w = compute_vd_weights_boost_only(
        abs_vd, [0.0] * R, R, alpha=alpha, max_w=max_w,
    )
    out = dict(row)
    out["vd_weights"] = w.tolist()
    return out


# Ordered list of (name, transform_fn). Transform takes (row_with_old_lp)
# and returns a row with vd_weights replaced. signed is the no-op baseline.
# Order matters for the recommended-variant selection: first all-pass
# wins. Boost-only (T2-2) listed first as it's GPT round-5's recommended
# design.
VARIANTS: list[tuple[str, callable]] = [
    ("boost_only", _with_boost_only_weights),  # T2-2 candidate
    ("signed", lambda row: row),
    ("abs", _with_abs_weights),
    ("abs_rms_preserve", _with_abs_rms_preserve_weights),
    ("abs_rms_preserve_wide", _with_abs_rms_preserve_wide_weights),
    ("abs_max_clip", _with_abs_max_clip_weights),
    ("abs_max_clip_renorm", _with_abs_max_clip_renorm_weights),
]


PASS_CRITERIA = {
    "frac_supp_lt_0.05": lambda p: (not math.isnan(p["frac_supp_neg_vd_neg_adv_mass"])
                                     and p["frac_supp_neg_vd_neg_adv_mass"] < 0.05),
    "vis_reject_mean_w_ge_1": lambda p: (
        p["quadrants"]["vis_reject_correction"]["n_tokens"] > 0
        and not math.isnan(p["quadrants"]["vis_reject_correction"]["mean_weight"])
        and p["quadrants"]["vis_reject_correction"]["mean_weight"] >= 1.0
    ),
    "corr_ge_-0.05": lambda p: (not math.isnan(p["corr_w_abs_adv_pooled"])
                                 and p["corr_w_abs_adv_pooled"] >= -0.05),
    "rho_l2_lt_1.5": lambda p: (not math.isnan(p["rho_l2_pooled"])
                                 and p["rho_l2_pooled"] < 1.5),
}


def scan_variants(diag_dir: Path, steps: list[int] | None, limit_rows: int) -> dict:
    """Run the audit once per variant in VARIANTS over the same diag data.
    Returns side-by-side result dict with per-variant summary + verdict.
    """
    files = sorted(diag_dir.glob("step_*.jsonl.gz"))
    files = [f for f in files if not SIDECAR_RE.search(f.name)]
    if not files:
        raise FileNotFoundError(f"no step_*.jsonl.gz in {diag_dir}")

    pooled: dict[str, Accumulator] = {name: Accumulator() for name, _ in VARIANTS}
    per_step: dict[str, dict[int, Accumulator]] = {name: {} for name, _ in VARIANTS}

    for f in files:
        m = STEP_RE.search(f.name)
        if not m:
            continue
        s = int(m.group(1))
        if steps and s not in steps:
            continue

        for name, _ in VARIANTS:
            per_step[name][s] = Accumulator()

        sidecar_lookup, _sidecar_stats = _load_sidecar_for_step(diag_dir, s)

        with gzip.open(f, "rt") as fin:
            n = 0
            for line in fin:
                if limit_rows and n >= limit_rows:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Inject old_log_probs from sidecar into the row up front so
                # weight transforms (which need adv = lp_full - old_lp) can
                # compute without re-implementing the join.
                row_with_olp = _ensure_old_lp(row, sidecar_lookup)
                for name, transform in VARIANTS:
                    transformed = transform(row_with_olp)
                    _process_sample(
                        transformed,
                        per_step[name][s],
                        pooled[name],
                        sidecar_lookup or None,
                    )
                n += 1

    out = {
        "diag_dir": str(diag_dir),
        "n_step_files_scanned": len(per_step["signed"]),
        "steps_scanned": sorted(per_step["signed"].keys()),
        "variants": {},
    }
    for name, _ in VARIANTS:
        out["variants"][name] = {
            "pooled": _summarize(pooled[name]),
            "per_step": {str(s): _summarize(per_step[name][s])
                         for s in sorted(per_step[name].keys())},
        }

    # Verdict per variant.
    verdicts: dict[str, dict] = {}
    for name in out["variants"]:
        p = out["variants"][name]["pooled"]
        passes = [k for k, check in PASS_CRITERIA.items() if check(p)]
        fails = [k for k in PASS_CRITERIA if k not in passes]
        verdicts[name] = {
            "passed": passes,
            "failed": fails,
            "all_pass": len(fails) == 0,
        }
    out["verdicts"] = verdicts

    # Headline.
    interp = []
    interp.append("== variant summary ==")
    interp.append(f"  {'variant':<20} {'rho_l2':>7} {'corr':>7} {'frac_supp':>10} "
                  f"{'cond_supp':>10} {'visR_mw':>8}  passes")
    for name in out["variants"]:
        p = out["variants"][name]["pooled"]
        rho = p["rho_l2_pooled"]
        cor = p["corr_w_abs_adv_pooled"]
        fs = p["frac_supp_neg_vd_neg_adv_mass"]
        cs = p["conditional_supp_visual_rejection"]
        mw = p["quadrants"]["vis_reject_correction"]["mean_weight"]
        passes_str = "/".join(
            "✓" if c in verdicts[name]["passed"] else "✗"
            for c in PASS_CRITERIA
        )
        interp.append(
            f"  {name:<20} {rho:>7.3f} {cor:>+7.3f} {fs:>10.3f} "
            f"{cs:>10.3f} {mw:>8.3f}  {passes_str}"
        )
    interp.append(f"  pass-order: {list(PASS_CRITERIA.keys())}")

    winners = [n for n in out["variants"] if verdicts[n]["all_pass"]]
    losers = [n for n in out["variants"] if not verdicts[n]["all_pass"]]
    if winners:
        # First winner in VARIANTS order = recommended.
        recommended = next(n for n, _ in VARIANTS if n in winners)
        interp.append(f"VERDICT: {len(winners)} variant(s) pass all 4 criteria: "
                      f"{winners}. RECOMMENDED: {recommended}")
    else:
        interp.append(f"VERDICT: no variant passes all 4 criteria. "
                      f"Failing: {[(n, verdicts[n]['failed']) for n in losers]}")
    out["headline_interpretation"] = interp

    return out


# Back-compat alias for the older smoke test.
def scan_both(diag_dir: Path, steps: list[int] | None, limit_rows: int) -> dict:
    """Legacy two-variant entry point (signed + abs only). Kept for the
    old smoke tests; new callers should use scan_variants."""
    out = scan_variants(diag_dir, steps, limit_rows)
    # Re-shape into the old {signed:..., abs:..., diff:...} schema.
    signed = out["variants"]["signed"]
    abs_ = out["variants"]["abs"]
    diff = {
        "delta_rho_l2": abs_["pooled"]["rho_l2_pooled"] - signed["pooled"]["rho_l2_pooled"],
        "delta_corr_w_abs_adv": (
            abs_["pooled"]["corr_w_abs_adv_pooled"]
            - signed["pooled"]["corr_w_abs_adv_pooled"]
        ),
        "delta_frac_supp_neg_vd_neg_adv_mass": (
            abs_["pooled"]["frac_supp_neg_vd_neg_adv_mass"]
            - signed["pooled"]["frac_supp_neg_vd_neg_adv_mass"]
        ),
        "delta_conditional_supp_visual_rejection": (
            abs_["pooled"]["conditional_supp_visual_rejection"]
            - signed["pooled"]["conditional_supp_visual_rejection"]
        ),
        "vis_reject_correction_mean_weight": {
            "signed": signed["pooled"]["quadrants"]["vis_reject_correction"]["mean_weight"],
            "abs": abs_["pooled"]["quadrants"]["vis_reject_correction"]["mean_weight"],
        },
        "vis_reject_correction_frac_w_below_1": {
            "signed": signed["pooled"]["quadrants"]["vis_reject_correction"]["frac_w_below_1"],
            "abs": abs_["pooled"]["quadrants"]["vis_reject_correction"]["frac_w_below_1"],
        },
    }
    return {
        "diag_dir": out["diag_dir"],
        "n_step_files_scanned": out["n_step_files_scanned"],
        "steps_scanned": out["steps_scanned"],
        "signed": signed,
        "abs": abs_,
        "diff": diff,
        "headline_interpretation": out["headline_interpretation"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--diag-dir", required=True,
                    help="<run>/diagnostics dir containing step_*.jsonl.gz "
                         "(needs sidecar .adv_dp*.jsonl.gz too for adv recovery)")
    ap.add_argument("--steps", default=None,
                    help="comma list of step numbers to include (default: all)")
    ap.add_argument("--limit-rows-per-step", type=int, default=0,
                    help="0 = all rows; >0 = subsample first N for speed")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    steps = None
    if args.steps:
        steps = [int(s) for s in args.steps.split(",") if s.strip()]

    out = scan_variants(Path(args.diag_dir), steps, args.limit_rows_per_step)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out_json).open("w") as f:
            json.dump(out, f, indent=2, allow_nan=True)
        print(f"json -> {args.out_json}", file=sys.stderr)
    else:
        # Strip per-step blocks for stdout brevity; keep pooled per variant.
        compact = {
            "diag_dir": out["diag_dir"],
            "n_step_files_scanned": out["n_step_files_scanned"],
            "steps_scanned": out["steps_scanned"],
            "variants_pooled": {k: v["pooled"] for k, v in out["variants"].items()},
            "verdicts": out["verdicts"],
            "headline_interpretation": out["headline_interpretation"],
        }
        print(json.dumps(compact, indent=2, default=str))

    signed_p = out["variants"]["signed"]["pooled"]
    print(f"\n=== T2-1 COUNTERFACTUAL VARIANTS "
          f"({signed_p['n_samples_used']} samples, "
          f"{signed_p['n_tokens_used']} tokens) ===",
          file=sys.stderr)
    for line in out["headline_interpretation"]:
        print(line, file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
