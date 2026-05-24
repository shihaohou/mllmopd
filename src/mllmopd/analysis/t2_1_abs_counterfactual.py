"""T2-1Abs offline counterfactual: would `vd → |vd|` fix the mis-target?

GPT round-4 ask (before committing 3-4h on T2-1Abs 230-step training):
on the existing T2-1 pilot diagnostics (10 steps × 8 dp ranks =
576 samples / 935814 tokens), substitute `vd → |vd|` in the PGPO
Eq 6+7 formulas and recompute weights, then re-evaluate the three
A0 metrics + quadrant table. If on this counterfactual:

  (a) `frac_supp_neg_vd_neg_adv ≈ 0`        — visual-rejection no longer suppressed
  (b) `mean_weight` on the `vis_reject_correction` quadrant ≥ 1 — actually boosted
  (c) `corr(w, |adv|) ≥ 0` (or at least less negative)
  (d) `rho_l2` not catastrophically high (e.g. < 1.5)

then T2-1Abs is justified as the next experiment. If they don't all
hold, the next step needs more design work (Abs+RMS, quadrant-aware,
hard top-k by |vd|, ...).

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
from mllmopd.training.vd_weighting import compute_vd_weights


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


def scan_both(diag_dir: Path, steps: list[int] | None, limit_rows: int) -> dict:
    """Run the audit twice over the same data: once signed (original
    weights from row), once abs (weights recomputed from |vd|).
    Returns side-by-side result dict.
    """
    files = sorted(diag_dir.glob("step_*.jsonl.gz"))
    files = [f for f in files if not SIDECAR_RE.search(f.name)]
    if not files:
        raise FileNotFoundError(f"no step_*.jsonl.gz in {diag_dir}")

    pooled_signed = Accumulator()
    pooled_abs = Accumulator()
    per_step_signed: dict[int, Accumulator] = {}
    per_step_abs: dict[int, Accumulator] = {}

    for f in files:
        m = STEP_RE.search(f.name)
        if not m:
            continue
        s = int(m.group(1))
        if steps and s not in steps:
            continue

        acc_signed = Accumulator()
        acc_abs = Accumulator()
        per_step_signed[s] = acc_signed
        per_step_abs[s] = acc_abs

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
                # SIGNED path: use weights from row as-is.
                _process_sample(row, acc_signed, pooled_signed, sidecar_lookup or None)
                # ABS path: same row but with weights recomputed from |vd|.
                # _process_sample re-reads vd / vd_weights / lp_full / old_lp_student,
                # but adv computation depends only on lp_full + old_lp_student (unchanged).
                row_abs = _with_abs_weights(row)
                _process_sample(row_abs, acc_abs, pooled_abs, sidecar_lookup or None)
                n += 1

    out = {
        "diag_dir": str(diag_dir),
        "n_step_files_scanned": len(per_step_signed),
        "steps_scanned": sorted(per_step_signed.keys()),
        "signed": {
            "pooled": _summarize(pooled_signed),
            "per_step": {str(s): _summarize(per_step_signed[s])
                         for s in sorted(per_step_signed.keys())},
        },
        "abs": {
            "pooled": _summarize(pooled_abs),
            "per_step": {str(s): _summarize(per_step_abs[s])
                         for s in sorted(per_step_abs.keys())},
        },
    }

    ps = out["signed"]["pooled"]
    pa = out["abs"]["pooled"]
    out["diff"] = {
        "delta_rho_l2": pa["rho_l2_pooled"] - ps["rho_l2_pooled"],
        "delta_corr_w_abs_adv": (
            pa["corr_w_abs_adv_pooled"] - ps["corr_w_abs_adv_pooled"]
        ),
        "delta_frac_supp_neg_vd_neg_adv_mass": (
            pa["frac_supp_neg_vd_neg_adv_mass"] - ps["frac_supp_neg_vd_neg_adv_mass"]
        ),
        "delta_conditional_supp_visual_rejection": (
            pa["conditional_supp_visual_rejection"]
            - ps["conditional_supp_visual_rejection"]
        ),
        "vis_reject_correction_mean_weight": {
            "signed": ps["quadrants"]["vis_reject_correction"]["mean_weight"],
            "abs": pa["quadrants"]["vis_reject_correction"]["mean_weight"],
        },
        "vis_reject_correction_frac_w_below_1": {
            "signed": ps["quadrants"]["vis_reject_correction"]["frac_w_below_1"],
            "abs": pa["quadrants"]["vis_reject_correction"]["frac_w_below_1"],
        },
    }

    # Headline interpretation: does Abs counterfactual fix the mis-target?
    interp = []
    p = out["diff"]
    fs_signed = ps["frac_supp_neg_vd_neg_adv_mass"]
    fs_abs = pa["frac_supp_neg_vd_neg_adv_mass"]
    if not math.isnan(fs_signed) and not math.isnan(fs_abs):
        interp.append(
            f"frac_supp_neg_vd_neg_adv_mass: signed={fs_signed:.3f} → abs={fs_abs:.3f} "
            f"(Δ={p['delta_frac_supp_neg_vd_neg_adv_mass']:+.3f})"
        )
    cs_signed = ps["conditional_supp_visual_rejection"]
    cs_abs = pa["conditional_supp_visual_rejection"]
    if not math.isnan(cs_signed) and not math.isnan(cs_abs):
        interp.append(
            f"conditional_supp_visual_rejection: signed={cs_signed:.3f} → abs={cs_abs:.3f} "
            f"(Δ={p['delta_conditional_supp_visual_rejection']:+.3f})"
        )
    mw_signed = p["vis_reject_correction_mean_weight"]["signed"]
    mw_abs = p["vis_reject_correction_mean_weight"]["abs"]
    if not math.isnan(mw_signed) and not math.isnan(mw_abs):
        interp.append(
            f"vis_reject_correction mean_weight: signed={mw_signed:.3f} → abs={mw_abs:.3f}"
        )
    interp.append(
        f"rho_l2: signed={ps['rho_l2_pooled']:.3f} → abs={pa['rho_l2_pooled']:.3f} "
        f"(Δ={p['delta_rho_l2']:+.3f})"
    )
    interp.append(
        f"corr(w,|adv|): signed={ps['corr_w_abs_adv_pooled']:+.3f} → "
        f"abs={pa['corr_w_abs_adv_pooled']:+.3f} (Δ={p['delta_corr_w_abs_adv']:+.3f})"
    )
    # Verdict line
    verdict_passes = []
    if not math.isnan(fs_abs) and fs_abs < 0.05:
        verdict_passes.append("frac_supp_abs<0.05")
    if not math.isnan(mw_abs) and mw_abs >= 1.0:
        verdict_passes.append("vis_reject mean_w_abs≥1")
    if not math.isnan(pa["corr_w_abs_adv_pooled"]) and pa["corr_w_abs_adv_pooled"] >= -0.05:
        verdict_passes.append("corr_abs≥-0.05")
    if not math.isnan(pa["rho_l2_pooled"]) and pa["rho_l2_pooled"] < 1.5:
        verdict_passes.append("rho_l2_abs<1.5")
    if len(verdict_passes) == 4:
        interp.append(f"VERDICT: T2-1Abs likely justified — all 4 conditions hold "
                      f"({', '.join(verdict_passes)})")
    else:
        failed = {"frac_supp_abs<0.05", "vis_reject mean_w_abs≥1",
                  "corr_abs≥-0.05", "rho_l2_abs<1.5"} - set(verdict_passes)
        interp.append(f"VERDICT: not all conditions hold — passed [{', '.join(verdict_passes)}], "
                      f"failed [{', '.join(failed)}]; revisit design")
    out["headline_interpretation"] = interp

    return out


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

    out = scan_both(Path(args.diag_dir), steps, args.limit_rows_per_step)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out_json).open("w") as f:
            json.dump(out, f, indent=2, allow_nan=True)
        print(f"json -> {args.out_json}", file=sys.stderr)
    else:
        # Strip per_step for stdout brevity.
        compact = {k: v for k, v in out.items() if k != "signed" and k != "abs"}
        compact["signed_pooled"] = out["signed"]["pooled"]
        compact["abs_pooled"] = out["abs"]["pooled"]
        print(json.dumps(compact, indent=2, default=str))

    print(f"\n=== T2-1Abs COUNTERFACTUAL "
          f"({out['signed']['pooled']['n_samples_used']} samples, "
          f"{out['signed']['pooled']['n_tokens_used']} tokens) ===",
          file=sys.stderr)
    for line in out["headline_interpretation"]:
        print(f"  {line}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
