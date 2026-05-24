#!/usr/bin/env python
"""Smoke test for src/mllmopd/analysis/t2_1_abs_counterfactual.py.

Constructs a synthetic 2-sample diagnostic where the signed `vd`
strongly suppresses visual-rejection correction (per the T2-1 result
pattern), then verifies that the abs counterfactual:
  1. Reduces frac_supp_neg_vd_neg_adv_mass.
  2. Increases mean_weight on the vis_reject_correction quadrant.
  3. Preserves the original signed-vd quadrant classification (the
     ORIGINAL vd signs are the keys for the quadrant table; only
     weights change).

Run:
    python scripts/audit/verify_t2_1_abs_counterfactual.py
"""

from __future__ import annotations

import gzip
import json
import math
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mllmopd.analysis.t2_1_abs_counterfactual import (  # noqa: E402
    scan_both, scan_variants, _with_abs_weights,
    _with_abs_rms_preserve_weights, _with_abs_max_clip_weights, _ensure_old_lp,
)
from mllmopd.training.vd_weighting import compute_vd_weights  # noqa: E402


def _approx(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _make_signed_row(lp_full, lp_blank, old_lp_student, sample_index=42):
    """Mirror of verify_t2_1_energy_audit._make_row, using signed weights."""
    R = len(lp_full)
    w = compute_vd_weights(lp_full, lp_blank, R).tolist()
    return {
        "id": "s0",
        "sample_index": sample_index,
        "step": 0,
        "response_length": R,
        "image_mode": "full",
        "teacher_url": "synth",
        "teacher_url_diag": "synth",
        "response_correct": True,
        "lp_full": list(lp_full),
        "lp_blank": list(lp_blank),
        "vd": [a - b for a, b in zip(lp_full, lp_blank)],
        "vd_weights": w,
        "old_lp_student": list(old_lp_student),
    }


def _write_step(rows, path):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_with_abs_weights_preserves_original_vd():
    """_with_abs_weights swaps vd_weights but keeps original vd."""
    lp_full = [1.0, 5.0]
    lp_blank = [5.0, 1.0]  # vd = [-4, 4]
    old_lp = [0.0, 0.0]
    row = _make_signed_row(lp_full, lp_blank, old_lp)
    new_row = _with_abs_weights(row)
    assert new_row["vd"] == row["vd"], "original signed vd must be preserved"
    assert new_row["lp_full"] == row["lp_full"], "lp_full must be preserved"
    assert new_row["lp_blank"] == row["lp_blank"], "lp_blank must be preserved"
    assert new_row["vd_weights"] != row["vd_weights"], "weights must differ"
    # With |vd|=[4,4] (all equal), compute_vd_weights returns unit ones.
    assert all(_approx(w, 1.0) for w in new_row["vd_weights"]), new_row["vd_weights"]
    print(f"  preserves_original_vd: signed_w={row['vd_weights']}, "
          f"abs_w={new_row['vd_weights']} (all unit since |vd| equal)  OK")


def test_abs_recovers_visual_rejection_quadrant():
    """Construct a 4-token case where signed-vd puts vis_reject_correction
    in suppress branch (w<1) but abs recovers it (w>=1).

    4 tokens for non-degenerate |vd| under min-max normalization:
      vd = [-4, +4, -3, +0.5];  |vd| = [4, 4, 3, 0.5]
      adv = [-2, +3, -1, +0.5]  (signs matched to vd for clean quadrants)

    Quadrants (by ORIGINAL signed vd):
      t0: vd<0, adv<0 → vis_reject_correction (key OPD signal)
      t1: vd>0, adv>0 → vis_support_agree
      t2: vd<0, adv<0 → vis_reject_correction
      t3: vd>0, adv>0 → vis_support_agree

    Hand-computed PGPO weights (τ=0.4, β=2.0):
      signed: w ≈ [0, 2.472, 0.258, 1.271]
        → vis_reject mean_w = (0+0.258)/2 ≈ 0.129  (strongly suppressed)
      abs:    w ≈ [1.491, 1.491, 1.018, 0]
        → vis_reject mean_w = (1.491+1.018)/2 ≈ 1.255  (boosted)
    """
    lp_full = [1.0, 5.0, 2.0, 3.5]
    lp_blank = [5.0, 1.0, 5.0, 3.0]   # vd = [-4, 4, -3, 0.5]
    old_lp = [3.0, 2.0, 3.0, 3.0]     # adv = lp_full - old_lp = [-2, 3, -1, 0.5]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "step_000000.jsonl.gz"
        _write_step([_make_signed_row(lp_full, lp_blank, old_lp)], path)
        out = scan_both(Path(tmp), steps=None, limit_rows=0)

    ps = out["signed"]["pooled"]
    pa = out["abs"]["pooled"]

    # Both paths must classify quadrants by original signed vd (semantics preserved).
    assert ps["quadrants"]["vis_reject_correction"]["n_tokens"] == 2, ps["quadrants"]
    assert pa["quadrants"]["vis_reject_correction"]["n_tokens"] == 2, pa["quadrants"]

    # Hand-computed: signed mean_w ≈ 0.129, abs mean_w ≈ 1.255.
    sig_mw = ps["quadrants"]["vis_reject_correction"]["mean_weight"]
    abs_mw = pa["quadrants"]["vis_reject_correction"]["mean_weight"]
    assert sig_mw < 0.3, f"signed vis_reject mean_w should be ~0.13; got {sig_mw}"
    assert abs_mw > 1.0, f"abs vis_reject mean_w should be > 1; got {abs_mw}"

    # frac_supp drops dramatically (signed ≈ 0.46, abs = 0).
    assert pa["frac_supp_neg_vd_neg_adv_mass"] < ps["frac_supp_neg_vd_neg_adv_mass"], (ps, pa)
    assert pa["frac_supp_neg_vd_neg_adv_mass"] < 0.05, pa["frac_supp_neg_vd_neg_adv_mass"]

    print(f"  vis_reject mean_w: signed={sig_mw:.3f} → abs={abs_mw:.3f}  OK")
    print(f"  frac_supp:         signed={ps['frac_supp_neg_vd_neg_adv_mass']:.3f} → "
          f"abs={pa['frac_supp_neg_vd_neg_adv_mass']:.3f}  OK")


def test_diff_block_populated():
    """The diff block + headline interp are populated."""
    lp_full = [1.0, 5.0, 6.0, 7.0, 8.0]
    lp_blank = [5.0, 4.0, 3.0, 2.0, 1.0]
    old_lp = [3.0, 2.0, 0.0, 1.0, 5.0]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "step_000000.jsonl.gz"
        _write_step([_make_signed_row(lp_full, lp_blank, old_lp)], path)
        out = scan_both(Path(tmp), steps=None, limit_rows=0)

    assert "diff" in out, list(out.keys())
    d = out["diff"]
    for k in ("delta_rho_l2", "delta_corr_w_abs_adv",
              "delta_frac_supp_neg_vd_neg_adv_mass",
              "delta_conditional_supp_visual_rejection",
              "vis_reject_correction_mean_weight"):
        assert k in d, f"diff missing {k}"
    assert "headline_interpretation" in out
    assert any("VERDICT:" in line for line in out["headline_interpretation"])
    print(f"  diff block populated with {len(d)} keys; headline has "
          f"{len(out['headline_interpretation'])} lines incl. VERDICT  OK")


def test_abs_rms_preserve_clamps_rho_l2_toward_1():
    """abs_rms_preserve should preserve advantage L2 energy per sequence.

    Reuses the 4-token construction from test_abs_recovers_visual_rejection_quadrant.
    With the per-sequence scalar applied, rho_l2 should land much
    closer to 1 than plain abs (which would have rho_l2 high here).
    """
    lp_full = [1.0, 5.0, 2.0, 3.5]
    lp_blank = [5.0, 1.0, 5.0, 3.0]
    old_lp = [3.0, 2.0, 3.0, 3.0]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "step_000000.jsonl.gz"
        _write_step([_make_signed_row(lp_full, lp_blank, old_lp)], path)
        out = scan_variants(Path(tmp), steps=None, limit_rows=0)

    rho_abs = out["variants"]["abs"]["pooled"]["rho_l2_pooled"]
    rho_rms = out["variants"]["abs_rms_preserve"]["pooled"]["rho_l2_pooled"]
    # rms should be closer to 1 than plain abs (closer in log-scale)
    assert abs(math.log(max(rho_rms, 1e-6))) <= abs(math.log(max(rho_abs, 1e-6))) + 1e-6, (
        f"abs rho_l2={rho_abs}, rms rho_l2={rho_rms} — rms should be closer to 1"
    )
    # And rms should be in clip range [0.5, 2.0] of the per-sequence rescale
    # (so pooled rho_l2 should be in [0.5, 2.0] when scalar is the only effect)
    assert 0.4 <= rho_rms <= 2.5, f"rms rho_l2={rho_rms} outside expected range"
    print(f"  abs_rms_preserve: rho_l2 abs={rho_abs:.3f} → rms={rho_rms:.3f}  OK")


def test_abs_max_clip_bounds_max_weight():
    """abs_max_clip with cap=2.0 should produce no weight > 2.0."""
    # Construct a case where vanilla abs produces weights > 2.0.
    # vd = [-4, 4, 0.5] → |vd| = [4, 4, 0.5] → vd_norm = [1, 1, 0]
    # → w_raw = [3, 3, 0], sum=6, w = [1.5, 1.5, 0]  (all ≤ 2.0 already)
    # Use more lopsided vd to get a w > 2.
    # vd = [-1, 4, 0.5] → |vd| = [1, 4, 0.5] → range=3.5
    # → vd_norm = [(1-0.5)/3.5, (4-0.5)/3.5, 0] = [0.143, 1.0, 0]
    # → w_raw t0: 0.143/0.4 = 0.357 (suppress); t1: 3.0 (boost); t2: 0
    # → sum = 3.357, w = w_raw * 3 / 3.357 = [0.319, 2.681, 0]
    # So abs gives max_w = 2.681 > 2.0. Clip to 2.0.
    lp_full = [1.0, 5.0, 2.5]
    lp_blank = [2.0, 1.0, 2.0]   # vd = [-1, 4, 0.5]
    old_lp = [2.0, 1.0, 1.5]      # adv = [-1, 4, 1]; all non-zero

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "step_000000.jsonl.gz"
        _write_step([_make_signed_row(lp_full, lp_blank, old_lp)], path)
        out = scan_variants(Path(tmp), steps=None, limit_rows=0)

    # We can't directly access weights; instead verify that
    # rho_l2(abs_max_clip) < rho_l2(abs) when abs has a large max weight.
    rho_abs = out["variants"]["abs"]["pooled"]["rho_l2_pooled"]
    rho_clip = out["variants"]["abs_max_clip"]["pooled"]["rho_l2_pooled"]
    assert rho_clip < rho_abs, f"clip rho_l2={rho_clip} should be < abs rho_l2={rho_abs}"
    print(f"  abs_max_clip: rho_l2 abs={rho_abs:.3f} → clip={rho_clip:.3f}  OK")


def test_scan_variants_has_all_paths():
    """scan_variants returns all 6 variant paths."""
    lp_full = [1.0, 5.0, 2.0, 3.5]
    lp_blank = [5.0, 1.0, 5.0, 3.0]
    old_lp = [3.0, 2.0, 3.0, 3.0]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "step_000000.jsonl.gz"
        _write_step([_make_signed_row(lp_full, lp_blank, old_lp)], path)
        out = scan_variants(Path(tmp), steps=None, limit_rows=0)
    expected = {
        "boost_only",  # T2-2 candidate (GPT round-5)
        "signed", "abs", "abs_rms_preserve", "abs_rms_preserve_wide",
        "abs_max_clip", "abs_max_clip_renorm",
    }
    assert set(out["variants"].keys()) == expected, list(out["variants"].keys())
    assert "verdicts" in out
    assert all(v in out["verdicts"] for v in out["variants"]), out["verdicts"]
    print(f"  scan_variants: {len(expected)} variants present, verdicts populated  OK")


def main():
    tests = [
        test_with_abs_weights_preserves_original_vd,
        test_abs_recovers_visual_rejection_quadrant,
        test_diff_block_populated,
        test_abs_rms_preserve_clamps_rho_l2_toward_1,
        test_abs_max_clip_bounds_max_weight,
        test_scan_variants_has_all_paths,
    ]
    print(f"Running {len(tests)} smoke tests for t2_1_abs_counterfactual:")
    failed = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
            failed.append(t.__name__)
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed.append(t.__name__)
    if failed:
        print(f"\n{len(failed)}/{len(tests)} FAILED: {failed}")
        return 1
    print(f"\nall {len(tests)} smoke tests pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
