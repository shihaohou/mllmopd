#!/usr/bin/env python
"""Smoke test for src/mllmopd/analysis/t2_1_energy_audit.py.

Synthesizes diagnostics .jsonl.gz rows with known lp_full, lp_blank,
old_lp_student, vd_weights (computed via the real compute_vd_weights),
runs the audit module, and verifies pooled rho_l2 / corr(w,|adv|) /
frac_supp against hand-computed reference values for the canonical
5-token case in the module docstring.

Run:
    python scripts/audit/verify_t2_1_energy_audit.py
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mllmopd.training.vd_weighting import compute_vd_weights  # noqa: E402
from mllmopd.analysis.t2_1_energy_audit import scan, VD_NORM_BINS  # noqa: E402


def _approx(a: float, b: float, tol: float = 1e-3) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _make_row(lp_full, lp_blank, old_lp_student, step=0, sid="s0", sample_index=None):
    R = len(lp_full)
    w = compute_vd_weights(lp_full, lp_blank, R).tolist()
    return {
        "id": sid,
        "sample_index": sample_index,
        "step": step,
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


def _write_step_file(rows, path):
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_sidecar_file(rows, path, step, dp_rank=0):
    """rows: list of (sample_index, old_log_probs)."""
    with gzip.open(path, "wt") as f:
        for sample_index, old_lp in rows:
            row = {"sample_index": sample_index, "step": step,
                   "dp_rank": dp_rank, "old_log_probs": list(old_lp)}
            f.write(json.dumps(row) + "\n")


def test_canonical_5token():
    """Hand-computed reference for the canonical R=5 case in audit docstring.

    R=5, lp_full=[1,2,3,4,5], lp_blank=[3,2,1,0,-1], old_lp_student=[2,1,0,-1,-2]
    Then vd=[-2,0,2,4,6], vd_norm=[0,0.25,0.5,0.75,1.0], τ=0.4,β=2.0:
      w_raw = [0, 0.625, 1.333, 2.167, 3.0]; sum=7.125
      w = w_raw * 5/7.125 = [0, 0.4386, 0.9357, 1.5205, 2.1053]
    adv = lp_full - old_lp_student = [-1, 1, 3, 5, 7]; |adv|=[1,1,3,5,7]
    Expected:
      sum_adv2     = 1+1+9+25+49 = 85
      sum_w2_adv2  = sum(w²·adv²) ≈ 283.04
      rho_l2       = sqrt(283.04/85) ≈ 1.825
      corr(w,|adv|) ≈ +0.982  (both monotone-increasing in t)
      frac_supp_neg_vd_neg_adv = |adv[0]|/sum|adv| = 1/17 ≈ 0.0588
    """
    lp_full = [1.0, 2.0, 3.0, 4.0, 5.0]
    lp_blank = [3.0, 2.0, 1.0, 0.0, -1.0]
    old_lp = [2.0, 1.0, 0.0, -1.0, -2.0]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "step_000001.jsonl.gz"
        _write_step_file([_make_row(lp_full, lp_blank, old_lp)], path)
        out = scan(Path(tmp), steps=None, limit_rows=0)

    p = out["pooled"]
    assert p["n_samples_used"] == 1, p
    assert p["n_tokens_used"] == 5, p
    assert _approx(p["rho_l2_pooled"], 1.825), p["rho_l2_pooled"]
    assert _approx(p["corr_w_abs_adv_pooled"], 0.982), p["corr_w_abs_adv_pooled"]
    assert _approx(p["frac_supp_neg_vd_neg_adv_mass"], 1.0 / 17.0), p["frac_supp_neg_vd_neg_adv_mass"]
    print(f"  canonical-5token: rho_l2={p['rho_l2_pooled']:.4f}, "
          f"corr={p['corr_w_abs_adv_pooled']:+.4f}, "
          f"frac_supp={p['frac_supp_neg_vd_neg_adv_mass']:.4f}  OK")


def test_anti_correlated():
    """Pathological case: weights HIGH where |adv| is LOW.

    Construct so that vd is LARGE where adv is SMALL → boost branch
    fires on low-adv tokens → corr(w,|adv|) negative AND rho_l2 < 1.
    """
    # vd big at end → boost at end. Make adv small at end.
    lp_full = [1.0, 2.0, 3.0, 4.0, 5.0]
    lp_blank = [3.0, 2.0, 1.0, 0.0, -1.0]  # vd ascending
    old_lp = [-6.0, -3.0, 0.0, 3.0, 5.0]  # adv = lp_full - old_lp = [7, 5, 3, 1, 0]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "step_000001.jsonl.gz"
        _write_step_file([_make_row(lp_full, lp_blank, old_lp)], path)
        out = scan(Path(tmp), steps=None, limit_rows=0)

    p = out["pooled"]
    assert p["n_samples_used"] == 1, p
    # adv = [7, 5, 3, 1, 0]; |adv| ascending↔descending sanity:
    #   adv positive everywhere → frac_supp = 0
    #   weights ascending (big at end), |adv| descending → corr < 0
    #   rho_l2: high-weight tokens have small adv → energy drops sharply
    assert p["corr_w_abs_adv_pooled"] < -0.5, p["corr_w_abs_adv_pooled"]
    assert p["rho_l2_pooled"] < 0.8, p["rho_l2_pooled"]
    assert p["frac_supp_neg_vd_neg_adv_mass"] == 0.0, p["frac_supp_neg_vd_neg_adv_mass"]
    print(f"  anti-correlated: rho_l2={p['rho_l2_pooled']:.4f}, "
          f"corr={p['corr_w_abs_adv_pooled']:+.4f}  OK")


def test_visual_rejection_suppression():
    """Visual-rejection token suppressed: vd<0 ∧ adv<0 ∧ w<1.

    The most important test for the signed-proxy-failure hypothesis.
    """
    # vd[0] = -4 (most negative), adv[0] = -2 (visual-rejection signal).
    # Other tokens positive vd, positive adv. PGPO suppresses vd[0]
    # (vd_norm=0 → w=0) and the |adv| of t=0 contributes to frac_supp.
    lp_full = [1.0, 5.0, 6.0, 7.0, 8.0]
    lp_blank = [5.0, 4.0, 3.0, 2.0, 1.0]  # vd = [-4, 1, 3, 5, 7]; vd_norm[0]=0 → w[0]=0
    old_lp = [3.0, 2.0, 0.0, 1.0, 5.0]  # adv = [-2, 3, 6, 6, 3]; |adv|=[2,3,6,6,3]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "step_000001.jsonl.gz"
        _write_step_file([_make_row(lp_full, lp_blank, old_lp)], path)
        out = scan(Path(tmp), steps=None, limit_rows=0)

    p = out["pooled"]
    # frac = |adv[0]| / sum|adv| = 2 / (2+3+6+6+3) = 2/20 = 0.10
    assert _approx(p["frac_supp_neg_vd_neg_adv_mass"], 0.10), p["frac_supp_neg_vd_neg_adv_mass"]
    print(f"  visual-rejection-suppression: frac_supp={p['frac_supp_neg_vd_neg_adv_mass']:.4f}  OK")


def test_skips_unit_weights():
    """All-equal vd → w returned as all-ones → sample should be flagged unit_w."""
    lp_full = [1.0, 2.0, 3.0]
    lp_blank = [1.0, 2.0, 3.0]  # vd all-zero
    old_lp = [0.5, 1.0, 1.5]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "step_000001.jsonl.gz"
        _write_step_file([_make_row(lp_full, lp_blank, old_lp)], path)
        out = scan(Path(tmp), steps=None, limit_rows=0)
    p = out["pooled"]
    assert p["n_samples_seen"] == 1, p
    assert p["n_samples_used"] == 0, p
    assert p["n_samples_unit_w"] == 1, p
    print(f"  unit-weights-skip: n_unit_w={p['n_samples_unit_w']}  OK")


def test_sidecar_join_recovers_old_lp():
    """Diag row has empty old_lp_student; sidecar with matching sample_index
    provides it; audit should join and produce same numbers as if it were inline."""
    lp_full = [1.0, 2.0, 3.0, 4.0, 5.0]
    lp_blank = [3.0, 2.0, 1.0, 0.0, -1.0]
    old_lp = [2.0, 1.0, 0.0, -1.0, -2.0]

    # Reference: inline old_lp_student case (canonical test result).
    with tempfile.TemporaryDirectory() as tmp1:
        path1 = Path(tmp1) / "step_000001.jsonl.gz"
        _write_step_file([_make_row(lp_full, lp_blank, old_lp, sample_index=42)], path1)
        ref_out = scan(Path(tmp1), steps=None, limit_rows=0)
    ref_rho = ref_out["pooled"]["rho_l2_pooled"]

    # Now: diag row with EMPTY old_lp_student, sidecar provides it.
    with tempfile.TemporaryDirectory() as tmp2:
        diag_path = Path(tmp2) / "step_000001.jsonl.gz"
        sidecar_path = Path(tmp2) / "step_000001.adv_dp0.jsonl.gz"
        row = _make_row(lp_full, lp_blank, [], sample_index=42)  # empty old_lp
        _write_step_file([row], diag_path)
        _write_sidecar_file([(42, old_lp)], sidecar_path, step=1, dp_rank=0)
        out = scan(Path(tmp2), steps=None, limit_rows=0)

    p = out["pooled"]
    assert p["n_samples_used"] == 1, p
    assert p["n_samples_old_lp_from_sidecar"] == 1, p
    assert out["n_steps_with_sidecar"] == 1, out
    assert _approx(p["rho_l2_pooled"], ref_rho), (p["rho_l2_pooled"], ref_rho)
    print(f"  sidecar-join: from_sidecar={p['n_samples_old_lp_from_sidecar']}, "
          f"rho_l2={p['rho_l2_pooled']:.4f} matches inline {ref_rho:.4f}  OK")


def test_sidecar_multi_dp_merge():
    """Two dp ranks each contribute disjoint sample_indices for the same step;
    audit merges them and processes all samples."""
    lp_full = [1.0, 2.0, 3.0, 4.0, 5.0]
    lp_blank = [3.0, 2.0, 1.0, 0.0, -1.0]
    old_lp = [2.0, 1.0, 0.0, -1.0, -2.0]

    with tempfile.TemporaryDirectory() as tmp:
        diag_path = Path(tmp) / "step_000001.jsonl.gz"
        rows = [
            _make_row(lp_full, lp_blank, [], sample_index=10),
            _make_row(lp_full, lp_blank, [], sample_index=20),
        ]
        _write_step_file(rows, diag_path)
        # Two dp ranks, each with one sample.
        _write_sidecar_file([(10, old_lp)], Path(tmp) / "step_000001.adv_dp0.jsonl.gz",
                            step=1, dp_rank=0)
        _write_sidecar_file([(20, old_lp)], Path(tmp) / "step_000001.adv_dp1.jsonl.gz",
                            step=1, dp_rank=1)
        out = scan(Path(tmp), steps=None, limit_rows=0)

    p = out["pooled"]
    assert p["n_samples_used"] == 2, p
    assert p["n_samples_old_lp_from_sidecar"] == 2, p
    print(f"  sidecar-multi-dp: merged 2 dp ranks, "
          f"n_used={p['n_samples_used']}  OK")


def test_sidecar_missing_index_falls_through():
    """Sidecar present but no matching sample_index → audit reports no_old_lp."""
    lp_full = [1.0, 2.0, 3.0, 4.0, 5.0]
    lp_blank = [3.0, 2.0, 1.0, 0.0, -1.0]
    old_lp = [2.0, 1.0, 0.0, -1.0, -2.0]

    with tempfile.TemporaryDirectory() as tmp:
        diag_path = Path(tmp) / "step_000001.jsonl.gz"
        _write_step_file([_make_row(lp_full, lp_blank, [], sample_index=42)], diag_path)
        # Sidecar has a DIFFERENT sample_index.
        _write_sidecar_file([(99, old_lp)], Path(tmp) / "step_000001.adv_dp0.jsonl.gz",
                            step=1, dp_rank=0)
        out = scan(Path(tmp), steps=None, limit_rows=0)

    p = out["pooled"]
    assert p["n_samples_used"] == 0, p
    assert p["n_samples_no_old_lp"] == 1, p
    print(f"  sidecar-missing-index: no_old_lp={p['n_samples_no_old_lp']}  OK")


def test_skips_missing_lp_full():
    """Sentinel-failed sample (lp_full=[]) should go to no_adv bucket."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "step_000001.jsonl.gz"
        row = {
            "id": "failed",
            "sample_index": 0,
            "step": 0,
            "response_length": 5,
            "image_mode": "full",
            "teacher_url": "synth",
            "teacher_url_diag": "synth",
            "response_correct": True,
            "lp_full": [],
            "lp_blank": [],
            "vd": [],
            "vd_weights": [],
            "old_lp_student": [0.0] * 5,
        }
        with gzip.open(path, "wt") as f:
            f.write(json.dumps(row) + "\n")
        out = scan(Path(tmp), steps=None, limit_rows=0)
    p = out["pooled"]
    assert p["n_samples_seen"] == 1, p
    assert p["n_samples_used"] == 0, p
    assert p["n_samples_no_adv"] == 1, p
    print(f"  missing-lp-full: n_no_adv={p['n_samples_no_adv']}  OK")


def test_multi_step_aggregation():
    """Two steps, two rows each — pooled should aggregate, per-step disaggregate."""
    lp_full = [1.0, 2.0, 3.0, 4.0, 5.0]
    lp_blank = [3.0, 2.0, 1.0, 0.0, -1.0]
    old_lp = [2.0, 1.0, 0.0, -1.0, -2.0]
    row = _make_row(lp_full, lp_blank, old_lp)
    with tempfile.TemporaryDirectory() as tmp:
        for s in [10, 50]:
            path = Path(tmp) / f"step_{s:06d}.jsonl.gz"
            _write_step_file([row, row], path)
        out = scan(Path(tmp), steps=None, limit_rows=0)
    p = out["pooled"]
    assert p["n_samples_used"] == 4, p
    assert p["n_tokens_used"] == 20, p
    assert set(out["per_step"].keys()) == {"10", "50"}, out["per_step"].keys()
    for step_key in ["10", "50"]:
        st = out["per_step"][step_key]
        assert st["n_samples_used"] == 2, st
        # pooled rho_l2 must equal per-step (same data) within float tol
        assert _approx(st["rho_l2_pooled"], p["rho_l2_pooled"]), (st, p)
    print(f"  multi-step-aggregation: pooled rho_l2={p['rho_l2_pooled']:.4f}, "
          f"per-step rho_l2={out['per_step']['10']['rho_l2_pooled']:.4f}  OK")


def test_vd_bin_assignment():
    """All tokens land in some VD-norm bin; bin totals == n_tokens_used."""
    lp_full = [1.0, 2.0, 3.0, 4.0, 5.0]
    lp_blank = [3.0, 2.0, 1.0, 0.0, -1.0]
    old_lp = [2.0, 1.0, 0.0, -1.0, -2.0]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "step_000001.jsonl.gz"
        _write_step_file([_make_row(lp_full, lp_blank, old_lp)], path)
        out = scan(Path(tmp), steps=None, limit_rows=0)
    p = out["pooled"]
    bins = p["abs_adv_by_vd_norm_bin"]
    total_in_bins = sum(b["n_tokens"] for b in bins.values())
    assert total_in_bins == p["n_tokens_used"], (bins, p["n_tokens_used"])
    print(f"  vd-bin-assignment: total_tokens_in_bins={total_in_bins} == n_tokens_used  OK")


def main():
    tests = [
        test_canonical_5token,
        test_anti_correlated,
        test_visual_rejection_suppression,
        test_skips_unit_weights,
        test_sidecar_join_recovers_old_lp,
        test_sidecar_multi_dp_merge,
        test_sidecar_missing_index_falls_through,
        test_skips_missing_lp_full,
        test_multi_step_aggregation,
        test_vd_bin_assignment,
    ]
    print(f"Running {len(tests)} smoke tests for t2_1_energy_audit:")
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
