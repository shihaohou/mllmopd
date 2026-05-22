#!/usr/bin/env python
"""Smoke test for src/mllmopd/training/vd_weighting.py.

Synthetic tests cover all branches of compute_vd_weights:
  - mass-preservation (sum == response_length)
  - non-negativity
  - monotonicity (higher vd → higher weight)
  - threshold behavior at tau
  - degenerate cases (empty, length=1, length-mismatch, all-equal vd)

Optional integration: if MLLMOPD_VD_VERIFY_JSONL points at a real
diagnostics .jsonl.gz (produced by opd_diagnostics_hook on H800),
also compute weights on every row and report distribution stats.

Run:
    python scripts/train/verify_vd_weighting.py
    MLLMOPD_VD_VERIFY_JSONL=runs/.../step_000010.jsonl.gz \\
        python scripts/train/verify_vd_weighting.py
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from mllmopd.training.vd_weighting import compute_vd_weights  # noqa: E402


def _approx(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def test_mass_preservation_uniform_increase():
    R = 16
    lp_full = [float(i) for i in range(R)]
    lp_blank = [0.0] * R
    w = compute_vd_weights(lp_full, lp_blank, R)
    assert w.shape == (R,), w.shape
    assert (w >= 0).all().item(), "weights must be non-negative"
    assert _approx(w.sum().item(), R), f"sum {w.sum().item():.6f} != {R}"
    diffs = w[1:] - w[:-1]
    assert (diffs >= -1e-5).all().item(), "monotonically-increasing vd → monotone weights"


def test_threshold_step():
    R = 10
    lp_full = [0.0] * (R // 2) + [1.0] * (R // 2)
    lp_blank = [0.0] * R
    w = compute_vd_weights(lp_full, lp_blank, R, tau=0.4, beta=2.0)
    low, high = w[: R // 2], w[R // 2 :]
    assert (low < 1.0).all().item(), f"below-tau should suppress: low={low.tolist()}"
    assert (high > 1.0).all().item(), f"at/above-tau should boost: high={high.tolist()}"
    assert _approx(w.sum().item(), R)


def test_all_equal_vd_returns_ones():
    R = 8
    lp_full = [3.14] * R
    lp_blank = [2.71] * R
    w = compute_vd_weights(lp_full, lp_blank, R)
    assert torch.allclose(w, torch.ones(R)), f"all-equal vd should give unit weights; got {w.tolist()}"


def test_response_length_one():
    w = compute_vd_weights([0.5], [0.1], 1)
    assert w.shape == (1,) and w.item() == 1.0


def test_response_length_zero():
    w = compute_vd_weights([], [], 0)
    assert w.numel() == 0


def test_length_mismatch_returns_ones():
    w = compute_vd_weights([1.0, 2.0], [0.0, 0.0, 0.0], 3)
    assert torch.allclose(w, torch.ones(3))


def test_pgpo_eq6_hand_computed():
    """Verify PGPO Eq 6 + Eq 7 against hand-computed weights on a 5-token row.

    Inputs span vd_norm = {0, 0.25, 0.4, 0.7, 1.0} with tau=0.4, beta=2.0.
    Eq 6 pre-renorm weights:
        0.0:   suppress, w = 0   /0.4 = 0.0
        0.25:  suppress, w = 0.25/0.4 = 0.625
        0.4:   boost,    w = 1 + 2*0/0.6 = 1.0       (continuity at tau)
        0.7:   boost,    w = 1 + 2*0.3/0.6 = 2.0
        1.0:   boost,    w = 1 + 2*0.6/0.6 = 3.0
    sum_raw = 6.625, R/sum = 5/6.625 ≈ 0.75472
    Expected post-renorm:
        [0.0, 0.47170, 0.75472, 1.50943, 2.26415]
    """
    R = 5
    lp_full = [0.0, 0.25, 0.4, 0.7, 1.0]
    lp_blank = [0.0] * R
    w = compute_vd_weights(lp_full, lp_blank, R, tau=0.4, beta=2.0)
    expected = torch.tensor([0.0, 0.47170, 0.75472, 1.50943, 2.26415])
    assert torch.allclose(w, expected, atol=1e-3), f"got {w.tolist()}, expected {expected.tolist()}"
    assert _approx(w.sum().item(), R), f"mass not preserved: {w.sum().item()}"


def test_random_inputs_no_nan():
    torch.manual_seed(0)
    for _ in range(50):
        R = int(torch.randint(2, 256, (1,)).item())
        lp_full = torch.randn(R).tolist()
        lp_blank = torch.randn(R).tolist()
        w = compute_vd_weights(lp_full, lp_blank, R)
        assert not torch.isnan(w).any(), "NaN in weights"
        assert (w >= 0).all().item(), "negative weight produced"
        assert _approx(w.sum().item(), R, tol=1e-3)


def test_env_var_override():
    os.environ["MLLMOPD_VD_TAU"] = "0.5"
    os.environ["MLLMOPD_VD_BETA"] = "1.0"
    try:
        from mllmopd.training.vd_weighting import _resolve_hyperparams
        tau, beta = _resolve_hyperparams()
        assert tau == 0.5 and beta == 1.0
    finally:
        os.environ.pop("MLLMOPD_VD_TAU")
        os.environ.pop("MLLMOPD_VD_BETA")


def _run_unit_tests():
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failures = []
    for t in tests:
        try:
            t()
            print(f"  ok   {t.__name__}")
        except AssertionError as e:
            failures.append((t.__name__, str(e)))
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:
            failures.append((t.__name__, f"{type(e).__name__}: {e}"))
            print(f"  ERR  {t.__name__}: {type(e).__name__}: {e}")
    return failures


def _run_jsonl_integration(jsonl_path: str):
    """Per-row weight stats on a real diagnostics dump."""
    path = Path(jsonl_path)
    if not path.exists():
        print(f"  jsonl not found at {path} — skipping")
        return
    print(f"\nIntegration on {path}:")
    n_rows = 0
    n_degen = 0
    sparsities = []
    max_weights = []
    with gzip.open(path, "rt") as fin:
        for line in fin:
            row = json.loads(line)
            lp_full = row.get("lp_full") or []
            lp_blank = row.get("lp_blank") or []
            R = int(row.get("response_length", 0))
            if R == 0 or len(lp_full) != R or len(lp_blank) != R:
                n_degen += 1
                continue
            w = compute_vd_weights(lp_full, lp_blank, R)
            n_rows += 1
            sparsity = (w < 1.0).float().mean().item()
            sparsities.append(sparsity)
            max_weights.append(w.max().item())

    if n_rows == 0:
        print("  no usable rows — all degenerate?")
        return
    s_mean = sum(sparsities) / len(sparsities)
    mw_mean = sum(max_weights) / len(max_weights)
    print(f"  rows                : {n_rows} ({n_degen} degenerate)")
    print(f"  mean sparsity       : {s_mean:.3f}  (fraction of tokens with w<1)")
    print(f"  mean max-weight     : {mw_mean:.3f}  (peak boost per row)")
    print(f"  sparsity 5/50/95 %  : "
          f"{sorted(sparsities)[len(sparsities)//20]:.3f} / "
          f"{sorted(sparsities)[len(sparsities)//2]:.3f} / "
          f"{sorted(sparsities)[len(sparsities)*19//20]:.3f}")


def main():
    print("Unit tests (compute_vd_weights):")
    failures = _run_unit_tests()

    jsonl = os.environ.get("MLLMOPD_VD_VERIFY_JSONL")
    if jsonl:
        _run_jsonl_integration(jsonl)
    else:
        print("\n(no MLLMOPD_VD_VERIFY_JSONL set; skipping integration on real diagnostics)")

    if failures:
        print(f"\n{len(failures)} FAILURES")
        sys.exit(1)
    print(f"\nAll {sum(1 for k in globals() if k.startswith('test_'))} unit tests passed.")


if __name__ == "__main__":
    main()
