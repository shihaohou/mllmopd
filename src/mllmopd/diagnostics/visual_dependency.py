"""Token-level visual dependency utilities.

`vis_dep(t)` = KL( p(.|x, image)[t]  ||  p(.|x, blank)[t] )

Operates on numpy log-prob arrays so it's import-friendly on Mac. The actual
log-prob extraction (forced-decoding the teacher over a fixed response while
varying the image) belongs in run_audit_pass.py on the devbox.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def kl_per_token(logp_a: np.ndarray, logp_b: np.ndarray) -> np.ndarray:
    """KL(P_a || P_b) per token. Both inputs shape (T, V) of log-probabilities."""
    assert logp_a.shape == logp_b.shape, f"shape mismatch: {logp_a.shape} vs {logp_b.shape}"
    pa = np.exp(logp_a)
    return (pa * (logp_a - logp_b)).sum(axis=-1)


def quantile_bins(values: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Return bin index (0..n_bins-1) for each value, using equal-frequency bins."""
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9  # include the max
    return np.clip(np.digitize(values, edges[1:-1]), 0, n_bins - 1)


def loss_mass_by_bin(loss: np.ndarray, bins: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Sum of loss within each bin, normalized to sum to 1."""
    out = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_bins):
        out[i] = float(loss[bins == i].sum())
    total = out.sum()
    return out / total if total > 0 else out


def summarize_records(records: Iterable[dict]) -> dict:
    """Aggregate per-token vis_dep across many examples.

    Each record must have keys: `vis_dep` (T,) and `opd_loss` (T,) numpy arrays.
    """
    all_vd, all_loss = [], []
    for r in records:
        all_vd.append(np.asarray(r["vis_dep"]))
        all_loss.append(np.asarray(r["opd_loss"]))
    vd = np.concatenate(all_vd)
    loss = np.concatenate(all_loss)
    bins = quantile_bins(vd, n_bins=5)
    return {
        "n_tokens": int(vd.size),
        "vis_dep_mean": float(vd.mean()),
        "vis_dep_median": float(np.median(vd)),
        "loss_mass_per_bin": loss_mass_by_bin(loss, bins, n_bins=5).tolist(),
    }
