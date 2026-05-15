"""Token-level visual dependency utilities.

Two flavors of per-token visual dependency are supported, with very different
storage costs. The audit pipeline saves the cheap one by default and uses the
expensive one only for hand-picked case studies.

(1) Default — `vis_dep_generated`:

    vd[t] = | logp_full(y_t | x, y_<t) - logp_blank(y_t | x, y_<t) |

    Only the log-prob of the *actually generated* token is stored under each
    image conditioning. Cost: (T,) scalar arrays per sample. This is what
    `summarize_records` consumes and what audit JSONL writes.

(2) Case study — `kl_per_token`:

    vd[t] = KL( p_full(.|x, y_<t) || p_blank(.|x, y_<t) )

    Full-vocab KL. Cost: (T, V) log-prob arrays per sample — for Qwen2.5-VL
    with V≈152K and T≈1024 that's ~1.2 GB float32 per sample, so this is only
    feasible on ~10-100 case study examples, not at audit scale.

Operates on numpy arrays so it's import-friendly on Mac. The log-prob
extraction itself (forced-decoding the teacher over a fixed response while
varying the image) lives in run_audit_pass.py on the devbox.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


def vis_dep_generated(logp_full: np.ndarray, logp_blank: np.ndarray) -> np.ndarray:
    """Cheap scalar visual dependency from generated-token log-probs.

    Both inputs are 1-D arrays of shape (T,) — the log-prob of the actually
    generated token y_t under (a) full-image and (b) blank-image conditioning.
    """
    assert logp_full.shape == logp_blank.shape, f"shape mismatch: {logp_full.shape} vs {logp_blank.shape}"
    return np.abs(np.asarray(logp_full) - np.asarray(logp_blank))


def kl_per_token(logp_a: np.ndarray, logp_b: np.ndarray) -> np.ndarray:
    """KL(P_a || P_b) per token. Both inputs shape (T, V) of log-probabilities.

    Case-study only — storage is (T, V) per sample. Use `vis_dep_generated` for
    the audit pipeline."""
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
