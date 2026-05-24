"""T2-1 / T2-2: per-token VD (Vision Dependency) weighting for OPD advantages.

Two compute paths:

  * `compute_vd_weights`           — T2-1 (PGPO Eq 6 + 7, signed vd,
                                     threshold-gated piecewise +
                                     per-sequence mass-preserving renorm).
                                     CONFIRMED MIS-TARGETING in OPD by
                                     A0 audit (cond_supp=0.997) — kept
                                     for reproducibility and ablation.
  * `compute_vd_weights_boost_only` — T2-2 (boost-only, |vd|, percentile-
                                     rank score, no suppress, no renorm).
                                     Designed per GPT round-5 to fix
                                     the signed-proxy mis-target and
                                     the structural frac_supp ~7% floor.
                                     See `docs/t2_2_design.md`.

The dispatch lives in `opd_diagnostics_hook.post_process_rewards_with_diagnostics`
and is gated by env `MLLMOPD_VD_MODE` (signed | boost_only, default signed).

Inputs: per-token teacher logprobs under full-image (`lp_full`) and
blank-image (`lp_blank`) conditions, aligned to the response segment
(length = response_length). Output: a per-token weight tensor of the
same length to multiply into the OPD advantage in
`compute_advantages_and_returns`.

Formula (PGPO Eq 6 + Eq 7, on min-max-normalized vd):

  vd_t        = lp_full(t) - lp_blank(t)
  vd_norm_t   = (vd_t - vd_min) / (vd_max - vd_min + eps)   per-sequence
  w_raw_t     = vd_norm_t / (tau + eps)                     if vd_norm_t < tau
              = 1 + beta * (vd_norm_t - tau) / (1 - tau + eps)  if vd_norm_t >= tau
  w_t         = w_raw_t * N / sum(w_raw_t)                  per-sequence

Defaults: tau=0.4, beta=2.0 (PGPO paper Table 2 winner; env-overridable).

Why min-max (not raw vd): PGPO's I_t is a non-negative KL divergence in
[0, +inf); their threshold tau=0.4 is meaningful on that scale. Our
vd_t = lp_full - lp_blank can be negative and is scale-free, so we
normalize per-sequence to bring tau back into [0, 1]. Percentile-based
threshold (e.g., 40th-percentile of vd_t) is an equivalent alternative
considered and rejected because min-max keeps the proportionality
between adjacent token magnitudes that PGPO's piecewise function relies
on.

Why per-sequence mass conservation (Eq 7): without it, the effective
learning rate drifts with the per-sequence VD distribution; PGPO's own
no-renorm ablation drops ~1.9 pts. Per-sequence (not per-batch) avoids
coupling the loss to batch composition (a VPPO-TAS pathology).

Edge cases (returns unit-weight no-op tensor):
  - Empty inputs / response_length == 0
  - Length mismatch (logs warning)
  - All-equal vd (vd_max - vd_min < eps): no reweighting signal available
  - response_length == 1: nothing to compare against
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

import torch

logger = logging.getLogger(__name__)

_EPS = 1e-8


def _resolve_hyperparams() -> tuple[float, float]:
    """Read tau, beta from env (set by launcher). Defaults match PGPO Table 2."""
    tau = float(os.environ.get("MLLMOPD_VD_TAU", "0.4"))
    beta = float(os.environ.get("MLLMOPD_VD_BETA", "2.0"))
    if not (0.0 < tau < 1.0):
        raise ValueError(f"MLLMOPD_VD_TAU must be in (0, 1); got {tau}")
    if beta < 0.0:
        raise ValueError(f"MLLMOPD_VD_BETA must be non-negative; got {beta}")
    return tau, beta


def compute_vd_weights(
    lp_full: Sequence[float] | torch.Tensor,
    lp_blank: Sequence[float] | torch.Tensor,
    response_length: int,
    *,
    tau: float | None = None,
    beta: float | None = None,
) -> torch.Tensor:
    """Compute per-token PGPO-style VD weights.

    Args:
        lp_full: per-token teacher logprobs under full image, length = response_length.
        lp_blank: per-token teacher logprobs under blank image, length = response_length.
        response_length: expected output length. Used as N in mass-preserve renorm
            (Eq 7) and to validate the inputs.
        tau, beta: optional override of env hyperparameters. Mainly for testing.

    Returns:
        torch.float32 tensor of shape `[response_length]`. All-ones (unit no-op
        weight) on any degenerate case. Always non-negative, always sums to
        `response_length` within float tolerance when computed normally.
    """
    if tau is None or beta is None:
        env_tau, env_beta = _resolve_hyperparams()
        tau = tau if tau is not None else env_tau
        beta = beta if beta is not None else env_beta

    if response_length <= 0:
        return torch.zeros(0, dtype=torch.float32)

    if response_length == 1:
        return torch.ones(1, dtype=torch.float32)

    lp_full_t = torch.as_tensor(lp_full, dtype=torch.float32)
    lp_blank_t = torch.as_tensor(lp_blank, dtype=torch.float32)

    if lp_full_t.numel() != response_length or lp_blank_t.numel() != response_length:
        logger.warning(
            "[vd_weighting] length mismatch: lp_full=%d, lp_blank=%d, expected=%d. "
            "Returning unit weights (no-op).",
            lp_full_t.numel(), lp_blank_t.numel(), response_length,
        )
        return torch.ones(response_length, dtype=torch.float32)

    vd = lp_full_t - lp_blank_t

    vd_min = vd.min()
    vd_max = vd.max()
    vd_range = vd_max - vd_min

    if vd_range.item() < _EPS:
        return torch.ones(response_length, dtype=torch.float32)

    vd_norm = (vd - vd_min) / (vd_range + _EPS)

    suppress_mask = vd_norm < tau
    boost_mask = ~suppress_mask

    w_raw = torch.empty_like(vd_norm)
    w_raw[suppress_mask] = vd_norm[suppress_mask] / (tau + _EPS)
    w_raw[boost_mask] = 1.0 + beta * (vd_norm[boost_mask] - tau) / (1.0 - tau + _EPS)

    w_sum = w_raw.sum()
    if w_sum.item() < _EPS:
        return torch.ones(response_length, dtype=torch.float32)

    w = w_raw * (response_length / w_sum)
    return w


def _resolve_boost_only_hyperparams() -> tuple[float, float]:
    """Read α, max_w from env (set by launcher). Defaults per GPT round-5."""
    alpha = float(os.environ.get("MLLMOPD_VD_ALPHA", "1.0"))
    max_w = float(os.environ.get("MLLMOPD_VD_MAX_W", "2.0"))
    if alpha < 0.0:
        raise ValueError(f"MLLMOPD_VD_ALPHA must be non-negative; got {alpha}")
    if max_w < 1.0:
        raise ValueError(f"MLLMOPD_VD_MAX_W must be >= 1.0 (boost-only floor); got {max_w}")
    return alpha, max_w


def compute_vd_weights_boost_only(
    lp_full: Sequence[float] | torch.Tensor,
    lp_blank: Sequence[float] | torch.Tensor,
    response_length: int,
    *,
    alpha: float | None = None,
    max_w: float | None = None,
) -> torch.Tensor:
    """T2-2: boost-only |vd| weights with per-sequence percentile rank.

    Formula::

        score_t = rank(|vd_t|) / (R - 1)        # per-sequence percentile rank ∈ [0, 1]
        w_t     = clamp(1 + α · score_t, 1, max_w)

    Key properties (vs T2-1 PGPO Eq 6+7):
      * w_t ≥ 1 everywhere → NO suppression → frac_supp_neg_vd_neg_adv ≡ 0
        BY CONSTRUCTION. The structural ~7% floor of T2-1 disappears.
      * NO mass-preserving renormalization → Σw > N (total monotone
        increase). This means more energy enters the gradient, but ALL
        of it is base-OPD-correction-preserving boost rather than
        redistribution. Effective LR rises by ~E[w]; control via
        global LR or grad clipping if needed.
      * Percentile rank (not min-max) is OUTLIER-ROBUST: one extreme
        |vd| does not compress everyone else into a tiny region.
      * Uses |vd| (non-negative magnitude) so visual-rejection tokens
        (vd<0, |vd| large) get the same boost as visual-support tokens
        (vd>0, |vd| large) — exactly what OPD needs (the sign is
        already in adv_t = teacher_logp - student_logp).

    Defaults: α=1.0 → w_t ∈ [1, 2] for max_w=2.0; mean ≈ 1.5 under
    uniform percentile rank. Override per env vars MLLMOPD_VD_ALPHA
    and MLLMOPD_VD_MAX_W.

    Edge cases (return all-ones, no-op):
      * Response length 0 or 1
      * Length mismatch (logs warning)
      * Tie-breaking in rank uses torch.argsort's stable behavior
        (deterministic). All-equal |vd| → ranks 0/1/2/.../(R-1)/(R-1)
        → after normalize all but max get a real score. Acceptable.
    """
    if alpha is None or max_w is None:
        env_alpha, env_max_w = _resolve_boost_only_hyperparams()
        alpha = alpha if alpha is not None else env_alpha
        max_w = max_w if max_w is not None else env_max_w

    if response_length <= 0:
        return torch.zeros(0, dtype=torch.float32)
    if response_length == 1:
        return torch.ones(1, dtype=torch.float32)

    lp_full_t = torch.as_tensor(lp_full, dtype=torch.float32)
    lp_blank_t = torch.as_tensor(lp_blank, dtype=torch.float32)

    if lp_full_t.numel() != response_length or lp_blank_t.numel() != response_length:
        logger.warning(
            "[vd_weighting/boost_only] length mismatch: lp_full=%d, lp_blank=%d, "
            "expected=%d. Returning unit weights (no-op).",
            lp_full_t.numel(), lp_blank_t.numel(), response_length,
        )
        return torch.ones(response_length, dtype=torch.float32)

    abs_vd = (lp_full_t - lp_blank_t).abs()

    # Per-sequence percentile rank via argsort (vectorized).
    # sorted_idx[k] = position of the k-th smallest abs_vd in the original tensor.
    # We want rank[i] = position of abs_vd[i] in the sorted order, normalized
    # to [0, 1]. Use scatter via inverse permutation.
    sorted_idx = torch.argsort(abs_vd)
    ranks = torch.empty_like(abs_vd)
    ranks[sorted_idx] = torch.arange(response_length, dtype=torch.float32) / (response_length - 1)

    w = (1.0 + alpha * ranks).clamp(min=1.0, max=max_w)
    return w
