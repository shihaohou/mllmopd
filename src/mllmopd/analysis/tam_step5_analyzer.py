"""Step 5 evidence-alignment analyzer.

Inputs:
  - `runs/audit/tam_step5_<TS>/alignment.jsonl` (output of
    `scripts/audit/tam_step5_evidence_alignment.py`).

Outputs (under `--out-dir`, default `docs/figures/step5/`):
  - `tables/overall.csv`           — overall paired Δ(JS,IoU,Cos)
  - `tables/per_bucket.csv`        — by bucket
  - `tables/per_category.csv`      — by token_category
  - `fig1_alignment_delta_overall.png`
  - `fig2_per_bucket.png`
  - `fig3_per_token_category.png`
  - `qualitative_cases.jsonl`      — picks for `tam_render_overlays.py`
  - `step5-results.md`             — auto-generated decision-tree report

The headline metric is paired Δ across **samples** (cluster bootstrap),
not across tokens, because tokens within a sample are not independent.

Usage::

    python -m mllmopd.analysis.tam_step5_analyzer \\
        --alignment runs/audit/tam_step5_<TS>/alignment.jsonl \\
        --out-dir docs/figures/step5/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ============================================================================
# Module-level constants — must precede any function that uses them as default
# ============================================================================
# Reliability filter threshold for `tam_entropy_norm_T < threshold`. v0.1.3
# TAM on long-CoT MMR1 outputs has entropy saturated at 0.99+ (smoke
# 2026-05-28 on 4355 tokens: min=0.9936, p25=0.9964, median=0.9974,
# p75=0.9982, p90=0.9988, max=0.9996). The Step 0 / Step 2 calibration
# (0.85-0.99, threshold = 0.95) was on v0.1.2 + 3B + short prompt (R ≈ 80)
# and is NOT applicable to the long-CoT 7B-teacher regime — 0.95 filters
# out every token. Default raised to 0.998 (≈ p75 of v0.1.3 distribution
# on this regime), overridable via --reliability-thresh.
RELIABILITY_THRESH_DEFAULT = 0.998


# ============================================================================
# Data loading
# ============================================================================
def load_alignment(path: Path) -> list[dict]:
    """Load alignment JSONL; lightly validate per-sample structure."""
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Sanity: alignment keys present
            if "align" not in rec or "S0_T" not in rec["align"] or "S1_T" not in rec["align"]:
                continue
            rows.append(rec)
    return rows


def _per_sample_means(rec: dict, *,
                      token_filter=None,
                      reliability_thresh: float | None = RELIABILITY_THRESH_DEFAULT
                      ) -> dict | None:
    """Compute sample-level mean of (S1−S0 vs T) for IoU / JS / Cos over
    tokens that satisfy `token_filter` AND have valid TAM on all three
    models, AND (optionally) the teacher map's `tam_entropy_norm` <
    `reliability_thresh`.

    Returns None if no tokens survive the filter.
    """
    R = rec["response_length"]
    valid = [
        rec["tam_valid_T"][t]
        and rec["tam_valid_S0"][t]
        and rec["tam_valid_S1"][t]
        for t in range(R)
    ]
    if reliability_thresh is not None:
        entropy_norm_T = rec["T"].get("tam_entropy_norm", [None] * R)
        valid = [
            valid[t] and entropy_norm_T[t] is not None
            and entropy_norm_T[t] < reliability_thresh
            for t in range(R)
        ]
    if token_filter is not None:
        valid = [valid[t] and token_filter(rec, t) for t in range(R)]
    if not any(valid):
        return None

    a = rec["align"]
    # ΔJS = JS(S0,T) − JS(S1,T)   (positive = OPD improves alignment)
    # ΔIoU= IoU(S1,T) − IoU(S0,T)
    # ΔCos= Cos(S1,T) − Cos(S0,T)
    js_s0 = [a["S0_T"]["js"][t] for t in range(R) if valid[t] and a["S0_T"]["js"][t] is not None]
    js_s1 = [a["S1_T"]["js"][t] for t in range(R) if valid[t] and a["S1_T"]["js"][t] is not None]
    iou_s0 = [a["S0_T"]["iou_top20"][t] for t in range(R) if valid[t] and a["S0_T"]["iou_top20"][t] is not None]
    iou_s1 = [a["S1_T"]["iou_top20"][t] for t in range(R) if valid[t] and a["S1_T"]["iou_top20"][t] is not None]
    cos_s0 = [a["S0_T"]["cos"][t] for t in range(R) if valid[t] and a["S0_T"]["cos"][t] is not None]
    cos_s1 = [a["S1_T"]["cos"][t] for t in range(R) if valid[t] and a["S1_T"]["cos"][t] is not None]
    if not (js_s0 and js_s1 and iou_s0 and iou_s1):
        return None

    return {
        "n_tokens":    sum(valid),
        "mean_js_S0":  float(np.mean(js_s0)),
        "mean_js_S1":  float(np.mean(js_s1)),
        "mean_iou_S0": float(np.mean(iou_s0)),
        "mean_iou_S1": float(np.mean(iou_s1)),
        "mean_cos_S0": float(np.mean(cos_s0)),
        "mean_cos_S1": float(np.mean(cos_s1)),
        "delta_js":   float(np.mean(js_s0) - np.mean(js_s1)),
        "delta_iou":  float(np.mean(iou_s1) - np.mean(iou_s0)),
        "delta_cos":  float(np.mean(cos_s1) - np.mean(cos_s0)),
    }


# ============================================================================
# Cluster bootstrap CI + null calibration + TOST
# ============================================================================
# Expected IoU of two independent random top-20% patch sets on N patches:
#   E[|A∩B|] = N·p²,  E[|A∪B|] = N(1−(1−p)²) = N(2p−p²)
#   E[IoU]  ≈ p² / (2p−p²) = p / (2−p) = 0.2/1.8 ≈ 0.1111
# (Reported alongside raw IoU for chance correction.)
RANDOM_IOU_BASELINE_TOP20 = 0.20 / (2 - 0.20)  # = 0.1111...

# RELIABILITY_THRESH_DEFAULT defined near module top (must precede
# `_per_sample_means` default).


def cluster_bootstrap_ci(per_sample_values: list[float], n_iter: int = 10000,
                         alpha: float = 0.05, seed: int = 1729
                         ) -> tuple[float, float, float, float]:
    """Returns (mean, low, high, bootstrap_sd) at the (1-alpha) bootstrap CI.

    bootstrap_sd is the std of the per-iter bootstrap means; used as a
    proxy for SE(Δ) in MDE computation (MDE_95 ≈ 1.96·SE).
    """
    if not per_sample_values:
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_sample_values, dtype=np.float64)
    n = len(arr)
    means = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(low), float(high), float(means.std())


def calibrate_null_threshold(per_sample_values: list[float],
                             n_iter: int = 1000,
                             alpha: float = 0.05,
                             seed: int = 7777
                             ) -> float:
    """Sign-flip permutation null (Null A in design §6.4).

    Under H0 (S0 and S1 equally aligned with T), the per-sample Δ is
    sign-symmetric. Randomly flip each sample's Δ sign, take the mean,
    repeat n_iter times. δ = (1−alpha) percentile of |null means|.

    This gives a per-metric noise floor that's appropriate to the actual
    cluster size and per-sample variance, vs hardcoded thresholds.
    """
    if not per_sample_values:
        return float("nan")
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_sample_values, dtype=np.float64)
    n = len(arr)
    null_means = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        signs = rng.choice([-1.0, 1.0], size=n)
        null_means[i] = (arr * signs).mean()
    return float(np.quantile(np.abs(null_means), 1 - alpha))


def tost_equivalence(mean: float, ci_low: float, ci_high: float,
                     delta: float) -> str:
    """Two-one-sided-test (TOST) on a metric Δ, threshold δ.

    Returns one of:
      - 'aligned'     : mean > δ AND ci_low > 0  (positive effect, beyond noise)
      - 'anti'        : mean < −δ AND ci_high < 0 (negative effect; pathological)
      - 'equivalent'  : ci_low > −δ AND ci_high < +δ (TOST passes: bounded by δ)
      - 'inconclusive': otherwise (underpowered or large CI)
    """
    if not (np.isfinite(mean) and np.isfinite(ci_low) and np.isfinite(ci_high)
            and np.isfinite(delta)):
        return "inconclusive"
    if mean > delta and ci_low > 0:
        return "aligned"
    if mean < -delta and ci_high < 0:
        return "anti"
    if ci_low > -delta and ci_high < +delta:
        return "equivalent"
    return "inconclusive"


# ============================================================================
# Aggregations
# ============================================================================
# Full v0.1.3 token category set (13 categories — per design §6.6).
ALL_TOKEN_CATEGORIES = (
    "content_noun", "visual_attribute", "proper_noun",       # C_local
    "visual_number", "answer_token", "spatial_relation",
    "template_token", "punctuation", "pronoun",
    "meta_cot_token", "special_token", "ocr_text", "other",
)
C_LOCAL_CATEGORIES = ("content_noun", "visual_attribute", "proper_noun")


def _metric_block(vals: list[float]) -> dict:
    """Returns {mean, ci_low, ci_high, sd, mde_95}.

    MDE_95 uses bootstrap percentile half-width (max distance from mean
    to CI endpoint), which is more robust to skewed bootstrap
    distributions than 1.96·sd. Per design §6.5 (updated per GPT review
    of c2066f5 — bootstrap_sd can underestimate when the per-sample
    distribution is skewed).
    """
    m, lo, hi, sd = cluster_bootstrap_ci(vals)
    if np.isfinite(m) and np.isfinite(lo) and np.isfinite(hi):
        mde = max(abs(m - lo), abs(hi - m))
    else:
        mde = float("nan")
    return {"mean": m, "ci_low": lo, "ci_high": hi, "sd": sd, "mde_95": mde}


def aggregate_overall(rows: list[dict], reliability: bool = False,
                      reliability_thresh: float = RELIABILITY_THRESH_DEFAULT
                      ) -> dict:
    """Overall paired Δ — no stratification."""
    thresh = reliability_thresh if reliability else None
    per_sample = [_per_sample_means(r, reliability_thresh=thresh) for r in rows]
    per_sample = [s for s in per_sample if s is not None]
    out: dict = {"n_samples": len(per_sample),
                 "reliability_filtered": bool(reliability)}
    for k in ("delta_js", "delta_iou", "delta_cos"):
        vals = [s[k] for s in per_sample]
        out[k] = _metric_block(vals)
        # Per-metric null calibration is computed once at aggregator-top-level;
        # here we just stash per-bucket arrays for later use.
        out[f"_{k}_per_sample"] = vals
    # Raw S0/S1 means
    for k in ("mean_js_S0", "mean_js_S1",
              "mean_iou_S0", "mean_iou_S1",
              "mean_cos_S0", "mean_cos_S1"):
        out[k] = float(np.mean([s[k] for s in per_sample])) if per_sample else float("nan")
    return out


def aggregate_per_bucket(rows: list[dict], reliability: bool = False,
                         reliability_thresh: float = RELIABILITY_THRESH_DEFAULT
                         ) -> dict:
    thresh = reliability_thresh if reliability else None
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        s = _per_sample_means(r, reliability_thresh=thresh)
        if s is None:
            continue
        by_bucket[r.get("bucket", "unknown")].append(s)
    out: dict = {}
    for bucket, lst in by_bucket.items():
        bk: dict = {"n_samples": len(lst), "reliability_filtered": bool(reliability)}
        for k in ("delta_js", "delta_iou", "delta_cos"):
            vals = [s[k] for s in lst]
            bk[k] = _metric_block(vals)
            bk[f"_{k}_per_sample"] = vals
        out[bucket] = bk
    return out


def aggregate_per_category(rows: list[dict], reliability: bool = False,
                           min_samples: int = 10,
                           reliability_thresh: float = RELIABILITY_THRESH_DEFAULT
                           ) -> dict:
    """Per token_category: filter tokens by category at sample level, then
    cluster-bootstrap across samples that contributed any tokens to that
    category."""
    thresh = reliability_thresh if reliability else None
    out: dict = {}
    for cat in ALL_TOKEN_CATEGORIES:
        def _filt(rec, t, c=cat):
            tc = rec.get("token_category", [])
            return t < len(tc) and tc[t] == c

        per_sample = [_per_sample_means(r, token_filter=_filt,
                                        reliability_thresh=thresh)
                      for r in rows]
        per_sample = [s for s in per_sample if s is not None]
        if len(per_sample) < min_samples:
            continue
        bk: dict = {
            "n_samples": len(per_sample),
            "n_tokens_total": sum(s["n_tokens"] for s in per_sample),
            "reliability_filtered": bool(reliability),
        }
        for k in ("delta_js", "delta_iou", "delta_cos"):
            vals = [s[k] for s in per_sample]
            bk[k] = _metric_block(vals)
        out[cat] = bk
    return out


def _decode_b64_map_raw(b64_str: str, h: int, w: int) -> np.ndarray:
    """Decode a uint8 b64 map back to [0,1] float32 H×W."""
    import base64 as _b64
    raw = _b64.b64decode(b64_str)
    return np.frombuffer(raw, dtype=np.uint8).reshape(h, w).astype(np.float32) / 255.0


# Map-level metric helpers (mirror of scripts/audit/tam_step5_evidence_alignment.py
# — re-defined here so the analyzer can compute Null B/C without importing the
# runner module). Numerically identical: same sum-normalize for JS, same
# argpartition for IoU.
_EPS = 1e-12


def _js_div(m1: np.ndarray, m2: np.ndarray) -> float:
    p = m1.flatten().astype(np.float64)
    q = m2.flatten().astype(np.float64)
    p = np.clip(p, 0.0, None)
    q = np.clip(q, 0.0, None)
    ps = p.sum()
    qs = q.sum()
    if ps < _EPS:
        p = np.full_like(p, 1.0 / p.size)
    else:
        p = p / ps
    if qs < _EPS:
        q = np.full_like(q, 1.0 / q.size)
    else:
        q = q / qs
    avg = 0.5 * (p + q)

    def _kl(a, b):
        a = np.clip(a, _EPS, 1.0)
        b = np.clip(b, _EPS, 1.0)
        return float((a * (np.log(a) - np.log(b))).sum())

    return 0.5 * _kl(p, avg) + 0.5 * _kl(q, avg)


def _iou_topk(m1: np.ndarray, m2: np.ndarray, frac: float = 0.20) -> float:
    flat1 = m1.flatten()
    flat2 = m2.flatten()
    n = flat1.size
    k = max(1, int(round(n * frac)))
    idx1 = set(np.argpartition(-flat1, k - 1)[:k].tolist())
    idx2 = set(np.argpartition(-flat2, k - 1)[:k].tolist())
    inter = idx1 & idx2
    union = idx1 | idx2
    if not union:
        return 0.0
    return len(inter) / len(union)


def _load_primary_sample_maps(rows: list[dict],
                              entropy_thresh: float = RELIABILITY_THRESH_DEFAULT) -> list[dict]:
    """Decode + reliability-filter maps for the OPD_improved primary cell.

    Returns a list of per-sample dicts: {T_maps, S0_maps, S1_maps,
    keep_idx}. Used by Null B (cross-sample pairing) and Null C
    (spatial scramble). Only emitted for rows in OPD_improved bucket
    that survive sample-level validity + at least one reliable token.
    """
    out: list[dict] = []
    for r in rows:
        if r.get("bucket") != "OPD_improved":
            continue
        h, w = r.get("map_h"), r.get("map_w")
        maps_b64 = r.get("maps_b64", {})
        if not (h and w and maps_b64):
            continue
        T_b64 = maps_b64.get("T", [])
        S0_b64 = maps_b64.get("S0", [])
        S1_b64 = maps_b64.get("S1", [])
        R = r.get("response_length", 0)
        if min(len(T_b64), len(S0_b64), len(S1_b64)) < R:
            continue
        valid_T = r.get("tam_valid_T", [False] * R)
        valid_S0 = r.get("tam_valid_S0", [False] * R)
        valid_S1 = r.get("tam_valid_S1", [False] * R)
        ent_T = r.get("T", {}).get("tam_entropy_norm", [None] * R)
        # Keep only tokens passing reliability + 3-model validity
        keep: list[int] = []
        for t in range(R):
            if not (valid_T[t] and valid_S0[t] and valid_S1[t]):
                continue
            e = ent_T[t] if t < len(ent_T) else None
            if e is None or e >= entropy_thresh:
                continue
            keep.append(t)
        if not keep:
            continue
        # Decode only the kept tokens to save memory
        T_maps = [_decode_b64_map_raw(T_b64[t], h, w) for t in keep]
        S0_maps = [_decode_b64_map_raw(S0_b64[t], h, w) for t in keep]
        S1_maps = [_decode_b64_map_raw(S1_b64[t], h, w) for t in keep]
        out.append({
            "id": r.get("id"),
            "shape": (int(h), int(w)),         # NEW — Null B partner filter
            "T": T_maps, "S0": S0_maps, "S1": S1_maps,
            "n_tokens": len(keep),
        })
    return out


def _random_derangement(n: int, rng: np.random.Generator,
                        max_retries: int = 64) -> np.ndarray | None:
    """Return a random derangement of [0, n) — a permutation with no fixed
    points (perm[i] != i for all i). For n ∈ {0,1} return None.

    Retries up to `max_retries` (the probability of a random permutation
    being a derangement is ≈ 1/e ≈ 0.37, so 64 retries is essentially
    guaranteed).
    """
    if n < 2:
        return None
    for _ in range(max_retries):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
    # Fallback: cycle shift (worst case n=2, always works)
    return (np.arange(n) + 1) % n


def _pair_delta_means(T_maps, S0_partner_maps, S1_partner_maps) -> dict | None:
    """Mean per-token (Δ_js, Δ_iou) on a paired set of maps (length-aligned
    by min). Used by Null B."""
    R = min(len(T_maps), len(S0_partner_maps), len(S1_partner_maps))
    if R == 0:
        return None
    djs, diou = [], []
    for t in range(R):
        Tm = T_maps[t]
        js_s0 = _js_div(S0_partner_maps[t], Tm)
        js_s1 = _js_div(S1_partner_maps[t], Tm)
        djs.append(js_s0 - js_s1)
        iou_s0 = _iou_topk(S0_partner_maps[t], Tm)
        iou_s1 = _iou_topk(S1_partner_maps[t], Tm)
        diou.append(iou_s1 - iou_s0)
    if not djs:
        return None
    return {"delta_js": float(np.mean(djs)),
            "delta_iou": float(np.mean(diou))}


def calibrate_null_b_cross_sample(samples_with_maps: list[dict],
                                  n_iter: int = 100,
                                  alpha: float = 0.05,
                                  seed: int = 8888) -> dict:
    """Null B — cross-sample pairing.

    For each iter, partition samples by `shape = (map_h, map_w)` (Qwen-VL
    dynamic vision grid produces variable patch grids per image), then
    within each same-shape group of size ≥ 2 generate a random
    *derangement* (perm[i] != i for all i) so that T_i is paired with
    S0_{perm[i]} and S1_{perm[i]} for j ≠ i. Compute the per-sample Δ
    on the mis-paired triple, then the mean over samples.
    `δ = (1 - alpha)` percentile of `|Δ_null_mean|`.

    Cost-controlled: n_iter=100 with full reliable-token coverage by
    default; per-shape grouping is the launch-blocker shape guard.
    """
    if len(samples_with_maps) < 2:
        return {k: float("nan") for k in ("delta_js", "delta_iou")}

    # Group by (map_h, map_w) — only cross-pair within group.
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, s in enumerate(samples_with_maps):
        groups[s.get("shape", (0, 0))].append(idx)
    usable_groups = [g for g in groups.values() if len(g) >= 2]
    if not usable_groups:
        return {k: float("nan") for k in ("delta_js", "delta_iou")}

    # Diagnostic: report shape distribution + how many samples actually
    # participate in Null B.
    n_in_groups = sum(len(g) for g in usable_groups)
    n_total = len(samples_with_maps)
    if n_in_groups < n_total:
        import sys as _sys
        print(f"    [null B] {n_total - n_in_groups}/{n_total} samples "
              f"in singleton shape groups; Null B uses {n_in_groups} samples "
              f"across {len(usable_groups)} same-shape groups.",
              file=_sys.stderr)

    rng = np.random.default_rng(seed)
    null_djs, null_diou = [], []
    for _ in range(n_iter):
        per_sample_djs, per_sample_diou = [], []
        for group in usable_groups:
            n_g = len(group)
            perm_s0 = _random_derangement(n_g, rng)
            perm_s1 = _random_derangement(n_g, rng)
            if perm_s0 is None or perm_s1 is None:
                continue
            for k, i in enumerate(group):
                j_s0 = group[perm_s0[k]]
                j_s1 = group[perm_s1[k]]
                T = samples_with_maps[i]["T"]
                S0p = samples_with_maps[j_s0]["S0"]
                S1p = samples_with_maps[j_s1]["S1"]
                res = _pair_delta_means(T, S0p, S1p)
                if res is not None:
                    per_sample_djs.append(res["delta_js"])
                    per_sample_diou.append(res["delta_iou"])
        if per_sample_djs:
            null_djs.append(float(np.mean(per_sample_djs)))
            null_diou.append(float(np.mean(per_sample_diou)))
    return {
        "delta_js": float(np.quantile(np.abs(null_djs), 1 - alpha)) if null_djs else float("nan"),
        "delta_iou": float(np.quantile(np.abs(null_diou), 1 - alpha)) if null_diou else float("nan"),
        "_n_usable_samples": n_in_groups,
        "_n_singleton_excluded": n_total - n_in_groups,
        "_n_groups": len(usable_groups),
    }


def calibrate_null_c_spatial_scramble(samples_with_maps: list[dict],
                                      n_iter: int = 100,
                                      alpha: float = 0.05,
                                      seed: int = 9999) -> dict:
    """Null C — spatial scramble. For each iter, permute the patches
    within each S0 and S1 map (preserves TAM concentration, destroys
    spatial alignment with T). Recompute Δ on the scrambled S0/S1
    vs the original T."""
    if not samples_with_maps:
        return {k: float("nan") for k in ("delta_js", "delta_iou")}
    rng = np.random.default_rng(seed)
    null_djs, null_diou = [], []
    for _ in range(n_iter):
        per_sample_djs, per_sample_diou = [], []
        for s in samples_with_maps:
            R = s["n_tokens"]
            djs, diou = [], []
            for t in range(R):
                T = s["T"][t]
                S0m, S1m = s["S0"][t], s["S1"][t]
                shape = S0m.shape
                n_pix = shape[0] * shape[1]
                perm0 = rng.permutation(n_pix)
                perm1 = rng.permutation(n_pix)
                S0_scr = S0m.flatten()[perm0].reshape(shape)
                S1_scr = S1m.flatten()[perm1].reshape(shape)
                djs.append(_js_div(S0_scr, T) - _js_div(S1_scr, T))
                diou.append(_iou_topk(S1_scr, T) - _iou_topk(S0_scr, T))
            if djs:
                per_sample_djs.append(float(np.mean(djs)))
                per_sample_diou.append(float(np.mean(diou)))
        if per_sample_djs:
            null_djs.append(float(np.mean(per_sample_djs)))
            null_diou.append(float(np.mean(per_sample_diou)))
    return {
        "delta_js": float(np.quantile(np.abs(null_djs), 1 - alpha)) if null_djs else float("nan"),
        "delta_iou": float(np.quantile(np.abs(null_diou), 1 - alpha)) if null_diou else float("nan"),
    }


def calibrate_deltas(rows: list[dict],
                     n_iter_b: int = 100,
                     n_iter_c: int = 100,
                     reliability_thresh: float = RELIABILITY_THRESH_DEFAULT
                     ) -> dict:
    """Compute null δ per metric across the full (reliability-filtered)
    OPD_improved bucket. Headline δ = max(δ_A, δ_B, δ_C) per design §6.4.

    - Null A: sign-flip permutation on per-sample Δ (cheap, always runs).
    - Null B: cross-sample T↔S map pairing (loads maps; ~minutes).
    - Null C: spatial scramble of S0/S1 patches (loads maps; ~minutes).

    Returns a dict with per-metric `delta` (max over A/B/C), plus
    `delta_A`, `delta_B`, `delta_C` for diagnostic.
    """
    deltas: dict = {}
    primary_samples = []
    for r in rows:
        if r.get("bucket") != "OPD_improved":
            continue
        s = _per_sample_means(r, reliability_thresh=reliability_thresh)
        if s is None:
            continue
        primary_samples.append(s)

    # Null A — cheap, computed on the per-sample mean Δ already in hand
    null_A: dict = {}
    for k in ("delta_js", "delta_iou", "delta_cos"):
        vals = [s[k] for s in primary_samples]
        null_A[k] = calibrate_null_threshold(vals)

    # Null B and Null C — need map decoding from the primary rows
    print(f"  [delta calibration] decoding maps for {len(primary_samples)} samples...",
          file=sys.stderr)
    samples_with_maps = _load_primary_sample_maps(rows, entropy_thresh=reliability_thresh)
    print(f"  [delta calibration] running Null B (cross-sample) n_iter={n_iter_b}...",
          file=sys.stderr)
    null_B = calibrate_null_b_cross_sample(samples_with_maps, n_iter=n_iter_b)
    print(f"  [delta calibration] running Null C (spatial scramble) n_iter={n_iter_c}...",
          file=sys.stderr)
    null_C = calibrate_null_c_spatial_scramble(samples_with_maps, n_iter=n_iter_c)

    # Headline δ = max(A, B, C); Cos has no B/C (no map computation) so falls back to A
    for k in ("delta_js", "delta_iou"):
        vals = [null_A[k], null_B[k], null_C[k]]
        vals_finite = [v for v in vals if np.isfinite(v)]
        deltas[k] = max(vals_finite) if vals_finite else float("nan")
    # Cos: only A available
    deltas["delta_cos"] = null_A["delta_cos"]

    # Diagnostic breakdowns
    deltas["_null_A"] = null_A
    deltas["_null_B"] = null_B
    deltas["_null_C"] = null_C
    deltas["_n_samples_for_calibration"] = len(primary_samples)
    deltas["_n_samples_with_maps"] = len(samples_with_maps)
    return deltas


# ============================================================================
# Plotting
# ============================================================================
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_overall(overall: dict, overall_rel: dict, out_path: Path) -> None:
    plt = _setup_mpl()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    metrics = ["delta_js", "delta_iou", "delta_cos"]
    titles = [r"$\Delta$JS = JS(S0,T) − JS(S1,T)",
              r"$\Delta$IoU = IoU(S1,T) − IoU(S0,T)",
              r"$\Delta$Cos = Cos(S1,T) − Cos(S0,T)"]
    for ax, key, title in zip(axes, metrics, titles):
        means = [overall[key]["mean"], overall_rel[key]["mean"]]
        los   = [overall[key]["ci_low"], overall_rel[key]["ci_low"]]
        his   = [overall[key]["ci_high"], overall_rel[key]["ci_high"]]
        x = np.arange(2)
        errs = [[m - lo for m, lo in zip(means, los)],
                [hi - m for m, hi in zip(means, his)]]
        ax.bar(x, means, yerr=errs, capsize=6,
               color=["#3b82f6", "#10b981"],
               edgecolor="black", linewidth=0.6)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"all tokens\n(n={overall['n_samples']})",
                            f"reliability filt.\n(n={overall_rel['n_samples']})"])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Δ (positive = OPD aligned)")
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    fig.suptitle("Step 5 — overall TAM evidence alignment (cluster bootstrap 95% CI)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_per_bucket(per_bucket: dict, out_path: Path) -> None:
    plt = _setup_mpl()
    buckets = ["OPD_improved", "OPD_failed", "Teacher_advantage", "Dataset_diversity"]
    metrics = ["delta_js", "delta_iou", "delta_cos"]
    titles = [r"$\Delta$JS", r"$\Delta$IoU", r"$\Delta$Cos"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"OPD_improved": "#10b981", "OPD_failed": "#ef4444",
              "Teacher_advantage": "#3b82f6", "Dataset_diversity": "#f59e0b"}
    for ax, key, title in zip(axes, metrics, titles):
        bs = [b for b in buckets if b in per_bucket]
        means = [per_bucket[b][key]["mean"] for b in bs]
        los   = [per_bucket[b][key]["ci_low"] for b in bs]
        his   = [per_bucket[b][key]["ci_high"] for b in bs]
        ns    = [per_bucket[b]["n_samples"] for b in bs]
        x = np.arange(len(bs))
        errs = [[m - lo for m, lo in zip(means, los)],
                [hi - m for m, hi in zip(means, his)]]
        bars = ax.bar(x, means, yerr=errs, capsize=5,
                      color=[colors[b] for b in bs],
                      edgecolor="black", linewidth=0.6)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(bs, ns)],
                           rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Δ (positive = OPD aligned)")
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    fig.suptitle("Step 5 — TAM evidence alignment by sample bucket", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_per_category(per_category: dict, out_path: Path) -> None:
    plt = _setup_mpl()
    # Order: C_local first (the Step 2 causal-positive set), then others
    c_local = ["content_noun", "visual_attribute", "proper_noun"]
    other_order = ["visual_number", "answer_token", "spatial_relation",
                   "template_token", "punctuation", "pronoun", "meta_cot_token",
                   "ocr_text", "other"]
    cats = [c for c in (c_local + other_order) if c in per_category]
    metrics = ["delta_js", "delta_iou", "delta_cos"]
    titles = [r"$\Delta$JS", r"$\Delta$IoU", r"$\Delta$Cos"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    for ax, key, title in zip(axes, metrics, titles):
        means = [per_category[c][key]["mean"] for c in cats]
        los   = [per_category[c][key]["ci_low"] for c in cats]
        his   = [per_category[c][key]["ci_high"] for c in cats]
        ns    = [per_category[c]["n_samples"] for c in cats]
        x = np.arange(len(cats))
        errs = [[m - lo for m, lo in zip(means, los)],
                [hi - m for m, hi in zip(means, his)]]
        colors = ["#10b981" if c in c_local else "#94a3b8" for c in cats]
        ax.bar(x, means, yerr=errs, capsize=4,
               color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n(n={n})" for c, n in zip(cats, ns)],
                           rotation=25, ha="right", fontsize=8)
        ax.set_title(f"{title} per token_category (green = C_local)",
                     fontsize=10)
        ax.set_ylabel("Δ")
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    fig.suptitle("Step 5 — TAM evidence alignment by token category",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ============================================================================
# Qualitative picks (for tam_render_overlays.py)
# ============================================================================
def pick_qualitative(rows: list[dict], k_per_bucket: int = 3,
                     tokens_per_case: int = 4,
                     reliability_thresh: float = RELIABILITY_THRESH_DEFAULT
                     ) -> list[dict]:
    """Pick representative cases per bucket. Within each sample, pick a
    handful of tokens that (a) are C_local, (b) have valid TAM, (c)
    teacher map is reasonably concentrated."""
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_bucket[r.get("bucket", "unknown")].append(r)

    picks: list[dict] = []
    c_local = {"content_noun", "visual_attribute", "proper_noun"}
    for bucket, lst in by_bucket.items():
        # rank samples by mean ΔJS (largest first — most visually different)
        scored: list[tuple[float, dict]] = []
        for r in lst:
            s = _per_sample_means(r, reliability_thresh=reliability_thresh)
            if s is None:
                continue
            scored.append((s["delta_js"], r))
        scored.sort(key=lambda x: -abs(x[0]))   # largest |Δ| first
        for _, r in scored[:k_per_bucket]:
            R = r["response_length"]
            candidates: list[tuple[int, str, float]] = []
            for t in range(R):
                if not (r["tam_valid_T"][t] and r["tam_valid_S0"][t]
                        and r["tam_valid_S1"][t]):
                    continue
                cat = (r.get("token_category") or ["other"] * R)[t]
                if cat not in c_local:
                    continue
                ent = r["T"]["tam_entropy_norm"][t]
                if ent is None:
                    continue
                candidates.append((t, cat, ent))
            # Rank by entropy ascending (most concentrated first) and take
            # top-K. No absolute threshold — v0.1.3 long-CoT TAM has
            # entropy saturated at 0.99+, so a hardcoded 0.92 cutoff (from
            # Step 0 / short-prompt era) would always reject every token.
            # Per-sample ranking instead picks the locally-most-concentrated
            # tokens regardless of distribution shift.
            candidates.sort(key=lambda x: x[2])
            tok_indices = [t for t, _, _ in candidates[:tokens_per_case]]
            picks.append({
                "id":     r["id"],
                "bucket": bucket,
                "benchmark": r.get("benchmark"),
                "image_path":   r["image_path"],
                "question":     r.get("question"),
                "answer":       r.get("answer"),
                "response_text": r["response_text"],
                "s0_correct":   r.get("s0_correct"),
                "s1_correct":   r.get("s1_correct"),
                "tok_indices":  tok_indices,
                "tok_texts":    [r["tokens"][i] for i in tok_indices] if tok_indices else [],
                "tok_categories": [(r.get("token_category") or [])[i] for i in tok_indices],
            })
    return picks


# ============================================================================
# CSV / Markdown writers
# ============================================================================
def write_overall_csv(overall: dict, overall_rel: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("filter,n_samples,metric,mean,ci_low,ci_high,sd,mde_95\n")
        for filt, o in [("all_tokens", overall), ("reliability_filt", overall_rel)]:
            for k in ("delta_js", "delta_iou", "delta_cos"):
                b = o[k]
                f.write(f"{filt},{o['n_samples']},{k},"
                        f"{b['mean']:.6f},{b['ci_low']:.6f},{b['ci_high']:.6f},"
                        f"{b['sd']:.6f},{b['mde_95']:.6f}\n")


def write_bucket_csv(per_bucket: dict, per_bucket_rel: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("bucket,filter,n_samples,metric,mean,ci_low,ci_high,sd,mde_95\n")
        for filt, src in [("all_tokens", per_bucket),
                          ("reliability_filt", per_bucket_rel)]:
            for b, bk in src.items():
                for k in ("delta_js", "delta_iou", "delta_cos"):
                    mb = bk[k]
                    f.write(f"{b},{filt},{bk['n_samples']},{k},"
                            f"{mb['mean']:.6f},{mb['ci_low']:.6f},{mb['ci_high']:.6f},"
                            f"{mb['sd']:.6f},{mb['mde_95']:.6f}\n")


def write_category_csv(per_category: dict, per_category_rel: dict,
                       path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("category,filter,n_samples,n_tokens,metric,mean,ci_low,ci_high,sd,mde_95\n")
        for filt, src in [("all_tokens", per_category),
                          ("reliability_filt", per_category_rel)]:
            for c, bk in src.items():
                for k in ("delta_js", "delta_iou", "delta_cos"):
                    mb = bk[k]
                    f.write(f"{c},{filt},{bk['n_samples']},{bk['n_tokens_total']},"
                            f"{k},{mb['mean']:.6f},{mb['ci_low']:.6f},"
                            f"{mb['ci_high']:.6f},{mb['sd']:.6f},{mb['mde_95']:.6f}\n")


def _write_primary_category_csv(per_category_primary_rel: dict,
                                path: Path) -> None:
    """OPD_improved × reliability per_category breakdown — the actual
    branch (c) decision input. Separate from per_category.csv to avoid
    confusion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("category,n_samples,n_tokens,metric,mean,ci_low,ci_high,sd,mde_95\n")
        for c, bk in per_category_primary_rel.items():
            for k in ("delta_js", "delta_iou", "delta_cos"):
                mb = bk[k]
                f.write(f"{c},{bk['n_samples']},{bk['n_tokens_total']},"
                        f"{k},{mb['mean']:.6f},{mb['ci_low']:.6f},"
                        f"{mb['ci_high']:.6f},{mb['sd']:.6f},{mb['mde_95']:.6f}\n")


def write_calibration_csv(deltas: dict, teacher_reliability: tuple[float, int, int],
                          path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    null_A = deltas.get("_null_A", {})
    null_B = deltas.get("_null_B", {})
    null_C = deltas.get("_null_C", {})
    with path.open("w") as f:
        f.write("metric,null_A,null_B,null_C,delta_headline\n")
        for k in ("delta_js", "delta_iou", "delta_cos"):
            f.write(f"{k},"
                    f"{null_A.get(k, float('nan')):.6f},"
                    f"{null_B.get(k, float('nan')):.6f},"
                    f"{null_C.get(k, float('nan')):.6f},"
                    f"{deltas.get(k, float('nan')):.6f}\n")
        f.write(f"_n_samples_for_calibration,{deltas.get('_n_samples_for_calibration', 0)},,,\n")
        f.write(f"_n_samples_with_maps,{deltas.get('_n_samples_with_maps', 0)},,,\n")
        rel_rate, n_rel, n_tot = teacher_reliability
        f.write(f"teacher_reliability_rate,{rel_rate:.4f},,,\n")
        f.write(f"teacher_reliable_tokens,{n_rel},,,\n")
        f.write(f"teacher_total_tokens,{n_tot},,,\n")
        f.write(f"random_iou_baseline_top20,{RANDOM_IOU_BASELINE_TOP20:.4f},,,\n")


def compute_teacher_reliability_rate(rows: list[dict],
                                     bucket: str = "OPD_improved",
                                     thresh: float = RELIABILITY_THRESH_DEFAULT) -> tuple[float, int, int]:
    """Fraction of OPD_improved tokens where teacher map satisfies
    tam_entropy_norm_T < thresh. Used to trigger §8 branch (d)."""
    n_total = 0
    n_reliable = 0
    for r in rows:
        if r.get("bucket") != bucket:
            continue
        ent = r.get("T", {}).get("tam_entropy_norm", []) or []
        valid = r.get("tam_valid_T", []) or []
        for t in range(min(len(ent), len(valid))):
            if not valid[t] or ent[t] is None:
                continue
            n_total += 1
            if ent[t] < thresh:
                n_reliable += 1
    if n_total == 0:
        return float("nan"), 0, 0
    return n_reliable / n_total, n_reliable, n_total


def decide_branch(per_bucket_rel: dict,
                  per_category_rel: dict,
                  deltas: dict,
                  teacher_reliability: tuple[float, int, int],
                  reliability_thresh: float = RELIABILITY_THRESH_DEFAULT,
                  ) -> dict:
    """Step 5 §8 decision via TOST on the PRIMARY cell
    (OPD_improved bucket × reliability-filtered).

    Returns a dict:
        {
          'branch':         'aligned' | 'flat' | 'split' | 'unreliable' | 'inconclusive',
          'label':          '(a) ...' (human-readable),
          'metric_status':  {'delta_js': 'aligned'|'equivalent'|..., 'delta_iou': ..., 'delta_cos': ...},
          'reasoning':      '...',
          'used_cell':      'OPD_improved x reliability',
          'deltas':         {δ_JS, δ_IoU, δ_Cos},
          'teacher_reliability_rate': ...,
        }
    """
    rel_rate, n_rel, n_tot = teacher_reliability

    # (d) — teacher TAM unreliable on this prompt distribution.
    # Recalibrated for v0.1.3 long-CoT regime (smoke 2026-05-28): entropy
    # is saturated at 0.99+ across the entire token distribution, so the
    # original 50% pass-rate cutoff (designed for v0.1.2 [0.85, 0.99]
    # bi-modal regime) became a design artifact that trips on healthy
    # data. New trigger: only (d) if either the absolute count of
    # reliable tokens is below a usable-statistics floor (30) OR the
    # rate falls below a true-saturation floor (20% — i.e. <1/5 of
    # tokens have any concentration, way below what we saw in smoke).
    MIN_RELIABLE_TOKENS = 30
    RATE_FLOOR = 0.20
    if n_tot > 0 and (n_rel < MIN_RELIABLE_TOKENS or rel_rate < RATE_FLOOR):
        return {
            "branch": "unreliable",
            "label": ("(d) Teacher TAM unreliable on OPD_improved "
                      "(rate={:.2f}, {}/{}).".format(rel_rate, n_rel, n_tot)),
            "metric_status": {},
            "reasoning": (
                f"Reliable-token floor not met: n_reliable={n_rel} < "
                f"{MIN_RELIABLE_TOKENS} OR rate={rel_rate:.3f} < {RATE_FLOOR}. "
                f"Threshold = entropy_norm_T < {reliability_thresh:.3f}. "
                f"TAM signal too diffuse to ground EA-OPD; main §Method = "
                f"T2-2/v3 only."
            ),
            "used_cell": "OPD_improved x reliability",
            "deltas": deltas,
            "teacher_reliability_rate": rel_rate,
            "min_reliable_tokens_required": MIN_RELIABLE_TOKENS,
            "rate_floor": RATE_FLOOR,
        }

    primary = per_bucket_rel.get("OPD_improved")
    if primary is None or primary.get("n_samples", 0) == 0:
        return {
            "branch": "inconclusive",
            "label": "(inconclusive) OPD_improved bucket empty after reliability filter.",
            "metric_status": {},
            "reasoning": "No usable samples in the primary decision cell.",
            "used_cell": "OPD_improved x reliability",
            "deltas": deltas,
            "teacher_reliability_rate": rel_rate,
        }

    # Min-n guard per GPT review of 6a081aa — bootstrap CI / TOST are
    # unstable below ~10 samples; refuse to drive the decision tree on
    # a tiny primary cell.
    MIN_PRIMARY_N = 10
    if primary["n_samples"] < MIN_PRIMARY_N:
        return {
            "branch": "inconclusive",
            "label": (f"(e) underpowered — OPD_improved × reliability "
                      f"n={primary['n_samples']} < {MIN_PRIMARY_N}."),
            "metric_status": {},
            "reasoning": (f"Primary cell has only {primary['n_samples']} samples; "
                          f"TOST / bootstrap unreliable below n={MIN_PRIMARY_N}. "
                          "Extend the candidate pool or accept the audit cannot decide."),
            "used_cell": "OPD_improved x reliability",
            "deltas": deltas,
            "teacher_reliability_rate": rel_rate,
            "min_primary_n_required": MIN_PRIMARY_N,
        }

    # TOST per metric
    metric_status: dict = {}
    for k in ("delta_js", "delta_iou", "delta_cos"):
        m = primary[k]
        d = deltas.get(k, float("nan"))
        metric_status[k] = tost_equivalence(m["mean"], m["ci_low"], m["ci_high"], d)

    js_st = metric_status["delta_js"]
    iou_st = metric_status["delta_iou"]
    cos_st = metric_status["delta_cos"]

    # (a) aligned — JS AND IoU both aligned (drop raw-cos confirmation per
    # GPT review of c2066f5: raw cosine on non-mean-centered maps is
    # prone to false positives from globally-active maps).
    if js_st == "aligned" and iou_st == "aligned":
        return {
            "branch": "aligned",
            "label": "(a) OPD already implicitly aligns visual evidence — drop EA-OPD line.",
            "metric_status": metric_status,
            "reasoning": "Both ΔJS and ΔIoU aligned (mean > δ AND ci_low > 0).",
            "used_cell": "OPD_improved x reliability",
            "deltas": deltas,
            "teacher_reliability_rate": rel_rate,
        }

    # (c) split — alignment on easy categories but equivalence on C_local
    easy = [c for c in ("template_token", "answer_token", "punctuation")
            if c in per_category_rel]
    c_local = [c for c in C_LOCAL_CATEGORIES if c in per_category_rel]
    if easy and c_local:
        easy_aligned = all(
            tost_equivalence(per_category_rel[c]["delta_js"]["mean"],
                             per_category_rel[c]["delta_js"]["ci_low"],
                             per_category_rel[c]["delta_js"]["ci_high"],
                             deltas.get("delta_js", float("nan"))) == "aligned"
            for c in easy
        )
        c_local_equiv = all(
            tost_equivalence(per_category_rel[c]["delta_js"]["mean"],
                             per_category_rel[c]["delta_js"]["ci_low"],
                             per_category_rel[c]["delta_js"]["ci_high"],
                             deltas.get("delta_js", float("nan"))) == "equivalent"
            for c in c_local
        )
        if easy_aligned and c_local_equiv:
            return {
                "branch": "split",
                "label": "(c) OPD aligns easy tokens (template/answer/punct) but not C_local — both methods coexist.",
                "metric_status": metric_status,
                "reasoning": "TOST: aligned on easy categories, equivalent on C_local.",
                "used_cell": "OPD_improved x reliability (per-category)",
                "deltas": deltas,
                "teacher_reliability_rate": rel_rate,
            }

    # (b) flat (TOST equivalence + powered enough)
    mde_ok = primary["delta_js"]["mde_95"] <= deltas.get("delta_js", float("inf"))
    if js_st == "equivalent" and mde_ok:
        return {
            "branch": "flat",
            "label": "(b) OPD trains output but NOT evidence — strong EA-OPD motivation.",
            "metric_status": metric_status,
            "reasoning": ("TOST passes on ΔJS (CI ⊆ [-δ, +δ]) AND audit is "
                          "powered enough (MDE_95 ≤ δ_JS)."),
            "used_cell": "OPD_improved x reliability",
            "deltas": deltas,
            "teacher_reliability_rate": rel_rate,
        }

    # (e) inconclusive / underpowered
    return {
        "branch": "inconclusive",
        "label": "(e) Audit inconclusive — neither aligned nor TOST-equivalent (or underpowered).",
        "metric_status": metric_status,
        "reasoning": (f"ΔJS: {js_st}; MDE_95={primary['delta_js']['mde_95']:.4f} "
                      f"vs δ_JS={deltas.get('delta_js', float('nan')):.4f}. "
                      "Increase n or accept the audit cannot decide."),
        "used_cell": "OPD_improved x reliability",
        "deltas": deltas,
        "teacher_reliability_rate": rel_rate,
    }


def write_results_md(overall: dict, overall_rel: dict,
                     per_bucket: dict, per_bucket_rel: dict,
                     per_category: dict, per_category_rel: dict,
                     per_category_primary_rel: dict,
                     picks: list[dict], decision: dict,
                     deltas: dict, teacher_reliability: tuple[float, int, int],
                     out_path: Path,
                     reliability_thresh: float = RELIABILITY_THRESH_DEFAULT) -> None:
    def _fmt(d):
        return (f"{d['mean']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
                f"  sd={d['sd']:.4f}")

    rel_rate, n_rel, n_tot = teacher_reliability
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("# Step 5 — TAM Evidence Alignment results\n\n")
        f.write(f"Generated by `mllmopd.analysis.tam_step5_analyzer` "
                f"(commit={os.environ.get('MLLMOPD_CODE_COMMIT', 'unknown')})\n\n")

        # ----- Decision header (banner at top) -----
        f.write("## 0. Decision\n\n")
        f.write(f"**Branch:** `{decision['branch']}`  \n")
        f.write(f"**Label:** {decision['label']}  \n")
        f.write(f"**Decision cell:** {decision['used_cell']}  \n")
        f.write(f"**Reasoning:** {decision['reasoning']}\n\n")
        if decision.get("metric_status"):
            f.write("**Per-metric TOST status:**  \n")
            for k, v in decision["metric_status"].items():
                f.write(f"- `{k}`: **{v}**  \n")
            f.write("\n")
        f.write("(See §8 of `docs/step5-evidence-alignment-design.md` for the "
                "full decision tree.)\n\n")

        # ----- Null calibration + teacher reliability -----
        null_A = deltas.get("_null_A", {})
        null_B = deltas.get("_null_B", {})
        null_C = deltas.get("_null_C", {})
        f.write("## 1. Null calibration + teacher TAM reliability\n\n")
        f.write(f"Headline null thresholds δ = max(δ_A, δ_B, δ_C) per design §6.4. "
                f"n_samples for calibration = "
                f"{deltas.get('_n_samples_for_calibration', 0)} on OPD_improved × reliability "
                f"({deltas.get('_n_samples_with_maps', 0)} of those had decoded maps for "
                f"Null B/C).\n\n")
        f.write("| metric | δ_A (sign-flip) | δ_B (cross-sample) | δ_C (spatial scramble) | **δ_headline (max)** |\n")
        f.write("|---|---|---|---|---|\n")
        for k in ("delta_js", "delta_iou", "delta_cos"):
            f.write(f"| {k} | "
                    f"{null_A.get(k, float('nan')):.4f} | "
                    f"{null_B.get(k, float('nan')):.4f} | "
                    f"{null_C.get(k, float('nan')):.4f} | "
                    f"**{deltas.get(k, float('nan')):.4f}** |\n")
        f.write("\n(Null B/C are not computed for `delta_cos` because the map-level "
                "decomposition is only meaningful for JS/IoU — Cos δ falls back to Null A.)\n\n")
        f.write(f"Teacher TAM reliability rate on `OPD_improved` "
                f"(entropy_norm_T < {reliability_thresh:.3f}): **{rel_rate:.3f}** "
                f"({n_rel}/{n_tot} tokens).\n\n")
        f.write(f"Random IoU baseline (top-20%, independent patch sets): "
                f"**{RANDOM_IOU_BASELINE_TOP20:.4f}**. Chance-correct IoU "
                f"by subtracting this when comparing absolute IoU(S0,T) / IoU(S1,T).\n\n")

        # ----- Overall -----
        f.write("## 2. Overall (paired ΔS1−S0 vs Teacher, cluster bootstrap 95% CI)\n\n")
        f.write("| Filter | n_samples | ΔJS | ΔIoU | ΔCos |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| all tokens | {overall['n_samples']} | "
                f"{_fmt(overall['delta_js'])} | "
                f"{_fmt(overall['delta_iou'])} | "
                f"{_fmt(overall['delta_cos'])} |\n")
        f.write(f"| reliability (entropy_norm_T < {reliability_thresh:.3f}) — decision input | {overall_rel['n_samples']} | "
                f"{_fmt(overall_rel['delta_js'])} | "
                f"{_fmt(overall_rel['delta_iou'])} | "
                f"{_fmt(overall_rel['delta_cos'])} |\n\n")

        # ----- By bucket (with reliability variant)  -----
        f.write("## 3. By bucket — reliability-filtered drives §8 decision; "
                "all-tokens row reported for sanity\n\n")
        f.write("| Bucket | Filter | n | ΔJS | ΔIoU | ΔCos |\n")
        f.write("|---|---|---|---|---|---|\n")
        for b in ("OPD_improved", "OPD_failed", "Teacher_advantage", "Dataset_diversity"):
            for filt_name, src in (("reliability", per_bucket_rel),
                                   ("all_tokens", per_bucket)):
                if b not in src:
                    continue
                bk = src[b]
                marker = " **(decision cell)**" if (b == "OPD_improved" and filt_name == "reliability") else ""
                f.write(f"| {b}{marker} | {filt_name} | {bk['n_samples']} | "
                        f"{_fmt(bk['delta_js'])} | {_fmt(bk['delta_iou'])} | "
                        f"{_fmt(bk['delta_cos'])} |\n")
        f.write("\n")

        # ----- By category (reliability-filtered shown first) -----
        # ----- Branch-(c) decision input: OPD_improved × reliability × category -----
        f.write("## 4. By token category — **OPD_improved × reliability** "
                "(branch (c) decision input)\n\n")
        f.write("| Category | n_samples | n_tokens | ΔJS | ΔIoU | ΔCos |\n")
        f.write("|---|---|---|---|---|---|\n")
        for c in ALL_TOKEN_CATEGORIES:
            if c not in per_category_primary_rel:
                continue
            bk = per_category_primary_rel[c]
            marker = " (C_local)" if c in C_LOCAL_CATEGORIES else ""
            f.write(f"| {c}{marker} | {bk['n_samples']} | {bk['n_tokens_total']} | "
                    f"{_fmt(bk['delta_js'])} | {_fmt(bk['delta_iou'])} | "
                    f"{_fmt(bk['delta_cos'])} |\n")
        f.write("\n")

        f.write("### 4b. By token category — all-buckets × reliability (exploratory)\n\n")
        f.write("| Category | n_samples | n_tokens | ΔJS | ΔIoU | ΔCos |\n")
        f.write("|---|---|---|---|---|---|\n")
        for c in ALL_TOKEN_CATEGORIES:
            if c not in per_category_rel:
                continue
            bk = per_category_rel[c]
            marker = " (C_local)" if c in C_LOCAL_CATEGORIES else ""
            f.write(f"| {c}{marker} | {bk['n_samples']} | {bk['n_tokens_total']} | "
                    f"{_fmt(bk['delta_js'])} | {_fmt(bk['delta_iou'])} | "
                    f"{_fmt(bk['delta_cos'])} |\n")
        f.write("\n")
        f.write("### 4c. By token category — all-buckets × all-tokens (exploratory)\n\n")
        f.write("| Category | n_samples | n_tokens | ΔJS | ΔIoU | ΔCos |\n")
        f.write("|---|---|---|---|---|---|\n")
        for c in ALL_TOKEN_CATEGORIES:
            if c not in per_category:
                continue
            bk = per_category[c]
            marker = " (C_local)" if c in C_LOCAL_CATEGORIES else ""
            f.write(f"| {c}{marker} | {bk['n_samples']} | {bk['n_tokens_total']} | "
                    f"{_fmt(bk['delta_js'])} | {_fmt(bk['delta_iou'])} | "
                    f"{_fmt(bk['delta_cos'])} |\n")
        f.write("\n")

        # ----- Qualitative -----
        f.write("## 5. Qualitative picks (input for `tam_render_overlays.py`)\n\n")
        f.write(f"{len(picks)} cases selected (~3 per bucket, "
                "ranked by |ΔJS|, C_local tokens with concentrated teacher TAM).\n\n")
        f.write("See `qualitative_cases.jsonl` for the picker output.\n")


# ============================================================================
# Main
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alignment", type=Path, required=True,
                    help="alignment.jsonl from tam_step5_evidence_alignment")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("docs/figures/step5/"))
    ap.add_argument("--reliability-thresh", type=float,
                    default=RELIABILITY_THRESH_DEFAULT,
                    help=(f"Token-level reliability filter: keep tokens where "
                          f"`tam_entropy_norm_T < THRESH`. Default "
                          f"{RELIABILITY_THRESH_DEFAULT} — calibrated for v0.1.3 "
                          f"TAM on long-CoT 7B teacher (smoke 2026-05-28: "
                          f"min=0.9936, p25=0.9964, p75=0.9982). Step 0/2 used "
                          f"0.95 on v0.1.2 + short prompt — NOT applicable here."))
    args = ap.parse_args(argv)
    THRESH = args.reliability_thresh
    print(f">>> reliability_thresh = {THRESH:.4f} "
          f"(tokens kept iff entropy_norm_T < THRESH)", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tables").mkdir(parents=True, exist_ok=True)

    rows = load_alignment(args.alignment)
    print(f">>> loaded {len(rows)} alignment rows from {args.alignment}",
          file=sys.stderr)
    if not rows:
        sys.exit("!! no rows — abort")

    # --- Aggregate (both filters, all axes) ---
    print(">>> computing overall (all tokens)", file=sys.stderr)
    overall = aggregate_overall(rows, reliability=False, reliability_thresh=THRESH)
    print(">>> computing overall (reliability filter)", file=sys.stderr)
    overall_rel = aggregate_overall(rows, reliability=True, reliability_thresh=THRESH)
    print(">>> per-bucket (both filters)", file=sys.stderr)
    per_bucket = aggregate_per_bucket(rows, reliability=False, reliability_thresh=THRESH)
    per_bucket_rel = aggregate_per_bucket(rows, reliability=True, reliability_thresh=THRESH)
    print(">>> per-category (both filters)", file=sys.stderr)
    per_category = aggregate_per_category(rows, reliability=False, reliability_thresh=THRESH)
    per_category_rel = aggregate_per_category(rows, reliability=True, reliability_thresh=THRESH)

    # --- Null calibration on the PRIMARY cell (max of A/B/C per §6.4) ---
    print(">>> calibrating null thresholds δ on OPD_improved × reliability",
          file=sys.stderr)
    deltas = calibrate_deltas(rows, reliability_thresh=THRESH)
    teacher_reliability = compute_teacher_reliability_rate(
        rows, bucket="OPD_improved", thresh=THRESH,
    )
    print(f"    δ_JS={deltas.get('delta_js', float('nan')):.4f}  "
          f"δ_IoU={deltas.get('delta_iou', float('nan')):.4f}  "
          f"δ_Cos={deltas.get('delta_cos', float('nan')):.4f}  "
          f"reliability_rate={teacher_reliability[0]:.3f}",
          file=sys.stderr)
    if "_null_A" in deltas:
        print(f"    null breakdown (JS):  "
              f"A={deltas['_null_A'].get('delta_js', float('nan')):.4f}  "
              f"B={deltas['_null_B'].get('delta_js', float('nan')):.4f}  "
              f"C={deltas['_null_C'].get('delta_js', float('nan')):.4f}",
              file=sys.stderr)

    # Per-category aggregation restricted to OPD_improved rows for the
    # decision tree (fixes the branch-(c) scope bug noted in GPT review
    # of c2066f5). The all-rows per_category_rel is still kept for the
    # report's exploratory table.
    primary_rows = [r for r in rows if r.get("bucket") == "OPD_improved"]
    per_category_primary_rel = aggregate_per_category(
        primary_rows, reliability=True, reliability_thresh=THRESH,
    )

    # --- Tables ---
    write_overall_csv(overall, overall_rel,
                      args.out_dir / "tables" / "overall.csv")
    write_bucket_csv(per_bucket, per_bucket_rel,
                     args.out_dir / "tables" / "per_bucket.csv")
    write_category_csv(per_category, per_category_rel,
                       args.out_dir / "tables" / "per_category.csv")
    write_calibration_csv(deltas, teacher_reliability,
                          args.out_dir / "tables" / "null_calibration.csv")

    # --- Figures (drive from primary = reliability-filtered) ---
    print(">>> rendering fig1 / fig2 / fig3", file=sys.stderr)
    plot_overall(overall, overall_rel,
                 args.out_dir / "fig1_alignment_delta_overall.png")
    plot_per_bucket(per_bucket_rel,
                    args.out_dir / "fig2_per_bucket.png")
    plot_per_category(per_category_rel,
                      args.out_dir / "fig3_per_token_category.png")

    # --- Qualitative picks ---
    picks = pick_qualitative(rows, k_per_bucket=3, tokens_per_case=4,
                             reliability_thresh=THRESH)
    with (args.out_dir / "qualitative_cases.jsonl").open("w") as f:
        for p in picks:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # --- Decision via TOST on the primary cell. branch (c) per-category
    #     comparison uses OPD_improved-only aggregation (per design §8). ---
    decision = decide_branch(per_bucket_rel, per_category_primary_rel,
                             deltas, teacher_reliability,
                             reliability_thresh=THRESH)
    write_results_md(overall, overall_rel,
                     per_bucket, per_bucket_rel,
                     per_category, per_category_rel,
                     per_category_primary_rel,
                     picks, decision,
                     deltas, teacher_reliability,
                     args.out_dir / "step5-results.md",
                     reliability_thresh=THRESH)
    # Sidecar CSV with the primary cell category breakdown (branch (c) input)
    _write_primary_category_csv(per_category_primary_rel,
                                args.out_dir / "tables" / "per_category_primary.csv")

    # Also emit the decision dict as JSON for downstream tooling
    with (args.out_dir / "decision.json").open("w") as f:
        json.dump(decision, f, indent=2, default=str)

    print(f"\n>>> outputs at {args.out_dir}/", file=sys.stderr)
    print(f"    branch = {decision['branch']}  ({decision['label']})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
