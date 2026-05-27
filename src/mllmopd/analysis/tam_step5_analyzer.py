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
                      reliability_thresh: float | None = 0.95
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
    m, lo, hi, sd = cluster_bootstrap_ci(vals)
    return {"mean": m, "ci_low": lo, "ci_high": hi, "sd": sd,
            "mde_95": 1.96 * sd if np.isfinite(sd) else float("nan")}


def aggregate_overall(rows: list[dict], reliability: bool = False) -> dict:
    """Overall paired Δ — no stratification."""
    thresh = 0.95 if reliability else None
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


def aggregate_per_bucket(rows: list[dict], reliability: bool = False) -> dict:
    thresh = 0.95 if reliability else None
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
                           min_samples: int = 10) -> dict:
    """Per token_category: filter tokens by category at sample level, then
    cluster-bootstrap across samples that contributed any tokens to that
    category."""
    thresh = 0.95 if reliability else None
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


def calibrate_deltas(rows: list[dict]) -> dict:
    """Compute the null δ per metric across the full (reliability-filtered)
    OPD_improved bucket. This is the primary decision threshold per §6.4."""
    deltas: dict = {}
    primary_samples = []
    for r in rows:
        if r.get("bucket") != "OPD_improved":
            continue
        s = _per_sample_means(r, reliability_thresh=0.95)
        if s is None:
            continue
        primary_samples.append(s)
    for k in ("delta_js", "delta_iou", "delta_cos"):
        vals = [s[k] for s in primary_samples]
        deltas[k] = calibrate_null_threshold(vals)
    deltas["_n_samples_for_calibration"] = len(primary_samples)
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
                     tokens_per_case: int = 4) -> list[dict]:
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
            s = _per_sample_means(r, reliability_thresh=0.95)
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
                if ent is None or ent > 0.92:
                    continue
                candidates.append((t, cat, ent))
            candidates.sort(key=lambda x: x[2])    # most concentrated first
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


def write_calibration_csv(deltas: dict, teacher_reliability: tuple[float, int, int],
                          path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("metric,null_delta\n")
        for k in ("delta_js", "delta_iou", "delta_cos"):
            f.write(f"{k},{deltas.get(k, float('nan')):.6f}\n")
        f.write(f"_n_samples_for_calibration,{deltas.get('_n_samples_for_calibration', 0)}\n")
        rel_rate, n_rel, n_tot = teacher_reliability
        f.write(f"teacher_reliability_rate,{rel_rate:.4f}\n")
        f.write(f"teacher_reliable_tokens,{n_rel}\n")
        f.write(f"teacher_total_tokens,{n_tot}\n")
        f.write(f"random_iou_baseline_top20,{RANDOM_IOU_BASELINE_TOP20:.4f}\n")


def compute_teacher_reliability_rate(rows: list[dict],
                                     bucket: str = "OPD_improved",
                                     thresh: float = 0.95) -> tuple[float, int, int]:
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
                  teacher_reliability: tuple[float, int, int]) -> dict:
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

    # (d) — teacher TAM unreliable on this prompt distribution
    if n_tot > 0 and rel_rate < 0.50:
        return {
            "branch": "unreliable",
            "label": ("(d) Teacher TAM unreliable on OPD_improved "
                      "(rate={:.2f}, {}/{}).".format(rel_rate, n_rel, n_tot)),
            "metric_status": {},
            "reasoning": ("Less than 50% of OPD_improved tokens have a "
                          "concentrated teacher map (entropy_norm_T < 0.95). "
                          "Cannot ground EA-OPD; main §Method = T2-2/v3 only."),
            "used_cell": "OPD_improved x reliability",
            "deltas": deltas,
            "teacher_reliability_rate": rel_rate,
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

    # TOST per metric
    metric_status: dict = {}
    for k in ("delta_js", "delta_iou", "delta_cos"):
        m = primary[k]
        d = deltas.get(k, float("nan"))
        metric_status[k] = tost_equivalence(m["mean"], m["ci_low"], m["ci_high"], d)

    js_st = metric_status["delta_js"]
    iou_st = metric_status["delta_iou"]
    cos_st = metric_status["delta_cos"]

    # (a) aligned — JS strongly positive, with confirmation from at least one secondary
    if js_st == "aligned" and (iou_st == "aligned" or cos_st == "aligned"):
        return {
            "branch": "aligned",
            "label": "(a) OPD already implicitly aligns visual evidence — drop EA-OPD line.",
            "metric_status": metric_status,
            "reasoning": "ΔJS > δ_JS with ci_low > 0 AND at least one of "
                         "ΔIoU/ΔCos in the same direction.",
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
                     picks: list[dict], decision: dict,
                     deltas: dict, teacher_reliability: tuple[float, int, int],
                     out_path: Path) -> None:
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
        f.write("## 1. Null calibration + teacher TAM reliability\n\n")
        f.write(f"Null thresholds δ (sign-flip permutation, 95th percentile of |Δ_null|; "
                f"n_samples for calibration = {deltas.get('_n_samples_for_calibration', 0)} on OPD_improved × reliability):\n\n")
        f.write("| metric | δ_null |\n|---|---|\n")
        for k in ("delta_js", "delta_iou", "delta_cos"):
            f.write(f"| {k} | {deltas.get(k, float('nan')):.4f} |\n")
        f.write(f"\nTeacher TAM reliability rate on `OPD_improved` "
                f"(entropy_norm_T < 0.95): **{rel_rate:.3f}** "
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
        f.write(f"| reliability (entropy_norm_T < 0.95) **PRIMARY** | {overall_rel['n_samples']} | "
                f"{_fmt(overall_rel['delta_js'])} | "
                f"{_fmt(overall_rel['delta_iou'])} | "
                f"{_fmt(overall_rel['delta_cos'])} |\n\n")

        # ----- By bucket (with reliability variant)  -----
        f.write("## 3. By bucket (reliability-filtered PRIMARY, all-tokens for sanity)\n\n")
        f.write("| Bucket | Filter | n | ΔJS | ΔIoU | ΔCos |\n")
        f.write("|---|---|---|---|---|---|\n")
        for b in ("OPD_improved", "OPD_failed", "Teacher_advantage", "Dataset_diversity"):
            for filt_name, src in (("reliability", per_bucket_rel),
                                   ("all_tokens", per_bucket)):
                if b not in src:
                    continue
                bk = src[b]
                marker = " **PRIMARY**" if (b == "OPD_improved" and filt_name == "reliability") else ""
                f.write(f"| {b}{marker} | {filt_name} | {bk['n_samples']} | "
                        f"{_fmt(bk['delta_js'])} | {_fmt(bk['delta_iou'])} | "
                        f"{_fmt(bk['delta_cos'])} |\n")
        f.write("\n")

        # ----- By category (reliability-filtered shown first) -----
        f.write("## 4. By token category — reliability-filtered (decision-driving)\n\n")
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
        f.write("### 4b. By token category — all tokens (exploratory)\n\n")
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
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tables").mkdir(parents=True, exist_ok=True)

    rows = load_alignment(args.alignment)
    print(f">>> loaded {len(rows)} alignment rows from {args.alignment}",
          file=sys.stderr)
    if not rows:
        sys.exit("!! no rows — abort")

    # --- Aggregate (both filters, all axes) ---
    print(">>> computing overall (all tokens)", file=sys.stderr)
    overall = aggregate_overall(rows, reliability=False)
    print(">>> computing overall (reliability filter)", file=sys.stderr)
    overall_rel = aggregate_overall(rows, reliability=True)
    print(">>> per-bucket (both filters)", file=sys.stderr)
    per_bucket = aggregate_per_bucket(rows, reliability=False)
    per_bucket_rel = aggregate_per_bucket(rows, reliability=True)
    print(">>> per-category (both filters)", file=sys.stderr)
    per_category = aggregate_per_category(rows, reliability=False)
    per_category_rel = aggregate_per_category(rows, reliability=True)

    # --- Null calibration on the PRIMARY cell ---
    print(">>> calibrating null thresholds δ on OPD_improved × reliability",
          file=sys.stderr)
    deltas = calibrate_deltas(rows)
    teacher_reliability = compute_teacher_reliability_rate(
        rows, bucket="OPD_improved", thresh=0.95,
    )
    print(f"    δ_JS={deltas.get('delta_js', float('nan')):.4f}  "
          f"δ_IoU={deltas.get('delta_iou', float('nan')):.4f}  "
          f"δ_Cos={deltas.get('delta_cos', float('nan')):.4f}  "
          f"reliability_rate={teacher_reliability[0]:.3f}",
          file=sys.stderr)

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
    picks = pick_qualitative(rows, k_per_bucket=3, tokens_per_case=4)
    with (args.out_dir / "qualitative_cases.jsonl").open("w") as f:
        for p in picks:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # --- Decision via TOST on the primary cell ---
    decision = decide_branch(per_bucket_rel, per_category_rel,
                             deltas, teacher_reliability)
    write_results_md(overall, overall_rel,
                     per_bucket, per_bucket_rel,
                     per_category, per_category_rel,
                     picks, decision,
                     deltas, teacher_reliability,
                     args.out_dir / "step5-results.md")

    # Also emit the decision dict as JSON for downstream tooling
    with (args.out_dir / "decision.json").open("w") as f:
        json.dump(decision, f, indent=2, default=str)

    print(f"\n>>> outputs at {args.out_dir}/", file=sys.stderr)
    print(f"    branch = {decision['branch']}  ({decision['label']})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
