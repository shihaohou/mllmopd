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
# Cluster bootstrap CI
# ============================================================================
def cluster_bootstrap_ci(per_sample_values: list[float], n_iter: int = 10000,
                         alpha: float = 0.05, seed: int = 1729) -> tuple[float, float, float]:
    """Returns (mean, low, high) at the (1-alpha) bootstrap CI."""
    if not per_sample_values:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    arr = np.asarray(per_sample_values, dtype=np.float64)
    n = len(arr)
    means = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(arr.mean()), float(low), float(high)


# ============================================================================
# Aggregations
# ============================================================================
def aggregate_overall(rows: list[dict], reliability: bool = False) -> dict:
    """Overall paired Δ — no stratification."""
    thresh = 0.95 if reliability else None
    per_sample = [_per_sample_means(r, reliability_thresh=thresh) for r in rows]
    per_sample = [s for s in per_sample if s is not None]
    out: dict = {"n_samples": len(per_sample)}
    for k in ("delta_js", "delta_iou", "delta_cos"):
        vals = [s[k] for s in per_sample]
        m, lo, hi = cluster_bootstrap_ci(vals)
        out[k] = {"mean": m, "ci_low": lo, "ci_high": hi}
    # Also report raw mean JS / IoU / Cos for S0 vs S1
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
        bk: dict = {"n_samples": len(lst)}
        for k in ("delta_js", "delta_iou", "delta_cos"):
            vals = [s[k] for s in lst]
            m, lo, hi = cluster_bootstrap_ci(vals)
            bk[k] = {"mean": m, "ci_low": lo, "ci_high": hi}
        out[bucket] = bk
    return out


def aggregate_per_category(rows: list[dict], reliability: bool = False,
                           min_samples: int = 10) -> dict:
    """Per token_category: filter tokens by category at sample level, then
    cluster-bootstrap across samples that contributed any tokens to that
    category."""
    thresh = 0.95 if reliability else None
    cats = ["content_noun", "visual_attribute", "proper_noun", "visual_number",
            "answer_token", "template_token", "pronoun", "spatial_relation",
            "meta_cot_token", "punctuation", "ocr_text", "other"]
    out: dict = {}
    for cat in cats:
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
        }
        for k in ("delta_js", "delta_iou", "delta_cos"):
            vals = [s[k] for s in per_sample]
            m, lo, hi = cluster_bootstrap_ci(vals)
            bk[k] = {"mean": m, "ci_low": lo, "ci_high": hi}
        out[cat] = bk
    return out


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
        f.write("filter,n_samples,metric,mean,ci_low,ci_high\n")
        for filt, o in [("all_tokens", overall), ("reliability_filt", overall_rel)]:
            for k in ("delta_js", "delta_iou", "delta_cos"):
                f.write(f"{filt},{o['n_samples']},{k},"
                        f"{o[k]['mean']:.6f},{o[k]['ci_low']:.6f},{o[k]['ci_high']:.6f}\n")


def write_bucket_csv(per_bucket: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("bucket,n_samples,metric,mean,ci_low,ci_high\n")
        for b, bk in per_bucket.items():
            for k in ("delta_js", "delta_iou", "delta_cos"):
                f.write(f"{b},{bk['n_samples']},{k},"
                        f"{bk[k]['mean']:.6f},{bk[k]['ci_low']:.6f},{bk[k]['ci_high']:.6f}\n")


def write_category_csv(per_category: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("category,n_samples,n_tokens,metric,mean,ci_low,ci_high\n")
        for c, bk in per_category.items():
            for k in ("delta_js", "delta_iou", "delta_cos"):
                f.write(f"{c},{bk['n_samples']},{bk['n_tokens_total']},{k},"
                        f"{bk[k]['mean']:.6f},{bk[k]['ci_low']:.6f},{bk[k]['ci_high']:.6f}\n")


def decide_branch(overall: dict, per_category: dict) -> str:
    """Return the §6 decision-tree branch label."""
    djs = overall["delta_js"]
    # (d) reliability check
    if overall["n_samples"] == 0:
        return "(d) TAM signal unreliable (zero samples passed reliability filter)"
    # (a) strong positive
    if djs["mean"] > 0.05 and djs["ci_low"] > 0:
        return "(a) OPD already implicitly aligns visual evidence — drop EA-OPD line."
    # (c) split: positive on easy, ~0 on hard
    hard_cats = [c for c in ("content_noun", "visual_attribute") if c in per_category]
    easy_cats = [c for c in ("template_token", "punctuation", "answer_token") if c in per_category]
    if hard_cats and easy_cats:
        hard_mean = float(np.mean([per_category[c]["delta_js"]["mean"] for c in hard_cats]))
        easy_mean = float(np.mean([per_category[c]["delta_js"]["mean"] for c in easy_cats]))
        if easy_mean > 0.05 and abs(hard_mean) < 0.03:
            return "(c) Partial — OPD aligns easy tokens (template/answer) but not hard (content_noun/visual_attribute). Both methods coexist."
    # (b) flat overall
    if abs(djs["mean"]) < 0.03 and djs["ci_low"] < 0 < djs["ci_high"]:
        return "(b) ΔJS ≈ 0 — vanilla OPD trains output but not evidence. Strong EA-OPD motivation."
    return "(mixed) See per-bucket / per-category tables for direction."


def write_results_md(overall: dict, overall_rel: dict,
                     per_bucket: dict, per_category: dict,
                     picks: list[dict], decision: str,
                     out_path: Path) -> None:
    def _fmt(d):
        return f"{d['mean']:+.4f} [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("# Step 5 — TAM Evidence Alignment results\n\n")
        f.write(f"Generated by `mllmopd.analysis.tam_step5_analyzer` "
                f"(commit={os.environ.get('MLLMOPD_CODE_COMMIT', 'unknown')})\n\n")

        f.write("## 1. Overall (paired ΔS1−S0 vs Teacher, cluster bootstrap 95% CI)\n\n")
        f.write("| Filter | n_samples | ΔJS | ΔIoU | ΔCos |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| all tokens | {overall['n_samples']} | "
                f"{_fmt(overall['delta_js'])} | "
                f"{_fmt(overall['delta_iou'])} | "
                f"{_fmt(overall['delta_cos'])} |\n")
        f.write(f"| reliability (entropy_norm_T < 0.95) | {overall_rel['n_samples']} | "
                f"{_fmt(overall_rel['delta_js'])} | "
                f"{_fmt(overall_rel['delta_iou'])} | "
                f"{_fmt(overall_rel['delta_cos'])} |\n\n")

        f.write("## 2. By bucket\n\n")
        f.write("| Bucket | n | ΔJS | ΔIoU | ΔCos |\n")
        f.write("|---|---|---|---|---|\n")
        for b in ("OPD_improved", "OPD_failed", "Teacher_advantage", "Dataset_diversity"):
            if b not in per_bucket:
                continue
            bk = per_bucket[b]
            f.write(f"| {b} | {bk['n_samples']} | "
                    f"{_fmt(bk['delta_js'])} | {_fmt(bk['delta_iou'])} | "
                    f"{_fmt(bk['delta_cos'])} |\n")
        f.write("\n")

        f.write("## 3. By token category\n\n")
        f.write("| Category | n_samples | n_tokens | ΔJS | ΔIoU | ΔCos |\n")
        f.write("|---|---|---|---|---|---|\n")
        c_local = ("content_noun", "visual_attribute", "proper_noun")
        for c in c_local + ("visual_number", "answer_token", "spatial_relation",
                            "template_token", "punctuation", "pronoun",
                            "meta_cot_token", "ocr_text", "other"):
            if c not in per_category:
                continue
            bk = per_category[c]
            marker = " (C_local)" if c in c_local else ""
            f.write(f"| {c}{marker} | {bk['n_samples']} | {bk['n_tokens_total']} | "
                    f"{_fmt(bk['delta_js'])} | {_fmt(bk['delta_iou'])} | "
                    f"{_fmt(bk['delta_cos'])} |\n")
        f.write("\n")

        f.write("## 4. Qualitative picks (input for `tam_render_overlays.py`)\n\n")
        f.write(f"{len(picks)} cases selected (~3 per bucket, "
                "ranked by |ΔJS|, C_local tokens with concentrated teacher TAM).\n\n")
        f.write("See `qualitative_cases.jsonl` for the picker output.\n\n")

        f.write("## 5. Decision\n\n")
        f.write(f"**Branch:** {decision}\n\n")
        f.write("(See §6 of `docs/step5-evidence-alignment-design.md` for the "
                "full decision tree.)\n")


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

    # --- Aggregate ---
    print(">>> computing overall (all tokens)", file=sys.stderr)
    overall = aggregate_overall(rows, reliability=False)
    print(">>> computing overall (reliability filter)", file=sys.stderr)
    overall_rel = aggregate_overall(rows, reliability=True)
    print(">>> per-bucket", file=sys.stderr)
    per_bucket = aggregate_per_bucket(rows, reliability=False)
    print(">>> per-category", file=sys.stderr)
    per_category = aggregate_per_category(rows, reliability=False)

    # --- Tables ---
    write_overall_csv(overall, overall_rel, args.out_dir / "tables" / "overall.csv")
    write_bucket_csv(per_bucket, args.out_dir / "tables" / "per_bucket.csv")
    write_category_csv(per_category, args.out_dir / "tables" / "per_category.csv")

    # --- Figures ---
    print(">>> rendering fig1 / fig2 / fig3", file=sys.stderr)
    plot_overall(overall, overall_rel,
                 args.out_dir / "fig1_alignment_delta_overall.png")
    plot_per_bucket(per_bucket,
                    args.out_dir / "fig2_per_bucket.png")
    plot_per_category(per_category,
                      args.out_dir / "fig3_per_token_category.png")

    # --- Qualitative picks ---
    picks = pick_qualitative(rows, k_per_bucket=3, tokens_per_case=4)
    with (args.out_dir / "qualitative_cases.jsonl").open("w") as f:
        for p in picks:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # --- Results doc + decision ---
    decision = decide_branch(overall_rel, per_category)
    write_results_md(overall, overall_rel, per_bucket, per_category, picks,
                     decision, args.out_dir / "step5-results.md")

    print(f"\n>>> outputs at {args.out_dir}/", file=sys.stderr)
    print(f"    decision branch: {decision}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
