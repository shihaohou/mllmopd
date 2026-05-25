"""TAM-VD calibration analyzer (v0.1.2 schema-aware).

Reads `tam_sanity.jsonl` (Step 0) or `tam_calibration.jsonl` (Step 1
1a/1b) per `docs/tam_calibration_schema.md` v0.1.2.

Auto-detects mode by checking whether `vd[]` / `adv[]` / `quad[]` are
present:

- **Step 0 mode** (vd absent): reports TAM/attention scalar distributions
  per `token_category`, TAM-vs-attention head-to-head, peak/COM center-
  bias diagnostics, QC stats. No correlations / AUC / AP computable.

- **Step 1 mode** (vd / adv present): full calibration table —
  corr(`tam_mass_top20`, `|vd|`) and corr(`attention_baseline_mass_top20`,
  `|vd|`) per `token_category` per `(student_ckpt, response_source)`,
  ROC-AUC for high-`|vd|` detection, **AP / PR-AUC for the rare
  `quad==3` visual-rejection class**, per-quadrant breakdowns, by-VD-
  norm-bin breakdowns (reusing `t2_1_energy_audit.VD_NORM_BINS`),
  bootstrap CI **by sample** (default 1000 resamples).

Usage::

    python -m mllmopd.analysis.tam_calibration \\
        --jsonl runs/audit/tam_sanity_<TS>/tam_sanity.jsonl \\
        --out-json runs/analysis/tam_calibration_v0.json \\
        [--out-txt  runs/analysis/tam_calibration_v0.txt] \\
        [--bootstrap-n 1000]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Reuse the canonical 4-quadrant labels from t2_1_energy_audit so the
# downstream interpretation stays consistent with our existing T2-1 audit.
try:
    from mllmopd.analysis.t2_1_energy_audit import QUADRANTS, VD_NORM_BINS
except ImportError:                                                  # pragma: no cover
    # Fallback if run outside the package — duplicated definitions.
    QUADRANTS = [
        ("vis_support_agree",            "vd>=0 ∧ adv>=0"),
        ("vis_support_pushed_away",      "vd>=0 ∧ adv<0"),
        ("vis_reject_teacher_toward",    "vd<0 ∧ adv>=0"),
        ("vis_reject_correction",        "vd<0 ∧ adv<0"),
    ]
    VD_NORM_BINS = [
        (0.0, 0.20, "vd_norm_p00_p20"),
        (0.20, 0.40, "vd_norm_p20_p40"),
        (0.40, 0.60, "vd_norm_p40_p60"),
        (0.60, 0.80, "vd_norm_p60_p80"),
        (0.80, 1.00, "vd_norm_p80_p100"),
    ]


# ============================================================================
# Loading / flattening
# ============================================================================
def load_rows(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get(row: dict, key: str, R: int, default=None) -> list:
    """Pull a per-token array of length R; pad/clip to R if needed."""
    arr = row.get(key)
    if arr is None:
        return [default] * R
    if len(arr) < R:
        return list(arr) + [default] * (R - len(arr))
    return list(arr[:R])


def flatten_tokens(rows: list[dict]) -> list[dict]:
    """One record per token, with sample-level fields denormalized.

    Skips rows where `tam_valid` is false (their per-token arrays are
    unreliable). Tokens with `is_template_token` / `is_special_token` are
    kept but typically excluded by analyzers; we keep them for QC."""
    out: list[dict] = []
    for row in rows:
        if not row.get("tam_valid"):
            continue
        R = int(row.get("response_length", 0))
        if R <= 0:
            continue
        sample_id = row.get("id")
        student_ckpt = row.get("student_ckpt") or "step0_no_student"
        response_source = row.get("response_source", "teacher_greedy")

        lp_full       = _get(row, "lp_full", R)
        lp_blank      = _get(row, "lp_blank", R)
        vd            = _get(row, "vd", R)
        student_lp    = _get(row, "student_lp", R)
        adv           = _get(row, "adv", R)
        quad          = _get(row, "quad", R)

        tam_top10     = _get(row, "tam_mass_top10", R)
        tam_top20     = _get(row, "tam_mass_top20", R)
        tam_top40     = _get(row, "tam_mass_top40", R)
        # v0.1.1 back-compat: schema had single `tam_mass` (top-20%)
        if all(v is None for v in tam_top20) and row.get("tam_mass"):
            tam_top20 = list(row["tam_mass"][:R]) + [None] * max(0, R - len(row["tam_mass"]))
        tam_H_norm    = _get(row, "tam_entropy_norm", R)
        tam_peak_xy   = _get(row, "tam_peak_xy", R)
        tam_com_xy    = _get(row, "tam_center_of_mass_xy", R)

        attn_top20    = _get(row, "attention_baseline_mass_top20", R)
        attn_H_norm   = _get(row, "attention_baseline_entropy_norm", R)

        category      = _get(row, "token_category", R, default="other")
        tokens        = _get(row, "tokens", R, default="")
        teacher_H     = _get(row, "teacher_entropy_full", R)
        teacher_marg  = _get(row, "teacher_top1_margin_full", R)

        is_answer     = _get(row, "is_answer_token", R, default=False)
        is_think      = _get(row, "is_think_token", R, default=False)
        is_blankness  = _get(row, "is_blankness_token", R, default=False)

        for t in range(R):
            abs_vd = abs(vd[t]) if vd[t] is not None else None
            abs_adv = abs(adv[t]) if adv[t] is not None else None
            out.append({
                "sample_id": sample_id,
                "student_ckpt": student_ckpt,
                "response_source": response_source,
                "ckpt_source": f"{student_ckpt}@{response_source}",
                "token_idx": t,
                "token": tokens[t],
                "token_category": category[t] or "other",
                "is_answer_token": bool(is_answer[t]),
                "is_think_token":  bool(is_think[t]),
                "is_blankness_token": bool(is_blankness[t]),
                "lp_full":  lp_full[t],
                "lp_blank": lp_blank[t],
                "vd":       vd[t],
                "abs_vd":   abs_vd,
                "student_lp": student_lp[t],
                "adv":      adv[t],
                "abs_adv":  abs_adv,
                "quad":     quad[t],
                "tam_mass_top10": tam_top10[t],
                "tam_mass_top20": tam_top20[t],
                "tam_mass_top40": tam_top40[t],
                "tam_entropy_norm": tam_H_norm[t],
                "tam_peak_xy":  tam_peak_xy[t],
                "tam_com_xy":   tam_com_xy[t],
                "attn_mass_top20":   attn_top20[t],
                "attn_entropy_norm": attn_H_norm[t],
                "teacher_entropy": teacher_H[t],
                "teacher_margin":  teacher_marg[t],
            })
    return out


# ============================================================================
# Statistics helpers (stdlib-only — no scipy / numpy required)
# ============================================================================
def _pearson(xs: list, ys: list) -> float | None:
    """Pearson r over paired non-None values; None if < 3 pairs or zero variance."""
    pairs = [(x, y) for x, y in zip(xs, ys)
             if x is not None and y is not None
             and not (isinstance(x, float) and math.isnan(x))
             and not (isinstance(y, float) and math.isnan(y))]
    if len(pairs) < 3:
        return None
    n = len(pairs)
    mx = sum(p[0] for p in pairs) / n
    my = sum(p[1] for p in pairs) / n
    num = sum((p[0] - mx) * (p[1] - my) for p in pairs)
    dx = sum((p[0] - mx) ** 2 for p in pairs)
    dy = sum((p[1] - my) ** 2 for p in pairs)
    if dx <= 0 or dy <= 0:
        return None
    return num / (math.sqrt(dx) * math.sqrt(dy))


def _roc_auc(scores: list, labels: list) -> float | None:
    """ROC AUC via concordance counting. labels ∈ {0,1}, scores arbitrary."""
    pairs = [(s, l) for s, l in zip(scores, labels)
             if s is not None and l is not None]
    if not pairs:
        return None
    pos = [s for s, l in pairs if l]
    neg = [s for s, l in pairs if not l]
    if not pos or not neg:
        return None
    concordant = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                concordant += 1
            elif sp == sn:
                concordant += 0.5
    return concordant / (len(pos) * len(neg))


def _average_precision(scores: list, labels: list) -> float | None:
    """Average Precision (PR-AUC). Robust to rare positives."""
    pairs = [(s, l) for s, l in zip(scores, labels)
             if s is not None and l is not None]
    if not pairs:
        return None
    pairs.sort(key=lambda p: -p[0])  # sort by score desc
    n_pos = sum(1 for _, l in pairs if l)
    if n_pos == 0:
        return None
    tp = 0
    fp = 0
    last_recall = 0.0
    ap = 0.0
    for s, l in pairs:
        if l:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        if recall > last_recall:
            ap += precision * (recall - last_recall)
            last_recall = recall
    return ap


def _dist_summary(vals: list) -> dict:
    """Return {n, mean, p5, p50, p95, min, max} for a numeric vector."""
    vs = sorted(v for v in vals
                if v is not None
                and not (isinstance(v, float) and math.isnan(v)))
    n = len(vs)
    if n == 0:
        return {"n": 0}
    return {
        "n":   n,
        "mean": sum(vs) / n,
        "p5":  vs[max(0, (n * 5) // 100)],
        "p50": vs[n // 2],
        "p95": vs[min(n - 1, (n * 95) // 100)],
        "min": vs[0],
        "max": vs[-1],
    }


def _quad_label_from_int(q):
    if q is None or not isinstance(q, int) or q < 0 or q >= 4:
        return None
    return QUADRANTS[q][0]


# ============================================================================
# Per-bucket statistics
# ============================================================================
def _bucket_stats(records: list[dict]) -> dict:
    """Compute the analyzer-output block for one bucket of token records.
    The bucket can be: pooled, per (ckpt,source), per token_category, etc."""
    n_tokens = len(records)
    n_samples = len({r["sample_id"] for r in records})

    tam20      = [r["tam_mass_top20"] for r in records]
    tam10      = [r["tam_mass_top10"] for r in records]
    tam40      = [r["tam_mass_top40"] for r in records]
    tam_Hn     = [r["tam_entropy_norm"] for r in records]
    attn20     = [r["attn_mass_top20"] for r in records]
    attn_Hn    = [r["attn_entropy_norm"] for r in records]
    abs_vd     = [r["abs_vd"] for r in records]
    abs_adv    = [r["abs_adv"] for r in records]
    quads      = [r["quad"] for r in records]
    teacher_H  = [r["teacher_entropy"] for r in records]
    teacher_M  = [r["teacher_margin"] for r in records]

    block: dict = {
        "n_tokens": n_tokens,
        "n_samples": n_samples,
        "dist": {
            "tam_mass_top20":            _dist_summary(tam20),
            "tam_mass_top10":            _dist_summary(tam10),
            "tam_mass_top40":            _dist_summary(tam40),
            "tam_entropy_norm":          _dist_summary(tam_Hn),
            "attention_baseline_top20":  _dist_summary(attn20),
            "attention_baseline_Hnorm":  _dist_summary(attn_Hn),
            "teacher_entropy_full":      _dist_summary(teacher_H),
            "teacher_top1_margin_full":  _dist_summary(teacher_M),
        },
        # TAM-vs-attention head-to-head (works on Step 0 too)
        "corr_tam20_attn20": _pearson(tam20, attn20),
    }
    # Center-bias diagnostics: mean peak xy over all valid peaks in bucket
    peaks = [r["tam_peak_xy"] for r in records
             if isinstance(r.get("tam_peak_xy"), list)]
    if peaks:
        xs = [p[0] for p in peaks if len(p) >= 2]
        ys = [p[1] for p in peaks if len(p) >= 2]
        if xs and ys:
            block["tam_peak_xy_mean"] = [sum(xs) / len(xs), sum(ys) / len(ys)]
            block["tam_peak_xy_var"]  = [
                sum((x - block["tam_peak_xy_mean"][0]) ** 2 for x in xs) / len(xs),
                sum((y - block["tam_peak_xy_mean"][1]) ** 2 for y in ys) / len(ys),
            ]

    # Step 1 metrics — only if abs_vd is available
    has_vd = any(v is not None for v in abs_vd)
    if has_vd:
        block["corr_tam20_abs_vd"]  = _pearson(tam20, abs_vd)
        block["corr_attn20_abs_vd"] = _pearson(attn20, abs_vd)
        # ΔTAM-vs-attn (positive = TAM beats attention on this bucket)
        if block["corr_tam20_abs_vd"] is not None and block["corr_attn20_abs_vd"] is not None:
            block["tam_beats_attn_delta"] = (
                block["corr_tam20_abs_vd"] - block["corr_attn20_abs_vd"]
            )

        # High-|VD| token detection (binary: top quartile)
        vd_valid = [v for v in abs_vd if v is not None]
        if len(vd_valid) >= 10:
            tau = sorted(vd_valid)[(len(vd_valid) * 75) // 100]
            high_vd_label = [
                (1 if (v is not None and v >= tau) else 0 if v is not None else None)
                for v in abs_vd
            ]
            block["auc_tam20_high_abs_vd"]   = _roc_auc(tam20, high_vd_label)
            block["auc_attn20_high_abs_vd"]  = _roc_auc(attn20, high_vd_label)
            block["high_abs_vd_threshold"]   = tau

        # Visual-rejection class (quad==3) as binary
        rej_label = [(1 if q == 3 else 0 if q is not None else None) for q in quads]
        n_rej = sum(1 for x in rej_label if x == 1)
        block["visual_rejection_n"]          = n_rej
        block["visual_rejection_prevalence"] = (
            n_rej / max(1, sum(1 for x in rej_label if x is not None))
        )
        if n_rej >= 5:
            block["auc_tam20_visual_rejection"]  = _roc_auc(tam20, rej_label)
            block["auc_attn20_visual_rejection"] = _roc_auc(attn20, rej_label)
            block["ap_tam20_visual_rejection"]   = _average_precision(tam20, rej_label)
            block["ap_attn20_visual_rejection"]  = _average_precision(attn20, rej_label)
    return block


# ============================================================================
# Bootstrap CI by sample
# ============================================================================
def _bootstrap_ci_by_sample(
    records: list[dict],
    metric_fn,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    rng_seed: int = 1729,
) -> dict | None:
    """Bootstrap CI at sample granularity (NOT token-level — long responses
    must not get extra weight). metric_fn takes records-list and returns a
    float-or-None scalar."""
    sample_ids = sorted({r["sample_id"] for r in records})
    if len(sample_ids) < 5:
        return None
    by_sample: dict[str, list] = defaultdict(list)
    for r in records:
        by_sample[r["sample_id"]].append(r)

    rng = random.Random(rng_seed)
    boot_vals = []
    for _ in range(n_resamples):
        chosen = rng.choices(sample_ids, k=len(sample_ids))
        resampled: list[dict] = []
        for sid in chosen:
            resampled.extend(by_sample[sid])
        v = metric_fn(resampled)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            boot_vals.append(v)

    if not boot_vals:
        return None
    boot_vals.sort()
    n = len(boot_vals)
    return {
        "mean":  sum(boot_vals) / n,
        "lo":    boot_vals[max(0, int(n * (alpha / 2)))],
        "hi":    boot_vals[min(n - 1, int(n * (1 - alpha / 2)))],
        "n_resamples_kept": n,
    }


# ============================================================================
# Headline tables
# ============================================================================
ANALYZED_CATEGORIES = [
    "content_noun", "proper_noun", "visual_number", "visual_attribute",
    "spatial_relation", "ocr_text",
    "answer_token", "pronoun", "meta_cot_token",
    "template_token", "special_token", "punctuation", "other",
]


def compute_per_ckpt_source(
    records: list[dict],
    bootstrap_n: int = 1000,
) -> dict:
    """Group by (student_ckpt, response_source) → token_category → stats."""
    out: dict = {}
    by_ckptsrc: dict[str, list] = defaultdict(list)
    for r in records:
        by_ckptsrc[r["ckpt_source"]].append(r)

    for key, recs in by_ckptsrc.items():
        ckpt_block: dict = {
            "n_tokens": len(recs),
            "n_samples": len({r["sample_id"] for r in recs}),
            "category_histogram": dict(Counter(r["token_category"] for r in recs)),
            "quad_token_counts":  dict(Counter(r["quad"] for r in recs if r["quad"] is not None)),
            "pooled":             _bucket_stats(recs),
            "by_token_category":  {},
        }
        if bootstrap_n > 0:
            ckpt_block["bootstrap_pooled_corr_tam20_abs_vd"] = _bootstrap_ci_by_sample(
                recs,
                lambda rs: _pearson([r["tam_mass_top20"] for r in rs],
                                    [r["abs_vd"] for r in rs]),
                n_resamples=bootstrap_n,
            )
            ckpt_block["bootstrap_pooled_corr_attn20_abs_vd"] = _bootstrap_ci_by_sample(
                recs,
                lambda rs: _pearson([r["attn_mass_top20"] for r in rs],
                                    [r["abs_vd"] for r in rs]),
                n_resamples=bootstrap_n,
            )
        for cat in ANALYZED_CATEGORIES:
            cat_recs = [r for r in recs if r["token_category"] == cat]
            if not cat_recs:
                continue
            ckpt_block["by_token_category"][cat] = _bucket_stats(cat_recs)

        # Per-quadrant breakdown (Step 1 only — needs vd & adv)
        quad_counts = ckpt_block["quad_token_counts"]
        if quad_counts:
            ckpt_block["by_quadrant"] = {}
            for q_idx, (q_key, _) in enumerate(QUADRANTS):
                q_recs = [r for r in recs if r["quad"] == q_idx]
                if not q_recs:
                    continue
                ckpt_block["by_quadrant"][q_key] = _bucket_stats(q_recs)

        out[key] = ckpt_block
    return out


def compute_qc(records: list[dict], rows: list[dict]) -> dict:
    """QC block — sanity checks across the run."""
    n_rows = len(rows)
    n_tam_failed = sum(1 for r in rows if not r.get("tam_valid"))
    n_attn_failed = sum(1 for r in rows if r.get("attn_baseline_valid") is False)
    cat_counts = Counter(r["token_category"] for r in records)
    return {
        "n_rows":                    n_rows,
        "n_tam_failed":              n_tam_failed,
        "n_attn_baseline_failed":    n_attn_failed,
        "n_tokens_total":            len(records),
        "category_histogram_global": dict(cat_counts),
        "pos_tagger":         _first_meta(rows, "pos_tagger"),
        "tam_preproc_version": _first_meta(rows, "tam_preproc_version"),
        "code_commit_run":    _first_meta(rows, "code_commit_run"),
        "attn_failure_reasons_seen": sorted({
            r.get("attn_baseline_failure_reason") or ""
            for r in rows if r.get("attn_baseline_failure_reason")
        }),
    }


def _first_meta(rows: list[dict], key: str):
    for r in rows:
        if key in r:
            return r[key]
    return None


# ============================================================================
# Mode detection + main
# ============================================================================
def detect_mode(rows: list[dict]) -> str:
    for r in rows:
        if not r.get("tam_valid"):
            continue
        vd = r.get("vd")
        if vd and any(v is not None for v in vd):
            return "step1"
    return "step0"


def emit_summary_txt(report: dict, out_txt: Path) -> None:
    lines: list[str] = []
    lines.append(f"# TAM calibration report  (mode={report['mode']})")
    qc = report["qc"]
    lines.append(f"n_rows = {qc['n_rows']}  tokens_total = {qc['n_tokens_total']}  "
                 f"tam_failed = {qc['n_tam_failed']}  attn_failed = {qc['n_attn_baseline_failed']}")
    lines.append(f"tam_preproc_version = {qc.get('tam_preproc_version')!r}  "
                 f"pos_tagger = {qc.get('pos_tagger')!r}  "
                 f"code_commit_run = {qc.get('code_commit_run')!r}")
    if qc.get("attn_failure_reasons_seen"):
        lines.append(f"attn failure reasons: {qc['attn_failure_reasons_seen']}")

    cat_hist = qc.get("category_histogram_global", {})
    if cat_hist:
        lines.append("\nGlobal token-category histogram:")
        for cat, n in sorted(cat_hist.items(), key=lambda x: -x[1]):
            lines.append(f"  {cat:20s} {n}")

    for key, block in report["per_ckpt_source"].items():
        lines.append(f"\n## {key}")
        lines.append(f"  n_tokens={block['n_tokens']}  n_samples={block['n_samples']}")
        pool = block["pooled"]
        lines.append(f"  pooled tam20.mean = {_fmt(pool['dist']['tam_mass_top20'].get('mean'))}  "
                     f"attn20.mean = {_fmt(pool['dist']['attention_baseline_top20'].get('mean'))}  "
                     f"corr_tam_attn = {_fmt(pool.get('corr_tam20_attn20'))}")
        if pool.get("corr_tam20_abs_vd") is not None:
            lines.append(f"  pooled corr(tam20, |vd|) = {_fmt(pool['corr_tam20_abs_vd'])}  "
                         f"corr(attn20, |vd|) = {_fmt(pool['corr_attn20_abs_vd'])}  "
                         f"Δ_TAM-attn = {_fmt(pool.get('tam_beats_attn_delta'))}")
        if pool.get("ap_tam20_visual_rejection") is not None:
            lines.append(f"  visual-rejection AP: tam={_fmt(pool['ap_tam20_visual_rejection'])}  "
                         f"attn={_fmt(pool['ap_attn20_visual_rejection'])}  "
                         f"(n_quad3={pool['visual_rejection_n']}, prevalence={_fmt(pool['visual_rejection_prevalence'])})")

        lines.append("  by token_category:")
        lines.append(f"    {'category':<22} {'n':>5}  "
                     f"{'tam20_mean':>10} {'attn20_mean':>11}  "
                     f"{'corr_tam':>8} {'corr_attn':>9} {'Δ':>7}")
        for cat in ANALYZED_CATEGORIES:
            cat_block = block["by_token_category"].get(cat)
            if not cat_block:
                continue
            tm = cat_block["dist"]["tam_mass_top20"].get("mean")
            am = cat_block["dist"]["attention_baseline_top20"].get("mean")
            ct = cat_block.get("corr_tam20_abs_vd")
            ca = cat_block.get("corr_attn20_abs_vd")
            dt = cat_block.get("tam_beats_attn_delta")
            lines.append(f"    {cat:<22} {cat_block['n_tokens']:>5d}  "
                         f"{_fmt(tm, 10)} {_fmt(am, 11)}  "
                         f"{_fmt(ct, 8)} {_fmt(ca, 9)} {_fmt(dt, 7)}")

        if block.get("by_quadrant"):
            lines.append("  by quadrant (vd × adv):")
            for q_key, q_block in block["by_quadrant"].items():
                tm = q_block["dist"]["tam_mass_top20"].get("mean")
                am = q_block["dist"]["attention_baseline_top20"].get("mean")
                lines.append(f"    {q_key:<32} n={q_block['n_tokens']:>5d}  "
                             f"tam20={_fmt(tm)}  attn20={_fmt(am)}")
    out_txt.write_text("\n".join(lines) + "\n")
    print(f">>> wrote summary: {out_txt}", file=sys.stderr)


def _fmt(x, width: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        s = "  NA"
    else:
        s = f"{x:+.3f}" if abs(x) < 100 else f"{x:.3f}"
    if width:
        s = f"{s:>{width}s}"
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jsonl", type=Path, required=True,
                    help="tam_sanity.jsonl (Step 0) or tam_calibration.jsonl (Step 1)")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-txt", type=Path, default=None,
                    help="Optional human-readable summary table.")
    ap.add_argument("--bootstrap-n", type=int, default=1000,
                    help="Bootstrap resamples for sample-level CI (Step 1 only). "
                         "0 to disable.")
    ap.add_argument("--mode", choices=["auto", "step0", "step1"], default="auto",
                    help="Force a specific mode (default: auto-detect from data).")
    args = ap.parse_args(argv)

    rows = load_rows(args.jsonl)
    if not rows:
        print(f"!! no rows in {args.jsonl}", file=sys.stderr)
        return 1
    mode = args.mode if args.mode != "auto" else detect_mode(rows)
    records = flatten_tokens(rows)
    qc = compute_qc(records, rows)
    per_ckpt_source = compute_per_ckpt_source(records, bootstrap_n=args.bootstrap_n)

    report = {
        "schema_version":   "v0.1.2",
        "mode":             mode,
        "jsonl":            str(args.jsonl),
        "qc":               qc,
        "per_ckpt_source":  per_ckpt_source,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f">>> wrote {args.out_json}", file=sys.stderr)

    if args.out_txt:
        args.out_txt.parent.mkdir(parents=True, exist_ok=True)
        emit_summary_txt(report, args.out_txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
