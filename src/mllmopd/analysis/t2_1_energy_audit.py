"""T2-1 energy audit: distinguish LR confound from signed-proxy failure.

GPT round-3 recommended an "A0: no-train energy audit" as the first
post-T2-1 diagnostic. This module reads T2-1's already-collected
per-step diagnostics JSONL.gz (lp_full, lp_blank, vd, vd_weights,
old_lp_student) and computes three decision metrics from the existing
data — NO retraining required.

Decision metrics
----------------

* **rho_l2** = sqrt(Σ_t (w_t·adv_t)^2 / Σ_t adv_t^2). Tests the
  effective-LR-halving hypothesis. PGPO Eq 7's sum-preserving renorm
  (Σ w_t = N) does NOT preserve advantage L2 energy; it can drop.
  rho_l2 ≈ 0.6 fully explains T2-1's step-1 grad_norm ratio (~60% of
  T1-2) as a mass-vs-energy mismatch artifact, not a method failure.

* **corr(w_t, |adv_t|)** — Pearson correlation per-sequence and
  pooled. If negative, weights amplify tokens where the OPD signal
  is small AND suppress tokens where it's large — the wrong
  reweighting target for OPD.

* **frac_supp_neg_vd_neg_adv** = Σ |adv_t|·𝟙[vd_t<0 ∧ adv_t<0 ∧ w_t<1] /
  Σ |adv_t|. GPT's signed-proxy-failure mechanism: when the teacher
  pushes the student AWAY from a token (adv<0) and the image is why
  (vd<0), the token is a "visual rejection" — PGPO's non-negative KL
  proxy would treat it as high-VD, but our signed `lp_full-lp_blank`
  proxy assigns it to the suppress branch. If frac is large, the
  proxy is fundamentally mis-targeting visual-rejection correction.

Also computes |adv| means per VD-norm quartile bin (anchored at the
PGPO τ=0.4 boundary) and per-step trajectories of all three metrics.

adv_t = lp_full[t] - old_lp_student[t], matching the Uni-OPD OPD loss
"adv_t = (lp_teacher_full - lp_student) · sentinel_mask · w_t" minus
the multiplier we're auditing.

old_lp_student source: the existing diag hook writes
`old_lp_student` BEFORE the trainer forward pass, so it's empty on
real production runs. To recover it, the patched runs (with
MLLMOPD_DUMP_OPD_ADV=1, P19 patch in loss.py) also write a sidecar
`step_NNNNNN.adv_dp{R}.jsonl.gz` carrying (sample_index, old_log_probs)
per sample. This module AUTO-DISCOVERS those sidecars by scanning the
same diag_dir for `step_NNNNNN.adv_dp*.jsonl.gz` and joining on
sample_index. The diag-side `old_lp_student` is preferred if present
and non-empty; falls back to the sidecar otherwise.

Sentinel handling: skip samples where lp_full or vd is absent /
length-mismatched. Also skip samples where vd_weights are all unit
(no signal to audit) or where old_log_probs cannot be recovered.
Counts are reported.

Usage::

    python -m mllmopd.analysis.t2_1_energy_audit \\
        --diag-dir ${MLLMOPD_RUNS}/t2_1_v0_T2_1_full_vd/diagnostics \\
        [--out-json runs/analysis/t2_1_energy_audit.json] \\
        [--steps 1,49,99,149,199,230]
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

STEP_RE = re.compile(r"step_(\d+)\.jsonl\.gz$")
SIDECAR_RE = re.compile(r"step_(\d+)\.adv_dp(\d+)\.jsonl\.gz$")

_EPS = 1e-12

# VD-norm quartile boundaries. We anchor at τ=0.4 (PGPO suppress/boost
# split) and put extra resolution around it. Bins are (lo, hi] except
# the first which is [0.0, hi].
VD_NORM_BINS: list[tuple[float, float, str]] = [
    (0.0, 0.20, "vd_norm_p00_p20"),
    (0.20, 0.40, "vd_norm_p20_p40"),  # PGPO suppress branch
    (0.40, 0.60, "vd_norm_p40_p60"),  # PGPO boost branch (low)
    (0.60, 0.80, "vd_norm_p60_p80"),
    (0.80, 1.00, "vd_norm_p80_p100"),
]

# 4-quadrant partition of (sign(vd), sign(adv)) per GPT round-4. Tokens
# with vd==0 or adv==0 are assigned by sign(...) >= 0 convention to keep
# the table exhaustive.
QUADRANTS: list[tuple[str, str]] = [
    # (key, human-readable description)
    ("vis_support_agree",
     "vd>=0 ∧ adv>=0: image supports current token AND teacher reinforces"),
    ("vis_support_teacher_pushes_away",
     "vd>=0 ∧ adv<0: image supports current token BUT teacher pushes student away"),
    ("vis_reject_teacher_pushes_toward",
     "vd<0 ∧ adv>=0: image rejects current token but teacher pushes student toward it"),
    ("vis_reject_correction",
     "vd<0 ∧ adv<0: image rejects current token AND teacher correctly pushes student away "
     "(the KEY OPD signal; signed proxy puts these in suppress branch)"),
]


def _quadrant_index(vd: float, adv: float) -> int:
    """Return 0..3 matching QUADRANTS order."""
    if vd >= 0.0:
        return 0 if adv >= 0.0 else 1
    else:
        return 2 if adv >= 0.0 else 3


@dataclass
class Accumulator:
    """Streaming sufficient statistics for one bucket (a step, or pooled)."""

    n_samples_seen: int = 0
    n_samples_used: int = 0
    n_samples_unit_w: int = 0  # vd_weights all == 1 (degenerate or weighting-off)
    n_samples_no_adv: int = 0  # lp_full or old_lp_student missing / mismatched
    n_samples_no_old_lp: int = 0  # old_lp_student missing from both diag row and sidecar
    n_samples_old_lp_from_sidecar: int = 0  # successfully joined from .adv_dp*.jsonl.gz
    n_tokens_used: int = 0

    sum_adv2: float = 0.0            # Σ adv²
    sum_w2_adv2: float = 0.0         # Σ (w·adv)²
    sum_abs_adv: float = 0.0         # Σ |adv|
    sum_supp_neg_vd_neg_adv: float = 0.0  # Σ |adv| · 𝟙[vd<0 ∧ adv<0 ∧ w<1]
    sum_neg_vd_neg_adv: float = 0.0       # Σ |adv| · 𝟙[vd<0 ∧ adv<0] (denominator alt)

    # Pearson corr(w, |adv|) sufficient stats
    n_corr: int = 0
    sum_w: float = 0.0
    sum_abs_a: float = 0.0
    sum_w2: float = 0.0
    sum_abs_a2: float = 0.0
    sum_w_abs_a: float = 0.0

    # |adv| per VD-norm bin: (sum, n) per bin
    bin_sum_abs_adv: list[float] = field(default_factory=lambda: [0.0] * len(VD_NORM_BINS))
    bin_n: list[int] = field(default_factory=lambda: [0] * len(VD_NORM_BINS))

    # 4-quadrant per (sign(vd), sign(adv)) (GPT round-4 ask). Index matches QUADRANTS.
    quad_n: list[int] = field(default_factory=lambda: [0] * 4)
    quad_sum_abs_adv: list[float] = field(default_factory=lambda: [0.0] * 4)
    quad_sum_w: list[float] = field(default_factory=lambda: [0.0] * 4)
    quad_n_w_below_1: list[int] = field(default_factory=lambda: [0] * 4)
    quad_sum_abs_vd: list[float] = field(default_factory=lambda: [0.0] * 4)

    # Per-sample distribution of rho_l2 / corr / frac_supp (so we can
    # report mean/median/p5/p95 in addition to pooled).
    per_sample_rho_l2: list[float] = field(default_factory=list)
    per_sample_corr: list[float] = field(default_factory=list)
    per_sample_frac_supp: list[float] = field(default_factory=list)


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _dist_summary(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": float("nan"), "p5": float("nan"),
                "p50": float("nan"), "p95": float("nan")}
    s = sorted(v for v in vals if not math.isnan(v))
    if not s:
        return {"n": 0, "mean": float("nan"), "p5": float("nan"),
                "p50": float("nan"), "p95": float("nan")}
    return {
        "n": len(s),
        "mean": sum(s) / len(s),
        "p5":  _percentile(s, 0.05),
        "p50": _percentile(s, 0.50),
        "p95": _percentile(s, 0.95),
    }


def _process_sample(
    row: dict,
    acc: Accumulator,
    pooled: Accumulator,
    sidecar_lookup: dict[int, list[float]] | None = None,
) -> None:
    """Update streaming stats for one diagnostics row in both per-step and pooled accumulators.

    sidecar_lookup: optional {sample_index → old_log_probs[float]} from the
    P19 adv-dump sidecar. Used when row["old_lp_student"] is empty (which
    is the normal production case — the existing diag hook fires before
    trainer forward).
    """
    R = int(row.get("response_length", 0))
    if R <= 1:
        return
    acc.n_samples_seen += 1
    pooled.n_samples_seen += 1

    lp_full = row.get("lp_full") or []
    old_lp_student = row.get("old_lp_student") or []
    vd = row.get("vd") or []
    w = row.get("vd_weights") or []

    # Try sidecar fallback for old_lp_student.
    used_sidecar = False
    if (not old_lp_student or len(old_lp_student) != R) and sidecar_lookup:
        sample_index = row.get("sample_index")
        if sample_index is not None:
            alt = sidecar_lookup.get(sample_index)
            if alt and len(alt) == R:
                old_lp_student = alt
                used_sidecar = True

    if not lp_full or len(lp_full) != R:
        acc.n_samples_no_adv += 1
        pooled.n_samples_no_adv += 1
        return
    if not old_lp_student or len(old_lp_student) != R:
        acc.n_samples_no_old_lp += 1
        pooled.n_samples_no_old_lp += 1
        return
    if not vd or len(vd) != R or not w or len(w) != R:
        acc.n_samples_no_adv += 1
        pooled.n_samples_no_adv += 1
        return

    if used_sidecar:
        acc.n_samples_old_lp_from_sidecar += 1
        pooled.n_samples_old_lp_from_sidecar += 1

    # Check unit-w (degenerate or weighting-off).
    if all(abs(x - 1.0) < 1e-6 for x in w):
        acc.n_samples_unit_w += 1
        pooled.n_samples_unit_w += 1
        return

    # Per-sequence min-max normalize vd, matching vd_weighting.py exactly.
    vd_min = min(vd)
    vd_max = max(vd)
    vd_range = vd_max - vd_min
    if vd_range < 1e-8:
        # Should not happen if w != ones, but defensive.
        acc.n_samples_unit_w += 1
        pooled.n_samples_unit_w += 1
        return
    vd_norm = [(v - vd_min) / vd_range for v in vd]

    # adv_t = lp_full - old_lp_student. We do NOT apply a sentinel mask
    # here — sentinel-failed samples have lp_full = [] (caught above);
    # the canonical Uni-OPD sentinel replacement happens in
    # sample.teacher_log_probs, not in the raw lp_full we read from
    # meta_info. So any lp_full present here is genuine teacher logp.
    adv = [lpf - lps for lpf, lps in zip(lp_full, old_lp_student)]

    s_adv2 = 0.0
    s_w2_adv2 = 0.0
    s_abs_adv = 0.0
    s_supp = 0.0           # numerator: |adv| · 𝟙[vd<0 ∧ adv<0 ∧ w<1]
    s_neg_vd_neg_adv = 0.0  # alt numerator: |adv| · 𝟙[vd<0 ∧ adv<0]
    n_corr = 0
    s_w = 0.0
    s_aa = 0.0
    s_w2 = 0.0
    s_aa2 = 0.0
    s_w_aa = 0.0

    for t in range(R):
        a = adv[t]
        wt = w[t]
        vdn = vd_norm[t]
        vt = vd[t]
        aa = abs(a)

        a2 = a * a
        s_adv2 += a2
        s_w2_adv2 += (wt * wt) * a2
        s_abs_adv += aa

        if vt < 0.0 and a < 0.0:
            s_neg_vd_neg_adv += aa
            if wt < 1.0:
                s_supp += aa

        # Pearson corr(w, |adv|) sufficient stats
        n_corr += 1
        s_w += wt
        s_aa += aa
        s_w2 += wt * wt
        s_aa2 += aa * aa
        s_w_aa += wt * aa

        # VD-norm bin (clamp 1.0 into the last bin).
        for i, (lo, hi, _) in enumerate(VD_NORM_BINS):
            if (vdn >= lo and vdn < hi) or (i == len(VD_NORM_BINS) - 1 and vdn >= hi):
                acc.bin_sum_abs_adv[i] += aa
                acc.bin_n[i] += 1
                pooled.bin_sum_abs_adv[i] += aa
                pooled.bin_n[i] += 1
                break

        # 4-quadrant by (sign(vd), sign(adv)).
        qi = _quadrant_index(vt, a)
        acc.quad_n[qi] += 1
        acc.quad_sum_abs_adv[qi] += aa
        acc.quad_sum_w[qi] += wt
        if wt < 1.0:
            acc.quad_n_w_below_1[qi] += 1
        acc.quad_sum_abs_vd[qi] += abs(vt)
        pooled.quad_n[qi] += 1
        pooled.quad_sum_abs_adv[qi] += aa
        pooled.quad_sum_w[qi] += wt
        if wt < 1.0:
            pooled.quad_n_w_below_1[qi] += 1
        pooled.quad_sum_abs_vd[qi] += abs(vt)

    if s_adv2 < _EPS:
        # All-zero advantage — uninformative, skip.
        return

    acc.n_samples_used += 1
    pooled.n_samples_used += 1
    acc.n_tokens_used += R
    pooled.n_tokens_used += R

    acc.sum_adv2 += s_adv2
    acc.sum_w2_adv2 += s_w2_adv2
    acc.sum_abs_adv += s_abs_adv
    acc.sum_supp_neg_vd_neg_adv += s_supp
    acc.sum_neg_vd_neg_adv += s_neg_vd_neg_adv

    pooled.sum_adv2 += s_adv2
    pooled.sum_w2_adv2 += s_w2_adv2
    pooled.sum_abs_adv += s_abs_adv
    pooled.sum_supp_neg_vd_neg_adv += s_supp
    pooled.sum_neg_vd_neg_adv += s_neg_vd_neg_adv

    acc.n_corr += n_corr
    acc.sum_w += s_w
    acc.sum_abs_a += s_aa
    acc.sum_w2 += s_w2
    acc.sum_abs_a2 += s_aa2
    acc.sum_w_abs_a += s_w_aa

    pooled.n_corr += n_corr
    pooled.sum_w += s_w
    pooled.sum_abs_a += s_aa
    pooled.sum_w2 += s_w2
    pooled.sum_abs_a2 += s_aa2
    pooled.sum_w_abs_a += s_w_aa

    # Per-sample summaries (for distribution view).
    rho_l2 = math.sqrt(s_w2_adv2 / (s_adv2 + _EPS))
    acc.per_sample_rho_l2.append(rho_l2)
    pooled.per_sample_rho_l2.append(rho_l2)

    # Per-sample Pearson corr(w, |adv|).
    if n_corr >= 2:
        var_w = s_w2 - (s_w * s_w) / n_corr
        var_a = s_aa2 - (s_aa * s_aa) / n_corr
        cov = s_w_aa - (s_w * s_aa) / n_corr
        denom = math.sqrt(max(var_w, 0.0) * max(var_a, 0.0))
        if denom > _EPS:
            corr = cov / denom
            acc.per_sample_corr.append(corr)
            pooled.per_sample_corr.append(corr)

    if s_abs_adv > _EPS:
        frac = s_supp / s_abs_adv
        acc.per_sample_frac_supp.append(frac)
        pooled.per_sample_frac_supp.append(frac)


def _summarize(acc: Accumulator) -> dict:
    """Reduce an Accumulator to the report dict."""
    rho_l2_pooled = (
        math.sqrt(acc.sum_w2_adv2 / acc.sum_adv2)
        if acc.sum_adv2 > _EPS else float("nan")
    )

    # Pooled Pearson corr(w, |adv|).
    if acc.n_corr >= 2 and acc.sum_w2 > _EPS and acc.sum_abs_a2 > _EPS:
        n = acc.n_corr
        var_w = acc.sum_w2 - (acc.sum_w * acc.sum_w) / n
        var_a = acc.sum_abs_a2 - (acc.sum_abs_a * acc.sum_abs_a) / n
        cov = acc.sum_w_abs_a - (acc.sum_w * acc.sum_abs_a) / n
        denom = math.sqrt(max(var_w, 0.0) * max(var_a, 0.0))
        corr_pooled = cov / denom if denom > _EPS else float("nan")
    else:
        corr_pooled = float("nan")

    frac_supp = (
        acc.sum_supp_neg_vd_neg_adv / acc.sum_abs_adv
        if acc.sum_abs_adv > _EPS else float("nan")
    )
    frac_neg_vd_neg_adv_total = (
        acc.sum_neg_vd_neg_adv / acc.sum_abs_adv
        if acc.sum_abs_adv > _EPS else float("nan")
    )
    # GPT round-4: report the conditional ratio explicitly. This is
    # "what fraction of visual-rejection correction mass is being
    # routed into the suppress branch (w<1)" — sharper framing than
    # the unconditional frac_supp.
    conditional_supp_visual_rejection = (
        acc.sum_supp_neg_vd_neg_adv / acc.sum_neg_vd_neg_adv
        if acc.sum_neg_vd_neg_adv > _EPS else float("nan")
    )

    quadrants = {}
    total_n = sum(acc.quad_n)
    total_abs_adv = sum(acc.quad_sum_abs_adv)
    for qi, (key, desc) in enumerate(QUADRANTS):
        n = acc.quad_n[qi]
        if n > 0:
            quadrants[key] = {
                "description": desc,
                "n_tokens": n,
                "frac_tokens": n / total_n if total_n > 0 else 0.0,
                "frac_abs_adv_mass": (
                    acc.quad_sum_abs_adv[qi] / total_abs_adv
                    if total_abs_adv > _EPS else 0.0
                ),
                "mean_weight": acc.quad_sum_w[qi] / n,
                "frac_w_below_1": acc.quad_n_w_below_1[qi] / n,
                "mean_abs_vd": acc.quad_sum_abs_vd[qi] / n,
                "mean_abs_adv": acc.quad_sum_abs_adv[qi] / n,
            }
        else:
            quadrants[key] = {
                "description": desc,
                "n_tokens": 0,
                "frac_tokens": 0.0,
                "frac_abs_adv_mass": 0.0,
                "mean_weight": float("nan"),
                "frac_w_below_1": float("nan"),
                "mean_abs_vd": float("nan"),
                "mean_abs_adv": float("nan"),
            }

    vd_bin_means = {}
    for i, (lo, hi, name) in enumerate(VD_NORM_BINS):
        if acc.bin_n[i] > 0:
            vd_bin_means[name] = {
                "range": [lo, hi],
                "n_tokens": acc.bin_n[i],
                "mean_abs_adv": acc.bin_sum_abs_adv[i] / acc.bin_n[i],
            }
        else:
            vd_bin_means[name] = {
                "range": [lo, hi],
                "n_tokens": 0,
                "mean_abs_adv": float("nan"),
            }

    return {
        "n_samples_seen": acc.n_samples_seen,
        "n_samples_used": acc.n_samples_used,
        "n_samples_unit_w": acc.n_samples_unit_w,
        "n_samples_no_adv": acc.n_samples_no_adv,
        "n_samples_no_old_lp": acc.n_samples_no_old_lp,
        "n_samples_old_lp_from_sidecar": acc.n_samples_old_lp_from_sidecar,
        "n_tokens_used": acc.n_tokens_used,

        "rho_l2_pooled": rho_l2_pooled,
        "corr_w_abs_adv_pooled": corr_pooled,
        "frac_supp_neg_vd_neg_adv_mass": frac_supp,
        "frac_neg_vd_neg_adv_mass_total": frac_neg_vd_neg_adv_total,
        "conditional_supp_visual_rejection": conditional_supp_visual_rejection,

        "rho_l2_per_sample": _dist_summary(acc.per_sample_rho_l2),
        "corr_w_abs_adv_per_sample": _dist_summary(acc.per_sample_corr),
        "frac_supp_per_sample": _dist_summary(acc.per_sample_frac_supp),

        "abs_adv_by_vd_norm_bin": vd_bin_means,
        "quadrants": quadrants,
    }


def _load_sidecar_for_step(diag_dir: Path, step: int) -> tuple[dict[int, list[float]], dict]:
    """Load (sample_index → old_log_probs) for one step by merging across dp ranks.

    Reads all step_NNNNNN.adv_dp*.jsonl.gz matching the given step. Returns
    (lookup, stats) where stats has join-sanity counters:
      n_sidecar_files, n_sidecar_rows, n_duplicate_sample_index.
    Empty dict on the lookup side means no sidecar available (caller
    falls back to row["old_lp_student"]).
    """
    out: dict[int, list[float]] = {}
    files = sorted(diag_dir.glob(f"step_{step:06d}.adv_dp*.jsonl.gz"))
    n_dups = 0
    n_rows = 0
    for f in files:
        try:
            with gzip.open(f, "rt") as fin:
                for line in fin:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    idx = row.get("sample_index")
                    olp = row.get("old_log_probs")
                    if idx is None or not olp:
                        continue
                    if idx in out:
                        n_dups += 1
                    out[idx] = olp
                    n_rows += 1
        except Exception:
            continue
    return out, {
        "n_sidecar_files": len(files),
        "n_sidecar_rows": n_rows,
        "n_duplicate_sample_index": n_dups,
    }


def scan(diag_dir: Path, steps: list[int] | None, limit_rows: int) -> dict:
    files = sorted(diag_dir.glob("step_*.jsonl.gz"))
    # The .adv_dp*.jsonl.gz sidecars also match step_*.jsonl.gz under the
    # naive glob; filter them out so we only process main diag files.
    files = [f for f in files if not SIDECAR_RE.search(f.name)]
    if not files:
        raise FileNotFoundError(f"no step_*.jsonl.gz in {diag_dir}")

    pooled = Accumulator()
    per_step: dict[int, Accumulator] = {}
    sidecar_steps_found = 0
    total_sidecar_files = 0
    total_sidecar_rows = 0
    total_sidecar_duplicates = 0

    for f in files:
        m = STEP_RE.search(f.name)
        if not m:
            continue
        s = int(m.group(1))
        if steps and s not in steps:
            continue
        acc = Accumulator()
        per_step[s] = acc

        sidecar_lookup, sidecar_stats = _load_sidecar_for_step(diag_dir, s)
        if sidecar_lookup:
            sidecar_steps_found += 1
        total_sidecar_files += sidecar_stats["n_sidecar_files"]
        total_sidecar_rows += sidecar_stats["n_sidecar_rows"]
        total_sidecar_duplicates += sidecar_stats["n_duplicate_sample_index"]

        with gzip.open(f, "rt") as fin:
            n = 0
            for line in fin:
                if limit_rows and n >= limit_rows:
                    break
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                _process_sample(row, acc, pooled, sidecar_lookup or None)
                n += 1

    out = {
        "diag_dir": str(diag_dir),
        "n_step_files_scanned": len(per_step),
        "n_steps_with_sidecar": sidecar_steps_found,
        "sidecar_join_stats": {
            "total_sidecar_files": total_sidecar_files,
            "total_sidecar_rows": total_sidecar_rows,
            "total_duplicate_sample_index": total_sidecar_duplicates,
            "n_samples_old_lp_from_sidecar_pooled": pooled.n_samples_old_lp_from_sidecar,
            "n_samples_no_old_lp_pooled": pooled.n_samples_no_old_lp,
        },
        "steps_scanned": sorted(per_step.keys()),
        "pooled": _summarize(pooled),
        "per_step": {str(s): _summarize(per_step[s]) for s in sorted(per_step.keys())},
    }

    # Headline interpretation: short, decisive.
    p = out["pooled"]
    rho = p.get("rho_l2_pooled", float("nan"))
    corr = p.get("corr_w_abs_adv_pooled", float("nan"))
    frac = p.get("frac_supp_neg_vd_neg_adv_mass", float("nan"))
    interp = []
    if not math.isnan(rho):
        if rho < 0.7:
            interp.append(f"rho_l2={rho:.3f} < 0.7 ⇒ effective-LR-drop confound strongly supported")
        elif rho < 0.85:
            interp.append(f"rho_l2={rho:.3f} ∈ [0.7, 0.85) ⇒ partial LR confound")
        else:
            interp.append(f"rho_l2={rho:.3f} ≥ 0.85 ⇒ LR confound unlikely")
    if not math.isnan(corr):
        if corr < -0.05:
            interp.append(f"corr(w,|adv|)={corr:+.3f} ⇒ weights anti-correlate with signal magnitude")
        elif corr > 0.05:
            interp.append(f"corr(w,|adv|)={corr:+.3f} ⇒ weights correlate with signal (expected)")
        else:
            interp.append(f"corr(w,|adv|)={corr:+.3f} ⇒ weights uncorrelated with signal magnitude")
    if not math.isnan(frac):
        if frac > 0.15:
            interp.append(f"frac_supp_neg_vd_neg_adv={frac:.3f} > 0.15 ⇒ signed-proxy is suppressing substantial visual-rejection mass")
        elif frac > 0.05:
            interp.append(f"frac_supp_neg_vd_neg_adv={frac:.3f} ⇒ modest visual-rejection suppression")
        else:
            interp.append(f"frac_supp_neg_vd_neg_adv={frac:.3f} ⇒ visual-rejection suppression small")
    cond = p.get("conditional_supp_visual_rejection", float("nan"))
    if not math.isnan(cond):
        interp.append(
            f"conditional_supp_visual_rejection={cond:.3f} "
            f"⇒ {cond*100:.0f}% of (vd<0 ∧ adv<0) correction mass is routed into suppress branch"
        )
    # Per-quadrant snapshot to drive headline.
    q = p.get("quadrants", {})
    if "vis_reject_correction" in q and q["vis_reject_correction"]["n_tokens"] > 0:
        vc = q["vis_reject_correction"]
        interp.append(
            f"visual-rejection-correction quadrant: "
            f"{vc['frac_tokens']*100:.1f}% of tokens, "
            f"{vc['frac_abs_adv_mass']*100:.1f}% of |adv| mass, "
            f"mean_weight={vc['mean_weight']:.3f} "
            f"(<1 means PGPO suppressing this quadrant)"
        )
    out["headline_interpretation"] = interp
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--diag-dir", required=True,
                    help="<run>/diagnostics dir containing step_*.jsonl.gz")
    ap.add_argument("--steps", default=None,
                    help="comma list of step numbers to include (default: all)")
    ap.add_argument("--limit-rows-per-step", type=int, default=0,
                    help="0 = all rows; >0 = subsample first N for speed")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)

    steps = None
    if args.steps:
        steps = [int(s) for s in args.steps.split(",") if s.strip()]

    out = scan(Path(args.diag_dir), steps, args.limit_rows_per_step)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.out_json).open("w") as f:
            json.dump(out, f, indent=2, allow_nan=True)
        print(f"json -> {args.out_json}", file=sys.stderr)
    else:
        print(json.dumps(out, indent=2, default=str))

    # Headline to stderr.
    print(f"\n=== T2-1 ENERGY AUDIT (pooled, {out['pooled']['n_samples_used']} samples, "
          f"{out['pooled']['n_tokens_used']} tokens) ===", file=sys.stderr)
    for line in out["headline_interpretation"]:
        print(f"  {line}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
