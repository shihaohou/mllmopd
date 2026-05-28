"""Step 5 — Pass 4 cross-rollout comparison (closes §13.1 design gap).

The Pass 1-3 audit (`tam_step5_evidence_alignment.py`) measures
**conditional** TAM divergence: all three models (T, S0, S1) are teacher-
forced to decode S1's rollout `R1`, and per-token alignment is computed
on that shared trajectory. The original §13.1 limitation is that this
cannot distinguish between

  (i) OPD did not move attention                            vs.
  (ii) OPD moved attention but only on tokens S0 wouldn't have produced.

Pass 4 closes the gap by running a second rollout `R0 = S0_greedy`,
re-running cross-model TAM on `R0`, and producing the alignment files
`alignment_R0.jsonl` (alongside the existing `alignment.jsonl` for R1).

This script consumes both alignment files and computes two paired
sample-level comparisons:

  Δ_self_traj  =  mean_align(S1@R1, T@R1) − mean_align(S0@R0, T@R0)
                  Self-trajectory teacher-alignment delta per sample.
                  Positive = OPD raises the student's teacher-alignment
                  on its own deployment trajectory.
  Δ_s0_cross   =  mean_align(S0@R0, T@R0) − mean_align(S0@R1, T@R1)
                  Same-student, two trajectories. Positive = S0 attends
                  more like teacher on ITS own trajectory than when
                  forced to S1's trajectory. Non-zero means the
                  conditional protocol systematically misses an S0-side
                  attention pattern.

Each Δ is aggregated across samples with cluster bootstrap (re-using
`mllmopd.analysis.tam_step5_analyzer.cluster_bootstrap_ci`), a sign-flip
null calibration, and a TOST equivalence verdict.

Reliability filter mirrors the primary Pass-3 cell: per-token, keep
tokens where `tam_entropy_norm_T < reliability_thresh`. Pass 4 means
are per-sample means of those reliable tokens.

Outputs (under `--out-dir`, default `docs/figures/step5/pass4/`):
  - `pass4_per_sample.jsonl`     — one row per id paired across R0+R1
  - `pass4_summary.json`         — overall + per-bucket bootstrap blocks
  - `pass4_decision.json`        — verdict per Δ + reasoning (mirrors
                                   the schema of decision.json)
  - `tables/pass4_overall.csv`
  - `tables/pass4_per_bucket.csv`
  - `pass4-results.md`           — human-readable report

Usage::

    python -m mllmopd.analysis.tam_step5_pass4_compare \\
        --alignment-r1 runs/audit/tam_step5_<TS>/alignment.jsonl \\
        --alignment-r0 runs/audit/tam_step5_<TS>/alignment_R0.jsonl \\
        --out-dir docs/figures/step5/pass4/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from mllmopd.analysis.tam_step5_analyzer import (
    RELIABILITY_THRESH_DEFAULT,
    cluster_bootstrap_ci,
    calibrate_null_threshold,
    tost_equivalence,
)


# ============================================================================
# Per-sample reliability-filtered mean (single rollout)
# ============================================================================
def per_sample_reliable_means(rec: dict,
                              reliability_thresh: float
                              ) -> dict | None:
    """For a single alignment row (one rollout, one sample), return the
    reliability-filtered mean alignment of {S0, S1} vs T over tokens where
    every model has tam_valid AND the teacher map satisfies
    `tam_entropy_norm_T < reliability_thresh`.

    Returns dict {n_tokens, mean_{iou,js,cos}_{S0,S1}} or None if no token
    survives the filter.
    """
    R = rec.get("response_length", 0)
    if R == 0:
        return None
    valid = [
        rec["tam_valid_T"][t]
        and rec["tam_valid_S0"][t]
        and rec["tam_valid_S1"][t]
        for t in range(R)
    ]
    entropy_T = rec.get("T", {}).get("tam_entropy_norm", [None] * R)
    reliable = [
        valid[t]
        and entropy_T[t] is not None
        and entropy_T[t] < reliability_thresh
        for t in range(R)
    ]
    if not any(reliable):
        return None

    a = rec["align"]
    js_s0  = [a["S0_T"]["js"][t]       for t in range(R) if reliable[t] and a["S0_T"]["js"][t] is not None]
    js_s1  = [a["S1_T"]["js"][t]       for t in range(R) if reliable[t] and a["S1_T"]["js"][t] is not None]
    iou_s0 = [a["S0_T"]["iou_top20"][t] for t in range(R) if reliable[t] and a["S0_T"]["iou_top20"][t] is not None]
    iou_s1 = [a["S1_T"]["iou_top20"][t] for t in range(R) if reliable[t] and a["S1_T"]["iou_top20"][t] is not None]
    cos_s0 = [a["S0_T"]["cos"][t]      for t in range(R) if reliable[t] and a["S0_T"]["cos"][t] is not None]
    cos_s1 = [a["S1_T"]["cos"][t]      for t in range(R) if reliable[t] and a["S1_T"]["cos"][t] is not None]
    if not (js_s0 and js_s1):
        return None

    return {
        "n_tokens":      sum(reliable),
        "mean_js_S0":    float(np.mean(js_s0)),
        "mean_js_S1":    float(np.mean(js_s1)),
        "mean_iou_S0":   float(np.mean(iou_s0)) if iou_s0 else float("nan"),
        "mean_iou_S1":   float(np.mean(iou_s1)) if iou_s1 else float("nan"),
        "mean_cos_S0":   float(np.mean(cos_s0)) if cos_s0 else float("nan"),
        "mean_cos_S1":   float(np.mean(cos_s1)) if cos_s1 else float("nan"),
    }


def _load_alignment(path: Path) -> dict[str, dict]:
    """Load alignment JSONL keyed by sample id. Light schema sanity."""
    out: dict[str, dict] = {}
    n_dropped = 0
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "id" not in rec or "align" not in rec:
                n_dropped += 1
                continue
            out[rec["id"]] = rec
    if n_dropped:
        print(f"!! {path.name}: dropped {n_dropped} rows missing id/align",
              file=sys.stderr)
    return out


# ============================================================================
# Cross-rollout pairing
# ============================================================================
def pair_per_sample(rows_r1: dict[str, dict], rows_r0: dict[str, dict],
                    reliability_thresh: float
                    ) -> list[dict]:
    """For every id present in both R0 and R1, return a per-sample dict
    of reliability-filtered means + the two cross-rollout deltas.

    Schema (each row)::

        {
          "id": str, "benchmark": str, "bucket": str,
          "n_tokens_R0": int, "n_tokens_R1": int,
          "R0": {mean_js_S0, mean_js_S1, mean_iou_*, mean_cos_*, response_length},
          "R1": {mean_js_S0, mean_js_S1, mean_iou_*, mean_cos_*, response_length},
          # Self-trajectory delta (sign matches Pass-3 ΔJS convention:
          # ΔJS > 0 means student S1 is more aligned than S0 — i.e.
          # JS(S0@R0,T) − JS(S1@R1,T) > 0):
          "delta_self_js":  float, "delta_self_iou": float, "delta_self_cos": float,
          # Same-student cross-trajectory delta (ΔJS > 0 means S0 is more
          # aligned to T when on its own trajectory than when forced to S1's
          # trajectory — i.e. JS(S0@R1,T) − JS(S0@R0,T) > 0):
          "delta_s0_cross_js": float, "delta_s0_cross_iou": float, "delta_s0_cross_cos": float,
        }
    """
    paired: list[dict] = []
    common_ids = set(rows_r1.keys()) & set(rows_r0.keys())
    n_only_r1 = len(set(rows_r1.keys()) - common_ids)
    n_only_r0 = len(set(rows_r0.keys()) - common_ids)
    if n_only_r1 or n_only_r0:
        print(f">>> pairing: |R1∩R0|={len(common_ids)}; "
              f"R1-only={n_only_r1}, R0-only={n_only_r0}",
              file=sys.stderr)

    for sid in sorted(common_ids):
        rec_r1 = rows_r1[sid]
        rec_r0 = rows_r0[sid]
        means_r1 = per_sample_reliable_means(rec_r1, reliability_thresh)
        means_r0 = per_sample_reliable_means(rec_r0, reliability_thresh)
        if means_r0 is None or means_r1 is None:
            continue

        # Self-trajectory: S1 on R1 vs S0 on R0
        delta_self_js  = means_r0["mean_js_S0"] - means_r1["mean_js_S1"]      # JS lower-is-better
        delta_self_iou = means_r1["mean_iou_S1"] - means_r0["mean_iou_S0"]    # IoU higher-is-better
        delta_self_cos = means_r1["mean_cos_S1"] - means_r0["mean_cos_S0"]
        # Same-student cross-trajectory: S0 on R0 vs S0 on R1
        # ΔJS positive => S0 looks MORE like T when on its own rollout.
        delta_s0_cross_js  = means_r1["mean_js_S0"] - means_r0["mean_js_S0"]
        delta_s0_cross_iou = means_r0["mean_iou_S0"] - means_r1["mean_iou_S0"]
        delta_s0_cross_cos = means_r0["mean_cos_S0"] - means_r1["mean_cos_S0"]

        paired.append({
            "id":           sid,
            "benchmark":    rec_r1.get("benchmark") or rec_r0.get("benchmark"),
            "bucket":       rec_r1.get("bucket")    or rec_r0.get("bucket"),
            "R1": {
                "response_length": rec_r1.get("response_length"),
                "n_tokens_reliable": means_r1["n_tokens"],
                "mean_js_S0":  means_r1["mean_js_S0"],
                "mean_js_S1":  means_r1["mean_js_S1"],
                "mean_iou_S0": means_r1["mean_iou_S0"],
                "mean_iou_S1": means_r1["mean_iou_S1"],
                "mean_cos_S0": means_r1["mean_cos_S0"],
                "mean_cos_S1": means_r1["mean_cos_S1"],
            },
            "R0": {
                "response_length": rec_r0.get("response_length"),
                "n_tokens_reliable": means_r0["n_tokens"],
                "mean_js_S0":  means_r0["mean_js_S0"],
                "mean_js_S1":  means_r0["mean_js_S1"],
                "mean_iou_S0": means_r0["mean_iou_S0"],
                "mean_iou_S1": means_r0["mean_iou_S1"],
                "mean_cos_S0": means_r0["mean_cos_S0"],
                "mean_cos_S1": means_r0["mean_cos_S1"],
            },
            "delta_self_js":      delta_self_js,
            "delta_self_iou":     delta_self_iou,
            "delta_self_cos":     delta_self_cos,
            "delta_s0_cross_js":  delta_s0_cross_js,
            "delta_s0_cross_iou": delta_s0_cross_iou,
            "delta_s0_cross_cos": delta_s0_cross_cos,
        })
    return paired


# ============================================================================
# Aggregation
# ============================================================================
def _block(vals: list[float]) -> dict:
    """Cluster bootstrap CI + null + MDE_95 (percentile half-width)."""
    m, lo, hi, sd = cluster_bootstrap_ci(vals)
    if np.isfinite(m) and np.isfinite(lo) and np.isfinite(hi):
        mde = max(abs(m - lo), abs(hi - m))
    else:
        mde = float("nan")
    delta = calibrate_null_threshold(vals)
    verdict = tost_equivalence(m, lo, hi, delta)
    return {
        "mean":     m,
        "ci_low":   lo,
        "ci_high":  hi,
        "sd":       sd,
        "mde_95":   mde,
        "delta":    delta,
        "verdict":  verdict,
    }


METRIC_KEYS_SELF = ("delta_self_js", "delta_self_iou", "delta_self_cos")
METRIC_KEYS_CROSS = ("delta_s0_cross_js", "delta_s0_cross_iou",
                     "delta_s0_cross_cos")


def aggregate_overall(paired: list[dict]) -> dict:
    out: dict = {"n_samples": len(paired)}
    for k in METRIC_KEYS_SELF + METRIC_KEYS_CROSS:
        out[k] = _block([p[k] for p in paired])
    # Raw means for color/context in the report
    for k in ("mean_js_S0", "mean_js_S1", "mean_iou_S0", "mean_iou_S1",
              "mean_cos_S0", "mean_cos_S1"):
        out[f"R1__{k}"] = (
            float(np.mean([p["R1"][k] for p in paired])) if paired else float("nan")
        )
        out[f"R0__{k}"] = (
            float(np.mean([p["R0"][k] for p in paired])) if paired else float("nan")
        )
    return out


def aggregate_per_bucket(paired: list[dict]) -> dict:
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for p in paired:
        by_bucket[p.get("bucket") or "unknown"].append(p)
    out: dict = {}
    for bucket, lst in by_bucket.items():
        out[bucket] = {"n_samples": len(lst)}
        for k in METRIC_KEYS_SELF + METRIC_KEYS_CROSS:
            out[bucket][k] = _block([p[k] for p in lst])
    return out


# ============================================================================
# Decision
# ============================================================================
def decide_pass4(per_bucket: dict, overall: dict,
                 primary_bucket: str = "OPD_improved") -> dict:
    """Pass 4 decision on the primary cell (OPD_improved × reliability).

    Produces a structured verdict mirroring `decision.json` schema where
    possible. Branches:

      (A1) self_aligned   : Δ_self_js > δ AND ci_low > 0 → OPD raises
                            self-trajectory teacher alignment
      (A2) self_flat      : Δ_self_js TOST-equivalent → OPD does NOT
                            raise self-trajectory teacher alignment
                            (strengthens Pass-3 branch (b))
      (A3) self_inconclusive
      (B1) cross_neutral  : Δ_s0_cross_js TOST-equivalent → forcing S0
                            onto S1's trajectory does NOT mask an S0-side
                            attention shift (Pass-3 protocol unbiased)
      (B2) cross_biased   : Δ_s0_cross_js > δ AND ci_low > 0 → S0 looks
                            more T-aligned on its own trajectory
                            (Pass-3 protocol systematically over-aligns
                             S0 → original Pass-3 Δ is over-stated)
      (B3) cross_anti_biased : opposite sign (Pass-3 under-stated)
      (B4) cross_inconclusive
    """
    primary = per_bucket.get(primary_bucket)
    if primary is None or primary.get("n_samples", 0) == 0:
        return {
            "primary_cell":   primary_bucket,
            "branch_self":    "inconclusive",
            "branch_cross":   "inconclusive",
            "self_label":     "(inconclusive) primary cell empty",
            "cross_label":    "(inconclusive) primary cell empty",
            "n_primary":      0,
        }

    MIN_PRIMARY_N = 10
    if primary["n_samples"] < MIN_PRIMARY_N:
        return {
            "primary_cell":   primary_bucket,
            "branch_self":    "underpowered",
            "branch_cross":   "underpowered",
            "self_label":     (f"(underpowered) n={primary['n_samples']} "
                               f"< {MIN_PRIMARY_N}"),
            "cross_label":    (f"(underpowered) n={primary['n_samples']} "
                               f"< {MIN_PRIMARY_N}"),
            "n_primary":      primary["n_samples"],
            "min_primary_n":  MIN_PRIMARY_N,
        }

    self_js = primary["delta_self_js"]
    cross_js = primary["delta_s0_cross_js"]

    # SELF branch
    if self_js["verdict"] == "aligned":
        branch_self = "self_aligned"
        self_label = ("(A1) OPD raises self-trajectory teacher alignment — "
                      "EA-OPD-style supervision may overlap with vanilla OPD.")
    elif self_js["verdict"] == "equivalent":
        branch_self = "self_flat"
        self_label = ("(A2) OPD does NOT raise self-trajectory teacher "
                      "alignment — Pass-3 branch (b) flat conclusion holds "
                      "under self-rollout. EA-OPD motivation strengthened.")
    elif self_js["verdict"] == "anti":
        branch_self = "self_anti"
        self_label = ("(A4) OPD LOWERS self-trajectory teacher alignment "
                      "(pathological) — vanilla OPD actively misaligns "
                      "visual evidence; EA-OPD motivation extremely strong.")
    else:
        branch_self = "self_inconclusive"
        self_label = ("(A3) self-trajectory Δ inconclusive — increase n or "
                      "accept Pass 4 cannot decide the §13.1 question.")

    # CROSS branch
    if cross_js["verdict"] == "equivalent":
        branch_cross = "cross_neutral"
        cross_label = ("(B1) Pass-3 protocol unbiased — forcing S0 onto "
                       "S1's trajectory does not mask an S0-side attention "
                       "shift. Original Pass-3 Δ is fairly comparable.")
    elif cross_js["verdict"] == "aligned":
        branch_cross = "cross_biased"
        cross_label = ("(B2) Pass-3 protocol BIASED — S0 looks more T-"
                       "aligned on its own trajectory than when forced to "
                       "S1's. Original Pass-3 Δ_JS over-states S0's gap.")
    elif cross_js["verdict"] == "anti":
        branch_cross = "cross_anti_biased"
        cross_label = ("(B3) Pass-3 protocol BIASED (opposite sign) — S0 "
                       "looks LESS T-aligned on its own trajectory. "
                       "Original Pass-3 Δ_JS under-states S0's gap.")
    else:
        branch_cross = "cross_inconclusive"
        cross_label = ("(B4) cross-trajectory Δ inconclusive — increase n.")

    return {
        "primary_cell":     primary_bucket,
        "branch_self":      branch_self,
        "branch_cross":     branch_cross,
        "self_label":       self_label,
        "cross_label":      cross_label,
        "delta_self_js":    self_js,
        "delta_s0_cross_js": cross_js,
        "n_primary":        primary["n_samples"],
    }


# ============================================================================
# IO — tables + per-sample + summary
# ============================================================================
def _fmt_block(b: dict) -> str:
    return (f"{b['mean']:+.4f} [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]  "
            f"sd={b['sd']:.4f}  δ={b['delta']:.4f}  → **{b['verdict']}**")


def write_overall_csv(overall: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("metric,mean,ci_low,ci_high,sd,mde_95,delta_null,verdict\n")
        for k in METRIC_KEYS_SELF + METRIC_KEYS_CROSS:
            b = overall[k]
            f.write(f"{k},{b['mean']:.6f},{b['ci_low']:.6f},{b['ci_high']:.6f},"
                    f"{b['sd']:.6f},{b['mde_95']:.6f},{b['delta']:.6f},"
                    f"{b['verdict']}\n")
        f.write(f"n_samples,{overall['n_samples']},,,,,,\n")


def write_per_bucket_csv(per_bucket: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("bucket,metric,n_samples,mean,ci_low,ci_high,sd,mde_95,"
                "delta_null,verdict\n")
        for bucket in sorted(per_bucket.keys()):
            bk = per_bucket[bucket]
            for k in METRIC_KEYS_SELF + METRIC_KEYS_CROSS:
                b = bk[k]
                f.write(f"{bucket},{k},{bk['n_samples']},"
                        f"{b['mean']:.6f},{b['ci_low']:.6f},{b['ci_high']:.6f},"
                        f"{b['sd']:.6f},{b['mde_95']:.6f},{b['delta']:.6f},"
                        f"{b['verdict']}\n")


def write_per_sample_jsonl(paired: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in paired:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def write_results_md(overall: dict, per_bucket: dict, decision: dict,
                     reliability_thresh: float,
                     args, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("# Step 5 — Pass 4 cross-rollout comparison\n\n")
        f.write(f"Generated by `mllmopd.analysis.tam_step5_pass4_compare` "
                f"(commit={os.environ.get('MLLMOPD_CODE_COMMIT', 'unknown')})\n\n")
        f.write("**What Pass 4 measures:** see "
                "`docs/step5-evidence-alignment-design.md` §13.\n\n")
        f.write(f"Inputs:\n- R1 alignment: `{args.alignment_r1}`\n"
                f"- R0 alignment: `{args.alignment_r0}`\n\n")
        f.write(f"Reliability filter: `tam_entropy_norm_T < "
                f"{reliability_thresh:.3f}`.\n\n")

        f.write("## 0. Decision\n\n")
        f.write(f"**Primary cell:** `{decision.get('primary_cell')} × "
                f"reliability`  (n={decision.get('n_primary', 0)})\n\n")
        f.write(f"**Self-trajectory branch:** `{decision['branch_self']}` — "
                f"{decision['self_label']}\n\n")
        f.write(f"**Cross-trajectory branch:** `{decision['branch_cross']}` — "
                f"{decision['cross_label']}\n\n")

        f.write("## 1. Overall (all paired samples)\n\n")
        f.write(f"n_samples paired (R0 ∩ R1, both reliability-filtered) = "
                f"**{overall['n_samples']}**\n\n")
        f.write("### Self-trajectory Δ:  align(S1@R1, T@R1) − align(S0@R0, T@R0)\n\n")
        f.write("| metric | mean [CI95] / sd / δ_null / TOST |\n|---|---|\n")
        for k in METRIC_KEYS_SELF:
            f.write(f"| `{k}` | {_fmt_block(overall[k])} |\n")
        f.write("\n### Cross-trajectory Δ (same-student S0):  "
                "align(S0@R0, T@R0) − align(S0@R1, T@R1)\n\n")
        f.write("| metric | mean [CI95] / sd / δ_null / TOST |\n|---|---|\n")
        for k in METRIC_KEYS_CROSS:
            f.write(f"| `{k}` | {_fmt_block(overall[k])} |\n")

        f.write("\n### Raw means (sanity)\n\n")
        f.write("| trajectory | mean JS(S0,T) | mean JS(S1,T) | mean IoU(S0,T) | mean IoU(S1,T) |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| R1 (S1 self) | {overall.get('R1__mean_js_S0', float('nan')):.4f} | "
                f"{overall.get('R1__mean_js_S1', float('nan')):.4f} | "
                f"{overall.get('R1__mean_iou_S0', float('nan')):.4f} | "
                f"{overall.get('R1__mean_iou_S1', float('nan')):.4f} |\n")
        f.write(f"| R0 (S0 self) | {overall.get('R0__mean_js_S0', float('nan')):.4f} | "
                f"{overall.get('R0__mean_js_S1', float('nan')):.4f} | "
                f"{overall.get('R0__mean_iou_S0', float('nan')):.4f} | "
                f"{overall.get('R0__mean_iou_S1', float('nan')):.4f} |\n\n")

        f.write("## 2. Per bucket\n\n")
        for bucket in ("OPD_improved", "OPD_failed", "Teacher_advantage",
                       "Dataset_diversity"):
            if bucket not in per_bucket:
                continue
            bk = per_bucket[bucket]
            f.write(f"### {bucket}  (n={bk['n_samples']})\n\n")
            f.write("**Self-trajectory Δ:**\n\n")
            f.write("| metric | mean [CI95] / sd / δ_null / TOST |\n|---|---|\n")
            for k in METRIC_KEYS_SELF:
                f.write(f"| `{k}` | {_fmt_block(bk[k])} |\n")
            f.write("\n**Cross-trajectory Δ (S0 only):**\n\n")
            f.write("| metric | mean [CI95] / sd / δ_null / TOST |\n|---|---|\n")
            for k in METRIC_KEYS_CROSS:
                f.write(f"| `{k}` | {_fmt_block(bk[k])} |\n")
            f.write("\n")

        f.write("## 3. Implications for §13.1 and §Method (EA-OPD)\n\n")
        bs = decision["branch_self"]
        bc = decision["branch_cross"]
        if bs == "self_flat" and bc == "cross_neutral":
            f.write("Best case for the §13.5 EA-OPD motivation: the original "
                    "Pass-3 branch (b) conclusion holds under self-rollout "
                    "(self_flat) AND the conditional protocol was not "
                    "biased (cross_neutral). Both the conditional and the "
                    "counterfactual readings now say: vanilla OPD does not "
                    "perform evidence alignment.\n")
        elif bs == "self_flat" and bc != "cross_neutral":
            f.write("Mixed: Pass-3 self-rollout conclusion holds (self_flat) "
                    "but the conditional protocol IS biased (cross != neutral). "
                    "Re-state Pass-3 ΔJS as protocol-dependent in §6.2.\n")
        elif bs == "self_aligned":
            f.write("Self-trajectory Δ is positive: OPD *does* raise self-"
                    "trajectory teacher alignment, contradicting Pass-3's "
                    "conditional flat. Original §6.2 honest-claim boundary "
                    "must be re-tightened: the Pass-3 conditional result is "
                    "the *more conservative* of the two.\n")
        elif bs == "self_anti":
            f.write("Self-trajectory Δ is NEGATIVE: OPD lowers self-trajectory "
                    "teacher alignment. EA-OPD becomes a more urgent §Method "
                    "candidate.\n")
        else:
            f.write("Self-trajectory Δ inconclusive — Pass 4 cannot settle "
                    "§13.1 at the current n. The Pass-3 branch (b) conclusion "
                    "remains the operative §Method input; report Pass-4 result "
                    "as a directional hint only.\n")


def write_summary_json(overall: dict, per_bucket: dict, decision: dict,
                       reliability_thresh: float,
                       args, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "overall":              overall,
        "per_bucket":           per_bucket,
        "decision":             decision,
        "reliability_thresh":   reliability_thresh,
        "alignment_r1_path":    str(args.alignment_r1),
        "alignment_r0_path":    str(args.alignment_r0),
        "code_commit":          os.environ.get("MLLMOPD_CODE_COMMIT", "unknown"),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def write_decision_json(decision: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2))


# ============================================================================
# Main
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alignment-r1", type=Path, required=True,
                    help="alignment.jsonl (R1 = S1 self rollout)")
    ap.add_argument("--alignment-r0", type=Path, required=True,
                    help="alignment_R0.jsonl (R0 = S0 self rollout)")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("docs/figures/step5/pass4/"))
    ap.add_argument("--reliability-thresh", type=float,
                    default=RELIABILITY_THRESH_DEFAULT,
                    help=(f"Per-token reliability filter "
                          f"(default {RELIABILITY_THRESH_DEFAULT}; same as "
                          f"Pass-3 primary cell)."))
    ap.add_argument("--primary-bucket", default="OPD_improved",
                    help="Decision-cell bucket (default OPD_improved).")
    args = ap.parse_args(argv)

    for p in (args.alignment_r1, args.alignment_r0):
        if not p.exists():
            print(f"!! missing input: {p}", file=sys.stderr)
            return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_r1 = _load_alignment(args.alignment_r1)
    rows_r0 = _load_alignment(args.alignment_r0)
    print(f">>> loaded: R1 n={len(rows_r1)}, R0 n={len(rows_r0)}",
          file=sys.stderr)

    paired = pair_per_sample(rows_r1, rows_r0, args.reliability_thresh)
    print(f">>> paired (both rollouts reliable): n={len(paired)}",
          file=sys.stderr)

    if not paired:
        print("!! no paired samples after reliability filter; "
              "lowering --reliability-thresh might help, but the conclusion "
              "is that Pass 4 cannot decide on this run.", file=sys.stderr)
        return 3

    overall    = aggregate_overall(paired)
    per_bucket = aggregate_per_bucket(paired)
    decision   = decide_pass4(per_bucket, overall,
                              primary_bucket=args.primary_bucket)

    write_per_sample_jsonl(paired, args.out_dir / "pass4_per_sample.jsonl")
    write_overall_csv(overall, args.out_dir / "tables" / "pass4_overall.csv")
    write_per_bucket_csv(per_bucket,
                         args.out_dir / "tables" / "pass4_per_bucket.csv")
    write_summary_json(overall, per_bucket, decision,
                       args.reliability_thresh, args,
                       args.out_dir / "pass4_summary.json")
    write_decision_json(decision, args.out_dir / "pass4_decision.json")
    write_results_md(overall, per_bucket, decision,
                     args.reliability_thresh, args,
                     args.out_dir / "pass4-results.md")

    print(f">>> outputs in {args.out_dir}")
    print(f"    decision.self  = {decision['branch_self']}: "
          f"{decision['self_label']}")
    print(f"    decision.cross = {decision['branch_cross']}: "
          f"{decision['cross_label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
