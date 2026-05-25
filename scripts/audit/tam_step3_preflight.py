"""Step 3a TAM-Boost OPD — offline pre-flight calibration.

Reads `tam_step1a.jsonl` and runs the Step 3a gate over the subset TAM maps
(K=20 stratified per sample × ckpt; see tam_step1a `tam_maps_subset`). For
each subset token, computes the gate g_t under a sweep of (τ, K, ρ) and
reports:

  - Gate-fire rate per category × ckpt
  - Gate-fire rate per quad × ckpt (cross-ref with Step 2b quad signal)
  - Coverage distribution (histogram + p25/p50/p75/p95)
  - Suggested α s.t. E[w_t] ≈ target_mean_w
  - τ recommendation s.t. gate-fire rate ≈ target_gate_fire_rate

**Caveat**: Step 1a stores ONLY K=20 stratified TAM maps per (sample, ckpt),
not all R maps. So E_x is estimated from those 20 maps rather than the full
response. The stratification over-samples C_local categories, which is fine
for calibrating gate-fire rate among C_local positions but means the
absolute pooled rate (across all categories of the response) is NOT
representative. To get a corpus-wide rate, run a v0.1.5 Step 1a that
serializes the full _response_maps_b64 cache.

Usage::

    python -m scripts.audit.tam_step3_preflight \\
        --step1a-jsonl runs/audit/tam_step1a_<TS>/tam_step1a.jsonl \\
        --out-json   runs/analysis/tam_step3_preflight_v0.json \\
        --out-txt    runs/analysis/tam_step3_preflight_v0.txt \\
        [--tau-sweep 0.3,0.4,0.5,0.6,0.7] \\
        [--K 0.20] [--rho 0.30] \\
        [--target-mean-w 1.10] \\
        [--target-gate-fire-rate 0.15]
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Repo-local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from mllmopd.training.tam_gate import (  # noqa: E402
    GateConfig, compute_weights, C_LOCAL_DEFAULT,
)


def _decode_uint8_map(b64: str, h: int, w: int) -> np.ndarray:
    """Decode `_b64_uint8_map` output back to float (0..1)."""
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size != h * w:
        raise ValueError(f"map size {arr.size} != H*W {h * w}")
    return (arr.reshape(h, w).astype(np.float32)) / 255.0


def _percentile(vs: list, p: float) -> float:
    if not vs:
        return float("nan")
    return float(np.percentile(vs, p))


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _gate_one_row(row: dict, K: float, rho: float, tau: float) -> dict:
    """Run gate over one row's subset; returns per-(subset-token) records.

    Skips rows with tam_valid=False or empty tam_maps_subset."""
    if not row.get("tam_valid"):
        return {"records": [], "skip_reason": "tam_invalid"}
    sub = row.get("tam_maps_subset") or {}
    maps_b64 = sub.get("maps_uint8_b64") or []
    tok_idxs = sub.get("token_indices") or []
    if not maps_b64 or not tok_idxs:
        return {"records": [], "skip_reason": "no_subset_maps"}
    h = int(row.get("map_h") or 0)
    w = int(row.get("map_w") or 0)
    if h == 0 or w == 0:
        return {"records": [], "skip_reason": "bad_map_shape"}

    cats_all = row.get("token_category") or []
    quads_all = row.get("quad") or []
    R = int(row.get("response_length") or 0)
    if R == 0 or len(cats_all) < R:
        return {"records": [], "skip_reason": "missing_categories"}

    # Decode subset maps + look up categories at the subset positions.
    tokens_all = row.get("tokens") or []
    tam_mass20_all = row.get("tam_mass_top20") or []
    tam_entropy_all = row.get("tam_entropy_norm") or []
    vd_all = row.get("vd") or []
    adv_all = row.get("adv") or []

    maps = []
    cats = []
    quads = []
    valid_idxs = []
    for b64, ti in zip(maps_b64, tok_idxs):
        if ti < 0 or ti >= R:
            continue
        try:
            m = _decode_uint8_map(b64, h, w)
        except Exception:  # noqa: BLE001
            continue
        maps.append(m)
        cats.append(cats_all[ti])
        quads.append(quads_all[ti] if ti < len(quads_all) else None)
        valid_idxs.append(ti)
    if not maps:
        return {"records": [], "skip_reason": "all_subset_decode_failed"}

    sample_id = row["id"]
    benchmark = row.get("benchmark")
    student_ckpt = row.get("student_ckpt") or "unknown"

    cfg = GateConfig(K=K, rho=rho, tau=tau, mode="main")
    weights, info = compute_weights(maps, cats, config=cfg, sample_id=sample_id)

    records = []
    for j, (ti, c, q, w_t, g, cov) in enumerate(zip(
        valid_idxs, cats, quads, weights, info["gate_fire"], info["coverage"],
    )):
        records.append({
            "sample_id":   sample_id,
            "benchmark":   benchmark,
            "student_ckpt": student_ckpt,
            "token_idx":   ti,
            "token":       tokens_all[ti] if ti < len(tokens_all) else None,
            "category":    c,
            "quad":        q if isinstance(q, int) else None,
            "vd":          vd_all[ti] if ti < len(vd_all) else None,
            "adv":         adv_all[ti] if ti < len(adv_all) else None,
            "tam_mass_top20":   tam_mass20_all[ti] if ti < len(tam_mass20_all) else None,
            "tam_entropy_norm": tam_entropy_all[ti] if ti < len(tam_entropy_all) else None,
            "weight":      float(w_t),
            "gate_fire":   int(g),
            "coverage":    None if math.isnan(cov) else float(cov),
            "tau":         tau,
        })
    return {"records": records,
            "E_x_size": info["E_x_size"],
            "n_C_local_positions_subset": info["n_C_local_positions"]}


def _aggregate(records: list[dict], cfg: GateConfig) -> dict:
    """Aggregate per-token records into category × quad × ckpt summary."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    by_quad: dict[int | None, list[dict]] = defaultdict(list)
    by_ckpt: dict[str, list[dict]] = defaultdict(list)
    by_cat_quad: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
        by_quad[r["quad"]].append(r)
        by_ckpt[r["student_ckpt"]].append(r)
        by_cat_quad[(r["category"], r["quad"])].append(r)

    def _summary(rs: list[dict]) -> dict:
        if not rs:
            return {"n": 0}
        n = len(rs)
        fires = sum(r["gate_fire"] for r in rs)
        covs = [r["coverage"] for r in rs if r["coverage"] is not None]
        mean_w = sum(r["weight"] for r in rs) / n
        return {
            "n":              n,
            "gate_fire_rate": fires / n,
            "mean_w":         mean_w,
            "coverage_n":     len(covs),
            "coverage_p25":   _percentile(covs, 25),
            "coverage_p50":   _percentile(covs, 50),
            "coverage_p75":   _percentile(covs, 75),
            "coverage_p95":   _percentile(covs, 95),
            "coverage_mean":  float(np.mean(covs)) if covs else float("nan"),
        }

    return {
        "config": {
            "K": cfg.K, "rho": cfg.rho, "tau": cfg.tau, "alpha": cfg.alpha,
            "C_local": list(cfg.C_local), "mode": cfg.mode,
        },
        "overall":      _summary(records),
        "by_category":  {c: _summary(rs) for c, rs in by_cat.items()},
        "by_quad":      {(str(q) if q is not None else "null"): _summary(rs)
                         for q, rs in by_quad.items()},
        "by_ckpt":      {c: _summary(rs) for c, rs in by_ckpt.items()},
        "by_cat_quad":  {f"{c}|q{q}": _summary(rs) for (c, q), rs in by_cat_quad.items()},
    }


def _suggest_alpha(gate_fire_rate: float, target_mean_w: float) -> float | None:
    """Pick α s.t. E[w_t] = 1 + α · gate_fire_rate ≈ target_mean_w."""
    if gate_fire_rate <= 1e-9:
        return None
    return max(0.0, (target_mean_w - 1.0) / gate_fire_rate)


def _tau_sweep(rows: list[dict], K: float, rho: float,
               tau_values: list[float], cfg_alpha: float,
               dump_records_path: Path | None = None) -> dict:
    """Run gate across multiple τ, report gate-fire rate at each.

    If `dump_records_path` is given, emit per-token records JSONL (one row per
    (sample, token, τ)) for downstream cell audits."""
    out: dict[float, dict] = {}
    fdump = None
    if dump_records_path is not None:
        dump_records_path.parent.mkdir(parents=True, exist_ok=True)
        fdump = dump_records_path.open("w")
    try:
        for tau in tau_values:
            records: list[dict] = []
            n_skip = Counter()
            for row in rows:
                res = _gate_one_row(row, K=K, rho=rho, tau=tau)
                if res.get("skip_reason"):
                    n_skip[res["skip_reason"]] += 1
                    continue
                records.extend(res["records"])
            if fdump is not None:
                for r in records:
                    fdump.write(json.dumps(r, ensure_ascii=False) + "\n")
            agg = _aggregate(records, GateConfig(K=K, rho=rho, tau=tau, alpha=cfg_alpha))
            agg["n_skipped"] = dict(n_skip)
            out[tau] = agg
    finally:
        if fdump is not None:
            fdump.close()
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step1a-jsonl", type=Path, required=True)
    ap.add_argument("--out-json",     type=Path, required=True)
    ap.add_argument("--out-txt",      type=Path, default=None)
    ap.add_argument("--tau-sweep",    type=str,
                    default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    ap.add_argument("--K",            type=float, default=0.20)
    ap.add_argument("--rho",          type=float, default=0.30)
    ap.add_argument("--alpha",        type=float, default=0.50,
                    help="alpha used for mean_w reporting only; actual α "
                         "recommendation is computed from gate-fire rate")
    ap.add_argument("--target-mean-w", type=float, default=1.10,
                    help="α calibration target — pick α s.t. mean(w_t)≈this")
    ap.add_argument("--target-gate-fire-rate", type=float, default=0.15,
                    help="τ recommendation target")
    ap.add_argument("--dump-records", type=Path, default=None,
                    help="If set, emit per-(sample,token,τ) records JSONL "
                         "for downstream cell audits (proper_noun|q3 etc.). "
                         "Includes token text + benchmark + vd/adv + tam scalars.")
    args = ap.parse_args(argv)

    tau_values = sorted(float(x) for x in args.tau_sweep.split(","))
    print(f">>> loading step1a JSONL: {args.step1a_jsonl}", file=sys.stderr)
    rows = _load_rows(args.step1a_jsonl)
    print(f">>> {len(rows)} rows", file=sys.stderr)

    print(f">>> τ sweep over {tau_values}, K={args.K}, ρ={args.rho}",
          file=sys.stderr)
    sweep = _tau_sweep(rows, K=args.K, rho=args.rho,
                        tau_values=tau_values, cfg_alpha=args.alpha,
                        dump_records_path=args.dump_records)

    # Build recommendation
    rec_tau = None
    rec_alpha = None
    rec_rationale = ""
    # Find τ whose pooled C_local gate-fire rate is closest to target
    cl = set(C_LOCAL_DEFAULT)
    target = args.target_gate_fire_rate
    best_diff = float("inf")
    for tau in tau_values:
        cl_summaries = [sweep[tau]["by_category"].get(c, {"n": 0})
                        for c in cl]
        cl_n = sum(s.get("n", 0) for s in cl_summaries)
        cl_fires = sum(s.get("n", 0) * s.get("gate_fire_rate", 0)
                       for s in cl_summaries)
        if cl_n == 0:
            continue
        cl_rate = cl_fires / cl_n
        diff = abs(cl_rate - target)
        if diff < best_diff:
            best_diff = diff
            rec_tau = tau
            # Suggest α from this τ's C_local gate-fire
            rec_alpha = _suggest_alpha(cl_rate, args.target_mean_w)
            rec_rationale = (f"C_local gate-fire rate {cl_rate:.3f} closest to "
                             f"target {target} at τ={tau}")

    report = {
        "step1a_jsonl":              str(args.step1a_jsonl),
        "K":                         args.K,
        "rho":                       args.rho,
        "tau_sweep":                 tau_values,
        "target_mean_w":             args.target_mean_w,
        "target_gate_fire_rate":     args.target_gate_fire_rate,
        "C_local":                   list(C_LOCAL_DEFAULT),
        "sweep":                     {str(k): v for k, v in sweep.items()},
        "recommendation": {
            "tau":         rec_tau,
            "alpha":       rec_alpha,
            "rationale":   rec_rationale,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f">>> wrote {args.out_json}", file=sys.stderr)

    if args.out_txt:
        lines: list[str] = []
        lines.append(f"# Step 3a TAM-Boost pre-flight — K={args.K}, ρ={args.rho}")
        lines.append(f"# step1a JSONL  = {args.step1a_jsonl}")
        lines.append(f"# n_rows         = {len(rows)}")
        lines.append(f"# C_local        = {list(C_LOCAL_DEFAULT)}")
        lines.append(f"# tau_sweep      = {tau_values}")
        lines.append(f"# target_mean_w  = {args.target_mean_w}")
        lines.append(f"# target_g_rate  = {args.target_gate_fire_rate}")
        lines.append("")

        for tau in tau_values:
            agg = sweep[tau]
            ov = agg["overall"]
            lines.append(f"## τ = {tau:.2f}")
            lines.append(f"  overall: n={ov.get('n',0)} "
                         f"gate_fire_rate={ov.get('gate_fire_rate',0):.4f}  "
                         f"mean_w={ov.get('mean_w',0):.4f}")

            lines.append(f"  {'category':<22} {'n':>6} {'fire_rate':>10}  "
                         f"{'mean_w':>8}  {'cov_p50':>8}  {'cov_p95':>8}")
            for c in sorted(agg["by_category"].keys()):
                s = agg["by_category"][c]
                if s.get("n", 0) == 0:
                    continue
                cl_mark = " ★" if c in cl else ""
                lines.append(f"  {c:<22} {s.get('n',0):>6}  "
                             f"{s.get('gate_fire_rate',0):>9.4f}  "
                             f"{s.get('mean_w',0):>+8.3f}  "
                             f"{s.get('coverage_p50', float('nan')):>+8.3f}  "
                             f"{s.get('coverage_p95', float('nan')):>+8.3f}"
                             f"{cl_mark}")
            lines.append("")
            # by quad cross-ref (q0/q1 should over-fire; q3 should under-fire)
            lines.append(f"  {'quad':<10} {'n':>6} {'fire_rate':>10}  "
                         f"{'cov_p50':>8}")
            for q in ("0", "1", "2", "3", "null"):
                s = agg["by_quad"].get(q, {"n": 0})
                if s.get("n", 0) == 0:
                    continue
                lines.append(f"  q={q:<8} {s.get('n',0):>6}  "
                             f"{s.get('gate_fire_rate',0):>9.4f}  "
                             f"{s.get('coverage_p50', float('nan')):>+8.3f}")
            lines.append("")

        lines.append("## RECOMMENDATION")
        lines.append(f"  τ      = {rec_tau}")
        lines.append(f"  α      = {rec_alpha:.4f}" if rec_alpha is not None else "  α = (n/a)")
        lines.append(f"  reason = {rec_rationale}")
        lines.append("")
        lines.append("  Caveat: E_x estimated from K=20 stratified subset maps per")
        lines.append("  (sample,ckpt), NOT full response. If full-response calibration")
        lines.append("  is needed, re-run step1a with full _response_maps_b64 export.")

        args.out_txt.parent.mkdir(parents=True, exist_ok=True)
        args.out_txt.write_text("\n".join(lines) + "\n")
        print(f">>> wrote summary: {args.out_txt}", file=sys.stderr)
    print(f">>> recommendation: τ={rec_tau}, α={rec_alpha}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
