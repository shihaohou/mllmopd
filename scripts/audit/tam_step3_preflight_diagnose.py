"""Diagnose q3 over-fire in tam_step3_preflight output.

Reads tam_step3_preflight_v0.json + (optional) tam_step1a_full_v013.json
and reports:
  1. Selection bias check: C_local fraction per quad across τ values
  2. Within-C_local fire rate per quad (corrects for the bias)
  3. Per (category × quad) fire rate at chosen τ
  4. Coverage distribution within (category × quad) — does q3's TAM hit E_x
     for the same REASON as q0's? (compare cov_mean / cov_p95)
  5. (if step1a-json given) Cross-reference: TAM scalars per quad — is q3's
     TAM intrinsically more diffuse / more concentrated than q0's?

Usage::
    python scripts/audit/tam_step3_preflight_diagnose.py \\
        --preflight-json runs/analysis/tam_step3_preflight_v0.json \\
        [--step1a-json   runs/analysis/tam_step1a_full_v013.json] \\
        [--tau 0.5,0.7,0.8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


C_LOCAL = ("content_noun", "proper_noun", "visual_attribute")


def _cell(s, key):
    return s.get(key, {"n": 0})


def _diag_one_tau(tau_str: str, sweep: dict, taus: list[str]) -> list[str]:
    s = sweep[tau_str]
    lines: list[str] = []
    lines.append(f"\n## τ = {tau_str}")

    # Per-quad bias decomposition
    lines.append(f"  {'quad':<4} {'n_tot':>6} {'n_Cloc':>7} {'%Cloc':>6} "
                 f"{'fire_pooled':>11} {'fire_Cloc':>9} {'cov_p50_Cloc':>13}")
    for q in (0, 1, 2, 3):
        q_block = s["by_quad"].get(str(q), {"n": 0})
        n_tot = q_block.get("n", 0)
        if n_tot == 0:
            continue
        n_Cloc = sum(_cell(s["by_cat_quad"], f"{c}|q{q}").get("n", 0)
                     for c in C_LOCAL)
        fires_tot = n_tot * q_block.get("gate_fire_rate", 0)
        fire_Cloc = (fires_tot / n_Cloc) if n_Cloc else 0
        frac_Cloc = n_Cloc / n_tot if n_tot else 0
        cov_p50 = q_block.get("coverage_p50", float("nan"))
        lines.append(f"  q{q:<3} {n_tot:>6d} {n_Cloc:>7d} {frac_Cloc:>5.1%}  "
                     f"{q_block.get('gate_fire_rate',0):>10.3f}  "
                     f"{fire_Cloc:>8.3f}  {cov_p50:>+13.3f}")
    return lines


def _diag_cat_quad(tau_str: str, sweep: dict) -> list[str]:
    s = sweep[tau_str]
    lines: list[str] = []
    lines.append(f"\n  Per (category × quad) at τ={tau_str}:")
    lines.append(f"  {'cell':<22} {'n':>5} {'fire':>6} {'cov_mean':>8} "
                 f"{'cov_p50':>8} {'cov_p95':>8}")
    for c in C_LOCAL:
        for q in (0, 1, 2, 3):
            key = f"{c}|q{q}"
            cell = s["by_cat_quad"].get(key, {"n": 0})
            n = cell.get("n", 0)
            if n == 0:
                continue
            lines.append(f"  {key:<22} {n:>5d}  "
                         f"{cell.get('gate_fire_rate',0):>5.3f}  "
                         f"{cell.get('coverage_mean',float('nan')):>+8.3f}  "
                         f"{cell.get('coverage_p50',float('nan')):>+8.3f}  "
                         f"{cell.get('coverage_p95',float('nan')):>+8.3f}")
    return lines


def _diag_step1a(step1a_path: Path) -> list[str]:
    """If we have Step 1a analyzer output, look for per-quad TAM scalar
    breakdowns that might explain coverage uniformity."""
    lines: list[str] = []
    try:
        data = json.loads(step1a_path.read_text())
    except Exception as e:  # noqa: BLE001
        lines.append(f"  (failed to load {step1a_path}: {e!r})")
        return lines
    # Step 1a analyzer JSON usually has by_quad blocks with mean tam_mass_top20
    # and tam_entropy_norm — if present, compare q0 vs q3.
    keys = ["by_quad", "per_quad", "quad_breakdown"]
    block = None
    for k in keys:
        if k in data:
            block = data[k]
            break
    if not block:
        # Search nested
        for k, v in data.items():
            if isinstance(v, dict) and any(qk in str(v.keys()) for qk in
                                            ["q0", "q1", "quad"]):
                lines.append(f"  found candidate block under '{k}': keys = "
                             f"{list(v.keys())[:5]}")
        lines.append("  no by_quad/per_quad block found in step1a JSON; "
                     "skipping TAM-scalar-per-quad cross-ref")
        return lines
    lines.append("\n## Step 1a TAM scalars per quad (cross-ref):")
    lines.append(f"  {'quad':<6} {'n':>6} {'mass_top20':>11} {'entropy_norm':>13}")
    for q, qblock in block.items():
        if isinstance(qblock, dict):
            n = qblock.get("n", qblock.get("count", 0))
            m20 = qblock.get("tam_mass_top20_mean",
                              qblock.get("mean_tam_mass_top20",
                                          qblock.get("mass_top20", "?")))
            ent = qblock.get("tam_entropy_norm_mean",
                              qblock.get("mean_tam_entropy_norm",
                                          qblock.get("entropy_norm", "?")))
            lines.append(f"  {str(q):<6} {n!s:>6} {m20!s:>11} {ent!s:>13}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preflight-json", type=Path, required=True)
    ap.add_argument("--step1a-json", type=Path, default=None)
    ap.add_argument("--tau", type=str, default="0.5,0.7,0.8")
    ap.add_argument("--out-txt", type=Path, default=None)
    args = ap.parse_args(argv)

    data = json.loads(args.preflight_json.read_text())
    sweep = data["sweep"]
    available_taus = sorted(sweep.keys(), key=float)
    requested = [t.strip() for t in args.tau.split(",")]
    target_taus = [t for t in requested if t in sweep]
    missing = [t for t in requested if t not in sweep]
    if missing:
        print(f"WARN: requested τ {missing} not in sweep; available = {available_taus}",
              file=sys.stderr)
    if not target_taus:
        target_taus = available_taus

    lines: list[str] = []
    lines.append(f"# Step 3a pre-flight q3-over-fire diagnostic")
    lines.append(f"# preflight: {args.preflight_json}")
    lines.append(f"# C_local: {list(C_LOCAL)}")

    for tau in target_taus:
        lines.extend(_diag_one_tau(tau, sweep, available_taus))
        lines.extend(_diag_cat_quad(tau, sweep))

    if args.step1a_json:
        lines.extend(_diag_step1a(args.step1a_json))

    out = "\n".join(lines) + "\n"
    if args.out_txt:
        args.out_txt.parent.mkdir(parents=True, exist_ok=True)
        args.out_txt.write_text(out)
        print(f">>> wrote {args.out_txt}", file=sys.stderr)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
