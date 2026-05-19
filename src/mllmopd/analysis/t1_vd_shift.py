"""T1 VD-shift analysis: did vanilla OPD push the student's visual dependency
distribution toward high-VD tokens?

Reads three `aggregate_vd.py` outputs:
  - teacher baseline (post-G1 MMR1-7B-RL on opd_target, the canonical
    `vd_summary.sysprompt.json`),
  - T1-2 student (FullTeacher OPD, scored on its own rollouts),
  - T1-3 student (BlankTeacher OPD, scored on its own rollouts).

For each (arm, benchmark) cell we compute the %tokens and %NLL-mass that
land in each of the 5 VD bins, plus the headline collapse to
`high + very_high`. The three headline shifts emitted as scalars:

    vd_shift_t1_2_vs_teacher   pp(T1-2 high+very_high %NLL) − pp(teacher ...)
    vd_shift_t1_3_vs_teacher   pp(T1-3 high+very_high %NLL) − pp(teacher ...)
    vd_shift_t1_2_vs_t1_3      pp(T1-2 ...) − pp(T1-3 ...)

The last is the T1-paper headline: if FullTeacher OPD genuinely pushes
more vision-conditioned reasoning into the student than BlankTeacher OPD,
this difference is positive and concentrated in the math/chart
benchmarks where Finding 2 already showed the teacher's NLL mass clusters
on high-VD tokens.

LIMITATION (read before quoting these numbers): this is a STUDENT-side VD
measurement. `vd(t) = logp_full(t) − logp_blank(t)` is computed on the
student's OWN predictions, so it depends on the student's output
distribution as well as on its internal vision reliance. If T1-2 and T1-3
diverge in scaffolding (`<think>` length, hedging style, choice of
phrasing) the VD bin shares can move because the tokens being scored
changed, not because the model became more visual. Cross-check the
absolute token counts per bin (`overall.{bin}.n`) — a 5pp NLL-mass shift
that comes with a 3x change in token count is mostly an output-shape
artifact, while a 5pp shift at near-constant token count is a real
internal reweighting.

Usage:
    python -m mllmopd.analysis.t1_vd_shift \\
        --teacher-baseline runs/audit/level1_v4_sysprompt_fixed/vd_summary.sysprompt.json \\
        --t1-2-summary    runs/audit/<run_id>/vd_summary.T1_2.json \\
        --t1-3-summary    runs/audit/<run_id>/vd_summary.T1_3.json \\
        [--out-json       runs/audit/<run_id>/vd_shift_t1.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# VD bin labels — MUST match aggregate_vd._VD_BINS verbatim. We don't import
# them so that this module stays a pure consumer of the JSON summaries; if
# the bin labels are renamed in aggregate_vd the assertion below will fire
# on every input and the operator will know to keep the two in sync.
_VD_BIN_LABELS = [
    "very_low (vd ≤ -1)",
    "low (-1 < vd ≤ 0)",
    "neutral (0 < vd ≤ 0.5)",
    "high (0.5 < vd ≤ 2.0)",
    "very_high (vd > 2)",
]
_HIGH_BINS = ["high (0.5 < vd ≤ 2.0)", "very_high (vd > 2)"]


def _bin_shares(bin_dict: dict) -> dict:
    """Convert {bin: {n, nll_full}} → {bin: {pct_tokens, pct_nll}} +
    high_plus_very_high collapse. Handles the empty-cell case (0 tokens)
    by returning None for percentages so downstream code can flag rather
    than divide by zero."""
    missing = [b for b in _VD_BIN_LABELS if b not in bin_dict]
    if missing:
        sys.exit(f"!! summary missing VD bins {missing}; out of sync with aggregate_vd._VD_BINS")
    tot_n = sum(bin_dict[b]["n"] for b in _VD_BIN_LABELS)
    tot_nll = sum(bin_dict[b]["nll_full"] for b in _VD_BIN_LABELS)
    out: dict = {"_tot_n": tot_n, "_tot_nll": tot_nll}
    for b in _VD_BIN_LABELS:
        n = bin_dict[b]["n"]
        nll = bin_dict[b]["nll_full"]
        out[b] = {
            "n": n,
            "nll_full": nll,
            "pct_tokens": (n / tot_n * 100) if tot_n else None,
            "pct_nll": (nll / tot_nll * 100) if tot_nll > 0 else None,
        }
    hi_n = sum(bin_dict[b]["n"] for b in _HIGH_BINS)
    hi_nll = sum(bin_dict[b]["nll_full"] for b in _HIGH_BINS)
    out["high_plus_very_high_pct_tokens"] = (hi_n / tot_n * 100) if tot_n else None
    out["high_plus_very_high_pct_nll"] = (hi_nll / tot_nll * 100) if tot_nll > 0 else None
    return out


def _shift_pp(a: float | None, b: float | None) -> float | None:
    """a − b in pp, propagating None if either side is undefined."""
    if a is None or b is None:
        return None
    return a - b


def _build_per_arm_shares(summary: dict) -> dict:
    """Apply `_bin_shares` to `overall` + every benchmark in `by_bench`."""
    cells: dict = {"overall": _bin_shares(summary["overall"])}
    for bench, bin_dict in summary.get("by_bench", {}).items():
        cells[bench] = _bin_shares(bin_dict)
    return cells


def _print_markdown_table(per_arm: dict[str, dict], scopes: list[str]) -> None:
    """Print a per-scope markdown table: arm vs %tokens / %NLL in high+v_high."""
    print()
    print("## T1 VD shift: high+very_high bin shares (% of NLL mass)")
    print()
    header = "| scope | teacher_baseline | T1-2 | T1-3 | T1-2 − teacher | T1-3 − teacher | T1-2 − T1-3 |"
    print(header)
    print("|" + "|".join(["---"] * 7) + "|")
    for scope in scopes:
        t_cell = per_arm["teacher_baseline"].get(scope)
        a_cell = per_arm["T1_2"].get(scope)
        b_cell = per_arm["T1_3"].get(scope)
        if t_cell is None or a_cell is None or b_cell is None:
            print(f"| {scope} | (missing) | (missing) | (missing) | - | - | - |")
            continue
        t_nll = t_cell["high_plus_very_high_pct_nll"]
        a_nll = a_cell["high_plus_very_high_pct_nll"]
        b_nll = b_cell["high_plus_very_high_pct_nll"]
        s_a_t = _shift_pp(a_nll, t_nll)
        s_b_t = _shift_pp(b_nll, t_nll)
        s_a_b = _shift_pp(a_nll, b_nll)

        def _fmt(v):
            return f"{v:6.2f}%" if v is not None else "  n/a "

        def _fmt_pp(v):
            return f"{v:+6.2f}pp" if v is not None else "  n/a "

        print(f"| {scope} | {_fmt(t_nll)} | {_fmt(a_nll)} | {_fmt(b_nll)} | "
              f"{_fmt_pp(s_a_t)} | {_fmt_pp(s_b_t)} | {_fmt_pp(s_a_b)} |")

    print()
    print("## T1 VD shift: high+very_high bin shares (% of tokens)")
    print()
    print(header)
    print("|" + "|".join(["---"] * 7) + "|")
    for scope in scopes:
        t_cell = per_arm["teacher_baseline"].get(scope)
        a_cell = per_arm["T1_2"].get(scope)
        b_cell = per_arm["T1_3"].get(scope)
        if t_cell is None or a_cell is None or b_cell is None:
            print(f"| {scope} | (missing) | (missing) | (missing) | - | - | - |")
            continue
        t_tok = t_cell["high_plus_very_high_pct_tokens"]
        a_tok = a_cell["high_plus_very_high_pct_tokens"]
        b_tok = b_cell["high_plus_very_high_pct_tokens"]
        s_a_t = _shift_pp(a_tok, t_tok)
        s_b_t = _shift_pp(b_tok, t_tok)
        s_a_b = _shift_pp(a_tok, b_tok)

        def _fmt(v):
            return f"{v:6.2f}%" if v is not None else "  n/a "

        def _fmt_pp(v):
            return f"{v:+6.2f}pp" if v is not None else "  n/a "

        print(f"| {scope} | {_fmt(t_tok)} | {_fmt(a_tok)} | {_fmt(b_tok)} | "
              f"{_fmt_pp(s_a_t)} | {_fmt_pp(s_b_t)} | {_fmt_pp(s_a_b)} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--teacher-baseline", type=Path,
        default=Path("runs/audit/level1_v4_sysprompt_fixed/vd_summary.sysprompt.json"),
        help="Teacher post-G1 VD distribution on opd_target (baseline anchor).",
    )
    ap.add_argument("--t1-2-summary", required=True, type=Path,
                    help="vd_summary.T1_2.json — FullTeacher arm student VD.")
    ap.add_argument("--t1-3-summary", required=True, type=Path,
                    help="vd_summary.T1_3.json — BlankTeacher arm student VD.")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Write the full comparison table as JSON. Defaults "
                         "to vd_shift_t1.json next to the T1-2 summary.")
    ap.add_argument("--print-table", action=argparse.BooleanOptionalAction, default=True,
                    help="Also print a markdown summary table to stdout.")
    args = ap.parse_args()

    if args.out_json is None:
        args.out_json = args.t1_2_summary.parent / "vd_shift_t1.json"

    for p in (args.teacher_baseline, args.t1_2_summary, args.t1_3_summary):
        if not p.exists():
            sys.exit(f"!! missing input summary: {p}")

    per_arm: dict[str, dict] = {
        "teacher_baseline": _build_per_arm_shares(json.loads(args.teacher_baseline.read_text())),
        "T1_2": _build_per_arm_shares(json.loads(args.t1_2_summary.read_text())),
        "T1_3": _build_per_arm_shares(json.loads(args.t1_3_summary.read_text())),
    }

    # Headline scalars: pp-shift in high+very_high %NLL on the `overall` cell.
    teacher_hi_nll = per_arm["teacher_baseline"]["overall"]["high_plus_very_high_pct_nll"]
    t1_2_hi_nll = per_arm["T1_2"]["overall"]["high_plus_very_high_pct_nll"]
    t1_3_hi_nll = per_arm["T1_3"]["overall"]["high_plus_very_high_pct_nll"]
    teacher_hi_tok = per_arm["teacher_baseline"]["overall"]["high_plus_very_high_pct_tokens"]
    t1_2_hi_tok = per_arm["T1_2"]["overall"]["high_plus_very_high_pct_tokens"]
    t1_3_hi_tok = per_arm["T1_3"]["overall"]["high_plus_very_high_pct_tokens"]

    headlines = {
        # %NLL-mass headlines — the canonical Finding-2 frame.
        "vd_shift_t1_2_vs_teacher_pct_nll": _shift_pp(t1_2_hi_nll, teacher_hi_nll),
        "vd_shift_t1_3_vs_teacher_pct_nll": _shift_pp(t1_3_hi_nll, teacher_hi_nll),
        "vd_shift_t1_2_vs_t1_3_pct_nll": _shift_pp(t1_2_hi_nll, t1_3_hi_nll),
        # %tokens headlines — sanity check on whether the shift is just
        # output-shape drift or actually a reweighting of which tokens
        # exist at all.
        "vd_shift_t1_2_vs_teacher_pct_tokens": _shift_pp(t1_2_hi_tok, teacher_hi_tok),
        "vd_shift_t1_3_vs_teacher_pct_tokens": _shift_pp(t1_3_hi_tok, teacher_hi_tok),
        "vd_shift_t1_2_vs_t1_3_pct_tokens": _shift_pp(t1_2_hi_tok, t1_3_hi_tok),
        # Absolute high+very_high shares per arm — useful for the writeup.
        "teacher_baseline_high_plus_very_high_pct_nll": teacher_hi_nll,
        "T1_2_high_plus_very_high_pct_nll": t1_2_hi_nll,
        "T1_3_high_plus_very_high_pct_nll": t1_3_hi_nll,
        "teacher_baseline_high_plus_very_high_pct_tokens": teacher_hi_tok,
        "T1_2_high_plus_very_high_pct_tokens": t1_2_hi_tok,
        "T1_3_high_plus_very_high_pct_tokens": t1_3_hi_tok,
        # Token counts per arm — limitation guard (see module docstring).
        "teacher_baseline_total_tokens": per_arm["teacher_baseline"]["overall"]["_tot_n"],
        "T1_2_total_tokens": per_arm["T1_2"]["overall"]["_tot_n"],
        "T1_3_total_tokens": per_arm["T1_3"]["overall"]["_tot_n"],
    }

    out: dict = {
        "inputs": {
            "teacher_baseline": str(args.teacher_baseline),
            "t1_2_summary": str(args.t1_2_summary),
            "t1_3_summary": str(args.t1_3_summary),
        },
        "headlines": headlines,
        "per_arm": per_arm,
        "limitation_note": (
            "Student-side VD: vd = logp_full(t) - logp_blank(t) computed on "
            "the student's OWN predictions. If T1-2 and T1-3 emit different "
            "output structures, VD bin shifts can reflect output-shape "
            "drift rather than internal visual reliance. Inspect "
            "per_arm.*.overall._tot_n: a large NLL-mass shift paired with a "
            "large token-count shift is mostly output drift, not a real "
            "internal reweighting."
        ),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, default=str))
    print(f">>> wrote VD-shift comparison -> {args.out_json}", file=sys.stderr)

    if args.print_table:
        # Print overall first, then each benchmark that appears in ALL three
        # arms (an arm-specific benchmark would just confuse the table).
        common_benches = sorted(
            set(per_arm["teacher_baseline"].keys())
            & set(per_arm["T1_2"].keys())
            & set(per_arm["T1_3"].keys())
            - {"overall"}
        )
        _print_markdown_table(per_arm, ["overall"] + common_benches)
        print()
        print("## Headlines")
        for k, v in headlines.items():
            if isinstance(v, float):
                print(f"  {k}: {v:+.3f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
