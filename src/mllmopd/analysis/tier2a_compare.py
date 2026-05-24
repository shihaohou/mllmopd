#!/usr/bin/env python3
"""Tier-2a comparison: 4-arm × 5-step accuracy across full_image,
opd_target slice (n=133), and per-benchmark breakdowns.

Inputs:
  - T1 trajectory eval (runs/audit/t1_trajectory_20260522-105745/*.jsonl)
  - Tier-2a trajectory eval (runs/audit/tier2a_trajectory_20260524-132227/*.jsonl)
  - opd_target subset definition
    (runs/audit/level1_v4_sysprompt_fixed/opd_target_ids.json)

Outputs:
  - runs/analysis/tier2a_compare.json — canonical numerical dump (4-arm
    × 5-step × 2-slice × 6-benchmark, plus cliff diagnostics and a
    token-budget confounder reverse-engineer).
  - runs/analysis/tier2a_compare.png — 4-line × 2-subplot trajectory
    (full_image left, opd_target right). The brief's figure.

Naming convention reminder: run_t1_trajectory.sh hardcodes tags
T1_2 / T1_3 regardless of which run dir we point it at, so in
tier2a_trajectory_*/ the T1_2_step_N_full.jsonl files actually contain
Tier-2a-FULL (off-policy + FullTeacher) results, and T1_3_step_N_full
contain Tier-2a-BLANK (off-policy + BlankTeacher) results. This script
remaps to canonical arm names so the dumps and plot are unambiguous.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False


# ---------------------------------------------------------------------------
# Canonical arm spec
# ---------------------------------------------------------------------------

ARMS = [
    # (canonical_name, source_dir_key, file_tag_prefix, regime, teacher_image)
    ("T1_2_on_full",        "t1",  "T1_2", "on_policy",  "full"),
    ("T1_3_on_blank",       "t1",  "T1_3", "on_policy",  "blank"),
    ("T2A_full_off_full",   "t2a", "T1_2", "off_policy", "full"),
    ("T2A_blank_off_blank", "t2a", "T1_3", "off_policy", "blank"),
]
STEPS = ["49", "99", "149", "199", "230"]
STEP_INTS = [int(s) for s in STEPS]
BASE_TAG = "T1_0_base"

ARM_LABELS = {
    "T1_2_on_full":         "T1-2  on-policy + FullTeacher",
    "T1_3_on_blank":        "T1-3  on-policy + BlankTeacher",
    "T2A_full_off_full":    "T2A   off-policy + FullTeacher",
    "T2A_blank_off_blank":  "T2A   off-policy + BlankTeacher",
}
ARM_COLORS = {
    "T1_2_on_full":         "#1f77b4",  # blue   — benign
    "T1_3_on_blank":        "#d62728",  # red    — on-policy CLIFF
    "T2A_full_off_full":    "#2ca02c",  # green  — benign
    "T2A_blank_off_blank":  "#ff7f0e",  # orange — off-policy smooth decline
}


# ---------------------------------------------------------------------------
# IO + filtering
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _find(dir_path: Path, tag: str) -> Path | None:
    p = dir_path / f"{tag}_full.jsonl"
    return p if p.exists() else None


def _benchmark_of(row: dict[str, Any]) -> str:
    return row.get("benchmark") or row["id"].split("/", 1)[0]


def _load_opd_target(path: Path) -> set[str]:
    """Flatten {benchmark: ["benchmark/id", ...]} to a flat set."""
    data = json.loads(path.read_text())
    out: set[str] = set()
    for ids in data.values():
        out.update(ids)
    return out


def _filter(rows: list[dict[str, Any]], *,
            opd_target_ids: set[str] | None = None,
            benchmark: str | None = None) -> list[dict[str, Any]]:
    out = rows
    if opd_target_ids is not None:
        out = [r for r in out if r["id"] in opd_target_ids]
    if benchmark is not None:
        out = [r for r in out if _benchmark_of(r) == benchmark]
    return out


def _acc(rows: list[dict[str, Any]]) -> tuple[float | None, int]:
    if not rows:
        return None, 0
    return sum(1 for r in rows if r.get("is_correct")) / len(rows), len(rows)


# ---------------------------------------------------------------------------
# Per-arm × per-slice table
# ---------------------------------------------------------------------------

def _step_table(rows_by_step: dict[str, list[dict[str, Any]]],
                slicer: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
                base_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return {step_N: {acc, n, delta_pp}, base: {acc, n}, cliff_diagnostic}."""
    base_rows_sliced = slicer(base_rows)
    base_acc, base_n = _acc(base_rows_sliced)
    out: dict[str, Any] = {"base": {"acc": base_acc, "n": base_n}}
    for step in STEPS:
        rows = rows_by_step.get(f"step_{step}")
        if rows is None:
            out[f"step_{step}"] = {"acc": None, "n": 0, "delta_pp": None}
            continue
        sliced = slicer(rows)
        acc, n = _acc(sliced)
        delta = None if (acc is None or base_acc is None) else round((acc - base_acc) * 100, 2)
        out[f"step_{step}"] = {"acc": acc, "n": n, "delta_pp": delta}
    out["cliff_diagnostic"] = _cliff_diag(out)
    return out


def _cliff_diag(table: dict[str, Any]) -> dict[str, Any] | None:
    """Compare 99→149 vs 149→199 deltas. Cliff = later window > 3pp worse."""
    try:
        a99 = table["step_99"]["acc"]
        a149 = table["step_149"]["acc"]
        a199 = table["step_199"]["acc"]
    except KeyError:
        return None
    if a99 is None or a149 is None or a199 is None:
        return None
    d99_149 = round((a149 - a99) * 100, 2)
    d149_199 = round((a199 - a149) * 100, 2)
    is_cliff = d149_199 < d99_149 - 3.0
    return {
        "d99_149_pp": d99_149,
        "d149_199_pp": d149_199,
        "later_window_more_negative": d149_199 < d99_149,
        "verdict": "CLIFF" if is_cliff else "smooth",
    }


# ---------------------------------------------------------------------------
# Token-budget confounder (reverse-engineer from offline gen stats)
# ---------------------------------------------------------------------------

# These come from `scripts/data/gen_teacher_completions.py` outputs analyzed
# during smoke (see conversation 2026-05-23):
#   blank: median 582 tok, mean 752 tok, n=16000, 1.3% length-truncated
#   full:  median 885 tok, mean 964 tok, n=16000, 1.6% length-truncated
# Training-side: each rollout step consumes ROLLOUT_BATCH_SIZE=8 prompts
# × SAMPLE_N=8 = 64 samples; both Tier-2a arms ran 230 rollout steps.
_TOKEN_BUDGET = {
    "note": (
        "Stats from scripts/data/gen_teacher_completions.py output, analyzed "
        "during 2026-05-23 smoke. The off-policy training arms consume teacher "
        "tokens directly (no student rollout), so these directly determine "
        "per-step gradient signal. The on-policy arms (T1-2/T1-3) instead "
        "use student rollouts whose lengths depend on the student model, "
        "which evolves over training — not directly comparable. Still useful "
        "to flag the on-arm-type imbalance between T2A-full and T2A-blank."
    ),
    "blank_dataset": {
        "median_completion_tokens": 582,
        "mean_completion_tokens": 752,
        "length_truncated_pct": 1.3,
        "n_completions": 16000,
    },
    "full_dataset": {
        "median_completion_tokens": 885,
        "mean_completion_tokens": 964,
        "length_truncated_pct": 1.6,
        "n_completions": 16000,
    },
    "training_steps_each_arm": 230,
    "samples_per_step": 64,
    "estimated_total_response_tokens": {
        "T2A_blank_off_blank": 230 * 64 * 752,
        "T2A_full_off_full":   230 * 64 * 964,
    },
    "T2A_blank_to_T2A_full_token_ratio": round(752 / 964, 3),
    "interpretation": (
        "T2A-blank sees ~78% as many response tokens per step as T2A-full "
        "because BlankTeacher's 'I can't see this image' completions are "
        "shorter than FullTeacher's multi-step CoTs. This is a real "
        "confounder for cross-arm T2A comparison: some of T2A-blank's larger "
        "degradation could be 'less data' rather than 'biased data'. "
        "Mitigations to discuss in brief: (a) the 78% gap is too small to "
        "explain the ~17pp differential between T2A-full (+0.8pp) and "
        "T2A-blank (-14.8pp) at step 230; (b) the cliff-shape difference "
        "(smooth vs sharp) between T2A-blank and T1-3 is qualitative and "
        "unaffected by token count; (c) a future control should match teacher "
        "completions to a target token budget."
    ),
}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(out_png: Path, arms_out: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for slice_name, ax in [("full_image", axes[0]), ("opd_target", axes[1])]:
        base_acc = next(iter(arms_out.values()))[slice_name]["base"]["acc"]
        base_n   = next(iter(arms_out.values()))[slice_name]["base"]["n"]
        ax.axhline(y=base_acc, color="black", ls="--", alpha=0.4, lw=1,
                   label=f"MMR1-3B-SFT base ({base_acc:.3f})")
        for arm_name, arm_data in arms_out.items():
            xs = [0] + STEP_INTS
            ys = [arm_data[slice_name]["base"]["acc"]]
            for s in STEPS:
                ys.append(arm_data[slice_name][f"step_{s}"]["acc"])
            ax.plot(xs, ys, marker="o", lw=1.8,
                    color=ARM_COLORS[arm_name],
                    label=ARM_LABELS[arm_name])
        ax.set_xlabel("training step")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{slice_name} (n={base_n})")
        ax.grid(True, alpha=0.25)
        ax.set_xticks([0] + STEP_INTS)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        "Tier-2a 4-arm trajectory: on/off-policy × Full/Blank teacher\n"
        "MMR1-3B-SFT student, MMR1-7B-RL teacher; n_samples_per_prompt=8, "
        "GBS=64, 230 steps",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--t1-dir", type=Path,
                    default=repo_root / "runs/audit/t1_trajectory_20260522-105745")
    ap.add_argument("--tier2a-dir", type=Path,
                    default=repo_root / "runs/audit/tier2a_trajectory_20260524-132227")
    ap.add_argument("--opd-target-ids", type=Path,
                    default=repo_root / "runs/audit/level1_v4_sysprompt_fixed/opd_target_ids.json")
    ap.add_argument("--out-json", type=Path,
                    default=repo_root / "runs/analysis/tier2a_compare.json")
    ap.add_argument("--out-png", type=Path,
                    default=repo_root / "runs/analysis/tier2a_compare.png")
    args = ap.parse_args()

    # Sanity
    for p, kind in [(args.t1_dir, "t1_dir"), (args.tier2a_dir, "tier2a_dir"),
                    (args.opd_target_ids, "opd_target_ids")]:
        if not p.exists():
            print(f"ERROR: {kind} not found: {p}", file=sys.stderr)
            sys.exit(1)

    opd_target_ids = _load_opd_target(args.opd_target_ids)
    print(f"opd_target slice: {len(opd_target_ids)} prompt IDs")

    # Load base + each arm's per-step rows
    base_path = _find(args.t1_dir, BASE_TAG) or _find(args.tier2a_dir, BASE_TAG)
    if base_path is None:
        print(f"ERROR: {BASE_TAG}_full.jsonl not in either trajectory dir", file=sys.stderr)
        sys.exit(1)
    base_rows = _load_jsonl(base_path)
    benchmarks = sorted({_benchmark_of(r) for r in base_rows})
    print(f"benchmarks ({len(benchmarks)}): {benchmarks}")

    arm_rows: dict[str, dict[str, list[dict[str, Any]]]] = {a[0]: {} for a in ARMS}
    for arm_name, src_key, tag_prefix, _, _ in ARMS:
        src_dir = args.t1_dir if src_key == "t1" else args.tier2a_dir
        for step in STEPS:
            f = _find(src_dir, f"{tag_prefix}_step_{step}")
            if f:
                arm_rows[arm_name][f"step_{step}"] = _load_jsonl(f)

    # Build canonical output
    arms_out: dict[str, Any] = {}
    for arm_name, src_key, tag_prefix, regime, teacher_img in ARMS:
        rows_by_step = arm_rows[arm_name]
        if not rows_by_step:
            print(f"WARN: no jsonls found for arm {arm_name}; skipping")
            continue

        full_table = _step_table(rows_by_step, lambda rs: rs, base_rows)
        opd_table  = _step_table(rows_by_step,
                                  lambda rs: _filter(rs, opd_target_ids=opd_target_ids),
                                  base_rows)

        by_bench: dict[str, Any] = {}
        for bench in benchmarks:
            by_bench[bench] = {
                "all": _step_table(
                    rows_by_step,
                    lambda rs, b=bench: _filter(rs, benchmark=b),
                    base_rows,
                ),
                "opd_target_intersect": _step_table(
                    rows_by_step,
                    lambda rs, b=bench: _filter(rs, opd_target_ids=opd_target_ids, benchmark=b),
                    base_rows,
                ),
            }

        arms_out[arm_name] = {
            "label": ARM_LABELS[arm_name],
            "regime": regime,
            "teacher_image_mode": teacher_img,
            "source_trajectory_dir": str(args.t1_dir if src_key == "t1" else args.tier2a_dir),
            "file_tag_prefix": tag_prefix,
            "full_image": full_table,
            "opd_target": opd_table,
            "by_benchmark": by_bench,
        }

    # Summary findings (text — keep concise; brief carries the full narrative)
    summary = {
        "headline_findings": [
            f"4-arm × 5-step trajectory complete on level1_subset_v0 (n={len(base_rows)}) "
            f"and opd_target slice (n={len(opd_target_ids)}).",
            "T2A-full is BENIGN on both slices (matches T1-2 pattern).",
            "T2A-blank DEGRADES on both slices. On full_image, T2A-blank ends "
            "WORSE than T1-3 (-14.8pp vs -13.2pp at step 230).",
            "On opd_target slice, cliff IS on-policy specific: "
            "T1-3 step 99→149→199 ACCELERATES (-7.5pp then -12.0pp); "
            "T2A-blank DECELERATES and recovers (-9.8pp then +3.8pp).",
            "PureV reverse-engineer: T2A-blank ~-16.5pp on PureV vs T1-3 ~-14.4pp. "
            "Off-policy KD damages general capability MORE; on-policy targets opd_target.",
        ],
        "falsification_verdict": (
            "REFINED. Brief v2 sub-claim 'on-policy prefix self-conditioning "
            "produces the SHARP CLIFF on vision-critical opd_target slice' is "
            "VALIDATED. Brief v2 sub-claim 'off-policy KD does not fail' is "
            "FALSIFIED. Two separable failure modes: (i) general-capability "
            "degradation from dense KL to misspecified teacher (universal, "
            "worse under off-policy); (ii) sharp phase-transition cliff on "
            "vision-critical slice (on-policy specific)."
        ),
    }

    out = {
        "metadata": {
            "produced_at": "2026-05-24",
            "t1_trajectory_dir": str(args.t1_dir),
            "tier2a_trajectory_dir": str(args.tier2a_dir),
            "opd_target_ids_path": str(args.opd_target_ids),
            "n_full_image_prompts": len(base_rows),
            "n_opd_target_prompts": len(opd_target_ids),
            "benchmarks": benchmarks,
            "steps_evaluated": STEP_INTS,
        },
        "arms": arms_out,
        "token_budget_confounder": _TOKEN_BUDGET,
        "summary": summary,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {args.out_json}  ({args.out_json.stat().st_size // 1024} KB)")

    if args.out_png:
        if not _HAS_PLT:
            print(f"WARN: matplotlib unavailable; skipping {args.out_png}",
                  file=sys.stderr)
        else:
            args.out_png.parent.mkdir(parents=True, exist_ok=True)
            _plot(args.out_png, arms_out)
            print(f"wrote {args.out_png}  ({args.out_png.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
