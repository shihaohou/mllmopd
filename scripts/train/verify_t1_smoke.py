"""T1 punch list #9 post-mortem verifier.

Run after `scripts/train/smoke_t1.sh` exits. Reads the diagnostics
JSONL gzipped files written by `opd_diagnostics_hook.py` plus the
train log, and asserts the smoke landed cleanly.

Exit 0 on PASS, 1 on any FAIL — keeping the same "hard stop" semantics
as T1 plan §8.1 #9.

Acceptance checks:
  1. ≥ 1 step JSONL.gz exists at <run_dir>/diagnostics/step_NNNNNN.jsonl.gz
  2. Each row has non-empty lp_full + lp_blank of length == response_length
  3. lp_full ≠ lp_blank somewhere (i.e., max |vd| > 0.1 for ≥ 1 sample) —
     confirms dual_teacher_get_reward actually scored both arms
  4. image_mode field matches the arm we launched
  5. train log contains optimizer-step lines and the loss trajectory
     across the captured steps is finite (no NaN / inf)
  6. clip_fraction (if logged) is < 0.9 across all steps (>0.9 means
     the teacher-student logp clip is firing on >90% of tokens — usually
     means teacher/student prompt mismatch)

The first three are hard checks. Loss/clip checks are advisory if the
train log format is unrecognized.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path


def _load_step_rows(run_dir: Path) -> list[tuple[int, list[dict]]]:
    diag_dir = run_dir / "diagnostics"
    if not diag_dir.exists():
        return []
    files = sorted(diag_dir.glob("step_*.jsonl.gz"))
    out: list[tuple[int, list[dict]]] = []
    for f in files:
        rows: list[dict] = []
        with gzip.open(f, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    print(f"!! {f.name}: bad json line: {e}", file=sys.stderr)
        m = re.search(r"step_(\d+)", f.stem)
        step_i = int(m.group(1)) if m else -1
        out.append((step_i, rows))
    return out


def _check_dual_teacher(rows: list[dict]) -> dict:
    """Confirm lp_full and lp_blank are both populated and DIFFER."""
    n_rows = len(rows)
    n_with_full = sum(1 for r in rows if r.get("lp_full"))
    n_with_blank = sum(1 for r in rows if r.get("lp_blank"))
    n_aligned = sum(
        1 for r in rows
        if r.get("lp_full") and r.get("lp_blank")
        and len(r["lp_full"]) == r.get("response_length", -1)
        and len(r["lp_blank"]) == r.get("response_length", -2)
    )
    max_abs_vd = 0.0
    for r in rows:
        vd = r.get("vd") or []
        for v in vd:
            if abs(v) > max_abs_vd:
                max_abs_vd = abs(v)
    return {
        "n_rows": n_rows,
        "n_with_lp_full": n_with_full,
        "n_with_lp_blank": n_with_blank,
        "n_lengths_aligned": n_aligned,
        "max_abs_vd": max_abs_vd,
    }


def _check_image_mode(rows: list[dict], expected: str) -> tuple[bool, set[str]]:
    seen = {r.get("image_mode") for r in rows}
    return (seen == {expected}, seen)


def _check_train_log(run_dir: Path) -> dict:
    log_dir = run_dir / "logs"
    if not log_dir.exists():
        return {"present": False, "reason": "no logs/ dir"}
    candidates = sorted(log_dir.glob("train_*.log"))
    if not candidates:
        return {"present": False, "reason": "no train_*.log"}
    log_path = candidates[-1]
    # Heuristic-grep for loss + clip lines. Format varies by Megatron version;
    # accept any of "loss=" / "loss:" / "pg_loss=" / "policy_loss=" patterns.
    losses: list[float] = []
    clips: list[float] = []
    nan_or_inf = False
    with log_path.open() as f:
        for line in f:
            for m in re.finditer(
                r"(?:^|[\s|])(?:pg_loss|policy_loss|loss)[=:]\s*(-?\d+\.?\d*(?:[eE][-+]?\d+)?|nan|inf|-inf)",
                line,
            ):
                tok = m.group(1)
                if tok in ("nan", "inf", "-inf"):
                    nan_or_inf = True
                else:
                    try:
                        losses.append(float(tok))
                    except ValueError:
                        pass
            for m in re.finditer(r"clip(?:_fraction|_frac|_ratio)[=:]\s*(\d+\.?\d*)", line):
                try:
                    clips.append(float(m.group(1)))
                except ValueError:
                    pass
    return {
        "present": True,
        "log_path": str(log_path),
        "n_loss_samples": len(losses),
        "loss_first_last": (losses[0], losses[-1]) if losses else None,
        "nan_or_inf": nan_or_inf,
        "n_clip_samples": len(clips),
        "max_clip": max(clips) if clips else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="${MLLMOPD_RUNS}/${OPD_RUN_NAME} of the smoke run")
    ap.add_argument("--arm", choices=("full", "blank"), required=True,
                    help="The arm the smoke launched (matches OPD_TEACHER_IMAGE_MODE)")
    ap.add_argument("--min-steps", type=int, default=1,
                    help="Minimum number of diagnostics step files required (default: 1)")
    ap.add_argument("--min-abs-vd", type=float, default=0.1,
                    help="Smallest acceptable max |vd| — anything below means "
                         "lp_full ≈ lp_blank, i.e. the blank-image swap didn't "
                         "actually change the teacher's view (default: 0.1)")
    args = ap.parse_args()

    fails: list[str] = []
    warns: list[str] = []
    passes: list[str] = []

    print(f"=== verifying smoke run at {args.run_dir} (arm={args.arm}) ===")

    steps = _load_step_rows(args.run_dir)
    print(f"  step JSONLs loaded: {len(steps)}")
    if len(steps) < args.min_steps:
        fails.append(f"only {len(steps)} step JSONLs; expected ≥ {args.min_steps}")
    else:
        passes.append(f"step jsonls present (n={len(steps)})")

    all_rows = [r for _, rows in steps for r in rows]
    if not all_rows:
        fails.append("zero rows across all step JSONLs")
    else:
        stats = _check_dual_teacher(all_rows)
        print(f"  rows total              : {stats['n_rows']}")
        print(f"  rows with lp_full       : {stats['n_with_lp_full']}")
        print(f"  rows with lp_blank      : {stats['n_with_lp_blank']}")
        print(f"  rows length-aligned     : {stats['n_lengths_aligned']}")
        print(f"  max |vd| across rows    : {stats['max_abs_vd']:.4f}")

        if stats["n_with_lp_full"] == 0:
            fails.append("no row has lp_full")
        if stats["n_with_lp_blank"] == 0:
            fails.append("no row has lp_blank — dual_teacher did NOT score the second arm")
        if stats["n_lengths_aligned"] != stats["n_rows"]:
            fails.append(
                f"{stats['n_rows'] - stats['n_lengths_aligned']}/{stats['n_rows']} rows have "
                f"lp_full/lp_blank length != response_length — teacher returned wrong-length "
                f"logprobs, would silently corrupt OPD loss (acceptance check #2)"
            )
        else:
            passes.append(
                f"all {stats['n_rows']} rows have lp_full/lp_blank length == response_length"
            )
        if stats["n_with_lp_blank"] > 0 and stats["max_abs_vd"] < args.min_abs_vd:
            fails.append(
                f"max |vd| = {stats['max_abs_vd']:.4f} < threshold "
                f"{args.min_abs_vd}; lp_full ≈ lp_blank suggests the blank-image "
                "substitution did not actually change the teacher's view"
            )
        elif stats["n_with_lp_blank"] > 0:
            passes.append(
                f"lp_full and lp_blank both populated and differ "
                f"(max |vd|={stats['max_abs_vd']:.2f})"
            )

        ok_mode, seen_modes = _check_image_mode(all_rows, args.arm)
        if not ok_mode:
            fails.append(
                f"image_mode mismatch: rows report {sorted(m for m in seen_modes if m)}, "
                f"expected only {args.arm!r}"
            )
        else:
            passes.append(f"image_mode == {args.arm!r} on all rows")

    log_stats = _check_train_log(args.run_dir)
    if log_stats.get("present"):
        print(f"  train log               : {log_stats['log_path']}")
        print(f"  loss samples parsed     : {log_stats['n_loss_samples']}")
        if log_stats["loss_first_last"]:
            lo, hi = log_stats["loss_first_last"]
            print(f"  loss first/last         : {lo:.4g} / {hi:.4g}")
        print(f"  nan/inf in losses       : {log_stats['nan_or_inf']}")
        print(f"  clip samples parsed     : {log_stats['n_clip_samples']}")
        if log_stats["max_clip"] is not None:
            print(f"  max clip_fraction       : {log_stats['max_clip']:.3f}")
        if log_stats["nan_or_inf"]:
            fails.append("training log contains nan/inf in loss — divergence")
        elif log_stats["n_loss_samples"] == 0:
            warns.append(
                "could not parse any loss samples from the train log "
                "(Megatron log format may have changed; inspect manually)"
            )
        else:
            passes.append("loss samples parsed and finite")
        if log_stats["max_clip"] is not None and log_stats["max_clip"] > 0.9:
            warns.append(
                f"max clip_fraction = {log_stats['max_clip']:.2f} > 0.9 "
                "(>90% of tokens clipped — possible teacher/student prompt mismatch)"
            )
    else:
        warns.append(f"train log not found: {log_stats.get('reason')}")

    print()
    print(f"PASSES: {len(passes)}")
    for p in passes:
        print(f"  + {p}")
    if warns:
        print(f"WARN:   {len(warns)}")
        for w in warns:
            print(f"  ? {w}")
    if fails:
        print(f"FAILS:  {len(fails)}")
        for f in fails:
            print(f"  ! {f}")
        print()
        print(">>> SMOKE FAILED — DO NOT proceed to T1 full runs (punch list #10/#11).")
        sys.exit(1)

    print()
    print(">>> SMOKE PASSED — green light for T1 full runs (punch list #10/#11).")
    sys.exit(0)


if __name__ == "__main__":
    main()
