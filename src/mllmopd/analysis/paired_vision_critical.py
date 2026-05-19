"""Paired vision-critical / teacher-advantage / OPD-target analysis.

Operates on a level1_v* audit run dir. For each benchmark, computes the prompt
subsets that matter for vision-conditioned OPD audit:

  vision_critical[M]   = {ids : M's full_image=correct AND M's blank_image=wrong}
                         → prompts where the IMAGE actually flips the answer
                         for model M.
  teacher_advantage    = {ids : T_RL_full=correct AND S_full=wrong}
                         → prompts where the OPD teacher gets it but student
                         misses (regardless of image dependency).
  opd_target           = vision_critical[T_RL] ∩ teacher_advantage
                         → prompts where the teacher's advantage IS visual:
                         teacher uses the image to get it right, the student
                         can't even with the image. THIS is the high-value
                         OPD training subset under the
                         "vision-conditioned capability transfer" hypothesis.

The framing comes from a 2026-05-19 review of level1_v4_sysprompt_fixed:
teacher's mean +6.8pt advantage on the 6-benchmark audit is concentrated in
full-image cells (mean teacher-student gap on full = +6.8pt, on blank/text-only
≈ 0 to +2pt). If OPD can transfer the teacher's vision-conditioned advantage,
the prompts in `opd_target` are where the signal is densest.

Usage:
    python -m mllmopd.analysis.paired_vision_critical \\
        --run_dir runs/audit/level1_v4_sysprompt_fixed \\
        --teacher MMR1-7B-RL \\
        --student MMR1-3B-SFT \\
        [--out-target-ids paired_opd_targets.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mllmopd.diagnostics import scorers


def _rescore_row(r: dict) -> bool | None:
    """Re-score a JSONL row with the current scorer. Mirrors
    aggregate_audit._rescore (list-gold unwrap + current scorer logic) so
    the paired tables agree with summary.json."""
    if r.get("scorer") in {"skip_missing_image", "skip_empty_gold"}:
        return r.get("is_correct")
    pred = r.get("prediction") or ""
    gold = r.get("gold")
    if isinstance(gold, list) and len(gold) == 1:
        gold = gold[0]
    is_c, _, _ = scorers.score_for_benchmark(
        r.get("benchmark", ""), pred, gold, choices=r.get("choices"),
    )
    return is_c


def _model_label(model_path: str) -> str:
    return model_path.rstrip("/").split("/")[-1]


def _load_correct_map(run_dir: Path) -> tuple[dict, list[str], list[str]]:
    """Returns ({(model_label, mode, benchmark, id): is_correct}, models, benches)."""
    correct_map: dict = {}
    models: set = set()
    benches: set = set()
    for jl in sorted(run_dir.glob("*.jsonl")):
        if jl.name == "summary.json":
            continue
        with jl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                m = _model_label(r["model"])
                models.add(m)
                benches.add(r["benchmark"])
                correct_map[(m, r["mode"], r["benchmark"], r["id"])] = _rescore_row(r)
    return correct_map, sorted(models), sorted(benches)


def _bench_subsets(correct_map, teacher, student, bench):
    """For one benchmark, return the various sets as id-frozensets."""
    def _get(model_label, mode):
        return {k[3]: v for k, v in correct_map.items()
                if k[0] == model_label and k[1] == mode and k[2] == bench and v is not None}

    t_full = _get(teacher, "full_image")
    t_blank = _get(teacher, "blank_image")
    t_text = _get(teacher, "text_only")
    s_full = _get(student, "full_image")
    s_blank = _get(student, "blank_image")
    s_text = _get(student, "text_only")

    common = set(t_full) & set(t_blank) & set(s_full) & set(s_blank)

    vc_t = {i for i in common if t_full.get(i) and not t_blank.get(i)}
    vc_s = {i for i in common if s_full.get(i) and not s_blank.get(i)}
    ta = {i for i in common if t_full.get(i) and not s_full.get(i)}
    opd_target = vc_t & ta
    # Also: prompts where teacher requires vision AND wins over student,
    # AND teacher fails on text-only — this is "pure vision-conditioned":
    pure_vc_t = (
        {i for i in common & set(t_text)
         if t_full.get(i) and not t_blank.get(i) and not t_text.get(i)}
    )
    opd_target_pure = pure_vc_t & ta

    return {
        "common": common,
        "vc_t": vc_t,
        "vc_s": vc_s,
        "teacher_advantage": ta,
        "opd_target": opd_target,
        "opd_target_pure": opd_target_pure,
        "t_full_acc": (sum(t_full.values()) / len(t_full)) if t_full else None,
        "s_full_acc": (sum(s_full.values()) / len(s_full)) if s_full else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True, type=Path)
    ap.add_argument("--teacher", default="MMR1-7B-RL",
                    help="Substring matched against model path basename.")
    ap.add_argument("--student", default="MMR1-3B-SFT")
    ap.add_argument("--out-target-ids", type=Path, default=None,
                    help="Optional path: dump opd_target ids per benchmark as JSON.")
    args = ap.parse_args()

    correct_map, models, benches = _load_correct_map(args.run_dir)
    print(f">>> loaded {len(correct_map)} (model,mode,bench,id) cells from {args.run_dir}", file=sys.stderr)

    teacher = next((m for m in models if args.teacher in m), None)
    student = next((m for m in models if args.student in m), None)
    if teacher is None or student is None:
        sys.exit(f"!! could not match teacher={args.teacher!r} or student={args.student!r} in {models}")
    print(f">>> teacher = {teacher}", file=sys.stderr)
    print(f">>> student = {student}", file=sys.stderr)
    print(f">>> benches = {benches}", file=sys.stderr)

    header = f"{'benchmark':18s} {'n':>4s} {'T_acc':>6s} {'S_acc':>6s} " \
             f"{'VC[T]':>6s} {'VC[S]':>6s} {'T_adv':>6s} {'OPD_t':>6s} {'PureV':>6s}"
    print()
    print(header)
    print("-" * len(header))
    target_ids_out: dict[str, list[str]] = {}
    totals = {"n": 0, "vc_t": 0, "vc_s": 0, "ta": 0, "opd_target": 0, "opd_target_pure": 0}
    for bench in benches:
        s = _bench_subsets(correct_map, teacher, student, bench)
        n = len(s["common"])
        target_ids_out[bench] = sorted(s["opd_target"])
        print(f"{bench:18s} {n:>4d} "
              f"{(s['t_full_acc'] or 0):>6.3f} {(s['s_full_acc'] or 0):>6.3f} "
              f"{len(s['vc_t']):>6d} {len(s['vc_s']):>6d} "
              f"{len(s['teacher_advantage']):>6d} "
              f"{len(s['opd_target']):>6d} {len(s['opd_target_pure']):>6d}")
        totals["n"] += n
        totals["vc_t"] += len(s["vc_t"])
        totals["vc_s"] += len(s["vc_s"])
        totals["ta"] += len(s["teacher_advantage"])
        totals["opd_target"] += len(s["opd_target"])
        totals["opd_target_pure"] += len(s["opd_target_pure"])

    print("-" * len(header))
    print(f"{'TOTAL':18s} {totals['n']:>4d}        "
          f"        {totals['vc_t']:>6d} {totals['vc_s']:>6d} "
          f"{totals['ta']:>6d} {totals['opd_target']:>6d} {totals['opd_target_pure']:>6d}")
    print()
    print("Legend:")
    print("  n              = prompts with valid scores in ALL of (T_full, T_blank, S_full, S_blank).")
    print("  T_acc / S_acc  = teacher / student full_image accuracy on `n`.")
    print("  VC[T] / VC[S]  = vision-critical for teacher/student: full=correct AND blank=wrong.")
    print("  T_adv          = teacher advantage: T_full=correct AND S_full=wrong.")
    print("  OPD_t          = OPD target = VC[T] ∩ T_adv (teacher's vision win that student misses).")
    print("  PureV          = pure vision-conditioned: T_full=✓, T_blank=✗, T_text_only=✗, AND in T_adv.")
    print()
    print("Interpretation key:")
    print("  - If OPD_t / T_adv is HIGH, most of the teacher's wins over student are visual:")
    print("    vanilla OPD's job is to transfer that visual-conditioned capability.")
    print("  - If OPD_t / T_adv is LOW, teacher's advantage is mostly language-prior:")
    print("    vanilla OPD might transfer that easily, but the visual story is weaker.")

    if args.out_target_ids:
        args.out_target_ids.parent.mkdir(parents=True, exist_ok=True)
        args.out_target_ids.write_text(json.dumps(target_ids_out, indent=2))
        n_total = sum(len(v) for v in target_ids_out.values())
        print(f"\n>>> wrote {n_total} opd_target ids -> {args.out_target_ids}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
