"""T2-1 per-prompt diff on canonical 133 opd_target subset.

For each of the canonical 133 opd_target prompts (from the T1 brief
baseline), compute T1-2 and T2-1 correctness and categorize into
WIN / LOSE / BOTH_CORRECT / BOTH_WRONG. Per GPT round-3, the McNemar
n=133 underpowered the headline (p=0.864) but the discordant set
b=16 / c=18 contains the signal — which 18 prompts T2-1 loses tells
us if T2-1 is hurting a specific class of vision-reasoning.

Output:
  * counts per benchmark × category (matrix)
  * full per-prompt records with id, benchmark, gold, T1-2 pred,
    T2-1 pred, T1-2 correct, T2-1 correct, category
  * LOSE-only and WIN-only excerpts for qualitative inspection

Rescoring matches t2_1_compare.py via scorers.score_for_benchmark so
the counts here reproduce t2_1_compare's McNemar b/c exactly.

Usage::
    python -m mllmopd.analysis.t2_1_per_prompt_diff \\
        --eval-dir runs/audit/t2_1_eval_20260523-003257 \\
        --opd-target-ids runs/audit/level1_v4_sysprompt_fixed/opd_target_ids.json \\
        --out-json runs/audit/t2_1_eval_20260523-003257/t2_1_per_prompt_diff.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mllmopd.diagnostics import scorers


T1_2_FILE = "T1_2_full.jsonl"
T2_1_FILE = "T2_1_full.jsonl"


def _rescore(row: dict) -> bool | None:
    """Same as t2_1_compare._rescore_row — kept inline for self-containment."""
    if row.get("scorer") in {"skip_missing_image", "skip_empty_gold"}:
        return row.get("is_correct")
    pred = row.get("prediction") or ""
    gold = row.get("gold")
    if isinstance(gold, list) and len(gold) == 1:
        gold = gold[0]
    is_c, _, _ = scorers.score_for_benchmark(
        row.get("benchmark", ""), pred, gold, choices=row.get("choices"),
    )
    return is_c


def _load_full_jsonl(path: Path) -> dict[str, dict]:
    """Returns {bench: {id: {is_correct, prediction, gold, num_tokens}}}."""
    out: dict[str, dict[str, dict]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            bench = r.get("benchmark", "")
            rid = r["id"]
            is_c = _rescore(r)
            out.setdefault(bench, {})[rid] = {
                "is_correct": bool(is_c) if is_c is not None else None,
                "prediction": r.get("prediction") or "",
                "gold": r.get("gold"),
                "num_tokens": r.get("num_tokens"),
            }
    return out


def categorize(t12_c: bool | None, t21_c: bool | None) -> str:
    if t12_c is None or t21_c is None:
        return "INCOMPLETE"
    if t12_c and t21_c:
        return "BOTH_CORRECT"
    if t12_c and not t21_c:
        return "LOSE"   # T2-1 wrong, T1-2 correct → treatment LOSE (matches t2_1_compare's c)
    if not t12_c and t21_c:
        return "WIN"    # T2-1 correct, T1-2 wrong → treatment WIN (matches t2_1_compare's b)
    return "BOTH_WRONG"


def diff(eval_dir: Path, opd_target_ids_path: Path) -> dict:
    t12 = _load_full_jsonl(eval_dir / T1_2_FILE)
    t21 = _load_full_jsonl(eval_dir / T2_1_FILE)
    ids = json.loads(opd_target_ids_path.read_text())

    records: list[dict] = []
    per_bench_counts: dict[str, dict[str, int]] = {}
    cat_counts = {"WIN": 0, "LOSE": 0, "BOTH_CORRECT": 0, "BOTH_WRONG": 0, "INCOMPLETE": 0}

    for bench in sorted(ids.keys()):
        per_bench_counts[bench] = {"WIN": 0, "LOSE": 0, "BOTH_CORRECT": 0, "BOTH_WRONG": 0, "INCOMPLETE": 0}
        for rid in ids[bench]:
            t12_row = t12.get(bench, {}).get(rid)
            t21_row = t21.get(bench, {}).get(rid)
            t12_c = t12_row["is_correct"] if t12_row else None
            t21_c = t21_row["is_correct"] if t21_row else None
            cat = categorize(t12_c, t21_c)
            cat_counts[cat] += 1
            per_bench_counts[bench][cat] += 1
            records.append({
                "id": rid,
                "benchmark": bench,
                "category": cat,
                "t1_2_correct": t12_c,
                "t2_1_correct": t21_c,
                "gold": (t12_row or t21_row or {}).get("gold"),
                "t1_2_prediction": (t12_row or {}).get("prediction"),
                "t2_1_prediction": (t21_row or {}).get("prediction"),
                "t1_2_num_tokens": (t12_row or {}).get("num_tokens"),
                "t2_1_num_tokens": (t21_row or {}).get("num_tokens"),
            })

    out = {
        "eval_dir": str(eval_dir),
        "opd_target_ids_path": str(opd_target_ids_path),
        "n_canonical": sum(cat_counts.values()),
        "counts": cat_counts,
        "per_benchmark_counts": per_bench_counts,
        "mcnemar_check": {
            "b_win_t2_1": cat_counts["WIN"],
            "c_lose_t2_1": cat_counts["LOSE"],
            "both_correct": cat_counts["BOTH_CORRECT"],
            "both_wrong": cat_counts["BOTH_WRONG"],
        },
        "records": records,
    }
    return out


def _truncate(s: str | None, n: int = 200) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", " ").replace("\r", " ")
    return s if len(s) <= n else s[:n] + "…"


def _print_text_summary(out: dict, max_excerpt: int = 6) -> None:
    print(f"\n=== T2-1 PER-PROMPT DIFF on canonical {out['n_canonical']} ===", file=sys.stderr)
    cc = out["counts"]
    print(f"  WIN  (T2-1 correct, T1-2 wrong):  {cc['WIN']}", file=sys.stderr)
    print(f"  LOSE (T2-1 wrong, T1-2 correct):  {cc['LOSE']}", file=sys.stderr)
    print(f"  BOTH_CORRECT:                     {cc['BOTH_CORRECT']}", file=sys.stderr)
    print(f"  BOTH_WRONG:                       {cc['BOTH_WRONG']}", file=sys.stderr)
    if cc.get("INCOMPLETE"):
        print(f"  INCOMPLETE (id missing in one arm): {cc['INCOMPLETE']}", file=sys.stderr)

    print("\n  per-benchmark WIN / LOSE:", file=sys.stderr)
    print(f"  {'benchmark':<18} {'WIN':>5} {'LOSE':>5} {'BOTH_C':>7} {'BOTH_W':>7} {'NET':>5}",
          file=sys.stderr)
    for bench, pb in sorted(out["per_benchmark_counts"].items()):
        net = pb["WIN"] - pb["LOSE"]
        print(f"  {bench:<18} {pb['WIN']:>5} {pb['LOSE']:>5} {pb['BOTH_CORRECT']:>7} "
              f"{pb['BOTH_WRONG']:>7} {net:>+5}",
              file=sys.stderr)

    # LOSE excerpt
    lose = [r for r in out["records"] if r["category"] == "LOSE"]
    if lose:
        print(f"\n  LOSE excerpt (first {min(len(lose), max_excerpt)} of {len(lose)}):",
              file=sys.stderr)
        for r in lose[:max_excerpt]:
            print(f"    [{r['benchmark']}] {r['id']}", file=sys.stderr)
            print(f"      gold:  {_truncate(r['gold'])}", file=sys.stderr)
            print(f"      T1-2 ({r['t1_2_num_tokens']}tok): {_truncate(r['t1_2_prediction'])}",
                  file=sys.stderr)
            print(f"      T2-1 ({r['t2_1_num_tokens']}tok): {_truncate(r['t2_1_prediction'])}",
                  file=sys.stderr)

    # WIN excerpt (also helpful: are wins on specific patterns?)
    win = [r for r in out["records"] if r["category"] == "WIN"]
    if win:
        print(f"\n  WIN excerpt (first {min(len(win), max_excerpt)} of {len(win)}):",
              file=sys.stderr)
        for r in win[:max_excerpt]:
            print(f"    [{r['benchmark']}] {r['id']}", file=sys.stderr)
            print(f"      gold:  {_truncate(r['gold'])}", file=sys.stderr)
            print(f"      T1-2 ({r['t1_2_num_tokens']}tok): {_truncate(r['t1_2_prediction'])}",
                  file=sys.stderr)
            print(f"      T2-1 ({r['t2_1_num_tokens']}tok): {_truncate(r['t2_1_prediction'])}",
                  file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-dir", required=True, type=Path,
                    help="dir containing T1_2_full.jsonl and T2_1_full.jsonl")
    ap.add_argument("--opd-target-ids", required=True, type=Path,
                    help="JSON of {bench: [ids]} (canonical 133)")
    ap.add_argument("--out-json", default=None, type=Path)
    ap.add_argument("--max-excerpt", type=int, default=6,
                    help="number of LOSE/WIN records to print to stderr")
    args = ap.parse_args(argv)

    out = diff(args.eval_dir, args.opd_target_ids)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"json -> {args.out_json}", file=sys.stderr)
    else:
        # Strip records from stdout JSON (too verbose) — they're only useful in file form
        compact = {k: v for k, v in out.items() if k != "records"}
        print(json.dumps(compact, indent=2, ensure_ascii=False))

    _print_text_summary(out, max_excerpt=args.max_excerpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
