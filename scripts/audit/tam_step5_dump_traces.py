"""Step 5 — dump full S0/S1 thinking traces for a subset of alignment.jsonl.

Use BEFORE deciding which cases to render: read the CoTs first, pick the
dramatic ones, then point the renderer at a smaller id list. The
qualitative renderer's `_context.md` only emits this for samples that
already made the picks file — this dump operates directly on
alignment.jsonl and supports benchmark/bucket-level filtering.

Usage (filter by benchmark + bucket)::

    python -m scripts.audit.tam_step5_dump_traces \\
        --alignment runs/audit/tam_step5_<TS>/alignment.jsonl \\
        --benchmark POPE \\
        --bucket OPD_improved \\
        --out-dir docs/figures/step5/narrative_picks/pope_traces/

Usage (filter by explicit id list)::

    python -m scripts.audit.tam_step5_dump_traces \\
        --alignment <...> \\
        --restrict-ids "POPE/1,POPE/2,POPE/3" \\
        --out-dir <...>

Output:
    traces.md       — ranked by ΔJS ASC (most evidence-regression first)
    traces.jsonl    — same data, machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mllmopd.analysis.tam_step5_analyzer import (
    RELIABILITY_THRESH_DEFAULT,
    _per_sample_means,
)


def _stream_filter(alignment_path: Path,
                   restrict_ids: set[str] | None,
                   benchmark_substr: str | None,
                   bucket: str | None) -> list[dict]:
    """Stream alignment.jsonl, keep rows matching all active filters.

    Avoids loading the whole 600MB+ file (load_alignment) for what is
    usually a tiny subset.
    """
    out: list[dict] = []
    with alignment_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            # Cheap pre-filter: id / benchmark / bucket all appear early
            # in each line (within first ~500 chars before the giant
            # maps_b64 payload).
            head = line[:500]
            if restrict_ids and not any(f'"{wid}"' in head
                                         for wid in restrict_ids):
                continue
            if benchmark_substr and benchmark_substr not in head:
                continue
            if bucket and f'"{bucket}"' not in head:
                continue
            rec = json.loads(line)
            if restrict_ids and rec.get("id") not in restrict_ids:
                continue
            if benchmark_substr:
                bench = rec.get("benchmark") or ""
                if benchmark_substr not in bench:
                    continue
            if bucket and rec.get("bucket") != bucket:
                continue
            out.append(rec)
    return out


def _safe_str(s) -> str:
    if s is None:
        return "(not stored)"
    if isinstance(s, list):
        return ", ".join(str(x) for x in s)
    return str(s)


def _mark(ok) -> str:
    if ok is True:
        return "✓"
    if ok is False:
        return "✗"
    return "?"


def write_traces_md(rows_with_metrics: list[tuple[dict, dict]],
                    out_md: Path,
                    benchmark_substr: str | None,
                    bucket: str | None) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    title_parts: list[str] = []
    if benchmark_substr:
        title_parts.append(f"benchmark~={benchmark_substr}")
    if bucket:
        title_parts.append(f"bucket={bucket}")
    title = " · ".join(title_parts) or "subset"

    lines: list[str] = []
    lines.append(f"# Step 5 traces — {title} (n={len(rows_with_metrics)})\n")
    lines.append("Ranked by ΔJS ASC — most evidence-regression first.\n")
    lines.append("Sign: ΔJS = JS(S0,T) − JS(S1,T); positive = OPD improves.\n")
    lines.append("---\n")
    for i, (r, s) in enumerate(rows_with_metrics, 1):
        lines.append(
            f"## {i}. `{r['id']}` "
            f"S0={_mark(r.get('s0_correct'))} "
            f"S1={_mark(r.get('s1_correct'))}  "
            f"ΔJS={s['delta_js']:+.4f} "
            f"ΔIoU={s['delta_iou']:+.4f} "
            f"ΔCos={s['delta_cos']:+.4f} "
            f"(n_tok={s['n_tokens']})\n"
        )
        lines.append(f"**Benchmark**: `{r.get('benchmark')}`  ")
        lines.append(f"**Bucket**: `{r.get('bucket')}`\n")
        lines.append("### Question\n")
        lines.append(f"> {_safe_str(r.get('question'))}\n")
        lines.append("### Gold\n")
        lines.append(f"> `{_safe_str(r.get('answer'))}`\n")

        s0_txt = r.get("s0_response_text")
        s1_txt = r.get("s1_response_text") or r.get("response_text")
        lines.append(f"### S0 — {_mark(r.get('s0_correct'))}\n")
        lines.append("```")
        lines.append(_safe_str(s0_txt))
        lines.append("```\n")
        lines.append(f"### S1 — {_mark(r.get('s1_correct'))}\n")
        lines.append("```")
        lines.append(_safe_str(s1_txt))
        lines.append("```\n")
        lines.append("---\n")
    out_md.write_text("\n".join(lines))


def write_traces_jsonl(rows_with_metrics: list[tuple[dict, dict]],
                       out_jsonl: Path) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w") as f:
        for r, s in rows_with_metrics:
            slim = {
                "id":           r["id"],
                "bucket":       r.get("bucket"),
                "benchmark":    r.get("benchmark"),
                "question":     r.get("question"),
                "answer":       r.get("answer"),
                "s0_response_text": r.get("s0_response_text"),
                "s1_response_text": (r.get("s1_response_text")
                                     or r.get("response_text")),
                "s0_correct":   r.get("s0_correct"),
                "s1_correct":   r.get("s1_correct"),
                "delta_js":     s["delta_js"],
                "delta_iou":    s["delta_iou"],
                "delta_cos":    s["delta_cos"],
                "n_tokens":     s["n_tokens"],
            }
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alignment", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--restrict-ids", default=None,
                    help="Comma-separated sample ids. If set, --benchmark / "
                         "--bucket are ANDed with this list.")
    ap.add_argument("--benchmark", default=None,
                    help="Substring match against rec['benchmark'] "
                         "(e.g. 'POPE' matches POPE_adversarial / POPE_random).")
    ap.add_argument("--bucket", default=None,
                    help="Exact bucket name "
                         "(OPD_improved / OPD_failed / Teacher_advantage / "
                         "Dataset_diversity).")
    ap.add_argument("--reliability-thresh", type=float,
                    default=RELIABILITY_THRESH_DEFAULT,
                    help="Per-token entropy filter for Δ-metrics, matches "
                         f"audit. Default {RELIABILITY_THRESH_DEFAULT}.")
    args = ap.parse_args(argv)

    if not (args.restrict_ids or args.benchmark or args.bucket):
        sys.exit("!! at least one of --restrict-ids / --benchmark / --bucket "
                 "is required (refuse to dump the entire alignment.jsonl).")

    restrict_ids: set[str] | None = None
    if args.restrict_ids:
        restrict_ids = {s.strip() for s in args.restrict_ids.split(",")
                        if s.strip()}
        print(f">>> restrict-ids: {len(restrict_ids)} ids", file=sys.stderr)
    if args.benchmark:
        print(f">>> benchmark substring: {args.benchmark!r}", file=sys.stderr)
    if args.bucket:
        print(f">>> bucket: {args.bucket!r}", file=sys.stderr)

    print(f">>> streaming {args.alignment}", file=sys.stderr)
    rows = _stream_filter(args.alignment, restrict_ids,
                          args.benchmark, args.bucket)
    print(f">>> matched {len(rows)} rows", file=sys.stderr)
    if not rows:
        sys.exit("!! no rows matched — check filters")

    with_metrics: list[tuple[dict, dict]] = []
    n_dropped = 0
    for r in rows:
        s = _per_sample_means(r, reliability_thresh=args.reliability_thresh)
        if s is None:
            n_dropped += 1
            continue
        with_metrics.append((r, s))
    print(f">>> per-sample metrics: kept {len(with_metrics)}, "
          f"dropped {n_dropped} (no surviving tokens after reliability filter)",
          file=sys.stderr)

    # Sort by ΔJS ASC — most regression first (matches N1 narrative ordering)
    with_metrics.sort(key=lambda x: x[1]["delta_js"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_traces_md(with_metrics, args.out_dir / "traces.md",
                    args.benchmark, args.bucket)
    write_traces_jsonl(with_metrics, args.out_dir / "traces.jsonl")
    print(f">>> wrote {args.out_dir / 'traces.md'}", file=sys.stderr)
    print(f">>> wrote {args.out_dir / 'traces.jsonl'}", file=sys.stderr)

    # Echo top 5 + bottom 5 to stdout so user sees range
    print(f"\n=== top 5 by ΔJS ASC (most regression) ===")
    for i, (r, s) in enumerate(with_metrics[:5], 1):
        print(f"  {i}. {r['id']:30s} ΔJS={s['delta_js']:+.4f} "
              f"ΔIoU={s['delta_iou']:+.4f} s0={r.get('s0_correct')} "
              f"s1={r.get('s1_correct')}")
    if len(with_metrics) > 10:
        print(f"\n=== bottom 5 (least regression / improvement) ===")
        for i, (r, s) in enumerate(with_metrics[-5:], len(with_metrics) - 4):
            print(f"  {i}. {r['id']:30s} ΔJS={s['delta_js']:+.4f} "
                  f"ΔIoU={s['delta_iou']:+.4f} s0={r.get('s0_correct')} "
                  f"s1={r.get('s1_correct')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
