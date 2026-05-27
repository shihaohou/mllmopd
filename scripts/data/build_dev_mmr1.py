"""Build the held-out dev_mmr1 evaluation set from MMR1-RL.

The mllmopd training data (data/opd_train/v0_2k/train.jsonl) is a 2k random
sample of MMR1-RL drawn with seed=42. For the Rethinking-OPD-style
training-dynamics atlas (Figure 1: per-checkpoint accuracy curves), we need a
~1k-prompt held-out dev set drawn from the SAME MMR1-RL pool but disjoint
from both the training rows AND the standard eval contamination index.

Pipeline (mirrors scripts/data/prep_opd_train_data.py):
  1. Load MMR1-RL via load_from_disk (fallback load_dataset).
  2. Read the training JSONL and collect source_row_idx into a hard exclusion
     set (train_exclusion_set). Any row already in training cannot land in dev.
  3. Build eval-side dedup index against data/audit/level1_subset_v0.jsonl
     (same _build_eval_index call used by prep_opd_train_data.py) so dev_mmr1
     ↔ level1 contamination is enforced to the same threshold.
  4. Scan MMR1-RL for clean rows := not in train_exclusion_set AND not flagged
     by layer-b (normalized question) or layer-d (image sha256) overlap.
  5. Hard-stop if clean_count < args.size OR layer-b rate > threshold.
  6. Random sample args.size rows with a DISTINCT seed (default 142 vs train's
     42) so the dev draw is independent of the training draw.
  7. Emit <out_dir>/dev.jsonl (NOT train.jsonl — distinct filename so a
     misconfigured launcher cannot silently train on the dev set) +
     <out_dir>/dedup_report.json with stats.

Schema per row matches prep_opd_train_data.py exactly so the same Uni-OPD /
miles Dataset readers can ingest the JSONL unchanged.

Usage (run on H800 where $MMR1_RL_DATA is available):

  python scripts/data/build_dev_mmr1.py \\
      --eval-subset data/audit/level1_subset_v0.jsonl \\
      --opd-target-ids data/audit/opd_target_ids.json \\
      --train-jsonl data/opd_train/v0_2k/train.jsonl \\
      --out-dir data/eval/dev_mmr1_v0_1k/ \\
      --size 1000 --seed 142
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Reuse helpers from prep_opd_train_data.py so dev_mmr1 normalization /
# hashing / chat-template formatting stays bit-identical to train.jsonl.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from prep_opd_train_data import (  # noqa: E402  (path-tweak above)
    MMR1_SYSTEM_PROMPT,
    _build_eval_index,
    _extract_question_text,
    normalize_question,
    sha256_pil,
)


def _load_train_exclusion(train_jsonl: Path) -> set[int]:
    """Read every `source_row_idx` from the existing training JSONL.

    Any MMR1-RL row already used for training is hard-excluded from dev so
    train ∩ dev = ∅ by construction.
    """
    exclude: set[int] = set()
    n_rows = 0
    n_missing = 0
    with train_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            rec = json.loads(line)
            src = rec.get("source_row_idx")
            if src is None:
                # Defensive: prep_opd_train_data.py emits source_row_idx for
                # every row; a missing field would mean the JSONL was produced
                # by an older/different script and we cannot guarantee the
                # train/dev disjointness.
                n_missing += 1
                continue
            exclude.add(int(src))
    if n_missing:
        sys.exit(
            f"ERROR: {n_missing}/{n_rows} train rows in {train_jsonl} are "
            f"missing 'source_row_idx'. Cannot guarantee train/dev disjoint "
            f"sets — regenerate train.jsonl with prep_opd_train_data.py."
        )
    print(
        f">>> train_exclusion_set: {len(exclude):,} source_row_idx values "
        f"loaded from {train_jsonl}",
        file=sys.stderr,
    )
    return exclude


def _resolve_mmr1_path(cli_value: str) -> str:
    """Resolve MMR1-RL path. CLI > env > fallback paths (matches prep script)."""
    if cli_value:
        return cli_value
    for guess in (
        "/home/web_server/antispam/project/houshihao/datasets/MMR1-RL",
        os.path.expanduser("~/datasets/MMR1-RL"),
    ):
        if os.path.isdir(guess):
            print(f">>> using fallback MMR1-RL path: {guess}", file=sys.stderr)
            return guess
    sys.exit(
        "ERROR: --mmr1-rl or $MMR1_RL_DATA must be set (or `source .env` first)"
    )


def _load_opd_target_ids(path: Path) -> set[str]:
    """Accept either a JSON list of ids OR a {benchmark: [ids]} dict — matches
    the dual-form handling in prep_opd_train_data.py.

    Returns an empty set (with a warning) when the file does not exist.
    The ids are only used to label contamination-hit attribution in the
    dedup report — dev_mmr1 sampling itself does not depend on them.
    """
    if not path.exists():
        print(
            f"WARNING: --opd-target-ids {path} does not exist; "
            f"continuing with empty opd_target set. The dedup report's "
            f"'n_opd_target' attribution column will read 0. "
            f"Sampling itself is unaffected.",
            file=sys.stderr,
        )
        return set()
    text = path.read_text()
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return set(parsed)
    if isinstance(parsed, dict):
        return set().union(*parsed.values())
    sys.exit(f"ERROR: unsupported opd_target_ids shape in {path}: {type(parsed)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mmr1-rl",
        type=str,
        default=os.environ.get("MMR1_RL_DATA", ""),
        help="Path to MMR1-RL dataset (load_from_disk format). "
             "Defaults to $MMR1_RL_DATA env var.",
    )
    ap.add_argument(
        "--eval-subset",
        type=Path,
        default=Path("data/audit/level1_subset_v0.jsonl"),
        help="Level-1 eval subset jsonl (the standard 1200-prompt "
             "contamination index used by prep_opd_train_data.py).",
    )
    ap.add_argument(
        "--opd-target-ids",
        type=Path,
        default=Path("data/audit/opd_target_ids.json"),
        help="opd_target_ids.json — used only for benchmark attribution in "
             "the dedup report; not a filter.",
    )
    ap.add_argument(
        "--train-jsonl",
        type=Path,
        default=Path("data/opd_train/v0_2k/train.jsonl"),
        help="Existing training JSONL. Every row's `source_row_idx` is "
             "hard-excluded from the dev draw.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/eval/dev_mmr1_v0_1k/"),
        help="Output directory. Will create {out_dir}/dev.jsonl, "
             "{out_dir}/images/, {out_dir}/dedup_report.json.",
    )
    ap.add_argument(
        "--size",
        type=int,
        default=1000,
        help="Number of dev prompts to sample (post-dedup).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=142,
        help="Random seed for the dev draw. MUST differ from the training "
             "seed (42) so dev sampling is independent of training sampling.",
    )
    ap.add_argument(
        "--layer-b-threshold",
        type=float,
        default=0.05,
        help="Hard-stop threshold for layer-b (normalized question) overlap "
             "rate — matches prep_opd_train_data.py.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, scan only the first N MMR1-RL rows (smoke testing).",
    )
    args = ap.parse_args()

    args.mmr1_rl = _resolve_mmr1_path(args.mmr1_rl)

    # Cheap pre-flight: refuse to overwrite an existing dev.jsonl, so a stray
    # rerun cannot quietly invalidate downstream atlas plots that referenced
    # the old file's seed.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "images").mkdir(exist_ok=True)
    report_path = args.out_dir / "dedup_report.json"
    out_jsonl = args.out_dir / "dev.jsonl"
    if out_jsonl.exists():
        sys.exit(
            f"ERROR: {out_jsonl} already exists. Refusing to overwrite — "
            f"delete it first if you really want to regenerate."
        )

    # 1. Load MMR1-RL (mirrors prep_opd_train_data.py loader logic).
    print(f">>> loading MMR1-RL from {args.mmr1_rl}", file=sys.stderr)
    loader_used = None
    raw = None
    try:
        from datasets import load_from_disk  # type: ignore
        raw = load_from_disk(args.mmr1_rl)
        loader_used = "load_from_disk"
    except Exception as e_disk:
        try:
            from datasets import load_dataset  # type: ignore
            raw = load_dataset(args.mmr1_rl)
            loader_used = "load_dataset"
        except Exception as e_hf:
            print(f"ERROR: could not load {args.mmr1_rl}", file=sys.stderr)
            print(f"  load_from_disk: {e_disk}", file=sys.stderr)
            print(f"  load_dataset:   {e_hf}", file=sys.stderr)
            sys.exit(1)
    print(f">>> loaded via {loader_used}", file=sys.stderr)

    if hasattr(raw, "keys") and not hasattr(raw, "features"):
        split = list(raw.keys())[0]
        mmr1 = raw[split]
        print(f">>> using split '{split}': {len(mmr1):,} rows", file=sys.stderr)
    else:
        mmr1 = raw
        print(f">>> single-split dataset: {len(mmr1):,} rows", file=sys.stderr)

    # 2. Training exclusion set.
    train_exclusion_set = _load_train_exclusion(args.train_jsonl)

    # 3. Eval-side dedup index (reuse prep script's function so logic is shared).
    opd_target_ids = _load_opd_target_ids(args.opd_target_ids)
    eval_idx = _build_eval_index(args.eval_subset, opd_target_ids)

    # 4. Scan MMR1-RL for clean residual (also exclude training rows).
    n_scan = args.limit if args.limit > 0 else len(mmr1)
    print(f">>> scanning {n_scan:,} MMR1-RL rows", file=sys.stderr)

    layer_b_hits: list[dict] = []
    layer_d_hits: list[dict] = []
    layer_bd_hits: list[dict] = []
    train_overlap_count = 0
    clean_indices: list[int] = []
    mmr1_img_sha_cache: dict[int, str] = {}

    t0 = time.time()
    for idx in range(n_scan):
        row = mmr1[idx]
        problem = row.get("problem", "")
        images = row.get("images") or []

        # Training-set exclusion takes precedence so the dedup report's
        # layer-b/d counters reflect the *eval-contamination* picture of the
        # train-disjoint residual, not contamination of the full pool.
        if idx in train_exclusion_set:
            train_overlap_count += 1
            continue

        qn = normalize_question(problem)
        hit_b = eval_idx["q_norm"].get(qn, [])

        img_sha = None
        if images:
            try:
                img_sha = sha256_pil(images[0])
                mmr1_img_sha_cache[idx] = img_sha
            except Exception as e:
                print(f"!! row {idx}: could not hash image: {e}", file=sys.stderr)
        hit_d = eval_idx["img_hash"].get(img_sha or "", [])

        if hit_b and hit_d:
            layer_bd_hits.append(
                {"idx": idx, "norm_q_match_ids": hit_b, "img_match_ids": hit_d}
            )
        elif hit_b:
            layer_b_hits.append({"idx": idx, "norm_q_match_ids": hit_b})
        elif hit_d:
            layer_d_hits.append({"idx": idx, "img_match_ids": hit_d})
        else:
            clean_indices.append(idx)

        if (idx + 1) % 1000 == 0:
            print(
                f"   ... {idx+1:,}/{n_scan:,}  "
                f"clean={len(clean_indices):,}  b={len(layer_b_hits):,}  "
                f"d={len(layer_d_hits):,}  b&d={len(layer_bd_hits):,}  "
                f"train_excl={train_overlap_count:,}  "
                f"({time.time() - t0:.1f}s)",
                file=sys.stderr,
            )

    n_layer_b = len(layer_b_hits) + len(layer_bd_hits)
    n_layer_d = len(layer_d_hits) + len(layer_bd_hits)
    # Rate denominator excludes training rows (we don't get to choose those
    # either way) so the rate matches what dev_mmr1 actually had to filter.
    denom = max(1, n_scan - train_overlap_count)
    rate_b = n_layer_b / denom
    rate_d = n_layer_d / denom

    # Benchmark attribution (mirrors prep_opd_train_data.py).
    import collections

    def _bench_breakdown(hits: list[dict], match_key: str) -> dict:
        per_bench: dict[str, int] = collections.Counter()
        n_opd_target = 0
        for h in hits:
            for eid in h.get(match_key, []):
                meta = eval_idx["eval_meta"][eid]
                per_bench[meta["benchmark"]] += 1
                if meta["is_opd_target"]:
                    n_opd_target += 1
        return {"by_benchmark": dict(per_bench), "n_opd_target": n_opd_target}

    layer_b_attr = _bench_breakdown(layer_b_hits + layer_bd_hits, "norm_q_match_ids")
    layer_d_attr = _bench_breakdown(layer_d_hits + layer_bd_hits, "img_match_ids")

    report: dict = {
        "mmr1_rl_path": args.mmr1_rl,
        "eval_subset_path": str(args.eval_subset),
        "train_jsonl_path": str(args.train_jsonl),
        "opd_target_ids_count": len(opd_target_ids),
        "eval_total": eval_idx["n_total"],
        "mmr1_scanned": n_scan,
        "train_overlap_count": train_overlap_count,
        "rate_denominator": denom,
        "layer_b_normalized_question": {
            "count": n_layer_b,
            "rate": rate_b,
            "attribution": layer_b_attr,
        },
        "layer_d_image_hash": {
            "count": n_layer_d,
            "rate": rate_d,
            "attribution": layer_d_attr,
        },
        "layer_b_and_d_intersection": len(layer_bd_hits),
        "clean_count": len(clean_indices),
        "hard_stop_threshold": args.layer_b_threshold,
        "hard_stop_triggered": rate_b > args.layer_b_threshold,
        "requested_size": args.size,
        "seed": args.seed,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(file=sys.stderr)
    print(f">>> dedup report -> {report_path}", file=sys.stderr)
    print(
        f"    train_exclusion:          {train_overlap_count:,} "
        f"({train_overlap_count*100.0/max(1,n_scan):.2f}% of scanned)",
        file=sys.stderr,
    )
    print(
        f"    layer-b (norm question):  {n_layer_b:,}/{denom:,} "
        f"({rate_b*100:.2f}%)  threshold {args.layer_b_threshold*100:.0f}%",
        file=sys.stderr,
    )
    print(
        f"    layer-d (image sha256):   {n_layer_d:,}/{denom:,} "
        f"({rate_d*100:.2f}%)",
        file=sys.stderr,
    )
    print(f"    layer-b ∩ layer-d:        {len(layer_bd_hits):,}", file=sys.stderr)
    print(f"    clean post-dedup:         {len(clean_indices):,}", file=sys.stderr)
    print(
        f"    layer-b benchmarks: {layer_b_attr['by_benchmark']}", file=sys.stderr
    )
    print(
        f"    layer-b opd_target hits: {layer_b_attr['n_opd_target']}",
        file=sys.stderr,
    )

    # 5. Hard stops.
    if report["hard_stop_triggered"]:
        print(file=sys.stderr)
        print(
            f"!! HARD STOP: layer-b overlap {rate_b*100:.2f}% exceeds threshold "
            f"{args.layer_b_threshold*100:.0f}%. Same policy as "
            f"prep_opd_train_data.py — operator must switch eval to held-out "
            f"benchmarks before relying on this dev set.",
            file=sys.stderr,
        )
        print("!! Dedup report written; dev.jsonl NOT written.", file=sys.stderr)
        sys.exit(1)

    if len(clean_indices) < args.size:
        sys.exit(
            f"!! only {len(clean_indices):,} clean rows but --size "
            f"{args.size:,} requested. Either reduce --size, lift --limit, or "
            f"shrink the training set to free up clean residual."
        )

    # 6. Sample, save images, emit jsonl.
    rng = random.Random(args.seed)
    sampled_indices = sorted(rng.sample(clean_indices, args.size))
    print(file=sys.stderr)
    print(
        f">>> sampling {args.size:,} clean rows (seed={args.seed}, "
        f"distinct from training seed=42)",
        file=sys.stderr,
    )

    # Sanity: ensure no sampled index leaked from training set. With
    # train_exclusion enforced in the scan loop, this should be impossible —
    # we still verify because a regression here would silently invalidate
    # downstream curves.
    overlap_with_train = set(sampled_indices) & train_exclusion_set
    if overlap_with_train:
        sys.exit(
            f"INTERNAL ERROR: {len(overlap_with_train)} sampled indices "
            f"overlap training set; this should be impossible. Aborting."
        )

    n_written = 0
    t0 = time.time()
    with out_jsonl.open("w") as fout:
        for out_idx, src_idx in enumerate(sampled_indices):
            row = mmr1[src_idx]
            problem = row.get("problem", "")
            answer = row.get("answer", "")
            images = row.get("images") or []
            if not images:
                # Defensive: row had no image when sampled; skip silently as
                # prep_opd_train_data.py does. Should be unreachable because
                # the scan only counts rows with an image as 'clean' (hashing
                # path), but rows without an image but without hits would also
                # land in clean_indices — guard against that.
                continue

            question_text = _extract_question_text(problem)
            full_problem = (
                f"{MMR1_SYSTEM_PROMPT.strip()} <image>\n{question_text}"
            )

            img = images[0]
            img_sha = mmr1_img_sha_cache.get(src_idx) or sha256_pil(img)
            img_path = args.out_dir / "images" / f"{img_sha}.png"
            if not img_path.exists():
                img.convert("RGB").save(img_path, format="PNG", optimize=False)

            # Note: id prefix is `mmr1_rl_dev_v0_` (not `mmr1_rl_v0_`) so a
            # downstream aggregator that joins train + dev cannot collide
            # ids even if the same out_idx slot is used.
            rec_id = f"mmr1_rl_dev_v0_{out_idx:06d}"
            record = {
                "id": rec_id,
                "problem": full_problem,
                "question_text": question_text,
                "images": [str(img_path)],
                "answer": answer,
                "teacher_model": "MMR1-7B-RL",
                "raw_problem": problem,
                "source_row_idx": src_idx,
                "metadata": {
                    "id": rec_id,
                    "source_row_idx": src_idx,
                    "teacher_model": "MMR1-7B-RL",
                    "split": "dev_mmr1_v0",
                },
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

            if (out_idx + 1) % 200 == 0:
                print(
                    f"   ... {out_idx+1:,}/{args.size:,}  "
                    f"({time.time() - t0:.1f}s)",
                    file=sys.stderr,
                )

    # Update the dedup report with the realized write count + sampled indices
    # summary so a downstream consumer can audit the draw without re-reading
    # the JSONL.
    report["n_written"] = n_written
    report["sampled_indices_summary"] = {
        "min": int(sampled_indices[0]) if sampled_indices else None,
        "max": int(sampled_indices[-1]) if sampled_indices else None,
        "count": len(sampled_indices),
    }
    report_path.write_text(json.dumps(report, indent=2))

    print(file=sys.stderr)
    print(f">>> wrote {n_written:,} dev rows -> {out_jsonl}", file=sys.stderr)
    print(
        f">>> images at {args.out_dir / 'images'} (unique by sha256)",
        file=sys.stderr,
    )
    print(
        ">>> dev_mmr1 ready. Pair with per-checkpoint eval (e.g. lmms-eval "
        "or a sglang accuracy pass) to plot Figure 1 of the atlas.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
