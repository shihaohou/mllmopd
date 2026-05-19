"""T1 punch list #2: build the OPD training JSONL from MMR1-RL.

Pipeline:
  1. Load MMR1-RL (via datasets.load_from_disk).
  2. Hash + normalize the entire 1200-prompt Level-1 eval subset; flag
     the 133 opd_target ids separately.
  3. Scan every MMR1-RL row, compute (normalized_question, image_sha256),
     check overlap against eval — layer (b) and layer (d).
  4. Write overlap report.
  5. Hard stop if layer-b (normalized-question) overlap > 5%, per T1 plan
     §2.5 / Risk #1.
  6. Random-sample N clean rows (seed-fixed), save images to disk as
     PNGs (hash-named, so identical images dedup automatically), emit
     JSONL ready for Uni-OPD `read_file` + `_build_messages`.

Output schema per row:
  {
    "id":            "mmr1_rl_v0_NNNNNN",  // synthetic, stable
    "problem":       "<MMR1_SYSTEM_PROMPT> <image>\\n...",  // sysprompt-before-image
    "images":        ["<out_dir>/images/<sha256>.png"],
    "answer":        str,                   // verbatim gold from MMR1-RL
    "teacher_model": "MMR1-7B-RL",
    "raw_problem":   str,                   // original MMR1-RL problem
    "source_row_idx": int,                  // MMR1-RL row index, for reproducibility
  }

Usage (on devbox, audit venv — only needs datasets + PIL):
  python scripts/data/prep_opd_train_data.py \\
      --eval-subset data/audit/level1_subset_v0.jsonl \\
      --opd-target-ids runs/audit/level1_v4_sysprompt_fixed/opd_target_ids.json \\
      --out-dir data/opd_train/v0_2k \\
      --size 2000 --seed 42
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import string
import sys
import time
from pathlib import Path
from typing import Iterable

# MMR1's training-time system prompt — verbatim from scripts/audit/run_smoke.sh:121.
# Identical to scripts/audit/rerun_h2_sysprompt.sh. KEEP IN SYNC if MMR1 retrains.
MMR1_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The User provides an image and asks a question. "
    "The Assistant first analyzes both the image and the question, then carefully thinks about the "
    "reasoning process step by step, and finally provides the User with an accurate answer. "
    "The Assistant must carefully checkout the correctness and validity of each reasoning step. "
    "If any errors or inconsistencies are found during the reasoning process, the Assistant "
    "reflects and corrects them logically. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here, with potential reflections and corrections </think>"
    "<answer> final answer here, with the key result enclosed in \\boxed{} </answer>."
)

_PLACEHOLDER_RE = re.compile(r"<(image|video|audio)>")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_question(q: str) -> str:
    """Strip multimodal placeholders, lowercase, drop punctuation, collapse ws."""
    if not isinstance(q, str):
        return ""
    q = _PLACEHOLDER_RE.sub(" ", q)
    q = q.lower().translate(_PUNCT_TABLE)
    return " ".join(q.split())


def sha256_pil(img) -> str:
    """Stable hash of a PIL.Image. Convert to RGB so encoding differences
    (mode='P' vs 'RGB' on the same pixels) don't produce different hashes."""
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash an image file by loading + RGB-converting + tobytes — matches
    sha256_pil so eval-side (path) and MMR1-side (PIL) hashes are comparable."""
    from PIL import Image  # type: ignore
    with Image.open(path) as img:
        return sha256_pil(img)


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_eval_index(eval_subset_path: Path, opd_target_ids: set[str]) -> dict:
    """Hash + normalize every eval row. Returns dict with sets + per-benchmark
    breakdown so the dedup report can attribute hits to benchmarks."""
    q_norm: dict[str, list[str]] = collections.defaultdict(list)   # norm_q -> [eval_id, ...]
    img_hash: dict[str, list[str]] = collections.defaultdict(list)  # img_sha -> [eval_id, ...]
    eval_meta: dict[str, dict] = {}  # eval_id -> {"benchmark", "is_opd_target"}
    n_loaded = 0

    print(f">>> hashing eval subset {eval_subset_path}", file=sys.stderr)
    t0 = time.time()
    for rec in load_jsonl(eval_subset_path):
        rid = rec["id"]
        bench = rec.get("benchmark", "?")
        eval_meta[rid] = {"benchmark": bench, "is_opd_target": rid in opd_target_ids}

        qn = normalize_question(rec.get("question", ""))
        if qn:
            q_norm[qn].append(rid)

        img_field = rec.get("image")
        if isinstance(img_field, list):
            img_field = img_field[0] if img_field else None
        if isinstance(img_field, str) and os.path.exists(img_field):
            try:
                img_hash[sha256_path(img_field)].append(rid)
            except Exception as e:
                print(f"!! could not hash eval image {img_field}: {e}", file=sys.stderr)

        n_loaded += 1
        if n_loaded % 200 == 0:
            print(f"   ... {n_loaded} eval rows  ({time.time() - t0:.1f}s)", file=sys.stderr)

    print(f">>> {n_loaded} eval rows hashed in {time.time() - t0:.1f}s "
          f"({len(q_norm)} unique norm-questions, {len(img_hash)} unique images)",
          file=sys.stderr)
    return {"q_norm": q_norm, "img_hash": img_hash, "eval_meta": eval_meta,
            "n_total": n_loaded}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mmr1-rl", default=os.environ.get("MMR1_RL_DATA", ""),
                    help="Path to MMR1-RL dataset (load_from_disk format). "
                         "Defaults to $MMR1_RL_DATA env var.")
    ap.add_argument("--eval-subset", type=Path, required=True,
                    help="Level-1 eval subset jsonl (1200 prompts).")
    ap.add_argument("--opd-target-ids", type=Path, required=True,
                    help="opd_target_ids.json (133 prompt ids, the dev subset).")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory. Will create {out_dir}/train.jsonl, "
                         "{out_dir}/images/, {out_dir}/dedup_report.json.")
    ap.add_argument("--size", type=int, default=2000,
                    help="Number of training prompts to sample (post-dedup).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--layer-b-threshold", type=float, default=0.05,
                    help="Hard-stop threshold for layer-b (normalized question) "
                         "overlap rate. If MMR1-RL→eval overlap exceeds this, "
                         "exit without writing training data — operator must "
                         "switch eval to held-out benchmarks.")
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, scan only the first N MMR1-RL rows (for "
                         "smoke testing).")
    args = ap.parse_args()

    if not args.mmr1_rl:
        sys.exit("ERROR: --mmr1-rl or $MMR1_RL_DATA must be set")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "images").mkdir(exist_ok=True)
    report_path = args.out_dir / "dedup_report.json"
    out_jsonl = args.out_dir / "train.jsonl"

    # 1. Load MMR1-RL.
    from datasets import load_from_disk  # type: ignore
    print(f">>> loading MMR1-RL from {args.mmr1_rl}", file=sys.stderr)
    raw = load_from_disk(args.mmr1_rl)
    if hasattr(raw, "keys") and not hasattr(raw, "features"):
        split = list(raw.keys())[0]
        mmr1 = raw[split]
        print(f">>> using split '{split}': {len(mmr1):,} rows", file=sys.stderr)
    else:
        mmr1 = raw
        print(f">>> single-split dataset: {len(mmr1):,} rows", file=sys.stderr)

    # 2. Build eval-side dedup index.
    opd_target_ids = set(json.loads(args.opd_target_ids.read_text())) \
        if args.opd_target_ids.read_text().lstrip().startswith("[") \
        else set().union(*json.loads(args.opd_target_ids.read_text()).values())
    eval_idx = _build_eval_index(args.eval_subset, opd_target_ids)

    # 3. Scan MMR1-RL.
    n_scan = args.limit if args.limit > 0 else len(mmr1)
    print(f">>> scanning {n_scan:,} MMR1-RL rows", file=sys.stderr)

    layer_b_hits: list[dict] = []   # by normalized question
    layer_d_hits: list[dict] = []   # by image sha256
    layer_bd_hits: list[dict] = []  # both
    clean_indices: list[int] = []
    mmr1_img_sha_cache: dict[int, str] = {}

    t0 = time.time()
    for idx in range(n_scan):
        row = mmr1[idx]
        problem = row.get("problem", "")
        images = row.get("images") or []
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
            layer_bd_hits.append({"idx": idx, "norm_q_match_ids": hit_b,
                                  "img_match_ids": hit_d})
        elif hit_b:
            layer_b_hits.append({"idx": idx, "norm_q_match_ids": hit_b})
        elif hit_d:
            layer_d_hits.append({"idx": idx, "img_match_ids": hit_d})
        else:
            clean_indices.append(idx)

        if (idx + 1) % 1000 == 0:
            print(f"   ... {idx+1:,}/{n_scan:,}  "
                  f"clean={len(clean_indices):,}  b={len(layer_b_hits):,}  "
                  f"d={len(layer_d_hits):,}  b&d={len(layer_bd_hits):,}  "
                  f"({time.time() - t0:.1f}s)", file=sys.stderr)

    n_layer_b = len(layer_b_hits) + len(layer_bd_hits)
    n_layer_d = len(layer_d_hits) + len(layer_bd_hits)
    rate_b = n_layer_b / n_scan
    rate_d = n_layer_d / n_scan

    # 4. Per-benchmark + opd-target hit attribution.
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

    report = {
        "mmr1_rl_path": args.mmr1_rl,
        "eval_subset_path": str(args.eval_subset),
        "opd_target_ids_count": len(opd_target_ids),
        "eval_total": eval_idx["n_total"],
        "mmr1_scanned": n_scan,
        "layer_b_normalized_question": {
            "count": n_layer_b, "rate": rate_b, "attribution": layer_b_attr,
        },
        "layer_d_image_hash": {
            "count": n_layer_d, "rate": rate_d, "attribution": layer_d_attr,
        },
        "layer_b_and_d_intersection": len(layer_bd_hits),
        "clean_count": len(clean_indices),
        "hard_stop_threshold": args.layer_b_threshold,
        "hard_stop_triggered": rate_b > args.layer_b_threshold,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(file=sys.stderr)
    print(f">>> dedup report -> {report_path}", file=sys.stderr)
    print(f"    layer-b (norm question):  {n_layer_b:,}/{n_scan:,} "
          f"({rate_b*100:.2f}%)  threshold {args.layer_b_threshold*100:.0f}%",
          file=sys.stderr)
    print(f"    layer-d (image sha256):   {n_layer_d:,}/{n_scan:,} "
          f"({rate_d*100:.2f}%)", file=sys.stderr)
    print(f"    layer-b ∩ layer-d:        {len(layer_bd_hits):,}", file=sys.stderr)
    print(f"    clean post-dedup:         {len(clean_indices):,}",
          file=sys.stderr)
    print(f"    layer-b benchmarks: {layer_b_attr['by_benchmark']}",
          file=sys.stderr)
    print(f"    layer-b opd_target hits: {layer_b_attr['n_opd_target']}",
          file=sys.stderr)

    # 5. Hard stop?
    if report["hard_stop_triggered"]:
        print(file=sys.stderr)
        print(f"!! HARD STOP: layer-b overlap {rate_b*100:.2f}% exceeds "
              f"threshold {args.layer_b_threshold*100:.0f}%.", file=sys.stderr)
        print("!! Per T1 plan §2.5, switch eval to held-out benchmarks "
              "(LogicVista / CharXiv / MMMU) before training.", file=sys.stderr)
        print("!! Dedup report written; training data NOT written.",
              file=sys.stderr)
        sys.exit(1)

    if len(clean_indices) < args.size:
        sys.exit(f"!! only {len(clean_indices):,} clean rows but --size "
                 f"{args.size:,} requested; reduce --size or use a bigger pool")

    # 6. Sample, save images, emit jsonl.
    random.seed(args.seed)
    sampled_indices = sorted(random.sample(clean_indices, args.size))
    print(file=sys.stderr)
    print(f">>> sampling {args.size:,} clean rows (seed={args.seed})", file=sys.stderr)

    n_written = 0
    t0 = time.time()
    with out_jsonl.open("w") as fout:
        for out_idx, src_idx in enumerate(sampled_indices):
            row = mmr1[src_idx]
            problem = row.get("problem", "")
            answer = row.get("answer", "")
            images = row.get("images") or []
            if not images:
                continue

            # Prepend MMR1 sysprompt — Uni-OPD's _build_messages will split
            # the resulting string on the <image> placeholder, producing
            # [text:"<sys> ", image, text:"\nquestion"] — byte-identical
            # to audit's _build_messages output ordering.
            full_problem = f"{MMR1_SYSTEM_PROMPT.strip()} {problem}"

            img = images[0]
            img_sha = mmr1_img_sha_cache.get(src_idx) or sha256_pil(img)
            img_path = args.out_dir / "images" / f"{img_sha}.png"
            if not img_path.exists():
                img.convert("RGB").save(img_path, format="PNG", optimize=False)

            record = {
                "id": f"mmr1_rl_v0_{out_idx:06d}",
                "problem": full_problem,
                "images": [str(img_path)],
                "answer": answer,
                "teacher_model": "MMR1-7B-RL",
                "raw_problem": problem,
                "source_row_idx": src_idx,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

            if (out_idx + 1) % 200 == 0:
                print(f"   ... {out_idx+1:,}/{args.size:,}  "
                      f"({time.time() - t0:.1f}s)", file=sys.stderr)

    print(file=sys.stderr)
    print(f">>> wrote {n_written:,} training rows -> {out_jsonl}", file=sys.stderr)
    print(f">>> images at {args.out_dir / 'images'} (unique by sha256)",
          file=sys.stderr)
    print(file=sys.stderr)
    print(">>> NEXT: punch list #3 — byte-level verify that Uni-OPD's "
          "Dataset processes this JSONL into the same chat-template prefix "
          "as run_audit_pass_sglang._build_chat_text with MMR1_SYSTEM_PROMPT.",
          file=sys.stderr)


if __name__ == "__main__":
    main()
