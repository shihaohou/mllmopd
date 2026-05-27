"""Step 5 sample selector — bucket candidates by S0 vs S1 correctness.

For each candidate in the input pool, runs S0 + S1 greedy generation and
classifies the result into one of four buckets:

  - OPD_improved        : S0 wrong, S1 correct
  - OPD_failed          : S0 wrong, S1 wrong
  - Teacher_advantage   : id ∈ opd_target_ids, disjoint from above
  - Dataset_diversity   : ChartQA + MathVista, disjoint, visual-evidence dense

Outputs a stratified subset (default 200 rows) ready to be consumed by
`tam_step5_evidence_alignment.py`. Cached S0/S1 responses are kept so
the main runner can stratify alignment metrics by (s0_correct,
s1_correct) without re-running generation.

Two-stage flow inside one Python call:

  Stage 1: iterate candidates, run S0 + S1 generation, write
           `pool_predictions_<TS>.jsonl` with both responses + correctness
           judgments. Resumable: previously-judged ids are skipped.

  Stage 2: load all predictions, group into buckets, sample stratified
           {70/60/30/40} → write `tam_step5_samples_v0.jsonl`.

Stage 1 dominates wall clock (~2k candidates × 2 generations on multi-GPU);
Stage 2 is sub-second.

See `docs/step5-evidence-alignment-design.md` for the design rationale.

Usage::

    python -m scripts.audit.tam_step5_sample_selector \\
        --candidates data/audit/level1_subset_v0.jsonl \\
        --candidates data/eval/dev_mmr1_v0_1k.jsonl \\
        --opd-target-ids runs/audit/<t1_eval>/opd_target_ids_T1_3_vs_T1_0.json \\
        --s0 "$MMR1_3B_SFT_CKPT" \\
        --s1 "$MLLMOPD_RUNS/t1_v1p5b_T1_2_full_mm/ckpt/hf/step_230" \\
        --predictions-out data/audit/tam_step5_predictions_v0.jsonl \\
        --out data/audit/tam_step5_samples_v0.jsonl \\
        --n-improved 70 --n-failed 60 --n-teacher-advantage 30 \\
        --n-diversity 40 \\
        [--shard-id 0 --num-shards 8]   # Stage 1 only; Stage 2 always single-process
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Reuse heavy lifting from tam_sanity — same model loader, sysprompt placement.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tam_sanity import (  # noqa: E402
    _build_model, _build_messages, MMR1_SYSTEM_PROMPT,
)


# ============================================================================
# Correctness judge
# ============================================================================
_BOXED_RE = re.compile(r"\\boxed\s*\{([^}]*)\}")
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_CHOICE_LETTER_RE = re.compile(r"\b([A-D])\b")


def _extract_predicted(response: str) -> str:
    """Extract the model's predicted answer from a free-form MMR1 response.

    Priority: \\boxed{...} > <answer>...</answer> > last line stripped.
    Returns normalized lowercase string."""
    # 1. \\boxed{X}
    matches = _BOXED_RE.findall(response)
    if matches:
        return _normalize_answer(matches[-1])

    # 2. <answer>...</answer>
    tag_matches = _ANSWER_TAG_RE.findall(response)
    if tag_matches:
        inner = tag_matches[-1]
        # boxed inside answer tag is common in MMR1 outputs
        inner_boxed = _BOXED_RE.findall(inner)
        if inner_boxed:
            return _normalize_answer(inner_boxed[-1])
        return _normalize_answer(inner)

    # 3. Last non-empty line
    lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
    if lines:
        return _normalize_answer(lines[-1])
    return ""


def _normalize_answer(s: str) -> str:
    s = s.strip().lower()
    # Strip trailing punctuation / quotes
    s = s.strip(" .,:;!?'\"()[]")
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def _judge_correct(predicted: str, gold: str, qtype: str | None = None) -> bool:
    """Lightweight correctness judge.

    Multi-choice: extract A/B/C/D from both and compare.
    Numeric: parse float, tolerate small relative error.
    Free text: normalized exact-match OR pred-contains-gold."""
    if not gold:
        return False
    p = _normalize_answer(predicted)
    g = _normalize_answer(gold)

    # Multi-choice short-form
    if qtype == "multi_choice" or (len(g) == 1 and g in "abcd"):
        p_letter = _CHOICE_LETTER_RE.findall(p.upper())
        g_letter = _CHOICE_LETTER_RE.findall(g.upper())
        if g_letter:
            return bool(p_letter) and p_letter[0] == g_letter[0]

    # Numeric (parse both as float)
    try:
        # Strip common units / commas
        p_num = float(re.sub(r"[,$%]", "", p))
        g_num = float(re.sub(r"[,$%]", "", g))
        if abs(g_num) < 1e-9:
            return abs(p_num - g_num) < 1e-3
        rel = abs(p_num - g_num) / max(abs(g_num), 1e-9)
        return rel < 0.02  # 2% relative tolerance
    except (ValueError, TypeError):
        pass

    # Free text
    if p == g:
        return True
    # Permit pred containing gold (e.g. "the answer is 42" vs "42")
    if g and g in p:
        return True
    return False


# ============================================================================
# Generation
# ============================================================================
def _generate_one(processor, model, image, question: str, sysprompt: str,
                  max_new_tokens: int) -> str:
    import torch
    messages = _build_messages(question, image, sysprompt)
    chat = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(
        text=[chat], images=[image], return_tensors="pt", padding=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
                or processor.tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    response_ids = outputs.sequences[0][input_len:].cpu().tolist()
    eos_id = processor.tokenizer.eos_token_id
    if eos_id is not None and eos_id in response_ids:
        response_ids = response_ids[: response_ids.index(eos_id) + 1]
    return processor.tokenizer.decode(response_ids, skip_special_tokens=True)


# ============================================================================
# Pool loading
# ============================================================================
def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_opd_target_ids(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    with path.open() as f:
        data = json.load(f)
    out: set[str] = set()
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                out.update(str(x) for x in v)
    elif isinstance(data, list):
        out.update(str(x) for x in data)
    return out


def _assemble_candidates(candidate_paths: list[Path]) -> list[dict]:
    """Concatenate + dedupe by id. Keeps first occurrence (caller's order
    is intentional: dev_mmr1 first, then level1, then opd_target)."""
    seen: set[str] = set()
    out: list[dict] = []
    for p in candidate_paths:
        if not p.exists():
            print(f">>> WARNING: candidate file missing: {p}", file=sys.stderr)
            continue
        for rec in _load_jsonl(p):
            if "id" not in rec:
                continue
            if rec["id"] in seen:
                continue
            seen.add(rec["id"])
            out.append(rec)
    return out


# ============================================================================
# Stage 1 — prediction generation
# ============================================================================
def _load_existing_predictions(path: Path) -> dict:
    """Load already-judged predictions from a previous (interrupted) run."""
    if not path.exists():
        return {}
    out: dict = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def _resolve_image_path(rec: dict, image_root: Path) -> Path:
    image_path = rec["image"]
    p = Path(image_path)
    if not p.is_absolute():
        p = (image_root / p).resolve()
    if not p.exists():
        s = str(p).replace("\\", "/")
        if "data/audit/images/" in s:
            tail = s.rsplit("data/audit/images/", 1)[-1]
            cand = image_root / "data" / "audit" / "images" / tail
            if cand.exists():
                p = cand
    return p


def run_stage1(args) -> None:
    """Generate S0 + S1 responses and judge correctness for the candidate
    pool. Writes predictions JSONL incrementally; resumable."""
    from PIL import Image

    candidates = _assemble_candidates([Path(p) for p in args.candidates])
    print(f">>> assembled {len(candidates)} unique candidates from "
          f"{len(args.candidates)} files", file=sys.stderr)

    # Shard for multi-GPU
    if args.num_shards > 1:
        candidates = candidates[args.shard_id::args.num_shards]
        print(f">>> shard {args.shard_id}/{args.num_shards}: "
              f"{len(candidates)} candidates", file=sys.stderr)

    predictions_path = Path(args.predictions_out)
    if args.num_shards > 1:
        predictions_path = predictions_path.with_suffix(
            f".shard_{args.shard_id}.jsonl"
        )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    done = _load_existing_predictions(predictions_path)
    if done:
        print(f">>> resuming: {len(done)} already-predicted ids in "
              f"{predictions_path}", file=sys.stderr)
    todo = [r for r in candidates if r["id"] not in done]

    if not todo:
        print(">>> all candidates already predicted; nothing to do",
              file=sys.stderr)
        return

    image_root = Path(args.image_root)

    # Phase 1A: S0
    print(f">>> loading S0 ← {args.s0}", file=sys.stderr)
    s0_proc, s0_model = _build_model(args.s0)
    s0_responses: dict[str, str] = {}
    with predictions_path.open("a") as fout:
        for k, rec in enumerate(todo):
            try:
                img_path = _resolve_image_path(rec, image_root)
                image = Image.open(img_path).convert("RGB")
                s0_resp = _generate_one(
                    s0_proc, s0_model, image, rec["question"],
                    args.system_prompt, args.max_new_tokens,
                )
                s0_responses[rec["id"]] = s0_resp
            except Exception as e:  # noqa: BLE001
                print(f"!! S0 gen failed on {rec['id']}: {e!r}", file=sys.stderr)
                s0_responses[rec["id"]] = ""
            if (k + 1) % 50 == 0:
                print(f"  S0 [{k+1}/{len(todo)}]", file=sys.stderr)
    # Free S0 memory
    try:
        import torch
        del s0_model
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    # Phase 1B: S1
    print(f">>> loading S1 ← {args.s1}", file=sys.stderr)
    s1_proc, s1_model = _build_model(args.s1)
    with predictions_path.open("a") as fout:
        for k, rec in enumerate(todo):
            rid = rec["id"]
            if rid in done:
                continue
            try:
                img_path = _resolve_image_path(rec, image_root)
                image = Image.open(img_path).convert("RGB")
                s1_resp = _generate_one(
                    s1_proc, s1_model, image, rec["question"],
                    args.system_prompt, args.max_new_tokens,
                )
            except Exception as e:  # noqa: BLE001
                print(f"!! S1 gen failed on {rid}: {e!r}", file=sys.stderr)
                s1_resp = ""

            s0_resp = s0_responses.get(rid, "")
            gold = rec.get("answer", "")
            qtype = rec.get("meta", {}).get("question_type")
            s0_pred = _extract_predicted(s0_resp)
            s1_pred = _extract_predicted(s1_resp)
            s0_correct = _judge_correct(s0_pred, gold, qtype)
            s1_correct = _judge_correct(s1_pred, gold, qtype)

            row = {
                "id": rid,
                "benchmark": rec.get("benchmark"),
                "image": rec.get("image"),
                "question": rec.get("question"),
                "answer": gold,
                "meta": rec.get("meta", {}),
                "s0_response_text": s0_resp,
                "s1_response_text": s1_resp,
                "s0_predicted": s0_pred,
                "s1_predicted": s1_pred,
                "s0_correct": bool(s0_correct),
                "s1_correct": bool(s1_correct),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if (k + 1) % 50 == 0:
                print(f"  S1 [{k+1}/{len(todo)}]", file=sys.stderr)


# ============================================================================
# Stage 2 — bucket assignment + stratified sampling
# ============================================================================
def _merge_shards(base_path: Path, num_shards: int) -> list[dict]:
    """Merge per-shard predictions files into a unified list."""
    out: list[dict] = []
    seen: set[str] = set()
    if num_shards <= 1:
        # Single-shard mode — base file is canonical
        if base_path.exists():
            for rec in _load_jsonl(base_path):
                if rec["id"] not in seen:
                    seen.add(rec["id"])
                    out.append(rec)
        return out
    for i in range(num_shards):
        sp = base_path.with_suffix(f".shard_{i}.jsonl")
        if not sp.exists():
            print(f">>> WARNING: missing shard predictions {sp}", file=sys.stderr)
            continue
        for rec in _load_jsonl(sp):
            if rec["id"] not in seen:
                seen.add(rec["id"])
                out.append(rec)
    return out


def run_stage2(args) -> None:
    """Bucket the predictions + stratified pick."""
    rng = random.Random(args.seed)

    predictions_path = Path(args.predictions_out)
    preds = _merge_shards(predictions_path, args.num_shards)
    if not preds:
        sys.exit(f"!! no predictions found at {predictions_path}* — run Stage 1 first")
    print(f">>> loaded {len(preds)} judged candidates", file=sys.stderr)

    opd_target_ids = _load_opd_target_ids(
        Path(args.opd_target_ids) if args.opd_target_ids else None
    )
    if opd_target_ids:
        print(f">>> opd_target ids: {len(opd_target_ids)}", file=sys.stderr)
    else:
        print(">>> no opd_target ids; Teacher_advantage bucket will draw from "
              "general visual-critical fallback", file=sys.stderr)

    diversity_benchmarks = {"ChartQA", "MathVista"}

    # ----- Bucket assignment (each candidate gets exactly one bucket) -----
    by_bucket: dict[str, list[dict]] = {
        "OPD_improved": [],
        "OPD_failed": [],
        "Teacher_advantage": [],
        "Dataset_diversity": [],
    }
    used_ids: set[str] = set()

    # 1. Improved (S0 wrong → S1 correct)
    for r in preds:
        if not r["s0_correct"] and r["s1_correct"]:
            by_bucket["OPD_improved"].append(r)
            used_ids.add(r["id"])

    # 2. Failed (S0 wrong & S1 wrong)
    for r in preds:
        if r["id"] in used_ids:
            continue
        if not r["s0_correct"] and not r["s1_correct"]:
            by_bucket["OPD_failed"].append(r)
            used_ids.add(r["id"])

    # 3. Teacher_advantage (id ∈ opd_target_ids, disjoint)
    for r in preds:
        if r["id"] in used_ids:
            continue
        if r["id"] in opd_target_ids:
            by_bucket["Teacher_advantage"].append(r)
            used_ids.add(r["id"])

    # 4. Dataset_diversity (ChartQA + MathVista, disjoint)
    for r in preds:
        if r["id"] in used_ids:
            continue
        if r.get("benchmark") in diversity_benchmarks:
            by_bucket["Dataset_diversity"].append(r)
            used_ids.add(r["id"])

    print("\n>>> bucket sizes (raw pool):", file=sys.stderr)
    for b, lst in by_bucket.items():
        print(f"    {b}: {len(lst)}", file=sys.stderr)

    # ----- Stratified pick -----
    targets = {
        "OPD_improved": args.n_improved,
        "OPD_failed": args.n_failed,
        "Teacher_advantage": args.n_teacher_advantage,
        "Dataset_diversity": args.n_diversity,
    }
    picked: list[dict] = []
    deficits: dict[str, int] = {}
    for b, n in targets.items():
        pool = by_bucket[b]
        rng.shuffle(pool)
        take = pool[:n]
        for r in take:
            r2 = dict(r)
            r2["bucket"] = b
            picked.append(r2)
        if len(take) < n:
            deficits[b] = n - len(take)
            print(f"!! deficit on {b}: needed {n}, got {len(take)}",
                  file=sys.stderr)

    # ----- Re-fill deficits from Dataset_diversity surplus -----
    deficit_total = sum(deficits.values())
    if deficit_total > 0:
        used_so_far = {r["id"] for r in picked}
        diversity_extras = [r for r in by_bucket["Dataset_diversity"]
                            if r["id"] not in used_so_far]
        rng.shuffle(diversity_extras)
        for r in diversity_extras[:deficit_total]:
            r2 = dict(r)
            r2["bucket"] = "Dataset_diversity"
            picked.append(r2)
        if len(picked) < sum(targets.values()):
            # Final fallback: any remaining OPD_failed (usually plentiful)
            failed_extras = [r for r in by_bucket["OPD_failed"]
                             if r["id"] not in {x["id"] for x in picked}]
            rng.shuffle(failed_extras)
            need = sum(targets.values()) - len(picked)
            for r in failed_extras[:need]:
                r2 = dict(r)
                r2["bucket"] = "OPD_failed"
                picked.append(r2)

    print(f"\n>>> final stratified pick: n = {len(picked)}", file=sys.stderr)
    cnt = Counter(r["bucket"] for r in picked)
    for b, n in cnt.items():
        print(f"    {b}: {n}", file=sys.stderr)

    # ----- Write output -----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f">>> wrote {len(picked)} samples to {out_path}", file=sys.stderr)

    # Also write a summary
    summary_path = out_path.with_suffix(".summary.txt")
    with summary_path.open("w") as f:
        f.write(f"# Step 5 sample selector  (commit="
                f"{os.environ.get('MLLMOPD_CODE_COMMIT', 'unknown')})\n")
        f.write(f"candidates input: {args.candidates}\n")
        f.write(f"S0 = {args.s0}\n")
        f.write(f"S1 = {args.s1}\n")
        f.write(f"opd_target ids = {len(opd_target_ids)}\n")
        f.write(f"n_predictions = {len(preds)}\n\n")
        f.write("raw bucket sizes:\n")
        for b, lst in by_bucket.items():
            f.write(f"  {b}: {len(lst)}\n")
        f.write("\nfinal pick:\n")
        for b, n in cnt.items():
            f.write(f"  {b}: {n}\n")
        if deficits:
            f.write("\ndeficits (filled from Dataset_diversity):\n")
            for b, n in deficits.items():
                f.write(f"  {b}: short by {n}\n")
        f.write(f"\noutput JSONL: {out_path}\n")


# ============================================================================
# Main
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", action="append", required=True,
                    help="Candidate JSONL (repeat for multiple sources)")
    ap.add_argument("--s0", required=True, help="S0 base student ckpt path")
    ap.add_argument("--s1", required=True, help="S1 OPD student ckpt path")
    ap.add_argument("--predictions-out", required=True,
                    help="Intermediate predictions JSONL (cached S0+S1 responses)")
    ap.add_argument("--out", required=True,
                    help="Final stratified samples JSONL")
    ap.add_argument("--opd-target-ids", default=None,
                    help="JSON file mapping benchmark→list of opd_target ids")
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--system-prompt", default=MMR1_SYSTEM_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--n-improved", type=int, default=70)
    ap.add_argument("--n-failed", type=int, default=60)
    ap.add_argument("--n-teacher-advantage", type=int, default=30)
    ap.add_argument("--n-diversity", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--stage", choices=["1", "2", "both"], default="both",
                    help="1=predict only, 2=bucket only, both=run both")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args(argv)

    if args.stage in ("1", "both"):
        print(">>> Stage 1: generating S0 + S1 predictions", file=sys.stderr)
        t0 = time.time()
        run_stage1(args)
        print(f">>> Stage 1 done in {time.time() - t0:.0f}s", file=sys.stderr)

    if args.stage in ("2", "both"):
        # If we just finished Stage 1 multi-shard, the shards are not yet
        # merged. Stage 2 explicitly handles that via _merge_shards.
        if args.num_shards > 1 and args.stage == "both":
            print(">>> NOTE: multi-shard Stage 1 — Stage 2 should be run once "
                  "after all shards complete (separately, with --stage 2).",
                  file=sys.stderr)
            if args.shard_id != 0:
                # Only rank-0 runs Stage 2 in multi-shard "both" mode; others exit.
                print(">>> shard != 0; skipping Stage 2", file=sys.stderr)
                return 0
        print("\n>>> Stage 2: bucketing + stratified sampling", file=sys.stderr)
        run_stage2(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
