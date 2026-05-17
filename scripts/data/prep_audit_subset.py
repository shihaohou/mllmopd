"""
Build the Level-1 audit subset (1K-3K prompts) by sampling from multiple
multimodal benchmarks. Runs locally on Mac (only datasets API needed; no GPU).

Usage:
    python scripts/data/prep_audit_subset.py \
        --out data/audit/audit_subset_v0.jsonl \
        --size 2000

The output is a JSONL where each line is:
    {
        "id": str,
        "benchmark": str,
        "image_path_or_url": str | list[str],
        "question": str,
        "answer": str | None,
        "split": str,
        "meta": {...}
    }

Concrete loader functions are stubs; fill them in with the actual HF
dataset ids you use. We deliberately keep them as separate functions so you
can run a single benchmark while developing.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

try:
    from datasets import load_dataset  # type: ignore
except ImportError:  # pragma: no cover
    load_dataset = None  # the script can still print the plan without datasets installed


import os


# (benchmark name, default HF id, env var that, if set, overrides with a local dir)
BENCHMARK_REGISTRY = {
    "MathVista":        ("AI4Math/MathVista",       "MATHVISTA_PATH",         "testmini"),
    "MathVision":       ("MathLLMs/MathVision",     "MATHVISION_PATH",        "test"),
    "MathVerse":        ("AI4Math/MathVerse",       "MATHVERSE_PATH",         "testmini"),
    "LogicVista":       ("renjiepi/LogicVista",     "LOGICVISTA_PATH",        "test"),
    "ChartQA":          ("HuggingFaceM4/ChartQA",   "CHARTQA_PATH",           "test"),
    "HallusionBench":   ("PahaII/HallusionBench",   "HALLUSIONBENCH_PATH",    "test"),
    "CharXiv":          ("princeton-nlp/CharXiv",   "CHARXIV_PATH",           "validation"),
    "MMMU":             ("MMMU/MMMU",               "MMMU_PATH",              "validation"),
    "POPE_adversarial": ("lmms-lab/POPE",           "POPE_PATH",              "adversarial"),
}

BENCHMARK_MIX = {
    "MathVista":      0.20,
    "MathVision":     0.15,
    "MathVerse":      0.10,
    "LogicVista":     0.10,
    "ChartQA":        0.15,
    "HallusionBench": 0.15,
    "CharXiv":        0.10,
    "MMMU":           0.05,
}


def _open_dataset(local: str, hf_id: str, split: str):
    """Load a benchmark, trying loaders in order:

    (a) `load_from_disk(local)` when `local` contains `dataset_info.json`
        — this is the `Dataset.save_to_disk()` format (`data-*.arrow +
        dataset_info.json + state.json`), produced by e.g. an earlier
        `ds.save_to_disk(...)` snapshot. **Most of the local MMR1 dataset
        dumps on the dev box are this format.**
    (b) `load_dataset(local, split=split)` for a HF repo unpacked on disk.
    (c) `load_dataset("parquet", data_dir=local)` for a raw parquet dump.
    (d) `load_dataset(hf_id, split=split)` over the network as a last resort.

    When (a) returns a `DatasetDict`, we pick the matching split or the first
    available; when it returns a single `Dataset`, we use it as-is (and ignore
    `split` since `save_to_disk` outputs are already split-specific in practice).
    """
    from datasets import Dataset  # type: ignore

    if local and os.path.isdir(local):
        if os.path.isfile(os.path.join(local, "dataset_info.json")):
            from datasets import load_from_disk  # type: ignore
            ds = load_from_disk(local)
            if hasattr(ds, "keys"):
                key = split if split in ds else next(iter(ds.keys()))
                ds = ds[key]
            return ds
        try:
            return load_dataset(local, split=split)
        except Exception as e:
            print(f"!! load_dataset({local!r}) failed ({e}); trying parquet fallback", flush=True)
            return load_dataset("parquet", data_dir=local, split="train")
    return load_dataset(hf_id, split=split)


def load_records(name: str, n: int, seed: int, image_dir: Path) -> Iterable[dict]:
    """Load `n` records from `name`. Tries $<NAME>_PATH first, falls back to HF id.
    PIL images in the source dataset are persisted under `image_dir/<benchmark>/`
    and the JSONL stores the resulting path (not the PIL object)."""
    if load_dataset is None:
        raise RuntimeError("Install `datasets` first:  pip install datasets")
    if name not in BENCHMARK_REGISTRY:
        raise ValueError(f"Unknown benchmark: {name}")
    rng = random.Random(seed)

    hf_id, env_var, split = BENCHMARK_REGISTRY[name]
    local = os.environ.get(env_var, "")
    ds = _open_dataset(local, hf_id, split)
    print(f">>> {name}: {len(ds)} rows  ({'local' if local else 'hf'} {local or hf_id})", flush=True)

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    for idx in indices[:n]:
        rec = ds[idx]
        yield _normalize(name, idx, rec, image_dir)


def _save_image(img, image_dir: Path, benchmark: str, idx) -> str | list[str] | None:
    """Persist a PIL image (or list of them) to disk; return the path(s).

    No-op for None / already-string paths."""
    if img is None:
        return None
    if isinstance(img, str):
        return img
    if isinstance(img, list):
        out = []
        for j, item in enumerate(img):
            saved = _save_image(item, image_dir, benchmark, f"{idx}_{j}")
            if saved is not None:
                out.append(saved)
        return out or None
    if hasattr(img, "save"):
        sub = image_dir / benchmark
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / f"{idx}.png"
        if not path.exists():
            try:
                img.convert("RGB").save(path, format="PNG")
            except Exception as e:
                print(f"!! could not save {benchmark}/{idx}: {e}", flush=True)
                return None
        return str(path)
    return None


_CHOICE_MARKERS = (" A)", " A.", "(A)", "(A.", "\nA)", "\nA.")


def _maybe_append_choices(question: str, rec: dict) -> str:
    """If the benchmark provides MCQ choices that aren't already in the question
    text, append them with letter labels so the model knows what to pick.

    Conservative: only triggered when `choices` / `options` is a non-empty list
    AND the question doesn't already show standalone letter markers."""
    choices = rec.get("choices") or rec.get("options")
    if not choices or not isinstance(choices, list):
        return question
    if any(marker in question for marker in _CHOICE_MARKERS):
        return question
    labels = "ABCDEFGHIJ"
    lines = [f"({labels[i]}) {c}" for i, c in enumerate(choices) if i < len(labels)]
    return question.rstrip() + "\nChoices:\n" + "\n".join(lines)


def _normalize(benchmark: str, idx, rec: dict, image_dir: Path) -> dict:
    """Map heterogeneous benchmark schemas to the audit schema."""
    img = rec.get("image") or rec.get("images") or rec.get("decoded_image")
    question = (
        rec.get("question")
        or rec.get("query")
        or rec.get("problem")
        or rec.get("prompt")
        or ""
    )
    question = _maybe_append_choices(question, rec)
    answer = rec.get("answer") or rec.get("label") or rec.get("solution")
    image_path = _save_image(img, image_dir, benchmark, idx)
    # Drop the PIL image from meta (json.dumps default=str would mangle it)
    meta = {
        k: v for k, v in rec.items()
        if k not in {"image", "images", "decoded_image", "question", "answer"}
    }
    return {
        "id": f"{benchmark}/{idx}",
        "benchmark": benchmark,
        "image": image_path,
        "question": question,
        "answer": answer,
        "meta": meta,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260514)
    ap.add_argument("--only", nargs="*", help="optional: restrict to a subset of benchmarks")
    ap.add_argument("--image-dir", type=Path, default=None,
                    help="where to persist PIL images (default: <out_dir>/images/)")
    args = ap.parse_args()
    image_dir = args.image_dir or (args.out.parent / "images")
    image_dir.mkdir(parents=True, exist_ok=True)

    if args.only:
        # Treat --only as a uniform mix over the named benchmarks. This lets us
        # request POPE_adversarial / VLMBias / etc. without them being in the
        # default math/reasoning BENCHMARK_MIX.
        only_set = set(args.only)
        unknown = only_set - set(BENCHMARK_REGISTRY)
        if unknown:
            raise SystemExit(f"Unknown benchmark(s) in --only: {sorted(unknown)}")
        weights = {k: 1.0 / len(only_set) for k in only_set}
    else:
        weights = BENCHMARK_MIX

    plan = {k: max(1, int(round(args.size * w))) for k, w in weights.items()}
    print(">>> Plan:")
    for k, v in plan.items():
        print(f"    {k:<16s} {v}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for bench, n in plan.items():
            for rec in load_records(bench, n, seed=args.seed, image_dir=image_dir):
                f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
    print(f">>> Wrote {args.out}  (images under {image_dir})")


if __name__ == "__main__":
    main()
