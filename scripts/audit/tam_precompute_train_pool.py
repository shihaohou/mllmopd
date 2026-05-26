"""Step 3a Phase 2 — offline TAM cache builder for training-pool samples.

Converts a `tam_step1a.py --skip-student` teacher_cache.jsonl into the
hook-ready cache JSONL consumed by
`src/mllmopd/training/tam_boost_hook.py`. The converter adds:

  - response_hash (sha256 of response_ids)        ← matches hook lookup key
  - tam_config_hash (sha256 of GateConfig + classifier version)
  - tokenizer_id  (model name + vocab hash)
  - tam_preproc_version pass-through

Two-step workflow:

    # 1. Run tam_step1a in teacher-only mode on the training pool:
    SKIP_STUDENT=1 SUBSET=data/audit/train_pool_subset.jsonl \\
        RUN_ID=tam_precompute_<TS> NUM_GPUS=8 \\
        bash scripts/audit/run_tam_step1a.sh

    # 2. Convert teacher_cache → hook-ready tam_cache:
    python -m scripts.audit.tam_precompute_train_pool \\
        --teacher-cache runs/audit/tam_precompute_<TS>/teacher_cache.jsonl \\
        --tokenizer-id "MMR1-7B-RL@<vocab_hash:16>" \\
        --out-jsonl runs/audit/tam_precompute_<TS>/tam_cache.jsonl \\
        [--K 0.20 --rho 0.30 --tau 0.70 --alpha 0.50]   # config_hash inputs

Then point the launcher at `runs/audit/tam_precompute_<TS>/tam_cache.jsonl`
via `MLLMOPD_TAM_CACHE_JSONL`.

NOTE: This intentionally does NOT re-run inference. Inference is the
expensive part (TAM extraction needs eager attention; ~17 min teacher
pass per 200 samples × 1 ckpt on H800 8-GPU). Re-converting after a
GateConfig knob change is fast — the per-token TAM maps don't depend on
K/ρ/τ/α (those are applied at hook time).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hash_response_ids(ids: list[int]) -> str:
    return _sha256_hex(json.dumps(list(ids)).encode("utf-8"))


def _compute_tam_config_hash(K: float, rho: float, tau: float, alpha: float,
                              C_local: tuple[str, ...],
                              classifier_version: str,
                              tam_preproc_version: str) -> str:
    """Stable hash over the gate config + classifier identity. Hook
    compares this against its runtime config to detect post-precompute
    drift (e.g. someone changed τ without rebuilding cache)."""
    blob = json.dumps({
        "K": K, "rho": rho, "tau": tau, "alpha": alpha,
        "C_local": list(C_local),
        "classifier_version":  classifier_version,
        "tam_preproc_version": tam_preproc_version,
    }, sort_keys=True).encode("utf-8")
    return _sha256_hex(blob)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--teacher-cache", type=Path, required=True,
                    help="tam_step1a teacher_cache.jsonl (from "
                         "--skip-student run on training pool)")
    ap.add_argument("--out-jsonl", type=Path, required=True,
                    help="Hook-ready cache JSONL output path")
    ap.add_argument("--tokenizer-id", type=str, required=True,
                    help="Model tokenizer fingerprint, e.g. "
                         "'MMR1-7B-RL@sha256:abc...'. Echoed into every "
                         "cache entry; hook re-checks at training time.")
    # GateConfig knobs that compose the tam_config_hash
    ap.add_argument("--K", type=float, default=0.20)
    ap.add_argument("--rho", type=float, default=0.30)
    ap.add_argument("--tau", type=float, default=0.70)
    ap.add_argument("--alpha", type=float, default=0.50)
    ap.add_argument("--C-local", type=str,
                    default="content_noun,visual_attribute,proper_noun")
    ap.add_argument("--classifier-version", type=str,
                    default="regex+spacy_align:v0.1.3")
    ap.add_argument("--tam-preproc-version", type=str, default="v0.1.3")
    args = ap.parse_args(argv)

    C_local = tuple(c.strip() for c in args.C_local.split(",") if c.strip())
    tam_config_hash = _compute_tam_config_hash(
        K=args.K, rho=args.rho, tau=args.tau, alpha=args.alpha,
        C_local=C_local,
        classifier_version=args.classifier_version,
        tam_preproc_version=args.tam_preproc_version,
    )
    print(f">>> tam_config_hash = {tam_config_hash[:16]}...", file=sys.stderr)
    print(f">>> tokenizer_id    = {args.tokenizer_id}", file=sys.stderr)

    n_in = 0
    n_out = 0
    n_skip_no_tam = 0
    n_skip_no_maps = 0
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.teacher_cache.open() as fin, args.out_jsonl.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            tcache = json.loads(line)

            # Skip invalid teacher passes
            if not tcache.get("tam_valid"):
                n_skip_no_tam += 1
                continue

            response_ids = tcache.get("response_ids") or []
            R = int(tcache.get("response_length") or 0)
            if R == 0 or len(response_ids) < R:
                n_skip_no_tam += 1
                continue
            response_ids = list(response_ids)[:R]

            # Full per-token TAM maps live under `_response_maps_b64` in
            # teacher_cache (see tam_step1a:459). For Step 3a we need ALL R
            # maps, not the K=20 subset.
            maps_b64 = tcache.get("_response_maps_b64") or []
            if not maps_b64 or len(maps_b64) < R:
                n_skip_no_maps += 1
                continue

            classification = tcache.get("classification") or {}
            token_categories = classification.get("token_category") or []
            if len(token_categories) < R:
                # Pad / truncate defensively
                token_categories = (token_categories + ["other"] * R)[:R]

            grid_thw = tcache.get("image_grid_thw") or []
            vision_shape = tcache.get("vision_shape") or []
            map_h = int(vision_shape[0]) if vision_shape else 0
            map_w = int(vision_shape[1]) if vision_shape else 0

            row_out = {
                "sample_id":           tcache["id"],
                "response_hash":       _hash_response_ids(response_ids),
                "response_length":     R,
                "response_ids":        response_ids,
                "token_categories":    token_categories[:R],
                "tam_maps_uint8_b64":  maps_b64[:R],
                "image_grid_thw":      grid_thw,
                "map_h":               map_h,
                "map_w":               map_w,
                "tokenizer_id":        args.tokenizer_id,
                "tam_config_hash":     tam_config_hash,
                "tam_preproc_version": args.tam_preproc_version,
                "classifier_version":  args.classifier_version,
            }
            fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            n_out += 1

    print(f">>> wrote {n_out}/{n_in} entries to {args.out_jsonl}", file=sys.stderr)
    print(f"    skipped: no_tam={n_skip_no_tam}  no_maps={n_skip_no_maps}",
          file=sys.stderr)
    if n_out == 0:
        print("!! WARNING: 0 cache entries written; hook will fall back to "
              "ones on every sample.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
