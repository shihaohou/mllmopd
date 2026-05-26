"""Step 3a — TAM-Boost post-process hook for Uni-OPD `--custom-reward-post-process-path`.

Wires precomputed TAM evidence (offline via `scripts/audit/tam_precompute_train_pool.py`)
into the training loop. Per-sample lookup → gate computation → unconditional
attach `sample.teacher_tam_weights` (ones on cache miss). Patches P20/P21/P22
in `scripts/setup/patch_uni_opd.sh` then plumb the tensor through to
`loss.py` where it multiplies the per-token OPD advantage.

Per docs/step3a-design-2026-05-26.md (v2) + GPT Phase-2 verdict refine §3:
  - Mandatory point #3: unconditional attach — cache miss → ones tensor
    (NOT skip), mirrors VD hook precedent in opd_diagnostics_hook.py
  - Mandatory point #4: cache key = (sample_id, response_hash) with
    response_length + tokenizer_id + tam_config_hash validation
  - Activation: MLLMOPD_USE_TAM_BOOST=1 env var (mirrors USE_VD_WEIGHTING)

This hook is a SUPERSET of opd_diagnostics_hook.post_process_rewards_with_diagnostics:
- Always runs canonical post_process_rewards (teacher_log_probs + response_correct)
- If MLLMOPD_USE_VD_WEIGHTING=1, delegates to the diagnostics hook for VD
- Then layers TAM-Boost on top
- A0 baseline = hook active but MLLMOPD_USE_TAM_BOOST=0 → identity over diagnostics
- A1 main = MLLMOPD_USE_TAM_BOOST=1 + cache JSONL set
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

# We deliberately avoid importing torch at module-import time; the hook is
# loaded by Uni-OPD's launcher in a context that may not have torch yet.
# torch.Tensor instantiation is deferred to the per-sample step.

from mllmopd.training.tam_gate import GateConfig, compute_weights, C_LOCAL_DEFAULT

# Re-export the canonical diagnostics hook so users can chain or fall back.
from mllmopd.training.opd_diagnostics_hook import (
    post_process_rewards,
    post_process_rewards_with_diagnostics,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Cache loading
# ============================================================================
_CACHE: dict | None = None        # dict[(sample_id, response_hash) → cache entry]
_CACHE_PATH: str | None = None    # the path we loaded from (for diagnostics)
_CACHE_META: dict | None = None   # global meta (tam_config_hash, tokenizer_id, ...)


def _decode_uint8_map(b64: str, h: int, w: int) -> np.ndarray:
    """Decode `_b64_uint8_map` output back to float (0..1). Mirror of
    tam_step3_preflight._decode_uint8_map."""
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size != h * w:
        raise ValueError(f"map size {arr.size} != H*W {h * w}")
    return arr.reshape(h, w).astype(np.float32) / 255.0


def _load_cache(path: str) -> tuple[dict, dict]:
    """Load TAM precompute JSONL. Returns (entries_by_key, global_meta).

    Cache file shape (one JSON per line; per `tam_precompute_train_pool.py`):
        {
          "sample_id":           str,
          "response_hash":       str (SHA256 of response_ids),
          "response_length":     int,
          "token_categories":    list[str] (length response_length),
          "tam_maps_uint8_b64":  list[str] (length response_length),
          "image_grid_thw":      [1, H_pre, W_pre],
          "map_h":               int,
          "map_w":               int,
          "tokenizer_id":        str,
          "tam_config_hash":     str,
          "tam_preproc_version": str,
        }
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"TAM cache JSONL not found: {p}")
    entries: dict = {}
    global_meta: dict = {}
    n_rows = 0
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_rows += 1
            key = (rec["sample_id"], rec["response_hash"])
            entries[key] = rec
            # First row sets global meta; subsequent rows should match.
            if not global_meta:
                global_meta = {
                    "tokenizer_id":        rec.get("tokenizer_id"),
                    "tam_config_hash":     rec.get("tam_config_hash"),
                    "tam_preproc_version": rec.get("tam_preproc_version"),
                }
    logger.info("[TAM-Boost] loaded cache %s: n_rows=%d, n_unique_keys=%d, "
                "tokenizer_id=%s, tam_config_hash=%s",
                p, n_rows, len(entries),
                global_meta.get("tokenizer_id"),
                global_meta.get("tam_config_hash"))
    return entries, global_meta


def _ensure_cache_loaded() -> tuple[dict | None, dict | None]:
    """Lazy-load cache on first hook invocation. Returns (None, None) if
    MLLMOPD_TAM_CACHE_JSONL not set."""
    global _CACHE, _CACHE_PATH, _CACHE_META
    if _CACHE is not None:
        return _CACHE, _CACHE_META
    cache_path = os.environ.get("MLLMOPD_TAM_CACHE_JSONL", "").strip()
    if not cache_path:
        logger.warning("[TAM-Boost] MLLMOPD_USE_TAM_BOOST=1 but "
                       "MLLMOPD_TAM_CACHE_JSONL unset; gate will run "
                       "with empty cache (every sample gets ones).")
        _CACHE = {}
        _CACHE_PATH = "<unset>"
        _CACHE_META = {}
        return _CACHE, _CACHE_META
    _CACHE, _CACHE_META = _load_cache(cache_path)
    _CACHE_PATH = cache_path
    return _CACHE, _CACHE_META


def _runtime_config() -> GateConfig:
    """Build GateConfig from env overrides (defaults match design lock)."""
    return GateConfig(
        K=float(os.environ.get("MLLMOPD_TAM_K", "0.20")),
        rho=float(os.environ.get("MLLMOPD_TAM_RHO", "0.30")),
        tau=float(os.environ.get("MLLMOPD_TAM_TAU", "0.70")),
        alpha=float(os.environ.get("MLLMOPD_TAM_ALPHA", "0.50")),
        mode=os.environ.get("MLLMOPD_TAM_MODE", "main"),
        C_local=C_LOCAL_DEFAULT,
        random_region_rate=float(os.environ.get("MLLMOPD_TAM_RANDOM_RATE", "0.40")),
        seed=int(os.environ.get("MLLMOPD_TAM_SEED", "4096")),
    )


def _hash_response_ids(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(list(ids)).encode("utf-8")).hexdigest()


# ============================================================================
# Per-sample TAM weight computation
# ============================================================================
def _compute_tam_weights_for_sample(
    sample: Any,
    cache: dict,
    cache_meta: dict,
    cfg: GateConfig,
    metrics: Counter,
) -> tuple[list[float], dict]:
    """Look up cache + compute weights for one sample. Returns
    (weights, diagnostics_dict). Always returns a list of length
    response_length (ones on any failure path).

    Increments fields on the shared metrics Counter:
        cache_hit, cache_miss, len_mismatch, hash_mismatch, config_hash_mismatch,
        empty_E_x, fallback_to_ones, success
    """
    R = int(getattr(sample, "response_length", 0) or 0)
    if R <= 0:
        metrics["fallback_to_ones"] += 1
        return [], {"reason": "zero_response_length"}

    # Recover response_ids if available; else use stored response_hash.
    response_ids = getattr(sample, "response_ids", None)
    if response_ids is None and hasattr(sample, "reward"):
        # Uni-OPD stores response_ids on sample directly; this branch is a
        # safety net. If neither path resolves, the hash-match will simply
        # fail and we'll fall back to ones.
        response_ids = None

    sample_id = getattr(sample, "index", None) or getattr(sample, "id", None) or "?"

    # Lookup: try (sample_id, response_hash) first; if response_ids exposed,
    # we can re-hash and confirm. Otherwise rely on sample-attached hash.
    response_hash = getattr(sample, "response_hash", None)
    if response_hash is None and response_ids is not None:
        response_hash = _hash_response_ids(list(response_ids))

    cache_entry = None
    if response_hash is not None:
        cache_entry = cache.get((sample_id, response_hash))

    if cache_entry is None:
        metrics["cache_miss"] += 1
        metrics["fallback_to_ones"] += 1
        return [1.0] * R, {"reason": "cache_miss", "sample_id": sample_id,
                           "response_hash": response_hash}

    # Validate length
    cache_R = int(cache_entry.get("response_length") or 0)
    if cache_R != R:
        metrics["len_mismatch"] += 1
        metrics["fallback_to_ones"] += 1
        return [1.0] * R, {"reason": "len_mismatch", "cache_R": cache_R, "R": R}

    # Validate config hash (warns if config was changed since precompute)
    if cache_meta and cache_entry.get("tam_config_hash") != cache_meta.get("tam_config_hash"):
        metrics["config_hash_mismatch"] += 1
        # Don't fallback here — config_hash mismatch usually means cache
        # was built with slightly different params, which is OK for smoke
        # but should bump tam_config_hash_mismatch_rate in metrics.

    # Decode maps + categories
    maps_b64 = cache_entry.get("tam_maps_uint8_b64") or []
    cats = cache_entry.get("token_categories") or []
    h = int(cache_entry.get("map_h") or 0)
    w = int(cache_entry.get("map_w") or 0)
    if not maps_b64 or len(maps_b64) != R or len(cats) != R or h == 0 or w == 0:
        metrics["fallback_to_ones"] += 1
        return [1.0] * R, {"reason": "malformed_cache_entry"}

    try:
        maps = [_decode_uint8_map(b64, h, w) for b64 in maps_b64]
    except Exception as e:  # noqa: BLE001
        metrics["fallback_to_ones"] += 1
        return [1.0] * R, {"reason": f"decode_error: {e!r}"}

    # Compute gate
    try:
        weights, info = compute_weights(
            maps, cats, config=cfg, sample_id=sample_id,
        )
    except Exception as e:  # noqa: BLE001
        metrics["fallback_to_ones"] += 1
        return [1.0] * R, {"reason": f"compute_weights_error: {e!r}"}

    metrics["cache_hit"] += 1
    metrics["success"] += 1
    if info.get("E_x_size", 0) == 0:
        metrics["empty_E_x"] += 1
    return weights, {
        "reason": "ok",
        "n_gate_fire": sum(info["gate_fire"]),
        "n_C_local":   info["n_C_local_positions"],
        "E_x_size":    info["E_x_size"],
    }


# ============================================================================
# Hook entry point
# ============================================================================
def post_process_rewards_with_tam_boost(args, samples, **kwargs):
    """Step 3a hook: diagnostics (+ optional VD weighting) + cached TAM-Boost.

    Order of operations per sample:
      1. Canonical reward post-process (writes sample.teacher_log_probs,
         sample.response_correct) — via opd_diagnostics_hook.
      2. If MLLMOPD_USE_VD_WEIGHTING=1: VD weights attached (via
         opd_diagnostics_hook chain).
      3. If MLLMOPD_USE_TAM_BOOST=1: cache lookup + gate compute →
         sample.teacher_tam_weights (ones on any failure path).

    A0 = hook active, MLLMOPD_USE_TAM_BOOST=0   → identity over (1)+(2)
    A1 = hook active, MLLMOPD_USE_TAM_BOOST=1   → +TAM weights
    """
    import torch

    # --- Step 1+2: delegate to the existing diagnostics hook ---
    result = post_process_rewards_with_diagnostics(args, samples, **kwargs)
    # Diagnostics hook returns (samples, ...) or just mutates in-place;
    # we mirror whatever shape it uses.

    use_tam_boost = os.environ.get("MLLMOPD_USE_TAM_BOOST", "0") == "1"
    if not use_tam_boost:
        return result

    # --- Step 3: TAM-Boost attach ---
    cache, cache_meta = _ensure_cache_loaded()
    cfg = _runtime_config()
    metrics: Counter = Counter()
    n_total = 0

    for sample in samples:
        R = int(getattr(sample, "response_length", 0) or 0)
        if R <= 0:
            # Attach empty tensor; downstream P22 guards on shape match.
            sample.teacher_tam_weights = torch.zeros(0, dtype=torch.float32)
            continue

        weights, _diag = _compute_tam_weights_for_sample(
            sample, cache or {}, cache_meta or {}, cfg, metrics,
        )
        # UNCONDITIONAL attach (GPT verdict §3 mandatory point 3)
        if len(weights) != R:
            # Defensive: any path that returned a different length gets unit
            weights = [1.0] * R
            metrics["len_mismatch_post"] += 1
        sample.teacher_tam_weights = torch.tensor(weights, dtype=torch.float32)
        n_total += 1

    # --- Log rolling metrics every batch ---
    if n_total > 0:
        hit = metrics.get("cache_hit", 0)
        miss = metrics.get("cache_miss", 0)
        len_mm = metrics.get("len_mismatch", 0)
        cfg_mm = metrics.get("config_hash_mismatch", 0)
        fallback = metrics.get("fallback_to_ones", 0)
        success = metrics.get("success", 0)
        empty_ex = metrics.get("empty_E_x", 0)
        logger.info(
            "[TAM-Boost] batch n=%d  hit=%.3f  miss=%.3f  len_mm=%.3f  "
            "cfg_mm=%.3f  fallback=%.3f  success=%.3f  empty_E_x=%.3f",
            n_total,
            hit / n_total, miss / n_total,
            len_mm / n_total, cfg_mm / n_total,
            fallback / n_total, success / n_total,
            empty_ex / n_total,
        )

    return result
