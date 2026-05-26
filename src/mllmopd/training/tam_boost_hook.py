"""Step 3a — TAM-Boost post-process hook for Uni-OPD `--custom-reward-post-process-path`.

After the 2026-05-26 PM pivot (Path C → Path b, see
docs/step3a-design-2026-05-26-v3.md), the hook supports two modes:

  - `onpolicy_category` (v3 main path): classifies student rollout tokens
    on the fly (regex+spaCy v0.1.3 from scripts/audit/tam_sanity) and
    applies `w_t = 1 + α · 1[c_t ∈ C_local]`. No cache, no TAM, no
    teacher forward at train time. This is the deployed §Method.

  - `random_rate_matched` (v3 B2 control): fires on a rate-matched random
    subset of response tokens (NOT category-driven). Critical control —
    proves B1 vs B0 benefit isn't just from "more loss energy at this rate".

  - `cached_spatial` (v2 diagnostic): the original Path C code path that
    looks up precomputed TAM cache by (sample_id, response_hash) and
    applies the full spatial gate. Demoted to diagnostic / off-policy
    ablation only — retained for paper §Future Work / supplementary.

Mode selection: `MLLMOPD_TAM_HOOK_MODE` env var, default `onpolicy_category`.

UNCONDITIONAL ATTACH (all modes): cache miss / classification error /
length mismatch → fallback to ones(response_length). This mirrors the
existing VD hook pattern in opd_diagnostics_hook.py that prevents
whole-batch downgrade when any single sample fails.
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

from mllmopd.training.tam_gate import (
    GateConfig, compute_weights, C_LOCAL_DEFAULT, _stable_uniform,
)

# Re-export the canonical diagnostics hook so users can chain or fall back.
from mllmopd.training.opd_diagnostics_hook import (
    post_process_rewards,
    post_process_rewards_with_diagnostics,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Lazy-loaded classifier (for onpolicy_category mode)
# ============================================================================
_CLASSIFIER_FN = None
_LABEL_FN = None
_TOKENIZER_CACHE: dict[str, object] = {}


def _get_classifier_fns():
    """Lazy-import scripts/audit/tam_sanity classifier (regex+spaCy v0.1.3).

    The classifier lives in scripts/audit/ (outside src/) because it was
    built for offline audits. v3 hook needs it at training time, so we
    sys.path-insert and import directly. Module-level state (spaCy nlp,
    template regexes) is loaded once at first import.
    """
    global _CLASSIFIER_FN, _LABEL_FN
    if _CLASSIFIER_FN is not None:
        return _CLASSIFIER_FN, _LABEL_FN
    import sys
    audit_dir = Path(__file__).resolve().parents[3] / "scripts" / "audit"
    if str(audit_dir) not in sys.path:
        sys.path.insert(0, str(audit_dir))
    from tam_sanity import (  # type: ignore[import-not-found]
        _classify_tokens_v012,
        _label_response_tokens,
        _SPACY_AVAILABLE,
        _SPACY_VERSION,
    )
    _CLASSIFIER_FN = _classify_tokens_v012
    _LABEL_FN = _label_response_tokens
    logger.info("[TAM-Boost/onpolicy] loaded classifier from %s; "
                "spaCy_available=%s version=%s",
                audit_dir, _SPACY_AVAILABLE, _SPACY_VERSION)
    return _CLASSIFIER_FN, _LABEL_FN


def _get_tokenizer(args):
    """Load + cache HF tokenizer from args.hf_checkpoint (one per ckpt path).
    Onpolicy-category mode needs this to decode response_ids into the text
    pieces spaCy expects.
    """
    ckpt = getattr(args, "hf_checkpoint", None) or getattr(args, "load", None)
    if not ckpt:
        raise ValueError(
            "[TAM-Boost/onpolicy] cannot resolve tokenizer: no hf_checkpoint "
            "or --load in args. The hook needs args.hf_checkpoint to decode "
            "student rollout tokens for classification."
        )
    if ckpt in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[ckpt]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    _TOKENIZER_CACHE[ckpt] = tok
    logger.info("[TAM-Boost/onpolicy] loaded tokenizer from %s (vocab_size=%d)",
                ckpt, len(tok))
    return tok


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
# Mode: onpolicy_category (v3 main path) — classify student rollout tokens
# at training time, fire on C_local categories only. No cache, no TAM, no
# teacher forward at training time.
# ============================================================================
def _weights_onpolicy_category(
    sample: Any,
    tokenizer: Any,
    alpha: float,
    c_local: set,
    metrics: Counter,
) -> tuple[list[float], dict]:
    """Per-sample weights for onpolicy_category mode.

    Reads student's response_ids → decodes + classifies via spaCy+regex
    v0.1.3 (same classifier as Step 1a precompute) → fires on C_local
    categories. Returns ones-vector on any failure path.
    """
    R = int(getattr(sample, "response_length", 0) or 0)
    if R <= 0:
        metrics["fallback_to_ones"] += 1
        return [], {"reason": "zero_response_length"}

    response_ids = getattr(sample, "response_ids", None)
    if response_ids is None:
        metrics["no_response_ids"] += 1
        metrics["fallback_to_ones"] += 1
        return [1.0] * R, {"reason": "no_response_ids"}
    response_ids = list(response_ids)[-R:]   # last R = the response (drop prompt)

    try:
        classifier_fn, label_fn = _get_classifier_fns()
        labels = label_fn(response_ids, tokenizer)
        classification = classifier_fn(
            response_ids, tokenizer, labels["is_answer_token"]
        )
        cats = classification["token_category"]
    except Exception as e:  # noqa: BLE001
        metrics["classify_error"] += 1
        metrics["fallback_to_ones"] += 1
        return [1.0] * R, {"reason": f"classify_error: {type(e).__name__}: {e!s:.80}"}

    cats = list(cats)[:R]
    while len(cats) < R:
        cats.append("other")  # pad if classifier returned fewer

    weights: list[float] = []
    n_fire = 0
    for c in cats:
        if c in c_local:
            weights.append(1.0 + alpha)
            n_fire += 1
        else:
            weights.append(1.0)

    metrics["success"] += 1
    metrics["n_fire_total"] += n_fire
    metrics["n_tokens_total"] += R
    return weights, {"n_fire": n_fire, "R": R}


# ============================================================================
# Mode: random_rate_matched (v3 B2 control) — fires on a deterministic
# random subset of response tokens, ignoring category. Same fire rate as
# B1 → proves B1's benefit comes from CATEGORY, not just "more loss energy".
# ============================================================================
def _weights_random_rate_matched(
    sample: Any,
    alpha: float,
    target_rate: float,
    seed_base: int,
    metrics: Counter,
) -> tuple[list[float], dict]:
    """Per-sample weights for random_rate_matched mode.

    For each token at position t, fires if a stable_uniform(seed, sid, t)
    draw is < target_rate. Per-(sample, token) deterministic so the same
    set fires across multiple training epochs.
    """
    R = int(getattr(sample, "response_length", 0) or 0)
    if R <= 0:
        metrics["fallback_to_ones"] += 1
        return [], {"reason": "zero_response_length"}

    sample_id = getattr(sample, "index", None) or getattr(sample, "id", None) or "?"
    if not isinstance(sample_id, str):
        sample_id = str(sample_id)

    weights: list[float] = []
    n_fire = 0
    for t in range(R):
        u = _stable_uniform(seed_base, sample_id, t)
        if u < target_rate:
            weights.append(1.0 + alpha)
            n_fire += 1
        else:
            weights.append(1.0)

    metrics["success"] += 1
    metrics["n_fire_total"] += n_fire
    metrics["n_tokens_total"] += R
    return weights, {"n_fire": n_fire, "R": R}


# ============================================================================
# Hook entry point
# ============================================================================
VALID_HOOK_MODES = ("onpolicy_category", "random_rate_matched", "cached_spatial")


def post_process_rewards_with_tam_boost(args, samples, **kwargs):
    """Step 3a v3 hook: diagnostics + mode-dispatched per-token weight attach.

    Order per sample:
      1. Canonical reward post-process (sample.teacher_log_probs +
         sample.response_correct) — via opd_diagnostics_hook.
      2. If MLLMOPD_USE_VD_WEIGHTING=1: VD weights (delegated to diagnostics).
      3. If MLLMOPD_USE_TAM_BOOST=1: dispatch on MLLMOPD_TAM_HOOK_MODE:
           - onpolicy_category (v3 main, default)
           - random_rate_matched (v3 B2 control)
           - cached_spatial (v2 diagnostic, demoted)

    B0 = MLLMOPD_USE_TAM_BOOST=0     → no TAM-Boost path (identity)
    B1 = USE_TAM_BOOST=1 + mode=onpolicy_category
    B2 = USE_TAM_BOOST=1 + mode=random_rate_matched + MLLMOPD_TAM_TARGET_RATE=<B1 rate>
    """
    import torch

    # --- Step 1+2: delegate to the existing diagnostics hook ---
    result = post_process_rewards_with_diagnostics(args, samples, **kwargs)

    use_tam_boost = os.environ.get("MLLMOPD_USE_TAM_BOOST", "0") == "1"
    if not use_tam_boost:
        return result

    mode = os.environ.get("MLLMOPD_TAM_HOOK_MODE", "onpolicy_category").strip()
    if mode not in VALID_HOOK_MODES:
        raise ValueError(
            f"[TAM-Boost] invalid MLLMOPD_TAM_HOOK_MODE={mode!r}; "
            f"must be one of {VALID_HOOK_MODES}"
        )
    alpha = float(os.environ.get("MLLMOPD_TAM_ALPHA", "0.50"))
    metrics: Counter = Counter()
    n_total = 0

    # --- Mode-specific setup (resources loaded once per batch) ---
    cache = cache_meta = cfg = None
    tokenizer = None
    c_local = None
    target_rate = None
    seed_base = None
    if mode == "cached_spatial":
        cache, cache_meta = _ensure_cache_loaded()
        cfg = _runtime_config()
    elif mode == "onpolicy_category":
        tokenizer = _get_tokenizer(args)
        c_local = set(
            os.environ.get(
                "MLLMOPD_TAM_C_LOCAL",
                ",".join(C_LOCAL_DEFAULT),
            ).split(",")
        )
        # Strip empties / whitespace
        c_local = {c.strip() for c in c_local if c.strip()}
    elif mode == "random_rate_matched":
        target_rate = float(os.environ.get("MLLMOPD_TAM_TARGET_RATE", "0.12"))
        seed_base = int(os.environ.get("MLLMOPD_TAM_SEED", "4096"))

    # --- Per-sample dispatch ---
    for sample in samples:
        R = int(getattr(sample, "response_length", 0) or 0)
        if R <= 0:
            sample.teacher_tam_weights = torch.zeros(0, dtype=torch.float32)
            continue

        if mode == "cached_spatial":
            weights, _diag = _compute_tam_weights_for_sample(
                sample, cache or {}, cache_meta or {}, cfg, metrics,
            )
        elif mode == "onpolicy_category":
            weights, _diag = _weights_onpolicy_category(
                sample, tokenizer, alpha, c_local, metrics,
            )
        elif mode == "random_rate_matched":
            weights, _diag = _weights_random_rate_matched(
                sample, alpha, target_rate, seed_base, metrics,
            )

        if len(weights) != R:
            weights = [1.0] * R
            metrics["len_mismatch_post"] += 1
        sample.teacher_tam_weights = torch.tensor(weights, dtype=torch.float32)
        n_total += 1

    # --- Mode-aware metrics log ---
    if n_total > 0:
        if mode == "cached_spatial":
            hit = metrics.get("cache_hit", 0)
            miss = metrics.get("cache_miss", 0)
            len_mm = metrics.get("len_mismatch", 0)
            cfg_mm = metrics.get("config_hash_mismatch", 0)
            fallback = metrics.get("fallback_to_ones", 0)
            success = metrics.get("success", 0)
            empty_ex = metrics.get("empty_E_x", 0)
            logger.info(
                "[TAM-Boost/cached_spatial] batch n=%d  hit=%.3f  miss=%.3f  "
                "len_mm=%.3f  cfg_mm=%.3f  fallback=%.3f  success=%.3f  empty_E_x=%.3f",
                n_total,
                hit / n_total, miss / n_total,
                len_mm / n_total, cfg_mm / n_total,
                fallback / n_total, success / n_total,
                empty_ex / n_total,
            )
        else:  # onpolicy_category or random_rate_matched
            n_fire = metrics.get("n_fire_total", 0)
            n_tok = metrics.get("n_tokens_total", 1)
            fallback = metrics.get("fallback_to_ones", 0)
            classify_err = metrics.get("classify_error", 0)
            fire_rate = n_fire / max(1, n_tok)
            mean_w_est = 1.0 + alpha * fire_rate
            logger.info(
                "[TAM-Boost/%s] batch n=%d  fire_rate=%.4f  mean_w_est=%.4f  "
                "α=%.3f  n_fire=%d/%d  fallback=%d  classify_err=%d",
                mode, n_total, fire_rate, mean_w_est, alpha,
                n_fire, n_tok, fallback, classify_err,
            )

    return result
