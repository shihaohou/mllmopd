"""Step 5 TAM evidence alignment runner — three-model audit on a shared rollout.

For each sample in the input JSONL (output of tam_step5_sample_selector):

  Pass 1: S1 generates a greedy rollout `y` (max-new=4096) under the
          MMR1 system prompt. Cached to rollout_cache.jsonl so the three
          model passes can reuse it.

  Pass 2: One model at a time (T, S0, S1) is loaded; teacher-forced
          forward on (prompt + y), then `_tam_core.TAM` produces a
          per-token activation map for every response token. Per-model
          output is written to `tam_per_model_<X>.jsonl`.

  Pass 3: Per-token alignment metrics (Top20% IoU, JS, Cosine) are
          computed offline by merging the three per-model files. Written
          to `alignment.jsonl`. Single-process; cheap.

Design doc:    docs/step5-evidence-alignment-design.md
Schema bumped: step5_schema_version v0.1

Heavy reuse of tam_sanity helpers (classifier v0.1.3, model loader, sysprompt,
QC, b64 encoding). The forward-path inside `model_pass_with_tam` mirrors the
v0.1.3 OOM-safe pattern from tam_step1a (bare generate + single teacher-
forced fwd), but since Pass 2 starts from a cached response_ids list it
SKIPS the generate phase entirely.

Usage::

    python -m scripts.audit.tam_step5_evidence_alignment \\
        --samples data/audit/tam_step5_samples_v0.jsonl \\
        --teacher "$MMR1_7B_RL_CKPT" \\
        --s0      "$MMR1_3B_SFT_CKPT" \\
        --s1      "$MLLMOPD_RUNS/t1_v1p5b_T1_2_full_mm/ckpt/hf/step_230" \\
        --out-dir "$MLLMOPD_RUNS/audit/tam_step5_$(date +%Y%m%d-%H%M%S)" \\
        --max-new-tokens 4096 \\
        [--shard-id 0 --num-shards 8]   # data-parallel fan-out
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tam_sanity import (  # noqa: E402
    _build_model, _build_messages,
    _classify_tokens_v012,
    _label_response_tokens,
    _b64_uint8_map, _image_sha256, _tokenizer_vocab_hash,
    _sha256_hex,
    MMR1_SYSTEM_PROMPT, QWEN_VL_SPECIAL_IDS,
)
from _tam_core import TAM, tam_scalars  # noqa: E402


# ============================================================================
# Image loading (parity with tam_step1a._load_image_for_sample, sans corruption)
# ============================================================================
def _load_image_for_rec(rec: dict, image_root: Path):
    from PIL import Image
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
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")
    return Image.open(p).convert("RGB"), p


# ============================================================================
# Pass 1 — S1 greedy rollout
# ============================================================================
def s1_rollout(processor, model, rec, image, args) -> dict:
    """Greedy generation from S1; returns response_ids + text + length.

    Bare generate — no hidden_states / scores / attentions. Memory-safe.
    """
    import torch
    messages = _build_messages(rec["question"], image, args.system_prompt)
    chat = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(
        text=[chat], images=[image], return_tensors="pt", padding=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    t0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
                or processor.tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    dt_gen = time.time() - t0

    response_ids = outputs.sequences[0][input_len:].cpu().tolist()
    eos_id = processor.tokenizer.eos_token_id
    response_length = len(response_ids)
    if eos_id is not None and eos_id in response_ids:
        response_length = response_ids.index(eos_id) + 1
    response_ids = response_ids[:response_length]
    response_text = processor.tokenizer.decode(response_ids, skip_special_tokens=False)
    response_hash = _sha256_hex(json.dumps(response_ids).encode())

    del outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "response_ids":   response_ids,
        "response_length": response_length,
        "response_text":   response_text,
        "response_hash":   response_hash,
        "rollout_gen_s":   dt_gen,
        "prompt_len":      input_len,
    }


# ============================================================================
# Pass 2 — per-model TAM extraction on the SHARED rollout
# ============================================================================
def model_pass_with_tam(processor, model, rec, image,
                        response_ids: list[int], args) -> dict:
    """Teacher-forced forward on (prompt + response_ids) under the given
    model. Returns per-response-token TAM maps + scalars + lp.

    Mirrors the v0.1.3 OOM-safe pattern from tam_step1a.teacher_pass:
    instead of `generate(output_hidden_states=True)` (full-seq HS each
    step → OOM on high-res), we do a single teacher-forced forward and
    rebuild `logit_list` with the TAM-expected shape:
        logit_list[0]   : (1, prompt_len, V)   — all prompt positions
        logit_list[r>0] : (1, 1, V)            — single new token

    Attention baseline is INTENTIONALLY SKIPPED (Step 5 doesn't need it;
    eager-attn cost is ~3-5× and saves no comparable signal — already
    settled by Step 0 Pearson r = 0.032).
    """
    import torch

    # ----- Build the (prompt + response) input -----
    messages = _build_messages(rec["question"], image, args.system_prompt)
    chat = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(
        text=[chat], images=[image], return_tensors="pt", padding=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    input_ids_full = inputs["input_ids"][0].tolist()

    response_tensor = torch.tensor(
        response_ids, device=model.device
    ).unsqueeze(0)
    full_ids = torch.cat([inputs["input_ids"], response_tensor], dim=1)
    full_mask = torch.cat([
        inputs["attention_mask"],
        torch.ones_like(response_tensor),
    ], dim=1)
    fwd_kwargs = dict(inputs)
    fwd_kwargs["input_ids"] = full_ids
    fwd_kwargs["attention_mask"] = full_mask
    response_length = len(response_ids)

    t0 = time.time()
    with torch.inference_mode():
        out_full = model(
            **fwd_kwargs,
            output_hidden_states=False,   # logits suffice for TAM + lp
            output_attentions=False,      # skip — Step 5 does not need
        )
    dt_fwd = time.time() - t0

    full_logits_2d = out_full.logits[0]  # (T_full, V)
    T_full = full_logits_2d.shape[0]

    # Per-token lp on response (predictor for response[t] sits at index
    # input_len + t - 1).
    lp: list[float] = []
    for t in range(response_length):
        pred_idx = input_len + t - 1
        log_probs = torch.log_softmax(full_logits_2d[pred_idx], dim=-1)
        lp.append(float(log_probs[response_ids[t]]))

    # Build logit_list with the v0.1.3 TAM-expected shape
    full_logits_3d = full_logits_2d.unsqueeze(0)  # (1, T_full, V) — view
    logit_list: list = []
    for r in range(response_length):
        if r == 0:
            logit_list.append(full_logits_3d[:, :input_len, :])
        else:
            logit_list.append(
                full_logits_3d[:, input_len + r - 1: input_len + r, :]
            )

    grid_thw = inputs["image_grid_thw"][0].tolist()
    vision_shape = (grid_thw[1] // 2, grid_thw[2] // 2)
    n_patches = int(vision_shape[0] * vision_shape[1])
    tokens_full_list = full_ids[0, : input_len + response_length].cpu().tolist()

    # ----- Run TAM per response token -----
    # Length-alignment invariant: every returned list MUST have length
    # response_length. If TAM throws mid-loop, we pad the remainder with
    # zero-maps + tam_scalars-of-zeros and mark `_per_token_valid[t]=False`.
    # Pass 3 will refuse to use any sample where any of T/S0/S1 has
    # `tam_valid=False` (sample-level drop, not token-level).
    t1 = time.time()
    img_scores_list: list = []
    response_maps: list = []
    per_token_valid: list[bool] = []
    first_failure_idx: int | None = None
    tam_failure = None
    for i in range(response_length):
        try:
            m = TAM(
                tokens=tokens_full_list,
                vision_shape=vision_shape,
                logit_list=logit_list,
                special_ids=QWEN_VL_SPECIAL_IDS,
                processor=processor,
                target_token=i,
                img_scores_list=img_scores_list,
                out_prompt_maps=None,  # Step 5 doesn't store prompt-token maps
            )
            response_maps.append(m)
            per_token_valid.append(bool(np.asarray(m).sum() > 0))
        except Exception as e:  # noqa: BLE001
            if first_failure_idx is None:
                first_failure_idx = i
                tam_failure = f"runtime@t={i}:{type(e).__name__}:{e!s:.120}"
            # Pad with zero map so downstream length invariant holds
            response_maps.append(np.zeros(vision_shape, dtype=np.float32))
            per_token_valid.append(False)
    dt_tam = time.time() - t1

    assert len(response_maps) == response_length, \
        f"length invariant broken: maps={len(response_maps)} vs R={response_length}"
    assert len(per_token_valid) == response_length

    # Sample-level tam_valid: TRUE only if no per-token failure AND at
    # least one token has a non-degenerate map (handles all-zero
    # degenerate runs).
    n_per_token_valid = sum(per_token_valid)
    tam_valid = (tam_failure is None) and (n_per_token_valid > 0)

    # Per-token scalars (computed on the padded maps; tam_scalars of a
    # zero-map returns the documented uniform-equivalent defaults).
    response_scalars = [tam_scalars(m) for m in response_maps]

    # Cleanup before returning
    del logit_list, full_logits_3d, full_logits_2d
    try:
        del out_full
    except NameError:
        pass
    del response_tensor, full_ids, full_mask, fwd_kwargs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "lp":                       lp,
        "tam_mass_top10":           [s["tam_mass_top10"] for s in response_scalars],
        "tam_mass_top20":           [s["tam_mass_top20"] for s in response_scalars],
        "tam_mass_top40":           [s["tam_mass_top40"] for s in response_scalars],
        "tam_entropy":              [s["tam_entropy"] for s in response_scalars],
        "tam_entropy_norm":         [s["tam_entropy_norm"] for s in response_scalars],
        "tam_effective_patch_frac": [s["tam_effective_patch_frac"] for s in response_scalars],
        "_response_maps_b64":       [_b64_uint8_map(m) for m in response_maps],
        "_response_maps_arrays":    response_maps,  # in-memory only, NOT serialized
        "_per_token_valid":         per_token_valid,
        "n_per_token_valid":        n_per_token_valid,
        "tam_valid":                bool(tam_valid),
        "tam_failure_reason":       tam_failure,
        "tam_failure_first_idx":    first_failure_idx,
        "vision_shape":             list(vision_shape),
        "n_patches":                n_patches,
        "_timings":                 {"fwd_s": dt_fwd, "tam_s": dt_tam},
    }


# ============================================================================
# Pass 3 — alignment metrics
# ============================================================================
def _normalize_map_to_prob(m: np.ndarray) -> np.ndarray:
    """Convert a non-negative map into a patch-probability vector."""
    flat = m.flatten().astype(np.float64)
    flat = np.clip(flat, 0.0, None)
    total = flat.sum()
    if total < 1e-12:
        return np.full_like(flat, 1.0 / flat.size)
    return flat / total


def _iou_topk(m1: np.ndarray, m2: np.ndarray, frac: float = 0.20) -> float:
    """IoU of the top-`frac` patch sets of two maps."""
    flat1 = m1.flatten()
    flat2 = m2.flatten()
    n = flat1.size
    k = max(1, int(round(n * frac)))
    # Argpartition for top-k indices (no need to sort)
    idx1 = set(np.argpartition(-flat1, k - 1)[:k].tolist())
    idx2 = set(np.argpartition(-flat2, k - 1)[:k].tolist())
    inter = idx1 & idx2
    union = idx1 | idx2
    if not union:
        return 0.0
    return len(inter) / len(union)


def _js_div(m1: np.ndarray, m2: np.ndarray) -> float:
    """Jensen-Shannon divergence in patch-probability space (nats)."""
    p = _normalize_map_to_prob(m1)
    q = _normalize_map_to_prob(m2)
    avg = 0.5 * (p + q)

    def _kl(a, b):
        a = np.clip(a, 1e-12, 1.0)
        b = np.clip(b, 1e-12, 1.0)
        return float((a * (np.log(a) - np.log(b))).sum())

    return 0.5 * _kl(p, avg) + 0.5 * _kl(q, avg)


def _cosine(m1: np.ndarray, m2: np.ndarray) -> float:
    a = m1.flatten().astype(np.float64)
    b = m2.flatten().astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compute_alignment_per_token(
    maps_T: list[np.ndarray],
    maps_X: list[np.ndarray],
    valid_T: list[bool],
    valid_X: list[bool],
) -> dict:
    """Return per-token IoU/JS/Cos lists comparing teacher vs student X.

    Tokens with either map invalid get None at that position so the
    arrays stay token-aligned.
    """
    R = len(maps_T)
    iou: list = [None] * R
    js:  list = [None] * R
    cos: list = [None] * R
    for t in range(R):
        if not (valid_T[t] and valid_X[t]):
            continue
        iou[t] = _iou_topk(maps_T[t], maps_X[t], frac=0.20)
        js[t]  = _js_div(maps_T[t], maps_X[t])
        cos[t] = _cosine(maps_T[t], maps_X[t])
    return {"iou_top20": iou, "js": js, "cos": cos}


# ============================================================================
# Orchestration
# ============================================================================
def _load_subset(path: Path, limit: int,
                 shard_id: int, num_shards: int) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if limit > 0:
        rows = rows[:limit]
    if num_shards > 1:
        rows = rows[shard_id::num_shards]
    return rows


def _load_jsonl(path: Path) -> dict:
    """Load a JSONL of {id: <rec>} for resume support."""
    if not path.exists():
        return {}
    out: dict = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def run_pass1(args, subset: list[dict], rollout_path: Path) -> dict:
    """Pass 1: S1 greedy rollout for every sample. Resumable."""
    import torch

    done = _load_jsonl(rollout_path)
    if done:
        print(f">>> Pass 1: resume — {len(done)} rollouts cached in "
              f"{rollout_path}", file=sys.stderr)
    todo = [r for r in subset if r["id"] not in done]
    if not todo:
        print(f">>> Pass 1: all {len(subset)} rollouts cached; skipping",
              file=sys.stderr)
        return done

    print(f">>> Pass 1: loading S1 ← {args.s1}", file=sys.stderr)
    s1_proc, s1_model = _build_model(args.s1)
    image_root = Path(args.image_root)

    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    open_mode = "a" if done else "w"
    n = len(todo)
    with rollout_path.open(open_mode) as fout:
        for k, rec in enumerate(todo):
            try:
                image, img_path = _load_image_for_rec(rec, image_root)
                r = s1_rollout(s1_proc, s1_model, rec, image, args)
                row = {
                    "id":             rec["id"],
                    "image_path":     str(img_path),
                    "image_sha256":   _image_sha256(img_path),
                    "response_ids":   r["response_ids"],
                    "response_length": r["response_length"],
                    "response_text":   r["response_text"],
                    "response_hash":   r["response_hash"],
                    "prompt_len":      r["prompt_len"],
                    "rollout_gen_s":   r["rollout_gen_s"],
                }
                done[rec["id"]] = row
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                rid_short = str(rec["id"])[:36]
                print(f"  rollout [{k+1:4d}/{n}] {r['rollout_gen_s']:5.1f}s  "
                      f"R={r['response_length']:4d}  {rid_short}",
                      file=sys.stderr, flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"!! rollout failed on {rec['id']}: {e!r}",
                      file=sys.stderr, flush=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    # Free S1 — Pass 2 reloads in canonical order
    try:
        del s1_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return done


def run_pass2_one_model(args, subset: list[dict], rollout_cache: dict,
                        model_name: str, model_path: str,
                        out_path: Path) -> dict:
    """Pass 2 for ONE model (T / S0 / S1). Loads the model once, iterates
    samples, writes per-model JSONL. Resumable."""
    import torch

    done = _load_jsonl(out_path)
    if done:
        print(f">>> Pass 2 [{model_name}]: resume — {len(done)} samples "
              f"cached in {out_path.name}", file=sys.stderr)
    todo = [r for r in subset
            if r["id"] in rollout_cache and r["id"] not in done]
    if not todo:
        print(f">>> Pass 2 [{model_name}]: all done; skipping",
              file=sys.stderr)
        return done

    print(f">>> Pass 2 [{model_name}]: loading ← {model_path}",
          file=sys.stderr)
    proc, model = _build_model(model_path)
    image_root = Path(args.image_root)

    open_mode = "a" if done else "w"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = len(todo)
    with out_path.open(open_mode) as fout:
        for k, rec in enumerate(todo):
            rid = rec["id"]
            rollout = rollout_cache[rid]
            t0 = time.time()
            try:
                image, _ = _load_image_for_rec(rec, image_root)
                result = model_pass_with_tam(
                    proc, model, rec, image,
                    rollout["response_ids"], args,
                )
                # Drop in-memory arrays (we only serialize b64 + scalars)
                _arrays = result.pop("_response_maps_arrays")
                # But keep them for in-process Pass 3 if same process
                row = {
                    "id":           rid,
                    "model_name":   model_name,
                    "model_path":   model_path,
                    "response_hash": rollout["response_hash"],
                    **result,
                }
                done[rid] = {**row, "_response_maps_arrays": _arrays}
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                dt = time.time() - t0
                R = rollout["response_length"]
                tams = result["_timings"]
                rid_short = str(rid)[:36]
                tam_valid = "ok" if result["tam_valid"] else "INVALID"
                print(f"  {model_name} [{k+1:4d}/{n}] {dt:5.1f}s  "
                      f"(fwd={tams['fwd_s']:.1f}s tam={tams['tam_s']:.1f}s) "
                      f"R={R:4d}  {tam_valid}  {rid_short}",
                      file=sys.stderr, flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"!! pass2 [{model_name}] failed on {rid}: {e!r}",
                      file=sys.stderr, flush=True)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

    try:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return done


def _decode_b64_map(b64_str: str, h: int, w: int) -> np.ndarray:
    """Inverse of `_b64_uint8_map` — uint8 → [0,1] float32 H×W."""
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
    return arr.astype(np.float32) / 255.0


def run_pass3(args, subset: list[dict], rollout_cache: dict,
              per_model: dict[str, dict], out_path: Path,
              processor=None) -> None:
    """Pass 3: merge per-model outputs + compute alignment metrics.

    Uses in-memory map arrays if available (same-process run); otherwise
    decodes from inline b64. Single-process; fast.
    """
    # We need a tokenizer for token_category classification. Use the S1
    # processor (any tokenizer in Qwen2.5-VL family is equivalent at
    # vocab level; v0.1.3 classifier is regex+spaCy on tokens).
    if processor is None:
        # Last-resort load (cheap, processor only)
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.s1, trust_remote_code=True)
    tokenizer = processor.tokenizer

    image_root = Path(args.image_root)

    n_written = 0
    n_skipped_missing = 0
    n_skipped_tam_invalid = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fout:
        for rec in subset:
            rid = rec["id"]
            if rid not in rollout_cache:
                n_skipped_missing += 1
                continue
            if not all(rid in per_model[m] for m in ("T", "S0", "S1")):
                n_skipped_missing += 1
                continue
            rollout = rollout_cache[rid]
            blocks = {m: per_model[m][rid] for m in ("T", "S0", "S1")}

            # ---- Sample-level drop: any model with tam_valid=False kills
            # the sample (per design §11: token-level partial failure padded
            # to length, sample-level dropped — never half-valid rows).
            invalid_models = [m for m in ("T", "S0", "S1")
                              if not blocks[m].get("tam_valid", False)]
            if invalid_models:
                n_skipped_tam_invalid += 1
                reasons = ",".join(
                    f"{m}={blocks[m].get('tam_failure_reason', 'unknown')}"
                    for m in invalid_models
                )
                print(f"  skip {rid}: invalid TAM on {invalid_models}; {reasons}",
                      file=sys.stderr)
                continue

            response_ids = rollout["response_ids"]
            R = rollout["response_length"]
            response_tokens = tokenizer.convert_ids_to_tokens(response_ids)

            # Resolve token-aligned maps for the three models. After the
            # tam_valid drop above, every model's arrays have length R
            # (model_pass_with_tam enforces the invariant).
            maps_dict: dict[str, list[np.ndarray]] = {}
            valid_dict: dict[str, list[bool]] = {}
            length_ok = True
            for m_name in ("T", "S0", "S1"):
                b = blocks[m_name]
                if "_response_maps_arrays" in b:
                    arrs = b["_response_maps_arrays"]
                else:
                    h, w = b["vision_shape"]
                    arrs = [_decode_b64_map(s, h, w)
                            for s in b["_response_maps_b64"]]
                if len(arrs) != R:
                    print(f"!! length-invariant violation on {rid}/{m_name}: "
                          f"maps={len(arrs)} R={R} — dropping sample",
                          file=sys.stderr)
                    length_ok = False
                    break
                pv = b.get("_per_token_valid")
                if pv is None or len(pv) != R:
                    pv = [bool(np.asarray(a).sum() > 0) for a in arrs]
                maps_dict[m_name] = arrs
                valid_dict[m_name] = pv
            if not length_ok:
                n_skipped_tam_invalid += 1
                continue

            # Per-token alignment
            align_S0_T = compute_alignment_per_token(
                maps_dict["T"], maps_dict["S0"],
                valid_dict["T"], valid_dict["S0"],
            )
            align_S1_T = compute_alignment_per_token(
                maps_dict["T"], maps_dict["S1"],
                valid_dict["T"], valid_dict["S1"],
            )

            # Token classification (v0.1.3 logic)
            labels = _label_response_tokens(response_ids, tokenizer)
            classification = _classify_tokens_v012(
                response_ids, tokenizer, labels["is_answer_token"],
            )
            token_category = classification.get("token_category", ["other"] * R)

            row = {
                "id":          rid,
                "benchmark":   rec.get("benchmark"),
                "bucket":      rec.get("bucket"),
                "image_path":  rollout["image_path"],
                "image_sha256": rollout["image_sha256"],
                "question":    rec.get("question"),
                "answer":      rec.get("answer"),

                # cached predictions / correctness from selector
                "s0_response_text": rec.get("s0_response_text"),
                "s1_response_text": rec.get("s1_response_text"),
                "s0_correct":       rec.get("s0_correct"),
                "s1_correct":       rec.get("s1_correct"),

                # rollout (the shared sequence)
                "rollout_source":   "S1_greedy",
                "rollout_model":    args.s1,
                "response_ids":     response_ids,
                "response_text":    rollout["response_text"],
                "response_length":  R,
                "response_hash":    rollout["response_hash"],
                "tokens":           response_tokens,

                # token category from v0.1.3 classifier
                "token_category":   token_category,
                "is_answer_token":  labels["is_answer_token"],
                "is_blankness_token": labels.get("is_blankness_token", [False] * R),

                # per-model scalars
                "T": {
                    "tam_mass_top10":   blocks["T"]["tam_mass_top10"],
                    "tam_mass_top20":   blocks["T"]["tam_mass_top20"],
                    "tam_mass_top40":   blocks["T"]["tam_mass_top40"],
                    "tam_entropy_norm": blocks["T"]["tam_entropy_norm"],
                    "tam_effective_patch_frac":
                        blocks["T"]["tam_effective_patch_frac"],
                    "lp":               blocks["T"]["lp"],
                    "tam_valid":        blocks["T"]["tam_valid"],
                },
                "S0": {
                    "tam_mass_top10":   blocks["S0"]["tam_mass_top10"],
                    "tam_mass_top20":   blocks["S0"]["tam_mass_top20"],
                    "tam_mass_top40":   blocks["S0"]["tam_mass_top40"],
                    "tam_entropy_norm": blocks["S0"]["tam_entropy_norm"],
                    "tam_effective_patch_frac":
                        blocks["S0"]["tam_effective_patch_frac"],
                    "lp":               blocks["S0"]["lp"],
                    "tam_valid":        blocks["S0"]["tam_valid"],
                },
                "S1": {
                    "tam_mass_top10":   blocks["S1"]["tam_mass_top10"],
                    "tam_mass_top20":   blocks["S1"]["tam_mass_top20"],
                    "tam_mass_top40":   blocks["S1"]["tam_mass_top40"],
                    "tam_entropy_norm": blocks["S1"]["tam_entropy_norm"],
                    "tam_effective_patch_frac":
                        blocks["S1"]["tam_effective_patch_frac"],
                    "lp":               blocks["S1"]["lp"],
                    "tam_valid":        blocks["S1"]["tam_valid"],
                },

                # per-token alignment (the headline)
                "align": {"S0_T": align_S0_T, "S1_T": align_S1_T},

                # inline maps (kept for renderer; analyzer can ignore)
                "maps_b64": {
                    "T":  blocks["T"]["_response_maps_b64"],
                    "S0": blocks["S0"]["_response_maps_b64"],
                    "S1": blocks["S1"]["_response_maps_b64"],
                },

                # QC
                "tam_valid_T":  valid_dict["T"],
                "tam_valid_S0": valid_dict["S0"],
                "tam_valid_S1": valid_dict["S1"],

                # geometry
                "vision_shape": blocks["T"]["vision_shape"],
                "n_patches":    blocks["T"]["n_patches"],
                "map_h":        int(blocks["T"]["vision_shape"][0]),
                "map_w":        int(blocks["T"]["vision_shape"][1]),
                "patch_index_order": "row_major_top_left",

                # version + commit
                "tam_preproc_version":    "v0.1.3",
                "step5_schema_version":   "v0.1",
                "code_commit_run":        os.environ.get("MLLMOPD_CODE_COMMIT", "unknown"),
                "tokenizer_vocab_hash":   _tokenizer_vocab_hash(processor),
                "pos_tagger_version":     classification.get("_pos_tagger_version"),
                "token_category_source":  "regex+spacy_align:v0.1.3",
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    print(f">>> Pass 3: wrote {n_written} alignment rows to {out_path}; "
          f"skipped {n_skipped_missing} (missing per-model output) + "
          f"{n_skipped_tam_invalid} (TAM invalid)",
          file=sys.stderr)
    # Write a sidecar drop log for the launcher's row-count check
    drop_log = out_path.with_suffix(".drops.txt")
    with drop_log.open("w") as f:
        f.write(f"n_written={n_written}\n")
        f.write(f"n_skipped_missing={n_skipped_missing}\n")
        f.write(f"n_skipped_tam_invalid={n_skipped_tam_invalid}\n")
        f.write(f"n_expected={len(subset)}\n")


# ============================================================================
# Main
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=Path, required=True,
                    help="Stratified samples JSONL (from tam_step5_sample_selector)")
    ap.add_argument("--teacher", required=True, help="T = teacher ckpt path")
    ap.add_argument("--s0",      required=True, help="S0 = base student ckpt path")
    ap.add_argument("--s1",      required=True, help="S1 = OPD student ckpt path")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--system-prompt", default=MMR1_SYSTEM_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--pass", dest="which_pass",
                    choices=["1", "2T", "2S0", "2S1", "3", "all"],
                    default="all",
                    help="Restrict to one pass — useful for multi-box "
                         "orchestration. Default: all four")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    subset = _load_subset(args.samples, args.limit,
                          args.shard_id, args.num_shards)
    print(f">>> subset: {len(subset)} samples "
          f"(shard {args.shard_id}/{args.num_shards})", file=sys.stderr)

    rollout_path = args.out_dir / "rollout_cache.jsonl"
    per_model_paths = {
        "T":  args.out_dir / "tam_per_model_T.jsonl",
        "S0": args.out_dir / "tam_per_model_S0.jsonl",
        "S1": args.out_dir / "tam_per_model_S1.jsonl",
    }
    alignment_path = args.out_dir / "alignment.jsonl"

    # ----- Pass 1 -----
    rollout_cache: dict = {}
    if args.which_pass in ("1", "all"):
        rollout_cache = run_pass1(args, subset, rollout_path)
    else:
        rollout_cache = _load_jsonl(rollout_path)

    if args.which_pass == "1":
        return 0

    # ----- Pass 2 (T / S0 / S1) -----
    per_model: dict[str, dict] = {"T": {}, "S0": {}, "S1": {}}
    if args.which_pass in ("2T", "all"):
        per_model["T"] = run_pass2_one_model(
            args, subset, rollout_cache, "T", args.teacher, per_model_paths["T"]
        )
    else:
        per_model["T"] = _load_jsonl(per_model_paths["T"])

    if args.which_pass in ("2S0", "all"):
        per_model["S0"] = run_pass2_one_model(
            args, subset, rollout_cache, "S0", args.s0, per_model_paths["S0"]
        )
    else:
        per_model["S0"] = _load_jsonl(per_model_paths["S0"])

    if args.which_pass in ("2S1", "all"):
        per_model["S1"] = run_pass2_one_model(
            args, subset, rollout_cache, "S1", args.s1, per_model_paths["S1"]
        )
    else:
        per_model["S1"] = _load_jsonl(per_model_paths["S1"])

    if args.which_pass.startswith("2"):
        return 0

    # ----- Pass 3 -----
    if args.which_pass in ("3", "all"):
        run_pass3(args, subset, rollout_cache, per_model, alignment_path)

    # Brief summary
    summary_path = args.out_dir / "summary.txt"
    with summary_path.open("w") as fs:
        fs.write(f"# Step 5 evidence alignment  "
                 f"(commit={os.environ.get('MLLMOPD_CODE_COMMIT', 'unknown')})\n")
        fs.write(f"teacher = {args.teacher}\n")
        fs.write(f"s0      = {args.s0}\n")
        fs.write(f"s1      = {args.s1}\n")
        fs.write(f"samples = {args.samples}\n")
        fs.write(f"n_samples (shard) = {len(subset)}\n")
        fs.write(f"rollout cache     = {rollout_path}\n")
        for m, p in per_model_paths.items():
            n = sum(1 for _ in p.open()) if p.exists() else 0
            fs.write(f"per_model[{m}]    = {p}  (n={n})\n")
        fs.write(f"alignment         = {alignment_path}\n")
    print(f">>> summary: {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
