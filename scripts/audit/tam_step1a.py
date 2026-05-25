"""Step 1a TAM-VD calibration runner — teacher_greedy mode.

For each sample in the calibration subset, produces one row per student
ckpt with the v0.1.2 schema. The teacher work (full + blank forward,
TAM, attention baseline, lp_full, lp_blank) is checkpoint-invariant and
cached into a teacher sidecar JSONL so that adding a new student ckpt
later doesn't require re-running the teacher.

Pipeline per sample:
  1. Teacher full forward (greedy generation w/ output_hidden_states +
     output_attentions + output_scores)
     → response_text/ids, lp_full, teacher_entropy_full,
       teacher_top1_margin_full, TAM scalars + maps subset,
       attention baseline + maps subset, peak meta, prompt-token TAM
  2. Teacher blank-image scoring on the SAME response_ids (teacher-
     forced forward, no generation)
     → lp_blank → vd = lp_full - lp_blank
  3. For each student_ckpt c (Pass 2, after teacher cache is written):
     - student teacher-forced forward on (prompt + response_ids)
     - → student_lp, student_entropy
     - adv = lp_full - student_lp; quad from sign(vd, adv)
     - Emit one row with `student_ckpt=c, response_source=teacher_greedy`

Heavy reuse of tam_sanity.py helpers (POS classification, attention
extraction, peak meta, scalar computation, span labels).

Usage::

    python -m scripts.audit.tam_step1a \\
        --subset data/audit/tam_calibration_subset_v0.jsonl \\
        --teacher MMR1/MMR1-7B-RL \\
        --students T1_0=MMR1/MMR1-3B-SFT \\
        --students T1_2=<HF path or run dir> \\
        --students T1_3=<HF path or run dir> \\
        --out-dir runs/audit/tam_step1a_<TS> \\
        [--teacher-cache <path>]  # reuse if exists; else write here
        [--limit N]

Output:
  <out-dir>/teacher_cache.jsonl   (one row per sample; teacher signals)
  <out-dir>/tam_step1a.jsonl      (one row per sample × student_ckpt)
  <out-dir>/summary.txt
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Reuse heavy lifting from tam_sanity.py — keeps both files in lockstep.
from tam_sanity import (  # noqa: E402
    _build_model, _build_messages,
    _classify_tokens_v012, _tam_peak_meta, _attention_baseline,
    _per_token_teacher_stats,
    _label_response_tokens, _compute_prompt_segments,
    _select_tokens_for_overlays,
    _b64_uint8_map, _image_sha256, _tokenizer_vocab_hash,
    _sha256_hex,
    MMR1_SYSTEM_PROMPT, QWEN_VL_SPECIAL_IDS,
    _SPACY_AVAILABLE, _SPACY_VERSION, _SPACY_LOAD_ERROR,
)
from _tam_core import TAM, tam_scalars  # noqa: E402


# ============================================================================
# Image-corruption helpers (for negative controls)
# ============================================================================
def _load_image_for_sample(rec: dict, image_root: Path, swap_lookup: dict):
    """Returns (PIL.Image, source_tag). Honors rec["image_corruption"]:
      - "blank_image": replace with uniform gray PIL of same size
      - "swap_image": load swap_with_image path
      - None / "full_image": load rec["image"] as-is"""
    from PIL import Image
    corruption = rec.get("image_corruption")
    image_path = rec["image"]
    if corruption == "swap_image" and rec.get("swap_with_image"):
        image_path = rec["swap_with_image"]
    p = Path(image_path)
    if not p.is_absolute():
        p = (image_root / p).resolve()
    # Foreign-path fallback (same trick as the renderer)
    if not p.exists():
        s = str(p).replace("\\", "/")
        if "data/audit/images/" in s:
            tail = s.rsplit("data/audit/images/", 1)[-1]
            cand = image_root / "data" / "audit" / "images" / tail
            if cand.exists():
                p = cand
    if not p.exists():
        raise FileNotFoundError(f"image not found: {p}")
    img = Image.open(p).convert("RGB")
    if corruption == "blank_image":
        # Uniform gray (mid-128) — keeps the vision encoder happy while
        # carrying no useful signal.
        img = Image.new("RGB", img.size, (128, 128, 128))
        return img, "blank_image"
    if corruption == "swap_image":
        return img, f"swap_image:{rec.get('swap_with_id')}"
    return img, "full_image"


# ============================================================================
# Teacher pass — one sample
# ============================================================================
def teacher_pass(processor, model, rec: dict, image, args) -> dict:
    """Returns the teacher-side block to be merged into the final row(s)."""
    import torch

    messages = _build_messages(rec["question"], image, args.system_prompt)
    chat = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(
        text=[chat], images=[image], return_tensors="pt", padding=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    input_ids_full = inputs["input_ids"][0].tolist()

    # --- Full-image generation ---
    t0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            output_hidden_states=True,
            output_scores=True,
            output_attentions=True,
            return_dict_in_generate=True,
        )
    dt_gen_full = time.time() - t0

    sequences = outputs.sequences[0]
    response_ids = sequences[input_len:].cpu().tolist()
    eos_id = processor.tokenizer.eos_token_id
    response_length = len(response_ids)
    if eos_id is not None and eos_id in response_ids:
        response_length = response_ids.index(eos_id) + 1
    response_ids = response_ids[:response_length]

    scores_trimmed = tuple(outputs.scores[:response_length])
    lp_full, teacher_entropy_full, teacher_top1_margin_full = _per_token_teacher_stats(
        scores_trimmed, sequences, input_len,
    )

    # logit_list for TAM
    t1 = time.time()
    logit_list = []
    for r, feats in enumerate(outputs.hidden_states[:response_length]):
        last_h = feats[-1]
        with torch.inference_mode():
            logit_list.append(model.lm_head(last_h))
    dt_logit = time.time() - t1

    grid_thw = inputs["image_grid_thw"][0].tolist()
    vision_shape = (grid_thw[1] // 2, grid_thw[2] // 2)
    n_patches = int(vision_shape[0] * vision_shape[1])
    tokens_full_list = sequences[: input_len + response_length].cpu().tolist()

    # TAM per response token + prompt token capture
    t2 = time.time()
    img_scores_list: list = []
    prompt_maps: list = []
    response_maps: list = []
    tam_failure = None
    try:
        for i in range(response_length):
            m = TAM(
                tokens=tokens_full_list, vision_shape=vision_shape,
                logit_list=logit_list, special_ids=QWEN_VL_SPECIAL_IDS,
                processor=processor, target_token=i,
                img_scores_list=img_scores_list,
                out_prompt_maps=prompt_maps if i == 0 else None,
            )
            response_maps.append(m)
    except Exception as e:  # noqa: BLE001
        tam_failure = f"runtime:{type(e).__name__}:{e!s:.120}"
    dt_tam = time.time() - t2
    tam_valid = (tam_failure is None and len(response_maps) == response_length)

    response_scalars = [tam_scalars(m) for m in response_maps]
    prompt_scalars   = [tam_scalars(m) for m in prompt_maps]
    response_peak    = [_tam_peak_meta(m) for m in response_maps]

    # Attention baseline
    img_start_idx_full = input_ids_full.index(QWEN_VL_SPECIAL_IDS["img_id"][0]) if QWEN_VL_SPECIAL_IDS["img_id"][0] in input_ids_full else -1
    img_end_idx_full = input_ids_full.index(QWEN_VL_SPECIAL_IDS["img_id"][1]) if QWEN_VL_SPECIAL_IDS["img_id"][1] in input_ids_full else -1
    t3 = time.time()
    if img_start_idx_full >= 0 and img_end_idx_full >= 0 and outputs.attentions is not None:
        attn_maps, attn_scalars, attn_peak, attn_failure = _attention_baseline(
            outputs.attentions[:response_length], response_length, input_len,
            img_start_idx_full, img_end_idx_full, vision_shape,
        )
    else:
        attn_maps = [np.zeros(vision_shape, dtype=np.float32)] * response_length
        attn_scalars = [tam_scalars(m) for m in attn_maps]
        attn_peak = [_tam_peak_meta(m) for m in attn_maps]
        attn_failure = "vision_span_or_attentions_missing"
    dt_attn = time.time() - t3
    attn_baseline_valid = (attn_failure is None)

    # --- Blank-image scoring on the SAME response_ids ---
    # Build a blank-image chat with same prompt; teacher-forced forward.
    from PIL import Image
    blank_pil = Image.new("RGB", image.size, (128, 128, 128))
    blank_chat = chat  # same prompt text (image-token positions are processor-driven)
    blank_inputs = processor(
        text=[blank_chat], images=[blank_pil], return_tensors="pt", padding=True,
    ).to(model.device)
    # Append response_ids to the prompt for teacher-forced scoring.
    import torch
    response_tensor = torch.tensor(response_ids, device=model.device).unsqueeze(0)
    blank_input_ids = torch.cat([blank_inputs["input_ids"], response_tensor], dim=1)
    blank_attn_mask = torch.cat([
        blank_inputs["attention_mask"],
        torch.ones_like(response_tensor),
    ], dim=1)
    blank_kwargs = dict(blank_inputs)
    blank_kwargs["input_ids"] = blank_input_ids
    blank_kwargs["attention_mask"] = blank_attn_mask

    t4 = time.time()
    lp_blank: list[float] = []
    try:
        with torch.inference_mode():
            blank_outputs = model(**blank_kwargs)
        blank_input_len = blank_inputs["input_ids"].shape[1]
        # logits[i] predicts position i+1; so predictor for response[t]
        # (which is at full position blank_input_len + t) is at index
        # (blank_input_len + t - 1).
        logits_blank = blank_outputs.logits[0]   # (T, V)
        for t in range(response_length):
            pred_idx = blank_input_len + t - 1
            log_probs = torch.log_softmax(logits_blank[pred_idx], dim=-1)
            lp_blank.append(float(log_probs[response_ids[t]]))
    except Exception as e:  # noqa: BLE001
        lp_blank = [None] * response_length
        print(f"!! blank-image scoring failed: {type(e).__name__}: {e!s:.120}",
              file=sys.stderr)
    dt_blank = time.time() - t4

    vd = [(lf - lb) if (lf is not None and lb is not None) else None
          for lf, lb in zip(lp_full, lp_blank)]

    # Token labels + classification
    tokenizer = processor.tokenizer
    response_tokens = tokenizer.convert_ids_to_tokens(response_ids)
    response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
    labels = _label_response_tokens(response_ids, tokenizer)
    classification = _classify_tokens_v012(
        response_ids, tokenizer, labels["is_answer_token"],
    )

    # Prompt-token TAM (question span only)
    prompt_segments = _compute_prompt_segments(input_ids_full, QWEN_VL_SPECIAL_IDS)
    question_span = prompt_segments.get("question")
    if question_span:
        tokens_prompt_ids = input_ids_full[question_span[0]: question_span[1]]
        tokens_prompt = tokenizer.convert_ids_to_tokens(tokens_prompt_ids)
    else:
        tokens_prompt_ids = []
        tokens_prompt = []
    if len(prompt_maps) > len(tokens_prompt):
        prompt_maps = prompt_maps[: len(tokens_prompt)]
        prompt_scalars = prompt_scalars[: len(tokens_prompt)]

    response_hash = _sha256_hex(json.dumps(response_ids).encode())

    return {
        "response_ids": response_ids,
        "response_length": response_length,
        "response_text": response_text,
        "response_hash": response_hash,
        "tokens": response_tokens,
        "lp_full": lp_full,
        "lp_blank": lp_blank,
        "vd": vd,
        "tam_mass_top10": [s["tam_mass_top10"] for s in response_scalars],
        "tam_mass_top20": [s["tam_mass_top20"] for s in response_scalars],
        "tam_mass_top40": [s["tam_mass_top40"] for s in response_scalars],
        "tam_entropy":              [s["tam_entropy"] for s in response_scalars],
        "tam_entropy_norm":         [s["tam_entropy_norm"] for s in response_scalars],
        "tam_effective_patch_frac": [s["tam_effective_patch_frac"] for s in response_scalars],
        "tam_peak_patch_idx":     [p["tam_peak_patch_idx"] for p in response_peak],
        "tam_peak_xy":            [p["tam_peak_xy"] for p in response_peak],
        "tam_center_of_mass_xy":  [p["tam_center_of_mass_xy"] for p in response_peak],
        "attention_baseline_mass_top10":   [s["tam_mass_top10"] for s in attn_scalars],
        "attention_baseline_mass_top20":   [s["tam_mass_top20"] for s in attn_scalars],
        "attention_baseline_mass_top40":   [s["tam_mass_top40"] for s in attn_scalars],
        "attention_baseline_entropy_norm": [s["tam_entropy_norm"] for s in attn_scalars],
        "teacher_entropy_full":     teacher_entropy_full,
        "teacher_top1_margin_full": teacher_top1_margin_full,
        "prompt_tam_scope":       "question_only",
        "prompt_length":          len(tokens_prompt),
        "tokens_prompt":          tokens_prompt,
        "tam_mass_top20_prompt":  [s["tam_mass_top20"] for s in prompt_scalars],
        "tam_entropy_norm_prompt": [s["tam_entropy_norm"] for s in prompt_scalars],
        "prompt_segments":        prompt_segments,
        "prompt_full_length":     len(input_ids_full),
        "image_grid_thw":         grid_thw,
        "vision_shape":           list(vision_shape),
        "n_patches":              n_patches,
        "map_h":                  int(vision_shape[0]),
        "map_w":                  int(vision_shape[1]),
        "patch_index_order":      "row_major_top_left",
        "tam_valid":              bool(tam_valid),
        "tam_failure_reason":     tam_failure,
        "attn_baseline_valid":          bool(attn_baseline_valid),
        "attn_baseline_failure_reason": attn_failure,
        "labels":   labels,             # nested; merged into row later
        "classification": classification,
        # Subset selection — same strata as Step 0 but we'll do it after VD is in
        "_response_maps_b64": [_b64_uint8_map(m) for m in response_maps],
        "_attn_maps_b64":     [_b64_uint8_map(m) for m in attn_maps],
        "_timings": {
            "gen_full_s":   dt_gen_full,
            "logit_s":      dt_logit,
            "tam_s":        dt_tam,
            "attn_s":       dt_attn,
            "blank_s":      dt_blank,
        },
    }


# ============================================================================
# Student pass — teacher-forced scoring of teacher's response under the student
# ============================================================================
def student_pass(processor, model, rec, image, response_ids: list[int], args) -> dict:
    """Returns student_lp + student_entropy aligned to response_ids."""
    import torch
    messages = _build_messages(rec["question"], image, args.system_prompt)
    chat = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(
        text=[chat], images=[image], return_tensors="pt", padding=True,
    ).to(model.device)
    response_tensor = torch.tensor(response_ids, device=model.device).unsqueeze(0)
    full_ids = torch.cat([inputs["input_ids"], response_tensor], dim=1)
    full_mask = torch.cat([
        inputs["attention_mask"],
        torch.ones_like(response_tensor),
    ], dim=1)
    kw = dict(inputs)
    kw["input_ids"] = full_ids
    kw["attention_mask"] = full_mask

    input_len = inputs["input_ids"].shape[1]
    R = len(response_ids)
    student_lp: list[float] = [None] * R   # type: ignore[list-item]
    student_entropy: list[float] = [None] * R   # type: ignore[list-item]
    try:
        with torch.inference_mode():
            out = model(**kw)
        logits = out.logits[0]
        for t in range(R):
            pred_idx = input_len + t - 1
            log_probs = torch.log_softmax(logits[pred_idx], dim=-1)
            probs = torch.softmax(logits[pred_idx].float(), dim=-1)
            student_lp[t] = float(log_probs[response_ids[t]])
            student_entropy[t] = float(-(probs * torch.log(probs + 1e-12)).sum())
    except Exception as e:  # noqa: BLE001
        print(f"!! student scoring failed: {type(e).__name__}: {e!s:.120}",
              file=sys.stderr)
    return {"student_lp": student_lp, "student_entropy": student_entropy}


# ============================================================================
# Row builder
# ============================================================================
def _quad_int(vd: float | None, adv: float | None) -> int | None:
    if vd is None or adv is None:
        return None
    if vd >= 0.0:
        return 0 if adv >= 0.0 else 1
    return 2 if adv >= 0.0 else 3


def _build_row(rec, image, image_path, tcache, student_ckpt, student_block,
               args, processor) -> dict:
    """Assemble a v0.1.2 row from teacher cache + student forward results."""
    R = tcache["response_length"]
    student_lp = student_block["student_lp"]
    student_entropy = student_block["student_entropy"]
    lp_full = tcache["lp_full"]
    vd = tcache["vd"]
    adv = [(lf - sl) if (lf is not None and sl is not None) else None
           for lf, sl in zip(lp_full, student_lp)]
    quad = [_quad_int(v, a) for v, a in zip(vd, adv)]

    # Subset selection (Step 1 strata: top-|vd| + top-|adv| + blankness + answer)
    is_blank = tcache["labels"]["is_blankness_token"]
    is_ans   = tcache["labels"]["is_answer_token"]

    abs_vd  = [abs(v) if v is not None else -1 for v in vd]
    abs_adv = [abs(a) if a is not None else -1 for a in adv]
    top_vd_idx  = sorted(range(R), key=lambda i: -abs_vd[i])[:5]
    top_adv_idx = sorted(range(R), key=lambda i: -abs_adv[i])[:5]
    blank_idxs  = [i for i in range(R) if is_blank[i]][:5]
    answer_idxs = [i for i in range(R) if is_ans[i]][:5]

    seen: dict[int, list] = {}
    for stratum, idxs, score_arr in [
        ("top_abs_vd",        top_vd_idx,  abs_vd),
        ("top_abs_adv",       top_adv_idx, abs_adv),
        ("blankness",         blank_idxs,  [1.0] * R),
        ("answer_critical",   answer_idxs, [1.0] * R),
    ]:
        for rank, idx in enumerate(idxs):
            if idx in seen:
                seen[idx][3].append(stratum)
                continue
            seen[idx] = [stratum, rank, float(score_arr[idx]), []]
    selected = list(seen.items())[:20]
    sub_idx       = [int(i) for i, _ in selected]
    sub_strata    = [v[0] for _, v in selected]
    sub_rank      = [int(v[1]) for _, v in selected]
    sub_score     = [float(v[2]) for _, v in selected]
    sub_dedup     = [list(v[3]) for _, v in selected]
    sub_tam_maps  = [tcache["_response_maps_b64"][i] for i in sub_idx]
    sub_attn_maps = [tcache["_attn_maps_b64"][i] for i in sub_idx]

    response_hash = tcache["response_hash"]
    token_uid = [
        f"{rec['id']}:{student_ckpt}:teacher_greedy:{response_hash}:{t}"
        for t in range(R)
    ]

    row = {
        # identity
        "id":           rec["id"],
        "benchmark":    rec.get("benchmark"),
        "split_tag":    rec.get("split_tag", "step1a"),
        "image_path":   str(image_path),
        "image_sha256": _image_sha256(image_path),
        "question":     rec.get("question"),
        "answer":       rec.get("answer"),
        "image_corruption": rec.get("image_corruption"),

        # mode / checkpoints
        "response_source": "teacher_greedy",
        "teacher_ckpt":    args.teacher,
        "student_ckpt":    student_ckpt,

        # run metadata (v0.1.2)
        "tokenizer_name_or_path":  args.teacher,
        "tokenizer_vocab_hash":    _tokenizer_vocab_hash(processor),
        "processor_name_or_path":  args.teacher,
        "tam_preproc_version":     "v0.1.2",
        "code_commit_run":         os.environ.get("MLLMOPD_CODE_COMMIT", "unknown"),
        "code_commit_analyzed":    None,
        "pos_tagger":              "spacy/en_core_web_sm" if _SPACY_AVAILABLE else "none",
        "pos_tagger_version":      _SPACY_VERSION,
        "pos_tagger_load_error":   _SPACY_LOAD_ERROR,
        "token_category_source":   "regex+spacy_align:v0.1.2",
        "attention_baseline_method": "last_layer_avg_heads:v0.1.2",
        "attention_baseline_layers": [-1],
        "attention_baseline_heads":  "all",

        # response (teacher_greedy → same across student_ckpt rows of same sample)
        "response_text":   tcache["response_text"],
        "response_ids":    tcache["response_ids"],
        "response_length": R,
        "response_hash":   response_hash,
        "tokens":          tcache["tokens"],
        "token_idx":       list(range(R)),
        "token_uid":       token_uid,

        # teacher signals
        "lp_full":   lp_full,
        "lp_blank":  tcache["lp_blank"],
        "vd":        vd,
        "tam_mass_top10": tcache["tam_mass_top10"],
        "tam_mass_top20": tcache["tam_mass_top20"],
        "tam_mass_top40": tcache["tam_mass_top40"],
        "tam_entropy":              tcache["tam_entropy"],
        "tam_entropy_norm":         tcache["tam_entropy_norm"],
        "tam_effective_patch_frac": tcache["tam_effective_patch_frac"],
        "tam_peak_patch_idx":     tcache["tam_peak_patch_idx"],
        "tam_peak_xy":            tcache["tam_peak_xy"],
        "tam_center_of_mass_xy":  tcache["tam_center_of_mass_xy"],
        "attention_baseline_mass_top10":   tcache["attention_baseline_mass_top10"],
        "attention_baseline_mass_top20":   tcache["attention_baseline_mass_top20"],
        "attention_baseline_mass_top40":   tcache["attention_baseline_mass_top40"],
        "attention_baseline_entropy_norm": tcache["attention_baseline_entropy_norm"],
        "teacher_entropy_full":     tcache["teacher_entropy_full"],
        "teacher_top1_margin_full": tcache["teacher_top1_margin_full"],

        # prompt-token TAM
        "prompt_tam_scope":         tcache["prompt_tam_scope"],
        "prompt_length":            tcache["prompt_length"],
        "tokens_prompt":            tcache["tokens_prompt"],
        "tam_mass_top20_prompt":    tcache["tam_mass_top20_prompt"],
        "tam_entropy_norm_prompt":  tcache["tam_entropy_norm_prompt"],
        "prompt_segments":          tcache["prompt_segments"],
        "prompt_full_length":       tcache["prompt_full_length"],

        # student signals
        "student_lp":      student_lp,
        "student_entropy": student_entropy,
        "adv":             adv,
        "quad":            quad,

        # labels + classification
        **tcache["labels"],
        **tcache["classification"],

        # image metadata
        "image_grid_thw":    tcache["image_grid_thw"],
        "vision_shape":      tcache["vision_shape"],
        "n_patches":         tcache["n_patches"],
        "map_h":             tcache["map_h"],
        "map_w":             tcache["map_w"],
        "patch_index_order": tcache["patch_index_order"],

        # QC
        "tam_valid":           tcache["tam_valid"],
        "tam_failure_reason":  tcache["tam_failure_reason"],
        "attn_baseline_valid":          tcache["attn_baseline_valid"],
        "attn_baseline_failure_reason": tcache["attn_baseline_failure_reason"],

        # subset maps
        "tam_maps_subset": {
            "token_indices":       sub_idx,
            "selection_strata":    sub_strata,
            "selection_rank":      sub_rank,
            "selection_score":     sub_score,
            "deduped_from_strata": sub_dedup,
            "maps_uint8_b64":      sub_tam_maps,
            "attention_maps_uint8_b64": sub_attn_maps,
        },

        "_timings": tcache["_timings"],
    }
    return row


# ============================================================================
# Main orchestration
# ============================================================================
def _parse_student_args(student_args: list[str]) -> list[tuple[str, str]]:
    """--students name=path → [(name, path), ...]"""
    out = []
    for s in student_args:
        if "=" not in s:
            raise ValueError(f"--students requires NAME=PATH form, got {s!r}")
        name, path = s.split("=", 1)
        out.append((name.strip(), path.strip()))
    return out


def _load_subset(subset_path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with subset_path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if limit > 0:
        rows = rows[:limit]
    return rows


def _swap_image_lookup(subset: list[dict]) -> dict:
    return {r["id"]: r.get("image") for r in subset}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--teacher", required=True, help="Teacher HF id or local ckpt path")
    ap.add_argument("--students", action="append", default=[],
                    help="NAME=PATH for each student ckpt; repeat. e.g. T1_0=MMR1/MMR1-3B-SFT")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--teacher-cache", type=Path, default=None,
                    help="Reuse pre-computed teacher cache if exists; else "
                         "default to <out-dir>/teacher_cache.jsonl")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--system-prompt", default=MMR1_SYSTEM_PROMPT)
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-student", action="store_true",
                    help="Run teacher pass only; write teacher cache and exit. "
                         "Useful for splitting wall-time across sessions.")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    teacher_cache_path = args.teacher_cache or (args.out_dir / "teacher_cache.jsonl")
    out_jsonl = args.out_dir / "tam_step1a.jsonl"
    summary_path = args.out_dir / "summary.txt"

    students = _parse_student_args(args.students)
    if not students and not args.skip_student:
        ap.error("Need at least one --students NAME=PATH (or use --skip-student).")

    subset = _load_subset(args.subset, args.limit)
    print(f">>> subset: {len(subset)} samples", file=sys.stderr)
    swap_lookup = _swap_image_lookup(subset)
    image_root = Path(args.image_root)

    # --- Pass 1: teacher cache ---
    teacher_cache: dict[str, dict] = {}
    if teacher_cache_path.exists():
        print(f">>> loading existing teacher cache: {teacher_cache_path}", file=sys.stderr)
        with teacher_cache_path.open() as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    teacher_cache[rec["id"]] = rec
        print(f">>> reused {len(teacher_cache)} teacher cache entries", file=sys.stderr)
    else:
        print(f">>> loading teacher: {args.teacher}", file=sys.stderr)
        t_proc, t_model = _build_model(args.teacher)
        teacher_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with teacher_cache_path.open("w") as tcf:
            for k, rec in enumerate(subset):
                print(f"--- [{k+1}/{len(subset)}] teacher pass on {rec['id']} "
                      f"({rec.get('split_tag')}) ---", file=sys.stderr)
                try:
                    image, source_tag = _load_image_for_sample(rec, image_root, swap_lookup)
                    tblock = teacher_pass(t_proc, t_model, rec, image, args)
                    tblock["id"] = rec["id"]
                    teacher_cache[rec["id"]] = tblock
                    tcf.write(json.dumps(tblock, ensure_ascii=False) + "\n")
                    tcf.flush()
                except Exception as e:  # noqa: BLE001
                    print(f"!! teacher pass failed on {rec['id']}: {e!r}",
                          file=sys.stderr)
        # Free teacher GPU memory before student loads
        try:
            import torch
            del t_model
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    if args.skip_student:
        print(">>> --skip-student set; wrote teacher cache and exiting.",
              file=sys.stderr)
        return 0

    # --- Pass 2: student per-ckpt ---
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    with out_jsonl.open("w") as fout:
        for student_name, student_path in students:
            print(f"\n>>> loading student {student_name} ← {student_path}",
                  file=sys.stderr)
            s_proc, s_model = _build_model(student_path)
            for k, rec in enumerate(subset):
                tcache = teacher_cache.get(rec["id"])
                if tcache is None:
                    continue
                try:
                    image, _ = _load_image_for_sample(rec, image_root, swap_lookup)
                    sblock = student_pass(
                        s_proc, s_model, rec, image,
                        tcache["response_ids"], args,
                    )
                    # Re-load image_path (full version) to record image_sha256
                    image_path_local = Path(rec["image"])
                    if not image_path_local.is_absolute():
                        image_path_local = (image_root / image_path_local).resolve()
                    row = _build_row(rec, image, image_path_local, tcache,
                                     student_name, sblock, args, s_proc)
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                    total_rows += 1
                    if (k + 1) % 25 == 0:
                        print(f"  [{student_name}] {k+1}/{len(subset)} done",
                              file=sys.stderr)
                except Exception as e:  # noqa: BLE001
                    print(f"!! row failed on {student_name}/{rec['id']}: {e!r}",
                          file=sys.stderr)
            try:
                import torch
                del s_model
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    # Brief summary
    with summary_path.open("w") as fs:
        fs.write(f"# Step 1a teacher_greedy  (commit={os.environ.get('MLLMOPD_CODE_COMMIT', 'unknown')})\n")
        fs.write(f"teacher = {args.teacher}\n")
        for name, path in students:
            fs.write(f"student {name} = {path}\n")
        fs.write(f"subset    = {args.subset}\n")
        fs.write(f"n_samples = {len(subset)}\n")
        fs.write(f"n_rows    = {total_rows}\n")
        fs.write(f"teacher_cache = {teacher_cache_path}\n")
        fs.write(f"out_jsonl     = {out_jsonl}\n")
    print(f"\n>>> wrote {total_rows} rows to {out_jsonl}", file=sys.stderr)
    print(f">>> summary: {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
