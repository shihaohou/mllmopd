"""Step 2 TAM causal masking audit.

For each selected (sample, target_token), runs teacher-forced forward
under multiple image-mask strategies and records lp_full_under_mask.

The calibration question Step 1a answered:
  Q: Does TAM mass correlate with |VD|?
  A: NO at per-token Pearson level (content_noun r=-0.001 on n=9118).

The calibration question Step 2 answers:
  Q: Does masking the top-TAM patches change teacher's logprob MORE
     than masking random patches?
  A: TBD — measures *causal* relevance of the TAM-identified region,
     which is orthogonal to Pearson r on scalar mass.

Per (sample, target_token):
  1. Teacher pass on full image → tam_map[target_token] + lp_full_baseline
  2. For each mask strategy:
       masked_image = apply_patch_mask(image, tam_map, strategy)
       lp_masked = teacher_forced_forward(masked_image, prompt, response[:target+1])
       logp_drop = lp_full - lp_masked  (positive = removed visual evidence)

Target token selection per sample (~4 tokens × N samples = ~200 tokens):
  - "content_noun_top_vd": content_noun w/ highest |vd|  (n=2)
  - "quad3_top_vd":        quad==3 (vis_reject_corr) w/ highest |vd|  (n=1)
  - "proper_noun_top_vd":  proper_noun w/ highest |vd|  (only Step 1a +corr; n=1)

Mask strategies (run on EVERY target token):
  - top_tam_20pct:           top 20% TAM patches → gray
  - random_20pct_seed_{42,43,44}: 3 random masks (matched count)
  - keep_top_tam_20pct:      gray everything OUTSIDE top-20% (inverse)
  - bottom_tam_20pct:        gray bottom-20% (negative control — should NOT hurt)

Reuses tam_step1a helpers; per-sample batching gives 6 forward passes per
sample (1 baseline + 5 mask strategies), with each forward giving lp at
ALL target tokens of that sample (teacher-forced on full response).

Usage::

    python -m scripts.audit.tam_step2 \\
        --teacher MMR1/MMR1-7B-RL \\
        --subset data/audit/tam_calibration_subset_v0.jsonl \\
        --student-ckpt $CKPT_T1_0 \\
        --out-dir runs/audit/tam_step2_<TS> \\
        [--limit N] [--shard-id i --num-shards K]

Output schema (one row per sample × target_token × mask_strategy):
  { id, benchmark, image_path,
    response_source="teacher_greedy", teacher_ckpt, student_ckpt,
    response_hash, token_uid, token_idx, token, token_category,
    vd, adv, quad,
    tam_mass_top20, tam_peak_xy, tam_peak_patch_idx,
    mask_strategy, mask_n_patches, mask_patch_indices (b64),
    lp_full_baseline, lp_full_masked,
    logp_drop = lp_full_baseline - lp_full_masked,
    vision_shape, image_grid_thw, map_h, map_w,
    code_commit_run, tam_preproc_version }
"""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tam_sanity import (  # noqa: E402
    _build_model, _build_messages,
    _per_token_teacher_stats,
    _label_response_tokens, _compute_prompt_segments,
    _classify_tokens_v012,
    _b64_uint8_map, _image_sha256, _tokenizer_vocab_hash,
    _sha256_hex,
    MMR1_SYSTEM_PROMPT, QWEN_VL_SPECIAL_IDS,
    _SPACY_AVAILABLE, _SPACY_VERSION, _SPACY_LOAD_ERROR,
)
from tam_step1a import (  # noqa: E402
    _load_image_for_sample, _quad_int, _load_subset,
)
from _tam_core import TAM, tam_scalars  # noqa: E402
# _tam_peak_meta is defined in tam_sanity, NOT in _tam_core.
from tam_sanity import _tam_peak_meta as _tam_peak  # noqa: E402


# ============================================================================
# Mask strategies
# ============================================================================
MASK_STRATEGIES = [
    "top_tam_20pct",
    "random_20pct_seed_42",
    "random_20pct_seed_43",
    "random_20pct_seed_44",
    # GPT round on `2f10687`: scrambled-TAM is the "near-mandatory" control.
    # Same TAM value distribution, randomized positions. Operationally
    # equivalent to random_20pct_seed_X (top-K of value-shuffled map ==
    # uniform-random patch selection), but stored as a distinct condition
    # to make the equivalence explicit in the paper: "TAM value
    # distribution alone doesn't drive the effect; position assignment
    # does."
    "scrambled_tam_seed_142",
    "scrambled_tam_seed_143",
    "scrambled_tam_seed_144",
    "keep_top_tam_20pct",
    "bottom_tam_20pct",
]


def _build_patch_mask(tam_map: np.ndarray, strategy: str,
                      target_frac: float = 0.2) -> np.ndarray:
    """Return boolean array of shape vision_shape; True = MASKED (gray out)."""
    H, W = tam_map.shape
    flat = tam_map.flatten()
    n_total = flat.size
    n_mask = max(1, int(round(n_total * target_frac)))
    sort_idx = np.argsort(flat)  # ascending
    mask_flat = np.zeros(n_total, dtype=bool)

    if strategy == "top_tam_20pct":
        mask_flat[sort_idx[-n_mask:]] = True
    elif strategy == "bottom_tam_20pct":
        mask_flat[sort_idx[:n_mask]] = True
    elif strategy == "keep_top_tam_20pct":
        # Mask everything EXCEPT top n_mask
        keep = set(sort_idx[-n_mask:].tolist())
        for i in range(n_total):
            if i not in keep:
                mask_flat[i] = True
    elif strategy.startswith("random_20pct_seed_"):
        seed = int(strategy.split("_")[-1])
        rng = np.random.default_rng(seed)
        chosen = rng.choice(n_total, size=n_mask, replace=False)
        mask_flat[chosen] = True
    elif strategy.startswith("scrambled_tam_seed_"):
        # Shuffle TAM values across positions, then take top-K of the
        # SHUFFLED map. Mathematically equivalent to random_20pct (uniform
        # random patch selection) — but explicit as a transparency control.
        seed = int(strategy.split("_")[-1])
        rng = np.random.default_rng(seed)
        shuffled = flat.copy()
        rng.shuffle(shuffled)
        perm_sort_idx = np.argsort(shuffled)
        mask_flat[perm_sort_idx[-n_mask:]] = True
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    return mask_flat.reshape(H, W)


def _apply_patch_mask_to_image(image, patch_mask, image_grid_thw,
                                gray_value: int = 128):
    """Resize image to processor target size, gray out 2×2 super-patches per
    vision_shape mask entry. Returns PIL.Image."""
    from PIL import Image

    # image_grid_thw = [1, h_pre_merge, w_pre_merge] (pre-2x2-merge patch count)
    # Each pre-merge patch = 14 pixels. After 2x2 merge → vision_shape entries
    # cover 28x28 pixels each.
    H_pre = image_grid_thw[1]
    W_pre = image_grid_thw[2]
    H_pix = H_pre * 14
    W_pix = W_pre * 14

    img_resized = image.resize((W_pix, H_pix))
    arr = np.array(img_resized).copy()

    Hp, Wp = patch_mask.shape  # vision_shape (post 2x2 merge)
    patch_pixel = 28           # 2 * 14
    for r in range(Hp):
        for c in range(Wp):
            if patch_mask[r, c]:
                arr[r * patch_pixel:(r + 1) * patch_pixel,
                    c * patch_pixel:(c + 1) * patch_pixel] = gray_value
    return Image.fromarray(arr)


# ============================================================================
# Teacher passes
# ============================================================================
def teacher_pass_with_tam(processor, model, image, question, system_prompt,
                          args) -> dict:
    """v0.1.3 pattern (per tam_step1a:148-260) — bare generate + single
    teacher-forced forward. Qwen2.5-VL's `generate(output_hidden_states=True)`
    returns FULL-sequence hidden states at each step (1, P+r, hidden) — for
    high-res prompts (HallusionBench/ChartQA, P>1000) this quadratically
    blows past 100 GB peak. v0.1.3 fix: bare generate gives KV-cached cheap
    response_ids, then ONE teacher-forced forward gives all hidden_states /
    logits in a single ~16 GB pass.

    Also matches `_tam_core.TAM`'s expected `logit_list` shape:
      logit_list[0]   = (1, prompt_len, V)    — all prompt positions
      logit_list[r>0] = (1, 1, V)              — single new token logit
    """
    import torch

    messages = _build_messages(question, image, system_prompt)
    chat = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(
        text=[chat], images=[image], return_tensors="pt", padding=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    input_ids_full = inputs["input_ids"][0].tolist()

    # --- Phase 1: BARE generate (no hidden / scores / attentions) ---
    with torch.inference_mode():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
    sequences = gen_out.sequences[0]
    response_ids = sequences[input_len:].cpu().tolist()
    eos_id = processor.tokenizer.eos_token_id
    response_length = len(response_ids)
    if eos_id is not None and eos_id in response_ids:
        response_length = response_ids.index(eos_id) + 1
    response_ids = response_ids[:response_length]
    del gen_out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if response_length == 0:
        # Edge case: empty response. Bail with minimal block.
        grid_thw = inputs["image_grid_thw"][0].tolist()
        return {
            "response_ids": [], "response_length": 0,
            "lp_full": [], "teacher_entropy_full": [],
            "tam_maps": [], "vision_shape": (grid_thw[1] // 2, grid_thw[2] // 2),
            "image_grid_thw": grid_thw, "prompt_len": input_len,
            "input_ids_full": input_ids_full, "chat_template": chat,
        }

    # --- Phase 2: single teacher-forced forward (logits only) ---
    # Step 2 does NOT need hidden_states. We use `out_full.logits` (already
    # the lm_head projection) directly as `logit_list` for TAM. This is
    # numerically identical to `model.lm_head(out_full.hidden_states[-1])`
    # but skips an extra ~600 MB tensor allocation. (Step 1a kept hidden
    # states because it ALSO needed eager-attention output and a separate
    # blank-image forward; Step 2 has neither.)
    response_tensor = torch.tensor(response_ids, device=model.device).unsqueeze(0)
    full_ids = torch.cat([inputs["input_ids"], response_tensor], dim=1)
    full_mask = torch.cat([
        inputs["attention_mask"],
        torch.ones_like(response_tensor),
    ], dim=1)
    fwd_kwargs = dict(inputs)
    fwd_kwargs["input_ids"] = full_ids
    fwd_kwargs["attention_mask"] = full_mask
    with torch.inference_mode():
        out_full = model(
            **fwd_kwargs,
            output_hidden_states=False,
            output_attentions=False,
        )

    # lp_full per response token (predictor index = input_len + t - 1)
    full_logits_2d = out_full.logits[0]   # (T_full, V)
    lp_full: list[float] = []
    teacher_entropy: list[float] = []
    for t in range(response_length):
        pred_idx = input_len + t - 1
        row = full_logits_2d[pred_idx]
        log_probs = torch.log_softmax(row, dim=-1)
        probs = torch.softmax(row.float(), dim=-1)
        lp_full.append(float(log_probs[response_ids[t]]))
        teacher_entropy.append(float(-(probs * torch.log(probs + 1e-12)).sum()))

    # logit_list as VIEWS into full_logits with the shape _tam_core.TAM expects
    full_logits_3d = full_logits_2d.unsqueeze(0)   # (1, T_full, V)
    logit_list = []
    for r in range(response_length):
        if r == 0:
            logit_list.append(full_logits_3d[:, :input_len, :])
        else:
            logit_list.append(full_logits_3d[:, input_len + r - 1: input_len + r, :])

    grid_thw = inputs["image_grid_thw"][0].tolist()
    vision_shape = (grid_thw[1] // 2, grid_thw[2] // 2)
    tokens_full_list = sequences[: input_len + response_length].cpu().tolist()

    # TAM per response token
    img_scores_list: list = []
    response_maps: list = []
    for i in range(response_length):
        m = TAM(
            tokens=tokens_full_list, vision_shape=vision_shape,
            logit_list=logit_list, special_ids=QWEN_VL_SPECIAL_IDS,
            processor=processor, target_token=i,
            img_scores_list=img_scores_list,
        )
        response_maps.append(m)

    # Aggressive cleanup so each mask-strategy forward starts from baseline
    del logit_list, full_logits_3d, full_logits_2d
    try:
        del out_full
    except NameError:
        pass
    try:
        del response_tensor, full_ids, full_mask, fwd_kwargs
    except NameError:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "response_ids": response_ids,
        "response_length": response_length,
        "lp_full": lp_full,
        "teacher_entropy_full": teacher_entropy,
        "tam_maps": response_maps,
        "vision_shape": vision_shape,
        "image_grid_thw": grid_thw,
        "prompt_len": input_len,
        "input_ids_full": input_ids_full,
        "chat_template": chat,
    }


def teacher_forced_lp_all(processor, model, image, question, response_ids,
                          system_prompt) -> list:
    """Single forward on (prompt + full response) under given image; returns
    lp at each response position. Used for mask experiments — one forward per
    image gives all target positions' lp."""
    import torch

    messages = _build_messages(question, image, system_prompt)
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
    lp = [None] * R   # type: ignore[list-item]
    try:
        with torch.inference_mode():
            out = model(**kw)
        logits = out.logits[0]
        for t in range(R):
            pred_idx = input_len + t - 1
            log_probs = torch.log_softmax(logits[pred_idx], dim=-1)
            lp[t] = float(log_probs[response_ids[t]])
        del out, logits
    except Exception as e:  # noqa: BLE001
        print(f"!! teacher_forced_lp_all failed: {type(e).__name__}: {e!s:.120}",
              file=sys.stderr)
    if torch.cuda.is_available():
        import torch as _torch
        _torch.cuda.empty_cache()
    return lp


def student_pass(processor, model, image, question, response_ids,
                 system_prompt) -> list:
    """Same as teacher_forced_lp_all but for student model."""
    return teacher_forced_lp_all(processor, model, image, question,
                                  response_ids, system_prompt)


# ============================================================================
# Target token selection
# ============================================================================
def select_target_tokens(tblock, classification, labels) -> list[dict]:
    """Pick a stratified set of target tokens within a sample.

    Strata (per docs/tam_calibration_schema.md §Step 2 sketch):
      - content_noun + top-|vd|     n=2
      - quad==3 visual_rejection    n=1
      - proper_noun + top-|vd|      n=1

    Returns list of dicts: {token_idx, stratum, score}."""
    R = tblock["response_length"]
    if R == 0:
        return []

    categories = classification["token_category"]
    is_blank = labels["is_blankness_token"]
    abs_vd_proxy = [abs(s["tam_mass_top20"] - 0.2) for s in
                    [tam_scalars(m) for m in tblock["tam_maps"]]]
    # Use TAM mass as a SAMPLE-LOCAL ranking proxy when vd is not yet available
    # (will be re-ranked by |vd| once student pass + blank pass populate vd/adv).
    # For Step 2 stand-alone use, we rank by tam_mass within stratum.

    out: list[dict] = []
    by_cat: dict[str, list[int]] = {}
    for t in range(R):
        by_cat.setdefault(categories[t], []).append(t)

    def pick_top_k(idxs: list[int], k: int, stratum: str):
        ranked = sorted(idxs, key=lambda i: -abs_vd_proxy[i])[:k]
        for rank, idx in enumerate(ranked):
            out.append({
                "token_idx": idx,
                "stratum": stratum,
                "rank_within_stratum": rank,
                "score": float(abs_vd_proxy[idx]),
            })

    pick_top_k(by_cat.get("content_noun", []), 2, "content_noun_top_tam")
    pick_top_k(by_cat.get("proper_noun", []),  1, "proper_noun_top_tam")
    # quad==3 stratum needs vd+adv; for Step 2 standalone we approximate by
    # "high tam_mass + non-template token". A future v0.2 of Step 2 should
    # take an explicit step1a JSONL to filter by quad==3.
    non_struct = [t for t in range(R)
                  if categories[t] not in
                  ("template_token", "special_token", "punctuation", "other")]
    pick_top_k(non_struct, 1, "non_struct_top_tam")

    # Dedup by token_idx (a token may qualify for multiple strata)
    seen = set()
    deduped: list[dict] = []
    for d in out:
        if d["token_idx"] not in seen:
            seen.add(d["token_idx"])
            deduped.append(d)
    return deduped


# ============================================================================
# Per-sample orchestration
# ============================================================================
def process_one_sample(rec, image, processor, model, args, image_root) -> list[dict]:
    """Run Step 2 on one sample. Returns list of rows (target_token × mask_strategy)."""
    import torch
    # 1. Baseline teacher pass: get tam_maps + lp_full
    tblock = teacher_pass_with_tam(
        processor, model, image, rec["question"], args.system_prompt, args,
    )
    response_ids = tblock["response_ids"]
    R = tblock["response_length"]
    if R == 0:
        return []

    labels = _label_response_tokens(response_ids, processor.tokenizer)
    classification = _classify_tokens_v012(
        response_ids, processor.tokenizer, labels["is_answer_token"],
    )

    targets = select_target_tokens(tblock, classification, labels)
    if not targets:
        return []
    target_idxs = sorted({t["token_idx"] for t in targets})

    # 2. For each mask strategy:
    rows: list[dict] = []
    image_grid_thw = tblock["image_grid_thw"]
    vision_shape = tblock["vision_shape"]
    tokens = processor.tokenizer.convert_ids_to_tokens(response_ids)
    response_hash = _sha256_hex(json.dumps(response_ids).encode())

    # Strategy lp_masked: keyed by (strategy, target_idx) → lp_masked
    strategy_lp: dict[str, list] = {}

    for strategy in MASK_STRATEGIES:
        # For each target token, mask is built FROM THAT TOKEN'S TAM map.
        # Since masks differ per target token, we need one forward per
        # (target_token, strategy). N_targets × N_strategies × 1 forward each.
        per_target_lp: list = []
        for t in target_idxs:
            tam_map = tblock["tam_maps"][t]
            patch_mask = _build_patch_mask(tam_map, strategy, target_frac=0.2)
            masked_image = _apply_patch_mask_to_image(
                image, patch_mask, image_grid_thw, gray_value=128,
            )
            lp_all = teacher_forced_lp_all(
                processor, model, masked_image, rec["question"],
                response_ids, args.system_prompt,
            )
            per_target_lp.append({
                "target_idx": t,
                "lp_at_target": lp_all[t] if t < len(lp_all) else None,
                "n_masked_patches": int(patch_mask.sum()),
            })
            # Aggressive cleanup
            del masked_image, lp_all
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        strategy_lp[strategy] = per_target_lp

    # 3. Emit rows
    image_path_local = Path(rec["image"])
    if not image_path_local.is_absolute():
        image_path_local = (image_root / image_path_local).resolve()

    for t_meta in targets:
        t = t_meta["token_idx"]
        if t >= R:
            continue
        peak_meta = _tam_peak(tblock["tam_maps"][t])
        s_t = tam_scalars(tblock["tam_maps"][t])
        for strategy in MASK_STRATEGIES:
            entry = next((x for x in strategy_lp[strategy] if x["target_idx"] == t), None)
            if entry is None:
                continue
            lp_baseline = tblock["lp_full"][t] if t < len(tblock["lp_full"]) else None
            lp_masked = entry["lp_at_target"]
            logp_drop = (lp_baseline - lp_masked) if (lp_baseline is not None and lp_masked is not None) else None

            rows.append({
                "id": rec["id"],
                "benchmark": rec.get("benchmark"),
                "split_tag": rec.get("split_tag", "step2"),
                "image_path": str(image_path_local),
                "image_corruption": rec.get("image_corruption"),
                "question": rec["question"],

                "response_source": "teacher_greedy",
                "teacher_ckpt": args.teacher,
                "tam_preproc_version": "v0.1.3-step2",
                "code_commit_run": os.environ.get("MLLMOPD_CODE_COMMIT", "unknown"),

                "response_hash": response_hash,
                "response_length": R,
                "token_idx": t,
                "token": tokens[t] if t < len(tokens) else "",
                "token_uid": f"{rec['id']}:teacher_greedy:{response_hash}:{t}",
                "token_category": classification["token_category"][t],
                "is_answer_token": bool(labels["is_answer_token"][t]),
                "is_think_token":  bool(labels["is_think_token"][t]),
                "is_blankness_token": bool(labels["is_blankness_token"][t]),

                "stratum": t_meta["stratum"],
                "rank_within_stratum": t_meta["rank_within_stratum"],

                "tam_mass_top10": s_t["tam_mass_top10"],
                "tam_mass_top20": s_t["tam_mass_top20"],
                "tam_mass_top40": s_t["tam_mass_top40"],
                "tam_entropy_norm": s_t["tam_entropy_norm"],
                "tam_peak_patch_idx": peak_meta["tam_peak_patch_idx"],
                "tam_peak_xy": peak_meta["tam_peak_xy"],
                "tam_center_of_mass_xy": peak_meta["tam_center_of_mass_xy"],

                "mask_strategy": strategy,
                "mask_target_frac": 0.2,
                "mask_n_patches": entry["n_masked_patches"],
                "lp_full_baseline": lp_baseline,
                "lp_full_masked":   lp_masked,
                "logp_drop":        logp_drop,

                "vision_shape": list(vision_shape),
                "image_grid_thw": image_grid_thw,
                "map_h": int(vision_shape[0]),
                "map_w": int(vision_shape[1]),
            })

    # Cleanup big objects
    del tblock
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return rows


# ============================================================================
# CLI
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--system-prompt", default=MMR1_SYSTEM_PROMPT)
    ap.add_argument("--image-root", default=".")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.out_dir / "tam_step2.jsonl"
    summary   = args.out_dir / "summary.txt"

    subset = _load_subset(args.subset, args.limit,
                          shard_id=args.shard_id, num_shards=args.num_shards)
    if args.num_shards > 1:
        print(f">>> shard {args.shard_id}/{args.num_shards}: {len(subset)} samples",
              file=sys.stderr)
    else:
        print(f">>> subset: {len(subset)} samples", file=sys.stderr)

    print(f">>> loading teacher: {args.teacher}", file=sys.stderr)
    processor, model = _build_model(args.teacher)

    image_root = Path(args.image_root)
    swap_lookup = {r["id"]: r.get("image") for r in subset}

    total_rows = 0
    n_done = 0
    t_start = time.time()
    with out_jsonl.open("w") as fout:
        for k, rec in enumerate(subset):
            print(f"--- [{k+1}/{len(subset)}] sample {rec['id']} "
                  f"({rec.get('split_tag')}) ---", file=sys.stderr)
            try:
                image, _ = _load_image_for_sample(rec, image_root, swap_lookup)
                rows = process_one_sample(rec, image, processor, model, args, image_root)
                for row in rows:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                total_rows += len(rows)
                n_done += 1
            except Exception as e:  # noqa: BLE001
                print(f"!! sample {rec['id']} FAILED: {e!r}", file=sys.stderr)
            finally:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                if torch.cuda.is_available():
                    mb_alloc = torch.cuda.memory_allocated() / 1024**2
                    mb_resv  = torch.cuda.memory_reserved() / 1024**2
                    print(f"    [cuda] alloc={mb_alloc:.0f} MB  reserved={mb_resv:.0f} MB",
                          file=sys.stderr)

    dt = time.time() - t_start
    with summary.open("w") as fs:
        fs.write(f"# Step 2 TAM causal masking  (commit={os.environ.get('MLLMOPD_CODE_COMMIT','unknown')})\n")
        fs.write(f"teacher = {args.teacher}\n")
        fs.write(f"subset  = {args.subset}\n")
        fs.write(f"n_samples = {len(subset)}\n")
        fs.write(f"n_samples_done = {n_done}\n")
        fs.write(f"n_rows  = {total_rows}\n")
        fs.write(f"wall_s  = {dt:.1f}\n")
    print(f">>> wrote {total_rows} rows ({n_done}/{len(subset)} samples) "
          f"in {dt:.1f}s → {out_jsonl}", file=sys.stderr)
    print(f">>> summary: {summary}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
