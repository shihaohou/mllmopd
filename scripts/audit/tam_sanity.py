"""Step 0 TAM sanity check on Qwen2.5-VL (MMR1-3B-SFT default).

v0.1.2 (incorporates GPT review on commit 31b67ac of Step 0 brief):
  - token_category enum × R (12 categories) via spaCy POS + regex pre-pass
  - attention baseline (last-layer-avg-heads) with same scalar family as TAM
  - TAM peak metadata (peak_patch_idx, peak_xy, center_of_mass_xy)
  - prompt_tam_scope = "question_only" (locked vs v0.1.1 doc-vs-code drift)
  - tam_maps_subset.attention_maps_uint8_b64 for direct visual comparison

For each probe in `data/audit/tam_probes.jsonl`:
  1. Load Qwen2.5-VL via HF transformers
  2. Build chat-template prompt (MMR1 system_prompt before image, per
     run_audit_pass.py:_build_messages)
  3. Run `model.generate(..., output_hidden_states=True, output_scores=True,
     output_attentions=True, return_dict_in_generate=True)`
  4. Compute `logits = [model.lm_head(feats[-1]) for feats in
     outputs.hidden_states]` (TAM input)
  5. Compute `lp_full`, `teacher_entropy_full`, `teacher_top1_margin_full`
     from `outputs.scores`
  6. For each generated token (and via ECI recursion, each question-prompt
     token), call `_tam_core.TAM` → get normalized [0,1] map → compute
     scalars + peak meta + attention baseline → optionally save PNG overlay
  7. Classify each response token (POS + regex → token_category enum)
  8. Emit JSONL row matching v0.1.2 Step 0 schema subset (see
     docs/tam_calibration_schema.md)

Output dir: $MLLMOPD_RUNS/tam_sanity_<TS>/
  ├── tam_sanity.jsonl       — one row per probe (v0.1.1 Step 0 subset)
  ├── overlays/<id>/<idx>_<token>.png — PNG overlays for selected tokens
  └── summary.txt            — eyeball summary (per-probe top-5 mass tokens)

Usage (devbox):
  python -m scripts.audit.tam_sanity \\
      --probes data/audit/tam_probes.jsonl \\
      --model "${MMR1_3B_SFT_CKPT:-MMR1/MMR1-3B-SFT}" \\
      --out-dir "${MLLMOPD_RUNS}/tam_sanity_$(date +%Y%m%d-%H%M%S)" \\
      --max-new-tokens 256

Decision criterion (go/no-go for Step 1):
  PASS if for the 3 POPE probes, the answer-token TAM heatmap visibly
  concentrates near the asked-about object (tennis racket / knife / car).
  FAIL if heatmaps are uniform or focus on the wrong region — abandon
  the method-tier port of TAM.
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
from pathlib import Path

import numpy as np

# Vendored TAM core sits next to this file. When invoked as `python -m
# scripts.audit.tam_sanity`, the package import works; when invoked as a
# script path, ensure the parent dir is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _tam_core import TAM, tam_scalars  # noqa: E402  (after sys.path tweak)

# v0.1.2: spaCy POS tagger for token_category. Optional — fall back to
# regex-only categorization if unavailable; the run is still valid but
# content_noun / pronoun / visual_attribute become "other".
try:
    import spacy as _spacy_mod
    _SPACY_NLP = _spacy_mod.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
    _SPACY_VERSION = _spacy_mod.__version__
    _SPACY_LOAD_ERROR = None
except Exception as _spacy_err:  # noqa: BLE001
    _SPACY_NLP = None
    _SPACY_AVAILABLE = False
    _SPACY_VERSION = None
    _SPACY_LOAD_ERROR = f"{type(_spacy_err).__name__}: {_spacy_err!s:.160}"


# Qwen2.5-VL shares Qwen2-VL's tokenizer + special-token IDs.
# See _tam_core docstring for derivation.
QWEN_VL_SPECIAL_IDS = {
    "img_id":    [151652, 151653],                          # <|vision_start|>, <|vision_end|>
    "prompt_id": [151653, [151645, 198, 151644, 77091]],     # <|vision_end|> → <|im_end|>\n<|im_start|>assistant
    "answer_id": [[198, 151644, 77091, 198], -1],            # \n<|im_start|>assistant\n → end
}

# Canonical MMR1 training-time sysprompt (verbatim from
# scripts/audit/run_t1_trajectory.sh:62). Passed via --system-prompt when we
# want MMR1 models to emit <think>...</think><answer>...</answer>.
MMR1_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The User provides an image "
    "and asks a question. The Assistant first analyzes both the image and "
    "the question, then carefully thinks about the reasoning process step "
    "by step, and finally provides the User with an accurate answer. The "
    "Assistant must carefully checkout the correctness and validity of "
    "each reasoning step. If any errors or inconsistencies are found during "
    "the reasoning process, the Assistant reflects and corrects them "
    "logically. The reasoning process and answer are enclosed within "
    "<think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here, with potential reflections and "
    "corrections </think><answer> final answer here, with the key result "
    "enclosed in \\boxed{} </answer>."
)

# BLANK_RE copied from src/mllmopd/analysis/t1_blankness_trajectory.py:39-52
# to keep _tam_core self-contained without pulling the analysis package.
BLANK_PHRASES = [
    r"\bblank\b", r"completely white", r"no information",
    r"cannot see", r"no visible", r"\bplaceholder\b",
    r"I cannot determine", r"image is empty",
    r"no chart\b", r"no image\b", r"\birrelevant\b",
]
BLANK_RE = re.compile("|".join(BLANK_PHRASES), re.IGNORECASE)

# v0.1.2: token-category constants (per docs/tam_calibration_schema.md §token_category)
# Qwen2.5-VL special IDs that mark non-content positions in the response.
QWEN_SPECIAL_IDS_SET = {
    151643, 151644, 151645, 151652, 151653, 151654, 151655, 151656, 151657, 151658,
}
TEMPLATE_TOKEN_RE = re.compile(
    r"<think>|</think>|<answer>|</answer>|\\boxed|:\*\*|\*\*:"
)
# v0.1.3 classifier: MMR1's "\boxed{answer}" template tokenizes to BPE pieces
# `\`, `boxed`, `{`, `answer`, `}` — TEMPLATE_TOKEN_RE catches `\boxed` as the
# combined string, but the BPE-split bare pieces `boxed` and `answer` slip
# through and reach spaCy, which tags them as PROPN → proper_noun. Pre-flight
# audit on tam_step1a_20260525-190333 found this contaminated 63 (sample,
# token) pairs across ChartQA / HallusionBench / POPE / MathVista all with
# vd≈adv≈0. Catch them here as template_token regardless of spaCy POS.
MMR1_BOXED_BARE_RE = re.compile(
    r"^\s*(?:boxed|Boxed|BOXED|answer|Answer|ANSWER)\s*$"
)
META_COT_WORDS = {
    "Crop", "crop", "Looking", "examine", "analyze", "carefully",
    "image", "based", "according", "user",
}
SPATIAL_RELATION_WORDS = {
    "above", "below", "left", "right", "near", "far", "between", "behind",
    "front", "beneath", "underneath", "over", "around", "inside", "outside",
    "top", "bottom", "center", "middle", "edge", "beside", "next",
}
ADJ_STOPLIST = {
    "good", "bad", "many", "some", "few", "much", "more", "less",
    "any", "every", "such", "other",
}
ANSWER_COMMIT_RE = re.compile(r"^\s*(Yes|No|yes|no)[,.]?\s*$")


# ============================================================================
# Model loading (parity with run_audit_pass.py:_build_model)
# ============================================================================
def _build_model(model_id: str):
    """v0.1.2: honor MLLMOPD_ATTN_IMPL env. Default is "eager" for tam_sanity
    because FA2 and SDPA both silently return None for outputs.attentions
    when output_attentions=True is requested, which breaks the v0.1.2
    attention baseline. eager is ~3-5× slower than FA2 but Step 0 only
    runs 4 probes, so the cost is ~5-10 min instead of ~1-2 min.

    Override with `MLLMOPD_ATTN_IMPL=flash_attention_2` (or `sdpa`) if you
    do not need the attention baseline (e.g. you are only checking
    TAM scalars, not the baseline)."""
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        proc_kwargs = {
            "trust_remote_code": True,
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 1280 * 28 * 28,
        }
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelCls
        proc_kwargs = {"trust_remote_code": True}

    processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)

    common = dict(
        torch_dtype=torch.bfloat16,
        # v0.1.2 fix: force single-GPU. device_map="auto" was spreading
        # layers across all visible H800s unevenly, OOM'ing one card while
        # the other 7 sat idle. 7B + 3B easily fit on one H800 (140 GB).
        # Honors CUDA_VISIBLE_DEVICES (device 0 == first visible GPU).
        device_map={"": 0},
        trust_remote_code=True,
    )
    attn_impl = os.environ.get("MLLMOPD_ATTN_IMPL", "eager")
    print(f">>> attn_implementation = {attn_impl} (override via MLLMOPD_ATTN_IMPL)",
          file=sys.stderr)
    try:
        model = ModelCls.from_pretrained(
            model_id, attn_implementation=attn_impl, **common
        )
    except (ImportError, ValueError, RuntimeError) as e:
        # Only fall back if user did NOT explicitly choose eager (eager is
        # what enables the attention baseline; if it fails we want a hard
        # error, not silent degradation).
        if attn_impl == "eager" and os.environ.get("MLLMOPD_ATTN_IMPL") is None:
            print(f">>> eager unavailable ({e!s:.80}); falling back to sdpa "
                  f"— attention baseline will be invalid",
                  file=sys.stderr)
            model = ModelCls.from_pretrained(
                model_id, attn_implementation="sdpa", **common
            )
        else:
            raise
    model.eval()
    return processor, model


def _build_messages(question: str, image, system_prompt: str):
    """Parity with run_audit_pass.py:_build_messages — sysprompt is prepended
    INTO the user-turn (NOT a separate system role), so MMR1 stays in its
    trained behavior."""
    content: list = []
    if system_prompt:
        content.append({"type": "text", "text": system_prompt.strip() + " "})
    content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


# ============================================================================
# Helpers
# ============================================================================
def _sha256_hex(data: bytes, n_chars: int = 16) -> str:
    return hashlib.sha256(data).hexdigest()[:n_chars]


def _image_sha256(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _tokenizer_vocab_hash(processor) -> str:
    """Stable hash of the tokenizer's vocab (sorted token-string list)."""
    vocab = processor.tokenizer.get_vocab()
    # token -> id; sort by id for determinism
    items = sorted(vocab.items(), key=lambda kv: kv[1])
    joined = "\n".join(f"{tok}\t{idx}" for tok, idx in items).encode("utf-8")
    return "sha256:" + hashlib.sha256(joined).hexdigest()[:16]


def _b64_uint8_map(normalized_map_2d: np.ndarray) -> str:
    """Encode a [0,1] H×W map as base64 uint8 for inline JSONL storage.
    H,W are recoverable from `vision_shape`/`map_h`/`map_w` in the row."""
    u8 = np.clip(normalized_map_2d * 255.0, 0, 255).astype(np.uint8)
    return base64.b64encode(u8.tobytes()).decode("ascii")


def _overlay_png(image_pil_rgb, normalized_map_2d, out_path: Path, alpha: float = 0.5) -> None:
    """Render a colormap heatmap blended on top of the image. PNG-only (no
    LaTeX text-vis from upstream TAM)."""
    try:
        import cv2
    except ImportError:
        print("!! cv2 unavailable; skipping PNG overlay", file=sys.stderr)
        return
    img_rgb = np.array(image_pil_rgb.convert("RGB"))
    H_img, W_img, _ = img_rgb.shape
    # Heatmap → uint8 → JET → resize to original
    u8 = np.clip(normalized_map_2d * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    heat_bgr = cv2.resize(heat_bgr, (W_img, H_img), interpolation=cv2.INTER_CUBIC)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    blended = (heat_bgr.astype(np.float32) * alpha +
               img_bgr.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blended)


# ============================================================================
# Span / label computation
# ============================================================================
def _find_spans(text: str, pattern: re.Pattern) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in pattern.finditer(text)]


# Match <answer>...</answer> and <think>...</think> with non-greedy body.
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE  = re.compile(r"<think>(.*?)</think>",   re.DOTALL | re.IGNORECASE)


def _char_spans_to_token_spans(char_spans, tokenizer, token_ids: list[int]):
    """Best-effort char-span → token-span mapping. Re-decodes the prefix
    of token_ids up to each token boundary to learn the cumulative-char-len
    at each boundary; uses that to map char-spans → inclusive-token-spans.

    This is an O(R²) char re-decode loop but R is at most a few hundred for
    Step 0 sanity, so it's fine."""
    # Cumulative char length at each token boundary (boundary i = after token i-1)
    boundaries = [0]
    for r in range(1, len(token_ids) + 1):
        decoded = tokenizer.decode(token_ids[:r], skip_special_tokens=False)
        boundaries.append(len(decoded))

    token_spans = []
    for cstart, cend in char_spans:
        # First token whose end-boundary > cstart
        tstart = next((t for t in range(len(token_ids))
                       if boundaries[t + 1] > cstart), len(token_ids))
        # Last token whose start-boundary < cend, then +1 for exclusive
        tend = next((t for t in range(len(token_ids) - 1, -1, -1)
                     if boundaries[t] < cend), tstart) + 1
        token_spans.append([int(tstart), int(tend)])
    return token_spans


def _label_response_tokens(
    response_ids: list[int],
    tokenizer,
) -> dict:
    """Compute is_blankness/answer/think + answer/think token-spans."""
    R = len(response_ids)
    full_text = tokenizer.decode(response_ids, skip_special_tokens=False)

    blank_char_spans  = _find_spans(full_text, BLANK_RE)
    answer_char_spans = _find_spans(full_text, _ANSWER_TAG_RE)
    think_char_spans  = _find_spans(full_text, _THINK_TAG_RE)

    blank_token_spans  = _char_spans_to_token_spans(blank_char_spans, tokenizer, response_ids)
    answer_token_spans = _char_spans_to_token_spans(answer_char_spans, tokenizer, response_ids)
    think_token_spans  = _char_spans_to_token_spans(think_char_spans, tokenizer, response_ids)

    is_blankness = [False] * R
    is_answer    = [False] * R
    is_think     = [False] * R
    for s, e in blank_token_spans:
        for i in range(s, min(e, R)):
            is_blankness[i] = True
    for s, e in answer_token_spans:
        for i in range(s, min(e, R)):
            is_answer[i] = True
    for s, e in think_token_spans:
        for i in range(s, min(e, R)):
            is_think[i] = True

    return {
        "is_blankness_token": is_blankness,
        "is_answer_token":    is_answer,
        "is_think_token":     is_think,
        "answer_token_spans": answer_token_spans,
        "think_token_spans":  think_token_spans,
        "answer_span_source": "regex_tag",  # <answer>...</answer> regex
    }


def _compute_prompt_segments(input_ids_full: list[int], special_ids) -> dict:
    """Locate system / image_placeholder / question spans in the FULL input
    sequence (not in TAM's question-only prompt). Half-open intervals."""
    vision_start = special_ids["img_id"][0]   # 151652
    vision_end   = special_ids["img_id"][1]   # 151653
    im_end       = 151645
    im_start     = 151644

    try:
        vstart_idx = input_ids_full.index(vision_start)
    except ValueError:
        vstart_idx = -1
    try:
        vend_idx = input_ids_full.index(vision_end)
    except ValueError:
        vend_idx = -1

    # First <|im_end|> after vision_end marks end of question
    question_end = -1
    if vend_idx >= 0:
        for i in range(vend_idx + 1, len(input_ids_full)):
            if input_ids_full[i] == im_end:
                question_end = i
                break

    segments = {}
    if vstart_idx >= 0:
        segments["system"] = [0, int(vstart_idx)]
        segments["image_placeholder"] = [int(vstart_idx), int(vend_idx) + 1] if vend_idx >= 0 else [int(vstart_idx), int(vstart_idx)]
    if vend_idx >= 0 and question_end >= 0:
        segments["question"] = [int(vend_idx) + 1, int(question_end)]
    return segments


# ============================================================================
# v0.1.2: token-category classification (regex pre-pass + spaCy POS)
# ============================================================================
def _classify_tokens_v012(
    response_ids: list[int],
    tokenizer,
    is_answer_token: list[bool],
) -> dict:
    """Return per-token (pos_tag, word_idx, word_text, token_category)
    per v0.1.2 schema. Regex pre-pass for mechanical categories first;
    spaCy POS for the linguistic ones; fallback to "other" if spaCy
    unavailable."""
    R = len(response_ids)
    full_text = tokenizer.decode(response_ids, skip_special_tokens=False)

    # Cumulative char-length at each subword boundary
    boundaries = [0]
    for r in range(1, R + 1):
        boundaries.append(len(tokenizer.decode(response_ids[:r], skip_special_tokens=False)))

    # spaCy doc on the full response text (response is already plain text after
    # decode; special-token text like "<answer>" is decoded literally and POS-
    # tagged as a tag-shaped token, which we override below).
    spacy_doc = _SPACY_NLP(full_text) if _SPACY_AVAILABLE else None

    pos_tag = [""] * R
    word_idx = [-1] * R
    word_text = [""] * R
    token_category = ["other"] * R

    for t in range(R):
        tid = response_ids[t]
        tok_str = tokenizer.decode([tid], skip_special_tokens=False)
        char_start = boundaries[t]
        char_end = boundaries[t + 1]
        char_mid = (char_start + char_end) // 2

        # 1. special_token by id
        if tid in QWEN_SPECIAL_IDS_SET:
            token_category[t] = "special_token"
            continue

        # 2. template_token by text pattern
        if TEMPLATE_TOKEN_RE.search(tok_str):
            token_category[t] = "template_token"
            continue

        # 2b. v0.1.3: MMR1 \boxed{answer} BPE-split pieces. The bare
        # `boxed` / `answer` tokens decode without the surrounding `\`/`{}` and
        # spaCy would call them PROPN. Override before reaching spaCy.
        if MMR1_BOXED_BARE_RE.match(tok_str):
            token_category[t] = "template_token"
            continue

        # 3. punctuation
        stripped = tok_str.strip()
        if stripped and all(not c.isalnum() for c in stripped):
            token_category[t] = "punctuation"
            pos_tag[t] = "PUNCT"
            continue

        # 4. answer_token marker (Yes/No commitment inside <answer>)
        if is_answer_token[t] and ANSWER_COMMIT_RE.match(tok_str):
            token_category[t] = "answer_token"
            continue

        # 5. spaCy POS lookup via char midpoint
        if spacy_doc is None:
            token_category[t] = "other"
            continue

        matched_word = None
        matched_idx = -1
        for i, w in enumerate(spacy_doc):
            if w.idx <= char_mid < w.idx + len(w.text):
                matched_word = w
                matched_idx = i
                break
        if matched_word is None:
            continue
        pos_tag[t] = matched_word.pos_
        word_idx[t] = matched_idx
        word_text[t] = matched_word.text

        # 6. Meta-CoT keyword override (e.g. "Crop", "Looking")
        if matched_word.text in META_COT_WORDS:
            token_category[t] = "meta_cot_token"
            continue

        # 7. POS → category
        text_lower = matched_word.text.lower()
        if matched_word.pos_ == "PRON":
            token_category[t] = "pronoun"
        elif matched_word.pos_ == "PROPN":
            token_category[t] = "proper_noun"
        elif matched_word.pos_ == "NUM" or re.match(r"^[0-9]+(\.[0-9]+)?$", matched_word.text):
            token_category[t] = "visual_number"
        elif matched_word.pos_ == "NOUN":
            token_category[t] = "content_noun"
        elif matched_word.pos_ == "ADJ" and text_lower not in ADJ_STOPLIST:
            token_category[t] = "visual_attribute"
        elif text_lower in SPATIAL_RELATION_WORDS:
            token_category[t] = "spatial_relation"
        else:
            token_category[t] = "other"

    return {
        "pos_tag": pos_tag,
        "word_idx": word_idx,
        "word_text": word_text,
        "token_category": token_category,
        # Derived booleans (per v0.1.2 schema for analyzer convenience)
        "is_template_token":  [c == "template_token"  for c in token_category],
        "is_special_token":   [c == "special_token"   for c in token_category],
        "is_pronoun":         [c == "pronoun"         for c in token_category],
        "is_meta_cot_token":  [c == "meta_cot_token"  for c in token_category],
    }


# ============================================================================
# v0.1.2: TAM peak metadata
# ============================================================================
def _tam_peak_meta(normalized_map_2d: np.ndarray) -> dict:
    """Return {tam_peak_patch_idx, tam_peak_xy, tam_center_of_mass_xy} for
    one TAM map in [0,1]^(H×W). xy normalized to [0,1]: x=col/(W-1),
    y=row/(H-1)."""
    H, W = normalized_map_2d.shape
    flat = normalized_map_2d.flatten()
    if flat.sum() < 1e-12:
        return {
            "tam_peak_patch_idx": -1,
            "tam_peak_xy": [0.0, 0.0],
            "tam_center_of_mass_xy": [0.5, 0.5],
        }
    peak_flat = int(np.argmax(flat))
    peak_r, peak_c = divmod(peak_flat, W)
    x_peak = peak_c / max(1, W - 1)
    y_peak = peak_r / max(1, H - 1)

    # Center of mass — weighted by activation
    rows = np.arange(H).reshape(-1, 1).repeat(W, axis=1)
    cols = np.arange(W).reshape(1, -1).repeat(H, axis=0)
    total = float(normalized_map_2d.sum())
    cm_r = float((rows * normalized_map_2d).sum() / total)
    cm_c = float((cols * normalized_map_2d).sum() / total)
    return {
        "tam_peak_patch_idx": peak_flat,
        "tam_peak_xy": [float(x_peak), float(y_peak)],
        "tam_center_of_mass_xy": [cm_c / max(1, W - 1), cm_r / max(1, H - 1)],
    }


# ============================================================================
# v0.1.2: attention baseline — last-layer-avg-heads from outputs.attentions
# ============================================================================
def _attention_baseline(
    attentions_tuple,        # outputs.attentions: tuple of length response_length
    response_length: int,
    input_len: int,
    img_start_idx_full: int,
    img_end_idx_full: int,
    vision_shape: tuple,
):
    """Compute per-response-token attention-baseline map (same H×W as TAM),
    plus per-token scalars and peak metadata. Last-layer attention, averaged
    over heads, sliced to image-patch key positions, normalized to [0,1].

    Returns (maps_list[R], scalars_list[R], peak_list[R], failure_reason_or_None).
    Maps that fail (e.g. attentions missing for that step) get a zero map."""
    import torch  # noqa: F401
    Hp, Wp = vision_shape
    n_patches = Hp * Wp
    img_slice = slice(img_start_idx_full + 1, img_end_idx_full)
    expected_n = img_end_idx_full - (img_start_idx_full + 1)

    maps_list = []
    scalars_list = []
    peak_list = []
    failure = None

    if expected_n != n_patches:
        failure = (
            f"image_patch_count_mismatch: expected_n={expected_n} "
            f"vision_shape={vision_shape} (Hp*Wp={n_patches})"
        )

    # Early-fail check: if attentions are None (FA2 / SDPA silently dropped
    # them), abort cleanly with a clear remediation message.
    if (len(attentions_tuple) > 0
            and attentions_tuple[0] is not None
            and len(attentions_tuple[0]) > 0
            and attentions_tuple[0][-1] is None):
        failure = (
            "outputs.attentions[t][-1] is None — current attn_implementation "
            "does not support output_attentions=True. Set MLLMOPD_ATTN_IMPL=eager "
            "and re-run; FA2 / SDPA silently drop attention tensors."
        )
        zero_maps = [np.zeros(vision_shape, dtype=np.float32)] * response_length
        zero_scalars = [tam_scalars(m) for m in zero_maps]
        zero_peaks = [_tam_peak_meta(m) for m in zero_maps]
        return zero_maps, zero_scalars, zero_peaks, failure

    for t in range(response_length):
        try:
            per_layer = attentions_tuple[t]
            last_layer = per_layer[-1]                  # (1, H, q, k)
            avg_heads = last_layer.float().mean(dim=1)  # (1, q, k)
            if t == 0:
                att_q = avg_heads[0, -1, :]             # query = last prompt position
            else:
                att_q = avg_heads[0, 0, :]              # query = new token
            img_attn = att_q[img_slice].detach().cpu().numpy()
            if img_attn.size != n_patches:
                # Pad / truncate defensively to vision_shape
                vec = np.zeros(n_patches, dtype=np.float32)
                k = min(img_attn.size, n_patches)
                vec[:k] = img_attn[:k]
                img_attn = vec
            m = img_attn.reshape(Hp, Wp).astype(np.float32)
            mx = float(m.max())
            if mx < 1e-12:
                m_norm = m
            else:
                m_norm = m / mx
        except Exception as e:  # noqa: BLE001
            if failure is None:
                failure = f"step {t}: {type(e).__name__}: {e!s:.120}"
            m_norm = np.zeros(vision_shape, dtype=np.float32)
        maps_list.append(m_norm)
        scalars_list.append(tam_scalars(m_norm))
        peak_list.append(_tam_peak_meta(m_norm))
    return maps_list, scalars_list, peak_list, failure


# ============================================================================
# Per-token logp / entropy / margin from generate scores
# ============================================================================
def _per_token_teacher_stats(scores, sequences, input_len: int) -> tuple[list, list, list]:
    import torch
    lp_full = []
    teacher_entropy = []
    teacher_top1_margin = []
    for t, score_t in enumerate(scores):
        log_probs = torch.log_softmax(score_t, dim=-1)[0]  # (V,)
        token_id = sequences[input_len + t].item()
        lp_full.append(float(log_probs[token_id]))
        probs = torch.softmax(score_t.float(), dim=-1)[0]
        H = float(-(probs * torch.log(probs + 1e-12)).sum())
        teacher_entropy.append(H)
        top2_logp = log_probs.topk(2).values
        teacher_top1_margin.append(float(top2_logp[0] - top2_logp[1]))
    return lp_full, teacher_entropy, teacher_top1_margin


# ============================================================================
# Token-selection rule for tam_maps_subset (Step 0)
# ============================================================================
def _select_tokens_for_overlays(
    response_length: int,
    scalars_per_token: list[dict],
    is_blankness: list[bool],
    is_answer: list[bool],
    is_think: list[bool],
) -> tuple[list[int], list[str], list[int], list[float], list[list[str]]]:
    """Step 0 stratified selection:
      5 top-tam_mass_top20  (highest TAM concentration)
      5 first response tokens (often "<think>" start, "Yes"/"No")
      5 blankness tokens (if any; pad with answer tokens if fewer)
      5 answer tokens (within <answer>...</answer> span)

    Step 1 will swap strata for top-|vd| / top-|adv|; Step 0 has no vd/adv.

    Returns lists of (idx, stratum, rank, score, deduped_from_strata)."""
    top_mass_ranked = sorted(
        range(response_length),
        key=lambda i: scalars_per_token[i].get("tam_mass_top20", 0.0),
        reverse=True,
    )
    first5 = list(range(min(5, response_length)))
    blank_idxs  = [i for i in range(response_length) if is_blankness[i]]
    answer_idxs = [i for i in range(response_length) if is_answer[i]]

    bins = {
        "top_tam_mass":   top_mass_ranked[:5],
        "first_response": first5,
        "blankness":      blank_idxs[:5],
        "answer":         answer_idxs[:5],
    }

    seen = {}  # idx -> (primary_stratum, primary_rank, primary_score, [other_strata])
    for stratum, idxs in bins.items():
        for rank, idx in enumerate(idxs):
            if idx in seen:
                seen[idx][3].append(stratum)
                continue
            score = (scalars_per_token[idx].get("tam_mass_top20", 0.0)
                     if stratum == "top_tam_mass" else 1.0)
            seen[idx] = [stratum, rank, score, []]

    # Preserve insertion order; cap K=20
    selected = list(seen.items())[:20]
    token_indices       = [int(i) for i, _ in selected]
    selection_strata    = [v[0] for _, v in selected]
    selection_rank      = [int(v[1]) for _, v in selected]
    selection_score     = [float(v[2]) for _, v in selected]
    deduped_from_strata = [list(v[3]) for _, v in selected]
    return (token_indices, selection_strata, selection_rank,
            selection_score, deduped_from_strata)


# ============================================================================
# Main per-probe processor
# ============================================================================
def process_probe(model, processor, probe: dict, args, out_dir: Path) -> dict:
    import torch
    from PIL import Image

    image_path = Path(probe["image"])
    if not image_path.is_absolute():
        image_path = (Path(args.image_root) / probe["image"]).resolve() if args.image_root else image_path
    image = Image.open(image_path).convert("RGB")

    messages = _build_messages(probe["question"], image, args.system_prompt)
    chat = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = processor(
        text=[chat],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]
    input_ids_full = inputs["input_ids"][0].tolist()

    t0 = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
            output_hidden_states=True,
            output_scores=True,
            output_attentions=True,   # v0.1.2: needed for attention baseline
            return_dict_in_generate=True,
        )
    dt_gen = time.time() - t0

    sequences = outputs.sequences[0]
    response_ids = sequences[input_len:].cpu().tolist()
    # Trim at EOS
    eos_id = processor.tokenizer.eos_token_id
    response_length = len(response_ids)
    if eos_id is not None and eos_id in response_ids:
        response_length = response_ids.index(eos_id) + 1
    response_ids = response_ids[:response_length]

    # Per-token teacher signals (operate on the un-trimmed scores tuple; we
    # also have to trim those to match response_length).
    scores_trimmed = tuple(outputs.scores[:response_length])
    lp_full, teacher_entropy_full, teacher_top1_margin_full = _per_token_teacher_stats(
        scores_trimmed, sequences, input_len,
    )

    # TAM input: logits per round computed via lm_head(feats[-1])
    t1 = time.time()
    logit_list = []
    for r, feats in enumerate(outputs.hidden_states[:response_length]):
        # feats is tuple of (num_layers+1) per-layer hidden states; use [-1] = last layer
        last_h = feats[-1]
        with torch.inference_mode():
            logit_list.append(model.lm_head(last_h))
    dt_logit = time.time() - t1

    # vision_shape (Qwen2.5-VL 2x2 packing)
    grid_thw = inputs["image_grid_thw"][0].tolist()
    vision_shape = (grid_thw[1] // 2, grid_thw[2] // 2)
    n_patches = int(vision_shape[0] * vision_shape[1])

    # Tokens for full sequence (TAM works on this)
    tokens_full_list = sequences[: input_len + response_length].cpu().tolist()

    # Run TAM for all response tokens; collect prompt-token TAM during round-0
    t2 = time.time()
    img_scores_list: list = []
    prompt_maps: list = []
    response_maps: list = []
    tam_failure_reason = None
    try:
        for i in range(response_length):
            m = TAM(
                tokens=tokens_full_list,
                vision_shape=vision_shape,
                logit_list=logit_list,
                special_ids=QWEN_VL_SPECIAL_IDS,
                processor=processor,
                target_token=i,
                img_scores_list=img_scores_list,
                out_prompt_maps=prompt_maps if i == 0 else None,
            )
            response_maps.append(m)
    except Exception as e:  # noqa: BLE001
        tam_failure_reason = f"runtime:{type(e).__name__}:{e!s:.120}"
        print(f"!! TAM runtime failure on probe {probe['id']}: {tam_failure_reason}", file=sys.stderr)
    dt_tam = time.time() - t2

    tam_valid = (tam_failure_reason is None and len(response_maps) == response_length)

    # Scalars per token + TAM peak metadata (v0.1.2)
    response_scalars = [tam_scalars(m) for m in response_maps]
    prompt_scalars   = [tam_scalars(m) for m in prompt_maps]
    response_peak    = [_tam_peak_meta(m) for m in response_maps]

    # Detokenize for labels + display
    tokenizer = processor.tokenizer
    response_tokens = tokenizer.convert_ids_to_tokens(response_ids)
    response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
    labels = _label_response_tokens(response_ids, tokenizer)

    # v0.1.2: token category classification (POS + regex). Depends on labels
    # (for is_answer_token gating of answer_token category).
    classification = _classify_tokens_v012(
        response_ids, tokenizer, labels["is_answer_token"],
    )

    # v0.1.2: attention baseline (last-layer-avg-heads on image-patch positions).
    img_start_idx_full = input_ids_full.index(QWEN_VL_SPECIAL_IDS["img_id"][0]) if QWEN_VL_SPECIAL_IDS["img_id"][0] in input_ids_full else -1
    img_end_idx_full = input_ids_full.index(QWEN_VL_SPECIAL_IDS["img_id"][1]) if QWEN_VL_SPECIAL_IDS["img_id"][1] in input_ids_full else -1
    t3 = time.time()
    if img_start_idx_full >= 0 and img_end_idx_full >= 0 and outputs.attentions is not None:
        attn_maps, attn_scalars, attn_peak, attn_failure = _attention_baseline(
            outputs.attentions[:response_length],
            response_length,
            input_len,
            img_start_idx_full,
            img_end_idx_full,
            vision_shape,
        )
    else:
        attn_maps = [np.zeros(vision_shape, dtype=np.float32)] * response_length
        attn_scalars = [tam_scalars(m) for m in attn_maps]
        attn_peak = [_tam_peak_meta(m) for m in attn_maps]
        attn_failure = "vision_span_or_attentions_missing"
    dt_attn = time.time() - t3
    attn_baseline_valid = (attn_failure is None)

    # Prompt-side detokenization (TAM's prompt range = between vision_end and im_end)
    prompt_segments = _compute_prompt_segments(input_ids_full, QWEN_VL_SPECIAL_IDS)
    question_span = prompt_segments.get("question")
    if question_span:
        tokens_prompt_ids = input_ids_full[question_span[0] : question_span[1]]
        tokens_prompt = tokenizer.convert_ids_to_tokens(tokens_prompt_ids)
    else:
        tokens_prompt_ids = []
        tokens_prompt = []

    # Align prompt_maps (TAM iterates len(prompt)+1) with tokens_prompt (len P)
    # Upstream's loop is `for t in range(len(prompt)+1)`, so len(prompt_maps) = P+1.
    # The extra map is the "prompt-end placeholder" — drop for storage.
    if len(prompt_maps) > len(tokens_prompt):
        prompt_maps = prompt_maps[: len(tokens_prompt)]
        prompt_scalars = prompt_scalars[: len(tokens_prompt)]

    # tam_maps_subset selection (Step 0 strata)
    (subset_idx, subset_strata, subset_rank, subset_score, subset_dedup) = _select_tokens_for_overlays(
        response_length,
        response_scalars,
        labels["is_blankness_token"],
        labels["is_answer_token"],
        labels["is_think_token"],
    )
    subset_maps_b64 = [_b64_uint8_map(response_maps[i]) for i in subset_idx]

    # Save PNG overlays for selected tokens
    overlays_dir = out_dir / "overlays" / probe["id"].replace("/", "_")
    for i, stratum in zip(subset_idx, subset_strata):
        tok_str = response_tokens[i] if i < len(response_tokens) else f"tok{i}"
        tok_safe = re.sub(r"[^A-Za-z0-9_-]", "_", tok_str)[:24]
        out_path = overlays_dir / f"resp_{i:03d}_{stratum}_{tok_safe}.png"
        _overlay_png(image, response_maps[i], out_path)

    # Save overlays for question-prompt tokens too (caption-probe sanity)
    for i, tok_id in enumerate(tokens_prompt_ids[: len(prompt_maps)]):
        tok_str = tokens_prompt[i] if i < len(tokens_prompt) else f"ptok{i}"
        tok_safe = re.sub(r"[^A-Za-z0-9_-]", "_", tok_str)[:24]
        out_path = overlays_dir / f"prompt_{i:03d}_{tok_safe}.png"
        _overlay_png(image, prompt_maps[i], out_path)

    # Assemble row (v0.1.2 Step 0 subset)
    response_hash = _sha256_hex(json.dumps(response_ids).encode())
    row = {
        # identity
        "id":           probe["id"],
        "benchmark":    probe["benchmark"],
        "split_tag":    probe.get("split_tag", "step0_sanity"),
        "image_path":   str(image_path),
        "image_sha256": _image_sha256(image_path),
        "question":     probe["question"],
        "answer":       probe.get("answer"),
        "probe_note":   probe.get("probe_note"),

        # mode / checkpoints
        "response_source": "teacher_greedy",
        "teacher_ckpt":    args.model,
        "student_ckpt":    None,  # Step 0: no student

        # run metadata (v0.1.2)
        "tokenizer_name_or_path":  args.model,
        "tokenizer_vocab_hash":    _tokenizer_vocab_hash(processor),
        "processor_name_or_path":  args.model,
        "tam_preproc_version":     "v0.1.2",
        "code_commit_run":         os.environ.get("MLLMOPD_CODE_COMMIT", "unknown"),
        "code_commit_analyzed":    None,
        "pos_tagger":              "spacy/en_core_web_sm" if _SPACY_AVAILABLE else "none",
        "pos_tagger_version":      _SPACY_VERSION,
        "pos_tagger_load_error":   _SPACY_LOAD_ERROR,
        "token_category_source":   "regex+spacy_align:v0.1.3",
        "attention_baseline_method": "last_layer_avg_heads:v0.1.2",
        "attention_baseline_layers": [-1],
        "attention_baseline_heads":  "all",
        "tam_score_def": {
            "mass_top_K": [10, 20, 40],
            "entropy": (
                "H = -Σ softmax(act_i)·log softmax(act_i) over patches "
                "(post rank-Gaussian); tam_entropy_norm = H / log(N_patches)"
            ),
            "map_pipeline": (
                "lm_head(last_hidden)[:, img_idx, cls_id] → ECI subtract → "
                "clip ≥0 → rank_gaussian_filter(3) → normalize 0-1"
            ),
            "peak": (
                "tam_peak_patch_idx = argmax(map.flatten()); "
                "tam_peak_xy = (col/(W-1), row/(H-1)) in [0,1]; "
                "tam_center_of_mass_xy = Σ p_i·(x_i, y_i) / Σ p_i"
            ),
        },

        # response
        "response_text":   response_text,
        "response_ids":    response_ids,
        "response_length": response_length,
        "response_hash":   response_hash,
        "tokens":          response_tokens,
        "token_idx":       list(range(response_length)),
        "token_uid": [
            f"{probe['id']}:teacher_greedy:{response_hash}:{i}"
            for i in range(response_length)
        ],

        # per-token teacher signals (response)
        "lp_full":     lp_full,
        "lp_blank":    None,   # Step 0: no blank forward
        "vd":          None,   # depends on lp_blank
        "tam_mass_top10": [s["tam_mass_top10"] for s in response_scalars],
        "tam_mass_top20": [s["tam_mass_top20"] for s in response_scalars],
        "tam_mass_top40": [s["tam_mass_top40"] for s in response_scalars],
        "tam_entropy":              [s["tam_entropy"] for s in response_scalars],
        "tam_entropy_norm":         [s["tam_entropy_norm"] for s in response_scalars],
        "tam_effective_patch_frac": [s["tam_effective_patch_frac"] for s in response_scalars],

        # v0.1.2: TAM peak metadata per response token
        "tam_peak_patch_idx":     [p["tam_peak_patch_idx"] for p in response_peak],
        "tam_peak_xy":            [p["tam_peak_xy"] for p in response_peak],
        "tam_center_of_mass_xy":  [p["tam_center_of_mass_xy"] for p in response_peak],

        # v0.1.2: attention baseline (same scalar family as TAM)
        "attention_baseline_mass_top10":   [s["tam_mass_top10"] for s in attn_scalars],
        "attention_baseline_mass_top20":   [s["tam_mass_top20"] for s in attn_scalars],
        "attention_baseline_mass_top40":   [s["tam_mass_top40"] for s in attn_scalars],
        "attention_baseline_entropy_norm": [s["tam_entropy_norm"] for s in attn_scalars],

        # confound disentanglers
        "teacher_entropy_full":     teacher_entropy_full,
        "teacher_top1_margin_full": teacher_top1_margin_full,

        # prompt-token TAM (v0.1.2: scope locked to question_only per schema)
        "prompt_tam_scope":         "question_only",
        "prompt_length":            len(tokens_prompt),
        "tokens_prompt":            tokens_prompt,
        "tam_mass_top20_prompt":    [s["tam_mass_top20"] for s in prompt_scalars],
        "tam_entropy_norm_prompt":  [s["tam_entropy_norm"] for s in prompt_scalars],
        "prompt_segments":          prompt_segments,
        "prompt_full_length":       len(input_ids_full),

        # student signals — Step 0: none
        "student_lp":      None,
        "student_entropy": None,
        "adv":             None,
        "quad":            None,

        # token-text labels (v0.1.1) + token category (v0.1.2)
        **labels,
        **classification,    # pos_tag, word_idx, word_text, token_category,
                              # is_template_token, is_special_token, is_pronoun,
                              # is_meta_cot_token

        # image metadata
        "image_grid_thw":    grid_thw,
        "vision_shape":      list(vision_shape),
        "n_patches":         n_patches,
        "map_h":             int(vision_shape[0]),
        "map_w":             int(vision_shape[1]),
        "patch_index_order": "row_major_top_left",

        # QC (v0.1.1 + v0.1.2 attention baseline QC)
        "tam_valid":          bool(tam_valid),
        "tam_failure_reason": tam_failure_reason,
        "attn_baseline_valid":          bool(attn_baseline_valid),
        "attn_baseline_failure_reason": attn_failure,

        # tam_maps_subset (Step 0 strata + v0.1.2 attention maps for direct comparison)
        "tam_maps_subset": {
            "token_indices":       subset_idx,
            "selection_strata":    subset_strata,
            "selection_rank":      subset_rank,
            "selection_score":     subset_score,
            "deduped_from_strata": subset_dedup,
            "maps_uint8_b64":      subset_maps_b64,
            "attention_maps_uint8_b64": [_b64_uint8_map(attn_maps[i]) for i in subset_idx],
        },

        # timings (not in schema; useful for cost validation)
        "_timings": {
            "generate_s":      dt_gen,
            "logit_compute_s": dt_logit,
            "tam_compute_s":   dt_tam,
            "attn_compute_s":  dt_attn,
        },
    }
    return row


# ============================================================================
# CLI
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probes", type=Path,
                    default=Path("data/audit/tam_probes.jsonl"),
                    help="Probe JSONL (default: data/audit/tam_probes.jsonl)")
    ap.add_argument("--model", default=os.environ.get("MMR1_3B_SFT_CKPT", "MMR1/MMR1-3B-SFT"),
                    help="HF model id or local ckpt path (default: $MMR1_3B_SFT_CKPT or MMR1/MMR1-3B-SFT)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory (e.g. $MLLMOPD_RUNS/tam_sanity_$(date +%%Y%%m%%d-%%H%%M%%S))")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="Generation cap. POPE answers are short; keep small for fast sanity (default 256).")
    ap.add_argument("--system-prompt", default=MMR1_SYSTEM_PROMPT,
                    help=(
                        "MMR1 system prompt text prepended to the user turn "
                        "(NOT a separate system role; matches "
                        "run_audit_pass.py:_build_messages). Pass empty "
                        "string for plain Qwen2.5-VL behavior."
                    ))
    ap.add_argument("--image-root", default=".",
                    help="Resolve relative image paths against this root "
                         "(default cwd).")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 = all probes (default).")
    args = ap.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "tam_sanity.jsonl"
    summary   = out_dir / "summary.txt"

    print(f">>> loading {args.model}", file=sys.stderr)
    processor, model = _build_model(args.model)
    print(f">>> loaded; model device={model.device} dtype={model.dtype}", file=sys.stderr)

    probes: list[dict] = []
    with args.probes.open() as f:
        for line in f:
            line = line.strip()
            if line:
                probes.append(json.loads(line))

    if args.limit:
        probes = probes[: args.limit]

    print(f">>> {len(probes)} probes; out_dir={out_dir}", file=sys.stderr)

    with out_jsonl.open("w") as fout, summary.open("w") as fsum:
        fsum.write(f"# TAM sanity ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
        fsum.write(f"model = {args.model}\n\n")

        for k, probe in enumerate(probes):
            print(f"--- [{k+1}/{len(probes)}] probe {probe['id']} ---", file=sys.stderr)
            try:
                row = process_probe(model, processor, probe, args, out_dir)
            except Exception as e:  # noqa: BLE001
                print(f"!! probe {probe['id']} FAILED: {e!r}", file=sys.stderr)
                row = {
                    "id": probe["id"],
                    "benchmark": probe["benchmark"],
                    "tam_valid": False,
                    "tam_failure_reason": f"orchestrator:{type(e).__name__}:{e!s:.200}",
                }

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

            # Summary: top-5 tokens by tam_mass_top20
            fsum.write(f"\n## {probe['id']}\n")
            fsum.write(f"  question:      {probe['question']!r}\n")
            fsum.write(f"  response_text: {row.get('response_text', '')!r}\n")
            fsum.write(f"  tam_valid:     {row.get('tam_valid')}\n")
            if row.get("tam_valid"):
                tm = row["tam_mass_top20"]
                ranked = sorted(range(len(tm)), key=lambda i: tm[i], reverse=True)[:5]
                fsum.write("  top-5 response tokens by tam_mass_top20:\n")
                for r_idx in ranked:
                    tok = row["tokens"][r_idx]
                    fsum.write(f"    [{r_idx:3d}] {tok!r:24s} mass20={tm[r_idx]:.3f}  H_norm={row['tam_entropy_norm'][r_idx]:.3f}\n")
                fsum.write(f"  expected region (from probe note): {probe.get('expected_object_region', 'n/a')!r}\n")
            fsum.flush()

    print(f">>> wrote {out_jsonl}", file=sys.stderr)
    print(f">>> summary: {summary}", file=sys.stderr)
    print(f">>> overlays under {out_dir}/overlays/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
