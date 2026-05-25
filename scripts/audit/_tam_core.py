"""Vendored slim copy of TAM core algorithm (xmed-lab/TAM, ICCV 2025).

Upstream: https://github.com/xmed-lab/TAM/blob/main/tam.py (commit on 2026-05-25)
Paper:    Li et al., "Token Activation Map to Visually Explain Multimodal LLMs",
          ICCV 2025 (Oral). arXiv:2506.23270.

What we kept:
  - rank_guassian_filter(img, kernel_size)   — robust spatial denoiser
  - least_squares(map1, map2)                — Estimated Causal Inference scaling
  - id2idx(inp_id, target_id, return_last)   — token-position locator
  - TAM(...)                                 — recursive entry, returns float [0,1] map
  - multimodal_process_image(...)            — single-image branch only

What we removed:
  - generate_latex / compile_latex_to_jpg / vis_text  (LaTeX/xelatex/fitz deps)
  - Video and multi-image branches of multimodal_process
  - cv2-based PNG overlay (moved to caller — _tam_overlay.tam_overlay)

What we added:
  - tam_scalars(normalized_map)              — per v0.1.1 schema:
      tam_mass_top{10,20,40}, tam_entropy, tam_entropy_norm,
      tam_effective_patch_frac

Original TAM was demoed on Qwen2-VL-2B-Instruct. The token-ID protocol is
shared with Qwen2.5-VL (same vocab/special tokens), so we reuse:
  img_id = [151652, 151653]                                  # <|vision_start|>, <|vision_end|>
  prompt_id = [151653, [151645, 198, 151644, 77091]]          # <|vision_end|> → <|im_end|>\\n<|im_start|>assistant
  answer_id = [[198, 151644, 77091, 198], -1]                 # \\n<|im_start|>assistant\\n → end

License: TAM is released under the upstream repo's license. This vendored
copy is for research/calibration use inside mllmopd. Attribution stays in
this docstring.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize_scalar

_EPS = 1e-12


# ============================================================================
# rank-Gaussian filter (upstream rank_guassian_filter, name kept for parity)
# ============================================================================
def rank_guassian_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Rank-based Gaussian-weighted filter. sigma = coefficient of variation
    (std / mean) of the sorted local window — invariant to global scale,
    robust to outliers."""
    filtered = np.zeros_like(img)
    pad = kernel_size // 2
    padded = np.pad(img, pad, mode="reflect")
    ax = np.arange(kernel_size ** 2) - kernel_size ** 2 // 2

    for i in range(pad, img.shape[0] + pad):
        for j in range(pad, img.shape[1] + pad):
            window = padded[i - pad:i + pad + 1, j - pad:j + pad + 1]
            sorted_win = np.sort(window.flatten())
            mean = sorted_win.mean()
            if mean > 0:
                sigma = sorted_win.std() / mean
                kernel = np.exp(-(ax ** 2) / (2 * sigma ** 2 + _EPS))
                kernel = kernel / np.sum(kernel)
                value = (sorted_win * kernel).sum()
            else:
                value = 0
            filtered[i - pad, j - pad] = value
    return filtered


def least_squares(map1: np.ndarray, map2: np.ndarray) -> float:
    """Optimal scalar `x` minimizing ||map1 − x·map2||²  via scipy."""
    def diff(x, map1, map2):
        return np.sum((map1 - map2 * x) ** 2)
    result = minimize_scalar(diff, args=(map1, map2))
    return float(result.x)


def id2idx(inp_id, target_id, return_last: bool = False) -> int:
    """Locate the index of `target_id` (or a token sequence) in `inp_id`."""
    if isinstance(target_id, list):
        n = len(target_id)
        indexes = [i for i in range(len(inp_id) - n + 1)
                   if inp_id[i:i + n] == target_id]
        if not indexes:
            return -1
        idx = indexes[-1]
        if return_last:
            idx += n - 1
        return idx
    try:
        return inp_id.index(target_id)
    except ValueError:
        return -1


# ============================================================================
# Per-token scalar computation (v0.1.1 schema)
# ============================================================================
def tam_scalars(normalized_map_2d: np.ndarray) -> dict:
    """Compute per-token scalars from a normalized [0,1] activation map.

    Args:
        normalized_map_2d: post-ECI, post-rank-Gaussian, normalized to [0,1].
                           Shape (H, W) where H*W = N_patches.

    Returns:
        dict with keys per v0.1.1 schema:
          tam_mass_top10  = sum(top-10% patches) / sum(all)
          tam_mass_top20  = sum(top-20%) / sum(all)              [primary]
          tam_mass_top40  = sum(top-40%) / sum(all)
          tam_entropy     = −Σ softmax(act)·log softmax(act)
          tam_entropy_norm = tam_entropy / log(N_patches)        [∈ [0,1]]
          tam_effective_patch_frac = exp(tam_entropy) / N_patches
    """
    flat = normalized_map_2d.flatten().astype(np.float64)
    n_patches = flat.size
    total = flat.sum()
    out = {
        "tam_mass_top10": 0.1,
        "tam_mass_top20": 0.2,
        "tam_mass_top40": 0.4,
        "tam_entropy": 0.0,
        "tam_entropy_norm": 0.0,
        "tam_effective_patch_frac": 1.0 / n_patches,
    }
    if total < _EPS:
        # Degenerate map (all zeros). Return uniform-equivalent defaults so the
        # caller can flag tam_valid=false via QC, but downstream maths don't NaN.
        return out

    sorted_desc = np.sort(flat)[::-1]
    # Top-K% mass concentration ratios
    for pct, key in [(10, "tam_mass_top10"),
                     (20, "tam_mass_top20"),
                     (40, "tam_mass_top40")]:
        k = max(1, int(round(n_patches * pct / 100)))
        out[key] = float(sorted_desc[:k].sum() / total)

    # Entropy in softmax(activation) space
    # Use stable softmax: subtract max
    a = flat - flat.max()
    p = np.exp(a) / (np.exp(a).sum() + _EPS)
    p = np.clip(p, _EPS, 1.0)
    H = float(-(p * np.log(p)).sum())
    out["tam_entropy"] = H
    out["tam_entropy_norm"] = H / float(np.log(n_patches))
    out["tam_effective_patch_frac"] = float(np.exp(H) / n_patches)
    return out


# ============================================================================
# multimodal_process — single-image only, returns float map [0,1]
# ============================================================================
def multimodal_process_image(
    vision_shape,
    img_scores: np.ndarray,
    txt_scores: np.ndarray,
) -> np.ndarray:
    """Single-image branch of upstream's multimodal_process. Returns the
    post-rank-Gaussian normalized [0,1] activation map (shape vision_shape).

    Removes upstream's overlay/text-vis logic — caller renders PNG separately
    via _tam_overlay.tam_overlay() on the returned map.

    Args:
        vision_shape: (H_patch, W_patch) after 2x2 packing — TAM's vision_shape
        img_scores:   1-D length H_patch*W_patch, per-patch activations
                       (pre-filter)
        txt_scores:   text-token activations including self — used for joint
                       normalization with img_scores
    """
    # Joint normalize img + text (upstream behavior — keeps the two modalities
    # on the same scale before isolating img_scores).
    txt_scores = txt_scores[:-1]  # drop self-score
    all_scores = np.concatenate([img_scores, txt_scores], 0)
    rng = all_scores.max() - all_scores.min()
    if rng < _EPS:
        # Degenerate joint distribution; return zero map for QC failure.
        return np.zeros(vision_shape, dtype=np.float32)
    all_scores = (all_scores - all_scores.min()) / rng
    img_scores = all_scores[:len(img_scores)]

    t_h, t_w = vision_shape
    # Apply the rank-Gaussian filter (upstream uses kernel_size=3)
    filt = rank_guassian_filter(img_scores.reshape(t_h, t_w), kernel_size=3)
    # Normalize the post-filter map to [0,1] to match v0.1.1 schema's stated
    # pipeline ("normalize 0-1"). The downstream uint8 conversion (for PNG
    # overlay) is done in the overlay helper, not here.
    fmax = filt.max()
    if fmax < _EPS:
        return np.zeros(vision_shape, dtype=np.float32)
    return (filt / fmax).astype(np.float32)


# ============================================================================
# TAM — recursive entry (preserves upstream algorithm)
# ============================================================================
def TAM(
    tokens,
    vision_shape,
    logit_list,
    special_ids,
    processor,
    target_token,
    img_scores_list,
    out_prompt_maps: list | None = None,
) -> np.ndarray:
    """Generate the Token Activation Map for one (sample, target_token) pair,
    applying Estimated Causal Inference using `img_scores_list` (accumulated
    by the caller across rounds).

    Args:
        tokens:           list[int] full token sequence (prompt + generated)
        vision_shape:     (H_patch, W_patch) tuple from
                          (image_grid_thw[0,1]//2, image_grid_thw[0,2]//2)
        logit_list:       list of per-round (1, seq_or_1, V) logits tensors:
                          logits = [model.lm_head(feats[-1])
                                    for feats in outputs.hidden_states]
        special_ids:      {'img_id': [...], 'prompt_id': [...], 'answer_id': [...]}
                          For Qwen2.5-VL: see module docstring.
        processor:        HF AutoProcessor (used for tokenizer.tokenize)
        target_token:     For non-first round: int round_idx.
                          For round 0: tuple (round_idx=0, prompt_token_idx).
        img_scores_list:  Mutable list, accumulates per-round img_scores for
                          ECI. Caller passes [] for the first round of each
                          sample.
        out_prompt_maps:  Optional mutable list. When provided AND the current
                          call is part of the round-0 prompt recursion (i.e.
                          `target_token` is a tuple), the returned
                          normalized_map_2d is appended (in prompt-token
                          order, indices 0..P). Used by Step 0/1 to capture
                          per-prompt-token TAM without re-running TAM.

    Returns:
        normalized_map_2d: float32 array of shape vision_shape, ∈ [0,1].
                           tam_scalars() consumes this directly.
    """
    img_id = special_ids["img_id"]
    prompt_id = special_ids["prompt_id"]
    answer_id = special_ids["answer_id"]

    if len(img_id) == 1:
        img_idx = (np.array(tokens) == img_id[0]).nonzero()[0]
    else:
        img_idx = [id2idx(tokens, img_id[0], True), id2idx(tokens, img_id[1])]

    prompt_idx = [id2idx(tokens, prompt_id[0], True),
                  id2idx(tokens, prompt_id[1])]
    answer_idx = [id2idx(tokens, answer_id[0], True),
                  id2idx(tokens, answer_id[1])]

    # Decode prompt + answer token sequences for ECI repetition match
    prompt = processor.tokenizer.tokenize(
        processor.batch_decode(
            [tokens[prompt_idx[0] + 1: prompt_idx[1]]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
    )
    answer = processor.tokenizer.tokenize(
        processor.batch_decode(
            [tokens[answer_idx[0] + 1:]],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
    )
    txt_all = prompt + answer

    round_idx = -1
    this_token_idx = 0

    if isinstance(target_token, int):
        round_idx = target_token
        this_token_idx = -1
        vis_token_idx = len(prompt) + target_token
    else:
        round_idx, prompt_token_idx = target_token
        this_token_idx = prompt_idx[0] + prompt_token_idx + 1
        vis_token_idx = prompt_token_idx

    # Round 0: recurse through all prompt tokens to build the ECI history
    if round_idx == 0 and isinstance(target_token, int):
        first_ori = None
        for t in range(len(prompt) + 1):
            m = TAM(tokens, vision_shape, logit_list, special_ids, processor,
                    [0, t], img_scores_list, out_prompt_maps=out_prompt_maps)
            if t == 0:
                first_ori = m
        return first_ori if first_ori is not None else np.zeros(vision_shape, dtype=np.float32)

    # Class-id selection — upstream's logic for the four cases
    if round_idx == 0:
        if prompt_token_idx == len(prompt):
            this_token_idx = logit_list[0].shape[1] - 1
            cls_id = tokens[this_token_idx]
        elif prompt_token_idx == 0:
            cls_id = int(logit_list[0][0, prompt_idx[0] + 1].argmax(0))
        else:
            cls_id = tokens[this_token_idx]
    else:
        cls_id = tokens[answer_idx[0] + round_idx + 1]

    # Concatenate cls_id logit across all rounds processed so far
    scores = torch.cat(
        [logit_list[r][0, :, cls_id] for r in range(round_idx + 1)],
        dim=-1,
    ).clip(min=0)
    scores = scores.detach().cpu().float().numpy()

    prompt_scores = scores[prompt_idx[0] + 1: prompt_idx[1]]
    last_prompt = scores[logit_list[0].shape[1] - 1: logit_list[0].shape[1]]
    answer_scores = scores[answer_idx[0] + 1:]
    txt_scores = np.concatenate([prompt_scores, last_prompt, answer_scores], -1)

    if isinstance(img_idx, list):
        img_scores = scores[img_idx[0] + 1: img_idx[1]]
    else:
        img_scores = scores[img_idx]

    img_scores_list.append(img_scores.copy())

    # ECI: subtract weighted interference from non-repeating prior tokens
    if len(img_scores_list) > 1 and vis_token_idx < len(txt_all):
        non_repeat_idx = [
            i for i in range(vis_token_idx)
            if i < len(txt_all) and txt_all[i] != txt_all[vis_token_idx]
        ]
        if non_repeat_idx:
            txt_scores_ = txt_scores[non_repeat_idx]
            img_scores_list_ = [img_scores_list[k] for k in non_repeat_idx]
            w = txt_scores_
            w = w / (w.sum() + _EPS)
            interf_img_scores = (
                np.stack(img_scores_list_, 0) * w.reshape(-1, 1)
            ).sum(0)
            scaled = least_squares(img_scores, interf_img_scores)
            img_scores = np.clip(img_scores - interf_img_scores * scaled, a_min=0.0, a_max=None)

    normalized_map = multimodal_process_image(vision_shape, img_scores, txt_scores)

    # Expose per-prompt-token maps to the caller during round-0 recursion.
    if out_prompt_maps is not None and not isinstance(target_token, int):
        out_prompt_maps.append(normalized_map)

    return normalized_map
