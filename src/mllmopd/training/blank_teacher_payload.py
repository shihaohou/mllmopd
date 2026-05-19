"""Helper: produce base64-encoded blank-image variants of a Sample's
visual content, ready to drop into a teacher request payload.

This is the T1 BlankTeacher intervention. The student rollout always
sees the *full* image (untouched); only the teacher's scoring prefix
gets the white-blank substitution. By replacing `payload["image_data"]`
with same-size white blanks we make the teacher produce
`logp(token | prefix, blank_image)` — identical in form to
`logp(token | prefix, full_image)` but with the visual evidence
zeroed out. The pointwise difference is the per-token visual
dependency we measured in the H2 audit.

Used by `dual_teacher_get_reward.py`. Keeps `mllm_corruptions.blank_image`'s
default color (white `(255,255,255)`) so the training intervention is
*byte-identical* to the audit/H2 blank canvas.
"""

from __future__ import annotations

from PIL import Image

_BLANK_CACHE: dict[tuple[int, int], Image.Image] = {}


def _get_blank(size: tuple[int, int]) -> Image.Image:
    """Return a cached same-size white PIL.Image. Matches
    `mllm_corruptions.blank_image()` default color (255,255,255) so the
    training intervention is identical to the audit blank canvas."""
    if size not in _BLANK_CACHE:
        _BLANK_CACHE[size] = Image.new("RGB", size, (255, 255, 255))
    return _BLANK_CACHE[size]


def make_blank_image_data(sample) -> list[str]:
    """Encode same-size white blanks for each image in `sample` using
    Uni-OPD's own `encode_image_for_rollout_engine` (so the blob format
    matches what the teacher server expects).

    Returns the list of base64 PNG strings ready to assign to
    `payload["image_data"]`. Returns `[]` if the sample has no images.
    """
    # Import lazily — at module-load time we can't assume Uni-OPD's
    # `miles` package is on PYTHONPATH (the smoke tests run outside the
    # training launcher).
    from miles.utils.processing_utils import encode_image_for_rollout_engine

    if not getattr(sample, "multimodal_inputs", None):
        return []
    images = sample.multimodal_inputs.get("images")
    if not images:
        return []
    return [encode_image_for_rollout_engine(_get_blank(img.size)) for img in images]
