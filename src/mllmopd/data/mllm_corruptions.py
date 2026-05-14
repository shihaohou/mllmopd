"""Image corruptions for Level-1 audit modes.

Three modes used by `scripts/audit/run_level1.sh`:
- full_image:    pass through unchanged
- blank_image:   replace with a same-size white image (preserves dimensions
                 so vision tokenizer produces the same number of patches)
- oracle_caption: replace image with a blank and prepend a caption to the text
- text_only:     no image at all (drop from input)

`swap_image` and `irrelevant_image` are provided for follow-up shortcut tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover
    Image = None  # type: ignore


def _require_pil() -> None:
    if Image is None:
        raise ImportError("Pillow is required for image corruptions. pip install pillow")


def load(image_path: str | Path) -> "Image.Image":
    _require_pil()
    return Image.open(image_path).convert("RGB")


def blank_image(reference: "Image.Image", color: tuple[int, int, int] = (255, 255, 255)) -> "Image.Image":
    """Same-size solid image. Default white; pass (0,0,0) for black."""
    _require_pil()
    return Image.new("RGB", reference.size, color=color)


def swap_image(_unused: "Image.Image", other: "Image.Image") -> "Image.Image":
    """Replace with an image from a different (matched-modality) sample."""
    return other


def irrelevant_image(reference: "Image.Image", noise_seed: int = 0) -> "Image.Image":
    """Same-size random noise — useful as a baseline-irrelevant visual input."""
    _require_pil()
    import random

    rng = random.Random(noise_seed)
    img = Image.new("RGB", reference.size)
    pixels = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            pixels[x, y] = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    return img


def apply_mode(
    image: "Image.Image",
    mode: str,
    *,
    other_image: Optional["Image.Image"] = None,
    caption: Optional[str] = None,
) -> tuple[Optional["Image.Image"], Optional[str]]:
    """Return (image_or_none, prefix_text_or_none) for a given audit mode."""
    if mode == "full_image":
        return image, None
    if mode == "blank_image":
        return blank_image(image), None
    if mode == "text_only":
        return None, None
    if mode == "oracle_caption":
        if not caption:
            raise ValueError("oracle_caption mode needs a non-empty caption")
        return blank_image(image), f"[Image description: {caption}]\n"
    if mode == "swap_image":
        if other_image is None:
            raise ValueError("swap_image mode needs other_image")
        return other_image, None
    if mode == "irrelevant_image":
        return irrelevant_image(image), None
    raise ValueError(f"Unknown audit mode: {mode}")
