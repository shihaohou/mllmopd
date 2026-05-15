"""Image corruptions for Level-1 audit modes.

The five primary modes used by the audit form a 3-way decomposition of the
(image, caption) input axes:

|                        | no caption          | + caption                 |
| ---------------------- | ------------------- | ------------------------- |
| no image               | text_only           | (skip — same as caption_only_blank without the blank) |
| blank image            | blank_image         | caption_only_blank        |
| full image             | full_image          | image_plus_caption        |

This decomposition is what lets the H1 quadrant separate perception-hard from
reasoning-hard prompts cleanly: comparing full_image to caption_only_blank
isolates "did the model need to *see*, or did it just need the caption-level
facts?". Earlier versions had only `oracle_caption` (= blank+caption) which
collapsed two axes into one cell and made the quadrant definition incoherent.

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


_CAPTION_PREFIX = "[Image description: {caption}]\n"

# Old name `oracle_caption` is kept as an alias of caption_only_blank for any
# external configs already authored against the previous schema.
_MODE_ALIASES = {"oracle_caption": "caption_only_blank"}


def apply_mode(
    image: "Image.Image",
    mode: str,
    *,
    other_image: Optional["Image.Image"] = None,
    caption: Optional[str] = None,
) -> tuple[Optional["Image.Image"], Optional[str]]:
    """Return (image_or_none, prefix_text_or_none) for a given audit mode."""
    mode = _MODE_ALIASES.get(mode, mode)
    if mode == "full_image":
        return image, None
    if mode == "blank_image":
        return blank_image(image), None
    if mode == "text_only":
        return None, None
    if mode == "caption_only_blank":
        if not caption:
            raise ValueError("caption_only_blank mode needs a non-empty caption")
        return blank_image(image), _CAPTION_PREFIX.format(caption=caption)
    if mode == "image_plus_caption":
        if not caption:
            raise ValueError("image_plus_caption mode needs a non-empty caption")
        return image, _CAPTION_PREFIX.format(caption=caption)
    if mode == "swap_image":
        if other_image is None:
            raise ValueError("swap_image mode needs other_image")
        return other_image, None
    if mode == "irrelevant_image":
        return irrelevant_image(image), None
    raise ValueError(f"Unknown audit mode: {mode}")
