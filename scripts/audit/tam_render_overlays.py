"""Offline PNG-overlay renderer for tam_sanity.jsonl.

Reads `tam_maps_subset.maps_uint8_b64` from each row and writes one PNG per
stored token map. Uses Pillow + numpy + matplotlib (no cv2 dependency) so
this runs on the Mac scaffolding box or any venv without opencv.

The original `tam_sanity.py` skips overlays when cv2 is missing — but it
still stores the per-token heatmaps in JSONL as base64 uint8. This script
materializes them.

Usage:
  python scripts/audit/tam_render_overlays.py \\
      --run-dir runs/audit/tam_sanity_20260525-133802 \\
      [--image-root .] \\
      [--alpha 0.5]

Output:
  <run_dir>/overlays/<id>/<stratum>_<token_idx>_<token_text>.png
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import numpy as np


def _safe_name(s: str, max_len: int = 24) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s)[:max_len]


def _decode_map(b64: str, h: int, w: int) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size != h * w:
        raise ValueError(f"map size {arr.size} != H*W {h*w}")
    return arr.reshape(h, w)


def _overlay_pil(image_rgb, map_uint8_HW, alpha: float = 0.5):
    """Pillow + matplotlib JET colormap. Returns PIL.Image RGB."""
    from PIL import Image
    from matplotlib import cm

    W_img, H_img = image_rgb.size
    # Upsample map to image size with bilinear
    map_pil = Image.fromarray(map_uint8_HW, mode="L").resize(
        (W_img, H_img), resample=Image.BILINEAR,
    )
    map_np = np.array(map_pil).astype(np.float32) / 255.0
    # JET colormap → RGBA → drop alpha → uint8
    jet_rgba = cm.get_cmap("jet")(map_np)
    jet_rgb = (jet_rgba[..., :3] * 255.0).astype(np.uint8)
    img_np = np.array(image_rgb).astype(np.float32)
    blended = (jet_rgb.astype(np.float32) * alpha
               + img_np * (1.0 - alpha)).astype(np.uint8)
    return Image.fromarray(blended)


def _render_one_row(row: dict, image_root: Path, out_dir: Path, alpha: float):
    from PIL import Image

    if not row.get("tam_valid"):
        return 0, f"skip (tam_valid=false): {row.get('tam_failure_reason')}"
    subset = row.get("tam_maps_subset") or {}
    indices = subset.get("token_indices") or []
    if not indices:
        return 0, "no tam_maps_subset.token_indices"

    map_h = int(row["map_h"])
    map_w = int(row["map_w"])
    image_path = Path(row["image_path"])
    if not image_path.is_absolute():
        image_path = (image_root / image_path).resolve()
    # Fallback: JSONL may carry a foreign-host absolute path (e.g. H800
    # cluster path while we render on Mac). Try to reconstruct the local
    # path under image_root using the `data/audit/images/<...>` suffix.
    if not image_path.exists():
        s = str(image_path).replace("\\", "/")
        marker = "data/audit/images/"
        if marker in s:
            tail = s.rsplit(marker, 1)[-1]      # "POPE_adversarial/1636.png"
            candidate = image_root / "data" / "audit" / "images" / tail
            if candidate.exists():
                image_path = candidate
    if not image_path.exists():
        return 0, f"image missing: {image_path}"
    image = Image.open(image_path).convert("RGB")

    probe_dir = out_dir / row["id"].replace("/", "_")
    probe_dir.mkdir(parents=True, exist_ok=True)

    tokens     = row.get("tokens") or []
    strata     = subset.get("selection_strata") or [""] * len(indices)
    maps_b64   = subset["maps_uint8_b64"]

    n = 0
    for idx, stratum, b64 in zip(indices, strata, maps_b64):
        try:
            m = _decode_map(b64, map_h, map_w)
        except Exception as e:  # noqa: BLE001
            print(f"  ! idx={idx}: decode failed: {e}", file=sys.stderr)
            continue
        tok_str = tokens[idx] if idx < len(tokens) else f"tok{idx}"
        out_name = f"resp_{idx:03d}_{stratum}_{_safe_name(tok_str)}.png"
        ovl = _overlay_pil(image, m, alpha=alpha)
        ovl.save(probe_dir / out_name)
        n += 1
    return n, "ok"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Output dir of run_tam_sanity.sh "
                         "(contains tam_sanity.jsonl)")
    ap.add_argument("--image-root", type=Path, default=Path("."),
                    help="Resolve relative image_path against this root")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Heatmap blend alpha (default 0.5)")
    args = ap.parse_args(argv)

    jsonl = args.run_dir / "tam_sanity.jsonl"
    if not jsonl.exists():
        print(f"!! {jsonl} not found", file=sys.stderr)
        return 1
    out_dir = args.run_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n, msg = _render_one_row(row, args.image_root, out_dir, args.alpha)
            total += n
            print(f"  {row['id']}: {n} overlays  ({msg})", file=sys.stderr)
    print(f">>> wrote {total} PNG overlays under {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
