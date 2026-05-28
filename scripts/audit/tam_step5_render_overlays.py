"""Step 5 qualitative case renderer.

Reads `qualitative_cases.jsonl` (output of tam_step5_analyzer) and
materializes a 3-panel comparison figure per (sample, token):

    [original image] | [T overlay] | [S0 overlay] | [S1 overlay]
    + per-token caption: token text, category, IoU/JS/Cos against T

Each output PNG shows one token's evidence map across the three models
side by side. Useful for paper qualitative figure + presentation slides.

Output layout:
    <out-dir>/<bucket>/<sample_id>/<tok_idx>_<token_text>.png

Usage::

    python -m scripts.audit.tam_step5_render_overlays \\
        --alignment runs/audit/tam_step5_<TS>/alignment.jsonl \\
        --picks docs/figures/step5/qualitative_cases.jsonl \\
        --out-dir docs/figures/step5/qualitative_overlays/ \\
        [--image-root .] [--alpha 0.5]
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


def _decode_b64_map(b64: str, h: int, w: int) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size != h * w:
        raise ValueError(f"map size {arr.size} != H*W {h*w}")
    return arr.reshape(h, w)


def _overlay_pil(image_rgb, map_uint8_HW: np.ndarray, alpha: float = 0.5):
    """Pillow + JET colormap → blended overlay. Returns PIL.Image RGB."""
    from PIL import Image
    from matplotlib import cm

    W_img, H_img = image_rgb.size
    map_pil = Image.fromarray(map_uint8_HW, mode="L").resize(
        (W_img, H_img), resample=Image.BILINEAR,
    )
    map_np = np.array(map_pil).astype(np.float32) / 255.0
    jet_rgba = cm.get_cmap("jet")(map_np)
    jet_rgb = (jet_rgba[..., :3] * 255.0).astype(np.uint8)
    img_np = np.array(image_rgb).astype(np.float32)
    blended = (jet_rgb.astype(np.float32) * alpha
               + img_np * (1.0 - alpha)).astype(np.uint8)
    return Image.fromarray(blended)


def _resolve_image(image_path_str: str, image_root: Path):
    """Mirror of tam_step5_evidence_alignment._load_image_for_rec resolution."""
    from PIL import Image
    p = Path(image_path_str)
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
        # Last-resort: foreign absolute path that included data/audit/images
        # is already handled; here just bail with a clear error.
        raise FileNotFoundError(f"image not found: {p}")
    return Image.open(p).convert("RGB"), p


def _stream_lookup(alignment_path: Path, wanted_ids: set[str]) -> dict:
    """Stream the (large) alignment.jsonl once, pulling only rows whose
    id is in `wanted_ids`. Returns dict keyed by id."""
    out: dict = {}
    with alignment_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            # Cheap pre-filter: check for any of the wanted ids in the
            # raw line before json.loads (saves ~50% time on large file).
            if not any(f'"{wid}"' in line[:200] for wid in wanted_ids):
                continue
            rec = json.loads(line)
            if rec.get("id") in wanted_ids:
                out[rec["id"]] = rec
                if len(out) == len(wanted_ids):
                    break
    return out


def _compose_triplet(image, maps_T, maps_S0, maps_S1, alpha: float,
                     caption: str):
    """Compose [orig, T, S0, S1] horizontal panel with caption strip."""
    from PIL import Image, ImageDraw, ImageFont

    overlays = [
        ("orig", image),
        ("T", _overlay_pil(image, maps_T, alpha=alpha)),
        ("S0", _overlay_pil(image, maps_S0, alpha=alpha)),
        ("S1", _overlay_pil(image, maps_S1, alpha=alpha)),
    ]

    # Resize all panels to a common height for clean horizontal stacking
    target_h = 256
    resized = []
    for name, im in overlays:
        w, h = im.size
        new_w = int(w * (target_h / h))
        resized.append((name, im.resize((new_w, target_h), Image.BILINEAR)))

    pad = 8
    label_h = 20
    caption_h = 60
    total_w = sum(im.size[0] for _, im in resized) + pad * (len(resized) + 1)
    total_h = target_h + label_h + caption_h + pad * 3

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11
        )
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
        font_small = font

    draw = ImageDraw.Draw(canvas)
    x = pad
    for name, im in resized:
        canvas.paste(im, (x, label_h + pad))
        draw.text((x + 6, 2), name, fill=(0, 0, 0), font=font)
        x += im.size[0] + pad

    # Caption strip below the row
    cap_y = label_h + pad + target_h + pad
    # Word-wrap caption manually
    max_chars_per_line = total_w // 7  # rough px → chars
    words = caption.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + 1 > max_chars_per_line:
            lines.append(cur)
            cur = w
        else:
            cur = w if not cur else cur + " " + w
    if cur:
        lines.append(cur)
    for i, line in enumerate(lines[:3]):
        draw.text((pad, cap_y + i * 16), line, fill=(0, 0, 0), font=font_small)
    return canvas


def render_pick(pick: dict, row: dict, image_root: Path, out_dir: Path,
                alpha: float) -> tuple[int, str]:
    """Render all tok_indices in a single pick. Returns (n_rendered, msg)."""
    if row is None:
        return 0, f"no alignment row for id={pick['id']}"
    tok_indices = pick.get("tok_indices") or []
    if not tok_indices:
        return 0, f"no tok_indices for id={pick['id']}"

    image_path_str = row.get("image_path")
    if not image_path_str:
        return 0, f"no image_path for id={pick['id']}"
    try:
        image, _ = _resolve_image(image_path_str, image_root)
    except FileNotFoundError as e:
        return 0, str(e)

    map_h = int(row["map_h"])
    map_w = int(row["map_w"])
    bucket = pick.get("bucket", "unknown")
    sample_dir = out_dir / bucket / _safe_name(pick["id"])
    sample_dir.mkdir(parents=True, exist_ok=True)

    maps_b64 = row.get("maps_b64", {})
    T_b64 = maps_b64.get("T", [])
    S0_b64 = maps_b64.get("S0", [])
    S1_b64 = maps_b64.get("S1", [])
    tokens = row.get("tokens", [])
    cat = row.get("token_category", [])
    align_S0 = row.get("align", {}).get("S0_T", {})
    align_S1 = row.get("align", {}).get("S1_T", {})

    n_rendered = 0
    for t in tok_indices:
        if t >= len(T_b64) or t >= len(S0_b64) or t >= len(S1_b64):
            continue
        try:
            M_T = _decode_b64_map(T_b64[t], map_h, map_w)
            M_S0 = _decode_b64_map(S0_b64[t], map_h, map_w)
            M_S1 = _decode_b64_map(S1_b64[t], map_h, map_w)
        except Exception as e:  # noqa: BLE001
            print(f"!! decode failed t={t}: {e!r}", file=sys.stderr)
            continue

        token_txt = tokens[t] if t < len(tokens) else "?"
        token_cat = cat[t] if t < len(cat) else "?"
        js_s0 = (align_S0.get("js") or [None] * (t + 1))[t]
        js_s1 = (align_S1.get("js") or [None] * (t + 1))[t]
        iou_s0 = (align_S0.get("iou_top20") or [None] * (t + 1))[t]
        iou_s1 = (align_S1.get("iou_top20") or [None] * (t + 1))[t]

        def _fmt(v):
            return f"{v:.3f}" if isinstance(v, (int, float)) else "—"

        caption = (
            f"{pick['id']} | bucket={bucket} | t={t} "
            f"tok={token_txt!r} cat={token_cat} | "
            f"S0_correct={pick.get('s0_correct')} "
            f"S1_correct={pick.get('s1_correct')} | "
            f"S0(base)→T: JS={_fmt(js_s0)} IoU={_fmt(iou_s0)} | "
            f"S1(OPD)→T: JS={_fmt(js_s1)} IoU={_fmt(iou_s1)} | "
            f"T=MMR1-7B-RL teacher | S0=MMR1-3B-SFT base | S1=T1-Full OPD"
        )

        try:
            panel = _compose_triplet(image, M_T, M_S0, M_S1, alpha, caption)
            out_path = sample_dir / f"{t:04d}_{_safe_name(token_txt)}.png"
            panel.save(out_path, "PNG")
            n_rendered += 1
        except Exception as e:  # noqa: BLE001
            print(f"!! compose failed id={pick['id']} t={t}: {e!r}",
                  file=sys.stderr)

    return n_rendered, f"id={pick['id']}: {n_rendered}/{len(tok_indices)}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alignment", type=Path, required=True,
                    help="alignment.jsonl from tam_step5_evidence_alignment")
    ap.add_argument("--picks", type=Path, required=True,
                    help="qualitative_cases.jsonl from tam_step5_analyzer")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, default=Path("."))
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="overlay blend: alpha*heatmap + (1-alpha)*image")
    args = ap.parse_args(argv)

    picks: list[dict] = []
    with args.picks.open() as f:
        for line in f:
            if line.strip():
                picks.append(json.loads(line))
    print(f">>> loaded {len(picks)} qualitative picks from {args.picks}",
          file=sys.stderr)

    wanted_ids = {p["id"] for p in picks}
    print(f">>> streaming alignment.jsonl for {len(wanted_ids)} ids "
          f"(this can take ~30s on a 600MB file)...",
          file=sys.stderr)
    rows = _stream_lookup(args.alignment, wanted_ids)
    print(f">>> matched {len(rows)}/{len(wanted_ids)} ids", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_total = 0
    for pick in picks:
        n, msg = render_pick(pick, rows.get(pick["id"]),
                             args.image_root, args.out_dir, args.alpha)
        n_total += n
        print(f"  {msg}", file=sys.stderr)

    print(f"\n>>> wrote {n_total} PNGs to {args.out_dir}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
