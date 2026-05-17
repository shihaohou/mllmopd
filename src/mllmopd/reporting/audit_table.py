"""Pretty-print the Level-1 audit summary as a table on stdout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _model_short(name: str) -> str:
    """Display name for a model: basename if it looks like a filesystem path,
    otherwise the trailing component after the last `/` (HF hub id).
    Keeps JSONL field untouched; this is just for the table."""
    if not name:
        return name
    if name.startswith("/") or os.sep in name:
        return os.path.basename(name.rstrip("/"))
    return name.split("/")[-1] if "/" in name else name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=Path)
    args = ap.parse_args()

    s = json.loads((args.run_dir / "summary.json").read_text())
    cells = s["cells"]

    def _f(x):
        return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))

    headers = ["model", "mode", "benchmark", "n", "acc", "tokens_mean", "acc/tok"]
    widths = [22, 20, 16, 5, 6, 11, 9]
    fmt = "  ".join(f"%-{w}s" for w in widths)
    print(fmt % tuple(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))

    for c in cells:
        print(fmt % (
            _model_short(c["model"])[:widths[0]],
            c["mode"][:widths[1]],
            c["benchmark"][:widths[2]],
            c["n"],
            _f(c["accuracy"]),
            _f(c["tokens_mean"]),
            _f(c["acc_per_token"]),
        ))

    paired = s.get("paired_full_blank", [])
    if paired:
        print()
        print("=== full vs blank, paired by prompt id ===")
        p_headers = ["model", "benchmark", "n", "both_ok", "full_only", "blank_only", "both_wrong",
                     "img_lift", "blank_shortcut"]
        p_widths = [22, 16, 5, 8, 9, 10, 10, 9, 14]
        p_fmt = "  ".join(f"%-{w}s" for w in p_widths)
        print(p_fmt % tuple(p_headers))
        print("-" * (sum(p_widths) + 2 * (len(p_widths) - 1)))
        for p in paired:
            print(p_fmt % (
                _model_short(p["model"])[:p_widths[0]],
                p["benchmark"][:p_widths[1]],
                p["n_paired"],
                p["both_correct"], p["full_only"], p["blank_only"], p["both_wrong"],
                _f(p["image_lift_rate"]),
                _f(p["blank_shortcut_rate"]),
            ))


if __name__ == "__main__":
    main()
