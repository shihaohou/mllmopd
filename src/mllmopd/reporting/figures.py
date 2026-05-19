"""Audit figures for the vision-conditioned OPD framing (post 2026-05-19 GPT review).

Three figures consumed straight from `summary.json` of a level1_v* audit run:

  fig1_acc_by_mode    — 4 models × 3 modes × 6 benchmarks accuracy panel.
                        Shows that full_image >> blank_image ≈ text_only across
                        all models on math/chart benchmarks: the tasks are
                        genuinely vision-dependent.
  fig2_teacher_gap    — Per-benchmark T_RL − S accuracy gap, decomposed by
                        mode. The gap is large on full_image, small/zero on
                        blank/text-only — supports the claim that the
                        teacher's advantage is vision-conditioned.
  fig3_acc_vs_tokens  — Per-benchmark accuracy vs mean output tokens scatter
                        for full_image. Shows that 7B-RL is both more accurate
                        AND not dramatically longer than 7B-SFT: no obvious
                        over-thinking artifact attributable to RL.

Usage:
    python -m mllmopd.reporting.figures \\
        --summary runs/audit/level1_v4_sysprompt_fixed/summary.json \\
        --out_dir runs/audit/level1_v4_sysprompt_fixed/figures
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Canonical short labels for the 4 audit models. Order = post-training axis.
_MODEL_ORDER = ["Base", "3B SFT", "7B SFT", "7B RL"]
_MODEL_COLOR = {
    "Base": "#888888",
    "3B SFT": "#4C72B0",
    "7B SFT": "#55A868",
    "7B RL": "#C44E52",
}
_MODE_ORDER = ["full_image", "blank_image", "text_only"]
_MODE_LABEL = {"full_image": "full", "blank_image": "blank", "text_only": "text-only"}
_MODE_HATCH = {"full_image": "", "blank_image": "//", "text_only": ".."}


def _short_model(path: str) -> str | None:
    """Map the JSONL `model` (absolute path) → canonical label, or None
    if it doesn't match any of the four expected models."""
    base = os.path.basename(path.rstrip("/"))
    if "Qwen2.5-VL-7B-Instruct" in base:
        return "Base"
    if "MMR1-3B-SFT" in base:
        return "3B SFT"
    if "MMR1-7B-SFT" in base:
        return "7B SFT"
    if "MMR1-7B-RL" in base:
        return "7B RL"
    return None


def _index_cells(cells: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Returns {(model_label, mode, benchmark): cell_dict}, dropping rows
    that don't match one of the four canonical models."""
    out: dict[tuple[str, str, str], dict] = {}
    for c in cells:
        m = _short_model(c["model"])
        if m is None:
            continue
        out[(m, c["mode"], c["benchmark"])] = c
    return out


def fig1_acc_by_mode(summary_path: Path, out: Path) -> None:
    """2×3 grid: per-benchmark grouped bars; x = 4 models, hue = 3 modes."""
    s = json.loads(summary_path.read_text())
    idx = _index_cells(s["cells"])
    benches = sorted({k[2] for k in idx})
    n_b = len(benches)
    n_rows = 2
    n_cols = (n_b + 1) // n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), sharey=False)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    bar_w = 0.25
    x_positions = np.arange(len(_MODEL_ORDER))

    for ax, bench in zip(axes_flat, benches):
        for j, mode in enumerate(_MODE_ORDER):
            heights = []
            for m in _MODEL_ORDER:
                cell = idx.get((m, mode, bench))
                heights.append(cell["accuracy"] if cell and cell.get("accuracy") is not None else 0)
            ax.bar(
                x_positions + (j - 1) * bar_w, heights, bar_w,
                label=_MODE_LABEL[mode],
                hatch=_MODE_HATCH[mode],
                color="#ddd",
                edgecolor=[_MODEL_COLOR[m] for m in _MODEL_ORDER],
                linewidth=1.5,
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(_MODEL_ORDER, rotation=0, fontsize=9)
        ax.set_title(bench, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("accuracy", fontsize=9)
        ax.grid(axis="y", lw=0.3, alpha=0.5)

    # Hide unused subplots
    for ax in axes_flat[n_b:]:
        ax.set_visible(False)

    # Single shared legend at the top
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#ddd", hatch=_MODE_HATCH[m],
                      edgecolor="black", linewidth=1.0)
        for m in _MODE_ORDER
    ]
    fig.legend(handles, [_MODE_LABEL[m] for m in _MODE_ORDER],
               loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Fig 1 — Per-benchmark accuracy by model × mode (level1_v4)",
                 fontsize=12, y=1.06)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f">>> wrote {out}")


def fig2_teacher_gap(summary_path: Path, out: Path) -> None:
    """T_RL − S accuracy gap per benchmark, decomposed by input mode.
    The story: gap on full >> gap on blank ≈ gap on text-only."""
    s = json.loads(summary_path.read_text())
    idx = _index_cells(s["cells"])
    benches = sorted({k[2] for k in idx})

    teacher, student = "7B RL", "3B SFT"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(benches))
    bar_w = 0.25
    mode_colors = {"full_image": "#C44E52", "blank_image": "#4C72B0", "text_only": "#55A868"}
    for j, mode in enumerate(_MODE_ORDER):
        gaps = []
        for bench in benches:
            t = idx.get((teacher, mode, bench))
            s_cell = idx.get((student, mode, bench))
            if not (t and s_cell and t.get("accuracy") is not None and s_cell.get("accuracy") is not None):
                gaps.append(0)
                continue
            gaps.append(t["accuracy"] - s_cell["accuracy"])
        ax.bar(
            x + (j - 1) * bar_w, gaps, bar_w,
            label=_MODE_LABEL[mode], color=mode_colors[mode], edgecolor="black", linewidth=0.5,
        )

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(benches, rotation=15, fontsize=10)
    ax.set_ylabel(f"acc({teacher}) − acc({student})", fontsize=10)
    ax.set_title(f"Fig 2 — Teacher–student gap by input mode (level1_v4)\n"
                 f"Vision-conditioned story: the gap is concentrated in full_image.",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", lw=0.3, alpha=0.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f">>> wrote {out}")


def fig3_acc_vs_tokens(summary_path: Path, out: Path) -> None:
    """Per-benchmark scatter: x = mean output tokens (full_image), y = accuracy.
    Shows whether RL teacher is over-thinking (high tokens, similar acc) or
    actually more efficient."""
    s = json.loads(summary_path.read_text())
    idx = _index_cells(s["cells"])
    benches = sorted({k[2] for k in idx})
    n_b = len(benches)
    n_rows = 2
    n_cols = (n_b + 1) // n_rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharey=False)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, bench in zip(axes_flat, benches):
        for m in _MODEL_ORDER:
            cell = idx.get((m, "full_image", bench))
            if not cell or cell.get("tokens_mean") is None or cell.get("accuracy") is None:
                continue
            ax.scatter([cell["tokens_mean"]], [cell["accuracy"]],
                       s=120, color=_MODEL_COLOR[m], edgecolor="black", linewidth=0.5,
                       label=m, zorder=3)
            ax.annotate(m, (cell["tokens_mean"], cell["accuracy"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_title(bench, fontsize=10)
        ax.set_xlabel("mean output tokens", fontsize=9)
        ax.set_ylabel("accuracy", fontsize=9)
        ax.grid(lw=0.3, alpha=0.5)
        ax.set_ylim(0, 1.0)

    for ax in axes_flat[n_b:]:
        ax.set_visible(False)

    fig.suptitle("Fig 3 — Accuracy vs response length (full_image, level1_v4)\n"
                 "RL more efficient than 7B-SFT? on/below the SFT curve, not above.",
                 fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f">>> wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", required=True, type=Path,
                    help="summary.json from a level1_v* audit run")
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig1_acc_by_mode(args.summary, args.out_dir / "fig1_acc_by_mode.png")
    fig2_teacher_gap(args.summary, args.out_dir / "fig2_teacher_gap.png")
    fig3_acc_vs_tokens(args.summary, args.out_dir / "fig3_acc_vs_tokens.png")


if __name__ == "__main__":
    main()
