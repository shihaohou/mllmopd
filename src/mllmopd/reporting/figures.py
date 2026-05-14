"""Generators for the five key audit figures (see docs/experiment-protocol.md).

Each function takes a `summary.json` path and writes a PDF/PNG. Plot styling is
deliberately minimal — readability > prettiness for now.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def fig1_accuracy_vs_length(summary_path: Path, out: Path) -> None:
    """Per-benchmark accuracy gain vs response-length gain, T_RL vs T_SFT."""
    s = json.loads(summary_path.read_text())
    by_bench: dict[str, dict[str, dict]] = defaultdict(dict)
    for c in s["cells"]:
        if c["mode"] != "full_image":
            continue
        by_bench[c["benchmark"]][c["model"]] = c

    xs, ys, labels = [], [], []
    for bench, models in by_bench.items():
        t_rl = next((c for k, c in models.items() if "RL" in k.upper()), None)
        t_sft = next((c for k, c in models.items() if "SFT" in k.upper()), None)
        if t_rl is None or t_sft is None:
            continue
        if t_rl["accuracy"] is None or t_sft["accuracy"] is None:
            continue
        if t_rl["tokens_mean"] is None or t_sft["tokens_mean"] is None:
            continue
        xs.append((t_rl["tokens_mean"] - t_sft["tokens_mean"]) / max(1, t_sft["tokens_mean"]))
        ys.append(t_rl["accuracy"] - t_sft["accuracy"])
        labels.append(bench)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(0, lw=0.5, color="gray")
    ax.axvline(0, lw=0.5, color="gray")
    ax.scatter(xs, ys, s=60)
    for x, y, l in zip(xs, ys, labels):
        ax.annotate(l, (x, y), fontsize=9, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Δ response length (RL − SFT, rel.)")
    ax.set_ylabel("Δ accuracy (RL − SFT)")
    ax.set_title("Fig 1 — RL teacher: accuracy gain vs length gain")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out)
    print(f">>> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    fig1_accuracy_vs_length(args.summary, args.out_dir / "fig1_acc_vs_length.pdf")


if __name__ == "__main__":
    main()
