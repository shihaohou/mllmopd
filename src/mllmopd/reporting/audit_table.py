"""Pretty-print the Level-1 audit summary as a table on stdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=Path)
    args = ap.parse_args()

    s = json.loads((args.run_dir / "summary.json").read_text())
    cells = s["cells"]

    headers = ["model", "mode", "benchmark", "n", "acc", "tokens_mean", "acc/tok"]
    widths = [22, 16, 16, 5, 6, 11, 9]
    fmt = "  ".join(f"%-{w}s" for w in widths)
    print(fmt % tuple(headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))

    def _f(x):
        return "-" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))

    for c in cells:
        print(fmt % (
            c["model"][:widths[0]],
            c["mode"][:widths[1]],
            c["benchmark"][:widths[2]],
            c["n"],
            _f(c["accuracy"]),
            _f(c["tokens_mean"]),
            _f(c["acc_per_token"]),
        ))


if __name__ == "__main__":
    main()
