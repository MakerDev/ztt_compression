#!/usr/bin/env python
"""Generate placeholder plots from run summaries."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots for paper")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    placeholder = output_dir / "README.md"
    placeholder.write_text(
        "# Plots\n\nGenerated plots will be saved here by scripts/make_plots.py.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
