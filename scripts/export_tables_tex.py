#!/usr/bin/env python
"""Export results summary to a LaTeX table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export metrics summary to LaTeX")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        output_path.write_text("% No summary available\n", encoding="utf-8")
        return

    summary = json.loads(summary_path.read_text())
    lines = ["\\begin{tabular}{ll}", "Run & Metric Keys \\\", "\\hline"]
    for entry in summary:
        run_name = entry.get("run", "-")
        metric_keys = ", ".join(sorted(entry.get("metrics", {}).keys())) or "-"
        lines.append(f"{run_name} & {metric_keys} \\")
    lines.append("\\end{tabular}")
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
