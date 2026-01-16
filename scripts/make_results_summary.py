#!/usr/bin/env python
"""Aggregate metrics from run directories into a summary file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize run metrics")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    summary = []
    for run_dir in sorted(runs_root.glob("*")):
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.resolved.yaml"
        if not metrics_path.exists() or not config_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        summary.append({"run": run_dir.name, "metrics": metrics})
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
