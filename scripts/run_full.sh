#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

python -m src.train.harness \
  --scheme "${ROOT_DIR}/configs/schemes/ours.yaml" \
  --baseline "${ROOT_DIR}/configs/baselines/llama_1b.yaml" \
  --train "${ROOT_DIR}/configs/train/full.yaml" \
  --run-name "full_${TIMESTAMP}"

python "${ROOT_DIR}/scripts/make_results_summary.py" --runs-root "${ROOT_DIR}/runs" --output "${ROOT_DIR}/paper/results_summary.json"
python "${ROOT_DIR}/scripts/make_plots.py" --runs-root "${ROOT_DIR}/runs" --output-dir "${ROOT_DIR}/paper/figures"
python "${ROOT_DIR}/scripts/export_tables_tex.py" --summary "${ROOT_DIR}/paper/results_summary.json" --output "${ROOT_DIR}/paper/tables.tex"
