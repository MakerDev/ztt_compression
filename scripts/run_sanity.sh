#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

python -m src.train.harness \
  --scheme "${ROOT_DIR}/configs/schemes/ours.yaml" \
  --baseline "${ROOT_DIR}/configs/baselines/llama_1b.yaml" \
  --train "${ROOT_DIR}/configs/train/sanity.yaml" \
  --run-name "sanity_${TIMESTAMP}"
