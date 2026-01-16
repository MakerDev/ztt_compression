#!/usr/bin/env bash
#SBATCH --job-name=ztt_sanity
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_sanity_%j.out
#SBATCH --error=slurm_sanity_%j.err

set -euo pipefail

bash scripts/run_sanity.sh
