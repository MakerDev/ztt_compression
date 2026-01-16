#!/usr/bin/env bash
#SBATCH --job-name=ztt_full
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_full_%j.out
#SBATCH --error=slurm_full_%j.err

set -euo pipefail

bash scripts/run_full.sh
