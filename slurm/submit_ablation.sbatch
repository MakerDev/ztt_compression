#!/usr/bin/env bash
#SBATCH --job-name=ztt_ablation
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm_ablation_%j.out
#SBATCH --error=slurm_ablation_%j.err

set -euo pipefail

bash scripts/run_ablation.sh
