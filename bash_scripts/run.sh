#!/bin/bash
#SBATCH --job-name=sang
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/bash_scripts/env.sh"
echo "[job] ${SLURM_JOB_ID:-local} on $(hostname) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, -) | env: ${CONDA_DEFAULT_ENV:-?} | commit: $(git rev-parse --short HEAD 2>/dev/null) | $(date '+%F %T')"
echo "[job] python -u $*"
python -u "$@"
