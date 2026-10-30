#!/bin/bash
# Every SANG job, training and evaluation alike:
#   sbatch [--time=.. --job-name=..] bash_scripts/run.sh scripts/<script>.py [args]
#SBATCH --job-name=sang
#SBATCH --partition=ifn
#SBATCH --account=ifn
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --comment=force_cpus
#SBATCH --mem=64gb
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/bash_scripts/env.sh"
echo "[job] ${SLURM_JOB_ID:-local} on $(hostname) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, -) | cpus: ${SLURM_CPUS_PER_TASK:-?} | env: ${CONDA_DEFAULT_ENV:-?} | commit: $(git rev-parse --short HEAD 2>/dev/null) | $(date '+%F %T')"
echo "[job] python -u $*"
python -u "$@"
