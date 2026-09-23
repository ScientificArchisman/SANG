#!/bin/bash
#SBATCH --job-name=sang_motion
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
# Usage: sbatch bash_scripts/train_motion.sh [--config configs/train_motion.yaml] [key=value ...]
# 53 M params over 74-frame windows: memory is small at batch 256, so the 40 GB card also works
# (--gres=gpu:a100:1). The log is copied to results/train/ for the record.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/bash_scripts/env.sh"
mkdir -p results/train
python -u scripts/train_motion.py "$@" 2>&1 | tee "results/train/train_motion_${SLURM_JOB_ID:-local}.txt"
