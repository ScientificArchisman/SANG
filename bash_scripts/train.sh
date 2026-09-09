#!/bin/bash
#SBATCH --job-name=sang
#SBATCH --partition=ifn
#SBATCH --account=ifn
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120gb
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
# Usage: sbatch bash_scripts/train.sh [--config configs/train.yaml] [--set k=v ...]
set -euo pipefail
# sbatch runs a copy of this script from /var/spool, so resolve env.sh from the submit dir
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/bash_scripts/env.sh"
mkdir -p results/train
python -u scripts/train.py "${@:---config configs/train.yaml}" \
  | tee "results/train/${SLURM_JOB_ID:-manual}.txt"
