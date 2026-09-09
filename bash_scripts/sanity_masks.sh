#!/bin/bash
#SBATCH --job-name=sanity_masks
#SBATCH --partition=ifn
#SBATCH --account=ifn
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slurm_logs results/sanity
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate avcodec
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CONFIG="${CONFIG:-configs/train_stream_v3.yaml}"
python -u scripts/sanity.py masks --config "$CONFIG" "$@" | tee "results/sanity/masks_${SLURM_JOB_ID:-manual}.txt"
