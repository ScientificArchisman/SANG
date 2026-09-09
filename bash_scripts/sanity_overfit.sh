#!/bin/bash
#SBATCH --job-name=sanity_overfit
#SBATCH --partition=ifn
#SBATCH --account=ifn
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slurm_logs results/sanity
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate avcodec
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/gl/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CONFIG="${CONFIG:-configs/train_stream_v3.yaml}"
STEPS="${STEPS:-400}"
python -u scripts/sanity.py overfit --config "$CONFIG" --steps "$STEPS" --device cuda "$@" \
  | tee "results/sanity/overfit_${SLURM_JOB_ID:-manual}.txt"
