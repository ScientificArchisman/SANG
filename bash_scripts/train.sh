#!/bin/bash
#SBATCH --job-name=sang_train
#SBATCH --partition=ifn
#SBATCH --account=ifn
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48gb
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slurm_logs results/train
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate avcodec
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/gl/lib:${LD_LIBRARY_PATH:-}"  # libGLESv2 for mediapipe (E2 face cond)
python scripts/train.py "$@" | tee "results/train/train_${SLURM_JOB_ID:-manual}.txt"
