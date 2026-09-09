#!/bin/bash
#SBATCH --job-name=stream
#SBATCH --partition=ifn
#SBATCH --account=ifn
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80gb
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slurm_logs results/train
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate avcodec
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/gl/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -u scripts/train.py --config configs/train_stream_v3.yaml "$@" | tee "results/train/train_stream_${SLURM_JOB_ID:-manual}.txt"
