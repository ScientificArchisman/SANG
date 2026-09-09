#!/bin/bash
#SBATCH --job-name=stream_test
#SBATCH --partition=ifn
#SBATCH --account=ifn
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slurm_logs results/test
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate avcodec
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/gl/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CKPT="${CKPT:-runs/stream_v3/best.pt}"
N="${N:-32}"
python -u scripts/test.py --config configs/train_stream_v3.yaml \
  --ckpt "$CKPT" --n "$N" \
  --out "results/test/test_stream_${SLURM_JOB_ID:-manual}.json" \
  "$@" | tee "results/test/test_stream_${SLURM_JOB_ID:-manual}.txt"
