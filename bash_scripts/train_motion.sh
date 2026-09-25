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
#SBATCH --gres=gpu:pro6000b_96gb:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
# Usage: sbatch bash_scripts/train_motion.sh [--config configs/train_motion.yaml] [key=value ...]
# Default card: pro6000b_96gb (gpu10/gpu11, partition ifn). Override per job with --gres, e.g.
# sbatch --gres=gpu:a100_80gb:1 bash_scripts/train_motion.sh. The log is copied to results/train/.
set -euo pipefail
# PRO 6000 Blackwell (sm_120) needs torch built for CUDA 12.8+; avcodec has 2.4.1+cu121. sang_bw also
# runs on the A100s. Create it once with: bash bash_scripts/make_env_bw.sh
export SANG_ENV="${SANG_ENV:-sang_bw}"
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/bash_scripts/env.sh"
mkdir -p results/train
python -u scripts/train_motion.py "$@" 2>&1 | tee "results/train/train_motion_${SLURM_JOB_ID:-local}.txt"
