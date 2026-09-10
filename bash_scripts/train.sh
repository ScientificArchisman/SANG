#!/bin/bash
#SBATCH --job-name=sang
#SBATCH --partition=ifn
#SBATCH --account=ifn
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --comment=force_cpus
#SBATCH --mem=120gb
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
# Usage: sbatch bash_scripts/train.sh [--config configs/train.yaml] [--set k=v ...]
# 80 GB cards take the config as is: a100_80gb (gpu08 x8, default here) or h100 (gpu09 x2,
# --gres=gpu:h100:1). gpu:a100 on gpu06 is the 40 GB card: --gres=gpu:a100:1 --set batch_size=2 grad_accum=32
set -euo pipefail
# sbatch runs a copy of this script from /var/spool, so resolve env.sh from the submit dir
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/bash_scripts/env.sh"
mkdir -p results/train
python -u scripts/train.py "${@:---config configs/train.yaml}" \
  | tee "results/train/${SLURM_JOB_ID:-manual}.txt"
