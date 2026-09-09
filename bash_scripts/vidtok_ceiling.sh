#!/bin/bash
#SBATCH --job-name=vidtok_ceiling
#SBATCH --partition=ifn
#SBATCH --account=ifn
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=01:30:00
#SBATCH 
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate avcodec
CEILING_CPU=1 python -u scripts/vidtok_ceiling.py
