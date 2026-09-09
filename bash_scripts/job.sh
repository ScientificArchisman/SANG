#!/bin/bash
#SBATCH --job-name=sang_job
#SBATCH --partition=ifn
#SBATCH --account=ifn
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
# Any short GPU job. Usage: sbatch bash_scripts/job.sh scripts/ceiling.py --n 32
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
python -u "$@"
