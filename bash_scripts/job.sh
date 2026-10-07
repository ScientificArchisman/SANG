#!/bin/bash
#SBATCH --job-name=sang_job
#SBATCH --partition=ifn
#SBATCH --account=ifn
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --comment=force_cpus
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
# Any short GPU job. Usage: sbatch bash_scripts/job.sh scripts/ceiling.py --n 32
set -euo pipefail
# sbatch runs a copy of this script from /var/spool, so resolve env.sh from the submit dir
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/bash_scripts/env.sh"
# so every log says what produced it
echo "[job] ${SLURM_JOB_ID:-local} on $(hostname) | gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd, -) | cpus: ${SLURM_CPUS_PER_TASK:-?} | env: ${CONDA_DEFAULT_ENV:-?} | commit: $(git rev-parse --short HEAD 2>/dev/null) | $(date '+%F %T')"
echo "[job] python -u $*"
python -u "$@"
