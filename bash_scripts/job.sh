#!/bin/bash
#SBATCH --job-name=sang_job
#SBATCH --partition=ifn
#SBATCH --account=ifn
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:a100_80gb:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
# Any short GPU job. Usage: sbatch bash_scripts/job.sh scripts/ceiling.py --n 32
set -euo pipefail
# sbatch runs a copy of this script from /var/spool, so resolve env.sh from the submit dir
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/bash_scripts/env.sh"
python -u "$@"
