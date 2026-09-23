# Shared environment for every SANG job. Sourced, not executed.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slurm_logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate avcodec
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# mediapipe needs libGLESv2, which only the `gl` env ships
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/gl/lib:${LD_LIBRARY_PATH:-}"
# ffmpeg for syncnet_python, linked by bash_scripts/install_motion.sh when the system has none
export PATH="$HOME/.local/bin:$PATH"
