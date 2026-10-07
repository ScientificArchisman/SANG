# Shared environment for every SANG job. Sourced, not executed.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slurm_logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${SANG_ENV:-avcodec}"    # train_motion.sh sets sang_bw (Blackwell-capable torch)
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# compute nodes run with an ASCII locale: make Python's text I/O and subprocess decoding UTF-8
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
# mediapipe needs libGLESv2, which only the `gl` env ships
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/gl/lib:${LD_LIBRARY_PATH:-}"
# ffmpeg for syncnet_python, linked by bash_scripts/install_motion.sh when the system has none
export PATH="$HOME/.local/bin:$PATH"
