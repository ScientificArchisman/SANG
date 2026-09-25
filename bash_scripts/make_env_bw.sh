#!/bin/bash
# One-time: a training env whose PyTorch runs on the RTX PRO 6000 Blackwell cards (sm_120) as well as
# the A100s. Training imports only torch, numpy and pyyaml; LivePortrait, the cache and inference stay
# in avcodec. Run on the LOGIN node (needs internet, no GPU):  bash bash_scripts/make_env_bw.sh
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda env list | grep -qE "^sang_bw\s" || conda create -y -n sang_bw python=3.10
conda activate sang_bw
pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
pip install numpy pyyaml pytest
python -c "import torch; print('torch', torch.__version__, 'built for CUDA', torch.version.cuda, 'arch list', torch.cuda.get_arch_list() if torch.cuda.is_available() else '(no GPU on this node; sm_120 is in the cu128 wheel)')"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m pytest tests/test_motion_model.py -q
