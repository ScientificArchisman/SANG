#!/bin/bash
# One-time: Ditto (antgroup/ditto-talkinghead, ACM MM 2025) as an HDTF baseline, in its own conda env.
# Run on the LOGIN node (needs internet):  bash bash_scripts/install_ditto.sh
# Then on a GPU node:  SANG_ENV=ditto sbatch bash_scripts/job.sh scripts/ditto_hdtf.py --data <HDTF> --list configs/hdtf_test.csv
# Uses Ditto's PyTorch weights (no TensorRT build needed); its helper models are ONNX (onnxruntime-gpu).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"

DITTO=third_party/ditto-talkinghead
if [ ! -d "$DITTO/.git" ]; then
    git clone --depth 1 https://github.com/antgroup/ditto-talkinghead "$DITTO"
fi
echo "ditto commit: $(git -C "$DITTO" rev-parse --short HEAD)"

conda env list | grep -qE "^ditto\s" || conda create -y -n ditto python=3.10
conda activate ditto
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
pip install librosa tqdm filetype imageio imageio-ffmpeg opencv-python-headless scikit-image cython colored \
            "numpy<2.1" onnxruntime-gpu "huggingface_hub[cli]"

# weights: only the PyTorch models and the configs (skips the TensorRT engines and ONNX copies)
huggingface-cli download digital-avatar/ditto-talkinghead --include "ditto_pytorch/*" "ditto_cfg/*" \
    --local-dir "$DITTO/checkpoints"
ls "$DITTO/checkpoints/ditto_pytorch/models" "$DITTO/checkpoints/ditto_cfg"
python -c "import torch, onnxruntime, librosa, cv2; print('torch', torch.__version__, '| ort', onnxruntime.__version__)"
echo "ditto install ok (test it on a GPU node with scripts/ditto_hdtf.py)"
