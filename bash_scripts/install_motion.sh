#!/bin/bash
# One-time install for the motion track. Run on the LOGIN node (it needs GitHub + Hugging Face):
#   bash bash_scripts/install_motion.sh
# Licences: LivePortrait code MIT, weights on HF; its InsightFace detector is non-commercial.
# syncnet_python is used for LSE-C / LSE-D only.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate avcodec

[ -d third_party/LivePortrait ] || git clone --depth 1 https://github.com/KwaiVGI/LivePortrait third_party/LivePortrait
pip install onnxruntime-gpu insightface tyro pykalman scikit-image "imageio[ffmpeg]" pytorch-fid scipy
huggingface-cli download KlingTeam/LivePortrait --local-dir third_party/LivePortrait/pretrained_weights \
    --exclude "*animal*" "*.git*"

if [ ! -d third_party/syncnet_python ]; then
    git clone --depth 1 https://github.com/joonson/syncnet_python third_party/syncnet_python
    (cd third_party/syncnet_python && sh download_model.sh)
fi
pip install python_speech_features scenedetect
# syncnet_python shells out to `ffmpeg` by name; give it imageio-ffmpeg's binary if none is on PATH
if ! command -v ffmpeg >/dev/null; then
    mkdir -p "$HOME/.local/bin"
    ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$HOME/.local/bin/ffmpeg"
    echo "linked ffmpeg -> $HOME/.local/bin/ffmpeg (make sure it is on PATH in bash_scripts/env.sh)"
fi

# Pure-tensor checks, no GPU needed; then the API check needs the weights.
python sang/motion.py
python sang/bench.py
python -m pytest tests/test_motion_model.py -q
python -c "from sang.motion import MotionCodec; MotionCodec(device='cpu'); print('LivePortrait API ok')"
