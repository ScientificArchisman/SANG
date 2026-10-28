#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate avcodec

[ -d third_party/LivePortrait ] || git clone --depth 1 https://github.com/KwaiVGI/LivePortrait third_party/LivePortrait
pip install insightface tyro pykalman scikit-image "imageio[ffmpeg]" pytorch-fid scipy
pip uninstall -y onnxruntime onnxruntime-gpu || true
rm -rf "$(python -c 'import site; print(site.getsitepackages()[0])')/onnxruntime"
pip install onnxruntime-gpu==1.23.2
python -c "import onnxruntime as o; p = o.get_available_providers(); print('onnxruntime', o.__version__, p); assert 'CUDAExecutionProvider' in p"
huggingface-cli download KlingTeam/LivePortrait --local-dir third_party/LivePortrait/pretrained_weights \
    --exclude "*animal*" "*.git*"

[ -d third_party/syncnet_python ] || git clone --depth 1 https://github.com/joonson/syncnet_python third_party/syncnet_python
if [ ! -s third_party/syncnet_python/data/syncnet_v2.model ] || [ ! -s third_party/syncnet_python/detectors/s3fd/weights/sfd_face.pth ]; then
    (cd third_party/syncnet_python && sh download_model.sh)
fi
ls -la third_party/syncnet_python/data/syncnet_v2.model third_party/syncnet_python/detectors/s3fd/weights/sfd_face.pth
pip install python_speech_features scenedetect
if ! command -v ffmpeg >/dev/null; then
    mkdir -p "$HOME/.local/bin"
    ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$HOME/.local/bin/ffmpeg"
    echo "linked ffmpeg -> $HOME/.local/bin/ffmpeg (make sure it is on PATH in bash_scripts/env.sh)"
fi

python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', root='third_party/insightface_full', providers=['CPUExecutionProvider']).prepare(ctx_id=-1)"
ls third_party/insightface_full/models/buffalo_l/w600k_r50.onnx

huggingface-cli download facebook/wav2vec2-xlsr-53-espeak-cv-ft

python sang/motion.py
python sang/bench.py
python -m pytest tests/test_motion_model.py tests/test_naturalness.py -q

if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "from sang.motion import MotionCodec; MotionCodec(); print('LivePortrait API ok')"
else
    echo "No GPU here. Run the renderer check on a GPU node:"
    echo "  sbatch --time=00:20:00 bash_scripts/job.sh -c \"from sang.motion import MotionCodec; MotionCodec(); print('LivePortrait API ok')\""
fi
