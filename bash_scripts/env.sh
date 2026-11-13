#!/bin/bash
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slurm_logs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${SANG_ENV:-sang}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export PATH="$HOME/.local/bin:$PATH"

install() {
    [ -d third_party/LivePortrait ] || git clone --depth 1 https://github.com/KwaiVGI/LivePortrait third_party/LivePortrait
    pip install -r third_party/LivePortrait/requirements_base.txt -r requirements.txt
    python -c "import onnxruntime as o; p = o.get_available_providers(); print('onnxruntime', o.__version__, p); assert 'CUDAExecutionProvider' in p"
    python -c "from huggingface_hub import snapshot_download; snapshot_download('KlingTeam/LivePortrait', local_dir='third_party/LivePortrait/pretrained_weights', ignore_patterns=['*animal*', '*.git*'])"

    [ -d third_party/syncnet_python ] || git clone --depth 1 https://github.com/joonson/syncnet_python third_party/syncnet_python
    if [ ! -s third_party/syncnet_python/data/syncnet_v2.model ] || [ ! -s third_party/syncnet_python/detectors/s3fd/weights/sfd_face.pth ]; then
        (cd third_party/syncnet_python && sh download_model.sh)
    fi
    pip install scipy python_speech_features scenedetect
    if ! command -v ffmpeg >/dev/null; then
        mkdir -p "$HOME/.local/bin"
        ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$HOME/.local/bin/ffmpeg"
    fi

    mkdir -p checkpoints
    [ -s checkpoints/face_landmarker.task ] || curl -L -o checkpoints/face_landmarker.task \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

    python -m sang.model.target
    python -m sang.render.liveportrait
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    case "${1:-}" in
        install) install ;;
        *) echo "usage: source bash_scripts/env.sh | bash bash_scripts/env.sh install" >&2
           exit 1 ;;
    esac
fi
