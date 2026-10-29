#!/bin/bash
# The SANG environment.
#   source bash_scripts/env.sh          activate it (run.sh does this for every job); SANG_ENV picks the conda env
#   bash bash_scripts/env.sh install    one-time setup of the avcodec env: LivePortrait, SyncNet, ArcFace, I3D, tests
#   bash bash_scripts/env.sh ditto      one-time setup of the Ditto baseline in its own env (SANG_ENV=ditto to use it)
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p slurm_logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${SANG_ENV:-avcodec}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
export LD_LIBRARY_PATH="$HOME/miniconda3/envs/gl/lib:${LD_LIBRARY_PATH:-}"
export PATH="$HOME/.local/bin:$PATH"

install() {
    [ -d third_party/LivePortrait ] || git clone --depth 1 https://github.com/KwaiVGI/LivePortrait third_party/LivePortrait
    pip install insightface tyro pykalman scikit-image "imageio[ffmpeg]" pytorch-fid scipy yt-dlp \
                python_speech_features scenedetect
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
    if ! command -v ffmpeg >/dev/null; then
        mkdir -p "$HOME/.local/bin"
        ln -sf "$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$HOME/.local/bin/ffmpeg"
        echo "linked ffmpeg -> $HOME/.local/bin/ffmpeg"
    fi

    python -c "from insightface.app import FaceAnalysis; FaceAnalysis(name='buffalo_l', root='third_party/insightface_full', providers=['CPUExecutionProvider']).prepare(ctx_id=-1)"
    ls third_party/insightface_full/models/buffalo_l/w600k_r50.onnx
    huggingface-cli download facebook/wav2vec2-xlsr-53-espeak-cv-ft

    mkdir -p third_party/fvd
    if [ ! -s third_party/fvd/i3d_torchscript.pt ]; then
        curl -L -o third_party/fvd/i3d_torchscript.pt "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
    fi
    python -c "import torch; torch.jit.load('third_party/fvd/i3d_torchscript.pt'); print('I3D ok')"
    python -c "from pytorch_fid.inception import InceptionV3; InceptionV3([3]); print('Inception ok')"

    python -m sang.model.target
    python -m sang.render.liveportrait
    python -m sang.eval.metrics
    python -m pytest tests -q
    local check="from sang.render.liveportrait import MotionCodec; MotionCodec(); print('LivePortrait API ok')"
    if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
        python -c "$check"
    else
        echo "No GPU here. Check the renderer on a GPU node:"
        echo "  sbatch --time=00:20:00 bash_scripts/run.sh -c \"$check\""
    fi
}

ditto() {
    local DITTO=third_party/ditto-talkinghead
    [ -d "$DITTO/.git" ] || git clone --depth 1 https://github.com/antgroup/ditto-talkinghead "$DITTO"
    echo "ditto commit: $(git -C "$DITTO" rev-parse --short HEAD)"
    conda env list | grep -qE "^ditto\s" || conda create -y -n ditto python=3.10
    conda activate ditto
    pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
    pip install librosa tqdm filetype imageio imageio-ffmpeg opencv-python-headless scikit-image cython colored \
                einops mediapipe "numpy<2.1" onnxruntime-gpu "huggingface_hub[cli]"
    pip uninstall -y opencv-contrib-python opencv-python || true
    pip install --force-reinstall --no-deps opencv-python-headless
    python - "$DITTO/checkpoints" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download("digital-avatar/ditto-talkinghead", allow_patterns=["ditto_pytorch/*", "ditto_cfg/*"],
                  local_dir=sys.argv[1])
EOF
    ls "$DITTO/checkpoints/ditto_pytorch/models" "$DITTO/checkpoints/ditto_cfg"
    python -c "
import torch, onnxruntime, librosa, cv2, einops, numpy, mediapipe
from mediapipe.tasks.python import vision, BaseOptions
print('torch', torch.__version__, '| ort', onnxruntime.__version__, '| mediapipe', mediapipe.__version__,
      '| cv2', cv2.__version__, '| numpy', numpy.__version__)"
    echo "ditto install ok (test it on a GPU node with scripts/ditto_hdtf.py)"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -euo pipefail
    case "${1:-}" in
        install) install ;;
        ditto) ditto ;;
        *) echo "usage: source bash_scripts/env.sh | bash bash_scripts/env.sh install | bash bash_scripts/env.sh ditto" >&2
           exit 1 ;;
    esac
fi
