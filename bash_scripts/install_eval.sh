#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${SANG_ENV:-avcodec}"

pip install -U yt-dlp pytorch-fid scipy

mkdir -p third_party/fvd
if [ ! -s third_party/fvd/i3d_torchscript.pt ]; then
    curl -L -o third_party/fvd/i3d_torchscript.pt "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
fi
python -c "import torch; m = torch.jit.load('third_party/fvd/i3d_torchscript.pt'); print('I3D ok')"

python -c "from pytorch_fid.inception import InceptionV3; InceptionV3([3]); print('Inception ok')"

ls third_party/syncnet_python/data/syncnet_v2.model third_party/insightface_full/models/buffalo_l/w600k_r50.onnx

python -m pytest tests/test_eval.py -q
echo "eval install ok"
