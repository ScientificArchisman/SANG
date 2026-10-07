#!/bin/bash
# One-time install for the HDTF benchmark. Run on the LOGIN node (needs internet):
#   bash bash_scripts/install_eval.sh
# Then: python scripts/download_hdtf.py --out <folder>   (login node: YouTube)
#       sbatch --time=12:00:00 --cpus-per-task=8 bash_scripts/job.sh scripts/eval_hdtf.py --data <folder> ...
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${SANG_ENV:-avcodec}"

pip install -U yt-dlp pytorch-fid scipy

# FVD: StyleGAN-V's TorchScript port of the Kinetics-400 I3D (the standard FVD feature network)
mkdir -p third_party/fvd
if [ ! -s third_party/fvd/i3d_torchscript.pt ]; then
    curl -L -o third_party/fvd/i3d_torchscript.pt "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1"
fi
python -c "import torch; m = torch.jit.load('third_party/fvd/i3d_torchscript.pt'); print('I3D ok')"

# FID: pytorch-fid fetches its Inception weights on first use; do it here, where there is internet
python -c "from pytorch_fid.inception import InceptionV3; InceptionV3([3]); print('Inception ok')"

# LSE (syncnet_python) and CSIM (ArcFace) come from install_motion.sh; check they are there
ls third_party/syncnet_python/data/syncnet_v2.model third_party/insightface_full/models/buffalo_l/w600k_r50.onnx

python -m pytest tests/test_eval.py -q
echo "eval install ok"
