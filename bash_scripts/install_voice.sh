#!/bin/bash
# One-time install for voice cloning (sang/voice.py). Run on the LOGIN node (needs GitHub + HF):
#   bash bash_scripts/install_voice.sh            # add --asr to also fetch Whisper-large-v3 (~3 GB) for eval CER
# kNN-VC (Baas et al., arXiv 2305.18975) code is MIT; weights are fetched from its GitHub release into
# torch.hub's cache (~/.cache/torch/hub/checkpoints), which compute nodes then read offline.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${SANG_ENV:-avcodec}"

python -c "import torchaudio; print('torchaudio', torchaudio.__version__)" || {
    echo "kNN-VC imports torchaudio; install the build that matches your torch:  pip install torchaudio==<torch version>"; exit 1; }

[ -d third_party/knn-vc ] || git clone --depth 1 https://github.com/bshall/knn-vc third_party/knn-vc
# WavLM-Large (unilm) + prematched HiFi-GAN weights -> torch hub cache (CPU load is enough to fetch them)
python - <<'EOF'
import sys; sys.path.insert(0, "third_party/knn-vc")
from hubconf import knn_vc
knn_vc(pretrained=True, progress=True, prematched=True, device="cpu")
print("kNN-VC weights cached")
EOF

# speaker verification (enrolment filter + similarity metric)
huggingface-cli download microsoft/wavlm-base-plus-sv
if [ "${1:-}" = "--asr" ]; then
    huggingface-cli download openai/whisper-large-v3
fi

python -m pytest tests/test_voice.py -q
echo "voice install ok. Next: sbatch bash_scripts/job.sh scripts/eval_voice.py --targets 20"
