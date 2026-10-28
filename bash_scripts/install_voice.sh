#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${SANG_ENV:-avcodec}"

python -c "import torchaudio; print('torchaudio', torchaudio.__version__)" || {
    echo "kNN-VC imports torchaudio; install the build that matches your torch:  pip install torchaudio==<torch version>"; exit 1; }

[ -d third_party/knn-vc ] || git clone --depth 1 https://github.com/bshall/knn-vc third_party/knn-vc
python - <<'EOF'
import sys; sys.path.insert(0, "third_party/knn-vc")
from hubconf import knn_vc
knn_vc(pretrained=True, progress=True, prematched=True, device="cpu")
print("kNN-VC weights cached")
EOF

huggingface-cli download microsoft/wavlm-base-plus-sv
if [ "${1:-}" = "--asr" ]; then
    huggingface-cli download openai/whisper-large-v3
fi

python -m pytest tests/test_voice.py -q
echo "voice install ok. Next: sbatch bash_scripts/job.sh scripts/eval_voice.py --targets 20"
