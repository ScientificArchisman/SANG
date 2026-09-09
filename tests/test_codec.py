"""Round-trip sanity for the frozen Mimi tokenizer."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.codec import load_mimi, reconstruct


def test_roundtrip_shapes():
    """Reconstruction stays within one frame hop; codes have the requested depth and dtype."""
    mimi = load_mimi(device="cpu")
    hop = int(mimi.sample_rate / mimi.frame_rate)
    wav = torch.randn(1, 1, mimi.sample_rate)
    for k in (8, 32):
        assert abs(reconstruct(mimi, wav, k).shape[-1] - wav.shape[-1]) <= hop
    mimi.set_num_codebooks(8)
    codes = mimi.encode(wav)
    assert codes.shape[1] == 8 and codes.dtype == torch.int64


if __name__ == "__main__":
    test_roundtrip_shapes()
    print("ok")
