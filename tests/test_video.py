"""Round-trip sanity for the frozen VidTok video tokenizer."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.video import load_vidtok, reconstruct


def test_roundtrip_shapes():
    """Causal round-trip preserves [B,3,T,H,W]; FSQ indices have latent shape and valid range."""
    codebook = 4096
    model = load_vidtok(codebook=codebook, device="cpu")
    x = torch.rand(1, 3, 17, 64, 64) * 2 - 1
    rec, idx = reconstruct(model, x, return_indices=True)
    assert rec.shape == x.shape
    assert idx.shape == (1, 5, 8, 8)  # (17-1)/4+1 latent frames, 64/8 spatial
    assert idx.dtype in (torch.int32, torch.int64)
    assert 0 <= int(idx.min()) and int(idx.max()) < codebook


if __name__ == "__main__":
    test_roundtrip_shapes()
    print("ok")
