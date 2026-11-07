"""sang.render.calibrate on CPU: a stand-in decoder with LivePortrait's layout (SPADE-style blocks with conv_0 /
conv_1, conv_img = Sequential[Conv2d, PixelShuffle])."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("cv2")

from sang.render.calibrate import calibrated, haar2d, image_scale, image_score

S, LEVELS = 32, 3


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv_0, self.conv_1 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.conv_1(torch.relu(self.conv_0(x)))


class Decoder(nn.Module):
    def __init__(self, c=8):
        super().__init__()
        self.blocks = nn.Sequential(Block(c), Block(c))
        self.conv_img = nn.Sequential(nn.Conv2d(c, 12, 3, padding=1), nn.PixelShuffle(2))

    def forward(self, feature):
        return torch.sigmoid(self.conv_img(self.blocks(feature)))


def fake_codec():
    torch.manual_seed(0)
    G = Decoder()
    proj = torch.randn(63, 8 * 16 * 16) * 0.1
    return SimpleNamespace(device="cpu", lp=SimpleNamespace(spade_generator=G),
                           driving_keypoints=lambda s, m: m[:, :63].reshape(-1, 21, 3),
                           warped=lambda s, x_d, half=False: (x_d.flatten(1) @ proj).view(-1, 8, 16, 16))


def footage(T=40):
    g = np.random.default_rng(0)
    pattern = g.integers(60, 200, (S, S, 3)).astype(np.float32)
    frames = np.stack([pattern + g.normal(0, 8, (S, S, 3)) for _ in range(T)]).clip(0, 255).astype(np.uint8)
    return frames, torch.randn(T, 70, generator=torch.Generator().manual_seed(1))


def test_the_image_score_is_orthonormal_and_lowest_at_the_real_spread():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4, 3, 16, 16, generator=g)
    assert torch.allclose(sum(h.pow(2).sum() for h in haar2d(x, 2)), x.pow(2).sum(), rtol=1e-5)
    y, z = torch.randn(1500, 3, 8, 8, generator=g), torch.randn(1500, 4, 3, 8, 8, generator=g)
    c0, cb = image_scale(y, 2)
    assert cb.numel() == 3 * 3 * 2 + 3  # (level, band, RGB) blocks, then the coarse mean per channel
    score = {s: float(image_score(s * z, y, c0, cb, 2).mean()) for s in (0.62, 1.0, 1.3)}
    assert score[1.0] < score[0.62] and score[1.0] < score[1.3]


@pytest.mark.parametrize("mode", ["recon", "energy"])
def test_calibration_fits_the_decoder_and_restores_it(mode):
    codec = fake_codec()
    G = codec.lp.spade_generator
    frames, motion = footage()
    M = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    f = codec.warped(None, codec.driving_keypoints(None, motion[:4]))
    with torch.no_grad():
        before = G(feature=f)
    log = {}
    with calibrated(codec, {}, M, frames, motion, mode, steps=40, lr=1e-2, rank=2, renders=8, hold=8, levels=LEVELS,
                    log=log):
        assert not isinstance(G.blocks[0].conv_0, nn.Conv2d)
        with torch.no_grad():
            a, b = G(feature=f), G(feature=f)
        assert not torch.allclose(a, before) and torch.equal(a, b)  # energy: one noise map per clip
    assert isinstance(G.blocks[0].conv_0, nn.Conv2d) and isinstance(G.conv_img[0], nn.Conv2d)
    with torch.no_grad():
        assert torch.equal(G(feature=f), before)
    assert log["rfit_es"] < log["rfit_es0"] and {"rfit_es_10", "rfit_es_20", "rfit_es_30", "rfit_s"} <= log.keys()
    assert ("rfit_noise" in log) == (mode == "energy") and log.get("rfit_noise", 1.0) > 0


def test_too_little_footage_skips_the_fit():
    codec = fake_codec()
    frames, motion = footage(12)
    log = {}
    with calibrated(codec, {}, np.eye(3)[:2], frames, motion, hold=8, renders=8, log=log):
        assert isinstance(codec.lp.spade_generator.blocks[0].conv_0, nn.Conv2d)
    assert log == {"rfit_skipped": 1.0}
