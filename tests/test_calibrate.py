"""sang.render.calibrate on CPU: a stand-in decoder with LivePortrait's SPADE layout (mlp_gamma / mlp_beta), a
stand-in appearance extractor (first, down_blocks) and a stand-in keypoint reader."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sang.render import calibrate as cal


class SPADE(nn.Module):
    def __init__(self, c, label_nc=4):
        super().__init__()
        self.param_free_norm = nn.InstanceNorm2d(c, affine=False)
        self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, 8, 3, padding=1), nn.ReLU())
        self.mlp_gamma, self.mlp_beta = nn.Conv2d(8, c, 3, padding=1), nn.Conv2d(8, c, 3, padding=1)

    def forward(self, x, seg):
        a = self.mlp_shared(F.interpolate(seg, size=x.shape[2:], mode="nearest"))
        return self.param_free_norm(x) * (1 + self.mlp_gamma(a)) + self.mlp_beta(a)


class Decoder(nn.Module):
    """feature [B, 4, 8, 8] -> image [B, 3, 32, 32] in [0, 1]."""

    def __init__(self):
        super().__init__()
        self.fc, self.n0, self.n1 = nn.Conv2d(4, 6, 3, padding=1), SPADE(6), SPADE(6)
        self.conv_img = nn.Conv2d(6, 3, 3, padding=1)

    def forward(self, feature):
        x = F.interpolate(self.n0(self.fc(feature), feature), scale_factor=2)
        x = F.interpolate(self.n1(x, feature), scale_factor=2)
        return torch.sigmoid(self.conv_img(F.leaky_relu(x, 0.2)))


class Feats(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Conv2d(3, 4, 3, padding=1)
        self.down_blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.AvgPool2d(2))
                                          for _ in range(2)])


def kp_reader():
    P = torch.randn(48, 63, generator=torch.Generator().manual_seed(1))
    lp = SimpleNamespace(motion_extractor=lambda x: (F.adaptive_avg_pool2d(x, 4).flatten(1) @ P).view(-1, 21, 3),
                         transform_keypoint=lambda info: info)
    return lambda img: cal.keypoints(lp, img)


def test_zero_offsets_change_nothing_and_a_mismatch_is_refused():
    torch.manual_seed(0)
    G, X = Decoder(), torch.randn(2, 4, 8, 8)
    with torch.no_grad():
        y = G(feature=X)
        with cal.offsets(G, cal.zeros(G)):
            assert torch.equal(G(feature=X), y)
        with cal.offsets(G, [t + 0.1 for t in cal.zeros(G)]):
            assert not torch.allclose(G(feature=X), y)
        assert torch.equal(G(feature=X), y)
    assert [t.shape for t in cal.zeros(G)] == [(4, 6), (4, 6)]
    with pytest.raises(ValueError), cal.offsets(G, cal.zeros(G)[:1]):
        pass


def test_the_amplitude_term_ignores_a_small_shift_but_not_blur():
    h = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(0))
    shift, blurred = h.roll(1, -1), cal.blur(h)
    amp = lambda a, b: float((cal.log_amp(a) - cal.log_amp(b)).abs().mean())
    px = lambda a, b: float((a - b).abs().mean())
    assert amp(h, shift) / amp(h, blurred) < 0.5 * px(h, shift) / px(h, blurred)


def test_fit_moves_toward_a_calibrated_teacher_and_restores_the_decoder():
    torch.manual_seed(0)
    G, Fx, kp_of, w = Decoder(), Feats(), kp_reader(), torch.ones(32, 32)
    X = torch.randn(48, 4, 8, 8)
    star = [0.5 * torch.randn_like(t) for t in cal.zeros(G)]
    with torch.no_grad():
        with cal.offsets(G, star):
            Y = (G(feature=X) * 255).round().to(torch.uint8)
        kp = kp_of(G(feature=X))
    pairs = lambda j: {"X": X[j], "Y": Y[j], "kp": kp[j]}
    train, held = pairs(slice(0, 40)), pairs(slice(40, 48))
    pre = cal.check(G, Fx, held, w)
    assert set(pre) == {"pre_noise_0.25", "pre_noise_0.5", "pre_noise_1", "pre_blur", "pre_ok"}
    with torch.no_grad():
        before = G(feature=X[:2])
    theta, log = cal.fit(G, Fx, kp_of, train, held, X[:1], w, lr=3e-2)
    assert log["cal_held"] < log["cal_held0"] and log["cal_accept"] == 1.0 and log["cal_pairs"] == 40
    with torch.no_grad():
        assert torch.equal(G(feature=X[:2]), before)
        with cal.offsets(G, theta):
            assert not torch.allclose(G(feature=X[:2]), before)
    oracle, olog = cal.fit(G, Fx, kp_of, train, None, X[:1], w)
    assert "cal_accept" not in olog and len(oracle) == 2
