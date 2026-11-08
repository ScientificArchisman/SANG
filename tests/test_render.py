"""Rendering and paste-back (MotionCodec.render, sang.render.pipeline.GpuPaster). CPU only: the renderer's networks are
replaced by small deterministic stand-ins with LivePortrait's call signatures."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

cv2 = pytest.importorskip("cv2")

from sang.model.target import M_DIM, rotation_matrix

from sang.render.liveportrait import as3x3
from sang.render.pipeline import GpuPaster, Timer, crop_mask


def cv2_paste(frames, M_c2o, background, mask):
    """LivePortrait's own paste-back: warp each crop over the full frame with cv2, then blend with the mask."""
    H, W = background.shape[:2]
    M = as3x3(M_c2o)[:2]
    m = cv2.warpAffine(mask.astype(np.float32), M, (W, H))[..., None]
    return np.stack([np.clip(m * cv2.warpAffine(f, M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)
                             + (1.0 - m) * background.astype(np.float32), 0, 255).astype(np.uint8) for f in frames])


def test_gpu_paste_matches_cv2_paste_back():
    rng = np.random.default_rng(0)
    S, H, W = 64, 120, 160
    crops = rng.integers(0, 256, (3, S, S, 3), dtype=np.uint8)
    bg = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    ang = np.deg2rad(7)
    M = np.array([[1.3 * np.cos(ang), -1.3 * np.sin(ang), 30.5], [1.3 * np.sin(ang), 1.3 * np.cos(ang), 12.25], [0, 0, 1]])
    mask = crop_mask(S)
    ref = cv2_paste(crops, M, bg, mask)
    got = GpuPaster(M, bg, S, device="cpu", mask=mask)(torch.from_numpy(crops))
    d = np.abs(ref.astype(int) - got.astype(int))
    assert d.max() <= 3 and d.mean() < 0.05
    psnr = 10 * np.log10(255 ** 2 / max(((ref.astype(float) - got) ** 2).mean(), 1e-12))
    assert psnr > 50
    far = np.array([[1, 0, 1000.0], [0, 1, 1000.0], [0, 0, 1]])
    assert (GpuPaster(far, bg, S, device="cpu")(torch.from_numpy(crops)) == bg).all()


def test_the_scene_shows_where_neither_person_moves():
    rng = np.random.default_rng(1)
    S, H, W = 64, 120, 160
    crops = torch.from_numpy(rng.integers(0, 256, (2, S, S, 3), dtype=np.uint8))
    bg, live = rng.integers(0, 256, (H, W, 3), dtype=np.uint8), rng.integers(0, 256, (2, H, W, 3), dtype=np.uint8)
    M = np.array([[1.3, 0, 30.5], [0, 1.3, 12.25], [0, 0, 1]])  # the crop covers x 30.5-112.4, y 12.25-94.2
    p = GpuPaster(M, bg, S, device="cpu")
    still = p(crops)
    assert (p(crops, scene=(live, torch.ones(2, 8, 8))) == live).all() and p.live_share == 1.0
    held = p(crops, scene=(live, torch.zeros(2, 8, 8)))  # both persons everywhere: the paste inside the crop
    assert np.abs(held[:, 20:85, 40:100].astype(int) - still[:, 20:85, 40:100]).max() <= 1  # float weights, uint8 cast
    assert (held[:, :p.y0] == live[:, :p.y0]).all() and (held[:, :, p.x1:] == live[:, :, p.x1:]).all()
    assert abs(p.live_share - (1 - (1.3 * 63) ** 2 / (H * W))) < 0.02


class FakeDenseMotion(torch.nn.Module):
    """LivePortrait's dense motion network: a softmax mask over [still, 21 keypoints] per depth slice and pixel."""

    def forward(self, feature, kp_driving, kp_source):
        d = (kp_driving - kp_source).reshape(len(kp_driving), -1).mean(1)
        logits = torch.arange(22 * 2 * 16.0).reshape(1, 22, 2, 4, 4).sin() * (1 + d.reshape(-1, 1, 1, 1, 1))
        return {"mask": logits.softmax(1)}


class FakeWarping(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_motion_network = FakeDenseMotion()

    def forward(self, feature_3d, kp_source, kp_driving):
        self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving, kp_source=kp_source)
        b = kp_driving.shape[0]
        v = (kp_driving - kp_source).reshape(b, -1)[:, :8] + feature_3d.reshape(b, -1)[:, :8]
        return {"out": v[:, :, None, None].expand(b, 8, 4, 4)}


class FakeLP:
    """LivePortrait stand-in: stitching nudges x_d by a function of (x_s, x_d); the warping module and
    decoder turn keypoints into an image deterministically, per sample (so batching must not matter)."""

    def __init__(self):
        self.warping_module = FakeWarping()

    def stitching(self, x_s, x_d):
        return x_d + 0.01 * torch.tanh(x_s - x_d)

    def spade_generator(self, feature):
        img = torch.sigmoid(feature[:, :3].repeat_interleave(8, 2).repeat_interleave(8, 3))
        return img


def fake_codec(monkeypatch):
    from sang.render import liveportrait as motion

    camera = type(sys)("src.utils.camera")
    camera.get_rotation_matrix = lambda p, y, r: rotation_matrix(p, y, r)
    for name in ("src", "src.utils"):
        monkeypatch.setitem(sys.modules, name, type(sys)(name))
    monkeypatch.setitem(sys.modules, "src.utils.camera", camera)
    codec = object.__new__(motion.MotionCodec)
    codec.lp, codec.device = FakeLP(), "cpu"
    g = torch.Generator().manual_seed(0)
    state = {"f_s": torch.randn(1, 2, 2, 2, 2, generator=g), "x_s": torch.randn(1, 21, 3, generator=g),
             "x_c": torch.randn(1, 21, 3, generator=g)}
    m = torch.randn(10, M_DIM, generator=g) * 0.05
    m[:, 0] = 1.2
    m[:, 1:4] = torch.randn(10, 3, generator=g) * 5
    return codec, state, m


def test_render_does_not_depend_on_the_batch_size(monkeypatch):
    codec, state, m = fake_codec(monkeypatch)
    one = codec.render(state, m, batch=1)
    assert isinstance(one, torch.Tensor) and one.dtype == torch.uint8 and one.shape == (10, 32, 32, 3)
    for batch in (4, 16):
        assert torch.equal(codec.render(state, m, batch=batch), one)


def test_the_background_weight_is_the_still_channel_at_the_most_moving_depth(monkeypatch):
    codec, state, m = fake_codec(monkeypatch)
    bg = []
    assert len(torch.cat(list(codec.render_iter(state, m, batch=4, bg=bg)))) == 10 and [len(b) for b in bg] == [4, 4, 2]
    still = codec.background(state, m, batch=3)
    assert torch.allclose(torch.cat(bg), still)  # batch-free, and the same without decoding
    x_s = state["x_s"].expand(10, -1, -1)
    x_d = codec.lp.stitching(x_s, codec.driving_keypoints(state, m))
    dm = codec.lp.warping_module.dense_motion_network
    assert torch.allclose(still, dm(None, x_d, x_s)["mask"][:, 0].amin(1)) and not dm._forward_hooks


def test_timer_adds_up():
    t = Timer(cuda=False)
    with t("a"):
        pass
    with t("a"):
        pass
    assert set(t.t) == {"a"} and t.t["a"] >= 0

