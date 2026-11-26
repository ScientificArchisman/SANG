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


class FakeLP:
    """LivePortrait stand-in: stitching nudges x_d by a function of (x_s, x_d); the warping module and
    decoder turn keypoints into an image deterministically, per sample (so batching must not matter)."""

    def stitching(self, x_s, x_d):
        return x_d + 0.01 * torch.tanh(x_s - x_d)

    def warping_module(self, feature_3d, kp_source, kp_driving):
        b = kp_driving.shape[0]
        v = (kp_driving - kp_source).reshape(b, -1)[:, :8] + feature_3d.reshape(b, -1)[:, :8]
        return {"out": v[:, :, None, None].expand(b, 8, 4, 4)}

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


class FakeDense(torch.nn.Module):
    """LivePortrait's dense motion: slot logits from its `mask` layer, a softmax over [still, 21 keypoints], the
    slot-weighted flows as the deformation, and an occlusion map."""

    def __init__(self):
        super().__init__()
        self.mask = torch.nn.Identity()

    def forward(self, feature, kp_driving, kp_source):
        b = len(kp_driving)
        flows = torch.cat([torch.zeros(b, 1, 3), kp_source - kp_driving], 1)[:, :, None, None, None]  # [b,22,1,1,1,3]
        mask = self.mask(torch.ones(b, 22, 2, 64, 64)).softmax(1)
        return {"mask": mask, "deformation": (flows * mask[..., None]).sum(1),
                "occlusion_map": torch.full((b, 1, 64, 64), 0.7)}


class FakeWarping(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_motion_network = FakeDense()

    def forward(self, feature_3d, kp_source, kp_driving):
        o = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving, kp_source=kp_source)
        return {"out": o["deformation"][:, :, :4, :4, 0].repeat(1, 4, 1, 1) + o["occlusion_map"][:, :, :4, :4]}


def test_probe_sees_every_batch_and_the_lock_forces_the_identity_flow(monkeypatch):
    codec, state, m = fake_codec(monkeypatch)
    codec.lp.warping_module = FakeWarping()
    seen = []
    plain = codec.render(state, m, batch=4)
    probed = torch.cat(list(codec.render_iter(state, m, batch=4, probe=lambda o, i: seen.append((i, o)))))
    assert torch.equal(plain, probed) and [i for i, _ in seen] == [0, 4, 8]
    cells = torch.zeros(10, 64, 64)
    cells[:, :, :32] = 1  # the left half of every frame is still
    occ_ref = torch.full((1, 1, 64, 64), 0.2)
    locked = []
    list(codec.render_iter(state, m, batch=4, probe=lambda o, i: locked.append(o), lock=(cells, occ_ref)))
    dm = codec.lp.warping_module.dense_motion_network
    assert not dm._forward_hooks and not dm.mask._forward_hooks
    d = torch.cat([o["deformation"] for o in locked])
    free = torch.cat([o["deformation"] for _, o in seen])
    assert d[:, :, :, :32].abs().max() < 1e-6 and torch.allclose(d[:, :, :, 32:], free[:, :, :, 32:])
    occ = torch.cat([o["occlusion_map"] for o in locked])
    assert torch.allclose(occ[..., :32], torch.tensor(0.2)) and torch.allclose(occ[..., 32:], torch.tensor(0.7))


def test_background_metrics_self_check():
    from sang.eval.background import self_check
    self_check()


def test_timer_adds_up():
    t = Timer(cuda=False)
    with t("a"):
        pass
    with t("a"):
        pass
    assert set(t.t) == {"a"} and t.t["a"] >= 0

