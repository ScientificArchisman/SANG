"""Phase C speed track (sang/fastpath.py, MotionCodec.render_iter). CPU only: the renderer's networks are
replaced by small deterministic stand-ins with LivePortrait's call signatures."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

cv2 = pytest.importorskip("cv2")

from sang import fastpath
from sang.motion import M_DIM, paste_back


def test_gpu_paste_matches_cpu_paste_back():
    rng = np.random.default_rng(0)
    S, H, W = 64, 120, 160
    crops = rng.integers(0, 256, (3, S, S, 3), dtype=np.uint8)
    bg = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    ang = np.deg2rad(7)
    M = np.array([[1.3 * np.cos(ang), -1.3 * np.sin(ang), 30.5], [1.3 * np.sin(ang), 1.3 * np.cos(ang), 12.25], [0, 0, 1]])
    mask = fastpath.crop_mask(S)
    ref = paste_back(crops, M, bg, mask=mask)
    got = fastpath.GpuPaster(M, bg, S, device="cpu", mask=mask)(torch.from_numpy(crops))
    d = np.abs(ref.astype(int) - got.astype(int))
    assert d.max() <= 3 and d.mean() < 0.05                  # cv2 interpolates in 1/32-px fixed point
    psnr = 10 * np.log10(255 ** 2 / max(((ref.astype(float) - got) ** 2).mean(), 1e-12))
    assert psnr > 50
    far = np.array([[1, 0, 1000.0], [0, 1, 1000.0], [0, 0, 1]])  # crop entirely outside: the background
    assert (fastpath.GpuPaster(far, bg, S, device="cpu")(torch.from_numpy(crops)) == bg).all()


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

    def warp_decode(self, f_s, x_s, x_d):
        return {"out": self.spade_generator(self.warping_module(f_s, kp_source=x_s, kp_driving=x_d)["out"])}

    @staticmethod
    def parse_output(out):
        out = np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])
        return np.clip(np.clip(out, 0, 1) * 255, 0, 255).astype(np.uint8)


def fake_codec(monkeypatch):
    from sang import motion

    camera = type(sys)("src.utils.camera")
    camera.get_rotation_matrix = lambda p, y, r: motion.rotation_matrix(p, y, r)
    for name in ("src", "src.utils"):
        monkeypatch.setitem(sys.modules, name, type(sys)(name))
    monkeypatch.setitem(sys.modules, "src.utils.camera", camera)
    codec = object.__new__(motion.MotionCodec)
    codec.lp, codec.device = FakeLP(), "cpu"
    g = torch.Generator().manual_seed(0)
    info = {"exp": torch.randn(1, 21, 3, generator=g) * 0.02, "scale": torch.tensor([[1.1]]),
            "t": torch.randn(1, 3, generator=g) * 0.01, "kp": torch.randn(1, 21, 3, generator=g)}
    state = {"info": info, "R_s": motion.rotation_matrix(torch.tensor([3.0]), torch.tensor([-5.0]), torch.tensor([1.0])),
             "f_s": torch.randn(1, 2, 2, 2, 2, generator=g), "x_s": torch.randn(1, 21, 3, generator=g),
             "x_c": info["kp"]}
    m = torch.randn(10, M_DIM, generator=g) * 0.05
    m[:, 0] = 1.2
    m[:, 1:4] = torch.randn(10, 3, generator=g) * 5
    return codec, state, m


@pytest.mark.parametrize("relative", [False, True])
def test_batched_render_matches_the_per_frame_loop(monkeypatch, relative):
    codec, state, m = fake_codec(monkeypatch)
    slow = codec.render(None, m, relative=relative, stitch=True, state=state)
    for batch in (2, 4, 16):
        fast = codec.render(None, m, relative=relative, stitch=True, state=state, batch=batch)
        assert fast.shape == slow.shape and np.abs(fast.astype(int) - slow.astype(int)).max() <= 1
    gpu = codec.render(None, m, relative=relative, stitch=True, state=state, batch=4, out="torch")
    assert isinstance(gpu, torch.Tensor) and gpu.dtype == torch.uint8 and gpu.shape == slow.shape


def test_async_writer_writes_a_playable_video(tmp_path):
    rng = np.random.default_rng(1)
    w = fastpath.AsyncWriter(tmp_path / "a.mp4", fps=25)
    for _ in range(3):
        w.write(rng.integers(0, 256, (5, 64, 64, 3), dtype=np.uint8))
    out = w.close()
    import imageio_ffmpeg
    assert out.exists() and imageio_ffmpeg.count_frames_and_secs(str(out))[0] == 15


def test_timer_adds_up():
    t = fastpath.Timer(cuda=False)
    with t("a"):
        pass
    with t("a"):
        pass
    assert set(t.t) == {"a"} and t.t["a"] >= 0
