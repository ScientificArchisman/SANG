"""sang.render.pipeline.Talker end to end on CPU: tiny trained SANG runs (latent codes, GFT), the real MotionCodec
render code with a stand-in LivePortrait (tests/test_render.py), a stand-in WavLM."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

pytest.importorskip("cv2")

from sang.model.target import M_DIM, rotation_matrix
from sang.render.liveportrait import MotionCodec
from sang.render.pipeline import Talker
from test_render import FakeLP
from test_train_eval import make_cache


class FakeWavLM:
    def encode(self, wav):
        n = wav.shape[-1] // 320
        g = torch.Generator().manual_seed(int(n))
        return torch.randn(1, n, 1024, generator=g)


def fake_codec(monkeypatch, H=100, W=120, S=32):
    camera = type(sys)("src.utils.camera")
    camera.get_rotation_matrix = lambda p, y, r: rotation_matrix(p, y, r)
    for name in ("src", "src.utils"):
        monkeypatch.setitem(sys.modules, name, type(sys)(name))
    monkeypatch.setitem(sys.modules, "src.utils.camera", camera)
    codec = object.__new__(MotionCodec)
    codec.lp, codec.device = FakeLP(), "cpu"
    g = torch.Generator().manual_seed(0)
    M = np.array([[2.0, 0, 20.0], [0, 2.0, 15.0], [0, 0, 1]])
    calls = {"detect": 0}

    def crop_image(src):
        calls["detect"] += 1
        return {"img_crop_256x256": np.zeros((256, 256, 3), np.uint8), "M_c2o": M}

    m_src = torch.zeros(1, M_DIM)
    m_src[:, 0] = 1.2
    kp = torch.randn(1, 63, generator=g)

    def source_motion(src, crop=None):
        if crop is None:
            crop_image(src)
        return {"m": m_src.clone(), "kp": kp.clone()}


    def source_state(crop):
        return {"f_s": torch.randn(1, 2, 2, 2, 2, generator=torch.Generator().manual_seed(1)),
                "x_s": kp.reshape(1, 21, 3).clone(), "x_c": kp.reshape(1, 21, 3)}

    codec.crop_image, codec.source_motion, codec.source_state = crop_image, source_motion, source_state
    return codec, calls


@pytest.fixture(scope="module")
def tiny_cache(tmp_path_factory):
    cache = tmp_path_factory.mktemp("pipe") / "cache"
    make_cache(cache)
    return cache


def train(cache, name, *extra):
    over = [f"cache_dir={cache}", f"out_dir={cache.parent / name}", f"sync_offsets={cache / 'sync_offsets.json'}", "dim=32", "layers=1", "heads=2",
            "max_steps=3", "warmup_steps=1", "eval_every=3", "log_every=1", "batch_size=2", "workers=0",
            "eval_batches=1", "sample_steps=2", "val_frac=0.3", "windows_per_clip=2", f"latent={cache / 'pca.pt'}",
            *extra]
    log = subprocess.run([sys.executable, str(REPO / "scripts" / "train_motion.py"), *over],
                         capture_output=True, encoding="utf-8", cwd=REPO)
    assert log.returncode == 0, log.stderr[-2000:]
    return str(cache.parent / name / "last.pt"), log.stdout


@pytest.fixture(scope="module")
def tiny_ckpt(tiny_cache):
    return train(tiny_cache, "run")[0]


SRC = np.random.default_rng(0).integers(0, 256, (100, 120, 3), dtype=np.uint8)
WAV, N = torch.randn(16000 * 3) * 0.1, 75


def test_talker_detects_the_face_once_pastes_and_times_every_stage(monkeypatch, tiny_ckpt):
    codec, calls = fake_codec(monkeypatch)
    talker = Talker(tiny_ckpt, "g=2,steps=3", device="cpu", codec=codec, wavlm=FakeWavLM())
    frames, m, timer = talker.run(SRC, WAV, N, batch=8)
    assert calls["detect"] == 1
    assert frames.shape == (N, 100, 120, 3) and frames.dtype == np.uint8 and m.shape == (N, M_DIM)
    assert set(timer.t) == {"source", "audio", "motion", "render", "paste"}
    crops, m2, _ = talker.run(SRC, WAV, N, paste=False, batch=4)
    assert crops.shape == (N, 32, 32, 3) and torch.allclose(m, m2)


def test_a_smaller_latent_trains_and_talks(monkeypatch, tiny_cache):
    lat, log = train(tiny_cache, "lat", "latent_k=16")
    assert "target 16-d, GFT" in log and "std_r_place" in log and "EVAL step 3" in log
    codec, _ = fake_codec(monkeypatch)
    t = Talker(lat, "g=2,steps=3", device="cpu", codec=codec, wavlm=FakeWavLM())
    frames, m, _ = t.run(SRC, WAV, N)
    assert frames.shape == (N, 100, 120, 3) and torch.isfinite(m).all()
    ck = torch.load(lat, map_location="cpu", weights_only=False)
    assert ck["latent"]["K"] == 69 and ck["cfg"]["latent_k"] == 16

