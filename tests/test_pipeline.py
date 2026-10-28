"""sang.pipeline.Talker end to end on CPU: tiny trained SANG runs (42-d, GFT, motion latent), the real MotionCodec
render code with a stand-in LivePortrait (tests/test_render.py), a stand-in WavLM."""
import json
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

from sang.latent import fit_pca
from sang.motion import M_DIM, MotionCodec, rotation_matrix, to_target
from sang.pipeline import Talker
from test_render import FakeLP


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
        return {"img_crop_256x256": np.zeros((256, 256, 3), np.uint8), "M_c2o": M, "lmk": None}

    m_src = torch.zeros(1, M_DIM)
    m_src[:, 0] = 1.2
    kp = torch.randn(1, 63, generator=g)

    def source_motion(src, normalize_lip=True, crop=None):
        if crop is None:
            crop_image(src)
        return {"m": m_src.clone(), "kp": kp.clone()}

    info = {"exp": torch.zeros(1, 21, 3), "scale": torch.tensor([[1.2]]), "t": torch.zeros(1, 3),
            "kp": kp.reshape(1, 21, 3)}

    def source_state(src, precropped=False):
        if not precropped:
            crop_image(src)
        return {"info": info, "R_s": torch.eye(3)[None], "f_s": torch.randn(1, 2, 2, 2, 2, generator=torch.Generator().manual_seed(1)),
                "x_s": kp.reshape(1, 21, 3).clone(), "x_c": info["kp"], "M_c2o": M}

    codec.crop_image, codec.source_motion, codec.source_state = crop_image, source_motion, source_state
    return codec, calls


@pytest.fixture(scope="module")
def tiny_cache(tmp_path_factory):
    cache = tmp_path_factory.mktemp("pipe") / "cache"
    rows, g = [], torch.Generator().manual_seed(0)
    for s in range(6):
        for c in range(2):
            p = cache / f"spk{s}" / f"c{c}.pt"
            p.parent.mkdir(parents=True, exist_ok=True)
            T = 90
            mm = torch.randn(T, 70, generator=g) * 0.05
            mm[:, 0] = 1.2
            torch.save({"m": mm.half(), "kp": torch.randn(T, 63, generator=g).half(),
                        "audio": torch.randn(2 * T, 1024, generator=g).half(), "n": T}, p)
            rows.append({"clip": f"/x/spk{s}/c{c}.mp4", "path": str(p), "n": T})
    (cache / "index_000.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    lat = fit_pca(torch.randn(500, 69) * torch.linspace(2, 0.1, 69), K=69)
    torch.save(lat.pack(), cache / "pca.pt")
    return cache


def train(cache, name, *extra):
    over = [f"cache_dir={cache}", f"out_dir={cache.parent / name}", "sync_offsets=", "dim=32", "layers=1", "heads=2",
            "max_steps=3", "warmup_steps=1", "eval_every=3", "log_every=1", "batch_size=2", "workers=0",
            "eval_batches=1", "sample_steps=2", "val_frac=0.3", "windows_per_clip=2", *extra]
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
    talker = Talker(tiny_ckpt, "g=2,mouth=1.25,tau=0.5,avg=2,steps=3", device="cpu", codec=codec, wavlm=FakeWavLM())
    frames, m, timer = talker.run(SRC, WAV, N, start="source", batch=8)
    assert calls["detect"] == 1
    assert frames.shape == (N, 100, 120, 3) and frames.dtype == np.uint8 and m.shape == (N, M_DIM)
    assert set(timer.t) == {"source", "audio", "motion", "render", "paste"}
    crops, m2, _ = talker.run(SRC, WAV, N, start="source", paste=False, batch=4)
    assert crops.shape == (N, 32, 32, 3) and torch.allclose(m, m2)


def test_ramp_scales_the_rotation_about_the_photo(monkeypatch, tiny_ckpt):
    codec, _ = fake_codec(monkeypatch)
    runs = {}
    for amp in (1.0, 2.0):
        t = Talker(tiny_ckpt, f"g=2,steps=3,ramp={amp:g}", device="cpu", codec=codec, wavlm=FakeWavLM())
        _, m, _ = t.run(SRC, WAV, N, start="source", paste=False)
        runs[amp] = to_target(m.float())
        assert torch.allclose(to_target(t.sampled.float()), runs[1.0], atol=1e-3)
    r0 = to_target(codec.source_motion(SRC)["m"])[0, :3]
    assert torch.allclose(runs[2.0][:, :3] - r0, 2 * (runs[1.0][:, :3] - r0), atol=1e-3)
    assert torch.allclose(runs[2.0][:, 3:], runs[1.0][:, 3:], atol=1e-3)


def test_gft_and_latent_runs_train_and_talk(monkeypatch, tiny_cache):
    gft, log = train(tiny_cache, "gft", "gft=true")
    assert "GFT" in log and "EVAL step 3" in log
    lat, log = train(tiny_cache, "lat", f"latent={tiny_cache / 'pca.pt'}", "latent_k=16", "gft=true")
    assert "target 16-d" in log and "std_r_place" in log
    codec, _ = fake_codec(monkeypatch)
    for ckpt in (gft, lat):
        t = Talker(ckpt, "g=2,steps=3", device="cpu", codec=codec, wavlm=FakeWavLM())
        frames, m, _ = t.run(SRC, WAV, N, start="source")
        assert frames.shape == (N, 100, 120, 3) and torch.isfinite(m).all()
    ck = torch.load(lat, map_location="cpu", weights_only=False)
    assert ck["latent"]["K"] == 69 and ck["cfg"]["latent_k"] == 16
    with pytest.raises(ValueError):
        Talker(lat, "g=2,ramp=1.5", device="cpu", codec=codec, wavlm=FakeWavLM()).run(SRC, WAV, N)
    with pytest.raises(ValueError):
        Talker(lat, "mouth=1.25", device="cpu", codec=codec, wavlm=FakeWavLM())

