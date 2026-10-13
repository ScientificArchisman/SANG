"""sang.pipeline.Talker end to end on CPU: a tiny trained SANG (with the audio prior), the real MotionCodec
render code with a stand-in LivePortrait (tests/test_fastpath.py), a stand-in WavLM."""
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

from sang.fastpath import Timer
from sang.motion import M_DIM, MotionCodec, rotation_matrix
from sang.pipeline import Opts, Talker
from test_fastpath import FakeLP


class FakeWavLM:
    def encode(self, wav):                                   # [1, 1, N] 16 kHz -> [1, 2 * frames, 1024]
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
def tiny_ckpt(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pipe")
    cache = tmp / "cache"
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
    over = [f"cache_dir={cache}", f"out_dir={tmp / 'run'}", "dim=32", "layers=1", "heads=2", "max_steps=3",
            "warmup_steps=1", "eval_every=3", "log_every=1", "batch_size=2", "workers=0", "eval_batches=1",
            "sample_steps=2", "val_frac=0.3", "windows_per_clip=2", "prior=true", "prior_width=16"]
    log = subprocess.run([sys.executable, str(REPO / "scripts" / "train_motion.py"), *over],
                         capture_output=True, encoding="utf-8", cwd=REPO)
    assert log.returncode == 0, log.stderr[-2000:]
    return str(tmp / "run" / "last.pt")


def test_fast_pipeline_matches_the_original_and_streams(monkeypatch, tiny_ckpt):
    codec, calls = fake_codec(monkeypatch)
    talker = Talker(tiny_ckpt, "g=2,mouth=1.25,tau=0.5,avg=2,steps=3", device="cpu", codec=codec, wavlm=FakeWavLM())
    src = np.random.default_rng(0).integers(0, 256, (100, 120, 3), dtype=np.uint8)
    wav = torch.randn(16000 * 3) * 0.1
    n = 75
    calls["detect"] = 0
    slow, m_slow, t_slow = talker.run(src, wav, n, Opts(), start="source")
    assert calls["detect"] == 3                              # the original path finds the face three times
    calls["detect"] = 0
    got = []
    fast, m_fast, t_fast = talker.run(src, wav, n, Opts(cfg_batch=True, render_batch=8, half=True, gpu_paste=True,
                                                        reuse_crop=True), start="source", sink=got.append)
    assert calls["detect"] == 1                              # once
    assert slow.shape == fast.shape == (n, 100, 120, 3)
    assert (m_slow - m_fast).abs().max() < 1e-4              # same motion up to float rounding
    assert np.abs(slow.astype(int) - fast.astype(int)).max() <= 3
    assert sum(len(b) for b in got) == n and len(got) == -(-n // 8)   # streamed in render batches
    assert set(t_fast.t) == {"source", "audio", "motion", "render", "paste"}
    none, _, _ = talker.run(src, wav, n, Opts.fast(), keep=False, sink=lambda b: None)
    assert none is None
