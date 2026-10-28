"""sang.motion: ONNX Runtime sessions get an explicit thread count (the Slurm affinity failure), the report says
where the face models run, and decode_clip serves short requests from long clips. Stand-in onnxruntime and
video reader, CPU only."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import sang.motion as motion


def fake_ort():
    ort = type(sys)("onnxruntime")
    ort.__version__ = "9.9"

    class SessionOptions:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    class InferenceSession:
        def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kw):
            self.opts, self.providers = sess_options, providers or ["CPUExecutionProvider"]

        def get_providers(self):
            return self.providers

    class Pickable(InferenceSession):
        def __init__(self, path, **kw):
            super().__init__(path, **kw)

    ort.SessionOptions, ort.InferenceSession, ort.Pickable = SessionOptions, InferenceSession, Pickable
    return ort


def test_sessions_get_an_explicit_thread_count(monkeypatch):
    ort = fake_ort()
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    with motion.ort_threads(6):
        a = ort.Pickable("det.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        b = ort.InferenceSession("lmk.onnx", providers=["CUDAExecutionProvider"])
        own = ort.SessionOptions()
        c = ort.InferenceSession("x.onnx", own)
    assert a.opts.intra_op_num_threads == b.opts.intra_op_num_threads == 6
    assert a.opts.inter_op_num_threads == 1 and a.providers[0] == "CUDAExecutionProvider"
    assert c.opts is own and own.intra_op_num_threads == 0
    assert ort.InferenceSession("y.onnx").opts is None


def test_report_flags_face_models_on_the_cpu(monkeypatch):
    ort = fake_ort()
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    S = lambda p: ort.InferenceSession("m", providers=[p])
    cropper = type("C", (), {})()
    cropper.face_analysis_wrapper = type("F", (), {"models": {"detection": type("M", (), {"session": S("CUDAExecutionProvider")})()}})()
    cropper.human_landmark_runner = type("L", (), {"session": S("CUDAExecutionProvider")})()
    assert "insightface detection on CUDA" in motion.ort_report(cropper, 8) and "slow" not in motion.ort_report(cropper, 8)
    cropper.human_landmark_runner.session = S("CPUExecutionProvider")
    assert "landmark.onnx on CPU" in motion.ort_report(cropper, 8) and "onnxruntime-gpu" in motion.ort_report(cropper, 8)


def test_decode_clip_serves_short_requests(monkeypatch):
    class Batch:
        def __init__(self, idx):
            self.idx = idx

        def asnumpy(self):
            return np.array(self.idx)[:, None, None, None] * np.ones((1, 2, 2, 3), np.int64)

    class Reader:
        def __init__(self, n, fps):
            self.n, self.fps = n, fps

        def __len__(self):
            return self.n

        def get_avg_fps(self):
            return self.fps

        def get_batch(self, idx):
            return Batch(idx)

    video = type(sys)("sang.video")
    video.open_video = lambda path: Reader(int(path.split(":")[0]), float(path.split(":")[1]))
    monkeypatch.setitem(sys.modules, "sang.video", video)
    assert motion.decode_clip("2226:25", 1).shape == (1, 2, 2, 3)
    f = motion.decode_clip("600:50", 10, start=100)
    assert f[:, 0, 0, 0].tolist() == list(range(200, 220, 2))
    assert len(motion.decode_clip("600:25", 500, start=550)) == 50
    with pytest.raises(ValueError, match="too short"):
        motion.decode_clip("600:25", 500, start=595)
    with pytest.raises(ValueError, match="too short"):
        motion.decode_clip("5:25", 500)

