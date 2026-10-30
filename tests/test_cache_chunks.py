"""cache_motion.py --chunks: consecutive segments of a long clip, cut at shot cuts, resumable. The video
decoder and LivePortrait are stand-ins, so this runs on CPU."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def cm():
    spec = importlib.util.spec_from_file_location("cm", REPO / "scripts" / "cache_motion.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fake(T: int, cuts=()):
    """A T-frame clip with shot cuts before the given frames: decode() and extract() as in cache_motion."""
    calls = []

    def decode(clip, max_frames, start=0):
        n = min(max_frames, T - start)
        if n < 8:
            raise ValueError("too short")
        calls.append(start)
        return np.arange(start, start + n)

    def extract(frames):
        s = int(frames[0])
        stop = next((c - s for c in sorted(cuts) if s < c < s + len(frames)), len(frames))
        return {"m": torch.zeros(stop, 70), "span": (0, stop)}

    return decode, extract, calls


def run(T, chunks, cuts=(), prev=None, max_frames=250, min_frames=74):
    decode, extract, calls = fake(T, cuts)
    out = [(k, s, d["span"][1] - d["span"][0]) for k, s, d in
           cm().segments("v/clip.mp4", prev or {}, chunks, max_frames, min_frames, decode, extract)]
    return out, calls


def test_one_chunk_is_the_old_behaviour():
    assert run(600, 1)[0] == [(0, 0, 250)]
    assert run(100, 1)[0] == [(0, 0, 100)]
    with pytest.raises(ValueError, match="only 50 frames before a shot cut"):
        run(600, 1, cuts=(50,))
    with pytest.raises(ValueError, match="too short"):
        run(5, 1)


def test_chunks_follow_each_other_and_skip_short_shots():
    got, _ = run(600, 4, cuts=(300, 330))
    assert got == [(0, 0, 250), (1, 330, 250)]
    assert run(600, 2)[0] == [(0, 0, 250), (1, 250, 250)]
    assert run(1000, 10)[0][-1] == (3, 750, 250)


def test_resume_adds_only_later_chunks():
    got, calls = run(800, 3, prev={0: {"start": 0, "n": 240}})
    assert got == [(1, 240, 250), (2, 490, 250)] and calls == [240, 490]
    got, calls = run(800, 2, prev={0: {"start": 0, "n": 240}, 1: {"start": 240, "n": 250}})
    assert got == [] and calls == []


def test_chunk_names_keep_the_clip_times():
    m = cm()
    assert m.chunk_stem("ab-c_d_NA_1.500_31.000", 0) == "ab-c_d_NA_1.500_31.000"
    assert m.chunk_stem("ab-c_d_NA_1.500_31.000", 2) == "ab-c_d_NA_c2_1.500_31.000"
    assert m.chunk_stem("noTimes", 1) == "noTimes_c1"


def test_main_writes_chunks_resumes_from_any_index(tmp_path, monkeypatch):
    import json
    m = cm()
    decode, extract, _ = fake(600)
    monkeypatch.setattr(m, "decode_clip", lambda clip, max_frames, start=0: decode(clip, max_frames, start))

    class Codec:
        def __init__(self, device):
            pass

        def extract(self, frames, timings=None):
            d = extract(frames)
            return {**d, "kp": torch.zeros(len(d["m"]), 63)}

    class WavLM:
        def encode(self, x):
            return torch.zeros(1, x.shape[-1] // 320, 1024)

    codec_mod = type(sys)("sang.data.audio")
    codec_mod.load_wavlm = lambda device: WavLM()
    codec_mod.load_wav = lambda path, s, n: torch.zeros(n * 640)
    monkeypatch.setitem(sys.modules, "sang.data.audio", codec_mod)
    monkeypatch.setattr(m, "MotionCodec", Codec)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **k: self)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)

    a, b = "/data/clips/vidA/vidA_NA_0.100_24.100.mp4", "/data/clips/vidB/vidB_NA_5.000_29.000.mp4"
    out = tmp_path / "cache"
    out.mkdir()
    (out / "index_003.jsonl").write_text(json.dumps({"clip": "/old/place/vidA/vidA_NA_0.100_24.100.mp4",
                                                     "path": "x", "n": 240, "start": 0}) + "\n")
    lst = tmp_path / "list.txt"
    lst.write_text(f"{a}\n{b}\n")
    argv = ["cache_motion.py", "--clips", str(lst), "--out", str(out), "--chunks", "3", "--tag", "c3", "--max-frames", "250"]
    monkeypatch.setattr(sys, "argv", argv)
    m.main()
    rows = [json.loads(l) for l in (out / "index_c3_000.jsonl").read_text().splitlines()]
    got = sorted((Path(r["path"]).name, r["start"], r["n"], r.get("chunk", 0)) for r in rows)
    assert got == [("vidA_NA_c1_0.100_24.100.pt", 240, 250, 1), ("vidA_NA_c2_0.100_24.100.pt", 490, 110, 2),
                   ("vidB_NA_5.000_29.000.pt", 0, 250, 0), ("vidB_NA_c1_5.000_29.000.pt", 250, 250, 1),
                   ("vidB_NA_c2_5.000_29.000.pt", 500, 100, 2)]
    assert all((out / Path(r["path"]).parent.name / Path(r["path"]).name).exists() for r in rows)
    d = torch.load(out / "vidB" / "vidB_NA_c1_5.000_29.000.pt", weights_only=True)
    assert d["n"] == 250 and d["audio"].shape == (500, 1024) and d["m"].shape == (250, 70)
    m.main()
    assert len((out / "index_c3_000.jsonl").read_text().splitlines()) == 5

