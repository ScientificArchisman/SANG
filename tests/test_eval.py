"""HDTF eval plumbing: the frozen test list, metrics and the results table. CPU only; no network, weights, decord
or data."""
import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sang.eval.metrics import frechet


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("eval_hdtf")


def test_read_list_flags_missing_and_changed_files(tmp_path, capsys):
    (tmp_path / "crops512").mkdir()
    for s in ("X_0", "Y_0"):
        (tmp_path / "crops512" / f"{s}.mp4").write_bytes(s.encode())
    lines = ["clip,crop_rel,md5"] + [f"{s},crops512/{s}.mp4,{hashlib.md5(s.encode()).hexdigest()}" for s in ("X_0", "Y_0")]
    (tmp_path / "t.csv").write_text("\n".join(lines) + "\n")
    clips = ev.read_list(str(tmp_path / "t.csv"), tmp_path)
    assert [c.name for c in clips] == ["X_0.mp4", "Y_0.mp4"] and "WARNING" not in capsys.readouterr().out
    (tmp_path / "crops512" / "Y_0.mp4").write_bytes(b"changed")
    ev.read_list(str(tmp_path / "t.csv"), tmp_path)
    assert "1 changed" in capsys.readouterr().out


def test_the_shipped_test_list_has_75_clips_one_per_video():
    import csv
    rows = list(csv.DictReader(open(REPO / "configs/hdtf_test.csv", encoding="utf-8")))
    assert len(rows) == 75 and len({r["video"] for r in rows}) == 75


def test_frechet_and_paste_back():
    import torch
    from sang.render.pipeline import GpuPaster
    r = np.random.default_rng(0)
    a = r.normal(size=(400, 6))
    assert abs(frechet(a, a)) < 1e-3 and abs(frechet(a, a + 1) - 6) < 1e-2
    assert np.isnan(frechet(a[:1], a))
    M = np.array([[0.5, 0, 10], [0, 0.5, 20], [0, 0, 1.0]])
    out = GpuPaster(M, np.zeros((300, 300, 3), np.uint8), 512, device="cpu")(torch.full((1, 512, 512, 3), 255, dtype=torch.uint8))
    assert out.shape == (1, 300, 300, 3) and out[0, 150, 150, 0] == 255 and out[0, 5, 5, 0] == 0


def test_lse_survives_non_ascii_syncnet_output(tmp_path, monkeypatch):
    """Every HDTF clip was skipped with UnicodeDecodeError: syncnet's output was decoded with the node's
    ASCII locale. Progress-bar glyphs and even invalid bytes must not stop the three numbers parsing."""
    import sang.eval.metrics as bench
    fake = tmp_path / "syncnet_python"
    fake.mkdir()
    (fake / "run_pipeline.py").write_text(r"import sys; sys.stderr.buffer.write(b'faces \xe2\x96\x88\xe2\x96\x88 100% \xff\n')")
    (fake / "run_syncnet.py").write_text(
        r"import sys; sys.stderr.buffer.write(b'2026-10-07 10:01:02 INFO AV offset: \t-1 \xe2\x80\xa6\n"
        r"2026-10-07 10:01:02 INFO Min dist: \t7.25\n2026-10-07 10:01:02 INFO Confidence: \t6.50\n')")
    monkeypatch.setattr(bench, "SYNCNET", fake)
    assert bench.lse(tmp_path / "clip.mp4", tmp_path / "work") == (-1.0, 7.25, 6.5)
    made = tmp_path / "lse_auto"
    monkeypatch.setattr(bench.tempfile, "mkdtemp", lambda prefix: str(made))
    assert bench.lse(tmp_path / "clip.mp4") == (-1.0, 7.25, 6.5) and not made.exists()


def test_table():
    args = argparse.Namespace(name="l42", sampler="g=2,steps=6", gt_latent="runs/latent/pca.pt", gt_k=16)
    rows = {"real": {"n_clips": 75, "m": {"LSE-C": 7.161, "CSIM": 0.9}},
            "gt_motion": {"n_clips": 75, "m": {"FID": 16.1, "FVD": 28.0}},
            "sang": {"n_clips": 75, "m": {"FID": 17.466, "FVD": 37.607, "LSE-C": 7.454}}}
    md = ev.markdown(rows, args).splitlines()
    assert md[0].startswith("| Method | n | FID | FVD | CSIM | LSE-C") and len(md) == 5
    assert md[2] == "| Real video | 75 | – | – | 0.900 | 7.161 | – | – | – |"
    assert md[3].startswith("| Real motion → LivePortrait (ceiling) through 16 codes | 75 | 16.100 | 28.000")
    assert md[4].startswith("| SANG-M (l42: g=2,steps=6) | 75 | 17.466 | 37.607 | – | 7.454")
