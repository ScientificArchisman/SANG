"""HDTF download parsing, test-set selection, metric plumbing and the results table.

CPU only; no network, weights, decord or data."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sang.bench import frechet
from sang.motion import paste_back
from sang.sota_hdtf import METRICS, SOTA, markdown_table


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dl, ev = _load("download_hdtf"), _load("eval_hdtf")


def test_time_and_crop_parsing(tmp_path):
    assert dl.to_seconds("01:35") == 95.0 and dl.to_seconds("1:02:03") == 3723.0
    ann = tmp_path
    (ann / "RD_video_url.txt").write_text("﻿Radio11 https://www.youtube.com/watch?v=x\nRadio24 https://y\n")
    (ann / "RD_resolution.txt").write_text("Radio11.mp4 720\n")
    (ann / "RD_annotion_time.txt").write_text("Radio11.mp4 00:30-01:00 01:30-02:30\n")
    (ann / "RD_crop_wh.txt").write_text("Radio11_0.mp4 412 599 0 599\nRadio11_1.mp4 557 478 0 478\n")
    v = {x["name"]: x for x in dl.parse_annotations(ann, ("RD",))}
    assert v["Radio11"]["clips"] == [(30.0, 60.0), (90.0, 150.0)]
    assert v["Radio11"]["crops"] == {0: (412, 599, 0, 599), 1: (557, 478, 0, 478)}
    assert v["Radio24"]["clips"] == []                                   # URL without time stamps
    assert dl.crop_filter((412, 599, 0, 599), 720, 1280, 720) == "crop=598:598:412:0,scale=512:512:flags=lanczos"
    assert dl.crop_filter((483, 390, 0, 390), 720, 1920, 1080).startswith("crop=584:584:724:0")   # 1080p download


def test_test_list_one_clip_per_video_stable_order(tmp_path):
    for n in ("A_0", "A_1", "B_1", "B_0", "C_0", "D_2"):
        (tmp_path / f"{n}.mp4").write_bytes(b"")
    picked = ev.test_list(tmp_path, 10)
    assert sorted(p.stem for p in picked) == ["A_0", "B_0", "C_0", "D_2"]   # one per video, lowest index
    assert [p.stem for p in ev.test_list(tmp_path, 2)] == [p.stem for p in picked[:2]]


def test_frechet_and_paste_back():
    r = np.random.default_rng(0)
    a = r.normal(size=(400, 6))
    assert abs(frechet(a, a)) < 1e-3 and abs(frechet(a, a + 1) - 6) < 1e-2
    assert np.isnan(frechet(a[:1], a))
    M = np.array([[0.5, 0, 10], [0, 0.5, 20], [0, 0, 1.0]])               # 512 crop -> a 256 box at (10, 20)
    out = paste_back(np.full((1, 512, 512, 3), 255, np.uint8), M, np.zeros((300, 300, 3), np.uint8))
    assert out.shape == (1, 300, 300, 3) and out[0, 150, 150, 0] == 255 and out[0, 5, 5, 0] == 0


def test_lse_survives_non_ascii_syncnet_output(tmp_path, monkeypatch):
    """Every HDTF clip was skipped with UnicodeDecodeError: syncnet's output was decoded with the node's
    ASCII locale. Progress-bar glyphs and even invalid bytes must not stop the three numbers parsing."""
    import sang.bench as bench
    fake = tmp_path / "syncnet_python"
    fake.mkdir()
    (fake / "run_pipeline.py").write_text(r"import sys; sys.stderr.buffer.write(b'faces \xe2\x96\x88\xe2\x96\x88 100% \xff\n')")
    (fake / "run_syncnet.py").write_text(
        r"import sys; sys.stderr.buffer.write(b'2026-10-07 10:01:02 INFO AV offset: \t-1 \xe2\x80\xa6\n"
        r"2026-10-07 10:01:02 INFO Min dist: \t7.25\n2026-10-07 10:01:02 INFO Confidence: \t6.50\n')")
    monkeypatch.setattr(bench, "SYNCNET", fake)
    assert bench.lse(tmp_path / "clip.mp4", tmp_path / "work") == (-1.0, 7.25, 6.5)


def test_tables():
    assert {k for k, _ in METRICS} >= {"FID", "FVD", "CSIM", "LSE-C", "LSE-D"}
    for r in SOTA:                                                       # every published value is a known metric
        assert set(r["m"]) <= {k for k, _ in METRICS}, r
    summary = {"name": "g2", "protocol": {"seconds": 10, "cfg": None, "cfg_mouth": 1.25},
               "rows": {"real": {"n_clips": 75, "m": {"LSE-C": 8.1, "LSE-D": 6.9, "CSIM": 0.85}},
                        "sang": {"n_clips": 75, "m": {"FID": 12.0, "FVD": 150.0, "LSE-C": 7.0}}}}
    rows = ev.ours_rows([summary, summary])
    assert [r["method"] for r in rows].count("Real video") == 1                 # shared rows appear once
    assert "mouth γ 1.25" in rows[-1]["method"]
    md = markdown_table(rows)
    assert md.splitlines()[2].startswith("| Ours (n=75, 10 s) | Real video")
    assert "| KDTalker Tab. 1 | KDTalker |" in md and "9.756" in md
