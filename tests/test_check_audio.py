"""scripts/check_audio.py finds cached segments past the end of their clip's audio, and sync_offsets.py --merge
--exclude drops them from training. ffmpeg is a stand-in; CPU only."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_classify():
    ca = load("check_audio")
    r = lambda start, n: {"start": start, "n": n}
    assert ca.classify(r(0, 500), (20.3, 0.0), 0.5) == "ok"                  # 0-20 s, audio 20.3 s
    assert ca.classify(r(4500, 500), (230.0, 0.0), 0.5) == "ok"
    assert ca.classify(r(4500, 500), (185.0, 0.0), 0.5) == "partial"          # 180-200 s, audio ends at 185
    assert ca.classify(r(4500, 500), (179.0, 0.0), 0.5) == "none"
    assert ca.classify(r(0, 500), None, 0.5) == "unreadable"


def test_check_then_merge_drops_segments_without_audio(tmp_path, monkeypatch):
    ca, so = load("check_audio"), load("sync_offsets")
    cache = tmp_path / "cache"
    (cache / "sync").mkdir(parents=True)
    rows_old = [{"clip": "/c/V/V_NA_0.000_235.000.mp4", "path": "/x/V/V_NA_0.000_235.000.pt", "n": 250, "start": 0}]
    rows_new = [{"clip": "/c/V/V_NA_0.000_235.000.mp4", "path": f"/x/V/V_NA_c{k}_0.000_235.000.pt", "n": 500,
                 "start": 250 + 500 * (k - 1), "chunk": k} for k in range(1, 10)]   # chunk k: 10 + 20(k-1) s on
    (cache / "index_000.jsonl").write_text("\n".join(json.dumps(r) for r in rows_old))
    (cache / "index_c10_000.jsonl").write_text("\n".join(json.dumps(r) for r in rows_new))
    monkeypatch.setattr(ca, "probe", lambda path, ffmpeg: (120.0, 0.0))     # the audio stops at 120 s
    monkeypatch.setattr(sys, "argv", ["check_audio.py", "--cache", str(cache), "--workers", "2"])
    ca.main()
    rep = json.loads((cache / "audio_check.json").read_text())
    # chunk 6 is 110-130 s (partial); chunks 7-9 start at 130, 150, 170 s (none)
    assert {k.split("/")[1].split("_")[2]: v["why"] for k, v in rep["bad"].items()} == \
        {"c6": "partial", "c7": "none", "c8": "none", "c9": "none"}
    assert rep["summary"]["before --chunks"] == {"ok": 1} and rep["summary"]["c10"]["ok"] == 5

    (cache / "sync" / "offsets_000.jsonl").write_text(json.dumps({"key": "V/V_NA_0.000_235.000", "video": "V",
                                                                   "offset": 1.0, "conf": 6.0}))
    args = argparse.Namespace(min_conf=3.0, max_abs=10.0, max_spread=1.0, out=str(tmp_path / "s.json"),
                              exclude=str(cache / "audio_check.json"))
    so.do_merge(args, cache)
    res = json.loads((tmp_path / "s.json").read_text())
    kept = sorted(k.split("/")[1] for k, c in res["clips"].items() if c["keep"])
    assert kept == ["V_NA_0.000_235.000"] + [f"V_NA_c{k}_0.000_235.000" for k in range(1, 6)]
    assert res["meta"]["dropped"] == {"audio partial": 1, "audio none": 3} and res["meta"]["kept"] == 6
