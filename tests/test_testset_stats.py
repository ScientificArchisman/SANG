"""Frozen HDTF test set (scripts/hdtf_testset.py, eval_hdtf --list csv, ditto_hdtf) and the data
statistics (scripts/data_stats.py). CPU only; no video decoding, models or data."""
import hashlib
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ts, ev, dh, ds = _load("hdtf_testset"), _load("eval_hdtf"), _load("ditto_hdtf"), _load("data_stats")


def _probe(bad=(), short=(), silent=()):
    def probe(stem):
        if stem in bad:
            return None
        return {"frames": 100 if stem in short else 400, "fps": 25.0, "width": 512, "height": 512,
                "audio": stem not in silent}
    return probe


def test_selection_is_deterministic_one_per_video_and_skips_bad_clips():
    stems = ["A_0", "A_1", "B_0", "B_1", "C_0", "D_0", "D_2", "E_0"]
    rows = ts.select(stems, _probe(), 10, 250)
    assert [r["video"] for r in rows] == ts.video_order({"A", "B", "C", "D", "E"})       # md5 order
    assert all(r["clip"].endswith("_0") for r in rows)                                  # lowest clip index
    assert [r["clip"] for r in ts.select(list(reversed(stems)), _probe(), 10, 250)] == [r["clip"] for r in rows]
    rows2 = ts.select(stems, _probe(short={"A_0"}, silent={"C_0"}, bad={"E_0"}), 10, 250)
    got = {r["video"]: r["clip"] for r in rows2}
    assert got["A"] == "A_1" and "C" not in got and "E" not in got                      # next clip, or skip video
    assert [r["rank"] for r in rows2] == list(range(1, len(rows2) + 1))
    assert len(ts.select(stems, _probe(), 2, 250)) == 2


def test_subsets_and_csv_round_trip(tmp_path):
    ann = tmp_path / "annotations"
    ann.mkdir()
    (ann / "RD_video_url.txt").write_text("﻿Radio11.mp4 https://x\n")
    (ann / "WRA_video_url.txt").write_text("KellyAyotte.mp4 https://y\n")
    subs = ts.subsets_from_annotations(ann)
    assert subs == {"Radio11": "RD", "KellyAyotte": "WRA"}
    rows = ts.select(["Radio11_0", "KellyAyotte_0"], _probe(), 5, 250, subs)
    assert {r["clip"]: r["subset"] for r in rows} == {"Radio11_0": "RD", "KellyAyotte_0": "WRA"}
    for r in rows:
        r["crop_rel"] = f"crops512/{r['clip']}.mp4"
    ts.write_csv(rows, tmp_path / "t.csv")
    back = ts.read_csv(tmp_path / "t.csv")
    assert [b["clip"] for b in back] == [r["clip"] for r in rows] and back[0]["frames"] == "400"


def test_eval_and_ditto_read_the_same_csv_and_flag_changed_files(tmp_path, capsys):
    data = tmp_path / "HDTF"
    crops = data / "crops512"
    crops.mkdir(parents=True)
    for s in ("X_0", "Y_0"):
        (crops / f"{s}.mp4").write_bytes(s.encode())
    rows = [{"rank": i + 1, "clip": s, "video": s[0], "subset": "", "crop_rel": f"crops512/{s}.mp4", "frames": 400,
             "fps": 25, "seconds": 16, "width": 512, "height": 512, "audio": "yes",
             "md5": hashlib.md5(s.encode()).hexdigest()} for i, s in enumerate(("X_0", "Y_0"))]
    ts.write_csv(rows, tmp_path / "t.csv")
    clips = ev.read_list(str(tmp_path / "t.csv"), data, crops)
    assert [c.name for c in clips] == ["X_0.mp4", "Y_0.mp4"] and "WARNING" not in capsys.readouterr().out
    assert [c.name for c in dh.clip_paths(str(tmp_path / "t.csv"), data)] == ["X_0.mp4", "Y_0.mp4"]
    (crops / "Y_0.mp4").write_bytes(b"changed")
    ev.read_list(str(tmp_path / "t.csv"), data, crops)
    assert "1 changed" in capsys.readouterr().out
    assert "ditto" in ev.ROWS


def test_ditto_mux_finds_our_ffmpeg_first(tmp_path, monkeypatch):
    import shutil
    fake = tmp_path / "ffmpeg-linux-x86_64-v7"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setattr(dh, "ffmpeg_exe", lambda: str(fake))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    link = dh.ffmpeg_on_path(tmp_path / "_bin")
    assert shutil.which("ffmpeg") == link and Path(link).resolve() == fake.resolve()
    assert dh.ffmpeg_on_path(tmp_path / "_bin") == link                                  # rerun replaces the link


def test_talkvid_metadata_matches_our_clip_names_and_feeds_the_language_tables():
    from sang import talkvid
    entries = [{"id": "video--04ZSRBGcsk-scene3", "start-time": 1150.6236, "end-time": 1155.8458,
                "info": {"Person ID": "17", "Language": "English", "Gender": "Male",
                         "Video Link": "https://www.youtube.com/watch?v=-04ZSRBGcsk"}},
               {"id": "video--04ZSRBGcsk-scene4", "start-time": 1200.0, "end-time": 1206.0,
                "info": {"Person ID": "17", "Language": "English", "Video Link": "https://www.youtube.com/watch?v=-04ZSRBGcsk"}},
               {"id": "video-zzz-scene1", "start-time": 3.0, "end-time": 9.0,
                "info": {"Person ID": "99", "Language": "Chinese", "Video Link": "https://www.youtube.com/watch?v=zzz"}}]
    lk = talkvid.build_lookup(entries)
    assert talkvid.match(lk, "-04ZSRBGcsk", "-04ZSRBGcsk_NA_1150.624_1155.846")["language"] == "English"
    assert talkvid.match(lk, "-04ZSRBGcsk", "-04ZSRBGcsk_NA_1150.900_1155.846") is None        # start too far off
    assert talkvid.match(lk, "nope", "nope_NA_1.0_2.0") is None and talkvid.clip_times("no_times") is None

    rows = [{"clip": "/t/-04ZSRBGcsk/-04ZSRBGcsk_NA_1150.624_1155.846.mp4", "path": "/c/-04ZSRBGcsk/a.pt", "n": 300},
            {"clip": "/t/zzz/zzz_NA_3.000_9.000.mp4", "path": "/c/zzz/b.pt", "n": 100},
            {"clip": "/t/qqq/qqq_NA_1.000_5.000.mp4", "path": "/c/qqq/c.pt", "n": 100}]
    meta = {k: v for k, v in ((ds.clip_key(r), talkvid.match(lk, ds.video_of(r), Path(r["clip"]).stem)) for r in rows) if v}
    langs = {"-04ZSRBGcsk/a": {"lang": "en"}, "zzz/b": {"lang": "ja"}}
    s = ds.split_stats(rows, langs, None, meta)
    assert {b["label"]: b["pct_clips"] for b in s["talkvid"]["language"]} == {"English": 33.33, "Chinese": 33.33, "unmatched": 33.33}
    assert {b["label"]: b["pct_hours"] for b in s["talkvid"]["language"]}["English"] == 60.0
    assert s["talkvid"]["matched"] == 2 and s["talkvid"]["persons"] == 2 and s["talkvid"]["whisper_agrees"] == 0.5
    assert [b["label"] for b in s["language_whisper"]][:2] == ["English", "Japanese"]


def test_data_stats_split_summary():
    rows = [{"clip": f"/v/{v}/c{i}.mp4", "path": f"/c/{v}/c{i}.pt", "n": n}
            for v, ns in (("vidA", [250, 250, 125]), ("vidB", [500]), ("vidC", [100]))
            for i, n in enumerate(ns)]
    langs = {"vidA/c0": {"lang": "en"}, "vidA/c1": {"lang": "en"}, "vidA/c2": {"lang": "zh"}, "vidB/c0": {"lang": "zh"}}
    s = ds.split_stats(rows, langs, None)
    assert s["clips"] == 5 and s["videos"] == 3 and s["frames"] == 1225 and s["hours"] == round(1225 / 25 / 3600, 2)
    assert s["language"]["clips"] == {"en": 2, "zh": 2, "unk": 1}
    assert s["language"]["videos"] == {"en": 1, "zh": 1, "unk": 1}                     # majority per video
    assert {b["bin"]: b["count"] for b in s["clips_per_video"]["hist"]}["3-5"] == 1
    assert s["clip_seconds"]["max"] == 20.0
    assert [b["bin"] for b in ds.bin_counts([1], [(0, 0), (1, 1), (3, 10 ** 9)])] == ["0", "1", ">2"]
