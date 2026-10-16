"""scripts/talkvid_inventory.py on a synthetic TalkVid: matching clips on disk to metadata, and a download
list that keeps the val set intact (no val-split video, no val person) and spreads over new persons. CPU only."""
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def tm():
    spec = importlib.util.spec_from_file_location("tm", REPO / "scripts" / "train_motion.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make(tmp: Path, n_vid: int = 40, per_vid: int = 6):
    entries = []
    for v in range(n_vid):
        for k in range(per_vid):
            s = 1.0 + 15 * k
            entries.append({"start-time": s, "end-time": s + 10.0, "height": 1080, "width": 1920, "fps": 24.0,
                            "dover_scores": 8.5, "cotracker_ratio": 0.9,
                            "info": {"Video Link": f"https://www.youtube.com/watch?v=vid{v:02d}",
                                     "Person ID": f"{v // 2}.0" if v % 5 else str(v // 2), "Language": "English" if v % 3 else "Chinese",
                                     "Gender": "Female", "Ethnicity": "Asian", "Age Group": "31-45", "Video Category": "x"}})
    meta = tmp / "meta.json"
    meta.write_text(json.dumps(entries))
    clips, cache = tmp / "clips", tmp / "cache"
    rows = []
    for v in range(8):                                        # we have the first 3 clips of 8 videos
        d = clips / f"vid{v:02d}"
        d.mkdir(parents=True)
        for k in range(3):
            s = 1.0 + 15 * k
            stem = f"vid{v:02d}_NA_{s:.3f}_{s + 10:.3f}"
            (d / f"{stem}.mp4").write_bytes(b"\0" * 1000)
            (d / f"{stem}.m4a").write_bytes(b"\0" * 100)
            if k < 2:                                         # 2 of them cached, 6 s each
                rows.append({"clip": str(d / f"{stem}.mp4"), "path": str(cache / f"vid{v:02d}" / f"{stem}.pt"), "n": 150})
    cache.mkdir()
    (cache / "index_000.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    cfg = tmp / "cfg.yaml"
    cfg.write_text("val_frac: 0.3\nseed: 0\n")
    return meta, clips, cache, cfg, entries


def test_inventory_and_download_list(tmp_path):
    meta, clips, cache, cfg, entries = make(tmp_path)
    sel, rep = tmp_path / "sel.json", tmp_path / "rep.json"
    target = 16 * 6 / 3600 + 0.05                             # cached now + 0.05 h (18 s) -> 0.065 h with overshoot
    res = subprocess.run([sys.executable, str(REPO / "scripts" / "talkvid_inventory.py"), "--meta", str(meta),
                          "--clips", str(clips), "--cache", str(cache), "--config", str(cfg), "--probe", "0",
                          "--target-hours", str(target), "--select", str(sel), "--out", str(rep)],
                         capture_output=True, encoding="utf-8", cwd=REPO)
    assert res.returncode == 0, res.stderr[-3000:]
    r = json.loads(rep.read_text())
    assert r["talkvid"]["clips"] == 240 and abs(r["talkvid"]["hours"] - 240 * 10 / 3600) < 1e-9
    assert r["talkvid"]["persons"] == 20                      # '3.0' and '3' are one person
    assert r["ours"]["clips_on_disk"] == r["ours"]["matched"] == 24
    assert r["ours"]["cached_clips"] == 16 and r["ours"]["on_disk_not_cached"] == 8
    assert abs(r["ours"]["cut_off_hours"] - 16 * 4 / 3600) < 1e-9

    picked = json.loads(sel.read_text())
    assert picked and all({"info", "start-time", "end-time"} <= set(e) for e in picked)
    vid = lambda e: e["info"]["Video Link"].split("=")[1]
    person = lambda e: e["info"]["Person ID"].removesuffix(".0")
    have = {f"vid{v:02d}" for v in range(8)}
    tr, va = tm().split_by_speaker([{"path": f"/x/vid{v:02d}/c.pt"} for v in range(40)], 0.3, 0)
    val_vids = {Path(x["path"]).parent.name for x in va}
    assert val_vids and len(val_vids) < 40
    val_persons = {str(int(v[3:]) // 2) for v in val_vids & have}
    known = {str(int(v[3:]) // 2) for v in have}
    for e in picked:
        assert vid(e) not in val_vids                        # the val set stays as it is
        assert person(e) not in val_persons | known          # new persons only
    assert sum(min(e["end-time"] - e["start-time"], 20) for e in picked) >= 0.05 * 1.3 * 3600
    n = Counter(person(e) for e in picked)
    assert max(n.values()) - min(n.values()) <= 1            # one clip per person per round


def test_why_not_cached_and_cache_list(tmp_path):
    meta, clips, cache, cfg, _ = make(tmp_path)
    cfg.write_text("val_frac: 0.3\nseed: 9\n")                  # vid01 in val (so vid00's person too); 2-7 train
    name = lambda v: f"vid{v:02d}_NA_31.000_41.000.mp4"         # the uncached (third) clip of video v
    path = lambda v: str(clips / f"vid{v:02d}" / name(v))
    every = [str(p) for p in sorted(clips.glob("*/*.mp4"))]
    queued = [p for p in every if not p.startswith((str(clips / "vid06"), str(clips / "vid07")))]
    kept = [p for p in queued if Path(p).name not in (name(2), name(3))]
    (tmp_path / "all.txt").write_text("\n".join(queued))
    (tmp_path / "kept.txt").write_text("\n".join(kept))
    (tmp_path / "report.jsonl").write_text(json.dumps({"clip": path(2), "reject": "face too small"}) + "\n")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "x.out").write_text(f"  skip {name(4)}: ValueError: only 50 frames before a shot cut\n")
    out = tmp_path / "cache_list.txt"
    res = subprocess.run([sys.executable, str(REPO / "scripts" / "talkvid_inventory.py"), "--meta", str(meta),
                          "--clips", str(clips), "--cache", str(cache), "--config", str(cfg), "--probe", "0",
                          "--queued", str(tmp_path / "all.txt"), "--filtered", str(tmp_path / "kept.txt"),
                          "--filter-report", str(tmp_path / "report.jsonl"), "--cache-logs", str(tmp_path / "logs/*.out"),
                          "--cache-list", str(out), "--chunk-list", str(tmp_path / "chunks.txt"),
                          "--out", str(tmp_path / "rep.json")],
                         capture_output=True, encoding="utf-8", cwd=REPO)
    assert res.returncode == 0, res.stderr[-3000:]

    _, va = tm().split_by_speaker([{"path": f"/x/vid{v:02d}/c.pt"} for v in range(40)], 0.3, 9)
    val_vids = {Path(x["path"]).parent.name for x in va}
    assert {v for v in range(8) if f"vid{v:02d}" in val_vids} == {1}
    val_persons = {int(v[3:]) // 2 for v in val_vids if int(v[3:]) < 8}
    expect = {}
    for v in range(8):
        if f"vid{v:02d}" in val_vids:
            expect[v] = "left out: val-split video"
        elif v // 2 in val_persons:
            expect[v] = "left out: person in our val set"
        else:
            expect[v] = {6: "never queued (not filtered yet)", 7: "never queued (not filtered yet)",
                         2: "filter rejected: face too small", 3: "filter rejected: decode / no face",
                         4: "cache failed: ValueError: only N frames before a shot cut"}.get(v, "passed the filter, never cached")
    got = json.loads((tmp_path / "rep.json").read_text())["not_cached"]
    assert {k: v["clips"] for k, v in got.items()} == dict(Counter(expect.values()))
    assert len(set(expect.values())) == 7                      # every class is exercised
    usable = sorted(path(v) for v, why in expect.items() if why.startswith(("never queued", "passed")))
    assert sorted(l for l in out.read_text().splitlines() if l) == usable
    # cached: 6 s of each 10 s clip (k = 0, 1) -> 4 s left per clip; train side = videos 2-7
    long_ = sorted(str(clips / f"vid{v:02d}" / f"vid{v:02d}_NA_{1 + 15 * k:.3f}_{11 + 15 * k:.3f}.mp4")
                   for v in range(2, 8) for k in range(2))
    assert sorted(l for l in (tmp_path / "chunks.txt").read_text().splitlines() if l) == long_
    ch = {c["K"]: c for c in json.loads((tmp_path / "rep.json").read_text())["chunks"]}
    assert ch[1]["from_cached_h"] == 0 and abs(ch[2]["from_cached_h"] - 12 * 4 / 3600) < 1e-9
    assert abs(ch[1]["from_usable_h"] - len(usable) * 10 / 3600) < 1e-9
