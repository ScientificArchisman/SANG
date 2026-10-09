"""sang/paths.py: the motion cache can move without rewriting its index files."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sang import paths


def test_index_paths_follow_the_cache_and_missing_cache_falls_back(tmp_path, monkeypatch):
    new = tmp_path / "archi_data" / "talkvid" / "motion_lp"
    new.mkdir(parents=True)
    old = "/beegfs/work/someone/SANG/cache/motion_lp"
    rows = [{"clip": "/talkvid/vidA/c0.mp4", "path": f"{old}/vidA/c0.pt", "n": 250, "start": 0},
            {"clip": "/talkvid/vidB/c3.mp4", "path": f"{old}/vidB/c3.pt", "n": 120, "start": 5}]
    (new / "index_000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = paths.load_index(new)
    assert [r["path"] for r in got] == [str(new / "vidA" / "c0.pt"), str(new / "vidB" / "c3.pt")]
    assert [r["clip"] for r in got] == [r["clip"] for r in rows]                       # source clips untouched

    monkeypatch.setattr(paths, "MOTION_CACHE", new)
    assert paths.motion_cache("cache/motion_lp_gone") == new                           # old checkpoint configs
    assert paths.motion_cache(str(new)) == new and paths.motion_cache(None) == new
