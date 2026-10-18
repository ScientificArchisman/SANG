"""scripts/sync_offsets.py after the cache grows (cache_motion.py --chunks): only videos without enough
measurements are measured, and merge gives the new segments of a measured video its offset. CPU only."""
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.sync import clip_key, merge


def so():
    spec = importlib.util.spec_from_file_location("so", REPO / "scripts" / "sync_offsets.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def row(video, stem, n=250, start=0):
    return {"clip": f"/c/{video}/{stem}.mp4", "path": f"/cache/{video}/{stem}.pt", "n": n, "start": start}


def test_only_new_videos_are_measured_and_old_offsets_carry_over():
    rows = [row("A", "A_NA_1.000_60.000"), row("A", "A_NA_70.000_90.000"),            # measured twice before
            row("A", "A_NA_c1_1.000_60.000", start=250), row("A", "A_NA_c2_1.000_60.000", start=500),
            row("B", "B_NA_1.000_30.000"), row("B", "B_NA_c1_1.000_30.000", start=250),  # measured once
            row("C", "C_NA_5.000_40.000"), row("C", "C_NA_c1_5.000_40.000", start=250),  # new video
            row("D", "D_NA_1.000_4.000", n=80)]                                           # too short for SyncNet
    measured = [{"key": "A/A_NA_1.000_60.000", "video": "A", "offset": 1.0, "conf": 6.0},
                {"key": "A/A_NA_70.000_90.000", "video": "A", "offset": 1.5, "conf": 5.0},
                {"key": "B/B_NA_1.000_30.000", "video": "B", "offset": -1.0, "conf": 7.0}]
    todo = so().to_measure(rows, measured, per_video=2, min_frames=101)
    assert sorted(clip_key(r) for r in todo) == ["B/B_NA_c1_1.000_30.000", "C/C_NA_5.000_40.000", "C/C_NA_c1_5.000_40.000"]

    res = merge(measured, rows)["clips"]
    assert res["A/A_NA_c2_1.000_60.000"]["keep"] and res["A/A_NA_c2_1.000_60.000"]["offset"] == 1.25
    assert res["A/A_NA_c2_1.000_60.000"]["shift_ticks"] == 2 and res["B/B_NA_c1_1.000_30.000"]["offset"] == -1.0
    assert not res["C/C_NA_c1_5.000_40.000"]["keep"]                                 # until it is measured
