"""Where SANG's data lives: <repo>/data, or the folder in the environment variable SANG_DATA.

    <DATA_ROOT>/talkvid/clips/<video>/<clip>.mp4 + .m4a   segmented TalkVid clips (video and audio)
    <DATA_ROOT>/talkvid/preprocessed/                    scripts/preprocess/cache_motion.py: per clip LivePortrait
                                                         motion, keypoints and WavLM features, index_*.jsonl, and the
                                                         derived files (audio_check.json, sync_offsets.json, norm42.pt)
"""
import json
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("SANG_DATA", Path(__file__).resolve().parents[2] / "data"))
MOTION_CACHE = DATA_ROOT / "talkvid" / "preprocessed"


def load_index(cache_dir: Path) -> list[dict]:
    """All index_*.jsonl rows of a motion cache. Each row's 'path' is re-rooted at cache_dir
    (<cache>/<speaker>/<clip>.pt): index files written before the move hold the old absolute paths."""
    cache_dir = Path(cache_dir)
    rows = []
    for f in sorted(cache_dir.glob("index_*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"no index_*.jsonl in {cache_dir}; run scripts/preprocess/cache_motion.py first")
    for r in rows:
        p = Path(r["path"])
        r["path"] = str(cache_dir / p.parent.name / p.name)
    return rows

