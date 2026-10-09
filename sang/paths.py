"""Where SANG's data lives. One place, so moving the data means changing one line (or setting SANG_DATA).

    <DATA_ROOT>/talkvid/clips/<video>/<clip>.mp4 + .m4a   segmented TalkVid clips (video and audio)
    <DATA_ROOT>/talkvid/preprocessed/                    scripts/cache_motion.py: per clip LivePortrait motion,
                                                         keypoints and WavLM features, index_*.jsonl, and the
                                                         derived files (sync_offsets.json, lang_lid.json,
                                                         openness.json, naturalness_stats.json)
"""
import json
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("SANG_DATA", "/beegfs/work_fast/shared/li_shared/archi_data"))
TALKVID_CLIPS = DATA_ROOT / "talkvid" / "clips"
MOTION_CACHE = DATA_ROOT / "talkvid" / "preprocessed"


def motion_cache(path=None) -> Path:
    """A config's cache_dir, or MOTION_CACHE when that path does not exist: checkpoints trained before
    the move store the old relative 'cache/motion_lp'."""
    p = Path(path) if path else MOTION_CACHE
    if p.exists():
        return p
    print(f"cache {p} not found, using {MOTION_CACHE}", flush=True)
    return MOTION_CACHE


def load_index(cache_dir: Path) -> list[dict]:
    """All index_*.jsonl rows of a motion cache. Each row's 'path' is re-rooted at cache_dir
    (<cache>/<speaker>/<clip>.pt): index files written before the move hold the old absolute paths."""
    cache_dir = Path(cache_dir)
    rows = []
    for f in sorted(cache_dir.glob("index_*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit(f"no index_*.jsonl in {cache_dir}; run scripts/cache_motion.py first")
    for r in rows:
        p = Path(r["path"])
        r["path"] = str(cache_dir / p.parent.name / p.name)
    return rows
