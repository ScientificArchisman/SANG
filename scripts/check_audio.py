#!/usr/bin/env python3
"""Does every cached segment have its audio? Compares each segment's end with the length of its clip's .m4a.

    python scripts/check_audio.py                 # CPU, a few minutes with --workers 16
    # -> <cache>/audio_check.json; then: python scripts/sync_offsets.py --merge --exclude <cache>/audio_check.json

cache_motion.py reads a segment's audio from the clip's separate .m4a and pads with silence where the file
ends, so a segment past the end of the audio is a moving mouth with silent audio. With --chunks, segments
reach minutes into long clips (2026-10-18: SyncNet found no audio stream in a segment ~180 s into a 235 s
clip). For every distinct clip in the index: the .m4a's duration and start time from ffmpeg's banner. Per
segment (index row): end = (start + n) / 25 s; 'none' if it starts after the audio ends, 'partial' if the audio
ends more than --tol s before the segment does. Summary by index file group (index_000-style = cached before
--chunks, index_c10 = the --chunks run).
"""
import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.paths import MOTION_CACHE
from sang.sync import clip_key

FPS = 25


def probe(path: str, ffmpeg: str):
    """(duration s, start s) of a media file from ffmpeg's banner; None if it cannot be read."""
    r = subprocess.run([ffmpeg, "-hide_banner", "-i", path], capture_output=True, encoding="utf-8", errors="replace")
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?), start: (-?\d+(?:\.\d+)?)", r.stderr)
    if not m:
        return None
    h, mi, s, st = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s), float(st)


def classify(row: dict, audio, tol: float) -> str:
    """'ok', 'partial', 'none' (segment starts after the audio ends) or 'unreadable'."""
    if audio is None:
        return "unreadable"
    dur = audio[0]
    s, e = row.get("start", 0) / FPS, (row.get("start", 0) + row["n"]) / FPS
    if s >= dur - tol:
        return "none"
    return "partial" if e > dur + tol else "ok"


def group_of(index_file: Path) -> str:
    tag = index_file.stem[len("index_"):]
    return "before --chunks" if tag.isdigit() else tag.rsplit("_", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(MOTION_CACHE))
    ap.add_argument("--tol", type=float, default=0.5, help="seconds of missing audio tolerated at a segment's end")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default=None, help="default <cache>/audio_check.json")
    args = ap.parse_args()
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"

    cache = Path(args.cache)
    rows = []
    for f in sorted(cache.glob("index_*.jsonl")):
        g = group_of(f)
        rows += [{**json.loads(l), "_group": g} for l in f.read_text().splitlines() if l.strip()]
    clips = sorted({r["clip"] for r in rows})
    print(f"{len(rows):,} segments from {len(clips):,} clips; probing the .m4a files with {args.workers} workers", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        audio = dict(zip(clips, ex.map(lambda c: probe(str(Path(c).with_suffix(".m4a")), ffmpeg), clips)))

    stats = defaultdict(lambda: Counter())
    hours = defaultdict(lambda: Counter())
    bad, bad_clips = {}, set()
    for r in rows:
        a = audio[r["clip"]]
        c = classify(r, a, args.tol)
        stats[r["_group"]][c] += 1
        hours[r["_group"]][c] += r["n"] / FPS / 3600
        if c != "ok":
            bad_clips.add(r["clip"])
            k = r.get("start", 0) / FPS
            bad[clip_key(r)] = {"why": c, "segment_s": [round(k, 2), round(k + r["n"] / FPS, 2)],
                                "audio_s": round(a[0], 2) if a else None}
    starts = Counter(round(a[1], 1) for a in audio.values() if a)
    print(f"\n{'group':<18}{'segments':>10}{'ok':>9}{'partial':>9}{'none':>7}{'unreadable':>12}{'hours bad':>11}{'of hours':>10}")
    for g in sorted(stats):
        s, h = stats[g], hours[g]
        print(f"{g:<18}{sum(s.values()):>10,}{s['ok']:>9,}{s['partial']:>9,}{s['none']:>7,}{s['unreadable']:>12,}"
              f"{h['partial'] + h['none'] + h['unreadable']:>11.2f}{sum(h.values()):>10.2f}")
    print(f"\n.m4a start times (s, rounded): {dict(starts.most_common(6))}")
    out = Path(args.out) if args.out else cache / "audio_check.json"
    out.write_text(json.dumps({"tol": args.tol, "segments": len(rows), "clips": len(clips),
                               "summary": {g: dict(stats[g]) for g in stats}, "bad": bad}, indent=1))
    print(f"{len(bad):,} segments without full audio, in {len(bad_clips):,} clips -> {out}")


if __name__ == "__main__":
    main()

