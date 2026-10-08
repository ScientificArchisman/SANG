#!/usr/bin/env python3
"""Freeze the HDTF test set into a CSV, so every method (SANG, Ditto, ...) is scored on exactly the
same clips, the same seconds, the same files.

    python scripts/hdtf_testset.py --data /beegfs/work/$USER/HDTF            # -> configs/hdtf_test.csv
    sbatch ... scripts/eval_hdtf.py --data ... --list configs/hdtf_test.csv   # every eval reads it
    sbatch ... scripts/ditto_hdtf.py --data ... --list configs/hdtf_test.csv

Selection (the same rule eval_hdtf.py's test_list used, plus checks): videos in md5(video name) order;
per video its lowest-index clip that is readable, has an audio stream and at least --seconds of video;
one clip per video; the first --n (default 75, SoulX-FlashHead's HDTF test-set size). The rule depends
only on file names, so it is the same on any machine that has the same download.

Columns: rank, clip, video, subset (RD/WDA/WRA), crop_rel (path under --data), frames, fps, seconds,
width, height, audio, md5 (of the whole file: eval_hdtf.py warns if a file changed).
Commit the CSV (configs/ is tracked) so the laptop and the cluster agree on the test set.
"""
import argparse
import csv
import hashlib
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIELDS = ["rank", "clip", "video", "subset", "crop_rel", "frames", "fps", "seconds", "width", "height", "audio", "md5"]


def split_name(stem: str) -> tuple[str, int]:
    """'WRA_KellyAyotte_000_3' -> ('WRA_KellyAyotte_000', 3); no numeric suffix -> (stem, 0)."""
    vid, _, k = stem.rpartition("_")
    return (vid, int(k)) if vid and k.isdigit() else (stem, 0)


def video_order(videos) -> list[str]:
    return sorted(videos, key=lambda v: hashlib.md5(v.encode()).hexdigest())


def subsets_from_annotations(ann: Path) -> dict[str, str]:
    """video name -> RD / WDA / WRA, from the HDTF annotation files download_hdtf.py fetched."""
    out = {}
    for s in ("RD", "WDA", "WRA"):
        f = ann / f"{s}_video_url.txt"
        if f.exists():
            for line in f.read_text(encoding="utf-8-sig").splitlines():
                if line.split():
                    out[line.split()[0].removesuffix(".mp4")] = s
    return out


def select(stems: list[str], probe, n: int, min_frames: int, subset_of: dict | None = None) -> list[dict]:
    """stems: clip file stems. probe(stem) -> dict(frames, fps, width, height, audio) or None (unreadable).
    -> up to n rows in rank order (one clip per video, lowest passing clip index)."""
    by_video = {}
    for s in stems:
        vid, k = split_name(s)
        by_video.setdefault(vid, []).append((k, s))
    rows = []
    for vid in video_order(by_video):
        for _, stem in sorted(by_video[vid]):
            info = probe(stem)
            if info and info["audio"] and info["frames"] >= min_frames:
                m = re.match(r"^(RD|WDA|WRA)_", stem)
                subset = (subset_of or {}).get(vid) or (m.group(1) if m else "")
                rows.append({"rank": len(rows) + 1, "clip": stem, "video": vid, "subset": subset, **info})
                break
        if len(rows) >= n:
            break
    return rows


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def probe_file(path: Path) -> dict | None:
    """frames, fps, width, height via OpenCV; audio stream presence via ffmpeg's banner."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = cap.read()
    cap.release()
    if not ok or frames <= 0:
        return None
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"
    banner = subprocess.run([ffmpeg, "-hide_banner", "-i", str(path)], capture_output=True,
                            encoding="utf-8", errors="replace").stderr
    return {"frames": frames, "fps": round(fps, 3), "width": w, "height": h, "audio": "Audio:" in banner}


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def read_csv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="HDTF root from scripts/download_hdtf.py")
    ap.add_argument("--crops", default="crops512")
    ap.add_argument("--n", type=int, default=75)
    ap.add_argument("--seconds", type=float, default=10.0, help="minimum clip length; eval uses the first --seconds")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--out", default=str(REPO / "configs" / "hdtf_test.csv"))
    args = ap.parse_args()

    root = Path(args.data)
    crops = root / args.crops
    stems = sorted(p.stem for p in crops.glob("*.mp4"))
    if not stems:
        sys.exit(f"no clips in {crops}")
    rows = select(stems, lambda s: probe_file(crops / f"{s}.mp4"), args.n, int(round(args.seconds * args.fps)),
                  subsets_from_annotations(root / "annotations"))
    for r in rows:
        r["crop_rel"] = f"{args.crops}/{r['clip']}.mp4"
        r["seconds"] = round(r["frames"] / r["fps"], 2) if r["fps"] else ""
        r["audio"] = "yes" if r["audio"] else "no"
        r["md5"] = md5_file(crops / f"{r['clip']}.mp4")
    write_csv(rows, Path(args.out))
    subsets = {s: sum(r["subset"] == s for r in rows) for s in ("RD", "WDA", "WRA")}
    print(f"{len(rows)} clips from {len({r['video'] for r in rows})} videos ({len(stems)} clips on disk) -> {args.out}")
    print(f"subsets {subsets}; shortest {min(r['seconds'] for r in rows)} s; all >= {args.seconds:g} s with audio")
    if len(rows) < args.n:
        print(f"WARNING: only {len(rows)} usable videos (asked for {args.n})")


if __name__ == "__main__":
    main()
