#!/usr/bin/env python3
"""Measure the audio-video offset of the motion cache with SyncNet, then merge into one file that
training and evaluation read (sang/sync.py has the convention and the merge rule).

    # 1. measure: 2 clips per source video, 3 array tasks, 4 SyncNet runs at a time per GPU (resumable)
    sbatch --array=0-2 --time=12:00:00 --cpus-per-task=8 bash_scripts/job.sh scripts/sync_offsets.py --nshards 3
    # 2. merge (CPU, seconds) -> <cache>/sync_offsets.json
    python scripts/sync_offsets.py --merge

Offsets are constant within a source video (diagnosis job 176657), so 2 clips per video are enough:
the video's mean offset is applied to all its clips, at half-frame resolution. Clips where SyncNet
finds no sync or has confidence < 3 are dropped. Rerun step 1 after the cache grows; finished
clips are skipped.
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.sync import FPS, clip_key, merge, pick_clips, real_clip_mp4, video_of


def load_index(cache: Path) -> list[dict]:
    rows = []
    for f in sorted(cache.glob("index_*.jsonl")):
        rows += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not rows:
        sys.exit(f"no index_*.jsonl in {cache}")
    return rows


def measure(args, cache: Path) -> None:
    from sang.bench import SYNCNET, lse
    if not (SYNCNET / "run_syncnet.py").exists():
        sys.exit(f"no syncnet_python at {SYNCNET} (bash_scripts/install_motion.sh)")
    rows = load_index(cache)
    picks = pick_clips(rows, args.per_video, min_frames=args.min_frames)
    videos = sorted({video_of(r) for r in picks})
    mine = {v for i, v in enumerate(videos) if i % args.nshards == args.shard}
    todo = [r for r in picks if video_of(r) in mine]
    out = cache / "sync" / f"offsets_{args.shard:03d}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["key"] for l in out.read_text().splitlines() if l.strip()} if out.exists() else set()
    todo = [r for r in todo if clip_key(r) not in done]
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"sync_offsets_{os.environ.get('SLURM_JOB_ID', 'local')}_{args.shard}"
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} videos, {len(todo)} clips to measure "
          f"({len(done)} already done), {args.workers} workers, {args.seconds:g} s each -> {out}", flush=True)

    lock, t0, count = threading.Lock(), time.time(), [0]

    def one(r: dict) -> dict:
        k = clip_key(r)
        work = tmp / k.replace("/", "__")
        try:
            mp4 = real_clip_mp4(r, min(int(r["n"]), int(args.seconds * FPS)), work / "clip.mp4")
            off, dist, conf = lse(mp4, work / "syncnet")
        except Exception as e:
            print(f"  {k}: {type(e).__name__}: {e}", flush=True)
            off = dist = conf = float("nan")
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return {"key": k, "video": video_of(r), "offset": off, "dist": dist, "conf": conf}

    with ThreadPoolExecutor(args.workers) as ex, open(out, "a") as f:
        for fut in as_completed([ex.submit(one, r) for r in todo]):
            res = fut.result()
            with lock:
                f.write(json.dumps(res) + "\n")
                f.flush()
                count[0] += 1
                el = time.time() - t0
                print(f"[{count[0]}/{len(todo)}] {res['key']:<45} offset {res['offset']:+.0f}  conf {res['conf']:.2f}  "
                      f"({el:.0f} s, ~{el / count[0] * (len(todo) - count[0]) / 60:.0f} min left)", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"shard {args.shard} done in {time.time() - t0:.0f} s", flush=True)


def do_merge(args, cache: Path) -> None:
    measured = []
    for f in sorted((cache / "sync").glob("offsets_*.jsonl")):
        measured += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    if not measured:
        sys.exit(f"nothing measured under {cache / 'sync'}; run the measuring job first")
    rows = load_index(cache)
    res = merge(measured, rows, min_conf=args.min_conf, max_abs=args.max_abs, max_spread=args.max_spread)
    dst = Path(args.out) if args.out else cache / "sync_offsets.json"
    dst.write_text(json.dumps(res, indent=1))
    m = res["meta"]
    print(f"{len(measured)} measurements -> {dst}")
    print(f"clips: {m['kept']} kept / {m['clips']}; dropped {m['dropped']}")
    print(f"videos: {m['videos']} in cache, {m['videos_measured']} measured, {m['videos_usable']} usable, "
          f"{m['videos_inconsistent']} with clips disagreeing by > {args.max_spread:g} frame")
    print(f"kept clips needing a shift: |offset| >= 1 frame {m['frac_kept_abs_ge1']:.0%}, >= 2 frames {m['frac_kept_abs_ge2']:.0%}")
    print(f"offset histogram (frames, kept clips): {m['offset_hist_kept']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(REPO / "cache/motion_lp"))
    ap.add_argument("--merge", action="store_true", help="merge the measurements into <cache>/sync_offsets.json")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--per-video", type=int, default=2, help="clips measured per source video")
    ap.add_argument("--seconds", type=float, default=8.0, help="segment length given to SyncNet")
    ap.add_argument("--min-frames", type=int, default=101, help="SyncNet needs > 100 frames with a face")
    ap.add_argument("--workers", type=int, default=4, help="SyncNet runs at a time on the one GPU")
    ap.add_argument("--min-conf", type=float, default=3.0)
    ap.add_argument("--max-abs", type=float, default=10.0, help="|offset| at or above this = no sync found")
    ap.add_argument("--max-spread", type=float, default=1.0, help="frames a video's clips may disagree by")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cache = Path(args.cache)
    do_merge(args, cache) if args.merge else measure(args, cache)


if __name__ == "__main__":
    main()
