#!/usr/bin/env python3
"""Motion cache: every clip -> LivePortrait motion + canonical keypoints + WavLM features.

One file per clip, laid out as
<out>/<speaker>/<clip>.pt so the speaker-disjoint split works on the cache alone:
    m      [T, 70]    fp16   raw LivePortrait motion (training vectors are derived at load time,
                             so changing them never needs a re-cache)
    kp     [T, 63]    fp16   canonical keypoints x_c, the identity condition
    audio  [2T, 1024] fp16   WavLM-large at 50 Hz = exactly 2 ticks per 25 fps frame
    n      int               frames

Sharded, resumable, one bad clip never kills the job:
    sbatch --array=0-7 bash_scripts/run.sh scripts/cache_motion.py --nshards 8
    (each task reads SLURM_ARRAY_TASK_ID as its shard)

Then `index_<tag>_<shard>.jsonl` per shard lists what landed; train_motion.py reads all index_*.jsonl.

Long clips (--chunks K): a clip gives up to K consecutive segments, each up to --max-frames frames under its
own fixed box and truncated at its first shot cut; the next starts where the last stopped. Chunk 0 keeps the
clip's name (every cache before --chunks), chunk k is <video>_<n>_c<k>_<start>_<end>.pt with "chunk": k in its
index row, and every row's "start" is its first frame in the clip. Rows already in ANY index_*.jsonl are not
redone, so rerunning on cached clips with a larger K only adds their later chunks.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party"))

from sang.data.paths import MOTION_CACHE
from sang.render.liveportrait import MotionCodec, decode_clip

SR, FPS, TPF = 16000, 25, 2


def chunk_stem(stem: str, k: int) -> str:
    """File name of chunk k of a clip: the clip's own name for k = 0, else '<video>_<n>_c<k>_<start>_<end>', so the
    name still ends in the clip's TalkVid start and end times (names without them get '_c<k>')."""
    if k == 0:
        return stem
    parts = stem.rsplit("_", 2)
    try:
        float(parts[1]), float(parts[2])
    except (IndexError, ValueError):
        return f"{stem}_c{k}"
    return f"{parts[0]}_c{k}_{parts[1]}_{parts[2]}"


def clip_id(clip: str) -> tuple[str, str]:
    """(video folder, file name): stable when the clip folder moves."""
    p = Path(clip)
    return p.parent.name, p.name


def segments(clip: str, prev: dict, chunks: int, max_frames: int, min_frames: int, decode, extract):
    """Yield (k, start, d) for each chunk of `clip` still to cache. Chunk k is up to max_frames frames from
    where chunk k-1 stopped, under one fixed box and cut at its first shot cut (extract -> d with 'span');
    start = its first frame in the clip. prev = {k: index row} cached earlier: skipped, and the next chunk
    starts after it. With chunks = 1 a first shot shorter than min_frames is an error, as before --chunks;
    with more, a short shot is skipped and the next segment starts after it."""
    k = offset = tries = 0
    while k < chunks and tries <= 2 * chunks:
        if k in prev:
            offset = int(prev[k].get("start", 0)) + int(prev[k]["n"])
            k += 1
            continue
        tries += 1
        try:
            frames = decode(clip, max_frames, start=offset)
        except ValueError:
            if offset == 0:
                raise
            return
        d = extract(frames)
        a, b = d["span"]
        if b - a < min_frames:
            if chunks == 1:
                raise ValueError(f"only {b - a} frames before a shot cut")
            offset += max(b, 1)
            continue
        yield k, offset + a, d
        offset += b
        k += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(REPO / "data/clips_filtered_all.txt"))
    ap.add_argument("--out", default=str(MOTION_CACHE))
    ap.add_argument("--max-frames", type=int, default=500, help="20 s cap bounds host memory per clip")
    ap.add_argument("--min-frames", type=int, default=74, help="one training window: 64 + 10 prefix")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--chunks", type=int, default=1,
                    help="segments per clip, each up to --max-frames (module docstring); 1 = the first only")
    ap.add_argument("--tag", default="", help="index files index_<tag>_<shard>.jsonl, so runs never share one")
    args = ap.parse_args()

    clips = [l.strip() for l in Path(args.clips).read_text().splitlines() if l.strip()]
    mine = clips[args.shard::args.nshards]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = out / f"index_{args.tag + '_' if args.tag else ''}{args.shard:03d}.jsonl"
    prev_of = {}
    for f in sorted(out.glob("index_*.jsonl")):
        for l in f.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                prev_of.setdefault(clip_id(r["clip"]), {})[int(r.get("chunk", 0))] = r
    full = sum(all(k in prev_of.get(clip_id(c), {}) for k in range(args.chunks)) for c in mine)
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} clips, {full} already cached "
          f"(up to {args.chunks} chunk{'s' if args.chunks > 1 else ''} of {args.max_frames} frames) -> {index.name}",
          flush=True)

    from sang.data.audio import load_wav, load_wavlm
    codec = MotionCodec(device="cuda")
    wavlm = load_wavlm(device="cuda")

    ok = fail = seen = 0
    t0 = time.time()
    tm = {}
    def lap(key, since):
        tm[key] = tm.get(key, 0.0) + time.perf_counter() - since
        return time.perf_counter()

    def decode(clip, max_frames, start=0):
        t = time.perf_counter()
        frames = decode_clip(clip, max_frames, start=start)
        lap("decode", t)
        return frames

    extract = lambda frames: codec.extract(frames, timings=tm)
    with index.open("a") as log:
        for i, clip in enumerate(mine):
            prev = prev_of.get(clip_id(clip), {})
            if all(k in prev for k in range(args.chunks)):
                continue
            seen += 1
            new = 0
            try:
                for k, start, d in segments(clip, prev, args.chunks, args.max_frames, args.min_frames, decode, extract):
                    t = time.perf_counter()
                    n = int(d["m"].shape[0])
                    with torch.no_grad():
                        a = wavlm.encode(load_wav(Path(clip).with_suffix(".m4a"), start, n)[None, None].cuda())[0].float().cpu()
                    t = lap("audio", t)
                    a = a[: TPF * n]
                    if a.shape[0] < TPF * n:
                        a = torch.cat([a, a[-1:].expand(TPF * n - a.shape[0], -1)])
                    dst = out / Path(clip).parent.name / f"{chunk_stem(Path(clip).stem, k)}.pt"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"m": d["m"], "kp": d["kp"], "audio": a.half(), "n": n}, dst)
                    t = lap("save", t)
                    log.write(json.dumps({"clip": clip, "path": str(dst), "n": n, "start": start,
                                          **({"chunk": k} if k else {})}) + "\n")
                    log.flush()
                    ok += 1
                    new += 1
                if not new and not prev:
                    raise ValueError(f"no shot of {args.min_frames} frames")
            except Exception as e:
                fail += 1
                print(f"  skip {Path(clip).name}: {type(e).__name__}: {e}", flush=True)
            if (i + 1) % 50 == 0:
                n_done = max(1, seen)
                split = "  ".join(f"{k_} {v / n_done:.1f}" for k_, v in sorted(tm.items(), key=lambda x: -x[1]))
                print(f"  [{i + 1}/{len(mine)}] ok {ok} chunks, fail {fail} clips  {(time.time() - t0) / n_done:.1f} s/clip"
                      f"  | s/clip by stage: {split}", flush=True)
    n_done = max(1, seen)
    split = "  ".join(f"{k_} {v / n_done:.1f}" for k_, v in sorted(tm.items(), key=lambda x: -x[1]))
    print(f"shard {args.shard} done: {ok} chunks cached from {seen} clips, {fail} clips failed  | s/clip by stage: {split}",
          flush=True)


if __name__ == "__main__":
    main()

