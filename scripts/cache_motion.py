#!/usr/bin/env python3
"""Motion cache: every clip -> LivePortrait motion + canonical keypoints + WavLM features.

Run only after scripts/motion_ceiling.py has passed. One file per clip, laid out as
<out>/<speaker>/<clip>.pt so the speaker-disjoint split works on the cache alone:

    m      [T, 70]    fp16   raw LivePortrait motion (the 42-d target is derived at load time,
                             so 42 vs 70 stays an ablation, not a re-cache)
    kp     [T, 63]    fp16   canonical keypoints x_c, the identity condition
    audio  [2T, 1024] fp16   WavLM-large at 50 Hz = exactly 2 ticks per 25 fps frame
    n      int               frames

Sharded, resumable, one bad clip never kills the job:

    sbatch --array=0-7 bash_scripts/job.sh scripts/cache_motion.py --nshards 8
    (each task reads SLURM_ARRAY_TASK_ID as its shard)

Then `index.jsonl` per shard lists what landed; train_motion.py reads all of them.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party"))

from sang.motion import MotionCodec, decode_clip

SR, FPS, TPF = 16000, 25, 2


def load_audio(clip: str, start_frame: int, n_frames: int) -> torch.Tensor:
    """16 kHz mono for exactly frames [start, start + n) -> [1, 1, N]."""
    from decord import AudioReader
    wav = torch.from_numpy(AudioReader(str(Path(clip).with_suffix(".m4a")), sample_rate=SR,
                                       mono=True)[:].asnumpy()).float()
    a0, need = start_frame * SR // FPS, n_frames * SR // FPS
    wav = wav[:, a0:a0 + need]
    if wav.shape[1] < need:
        wav = torch.nn.functional.pad(wav, (0, need - wav.shape[1]))
    return wav.unsqueeze(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(REPO / "data/clips_filtered_all.txt"))
    ap.add_argument("--out", default=str(REPO / "cache/motion_lp"))
    ap.add_argument("--max-frames", type=int, default=500, help="20 s cap bounds host memory per clip")
    ap.add_argument("--min-frames", type=int, default=74, help="one training window: 64 + 10 prefix")
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--audio-encoder", default="wavlm-large")
    args = ap.parse_args()

    clips = [l.strip() for l in Path(args.clips).read_text().splitlines() if l.strip()]
    mine = clips[args.shard::args.nshards]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = out / f"index_{args.shard:03d}.jsonl"
    done = {json.loads(l)["clip"] for l in index.read_text().splitlines()} if index.exists() else set()
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} clips, {len(done)} already cached", flush=True)

    from sang.codec import load_wavlm
    codec = MotionCodec(device="cuda")
    wavlm = load_wavlm(args.audio_encoder, device="cuda")

    ok = fail = 0
    t0 = time.time()
    tm = {}                                               # seconds per stage, summed over clips
    def lap(key, since):
        tm[key] = tm.get(key, 0.0) + time.perf_counter() - since
        return time.perf_counter()
    with index.open("a") as log:
        for k, clip in enumerate(mine):
            if clip in done:
                continue
            dst = out / Path(clip).parent.name / f"{Path(clip).stem}.pt"
            try:
                t = time.perf_counter()
                frames = decode_clip(clip, args.max_frames)
                t = lap("decode", t)
                d = codec.extract(frames, timings=tm)       # fixed box, truncated at a shot cut
                t = time.perf_counter()
                n = int(d["m"].shape[0])
                if n < args.min_frames:
                    raise ValueError(f"only {n} frames before a shot cut")
                with torch.no_grad():
                    a = wavlm.encode(load_audio(clip, d["span"][0], n).cuda())[0].float().cpu()   # [~2n, D]
                t = lap("audio", t)
                a = a[: TPF * n]
                if a.shape[0] < TPF * n:                     # conv edge: pad the last tick(s)
                    a = torch.cat([a, a[-1:].expand(TPF * n - a.shape[0], -1)])
                dst.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"m": d["m"], "kp": d["kp"], "audio": a.half(), "n": n}, dst)
                t = lap("save", t)
                log.write(json.dumps({"clip": clip, "path": str(dst), "n": n, "start": d["span"][0]}) + "\n")
                log.flush()
                ok += 1
            except Exception as e:
                fail += 1
                print(f"  skip {Path(clip).name}: {type(e).__name__}: {e}", flush=True)
            if (k + 1) % 50 == 0:
                rate = (time.time() - t0) / max(1, ok + fail)
                n_done = max(1, ok + fail)
                split = "  ".join(f"{k_} {v / n_done:.1f}" for k_, v in sorted(tm.items(), key=lambda x: -x[1]))
                print(f"  [{k + 1}/{len(mine)}] ok {ok} fail {fail}  {rate:.1f} s/clip  | s/clip by stage: {split}", flush=True)
    n_done = max(1, ok + fail)
    split = "  ".join(f"{k_} {v / n_done:.1f}" for k_, v in sorted(tm.items(), key=lambda x: -x[1]))
    print(f"shard {args.shard} done: {ok} cached, {fail} failed  | s/clip by stage: {split}", flush=True)


if __name__ == "__main__":
    main()
