#!/usr/bin/env python3
"""Fit linear lip / eye openness readouts on the 42-d target (needed by sang/naturalness.py).

    sbatch --time=08:00:00 bash_scripts/job.sh scripts/calibrate_openness.py

Resumable and saves as it goes: raw (motion, ratio) pairs go to <cache>/openness_pairs.pt and
openness.json is refitted every --save-every clips, so a timeout keeps what was done and a rerun
continues from there. (The first run, 2026-09-29, spent ~125 s per clip and hit the 4 h limit at
110/120 clips with nothing written.)

For cached clips, pairs each frame's cached motion with LivePortrait's own landmark ratios on the
SAME raw frame (upstream retargeting_utils: lip gap / mouth width, eye lid gap / eye width), then
ridge-regresses ratio on the mouth / eye columns. Writes <cache>/openness.json with held-out-clip
R^2; a readout below R^2 ~0.7 is too weak to trust the metrics or the constraints built on it.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party"))

from sang.motion import REGIONS, MotionCodec, to_target
from sang.naturalness import eye_ratio, fit_readout, lip_ratio
from sang.paths import MOTION_CACHE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(MOTION_CACHE))
    ap.add_argument("--n", type=int, default=80, help="clips (identities matter more than frames)")
    ap.add_argument("--frames", type=int, default=150, help="per clip")
    ap.add_argument("--every", type=int, default=3, help="landmark every k-th frame")
    ap.add_argument("--save-every", type=int, default=10, help="clips between checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("tm", REPO / "scripts" / "train_motion.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    rows = random.Random(args.seed).sample(tm.load_index(Path(args.cache)), args.n)

    pairs_path = Path(args.cache) / "openness_pairs.pt"
    st = torch.load(pairs_path, weights_only=False) if pairs_path.exists() else \
        {"Y": [], "L": [], "E": [], "G": [], "done": []}
    done = {c for c in st["done"]}
    print(f"{len(done)} clips already done, {len(st['Y'])} frames", flush=True)

    from sang.video import open_video
    codec = MotionCodec(device="cuda")
    t0, tm_dec, tm_lmk, new = time.time(), 0.0, 0.0, 0
    for k, r in enumerate(rows):
        if r["clip"] in done:
            continue
        try:
            d = torch.load(r["path"], map_location="cpu", weights_only=True)
            s, n = r.get("start", 0), min(args.frames, int(d["n"]))
            y = to_target(d["m"][:n].float())
            t = time.perf_counter()
            vr = open_video(r["clip"])                      # decode ONLY the frames we landmark,
            stride = vr.get_avg_fps() / 25.0                # on decode_clip's 25 fps clock
            want = [i for i in range(0, n, args.every) if int(round((s + i) * stride)) < len(vr)]
            frames = vr.get_batch([int(round((s + i) * stride)) for i in want]).asnumpy()
            tm_dec += time.perf_counter() - t
            t = time.perf_counter()
            for i, fr in zip(want, frames):
                lmk = codec.landmarks(fr)
                if lmk is None:
                    continue
                st["Y"].append(y[i]); st["L"].append(float(lip_ratio(lmk)))
                st["E"].append(float(eye_ratio(lmk))); st["G"].append(k)
            tm_lmk += time.perf_counter() - t
        except Exception as e:
            print(f"  skip {Path(r['clip']).name}: {type(e).__name__}: {e}", flush=True)
        st["done"].append(r["clip"])
        new += 1
        if new % args.save_every == 0:
            torch.save(st, pairs_path)
            fit_and_write(st, Path(args.cache))
            print(f"  [{len(st['done'])}/{len(rows)}] {len(st['Y'])} frames, {time.time() - t0:.0f} s  "
                  f"(per clip: decode {tm_dec / new:.1f} s, landmarks {tm_lmk / new:.1f} s)", flush=True)
    torch.save(st, pairs_path)
    fit_and_write(st, Path(args.cache))


def fit_and_write(st: dict, cache: Path) -> None:
    if len(set(st["G"])) < 10:
        return                                          # too few clips for a held-out R^2
    Y, G = torch.stack(st["Y"]), np.array(st["G"])
    out = {}
    for name, target, cols in (("lip", st["L"], REGIONS["mouth"]), ("eye", st["E"], REGIONS["eyes"])):
        ro = fit_readout(Y, torch.tensor(target), cols, G)
        out[name] = {**ro.to_dict(), "ratio_p5": float(np.percentile(target, 5)),
                     "ratio_p50": float(np.median(target)), "ratio_p95": float(np.percentile(target, 95))}
        flag = "ok" if ro.r2 >= 0.7 else "WEAK -- do not trust constraints on it"
        print(f"  {name}: held-out-clip R^2 {ro.r2:.3f} ({flag}); ratio p5/p50/p95 "
              f"{out[name]['ratio_p5']:.3f} / {out[name]['ratio_p50']:.3f} / {out[name]['ratio_p95']:.3f}", flush=True)
    out["frames"], out["clips"] = len(Y), int(len(np.unique(G)))
    dst = cache / "openness.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"  wrote {dst} ({out['clips']} clips, {out['frames']} frames)", flush=True)


if __name__ == "__main__":
    main()
