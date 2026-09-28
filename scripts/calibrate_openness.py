#!/usr/bin/env python3
"""Fit linear lip / eye openness readouts on the 42-d target (needed by sang/naturalness.py).

    sbatch bash_scripts/job.sh scripts/calibrate_openness.py --n 120

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

from sang.motion import REGIONS, MotionCodec, decode_clip, to_target
from sang.naturalness import eye_ratio, fit_readout, lip_ratio


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(REPO / "cache/motion_lp"))
    ap.add_argument("--n", type=int, default=120, help="clips")
    ap.add_argument("--frames", type=int, default=200, help="per clip")
    ap.add_argument("--every", type=int, default=2, help="landmark every k-th frame")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("tm", REPO / "scripts" / "train_motion.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    rows = random.Random(args.seed).sample(tm.load_index(Path(args.cache)), args.n)

    codec = MotionCodec(device="cuda")
    Y, L, E, G = [], [], [], []
    t0 = time.time()
    for k, r in enumerate(rows):
        try:
            d = torch.load(r["path"], map_location="cpu", weights_only=True)
            s, n = r.get("start", 0), min(args.frames, int(d["n"]))
            frames = decode_clip(r["clip"], s + n)
            y = to_target(d["m"][:n].float())
            for i in range(0, min(n, len(frames) - s), args.every):
                lmk = codec.landmarks(frames[s + i])
                if lmk is None:
                    continue
                Y.append(y[i]); L.append(float(lip_ratio(lmk))); E.append(float(eye_ratio(lmk))); G.append(k)
        except Exception as e:
            print(f"  skip {Path(r['clip']).name}: {type(e).__name__}: {e}", flush=True)
        if (k + 1) % 10 == 0:
            print(f"  [{k + 1}/{len(rows)}] {len(Y)} frames, {time.time() - t0:.0f} s", flush=True)

    Y, G = torch.stack(Y), np.array(G)
    out = {}
    for name, target, cols in (("lip", L, REGIONS["mouth"]), ("eye", E, REGIONS["eyes"])):
        r = torch.tensor(target)
        ro = fit_readout(Y, r, cols, G)
        out[name] = {**ro.to_dict(), "ratio_p5": float(np.percentile(target, 5)),
                     "ratio_p50": float(np.median(target)), "ratio_p95": float(np.percentile(target, 95))}
        flag = "ok" if ro.r2 >= 0.7 else "WEAK -- do not trust constraints on it"
        print(f"{name}: held-out-clip R^2 {ro.r2:.3f} ({flag}); ratio p5/p50/p95 "
              f"{out[name]['ratio_p5']:.3f} / {out[name]['ratio_p50']:.3f} / {out[name]['ratio_p95']:.3f}", flush=True)
    out["frames"], out["clips"] = len(Y), int(len(np.unique(G)))
    dst = Path(args.cache) / "openness.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
