#!/usr/bin/env python3
"""Start-up jitter in side-by-side demos (scripts/demo_motion.py): real left, generated right.

    python scripts/video_jitter.py results/extras/demo/*_sbs.mp4

Jitter = pixel ACCELERATION |x[t+1] - 2 x[t] + x[t-1]| on a 128x128 grayscale copy of each half:
smooth motion has small acceleration, frame-to-frame shaking has large. Reported per frame bin and
as generated / real, so camera and background motion in the real half set the scale. Before the
first-window fix (2026-09-28, runs/motion_12k_anneal, 6 val clips): frames 0-10 median 2.7x real
(per frame 5.1x at frame 1 down to 1.4x at frame 8), frames 10+ 0.5-0.8x.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

BINS = [(0, 10), (10, 25), (25, 64), (64, 74), (74, 10_000)]


def accel(path: Path) -> dict[str, np.ndarray]:
    cap, fr = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    v = np.stack(fr)
    w = v.shape[2] // 2
    out = {}
    for name, half in (("real", v[:, :, :w]), ("gen", v[:, :, w:])):
        g = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (128, 128), interpolation=cv2.INTER_AREA)
                      for f in half]).astype(np.float32)
        out[name] = np.abs(g[2:] - 2 * g[1:-1] + g[:-2]).mean((1, 2))       # index i = frame i + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    args = ap.parse_args()
    ratios = {b: [] for b in BINS}
    print("generated / real pixel acceleration, by frame range")
    print(f"{'clip':<34}" + "".join(f"{(f'{a}-{b}' if b < 10_000 else f'{a}+'):>12}" for a, b in BINS))
    first = []
    for p in map(Path, args.videos):
        r = accel(p)
        line = f"{p.stem[:34]:<34}"
        for a, b in BINS:
            g, re = r["gen"][max(0, a - 1):b - 1], r["real"][max(0, a - 1):b - 1]
            if len(g) == 0:
                line += f"{'-':>12}"
                continue
            ratios[(a, b)].append(g.mean() / max(re.mean(), 1e-6))
            line += f"{ratios[(a, b)][-1]:>12.2f}"
        print(line)
        first.append(r["gen"][:25] / max(r["real"].mean(), 1e-6))
    print(f"{'median gen/real':<34}" + "".join(f"{np.median(ratios[b]) if ratios[b] else float('nan'):>12.2f}" for b in BINS))
    n = min(len(f) for f in first)
    print("\nper-frame gen accel / clip's mean real accel, frames 1..%d (median over clips):" % n)
    print(" ".join(f"{x:.1f}" for x in np.median(np.stack([f[:n] for f in first]), 0)))


if __name__ == "__main__":
    main()
