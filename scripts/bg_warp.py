#!/usr/bin/env python3
"""Background warping in side-by-side demos (scripts/demo_motion.py): real left, generated right.

    python scripts/bg_warp.py results/extras/demo_guided/*_sbs.mp4

Uses sang.bench.background_motion: mean |frame difference| on textured pixels that are static in
the real video and off the head, generated vs real. bg_ratio ~1 = as still as the real video.
First measurement (2026-09-29, runs/motion_12k_anneal, 6 val clips, 512 px, stitching off, no
paste-back, --guide both --start source): median 2.35x, range 1.42-11.97x. The plain studio backdrop
that is almost perfectly still in the real video reads 7.75x with the eps floor (22.6x without it).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sang.bench import background_motion


def halves(path: Path) -> tuple[np.ndarray, np.ndarray]:
    cap, fr = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f[..., ::-1])
    v = np.stack(fr)
    w = v.shape[2] // 2
    return v[:, :, :w], v[:, :, w:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    args = ap.parse_args()
    ratios = []
    print(f"{'clip':<44}{'bg real':>9}{'bg gen':>9}{'ratio':>8}")
    for p in map(Path, args.videos):
        r = background_motion(*halves(p))
        ratios.append(r["bg_ratio"])
        print(f"{p.stem[:44]:<44}{r['bg_real']:>9.3f}{r['bg_gen']:>9.3f}{r['bg_ratio']:>8.2f}")
    print(f"{'median':<44}{'':>18}{np.nanmedian(ratios):>8.2f}")


if __name__ == "__main__":
    main()
