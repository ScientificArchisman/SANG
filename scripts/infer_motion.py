#!/usr/bin/env python3
"""Source image + speech -> talking-head mp4 (512x512 face crop), through sang.render.pipeline.Talker.

    python scripts/infer_motion.py --ckpt runs/l42_gft/last.pt --image face.jpg --audio speech.wav --out out.mp4 \
        --sampler g=2,steps=6
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.data.audio import load_wav
from sang.eval.metrics import write_mp4
from sang.render.pipeline import FPS, SR, Talker


def load_image(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sampler", default=None, help="sampler spec (sang.model.flow.SAMPLER_KEYS); default: the run's")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    talker = Talker(args.ckpt, args.sampler, dev)
    wav = load_wav(args.audio)
    n = int(len(wav) / SR * FPS)
    frames, _, _ = talker.run(load_image(args.image), wav, n, seed=args.seed, paste=False)
    write_mp4(frames, Path(args.out), fps=FPS, audio=Path(args.audio))
    print(f"wrote {args.out}: {len(frames)} frames, {n / FPS:.1f} s")


if __name__ == "__main__":
    main()

