#!/usr/bin/env python3
"""Animate a face with speech: a photo or a video of the person, plus an audio file, gives a talking-head mp4.

    python scripts/infer.py --ckpt checkpoints/sang.pt --ref face.jpg --audio speech.wav --out out.mp4

With a video as --ref, its first frame is the face that is animated, and up to --footage seconds of it, with its
audio, show the model how this person moves. The output is the 512x512 face crop, or the whole frame with --paste.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.data.audio import load_wav
from sang.data.video import write_mp4
from sang.render.liveportrait import decode_clip
from sang.render.pipeline import FPS, SR, Talker

VIDEO = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="a SANG checkpoint")
    ap.add_argument("--ref", required=True, help="a photo, or a video of the person")
    ap.add_argument("--audio", required=True, help="the speech (wav, mp3, m4a, or any file with an audio track)")
    ap.add_argument("--out", required=True, help="the output mp4")
    ap.add_argument("--footage", type=float, default=30.0, help="seconds of a video --ref the model reads, 0 for none")
    ap.add_argument("--sampler", default="g=2,steps=6", help="audio guidance scale g and Euler steps")
    ap.add_argument("--paste", action="store_true", help="paste the face back into the full frame")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    demo = None
    if Path(args.ref).suffix.lower() in VIDEO:
        frames = decode_clip(args.ref, max(1, round(args.footage * FPS)))
        photo = frames[0]
        if args.footage > 0:
            try:
                demo = (frames, load_wav(args.ref, 0, len(frames)))
            except Exception as e:
                sys.exit(f"cannot read the audio of {args.ref} ({e}); a video without audio needs --footage 0")
    else:
        from PIL import Image
        photo = np.asarray(Image.open(args.ref).convert("RGB"))

    talker = Talker(args.ckpt, args.sampler, args.device)
    wav = load_wav(args.audio)
    n = int(len(wav) / SR * FPS)
    out, _, _ = talker.run(photo, wav, n, seed=args.seed, paste=args.paste, demo=demo)
    write_mp4(out, Path(args.out), fps=FPS, audio=Path(args.audio))
    print(f"wrote {args.out}: {len(out)} frames, {n / FPS:.1f} s"
          + (f", read {talker.demo_frames} frames of footage" if demo else ""))


if __name__ == "__main__":
    main()
