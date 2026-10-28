#!/usr/bin/env python3
"""Source image + speech -> talking-head mp4 (512x512 face crop), through sang.pipeline.Talker.

    python scripts/infer_motion.py --ckpt runs/d1_97h/last.pt --image face.jpg --audio speech.wav --out out.mp4 \
        --sampler g=2,mouth=1.25,tau=0.5,avg=4,steps=6

Voice cloning (sang/voice.py): --voice voices/<name> (a bank from scripts/enroll_voice.py) or
--voice <folder or file of the person's audio> (enrolled into voices/<folder name>/ on the fly and
kept). The face is still driven by the ORIGINAL --audio; only the audio track muxed into the mp4 is
converted (kNN-VC keeps the timing, so the lips stay aligned), and also written as <out>.voice.wav.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.bench import write_mp4
from sang.codec import load_wav
from sang.pipeline import FPS, SR, Talker


def load_image(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sampler", default=None, help="sampler spec (sang.motion_model.SAMPLER_KEYS); default: the run's")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", default="source", choices=["null", "source"],
                    help="first window: 'source' = continue from the photo's own motion after 0.4 s of silence; "
                         "'null' = the dropped-prefix start training uses")
    ap.add_argument("--stitch", action=argparse.BooleanOptionalAction, default=True,
                    help="LivePortrait's stitching module: keeps the background still")
    ap.add_argument("--voice", default=None,
                    help="speak in this person's voice: a voice bank dir (voices/<name>) or their audio files")
    ap.add_argument("--voice-k", type=int, default=None, help="kNN-VC k; default grows with the bank size")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    talker = Talker(args.ckpt, args.sampler, dev)
    wav = load_wav(args.audio)
    n = int(len(wav) / SR * FPS)
    frames, _, _ = talker.run(load_image(args.image), wav, n, seed=args.seed, stitch=args.stitch, paste=False,
                              start=args.start)
    track = Path(args.audio)
    if args.voice:
        from sang.voice import SpeakerEncoder, convert, load_knnvc, save_wav, voice_bank
        knnvc = load_knnvc(dev)
        bank = voice_bank(args.voice, knnvc, None if (Path(args.voice) / "bank.pt").exists() else SpeakerEncoder(dev))
        track = save_wav(convert(wav, bank, knnvc, k=args.voice_k), Path(args.out).with_suffix(".voice.wav"))
        print(f"voice: {bank.dir.name}, bank {bank.seconds:.0f} s -> {track}")
    write_mp4(frames, Path(args.out), fps=FPS, audio=track)
    print(f"wrote {args.out}: {len(frames)} frames, {n / FPS:.1f} s")


if __name__ == "__main__":
    main()

