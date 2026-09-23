#!/usr/bin/env python3
"""Source image + speech -> talking-head mp4, through the trained motion model and frozen renderer.

    python scripts/infer_motion.py --ckpt runs/motion_bidir/best.pt \
        --image face.jpg --audio speech.wav --out out.mp4

Also the unit the HDTF evaluation runs per clip (first frame as source, the clip's own audio,
KDTalker's protocol), after which sang/bench.py scores CSIM / LSE / FID on the written mp4s.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party"))

from sang.bench import write_mp4
from sang.motion import MotionCodec, from_target, to_target
from sang.motion_model import Norm, build, generate

SR, FPS = 16000, 25


def load_image(path: str) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def load_audio(path: str) -> torch.Tensor:
    """Any container decord can read (wav, m4a, mp4) -> [1, 1, N] at 16 kHz mono."""
    from decord import AudioReader
    return torch.from_numpy(AudioReader(path, sample_rate=SR, mono=True)[:].asnumpy()).float()[None]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None, help="NFE; default = the run's sample_steps")
    ap.add_argument("--cfg", type=float, default=None, help="audio guidance; default = the run's cfg_audio")
    ap.add_argument("--no-lip-norm", action="store_true", help="skip LivePortrait's flag_normalize_lip")
    ap.add_argument("--raw", action="store_true", help="use the training weights instead of the EMA")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    model = build(cfg).to(dev).eval()
    model.load_state_dict(ck["model" if args.raw else "ema"])
    norm = Norm(**ck["norm"]).to(dev)

    from sang.codec import load_wavlm
    codec = MotionCodec(device=dev)
    wavlm = load_wavlm(cfg.get("audio_encoder", "wavlm-large"), device=dev)

    src = load_image(args.image)
    sm = codec.source_motion(src, normalize_lip=not args.no_lip_norm)
    m_src, kp_src = sm["m"].to(dev), sm["kp"].to(dev)
    ref = norm.ref(kp_src, to_target(m_src))                          # [1, REF_DIM]

    wav = load_audio(args.audio)
    n_frames = int(wav.shape[-1] / SR * FPS)
    with torch.no_grad():
        audio = wavlm.encode(wav.to(dev)).float()                    # [1, ~2n, D]
    g = torch.Generator(device=dev).manual_seed(args.seed)
    y = generate(model, audio, ref, n_frames, window=cfg["frames"], n_prefix=cfg["prefix"],
                 steps=args.steps or cfg["sample_steps"],
                 cfg_audio=cfg["cfg_audio"] if args.cfg is None else args.cfg, generator=g)
    m = from_target(norm.untarget(y[0]), m_src)                       # [n, 70], source scale/t/shape
    frames = codec.render(src, m.cpu(), relative=False, stitch=True)
    write_mp4(frames, Path(args.out), fps=FPS, audio=Path(args.audio))
    print(f"wrote {args.out}: {len(frames)} frames, {n_frames / FPS:.1f} s")


if __name__ == "__main__":
    main()
