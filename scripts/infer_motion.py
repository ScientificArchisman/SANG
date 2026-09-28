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
from sang.motion_model import Norm, build
from sang.naturalness import Guide, guided_generate

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
    ap.add_argument("--stitch", action="store_true",
                    help="LivePortrait's stitching: only for pasting back into the full photo. It pulls "
                         "keypoints toward the source and cost 5.4 dB PSNR in M0 (jobs 171518 vs 171519)")
    ap.add_argument("--raw", action="store_true", help="use the training weights instead of the EMA")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guide", default="none", choices=["none", "lips", "blinks", "both"],
                    help="rule constraints at sampling (sang/naturalness.py)")
    ap.add_argument("--start", default="null", choices=["null", "source"],
                    help="first window: 'null' = the dropped-prefix configuration training used; "
                         "'source' = continue from the source photo's own motion after 0.4 s of silence")
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
    start, lead = None, 0
    if args.start == "source":                                       # the photo, still and silent, for P frames
        lead = cfg["prefix"]
        start = norm.target(to_target(m_src)).unsqueeze(1).expand(1, lead, -1)
    with torch.no_grad():
        audio = wavlm.encode(torch.nn.functional.pad(wav, (lead * SR // FPS, 0)).to(dev)).float()   # [1, ~2(lead+n), D]
    guide = Guide.load(Path(cfg["cache_dir"]), dev) if args.guide != "none" else None
    y, info = guided_generate(model, audio, ref, n_frames, cfg, norm, wav.flatten(), guide, args.guide,
                              args.seed, steps=args.steps, cfg_audio=args.cfg, start=start)
    if info:
        print(f"constraints: {info}")
    m = from_target(norm.untarget(y[0]), m_src)                       # [n, 70], source scale/t/shape
    frames = codec.render(src, m.cpu(), relative=False, stitch=args.stitch)
    write_mp4(frames, Path(args.out), fps=FPS, audio=Path(args.audio))
    print(f"wrote {args.out}: {len(frames)} frames, {n_frames / FPS:.1f} s")


if __name__ == "__main__":
    main()
