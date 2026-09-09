#!/usr/bin/env python3
"""Reconstruction ceiling: the best any token/latent predictor could achieve.

Encode real clips and decode straight back, no transformer. Replaces the separate
vidtok_ceiling.py / wan_ceiling.py, and adds the axis that turned out to matter most --
whether the frames are face-cropped or blind centre-cropped.

    python scripts/ceiling.py --n 8
    python scripts/ceiling.py --n 8 --tokenizers wan --res 256

mediapipe needs libGLESv2: export LD_LIBRARY_PATH=$CONDA_PREFIX/../gl/lib:$LD_LIBRARY_PATH
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party"))

from sang.video import crop_resize, decode_frames, encode, face_box, load_vidtok, to_uint8_frames

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 20 * np.log10(255.0) - 10 * np.log10(mse)


def roundtrip(kind: str, model, gt: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if kind == "wan":
            return model.decode_video(model.encode_video(gt))
        return model.decode(encode(model, gt), decode_from_indices=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(REPO / "data/clips_8000.txt"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--res", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--tokenizers", nargs="+", default=["fsq32768", "wan"])
    args = ap.parse_args()

    paths = [l.strip() for l in Path(args.clips).read_text().splitlines() if l.strip()][: args.n]
    clips = []
    for p in paths:
        try:
            fr, _, _ = decode_frames(p, args.frames, start=None, fps=25)
        except Exception:
            continue
        box = face_box(fr)
        clips.append((fr, box))
    n_face = sum(b is not None for _, b in clips)
    print(f"{len(clips)} clips, face detected in {n_face}\n")
    print(f"{'tokenizer':>10} {'res':>5} {'crop':>7} {'PSNR dB':>9}")
    print("-" * 36)

    for res in args.res:
        for tok in args.tokenizers:
            if tok == "wan":
                from sang.vae import load_wan_vae
                model, kind = load_wan_vae(device=DEV), "wan"
            else:
                cb = int(tok.replace("fsq", ""))
                model, kind = load_vidtok(codebook=cb, device=DEV), "fsq"
            for crop in ("blind", "face"):
                scores = []
                for fr, box in clips:
                    if crop == "face" and box is None:
                        continue
                    gt = crop_resize(fr, res, box=box if crop == "face" else None).to(DEV)
                    rec = roundtrip(kind, model, gt)
                    scores.append(psnr(to_uint8_frames(gt), to_uint8_frames(rec)))
                if scores:
                    print(f"{tok:>10} {res:>5} {crop:>7} {np.mean(scores):>9.2f}")
            del model
            if DEV == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
