"""Wan2.1 VAE reconstruction ceiling vs VidTok FSQ. Run: python scripts/wan_ceiling.py

Needs decord (avcodec env). avasr can load the VAE but not decode mp4s.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sang.video import decode_frames, crop_resize, to_uint8_frames
from sang.vae import load_wan_vae

PATH = "/beegfs/work_fast/shared/li_shared/archi_data/talkvid/clips/-04ZSRBGcsk/-04ZSRBGcsk_NA_1150.624_1155.846.mp4"


def psnr_u8(a, b):
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()
    return 10 * math.log10(255 ** 2 / max(mse, 1e-8))


def main():
    vae = load_wan_vae(device="cpu")
    frames, _, _ = decode_frames(PATH, frames=17, fps=25)
    for res in (128, 256):
        gt = crop_resize(frames, res)
        z = vae.encode_video(gt)
        rec = vae.decode_video(z)
        gt_u8 = to_uint8_frames(gt)
        rec_u8 = to_uint8_frames(rec)
        print(f"WanVAE  res={res}  latent={tuple(z.shape)}  PSNR={psnr_u8(gt_u8, rec_u8):.2f} dB")
        if res == 256:
            side = np.concatenate([gt_u8, rec_u8], axis=2)  # [T,H,2W,3]
            out = REPO / "results" / "extras" / "ceiling_wan_256.npy"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, side)
            print(f"saved {out} (left=GT right=Wan recon)")


if __name__ == "__main__":
    main()
