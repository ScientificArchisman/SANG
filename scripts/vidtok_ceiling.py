"""Compare VidTok reconstruction PSNR across codebook size and resolution.

Answers: how much of the 'melting' is fixed by (a) bigger codebook, (b) higher res?
Run on CPU (GPU node conv3d is broken): CEILING_CPU=1 python scripts/vidtok_ceiling.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sang.video import load_vidtok, decode_frames, crop_resize, encode, to_uint8_frames

DEV = "cpu" if os.environ.get("CEILING_CPU") else "cuda"
PATH = "/beegfs/work_fast/shared/li_shared/archi_data/talkvid/clips/-04ZSRBGcsk/-04ZSRBGcsk_NA_1150.624_1155.846.mp4"


def psnr(a, b):
    mse = ((a.astype(np.float32) - b.astype(np.float32)) ** 2).mean()
    return 10 * np.log10(255 ** 2 / max(mse, 1e-8))


frames, _, _ = decode_frames(PATH, frames=17, fps=25)
print(f"{'codebook':>9} {'res':>5} {'PSNR dB':>8}")
for res in (128, 256):
    gt = crop_resize(frames, res)
    gt_u8 = to_uint8_frames(gt)[0]
    for cb in (32768, 262144):
        vt = load_vidtok(codebook=cb, device=DEV)
        with torch.no_grad():
            idx = encode(vt, gt)
            rec = vt.decode(idx, decode_from_indices=True)
        rec_u8 = to_uint8_frames(rec)[0]
        print(f"{cb:>9} {res:>5} {psnr(gt_u8, rec_u8):>8.2f}")
        if res == 256 and cb == 262144:
            # D13: gt_u8/rec_u8 are [H,W,3]; axis=2 concatenated CHANNELS (a 6-channel
            # array), not a side-by-side image. Older .npy files in results/extras use that
            # broken layout -- split them on channels to recover the pair.
            side = np.concatenate([gt_u8, rec_u8], axis=1)
            np.save(REPO / "results" / "extras" / "ceiling_256_262144.npy", side)
            try:
                from PIL import Image
                Image.fromarray(side).save(REPO / "results" / "extras" / "ceiling_256_262144.png")
            except Exception:
                pass
        del vt
        torch.cuda.empty_cache() if DEV == "cuda" else None
print("saved results/extras/ceiling_256_262144.npy (left=GT right=recon)")
