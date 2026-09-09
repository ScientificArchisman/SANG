#!/usr/bin/env python3
"""One-shot talking-head inference: reference image + audio -> mp4."""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from decord import AudioReader
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from sang.data import audio_samples, video_struct_windows
from sang.model import build_talking_head
from sang.video import (crop_resize, encode, encode_latent, face_box, load_vidtok,
                        to_uint8_frames, upscale_to_original)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ref_grid(vidtok, img: np.ndarray, frames: int, res: int, continuous: bool = False,
             face_crop: bool = False):
    """Single RGB image [H,W,3] -> VidTok identity tokens [1,h,w] (discrete) or latent [1,z_ch,h,w]
    (continuous). Static ref: replicate over time, take frame 0.

    `face_crop` must match how the checkpoint was trained, or the reference face is framed
    completely differently from anything the model saw."""
    stack = np.repeat(img[None], frames, axis=0)
    box = face_box(stack) if face_crop else None
    if face_crop and box is None:
        print("WARNING: face_crop is on but no face was detected in the reference image; "
              "falling back to the centre crop (identity framing will not match training)")
    vid = crop_resize(stack, res, box=box).to(DEVICE)
    if continuous:
        return encode_latent(vidtok, vid)[:, :, 0], box  # [1, z_ch, h, w]
    return encode(vidtok, vid)[:, 0], box


def audio_windows(wav: torch.Tensor, n: int) -> list[torch.Tensor]:
    """[1, N] waveform -> list of [1, n] windows (last zero-padded)."""
    wins = []
    for a0 in range(0, wav.shape[-1], n):
        win = wav[:, a0:a0 + n]
        if win.shape[-1] < n:
            win = F.pad(win, (0, n - win.shape[-1]))
        wins.append(win)
    return wins


def write_mp4(frames: np.ndarray, fps: float, audio: str, out: str) -> None:
    """[T,H,W,3] uint8 frames + audio file -> mp4 (silent x264, then ffmpeg audio mux)."""
    import imageio.v2 as imageio
    silent = str(Path(out).with_suffix(".silent.mp4"))
    with imageio.get_writer(silent, fps=fps, codec="libx264", macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)
    subprocess.run(["ffmpeg", "-y", "-i", silent, "-i", audio, "-c:v", "copy",
                    "-c:a", "aac", "-shortest", out], check=True, capture_output=True)
    os.remove(silent)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--ckpt", default="runs/stream_v3/best.pt", help="best val_acc among saved ckpts")
    ap.add_argument("--drive_video", default=None, help="E2: drive pose/expression from this clip (reenactment)")
    ap.add_argument("--audio_cfg", type=float, default=None,
                    help="classifier-free audio guidance (v3: try 1.5-2.5; default = ckpt cfg)")
    ap.add_argument("--out", default="results/extras/infer.mp4")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = REPO / ckpt
    state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    # ckpt carries its own training cfg (v3 token or v4 continuous); fall back to v3 yaml for old ckpts
    # the checkpoint carries its own training cfg; fall back to the default for older ones
    cfg = {**yaml.safe_load(open(REPO / "configs" / "train.yaml")), **state["cfg"]}

    continuous = cfg.get("continuous", False)
    if continuous:
        from sang.vae import load_wan_vae
        vidtok = load_wan_vae(device=DEVICE)
        fsq_codes = None
    else:
        vidtok = load_vidtok(codebook=cfg["codebook"], device=DEVICE)
        fsq_codes = getattr(vidtok.regularization, "implicit_codebook", None)
        if fsq_codes is not None:
            fsq_codes = fsq_codes.to(DEVICE).float()
    from sang.codec import load_wavlm
    audio_enc = load_wavlm(cfg["audio_encoder"], device=DEVICE)
    model = build_talking_head(cfg, fsq_codes=fsq_codes).to(DEVICE).eval()
    model.load_state_dict(state["model"])
    t0 = time.time()
    img = np.array(Image.open(args.image).convert("RGB"))
    orig_h, orig_w = img.shape[:2]
    ref, ref_box = ref_grid(vidtok, img, cfg["frames"], cfg["res"], continuous,
                            face_crop=cfg.get("face_crop", False))
    wav = torch.from_numpy(AudioReader(args.audio, sample_rate=audio_enc.sample_rate, mono=True)[:].asnumpy())
    # Must match sang.data.audio_samples used at cache time, or Ta differs from training.
    win_sec = (cfg["frames"] - 1) / cfg["fps"]
    n = audio_samples(cfg["frames"], cfg["fps"], audio_enc.sample_rate)
    wins = audio_windows(wav, n)[: max(1, round(args.seconds / win_sec))]
    struct = None
    if args.drive_video:
        struct = video_struct_windows(args.drive_video, vidtok, cfg["frames"], cfg["res"], cfg["fps"], max_windows=len(wins))
        wins = wins[: len(struct)]  # generate only where we have driving structure
    audio_cfg = args.audio_cfg if args.audio_cfg is not None else cfg.get("audio_cfg", 1.0)
    ctv = (cfg["frames"] - 1) // 4 + 1  # content slices per window
    ctx, raw = None, []
    if continuous:
        gen_kw = dict(steps=cfg.get("decode_steps", 8), audio_cfg=audio_cfg)
        for i, c in enumerate(wins):
            lat = model.generate_continuous(audio_enc.encode(c[None].to(DEVICE)), ref,
                                            struct=struct[i] if struct else None, ctx=ctx, **gen_kw)
            # lat [1, ctv, z_ch, h, w] -> [1, z_ch, ctv, h, w] -> pixels
            ctx = lat[:, -1]
            px = vidtok.decode_video(lat.permute(0, 2, 1, 3, 4).contiguous())
            raw.append(to_uint8_frames(px))
    else:
        gen_kw = dict(steps=cfg.get("decode_steps", 8), temperature=cfg.get("decode_temperature", 1.0),
                      gumbel_temp=cfg.get("decode_gumbel", 4.5), audio_cfg=audio_cfg)
        for i, c in enumerate(wins):
            grid = model.generate(audio_enc.encode(c[None].to(DEVICE)), ref,
                                  struct=struct[i] if struct else None, ctx=ctx, **gen_kw)
            ctx = grid[:, -1:]  # motion context handed to the next window (no 0.68s reset)
            raw.append(to_uint8_frames(vidtok.decode(grid[:, grid.shape[1] - ctv:], decode_from_indices=True)))
    frames = upscale_to_original(np.concatenate(raw, 0), orig_h, orig_w, box=ref_box)
    tmp = Path(args.out).with_suffix(".tmp.mp4")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    write_mp4(frames, cfg["fps"], args.audio, str(tmp))
    shutil.move(tmp, args.out)
    print(f"weights: {ckpt.name} step {state.get('step', '?')} val_acc {state.get('val_acc', float('nan')):.4f}")
    print(f"generated {len(wins) * win_sec:.1f}s ({len(wins)} windows) in {time.time() - t0:.1f}s on {DEVICE}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
