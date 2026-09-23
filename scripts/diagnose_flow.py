#!/usr/bin/env python3
"""Diagnose the continuous (Wan-VAE + flow) track on real val windows.

Answers three questions about a checkpoint, with numbers:
  1. Is SyncNet being gamed?  Score the frozen StableSyncNet on (a) ground truth, (b) generated
     video with the matched mel, (c) generated video with a shuffled mel, (d) the one-step x0
     estimate the training loss actually sees. A healthy model has (b) << (c); a gamed loss has
     (b) ~ (d) < GT with (c) also low (the loss is satisfied by a texture, not by lip motion).
  2. Train/inference gap.  Teacher-forced one-step x0 latent MSE at several t, versus the MSE of
     a full sequential sample. A large gap = exposure bias / sampler error, not underfitting.
  3. Does the output move?  Inter-slice latent change of generated vs GT content slices, and
     per-frame pixel change in the mouth band. Static output shows up as near-zero motion.

    sbatch bash_scripts/job.sh scripts/diagnose_flow.py --ckpt runs/face256_wan_165551/best.pt
"""
import argparse
import glob
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from sang.diffusion import add_noise, x0_from_v
from sang.model import build_talking_head
from sang.syncnet import StableSyncNet
from sang.losses import SYNCNET

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def psnr_px(a, b):
    return 10 * math.log10(4.0 / max(F.mse_loss(a, b).item(), 1e-8))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=24, help="val windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "results/diagnose_flow.json"))
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = build_talking_head(cfg).to(DEV)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    print(f"ckpt step {ck.get('step')} val_psnr {ck.get('val_acc')}")

    from sang.vae import load_wan_vae
    vae = load_wan_vae(device=DEV)
    sync = StableSyncNet.from_checkpoint(SYNCNET, device=DEV)

    # val windows: same speaker split as train.py -> just take files whose key is in the val set.
    # train.py does not persist the split, so we re-derive it exactly.
    src = Path(cfg["data_glob"])
    clips = ([l.strip() for l in src.read_text().splitlines() if l.strip()]
             if src.suffix == ".txt" and src.exists() else sorted(glob.glob(cfg["data_glob"])))
    random.Random(cfg["seed"]).shuffle(clips)
    if cfg.get("max_clips"):
        clips = clips[: cfg["max_clips"]]
    speakers = sorted({Path(c).parent.name for c in clips})
    random.Random(cfg["seed"]).shuffle(speakers)
    n_val = max(1, int(len(speakers) * cfg["val_frac"]))
    val_spk = set(speakers[:n_val])
    import hashlib
    cache = REPO / cfg["cache_dir"]
    # one directory listing (96k files on BeeGFS: a glob per clip took >10 min in job 165645)
    by_key = {}
    for name in os.listdir(cache):
        if name.endswith(".pt"):
            by_key.setdefault(name.split("_")[0], []).append(cache / name)
    files = []
    for c in clips:
        if Path(c).parent.name in val_spk:
            files += sorted(by_key.get(hashlib.md5(str(c).encode()).hexdigest(), []))
    random.Random(args.seed).shuffle(files)
    files = files[: args.n]
    print(f"{len(files)} val windows from {len(val_spk)} unseen speakers")

    n_cond = model.ref_slices
    acc = {k: [] for k in ["sync_gt", "sync_gen", "sync_gen_shuf", "sync_x0_t25", "sync_x0_t50",
                           "sync_x0_t75", "sync_freeze_ctx",
                           "mse_gen", "mse_freeze_ctx", "mse_tf_t25", "mse_tf_t50", "mse_tf_t75",
                           "mse_tf_t90", "psnr_gen", "psnr_freeze_ctx",
                           "motion_gt", "motion_gen", "mouth_motion_gt", "mouth_motion_gen"]}
    mels = []
    for f in files:
        d = torch.load(f, weights_only=True)
        mels.append(d["mel"].float())
    for i, f in enumerate(files):
        d = torch.load(f, weights_only=True)
        video = d["video"].float().permute(1, 0, 2, 3)          # [tv_content, z, h, w]
        video = torch.cat([d["ref"][None].float(), d["ctx"][None].float(), video], 0)[None].to(DEV)
        audio = d["audio"].float()[None].to(DEV)
        mel = mels[i][None].to(DEV)
        mel_shuf = mels[(i + 7) % len(mels)][None].to(DEV)
        gt = video[:, n_cond:]                                   # [1, ctv, z, h, w]

        gen = model.generate_continuous(audio, video[:, 0], ctx=video[:, 1],
                                        steps=cfg.get("decode_steps", 12),
                                        audio_cfg=cfg.get("audio_cfg", 1.0),
                                        bridge_t=cfg.get("bridge_t", 1.0))
        frz = video[:, 1:2].repeat(1, gt.shape[1], 1, 1, 1)
        acc["mse_gen"].append(F.mse_loss(gen, gt).item())
        acc["mse_freeze_ctx"].append(F.mse_loss(frz, gt).item())

        dec = lambda z: vae.decode_video(z.permute(0, 2, 1, 3, 4).contiguous())  # [1,3,T,H,W]
        gt_px, gen_px, frz_px = dec(gt), dec(gen), dec(frz)
        acc["psnr_gen"].append(psnr_px(gen_px, gt_px))
        acc["psnr_freeze_ctx"].append(psnr_px(frz_px, gt_px))
        acc["sync_gt"].append(sync.loss(gt_px, mel).item())
        acc["sync_gen"].append(sync.loss(gen_px, mel).item())
        acc["sync_gen_shuf"].append(sync.loss(gen_px, mel_shuf).item())
        acc["sync_freeze_ctx"].append(sync.loss(frz_px, mel).item())

        # motion: mean |z_s - z_{s-1}| across content slices; mouth band px change
        acc["motion_gt"].append((gt[:, 1:] - gt[:, :-1]).abs().mean().item())
        acc["motion_gen"].append((gen[:, 1:] - gen[:, :-1]).abs().mean().item())
        H = gt_px.shape[-2]
        band = slice(int(0.58 * H), int(0.78 * H))
        acc["mouth_motion_gt"].append((gt_px[:, :, 1:, band] - gt_px[:, :, :-1, band]).abs().mean().item())
        acc["mouth_motion_gen"].append((gen_px[:, :, 1:, band] - gen_px[:, :, :-1, band]).abs().mean().item())

        # teacher-forced one-step x0 at fixed t (what the training loss + SyncNet see)
        known = torch.zeros(1, model.tv * model.r, dtype=torch.bool, device=DEV)
        known[:, : n_cond * model.r] = True
        hs = model.hidden_states(video, audio, known)
        h_c = model.norm(hs[:, n_cond:].reshape(1, -1, hs.shape[-1]))
        z0 = model._content_latents(video)
        for tval in (0.25, 0.5, 0.75, 0.9):
            t = torch.full((1,), tval, device=DEV)
            z_t, _ = add_noise(z0, t, noise=torch.randn_like(z0))
            v = model.head(z_t, t, h_c)
            x0 = x0_from_v(z_t, t, v)
            ctv = model.tv - n_cond
            x0 = x0.reshape(1, ctv, model.r, model.z_ch).permute(0, 1, 3, 2).reshape(1, ctv, model.z_ch, model.spatial, model.spatial)
            acc[f"mse_tf_t{int(tval*100)}"].append(F.mse_loss(x0, gt).item())
            if tval in (0.25, 0.5, 0.75):
                acc[f"sync_x0_t{int(tval*100)}"].append(sync.loss(dec(x0), mel).item())
        if i % 4 == 0:
            print(f"[{i}] psnr gen {acc['psnr_gen'][-1]:.2f} frz {acc['psnr_freeze_ctx'][-1]:.2f} | "
                  f"sync gt {acc['sync_gt'][-1]:.2f} gen {acc['sync_gen'][-1]:.2f} shuf {acc['sync_gen_shuf'][-1]:.2f} "
                  f"x0@.5 {acc['sync_x0_t50'][-1]:.2f}", flush=True)

    summary = {k: (sum(v) / len(v) if v else None) for k, v in acc.items()}
    summary["n"] = len(files)
    summary["ckpt"] = args.ckpt
    summary["step"] = ck.get("step")
    print(json.dumps(summary, indent=2))
    Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
