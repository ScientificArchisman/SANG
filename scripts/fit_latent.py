#!/usr/bin/env python3
"""Fit the motion latents (sang.latent) on the training speakers and measure how many codes motion needs.

    sbatch bash_scripts/job.sh scripts/fit_latent.py --out runs/latent

Fits a PCA (closed form) and an ordered autoencoder (nested dropout) over the 69 numbers LivePortrait renders,
relative to a random frame of the same clip, then prints the fraction of variance kept on held-out VAL speakers
for k = 1..K codes, overall and per region (rot, brow, eyes, mouth, shape, place), in standardised units. The
hand-picked 42-d target keeps rot, brow, eyes and mouth and holds shape and place at the photo's: its row is
printed for comparison.

Writes <out>/pca.pt, <out>/ae.pt (MotionLatent.pack) and <out>/curve.json. Next, the render ceiling per k:
    sbatch ... scripts/eval_hdtf.py ... --rows gt_motion --gt-latent <out>/ae.pt --gt-k <k> --name lat_ae_k<k>
and training on the chosen k:
    sbatch bash_scripts/train_motion.sh out_dir=runs/<name> latent=<out>/ae.pt latent_k=<k>
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.data import load_cfg, split_by_speaker, wide_deltas
from sang.latent import curve, fit_ae, fit_pca
from sang.motion import WIDE_REGIONS
from sang.paths import load_index, motion_cache

KS = (1, 2, 4, 8, 12, 16, 24, 32, 42, 48, 64, 69)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/train_motion.yaml"))
    ap.add_argument("--out", default=str(REPO / "runs/latent"))
    ap.add_argument("--clips", type=int, default=4000, help="training clips to fit on")
    ap.add_argument("--val-clips", type=int, default=400)
    ap.add_argument("--K", type=int, default=64, help="autoencoder code size (the PCA keeps all 69)")
    ap.add_argument("--steps", type=int, default=20000)
    args = ap.parse_args()
    cfg = load_cfg(args.config, [])
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train, val = split_by_speaker(load_index(motion_cache(cfg["cache_dir"])), cfg["val_frac"], cfg["seed"])
    d_train, d_val = wide_deltas(train, args.clips, seed=0), wide_deltas(val, args.val_clips, seed=1)
    print(f"{len(d_train)} training / {len(d_val)} val frames (deltas to a random frame of the same clip)", flush=True)

    lats = {"pca": fit_pca(d_train).to(dev), "ae": fit_ae(d_train, args.K, steps=args.steps, device=dev)}
    res = {}
    for name, lat in lats.items():
        torch.save(lat.cpu().pack(), out / f"{name}.pt")
        res[name] = curve(lat.to(dev), d_val, [k for k in KS if k <= lat.K])
    var = (d_val / lats["pca"].sd.cpu()).var(0)
    kept = [i for n in ("rot", "brow", "eyes", "mouth") for i in WIDE_REGIONS[n]]
    hand = {"k": 42, "r2": float(var[kept].sum() / var.sum()),
            **{f"r2_{n}": float(n in ("rot", "brow", "eyes", "mouth")) for n in WIDE_REGIONS}}
    (out / "curve.json").write_text(json.dumps({**res, "hand_picked_42": hand}, indent=1))

    cols = ["r2"] + [f"r2_{n}" for n in WIDE_REGIONS]
    print("\nfraction of variance kept on val speakers, by codes k")
    print(f"{'latent':<14}{'k':>4}" + "".join(f"{c:>12}" for c in cols))
    for name, rows in [*res.items(), ("hand-picked", [hand])]:
        for r in rows:
            print(f"{name:<14}{r['k']:>4}" + "".join(f"{r[c]:>12.3f}" for c in cols))
    print(f"\nwrote {out}/pca.pt, ae.pt, curve.json", flush=True)


if __name__ == "__main__":
    main()

