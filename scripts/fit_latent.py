#!/usr/bin/env python3
"""Fit the motion latent (sang.model.latent) on the training speakers.

    sbatch bash_scripts/run.sh scripts/fit_latent.py --out runs/latent

A PCA (closed form) over the 69 numbers LivePortrait renders, relative to a random frame of the same clip.
Writes <out>/pca.pt (MotionLatent.pack), which configs/train_motion.yaml reads.
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.data.dataset import load_cfg, split_by_speaker, wide_deltas
from sang.data.paths import load_index
from sang.model.latent import fit_pca


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/train_motion.yaml"))
    ap.add_argument("--out", default=str(REPO / "runs/latent"))
    ap.add_argument("--clips", type=int, default=4000, help="training clips to fit on")
    args = ap.parse_args()
    cfg = load_cfg(args.config, [])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train, _ = split_by_speaker(load_index(Path(cfg["cache_dir"])), cfg["val_frac"], cfg["seed"])
    d = wide_deltas(train, args.clips, seed=0)
    print(f"{len(d)} training frames (deltas to a random frame of the same clip)", flush=True)
    torch.save(fit_pca(d).pack(), out / "pca.pt")
    print(f"wrote {out}/pca.pt", flush=True)


if __name__ == "__main__":
    main()

