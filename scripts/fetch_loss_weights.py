#!/usr/bin/env python3
"""Download SyncNet (LatentSync) + VideoMAE-v2 (TREPA) pretrained weights into third_party/.

Usage:
    export HF_TOKEN=hf_...   # or pass --token
    python scripts/fetch_loss_weights.py

Targets:
    third_party/syncnet/stable_syncnet.pt   # ByteDance/LatentSync-1.6 (94% acc lip-sync scorer)
    third_party/videomaev2/                # OpenGVLab/VideoMAEv2-Large (TREPA temporal encoder)
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

WEIGHTS = [
    {
        "repo_id": "ByteDance/LatentSync-1.6",
        "files": ["stable_syncnet.pt"],
        "dest": REPO / "third_party" / "syncnet",
    },
    {
        "repo_id": "OpenGVLab/VideoMAEv2-Large",
        # Large is the default TREPA encoder; Base is ~4x smaller if Large is too heavy.
        "files": ["model.safetensors", "config.json", "modeling_videomaev2.py",
                  "modeling_config.py", "preprocessor_config.json"],
        "dest": REPO / "third_party" / "videomaev2",
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="Hugging Face token")
    ap.add_argument("--videomae-size", choices=["Large", "Base"], default="Large",
                    help="VideoMAE-v2 size (Base is lighter for TREPA)")
    args = ap.parse_args()
    if not args.token:
        sys.exit("Set HF_TOKEN env var or pass --token")

    from huggingface_hub import hf_hub_download

    for w in WEIGHTS:
        dest = Path(w["dest"])
        dest.mkdir(parents=True, exist_ok=True)
        for fn in w["files"]:
            out = hf_hub_download(repo_id=w["repo_id"], filename=fn,
                                  local_dir=dest, token=args.token)
            print(f"  {w['repo_id']}/{fn} -> {out}")

    print(f"\nDone. Weights are under {REPO / 'third_party'}")
    print(f"  SyncNet:  {REPO / 'third_party' / 'syncnet' / 'stable_syncnet.pt'}")
    print(f"  VideoMAE: {REPO / 'third_party' / 'videomaev2' / 'model.safetensors'}")


if __name__ == "__main__":
    main()
