#!/usr/bin/env python3
"""Download the SyncNet lip-sync scorer into third_party/.

    export HF_TOKEN=hf_...        # or pass --token
    python scripts/fetch_loss_weights.py
"""
import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEST = REPO / "third_party" / "syncnet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()
    if not args.token:
        sys.exit("set HF_TOKEN or pass --token")

    from huggingface_hub import hf_hub_download

    DEST.mkdir(parents=True, exist_ok=True)
    out = hf_hub_download(repo_id="ByteDance/LatentSync-1.6", filename="stable_syncnet.pt",
                          local_dir=DEST, token=args.token)
    print(f"SyncNet -> {out}")


if __name__ == "__main__":
    main()
