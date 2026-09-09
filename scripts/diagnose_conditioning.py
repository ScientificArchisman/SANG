"""Conditioning diagnostics: does the model actually use audio, and can it vary across slices?

Reproduces the D1/D2 measurements in docs/v3_improvement_plan.md Part VI. Read-only, CPU-friendly.

    python scripts/diagnose_conditioning.py --ckpt runs/stream_v3_161491/last.pt
    python scripts/diagnose_conditioning.py --ckpt ... --syncnet   # also check the SyncNet margin

Interpretation:
  * slice spread == 0            -> bridge_init has collapsed all content slices (D1). Fatal:
                                    the model cannot produce motion regardless of audio.
  * audio sensitivity ~= 0       -> the model ignores audio content (D2). No lip-sync is possible.
  * shuffled ~= zeroed           -> it responds to the presence of audio, not its content.
  * syncnet margin < ~0.3        -> the SyncNet loss is non-discriminative (D3); do not trust it.
"""
import argparse
import glob
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party"))

from sang.config import load_config
from sang.model import build_talking_head


def load(ckpt: str, config: str):
    cfg = load_config(REPO / config)
    model = build_talking_head(cfg, fsq_codes=None)
    ck = torch.load(REPO / ckpt if not Path(ckpt).is_absolute() else ckpt,
                    map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"ckpt {Path(ckpt).name}  step={ck.get('step')}  "
          f"(missing {len(missing)}, unexpected {len(unexpected)} — head needs the FSQ codec)")
    print(f"cfg  bridge_init={cfg.get('bridge_init')} motion_ctx={cfg.get('motion_ctx')} "
          f"face_cond={cfg.get('face_cond')} audio_lookahead={cfg.get('audio_lookahead')}")
    return cfg, model.eval()


def batch(cache_glob: str, n: int, seed: int):
    files = sorted(glob.glob(cache_glob))
    if not files:
        raise SystemExit(f"no cached windows matched {cache_glob}")
    random.Random(seed).shuffle(files)
    vids, auds, strs = [], [], []
    for f in files[:n]:
        d = torch.load(f, map_location="cpu", weights_only=False)
        vids.append(torch.cat([d["ref"][None].long(), d["ctx"][None].long(), d["video"].long()], 0))
        auds.append(d["audio"].float())
        strs.append(d["struct"].long())
    return torch.stack(vids), torch.stack(auds), torch.stack(strs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/train_stream_v3.yaml")
    ap.add_argument("--cache", default=str(REPO / "cache/v3_128_32768_ctx_wavlm/*.pt"))
    ap.add_argument("--n", type=int, default=8, help="windows to average over")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--syncnet", action="store_true",
                    help="also measure the SyncNet matched/mismatched margin (needs VidTok)")
    args = ap.parse_args()

    cfg, model = load(args.ckpt, args.config)
    video, audio, struct = batch(args.cache, args.n, args.seed)
    B, tv, h, w = video.shape
    r, n_cond = h * w, model.ref_slices
    print(f"grid={tuple(video.shape)} audio={tuple(audio.shape)} tv={tv} r={r} ref_slices={n_cond}\n")

    # generation regime: only the conditioning slices are known
    known = torch.zeros(B, tv * r, dtype=torch.bool)
    known[:, : n_cond * r] = True

    print("=" * 68)
    print("D1  can the model differentiate content slices at generation time?")
    print("=" * 68)
    with torch.no_grad():
        e_with = model._embed_grid(video, struct, known)
        e_none = model._embed_grid(video, None, known)
    masked = ~known
    d_struct = (e_with - e_none)[masked].abs().max().item()
    ec = e_with.reshape(B, tv, r, -1)[:, n_cond:]
    spread = (ec - ec[:, :1]).abs().max().item()
    print(f"  |embed(struct) - embed(no struct)| on masked positions : {d_struct:.6f}"
          f"   {'<- struct DISCARDED (D1)' if d_struct == 0 else ''}")
    print(f"  spread across the {ec.shape[1]} content slices             : {spread:.6f}"
          f"   {'<- ALL SLICES IDENTICAL (D1, fatal)' if spread == 0 else ''}\n")

    print("=" * 68)
    print("D2  does the model use audio content?")
    print("=" * 68)
    with torch.no_grad():
        def hs(a, st=struct):
            return model.hidden_states(video, a, known, struct=st)
        ref_h = hs(audio)
        perm = torch.stack([a[torch.randperm(a.shape[0])] for a in audio])
        variants = {"shuffled audio": hs(perm), "zeroed audio": hs(torch.zeros_like(audio)),
                    "struct removed": hs(audio, st=None)}

    def rel(x):
        a = ref_h.reshape(B, tv, r, -1)[:, n_cond:]
        b = x.reshape(B, tv, r, -1)[:, n_cond:]
        return ((a - b).norm() / a.norm()).item() * 100

    for name, hv in variants.items():
        print(f"  relative change in content hidden states, {name:<15}: {rel(hv):7.3f} %")
    print("  (shuffled ~= zeroed means the model reads presence, not content)\n")

    if args.syncnet:
        print("=" * 68)
        print("D3  is the SyncNet loss discriminative on ground-truth video?")
        print("=" * 68)
        from sang.video import load_vidtok
        from sang.syncnet import StableSyncNet
        mel = torch.stack([torch.load(f, map_location="cpu", weights_only=False)["mel"].float()
                           for f in sorted(glob.glob(args.cache))[: args.n]])
        vidtok = load_vidtok(cfg["codebook"], device="cpu")
        sn = StableSyncNet.from_checkpoint(str(REPO / "third_party/syncnet/stable_syncnet.pt"), "cpu")
        with torch.no_grad():
            px = vidtok.decode(video[:, n_cond:], decode_from_indices=True)
            matched = sn.loss(px, mel).item()
            mismatched = sn.loss(px, mel.flip(0)).item()
        print(f"  GT video + MATCHED audio    : {matched:.4f}")
        print(f"  GT video + MISMATCHED audio : {mismatched:.4f}")
        print(f"  discriminative margin       : {mismatched - matched:.4f}"
              f"   {'<- NON-DISCRIMINATIVE (D3)' if mismatched - matched < 0.3 else ''}")


if __name__ == "__main__":
    main()
