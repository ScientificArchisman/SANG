#!/usr/bin/env python3
"""Quality-filter the clip list before caching (Part VI, D9).

SANG did no filtering at all: the only rejection was a try/except around decode, so an unknown
fraction of cached windows contain no face, an off-screen speaker, music, or overdubbed audio.
Every comparable system filters -- Teller drops clips with >50% facial movement and screens on
Sync-C/Sync-D; SoulX-FlashHead adds optical-flow and pose-occlusion gates.

    python scripts/filter_clips.py --clips data/clips_8000.txt --out data/clips_filtered.txt
    python scripts/filter_clips.py --clips data/clips_8000.txt --out data/clips_filtered.txt --sync

mediapipe needs libGLESv2, which is absent from the login node:
    export LD_LIBRARY_PATH=$CONDA_PREFIX/../gl/lib:$LD_LIBRARY_PATH

Note: training selects clips with sorted(glob(data_glob)), so point `data_glob` at the filtered
list's directory or feed it through --clips. data/clips_8000.txt is currently referenced by
nothing in the code.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party"))

from sang.video import decode_frames

PROBE_FRAMES = 16
FPS = 25.0
SR = 16000


def face_stats(frames: np.ndarray) -> dict | None:
    """Per-clip face geometry on the ORIGINAL frames: size, centre drift, detection rate."""
    from sang import face

    areas, cx, cy, seen = [], [], [], 0
    H, W = frames.shape[1], frames.shape[2]
    for f in frames:
        pts = face.landmarks_px(np.ascontiguousarray(f))
        if pts is None:
            continue
        seen += 1
        w, h = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
        areas.append((w * h) / (H * W))
        cx.append(pts[:, 0].mean() / W)
        cy.append(pts[:, 1].mean() / H)
    if not areas:
        return None
    return {
        "detect_rate": seen / len(frames),
        "face_frac": float(np.median(areas)),
        # centre drift as a fraction of frame size: Teller's "too much movement" proxy
        "drift": float(np.hypot(np.ptp(cx), np.ptp(cy))),
        "scale_var": float(np.std(areas) / (np.mean(areas) + 1e-8)),
    }


def sync_conf(path: Path, sn, frames: np.ndarray) -> float | None:
    """Discriminative margin proxy: matched vs shifted audio on this clip's own face crop."""
    import torchaudio
    from decord import AudioReader
    from sang.video import crop_resize, face_box

    box = face_box(frames)
    if box is None:
        return None
    vid = crop_resize(frames, 128, box=box)[0]
    try:
        wav = torch.from_numpy(
            AudioReader(str(path.with_suffix(".m4a")), sample_rate=SR, mono=True)[:].asnumpy())
    except Exception:
        return None
    n = round((PROBE_FRAMES - 1) / FPS * SR)
    if wav.shape[-1] < 2 * n:
        return None
    mel_t = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=1024, hop_length=160, win_length=400, n_mels=80,
        f_min=0, f_max=SR // 2, power=2.0)

    def mel(w):
        m = torch.log(mel_t(w).clamp_min(1e-5))
        return F.interpolate(m.unsqueeze(0), size=(80, 52), mode="bilinear", align_corners=False)[0]

    matched, shifted = mel(wav[:, :n]), mel(wav[:, n:2 * n])
    px = vid[None]
    with torch.no_grad():
        return float(sn.loss(px, shifted[None]) - sn.loss(px, matched[None]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(REPO / "data/clips_8000.txt"),
                    help="manifest file, or a glob pattern")
    ap.add_argument("--out", default=str(REPO / "data/clips_filtered.txt"))
    ap.add_argument("--report", default=None, help="write per-clip stats as JSONL")
    ap.add_argument("--min-face-frac", type=float, default=0.04,
                    help="median face bbox area as a fraction of the frame (measured median ~0.10)")
    ap.add_argument("--min-detect-rate", type=float, default=0.9)
    ap.add_argument("--max-drift", type=float, default=0.35, help="face-centre travel across the probe")
    ap.add_argument("--max-scale-var", type=float, default=0.5, help="cuts/zooms proxy")
    ap.add_argument("--sync", action="store_true", help="also screen on the SyncNet margin (slow)")
    ap.add_argument("--min-sync-margin", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.clips)
    clips = ([l.strip() for l in src.read_text().splitlines() if l.strip()]
             if src.exists() else sorted(glob.glob(args.clips)))
    if args.limit:
        clips = clips[: args.limit]
    print(f"{len(clips)} candidate clips")

    sn = None
    if args.sync:
        from sang.syncnet import StableSyncNet
        sn = StableSyncNet.from_checkpoint(str(REPO / "third_party/syncnet/stable_syncnet.pt"), "cpu")

    kept, reasons, rows = [], {}, []
    for i, c in enumerate(clips):
        if i % 200 == 0:
            print(f"  {i}/{len(clips)} kept={len(kept)}", flush=True)
        p = Path(c)
        try:
            frames, _, _ = decode_frames(str(p), PROBE_FRAMES, start=None, fps=FPS)
        except Exception as e:
            reasons["decode"] = reasons.get("decode", 0) + 1
            continue
        st = face_stats(frames)
        if st is None:
            reasons["no face"] = reasons.get("no face", 0) + 1
            continue
        why = None
        if st["detect_rate"] < args.min_detect_rate:
            why = "intermittent face"
        elif st["face_frac"] < args.min_face_frac:
            why = "face too small"
        elif st["drift"] > args.max_drift:
            why = "too much movement"
        elif st["scale_var"] > args.max_scale_var:
            why = "scale jump / cut"
        if why is None and sn is not None:
            m = sync_conf(p, sn, frames)
            st["sync_margin"] = m
            if m is None or m < args.min_sync_margin:
                why = "bad sync"
        rows.append({"clip": c, **st, "reject": why})
        if why:
            reasons[why] = reasons.get(why, 0) + 1
        else:
            kept.append(c)

    Path(args.out).write_text("\n".join(kept) + "\n")
    if args.report:
        Path(args.report).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\nkept {len(kept)}/{len(clips)} ({100 * len(kept) / max(1, len(clips)):.1f}%) -> {args.out}")
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  dropped {v:5d}  {k}")


if __name__ == "__main__":
    main()
