#!/usr/bin/env python3
"""Quality-filter the clip list before caching: drop clips whose face is missing, small, moving out of
frame or cut. Audio-video sync is screened later, per source video, by scripts/sync_offsets.py.

    python scripts/filter_clips.py --clips data/clips_all.txt --out data/clips_filtered_all.txt

mediapipe needs libGLESv2, which is absent from the login node:
    export LD_LIBRARY_PATH=$CONDA_PREFIX/../gl/lib:$LD_LIBRARY_PATH
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sang.video import decode_frames

PROBE_FRAMES = 16
FPS = 25.0
LANDMARKER = REPO / "checkpoints" / "face_landmarker.task"
_landmarker = None


def landmarks_px(frame: np.ndarray):
    """[H, W, 3] uint8 RGB -> [478, 2] float32 pixel landmarks (MediaPipe FaceLandmarker), or None if no face."""
    global _landmarker
    import mediapipe as mp
    if _landmarker is None:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        if not LANDMARKER.exists():
            raise FileNotFoundError(f"missing {LANDMARKER}; download face_landmarker.task into checkpoints/")
        _landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(LANDMARKER)),
            running_mode=vision.RunningMode.IMAGE, num_faces=1, output_face_blendshapes=False))
    res = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame)))
    if not res.face_landmarks:
        return None
    h, w = frame.shape[:2]
    return np.array([[lm.x * w, lm.y * h] for lm in res.face_landmarks[0]], np.float32)


def face_stats(frames: np.ndarray) -> dict | None:
    """Per-clip face geometry on the ORIGINAL frames: size, centre drift (Teller's "too much movement"
    proxy, as a fraction of the frame), size variation and detection rate."""
    areas, cx, cy, seen = [], [], [], 0
    H, W = frames.shape[1], frames.shape[2]
    for f in frames:
        pts = landmarks_px(f)
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
        "drift": float(np.hypot(np.ptp(cx), np.ptp(cy))),
        "scale_var": float(np.std(areas) / (np.mean(areas) + 1e-8)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=str(REPO / "data/clips_all.txt"), help="manifest file, or a glob pattern")
    ap.add_argument("--out", default=str(REPO / "data/clips_filtered_all.txt"))
    ap.add_argument("--report", default=None, help="write per-clip stats as JSONL")
    ap.add_argument("--min-face-frac", type=float, default=0.04,
                    help="median face bbox area as a fraction of the frame (measured median ~0.10)")
    ap.add_argument("--min-detect-rate", type=float, default=0.9)
    ap.add_argument("--max-drift", type=float, default=0.35, help="face-centre travel across the probe")
    ap.add_argument("--max-scale-var", type=float, default=0.5, help="cuts/zooms proxy")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.clips)
    clips = ([l.strip() for l in src.read_text().splitlines() if l.strip()]
             if src.exists() else sorted(glob.glob(args.clips)))
    if args.limit:
        clips = clips[: args.limit]
    print(f"{len(clips)} candidate clips")

    kept, reasons, rows = [], {}, []
    for i, c in enumerate(clips):
        if i % 200 == 0:
            print(f"  {i}/{len(clips)} kept={len(kept)}", flush=True)
        try:
            frames, _, _ = decode_frames(c, PROBE_FRAMES, start=None, fps=FPS)
        except Exception:
            reasons["decode"] = reasons.get("decode", 0) + 1
            continue
        st = face_stats(frames)
        if st is None:
            reasons["no face"] = reasons.get("no face", 0) + 1
            continue
        why = ("intermittent face" if st["detect_rate"] < args.min_detect_rate else
               "face too small" if st["face_frac"] < args.min_face_frac else
               "too much movement" if st["drift"] > args.max_drift else
               "scale jump / cut" if st["scale_var"] > args.max_scale_var else None)
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

